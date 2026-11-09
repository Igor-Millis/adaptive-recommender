"""Motor de recomendação de produção (independente do Streamlit).

Carrega o estado ativo (supervisionado versionado, clusters, estado bayesiano
incremental e gating) e ajusta os especialistas determinísticos a partir do
banco. Expõe recomendação explicável, comparação entre especialistas e
atualização após feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import joblib
import numpy as np
import pandas as pd

from config.settings import EXPERT_LABELS, EXPERTS, RANKING, ROOT_DIR, RankingConfig
from database.repositories import Repositories
from src.adaptive.ensemble import GatingExplanation, combine, explain_weights, normalize_expert_scores
from src.adaptive.gating import FixedGating, LearnedGating, RuleGating
from src.adaptive.model_selection import compute_signals
from src.learning.retraining import segmentation_from_db
from src.preprocessing.snapshot import Dataset
from src.ranking.diversification import mmr_rank
from src.ranking.filtering import candidate_mask, fatigue_penalty
from src.ranking.scoring import ranking_scores
from src.recommenders.suite import ExpertSuite
from src.recommenders.supervised import SupervisedRecommender
from src.utils.logging_setup import get_logger
from src.utils.validation import validate_probabilities, validate_product_ids, validate_weights

log = get_logger("recommendation_service")

GATING_METHODS = {
    "learned": "Adaptativo (gating treinado)",
    "rules": "Adaptativo (regras)",
    "fixed": "Pesos fixos (ajustados)",
    "equal": "Pesos iguais",
}


@dataclass
class RecommendationResult:
    customer_id: int
    items: pd.DataFrame
    explanation: GatingExplanation
    method: str
    config: RankingConfig
    run_id: int | None = None
    recommendation_ids: list = field(default_factory=list)
    exclusions: dict = field(default_factory=dict)
    as_of: pd.Timestamp | None = None


class RecommendationEngine:
    def __init__(self, engine):
        self.engine = engine
        self.repos = Repositories(engine)
        self.load()

    # ------------------------------------------------------------------ carga do estado
    def load(self) -> None:
        r = self.repos
        sup_v = r.models.active("supervised")
        if sup_v.empty:
            raise RuntimeError("Nenhum modelo supervisionado ativo — execute scripts/train_models.py.")
        self.sup_version = sup_v.iloc[0].to_dict()
        art = joblib.load(ROOT_DIR / self.sup_version["artifact_path"])
        self.supervised = SupervisedRecommender.from_artifact(art)
        self.segmentation = segmentation_from_db(r)
        g_v = r.models.active("gating")
        self.gating_version = g_v.iloc[0].to_dict() if not g_v.empty else None
        payload = json.loads((ROOT_DIR / self.gating_version["artifact_path"]).read_text(encoding="utf-8")) if self.gating_version else {}
        self.gatings = {"rules": RuleGating(), "equal": FixedGating({e: 1.0 for e in EXPERTS})}
        if payload:
            self.gatings["learned"] = LearnedGating.from_dict(payload["learned"])
            self.gatings["fixed"] = FixedGating(payload["fixed"])
        self.gating_payload = payload
        self._fit_state()

    def _fit_state(self) -> None:
        self.dataset = Dataset.from_db(self.engine)
        self.as_of = max(pd.Timestamp.now().floor("s"), self.dataset.max_date + pd.Timedelta(seconds=1)) + pd.Timedelta(seconds=1)
        self.snapshot = self.dataset.snapshot(self.as_of)
        self.suite = ExpertSuite(self.supervised).fit(self.snapshot, segmentation=self.segmentation,
                                                       bayes_state=self.repos.bayes.all())
        self.signals = compute_signals(self.suite)

    def model_versions(self) -> dict:
        df = self.repos.models.active()
        return {r.model_name: int(r.version) for r in df.itertuples()}

    # ------------------------------------------------------------------ recomendação
    def _weights(self, method: str) -> np.ndarray:
        g = self.gatings.get(method) or self.gatings["rules"]
        return validate_weights(g.weights(self.signals))

    def recommend(self, customer_id: int, config: RankingConfig = RANKING, method: str = "learned",
                  log_run: bool = True) -> RecommendationResult:
        s = self.snapshot
        if int(customer_id) not in s.cidx:
            raise ValueError(f"Cliente {customer_id} inexistente.")
        i = s.cidx[int(customer_id)]
        mask, excl_all = candidate_mask(s, config, self.suite.ctx.outcomes)
        excl = {k: int(v[i]) for k, v in excl_all.items()}
        raw = self.suite.scores(EXPERTS)
        norm = normalize_expert_scores(raw, mask)
        W = self._weights(method)
        W_rules = self.gatings["rules"].weights(self.signals)
        rel, contrib = combine(norm, W, EXPERTS)
        fatigue = fatigue_penalty(s, self.suite.ctx.outcomes, config)
        margins = s.products["margin"].to_numpy(float)
        final = ranking_scores(rel, margins, config.margin_weight, fatigue)
        cats = s.products["category"].to_numpy()
        sim = self.suite.experts["content_based"].item_similarity
        sel = mmr_rank(final[i], mask[i], sim, config.top_k, config.diversity_lambda, cats, config.max_per_category)
        prob = validate_probabilities(self.suite.experts["supervised"].score()[i], "probabilidade de conversão")
        rows = []
        for rank, j in enumerate(sel, start=1):
            pid = int(s.product_ids[j])
            p = s.products.loc[pid]
            c = {e: float(contrib[i, k, j]) for k, e in enumerate(EXPERTS)}
            tot = sum(c.values()) or 1.0
            share = {e: v / tot for e, v in c.items()}
            reasons = []
            for e in sorted(share, key=share.get, reverse=True)[:2]:
                if share[e] > 0.12:
                    reasons.append(f"[{EXPERT_LABELS[e]}] " + self.suite.experts[e].explain(customer_id, pid)["summary"])
            if fatigue[i, j] > 0:
                reasons.append("Exibido várias vezes recentemente sem resposta: score reduzido por fadiga.")
            if config.purchased_rule == "recent" and s.purchased_matrix()[i, j]:
                reasons.append("Recompra: produto já adquirido há mais de "
                               f"{config.recent_days} dias (regra de recompra ativa).")
            rows.append({"rank": rank, "product_id": pid, "name": p["name"], "category": p["category"],
                         "subcategory": p["subcategory"], "price": float(p["price"]), "stock": int(p["stock"]),
                         "margin": float(p["margin"]), "relevance": float(np.clip(rel[i, j], 0, 1)),
                         "final_score": float(final[i, j]), "conversion_probability": float(prob[j]),
                         "contributions": share, "reasons": reasons, "reasons_text": " | ".join(reasons)})
        items = pd.DataFrame(rows)
        validate_product_ids(items["product_id"] if len(items) else [], s.product_ids)
        adj = self.gatings["learned"].adjustments(self.signals) if method == "learned" and "learned" in self.gatings else None
        expl = explain_weights(int(customer_id), self.signals, W, W_rules, self.suite.confidences(EXPERTS), method, adj)
        res = RecommendationResult(int(customer_id), items, expl, method, config, exclusions=excl, as_of=self.as_of)
        if log_run and len(items):
            run_id, rec_ids = self.repos.recommendations.save_run(
                int(customer_id), self.as_of, method, {k: v for k, v in config.__dict__.items()}, self.model_versions(),
                " ".join(expl.justification), items, expl.weights,
                {k: float(v) for k, v in self.signals.loc[int(customer_id)].items()})
            res.run_id, res.recommendation_ids = run_id, rec_ids
            items["recommendation_id"] = rec_ids
        return res

    def expert_rankings(self, customer_id: int, k: int = 10, config: RankingConfig = RANKING) -> dict[str, list[int]]:
        """Top-k de cada especialista isolado e de cada estratégia de combinação (mesmos candidatos)."""
        s = self.snapshot
        i = s.cidx[int(customer_id)]
        mask, _ = candidate_mask(s, config, self.suite.ctx.outcomes)
        norm = normalize_expert_scores(self.suite.scores(EXPERTS), mask)
        out = {}
        for e in EXPERTS:
            sc = np.where(mask[i], norm[e][i], -np.inf)
            out[e] = [int(s.product_ids[j]) for j in np.argsort(-sc, kind="stable")[:k] if np.isfinite(sc[j])]
        for m in self.gatings:
            rel, _ = combine(norm, self._weights(m), EXPERTS)
            sc = np.where(mask[i], rel[i], -np.inf)
            out[f"hybrid_{m}"] = [int(s.product_ids[j]) for j in np.argsort(-sc, kind="stable")[:k] if np.isfinite(sc[j])]
        return out

    def weights_table(self, method: str = "learned") -> pd.DataFrame:
        return pd.DataFrame(self._weights(method), index=self.signals.index, columns=EXPERTS)

    # ------------------------------------------------------------------ aprendizado
    def refresh_after_feedback(self, customer_id: int) -> dict:
        """Recarrega dados do banco e atualiza incrementalmente o estado do cliente afetado."""
        self.dataset = Dataset.from_db(self.engine)
        self.as_of = max(pd.Timestamp.now().floor("s"), self.dataset.max_date + pd.Timedelta(seconds=1)) + pd.Timedelta(seconds=1)
        self.snapshot = self.dataset.snapshot(self.as_of)
        done = self.suite.refresh_customer(self.snapshot, int(customer_id), bayes_state=self.repos.bayes.all())
        self.signals = compute_signals(self.suite)
        return done

    def config_with(self, **kw) -> RankingConfig:
        return replace(RANKING, **kw)
