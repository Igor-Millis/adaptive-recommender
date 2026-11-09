"""Modelo supervisionado de propensão de compra P(compra de p por c nos próximos H dias).

Construção do dataset (ver MATHEMATICAL_FOUNDATIONS.md §6–8):
  * snapshots mensais t; features calculadas com dados < t; rótulo y = 1 se o
    cliente comprou p em [t, t+H). Exige t + H ≤ corte de treino (sem vazamento);
  * candidatos: produtos lançados até t e ainda NÃO comprados pelo cliente
    (tarefa de descoberta);
  * negativos não são "definitivos": um par negativo num snapshot pode ser
    positivo noutro, e só entram clientes *ativos* na janela (que fizeram ≥ 1
    pedido — houve oportunidade de compra). Exposição prévia (impressão,
    engajamento, rejeição) entra como feature;
  * negativos são subamostrados com taxa r e recebem peso 1/r, preservando a
    calibração da probabilidade (estimador de Horvitz-Thompson).
Seleção do algoritmo e calibração (Platt) usam um snapshot de *holdout*
temporal posterior aos snapshots de treino.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CATEGORIES, DATA_START, HORIZON_DAYS, RANDOM_SEED, SUP
from src.preprocessing.feature_engineering import POSITIVE_OUTCOMES
from src.recommenders.base import BaseRecommender, ExpertContext
from src.recommenders.context import build_context, summer_score
from src.utils.logging_setup import get_logger
from src.utils.validation import validate_probabilities

log = get_logger("supervised")

BANDS = ["Econômica", "Intermediária", "Premium"]


# ============================================================================ features
def pair_features(snapshot, ctx: ExpertContext, horizon_days: int = HORIZON_DAYS) -> tuple[np.ndarray, list[str]]:
    """Matriz de features para todos os pares (c, p), ordem c-major (n_c·n_p × d)."""
    s = snapshot
    nc, npr = s.n_customers, s.n_products
    cf = ctx.customer_features.reindex(s.customer_ids)
    pf = ctx.product_features.reindex(s.product_ids)
    prods = s.products
    cat_idx = prods["category"].map({c: k for k, c in enumerate(CATEGORIES)}).to_numpy()
    sub_codes, sub_uni = pd.factorize(prods["subcategory"])
    aud_codes, _ = pd.factorize(prods["target_audience"])
    band_idx = prods["price_band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()

    feats: dict[str, np.ndarray] = {}

    def cust(v):
        return np.repeat(np.asarray(v, float), npr)

    def prod(v):
        return np.tile(np.asarray(v, float), nc)

    # ---- cliente
    feats["c_log_orders"] = cust(np.log1p(cf["n_orders"]))
    feats["c_log_distinct"] = cust(np.log1p(cf["n_distinct_products"]))
    feats["c_log_ticket"] = cust(np.log1p(cf["avg_ticket"]))
    feats["c_frequency"] = cust(cf["frequency_per_month"])
    feats["c_log_recency"] = cust(np.log1p(cf["recency_days"].fillna(730)))
    feats["c_entropy"] = cust(cf["category_entropy"])
    feats["c_price_index"] = cust(cf["price_index"])
    feats["c_engagement_rate"] = cust(cf["engagement_rate"])
    feats["c_has_history"] = cust(cf["has_history"])
    feats["c_log_tenure"] = cust(np.log1p(cf["tenure_days"]))
    feats["c_drift"] = cust(cf["drift_js"])
    feats["c_seasonality"] = cust(cf["seasonality_index"])
    # ---- produto
    feats["p_log_price"] = prod(pf["log_price"])
    feats["p_rel_price"] = prod(pf["rel_price"])
    feats["p_log_pop90"] = prod(np.log1p(pf["popularity_90d"]))
    feats["p_log_pop_all"] = prod(np.log1p(pf["popularity_all"]))
    age = pf["age_days"].clip(lower=0)
    feats["p_log_age"] = prod(np.log1p(age))
    feats["p_new"] = prod((age < 90).astype(float))
    season_sign = prods["season"].map({"Verão": 1.0, "Inverno": -1.0, "Atemporal": 0.0}).to_numpy()
    feats["p_season_match"] = prod(season_sign * summer_score(s.as_of, horizon_days))
    feats["p_band"] = prod(band_idx)
    for k, c in enumerate(CATEGORIES):
        feats[f"p_cat_{k}"] = prod((cat_idx == k).astype(float))

    # ---- par cliente × produto
    def share_matrix(codes, n_levels, since=None):
        it = s.items if since is None else s.items[s.items["order_date"] >= since]
        M = np.zeros((nc, n_levels))
        if len(it):
            pj = it["product_id"].map(s.pidx).to_numpy()
            np.add.at(M, (it["customer_id"].map(s.cidx).to_numpy(), codes[pj]), it["quantity"].to_numpy(float))
        tot = M.sum(1, keepdims=True)
        return np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)

    cat_share = share_matrix(cat_idx, len(CATEGORIES))
    feats["x_cat_share"] = cat_share[:, cat_idx].ravel()
    feats["x_cat_share_recent"] = share_matrix(cat_idx, len(CATEGORIES), s.as_of - pd.Timedelta(days=120))[:, cat_idx].ravel()
    feats["x_subcat_share"] = share_matrix(sub_codes, len(sub_uni))[:, sub_codes].ravel()
    feats["x_audience_share"] = share_matrix(aud_codes, aud_codes.max() + 1)[:, aud_codes].ravel()
    feats["x_band_share"] = share_matrix(band_idx, 3)[:, band_idx].ravel()
    feats["x_price_gap"] = (np.asarray(pf["rel_price"], float)[None, :] - np.asarray(cf["price_index"], float)[:, None]).ravel()

    declared_cat = np.zeros((nc, len(CATEGORIES)))
    declared_band = np.zeros((nc, 3))
    for i, cid in enumerate(s.customer_ids):
        row = s.customers.loc[cid]
        for c in str(row.get("declared_categories") or "").split(";"):
            if c in CATEGORIES:
                declared_cat[i, CATEGORIES.index(c)] = 1
        if row.get("declared_price_band") in BANDS:
            declared_band[i, BANDS.index(row["declared_price_band"])] = 1
    feats["x_declared_cat"] = declared_cat[:, cat_idx].ravel()
    feats["x_declared_band"] = declared_band[:, band_idx].ravel()

    # engajamento/rejeição por categoria e exposição prévia ao produto
    out = ctx.outcomes
    eng = np.zeros((nc, len(CATEGORIES)))
    imp = np.zeros((nc, len(CATEGORIES)))
    rej = np.zeros((nc, len(CATEGORIES)))
    p_imp = np.zeros((nc, npr))
    p_eng = np.zeros((nc, npr))
    p_rej = np.zeros((nc, npr))
    if len(out):
        ci = out["customer_id"].map(s.cidx).to_numpy()
        pj = out["product_id"].map(s.pidx).to_numpy()
        ki = cat_idx[pj]
        pos = out["outcome"].isin(POSITIVE_OUTCOMES).to_numpy()
        rj = (out["outcome"] == "rejection").to_numpy()
        np.add.at(imp, (ci, ki), 1)
        np.add.at(eng, (ci[pos], ki[pos]), 1)
        np.add.at(rej, (ci[rj], ki[rj]), 1)
        np.add.at(p_imp, (ci, pj), 1)
        np.add.at(p_eng, (ci[pos], pj[pos]), 1)
        np.add.at(p_rej, (ci[rj], pj[rj]), 1)
    feats["x_cat_engagement"] = ((eng + 1) / (imp + 6))[:, cat_idx].ravel()
    feats["x_cat_rejection"] = ((rej + 0.2) / (imp + 6))[:, cat_idx].ravel()
    feats["x_prod_impressed"] = np.minimum(p_imp, 5).ravel()
    feats["x_prod_engaged"] = (p_eng > 0).astype(float).ravel()
    feats["x_prod_rejected"] = (p_rej > 0).astype(float).ravel()

    labels = ctx.cluster_labels.reindex(s.customer_ids).fillna(-1).astype(int).to_numpy()
    cl_share = np.zeros((nc, len(CATEGORIES)))
    for g in np.unique(labels[labels >= 0]):
        cl_share[labels == g] = cat_share[labels == g].mean(axis=0)
    feats["x_cluster_cat_share"] = cl_share[:, cat_idx].ravel()

    names = list(feats.keys())
    X = np.column_stack([feats[n] for n in names]).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


FEATURE_LABELS = {
    "x_cat_share": "participação da categoria nas compras", "x_cat_share_recent": "participação recente da categoria",
    "x_subcat_share": "afinidade com a subcategoria", "x_audience_share": "afinidade com o público-alvo",
    "x_band_share": "afinidade com a faixa de preço", "x_price_gap": "distância de preço ao padrão do cliente",
    "x_declared_cat": "categoria declarada no cadastro", "x_declared_band": "faixa de preço declarada",
    "x_cat_engagement": "engajamento com a categoria", "x_cat_rejection": "rejeições na categoria",
    "x_prod_impressed": "exposições anteriores ao produto", "x_prod_engaged": "engajou com o produto antes",
    "x_prod_rejected": "rejeitou o produto antes", "x_cluster_cat_share": "preferência do segmento pela categoria",
    "p_log_pop90": "popularidade recente do produto", "p_log_pop_all": "popularidade histórica",
    "p_season_match": "aderência à estação", "p_new": "lançamento recente", "p_rel_price": "preço relativo na categoria",
    "c_log_orders": "volume de pedidos do cliente", "c_frequency": "frequência de compra", "c_log_recency": "recência",
    "p_log_price": "preço do produto", "p_band": "faixa de preço", "p_log_age": "tempo desde o lançamento",
    "c_engagement_rate": "taxa de engajamento do cliente", "c_price_index": "sensibilidade a preço do cliente",
    "c_entropy": "diversidade de compras do cliente", "c_has_history": "existência de histórico",
    "c_log_distinct": "variedade de produtos comprados", "c_log_ticket": "ticket médio", "c_log_tenure": "tempo de cadastro",
    "c_drift": "mudança recente de preferência", "c_seasonality": "sazonalidade do cliente",
    **{f"p_cat_{k}": f"categoria {c}" for k, c in enumerate(CATEGORIES)},
}


def candidate_mask(snapshot) -> np.ndarray:
    return snapshot.launched[None, :] & ~snapshot.purchased_matrix()


# ============================================================================ dataset
def snapshot_dates(cutoff: pd.Timestamp, horizon: int = HORIZON_DAYS, step: int = SUP.snapshot_step_days,
                   min_history: int = SUP.min_history_days) -> list[pd.Timestamp]:
    last = cutoff - pd.Timedelta(days=horizon)
    first = DATA_START + pd.Timedelta(days=min_history)
    dates = []
    t = last
    while t >= first:
        dates.append(t.normalize())
        t -= pd.Timedelta(days=step)
    return sorted(dates)


def build_labeled_snapshot(dataset, t: pd.Timestamp, horizon: int, fixed_k: int | None, sample_rate: float | None,
                           rng: np.random.Generator, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    if t + pd.Timedelta(days=horizon) > cutoff:
        raise ValueError(f"Snapshot {t} com janela além do corte {cutoff} (vazamento).")
    snap = dataset.snapshot(t)
    ctx, _ = build_context(snap, fixed_k=fixed_k, fast=True)
    X, names = pair_features(snap, ctx, horizon)
    win = dataset.window_items(t, t + pd.Timedelta(days=horizon))
    Y = np.zeros((snap.n_customers, snap.n_products), dtype=bool)
    if len(win):
        Y[win["customer_id"].map(snap.cidx).to_numpy(), win["product_id"].map(snap.pidx).to_numpy()] = True
    active = np.zeros(snap.n_customers, dtype=bool)
    active[win["customer_id"].map(snap.cidx).unique()] = True
    cand = candidate_mask(snap) & active[:, None]
    y = Y.ravel()
    keep = cand.ravel()
    w = np.ones_like(y, dtype=float)
    if sample_rate is not None and sample_rate < 1:
        neg = keep & ~y
        drop = neg & (rng.random(len(y)) > sample_rate)
        keep = keep & ~drop
        w[neg] = 1.0 / sample_rate
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return None
    df = pd.DataFrame(X[idx], columns=names)
    df["y"] = y[idx].astype(int)
    df["w"] = w[idx]
    df["snapshot"] = t
    df["customer_id"] = np.repeat(snap.customer_ids, snap.n_products)[idx]
    df["product_id"] = np.tile(snap.product_ids, snap.n_customers)[idx]
    return df


# ============================================================================ modelos
def _make_model(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=15, max_features="sqrt", max_samples=0.5, n_jobs=-1,
                                      random_state=RANDOM_SEED)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=RANDOM_SEED, verbose=-1)
    raise ValueError(name)


class PlattCalibrator:
    """Calibração de Platt sobre o logit da probabilidade bruta: p' = σ(a·logit(p) + b)."""

    def fit(self, p, y, w=None):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(z[:, None], y, sample_weight=w)
        return self

    def transform(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.lr.predict_proba(z[:, None])[:, 1]


def classification_metrics(y, p, w=None, threshold: float | None = None) -> dict:
    y = np.asarray(y)
    p = validate_probabilities(p)
    out = {"roc_auc": float(roc_auc_score(y, p, sample_weight=w)), "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
           "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w)),
           "brier": float(brier_score_loss(y, p, sample_weight=w)), "base_rate": float(np.average(y, weights=w)),
           "n": int(len(y)), "n_pos": int(y.sum())}
    if threshold is None:
        # limiar que maximiza F1 (definido no holdout e reaplicado no teste)
        qs = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 60)))
        best = max(qs, key=lambda t: _f1(y, p >= t))
        threshold = float(best)
    pred = p >= threshold
    out.update({"threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0))})
    return out


def _f1(y, pred):
    tp = np.sum(pred & (y == 1))
    if tp == 0:
        return 0.0
    pr, rc = tp / pred.sum(), tp / (y == 1).sum()
    return 2 * pr * rc / (pr + rc)


def reliability(y, p, n_bins: int = 10) -> dict:
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean.tolist(), "fraction_positive": frac.tolist()}


class SupervisedRecommender(BaseRecommender):
    name = "supervised"
    label = "Supervisionado"

    def __init__(self, model=None, calibrator=None, feature_names=None, algorithm: str = "", report: dict | None = None):
        super().__init__()
        self.model = model
        self.calibrator = calibrator
        self.feature_names = feature_names or []
        self.algorithm = algorithm
        self.report = report or {}

    # ------------------------------------------------------------------ treino
    @classmethod
    def train(cls, dataset, cutoff: pd.Timestamp, fixed_k: int | None = None, horizon: int = HORIZON_DAYS,
              cfg=SUP) -> "SupervisedRecommender":
        rng = np.random.default_rng(RANDOM_SEED)
        dates = snapshot_dates(cutoff, horizon)
        if len(dates) < 3:
            raise ValueError("Histórico insuficiente para treinar o modelo supervisionado.")
        holdout_t = dates[-1]
        if fixed_k is None:  # k escolhido uma vez (no holdout) e reutilizado nos snapshots
            _, seg = build_context(dataset.snapshot(holdout_t))
            fixed_k = seg.k if seg is not None else None
        train_dates = [t for t in dates if t + pd.Timedelta(days=horizon) <= holdout_t]
        log.info("Supervisionado: corte=%s, %d snapshots de treino, holdout=%s", cutoff.date(), len(train_dates), holdout_t.date())
        tr = pd.concat([d for t in train_dates
                        if (d := build_labeled_snapshot(dataset, t, horizon, fixed_k, cfg.negative_sample_rate, rng, holdout_t)) is not None])
        ho = build_labeled_snapshot(dataset, holdout_t, horizon, fixed_k, None, rng, cutoff)
        names = [c for c in tr.columns if c not in ("y", "w", "snapshot", "customer_id", "product_id")]
        Xtr, ytr, wtr = tr[names].to_numpy(), tr["y"].to_numpy(), tr["w"].to_numpy()
        Xho, yho = ho[names].to_numpy(), ho["y"].to_numpy()
        candidates = {}
        for alg in cfg.candidate_models:
            t0 = time.perf_counter()
            m = _make_model(alg)
            fit_kw = {"logisticregression__sample_weight": wtr} if alg == "logistic_regression" else {"sample_weight": wtr}
            m.fit(Xtr, ytr, **fit_kw)
            p_raw = m.predict_proba(Xho)[:, 1]
            cal = PlattCalibrator().fit(p_raw, yho)
            p_cal = cal.transform(p_raw)
            metrics = classification_metrics(yho, p_cal)
            metrics["raw_log_loss"] = float(log_loss(yho, np.clip(p_raw, 1e-7, 1 - 1e-7)))
            metrics["fit_seconds"] = time.perf_counter() - t0
            candidates[alg] = {"model": m, "cal": cal, "metrics": metrics, "reliability": reliability(yho, p_cal),
                               "reliability_raw": reliability(yho, p_raw)}
            log.info("  %s: PR-AUC=%.4f ROC-AUC=%.4f logloss=%.4f", alg, metrics["pr_auc"], metrics["roc_auc"], metrics["log_loss"])
        best = max(candidates, key=lambda a: candidates[a]["metrics"]["pr_auc"])
        report = {
            "cutoff": str(cutoff.date()), "holdout_snapshot": str(holdout_t.date()),
            "train_snapshots": [str(t.date()) for t in train_dates], "n_train": int(len(tr)), "n_train_pos": int(ytr.sum()),
            "n_holdout": int(len(ho)), "selected": best,
            "candidates": {a: {"metrics": v["metrics"], "reliability": v["reliability"], "reliability_raw": v["reliability_raw"]}
                           for a, v in candidates.items()},
        }
        inst = cls(candidates[best]["model"], candidates[best]["cal"], names, best, report)
        inst.all_candidates = {a: (v["model"], v["cal"]) for a, v in candidates.items()}
        return inst

    def with_algorithm(self, alg: str) -> "SupervisedRecommender":
        m, c = self.all_candidates[alg]
        return SupervisedRecommender(m, c, self.feature_names, alg, self.report)

    # ------------------------------------------------------------------ aplicação
    def predict_matrix(self, snapshot, ctx) -> tuple[np.ndarray, np.ndarray]:
        X, names = pair_features(snapshot, ctx)
        if names != self.feature_names:
            raise ValueError("Features divergentes das usadas no treino do modelo supervisionado.")
        p = self.calibrator.transform(self.model.predict_proba(X)[:, 1])
        p = validate_probabilities(p, "probabilidade de conversão")
        return p.reshape(snapshot.n_customers, snapshot.n_products), X

    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        P, X = self.predict_matrix(snapshot, ctx)
        self._X = X
        self._scores = P
        return self

    def update_customer(self, snapshot, ctx, customer_id: int) -> None:
        """Recalcula as features (não os parâmetros) do cliente — o modelo só é retreinado na rotina controlada."""
        self.fit(snapshot, ctx)

    def confidence(self):
        f = self.ctx.customer_features.reindex(self.snapshot.customer_ids)
        n = f["n_orders"].fillna(0).to_numpy(float) + f["n_impressions"].fillna(0).to_numpy(float) / 10
        return 0.35 + 0.65 * (1 - np.exp(-n / 10.0))  # generaliza entre clientes: nunca é zero

    def contributions(self, customer_id: int, product_id: int, top: int = 5) -> pd.DataFrame:
        i, j = self._ci(customer_id), self._pj(product_id)
        x = self._X[i * self.snapshot.n_products + j][None, :]
        if self.algorithm == "lightgbm":
            c = self.model.booster_.predict(x, pred_contrib=True)[0][:-1]  # TreeSHAP (log-odds)
        elif self.algorithm == "logistic_regression":
            sc = self.model.named_steps["standardscaler"]
            lr = self.model.named_steps["logisticregression"]
            c = ((x - sc.mean_) / sc.scale_)[0] * lr.coef_[0]
        else:
            imp = self.model.feature_importances_
            mu = self._X.mean(axis=0)
            sd = self._X.std(axis=0) + 1e-9
            c = imp * (x[0] - mu) / sd
        df = pd.DataFrame({"feature": self.feature_names, "contribution": c, "value": x[0]})
        df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
        return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)

    def explain(self, customer_id, product_id):
        i, j = self._ci(customer_id), self._pj(product_id)
        p = float(self._scores[i, j])
        c = self.contributions(customer_id, product_id)
        pos = [r.label for r in c.itertuples() if r.contribution > 0][:3]
        neg = [r.label for r in c.itertuples() if r.contribution < 0][:2]
        summary = f"Probabilidade estimada de compra em {HORIZON_DAYS} dias: {p:.1%} ({self.algorithm.replace('_', ' ')} calibrado)."
        if pos:
            summary += " A favor: " + ", ".join(pos) + "."
        if neg:
            summary += " Contra: " + ", ".join(neg) + "."
        return {"summary": summary, "details": {"probabilidade": p, "contribuicoes": c.to_dict("records")}}

    # ------------------------------------------------------------------ persistência
    def to_artifact(self) -> dict:
        return {"model": self.model, "calibrator": self.calibrator, "feature_names": self.feature_names,
                "algorithm": self.algorithm, "report": self.report}

    @classmethod
    def from_artifact(cls, art: dict) -> "SupervisedRecommender":
        return cls(art["model"], art["calibrator"], art["feature_names"], art["algorithm"], art["report"])
