"""Registro de feedback e propagação das atualizações incrementais.

Fluxo de ``FeedbackService.register`` (uma transação no banco):
  1. valida cliente/produto/tipo de interação;
  2. grava a impressão (se ainda não existe na sessão) e o evento;
  3. se for compra: cria pedido + item e baixa o estoque;
  4. atualiza o estado bayesiano (Δ de evidência, conjugado) e registra o log;
  5. registra o evento de auditoria.
Depois da transação: recalcula as features comportamentais do cliente
(tabela ``customer_features``). A atualização do estado do recomendador
(perfil CB, linha da matriz CF, features do supervisionado) é feita pelo
``RecommendationEngine.refresh_after_feedback``.

O que NÃO acontece aqui (fica para ``learning.retraining``): reestimação de
priors, retreino do supervisionado, refatoração ALS, reclusterização e gating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import INTERACTION_TYPES
from database.repositories import Repositories
from src.learning.bayesian_updates import apply_update, evidence_delta, resolve_events
from src.utils.logging_setup import get_logger

log = get_logger("feedback")

UI_ACTIONS = {
    "Gostei": "interest",
    "Não tenho interesse": "rejection",
    "Adicionar ao carrinho": "add_to_cart",
    "Comprar": "purchase",
    "Ignorar": "impression",
    "Clique": "click",
}


@dataclass
class FeedbackResult:
    customer_id: int
    product_id: int
    category: str
    interaction_type: str
    session_id: int
    interaction_ids: list
    previous_outcome: str | None
    new_outcome: str | None
    bayes_changes: list
    order_id: int | None = None
    feature_changes: dict = field(default_factory=dict)


class FeedbackService:
    def __init__(self, repos: Repositories):
        self.repos = repos

    def new_session(self) -> int:
        return self.repos.interactions.next_session_id()

    def register(self, customer_id: int, product_id: int, interaction_type: str, session_id: int,
                 model_name: str = "hybrid_learned", recommendation_id: int | None = None, context: dict | None = None,
                 quantity: int | None = None, timestamp: pd.Timestamp | None = None) -> FeedbackResult:
        if interaction_type not in INTERACTION_TYPES:
            raise ValueError(f"Tipo de interação inválido: {interaction_type}")
        r = self.repos
        cust = r.customers.get(customer_id)
        prod = r.products.get(product_id)
        if cust is None or prod is None:
            raise ValueError("Cliente ou produto inexistente.")
        if interaction_type == "purchase" and int(prod["stock"]) <= 0:
            raise ValueError("Produto sem estoque: compra não pode ser registrada.")
        ts = timestamp or pd.Timestamp.now()
        category = prod["category"]
        before_feats = r.features.load()
        ctx_json = json.dumps(context or {"origem": "simulador"}, ensure_ascii=False)
        order_id = None
        with r.engine.begin() as conn:
            prev = conn.execute(text("SELECT interaction_type FROM interactions WHERE session_id = :s AND product_id = :p "
                                     "AND customer_id = :c ORDER BY interaction_id"),
                                {"s": session_id, "p": product_id, "c": customer_id}).scalars().all()
            prev_outcome = resolve_events(list(prev))
            ids = []
            base = {"session_id": session_id, "customer_id": customer_id, "product_id": product_id, "model_name": model_name,
                    "context": ctx_json, "recommendation_id": recommendation_id, "source": "app"}
            if not prev:
                ids.append(r.interactions.insert({**base, "interaction_type": "impression", "timestamp": ts}, conn=conn))
            if interaction_type != "impression" or prev:
                ids.append(r.interactions.insert({**base, "interaction_type": interaction_type,
                                                  "timestamp": ts + pd.Timedelta(seconds=1)}, conn=conn))
            new_outcome = resolve_events(list(prev) + (["impression"] if not prev else []) +
                                         ([interaction_type] if (interaction_type != "impression" or prev) else []))
            delta = evidence_delta(prev_outcome, new_outcome)
            changes = apply_update(r, conn, customer_id, category, delta, ids[-1] if ids else None,
                                   reason=f"{prev_outcome or '—'} → {new_outcome}") if delta else []
            if interaction_type == "purchase":
                q = quantity or self._default_quantity(customer_id)
                order_id = r.orders.create_order(customer_id, ts + pd.Timedelta(seconds=2),
                                                 [{"product_id": product_id, "quantity": q, "unit_price": float(prod["price"]),
                                                   "discount": 0.0}], channel="portal_b2b", source="app", conn=conn)
                r.products.decrement_stock(product_id, q, conn=conn)
            r.events.log("feedback_registered", "interactions", ids[-1] if ids else None,
                         {"customer_id": customer_id, "product_id": product_id, "type": interaction_type,
                          "session_id": session_id, "previous_outcome": prev_outcome, "new_outcome": new_outcome,
                          "order_id": order_id, "bayes": changes}, conn=conn)
        feat_changes = self._update_features(customer_id, before_feats)
        log.info("Feedback %s: cliente=%s produto=%s (%s → %s)", interaction_type, customer_id, product_id, prev_outcome, new_outcome)
        return FeedbackResult(customer_id, product_id, category, interaction_type, session_id, ids, prev_outcome, new_outcome,
                              changes, order_id, feat_changes)

    def _default_quantity(self, customer_id: int) -> int:
        f = self.repos.features.load()
        if customer_id in f.index and f.loc[customer_id, "n_lines"] > 0:
            return max(1, int(round(f.loc[customer_id, "avg_units_per_order"] / max(f.loc[customer_id, "avg_lines_per_order"], 1))))
        return 6

    def _update_features(self, customer_id: int, before: pd.DataFrame) -> dict:
        """Recalcula as features comportamentais (ponto no tempo = agora) e persiste a do cliente."""
        from src.preprocessing.feature_engineering import build_customer_features
        from src.preprocessing.snapshot import Dataset

        ds = Dataset.from_db(self.repos.engine)
        snap = ds.snapshot(pd.Timestamp.now() + pd.Timedelta(seconds=5))
        feats = build_customer_features(snap)
        self.repos.features.upsert(feats.loc[[customer_id]], snap.as_of)
        changes = {}
        if customer_id in before.index:
            for c in ["n_orders", "n_distinct_products", "n_impressions", "n_engagements", "n_rejections", "engagement_rate",
                      "avg_ticket", "recency_days"]:
                b, a = before.loc[customer_id].get(c), feats.loc[customer_id, c]
                try:
                    if b is None or not np.isclose(float(b), float(a)):
                        changes[c] = {"antes": None if b is None else float(b), "depois": float(a)}
                except (TypeError, ValueError):
                    pass
        return changes
