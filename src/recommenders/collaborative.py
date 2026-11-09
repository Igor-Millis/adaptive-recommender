"""Filtragem colaborativa baseada em usuários (user-kNN) com feedback implícito.

Matriz cliente-produto (implícita):
    R[c, p] = Σ_compras log(1+q)·0.5^{idade/h} + w_e · Σ_engajamentos 0.5^{idade/h}

Similaridade: cosseno com *shrinkage* pelo nº de itens em comum (Koren, 2008):
    s'(c, v) = cos(R_c, R_v) · n_cv / (n_cv + λ)
Predição (k vizinhos mais similares, R normalizada por linha para [0, 1]):
    r̂(c, p) = Σ_{v∈N_k(c)} s'(c,v)·R̃[v,p] / Σ_{v∈N_k(c)} s'(c,v)

Clientes sem histórico têm linha nula → score 0 e confiança 0 (o gating zera o peso).
"""
from __future__ import annotations

import numpy as np

from config.settings import CF
from src.preprocessing.feature_engineering import POSITIVE_OUTCOMES
from src.preprocessing.transformations import exp_decay
from src.recommenders.base import BaseRecommender, ExpertContext


def implicit_matrix(snapshot, outcomes, half_life: float, engagement_weight: float) -> np.ndarray:
    R = np.zeros((snapshot.n_customers, snapshot.n_products))
    it = snapshot.items
    if len(it):
        age = (snapshot.as_of - it["order_date"]).dt.days.to_numpy()
        w = np.log1p(it["quantity"].to_numpy(float)) * exp_decay(age, half_life)
        np.add.at(R, (it["customer_id"].map(snapshot.cidx).to_numpy(), it["product_id"].map(snapshot.pidx).to_numpy()), w)
    if len(outcomes) and engagement_weight > 0:
        pos = outcomes[outcomes["outcome"].isin(POSITIVE_OUTCOMES - {"purchase"})]
        if len(pos):
            age = (snapshot.as_of - pos["timestamp"]).dt.days.to_numpy()
            np.add.at(R, (pos["customer_id"].map(snapshot.cidx).to_numpy(), pos["product_id"].map(snapshot.pidx).to_numpy()),
                      engagement_weight * exp_decay(age, half_life))
    return R


class UserKNNRecommender(BaseRecommender):
    name = "collaborative"
    label = "Collaborative Filtering"

    def __init__(self, config=CF):
        super().__init__()
        self.cfg = config

    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        self.R = implicit_matrix(snapshot, ctx.outcomes, self.cfg.half_life_days, self.cfg.engagement_weight)
        self._similarities()
        self._predict_all()
        return self

    def _similarities(self) -> None:
        R = self.R
        norms = np.linalg.norm(R, axis=1)
        denom = np.outer(norms, norms)
        S = np.divide(R @ R.T, denom, out=np.zeros((len(R), len(R))), where=denom > 0)
        B = (R > 0).astype(float)
        common = B @ B.T
        S = S * common / (common + self.cfg.shrinkage)
        np.fill_diagonal(S, 0.0)
        self.S = S
        rmax = R.max(axis=1, keepdims=True)
        self.Rn = np.divide(R, rmax, out=np.zeros_like(R), where=rmax > 0)

    def _neighbors(self, i: int) -> np.ndarray:
        s = self.S[i]
        idx = np.argsort(-s)[: self.cfg.n_neighbors]
        return idx[s[idx] > 0]

    def _predict_row(self, i: int) -> np.ndarray:
        nb = self._neighbors(i)
        if len(nb) == 0:
            return np.zeros(self.R.shape[1])
        w = self.S[i, nb]
        return w @ self.Rn[nb] / w.sum()

    def _predict_all(self) -> None:
        self._scores = np.vstack([self._predict_row(i) for i in range(len(self.R))])

    def update_customer(self, snapshot, ctx: ExpertContext, customer_id: int) -> None:
        """Atualização incremental da linha do cliente na matriz e das similaridades afetadas."""
        self.snapshot, self.ctx = snapshot, ctx
        i = snapshot.cidx[int(customer_id)]
        R_new = implicit_matrix(snapshot, ctx.outcomes, self.cfg.half_life_days, self.cfg.engagement_weight)
        self.R[i] = R_new[i]
        self._similarities()
        affected = set([i]) | set(np.where(self.S[:, i] > 0)[0].tolist())
        for a in affected:
            self._scores[a] = self._predict_row(a)

    def confidence(self):
        conf = np.zeros(len(self.R))
        for i in range(len(self.R)):
            nb = self._neighbors(i)
            if len(nb):
                conf[i] = float(np.mean(self.S[i, nb]))
        return np.clip(conf, 0, 1)

    def similar_customers(self, customer_id: int, n: int = 5) -> list[tuple[int, float]]:
        i = self._ci(customer_id)
        idx = np.argsort(-self.S[i])[:n]
        return [(int(self.snapshot.customer_ids[k]), float(self.S[i, k])) for k in idx if self.S[i, k] > 0]

    def explain(self, customer_id, product_id):
        i, j = self._ci(customer_id), self._pj(product_id)
        nb = self._neighbors(i)
        buyers = [(int(self.snapshot.customer_ids[v]), float(self.S[i, v])) for v in nb if self.R[v, j] > 0]
        names = self.snapshot.customers["name"]
        if buyers:
            txt = ", ".join(f"{names.loc[c]} (sim. {s:.2f})" for c, s in buyers[:3])
            summary = f"{len(buyers)} de {len(nb)} lojistas com padrão de compra semelhante compraram este produto: {txt}."
        else:
            summary = "Nenhum vizinho próximo comprou este produto (sinal colaborativo fraco)."
        return {"summary": summary, "details": {"score": float(self._scores[i, j]), "vizinhos_compradores": buyers,
                                                "n_vizinhos": int(len(nb))}}
