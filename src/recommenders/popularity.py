"""Baseline de popularidade com decaimento temporal.

score(p) = Σ_{clientes distintos c que compraram p} 0.5^{idade_última_compra(c,p)/h}
           + 0.1 · (nº de compradores distintos histórico) / max
É o mesmo para todos os clientes — serve de referência e de *fallback* no cold start.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing.transformations import exp_decay
from src.recommenders.base import BaseRecommender, ExpertContext


class PopularityRecommender(BaseRecommender):
    name = "popularity"
    label = "Popularidade"

    def __init__(self, half_life_days: float = 60.0, window_days: int = 180):
        super().__init__()
        self.half_life = half_life_days
        self.window = window_days

    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        it = snapshot.items
        last = it.groupby(["customer_id", "product_id"])["order_date"].max().reset_index()
        age = (snapshot.as_of - last["order_date"]).dt.days.to_numpy()
        recent = age <= self.window
        pop = np.zeros(snapshot.n_products)
        pj = last["product_id"].map(snapshot.pidx).to_numpy()
        np.add.at(pop, pj[recent], exp_decay(age[recent], self.half_life))
        alltime = np.bincount(pj, minlength=snapshot.n_products).astype(float)
        if alltime.max() > 0:
            pop += 0.1 * alltime / alltime.max()
        self.product_scores = pop
        self.n_recent_buyers = np.bincount(pj[recent], minlength=snapshot.n_products)
        self._scores = np.tile(pop, (snapshot.n_customers, 1))
        return self

    def confidence(self):
        return np.full(self.snapshot.n_customers, 1.0)

    def explain(self, customer_id, product_id):
        j = self._pj(product_id)
        n = int(self.n_recent_buyers[j])
        return {"summary": f"Popular na base: comprado por {n} lojista(s) nos últimos {self.window} dias.",
                "details": {"compradores_recentes": n, "score_popularidade": float(self.product_scores[j])}}
