"""Consultas de inteligência de clientes (perfil observado × inferido)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CATEGORIES
from src.preprocessing.feature_engineering import SHARE_COLS, segment_by_history


class CustomerService:
    def __init__(self, rec_engine):
        self.e = rec_engine

    @property
    def features(self) -> pd.DataFrame:
        return self.e.suite.ctx.customer_features

    def directory(self) -> pd.DataFrame:
        s = self.e.snapshot
        c = s.customers.reset_index()
        f = self.features
        ctx = self.e.suite.ctx
        df = c[["customer_id", "name", "city", "state", "store_size", "business_segment"]].copy()
        df["cluster"] = df["customer_id"].map(lambda x: ctx.cluster_names.get(int(ctx.cluster_labels.get(x, -1)), "Sem histórico suficiente"))
        df["n_orders"] = df["customer_id"].map(f["n_orders"]).astype(int)
        df["total_revenue"] = df["customer_id"].map(f["total_revenue"])
        df["avg_ticket"] = df["customer_id"].map(f["avg_ticket"])
        df["recency_days"] = df["customer_id"].map(f["recency_days"])
        df["history_segment"] = df["customer_id"].map(segment_by_history(f))
        return df

    def cadastral(self, customer_id: int) -> dict:
        return self.e.snapshot.customers.loc[int(customer_id)].to_dict()

    def behavioral(self, customer_id: int) -> dict:
        return self.features.loc[int(customer_id)].to_dict()

    def history(self, customer_id: int) -> pd.DataFrame:
        it = self.e.snapshot.items
        d = it[it["customer_id"] == int(customer_id)].copy()
        d = d.merge(self.e.snapshot.products[["name"]], left_on="product_id", right_index=True, how="left")
        return d.sort_values("order_date", ascending=False)

    def orders_timeline(self, customer_id: int) -> pd.DataFrame:
        d = self.history(customer_id)
        if d.empty:
            return pd.DataFrame(columns=["order_date", "value", "units", "ticket_rolling"])
        o = d.groupby("order_id").agg(order_date=("order_date", "first"), value=("total_value", "sum"),
                                      units=("quantity", "sum")).sort_values("order_date")
        o["ticket_rolling"] = o["value"].rolling(5, min_periods=1).mean()
        o["days_between"] = o["order_date"].diff().dt.days
        return o.reset_index()

    def category_shares(self, customer_id: int) -> pd.DataFrame:
        f = self.features.loc[int(customer_id)]
        return pd.DataFrame({"category": CATEGORIES, "share": [float(f[c]) for c in SHARE_COLS]})

    def similar_customers(self, customer_id: int, n: int = 5) -> pd.DataFrame:
        cf = self.e.suite.experts["collaborative"]
        sims = cf.similar_customers(customer_id, n)
        names = self.e.snapshot.customers["name"]
        ctx = self.e.suite.ctx
        return pd.DataFrame([{"customer_id": c, "name": names.loc[c], "similarity": s,
                              "cluster": ctx.cluster_names.get(int(ctx.cluster_labels.get(c, -1)), "—")} for c, s in sims])

    def bayes_summary(self, customer_id: int) -> pd.DataFrame:
        return self.e.suite.experts["bayesian"].customer_summary(int(customer_id))

    def bayes_trajectory(self, customer_id: int, category: str) -> pd.DataFrame:
        return self.e.suite.experts["bayesian"].trajectory(int(customer_id), category)

    def cluster_info(self, customer_id: int) -> dict:
        ctx = self.e.suite.ctx
        k = int(ctx.cluster_labels.get(int(customer_id), -1))
        seg = self.e.segmentation
        prof = None
        if seg is not None and k >= 0:
            p = seg.profiles[seg.profiles["cluster_id"] == k]
            prof = p.iloc[0].to_dict() if len(p) else None
        return {"cluster_id": k, "label": ctx.cluster_names.get(k, "Sem histórico suficiente"), "profile": prof}

    def price_sensitivity_label(self, customer_id: int) -> str:
        pi = float(self.features.loc[int(customer_id), "price_index"])
        if self.features.loc[int(customer_id), "n_orders"] == 0:
            return "Indeterminada (sem histórico)"
        return "Alta" if pi < -0.35 else "Moderada" if pi < -0.1 else "Baixa"

    def neighbors_matrix(self) -> np.ndarray:
        return self.e.suite.experts["collaborative"].S
