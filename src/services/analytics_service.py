"""Indicadores agregados para o dashboard (calculados a partir do banco, com filtros)."""
from __future__ import annotations

import pandas as pd


class AnalyticsService:
    def __init__(self, rec_engine):
        self.e = rec_engine

    def filtered_items(self, start=None, end=None, categories=None, clusters=None) -> pd.DataFrame:
        it = self.e.snapshot.items
        ctx = self.e.suite.ctx
        it = it.assign(cluster=it["customer_id"].map(
            lambda c: ctx.cluster_names.get(int(ctx.cluster_labels.get(c, -1)), "Sem histórico suficiente")))
        if start is not None:
            it = it[it["order_date"] >= pd.Timestamp(start)]
        if end is not None:
            it = it[it["order_date"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        if categories:
            it = it[it["category"].isin(categories)]
        if clusters:
            it = it[it["cluster"].isin(clusters)]
        return it

    def kpis(self, items: pd.DataFrame) -> dict:
        orders = items.groupby("order_id")["total_value"].sum()
        return {"clientes": int(self.e.snapshot.n_customers), "clientes_ativos": int(items["customer_id"].nunique()),
                "produtos": int(self.e.snapshot.n_products), "pedidos": int(items["order_id"].nunique()),
                "faturamento": float(items["total_value"].sum()), "ticket_medio": float(orders.mean()) if len(orders) else 0.0,
                "unidades": int(items["quantity"].sum())}

    def monthly(self, items: pd.DataFrame) -> pd.DataFrame:
        if items.empty:
            return pd.DataFrame(columns=["month", "faturamento", "pedidos"])
        m = items.assign(month=items["order_date"].dt.to_period("M").dt.to_timestamp())
        out = m.groupby("month").agg(faturamento=("total_value", "sum"), pedidos=("order_id", "nunique")).reset_index()
        # mês corrente incompleto distorce a série (queda artificial): é marcado e excluído do gráfico
        last = items["order_date"].max()
        month_end = last.to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
        out["parcial"] = False
        if (month_end - last).days > 3:
            out.loc[out["month"] == last.to_period("M").to_timestamp(), "parcial"] = True
        return out

    def by_category(self, items: pd.DataFrame) -> pd.DataFrame:
        return items.groupby("category").agg(faturamento=("total_value", "sum"), unidades=("quantity", "sum"),
                                             pedidos=("order_id", "nunique")).sort_values("faturamento", ascending=False).reset_index()

    def cluster_distribution(self) -> pd.DataFrame:
        ctx = self.e.suite.ctx
        lab = ctx.cluster_labels.map(lambda k: ctx.cluster_names.get(int(k), "Sem histórico suficiente"))
        return lab.value_counts().rename_axis("cluster").reset_index(name="clientes")

    def customer_profile_points(self, items: pd.DataFrame) -> pd.DataFrame:
        if items.empty:
            return pd.DataFrame()
        g = items.groupby("customer_id").agg(faturamento=("total_value", "sum"), pedidos=("order_id", "nunique"),
                                             cluster=("cluster", "first"))
        g["ticket_medio"] = g["faturamento"] / g["pedidos"]
        g["name"] = self.e.snapshot.customers["name"].reindex(g.index)
        return g.reset_index()
