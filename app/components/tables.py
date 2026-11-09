"""Tabelas padronizadas (st.dataframe com column_config)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import EXPERT_LABELS


def recommendations_table(items: pd.DataFrame, height: int | str = "auto") -> None:
    df = items.copy()
    df["principal_especialista"] = df["contributions"].map(lambda c: EXPERT_LABELS[max(c, key=c.get)])
    cols = ["rank", "name", "category", "price", "stock", "relevance", "conversion_probability", "principal_especialista"]
    st.dataframe(
        df[cols], hide_index=True, width="stretch", height=height,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "name": st.column_config.TextColumn("Produto", width="large"),
            "category": st.column_config.TextColumn("Categoria"),
            "price": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
            "stock": st.column_config.NumberColumn("Estoque", format="%d"),
            "relevance": st.column_config.ProgressColumn("Relevância", min_value=0.0, max_value=1.0, format="%.2f",
                                                         help="Score híbrido combinado (0–1), sem margem comercial."),
            "conversion_probability": st.column_config.ProgressColumn(
                "P(compra 92d)", min_value=0.0, max_value=1.0, format="percent",
                help="Probabilidade calibrada do modelo supervisionado de o cliente comprar o produto em 92 dias."),
            "principal_especialista": st.column_config.TextColumn("Maior contribuição"),
        },
    )


def directory_table(df: pd.DataFrame, key: str):
    return st.dataframe(
        df, hide_index=True, width="stretch", height=360, key=key, on_select="rerun", selection_mode="single-row",
        column_config={
            "customer_id": st.column_config.NumberColumn("ID", width="small"),
            "name": st.column_config.TextColumn("Lojista", width="medium"),
            "city": "Cidade", "state": st.column_config.TextColumn("UF", width="small"),
            "store_size": "Porte", "business_segment": "Segmento", "cluster": "Cluster (inferido)",
            "n_orders": st.column_config.NumberColumn("Pedidos"),
            "total_revenue": st.column_config.NumberColumn("Faturamento", format="R$ %.0f"),
            "avg_ticket": st.column_config.NumberColumn("Ticket médio", format="R$ %.0f"),
            "recency_days": st.column_config.NumberColumn("Recência (dias)", format="%.0f"),
            "history_segment": "Histórico",
        },
    )
