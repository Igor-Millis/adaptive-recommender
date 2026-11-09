"""Barra de filtros (uma linha acima dos gráficos)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import CATEGORIES


def dashboard_filters(min_date: pd.Timestamp, max_date: pd.Timestamp, clusters: list[str]) -> dict:
    c1, c2, c3 = st.columns([1.3, 1.5, 1.5])
    with c1:
        period = st.date_input("Período", value=(min_date.date(), max_date.date()), min_value=min_date.date(),
                               max_value=max_date.date(), format="DD/MM/YYYY")
    with c2:
        cats = st.multiselect("Categorias", CATEGORIES, placeholder="Todas as categorias")
    with c3:
        segs = st.multiselect("Segmento (cluster)", clusters, placeholder="Todos os segmentos")
    start, end = (period if isinstance(period, tuple) and len(period) == 2 else (min_date.date(), max_date.date()))
    return {"start": start, "end": end, "categories": cats, "clusters": segs}
