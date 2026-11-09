"""Recursos compartilhados entre páginas (cache do motor e serviços)."""
from __future__ import annotations

import streamlit as st

from database.connection import get_engine
from src.services.analytics_service import AnalyticsService
from src.services.customer_service import CustomerService
from src.services.model_service import ModelService
from src.services.recommendation_service import RecommendationEngine


@st.cache_resource(show_spinner="Carregando modelos e estado do recomendador…")
def get_rec_engine() -> RecommendationEngine:
    return RecommendationEngine(get_engine())


def services():
    eng = get_rec_engine()
    return eng, CustomerService(eng), AnalyticsService(eng), ModelService(get_engine())


def reset_engine() -> None:
    get_rec_engine.clear()


def customer_options(eng) -> dict[str, int]:
    c = eng.snapshot.customers
    return {f"{name} · {c.loc[cid, 'city']}/{c.loc[cid, 'state']} (#{cid})": int(cid) for cid, name in c["name"].items()}


def customer_picker(eng, key: str, label: str = "Cliente") -> int:
    opts = customer_options(eng)
    labels = list(opts)
    default = st.session_state.get("selected_customer")
    idx = next((i for i, lab in enumerate(labels) if opts[lab] == default), 0)
    choice = st.selectbox(label, labels, index=idx, key=key)
    st.session_state["selected_customer"] = opts[choice]
    return opts[choice]
