"""Adaptive Recommender — aplicação Streamlit (ponto de entrada).

Executar:  streamlit run app/main.py   (ou run.bat)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Adaptive Recommender · B2B", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

from app.components.theme import inject_css  # noqa: E402
from database.connection import create_schema, get_engine  # noqa: E402
from database.init_db import is_populated  # noqa: E402
from src.services.model_service import ModelService  # noqa: E402

inject_css()

with st.sidebar:
    st.markdown('<div class="ar-brand">◆ Adaptive Recommender</div>'
                '<div class="ar-brand-sub">Inteligência comercial B2B · Distribuidora de moda</div>', unsafe_allow_html=True)
    with st.expander("Sobre este projeto"):
        st.caption("Projeto de fim de semana, feito por curiosidade: testar na prática vários modelos de "
                   "recomendação, brincar com um motor adaptativo e aprender coisas novas. Alguns modelos são mais "
                   "complexos do que o problema exigiria — de propósito, para explorar. Os dados da aplicação são "
                   "sintéticos; há também um experimento à parte com dados reais (Online Retail II, UCI).")

engine = get_engine()
create_schema(engine)
ready = is_populated(engine) and ModelService(engine).is_trained()

if not ready:
    st.title("Configuração inicial")
    st.write("O banco ainda não possui dados ou modelos treinados. A inicialização gera os dados sintéticos, "
             "carrega o banco e treina todos os modelos (≈ 1–2 minutos).")
    if st.button("Inicializar sistema", type="primary"):
        from database.init_db import load_synthetic
        from src.data_generation.pipeline import generate_all
        from src.learning.retraining import retrain_all

        with st.status("Inicializando…", expanded=True) as status:
            if not is_populated(engine):
                st.write("Gerando dados sintéticos…")
                ds = generate_all(save=True)
                st.write("Carregando banco…")
                load_synthetic(engine)
            st.write("Treinando modelos…")
            retrain_all(engine)
            status.update(label="Sistema pronto", state="complete")
        st.rerun()
    st.stop()

pages = [
    st.Page("views/dashboard.py", title="Dashboard Geral", icon=":material/space_dashboard:", default=True),
    st.Page("views/clientes.py", title="Inteligência dos Clientes", icon=":material/storefront:"),
    st.Page("views/recomendacoes.py", title="Central de Recomendações", icon=":material/auto_awesome:"),
    st.Page("views/simulador.py", title="Simulador de Interações", icon=":material/touch_app:"),
    st.Page("views/avaliacao.py", title="Avaliação dos Modelos", icon=":material/science:"),
]
nav = st.navigation(pages)

with st.sidebar:
    from app.components.state import get_rec_engine

    eng = get_rec_engine()
    v = eng.model_versions()
    st.caption(f"Estado em {eng.as_of:%d/%m/%Y %H:%M}")
    st.caption(f"Supervisionado v{v.get('supervised', '—')} ({eng.supervised.algorithm}) · Gating v{v.get('gating', '—')} · "
               f"Clusters v{v.get('clustering', '—')}")

nav.run()
