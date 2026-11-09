"""Identidade visual: tokens de cor (paleta validada), CSS global e cores fixas por especialista."""
from __future__ import annotations

import streamlit as st

# Paleta categórica de referência (ordem fixa, validada para CVD em pares adjacentes)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e8e7e2"
SURFACE = "#ffffff"
NEUTRAL = "#c3c2b7"

# A cor acompanha a entidade (especialista), nunca a posição no ranking
EXPERT_COLORS = {
    "content_based": SERIES[0],
    "collaborative": SERIES[1],
    "supervised": SERIES[2],
    "bayesian": SERIES[3],
    "popularity": SERIES[4],
    "collaborative_mf": SERIES[5],
    "hybrid_learned": SERIES[6],
    "hybrid_rules": "#9085e9",
    "hybrid_fixed": "#52514e",
    "hybrid_equal": "#8a8984",
    "random": "#c3c2b7",
}

CSS = """
<style>
:root { --ar-border:#e6e5e0; --ar-text-2:#52514e; --ar-muted:#8a8984; --ar-accent:#2a78d6; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #10213a; }
[data-testid="stSidebar"] * { color: #e9eef6 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(255,255,255,0.12); border-radius: 8px; }
.ar-brand { font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; margin: 0.2rem 0 0.1rem 0; }
.ar-brand-sub { font-size: 0.78rem; opacity: 0.75; margin-bottom: 0.6rem; }
.ar-page-title { font-size: 1.65rem; font-weight: 700; margin: 0; color: #0b0b0b; }
.ar-page-sub { color: var(--ar-text-2); margin: 0.15rem 0 1.1rem 0; font-size: 0.95rem; }
.ar-card { background:#fff; border:1px solid var(--ar-border); border-radius: 12px; padding: 14px 16px; height: 100%;
           box-shadow: 0 1px 2px rgba(16,33,58,0.04); }
.ar-kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ar-text-2); font-weight: 600; }
.ar-kpi-value { font-size: 1.6rem; font-weight: 700; color: #0b0b0b; margin-top: 2px; line-height: 1.2; }
.ar-kpi-foot { font-size: 0.78rem; color: var(--ar-muted); margin-top: 2px; }
.ar-section { font-size: 1.02rem; font-weight: 650; margin: 1.1rem 0 0.4rem 0; color: #0b0b0b; }
.ar-badge { display:inline-block; font-size: 0.68rem; font-weight: 650; padding: 2px 8px; border-radius: 999px;
            letter-spacing: 0.03em; vertical-align: middle; margin-left: 6px; }
.ar-badge-obs { background:#e7f0fb; color:#1c5cab; }
.ar-badge-inf { background:#efedfb; color:#4a3aa7; }
.ar-badge-cad { background:#efefec; color:#52514e; }
.ar-badge-ok { background:#e5f4e5; color:#006300; }
.ar-badge-warn { background:#fdf1d8; color:#7a5200; }
.ar-badge-low { background:#fbe7e7; color:#9c2525; }
.ar-kv { display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px dashed #efeee9; font-size: 0.88rem; }
.ar-kv span:first-child { color: var(--ar-text-2); }
.ar-kv span:last-child { font-weight: 600; color:#0b0b0b; text-align:right; }
.ar-callout { background:#f3f7fd; border:1px solid #d6e4f7; border-radius: 10px; padding: 12px 14px; font-size: 0.9rem; color:#1f2a3a; }
.ar-callout ul { margin: 0.3rem 0 0 1.1rem; padding: 0; }
.ar-reason { font-size: 0.86rem; color:#1f2a3a; margin: 3px 0; }
.ar-legend-dot { display:inline-block; width:10px; height:10px; border-radius: 3px; margin-right:6px; vertical-align: middle; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--ar-border); border-radius: 12px; padding: 10px 14px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
