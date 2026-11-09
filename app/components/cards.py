"""Cards, badges e blocos de texto da interface."""
from __future__ import annotations

import html

import streamlit as st

BADGES = {
    "observado": ("ar-badge-obs", "OBSERVADO"),
    "inferido": ("ar-badge-inf", "INFERIDO"),
    "cadastro": ("ar-badge-cad", "CADASTRO"),
}


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="ar-page-title">{html.escape(title)}</p><p class="ar-page-sub">{html.escape(subtitle)}</p>',
                unsafe_allow_html=True)


def badge(kind: str) -> str:
    cls, label = BADGES[kind]
    return f'<span class="ar-badge {cls}">{label}</span>'


def confidence_badge(level: str) -> str:
    cls = {"Alta": "ar-badge-ok", "Média": "ar-badge-warn", "Baixa": "ar-badge-low"}.get(level, "ar-badge-cad")
    icon = {"Alta": "●●●", "Média": "●●○", "Baixa": "●○○"}.get(level, "")
    return f'<span class="ar-badge {cls}">{icon} CONFIANÇA {level.upper()}</span>'


def kpi_card(label: str, value: str, foot: str = "", tooltip: str = "") -> str:
    t = f' title="{html.escape(tooltip)}"' if tooltip else ""
    return (f'<div class="ar-card"{t}><div class="ar-kpi-label">{html.escape(label)}</div>'
            f'<div class="ar-kpi-value">{html.escape(value)}</div><div class="ar-kpi-foot">{html.escape(foot)}</div></div>')


def kpi_row(cards: list[tuple]) -> None:
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        col.markdown(kpi_card(*c), unsafe_allow_html=True)


def section(title: str, kind: str | None = None, help_text: str | None = None) -> None:
    b = badge(kind) if kind else ""
    st.markdown(f'<div class="ar-section">{html.escape(title)}{b}</div>', unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def kv_card(title: str, items: list[tuple[str, str]], kind: str | None = None) -> None:
    rows = "".join(f'<div class="ar-kv"><span>{html.escape(k)}</span><span>{html.escape(v)}</span></div>' for k, v in items)
    b = badge(kind) if kind else ""
    st.markdown(f'<div class="ar-card"><div class="ar-kpi-label">{html.escape(title)}{b}</div>{rows}</div>',
                unsafe_allow_html=True)


def callout(title: str, bullets: list[str]) -> None:
    li = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
    st.markdown(f'<div class="ar-callout"><b>{html.escape(title)}</b><ul>{li}</ul></div>', unsafe_allow_html=True)


def brl(v: float, decimals: int = 0) -> str:
    s = f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def brl_compact(v: float) -> str:
    """R$ 29,8 mi / R$ 912 mil — cabe em cards estreitos."""
    if abs(v) >= 1e6:
        return "R$ " + f"{v / 1e6:,.1f}".replace(".", ",") + " mi"
    if abs(v) >= 1e4:
        return "R$ " + f"{v / 1e3:,.0f}".replace(",", ".") + " mil"
    return brl(v)


def num(v: float, decimals: int = 0) -> str:
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, decimals: int = 0) -> str:
    return f"{v * 100:.{decimals}f}%".replace(".", ",")
