"""Gráficos Plotly com estilo consistente (marcas finas, grade recessiva, um eixo por gráfico)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from app.components.theme import EXPERT_COLORS, GRID, NEUTRAL, SEQ_BLUE, SERIES, TEXT_PRIMARY, TEXT_SECONDARY
from config.settings import EXPERT_LABELS


def style(fig: go.Figure, height: int = 320, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=40 if title else 12, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff", font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=TEXT_SECONDARY),
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY), x=0, xanchor="left") if title else None,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=TEXT_PRIMARY, size=12)),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        bargap=0.28,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True)
    return fig


def line(df: pd.DataFrame, x: str, y: str, name: str = "", color: str = SERIES[0], hover_fmt: str = ",.0f",
         height: int = 300, yprefix: str = "", title: str | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=name, line=dict(color=color, width=2),
                               marker=dict(size=7, color=color, line=dict(color="#fff", width=2)),
                               hovertemplate=f"%{{x|%b/%Y}}<br>{name}: {yprefix}%{{y:{hover_fmt}}}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix=yprefix)
    return style(fig, height, legend=False, title=title)


def hbar(labels, values, color: str = SERIES[0], fmt: str = ",.0f", prefix: str = "", height: int | None = None,
         title: str | None = None, text: list | None = None, highlight: set | None = None) -> go.Figure:
    labels, values = list(labels), list(values)
    colors = [color if (highlight is None or lab in highlight) else NEUTRAL for lab in labels]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, line=dict(color="#fff", width=2),
                                                                                cornerradius=4),
                           text=text if text is not None else [f"{prefix}{v:{fmt}}" for v in values], textposition="outside",
                           textfont=dict(color=TEXT_SECONDARY, size=11), cliponaxis=False,
                           hovertemplate=f"%{{y}}: {prefix}%{{x:{fmt}}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    vmax = max(values) if values else 1
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, vmax * 1.28])
    return style(fig, height or max(180, 34 * len(labels) + 40), legend=False, title=title)


def weights_bar(weights: dict, height: int = 120, title: str | None = None) -> go.Figure:
    """Barra única empilhada com o peso de cada especialista (soma = 100%)."""
    fig = go.Figure()
    for e, w in weights.items():
        fig.add_trace(go.Bar(x=[w], y=[""], orientation="h", name=EXPERT_LABELS.get(e, e),
                             marker=dict(color=EXPERT_COLORS.get(e, NEUTRAL), line=dict(color="#fff", width=2)),
                             text=[f"{w:.0%}" if w >= 0.06 else ""], textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#fff", size=12),
                             hovertemplate=f"{EXPERT_LABELS.get(e, e)}: %{{x:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return style(fig, height, legend=True, title=title)


def weights_compare(before: dict, after: dict, height: int = 260) -> go.Figure:
    experts = list(before)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[EXPERT_LABELS[e] for e in experts], x=[before[e] for e in experts], orientation="h", name="Antes",
                         marker=dict(color=NEUTRAL, line=dict(color="#fff", width=2), cornerradius=4),
                         hovertemplate="%{y} — antes: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(y=[EXPERT_LABELS[e] for e in experts], x=[after[e] for e in experts], orientation="h", name="Depois",
                         marker=dict(color=SERIES[0], line=dict(color="#fff", width=2), cornerradius=4),
                         hovertemplate="%{y} — depois: %{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return style(fig, height)


def contributions_stacked(items: pd.DataFrame, experts: list[str], height: int | None = None) -> go.Figure:
    """Contribuição relativa de cada especialista para o score de cada produto recomendado."""
    labels = [f"#{r} {n[:38]}" for r, n in zip(items["rank"], items["name"])]
    fig = go.Figure()
    for e in experts:
        vals = [c.get(e, 0.0) for c in items["contributions"]]
        fig.add_trace(go.Bar(y=labels, x=vals, orientation="h", name=EXPERT_LABELS[e],
                             marker=dict(color=EXPERT_COLORS[e], line=dict(color="#fff", width=2)),
                             hovertemplate=f"%{{y}}<br>{EXPERT_LABELS[e]}: %{{x:.0%}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return style(fig, height or 60 + 28 * len(items))


def interval_chart(df: pd.DataFrame, label_col: str, mean_col: str, lo_col: str, hi_col: str, color: str = SERIES[6],
                   prior_col: str | None = None, height: int | None = None, xfmt: str = ".0%", title: str | None = None) -> go.Figure:
    """Estimativa pontual + intervalo de credibilidade (incerteza visível)."""
    d = df.sort_values(mean_col, ascending=True)
    fig = go.Figure()
    if prior_col:
        fig.add_trace(go.Scatter(x=d[prior_col], y=d[label_col], mode="markers", name="Prior (segmento)",
                                 marker=dict(symbol="line-ns", size=14, color=TEXT_SECONDARY, line=dict(width=2, color=TEXT_SECONDARY)),
                                 hovertemplate="%{y}<br>Prior: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=d[mean_col], y=d[label_col], mode="markers", name="Média posterior (IC 90%)",
        marker=dict(size=10, color=color, line=dict(color="#fff", width=2)),
        error_x=dict(type="data", symmetric=False, array=d[hi_col] - d[mean_col], arrayminus=d[mean_col] - d[lo_col],
                     color=color, thickness=2, width=0),
        customdata=np.c_[d[lo_col], d[hi_col]],
        hovertemplate="%{y}<br>Média: %{x:.1%}<br>IC 90%: %{customdata[0]:.1%} – %{customdata[1]:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=xfmt)
    return style(fig, height or 60 + 34 * len(d), legend=True, title=title)


def beta_densities(curves: list[tuple[str, float, float, str]], height: int = 280, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    x = np.linspace(0.0005, 0.9995, 600)
    for name, a, b, color in curves:
        fig.add_trace(go.Scatter(x=x, y=stats.beta.pdf(x, a, b), mode="lines", name=f"{name} (α={a:.1f}, β={b:.1f})",
                                 line=dict(color=color, width=2), hovertemplate=f"{name}<br>θ=%{{x:.2f}}<br>densidade=%{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="θ (probabilidade)", tickformat=".0%")
    fig.update_yaxes(title="densidade")
    return style(fig, height, title=title)


def trajectory_chart(traj: pd.DataFrame, stage: str = "interest", color: str = SERIES[6], height: int = 300,
                     title: str | None = None) -> go.Figure:
    t = traj.reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(t["step"]) + list(t["step"][::-1]), y=list(t[f"{stage}_high"]) + list(t[f"{stage}_low"][::-1]),
                             fill="toself", fillcolor="rgba(98,80,214,0.13)", line=dict(width=0), name="IC 90%",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["step"], y=t[f"{stage}_mean"], mode="lines", name="Média posterior",
                             line=dict(color=color, width=2), customdata=t["outcome"],
                             hovertemplate="Evento %{x}: %{customdata}<br>Média: %{y:.1%}<extra></extra>"))
    fig.update_xaxes(title="nº de eventos observados")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return style(fig, height, title=title)


def metric_bars_ci(df: pd.DataFrame, value: str = "value", lo: str = "ci_low", hi: str = "ci_high", height: int | None = None,
                   fmt: str = ".3f", title: str | None = None) -> go.Figure:
    d = df.sort_values(value, ascending=False)
    colors = [EXPERT_COLORS.get(m, NEUTRAL) for m in d["model_name"]]
    err = None
    if lo in d and d[lo].notna().all():
        err = dict(type="data", symmetric=False, array=d[hi] - d[value], arrayminus=d[value] - d[lo], color=TEXT_SECONDARY,
                   thickness=1.5, width=4)
    fig = go.Figure(go.Bar(y=[EXPERT_LABELS.get(m, m) for m in d["model_name"]], x=d[value], orientation="h",
                           marker=dict(color=colors, line=dict(color="#fff", width=2), cornerradius=4), error_x=err,
                           text=[f"{v:{fmt}}" for v in d[value]], textposition="inside", insidetextanchor="end",
                           textfont=dict(color="#fff", size=11),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return style(fig, height or 60 + 32 * len(d), legend=False, title=title)


def heatmap(df: pd.DataFrame, fmt: str = ".3f", height: int | None = None, title: str | None = None, zmin=None, zmax=None) -> go.Figure:
    z = df.to_numpy(float)
    fig = go.Figure(go.Heatmap(z=z, x=list(df.columns), y=list(df.index), colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
                               zmin=zmin, zmax=zmax, xgap=2, ygap=2, text=[[f"{v:{fmt}}" if np.isfinite(v) else "—" for v in row] for row in z],
                               texttemplate="%{text}", textfont=dict(size=11),
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>", colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return style(fig, height or 80 + 30 * len(df), legend=False, title=title)


def curve_chart(curves: list[tuple[str, list, list, str]], xtitle: str, ytitle: str, diagonal: bool = False, height: int = 320,
                title: str | None = None, fmt: str = ".2f") -> go.Figure:
    fig = go.Figure()
    if diagonal:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Referência", line=dict(color=NEUTRAL, width=1.5, dash="dot"),
                                 hoverinfo="skip"))
    for name, x, y, color in curves:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers" if len(x) <= 15 else "lines", name=name,
                                 line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="#fff", width=2)),
                                 hovertemplate=f"{name}<br>{xtitle}: %{{x:{fmt}}}<br>{ytitle}: %{{y:{fmt}}}<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    return style(fig, height, title=title)


def scatter_highlight(df: pd.DataFrame, x: str, y: str, size: str, hover: str, group: str, highlight: str | None,
                      height: int = 340, xtitle: str = "", ytitle: str = "", title: str | None = None) -> go.Figure:
    fig = go.Figure()
    s = df[size].to_numpy(float)
    sz = 8 + 22 * (s - s.min()) / (np.ptp(s) or 1)
    is_h = (df[group] == highlight) if highlight else pd.Series(True, index=df.index)
    for flag, color, name in [(False, NEUTRAL, "Demais clientes"), (True, SERIES[0], highlight or "Clientes")]:
        m = is_h == flag
        if not m.any():
            continue
        fig.add_trace(go.Scatter(x=df.loc[m, x], y=df.loc[m, y], mode="markers", name=name,
                                 marker=dict(size=sz[m.to_numpy()], color=color, opacity=0.85, line=dict(color="#fff", width=2)),
                                 customdata=np.c_[df.loc[m, hover], df.loc[m, group]],
                                 hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>" + f"{xtitle}: %{{x:,.0f}}<br>{ytitle}: %{{y:,.1f}}<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    return style(fig, height, legend=highlight is not None, title=title)
