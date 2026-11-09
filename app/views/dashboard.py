"""Página 1 — Dashboard Geral."""
from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.cards import brl, brl_compact, kpi_row, num, page_header, section
from app.components.filters import dashboard_filters
from app.components.state import services
from app.components.theme import SERIES

eng, cust_svc, analytics, _ = services()
page_header("Dashboard Geral", "Visão consolidada da operação comercial, calculada diretamente dos pedidos registrados.")

items_all = eng.snapshot.items
clusters = sorted(analytics.cluster_distribution()["cluster"].tolist())
flt = dashboard_filters(items_all["order_date"].min(), items_all["order_date"].max(), clusters)
items = analytics.filtered_items(flt["start"], flt["end"], flt["categories"], flt["clusters"])
k = analytics.kpis(items)

kpi_row([
    ("Clientes", num(k["clientes"]), f"{k['clientes_ativos']} com compras no filtro", "Lojistas cadastrados na base"),
    ("Produtos", num(k["produtos"]), f"{int(eng.snapshot.in_stock.sum())} com estoque", "Itens do catálogo"),
    ("Pedidos", num(k["pedidos"]), f"{num(k['unidades'])} unidades", "Pedidos no período filtrado"),
    ("Faturamento", brl_compact(k["faturamento"]), f"{brl(k['faturamento'])} líquido", "Soma dos itens de pedido"),
    ("Ticket médio", brl(k["ticket_medio"]), "por pedido", "Faturamento / nº de pedidos"),
])

if items.empty:
    st.info("Nenhum pedido encontrado para os filtros selecionados.")
    st.stop()

c1, c2 = st.columns([1.6, 1])
with c1:
    section("Evolução mensal do faturamento")
    m = analytics.monthly(items)
    st.plotly_chart(charts.line(m[~m["parcial"]], "month", "faturamento", "Faturamento", yprefix="R$ "), width="stretch")
    if m["parcial"].any():
        st.caption(f"Mês corrente ({m.loc[m['parcial'], 'month'].iloc[0]:%m/%Y}) ainda incompleto — omitido da série.")
with c2:
    section("Categorias mais vendidas", help_text="Faturamento por categoria no período")
    cat = analytics.by_category(items)
    st.plotly_chart(charts.hbar(cat["category"], cat["faturamento"], prefix="R$ ", fmt=",.0f",
                                text=[brl(v / 1000, 0) + " mil" for v in cat["faturamento"]]), width="stretch")

c3, c4 = st.columns([1, 1.6])
with c3:
    section("Clientes por segmento", "inferido", "Segmentos descobertos pela clusterização (K-Means)")
    dist = analytics.cluster_distribution()
    hl = set(flt["clusters"]) if flt["clusters"] else None
    st.plotly_chart(charts.hbar(dist["cluster"], dist["clientes"], color=SERIES[0], fmt=",.0f", highlight=hl), width="stretch")
with c4:
    section("Distribuição do perfil de consumo", help_text="Cada ponto é um lojista; tamanho = faturamento no período")
    pts = analytics.customer_profile_points(items)
    hl_opt = st.selectbox("Destacar segmento", ["(nenhum)"] + clusters, key="dash_hl", label_visibility="collapsed")
    st.plotly_chart(charts.scatter_highlight(pts, "ticket_medio", "pedidos", "faturamento", "name", "cluster",
                                             None if hl_opt == "(nenhum)" else hl_opt, xtitle="Ticket médio (R$)",
                                             ytitle="Pedidos no período"), width="stretch")

with st.expander("Ver dados agregados (tabela)"):
    st.dataframe(analytics.by_category(items), hide_index=True, width="stretch",
                 column_config={"faturamento": st.column_config.NumberColumn("Faturamento", format="R$ %.0f")})
