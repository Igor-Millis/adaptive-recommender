"""Página 2 — Inteligência dos Clientes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.cards import brl, kv_card, num, page_header, pct, section
from app.components.state import services
from app.components.tables import directory_table
from app.components.theme import SERIES
from config.settings import CATEGORIES

eng, cust_svc, _, _ = services()
page_header("Inteligência dos Clientes", "Perfil cadastral, comportamento observado e preferências inferidas de cada lojista.")

directory = cust_svc.directory()
q = st.text_input("Buscar lojista", placeholder="Nome, cidade ou segmento…", label_visibility="collapsed")
view = directory
if q:
    ql = q.lower()
    view = directory[directory.apply(lambda r: ql in f"{r['name']} {r['city']} {r['business_segment']} {r['cluster']}".lower(), axis=1)]
sel = directory_table(view, key="dir_table")
rows = sel.selection.rows if sel and sel.selection else []
if rows:
    st.session_state["selected_customer"] = int(view.iloc[rows[0]]["customer_id"])
cid = st.session_state.get("selected_customer", int(directory["customer_id"].iloc[0]))
if cid not in directory["customer_id"].values:
    cid = int(directory["customer_id"].iloc[0])
st.caption("Selecione uma linha da tabela para abrir o perfil do lojista.")

cad = cust_svc.cadastral(cid)
beh = cust_svc.behavioral(cid)
clu = cust_svc.cluster_info(cid)
st.markdown(f"### {cad['name']}")
st.caption(f"#{cid} · {cad['city']}/{cad['state']} · {cad['business_segment']} · Segmento inferido: **{clu['label']}**")

c1, c2, c3 = st.columns(3)
with c1:
    kv_card("Dados cadastrais", [
        ("Cadastro", pd.Timestamp(cad["registration_date"]).strftime("%d/%m/%Y")), ("Porte", cad["store_size"]),
        ("Faixa de faturamento", cad["revenue_band"]),
        ("Categorias declaradas", (cad.get("declared_categories") or "—").replace(";", ", ")),
        ("Faixa de preço declarada", cad.get("declared_price_band") or "—"),
        ("Frequência declarada", cad.get("declared_purchase_frequency") or "—")], kind="cadastro")
with c2:
    no_hist = beh["n_orders"] == 0
    kv_card("Comportamento observado", [
        ("Pedidos", num(beh["n_orders"])), ("Faturamento total", brl(beh["total_revenue"])),
        ("Ticket médio", brl(beh["avg_ticket"])), ("Itens (unid.) por pedido", num(beh["avg_units_per_order"], 1)),
        ("Frequência", f"{num(beh['frequency_per_month'], 2)} pedidos/mês"),
        ("Recência", "—" if no_hist else f"{num(beh['recency_days'])} dias"),
        ("Recomendações exibidas", num(beh["n_impressions"]))], kind="observado")
with c3:
    kv_card("Características inferidas", [
        ("Segmento (cluster)", clu["label"]), ("Sensibilidade a preço", cust_svc.price_sensitivity_label(cid)),
        ("Índice de preço pago", f"{beh['price_index']:+.2f} (log vs. mediana)"),
        ("Diversidade de categorias", f"{beh['category_entropy']:.2f}"),
        ("Sazonalidade", f"{beh['seasonality_index']:.2f}" + (f" (pico mês {int(beh['seasonal_peak_month'])})" if beh["seasonality_index"] > 0.06 else "")),
        ("Mudança de preferência (JS)", f"{beh['drift_js']:.2f}"),
        ("Taxa de engajamento", pct(beh["engagement_rate"], 1))], kind="inferido")

tab1, tab2, tab3, tab4 = st.tabs(["Histórico de compras", "Preferências e incerteza", "Segmento e similares", "Evolução bayesiana"])

with tab1:
    tl = cust_svc.orders_timeline(cid)
    if tl.empty:
        st.info("Cliente sem pedidos registrados (cold start). As recomendações usam o cadastro, a popularidade e o modelo supervisionado.")
    else:
        a, b = st.columns(2)
        with a:
            section("Evolução do ticket médio", "observado", "Valor do pedido e média móvel de 5 pedidos")
            st.plotly_chart(charts.line(tl, "order_date", "ticket_rolling", "Ticket (média móvel)", yprefix="R$ "), width="stretch")
        with b:
            section("Intervalo entre pedidos", "observado", "Dias desde o pedido anterior")
            st.plotly_chart(charts.line(tl.dropna(subset=["days_between"]), "order_date", "days_between", "Dias", color=SERIES[1],
                                        hover_fmt=".0f"), width="stretch")
        hist = cust_svc.history(cid)
        st.dataframe(hist[["order_date", "order_id", "name", "category", "quantity", "unit_price", "discount", "total_value"]],
                     hide_index=True, width="stretch", height=280,
                     column_config={"order_date": st.column_config.DatetimeColumn("Data", format="DD/MM/YYYY"),
                                    "order_id": "Pedido", "name": "Produto", "category": "Categoria", "quantity": "Qtd.",
                                    "unit_price": st.column_config.NumberColumn("Preço unit.", format="R$ %.2f"),
                                    "discount": st.column_config.NumberColumn("Desconto", format="percent"),
                                    "total_value": st.column_config.NumberColumn("Total", format="R$ %.2f")})

with tab2:
    a, b = st.columns(2)
    with a:
        section("Categorias preferidas", "observado", "Participação nas unidades compradas")
        sh = cust_svc.category_shares(cid).sort_values("share", ascending=False)
        st.plotly_chart(charts.hbar(sh["category"], sh["share"], fmt=".0%", text=[pct(v) for v in sh["share"]]), width="stretch")
    with b:
        section("Preferências estimadas", "inferido",
                "P(resposta positiva a uma recomendação da categoria): média posterior Beta com intervalo de credibilidade de 90%. "
                "O traço cinza é o prior do segmento; intervalos largos indicam pouca evidência.")
        bs = cust_svc.bayes_summary(cid)
        st.plotly_chart(charts.interval_chart(bs, "category", "interest_mean", "interest_low", "interest_high",
                                              prior_col="interest_prior_mean"), width="stretch")
    with st.expander("Tabela de estimativas bayesianas (ambos os estágios do funil)"):
        show = bs[["category", "interest_mean", "interest_low", "interest_high", "interest_n", "conversion_mean",
                   "conversion_low", "conversion_high", "purchase_mean", "purchase_low", "purchase_high"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in show.columns if c not in ("category", "interest_n")})

with tab3:
    a, b = st.columns([1, 1])
    with a:
        section("Segmento atribuído", "inferido")
        if clu["profile"]:
            st.markdown(f"**{clu['label']}** — {clu['profile']['description']}")
        else:
            st.write("Cliente sem histórico suficiente para segmentação (menos de 3 pedidos).")
    with b:
        section("Lojistas mais semelhantes", "inferido", "Similaridade de cosseno com shrinkage na matriz cliente-produto")
        sim = cust_svc.similar_customers(cid)
        if sim.empty:
            st.info("Sem vizinhos: o cliente ainda não possui compras suficientes.")
        else:
            st.dataframe(sim, hide_index=True, width="stretch", column_config={
                "customer_id": "ID", "name": "Lojista", "cluster": "Segmento",
                "similarity": st.column_config.ProgressColumn("Similaridade", min_value=0.0, max_value=1.0, format="%.2f")})

with tab4:
    section("Como a estimativa evoluiu com as evidências", "inferido",
            "Reprodução sequencial da atualização Beta-Binomial: cada interação com recomendação desloca a média e estreita o intervalo.")
    cat_sel = st.selectbox("Categoria", CATEGORIES, key="traj_cat")
    traj = cust_svc.bayes_trajectory(cid, cat_sel)
    stage = st.radio("Estágio do funil", ["interest", "conversion"], horizontal=True,
                     format_func=lambda s: {"interest": "Interesse (engajamento)", "conversion": "Conversão dado interesse"}[s])
    if len(traj) <= 1:
        st.info("Ainda não há interações nesta categoria: a estimativa é igual ao prior do segmento.")
    st.plotly_chart(charts.trajectory_chart(traj, stage), width="stretch")
