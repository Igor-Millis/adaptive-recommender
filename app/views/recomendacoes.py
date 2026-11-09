"""Página 3 — Central de Recomendações."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.cards import callout, confidence_badge, page_header, section
from app.components.state import customer_picker, services
from app.components.tables import recommendations_table
from app.components.theme import EXPERT_COLORS
from config.settings import EXPERT_LABELS, EXPERTS
from src.services.recommendation_service import GATING_METHODS

eng, _, _, _ = services()
page_header("Central de Recomendações", "Execute o motor adaptativo e veja por que cada produto foi recomendado.")

with st.form("rec_form"):
    c1, c2, c3 = st.columns([2.2, 1.3, 1])
    with c1:
        cid = customer_picker(eng, "rec_customer")
    with c2:
        method = st.selectbox("Estratégia de combinação", list(GATING_METHODS), format_func=GATING_METHODS.get,
                              help="Adaptativo: pesos diferentes por cliente. Fixos/iguais: mesmos pesos para todos (baselines).")
    with c3:
        top_k = st.slider("Nº de produtos", 5, 20, 10)
    with st.expander("Regras de ranking"):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            rule = st.radio("Produtos já comprados", ["all", "recent", "none"], index=0,
                            format_func={"all": "Excluir todos (descoberta)", "recent": "Permitir recompra antiga",
                                         "none": "Não excluir"}.get)
            recent_days = st.number_input("Recompra após (dias)", 30, 365, 120, step=15, disabled=False)
        with d2:
            lam = st.slider("Diversificação (MMR)", 0.0, 1.0, 1.0, 0.05,
                            help="1 = só relevância. Valores menores trocam um pouco de relevância por variedade (Maximal Marginal Relevance).")
            max_cat = st.number_input("Máx. por categoria (0 = sem limite)", 0, 20, 0)
        with d3:
            margin_w = st.slider("Peso da margem comercial", 0.0, 1.0, 0.0, 0.05,
                                 help="Objetivo separado da relevância: 0 = otimiza apenas o interesse do cliente.")
        with d4:
            req_stock = st.checkbox("Exigir estoque disponível", True)
    submitted = st.form_submit_button("Gerar recomendações", type="primary")

if submitted:
    cfg = eng.config_with(top_k=top_k, purchased_rule=rule, recent_days=int(recent_days), diversity_lambda=lam,
                          max_per_category=int(max_cat) or None, margin_weight=margin_w, require_stock=req_stock)
    with st.spinner("Calculando scores dos especialistas e combinando…"):
        st.session_state["rec_result"] = eng.recommend(cid, cfg, method)
        st.session_state["rec_compare"] = eng.expert_rankings(cid, top_k, cfg)

res = st.session_state.get("rec_result")
if res is None or res.customer_id != cid:
    st.info("Escolha o cliente e clique em **Gerar recomendações**.")
    st.stop()

expl = res.explanation
items = res.items
name = eng.snapshot.customers.loc[cid, "name"]
st.markdown(f"#### {name} {confidence_badge(expl.confidence_level)}", unsafe_allow_html=True)
st.caption(f"Execução #{res.run_id} · {GATING_METHODS[res.method]} · {len(items)} produtos · "
           f"exclusões: {', '.join(f'{k.replace('_', ' ')}={v}' for k, v in res.exclusions.items())}")

tab1, tab2, tab3 = st.tabs(["Recomendações", "Como o sistema chegou a essa recomendação", "Comparar estratégias"])

with tab1:
    if items.empty:
        st.warning("Nenhum produto elegível com as regras atuais.")
    else:
        recommendations_table(items)
        section("Motivos da recomendação", help_text="Explicações dos especialistas que mais contribuíram para cada produto")
        for r in items.itertuples():
            with st.expander(f"#{r.rank} · {r.name} — relevância {r.relevance:.2f} · P(compra) {r.conversion_probability:.0%}"):
                a, b = st.columns([1.4, 1])
                with a:
                    for reason in r.reasons:
                        st.markdown(f'<div class="ar-reason">• {reason}</div>', unsafe_allow_html=True)
                    st.caption(f"Categoria {r.category} · {r.subcategory} · R$ {r.price:.2f} · estoque {r.stock} · margem {r.margin:.0%}")
                with b:
                    st.plotly_chart(charts.weights_bar(r.contributions, height=110), width="stretch", key=f"contrib_{r.rank}")

with tab2:
    a, b = st.columns([1.2, 1])
    with a:
        section("Pesos efetivamente utilizados pelo motor adaptativo")
        st.plotly_chart(charts.weights_bar(expl.weights, height=130), width="stretch")
        if res.method == "learned":
            st.caption("Pesos das regras (antes do ajuste aprendido): " +
                       " · ".join(f"{EXPERT_LABELS[e]} {w:.0%}" for e, w in expl.rule_weights.items()))
        callout("Justificativa da seleção", expl.justification)
        if expl.discarded:
            st.caption("Especialistas descartados ou pouco utilizados (< 5%): " + ", ".join(EXPERT_LABELS[e] for e in expl.discarded))
    with b:
        section("Principais sinais considerados", "observado")
        sig = pd.DataFrame(expl.key_signals)
        st.dataframe(sig, hide_index=True, width="stretch",
                     column_config={"sinal": "Sinal", "valor": st.column_config.NumberColumn("Valor", format="%.2f")})
        st.caption(f"Confiança agregada = Σ peso × confiança do especialista = {expl.confidence_value:.2f} → {expl.confidence_level}.")
    if len(items):
        section("Contribuição de cada modelo por produto", help_text="Parcela do score híbrido vinda de cada especialista")
        st.plotly_chart(charts.contributions_stacked(items, EXPERTS), width="stretch")

with tab3:
    comp = st.session_state.get("rec_compare", {})
    names = eng.snapshot.products["name"]
    order = ["content_based", "collaborative", "supervised", "bayesian", "popularity", "hybrid_learned", "hybrid_rules", "hybrid_fixed"]
    order = [o for o in order if o in comp]
    labels = {**EXPERT_LABELS, "hybrid_learned": "Híbrido adaptativo (treinado)", "hybrid_rules": "Híbrido adaptativo (regras)",
              "hybrid_fixed": "Híbrido pesos fixos"}
    section("Top produtos por estratégia", help_text="Mesmos candidatos e regras; apenas o critério de ordenação muda")
    cols = st.columns(4)
    for i, key in enumerate(order):
        with cols[i % 4]:
            dot = f'<span class="ar-legend-dot" style="background:{EXPERT_COLORS.get(key, "#888")}"></span>'
            st.markdown(f"{dot}**{labels[key]}**", unsafe_allow_html=True)
            st.markdown("\n".join(f"{j + 1}. {names.loc[p][:42]}" for j, p in enumerate(comp[key])))
    section("Sobreposição entre listas (Jaccard)", help_text="1 = listas idênticas; 0 = nenhum produto em comum")
    mat = pd.DataFrame(index=[labels[o] for o in order], columns=[labels[o] for o in order], dtype=float)
    for x in order:
        for y in order:
            sx, sy = set(comp[x]), set(comp[y])
            mat.loc[labels[x], labels[y]] = len(sx & sy) / max(len(sx | sy), 1)
    st.plotly_chart(charts.heatmap(mat, fmt=".2f", zmin=0, zmax=1), width="stretch")
