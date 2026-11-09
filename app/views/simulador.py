"""Página 4 — Simulador de Interações (feedback real gravado no banco)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.cards import callout, page_header, section
from app.components.state import customer_picker, services
from app.components.theme import NEUTRAL, SERIES
from config.settings import EXPERTS, INTERACTION_LABELS
from src.learning.feedback import UI_ACTIONS, FeedbackService

eng, cust_svc, _, _ = services()
fb = FeedbackService(eng.repos)
page_header("Simulador de Interações", "Simule o comportamento do lojista: cada feedback é gravado no banco e atualiza os modelos de verdade.")

c1, c2 = st.columns([2, 1])
with c1:
    cid = customer_picker(eng, "sim_customer")
with c2:
    method = st.selectbox("Estratégia", ["learned", "rules"], format_func={"learned": "Adaptativo (gating treinado)",
                                                                           "rules": "Adaptativo (regras)"}.get)

key = f"sim_{cid}_{method}"
if key not in st.session_state:
    res = eng.recommend(cid, eng.config_with(top_k=10), method)
    st.session_state[key] = {"session_id": fb.new_session(), "current": res, "history": []}
state = st.session_state[key]
current = state["current"]

a, b = st.columns([1.5, 1])
with a:
    section("Recomendações atuais")
    show = current.items[["rank", "name", "category", "relevance", "conversion_probability"]]
    st.dataframe(show, hide_index=True, width="stretch", height=390, column_config={
        "rank": "#", "name": "Produto", "category": "Categoria",
        "relevance": st.column_config.ProgressColumn("Relevância", min_value=0.0, max_value=1.0, format="%.2f"),
        "conversion_probability": st.column_config.ProgressColumn("P(compra)", min_value=0.0, max_value=1.0, format="percent")})
with b:
    section("Registrar feedback")
    with st.form("fb_form"):
        opts = {f"#{r.rank} {r.name}": (int(r.product_id), int(r.recommendation_id)) for r in current.items.itertuples()}
        choice = st.selectbox("Produto recomendado", list(opts))
        action = st.radio("Interação", list(UI_ACTIONS), horizontal=False,
                          help="Ignorar = impressão sem resposta (evidência fraca, NÃO é rejeição). "
                               "Comprar cria um pedido real e baixa o estoque.")
        go = st.form_submit_button("Registrar feedback", type="primary")
    st.caption(f"Sessão de recomendação #{state['session_id']}")

if go:
    pid, rec_id = opts[choice]
    itype = UI_ACTIONS[action]
    before_res = current
    sig_before = eng.signals.loc[cid].copy()
    bayes_before = cust_svc.bayes_summary(cid)
    try:
        with st.spinner("Registrando evento e atualizando estimativas…"):
            result = fb.register(cid, pid, itype, state["session_id"], model_name=f"hybrid_{method}", recommendation_id=rec_id,
                                 context={"origem": "simulador", "estrategia": method})
            done = eng.refresh_after_feedback(cid)
            after_res = eng.recommend(cid, eng.config_with(top_k=10), method)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    state["history"].append({"result": result, "done": done, "before": before_res, "after": after_res,
                             "bayes_before": bayes_before, "bayes_after": cust_svc.bayes_summary(cid),
                             "product": eng.snapshot.products.loc[pid, "name"]})
    state["current"] = after_res
    st.rerun()

if not state["history"]:
    st.info("Registre um feedback para ver as atualizações calculadas pelo sistema.")
    st.stop()

last = state["history"][-1]
r = last["result"]
st.divider()
st.markdown(f"### Última atualização: {INTERACTION_LABELS[r.interaction_type]} em “{last['product']}”")
st.caption(f"Desfecho da recomendação nesta sessão: {r.previous_outcome or 'nenhum'} → **{r.new_outcome}** · "
           f"interações gravadas: {r.interaction_ids}" + (f" · pedido #{r.order_id} criado" if r.order_id else ""))

steps = [f"Evento registrado no banco (interações {r.interaction_ids})."]
if r.bayes_changes:
    for ch in r.bayes_changes:
        stage = "interesse" if ch["stage"] == "interest" else "conversão"
        steps.append(f"Beta de {stage} em {r.category}: α {ch['alpha_before']:.2f}→{ch['alpha_after']:.2f}, "
                     f"β {ch['beta_before']:.2f}→{ch['beta_after']:.2f}; média {ch['mean_before']:.1%}→{ch['mean_after']:.1%}.")
else:
    steps.append("Nenhuma evidência bayesiana nova (o desfecho resolvido da sessão não mudou).")
if r.feature_changes:
    steps.append("Features atualizadas: " + ", ".join(f"{k} {v['antes']:.2f}→{v['depois']:.2f}" for k, v in r.feature_changes.items()
                                                      if v["antes"] is not None))
steps += [f"{k}: {v}" for k, v in last["done"].items()]
steps.append("Recomendações recalculadas com o novo estado.")
callout("O que o sistema efetivamente recalculou", steps)

t1, t2, t3, t4 = st.tabs(["Estimativa bayesiana", "Pesos dos especialistas", "Recomendações antes × depois", "Histórico da sessão"])
with t1:
    if r.bayes_changes:
        cols = st.columns(len(r.bayes_changes))
        for col, ch in zip(cols, r.bayes_changes):
            with col:
                stage = "Interesse (engajamento)" if ch["stage"] == "interest" else "Conversão dado interesse"
                st.plotly_chart(charts.beta_densities([("Antes", ch["alpha_before"], ch["beta_before"], NEUTRAL),
                                                       ("Depois", ch["alpha_after"], ch["beta_after"], SERIES[6])],
                                                      title=f"{stage} — {r.category}"), width="stretch", key=f"dens_{ch['stage']}")
    log = eng.repos.bayes.update_log(cid)
    log = log[(log["category"] == r.category)]
    if len(log):
        section("Evolução das estimativas registradas no simulador", help_text="Média posterior e IC 90% após cada atualização incremental")
        stg = st.radio("Estágio", ["interest", "conversion"], horizontal=True, key="sim_stage",
                       format_func={"interest": "Interesse", "conversion": "Conversão"}.get)
        lg = log[log["stage"] == stg].reset_index(drop=True)
        if len(lg):
            traj = pd.DataFrame({"step": range(1, len(lg) + 1), f"{stg}_mean": lg["mean_after"], f"{stg}_low": lg["ci_low"],
                                 f"{stg}_high": lg["ci_high"], "outcome": lg["reason"]})
            st.plotly_chart(charts.trajectory_chart(traj, stg), width="stretch")
with t2:
    wb, wa = last["before"].explanation.weights, last["after"].explanation.weights
    st.plotly_chart(charts.weights_compare(wb, wa), width="stretch")
    diff = max(abs(wa[e] - wb[e]) for e in EXPERTS)
    if diff < 0.005:
        st.caption("Os pesos praticamente não mudaram: uma única interação raramente é evidência suficiente para alterar a "
                   "seleção de especialistas — isso só ocorre quando os sinais do cliente (interações, histórico) mudam de forma relevante.")
    else:
        st.caption(f"Maior variação de peso: {diff:.1%}.")
with t3:
    bf = last["before"].items[["product_id", "rank", "name"]].rename(columns={"rank": "antes"})
    af = last["after"].items[["product_id", "rank", "name"]].rename(columns={"rank": "depois"})
    m = bf.merge(af, on=["product_id", "name"], how="outer")

    def _move(row):
        if pd.isna(row["antes"]):
            return "novo"
        if pd.isna(row["depois"]):
            return "saiu"
        d = int(row["antes"] - row["depois"])
        return f"▲ {d}" if d > 0 else f"▼ {-d}" if d < 0 else "="
    m["mudança"] = m.apply(_move, axis=1)
    st.dataframe(m.sort_values(["depois", "antes"]).drop(columns="product_id"), hide_index=True, width="stretch",
                 column_config={"name": "Produto", "antes": st.column_config.NumberColumn("Posição antes", format="%d"),
                                "depois": st.column_config.NumberColumn("Posição depois", format="%d")})
with t4:
    rows = [{"#": i + 1, "produto": h["product"], "interação": INTERACTION_LABELS[h["result"].interaction_type],
             "desfecho": f"{h['result'].previous_outcome or '—'} → {h['result'].new_outcome}",
             "pedido": h["result"].order_id} for i, h in enumerate(state["history"])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
