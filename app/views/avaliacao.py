"""Página 5 — Avaliação dos Modelos (área técnica)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.cards import callout, page_header, section
from app.components.state import reset_engine, services
from app.components.theme import EXPERT_COLORS, SERIES
from config.settings import EXPERT_LABELS, EXPERTS

eng, _, _, model_svc = services()
page_header("Avaliação dos Modelos", "Validação temporal: treino < mai/2026, validação mai–jul/2026, teste ago–out/2026. "
            "Todos os números vêm do último experimento executado.")

ev = model_svc.latest_evaluation()
if ev is None:
    st.warning("Nenhuma avaliação encontrada.")
    if st.button("Executar avaliação agora (≈ 1 min)", type="primary"):
        with st.spinner("Executando experimento…"):
            model_svc.run_evaluation()
        st.rerun()
    st.stop()

R = ev["results"]
test = R[(R["stage"] == "teste") & (R["segment_type"] == "geral")]
proto = ev["protocol"]
st.caption(f"Tarefa: {proto['task']} · horizonte {proto['horizon_days']} dias · {proto['n_test_customers']} clientes avaliados no teste · "
           f"IC95% por bootstrap ({proto['bootstrap_samples']} reamostragens sobre clientes).")

tabs = st.tabs(["Ranking", "Comparações", "Por segmento", "Motor adaptativo", "Classificação", "Clusterização", "Operações"])
METRIC_LABELS = {"ndcg": "NDCG@K", "precision": "Precision@K", "recall": "Recall@K", "map": "MAP@K", "hit_rate": "Hit rate@K",
                 "coverage": "Cobertura do catálogo", "diversity": "Diversidade intra-lista"}

with tabs[0]:
    c1, c2 = st.columns([1, 1])
    metric = c1.selectbox("Métrica", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
    k = c2.select_slider("K", options=sorted(test["k"].dropna().unique().astype(int)), value=10)
    d = test[(test["metric"] == metric) & (test["k"] == k)]
    st.plotly_chart(charts.metric_bars_ci(d, title=f"{METRIC_LABELS[metric].replace('K', str(k))} — teste (barras: IC95%)"),
                    width="stretch")
    piv = test[test["k"] == k].pivot_table(index="model_name", columns="metric", values="value")
    piv = piv[[m for m in METRIC_LABELS if m in piv.columns]].sort_values("ndcg", ascending=False)
    piv.index = [EXPERT_LABELS.get(i, i) for i in piv.index]
    times = test[test["metric"] == "fit_seconds"].set_index("model_name")["value"]
    piv["tempo (s)"] = [times.get(next((m for m, lab in EXPERT_LABELS.items() if lab == i), i), float("nan")) for i in piv.index]
    st.dataframe(piv.round(4), width="stretch")
    st.caption("Tempo = ajuste + scoring no estágio de teste; híbridos incluem o tempo de todos os especialistas e do contexto.")
    callout("Conclusões geradas a partir dos resultados", ev["conclusions"])

with tabs[1]:
    section("Comparações pareadas por cliente", help_text="Diferença média de NDCG@10 (A − B) com IC95% por bootstrap pareado")
    pr = pd.DataFrame(ev["paired"])
    pr["A"] = pr["a"].map(lambda x: EXPERT_LABELS.get(x, x))
    pr["B"] = pr["b"].map(lambda x: EXPERT_LABELS.get(x, x))
    pr["veredito"] = pr.apply(lambda r: "A melhor (consistente)" if r["ci_low"] > 0 else "B melhor (consistente)" if r["ci_high"] < 0
                              else "Inconclusivo", axis=1)
    st.dataframe(pr[["A", "B", "diff", "ci_low", "ci_high", "p_le_zero", "veredito", "n"]], hide_index=True, width="stretch",
                 column_config={"diff": st.column_config.NumberColumn("Δ NDCG@10", format="%+.4f"),
                                "ci_low": st.column_config.NumberColumn("IC inf.", format="%+.4f"),
                                "ci_high": st.column_config.NumberColumn("IC sup.", format="%+.4f"),
                                "p_le_zero": st.column_config.NumberColumn("P(Δ ≤ 0)", format="%.3f"), "n": "Clientes"})
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=pr["diff"], y=pr["A"] + " − " + pr["B"], mode="markers",
                               marker=dict(size=10, color=SERIES[0], line=dict(color="#fff", width=2)),
                               error_x=dict(type="data", symmetric=False, array=pr["ci_high"] - pr["diff"],
                                            arrayminus=pr["diff"] - pr["ci_low"], color=SERIES[0], thickness=2, width=0),
                               hovertemplate="%{y}<br>Δ = %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#8a8984", dash="dot"))
    st.plotly_chart(charts.style(fig, 60 + 34 * len(pr), legend=False), width="stretch")

with tabs[2]:
    seg_types = {"historico": "Disponibilidade de histórico", "cenario_detectado": "Cenário detectado (dados observados)",
                 "cluster": "Cluster", "arquetipo_sintetico": "Arquétipo sintético (ground truth — só análise)"}
    st_sel = st.selectbox("Segmentar por", [s for s in seg_types if s in R["segment_type"].unique()], format_func=seg_types.get)
    m_sel = st.radio("Métrica", ["ndcg", "recall"], horizontal=True, format_func={"ndcg": "NDCG@10", "recall": "Recall@10"}.get)
    seg = R[(R["segment_type"] == st_sel) & (R["metric"] == m_sel)]
    piv = seg.pivot_table(index="model_name", columns="segment_value", values="value")
    n = seg.groupby("segment_value")["n_customers"].first()
    piv.columns = [f"{c} (n={int(n[c])})" for c in piv.columns]
    order = [m for m in ["hybrid_learned", "hybrid_rules", "hybrid_fixed", "hybrid_equal", "supervised", "content_based",
                         "collaborative", "collaborative_mf", "bayesian", "popularity", "random"] if m in piv.index]
    piv = piv.loc[order]
    piv.index = [EXPERT_LABELS.get(i, i) for i in piv.index]
    st.plotly_chart(charts.heatmap(piv, fmt=".3f"), width="stretch")
    st.caption("Segmentos pequenos (n < 5) têm alta variância: use-os como indício, não como conclusão.")

with tabs[3]:
    sel = ev["selection"]["teste"]
    lg = ev["learned_gating"]["train_info"]
    c1, c2 = st.columns(2)
    with c1:
        section("Pesos médios no teste por estratégia")
        mw = ev["mean_weights_test"]
        for name, w in mw.items():
            st.markdown(f"**{EXPERT_LABELS.get(name, name)}**")
            st.plotly_chart(charts.weights_bar(w, height=95), width="stretch", key=f"mw_{name}")
    with c2:
        section("Seleção do gating treinável (validação cruzada)", help_text="NDCG@10 fora da dobra; empates favorecem o mais simples")
        cv = pd.DataFrame(lg["cv"])
        cv["λ"] = cv["l2"].map(lambda v: "—" if pd.isna(v) else v)
        st.dataframe(cv[["variant", "λ", "cv_ndcg", "cv_ndcg_sd"]], hide_index=True, width="stretch",
                     column_config={"variant": "Variante", "cv_ndcg": st.column_config.NumberColumn("NDCG@10 CV", format="%.4f"),
                                    "cv_ndcg_sd": st.column_config.NumberColumn("Desvio", format="%.4f")})
        st.markdown(f"Variante escolhida na avaliação: **{lg.get('variant')}**.")
        callout("O gating escolhe bem?", [
            f"Oráculo (melhor especialista por cliente, a posteriori): NDCG@10 = {sel['oracle_ndcg']:.4f} — limite superior teórico da seleção.",
            *[f"{EXPERT_LABELS.get(m, m)}: maior peso = melhor especialista em {v['top_weight_is_best']:.0%} dos clientes; "
              f"correlação de Spearman média peso × desempenho = {v['mean_spearman_weight_vs_ndcg']:.2f}." for m, v in sel["methods"].items()
              if v["mean_spearman_weight_vs_ndcg"] is not None],
        ])
    section("Pesos médios por disponibilidade de histórico (gating por regras)")
    by = pd.DataFrame(sel["methods"]["hybrid_rules"]["mean_weights_by_segment"]).T
    by.columns = [EXPERT_LABELS[c] for c in by.columns]
    st.plotly_chart(charts.heatmap(by, fmt=".2f", zmin=0, zmax=1), width="stretch")
    best = pd.Series(sel["methods"]["hybrid_rules"]["best_expert_share"])
    st.caption("Especialista com melhor NDCG@10 por cliente no teste: " +
               " · ".join(f"{EXPERT_LABELS.get(k2, k2)} {v:.0%}" for k2, v in best.sort_values(ascending=False).items()))

with tabs[4]:
    clf = ev["classification"]
    algs = [a for a in clf if not a.startswith("_")]
    rows = [{"algoritmo": a + (" ✓" if a == clf["_selected"] else ""), **{k2: v for k2, v in clf[a]["metrics"].items()
                                                                           if k2 in ("roc_auc", "pr_auc", "log_loss", "brier", "precision", "recall", "base_rate", "threshold")}} for a in algs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in ["roc_auc", "pr_auc", "log_loss", "brier",
                                                                                             "precision", "recall", "base_rate", "threshold"]})
    st.caption("✓ = algoritmo selecionado (maior PR-AUC no holdout temporal do treino). Taxa base ≈ fração de pares positivos — "
               "PR-AUC deve ser comparado a ela. Limiar = máximo F1 no holdout, reaplicado no teste.")
    cols = {"logistic_regression": SERIES[0], "random_forest": SERIES[1], "lightgbm": SERIES[2]}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(charts.curve_chart([(a, clf[a]["roc"]["fpr"], clf[a]["roc"]["tpr"], cols.get(a, SERIES[3])) for a in algs],
                                           "Taxa de falsos positivos", "Taxa de verdadeiros positivos", diagonal=True, title="Curva ROC"),
                        width="stretch")
    with c2:
        st.plotly_chart(charts.curve_chart([(a, clf[a]["pr"]["recall"], clf[a]["pr"]["precision"], cols.get(a, SERIES[3])) for a in algs],
                                           "Recall", "Precisão", title="Curva Precisão-Recall"), width="stretch")
    with c3:
        st.plotly_chart(charts.curve_chart([(a, clf[a]["reliability"]["mean_predicted"], clf[a]["reliability"]["fraction_positive"],
                                             cols.get(a, SERIES[3])) for a in algs],
                                           "Probabilidade prevista", "Frequência observada", diagonal=True, title="Calibração (decis)"),
                        width="stretch")

with tabs[5]:
    cl = ev["clustering"]
    dg = pd.DataFrame(cl["diagnostics"])
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(charts.curve_chart([("Silhouette", dg["k"], dg["silhouette"], SERIES[0])], "k", "Silhouette", title="Silhouette por k", fmt=".3f"),
                    width="stretch")
    c2.plotly_chart(charts.curve_chart([("Estabilidade", dg["k"], dg["stability"], SERIES[1])], "k", "ARI médio (bootstrap)",
                                       title="Estabilidade por k", fmt=".3f"), width="stretch")
    c3.plotly_chart(charts.curve_chart([("Inertia", dg["k"], dg["inertia"], SERIES[2])], "k", "Inertia", title="Cotovelo (inertia)", fmt=".1f"),
                    width="stretch")
    h = cl["hierarchical"]
    st.markdown(f"k escolhido: **{cl['k']}** (máx. silhouette × estabilidade para k ≥ 3). Ward: silhouette {h['silhouette_ward']:.3f}; "
                f"concordância K-Means × Ward (ARI) = {h['ari_kmeans_vs_ward']:.3f}.")
    st.dataframe(pd.DataFrame(cl["profiles"])[["cluster_id", "label", "size", "description"]], hide_index=True, width="stretch",
                 column_config={"cluster_id": "ID", "label": "Segmento", "size": "Clientes", "description": "Descrição"})

with tabs[6]:
    section("Versões dos modelos")
    vers = model_svc.versions()
    st.dataframe(vers[["model_name", "version", "status", "trained_at", "train_cutoff", "artifact_path"]], hide_index=True,
                 width="stretch", height=260)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retreinamento controlado**")
        st.caption("Reestima priors bayesianos, retreina o supervisionado, reclusteriza e reaprende o gating. "
                   "Pedidos e interações são preservados; versões anteriores ficam arquivadas.")
        if st.button("Retreinar todos os modelos (≈ 1 min)"):
            with st.spinner("Retreinando…"):
                rep = model_svc.retrain()
            reset_engine()
            st.success(f"Retreinamento concluído em {rep['elapsed_seconds']:.0f}s — supervisionado: {rep['supervised']['algorithm']}.")
    with c2:
        st.markdown("**Reexecutar experimento de avaliação**")
        st.caption("Recalcula todas as métricas desta página e o arquivo docs/MODEL_EVALUATION.md.")
        if st.button("Executar avaliação (≈ 1 min)"):
            with st.spinner("Executando experimento temporal…"):
                model_svc.run_evaluation()
            st.rerun()
    section("Eventos recentes (auditoria)")
    st.dataframe(model_svc.recent_events(30)[["created_at", "event_type", "entity", "entity_id"]], hide_index=True, width="stretch")
