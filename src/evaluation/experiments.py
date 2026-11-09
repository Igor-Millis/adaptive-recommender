"""Experimento completo: compara especialistas, híbridos fixos e adaptativos no teste temporal.

Saídas: resultados no banco (evaluation_runs/results), JSON detalhado em
``models/metadata/evaluation_latest.json`` e relatório ``docs/MODEL_EVALUATION.md``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from config.settings import (
    CB,
    DATA_END,
    EVAL_JSON_PATH,
    EVAL_REPORT_PATH,
    EVAL_REPORT_TITLE,
    EVAL,
    EXPERT_LABELS,
    EXPERTS,
    HORIZON_DAYS,
    METADATA_DIR,
    RANDOM_SEED,
    TEST_START,
    VAL_START,
)
from src.adaptive.ensemble import combine
from src.adaptive.gating import FixedGating, LearnedGating, RuleGating, tune_fixed_weights
from src.data_generation.pipeline import load_ground_truth
from src.evaluation.comparison import evaluate_strategy, paired_comparisons, segment_table, selection_analysis
from src.evaluation.temporal_validation import Stage, build_stage
from src.preprocessing.feature_engineering import segment_by_history
from src.recommenders.supervised import SupervisedRecommender, classification_metrics, reliability
from src.utils.logging_setup import get_logger

log = get_logger("experiments")

SINGLE = ["popularity", "content_based", "collaborative", "collaborative_mf", "supervised", "bayesian"]
HYBRIDS = ["hybrid_equal", "hybrid_fixed", "hybrid_rules", "hybrid_learned"]


def strategy_scores(stage: Stage, gatings: dict) -> tuple[dict, dict, dict]:
    scores, weights, times = {}, {}, {}
    rng = np.random.default_rng(RANDOM_SEED)
    scores["random"] = rng.random(stage.mask.shape)
    for e in SINGLE:
        if e in stage.norm:
            scores[e] = stage.norm[e]
            times[e] = stage.timings.get(e, np.nan)
    base = stage.timings.get("context", 0) + sum(stage.timings.get(e, 0) for e in EXPERTS)
    for name, g in gatings.items():
        t = time.perf_counter()
        W = g.weights(stage.signals)
        S, _ = combine({e: stage.norm[e] for e in EXPERTS}, W, EXPERTS)
        times[name] = base + (time.perf_counter() - t)
        scores[name] = S
        weights[name] = W
    return scores, weights, times


def _segments(stage: Stage) -> dict[str, pd.Series]:
    f = stage.suite.ctx.customer_features
    segs = {"historico": segment_by_history(f)}
    labels = stage.suite.ctx.cluster_labels
    names = stage.suite.ctx.cluster_names
    segs["cluster"] = labels.map(lambda k: names.get(int(k), "Sem histórico suficiente"))
    segs["cenario_detectado"] = pd.Series(
        np.select([f["n_orders"] == 0, f["drift_js"] > CB.drift_threshold, f["seasonality_index"] > 0.06],
                  ["sem histórico", "mudança de preferência", "sazonal"], "padrão estável"), index=f.index)
    gt = load_ground_truth()
    if len(gt):
        segs["arquetipo_sintetico"] = gt.set_index("customer_id")["archetype"]
    return segs


def _classification_on_stage(stage: Stage, sup: SupervisedRecommender) -> dict:
    s = stage.snapshot
    # clientes ativos na janela = ao menos um pedido (inclui recompras) — houve oportunidade de compra
    rows = stage.mask & stage.active[:, None]
    y = stage.Y[rows].astype(int)
    out = {}
    thr = sup.report["candidates"][sup.algorithm]["metrics"]["threshold"]
    for alg in sup.report["candidates"]:
        m = sup.with_algorithm(alg) if hasattr(sup, "all_candidates") else sup
        P, _ = m.predict_matrix(s, stage.suite.ctx)
        p = P[rows]
        t_alg = sup.report["candidates"][alg]["metrics"]["threshold"]
        met = classification_metrics(y, p, threshold=t_alg)
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        sel = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
        sel2 = np.linspace(0, len(pr) - 1, min(120, len(pr))).astype(int)
        out[alg] = {"metrics": met, "reliability": reliability(y, p),
                    "roc": {"fpr": fpr[sel].tolist(), "tpr": tpr[sel].tolist()},
                    "pr": {"precision": pr[sel2].tolist(), "recall": rc[sel2].tolist()}}
    out["_selected"] = sup.algorithm
    out["_threshold"] = thr
    return out


def run_experiment(dataset, save: bool = True, repos=None) -> dict:
    t_start = time.perf_counter()
    log.info("Experimento: estágio A (validação)")
    sup_a = SupervisedRecommender.train(dataset, VAL_START)
    stage_a = build_stage(dataset, "validacao", VAL_START, TEST_START, sup_a)
    k_a = stage_a.suite.segmentation.k if stage_a.suite.segmentation else None

    learned = LearnedGating().fit(stage_a.signals, {e: stage_a.norm[e] for e in EXPERTS}, stage_a.Y, stage_a.mask)
    fixed_w, fixed_val = tune_fixed_weights({e: stage_a.norm[e] for e in EXPERTS}, stage_a.Y, stage_a.mask, stage_a.signals)
    gatings = {"hybrid_equal": FixedGating({e: 1.0 for e in EXPERTS}), "hybrid_fixed": FixedGating(fixed_w),
               "hybrid_rules": RuleGating(), "hybrid_learned": learned}

    sim_a = stage_a.suite.experts["content_based"].item_similarity
    scores_a, weights_a, _ = strategy_scores(stage_a, gatings)
    val_results = {n: evaluate_strategy(n, S, stage_a, sim_a, ks=(EVAL.main_k,), n_boot=500) for n, S in scores_a.items()}

    log.info("Experimento: estágio B (teste)")
    sup_b = SupervisedRecommender.train(dataset, TEST_START, fixed_k=k_a)
    stage_b = build_stage(dataset, "teste", TEST_START, DATA_END, sup_b)
    sim_b = stage_b.suite.experts["content_based"].item_similarity
    scores_b, weights_b, times_b = strategy_scores(stage_b, gatings)
    results = {n: evaluate_strategy(n, S, stage_b, sim_b) for n, S in scores_b.items()}

    segs = _segments(stage_b)
    seg_df = pd.concat([segment_table(results, segs, metric="ndcg"), segment_table(results, segs, metric="recall")])
    pairs = [("hybrid_learned", "hybrid_fixed"), ("hybrid_rules", "hybrid_fixed"), ("hybrid_learned", "hybrid_rules"),
             ("hybrid_learned", "supervised"), ("hybrid_rules", "supervised"), ("hybrid_fixed", "supervised"),
             ("hybrid_learned", "hybrid_equal"), ("supervised", "popularity"), ("collaborative", "collaborative_mf")]
    paired = paired_comparisons(results, pairs)
    selection = {"teste": selection_analysis(results, {k: weights_b[k] for k in ("hybrid_rules", "hybrid_learned")}, stage_b, segs["historico"]),
                 "validacao": selection_analysis(val_results, {k: weights_a[k] for k in ("hybrid_rules", "hybrid_learned")}, stage_a,
                                                 segment_by_history(stage_a.suite.ctx.customer_features))}
    classification = _classification_on_stage(stage_b, sup_b)

    summary_rows = [r for res in results.values() for r in res["summary"]]
    for n, t in times_b.items():
        summary_rows.append({"model_name": n, "metric": "fit_seconds", "k": None, "value": float(t), "ci_low": None,
                             "ci_high": None, "n_customers": None})
    summary = pd.DataFrame(summary_rows)
    summary["stage"] = "teste"
    summary["segment_type"] = "geral"
    summary["segment_value"] = "todos"
    val_rows = pd.DataFrame([r for res in val_results.values() for r in res["summary"]])
    val_rows["stage"] = "validacao"
    val_rows["segment_type"] = "geral"
    val_rows["segment_value"] = "todos"
    seg_df["stage"] = "teste"
    all_rows = pd.concat([summary, val_rows, seg_df], ignore_index=True)

    seg_b = stage_b.suite.segmentation
    report = {
        "protocol": {"train": f"< {VAL_START.date()}", "validation": f"[{VAL_START.date()}, {TEST_START.date()})",
                     "test": f"[{TEST_START.date()}, {DATA_END.date()})", "horizon_days": HORIZON_DAYS,
                     "task": "descoberta (produtos não comprados até o corte)", "ks": list(EVAL.ks),
                     "bootstrap_samples": EVAL.bootstrap_samples, "n_test_customers": len(stage_b.relevant),
                     "n_val_customers": len(stage_a.relevant)},
        "fixed_weights": fixed_w, "fixed_weights_val_ndcg": fixed_val,
        "learned_gating": learned.to_dict(),
        "paired": paired, "selection": selection, "classification": classification,
        "supervised_report_test_stage": sup_b.report,
        "clustering": {"k": seg_b.k, "diagnostics": seg_b.diagnostics.to_dict("records"),
                       "hierarchical": {k: v for k, v in seg_b.hierarchical.items() if k != "ward_labels"},
                       "profiles": seg_b.profiles.drop(columns=["centroid"]).to_dict("records")},
        "mean_weights_test": {n: dict(zip(EXPERTS, np.round(W.mean(axis=0), 4).tolist())) for n, W in weights_b.items()},
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    report["conclusions"] = conclusions(all_rows, paired, selection)
    if save:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        path = EVAL_JSON_PATH
        payload = dict(report)
        payload["results"] = all_rows.to_dict("records")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=_json_default)
        write_markdown_report(report, all_rows)
        if repos is not None:
            rid = repos.evaluations.save(report["protocol"], {"conclusions": report["conclusions"],
                                                              "fixed_weights": fixed_w, "paired": paired},
                                         all_rows, str(path))
            report["eval_run_id"] = rid
            repos.events.log("evaluation_completed", "evaluation_runs", rid, {"elapsed": report["elapsed_seconds"]})
    report["results"] = all_rows
    report["gatings"] = gatings
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return str(o)
    return str(o)


def conclusions(rows: pd.DataFrame, paired: list[dict], selection: dict) -> list[str]:
    main = rows[(rows["stage"] == "teste") & (rows["segment_type"] == "geral") & (rows["metric"] == "ndcg") & (rows["k"] == EVAL.main_k)]
    main = main.sort_values("value", ascending=False)
    out = []
    best = main.iloc[0]
    out.append(f"Melhor NDCG@{EVAL.main_k} no teste: {EXPERT_LABELS.get(best.model_name, best.model_name)} = {best.value:.4f} "
               f"(IC95% {best.ci_low:.4f}–{best.ci_high:.4f}).")
    for p in paired:
        a, b = EXPERT_LABELS.get(p["a"], p["a"]), EXPERT_LABELS.get(p["b"], p["b"])
        if abs(p["ci_low"]) < 1e-12 and abs(p["ci_high"]) < 1e-12:
            verdict = "rankings idênticos (o gating treinado escolheu a mesma variante das regras)"
        elif p["ci_low"] > 0:
            verdict = "ganho estatisticamente consistente (IC95% acima de zero)"
        elif p["ci_high"] < 0:
            verdict = "perda estatisticamente consistente (IC95% abaixo de zero)"
        else:
            verdict = "diferença NÃO conclusiva (IC95% inclui zero)"
        out.append(f"{a} − {b}: ΔNDCG@{p['k']} = {p['diff']:+.4f} (IC95% {p['ci_low']:+.4f} a {p['ci_high']:+.4f}) → {verdict}.")
    sel = selection.get("teste", {}).get("methods", {})
    for m, v in sel.items():
        out.append(f"{EXPERT_LABELS.get(m, m)}: o especialista de maior peso coincide com o melhor especialista do cliente em "
                   f"{v['top_weight_is_best']:.0%} dos casos (oráculo NDCG@{EVAL.main_k} = {selection['teste']['oracle_ndcg']:.4f}).")
    return out


def write_markdown_report(report: dict, rows: pd.DataFrame) -> None:
    k = EVAL.main_k
    test = rows[(rows["stage"] == "teste") & (rows["segment_type"] == "geral")]

    def table(metrics, kk):
        sub = test[test["metric"].isin(metrics) & ((test["k"] == kk) | test["k"].isna())]
        piv = sub.pivot_table(index="model_name", columns="metric", values="value", aggfunc="first")
        piv = piv.reindex(columns=[m for m in metrics if m in piv.columns])
        piv = piv.sort_values(metrics[0], ascending=False)
        lines = ["| Estratégia | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        for name, r in piv.iterrows():
            cells = []
            for m in piv.columns:
                ci = sub[(sub["model_name"] == name) & (sub["metric"] == m)]
                v = r[m]
                if len(ci) and pd.notna(ci.iloc[0]["ci_low"]):
                    cells.append(f"{v:.4f} [{ci.iloc[0]['ci_low']:.3f}, {ci.iloc[0]['ci_high']:.3f}]")
                else:
                    cells.append(f"{v:.4f}")
            lines.append(f"| {EXPERT_LABELS.get(name, name)} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def seg_table(seg_type: str) -> list[str]:
        seg = rows[(rows["stage"] == "teste") & (rows["segment_type"] == seg_type) & (rows["metric"] == "ndcg")]
        if seg.empty:
            return ["(sem dados)"]
        seg_piv = seg.pivot_table(index="model_name", columns="segment_value", values="value")
        seg_n = seg.groupby("segment_value")["n_customers"].first()
        lines = ["| Estratégia | " + " | ".join(f"{c} (n={int(seg_n[c])})" for c in seg_piv.columns) + " |",
                 "|---|" + "---|" * len(seg_piv.columns)]
        for name, r in seg_piv.sort_values(seg_piv.columns[0], ascending=False).iterrows():
            lines.append(f"| {EXPERT_LABELS.get(name, name)} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
        return lines

    seg_lines = seg_table("historico") + ["", f"### Por cenário detectado nos dados observados (NDCG@{k})", ""] + \
        seg_table("cenario_detectado") + ["", "Cenários: *sem histórico* (0 pedidos no corte), *mudança de preferência* "
                                          "(JS > 0,15), *sazonal* (índice > 0,06), *padrão estável* (demais). "
                                          "Segmentos com n < 10 são indicativos, não conclusivos."]

    clf = report["classification"]
    clf_lines = ["| Algoritmo | ROC-AUC | PR-AUC | Log Loss | Brier | Precisão | Recall | Taxa base |", "|---|---|---|---|---|---|---|---|"]
    for alg, v in clf.items():
        if alg.startswith("_"):
            continue
        m = v["metrics"]
        sel = " (selecionado)" if alg == clf["_selected"] else ""
        clf_lines.append(f"| {alg}{sel} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                         f"{m['precision']:.3f} | {m['recall']:.3f} | {m['base_rate']:.4f} |")
    lg = report["learned_gating"]["train_info"]
    cv_lines = [f"| {r.get('variant', '')} | {'—' if r['l2'] is None else r['l2']} | {r['cv_ndcg']:.4f} ± {r['cv_ndcg_sd']:.4f} |"
                for r in lg.get("cv", [])]
    mw = report["mean_weights_test"]
    w_lines = ["| Gating | " + " | ".join(EXPERT_LABELS[e] for e in EXPERTS) + " |", "|---|" + "---|" * len(EXPERTS)]
    for n, w in mw.items():
        w_lines.append(f"| {EXPERT_LABELS.get(n, n)} | " + " | ".join(f"{w[e]:.2f}" for e in EXPERTS) + " |")
    clus = report["clustering"]
    clus_lines = ["| k | Inertia | Silhouette | Estabilidade (ARI bootstrap) |", "|---|---|---|---|"]
    for d in clus["diagnostics"]:
        clus_lines.append(f"| {d['k']} | {d['inertia']:.1f} | {d['silhouette']:.3f} | {d['stability']:.3f} |")

    md = f"""# {EVAL_REPORT_TITLE}

> Documento **gerado automaticamente** por `scripts/run_evaluation.py` a partir dos resultados reais do experimento.
> Reexecutar o script atualiza todos os números.

## 1. Protocolo

- Treino: dados {report['protocol']['train']}; validação {report['protocol']['validation']}; teste {report['protocol']['test']}.
- Tarefa: **{report['protocol']['task']}**; relevante = produto comprado na janela de {HORIZON_DAYS} dias.
- Especialistas re-ajustados com dados anteriores ao teste; gating treinável e pesos fixos aprendidos **somente** na validação.
- Clientes avaliados no teste: {report['protocol']['n_test_customers']} (com ≥ 1 compra nova na janela). IC95% por bootstrap sobre clientes ({EVAL.bootstrap_samples} reamostragens).
- Pesos fixos ajustados na validação (grade no simplex, passo 0,1): {', '.join(f"{EXPERT_LABELS[e]} {w:.1f}" for e, w in report['fixed_weights'].items())}.

## 2. Ranking — teste (K = {k})

{table(['ndcg', 'precision', 'recall', 'map', 'hit_rate', 'coverage', 'diversity'], k)}

### K = 5
{table(['ndcg', 'precision', 'recall'], 5)}

### K = 20
{table(['ndcg', 'precision', 'recall'], 20)}

### Tempo de execução (segundos, ajuste + scoring no estágio de teste)
""" + "\n".join(f"- {EXPERT_LABELS.get(r.model_name, r.model_name)}: {r.value:.2f}s"
                 for r in test[test['metric'] == 'fit_seconds'].itertuples()) + f"""

## 3. Comparações pareadas (ΔNDCG@{k}, bootstrap pareado)

| A | B | Δ (A−B) | IC95% | P(Δ ≤ 0) |
|---|---|---|---|---|
""" + "\n".join(f"| {EXPERT_LABELS.get(p['a'], p['a'])} | {EXPERT_LABELS.get(p['b'], p['b'])} | {p['diff']:+.4f} | "
                 f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] | {p['p_le_zero']:.3f} |" for p in report["paired"]) + f"""

## 4. Por disponibilidade de histórico (NDCG@{k})

{chr(10).join(seg_lines)}

## 5. Seleção de especialistas pelo motor adaptativo

Pesos médios no teste:

{chr(10).join(w_lines)}

Gating treinável — seleção de complexidade por validação cruzada (5 dobras sobre clientes da validação;
empates de ±0,001 favorecem a variante mais simples):

| Variante | λ (L2 de θ) | NDCG@10 fora da dobra |
|---|---|---|
{chr(10).join(cv_lines)}

Variante escolhida: **{lg.get('variant')}** (λ = {lg.get('l2')}).

## 6. Modelo supervisionado — classificação no teste

{chr(10).join(clf_lines)}

Limiar de decisão definido no *holdout* do treino (máximo F1) e reaplicado no teste. Probabilidades calibradas por Platt.

## 7. Clusterização (corte do teste)

{chr(10).join(clus_lines)}

k escolhido = {clus['k']}. Silhouette Ward = {clus['hierarchical']['silhouette_ward']:.3f}; ARI K-Means × Ward = {clus['hierarchical']['ari_kmeans_vs_ward']:.3f}.

## 8. Conclusões (geradas a partir dos números acima)

""" + "\n".join(f"- {c}" for c in report["conclusions"]) + f"""

## 9. Limitações e interpretação

- Os dados são sintéticos: o gerador define a utilidade latente; ganhos aqui não garantem ganhos em dados reais.
- Clientes avaliados: {report['protocol']['n_test_customers']} no teste e {report['protocol']['n_val_customers']} na validação
  (onde o gating é aprendido). A largura dos IC95% acima indica a menor diferença que este tamanho de amostra
  consegue distinguir do ruído.
- O gating treinável é regularizado em direção às regras e só substitui as regras quando vence na validação
  cruzada; quando isso não ocorre, ele reproduz exatamente o gating por regras.
- O especialista bayesiano opera no nível de categoria (por desenho): é útil como sinal de resposta a
  recomendações, mas não discrimina produtos dentro da categoria.
"""
    EVAL_REPORT_PATH.write_text(md, encoding="utf-8")
