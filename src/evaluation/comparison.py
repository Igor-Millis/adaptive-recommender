"""Comparação entre estratégias: rankings, métricas com IC, segmentos e análise de seleção."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import EVAL, EXPERTS
from src.evaluation.metrics import (
    bootstrap_ci,
    catalog_coverage,
    intra_list_diversity,
    paired_bootstrap,
    per_customer_metrics,
)


def rankings_from_scores(S: np.ndarray, mask: np.ndarray, snapshot, k: int = 20) -> dict[int, list[int]]:
    M = np.where(mask, S, -np.inf)
    out = {}
    for i, c in enumerate(snapshot.customer_ids):
        idx = np.argsort(-M[i], kind="stable")[:k]
        out[int(c)] = [int(snapshot.product_ids[j]) for j in idx if np.isfinite(M[i, j])]
    return out


def evaluate_strategy(name: str, S: np.ndarray, stage, similarity: np.ndarray, ks=EVAL.ks, n_boot=EVAL.bootstrap_samples) -> dict:
    snap = stage.snapshot
    rk = rankings_from_scores(S, stage.mask, snap, max(ks))
    res = {"name": name, "rankings": rk, "per_customer": {}, "summary": []}
    for k in ks:
        pcm = per_customer_metrics(rk, stage.relevant, k)
        res["per_customer"][k] = pcm
        for m, v in pcm.items():
            if m == "customers":
                continue
            mean, lo, hi = bootstrap_ci(v, n_boot)
            res["summary"].append({"model_name": name, "metric": m, "k": k, "value": mean, "ci_low": lo, "ci_high": hi,
                                   "n_customers": len(v)})
        cov = catalog_coverage(rk, int(snap.launched.sum()), k)
        res["summary"].append({"model_name": name, "metric": "coverage", "k": k, "value": cov, "ci_low": None,
                               "ci_high": None, "n_customers": len(rk)})
        ild = np.array([intra_list_diversity(rk[c], similarity, snap.pidx, k) for c in pcm["customers"]])
        mean, lo, hi = bootstrap_ci(ild, n_boot)
        res["summary"].append({"model_name": name, "metric": "diversity", "k": k, "value": mean, "ci_low": lo,
                               "ci_high": hi, "n_customers": len(ild)})
    return res


def segment_table(results: dict, segments: dict[str, pd.Series], k: int = EVAL.main_k, metric: str = "ndcg",
                  n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for seg_type, seg in segments.items():
        for name, r in results.items():
            pcm = r["per_customer"][k]
            cust = pcm["customers"]
            vals = pd.Series(pcm[metric], index=cust)
            labels = seg.reindex(cust)
            for val, idx in vals.groupby(labels).groups.items():
                v = vals.loc[idx].to_numpy()
                mean, lo, hi = bootstrap_ci(v, n_boot)
                rows.append({"segment_type": seg_type, "segment_value": str(val), "model_name": name, "metric": metric,
                             "k": k, "value": mean, "ci_low": lo, "ci_high": hi, "n_customers": len(v)})
    return pd.DataFrame(rows)


def paired_comparisons(results: dict, pairs: list[tuple[str, str]], k: int = EVAL.main_k, metric: str = "ndcg",
                       n_boot: int = EVAL.bootstrap_samples) -> list[dict]:
    out = []
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        pa, pb = results[a]["per_customer"][k], results[b]["per_customer"][k]
        assert np.array_equal(pa["customers"], pb["customers"]), "comparação pareada exige mesmos clientes"
        r = paired_bootstrap(pa[metric], pb[metric], n_boot)
        r.update({"a": a, "b": b, "metric": metric, "k": k})
        out.append(r)
    return out


def selection_analysis(results: dict, weights: dict[str, np.ndarray], stage, segment: pd.Series, k: int = EVAL.main_k,
                       experts: list[str] = EXPERTS) -> dict:
    """O gating escolhe bem? Compara pesos com o desempenho real de cada especialista por cliente."""
    cust = results[experts[0]]["per_customer"][k]["customers"]
    perf = pd.DataFrame({e: results[e]["per_customer"][k]["ndcg"] for e in experts}, index=cust)
    oracle = perf.max(axis=1)
    idx = [stage.signals.index.get_loc(c) for c in cust]
    out = {"oracle_ndcg": float(oracle.mean()), "methods": {}}
    for method, W in weights.items():
        Wc = pd.DataFrame(W[idx], index=cust, columns=experts)
        top_w = Wc.idxmax(axis=1)
        best = perf.idxmax(axis=1)
        tie_best = perf.eq(perf.max(axis=1), axis=0)
        agree = np.mean([tie_best.loc[c, top_w.loc[c]] for c in cust])
        # correlação de Spearman (por cliente) entre pesos e NDCG dos especialistas
        cors = []
        for c in cust:
            if perf.loc[c].std() > 0 and Wc.loc[c].std() > 0:
                cors.append(pd.Series(Wc.loc[c]).rank().corr(perf.loc[c].rank()))
        by_seg = Wc.groupby(segment.reindex(cust)).mean()
        out["methods"][method] = {"top_weight_is_best": float(agree), "mean_spearman_weight_vs_ndcg": float(np.nanmean(cors)) if cors else None,
                                  "mean_weights_by_segment": by_seg.round(4).to_dict(orient="index"),
                                  "best_expert_share": best.value_counts(normalize=True).round(3).to_dict()}
    return out
