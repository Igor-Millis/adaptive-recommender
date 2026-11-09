"""Métricas de ranking, cobertura, diversidade e reamostragem bootstrap.

Relevância binária: rel(c, p) = 1 se p foi comprado por c na janela de avaliação
(e era candidato no corte). Todas as métricas são médias por cliente
(macro-média), avaliadas somente em clientes com ≥ 1 item relevante.
"""
from __future__ import annotations

import numpy as np


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    return len([p for p in ranked[:k] if p in relevant]) / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len([p for p in ranked[:k] if p in relevant]) / len(relevant)


def hit_rate_at_k(ranked: list, relevant: set, k: int) -> float:
    return float(any(p in relevant for p in ranked[:k]))


def dcg_at_k(ranked: list, relevant: set, k: int) -> float:
    return sum(1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg_at_k(ranked, relevant, k) / ideal


def average_precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits, s = 0, 0.0
    for i, p in enumerate(ranked[:k]):
        if p in relevant:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(relevant), k)


RANKING_METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def per_customer_metrics(rankings: dict[int, list], relevant: dict[int, set], k: int) -> dict[str, np.ndarray]:
    """Vetores de métricas por cliente (mesma ordem de ``sorted(relevant)``)."""
    cust = sorted(c for c in relevant if relevant[c])
    out = {m: np.array([f(rankings.get(c, []), relevant[c], k) for c in cust]) for m, f in RANKING_METRICS.items()}
    out["customers"] = np.array(cust)
    return out


def catalog_coverage(rankings: dict[int, list], n_catalog: int, k: int) -> float:
    items = {p for r in rankings.values() for p in r[:k]}
    return len(items) / n_catalog if n_catalog else 0.0


def intra_list_diversity(ranked: list, similarity, pidx: dict, k: int) -> float:
    """1 − similaridade média entre pares da lista (similaridade de conteúdo em [0, 1])."""
    idx = [pidx[p] for p in ranked[:k] if p in pidx]
    if len(idx) < 2:
        return 0.0
    S = similarity[np.ix_(idx, idx)]
    n = len(idx)
    return float(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Diferença média pareada (a − b) por cliente com IC bootstrap e P(diferença ≤ 0)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_zero": float((means <= 0).mean()), "n": int(len(d))}
