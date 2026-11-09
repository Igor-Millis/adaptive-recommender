"""Diversificação do ranking por Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    p* = argmax_{p ∉ L} [ λ · score(p) − (1 − λ) · max_{q ∈ L} sim(p, q) ]

λ = 1 reproduz o ranking por score. Opcionalmente limita itens por categoria.
"""
from __future__ import annotations

import numpy as np


def mmr_rank(scores: np.ndarray, candidates: np.ndarray, similarity: np.ndarray, k: int, lam: float = 1.0,
             categories: np.ndarray | None = None, max_per_category: int | None = None) -> list[int]:
    cand = [int(j) for j in np.where(candidates)[0]]
    if not cand:
        return []
    s = np.asarray(scores, float)
    if lam >= 1.0 and max_per_category is None:
        order = sorted(cand, key=lambda j: -s[j])
        return order[:k]
    # normaliza scores para [0,1] entre candidatos para que λ seja interpretável
    v = s[cand]
    rng = np.ptp(v)
    sn = {j: (s[j] - v.min()) / rng if rng > 0 else 0.5 for j in cand}
    selected: list[int] = []
    per_cat: dict = {}
    remaining = set(cand)
    while remaining and len(selected) < k:
        best, best_val = None, -np.inf
        for j in remaining:
            if max_per_category and categories is not None and per_cat.get(categories[j], 0) >= max_per_category:
                continue
            red = max((similarity[j, q] for q in selected), default=0.0)
            val = lam * sn[j] - (1 - lam) * red
            if val > best_val:
                best, best_val = j, val
        if best is None:
            break
        selected.append(best)
        remaining.discard(best)
        if categories is not None:
            per_cat[categories[best]] = per_cat.get(categories[best], 0) + 1
    return selected
