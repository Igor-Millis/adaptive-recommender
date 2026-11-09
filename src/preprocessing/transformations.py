"""Transformações numéricas reutilizáveis (entropia, divergências, normalizações)."""
from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray, axis: int = -1, normalized: bool = True) -> np.ndarray:
    """Entropia de Shannon; normalizada por log(K) quando ``normalized``."""
    p = np.asarray(p, dtype=float)
    s = p.sum(axis=axis, keepdims=True)
    q = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(q > 0, q * np.log(q), 0.0), axis=axis)
    if normalized:
        h = h / np.log(p.shape[axis])
    return h


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Divergência de Jensen-Shannon (base 2, em [0, 1])."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X, dtype=float), where=n > 0)


def rank_normalize(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Percentil (empates com rank médio) por linha, em [0, 1], restrito à máscara de candidatos.

    Linhas constantes (especialista sem informação) recebem 0,5 — não alteram a ordenação.
    Posições fora da máscara recebem 0.
    """
    S = np.asarray(scores, dtype=float)
    out = np.zeros_like(S)
    if mask is None:
        mask = np.ones_like(S, dtype=bool)
    for i in range(S.shape[0]):
        m = mask[i]
        n = int(m.sum())
        if n == 0:
            continue
        v = S[i, m]
        if np.ptp(v) < 1e-12:
            out[i, m] = 0.5
            continue
        order = v.argsort(kind="mergesort")
        ranks = np.empty(n)
        ranks[order] = np.arange(n, dtype=float)
        # média para empates
        uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if len(uniq) < n:
            sums = np.zeros(len(uniq))
            np.add.at(sums, inv, ranks)
            ranks = sums[inv] / counts[inv]
        out[i, m] = ranks / (n - 1) if n > 1 else 0.5
    return out


def minmax_normalize(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    S = np.asarray(scores, dtype=float)
    out = np.zeros_like(S)
    if mask is None:
        mask = np.ones_like(S, dtype=bool)
    for i in range(S.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        v = S[i, m]
        rng = np.ptp(v)
        out[i, m] = 0.5 if rng < 1e-12 else (v - v.min()) / rng
    return out


def exp_decay(age_days: np.ndarray, half_life: float) -> np.ndarray:
    return np.power(0.5, np.asarray(age_days, float) / half_life)


def circular_seasonality(month_counts: np.ndarray) -> tuple[float, int]:
    """Concentração sazonal pela 1ª harmônica (estatística circular), com correção de viés.

    R̄ = |Σ n_m e^{iθ_m}| / n.  Sob uniformidade E[n R̄²] ≈ 1, então usamos
    S = max(0, (n R̄² − 1)/(n − 1)) ∈ [0, 1]. Retorna (S, mês de pico 1..12).
    """
    n_m = np.asarray(month_counts, float)
    n = n_m.sum()
    if n < 4:
        return 0.0, 0
    theta = 2 * np.pi * np.arange(12) / 12
    c = np.sum(n_m * np.cos(theta))
    s = np.sum(n_m * np.sin(theta))
    r2 = (c * c + s * s) / (n * n)
    stat = max(0.0, (n * r2 - 1) / (n - 1))
    peak = int(round((np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 12)) % 12 + 1
    return float(stat), peak
