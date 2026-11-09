"""Score de ranking: relevância para o cliente × objetivo comercial (margem), separados.

    score_rank(c, p) = (1 − μ) · relevância(c, p) · (1 − fadiga(c, p)) + μ · margem_norm(p)

μ = ``margin_weight`` (padrão 0: otimiza apenas relevância). A relevância
continua sendo exibida separadamente — a margem nunca é apresentada como
"interesse do cliente".
"""
from __future__ import annotations

import numpy as np


def margin_scores(margins: np.ndarray) -> np.ndarray:
    m = np.asarray(margins, float)
    rng = np.ptp(m)
    return np.zeros_like(m) if rng < 1e-12 else (m - m.min()) / rng


def ranking_scores(relevance: np.ndarray, margins: np.ndarray, margin_weight: float = 0.0,
                   fatigue: np.ndarray | None = None) -> np.ndarray:
    if not 0.0 <= margin_weight <= 1.0:
        raise ValueError("margin_weight deve estar em [0, 1].")
    rel = relevance if fatigue is None else relevance * (1 - fatigue)
    return (1 - margin_weight) * rel + margin_weight * margin_scores(margins)[None, :]
