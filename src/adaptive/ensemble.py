"""Combinação dos especialistas (Mixture of Experts) e explicação dos pesos.

    s(c, p) = Σ_e w_e(c) · s̃_e(c, p),   w_e ≥ 0,  Σ_e w_e = 1

em que s̃_e é o score do especialista normalizado por percentil dentro dos
candidatos do cliente (escala comum [0, 1], robusta a escalas e caudas
diferentes entre modelos).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import EXPERT_LABELS, EXPERTS
from src.adaptive.model_selection import SIGNAL_LABELS, rule_justification
from src.preprocessing.transformations import minmax_normalize, rank_normalize
from src.utils.validation import validate_scores, validate_weights


def normalize_expert_scores(raw: dict[str, np.ndarray], mask: np.ndarray, method: str = "rank") -> dict[str, np.ndarray]:
    fn = rank_normalize if method == "rank" else minmax_normalize
    return {e: fn(validate_scores(S, e), mask) for e, S in raw.items()}


def combine(norm: dict[str, np.ndarray], W: np.ndarray, experts: list[str] = EXPERTS) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (score final n_c×n_p, contribuições n_c×E×n_p)."""
    validate_weights(W)
    stack = np.stack([norm[e] for e in experts], axis=1)
    contrib = W[:, :, None] * stack
    return contrib.sum(axis=1), contrib


@dataclass
class GatingExplanation:
    weights: dict
    rule_weights: dict
    method: str
    justification: list
    discarded: list
    confidence_level: str
    confidence_value: float
    key_signals: list


def explain_weights(customer_id: int, signals: pd.DataFrame, W: np.ndarray, W_rules: np.ndarray, confidences: dict,
                    method: str, adjustments: pd.DataFrame | None = None, experts: list[str] = EXPERTS) -> GatingExplanation:
    i = signals.index.get_loc(customer_id)
    w = dict(zip(experts, map(float, W[i])))
    wr = dict(zip(experts, map(float, W_rules[i])))
    sig = signals.loc[customer_id]
    just = rule_justification(sig, w)
    if method == "learned" and adjustments is not None:
        adj = adjustments.loc[customer_id]
        ups = [EXPERT_LABELS[e] for e in experts if adj[e] > 0.25 and w[e] > 0.02]
        downs = [EXPERT_LABELS[e] for e in experts if adj[e] < -0.25 and wr[e] > 0.02]
        if ups or downs:
            txt = "O gating treinado (aprendido numa janela passada com resultados já observados) ajustou as regras"
            if ups:
                txt += f": aumentou {', '.join(ups)}"
            if downs:
                txt += ("; " if ups else ": ") + f"reduziu {', '.join(downs)}"
            just.append(txt + ", com base no desempenho observado desses especialistas em clientes com sinais semelhantes.")
    conf = sum(w[e] * float(np.clip(confidences[e][i], 0, 1)) for e in experts if e in confidences)
    level = "Alta" if conf >= 0.6 else "Média" if conf >= 0.35 else "Baixa"
    discarded = [e for e in experts if w[e] < 0.05]
    key = [{"sinal": SIGNAL_LABELS.get(c, c), "valor": float(sig[c])} for c in
           ["n_orders", "n_distinct_products", "n_impressions", "neighbor_strength", "consistency", "drift", "bayes_evidence", "recency_days"]]
    return GatingExplanation(w, wr, method, just, discarded, level, float(conf), key)
