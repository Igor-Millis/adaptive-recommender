"""Protocolo de validação temporal.

    treino  : dados < VAL_START         → especialistas do estágio A
    validação: [VAL_START, TEST_START)   → seleção/calibração, gating treinável, pesos fixos
    teste   : [TEST_START, DATA_END)     → especialistas re-ajustados com dados < TEST_START,
                                           gating congelado do estágio A; avaliação final

Nenhum componente avaliado no teste viu dados ≥ TEST_START. Estoque não é usado
como filtro na avaliação (é um estado atual, não histórico — usá-lo seria vazamento).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import EXPERTS
from src.adaptive.ensemble import normalize_expert_scores
from src.adaptive.model_selection import compute_signals
from src.recommenders.suite import ExpertSuite
from src.recommenders.supervised import SupervisedRecommender, candidate_mask
from src.utils.validation import validate_temporal_split


@dataclass
class Stage:
    name: str
    cutoff: pd.Timestamp
    end: pd.Timestamp
    suite: ExpertSuite
    signals: pd.DataFrame
    mask: np.ndarray
    Y: np.ndarray
    relevant: dict
    norm: dict
    active: np.ndarray
    timings: dict = field(default_factory=dict)

    @property
    def snapshot(self):
        return self.suite.snapshot


def build_stage(dataset, name: str, cutoff: pd.Timestamp, end: pd.Timestamp, supervised: SupervisedRecommender,
                fixed_k: int | None = None) -> Stage:
    sup_cut = pd.Timestamp(supervised.report["cutoff"])
    validate_temporal_split(sup_cut, cutoff, end)
    snap = dataset.snapshot(cutoff)
    t0 = time.perf_counter()
    suite = ExpertSuite(supervised).fit(snap, fixed_k=fixed_k)
    timings = dict(suite.timings)
    timings["total_fit"] = time.perf_counter() - t0
    mask = candidate_mask(snap)
    win = dataset.window_items(cutoff, end)
    Y = np.zeros_like(mask)
    if len(win):
        Y[win["customer_id"].map(snap.cidx).to_numpy(), win["product_id"].map(snap.pidx).to_numpy()] = True
    Y &= mask
    active = np.zeros(snap.n_customers, dtype=bool)
    if len(win):
        active[win["customer_id"].map(snap.cidx).unique()] = True
    relevant = {int(c): {int(snap.product_ids[j]) for j in np.where(Y[i])[0]}
                for i, c in enumerate(snap.customer_ids) if Y[i].any()}
    names = [e for e in EXPERTS + ["collaborative_mf"] if e in suite.experts]
    norm = normalize_expert_scores(suite.scores(names), mask)
    return Stage(name, cutoff, end, suite, compute_signals(suite), mask, Y, relevant, norm, active, timings)
