"""Atualização bayesiana incremental (conjugada) a partir de um novo evento.

A evidência é definida sobre o *desfecho resolvido* de cada (sessão, produto).
Quando chega um novo evento, aplica-se a diferença

    Δ = evidência(desfecho_novo) − evidência(desfecho_anterior)

Assim, registrar "impressão" e depois "clique" na mesma sessão resulta
exatamente no mesmo estado que processar o histórico em lote (propriedade
verificada em ``tests/test_bayesian.py``). Como Beta é conjugada à Binomial, a
atualização é apenas α ← α + Δs, β ← β + Δf.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from config.settings import BAYES
from src.preprocessing.feature_engineering import OUTCOME_ORDER, POSITIVE_OUTCOMES
from src.recommenders.bayesian import STAGES, outcome_evidence


def resolve_events(types: list[str]) -> str | None:
    """Desfecho mais forte de uma lista de eventos (mesma regra de ``resolve_outcomes``)."""
    if not types:
        return None
    pos = [t for t in types if t in POSITIVE_OUTCOMES]
    if pos:
        return max(pos, key=lambda t: OUTCOME_ORDER[t])
    if "rejection" in types:
        return "rejection"
    return "impression"


def evidence_delta(prev_outcome: str | None, new_outcome: str | None, cfg=BAYES) -> dict[str, tuple[float, float]]:
    old = outcome_evidence(prev_outcome, cfg) if prev_outcome else {}
    new = outcome_evidence(new_outcome, cfg) if new_outcome else {}
    out = {}
    for st in STAGES:
        s0, f0 = old.get(st, (0.0, 0.0))
        s1, f1 = new.get(st, (0.0, 0.0))
        if (s1 - s0) != 0 or (f1 - f0) != 0:
            out[st] = (s1 - s0, f1 - f0)
    return out


def apply_update(repos, conn, customer_id: int, category: str, delta: dict, interaction_id: int | None,
                 reason: str, cfg=BAYES) -> list[dict]:
    """Aplica Δ ao estado persistido e registra o log de auditoria. Retorna antes/depois por estágio."""
    changes = []
    for st, (ds, df) in delta.items():
        cell = repos.bayes.get_cell(customer_id, category, st, conn=conn)
        if cell is None:
            raise RuntimeError(f"Estado bayesiano ausente para cliente {customer_id}/{category}/{st}: execute o treinamento.")
        a0, b0 = float(cell["alpha"]), float(cell["beta"])
        a1 = max(a0 + ds, float(cell["alpha_prior"]) * 1e-3)
        b1 = max(b0 + df, float(cell["beta_prior"]) * 1e-3)
        lo, hi = stats.beta.ppf([(1 - cfg.credible_level) / 2, (1 + cfg.credible_level) / 2], a1, b1)
        repos.bayes.upsert_cell({"customer_id": customer_id, "category": category, "stage": st,
                                 "alpha_prior": cell["alpha_prior"], "beta_prior": cell["beta_prior"], "alpha": a1, "beta": b1,
                                 "n_events": int(cell["n_events"]) + 1, "updated_at": pd.Timestamp.now(),
                                 "model_version_id": cell.get("model_version_id")}, conn)
        repos.bayes.log_update({"created_at": pd.Timestamp.now(), "customer_id": customer_id, "category": category, "stage": st,
                                "interaction_id": interaction_id, "alpha_before": a0, "beta_before": b0, "alpha_after": a1,
                                "beta_after": b1, "mean_after": a1 / (a1 + b1), "ci_low": float(lo), "ci_high": float(hi),
                                "evidence_s": ds, "evidence_f": df, "reason": reason}, conn)
        changes.append({"stage": st, "alpha_before": a0, "beta_before": b0, "alpha_after": a1, "beta_after": b1,
                        "mean_before": a0 / (a0 + b0), "mean_after": a1 / (a1 + b1), "delta_s": ds, "delta_f": df,
                        "ci_low": float(lo), "ci_high": float(hi)})
    return changes
