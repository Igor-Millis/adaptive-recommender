"""Validações de invariantes estatísticas e de integridade.

Essas funções são chamadas nos pontos críticos do pipeline (combinação de
especialistas, saída de probabilidades, ranking e particionamento temporal) e
levantam ``ValidationError`` em vez de deixar valores inválidos se propagarem.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Violação de uma invariante do sistema."""


def validate_weights(weights: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Pesos devem ser finitos, não negativos e somar 1 (por linha)."""
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValidationError("Pesos contêm valores não finitos.")
    if np.any(w < -atol):
        raise ValidationError(f"Pesos negativos encontrados: min={w.min():.4g}")
    sums = w.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol):
        raise ValidationError(f"Pesos não somam 1: {np.round(sums, 6)}")
    return np.clip(w, 0.0, None)


def validate_probabilities(p: np.ndarray, name: str = "probabilidades") -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name}: valores não finitos.")
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValidationError(f"{name}: valores fora de [0, 1] (min={arr.min():.4g}, max={arr.max():.4g}).")
    return arr


def validate_scores(s: np.ndarray, name: str = "scores") -> np.ndarray:
    arr = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name}: valores não finitos (NaN/inf).")
    return arr


def validate_product_ids(recommended: Iterable[int], valid_ids: Iterable[int]) -> None:
    rec = set(int(x) for x in recommended)
    valid = set(int(x) for x in valid_ids)
    missing = rec - valid
    if missing:
        raise ValidationError(f"Produtos inexistentes recomendados: {sorted(missing)[:10]}")


def validate_no_future(df: pd.DataFrame, date_col: str, cutoff: pd.Timestamp, name: str = "dados") -> None:
    """Garante que nenhum registro usado para treino é posterior ao corte."""
    if df.empty:
        return
    max_date = pd.to_datetime(df[date_col]).max()
    if max_date >= cutoff:
        raise ValidationError(f"Vazamento temporal em {name}: registro em {max_date} >= corte {cutoff}.")


def validate_temporal_split(train_end: pd.Timestamp, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> None:
    if not (train_end <= eval_start < eval_end):
        raise ValidationError(
            f"Partição temporal inválida: treino até {train_end}, avaliação [{eval_start}, {eval_end})."
        )
