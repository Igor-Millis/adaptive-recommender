"""Construção do contexto compartilhado (features, desfechos, segmentação) de um snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import HEMISPHERE
from src.preprocessing.feature_engineering import build_customer_features, build_product_features, resolve_outcomes
from src.preprocessing.snapshot import DataSnapshot
from src.recommenders.base import ExpertContext
from src.segmentation.clustering import SegmentationResult, fit_segmentation


def build_context(snapshot: DataSnapshot, segmentation: SegmentationResult | None = None, fixed_k: int | None = None,
                  fit_clusters: bool = True, fast: bool = False) -> tuple[ExpertContext, SegmentationResult | None]:
    cf = build_customer_features(snapshot)
    pf = build_product_features(snapshot)
    outcomes = resolve_outcomes(snapshot.interactions)
    seg = segmentation
    if seg is None and fit_clusters:
        k_range = [fixed_k] if (fast and fixed_k) else range(2, 11)
        seg = fit_segmentation(cf, snapshot.as_of, k_range=k_range, fixed_k=fixed_k)
    if seg is not None:
        labels = seg.labels.reindex(snapshot.customer_ids).fillna(-1).astype(int)
        names = seg.label_names
    else:
        labels = pd.Series(-1, index=snapshot.customer_ids, dtype=int)
        names = {}
    return ExpertContext(customer_features=cf, product_features=pf, outcomes=outcomes, cluster_labels=labels,
                         cluster_names=names), seg


def summer_score(start: pd.Timestamp, days: int) -> float:
    """Média de cos(2π(m−1)/12) nos meses da janela: +1 verão pleno, −1 inverno pleno."""
    months = pd.date_range(start, start + pd.Timedelta(days=days), freq="MS").month
    if len(months) == 0:
        months = [start.month]
    shift = 6 if HEMISPHERE == "north" else 0  # no hemisfério norte o verão é em julho
    return float(np.mean(np.cos(2 * np.pi * (np.asarray(months) - 1 - shift) / 12.0)))
