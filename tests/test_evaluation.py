"""Métricas de avaliação, ranking e partição temporal."""
import numpy as np
import pytest

from config.settings import RANKING
from src.evaluation.metrics import (
    average_precision_at_k,
    bootstrap_ci,
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
)
from src.ranking.diversification import mmr_rank
from src.ranking.filtering import candidate_mask
from src.ranking.scoring import ranking_scores
from src.utils.validation import ValidationError, validate_temporal_split


def test_precision_recall():
    assert precision_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == 0.5
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == pytest.approx(2 / 3)


def test_ndcg_known_values():
    assert ndcg_at_k([1, 2, 3], {1}, 3) == 1.0
    assert ndcg_at_k([2, 1, 3], {1}, 3) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k([5, 6], {1}, 2) == 0.0


def test_map():
    assert average_precision_at_k([1, 9, 2], {1, 2}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_coverage_and_diversity():
    assert catalog_coverage({1: [1, 2], 2: [2, 3]}, 10, 2) == 0.3
    sim = np.eye(3)
    assert intra_list_diversity([0, 1, 2], sim, {0: 0, 1: 1, 2: 2}, 3) == 1.0


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(1, 1, 200)
    m, lo, hi = bootstrap_ci(v)
    assert lo < m < hi


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 100)
    r = paired_bootstrap(a + 0.05, a)
    assert r["ci_low"] > 0


def test_temporal_split_validation():
    import pandas as pd

    with pytest.raises(ValidationError):
        validate_temporal_split(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-01"))


def test_mmr_lambda_one_is_sort():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert mmr_rank(s, np.ones(4, bool), np.eye(4), 3, 1.0) == [1, 3, 2]


def test_mmr_category_cap():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    cats = np.array(["A", "A", "A", "B"])
    out = mmr_rank(s, np.ones(4, bool), np.eye(4), 3, 1.0, cats, max_per_category=2)
    assert list(cats[out]).count("A") <= 2 and 3 in out


def test_mmr_diversifies():
    s = np.array([1.0, 0.99, 0.5])
    sim = np.array([[1, 0.99, 0], [0.99, 1, 0], [0, 0, 1.0]])
    assert mmr_rank(s, np.ones(3, bool), sim, 2, 0.3) == [0, 2]


def test_margin_is_separate_objective():
    rel = np.array([[0.9, 0.1]])
    margins = np.array([0.1, 0.9])
    assert np.argmax(ranking_scores(rel, margins, 0.0)) == 0
    assert np.argmax(ranking_scores(rel, margins, 1.0)) == 1
    with pytest.raises(ValueError):
        ranking_scores(rel, margins, 1.5)


def test_unavailable_products_excluded(snapshot_val):
    mask, reasons = candidate_mask(snapshot_val, RANKING)
    assert not mask[:, ~snapshot_val.in_stock].any()
    assert not mask[:, ~snapshot_val.launched].any()
    assert reasons["sem_estoque"].shape == (snapshot_val.n_customers,)
