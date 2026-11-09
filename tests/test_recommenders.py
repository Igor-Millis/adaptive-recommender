"""Funcionamento dos especialistas, prevenção de vazamento e tratamento de cold start."""
import numpy as np
import pandas as pd
import pytest

from config.settings import DATA_END, HORIZON_DAYS, VAL_START
from src.preprocessing.feature_engineering import build_customer_features
from src.recommenders.supervised import build_labeled_snapshot, candidate_mask, snapshot_dates
from src.segmentation.clustering import fit_segmentation
from src.utils.validation import ValidationError


def test_all_expert_scores_valid(suite_val):
    s = suite_val.snapshot
    for name, S in suite_val.scores().items():
        assert S.shape == (s.n_customers, s.n_products), name
        assert np.all(np.isfinite(S)), name
    for name, c in suite_val.confidences().items():
        assert np.all((c >= 0) & (c <= 1)), name


def test_supervised_probabilities_in_unit_interval(suite_val):
    P = suite_val.experts["supervised"].score()
    assert P.min() >= 0 and P.max() <= 1


def test_cold_start_handling(suite_val):
    s = suite_val.snapshot
    f = suite_val.ctx.customer_features
    # sem histórico = sem pedidos E sem interações (quem só engajou com recomendações já tem sinal colaborativo)
    cold = [c for c in s.customer_ids if f.loc[c, "n_orders"] == 0 and f.loc[c, "n_impressions"] == 0]
    assert cold, "deve haver clientes sem histórico no corte de validação"
    cf = suite_val.experts["collaborative"]
    for c in cold:
        i = s.cidx[c]
        assert np.allclose(cf.score()[i], 0) and cf.confidence()[i] == 0
        # Content-Based usa o cadastro (perfil declarado), não fica vazio
        assert np.linalg.norm(suite_val.experts["content_based"].profiles[i]) > 0
        # supervisionado ainda produz probabilidades diferenciadas
        assert np.ptp(suite_val.experts["supervised"].score()[i]) > 0


def test_content_based_explanation(suite_val):
    cb = suite_val.experts["content_based"]
    s = suite_val.snapshot
    cid = int(s.customer_ids[0])
    top = cb.top_n(cid, 1)[0]
    contrib = cb.feature_contributions(cid, top, top=200)
    assert contrib["share"].sum() <= 1.0 + 1e-9
    assert "summary" in cb.explain(cid, top)


def test_collaborative_neighbors_symmetric(suite_val):
    S = suite_val.experts["collaborative"].S
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 0)


def test_candidate_mask_excludes_purchased_and_unlaunched(snapshot_val):
    m = candidate_mask(snapshot_val)
    assert not (m & snapshot_val.purchased_matrix()).any()
    assert not m[:, ~snapshot_val.launched].any()


def test_snapshot_rejects_future_rows(dataset):
    from src.preprocessing.snapshot import DataSnapshot

    s = dataset.snapshot(VAL_START)
    with pytest.raises(ValidationError):
        DataSnapshot(as_of=VAL_START - pd.Timedelta(days=30), customers=s.customers, products=s.products,
                     items=s.items, interactions=s.interactions)


def test_labeled_snapshot_rejects_window_after_cutoff(dataset):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_labeled_snapshot(dataset, VAL_START - pd.Timedelta(days=10), HORIZON_DAYS, 5, None, rng, VAL_START)


def test_snapshot_dates_respect_horizon():
    for t in snapshot_dates(VAL_START):
        assert t + pd.Timedelta(days=HORIZON_DAYS) <= VAL_START


def test_clustering_quality(dataset):
    snap = dataset.snapshot(DATA_END)
    seg = fit_segmentation(build_customer_features(snap), snap.as_of)
    assert 3 <= seg.k <= 8
    assert seg.silhouette > 0.1
    assert set(seg.diagnostics.columns) >= {"inertia", "silhouette", "stability"}
    assert (seg.labels == -1).sum() >= 1  # clientes sem histórico não recebem cluster arbitrário
    assert len(set(seg.label_names.values())) == seg.k  # rótulos únicos e interpretáveis
