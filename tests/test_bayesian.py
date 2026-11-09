"""Modelo Beta-Binomial: fórmulas, tipos de evidência, Bayes empírico e consistência incremental × lote."""
import itertools

import numpy as np
import pandas as pd
import pytest

from config.settings import BAYES
from src.learning.bayesian_updates import evidence_delta, resolve_events
from src.recommenders.bayesian import BayesianCategoryModel, beta_summary, fit_empirical_bayes, outcome_evidence


def test_posterior_mean_formula():
    a, b, s, f = 2.0, 8.0, 3.0, 5.0
    post = beta_summary(a + s, b + f)
    assert post["mean"] == pytest.approx((a + s) / (a + b + s + f))
    assert post["ci_low"] < post["mean"] < post["ci_high"]


def test_credible_interval_shrinks_with_evidence():
    w1 = beta_summary(2 + 3, 8 + 7)
    w2 = beta_summary(2 + 30, 8 + 70)
    assert (w2["ci_high"] - w2["ci_low"]) < (w1["ci_high"] - w1["ci_low"])


def test_purchase_stronger_than_click():
    p, c = outcome_evidence("purchase"), outcome_evidence("click")
    assert p["interest"] == c["interest"] == (1.0, 0.0)
    assert p["conversion"][0] > c["conversion"][0]


def test_impression_is_not_conversion_failure():
    ev = outcome_evidence("impression")
    assert "conversion" not in ev
    assert ev["interest"] == (0.0, BAYES.impression_failure_weight)
    assert ev["interest"][1] < outcome_evidence("rejection")["interest"][1]


def test_empirical_bayes_recovers_mean():
    rng = np.random.default_rng(0)
    theta = rng.beta(4, 16, size=400)  # média 0,2, κ = 20
    n = rng.integers(5, 40, size=400)
    s = rng.binomial(n, theta).astype(float)
    mu, kappa = fit_empirical_bayes(s, n - s)
    assert mu == pytest.approx(0.2, abs=0.03)
    assert 8 < kappa < 50


EVENTS = ["impression", "click", "interest", "add_to_cart", "purchase", "rejection"]


@pytest.mark.parametrize("seq", [list(p) for r in (1, 2, 3) for p in itertools.product(EVENTS, repeat=r)])
def test_incremental_deltas_equal_batch(seq):
    """Σ Δ(evento) ao longo de uma sessão = evidência do desfecho resolvido final (propriedade de consistência)."""
    total = {"interest": np.zeros(2), "conversion": np.zeros(2)}
    prev, seen = None, []
    for e in seq:
        seen.append(e)
        new = resolve_events(seen)
        for st, (ds, df) in evidence_delta(prev, new).items():
            total[st] += (ds, df)
        prev = new
    final = outcome_evidence(resolve_events(seq))
    for st in total:
        assert np.allclose(total[st], final.get(st, (0.0, 0.0)))


def test_bayesian_model_posteriors(suite_val):
    m: BayesianCategoryModel = suite_val.experts["bayesian"]
    for st in ("interest", "conversion"):
        assert np.all(m.alpha[st] > 0) and np.all(m.beta[st] > 0)
        assert np.allclose(m.alpha[st], m.alpha0[st] + m.E[st][..., 0])
    assert np.all((m.purchase_mean >= 0) & (m.purchase_mean <= 1))


def test_state_roundtrip(suite_val):
    m = suite_val.experts["bayesian"]
    m2 = BayesianCategoryModel().load_state(suite_val.snapshot, suite_val.ctx, m.state_frame())
    assert np.allclose(m.purchase_mean, m2.purchase_mean)


def test_trajectory_ends_at_posterior(suite_val):
    m = suite_val.experts["bayesian"]
    s = suite_val.snapshot
    out = suite_val.ctx.outcomes
    cid = int(out["customer_id"].value_counts().index[0])
    cat = out[out["customer_id"] == cid]["category"].value_counts().index[0]
    traj = m.trajectory(cid, cat)
    i, k = s.cidx[cid], list(m.snapshot.products["category"].unique())
    from config.settings import CATEGORIES

    kk = CATEGORIES.index(cat)
    assert traj.iloc[-1]["interest_alpha"] == pytest.approx(m.alpha["interest"][i, kk])
    assert traj.iloc[-1]["interest_beta"] == pytest.approx(m.beta["interest"][i, kk])
