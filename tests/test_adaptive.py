"""Motor adaptativo: normalização de pesos, regras, gating treinável e combinação."""
import numpy as np
import pytest

from config.settings import EXPERTS
from src.adaptive.ensemble import combine, normalize_expert_scores
from src.adaptive.gating import FixedGating, LearnedGating, RuleGating, _simplex_grid
from src.adaptive.model_selection import rule_reliabilities
from src.preprocessing.transformations import rank_normalize
from src.utils.validation import ValidationError, validate_weights


def _check(W):
    assert np.all(W >= 0)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_rule_weights_valid(stage_val):
    _check(RuleGating().weights(stage_val.signals))


def test_fixed_weights_respect_availability(stage_val):
    W = FixedGating({e: 1.0 for e in EXPERTS}).weights(stage_val.signals)
    _check(W)
    no_hist = stage_val.signals["n_distinct_products"].to_numpy() == 0
    assert np.allclose(W[no_hist, EXPERTS.index("collaborative")], 0.0)


def test_rules_differ_between_customers(stage_val):
    W = RuleGating().weights(stage_val.signals)
    assert W.std(axis=0).max() > 0.02  # pesos diferentes para clientes diferentes


def test_cold_start_rules(stage_val):
    sig = stage_val.signals
    R = rule_reliabilities(sig)
    cold = sig["n_orders"] == 0
    assert (R.loc[cold, "collaborative"] == 0).all()
    W = RuleGating().weights(sig)
    pop = EXPERTS.index("popularity")
    assert W[cold.to_numpy(), pop].mean() > W[~cold.to_numpy(), pop].mean()


def test_validate_weights_rejects_invalid():
    with pytest.raises(ValidationError):
        validate_weights(np.array([[0.5, 0.6]]))
    with pytest.raises(ValidationError):
        validate_weights(np.array([[1.2, -0.2]]))


def test_learned_gating_zero_params_equals_rules(stage_val):
    g = LearnedGating()
    g.mu = stage_val.signals[g.features].mean().to_numpy()
    g.sd = stage_val.signals[g.features].std().to_numpy() + 1e-9
    assert np.allclose(g.weights(stage_val.signals), RuleGating().weights(stage_val.signals))


def test_learned_gating_gradient(stage_val):
    g = LearnedGating()
    sig = stage_val.signals
    X = sig[g.features].to_numpy(float)
    g.mu, g.sd = X.mean(0), X.std(0) + 1e-9
    Phi = (X - g.mu) / g.sd
    R = rule_reliabilities(sig).to_numpy()
    logR = np.where(R > 0, np.log(R + 1e-9), -1e4)
    norm = {e: stage_val.norm[e] for e in EXPERTS}
    stack, rows = g._prepare(norm, stage_val.Y, stage_val.mask, EXPERTS)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, len(EXPERTS) * (1 + Phi.shape[1]))
    f0, grad = g._loss_grad(x, stack, stage_val.Y, stage_val.mask, logR, Phi, rows, 0.5)
    for idx in rng.choice(len(x), 6, replace=False):
        e = np.zeros_like(x)
        e[idx] = 1e-5
        num = (g._loss_grad(x + e, stack, stage_val.Y, stage_val.mask, logR, Phi, rows, 0.5)[0] - f0) / 1e-5
        assert num == pytest.approx(grad[idx], rel=1e-2, abs=1e-5)


def test_learned_gating_fit_and_roundtrip(stage_val):
    norm = {e: stage_val.norm[e] for e in EXPERTS}
    g = LearnedGating().fit(stage_val.signals, norm, stage_val.Y, stage_val.mask, l2_grid=(1.0,), n_folds=3)
    _check(g.weights(stage_val.signals))
    g2 = LearnedGating.from_dict(g.to_dict())
    assert np.allclose(g.weights(stage_val.signals), g2.weights(stage_val.signals))
    assert g.train_info["variant"] in {"regras", "viés", "viés+contexto"}


def test_combine_bounds(stage_val):
    norm = {e: stage_val.norm[e] for e in EXPERTS}
    W = RuleGating().weights(stage_val.signals)
    S, contrib = combine(norm, W)
    assert S.min() >= -1e-12 and S.max() <= 1 + 1e-12
    assert np.allclose(contrib.sum(axis=1), S)


def test_rank_normalization():
    S = np.array([[3.0, 1.0, 2.0, 2.0], [5.0, 5.0, 5.0, 5.0]])
    N = rank_normalize(S)
    assert N[0, 0] == 1.0 and N[0, 1] == 0.0 and N[0, 2] == N[0, 3]
    assert np.allclose(N[1], 0.5)  # especialista sem informação não altera a ordem


def test_simplex_grid():
    g = _simplex_grid(5, 10)
    assert len(g) == 1001
    assert all(abs(w.sum() - 1) < 1e-9 and (w >= 0).all() for w in g)


def test_normalize_rejects_nan():
    with pytest.raises(ValidationError):
        normalize_expert_scores({"x": np.array([[np.nan, 1.0]])}, np.ones((1, 2), bool))
