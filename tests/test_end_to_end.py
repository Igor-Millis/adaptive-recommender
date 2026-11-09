"""Teste de ponta a ponta: banco → treino → recomendação → feedback → atualização → retreino."""
import numpy as np
import pandas as pd
import pytest

from config.settings import CATEGORIES
from database.repositories import Repositories
from src.learning.feedback import FeedbackService
from src.learning.retraining import retrain_all
from src.preprocessing.snapshot import Dataset
from src.recommenders.bayesian import BayesianCategoryModel
from src.services.recommendation_service import RecommendationEngine
from src.utils.validation import validate_weights


@pytest.mark.slow
def test_full_cycle(temp_engine, tmp_path):
    repos = Repositories(temp_engine)
    n_items_before = repos.order_items.count()
    n_inter_before = repos.interactions.count()

    rep = retrain_all(temp_engine, train_gating=False, trained_dir=tmp_path / "trained")
    assert rep["supervised"]["algorithm"] in {"logistic_regression", "random_forest", "lightgbm"}
    assert repos.order_items.count() == n_items_before  # treino não altera dados operacionais

    eng = RecommendationEngine(temp_engine)
    valid_ids = set(eng.snapshot.product_ids.tolist())

    # recomendação para todos os clientes (inclusive sem histórico)
    for cid in eng.snapshot.customer_ids:
        res = eng.recommend(int(cid), method="rules", log_run=False)
        assert len(res.items) == 10
        assert set(res.items["product_id"]) <= valid_ids
        assert (res.items["stock"] > 0).all()
        assert res.items["conversion_probability"].between(0, 1).all()
        validate_weights(np.array([list(res.explanation.weights.values())]))
        bought = set(eng.snapshot.items.loc[eng.snapshot.items["customer_id"] == cid, "product_id"])
        assert not (set(res.items["product_id"]) & bought)

    # ciclo de feedback
    cid = int(eng.snapshot.customer_ids[3])
    res = eng.recommend(cid, method="rules")
    assert res.run_id is not None and len(res.recommendation_ids) == 10
    pid, rid = int(res.items["product_id"].iloc[0]), int(res.recommendation_ids[0])
    cat = eng.snapshot.products.loc[pid, "category"]
    stock_before = repos.products.get(pid)["stock"]
    fb = FeedbackService(repos)
    sid = fb.new_session()
    r1 = fb.register(cid, pid, "click", sid, recommendation_id=rid)
    r2 = fb.register(cid, pid, "purchase", sid, recommendation_id=rid)
    assert r1.new_outcome == "click" and r2.new_outcome == "purchase"
    assert r2.order_id is not None
    assert repos.products.get(pid)["stock"] < stock_before
    pid2 = int(res.items["product_id"].iloc[1])
    r3 = fb.register(cid, pid2, "impression", sid)  # "Ignorar" = só impressão (evidência fraca)
    assert r3.bayes_changes[0]["delta_f"] == pytest.approx(0.25)
    assert repos.interactions.count() == n_inter_before + 4

    # estado incremental == recálculo em lote com os mesmos priors
    ds = Dataset.from_db(temp_engine)
    snap = ds.snapshot(pd.Timestamp.now() + pd.Timedelta(minutes=1))
    eng.refresh_after_feedback(cid)
    batch = BayesianCategoryModel()
    state = repos.bayes.all()
    batch.load_state(snap, eng.suite.ctx, state.assign(alpha=state["alpha_prior"], beta=state["beta_prior"], n_events=0))
    ev = batch.evidence_from_outcomes(eng.suite.ctx.outcomes, snap.n_customers, snap.cidx)
    i, k = snap.cidx[cid], CATEGORIES.index(cat)
    for st in ("interest", "conversion"):
        cell = repos.bayes.get_cell(cid, cat, st)
        assert cell["alpha"] == pytest.approx(batch.alpha0[st][i, k] + ev["E"][st][i, k, 0])
        assert cell["beta"] == pytest.approx(batch.beta0[st][i, k] + ev["E"][st][i, k, 1])

    # produto comprado sai das recomendações após a atualização
    after = eng.recommend(cid, method="rules", log_run=False)
    assert pid not in set(after.items["product_id"])

    # retreino controlado: novas versões, dados preservados
    n_items = repos.order_items.count()
    retrain_all(temp_engine, train_gating=False, trained_dir=tmp_path / "trained")
    assert repos.order_items.count() == n_items
    v = repos.models.active("supervised")
    assert len(v) == 1 and int(v.iloc[0]["version"]) == 2
    assert len(repos.models.list("model_name = 'supervised' AND status = 'archived'")) == 1
