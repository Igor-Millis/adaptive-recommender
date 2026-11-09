"""Fixtures compartilhadas: dados sintéticos em memória e banco SQLite temporário."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import TEST_START, VAL_START  # noqa: E402
from database.connection import create_schema, make_engine  # noqa: E402
from database.init_db import load_synthetic  # noqa: E402
from src.data_generation.pipeline import generate_all  # noqa: E402
from src.preprocessing.snapshot import Dataset  # noqa: E402
from src.recommenders.suite import ExpertSuite  # noqa: E402


# Base de teste menor e fixa: os testes verificam propriedades, não dependem do tamanho de produção
TEST_N_CUSTOMERS = 80


@pytest.fixture(scope="session")
def synthetic():
    return generate_all(seed=42, save=False, n_customers=TEST_N_CUSTOMERS, n_products=200)


@pytest.fixture(scope="session")
def dataset(synthetic):
    s = synthetic
    return Dataset.from_frames(s.customers, s.products, s.orders, s.order_items, s.interactions)


@pytest.fixture(scope="session")
def snapshot_val(dataset):
    return dataset.snapshot(VAL_START)


@pytest.fixture(scope="session")
def supervised_val(dataset):
    from src.recommenders.supervised import SupervisedRecommender

    return SupervisedRecommender.train(dataset, VAL_START, fixed_k=5)


@pytest.fixture(scope="session")
def suite_val(snapshot_val, supervised_val):
    return ExpertSuite(supervised_val).fit(snapshot_val, fixed_k=5)


@pytest.fixture(scope="session")
def stage_val(dataset, supervised_val):
    from src.evaluation.temporal_validation import build_stage

    return build_stage(dataset, "validacao", VAL_START, TEST_START, supervised_val, fixed_k=5)


@pytest.fixture()
def temp_engine(tmp_path, synthetic):
    eng = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    create_schema(eng)
    s = synthetic
    load_synthetic(eng, frames={"customers": s.customers.copy(), "products": s.products.copy(), "orders": s.orders.copy(),
                                "order_items": s.order_items.copy(), "interactions": s.interactions.copy()})
    yield eng
    eng.dispose()
