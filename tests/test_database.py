"""Esquema, CRUD e integridade referencial."""
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from database.connection import table_exists
from database.init_db import load_synthetic
from database.repositories import Repositories


def test_schema_tables(temp_engine):
    for t in ["customers", "products", "orders", "order_items", "interactions", "recommendations", "model_versions",
              "expert_weights", "bayesian_estimates", "customer_clusters", "evaluation_results", "v_order_items_flat"]:
        assert table_exists(temp_engine, t), t


def test_counts_loaded(temp_engine, synthetic):
    r = Repositories(temp_engine)
    assert r.customers.count() == len(synthetic.customers)
    assert r.products.count() == 200
    assert r.order_items.count() == len(synthetic.order_items)


def test_crud_customer(temp_engine):
    r = Repositories(temp_engine)
    cid = r.customers.insert({"name": "Loja Teste CRUD", "city": "Campinas", "state": "SP",
                              "registration_date": pd.Timestamp("2026-01-01"), "store_size": "Pequeno",
                              "business_segment": "Multimarcas", "revenue_band": "Até R$ 360 mil"})
    assert r.customers.get(cid)["city"] == "Campinas"
    r.customers.update(cid, {"city": "Santos"})
    assert r.customers.get(cid)["city"] == "Santos"
    assert r.customers.delete(cid) == 1
    assert r.customers.get(cid) is None


def test_foreign_key_enforced(temp_engine):
    r = Repositories(temp_engine)
    with pytest.raises(IntegrityError):
        r.orders.insert({"customer_id": 99999, "order_date": pd.Timestamp("2026-01-01"), "channel": "x", "source": "app"})


def test_check_constraints(temp_engine):
    r = Repositories(temp_engine)
    with pytest.raises(IntegrityError):
        r.products.update(1, {"stock": -5})
    with pytest.raises(IntegrityError):
        r.interactions.insert({"session_id": 1, "customer_id": 1, "product_id": 1, "model_name": "x",
                               "interaction_type": "nao_existe", "timestamp": pd.Timestamp.now(), "source": "app"})


def test_restrict_delete_with_orders(temp_engine):
    r = Repositories(temp_engine)
    cid = int(r.orders.list().iloc[0]["customer_id"])
    with pytest.raises(IntegrityError):
        r.customers.delete(cid)


def test_load_does_not_overwrite_existing(temp_engine):
    before = Repositories(temp_engine).order_items.count()
    assert load_synthetic(temp_engine) == {}  # banco populado: carga ignorada (dados preservados)
    assert Repositories(temp_engine).order_items.count() == before


def test_stock_never_negative(temp_engine):
    r = Repositories(temp_engine)
    r.products.decrement_stock(1, 10**6)
    assert r.products.get(1)["stock"] == 0
