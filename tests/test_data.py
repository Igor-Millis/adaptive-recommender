"""Integridade e coerência estatística dos dados sintéticos."""
import numpy as np
import pandas as pd
import pytest

from config.settings import CATEGORIES, TEST_START
from src.data_generation.pipeline import generate_all
from src.preprocessing.cleaning import data_quality_report


def test_sizes(synthetic):
    n = len(synthetic.customers)
    assert len(synthetic.products) == 200
    assert 150 * n <= len(synthetic.order_items) <= 260 * n  # ~200 linhas de pedido por cliente
    assert set(synthetic.products["category"]) == set(CATEGORIES)


@pytest.mark.parametrize("n", [50, 137, 300])
def test_customer_count_and_proportions(n):
    from src.data_generation.generate_customers import archetype_counts, generate_customers

    counts = archetype_counts(n)
    assert sum(counts.values()) == n and min(counts.values()) >= 1
    customers, latents = generate_customers(seed=1, n_customers=n)
    assert len(customers) == n and customers["name"].is_unique
    profiles = pd.Series([lat.history_profile for lat in latents]).value_counts()
    assert profiles.get("sem_historico", 0) >= 2 and profiles.get("novo_teste", 0) >= 2


def test_quality_report_all_pass(synthetic):
    s = synthetic
    rep = data_quality_report(s.customers, s.products, s.orders, s.order_items, s.interactions)
    for k in ["ids_unicos", "fk_pedidos_clientes", "fk_itens_pedidos", "fk_itens_produtos", "fk_interacoes",
              "quantidades_positivas", "totais_consistentes", "compras_apos_lancamento", "pedidos_apos_cadastro",
              "compras_interacao_em_pedidos"]:
        assert rep[k] is True, k


def test_cold_start_customers_exist(synthetic):
    s = synthetic
    n_orders = s.orders.groupby("customer_id").size().reindex(s.customers["customer_id"], fill_value=0)
    assert (n_orders == 0).sum() >= 1  # sem histórico algum
    first = s.orders.groupby("customer_id")["order_date"].min()
    assert (pd.to_datetime(first) >= TEST_START).sum() >= 1  # sem histórico no corte do teste


def test_interaction_types_and_funnel(synthetic):
    it = synthetic.interactions
    counts = it["interaction_type"].value_counts()
    assert counts["impression"] > counts["click"] > counts["purchase"]
    # ausência de clique NÃO é registrada como rejeição
    assert counts["rejection"] < counts["impression"] - counts["click"]


def test_patterns_are_identifiable(synthetic):
    """Clientes esportivos compram mais roupas esportivas que a média (estrutura controlada)."""
    gt = pd.DataFrame(synthetic.ground_truth).set_index("customer_id")
    m = synthetic.order_items.merge(synthetic.orders, on="order_id").merge(synthetic.products[["product_id", "category"]])
    share = m.assign(sp=m["category"] == "Roupas esportivas").groupby("customer_id")["sp"].mean()
    esportivo = share.reindex(gt.index[gt["archetype"] == "esportivo"]).dropna()
    others = share.reindex(gt.index[gt["archetype"] != "esportivo"]).dropna()
    assert esportivo.mean() > 2 * others.mean()


def test_reproducible():
    a = generate_all(seed=7, save=False, n_customers=40)
    b = generate_all(seed=7, save=False, n_customers=40)
    assert a.order_items["total_value"].sum() == b.order_items["total_value"].sum()
    assert np.array_equal(a.interactions["product_id"].to_numpy(), b.interactions["product_id"].to_numpy())
