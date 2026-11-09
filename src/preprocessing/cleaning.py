"""Limpeza e verificação de qualidade dos dados brutos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_items(items: pd.DataFrame, valid_products: set[int]) -> pd.DataFrame:
    """Remove linhas inválidas (quantidade não positiva, produto inexistente, valor negativo)."""
    df = items.copy()
    mask = (df["quantity"] > 0) & (df["unit_price"] > 0) & (df["total_value"] >= 0) & df["product_id"].isin(valid_products)
    return df.loc[mask].reset_index(drop=True)


def data_quality_report(customers, products, orders, items, interactions) -> dict:
    """Checagens de integridade e de coerência estatística do conjunto gerado."""
    rep: dict = {}
    rep["ids_unicos"] = bool(
        customers["customer_id"].is_unique and products["product_id"].is_unique and orders["order_id"].is_unique
        and items["item_id"].is_unique and interactions["interaction_id"].is_unique
    )
    rep["fk_pedidos_clientes"] = bool(orders["customer_id"].isin(customers["customer_id"]).all())
    rep["fk_itens_pedidos"] = bool(items["order_id"].isin(orders["order_id"]).all())
    rep["fk_itens_produtos"] = bool(items["product_id"].isin(products["product_id"]).all())
    rep["fk_interacoes"] = bool(
        interactions["customer_id"].isin(customers["customer_id"]).all()
        and interactions["product_id"].isin(products["product_id"]).all()
    )
    rep["quantidades_positivas"] = bool((items["quantity"] > 0).all())
    recomputed = (items["quantity"] * items["unit_price"] * (1 - items["discount"])).round(2)
    rep["totais_consistentes"] = bool(np.allclose(recomputed, items["total_value"], atol=0.011))
    m = items.merge(orders, on="order_id").merge(products[["product_id", "launch_date"]], on="product_id")
    rep["compras_apos_lancamento"] = bool((pd.to_datetime(m["order_date"]) >= pd.to_datetime(m["launch_date"])).all())
    reg = customers.set_index("customer_id")["registration_date"]
    od = orders.assign(reg=orders["customer_id"].map(reg))
    rep["pedidos_apos_cadastro"] = bool((pd.to_datetime(od["order_date"]) >= pd.to_datetime(od["reg"])).all())
    # Cada compra via recomendação deve existir como item de pedido do cliente
    pur = interactions[interactions["interaction_type"] == "purchase"][["customer_id", "product_id"]].drop_duplicates()
    bought = m[["customer_id", "product_id"]].drop_duplicates()
    rep["compras_interacao_em_pedidos"] = bool(len(pur.merge(bought)) == len(pur))
    n_orders = orders.groupby("customer_id").size().reindex(customers["customer_id"], fill_value=0)
    rep["clientes_sem_pedidos"] = int((n_orders == 0).sum())
    rep["clientes_menos_de_5_pedidos"] = int(((n_orders > 0) & (n_orders < 5)).sum())
    rep["itens_de_pedido"] = int(len(items))
    rep["esparsidade_matriz"] = round(1 - len(bought) / (len(customers) * len(products)), 3)
    return rep
