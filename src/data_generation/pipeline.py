"""Orquestra a geração completa dos dados sintéticos e salva em ``data/synthetic``."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from config.settings import N_CUSTOMERS, N_PRODUCTS, PROFILE, RANDOM_SEED, SYNTHETIC_DIR
from src.data_generation.generate_customers import generate_customers, latents_to_records
from src.data_generation.generate_orders import simulate
from src.data_generation.generate_products import generate_products
from src.utils.logging_setup import get_logger

log = get_logger("data_generation")


@dataclass
class SyntheticDataset:
    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    interactions: pd.DataFrame
    ground_truth: list[dict]


def generate_all(seed: int = RANDOM_SEED, save: bool = True, n_customers: int = N_CUSTOMERS,
                 n_products: int = N_PRODUCTS) -> SyntheticDataset:
    products = generate_products(n_products, seed=seed)
    customers, latents = generate_customers(seed=seed, n_customers=n_customers)
    orders, items, interactions, quality = simulate(customers, latents, products, seed=seed)
    gt = latents_to_records(latents)
    products_gt = products[["product_id", "_style"]].copy()
    products_gt["latent_quality"] = quality
    products_clean = products.drop(columns=["_style"])
    ds = SyntheticDataset(customers, products_clean, orders, items, interactions, gt)
    log.info(
        "Dados gerados: %d clientes, %d produtos, %d pedidos, %d itens, %d interações",
        len(customers), len(products_clean), len(orders), len(items), len(interactions),
    )
    if save:
        customers.to_csv(SYNTHETIC_DIR / "customers.csv", index=False)
        products_clean.to_csv(SYNTHETIC_DIR / "products.csv", index=False)
        orders.to_csv(SYNTHETIC_DIR / "orders.csv", index=False)
        items.to_csv(SYNTHETIC_DIR / "order_items.csv", index=False)
        interactions.to_csv(SYNTHETIC_DIR / "interactions.csv", index=False)
        with open(SYNTHETIC_DIR / "ground_truth_customers.json", "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=1, default=str)
        products_gt.to_csv(SYNTHETIC_DIR / "ground_truth_products.csv", index=False)
    return ds


def load_ground_truth() -> pd.DataFrame:
    """Ground truth dos arquétipos — uso exclusivo em análise/avaliação, nunca em modelos."""
    path = SYNTHETIC_DIR / "ground_truth_customers.json"
    if PROFILE != "synthetic" or not path.exists():
        return pd.DataFrame(columns=["customer_id", "archetype", "history_profile"])
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))
