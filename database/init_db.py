"""Inicialização e carga do banco a partir dos dados sintéticos."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import SYNTHETIC_DIR
from database.connection import create_schema
from database.repositories import Repositories
from src.utils.logging_setup import get_logger

log = get_logger("database")

LOAD_ORDER = ["customers", "products", "orders", "order_items", "interactions"]
DATE_COLS = {
    "customers": ["registration_date"],
    "products": ["launch_date"],
    "orders": ["order_date"],
    "interactions": ["timestamp"],
}


def is_populated(engine: Engine) -> bool:
    try:
        with engine.connect() as c:
            return int(c.execute(text("SELECT COUNT(*) FROM customers")).scalar_one()) > 0
    except Exception:
        return False


def load_synthetic(engine: Engine, frames: dict[str, pd.DataFrame] | None = None, reset: bool = False) -> dict:
    """Carrega CSVs sintéticos no banco.

    Por segurança (regra 15: preservar dados), só carrega em banco vazio, a não
    ser que ``reset=True`` seja passado explicitamente.
    """
    create_schema(engine)
    if is_populated(engine) and not reset:
        log.info("Banco já populado — carga ignorada (use reset=True para recriar).")
        return {}
    if reset:
        with engine.begin() as c:
            for t in ["bayesian_update_log", "expert_weights", "recommendations", "recommendation_runs", "interactions",
                      "order_items", "orders", "customer_clusters", "cluster_profiles", "cluster_runs", "bayesian_estimates",
                      "customer_features", "evaluation_results", "evaluation_runs", "model_versions", "models",
                      "event_log", "products", "customers"]:
                c.execute(text(f"DELETE FROM {t}"))
    repos = Repositories(engine)
    counts = {}
    for name in LOAD_ORDER:
        df = frames[name] if frames else pd.read_csv(SYNTHETIC_DIR / f"{name}.csv")
        for c in DATE_COLS.get(name, []):
            df[c] = pd.to_datetime(df[c], format="mixed")
        if name == "interactions":
            df = df.assign(source="synthetic", recommendation_id=None)
        if name == "orders":
            df = df.assign(source="synthetic")
        repo = {"customers": repos.customers, "products": repos.products, "orders": repos.orders,
                "order_items": repos.order_items, "interactions": repos.interactions}[name]
        counts[name] = repo.bulk_load(df)
    repos.events.log("database_loaded", payload=counts)
    log.info("Banco carregado: %s", counts)
    return counts
