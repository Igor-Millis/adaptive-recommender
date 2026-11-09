"""Snapshots temporais dos dados (ponto no tempo) — base da prevenção de vazamento.

``Dataset`` contém todos os registros. ``Dataset.snapshot(as_of)`` devolve um
``DataSnapshot`` em que pedidos e interações são estritamente anteriores a
``as_of``; todo treino de modelo recebe um snapshot, nunca o ``Dataset`` bruto.
Dados cadastrais (clientes, atributos de produtos) são considerados estáticos;
a disponibilidade de produtos é dada pela data de lançamento.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.utils.validation import validate_no_future


@dataclass
class DataSnapshot:
    as_of: pd.Timestamp
    customers: pd.DataFrame  # indexado por customer_id (ordenado)
    products: pd.DataFrame  # indexado por product_id (ordenado)
    items: pd.DataFrame  # linhas de pedido < as_of (com customer_id, order_date, category)
    interactions: pd.DataFrame  # interações < as_of
    customer_ids: np.ndarray = field(init=False)
    product_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        validate_no_future(self.items, "order_date", self.as_of, "itens de pedido")
        validate_no_future(self.interactions, "timestamp", self.as_of, "interações")
        self.customer_ids = self.customers.index.to_numpy()
        self.product_ids = self.products.index.to_numpy()
        self.cidx = {int(c): i for i, c in enumerate(self.customer_ids)}
        self.pidx = {int(p): j for j, p in enumerate(self.product_ids)}
        self.launched = (self.products["launch_date"] <= self.as_of).to_numpy()
        active = (self.products["is_active"].to_numpy() == 1) if "is_active" in self.products.columns \
            else np.ones(len(self.products), dtype=bool)
        self.in_stock = (self.products["stock"].to_numpy() > 0) & active

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    def purchased_matrix(self, since: pd.Timestamp | None = None) -> np.ndarray:
        """Matriz booleana cliente × produto: comprou (desde ``since``)."""
        M = np.zeros((self.n_customers, self.n_products), dtype=bool)
        it = self.items if since is None else self.items[self.items["order_date"] >= since]
        if len(it):
            ci = it["customer_id"].map(self.cidx).to_numpy()
            pj = it["product_id"].map(self.pidx).to_numpy()
            M[ci, pj] = True
        return M


@dataclass
class Dataset:
    customers: pd.DataFrame
    products: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame

    def __post_init__(self):
        self.customers = self.customers.set_index("customer_id").sort_index() if "customer_id" in self.customers else self.customers.sort_index()
        self.products = self.products.set_index("product_id").sort_index() if "product_id" in self.products else self.products.sort_index()
        self.customers["registration_date"] = pd.to_datetime(self.customers["registration_date"], format="mixed")
        self.products["launch_date"] = pd.to_datetime(self.products["launch_date"], format="mixed")
        self.items = self.items.copy()
        self.items["order_date"] = pd.to_datetime(self.items["order_date"], format="mixed")
        if "category" not in self.items:
            self.items["category"] = self.items["product_id"].map(self.products["category"])
        self.items = self.items.sort_values("order_date").reset_index(drop=True)
        self.interactions = self.interactions.copy()
        self.interactions["timestamp"] = pd.to_datetime(self.interactions["timestamp"], format="mixed")
        self.interactions["category"] = self.interactions["product_id"].map(self.products["category"])
        self.interactions = self.interactions.sort_values(["timestamp", "interaction_id"]).reset_index(drop=True)

    @property
    def max_date(self) -> pd.Timestamp:
        dates = [self.items["order_date"].max() if len(self.items) else pd.NaT,
                 self.interactions["timestamp"].max() if len(self.interactions) else pd.NaT]
        return max(d for d in dates if pd.notna(d))

    def snapshot(self, as_of: pd.Timestamp) -> DataSnapshot:
        as_of = pd.Timestamp(as_of)
        return DataSnapshot(
            as_of=as_of,
            customers=self.customers,
            products=self.products,
            items=self.items[self.items["order_date"] < as_of],
            interactions=self.interactions[self.interactions["timestamp"] < as_of],
        )

    def window_items(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        it = self.items
        return it[(it["order_date"] >= start) & (it["order_date"] < end)]

    @classmethod
    def from_db(cls, engine: Engine) -> "Dataset":
        from database.repositories import Repositories

        r = Repositories(engine)
        customers = r.customers.list()
        products = r.products.list()
        items = r.orders.flat_items()
        inter = r.interactions.list(order_by="timestamp, interaction_id")
        return cls(customers, products, items, inter)

    @classmethod
    def from_frames(cls, customers, products, orders, order_items, interactions) -> "Dataset":
        items = order_items.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id")
        return cls(customers.copy(), products.copy(), items, interactions.copy())
