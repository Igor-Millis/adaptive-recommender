"""Repositórios transacionais: pedidos, itens e interações."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from database.repositories.base import BaseRepository, _to_db


class OrderRepository(BaseRepository):
    table = "orders"
    pk = "order_id"
    date_cols = ("order_date",)

    def create_order(self, customer_id: int, order_date, items: list[dict], channel: str = "portal_b2b",
                     source: str = "app", conn=None) -> int:
        """Cria pedido + itens na mesma transação."""
        def _do(c):
            oid = self.insert({"customer_id": customer_id, "order_date": order_date, "channel": channel, "source": source}, conn=c)
            item_repo = OrderItemRepository(self.engine)
            for it in items:
                total = round(it["quantity"] * it["unit_price"] * (1 - it.get("discount", 0.0)), 2)
                item_repo.insert({"order_id": oid, "product_id": it["product_id"], "quantity": it["quantity"],
                                  "unit_price": it["unit_price"], "discount": it.get("discount", 0.0), "total_value": total}, conn=c)
            return oid
        if conn is not None:
            return _do(conn)
        with self.engine.begin() as c:
            return _do(c)

    def flat_items(self, customer_id: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM v_order_items_flat"
        params = {}
        if customer_id is not None:
            sql += " WHERE customer_id = :c"
            params["c"] = int(customer_id)
        df = self.query(sql + " ORDER BY order_date", params)
        df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
        return df


class OrderItemRepository(BaseRepository):
    table = "order_items"
    pk = "item_id"


class InteractionRepository(BaseRepository):
    table = "interactions"
    pk = "interaction_id"
    date_cols = ("timestamp",)

    def next_session_id(self, conn=None) -> int:
        sql = text("SELECT COALESCE(MAX(session_id), 0) + 1 FROM interactions")
        if conn is not None:
            return int(conn.execute(sql).scalar_one())
        with self.engine.connect() as c:
            return int(c.execute(sql).scalar_one())

    def by_customer(self, customer_id: int) -> pd.DataFrame:
        return self.list("customer_id = :c", {"c": int(customer_id)}, order_by="timestamp, interaction_id")

    def session_events(self, session_id: int, product_id: int) -> pd.DataFrame:
        return self.list("session_id = :s AND product_id = :p", {"s": int(session_id), "p": int(product_id)},
                         order_by="interaction_id")


__all__ = ["OrderRepository", "OrderItemRepository", "InteractionRepository", "_to_db"]
