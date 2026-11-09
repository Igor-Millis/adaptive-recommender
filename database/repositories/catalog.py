"""Repositórios de cadastro: clientes e produtos."""
from __future__ import annotations

from sqlalchemy import text

from database.repositories.base import BaseRepository


class CustomerRepository(BaseRepository):
    table = "customers"
    pk = "customer_id"
    date_cols = ("registration_date", "created_at")

    def search(self, term: str) -> "pd.DataFrame":  # noqa: F821
        like = f"%{term.strip()}%"
        return self.list("name LIKE :t OR city LIKE :t OR business_segment LIKE :t", {"t": like})


class ProductRepository(BaseRepository):
    table = "products"
    pk = "product_id"
    date_cols = ("launch_date",)

    def available(self):
        return self.list("stock > 0 AND is_active = 1")

    def decrement_stock(self, product_id: int, quantity: int, conn=None) -> int:
        """Baixa de estoque sem permitir valores negativos (CHECK stock >= 0)."""
        sql = text("UPDATE products SET stock = MAX(stock - :q, 0) WHERE product_id = :p")
        params = {"q": int(quantity), "p": int(product_id)}
        if conn is not None:
            return conn.execute(sql, params).rowcount
        with self.engine.begin() as c:
            return c.execute(sql, params).rowcount
