"""Repositório genérico: CRUD parametrizado sobre uma tabela.

Usa SQL textual com parâmetros nomeados (sem concatenação de valores) e
``INSERT ... RETURNING`` (suportado por SQLite ≥ 3.35 e PostgreSQL).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


def _to_db(v: Any) -> Any:
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


class BaseRepository:
    table: str = ""
    pk: str = ""
    date_cols: tuple[str, ...] = ()

    def __init__(self, engine: Engine):
        self.engine = engine

    # ------------------------------------------------------------------ leitura
    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params or {})
        for c in self.date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], format="mixed")
        return df

    def get(self, pk_value: Any) -> dict | None:
        df = self.query(f"SELECT * FROM {self.table} WHERE {self.pk} = :v", {"v": _to_db(pk_value)})
        return None if df.empty else df.iloc[0].to_dict()

    def list(self, where: str | None = None, params: dict | None = None, order_by: str | None = None) -> pd.DataFrame:
        sql = f"SELECT * FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order_by or self.pk}"
        return self.query(sql, params)

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {self.table}")).scalar_one())

    # ------------------------------------------------------------------ escrita
    def insert(self, values: dict, conn: Connection | None = None) -> Any:
        cols = list(values.keys())
        sql = text(
            f"INSERT INTO {self.table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)}) RETURNING {self.pk}"
        )
        params = {c: _to_db(values[c]) for c in cols}
        if conn is not None:
            return conn.execute(sql, params).scalar_one()
        with self.engine.begin() as c:
            return c.execute(sql, params).scalar_one()

    def insert_many(self, rows: Iterable[dict], conn: Connection | None = None) -> int:
        rows = list(rows)
        if not rows:
            return 0
        cols = list(rows[0].keys())
        sql = text(f"INSERT INTO {self.table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})")
        payload = [{c: _to_db(r.get(c)) for c in cols} for r in rows]
        if conn is not None:
            conn.execute(sql, payload)
        else:
            with self.engine.begin() as c:
                c.execute(sql, payload)
        return len(rows)

    def bulk_load(self, df: pd.DataFrame) -> int:
        return self.insert_many(df.to_dict("records"))

    def update(self, pk_value: Any, values: dict, conn: Connection | None = None) -> int:
        if not values:
            return 0
        sets = ", ".join(f"{c} = :{c}" for c in values)
        params = {c: _to_db(v) for c, v in values.items()}
        params["_pk"] = _to_db(pk_value)
        sql = text(f"UPDATE {self.table} SET {sets} WHERE {self.pk} = :_pk")
        if conn is not None:
            return conn.execute(sql, params).rowcount
        with self.engine.begin() as c:
            return c.execute(sql, params).rowcount

    def delete(self, pk_value: Any) -> int:
        with self.engine.begin() as c:
            return c.execute(text(f"DELETE FROM {self.table} WHERE {self.pk} = :v"), {"v": _to_db(pk_value)}).rowcount
