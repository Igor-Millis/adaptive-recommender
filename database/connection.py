"""Conexão com o banco via SQLAlchemy (SQLite por padrão, preparado para PostgreSQL)."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from config.settings import DATABASE_URL, SCHEMA_PATH


def make_engine(url: str = DATABASE_URL) -> Engine:
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()

    return engine


@lru_cache(maxsize=4)
def get_engine(url: str = DATABASE_URL) -> Engine:
    return make_engine(url)


def create_schema(engine: Engine) -> None:
    """Cria tabelas, índices e views (idempotente: ``IF NOT EXISTS``)."""
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    # remove comentários antes de separar instruções (comentários podem conter ';')
    sql = "\n".join(line.split("--", 1)[0] for line in raw.splitlines())
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            body = stmt.strip()
            if body:
                conn.execute(text(body))


@contextmanager
def transaction(engine: Engine):
    with engine.begin() as conn:
        yield conn


def table_exists(engine: Engine, name: str) -> bool:
    insp = inspect(engine)
    return name in insp.get_table_names() or name in insp.get_view_names()
