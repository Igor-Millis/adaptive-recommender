"""Configuração de logging compartilhada (arquivo + console)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOGS_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("adaptive")
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        fh = RotatingFileHandler(LOGS_DIR / "adaptive_recommender.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.WARNING)
        root.addHandler(fh)
        root.addHandler(sh)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"adaptive.{name}")
