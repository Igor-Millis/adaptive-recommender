"""Retreinamento controlado de todos os modelos (gera novas versões; preserva dados e histórico).

Uso:  python scripts/train_models.py [--no-gating]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.connection import create_schema, get_engine  # noqa: E402
from database.init_db import is_populated  # noqa: E402
from src.learning.retraining import retrain_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gating", action="store_true", help="não reaprende o gating (mais rápido)")
    args = ap.parse_args()
    engine = get_engine()
    create_schema(engine)
    if not is_populated(engine):
        sys.exit("Banco vazio — execute scripts/initialize_database.py primeiro.")
    rep = retrain_all(engine, train_gating=not args.no_gating)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
