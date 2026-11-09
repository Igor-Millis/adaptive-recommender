"""Cria o esquema e carrega os dados sintéticos no banco.

Uso:
    python scripts/initialize_database.py            # cria se não existir (preserva dados)
    python scripts/initialize_database.py --reset    # recria do zero (APAGA dados existentes)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SYNTHETIC_DIR  # noqa: E402
from database.connection import create_schema, get_engine  # noqa: E402
from database.init_db import is_populated, load_synthetic  # noqa: E402
from src.data_generation.pipeline import generate_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="apaga e recarrega todo o banco")
    args = ap.parse_args()
    engine = get_engine()
    create_schema(engine)
    if is_populated(engine) and not args.reset:
        print("Banco já inicializado — nada a fazer (dados preservados).")
        return
    if not (SYNTHETIC_DIR / "customers.csv").exists():
        print("Dados sintéticos ausentes — gerando...")
        generate_all(save=True)
    counts = load_synthetic(engine, reset=args.reset)
    print("Banco inicializado:", counts)


if __name__ == "__main__":
    main()
