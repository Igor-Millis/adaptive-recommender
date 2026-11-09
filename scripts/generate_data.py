"""Gera os dados sintéticos em data/synthetic (não altera o banco).

Uso:  python scripts/generate_data.py [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.pipeline import generate_all  # noqa: E402
from src.preprocessing.cleaning import data_quality_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--customers", type=int, default=None, help="nº de clientes (padrão: N_CUSTOMERS)")
    args = ap.parse_args()
    kw = {"n_customers": args.customers} if args.customers else {}
    ds = generate_all(seed=args.seed, save=True, **kw)
    print(f"Clientes: {len(ds.customers)} | Produtos: {len(ds.products)} | Pedidos: {len(ds.orders)} | "
          f"Itens: {len(ds.order_items)} | Interações: {len(ds.interactions)}")
    rep = data_quality_report(ds.customers, ds.products, ds.orders, ds.order_items, ds.interactions)
    for k, v in rep.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
