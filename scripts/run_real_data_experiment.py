"""Roda o MESMO experimento temporal do projeto sobre dados reais (Online Retail II, UCI).

Saídas: docs/REAL_DATA_EVALUATION.md e models/metadata/evaluation_online_retail.json
(não altera o banco nem os resultados do perfil sintético).

Uso:
    python scripts/download_real_data.py
    python scripts/run_real_data_experiment.py [--customers 500] [--products 400]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["ADAPTIVE_RECOMMENDER_PROFILE"] = "online_retail"  # antes de importar o projeto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import EVAL_REPORT_PATH, RAW_DIR  # noqa: E402
from src.data_sources.online_retail import build_dataset, load_raw  # noqa: E402
from src.evaluation.experiments import run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=500)
    ap.add_argument("--products", type=int, default=400)
    args = ap.parse_args()
    csv = RAW_DIR / "online_retail_ii" / "online_retail_ii.csv"
    if not csv.exists():
        sys.exit("Dataset ausente — execute primeiro: python scripts/download_real_data.py")
    ds, stats = build_dataset(load_raw(csv), n_customers=args.customers, n_products=args.products)
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=1, default=str))
    rep = run_experiment(ds, save=True)
    # acrescenta ao relatório a descrição dos dados reais usados
    extra = ["", "## 10. Dados reais utilizados", "",
             f"- Fonte: Online Retail II — UCI Machine Learning Repository (CC BY 4.0), dez/2009 a dez/2011.",
             f"- Linhas brutas: {stats.raw_rows:,}; após limpeza (sem cancelamentos, devoluções, clientes anônimos e "
             f"códigos administrativos): {stats.clean_rows:,}.",
             f"- Amostra: {stats.n_customers} clientes aleatórios e os {stats.n_products} produtos com mais compradores "
             f"antes da validação → {stats.n_orders:,} pedidos, {stats.selected_rows:,} linhas.",
             f"- Clientes cuja 1ª compra ocorre no teste (cold start real): {stats.first_purchase_in_test}.",
             "- Produtos por categoria (derivada da descrição): " + ", ".join(f"{k} {v}" for k, v in stats.category_counts.items()) + ".",
             *[f"- {n}" for n in stats.notes]]
    with open(EVAL_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(extra) + "\n")
    print(f"Concluído em {rep['elapsed_seconds']:.0f}s → {EVAL_REPORT_PATH}")
    for c in rep["conclusions"]:
        print(" -", c)


if __name__ == "__main__":
    main()
