"""Baixa o dataset público Online Retail II (UCI ML Repository, CC BY 4.0) e converte para CSV.

Fonte: https://archive.ics.uci.edu/dataset/502/online+retail+ii  (≈ 45 MB, xlsx com duas planilhas)
Destino: data/raw/online_retail_ii/online_retail_ii.csv

Uso:  python scripts/download_real_data.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DEST = ROOT / "data" / "raw" / "online_retail_ii"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    csv = DEST / "online_retail_ii.csv"
    if csv.exists():
        print(f"Já existe: {csv}")
        return
    zpath = DEST / "online_retail_ii.zip"
    if not zpath.exists():
        print(f"Baixando {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    print("Convertendo xlsx → csv (≈ 1 min)...")
    with zipfile.ZipFile(zpath) as z, z.open("online_retail_II.xlsx") as f:
        sheets = pd.read_excel(f, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"OK: {len(df):,} linhas → {csv}")


if __name__ == "__main__":
    sys.exit(main())
