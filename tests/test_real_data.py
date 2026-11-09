"""Adaptador de dados reais (Online Retail II) — testado com um mini-dataset fabricado (sem download)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_sources.online_retail import build_dataset, categorize, clean_transactions

ROOT = Path(__file__).resolve().parents[1]


def _raw() -> pd.DataFrame:
    rows = []
    inv = 500000
    for c, cust in enumerate([12346, 12347, 12348, 12349]):
        for m, month in enumerate(["2010-01-10", "2010-05-10", "2011-02-10", "2011-07-10", "2011-10-10"]):
            inv += 1
            for code, desc, price in [("85123", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55), ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
                                      ("85099", "JUMBO BAG RED RETROSPOT", 1.95), ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95)][: 2 + (c + m) % 3]:
                rows.append({"Invoice": str(inv), "StockCode": code, "Description": desc, "Quantity": 6 + c, "Price": price,
                             "InvoiceDate": pd.Timestamp(month), "Customer ID": float(cust), "Country": "United Kingdom"})
    rows.append({"Invoice": "C900000", "StockCode": "85123", "Description": "x", "Quantity": -6, "Price": 2.55,
                 "InvoiceDate": pd.Timestamp("2010-02-01"), "Customer ID": 12346.0, "Country": "United Kingdom"})
    rows.append({"Invoice": "900001", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "Price": 18.0,
                 "InvoiceDate": pd.Timestamp("2010-02-01"), "Customer ID": 12346.0, "Country": "France"})
    rows.append({"Invoice": "900002", "StockCode": "85123", "Description": "x", "Quantity": 1, "Price": 2.55,
                 "InvoiceDate": pd.Timestamp("2010-02-01"), "Customer ID": None, "Country": "United Kingdom"})
    return pd.DataFrame(rows)


def test_categorize_rules():
    assert categorize("JUMBO BAG RED RETROSPOT")[0] == "Bolsas e sacolas"
    assert categorize("PAPER CHAIN KIT 50'S CHRISTMAS")[0] == "Natal e datas sazonais"  # sazonal tem prioridade
    assert categorize("WHITE HANGING HEART T-LIGHT HOLDER")[0] == "Velas e iluminação"
    assert categorize("ZZZ UNKNOWN THING")[0] == "Outros"


def test_cleaning_removes_invalid_rows():
    clean = clean_transactions(_raw())
    assert not clean["Invoice"].str.startswith("C").any()
    assert (clean["Quantity"] > 0).all() and clean["Customer ID"].notna().all()
    assert "POST" not in set(clean["StockCode"])


def test_requires_real_data_profile():
    with pytest.raises(RuntimeError):
        build_dataset(_raw())  # sob o perfil sintético as categorias não batem


def test_build_dataset_under_profile(tmp_path):
    csv = tmp_path / "mini.csv"
    _raw().to_csv(csv, index=False)
    code = (
        "import json; from src.data_sources.online_retail import build_dataset, load_raw;"
        f"ds, st = build_dataset(load_raw(r'{csv}'), n_customers=10, n_products=10);"
        "print(json.dumps({'c': len(ds.customers), 'p': len(ds.products), 'i': len(ds.items), 'cats': sorted(set(ds.products['category']))}))"
    )
    env = {**os.environ, "ADAPTIVE_RECOMMENDER_PROFILE": "online_retail", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["c"] == 4 and res["p"] == 4 and res["i"] > 0
    assert "Bolsas e sacolas" in res["cats"]
