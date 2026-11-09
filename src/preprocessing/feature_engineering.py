"""Engenharia de atributos ponto-no-tempo para clientes e produtos.

Toda função recebe um ``DataSnapshot`` e usa somente registros anteriores a
``snapshot.as_of``. Os atributos se dividem em:

* **observados (histórico)** — contagens e médias diretas dos pedidos;
* **estimados (inferidos)** — índices que dependem de hipóteses (sensibilidade
  a preço, sazonalidade, deriva, consistência), documentados em
  ``docs/DATA_DICTIONARY.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CATEGORIES
from src.preprocessing.snapshot import DataSnapshot
from src.preprocessing.transformations import circular_seasonality, cosine, entropy, js_divergence

OUTCOME_ORDER = {"impression": 0, "rejection": 1, "click": 2, "interest": 3, "add_to_cart": 4, "purchase": 5}
POSITIVE_OUTCOMES = {"click", "interest", "add_to_cart", "purchase"}

OBSERVED_FEATURES = [
    "n_orders", "n_lines", "n_distinct_products", "total_revenue", "avg_ticket", "avg_units_per_order",
    "avg_lines_per_order", "avg_unit_price", "avg_discount", "recency_days", "frequency_per_month",
    "tenure_days", "n_impressions", "n_engagements", "n_rejections", "n_rec_purchases",
]
INFERRED_FEATURES = ["price_index", "category_entropy", "seasonality_index", "seasonal_peak_month",
                     "drift_js", "consistency", "engagement_rate"]
SHARE_COLS = [f"share_{c}" for c in CATEGORIES]


def resolve_outcomes(interactions: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (sessão, cliente, produto) com o desfecho mais forte observado.

    Impressão sem resposta permanece com desfecho ``impression`` (não é rejeição).
    """
    if interactions.empty:
        return pd.DataFrame(columns=["session_id", "customer_id", "product_id", "category", "outcome", "timestamp", "model_name"])
    df = interactions.copy()
    df["_rank"] = df["interaction_type"].map(OUTCOME_ORDER)
    pos_mask = df["interaction_type"].isin(POSITIVE_OUTCOMES)
    # se houve engajamento positivo, ele prevalece sobre rejeição na mesma sessão
    df.loc[~pos_mask & (df["interaction_type"] == "rejection"), "_rank"] = 1
    g = df.sort_values(["_rank", "interaction_id"]).groupby(["session_id", "customer_id", "product_id"], as_index=False)
    out = g.agg(category=("category", "first"), outcome=("interaction_type", "last"), timestamp=("timestamp", "min"),
                model_name=("model_name", "first"), _rank=("_rank", "max"))
    inv = {v: k for k, v in OUTCOME_ORDER.items()}
    out["outcome"] = out["_rank"].map(inv)
    return out.drop(columns="_rank").sort_values("timestamp").reset_index(drop=True)


def category_share_matrix(snapshot: DataSnapshot, since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> np.ndarray:
    """Participação de cada categoria nas unidades compradas (linhas somam 1 ou 0)."""
    it = snapshot.items
    if since is not None:
        it = it[it["order_date"] >= since]
    if until is not None:
        it = it[it["order_date"] < until]
    M = np.zeros((snapshot.n_customers, len(CATEGORIES)))
    if len(it):
        ci = it["customer_id"].map(snapshot.cidx).to_numpy()
        kj = it["category"].map({c: k for k, c in enumerate(CATEGORIES)}).to_numpy()
        np.add.at(M, (ci, kj), it["quantity"].to_numpy(dtype=float))
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


def build_customer_features(snapshot: DataSnapshot) -> pd.DataFrame:
    as_of = snapshot.as_of
    it = snapshot.items
    prods = snapshot.products
    cat_median = prods.groupby("category")["price"].median()
    rows = []
    outcomes = resolve_outcomes(snapshot.interactions)
    shares_all = category_share_matrix(snapshot)
    recent_cut = as_of - pd.Timedelta(days=120)
    shares_recent = category_share_matrix(snapshot, since=recent_cut)
    shares_old = category_share_matrix(snapshot, until=recent_cut)
    by_cust = dict(tuple(it.groupby("customer_id"))) if len(it) else {}
    out_by_cust = dict(tuple(outcomes.groupby("customer_id"))) if len(outcomes) else {}

    for i, cid in enumerate(snapshot.customer_ids):
        cust = snapshot.customers.loc[cid]
        reg = pd.Timestamp(cust["registration_date"])
        d = by_cust.get(cid)
        f: dict = {"customer_id": int(cid), "tenure_days": max((as_of - reg).days, 0)}
        if d is None or d.empty:
            f.update(n_orders=0, n_lines=0, n_distinct_products=0, total_revenue=0.0, avg_ticket=0.0,
                     avg_units_per_order=0.0, avg_lines_per_order=0.0, avg_unit_price=0.0, avg_discount=0.0,
                     recency_days=np.nan, frequency_per_month=0.0, price_index=0.0, category_entropy=0.0,
                     seasonality_index=0.0, seasonal_peak_month=0, drift_js=0.0, consistency=0.0)
        else:
            orders = d.groupby("order_id").agg(date=("order_date", "first"), value=("total_value", "sum"),
                                               units=("quantity", "sum"), lines=("product_id", "size"))
            first = orders["date"].min()
            months = max((as_of - max(first, reg)).days / 30.44, 1.0)
            rel = np.log(d["unit_price"].to_numpy() / d["category"].map(cat_median).to_numpy())
            w = d["quantity"].to_numpy(dtype=float)
            month_counts = np.bincount(orders["date"].dt.month.to_numpy() - 1, minlength=12)
            seas, peak = circular_seasonality(month_counts) if months >= 10 else (0.0, 0)
            # consistência: cosseno entre participação por categoria na 1ª e 2ª metade dos pedidos
            o_sorted = orders.sort_values("date")
            half = o_sorted.index[: len(o_sorted) // 2]
            first_half = d[d["order_id"].isin(half)]
            second_half = d[~d["order_id"].isin(half)]

            def _share(x):
                v = x.groupby("category")["quantity"].sum().reindex(CATEGORIES, fill_value=0).to_numpy(float)
                return v

            consistency = cosine(_share(first_half), _share(second_half)) if len(o_sorted) >= 4 else 0.0
            n_recent = (d["order_date"] >= recent_cut).sum()
            n_old = (d["order_date"] < recent_cut).sum()
            drift = js_divergence(shares_recent[i], shares_old[i]) if (n_recent >= 8 and n_old >= 8) else 0.0
            f.update(
                n_orders=int(len(orders)), n_lines=int(len(d)), n_distinct_products=int(d["product_id"].nunique()),
                total_revenue=float(orders["value"].sum()), avg_ticket=float(orders["value"].mean()),
                avg_units_per_order=float(orders["units"].mean()), avg_lines_per_order=float(orders["lines"].mean()),
                avg_unit_price=float(np.average(d["unit_price"], weights=w)), avg_discount=float(np.average(d["discount"], weights=w)),
                recency_days=float((as_of - orders["date"].max()).days), frequency_per_month=float(len(orders) / months),
                price_index=float(np.average(rel, weights=w)), category_entropy=float(entropy(shares_all[i])),
                seasonality_index=seas, seasonal_peak_month=peak, drift_js=float(drift), consistency=float(consistency),
            )
        o = out_by_cust.get(cid)
        if o is None or o.empty:
            f.update(n_impressions=0, n_engagements=0, n_rejections=0, n_rec_purchases=0, engagement_rate=0.0)
        else:
            n_imp = len(o)
            n_eng = int(o["outcome"].isin(POSITIVE_OUTCOMES).sum())
            f.update(n_impressions=n_imp, n_engagements=n_eng, n_rejections=int((o["outcome"] == "rejection").sum()),
                     n_rec_purchases=int((o["outcome"] == "purchase").sum()),
                     engagement_rate=float((n_eng + 1) / (n_imp + 10)))  # suavizado (Laplace-like)
        for k, c in enumerate(CATEGORIES):
            f[f"share_{c}"] = float(shares_all[i, k])
        f["has_history"] = int(f["n_orders"] > 0)
        rows.append(f)
    df = pd.DataFrame(rows).set_index("customer_id")
    return df


def build_product_features(snapshot: DataSnapshot) -> pd.DataFrame:
    as_of = snapshot.as_of
    p = snapshot.products.copy()
    it = snapshot.items
    recent = it[it["order_date"] >= as_of - pd.Timedelta(days=90)]
    p["popularity_90d"] = recent.groupby("product_id")["customer_id"].nunique().reindex(p.index, fill_value=0)
    p["popularity_all"] = it.groupby("product_id")["customer_id"].nunique().reindex(p.index, fill_value=0)
    p["units_90d"] = recent.groupby("product_id")["quantity"].sum().reindex(p.index, fill_value=0)
    p["age_days"] = (as_of - p["launch_date"]).dt.days
    p["launched"] = (p["age_days"] >= 0).astype(int)
    p["log_price"] = np.log(p["price"])
    p["rel_price"] = np.log(p["price"] / p.groupby("category")["price"].transform("median"))
    return p


def segment_by_history(features: pd.DataFrame) -> pd.Series:
    """Faixa de disponibilidade de histórico (usada em avaliação e explicações)."""
    n = features["n_orders"]
    return pd.Series(
        np.select([n == 0, n < 5, n < 15], ["sem histórico", "pouco histórico", "histórico moderado"], "histórico extenso"),
        index=features.index,
    )
