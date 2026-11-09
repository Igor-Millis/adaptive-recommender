"""Modelo de utilidade latente e de resposta a recomendações.

O mesmo modelo de utilidade governa (a) a escolha de produtos nos pedidos e
(b) a resposta do lojista às recomendações exibidas pelo sistema legado. Isso
garante coerência: o que o cliente compra organicamente e como ele reage a
recomendações são manifestações da mesma preferência latente.

Funil de resposta a uma impressão (``z`` = utilidade padronizada do produto):

    P(engajar)          = σ(-2.0 + 1.5 z)
    P(comprar | engajar) = σ(-0.6 + 1.0 z)   (somente em sessões que antecedem pedido)
    P(rejeitar | ¬engajar) = σ(-3.2 - 1.2 z)

Ausência de resposta permanece como "impressão" — não é registrada como rejeição.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CATEGORIES
from src.data_generation.generate_customers import AUDIENCES, STYLES, LatentCustomer

BANDS = ["Econômica", "Intermediária", "Premium"]
COMPLEMENTS = {
    "Calças": "Camisas",
    "Camisas": "Calças",
    "Vestidos": "Acessórios",
    "Saias": "Camisas",
    "Camisetas": "Bermudas",
    "Bermudas": "Camisetas",
    "Roupas esportivas": "Acessórios",
    "Jaquetas": "Calças",
}
CHANNELS = ["portal_b2b", "email_semanal", "representante_comercial"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ProductArrays:
    """Atributos do catálogo em arrays NumPy alinhados (para cálculo vetorizado)."""

    def __init__(self, products: pd.DataFrame, seed: int = 7):
        rng = np.random.default_rng(seed)
        p = products.sort_values("product_id").reset_index(drop=True)
        self.df = p
        self.ids = p["product_id"].to_numpy()
        self.cat_idx = p["category"].map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy()
        self.style_idx = p["_style"].map({s: i for i, s in enumerate(STYLES)}).to_numpy()
        self.aud_idx = p["target_audience"].map({a: i for i, a in enumerate(AUDIENCES)}).to_numpy()
        self.band_idx = p["price_band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
        cat_median = p.groupby("category")["price"].transform("median")
        self.log_rel_price = np.log(p["price"].to_numpy() / cat_median.to_numpy())
        self.price = p["price"].to_numpy()
        self.season_sign = p["season"].map({"Verão": 1.0, "Inverno": -1.0, "Atemporal": 0.0}).to_numpy()
        self.launch = p["launch_date"].to_numpy(dtype="datetime64[ns]")
        # "qualidade" latente do produto: gera produtos naturalmente mais vendidos
        self.quality = rng.normal(0.0, 0.45, size=len(p))
        self.cat_of = p["category"].to_numpy()


def drift_weight(lat: LatentCustomer, date: pd.Timestamp) -> float:
    if lat.drift_start is None:
        return 0.0
    s, e = pd.Timestamp(lat.drift_start), pd.Timestamp(lat.drift_end)
    if date <= s:
        return 0.0
    if date >= e:
        return 1.0
    return float((date - s) / (e - s))


def utility(lat: LatentCustomer, pa: ProductArrays, date: pd.Timestamp, bought_before: np.ndarray) -> np.ndarray:
    w = drift_weight(lat, date)
    cat_pref = np.array([(1 - w) * lat.cat_pref_start[c] + w * lat.cat_pref_end[c] for c in CATEGORIES])
    sty_pref = np.array([(1 - w) * lat.style_pref_start[s] + w * lat.style_pref_end[s] for s in STYLES])
    aud_pref = np.array([lat.audience_pref[a] for a in AUDIENCES])
    summer = np.cos(2 * np.pi * (date.month - 1) / 12.0)  # +1 em janeiro, -1 em julho
    days_since_launch = (np.datetime64(date) - pa.launch).astype("timedelta64[D]").astype(float)
    novelty = np.where((days_since_launch >= 0) & (days_since_launch < 90), 1.0, 0.0)
    band_pref = BANDS.index(lat.preferred_band)
    u = (
        1.3 * np.log(cat_pref[pa.cat_idx] + 1e-3)
        + 1.0 * np.log(sty_pref[pa.style_idx] + 0.02)
        + 0.6 * np.log(aud_pref[pa.aud_idx] + 0.02)
        - lat.price_sensitivity * pa.log_rel_price * 1.6
        + 0.5 * (pa.band_idx == band_pref)
        + 0.8 * pa.season_sign * summer
        + pa.quality
        + 0.35 * novelty
        + 0.9 * bought_before  # recompra é comum em atacado de moda
    )
    return u


def response_to_impression(z: float, rng: np.random.Generator, can_purchase: bool) -> list[str]:
    """Retorna a sequência de eventos após uma impressão (inclui 'impression')."""
    events = ["impression"]
    if rng.random() < _sigmoid(-2.0 + 1.5 * z):
        if can_purchase and rng.random() < _sigmoid(-0.6 + 1.0 * z):
            events += ["click", "add_to_cart", "purchase"]
        else:
            last = rng.choice(["click", "interest", "add_to_cart"], p=[0.55, 0.25, 0.20])
            events += ["click"] if last == "click" else ["click", str(last)]
    elif rng.random() < _sigmoid(-3.2 - 1.2 * z):
        events.append("rejection")
    return events
