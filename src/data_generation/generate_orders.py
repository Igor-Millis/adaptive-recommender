"""Simulação cronológica de pedidos e de interações com o recomendador legado.

1. Para cada cliente e mês, o nº de pedidos segue Poisson(taxa × fator sazonal).
2. Todos os eventos (pedidos e sessões de navegação) são ordenados no tempo e
   processados sequencialmente, mantendo contagens globais de vendas — o
   recomendador legado de popularidade só enxerga o passado.
3. Antes de cada pedido há uma sessão em que o legado exibe 6 produtos; as
   respostas seguem o funil de ``generate_interactions``. Produtos comprados via
   recomendação entram como itens do próprio pedido (coerência entre tabelas).
4. Os itens orgânicos são escolhidos por Gumbel-top-k sobre a utilidade latente,
   com adição de itens complementares.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np
import pandas as pd

from config.settings import DATA_END, DATA_START
from src.data_generation.generate_customers import LatentCustomer
from src.data_generation.generate_interactions import (
    CHANNELS,
    COMPLEMENTS,
    ProductArrays,
    response_to_impression,
    utility,
)

LEGACY_POP = "legado_popularidade"
LEGACY_RANDOM = "legado_exploracao"


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"))


def simulate(customers: pd.DataFrame, latents: list[LatentCustomer], products: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    pa = ProductArrays(products, seed=seed + 1)
    n_prod = len(pa.ids)
    reg = dict(zip(customers["customer_id"], pd.to_datetime(customers["registration_date"])))

    # ------------------------------------------------------------------ eventos
    events = []  # (timestamp, customer_id, kind)
    for lat in latents:
        if lat.never_orders:
            continue
        start = max(DATA_START, reg[lat.customer_id])
        for ms in _month_starts(start, DATA_END - pd.Timedelta(days=1)):
            month_end = min(ms + pd.offsets.MonthBegin(1), DATA_END)
            lo = max(ms, start)
            if lo >= month_end:
                continue
            frac = (month_end - lo).days / max((month_end - ms).days, 1)
            seas = 1.0 + lat.seasonal_amplitude * np.cos(2 * np.pi * (ms.month - lat.seasonal_peak_month) / 12.0)
            # novos clientes começam com taxa menor (rampa de 3 meses)
            months_active = (ms - reg[lat.customer_id]).days / 30.0
            ramp = 0.6 if months_active < 3 else 1.0
            lam = max(lat.orders_per_month * seas * frac * ramp, 0.03)
            n_orders = rng.poisson(lam)
            if months_active < 1 and n_orders == 0 and lat.history_profile.startswith("novo"):
                n_orders = 1  # garante 1º pedido de clientes recém-cadastrados
            span = (month_end - lo).total_seconds()
            for _ in range(n_orders):
                t = lo + pd.Timedelta(seconds=float(rng.uniform(0, span)))
                events.append((t, lat.customer_id, "order"))
            for _ in range(rng.poisson(0.5 * frac)):
                t = lo + pd.Timedelta(seconds=float(rng.uniform(0, span)))
                events.append((t, lat.customer_id, "browse"))
    events.sort(key=lambda e: e[0])

    lat_by_id = {lat.customer_id: lat for lat in latents}
    bought = {cid: np.zeros(n_prod) for cid in lat_by_id}
    last_buy = {cid: np.full(n_prod, np.datetime64("NaT"), dtype="datetime64[ns]") for cid in lat_by_id}
    sales_window: deque = deque()  # (data, idx produto) dos últimos 120 dias — popularidade do legado
    pop_counts = np.zeros(n_prod)

    orders, items, interactions = [], [], []
    order_id = item_id = inter_id = session_id = 0

    def legacy_slate(cid: int, t: pd.Timestamp, avail: np.ndarray) -> list[tuple[int, str]]:
        """Recomendador legado: 70% populares (últimos 120 dias), 30% exploração aleatória."""
        t64 = np.datetime64(t)
        recent = t64 - np.timedelta64(90, "D")
        not_recent = ~(last_buy[cid] >= recent)
        cand = np.where(avail & not_recent)[0]
        if len(cand) == 0:
            return []
        window = t64 - np.timedelta64(120, "D")
        while sales_window and sales_window[0][0] < window:  # janela deslizante (eventos em ordem cronológica)
            _, j_old = sales_window.popleft()
            pop_counts[j_old] -= 1
        counts = pop_counts
        pop_order = cand[np.argsort(-counts[cand], kind="stable")][:30]
        slate: list[tuple[int, str]] = []
        for pos in range(6):
            if rng.random() < 0.7 and len(pop_order):
                j = int(rng.choice(pop_order))
                src = LEGACY_POP
            else:
                j = int(rng.choice(cand))
                src = LEGACY_RANDOM
            if j not in [s[0] for s in slate]:
                slate.append((j, src))
        return slate

    for t, cid, kind in events:
        lat = lat_by_id[cid]
        t64 = np.datetime64(t)
        avail = pa.launch <= t64
        u = utility(lat, pa, t, (bought[cid] > 0).astype(float))
        u_av = u[avail]
        z_all = (u - u_av.mean()) / (u_av.std() + 1e-9)

        # ---------------- sessão de recomendação do legado
        session_id += 1
        channel = str(rng.choice(CHANNELS, p=[0.6, 0.25, 0.15]))
        session_start = t - pd.Timedelta(minutes=int(rng.integers(20, 240)))
        rec_purchases: list[int] = []
        for pos, (j, src) in enumerate(legacy_slate(cid, t, avail), start=1):
            evs = response_to_impression(float(z_all[j]), rng, can_purchase=(kind == "order"))
            for k, ev in enumerate(evs):
                inter_id += 1
                interactions.append(
                    {
                        "interaction_id": inter_id,
                        "session_id": session_id,
                        "customer_id": cid,
                        "product_id": int(pa.ids[j]),
                        "model_name": src,
                        "interaction_type": ev,
                        "timestamp": session_start + pd.Timedelta(minutes=pos + 2 * k),
                        "context": json.dumps({"canal": channel, "posicao": pos, "origem": "historico_simulado"}, ensure_ascii=False),
                        "recommendation_id": None,
                    }
                )
            if "purchase" in evs:
                rec_purchases.append(j)

        if kind != "order":
            continue

        # ---------------- pedido
        n_lines = int(min(1 + rng.poisson(max(0.8 * lat.lines_per_order - 1, 0.5)), 16))
        gumbel = u + rng.gumbel(size=n_prod)
        gumbel[~avail] = -np.inf
        chosen = list(dict.fromkeys(rec_purchases))
        for j in np.argsort(-gumbel):
            if len(chosen) >= n_lines + len(rec_purchases):
                break
            if int(j) not in chosen:
                chosen.append(int(j))
        # complementares
        for j in list(chosen):
            comp = COMPLEMENTS.get(pa.cat_of[j])
            if comp and rng.random() < lat.complementarity:
                mask = (pa.cat_of == comp) & avail
                mask[chosen] = False
                if mask.any():
                    g = np.where(mask, gumbel, -np.inf)
                    chosen.append(int(np.argmax(g)))

        order_id += 1
        orders.append({"order_id": order_id, "customer_id": cid, "order_date": t, "channel": channel})
        for j in chosen:
            qty = max(1, int(round(lat.qty_base * np.exp(rng.normal(0, 0.45)) * (1 / np.exp(pa.log_rel_price[j])) ** 0.3)))
            disc = 0.02 + 0.035 * lat.price_sensitivity + (0.02 if qty > 1.5 * lat.qty_base else 0.0) + rng.normal(0, 0.012)
            disc = float(np.round(np.clip(disc, 0.0, 0.22), 3))
            unit = float(pa.price[j])
            item_id += 1
            items.append(
                {
                    "item_id": item_id,
                    "order_id": order_id,
                    "product_id": int(pa.ids[j]),
                    "quantity": qty,
                    "unit_price": unit,
                    "discount": disc,
                    "total_value": round(qty * unit * (1 - disc), 2),
                }
            )
            bought[cid][j] += qty
            last_buy[cid][j] = t64
            sales_window.append((t64, j))
            pop_counts[j] += 1

    orders_df = pd.DataFrame(orders)
    items_df = pd.DataFrame(items)
    inter_df = pd.DataFrame(interactions)
    return orders_df, items_df, inter_df, pa.quality
