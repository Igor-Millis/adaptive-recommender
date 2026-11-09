"""Regras de elegibilidade de candidatos (o que PODE ser recomendado)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import RANKING, RankingConfig
from src.preprocessing.feature_engineering import POSITIVE_OUTCOMES


def candidate_mask(snapshot, cfg: RankingConfig = RANKING, outcomes: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Máscara (n_c × n_p) de candidatos + vetor (n_c,) de exclusões por motivo (por cliente).

    1. produto lançado até ``as_of``;
    2. com estoque e ativo (se ``require_stock``);
    3. regra de compra prévia: ``all`` (descoberta), ``recent`` (permite recompra após N dias) ou ``none``;
    4. bloqueio de produtos rejeitados explicitamente nos últimos ``rejection_block_days``.
    """
    nc, npr = snapshot.n_customers, snapshot.n_products
    mask = np.tile(snapshot.launched, (nc, 1))
    reasons = {"nao_lancado": np.full(nc, int((~snapshot.launched).sum()))}
    if cfg.require_stock:
        reasons["sem_estoque"] = (mask & ~snapshot.in_stock[None, :]).sum(axis=1)
        mask &= snapshot.in_stock[None, :]
    if cfg.purchased_rule == "all":
        bought = snapshot.purchased_matrix()
    elif cfg.purchased_rule == "recent":
        bought = snapshot.purchased_matrix(since=snapshot.as_of - pd.Timedelta(days=cfg.recent_days))
    else:
        bought = np.zeros((nc, npr), dtype=bool)
    reasons["ja_comprado"] = (mask & bought).sum(axis=1)
    mask &= ~bought
    if outcomes is not None and len(outcomes):
        cut = snapshot.as_of - pd.Timedelta(days=cfg.rejection_block_days)
        rej = outcomes[(outcomes["outcome"] == "rejection") & (outcomes["timestamp"] >= cut)]
        R = np.zeros((nc, npr), dtype=bool)
        if len(rej):
            R[rej["customer_id"].map(snapshot.cidx).to_numpy(), rej["product_id"].map(snapshot.pidx).to_numpy()] = True
        reasons["rejeitado_recentemente"] = (mask & R).sum(axis=1)
        mask &= ~R
    return mask, reasons


def fatigue_penalty(snapshot, outcomes: pd.DataFrame, cfg: RankingConfig = RANKING, days: int = 60) -> np.ndarray:
    """Penalidade multiplicativa para produtos exibidos repetidamente sem resposta (evita repetição)."""
    P = np.zeros((snapshot.n_customers, snapshot.n_products))
    if outcomes is None or outcomes.empty:
        return P
    o = outcomes[outcomes["timestamp"] >= snapshot.as_of - pd.Timedelta(days=days)]
    if o.empty:
        return P
    ignored = o[~o["outcome"].isin(POSITIVE_OUTCOMES)]
    cnt = ignored.groupby(["customer_id", "product_id"]).size()
    for (c, p), n in cnt.items():
        if n >= cfg.fatigue_impressions and c in snapshot.cidx and p in snapshot.pidx:
            P[snapshot.cidx[c], snapshot.pidx[p]] = cfg.fatigue_penalty
    return P
