"""Interface padronizada dos modelos especialistas.

Todo especialista:
  * ``fit(snapshot, ctx)`` — aprende usando somente o snapshot (dados < as_of);
  * ``score()``            — matriz (n_clientes × n_produtos) de scores brutos;
  * ``confidence()``       — vetor (n_clientes,) em [0, 1]: quanta evidência o
                             especialista tem para cada cliente (0 = indisponível);
  * ``explain(c, p)``      — dicionário com motivo legível e detalhes técnicos.

Novos especialistas (ex.: embeddings neurais, two-tower, sequenciais) só
precisam implementar esta interface para entrar no motor adaptativo — a
arquitetura de gating é agnóstica ao número de especialistas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.preprocessing.snapshot import DataSnapshot


@dataclass
class ExpertContext:
    """Informação compartilhada entre especialistas para um mesmo snapshot."""

    customer_features: pd.DataFrame
    product_features: pd.DataFrame
    outcomes: pd.DataFrame  # desfechos resolvidos por (sessão, produto)
    cluster_labels: pd.Series  # customer_id -> cluster (-1 sem cluster)
    cluster_names: dict = field(default_factory=dict)


class BaseRecommender(ABC):
    name: str = "base"
    label: str = "Base"

    def __init__(self):
        self.snapshot: DataSnapshot | None = None
        self.ctx: ExpertContext | None = None
        self._scores: np.ndarray | None = None

    @abstractmethod
    def fit(self, snapshot: DataSnapshot, ctx: ExpertContext) -> "BaseRecommender": ...

    def score(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError(f"{self.name}: chame fit() antes de score().")
        return self._scores

    def confidence(self) -> np.ndarray:
        return np.ones(self.snapshot.n_customers)

    def explain(self, customer_id: int, product_id: int) -> dict:
        return {"summary": "", "details": {}}

    # utilidades -----------------------------------------------------------
    def _ci(self, customer_id: int) -> int:
        return self.snapshot.cidx[int(customer_id)]

    def _pj(self, product_id: int) -> int:
        return self.snapshot.pidx[int(product_id)]

    def top_n(self, customer_id: int, n: int = 10, mask: np.ndarray | None = None) -> list[int]:
        s = self.score()[self._ci(customer_id)].copy()
        if mask is not None:
            s[~mask] = -np.inf
        idx = np.argsort(-s, kind="stable")[:n]
        return [int(self.snapshot.product_ids[j]) for j in idx if np.isfinite(s[j])]
