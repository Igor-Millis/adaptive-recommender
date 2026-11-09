"""Fatoração de matrizes para feedback implícito (ALS, Hu, Koren & Volinsky, 2008).

Preferência binária P = 1[R > 0] e confiança C = 1 + α·R. Minimiza
    Σ_{c,p} C_cp (P_cp − x_cᵀ y_p)² + λ (Σ‖x_c‖² + Σ‖y_p‖²)
por mínimos quadrados alternados, com solução fechada por linha:
    x_c = (YᵀY + Yᵀ(C^c − I)Y + λI)⁻¹ Yᵀ C^c P_c

Implementação própria em NumPy (dimensão pequena: 50 × 200). É avaliado como
especialista alternativo ao user-kNN (ver MODEL_EVALUATION.md).
"""
from __future__ import annotations

import numpy as np

from config.settings import CF, MF, RANDOM_SEED
from src.recommenders.base import BaseRecommender, ExpertContext
from src.recommenders.collaborative import implicit_matrix


class ImplicitALSRecommender(BaseRecommender):
    name = "collaborative_mf"
    label = "Fatoração de Matrizes (ALS)"

    def __init__(self, config=MF):
        super().__init__()
        self.cfg = config

    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        R = implicit_matrix(snapshot, ctx.outcomes, CF.half_life_days, CF.engagement_weight)
        self.R = R
        n_u, n_i = R.shape
        f, lam, alpha = self.cfg.n_factors, self.cfg.regularization, self.cfg.alpha
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(0, 0.1, (n_u, f))
        Y = rng.normal(0, 0.1, (n_i, f))
        P = (R > 0).astype(float)
        C = 1.0 + alpha * R
        I = np.eye(f)
        self.losses = []
        for _ in range(self.cfg.iterations):
            YtY = Y.T @ Y
            for u in range(n_u):
                cu = C[u]
                A = YtY + (Y.T * (cu - 1)) @ Y + lam * I
                X[u] = np.linalg.solve(A, (Y.T * cu) @ P[u])
            XtX = X.T @ X
            for i in range(n_i):
                ci = C[:, i]
                A = XtX + (X.T * (ci - 1)) @ X + lam * I
                Y[i] = np.linalg.solve(A, (X.T * ci) @ P[:, i])
            E = P - X @ Y.T
            self.losses.append(float((C * E * E).sum() + lam * ((X ** 2).sum() + (Y ** 2).sum())))
        has = R.sum(axis=1) > 0
        X[~has] = 0.0  # sem histórico: sem fator latente identificável
        self.Xu, self.Yi = X, Y
        self._scores = X @ Y.T
        return self

    def confidence(self):
        n = (self.R > 0).sum(axis=1)
        return np.where(n > 0, 1 - np.exp(-n / 20.0), 0.0)

    def explain(self, customer_id, product_id):
        i, j = self._ci(customer_id), self._pj(product_id)
        return {"summary": "Afinidade em fatores latentes aprendidos da matriz cliente-produto.",
                "details": {"score": float(self._scores[i, j])}}
