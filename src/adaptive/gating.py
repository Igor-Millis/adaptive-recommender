"""Mecanismos de gating (seleção/ponderação de especialistas).

* ``FixedGating``   — mesmos pesos para todos (baseline "híbrido de pesos fixos").
* ``RuleGating``    — w_e(c) = r_e(c) / Σ r(c) a partir de regras interpretáveis.
* ``LearnedGating`` — *gating residual* treinável:

      z_e(c) = log r_e(c) + b_e + θ_eᵀ φ̃(c),     w(c) = softmax(z(c))

  em que φ̃ são os sinais padronizados. Com b = θ = 0 recupera exatamente as
  regras; a regularização L2 (prior gaussiano centrado nas regras) encolhe os
  ajustes quando os dados de validação são escassos. Especialistas
  indisponíveis (r_e = 0) recebem peso 0 (logit −∞).

  Treino: perda listwise (softmax cross-entropy) sobre os candidatos de cada
  cliente na janela de validação,

      L = (1/|C|) Σ_c −(1/|P_c|) Σ_{p∈P_c} log softmax_q( s(c,q)/T )_p  +  (λ_b/2)‖b‖² + (λ/2)‖θ‖²,
      s(c,q) = Σ_e w_e(c) · s̃_e(c,q)

  com gradiente analítico e L-BFGS. O viés b_e (qualidade média do especialista,
  "desempenho histórico") tem penalidade fraca λ_b; os ajustes contextuais θ têm
  λ escolhido por validação cruzada K-fold sobre clientes (NDCG@10 fora da dobra).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from config.settings import EXPERTS, GATING, RANDOM_SEED
from src.adaptive.model_selection import SIGNAL_COLUMNS, normalize_weights, rule_reliabilities
from src.evaluation.metrics import ndcg_at_k
from src.utils.validation import validate_weights

EPS = 1e-9


class FixedGating:
    method = "fixed"

    def __init__(self, weights: dict[str, float], experts: list[str] = EXPERTS):
        self.experts = experts
        w = np.array([weights.get(e, 0.0) for e in experts], float)
        self.w = w / w.sum()

    def weights(self, signals: pd.DataFrame) -> np.ndarray:
        W = np.tile(self.w, (len(signals), 1))
        # respeita disponibilidade (ex.: CF sem histórico) e renormaliza
        avail = rule_reliabilities(signals, self.experts).to_numpy() > 0
        W = np.where(avail, W, 0.0)
        return validate_weights(normalize_weights(W))


class RuleGating:
    method = "rules"

    def __init__(self, experts: list[str] = EXPERTS):
        self.experts = experts

    def weights(self, signals: pd.DataFrame) -> np.ndarray:
        R = rule_reliabilities(signals, self.experts).to_numpy()
        return validate_weights(normalize_weights(R))


class LearnedGating:
    method = "learned"

    def __init__(self, experts: list[str] = EXPERTS, features: list[str] = SIGNAL_COLUMNS, temperature: float = GATING.temperature):
        self.experts = experts
        self.features = features
        self.T = temperature
        E, F = len(experts), len(features)
        self.b = np.zeros(E)
        self.theta = np.zeros((E, F))
        self.mu = np.zeros(F)
        self.sd = np.ones(F)
        self.l2 = None
        self.cv_table: list[dict] = []
        self.train_info: dict = {}

    # ------------------------------------------------------------------ forward
    def _phi(self, signals: pd.DataFrame) -> np.ndarray:
        return (signals[self.features].to_numpy(float) - self.mu) / self.sd

    def _logits(self, signals: pd.DataFrame, b=None, theta=None) -> np.ndarray:
        b = self.b if b is None else b
        theta = self.theta if theta is None else theta
        R = rule_reliabilities(signals, self.experts).to_numpy()
        with np.errstate(divide="ignore"):
            base = np.where(R > 0, np.log(R + EPS), -np.inf)
        return base + b[None, :] + self._phi(signals) @ theta.T

    def weights(self, signals: pd.DataFrame) -> np.ndarray:
        Z = self._logits(signals)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        W = np.exp(Z)
        return validate_weights(W / W.sum(axis=1, keepdims=True))

    def adjustments(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Ajuste aprendido no logit de cada especialista (positivo = ganhou peso em relação às regras)."""
        A = self.b[None, :] + self._phi(signals) @ self.theta.T
        return pd.DataFrame(A, index=signals.index, columns=self.experts)

    # ------------------------------------------------------------------ treino
    @staticmethod
    def _prepare(S: dict[str, np.ndarray], Y: np.ndarray, mask: np.ndarray, experts):
        stack = np.stack([np.where(mask, S[e], 0.0) for e in experts], axis=1)  # (n_c, E, n_p)
        rows = np.where((Y & mask).sum(axis=1) > 0)[0]
        return stack, rows

    def _loss_grad(self, params, stack, Y, mask, logR, Phi, rows, l2):
        E, F = len(self.experts), Phi.shape[1]
        b, theta = params[:E], params[E:].reshape(E, F)
        Z = logR[rows] + b[None, :] + Phi[rows] @ theta.T
        Z = Z - Z.max(axis=1, keepdims=True)
        W = np.exp(Z)
        W /= W.sum(axis=1, keepdims=True)
        St = stack[rows]  # (n, E, P)
        s = np.einsum("ne,nep->np", W, St) / self.T
        m = mask[rows]
        s = np.where(m, s, -np.inf)
        lse = logsumexp(s, axis=1)
        y = (Y[rows] & m).astype(float)
        npos = y.sum(axis=1)
        loss_c = -((np.where(m, s, 0.0) * y).sum(axis=1) / npos - lse)
        pi = np.where(m, np.exp(s - lse[:, None]), 0.0)
        dL_ds = (pi - y / npos[:, None]) / self.T  # (n, P)
        g = np.einsum("np,nep->ne", dL_ds, St)  # dL/dw
        dz = W * (g - (W * g).sum(axis=1, keepdims=True))
        n = len(rows)
        lb = GATING.bias_l2
        grad_b = dz.sum(axis=0) / n + lb * b
        grad_t = dz.T @ Phi[rows] / n + l2 * theta
        loss = loss_c.mean() + 0.5 * lb * np.sum(b ** 2) + 0.5 * l2 * np.sum(theta ** 2)
        return loss, np.concatenate([grad_b, grad_t.ravel()])

    def _fit_params(self, stack, Y, mask, logR, Phi, rows, l2):
        """l2 = inf → só o viés b é ajustado (θ ≡ 0): "pesos globais por especialista sobre as regras"."""
        E, F = len(self.experts), Phi.shape[1]
        if not np.isfinite(l2):
            def f_b(b):
                loss, g = self._loss_grad(np.concatenate([b, np.zeros(E * F)]), stack, Y, mask, logR, Phi, rows, 0.0)
                return loss, g[:E]
            res = minimize(f_b, np.zeros(E), jac=True, method="L-BFGS-B", options={"maxiter": 300})
            return res.x, np.zeros((E, F)), res
        x0 = np.zeros(E + E * F)
        res = minimize(self._loss_grad, x0, args=(stack, Y, mask, logR, Phi, rows, l2), jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        return res.x[:E], res.x[E:].reshape(E, F), res

    def fit(self, signals: pd.DataFrame, S: dict[str, np.ndarray], Y: np.ndarray, mask: np.ndarray,
            l2_grid=GATING.l2_grid, n_folds: int = GATING.cv_folds, k: int = 10) -> "LearnedGating":
        """S: scores normalizados (s̃) por especialista; Y: compras na janela; mask: candidatos."""
        X = signals[self.features].to_numpy(float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Phi = (X - self.mu) / self.sd
        R = rule_reliabilities(signals, self.experts).to_numpy()
        with np.errstate(divide="ignore"):
            logR = np.where(R > 0, np.log(R + EPS), -1e4)  # -1e4 ≈ -∞ numericamente estável
        stack, rows = self._prepare(S, Y, mask, self.experts)
        rng = np.random.default_rng(RANDOM_SEED)
        folds = np.array_split(rng.permutation(rows), n_folds)

        def ndcg_for(b, theta, eval_rows):
            Z = logR[eval_rows] + b[None, :] + Phi[eval_rows] @ theta.T
            Z -= Z.max(axis=1, keepdims=True)
            W = np.exp(Z)
            W /= W.sum(axis=1, keepdims=True)
            sc = np.einsum("ne,nep->np", W, stack[eval_rows])
            sc = np.where(mask[eval_rows], sc, -np.inf)
            vals = []
            for r, row in zip(eval_rows, sc):
                ranked = list(np.argsort(-row, kind="stable")[:k])
                vals.append(ndcg_at_k(ranked, set(np.where(Y[r] & mask[r])[0]), k))
            return float(np.mean(vals))

        # Seleção de complexidade por CV: regras puras (sem parâmetros), só viés (λ=∞) ou viés + θ(λ)
        rule_scores = [ndcg_for(np.zeros(len(self.experts)), np.zeros_like(self.theta), folds[f]) for f in range(n_folds)]
        self.cv_table = [{"variant": "regras", "l2": None, "cv_ndcg": float(np.mean(rule_scores)),
                          "cv_ndcg_sd": float(np.std(rule_scores))}]
        for l2 in [np.inf] + list(l2_grid):
            scores = []
            for f in range(n_folds):
                test = folds[f]
                train = np.concatenate([folds[j] for j in range(n_folds) if j != f])
                b, theta, _ = self._fit_params(stack, Y, mask, logR, Phi, train, l2)
                scores.append(ndcg_for(b, theta, test))
            self.cv_table.append({"variant": "viés" if not np.isfinite(l2) else "viés+contexto",
                                  "l2": None if not np.isfinite(l2) else l2,
                                  "cv_ndcg": float(np.mean(scores)), "cv_ndcg_sd": float(np.std(scores))})
        # empate (±0,001) favorece o modelo mais simples (ordem da tabela)
        top = max(r["cv_ndcg"] for r in self.cv_table)
        best = next(r for r in self.cv_table if r["cv_ndcg"] >= top - 1e-3)
        self.variant = best["variant"]
        if best["variant"] == "regras":
            self.l2 = None
            self.b, self.theta = np.zeros(len(self.experts)), np.zeros_like(self.theta)
            converged, final = True, None
        else:
            l2 = np.inf if best["l2"] is None else best["l2"]
            self.l2 = best["l2"]
            self.b, self.theta, res = self._fit_params(stack, Y, mask, logR, Phi, rows, l2)
            converged, final = bool(res.success), float(res.fun)
        self.train_info = {"n_customers": int(len(rows)), "variant": self.variant, "l2": self.l2, "cv": self.cv_table,
                           "cv_rules_ndcg": float(np.mean(rule_scores)), "converged": converged, "final_loss": final}
        return self

    # ------------------------------------------------------------------ persistência
    def to_dict(self) -> dict:
        return {"experts": self.experts, "features": self.features, "T": self.T, "b": self.b.tolist(),
                "theta": self.theta.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "l2": self.l2,
                "variant": getattr(self, "variant", None), "train_info": self.train_info}

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedGating":
        g = cls(d["experts"], d["features"], d["T"])
        g.b, g.theta = np.array(d["b"]), np.array(d["theta"])
        g.mu, g.sd = np.array(d["mu"]), np.array(d["sd"])
        g.l2, g.train_info = d.get("l2"), d.get("train_info", {})
        g.variant = d.get("variant")
        return g


def tune_fixed_weights(S: dict[str, np.ndarray], Y: np.ndarray, mask: np.ndarray, signals: pd.DataFrame,
                       experts: list[str] = EXPERTS, step: float = 0.1, k: int = 10) -> tuple[dict, float]:
    """Busca em grade no simplex (passo 0,1) dos pesos fixos que maximizam o NDCG@k médio na validação."""
    grid = _simplex_grid(len(experts), int(round(1 / step)))
    stack = np.stack([np.where(mask, S[e], 0.0) for e in experts], axis=1)
    rows = np.where((Y & mask).sum(axis=1) > 0)[0]
    rel = [set(np.where(Y[r] & mask[r])[0]) for r in rows]
    avail = rule_reliabilities(signals, experts).to_numpy() > 0
    best_w, best_v = None, -1.0
    for w in grid:
        W = np.where(avail[rows], w[None, :], 0.0)
        W = normalize_weights(W)
        sc = np.einsum("ne,nep->np", W, stack[rows])
        sc = np.where(mask[rows], sc, -np.inf)
        top = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        v = np.mean([ndcg_at_k(list(t), r, k) for t, r in zip(top, rel)])
        if v > best_v + 1e-9:
            best_w, best_v = w, v
    return dict(zip(experts, map(float, best_w))), float(best_v)


def _simplex_grid(n: int, total: int) -> list[np.ndarray]:
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(np.array(prefix + [remaining], float) / total)
            return
        for i in range(remaining + 1):
            rec(prefix + [i], remaining - i, slots - 1)

    rec([], total, n)
    return out
