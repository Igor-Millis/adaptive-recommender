"""Modelo bayesiano Beta-Binomial em dois estágios (funil) por cliente × categoria.

Para cada cliente c e categoria k:
    θᴵ_ck = P(engajamento positivo | recomendação de k)   ~ Beta(αᴵ, βᴵ)
    θᶜ_ck = P(compra | engajamento positivo)               ~ Beta(αᶜ, βᶜ)
    P(compra | recomendação) = θᴵ_ck · θᶜ_ck

Evidência (fracionária, ver MATHEMATICAL_FOUNDATIONS.md §10):
    estágio interesse  : engajamento → s += 1 ; rejeição → f += 1 ;
                         impressão sem resposta → f += η (η = 0,25: evidência fraca/ambígua)
    estágio conversão  : compra → s += 1 ; carrinho sem compra → f += 0,5 ;
                         clique/interesse sem compra → f += 1
Uma compra conta como sucesso nos DOIS estágios (evidência mais forte que um clique);
impressão sem clique nunca entra no estágio de conversão.

Prior hierárquico (Bayes empírico):
    * (μ_k, κ_k) por categoria e estágio maximizando a verossimilhança marginal
      Beta-Binomial entre clientes;
    * média do cluster com *leave-one-out* e encolhimento para μ_k;
    * α₀ = μ·κ, β₀ = (1−μ)·κ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit

from config.settings import BAYES, CATEGORIES, RANDOM_SEED
from src.recommenders.base import BaseRecommender, ExpertContext

STAGES = ("interest", "conversion")
STAGE_LABELS = {"interest": "Interesse (engajamento)", "conversion": "Conversão dado interesse"}
_POS = {"click", "interest", "add_to_cart", "purchase"}


def outcome_evidence(outcome: str, cfg=BAYES) -> dict[str, tuple[float, float]]:
    """Evidência (sucessos, fracassos) gerada por um desfecho, em cada estágio."""
    ev: dict[str, tuple[float, float]] = {}
    if outcome in _POS:
        ev["interest"] = (1.0, 0.0)
        if outcome == "purchase":
            ev["conversion"] = (1.0, 0.0)
        elif outcome == "add_to_cart":
            ev["conversion"] = (0.0, cfg.cart_no_purchase_failure_weight)
        else:
            ev["conversion"] = (0.0, 1.0)
    elif outcome == "rejection":
        ev["interest"] = (0.0, cfg.rejection_failure_weight)
    elif outcome == "impression":
        ev["interest"] = (0.0, cfg.impression_failure_weight)
    return ev


def beta_summary(a, b, level: float = BAYES.credible_level) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = stats.beta.ppf((1 - level) / 2, a, b)
    hi = stats.beta.ppf((1 + level) / 2, a, b)
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "sd": np.sqrt(var)}


def fit_empirical_bayes(s: np.ndarray, f: np.ndarray, cfg=BAYES, default_mu: float = 0.2) -> tuple[float, float]:
    """(μ, κ) que maximizam Σ_c log B(μκ+s_c, (1−μ)κ+f_c) − log B(μκ, (1−μ)κ)."""
    mask = (s + f) > 0
    s, f = s[mask], f[mask]
    if len(s) < 3:
        tot = s.sum() + f.sum()
        mu = (s.sum() + 1) / (tot + 2) if tot > 0 else default_mu
        return float(mu), 10.0

    def nll(theta):
        mu = expit(theta[0])
        kappa = np.exp(theta[1])
        a, b = mu * kappa, (1 - mu) * kappa
        return -np.sum(betaln(a + s, b + f) - betaln(a, b))

    mu0 = np.clip((s.sum() + 1) / (s.sum() + f.sum() + 2), 0.01, 0.99)
    res = minimize(nll, x0=[logit(mu0), np.log(10.0)], method="L-BFGS-B",
                   bounds=[(-8, 8), (np.log(cfg.min_prior_strength), np.log(cfg.max_prior_strength))])
    return float(expit(res.x[0])), float(np.exp(res.x[1]))


class BayesianCategoryModel(BaseRecommender):
    name = "bayesian"
    label = "Bayesiano"

    def __init__(self, config=BAYES):
        super().__init__()
        self.cfg = config
        self.K = len(CATEGORIES)

    # ------------------------------------------------------------------ evidência
    def evidence_from_outcomes(self, outcomes: pd.DataFrame, n_customers: int, cidx: dict) -> dict:
        E = {st: np.zeros((n_customers, self.K, 2)) for st in STAGES}
        N = {st: np.zeros((n_customers, self.K), dtype=int) for st in STAGES}
        if outcomes is None or outcomes.empty:
            return {"E": E, "N": N}
        kmap = {c: k for k, c in enumerate(CATEGORIES)}
        for row in outcomes[["customer_id", "category", "outcome"]].itertuples(index=False):
            i = cidx.get(int(row.customer_id))
            k = kmap.get(row.category)
            if i is None or k is None:
                continue
            for st, (sv, fv) in outcome_evidence(row.outcome, self.cfg).items():
                E[st][i, k, 0] += sv
                E[st][i, k, 1] += fv
                N[st][i, k] += 1
        return {"E": E, "N": N}

    # ------------------------------------------------------------------ ajuste
    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        ev = self.evidence_from_outcomes(ctx.outcomes, snapshot.n_customers, snapshot.cidx)
        self.E, self.N = ev["E"], ev["N"]
        labels = ctx.cluster_labels.reindex(snapshot.customer_ids).fillna(-1).astype(int).to_numpy()
        self.alpha0, self.beta0, self.global_prior = {}, {}, {}
        for st in STAGES:
            s, f = self.E[st][..., 0], self.E[st][..., 1]
            a0 = np.zeros_like(s)
            b0 = np.zeros_like(s)
            for k in range(self.K):
                mu_k, kappa_k = fit_empirical_bayes(s[:, k], f[:, k], self.cfg, default_mu=0.2 if st == "interest" else 0.35)
                self.global_prior[(st, CATEGORIES[k])] = (mu_k, kappa_k)
                for i in range(snapshot.n_customers):
                    g = labels[i]
                    if g >= 0:
                        members = (labels == g)
                        members[i] = False  # leave-one-out
                        S, F = s[members, k].sum(), f[members, k].sum()
                        lam = self.cfg.cluster_shrinkage
                        mu = (S + lam * mu_k) / (S + F + lam)
                    else:
                        mu = mu_k
                    a0[i, k] = mu * kappa_k
                    b0[i, k] = (1 - mu) * kappa_k
            self.alpha0[st], self.beta0[st] = a0, b0
        self._posterior_from_evidence()
        return self

    def _posterior_from_evidence(self) -> None:
        self.alpha = {st: self.alpha0[st] + self.E[st][..., 0] for st in STAGES}
        self.beta = {st: self.beta0[st] + self.E[st][..., 1] for st in STAGES}
        self._recompute_scores()

    def _recompute_scores(self) -> None:
        m_i = self.alpha["interest"] / (self.alpha["interest"] + self.beta["interest"])
        m_c = self.alpha["conversion"] / (self.alpha["conversion"] + self.beta["conversion"])
        self.purchase_mean = m_i * m_c  # E[θᴵθᶜ] = E[θᴵ]E[θᶜ] (posteriores independentes)
        cat_idx = self.snapshot.products["category"].map({c: k for k, c in enumerate(CATEGORIES)}).to_numpy()
        pop = self.ctx.product_features["popularity_90d"].reindex(self.snapshot.product_ids).fillna(0).to_numpy(float)
        tie = pop / pop.max() if pop.max() > 0 else pop
        self._cat_idx = cat_idx
        # desempate dentro da categoria por popularidade recente (ε pequeno: não altera a ordem entre categorias)
        self._scores = self.purchase_mean[:, cat_idx] + 1e-4 * tie[None, :]

    # ------------------------------------------------------------------ estado (banco)
    def state_frame(self) -> pd.DataFrame:
        rows = []
        for i, cid in enumerate(self.snapshot.customer_ids):
            for k, cat in enumerate(CATEGORIES):
                for st in STAGES:
                    rows.append({"customer_id": int(cid), "category": cat, "stage": st,
                                 "alpha_prior": float(self.alpha0[st][i, k]), "beta_prior": float(self.beta0[st][i, k]),
                                 "alpha": float(self.alpha[st][i, k]), "beta": float(self.beta[st][i, k]),
                                 "n_events": int(self.N[st][i, k])})
        return pd.DataFrame(rows)

    def load_state(self, snapshot, ctx: ExpertContext, state: pd.DataFrame) -> "BayesianCategoryModel":
        """Carrega o estado incremental persistido (produção) em vez de recalcular do zero."""
        self.snapshot, self.ctx = snapshot, ctx
        n = snapshot.n_customers
        self.alpha0 = {st: np.ones((n, self.K)) for st in STAGES}
        self.beta0 = {st: np.ones((n, self.K)) for st in STAGES}
        self.alpha = {st: np.ones((n, self.K)) for st in STAGES}
        self.beta = {st: np.ones((n, self.K)) for st in STAGES}
        self.N = {st: np.zeros((n, self.K), dtype=int) for st in STAGES}
        self.E = {st: np.zeros((n, self.K, 2)) for st in STAGES}
        kmap = {c: k for k, c in enumerate(CATEGORIES)}
        for r in state.itertuples(index=False):
            i, k = snapshot.cidx.get(int(r.customer_id)), kmap.get(r.category)
            if i is None or k is None:
                continue
            self.alpha0[r.stage][i, k], self.beta0[r.stage][i, k] = r.alpha_prior, r.beta_prior
            self.alpha[r.stage][i, k], self.beta[r.stage][i, k] = r.alpha, r.beta
            self.N[r.stage][i, k] = r.n_events
            self.E[r.stage][i, k] = (r.alpha - r.alpha_prior, r.beta - r.beta_prior)
        self._recompute_scores()
        return self

    # ------------------------------------------------------------------ consultas
    def confidence(self):
        n_eff = self.E["interest"].sum(axis=2)  # evidência efetiva por célula
        kappa = self.alpha0["interest"] + self.beta0["interest"]
        return (n_eff / (n_eff + kappa)).mean(axis=1)

    def customer_summary(self, customer_id: int, n_samples: int | None = None) -> pd.DataFrame:
        i = self._ci(customer_id)
        rng = np.random.default_rng(RANDOM_SEED)
        n_samples = n_samples or self.cfg.mc_samples
        lvl = self.cfg.credible_level
        rows = []
        for k, cat in enumerate(CATEGORIES):
            r = {"category": cat}
            draws = {}
            for st in STAGES:
                a, b = self.alpha[st][i, k], self.beta[st][i, k]
                a0, b0 = self.alpha0[st][i, k], self.beta0[st][i, k]
                lo, hi = stats.beta.ppf([(1 - lvl) / 2, (1 + lvl) / 2], a, b)
                r.update({f"{st}_mean": a / (a + b), f"{st}_low": lo, f"{st}_high": hi,
                          f"{st}_prior_mean": a0 / (a0 + b0), f"{st}_alpha": a, f"{st}_beta": b,
                          f"{st}_n": int(self.N[st][i, k])})
                draws[st] = rng.beta(a, b, size=n_samples)
            prod = draws["interest"] * draws["conversion"]
            r.update({"purchase_mean": float(self.purchase_mean[i, k]),
                      "purchase_low": float(np.quantile(prod, (1 - lvl) / 2)),
                      "purchase_high": float(np.quantile(prod, (1 + lvl) / 2))})
            rows.append(r)
        return pd.DataFrame(rows)

    def trajectory(self, customer_id: int, category: str, outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
        """Reproduz sequencialmente as atualizações (prior → posterior) de um cliente/categoria."""
        i = self._ci(customer_id)
        k = CATEGORIES.index(category)
        out = self.ctx.outcomes if outcomes is None else outcomes
        out = out[(out["customer_id"] == int(customer_id)) & (out["category"] == category)].sort_values("timestamp")
        a = {st: self.alpha0[st][i, k] for st in STAGES}
        b = {st: self.beta0[st][i, k] for st in STAGES}
        lvl = self.cfg.credible_level
        rows = [{"step": 0, "timestamp": None, "outcome": "prior", **self._traj_row(a, b, lvl)}]
        for n, r in enumerate(out.itertuples(index=False), start=1):
            for st, (sv, fv) in outcome_evidence(r.outcome, self.cfg).items():
                a[st] += sv
                b[st] += fv
            rows.append({"step": n, "timestamp": r.timestamp, "outcome": r.outcome, **self._traj_row(a, b, lvl)})
        return pd.DataFrame(rows)

    @staticmethod
    def _traj_row(a, b, lvl):
        out = {}
        for st in STAGES:
            lo, hi = stats.beta.ppf([(1 - lvl) / 2, (1 + lvl) / 2], a[st], b[st])
            out.update({f"{st}_mean": a[st] / (a[st] + b[st]), f"{st}_low": lo, f"{st}_high": hi,
                        f"{st}_alpha": a[st], f"{st}_beta": b[st]})
        return out

    def thompson_scores(self, seed: int | None = None) -> np.ndarray:
        """Amostragem de Thompson (exploração): uma amostra posterior por célula."""
        rng = np.random.default_rng(seed)
        si = rng.beta(self.alpha["interest"], self.beta["interest"])
        sc = rng.beta(self.alpha["conversion"], self.beta["conversion"])
        return (si * sc)[:, self._cat_idx]

    def explain(self, customer_id, product_id):
        i, j = self._ci(customer_id), self._pj(product_id)
        k = int(self._cat_idx[j])
        cat = CATEGORIES[k]
        a, b = self.alpha["interest"][i, k], self.beta["interest"][i, k]
        lo, hi = stats.beta.ppf([0.05, 0.95], a, b)
        n = int(self.N["interest"][i, k])
        summary = (f"Taxa estimada de resposta positiva a recomendações de {cat}: {a / (a + b):.0%} "
                   f"(IC 90%: {lo:.0%}–{hi:.0%}, {n} exposições observadas); "
                   f"P(compra | recomendação) ≈ {self.purchase_mean[i, k]:.1%}.")
        return {"summary": summary, "details": {"categoria": cat, "alpha": float(a), "beta": float(b),
                                                "p_compra": float(self.purchase_mean[i, k]), "n_exposicoes": n}}
