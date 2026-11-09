"""Sinais de evidência por cliente e seleção de especialistas baseada em regras.

Os sinais descrevem *quanto e que tipo* de informação existe sobre o cliente no
momento da recomendação (todos calculados com dados < as_of). A versão por
regras converte sinais em uma confiabilidade r_e(c) ∈ [0, 1] por especialista
com funções de saturação interpretáveis; os pesos são w_e = r_e / Σ r.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CB, EXPERT_LABELS, EXPERTS

SIGNAL_COLUMNS = ["log_orders", "log_distinct", "log_impressions", "density", "diversity", "neighbor_strength",
                  "log_recency", "drift", "consistency", "bayes_evidence", "has_history"]
SIGNAL_LABELS = {
    "n_orders": "pedidos no histórico", "n_distinct_products": "produtos distintos comprados",
    "n_impressions": "recomendações exibidas", "n_engagements": "interações positivas",
    "density": "densidade do histórico", "diversity": "diversidade de categorias",
    "neighbor_strength": "similaridade com outros clientes", "recency_days": "dias desde o último pedido",
    "drift": "mudança recente de preferência", "consistency": "consistência do histórico",
    "bayes_evidence": "evidência bayesiana acumulada",
}


def _sat(x, scale):
    return 1.0 - np.exp(-np.asarray(x, float) / scale)


def compute_signals(suite) -> pd.DataFrame:
    s = suite.snapshot
    f = suite.ctx.customer_features.reindex(s.customer_ids)
    conf = suite.confidences()
    n_launched = max(int(s.launched.sum()), 1)
    df = pd.DataFrame(index=pd.Index(s.customer_ids, name="customer_id"))
    df["n_orders"] = f["n_orders"].to_numpy()
    df["n_distinct_products"] = f["n_distinct_products"].to_numpy()
    df["n_impressions"] = f["n_impressions"].to_numpy()
    df["n_engagements"] = f["n_engagements"].to_numpy()
    df["density"] = df["n_distinct_products"] / n_launched
    df["diversity"] = f["category_entropy"].to_numpy()
    df["neighbor_strength"] = conf.get("collaborative", np.zeros(len(df)))
    df["recency_days"] = f["recency_days"].fillna(730).to_numpy()
    df["drift"] = f["drift_js"].to_numpy()
    df["consistency"] = f["consistency"].to_numpy()
    df["bayes_evidence"] = conf.get("bayesian", np.zeros(len(df)))
    df["has_history"] = f["has_history"].to_numpy()
    df["cb_has_profile"] = (np.linalg.norm(suite.experts["content_based"].profiles, axis=1) > 0).astype(int) \
        if "content_based" in suite.experts else 0
    # transformações usadas pelo gating treinável
    df["log_orders"] = np.log1p(df["n_orders"])
    df["log_distinct"] = np.log1p(df["n_distinct_products"])
    df["log_impressions"] = np.log1p(df["n_impressions"])
    df["log_recency"] = np.log1p(df["recency_days"])
    return df


def rule_reliabilities(signals: pd.DataFrame, experts: list[str] = EXPERTS) -> pd.DataFrame:
    """Confiabilidade a priori r_e(c) de cada especialista (0 = indisponível)."""
    g = signals
    n_dist, n_ord, n_imp = g["n_distinct_products"].to_numpy(float), g["n_orders"].to_numpy(float), g["n_impressions"].to_numpy(float)
    drift_flag = (g["drift"].to_numpy() > CB.drift_threshold).astype(float)
    r = pd.DataFrame(index=g.index)
    # Content-Based: histórico individual consistente aumenta a confiança; perfil declarado = confiança baixa
    cb = np.where(n_dist > 0, 0.35 + 0.45 * _sat(n_dist, 15) + 0.2 * g["consistency"].to_numpy(),
                  np.where(g["cb_has_profile"].to_numpy() > 0, 0.15, 0.0))
    r["content_based"] = cb
    # Collaborative: exige histórico mínimo e vizinhos realmente semelhantes
    nb = np.clip(g["neighbor_strength"].to_numpy() / 0.4, 0, 1)
    r["collaborative"] = np.where(n_dist >= 3, _sat(n_dist, 20) * nb, 0.0)
    # Supervisionado: generaliza entre clientes; ganha peso com interações e em clientes com mudança recente
    r["supervised"] = 0.4 + 0.3 * _sat(n_ord, 10) + 0.2 * _sat(n_imp, 60) + 0.15 * drift_flag
    # Bayesiano: proporcional à evidência acumulada nas interações com recomendações
    r["bayesian"] = 0.05 + 0.6 * g["bayes_evidence"].to_numpy()
    # Popularidade: fallback forte no cold start, residual quando há histórico
    r["popularity"] = 0.05 + 0.75 * np.exp(-n_dist / 5.0)
    return r[[e for e in experts if e in r.columns]].clip(lower=0.0)


def normalize_weights(R: np.ndarray) -> np.ndarray:
    s = R.sum(axis=1, keepdims=True)
    W = np.divide(R, s, out=np.full_like(R, 1.0 / R.shape[1]), where=s > 0)
    return W


def rule_justification(sig: pd.Series, weights: dict[str, float]) -> list[str]:
    """Frases explicativas a partir dos sinais que acionaram as regras."""
    out = []
    n_ord, n_dist, n_imp = int(sig["n_orders"]), int(sig["n_distinct_products"]), int(sig["n_impressions"])
    if n_ord == 0:
        out.append("Cliente sem histórico de compras (cold start): o sistema recorre à popularidade, ao perfil "
                   "declarado no cadastro e ao modelo supervisionado, que generaliza a partir de outros clientes.")
    elif n_ord < 5:
        out.append(f"Histórico ainda curto ({n_ord} pedidos, {n_dist} produtos): a filtragem colaborativa recebe pouca "
                   "confiança, pois a similaridade com outros clientes é instável com poucos dados.")
    else:
        out.append(f"Histórico de {n_ord} pedidos e {n_dist} produtos distintos.")
    if n_dist > 0 and sig["consistency"] > 0.85:
        out.append(f"Padrão de compra consistente ao longo do tempo (consistência {sig['consistency']:.2f}): reforça a afinidade de conteúdo.")
    if sig["neighbor_strength"] >= 0.3:
        out.append(f"Há lojistas com comportamento muito parecido (similaridade média {sig['neighbor_strength']:.2f}): "
                   "o sinal colaborativo é confiável.")
    elif n_dist >= 3:
        out.append(f"Vizinhos pouco semelhantes (similaridade média {sig['neighbor_strength']:.2f}): sinal colaborativo moderado.")
    if sig["drift"] > CB.drift_threshold:
        out.append(f"Mudança recente de preferência detectada (JS = {sig['drift']:.2f}): o perfil de conteúdo passa a "
                   "pesar mais as compras recentes e o modelo supervisionado ganha importância.")
    if n_imp >= 30:
        out.append(f"{n_imp} recomendações já exibidas: as estimativas bayesianas de resposta por categoria têm evidência própria.")
    elif n_imp > 0:
        out.append(f"Poucas interações com recomendações ({n_imp}): o modelo bayesiano depende muito do prior do segmento.")
    low = [EXPERT_LABELS[e] for e, w in weights.items() if w < 0.05]
    if low:
        out.append("Pouco utilizados ou descartados: " + ", ".join(low) + ".")
    return out
