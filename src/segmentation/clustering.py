"""Segmentação de clientes: K-Means (principal) e Hierárquico Ward (comparação).

Escolha de k combina três critérios: Silhouette (separação), estabilidade por
bootstrap (ARI médio entre re-ajustes) e Inertia (cotovelo, apenas diagnóstico).
Critério de seleção: maximizar ``silhouette × estabilidade`` entre k ∈ [3, 10] —
k = 2 tende a vencer em Silhouette mas gera segmentos pouco acionáveis.

Clientes sem histórico suficiente (< 3 pedidos) não são clusterizados: recebem
o rótulo -1 ("Sem histórico suficiente") em vez de uma atribuição arbitrária.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config.settings import CATEGORIES, RANDOM_SEED
from src.preprocessing.feature_engineering import SHARE_COLS
from src.segmentation.cluster_analysis import describe_clusters

MIN_ORDERS_TO_CLUSTER = 3
BASE_COLS = ["log_avg_ticket", "log_frequency", "category_entropy", "log_avg_unit_price", "price_index",
             "log_recency", "seasonality_index"]


def clustering_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    f = features[features["n_orders"] >= MIN_ORDERS_TO_CLUSTER].copy()
    X = pd.DataFrame(index=f.index)
    X["log_avg_ticket"] = np.log1p(f["avg_ticket"])
    X["log_frequency"] = np.log1p(f["frequency_per_month"])
    X["category_entropy"] = f["category_entropy"]
    X["log_avg_unit_price"] = np.log1p(f["avg_unit_price"])
    X["price_index"] = f["price_index"]
    X["log_recency"] = np.log1p(f["recency_days"].fillna(f["recency_days"].max()))
    X["seasonality_index"] = f["seasonality_index"]
    for c in SHARE_COLS:
        X[c] = f[c]
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.to_numpy())
    # bloco de participações (9 colunas composicionais) pesa como ~3 atributos
    share_idx = [X.columns.get_loc(c) for c in SHARE_COLS]
    Z[:, share_idx] *= np.sqrt(3 / len(SHARE_COLS))
    return X, Z, scaler


def _stability(Z: np.ndarray, k: int, ref: np.ndarray, n_boot: int = 30, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), size=len(Z), replace=True)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + b).fit(Z[idx])
        aris.append(adjusted_rand_score(ref, km.predict(Z)))
    return float(np.mean(aris))


@dataclass
class SegmentationResult:
    as_of: pd.Timestamp
    k: int
    labels: pd.Series  # customer_id -> cluster id (-1 = sem histórico)
    label_names: dict
    profiles: pd.DataFrame
    diagnostics: pd.DataFrame  # por k: inertia, silhouette, stability
    hierarchical: dict
    centroids: np.ndarray
    feature_names: list = field(default_factory=list)
    silhouette: float = 0.0

    def assignments(self) -> pd.DataFrame:
        return pd.DataFrame({"customer_id": self.labels.index, "cluster_id": self.labels.to_numpy(),
                             "cluster_label": [self.label_names.get(int(k), "Sem histórico suficiente") for k in self.labels]})


def fit_segmentation(features: pd.DataFrame, as_of: pd.Timestamp, k_range=range(2, 11), fixed_k: int | None = None,
                     previous_centroids: np.ndarray | None = None) -> SegmentationResult:
    X, Z, _ = clustering_matrix(features)
    n = len(Z)
    rows = []
    fits = {}
    for k in k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        stab = _stability(Z, k, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "stability": stab, "score": sil * stab})
        fits[k] = km
    diag = pd.DataFrame(rows)
    if fixed_k is not None and fixed_k in fits:
        k_best = fixed_k
    else:
        cand = diag[diag["k"] >= 3]
        k_best = int(cand.loc[cand["score"].idxmax(), "k"]) if len(cand) else int(diag.loc[diag["score"].idxmax(), "k"])
    km = fits[k_best]
    labels = km.labels_.copy()
    centroids = km.cluster_centers_
    if previous_centroids is not None and previous_centroids.shape == centroids.shape:
        labels, centroids = align_labels(labels, centroids, previous_centroids)

    # Ward com o mesmo k, para comparação
    ward = AgglomerativeClustering(n_clusters=k_best, linkage="ward").fit(Z)
    hier = {
        "silhouette_ward": float(silhouette_score(Z, ward.labels_)),
        "silhouette_kmeans": float(silhouette_score(Z, labels)),
        "ari_kmeans_vs_ward": float(adjusted_rand_score(labels, ward.labels_)),
        "ward_labels": dict(zip(map(int, X.index), map(int, ward.labels_))),
    }
    full = pd.Series(-1, index=features.index, dtype=int)
    full.loc[X.index] = labels
    profiles, names = describe_clusters(X, labels, features.loc[X.index])
    return SegmentationResult(as_of=as_of, k=k_best, labels=full, label_names=names, profiles=profiles,
                              diagnostics=diag, hierarchical=hier, centroids=centroids,
                              feature_names=list(X.columns), silhouette=hier["silhouette_kmeans"])


def align_labels(labels: np.ndarray, centroids: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reordena ids de cluster para casar com a execução anterior (Hungarian) — ids estáveis entre retreinos."""
    cost = ((centroids[:, None, :] - previous[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    mapping = dict(zip(r, c))
    new_labels = np.array([mapping[x] for x in labels])
    new_centroids = np.zeros_like(centroids)
    for old, new in mapping.items():
        new_centroids[new] = centroids[old]
    return new_labels, new_centroids


def cluster_category_shares(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Participação média por categoria em cada cluster (usado como contexto por outros modelos)."""
    df = features[SHARE_COLS].copy()
    df["cluster"] = labels
    out = df[df["cluster"] >= 0].groupby("cluster")[SHARE_COLS].mean()
    out.columns = CATEGORIES
    return out
