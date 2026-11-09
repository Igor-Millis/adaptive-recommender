"""Interpretação dos clusters: rótulos e descrições derivados dos centróides.

Os rótulos NÃO são atribuídos manualmente: cada cluster recebe "etiquetas"
candidatas quando seu centróide se afasta da média global (z-score) num
atributo interpretável; a etiqueta mais forte vira o nome do segmento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CATEGORIES
from src.preprocessing.feature_engineering import SHARE_COLS


def describe_clusters(X: pd.DataFrame, labels: np.ndarray, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df["cluster"] = labels
    glob_mean = df.mean(numeric_only=True)
    glob_std = df.std(numeric_only=True).replace(0, 1)
    rows, names, used = [], {}, set()
    for k in sorted(np.unique(labels)):
        g = df[df["cluster"] == k]
        m = g.mean(numeric_only=True)
        z = (m - glob_mean) / glob_std
        tags: list[tuple[float, str]] = []
        shares = m[SHARE_COLS].to_numpy()
        top = int(np.argmax(shares))
        top_cat = CATEGORIES[top]
        lift = shares[top] / max(glob_mean[SHARE_COLS].to_numpy()[top], 1e-9)
        if shares[top] > 0.28 and lift > 1.6:
            tags.append((lift, f"Especialistas em {top_cat.lower()}"))
        if z["avg_ticket"] > 0.7:
            tags.append((1 + z["avg_ticket"], "Lojistas de alto ticket"))
        if z["price_index"] < -0.7:
            tags.append((1 - z["price_index"], "Lojistas sensíveis a preço"))
        if z["seasonality_index"] > 0.7 and m["seasonality_index"] > 0.06:
            tags.append((1 + z["seasonality_index"], "Clientes sazonais"))
        if z["category_entropy"] > 0.6:
            tags.append((1 + z["category_entropy"], "Compradores diversificados"))
        if z["frequency_per_month"] > 0.8:
            tags.append((1 + z["frequency_per_month"], "Compradores frequentes"))
        if z["price_index"] > 0.7:
            tags.append((1 + z["price_index"], "Foco em linha premium"))
        tags.sort(reverse=True)
        name = next((t for _, t in tags if t not in used), None)
        if name is None:
            name = f"Perfil misto ({top_cat.lower()})"
            if name in used:
                name = f"{name} #{k}"
        used.add(name)
        names[int(k)] = name
        second = [t for _, t in tags if t != name][:2]
        desc = (
            f"{len(g)} clientes; ticket médio R$ {m['avg_ticket']:,.0f}; {m['frequency_per_month']:.1f} pedidos/mês; "
            f"categoria principal {top_cat} ({shares[top]:.0%} das unidades); índice de preço {m['price_index']:+.2f}; "
            f"diversidade {m['category_entropy']:.2f}"
        )
        if second:
            desc += f". Também: {', '.join(s.lower() for s in second)}."
        rows.append({"cluster_id": int(k), "label": name, "description": desc, "size": int(len(g)),
                     "centroid": {c: float(m[c]) for c in ["avg_ticket", "frequency_per_month", "price_index",
                                                           "category_entropy", "seasonality_index", "recency_days"] + SHARE_COLS},
                     "z_scores": {c: float(z[c]) for c in ["avg_ticket", "frequency_per_month", "price_index",
                                                           "category_entropy", "seasonality_index"]}})
    return pd.DataFrame(rows), names
