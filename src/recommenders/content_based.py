"""Filtragem baseada em conteúdo (Content-Based).

Representação dos produtos: concatenação de blocos, cada um normalizado em L2 e
multiplicado por √w_b (assim o produto interno total é Σ_b w_b·cos_b):
  * one-hot: categoria, subcategoria, público, faixa de preço, material, cor, estação;
  * TF-IDF sobre nome + descrição (termos como "academia", "corporativo", "praia").

Perfil do cliente (Rocchio ponderado):
    u_c = Σ_compras log(1+q)·0.5^{idade/h_c}·x_p + Σ_interações w_tipo·0.5^{idade/h_c}·x_p
com w_rejeição < 0. A meia-vida h_c é reduzida quando o cliente mostra mudança
de preferência (JS > limiar). Cold start: perfil a partir das categorias e da
faixa de preço declaradas no cadastro.

score(c, p) = cos(u_c, x_p). A contribuição de cada atributo é o termo
correspondente do produto interno normalizado, o que torna a explicação exata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import CB
from src.preprocessing.transformations import exp_decay, l2_normalize_rows
from src.recommenders.base import BaseRecommender, ExpertContext

BLOCK_COLUMNS = {
    "categoria": "category",
    "subcategoria": "subcategory",
    "publico": "target_audience",
    "faixa_preco": "price_band",
    "material": "material",
    "cor": "color",
    "estacao": "season",
}
STOPWORDS = ["de", "da", "do", "e", "a", "o", "em", "com", "para", "estilo", "dia"]


class ProductVectorizer:
    def __init__(self, block_weights: dict | None = None):
        self.block_weights = block_weights or CB.block_weights

    def fit_transform(self, products: pd.DataFrame) -> np.ndarray:
        blocks, names, block_of = [], [], []
        for block, col in BLOCK_COLUMNS.items():
            d = pd.get_dummies(products[col].astype(str), dtype=float)
            M = l2_normalize_rows(d.to_numpy()) * np.sqrt(self.block_weights[block])
            blocks.append(M)
            names += [f"{block}={v}" for v in d.columns]
            block_of += [block] * d.shape[1]
        text = (products["name"].astype(str) + " " + products["description"].astype(str)).str.lower()
        self.tfidf = TfidfVectorizer(strip_accents="unicode", min_df=2, stop_words=STOPWORDS, token_pattern=r"(?u)\b[a-zà-ú]{3,}\b")
        T = self.tfidf.fit_transform(text).toarray()
        blocks.append(l2_normalize_rows(T) * np.sqrt(self.block_weights["texto"]))
        names += [f"texto:{t}" for t in self.tfidf.get_feature_names_out()]
        block_of += ["texto"] * T.shape[1]
        self.feature_names = np.array(names)
        self.block_of = np.array(block_of)
        X = np.hstack(blocks)
        return l2_normalize_rows(X)


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"
    label = "Content-Based"

    def __init__(self, config=CB):
        super().__init__()
        self.cfg = config

    def fit(self, snapshot, ctx: ExpertContext):
        self.snapshot, self.ctx = snapshot, ctx
        self.vectorizer = ProductVectorizer(self.cfg.block_weights)
        self.X = self.vectorizer.fit_transform(snapshot.products)
        self.item_similarity = self.X @ self.X.T
        self.profiles = np.zeros((snapshot.n_customers, self.X.shape[1]))
        self.profile_source = np.array(["sem dados"] * snapshot.n_customers, dtype=object)
        self.half_lives = np.full(snapshot.n_customers, self.cfg.half_life_days)
        for i, cid in enumerate(snapshot.customer_ids):
            self._build_profile(i, int(cid))
        self._recompute_scores()
        return self

    # -------------------------------------------------------------- perfis
    def _build_profile(self, i: int, cid: int) -> None:
        s, ctx = self.snapshot, self.ctx
        drift = float(ctx.customer_features.loc[cid, "drift_js"]) if cid in ctx.customer_features.index else 0.0
        h = self.cfg.drift_half_life_days if drift > self.cfg.drift_threshold else self.cfg.half_life_days
        self.half_lives[i] = h
        prof = np.zeros(self.X.shape[1])
        it = s.items[s.items["customer_id"] == cid]
        if len(it):
            age = (s.as_of - it["order_date"]).dt.days.to_numpy()
            w = np.log1p(it["quantity"].to_numpy(float)) * exp_decay(age, h)
            pj = it["product_id"].map(s.pidx).to_numpy()
            prof += w @ self.X[pj]
        out = ctx.outcomes[ctx.outcomes["customer_id"] == cid] if len(ctx.outcomes) else ctx.outcomes
        if len(out):
            wt = out["outcome"].map(self.cfg.interaction_weights).fillna(0.0).to_numpy()
            age = (s.as_of - out["timestamp"]).dt.days.to_numpy()
            pj = out["product_id"].map(s.pidx).to_numpy()
            prof += (wt * exp_decay(age, h)) @ self.X[pj]
        if len(it):
            self.profile_source[i] = "histórico de compras e interações"
        elif np.linalg.norm(prof) > 0:
            self.profile_source[i] = "interações com recomendações"
        if np.linalg.norm(prof) <= 1e-12 or not len(it):
            declared = self._declared_profile(cid)
            if np.linalg.norm(declared) > 0:
                prof = prof + declared if np.linalg.norm(prof) > 0 else declared
                self.profile_source[i] = "cadastro (categorias e faixa de preço declaradas)" if not len(it) else self.profile_source[i]
        self.profiles[i] = prof

    def _declared_profile(self, cid: int) -> np.ndarray:
        cust = self.snapshot.customers.loc[cid]
        cats = [c for c in str(cust.get("declared_categories", "") or "").split(";") if c]
        band = cust.get("declared_price_band")
        prods = self.snapshot.products
        mask = prods["category"].isin(cats).to_numpy() & self.snapshot.launched
        if band and (mask & (prods["price_band"] == band).to_numpy()).any():
            mask = mask & (prods["price_band"] == band).to_numpy()
        if not mask.any():
            return np.zeros(self.X.shape[1])
        return self.X[mask].mean(axis=0)

    def _recompute_scores(self) -> None:
        P = l2_normalize_rows(self.profiles)
        self._scores = P @ self.X.T

    def update_customer(self, snapshot, ctx: ExpertContext, customer_id: int) -> None:
        """Atualização incremental: recalcula apenas o perfil de um cliente."""
        self.snapshot, self.ctx = snapshot, ctx
        i = snapshot.cidx[int(customer_id)]
        self._build_profile(i, int(customer_id))
        p = self.profiles[i]
        n = np.linalg.norm(p)
        self._scores[i] = (p / n) @ self.X.T if n > 0 else 0.0

    def confidence(self):
        f = self.ctx.customer_features.reindex(self.snapshot.customer_ids)
        n = f["n_distinct_products"].fillna(0).to_numpy(float)
        conf = 1 - np.exp(-n / 15.0)
        has_prof = np.linalg.norm(self.profiles, axis=1) > 0
        return np.where(n > 0, conf, np.where(has_prof, 0.15, 0.0))

    # -------------------------------------------------------------- explicação
    def feature_contributions(self, customer_id: int, product_id: int, top: int = 6) -> pd.DataFrame:
        i, j = self._ci(customer_id), self._pj(product_id)
        p = self.profiles[i]
        n = np.linalg.norm(p)
        if n == 0:
            return pd.DataFrame(columns=["feature", "block", "contribution", "share"])
        contrib = (p / n) * self.X[j]
        total = contrib.sum()
        df = pd.DataFrame({"feature": self.vectorizer.feature_names, "block": self.vectorizer.block_of, "contribution": contrib})
        df = df[df["contribution"] > 0].sort_values("contribution", ascending=False)
        df["share"] = df["contribution"] / total if total > 0 else 0.0
        return df.head(top).reset_index(drop=True)

    def explain(self, customer_id, product_id):
        i, j = self._ci(customer_id), self._pj(product_id)
        contrib = self.feature_contributions(customer_id, product_id)
        parts = []
        for r in contrib.head(3).itertuples():
            name = r.feature.split("=", 1)[-1] if "=" in r.feature else f"termo '{r.feature.split(':', 1)[-1]}'"
            parts.append(f"{name} ({r.share:.0%})")
        s = self.snapshot
        bought = s.items[s.items["customer_id"] == int(customer_id)]["product_id"].unique()
        similar_txt, similar_id = "", None
        if len(bought):
            bj = np.array([s.pidx[int(b)] for b in bought])
            sims = self.item_similarity[j, bj]
            k = int(np.argmax(sims))
            similar_id = int(bought[k])
            similar_txt = f" Mais parecido com '{s.products.loc[similar_id, 'name']}' (similaridade {sims[k]:.2f})."
        summary = f"Afinidade de conteúdo com o perfil ({self.profile_source[i]}): " + ", ".join(parts) + "." + similar_txt
        return {"summary": summary, "details": {"score": float(self._scores[i, j]), "contribuicoes": contrib.to_dict("records"),
                                                "produto_similar": similar_id, "meia_vida_dias": float(self.half_lives[i])}}
