"""Adaptador do dataset público *Online Retail II* (UCI Machine Learning Repository, CC BY 4.0).

Transações reais (dez/2009 – dez/2011) de um atacadista online do Reino Unido cujos clientes são,
em boa parte, revendedores — um cenário B2B próximo ao do projeto. O adaptador converte os dados
para o mesmo esquema do pipeline sintético (``Dataset``), de modo que os MESMOS modelos, o mesmo
motor adaptativo e o mesmo protocolo de avaliação sejam executados sem alterações.

O que o dataset NÃO tem (e como isso é tratado):
  * categorias/atributos de produto → derivados da descrição por regras de palavra-chave
    (categoria, subcategoria = palavra-chave, cor, material, público, estação);
  * logs de recomendação/interação → tabela de interações vazia: o especialista bayesiano fica
    sem evidência própria (opera só com o prior) — limitação declarada no relatório;
  * custo/margem e estoque → margem fixa e estoque disponível (não afetam a avaliação de relevância);
  * data de cadastro e dados declarados → desconhecidos (cadastro = início dos dados).

Seleção (evita vazamento): o catálogo é formado pelos produtos com mais compradores distintos
ANTES do início da validação; os clientes são uma amostra aleatória dos que compraram esses produtos
(o que naturalmente inclui clientes cuja 1ª compra ocorre no teste — cold start real).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CATEGORIES, DATA_START, RANDOM_SEED, VAL_START
from src.preprocessing.snapshot import Dataset

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Natal e datas sazonais", ["CHRISTMAS", "XMAS", "ADVENT", "EASTER", "HALLOWEEN", "SNOWFLAKE", "REINDEER", "SANTA", "BAUBLE"]),
    ("Bolsas e sacolas", ["BAG", "BAGS", "SHOPPER", "PURSE", "TOTE"]),
    ("Cozinha e confeitaria", ["CAKE", "CASES", "BAKING", "MUG", "CUP", "CUPS", "TEA", "TEACUP", "PLATE", "PLATES", "BOWL", "JAR",
                               "KITCHEN", "CUTLERY", "PANTRY", "LUNCH", "SPOON", "TRAY", "JUG", "APRON", "CAKESTAND", "BISCUIT",
                               "EGG", "COFFEE", "SUGAR", "RECIPE", "JAM", "BREAD", "CAKES"]),
    ("Velas e iluminação", ["CANDLE", "CANDLES", "T-LIGHT", "TLIGHT", "LIGHT", "LIGHTS", "LANTERN", "LAMP", "NIGHTLIGHT"]),
    ("Festas e presentes", ["PARTY", "BUNTING", "BALLOON", "BALLOONS", "NAPKINS", "RIBBON", "RIBBONS", "WRAP", "GIFT", "BIRTHDAY",
                            "CONFETTI", "STRAWS", "BANNER"]),
    ("Papelaria e cartões", ["CARD", "CARDS", "NOTEBOOK", "PEN", "PENS", "PENCIL", "PENCILS", "PAPER", "STICKER", "STICKERS",
                             "ENVELOPE", "CALENDAR", "BOOK", "CHALK", "ERASER", "RULER", "TISSUES", "NOTEPAD"]),
    ("Infantil e brinquedos", ["TOY", "TOYS", "CHILDREN", "CHILDRENS", "DOLL", "DOLLY", "GAME", "SPACEBOY", "PLAYHOUSE", "JIGSAW",
                               "PUZZLE", "TEDDY", "BABY", "KIDS", "SKIPPING", "SOLDIER", "DINOSAUR"]),
    ("Jardim e organização", ["GARDEN", "STORAGE", "BOX", "BOXES", "BASKET", "HOOK", "HOOKS", "PEG", "PEGS", "WATERING", "TIN",
                              "TINS", "DRAWER", "HANGER", "WICKER", "SHELF", "PLANT", "POT"]),
    ("Decoração da casa", ["SIGN", "FRAME", "CLOCK", "HEART", "HEARTS", "DECORATION", "MIRROR", "CUSHION", "DOORMAT", "HANGING",
                           "ORNAMENT", "WREATH", "VASE", "PLAQUE", "BIRD", "GARLAND", "DOORSTOP", "WALL", "BELL", "CHIME"]),
]
COLORS = ["RED", "PINK", "BLUE", "WHITE", "GREEN", "BLACK", "IVORY", "CREAM", "SILVER", "GOLD", "YELLOW", "PURPLE", "ORANGE"]
MATERIALS = {"METAL": "Metal", "WOODEN": "Madeira", "WOOD": "Madeira", "GLASS": "Vidro", "CERAMIC": "Cerâmica",
             "PAPER": "Papel", "FELT": "Feltro", "FELTCRAFT": "Feltro", "WICKER": "Vime", "ENAMEL": "Esmalte",
             "PORCELAIN": "Porcelana", "ZINC": "Zinco", "CANVAS": "Lona", "COTTON": "Algodão"}


def categorize(description: str) -> tuple[str, str]:
    words = re.findall(r"[A-Z\-]+", str(description).upper())
    for cat, keys in CATEGORY_RULES:
        for w in words:
            if w in keys:
                return cat, w.title()
    return "Outros", (words[-1].title() if words else "Outros")


def _attribute(description: str, vocab) -> str | None:
    words = re.findall(r"[A-Z]+", str(description).upper())
    for w in words:
        if w in vocab:
            return w
    return None


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelamentos, devoluções, clientes anônimos e códigos que não são produtos (POST, M, BANK CHARGES...)."""
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    mask = (~df["Invoice"].str.startswith("C") & df["Customer ID"].notna() & (df["Quantity"] > 0) & (df["Price"] > 0)
            & df["StockCode"].str.match(r"^\d{5}"))
    df = df.loc[mask].copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@dataclass
class RealDataStats:
    raw_rows: int
    clean_rows: int
    selected_rows: int
    n_customers: int
    n_products: int
    n_orders: int
    category_counts: dict = field(default_factory=dict)
    first_purchase_in_test: int = 0
    notes: list = field(default_factory=list)


def build_dataset(raw: pd.DataFrame, n_customers: int = 500, n_products: int = 400,
                  seed: int = RANDOM_SEED) -> tuple[Dataset, RealDataStats]:
    unknown = set(c for c, _ in CATEGORY_RULES) | {"Outros"}
    if set(CATEGORIES) != unknown:
        raise RuntimeError("Use o perfil 'online_retail' (ADAPTIVE_RECOMMENDER_PROFILE=online_retail) para estes dados.")
    df = clean_transactions(raw)
    before_val = df[df["InvoiceDate"] < VAL_START]
    buyers = before_val.groupby("StockCode")["Customer ID"].nunique().sort_values(ascending=False)
    catalog = list(buyers.head(n_products).index)
    sel = df[df["StockCode"].isin(catalog)]
    rng = np.random.default_rng(seed)
    pool = np.sort(sel["Customer ID"].unique())
    chosen = set(rng.choice(pool, size=min(n_customers, len(pool)), replace=False).tolist())
    sel = sel[sel["Customer ID"].isin(chosen)]

    # ---------------- produtos
    g = df[df["StockCode"].isin(catalog)].groupby("StockCode")
    desc = g["Description"].agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else "")
    price = g["Price"].median()
    first_sale = g["InvoiceDate"].min()
    code_to_pid = {c: i + 1 for i, c in enumerate(catalog)}
    rows = []
    for code in catalog:
        d = str(desc[code])
        cat, sub = categorize(d)
        color = _attribute(d, COLORS)
        mat = _attribute(d, MATERIALS)
        up = d.upper()
        season = "Inverno" if (cat == "Natal e datas sazonais" or "HOT WATER" in up) else \
            "Verão" if any(w in up for w in ("PICNIC", "BEACH", "SUMMER", "PARASOL")) else "Atemporal"
        audience = "Infantil" if cat == "Infantil e brinquedos" or any(w in up for w in ("CHILDRENS", "SPACEBOY", "DOLLY")) else "Geral"
        rows.append({"product_id": code_to_pid[code], "stock_code": code, "name": d.title(), "category": cat, "subcategory": sub,
                     "target_audience": audience, "price": float(price[code]), "color": (color or "Sem cor").title(),
                     "size_range": "Único", "material": MATERIALS.get(mat, "N/D") if mat else "N/D", "season": season,
                     "collection": "N/D", "launch_date": first_sale[code].normalize(), "stock": 1, "is_active": 1,
                     "description": d.lower()})
    products = pd.DataFrame(rows)
    products["cost"] = (products["price"] * 0.6).round(2)  # custo desconhecido no dataset (margem não é avaliada)
    products["margin"] = 0.4
    products["price_band"] = (products.groupby("category")["price"]
                              .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Econômica", "Intermediária", "Premium"])
                                         if len(s) >= 3 else pd.Series("Intermediária", index=s.index)).astype(str))

    # ---------------- clientes (dados cadastrais indisponíveis)
    cust = sel.groupby("Customer ID").agg(country=("Country", "first"), first=("InvoiceDate", "min"))
    customers = pd.DataFrame({"customer_id": cust.index.astype(int), "name": [f"Cliente {c}" for c in cust.index],
                              "city": cust["country"].to_numpy(), "state": "--", "registration_date": DATA_START,
                              "store_size": "N/D", "business_segment": "Varejo/atacado de presentes e casa",
                              "revenue_band": "N/D", "declared_categories": None, "declared_price_band": None,
                              "declared_purchase_frequency": None})

    # ---------------- pedidos e itens
    inv = sel.groupby("Invoice").agg(customer_id=("Customer ID", "first"), order_date=("InvoiceDate", "min")).reset_index()
    inv["order_id"] = np.arange(1, len(inv) + 1)
    inv["channel"] = "online"
    sel = sel.assign(value=sel["Quantity"] * sel["Price"])
    it = sel.groupby(["Invoice", "StockCode"]).agg(quantity=("Quantity", "sum"), value=("value", "sum")).reset_index()
    it = it.merge(inv[["Invoice", "order_id"]], on="Invoice")
    it["product_id"] = it["StockCode"].map(code_to_pid)
    it["unit_price"] = (it["value"] / it["quantity"]).round(4)
    it["discount"] = 0.0
    it["total_value"] = it["value"].round(2)
    it["item_id"] = np.arange(1, len(it) + 1)
    orders = inv[["order_id", "customer_id", "order_date", "channel"]]
    items = it[["item_id", "order_id", "product_id", "quantity", "unit_price", "discount", "total_value"]]
    interactions = pd.DataFrame(columns=["interaction_id", "session_id", "customer_id", "product_id", "model_name",
                                         "interaction_type", "timestamp", "context", "recommendation_id"])
    from config.settings import TEST_START

    stats = RealDataStats(
        raw_rows=int(len(raw)), clean_rows=int(len(df)), selected_rows=int(len(sel)), n_customers=int(len(customers)),
        n_products=int(len(products)), n_orders=int(len(orders)),
        category_counts=products["category"].value_counts().to_dict(),
        first_purchase_in_test=int((cust["first"] >= TEST_START).sum()),
        notes=["Sem logs de interação: especialista bayesiano opera apenas com o prior.",
               "Categorias derivadas por palavras-chave da descrição.",
               "Custo/margem e estoque desconhecidos (não afetam métricas de relevância)."],
    )
    ds = Dataset.from_frames(customers, products.drop(columns=["stock_code"]), orders, items, interactions)
    return ds, stats


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
