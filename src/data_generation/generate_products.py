"""Gerador do catálogo sintético de produtos.

Cada subcategoria tem preço-base de atacado, materiais plausíveis, públicos-alvo,
estação e um "estilo" (esportivo, social, casual, praia). O estilo não é exposto
como coluna — ele aparece apenas no texto descritivo, que é o que o modelo
Content-Based vê via TF-IDF. Assim o gerador cria estrutura estatística real
sem entregar a resposta pronta aos modelos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import DATA_END, DATA_START

# subcategoria: (categoria, preço base, materiais, públicos, estação, estilo, descritores)
SUBCATEGORIES: dict[str, tuple] = {
    # Camisetas
    "Camiseta básica": ("Camisetas", 24, ["Algodão", "Algodão orgânico"], ["Masculino", "Feminino", "Unissex"], "Atemporal", "casual", "básica gola careca conforto dia a dia"),
    "Camiseta estampada": ("Camisetas", 32, ["Algodão", "Viscose"], ["Masculino", "Feminino", "Unissex", "Infantil"], "Verão", "casual", "estampa gráfica jovem descolada"),
    "Camisa polo": ("Camisetas", 45, ["Piquet", "Algodão"], ["Masculino"], "Atemporal", "casual", "polo gola botões casual elegante"),
    "Regata": ("Camisetas", 20, ["Algodão", "Viscose"], ["Masculino", "Feminino"], "Verão", "praia", "regata leve calor praia verão"),
    "Camiseta oversized": ("Camisetas", 38, ["Algodão"], ["Unissex"], "Atemporal", "casual", "oversized ampla streetwear tendência"),
    # Camisas
    "Camisa social": ("Camisas", 72, ["Algodão egípcio", "Tricoline"], ["Masculino", "Feminino"], "Atemporal", "social", "social escritório corporativo elegante colarinho"),
    "Camisa casual": ("Camisas", 58, ["Algodão", "Viscose"], ["Masculino", "Feminino"], "Atemporal", "casual", "casual fim de semana botões"),
    "Camisa de linho": ("Camisas", 95, ["Linho"], ["Masculino", "Feminino"], "Verão", "social", "linho leve verão elegante respirável"),
    "Camisa xadrez flanela": ("Camisas", 64, ["Flanela"], ["Masculino", "Unissex"], "Inverno", "casual", "flanela xadrez inverno quente"),
    # Calças
    "Calça jeans": ("Calças", 85, ["Jeans", "Jeans com elastano"], ["Masculino", "Feminino", "Infantil"], "Atemporal", "casual", "jeans cinco bolsos resistente"),
    "Calça alfaiataria": ("Calças", 110, ["Alfaiataria", "Viscose"], ["Masculino", "Feminino"], "Atemporal", "social", "alfaiataria social corporativo elegante vinco"),
    "Calça chino": ("Calças", 78, ["Sarja", "Algodão"], ["Masculino"], "Atemporal", "casual", "chino sarja casual elegante"),
    "Calça jogger": ("Calças", 62, ["Moletom", "Sarja"], ["Masculino", "Feminino", "Unissex"], "Inverno", "esportivo", "jogger punho conforto esportivo"),
    # Bermudas
    "Bermuda jeans": ("Bermudas", 55, ["Jeans"], ["Masculino", "Feminino", "Infantil"], "Verão", "casual", "bermuda jeans verão casual"),
    "Bermuda sarja": ("Bermudas", 52, ["Sarja"], ["Masculino"], "Verão", "casual", "bermuda sarja casual verão"),
    "Bermuda moletom": ("Bermudas", 40, ["Moletom"], ["Masculino", "Unissex"], "Atemporal", "esportivo", "moletom conforto esportivo"),
    "Bermuda tactel": ("Bermudas", 35, ["Tactel", "Poliéster"], ["Masculino", "Infantil"], "Verão", "praia", "tactel praia secagem rápida verão"),
    # Vestidos
    "Vestido midi": ("Vestidos", 98, ["Viscose", "Crepe"], ["Feminino"], "Atemporal", "casual", "midi feminino fluido versátil"),
    "Vestido longo": ("Vestidos", 120, ["Viscose", "Seda sintética"], ["Feminino"], "Verão", "praia", "longo fluido verão praia estampado"),
    "Vestido curto": ("Vestidos", 75, ["Algodão", "Viscose"], ["Feminino", "Infantil"], "Verão", "casual", "curto leve verão jovem"),
    "Vestido de festa": ("Vestidos", 165, ["Crepe", "Cetim", "Renda"], ["Feminino"], "Atemporal", "social", "festa noite elegante sofisticado"),
    "Vestido tubinho": ("Vestidos", 105, ["Alfaiataria", "Crepe"], ["Feminino"], "Atemporal", "social", "tubinho corporativo social elegante"),
    # Saias
    "Saia midi": ("Saias", 70, ["Viscose", "Crepe"], ["Feminino"], "Atemporal", "casual", "saia midi fluida feminina"),
    "Saia plissada": ("Saias", 78, ["Poliéster", "Crepe"], ["Feminino"], "Atemporal", "social", "plissada elegante feminina"),
    "Saia jeans": ("Saias", 58, ["Jeans"], ["Feminino", "Infantil"], "Verão", "casual", "saia jeans casual jovem"),
    "Saia lápis": ("Saias", 82, ["Alfaiataria"], ["Feminino"], "Atemporal", "social", "lápis corporativa social escritório"),
    # Jaquetas
    "Jaqueta jeans": ("Jaquetas", 125, ["Jeans"], ["Masculino", "Feminino", "Unissex"], "Inverno", "casual", "jaqueta jeans casual meia estação"),
    "Jaqueta couro sintético": ("Jaquetas", 160, ["Couro sintético"], ["Masculino", "Feminino"], "Inverno", "casual", "couro sintético motociclista inverno"),
    "Corta-vento": ("Jaquetas", 95, ["Poliéster", "Nylon"], ["Masculino", "Feminino", "Unissex"], "Inverno", "esportivo", "corta vento leve esportivo impermeável"),
    "Jaqueta puffer": ("Jaquetas", 175, ["Nylon"], ["Masculino", "Feminino", "Infantil"], "Inverno", "casual", "puffer acolchoada inverno quente"),
    "Blazer": ("Jaquetas", 185, ["Alfaiataria", "Linho"], ["Masculino", "Feminino"], "Atemporal", "social", "blazer alfaiataria social corporativo"),
    # Roupas esportivas
    "Legging": ("Roupas esportivas", 48, ["Poliamida", "Elastano"], ["Feminino"], "Atemporal", "esportivo", "legging academia treino compressão"),
    "Top esportivo": ("Roupas esportivas", 38, ["Poliamida", "Elastano"], ["Feminino"], "Atemporal", "esportivo", "top academia treino sustentação"),
    "Short de corrida": ("Roupas esportivas", 36, ["Poliéster", "Dry-fit"], ["Masculino", "Feminino"], "Verão", "esportivo", "corrida treino leve respirável"),
    "Camiseta dry-fit": ("Roupas esportivas", 42, ["Dry-fit", "Poliéster"], ["Masculino", "Feminino", "Unissex"], "Atemporal", "esportivo", "dry fit treino secagem rápida academia"),
    "Agasalho esportivo": ("Roupas esportivas", 115, ["Poliéster", "Moletom"], ["Masculino", "Feminino", "Infantil"], "Inverno", "esportivo", "agasalho conjunto treino inverno"),
    # Acessórios
    "Boné": ("Acessórios", 28, ["Algodão", "Poliéster"], ["Unissex", "Infantil"], "Atemporal", "esportivo", "boné aba curva casual esportivo"),
    "Cinto": ("Acessórios", 42, ["Couro", "Couro sintético"], ["Masculino", "Feminino"], "Atemporal", "social", "cinto fivela couro social"),
    "Bolsa": ("Acessórios", 95, ["Couro sintético", "Lona"], ["Feminino"], "Atemporal", "casual", "bolsa transversal feminina"),
    "Meias (kit)": ("Acessórios", 18, ["Algodão", "Poliamida"], ["Unissex", "Infantil"], "Atemporal", "esportivo", "kit meias cano médio conforto"),
    "Lenço": ("Acessórios", 30, ["Seda sintética", "Viscose"], ["Feminino"], "Inverno", "social", "lenço estampado elegante"),
    "Óculos de sol": ("Acessórios", 55, ["Acetato", "Policarbonato"], ["Unissex"], "Verão", "praia", "óculos sol proteção uv praia verão"),
}

COLORS = ["Preto", "Branco", "Azul-marinho", "Cinza", "Bege", "Verde-oliva", "Vinho", "Azul-claro", "Rosa", "Terracota", "Mostarda", "Estampado"]
SIZES_BY_AUDIENCE = {
    "Masculino": ["P-GG", "P-XGG", "38-48"],
    "Feminino": ["PP-G", "P-GG", "34-44"],
    "Unissex": ["P-GG", "Único"],
    "Infantil": ["2-8", "4-14"],
}
NAME_SUFFIX = ["Essencial", "Urban", "Prime", "Lisboa", "Atlântico", "Serra", "Aurora", "Horizonte", "Pulse", "Classic", "Nativa", "Vértice"]


def generate_products(n_products: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subcats = list(SUBCATEGORIES.keys())
    # Garante cobertura: pelo menos 3 produtos por subcategoria, restante proporcional
    base = [s for s in subcats for _ in range(3)]
    extra = list(rng.choice(subcats, size=n_products - len(base), replace=True))
    chosen = base + extra
    rng.shuffle(chosen)

    rows = []
    span_days = (DATA_END - DATA_START).days
    for pid, sub in enumerate(chosen, start=1):
        cat, base_price, materials, audiences, season, style, desc = SUBCATEGORIES[sub]
        material = rng.choice(materials)
        audience = rng.choice(audiences)
        color = rng.choice(COLORS)
        size = rng.choice(SIZES_BY_AUDIENCE[audience])
        # Preço com variação log-normal (posicionamento de produto)
        price_mult = float(np.exp(rng.normal(0.0, 0.28)))
        if audience == "Infantil":
            price_mult *= 0.7
        price = round(base_price * price_mult, 2)
        cost = round(price * rng.uniform(0.42, 0.66), 2)

        # 60% do catálogo já existia no início; o restante é lançado em coleções
        if rng.random() < 0.6:
            launch = DATA_START - pd.Timedelta(days=int(rng.integers(30, 540)))
        else:
            # coleção de verão lançada em ago-out, inverno em fev-abr, atemporal qualquer mês
            offset = int(rng.integers(0, span_days - 20))
            launch = DATA_START + pd.Timedelta(days=offset)
            if season == "Verão":
                launch = pd.Timestamp(year=launch.year, month=int(rng.choice([8, 9, 10])), day=int(rng.integers(1, 28)))
            elif season == "Inverno":
                launch = pd.Timestamp(year=launch.year, month=int(rng.choice([2, 3, 4])), day=int(rng.integers(1, 28)))
            if launch >= DATA_END - pd.Timedelta(days=10):
                launch = DATA_END - pd.Timedelta(days=int(rng.integers(15, 60)))
        if season == "Verão":
            collection = f"Verão {launch.year + (1 if launch.month >= 8 else 0)}"
        elif season == "Inverno":
            collection = f"Inverno {launch.year}"
        else:
            collection = "Atemporal"

        stock = int(rng.negative_binomial(3, 0.012))
        if rng.random() < 0.06:
            stock = 0  # produtos indisponíveis
        name = f"{sub} {material} {color} {rng.choice(NAME_SUFFIX)}"
        rows.append(
            {
                "product_id": pid,
                "name": name,
                "category": cat,
                "subcategory": sub,
                "target_audience": audience,
                "price": price,
                "cost": cost,
                "color": color,
                "size_range": size,
                "material": material,
                "season": season,
                "collection": collection,
                "launch_date": launch.normalize(),
                "stock": stock,
                "description": f"{sub.lower()} {material.lower()} {color.lower()} {desc} estilo {style}",
            }
        )
    df = pd.DataFrame(rows)
    df["margin"] = ((df["price"] - df["cost"]) / df["price"]).round(4)
    # Faixa de preço relativa à categoria (tercis) — atributo de catálogo
    df["price_band"] = (
        df.groupby("category")["price"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Econômica", "Intermediária", "Premium"]))
        .astype(str)
    )
    style_map = {k: v[5] for k, v in SUBCATEGORIES.items()}
    df["_style"] = df["subcategory"].map(style_map)  # latente: removido antes de salvar no banco
    return df
