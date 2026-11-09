"""Gerador de lojistas (clientes B2B) com perfis latentes controlados.

Cada cliente recebe um *arquétipo* comportamental e parâmetros latentes
(preferência por categoria, estilo, sensibilidade a preço, sazonalidade, deriva
de preferência...). Esses parâmetros ficam APENAS no arquivo de ground truth
(``data/synthetic/ground_truth_customers.json``) e nunca são lidos pelos modelos.
O banco recebe somente dados cadastrais (observáveis no cadastro), e todas as
características comportamentais são recalculadas a partir dos pedidos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config.settings import CATEGORIES, DATA_START

STYLES = ["esportivo", "social", "casual", "praia"]
AUDIENCES = ["Masculino", "Feminino", "Unissex", "Infantil"]
BANDS = ["Econômica", "Intermediária", "Premium"]

# arquétipo -> (concentração por categoria, preferência de estilo, sensibilidade a preço (média), faixa preferida,
#               pedidos/mês, linhas/pedido, amplitude sazonal, mês de pico, força de complementaridade, segmento)
ARCHETYPES: dict[str, dict] = {
    "esportivo": dict(
        cats={"Roupas esportivas": 9, "Camisetas": 3, "Bermudas": 2.5, "Acessórios": 2.5, "Calças": 1.5},
        styles={"esportivo": 6, "casual": 1.5, "praia": 0.6, "social": 0.2},
        lam=1.0, band="Intermediária", rate=1.7, lines=6, amp=0.15, peak=1, comp=0.15, segment="Moda esportiva",
    ),
    "economico": dict(
        cats={"Camisetas": 5, "Bermudas": 3, "Calças": 3, "Acessórios": 2, "Vestidos": 1.5, "Saias": 1.5},
        styles={"casual": 4, "praia": 1.5, "esportivo": 1.2, "social": 0.4},
        lam=2.4, band="Econômica", rate=2.2, lines=5, amp=0.15, peak=12, comp=0.1, segment="Moda popular",
    ),
    "social": dict(
        cats={"Camisas": 7, "Calças": 5, "Saias": 3, "Vestidos": 3, "Jaquetas": 2.5, "Acessórios": 1.2},
        styles={"social": 6, "casual": 1.5, "esportivo": 0.15, "praia": 0.2},
        lam=0.5, band="Premium", rate=1.3, lines=6, amp=0.1, peak=3, comp=0.25, segment="Moda social",
    ),
    "sazonal": dict(
        cats={"Bermudas": 5, "Vestidos": 4, "Camisetas": 4, "Acessórios": 2.5, "Saias": 1.5, "Roupas esportivas": 1},
        styles={"praia": 5, "casual": 2.5, "esportivo": 0.8, "social": 0.3},
        lam=1.2, band="Intermediária", rate=1.4, lines=7, amp=0.95, peak=11, comp=0.15, segment="Moda praia",
    ),
    "diversificado": dict(
        cats={c: 2.5 for c in CATEGORIES},
        styles={"casual": 2, "social": 1.5, "esportivo": 1.5, "praia": 1.2},
        lam=1.1, band="Intermediária", rate=1.6, lines=8, amp=0.2, peak=12, comp=0.2, segment="Multimarcas",
    ),
    "premium": dict(
        cats={"Jaquetas": 4, "Vestidos": 4, "Camisas": 4, "Calças": 3, "Acessórios": 2.5, "Saias": 1.5},
        styles={"social": 3, "casual": 3, "praia": 0.8, "esportivo": 0.6},
        lam=0.3, band="Premium", rate=1.2, lines=7, amp=0.25, peak=6, comp=0.25, segment="Moda premium",
    ),
    "migrante": dict(  # mudança gradual de preferência (casual -> esportivo)
        cats={"Camisetas": 5, "Calças": 4, "Bermudas": 3, "Camisas": 2, "Acessórios": 1},
        cats_end={"Roupas esportivas": 8, "Acessórios": 3, "Camisetas": 2.5, "Bermudas": 2},
        styles={"casual": 4, "esportivo": 1.0, "praia": 0.8, "social": 0.6},
        styles_end={"esportivo": 5, "casual": 1.5, "praia": 0.5, "social": 0.2},
        lam=1.2, band="Intermediária", rate=1.6, lines=6, amp=0.15, peak=12, comp=0.1, segment="Moda casual",
    ),
    "complementar": dict(
        cats={"Calças": 4, "Camisas": 4, "Vestidos": 3, "Acessórios": 3, "Saias": 2, "Camisetas": 1.5},
        styles={"casual": 3, "social": 2.5, "praia": 0.5, "esportivo": 0.5},
        lam=0.9, band="Intermediária", rate=1.4, lines=5, amp=0.15, peak=5, comp=0.75, segment="Moda casual",
    ),
}
ARCHETYPE_COUNTS = {
    "esportivo": 7, "economico": 7, "social": 6, "sazonal": 6,
    "diversificado": 7, "premium": 6, "migrante": 6, "complementar": 5,
}

CITIES = [
    ("São Paulo", "SP"), ("Campinas", "SP"), ("Santos", "SP"), ("Ribeirão Preto", "SP"), ("Sorocaba", "SP"),
    ("Rio de Janeiro", "RJ"), ("Niterói", "RJ"), ("Cabo Frio", "RJ"), ("Belo Horizonte", "MG"), ("Uberlândia", "MG"),
    ("Juiz de Fora", "MG"), ("Curitiba", "PR"), ("Londrina", "PR"), ("Florianópolis", "SC"), ("Balneário Camboriú", "SC"),
    ("Joinville", "SC"), ("Porto Alegre", "RS"), ("Caxias do Sul", "RS"), ("Salvador", "BA"), ("Porto Seguro", "BA"),
    ("Recife", "PE"), ("Fortaleza", "CE"), ("Natal", "RN"), ("Goiânia", "GO"), ("Brasília", "DF"), ("Vitória", "ES"),
]
BEACH_CITIES = [("Cabo Frio", "RJ"), ("Balneário Camboriú", "SC"), ("Porto Seguro", "BA"), ("Natal", "RN"), ("Florianópolis", "SC"), ("Santos", "SP")]

NAME_PREFIX = {
    "Moda esportiva": ["Ativa", "Movimento", "Arena", "Pulso", "Meta"],
    "Moda popular": ["Bazar", "Mix", "Ponto", "Varejão", "Casa"],
    "Moda social": ["Alfaiataria", "Ateliê", "Maison", "Galeria", "Studio"],
    "Moda praia": ["Brisa", "Maré", "Areia", "Coral", "Onda"],
    "Multimarcas": ["Empório", "Espaço", "Vitrine", "Mercado", "Armazém"],
    "Moda premium": ["Maison", "Boutique", "Galeria", "Atelier", "Salão"],
    "Moda casual": ["Loja", "Estação", "Rua", "Vila", "Cantinho"],
}
NAME_CORE = [
    "Aurora", "Ipê", "Jacarandá", "Horizonte", "Bela Vista", "Primavera", "Dois Irmãos", "Sol Nascente", "Cedro",
    "Jatobá", "Lua Cheia", "Vento Sul", "Serra Azul", "Canela", "Girassol", "Alecrim", "Pitanga", "Manacá",
    "Bem-te-vi", "Sabiá", "Araucária", "Mandacaru", "Laguna", "Itapuã", "Arpoador", "Tamarindo", "Vereda",
    "Aroeira", "Cajueiro", "Margarida", "Ventania", "Estrela", "Oceano", "Pomar", "Quintal", "Varanda",
    "Bonança", "Campina", "Caravela", "Farol", "Garoa", "Lírio", "Mirante", "Orquídea", "Palmeira",
    "Recanto", "Seriema", "Tucano", "Urucum", "Violeta",
]
SIZE_QTY = {"Pequeno": 4, "Médio": 8, "Grande": 16}
REVENUE_BAND = {"Pequeno": "Até R$ 360 mil", "Médio": "R$ 360 mil a R$ 4,8 mi", "Grande": "Acima de R$ 4,8 mi"}


@dataclass
class LatentCustomer:
    customer_id: int
    archetype: str
    cat_pref_start: dict
    cat_pref_end: dict
    style_pref_start: dict
    style_pref_end: dict
    audience_pref: dict
    price_sensitivity: float
    preferred_band: str
    orders_per_month: float
    lines_per_order: float
    qty_base: int
    seasonal_amplitude: float
    seasonal_peak_month: int
    complementarity: float
    drift_start: str | None
    drift_end: str | None
    history_profile: str
    never_orders: bool


def _dirichlet(rng: np.random.Generator, conc: dict, keys: list[str], floor: float = 0.3) -> dict:
    alpha = np.array([conc.get(k, floor) for k in keys], dtype=float)
    p = rng.dirichlet(alpha * 3.0)  # escala 3: perfis coerentes, mas não idênticos
    return {k: float(v) for k, v in zip(keys, p)}


def archetype_counts(n_customers: int) -> dict[str, int]:
    """Distribui n clientes entre arquétipos na proporção da base de referência (50 clientes)."""
    total = sum(ARCHETYPE_COUNTS.values())
    raw = {a: n_customers * c / total for a, c in ARCHETYPE_COUNTS.items()}
    counts = {a: int(np.floor(v)) for a, v in raw.items()}
    for a in sorted(raw, key=lambda a: raw[a] - counts[a], reverse=True)[: n_customers - sum(counts.values())]:
        counts[a] += 1
    return counts


# Proporção de cada perfil de histórico (na base de 50: 3, 3, 2, 3 e 2 clientes)
HISTORY_SHARES = {"moderado": 0.06, "pouco": 0.06, "novo_validacao": 0.04, "novo_teste": 0.06, "sem_historico": 0.04}


def generate_customers(seed: int = 42, n_customers: int = 50) -> tuple[pd.DataFrame, list[LatentCustomer]]:
    rng = np.random.default_rng(seed)
    archetypes = [a for a, n in archetype_counts(n_customers).items() for _ in range(n)]
    rng.shuffle(archetypes)
    n = len(archetypes)

    # Sobreposição de perfis de histórico (cold start / pouco histórico) sobre os arquétipos
    idx = rng.permutation(n)
    history_profile = np.array(["completo"] * n, dtype=object)
    start = 0
    for prof, share in HISTORY_SHARES.items():
        k = max(2, int(round(share * n)))
        history_profile[idx[start:start + k]] = prof
        start += k

    names_used: set[str] = set()
    cores = list(NAME_CORE)
    rng.shuffle(cores)
    rows, latents = [], []
    for i, arch in enumerate(archetypes):
        cid = i + 1
        spec = ARCHETYPES[arch]
        segment = spec["segment"]
        if arch == "complementar":
            segment = str(rng.choice(["Moda casual", "Multimarcas"]))
        size = {
            "premium": rng.choice(["Médio", "Grande"], p=[0.3, 0.7]),
            "economico": rng.choice(["Pequeno", "Médio"], p=[0.6, 0.4]),
        }.get(arch, rng.choice(["Pequeno", "Médio", "Grande"], p=[0.4, 0.4, 0.2]))
        size = str(size)
        city, state = (BEACH_CITIES if arch == "sazonal" else CITIES)[int(rng.integers(0, len(BEACH_CITIES if arch == "sazonal" else CITIES)))]

        prof = history_profile[i]
        if prof == "completo":
            reg = DATA_START - pd.Timedelta(days=int(rng.integers(60, 1800)))
        elif prof == "moderado":
            reg = pd.Timestamp("2025-08-01") + pd.Timedelta(days=int(rng.integers(0, 120)))
        elif prof == "pouco":
            reg = pd.Timestamp("2026-02-10") + pd.Timedelta(days=int(rng.integers(0, 60)))
        elif prof == "novo_validacao":
            reg = pd.Timestamp("2026-05-03") + pd.Timedelta(days=int(rng.integers(0, 40)))
        elif prof == "novo_teste":
            reg = pd.Timestamp("2026-08-03") + pd.Timedelta(days=int(rng.integers(0, 40)))
        else:
            reg = pd.Timestamp("2026-10-15") + pd.Timedelta(days=int(rng.integers(0, 12)))

        cat_start = _dirichlet(rng, spec["cats"], CATEGORIES)
        cat_end = _dirichlet(rng, spec.get("cats_end", spec["cats"]), CATEGORIES) if "cats_end" in spec else cat_start
        sty_start = _dirichlet(rng, spec["styles"], STYLES, floor=0.1)
        sty_end = _dirichlet(rng, spec.get("styles_end", spec["styles"]), STYLES, floor=0.1) if "styles_end" in spec else sty_start
        aud_conc = {"Masculino": 2, "Feminino": 2, "Unissex": 1.5, "Infantil": 0.4}
        if segment in ("Moda social", "Moda premium"):
            aud_conc = {"Masculino": 2, "Feminino": 3, "Unissex": 0.8, "Infantil": 0.2}
        aud = _dirichlet(rng, aud_conc, AUDIENCES, floor=0.2)
        lam = float(np.clip(rng.normal(spec["lam"], 0.25), 0.05, 3.0))
        band = spec["band"] if rng.random() < 0.85 else str(rng.choice(BANDS))
        rate = float(np.clip(rng.normal(spec["rate"], 0.35), 0.5, 3.5))
        lines = float(np.clip(rng.normal(spec["lines"], 1.0), 2.5, 11))
        drift_start = drift_end = None
        if arch == "migrante":
            drift_start = str((pd.Timestamp("2025-07-01") + pd.Timedelta(days=int(rng.integers(0, 90)))).date())
            drift_end = str((pd.Timestamp(drift_start) + pd.Timedelta(days=int(rng.integers(240, 330)))).date())

        latent = LatentCustomer(
            customer_id=cid, archetype=arch, cat_pref_start=cat_start, cat_pref_end=cat_end,
            style_pref_start=sty_start, style_pref_end=sty_end, audience_pref=aud, price_sensitivity=lam,
            preferred_band=band, orders_per_month=rate, lines_per_order=lines, qty_base=SIZE_QTY[size],
            seasonal_amplitude=float(np.clip(rng.normal(spec["amp"], 0.08), 0.0, 1.0)),
            seasonal_peak_month=int((spec["peak"] - 1 + rng.integers(-1, 2)) % 12 + 1),
            complementarity=float(np.clip(rng.normal(spec["comp"], 0.07), 0.0, 0.95)),
            drift_start=drift_start, drift_end=drift_end, history_profile=prof,
            never_orders=prof == "sem_historico",
        )
        latents.append(latent)

        # ---- dados cadastrais (observáveis) ----
        prefix_pool = NAME_PREFIX[segment]
        name = None
        for _ in range(30):  # combinações prefixo × nome; com bases grandes acrescenta a cidade
            cand = f"{rng.choice(prefix_pool)} {rng.choice(cores)}"
            if cand not in names_used:
                name = cand
                break
        if name is None:
            base = f"{rng.choice(prefix_pool)} {rng.choice(cores)} {city}"
            name, k = base, 2
            while name in names_used:
                name, k = f"{base} {k}", k + 1
        names_used.add(name)
        # Categorias declaradas no cadastro: sorteadas pela preferência *inicial* com ruído
        p = np.array([cat_start[c] for c in CATEGORIES]) ** 1.3
        p = p / p.sum()
        declared = list(rng.choice(CATEGORIES, size=3, replace=False, p=p))
        declared_band = band if rng.random() < 0.75 else str(rng.choice(BANDS))
        freq_decl = rate * float(np.exp(rng.normal(0, 0.25)))
        freq_label = "Semanal" if freq_decl >= 3 else "Quinzenal" if freq_decl >= 1.7 else "Mensal" if freq_decl >= 0.8 else "Bimestral"
        rows.append(
            {
                "customer_id": cid,
                "name": name,
                "city": city,
                "state": state,
                "registration_date": reg.normalize(),
                "store_size": size,
                "business_segment": segment,
                "revenue_band": REVENUE_BAND[size],
                "declared_categories": ";".join(declared),
                "declared_price_band": declared_band,
                "declared_purchase_frequency": freq_label,
            }
        )
    return pd.DataFrame(rows), latents


def latents_to_records(latents: list[LatentCustomer]) -> list[dict]:
    return [asdict(lat) for lat in latents]
