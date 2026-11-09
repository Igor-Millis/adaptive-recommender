"""Configuração central do Adaptive Recommender.

Todas as constantes de caminho, datas do protocolo temporal e hiperparâmetros
ficam aqui, para que módulos de dados, modelos, avaliação e interface usem
exatamente os mesmos valores.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ROOT_DIR / "models"
TRAINED_DIR = MODELS_DIR / "trained"
METADATA_DIR = MODELS_DIR / "metadata"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"
SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"

# A URL do banco pode ser sobrescrita por variável de ambiente (ex.: PostgreSQL
# no futuro: postgresql+psycopg://user:pass@host/db). O SQL é ANSI na medida do
# possível; ver docs/ARCHITECTURE.md para notas de migração.
DATABASE_URL = os.environ.get(
    "ADAPTIVE_RECOMMENDER_DB_URL", f"sqlite:///{(DATA_DIR / 'adaptive_recommender.db').as_posix()}"
)

for _d in (RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, TRAINED_DIR, METADATA_DIR, LOGS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Perfil de domínio
# ---------------------------------------------------------------------------
# "synthetic" (padrão) = distribuidora de moda sintética. Outros perfis (config/profiles/<nome>.json)
# permitem rodar o MESMO pipeline em outro domínio (ex.: dados reais "online_retail") trocando categorias,
# datas do protocolo, hemisfério e caminhos de saída — sem tocar nos resultados do perfil sintético.
PROFILE = os.environ.get("ADAPTIVE_RECOMMENDER_PROFILE", "synthetic")
_PROFILE_CFG: dict = {}
if PROFILE != "synthetic":
    _profile_path = ROOT_DIR / "config" / "profiles" / f"{PROFILE}.json"
    _PROFILE_CFG = json.loads(_profile_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Linha do tempo dos dados e protocolo de validação temporal
# ---------------------------------------------------------------------------
DATA_START = pd.Timestamp(_PROFILE_CFG.get("data_start", "2024-11-01"))
DATA_END = pd.Timestamp(_PROFILE_CFG.get("data_end", "2026-11-01"))  # exclusivo: dados até 31/10/2026
VAL_START = pd.Timestamp(_PROFILE_CFG.get("val_start", "2026-05-01"))  # treino: [DATA_START, VAL_START)
TEST_START = pd.Timestamp(_PROFILE_CFG.get("test_start", "2026-08-01"))  # validação: [VAL_START, TEST_START); teste: [TEST_START, DATA_END)
HORIZON_DAYS = int(_PROFILE_CFG.get("horizon_days", 92))  # janela de previsão (≈ 3 meses) usada em rótulos e avaliação
HEMISPHERE = _PROFILE_CFG.get("hemisphere", "south")  # define o que é "verão" no calendário (sazonalidade)

RANDOM_SEED = 42

# Tamanho da base sintética (sobrescrevível por variável de ambiente para experimentos de escala)
# 300 clientes: escolhido por medição — o experimento completo roda em ~2,5 min e os IC95% de NDCG@10
# caem de ±0,07 (50 clientes) para ±0,015, o suficiente para o gating treinável ser aprendido e avaliado.
N_CUSTOMERS = int(os.environ.get("ADAPTIVE_RECOMMENDER_N_CUSTOMERS", 300))
N_PRODUCTS = int(os.environ.get("ADAPTIVE_RECOMMENDER_N_PRODUCTS", 200))

CATEGORIES = _PROFILE_CFG.get("categories", [
    "Camisetas",
    "Camisas",
    "Calças",
    "Bermudas",
    "Vestidos",
    "Saias",
    "Jaquetas",
    "Roupas esportivas",
    "Acessórios",
])

# Saídas da avaliação (cada perfil escreve nos seus próprios arquivos)
EVAL_JSON_PATH = METADATA_DIR / ("evaluation_latest.json" if PROFILE == "synthetic" else f"evaluation_{PROFILE}.json")
EVAL_REPORT_PATH = DOCS_DIR / _PROFILE_CFG.get("report_name", "MODEL_EVALUATION.md")
EVAL_REPORT_TITLE = _PROFILE_CFG.get("report_title", "Avaliação dos Modelos")

EXPERTS = ["content_based", "collaborative", "supervised", "bayesian", "popularity"]
EXPERT_LABELS = {
    "content_based": "Content-Based",
    "collaborative": "Collaborative Filtering",
    "supervised": "Supervisionado",
    "bayesian": "Bayesiano",
    "popularity": "Popularidade",
    "collaborative_mf": "Fatoração de Matrizes (ALS)",
    "random": "Aleatório",
    "hybrid_equal": "Híbrido pesos iguais",
    "hybrid_fixed": "Híbrido pesos fixos (ajustados)",
    "hybrid_rules": "Adaptativo (regras)",
    "hybrid_learned": "Adaptativo (gating treinado)",
}

INTERACTION_TYPES = ["impression", "click", "interest", "add_to_cart", "purchase", "rejection"]
INTERACTION_LABELS = {
    "impression": "Impressão",
    "click": "Clique",
    "interest": "Interesse declarado",
    "add_to_cart": "Adição ao carrinho",
    "purchase": "Compra",
    "rejection": "Rejeição explícita",
}


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContentBasedConfig:
    half_life_days: float = 180.0
    drift_half_life_days: float = 60.0  # usado quando o cliente apresenta mudança de preferência
    drift_threshold: float = 0.15  # divergência de Jensen-Shannon que caracteriza mudança
    block_weights: dict = field(
        default_factory=lambda: {
            "categoria": 1.0,
            "subcategoria": 0.8,
            "publico": 0.6,
            "faixa_preco": 0.6,
            "material": 0.35,
            "cor": 0.2,
            "estacao": 0.35,
            "texto": 0.4,
        }
    )
    interaction_weights: dict = field(
        default_factory=lambda: {
            "click": 0.25,
            "interest": 0.5,
            "add_to_cart": 0.7,
            "rejection": -0.7,
        }
    )


@dataclass(frozen=True)
class CollaborativeConfig:
    n_neighbors: int = 10
    shrinkage: float = 5.0
    half_life_days: float = 365.0
    engagement_weight: float = 0.3


@dataclass(frozen=True)
class MFConfig:
    n_factors: int = 12
    regularization: float = 0.5
    alpha: float = 8.0
    iterations: int = 15


@dataclass(frozen=True)
class SupervisedConfig:
    snapshot_step_days: int = 30
    min_history_days: int = 120  # 1º snapshot só após este tempo de dados
    negative_sample_rate: float = 0.35
    candidate_models: tuple = ("logistic_regression", "random_forest", "lightgbm")


@dataclass(frozen=True)
class BayesianConfig:
    # Evidência fracionária (ver MATHEMATICAL_FOUNDATIONS.md §10)
    impression_failure_weight: float = 0.25  # impressão sem resposta: evidência fraca e ambígua
    rejection_failure_weight: float = 1.0
    cart_no_purchase_failure_weight: float = 0.5
    min_prior_strength: float = 2.0
    max_prior_strength: float = 60.0
    cluster_shrinkage: float = 20.0  # pseudo-contagens da média global na média do cluster
    credible_level: float = 0.90
    mc_samples: int = 4000


@dataclass(frozen=True)
class GatingConfig:
    temperature: float = 0.1
    l2_grid: tuple = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)  # penalidade dos ajustes contextuais θ (escolhida por CV)
    bias_l2: float = 0.01  # penalidade fraca do viés global b_e ("desempenho histórico" de cada especialista)
    cv_folds: int = 5
    min_weight: float = 0.0


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 10
    purchased_rule: str = "all"  # "all" | "recent" | "none"
    recent_days: int = 120
    rejection_block_days: int = 180
    fatigue_impressions: int = 3  # impressões sem resposta em 60 dias antes de penalizar
    fatigue_penalty: float = 0.15
    diversity_lambda: float = 1.0  # 1 = sem diversificação (MMR)
    max_per_category: int | None = None
    margin_weight: float = 0.0  # 0 = otimiza relevância pura
    require_stock: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple = (5, 10, 20)
    main_k: int = 10
    bootstrap_samples: int = 2000
    ci_level: float = 0.95


CB = ContentBasedConfig()
CF = CollaborativeConfig()
MF = MFConfig()
SUP = SupervisedConfig(**_PROFILE_CFG.get("supervised", {}))
BAYES = BayesianConfig()
GATING = GatingConfig()
RANKING = RankingConfig()
EVAL = EvaluationConfig()
