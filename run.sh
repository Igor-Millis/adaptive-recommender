#!/usr/bin/env bash
# Adaptive Recommender — inicialização com um comando (macOS / Linux)
# 1. cria/ativa o ambiente virtual  2. instala dependências  3. prepara banco, modelos e avaliação  4. abre o app
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" PYTHONIOENCODING=utf-8

PY=${PYTHON:-python3}
if [ ! -x ".venv/bin/python" ]; then
  echo "[1/4] Criando ambiente virtual .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] Verificando dependências..."
if ! python -c "import streamlit, sklearn, lightgbm, plotly, sqlalchemy, scipy, pandas, joblib" 2>/dev/null; then
  python -m pip install --upgrade pip -q
  python -m pip install -r requirements.txt -q
fi

echo "[3/4] Verificando banco de dados, modelos e avaliação..."
python scripts/initialize_database.py
if ! python -c "import sys; from database.connection import get_engine; from src.services.model_service import ModelService; sys.exit(0 if ModelService(get_engine()).is_trained() else 1)"; then
  echo "      Treinando modelos pela primeira vez (~3 min)..."
  python scripts/train_models.py > /dev/null
fi
if [ ! -f "models/metadata/evaluation_latest.json" ]; then
  echo "      Executando a avaliação experimental (~3 min)..."
  python scripts/run_evaluation.py
fi

echo "[4/4] Abrindo http://localhost:8501 ..."
exec python -m streamlit run app/main.py --server.headless false
