@echo off
REM ==========================================================================
REM  Adaptive Recommender - inicializacao com um clique (Windows)
REM  1. verifica/cria o ambiente virtual   2. ativa   3. verifica dependencias
REM  4. inicializa banco/modelos se necessario   5. abre a aplicacao Streamlit
REM ==========================================================================
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Criando ambiente virtual .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERRO: Python nao encontrado. Instale o Python 3.11 ou superior e tente novamente.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Ambiente virtual encontrado.
)

echo [2/5] Ativando ambiente...
call ".venv\Scripts\activate.bat"

echo [3/5] Verificando dependencias...
python -c "import streamlit, sklearn, lightgbm, plotly, sqlalchemy, scipy, pandas, joblib" 2>nul
if errorlevel 1 (
    echo        Instalando dependencias (somente na primeira execucao^)...
    python -m pip install --upgrade pip -q
    python -m pip install -r requirements.txt -q
    if errorlevel 1 (
        echo ERRO: falha ao instalar dependencias.
        pause
        exit /b 1
    )
)

echo [4/5] Verificando banco de dados e modelos...
python scripts\initialize_database.py
if errorlevel 1 (
    echo ERRO ao inicializar o banco.
    pause
    exit /b 1
)
python -c "import sys; from database.connection import get_engine; from src.services.model_service import ModelService; sys.exit(0 if ModelService(get_engine()).is_trained() else 1)"
if errorlevel 1 (
    echo        Treinando modelos pela primeira vez (cerca de 1 minuto^)...
    python scripts\train_models.py >nul
    if errorlevel 1 (
        echo ERRO no treinamento dos modelos. Veja logs\adaptive_recommender.log
        pause
        exit /b 1
    )
)
if not exist "models\metadata\evaluation_latest.json" (
    echo        Executando a avaliacao experimental (cerca de 1 minuto^)...
    python scripts\run_evaluation.py
)

echo [5/5] Iniciando a aplicacao em http://localhost:8501 ...
python -m streamlit run app\main.py --server.headless false
endlocal
