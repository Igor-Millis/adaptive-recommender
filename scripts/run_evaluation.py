"""Executa o experimento de avaliação temporal completo e atualiza docs/MODEL_EVALUATION.md.

Uso:  python scripts/run_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.connection import get_engine  # noqa: E402
from database.repositories import Repositories  # noqa: E402
from src.evaluation.experiments import run_experiment  # noqa: E402
from src.preprocessing.snapshot import Dataset  # noqa: E402


def main() -> None:
    engine = get_engine()
    rep = run_experiment(Dataset.from_db(engine), save=True, repos=Repositories(engine))
    print(f"Avaliação concluída em {rep['elapsed_seconds']:.1f}s (run {rep.get('eval_run_id')}).")
    for c in rep["conclusions"]:
        print(" -", c)


if __name__ == "__main__":
    main()
