"""Operações de ciclo de vida de modelos: estado, treino, avaliação e versões."""
from __future__ import annotations

import json

import pandas as pd

from config.settings import EVAL_JSON_PATH
from database.repositories import Repositories
from src.learning.retraining import retrain_all
from src.preprocessing.snapshot import Dataset


class ModelService:
    def __init__(self, engine):
        self.engine = engine
        self.repos = Repositories(engine)

    def is_trained(self) -> bool:
        return not self.repos.models.active("supervised").empty and not self.repos.bayes.all().empty

    def retrain(self, train_gating: bool = True) -> dict:
        return retrain_all(self.engine, train_gating=train_gating)

    def run_evaluation(self) -> dict:
        from src.evaluation.experiments import run_experiment

        return run_experiment(Dataset.from_db(self.engine), save=True, repos=self.repos)

    def versions(self) -> pd.DataFrame:
        return self.repos.models.history()

    def latest_evaluation(self) -> dict | None:
        path = EVAL_JSON_PATH
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["results"] = pd.DataFrame(data.get("results", []))
        return data

    def recent_events(self, n: int = 30) -> pd.DataFrame:
        return self.repos.events.recent(n)
