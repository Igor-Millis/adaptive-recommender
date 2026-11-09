"""Ajuste conjunto de todos os especialistas sobre um mesmo snapshot."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.preprocessing.snapshot import DataSnapshot
from src.recommenders.bayesian import BayesianCategoryModel
from src.recommenders.collaborative import UserKNNRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.context import build_context
from src.recommenders.matrix_factorization import ImplicitALSRecommender
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.supervised import SupervisedRecommender
from src.segmentation.clustering import SegmentationResult
from src.utils.logging_setup import get_logger
from src.utils.validation import validate_scores

log = get_logger("suite")


class ExpertSuite:
    def __init__(self, supervised: SupervisedRecommender | None = None, include_mf: bool = True):
        self.supervised = supervised
        self.include_mf = include_mf
        self.experts: dict = {}
        self.timings: dict[str, float] = {}

    def fit(self, snapshot: DataSnapshot, segmentation: SegmentationResult | None = None,
            bayes_state: pd.DataFrame | None = None, fixed_k: int | None = None) -> "ExpertSuite":
        self.snapshot = snapshot
        t0 = time.perf_counter()
        self.ctx, self.segmentation = build_context(snapshot, segmentation, fixed_k=fixed_k)
        self.timings["context"] = time.perf_counter() - t0
        experts = [PopularityRecommender(), ContentBasedRecommender(), UserKNNRecommender()]
        if self.include_mf:
            experts.append(ImplicitALSRecommender())
        for e in experts:
            t = time.perf_counter()
            e.fit(snapshot, self.ctx)
            self.timings[e.name] = time.perf_counter() - t
            self.experts[e.name] = e
        t = time.perf_counter()
        bayes = BayesianCategoryModel()
        if bayes_state is not None and len(bayes_state):
            bayes.load_state(snapshot, self.ctx, bayes_state)
        else:
            bayes.fit(snapshot, self.ctx)
        self.timings["bayesian"] = time.perf_counter() - t
        self.experts["bayesian"] = bayes
        if self.supervised is not None:
            t = time.perf_counter()
            self.supervised.fit(snapshot, self.ctx)
            self.timings["supervised"] = time.perf_counter() - t
            self.experts["supervised"] = self.supervised
        for name, e in self.experts.items():
            validate_scores(e.score(), f"scores de {name}")
        return self

    def scores(self, names: list[str] | None = None) -> dict[str, np.ndarray]:
        names = names or list(self.experts)
        return {n: self.experts[n].score() for n in names if n in self.experts}

    def confidences(self, names: list[str] | None = None) -> dict[str, np.ndarray]:
        names = names or list(self.experts)
        return {n: np.asarray(self.experts[n].confidence(), float) for n in names if n in self.experts}

    def refresh_customer(self, snapshot: DataSnapshot, customer_id: int, bayes_state: pd.DataFrame | None = None) -> dict:
        """Atualização incremental após feedback: features, perfis CB, linha da matriz CF e estado bayesiano.

        Parâmetros do modelo supervisionado, fatores ALS e clusters NÃO são reajustados aqui
        (isso ocorre apenas na rotina controlada de retreinamento).
        """
        self.snapshot = snapshot
        ctx, _ = build_context(snapshot, self.segmentation)
        self.ctx = ctx
        done = {}
        for name in ("content_based", "collaborative"):
            if name in self.experts:
                self.experts[name].update_customer(snapshot, ctx, customer_id)
                done[name] = "perfil/linha do cliente atualizados"
        self.experts["popularity"].fit(snapshot, ctx)
        done["popularity"] = "contagens recalculadas"
        if "collaborative_mf" in self.experts:
            self.experts["collaborative_mf"].snapshot = snapshot
            done["collaborative_mf"] = "mantido (fatores só mudam no retreinamento)"
        if bayes_state is not None:
            self.experts["bayesian"].load_state(snapshot, ctx, bayes_state)
            done["bayesian"] = "posteriores carregadas do estado incremental"
        if "supervised" in self.experts:
            self.experts["supervised"].update_customer(snapshot, ctx, customer_id)
            done["supervised"] = "features recalculadas (parâmetros inalterados)"
        return done
