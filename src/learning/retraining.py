"""Rotina controlada de (re)treinamento de todos os componentes.

Nível de atualização por componente:
  | componente                   | após cada feedback        | nesta rotina                        |
  |------------------------------|---------------------------|-------------------------------------|
  | parâmetros bayesianos (α, β) | incremental (conjugado)   | priors reestimados (Bayes empírico) |
  | features do cliente          | recalculadas              | recalculadas para todos             |
  | perfil Content-Based         | perfil do cliente         | todos os perfis                     |
  | matriz colaborativa          | linha do cliente          | matriz completa                     |
  | supervisionado               | só features (predição)    | retreino + calibração + nova versão |
  | ALS / clusterização / gating | —                         | reajuste + nova versão              |

Garantias: lock de execução única; dados operacionais (pedidos, interações)
nunca são apagados; artefatos antigos são preservados (status "archived").
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd

from config.settings import CB, CF, EXPERTS, HORIZON_DAYS, MF, MODELS_DIR, TRAINED_DIR
from database.repositories import Repositories
from src.adaptive.gating import LearnedGating, tune_fixed_weights
from src.evaluation.temporal_validation import build_stage
from src.preprocessing.feature_engineering import build_customer_features
from src.preprocessing.snapshot import Dataset
from src.recommenders.bayesian import BayesianCategoryModel
from src.recommenders.context import build_context
from src.recommenders.supervised import SupervisedRecommender
from src.segmentation.clustering import SegmentationResult, fit_segmentation
from src.utils.logging_setup import get_logger

log = get_logger("retraining")
LOCK = MODELS_DIR / ".retraining.lock"


@contextmanager
def training_lock(stale_seconds: int = 1800):
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < stale_seconds:
        raise RuntimeError("Já existe um retreinamento em andamento (lock ativo).")
    LOCK.write_text(str(os.getpid()))
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def _rel(path) -> str:
    try:
        return str(path.relative_to(MODELS_DIR.parent))
    except ValueError:
        return str(path)


def segmentation_from_db(repos: Repositories) -> SegmentationResult | None:
    run = repos.clusters.active_run()
    if run is None:
        return None
    a = repos.clusters.assignments(run["cluster_run_id"])
    prof = repos.clusters.profiles(run["cluster_run_id"])
    params = json.loads(run["params_json"] or "{}")
    diag = json.loads(run["diagnostics_json"] or "{}")
    labels = a.set_index("customer_id")["cluster_id"].astype(int)
    names = dict(zip(prof["cluster_id"].astype(int), prof["label"]))
    profiles = prof.rename(columns={"centroid_json": "centroid"})
    profiles["centroid"] = profiles["centroid"].map(lambda s: json.loads(s) if s else {})
    return SegmentationResult(as_of=pd.Timestamp(run["as_of"]), k=int(run["k"]), labels=labels, label_names=names,
                              profiles=profiles, diagnostics=pd.DataFrame(diag.get("k_selection", [])),
                              hierarchical=diag.get("hierarchical", {}), centroids=np.array(params.get("centroids", [])),
                              feature_names=params.get("feature_names", []), silhouette=float(run["silhouette"] or 0))


def retrain_all(engine, as_of: pd.Timestamp | None = None, train_gating: bool = True, trained_dir=None) -> dict:
    repos = Repositories(engine)
    trained_dir = trained_dir or TRAINED_DIR
    trained_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with training_lock():
        ds = Dataset.from_db(engine)
        as_of = pd.Timestamp(as_of) if as_of is not None else max(pd.Timestamp.now().floor("s"), ds.max_date + pd.Timedelta(seconds=1))
        snap = ds.snapshot(as_of)
        report: dict = {"as_of": str(as_of)}
        repos.events.log("retraining_started", payload={"as_of": str(as_of)})

        # 1. features + segmentação (ids de cluster alinhados com a execução anterior)
        feats = build_customer_features(snap)
        repos.features.upsert(feats, as_of)
        prev = segmentation_from_db(repos)
        seg = fit_segmentation(feats, as_of, previous_centroids=prev.centroids if prev is not None and len(prev.centroids) else None)
        diag = {"k_selection": seg.diagnostics.to_dict("records"),
                "hierarchical": {k: v for k, v in seg.hierarchical.items() if k != "ward_labels"}}
        crun = repos.clusters.save_run(as_of, "kmeans", seg.k, seg.silhouette,
                                       {"centroids": seg.centroids.tolist(), "feature_names": seg.feature_names},
                                       diag, seg.assignments(), seg.profiles)
        repos.models.register_version("clustering", as_of, {"algorithm": "kmeans", "k": seg.k, "cluster_run_id": crun},
                                      {"silhouette": seg.silhouette, **diag["hierarchical"]}, None, "segmentation",
                                      "K-Means com escolha de k por silhouette × estabilidade")
        report["clustering"] = {"k": seg.k, "silhouette": seg.silhouette, "cluster_run_id": crun}

        # 2. supervisionado
        sup = SupervisedRecommender.train(ds, as_of, fixed_k=seg.k)
        v_next = len(repos.models.list("model_name = 'supervised'")) + 1
        path = trained_dir / f"supervised_v{v_next}.joblib"
        joblib.dump(sup.to_artifact(), path)
        sel = sup.report["candidates"][sup.algorithm]["metrics"]
        repos.models.register_version("supervised", as_of, {"algorithm": sup.algorithm, "n_train": sup.report["n_train"],
                                                            "snapshots": len(sup.report["train_snapshots"])},
                                      {k: v for k, v in sel.items() if isinstance(v, (int, float))},
                                      _rel(path), "expert",
                                      "Propensão de compra calibrada (melhor de LR/RF/LightGBM no holdout temporal)")
        with open(trained_dir / f"supervised_v{v_next}_report.json", "w", encoding="utf-8") as f:
            json.dump(sup.report, f, ensure_ascii=False, indent=1, default=str)
        report["supervised"] = {"algorithm": sup.algorithm, "holdout": sel, "artifact": str(path)}

        # 3. bayesiano (priors reestimados + posteriores recalculadas em lote)
        ctx, _ = build_context(snap, seg)
        bayes = BayesianCategoryModel().fit(snap, ctx)
        priors = {f"{st}|{cat}": {"mu": mu, "kappa": kp} for (st, cat), (mu, kp) in bayes.global_prior.items()}
        vid = repos.models.register_version("bayesian", as_of, {"granularity": "cliente×categoria×estágio", "priors": priors},
                                            {"cells": int(bayes.state_frame().shape[0])}, None, "expert",
                                            "Beta-Binomial em dois estágios com prior hierárquico")
        repos.bayes.replace_all(bayes.state_frame(), vid)
        with engine.begin() as conn:
            repos.events.log("bayesian_recalibrated", "bayesian_estimates", vid, {"as_of": str(as_of)}, conn=conn)

        # 4. especialistas reajustados a cada carga (determinísticos): registra parâmetros para auditoria
        for name, params in [("content_based", CB.__dict__), ("collaborative", CF.__dict__),
                             ("collaborative_mf", MF.__dict__), ("popularity", {"half_life_days": 60, "window_days": 180})]:
            repos.models.register_version(name, as_of, {k: (v if not isinstance(v, dict) else dict(v)) for k, v in params.items()},
                                          {"n_order_lines": int(len(snap.items)), "n_interactions": int(len(snap.interactions))},
                                          None, "expert", "Reajustado a partir do banco ao carregar o motor")

        # 5. gating: aprendido na janela completa mais recente [as_of − H, as_of)
        if train_gating:
            g_cut = (as_of - pd.Timedelta(days=HORIZON_DAYS)).normalize()
            sup_g = SupervisedRecommender.train(ds, g_cut, fixed_k=seg.k)
            stage = build_stage(ds, "gating_producao", g_cut, as_of, sup_g, fixed_k=seg.k)
            norm = {e: stage.norm[e] for e in EXPERTS}
            learned = LearnedGating().fit(stage.signals, norm, stage.Y, stage.mask)
            fixed_w, fixed_v = tune_fixed_weights(norm, stage.Y, stage.mask, stage.signals)
            payload = {"learned": learned.to_dict(), "fixed": fixed_w, "fixed_window_ndcg": fixed_v,
                       "window": [str(g_cut), str(as_of)], "n_customers": len(stage.relevant)}
            gv = len(repos.models.list("model_name = 'gating'")) + 1
            gpath = trained_dir / f"gating_v{gv}.json"
            gpath.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            repos.models.register_version("gating", as_of, {"variant": learned.train_info.get("variant"), "l2": learned.l2,
                                                            "fixed_weights": fixed_w, "window": payload["window"]},
                                          {"cv": learned.cv_table, "fixed_window_ndcg": fixed_v},
                                          _rel(gpath), "gating",
                                          "Gating residual treinável + pesos fixos ajustados")
            report["gating"] = {"variant": learned.train_info.get("variant"), "fixed": fixed_w}
        report["elapsed_seconds"] = time.perf_counter() - t0
        repos.events.log("retraining_completed", payload=report)
        log.info("Retreinamento concluído em %.1fs", report["elapsed_seconds"])
        return report
