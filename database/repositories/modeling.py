"""Repositórios de estado de modelos: registro/versões, clusters, Bayes e features."""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text

from database.repositories.base import BaseRepository, _to_db


class ModelRegistryRepository(BaseRepository):
    table = "model_versions"
    pk = "version_id"
    date_cols = ("trained_at", "train_cutoff")

    def ensure_model(self, model_name: str, model_type: str, description: str = "") -> None:
        with self.engine.begin() as c:
            c.execute(
                text("INSERT INTO models (model_name, model_type, description) VALUES (:n, :t, :d) "
                     "ON CONFLICT (model_name) DO UPDATE SET model_type = excluded.model_type, description = excluded.description"),
                {"n": model_name, "t": model_type, "d": description},
            )

    def register_version(self, model_name: str, train_cutoff, params: dict, metrics: dict, artifact_path: str | None,
                         model_type: str = "expert", description: str = "") -> int:
        """Cria nova versão ativa e arquiva as anteriores (histórico preservado)."""
        self.ensure_model(model_name, model_type, description)
        with self.engine.begin() as c:
            v = c.execute(text("SELECT COALESCE(MAX(version), 0) + 1 FROM model_versions WHERE model_name = :n"),
                          {"n": model_name}).scalar_one()
            c.execute(text("UPDATE model_versions SET status = 'archived' WHERE model_name = :n AND status = 'active'"),
                      {"n": model_name})
            return self.insert({"model_name": model_name, "version": int(v), "trained_at": pd.Timestamp.now(),
                                "train_cutoff": train_cutoff, "params_json": params, "metrics_json": metrics,
                                "artifact_path": artifact_path, "status": "active"}, conn=c)

    def active(self, model_name: str | None = None) -> pd.DataFrame:
        if model_name:
            return self.list("status = 'active' AND model_name = :n", {"n": model_name})
        return self.list("status = 'active'")

    def history(self) -> pd.DataFrame:
        return self.list(order_by="model_name, version DESC")


class ClusterRepository(BaseRepository):
    table = "cluster_runs"
    pk = "cluster_run_id"
    date_cols = ("created_at", "as_of")

    def save_run(self, as_of, algorithm: str, k: int, silhouette: float, params: dict, diagnostics: dict,
                 assignments: pd.DataFrame, profiles: pd.DataFrame) -> int:
        with self.engine.begin() as c:
            c.execute(text("UPDATE cluster_runs SET status = 'archived' WHERE status = 'active'"))
            rid = self.insert({"as_of": as_of, "algorithm": algorithm, "k": k, "silhouette": silhouette,
                               "params_json": params, "diagnostics_json": diagnostics, "status": "active"}, conn=c)
            for r in assignments.to_dict("records"):
                c.execute(text("INSERT INTO customer_clusters (cluster_run_id, customer_id, cluster_id, cluster_label) "
                               "VALUES (:r, :c, :k, :l)"),
                          {"r": rid, "c": int(r["customer_id"]), "k": int(r["cluster_id"]), "l": r["cluster_label"]})
            for r in profiles.to_dict("records"):
                c.execute(text("INSERT INTO cluster_profiles (cluster_run_id, cluster_id, label, description, size, centroid_json) "
                               "VALUES (:r, :k, :l, :d, :s, :j)"),
                          {"r": rid, "k": int(r["cluster_id"]), "l": r["label"], "d": r.get("description", ""),
                           "s": int(r["size"]), "j": _to_db(r.get("centroid", {}))})
        return rid

    def active_run(self) -> dict | None:
        df = self.list("status = 'active'", order_by="cluster_run_id DESC")
        return None if df.empty else df.iloc[0].to_dict()

    def assignments(self, run_id: int) -> pd.DataFrame:
        return self.query("SELECT * FROM customer_clusters WHERE cluster_run_id = :r ORDER BY customer_id", {"r": run_id})

    def profiles(self, run_id: int) -> pd.DataFrame:
        return self.query("SELECT * FROM cluster_profiles WHERE cluster_run_id = :r ORDER BY cluster_id", {"r": run_id})


class BayesianRepository(BaseRepository):
    table = "bayesian_estimates"
    pk = "customer_id"
    date_cols = ("updated_at",)

    def all(self) -> pd.DataFrame:
        return self.query("SELECT * FROM bayesian_estimates ORDER BY customer_id, category, stage")

    def for_customer(self, customer_id: int) -> pd.DataFrame:
        return self.query("SELECT * FROM bayesian_estimates WHERE customer_id = :c ORDER BY category, stage",
                          {"c": int(customer_id)})

    def get_cell(self, customer_id: int, category: str, stage: str, conn=None) -> dict | None:
        sql = text("SELECT * FROM bayesian_estimates WHERE customer_id = :c AND category = :k AND stage = :s")
        params = {"c": int(customer_id), "k": category, "s": stage}
        if conn is not None:
            row = conn.execute(sql, params).mappings().first()
        else:
            with self.engine.connect() as c:
                row = c.execute(sql, params).mappings().first()
        return dict(row) if row else None

    def replace_all(self, df: pd.DataFrame, model_version_id: int | None) -> None:
        with self.engine.begin() as c:
            c.execute(text("DELETE FROM bayesian_estimates"))
            rows = df.assign(model_version_id=model_version_id, updated_at=pd.Timestamp.now()).to_dict("records")
            if rows:
                cols = list(rows[0].keys())
                c.execute(text(f"INSERT INTO bayesian_estimates ({', '.join(cols)}) VALUES ({', '.join(':' + x for x in cols)})"),
                          [{k: _to_db(v) for k, v in r.items()} for r in rows])

    def upsert_cell(self, values: dict, conn) -> None:
        cols = list(values.keys())
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("customer_id", "category", "stage"))
        conn.execute(
            text(f"INSERT INTO bayesian_estimates ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)}) "
                 f"ON CONFLICT (customer_id, category, stage) DO UPDATE SET {updates}"),
            {c: _to_db(v) for c, v in values.items()},
        )

    def log_update(self, values: dict, conn) -> None:
        cols = list(values.keys())
        conn.execute(text(f"INSERT INTO bayesian_update_log ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"),
                     {c: _to_db(v) for c, v in values.items()})

    def update_log(self, customer_id: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM bayesian_update_log"
        params = {}
        if customer_id is not None:
            sql += " WHERE customer_id = :c"
            params["c"] = int(customer_id)
        df = self.query(sql + " ORDER BY log_id", params)
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
        return df


class FeatureRepository(BaseRepository):
    table = "customer_features"
    pk = "customer_id"
    date_cols = ("as_of", "updated_at")

    def upsert(self, features: pd.DataFrame, as_of) -> None:
        with self.engine.begin() as c:
            for cid, row in features.iterrows():
                c.execute(
                    text("INSERT INTO customer_features (customer_id, as_of, features_json, updated_at) VALUES (:c, :a, :f, :u) "
                         "ON CONFLICT (customer_id) DO UPDATE SET as_of = excluded.as_of, features_json = excluded.features_json, "
                         "updated_at = excluded.updated_at"),
                    {"c": int(cid), "a": _to_db(as_of), "u": _to_db(pd.Timestamp.now()),
                     "f": json.dumps({k: _to_db(v) for k, v in row.to_dict().items()}, ensure_ascii=False)},
                )

    def load(self) -> pd.DataFrame:
        df = self.list()
        if df.empty:
            return df
        feats = pd.DataFrame([json.loads(x) for x in df["features_json"]], index=df["customer_id"])
        return feats
