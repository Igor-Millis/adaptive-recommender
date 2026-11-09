"""Repositórios de resultados e auditoria: recomendações, pesos, avaliações e eventos."""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text

from database.repositories.base import BaseRepository, _to_db


class RecommendationRepository(BaseRepository):
    table = "recommendation_runs"
    pk = "run_id"
    date_cols = ("created_at", "as_of")

    def save_run(self, customer_id: int, as_of, gating_method: str, config: dict, model_versions: dict,
                 justification: str, recs: pd.DataFrame, weights: dict, signals: dict) -> tuple[int, list[int]]:
        """Persiste execução + itens + pesos dos especialistas numa única transação."""
        rec_ids: list[int] = []
        with self.engine.begin() as c:
            run_id = self.insert({"customer_id": customer_id, "as_of": as_of, "gating_method": gating_method,
                                  "config_json": config, "model_versions_json": model_versions,
                                  "justification": justification, "created_at": pd.Timestamp.now()}, conn=c)
            for r in recs.to_dict("records"):
                rid = c.execute(
                    text("INSERT INTO recommendations (run_id, product_id, rank, final_score, relevance_score, "
                         "conversion_probability, contributions_json, reasons) VALUES (:run, :p, :rank, :fs, :rs, :cp, :cj, :re) "
                         "RETURNING recommendation_id"),
                    {"run": run_id, "p": int(r["product_id"]), "rank": int(r["rank"]), "fs": float(r["final_score"]),
                     "rs": float(r["relevance"]), "cp": _to_db(r.get("conversion_probability")),
                     "cj": _to_db(r.get("contributions", {})), "re": r.get("reasons_text", "")},
                ).scalar_one()
                rec_ids.append(int(rid))
            for expert, w in weights.items():
                c.execute(text("INSERT INTO expert_weights (run_id, customer_id, expert, weight, method, signals_json) "
                               "VALUES (:r, :c, :e, :w, :m, :s)"),
                          {"r": run_id, "c": customer_id, "e": expert, "w": float(w), "m": gating_method, "s": _to_db(signals)})
        return int(run_id), rec_ids

    def runs_for_customer(self, customer_id: int) -> pd.DataFrame:
        return self.list("customer_id = :c", {"c": int(customer_id)}, order_by="run_id DESC")

    def items(self, run_id: int) -> pd.DataFrame:
        return self.query("SELECT * FROM recommendations WHERE run_id = :r ORDER BY rank", {"r": int(run_id)})

    def weights_history(self, customer_id: int) -> pd.DataFrame:
        df = self.query("SELECT w.*, r.created_at FROM expert_weights w JOIN recommendation_runs r ON r.run_id = w.run_id "
                        "WHERE w.customer_id = :c ORDER BY w.run_id", {"c": int(customer_id)})
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
        return df


class EvaluationRepository(BaseRepository):
    table = "evaluation_runs"
    pk = "eval_run_id"
    date_cols = ("created_at",)

    def save(self, protocol: dict, summary: dict, results: pd.DataFrame, report_path: str | None = None) -> int:
        with self.engine.begin() as c:
            rid = self.insert({"protocol_json": protocol, "summary_json": summary, "report_path": report_path,
                               "created_at": pd.Timestamp.now(), "status": "completed"}, conn=c)
            rows = results.assign(eval_run_id=rid).to_dict("records")
            if rows:
                cols = ["eval_run_id", "stage", "model_name", "segment_type", "segment_value", "metric", "k",
                        "value", "ci_low", "ci_high", "n_customers"]
                c.execute(text(f"INSERT INTO evaluation_results ({', '.join(cols)}) VALUES ({', '.join(':' + x for x in cols)})"),
                          [{k: _to_db(r.get(k)) for k in cols} for r in rows])
        return int(rid)

    def latest(self) -> dict | None:
        df = self.list(order_by="eval_run_id DESC")
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        row["protocol"] = json.loads(row["protocol_json"]) if row.get("protocol_json") else {}
        row["summary"] = json.loads(row["summary_json"]) if row.get("summary_json") else {}
        return row

    def results(self, eval_run_id: int) -> pd.DataFrame:
        return self.query("SELECT * FROM evaluation_results WHERE eval_run_id = :r", {"r": int(eval_run_id)})


class EventLogRepository(BaseRepository):
    table = "event_log"
    pk = "event_id"
    date_cols = ("created_at",)

    def log(self, event_type: str, entity: str | None = None, entity_id=None, payload: dict | None = None, conn=None) -> None:
        values = {"event_type": event_type, "entity": entity, "entity_id": None if entity_id is None else str(entity_id),
                  "payload_json": payload or {}, "created_at": pd.Timestamp.now()}
        self.insert(values, conn=conn)

    def recent(self, limit: int = 50) -> pd.DataFrame:
        return self.query(f"SELECT * FROM event_log ORDER BY event_id DESC LIMIT {int(limit)}")
