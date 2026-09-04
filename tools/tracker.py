"""
实验追踪 — SQLite 后端，零外部依赖。

用法:
  tracker = ExperimentTracker("my_project")
  run_id = tracker.create_run(config={...}, tags=["nano", "lr_test"])
  tracker.log_metric(run_id, "loss", 2.34, step=100)
  tracker.log_metrics(run_id, {"eval/ceval": 0.35, "eval/cmmlu": 0.32}, step=500)
  tracker.finish_run(run_id, note="first experiment")
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(self, project: str, db_path: str = "experiments.db"):
        self.project = project
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                config TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                status TEXT DEFAULT 'running',
                created_at REAL NOT NULL,
                finished_at REAL,
                note TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, step);
            CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
        """)
        self._conn.commit()

    def create_run(
        self, config: dict | None = None, tags: list[str] | None = None, run_name: str | None = None
    ) -> str:
        run_id = run_name or f"{self.project}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = time.time()
        self._conn.execute(
            "INSERT INTO runs (id, project, config, tags, status, created_at) VALUES (?, ?, ?, ?, 'running', ?)",
            (run_id, self.project, json.dumps(config or {}), json.dumps(tags or []), now),
        )
        self._conn.commit()
        return run_id

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        self._conn.execute(
            "INSERT INTO metrics (run_id, step, key, value, timestamp) VALUES (?, ?, ?, ?, ?)",
            (run_id, step, key, value, time.time()),
        )
        self._conn.commit()

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0):
        for k, v in metrics.items():
            self.log_metric(run_id, k, v, step)

    def finish_run(self, run_id: str, note: str = ""):
        self._conn.execute(
            "UPDATE runs SET status='finished', finished_at=?, note=? WHERE id=?",
            (time.time(), note, run_id),
        )
        self._conn.commit()

    def get_runs(self, project: str | None = None) -> list[dict]:
        project = project or self.project
        rows = self._conn.execute(
            "SELECT id, project, config, tags, status, created_at, finished_at, note "
            "FROM runs WHERE project=? ORDER BY created_at DESC",
            (project,),
        ).fetchall()
        return [
            {
                "id": r[0], "project": r[1], "config": json.loads(r[2]),
                "tags": json.loads(r[3]), "status": r[4],
                "created_at": r[5], "finished_at": r[6], "note": r[7],
            }
            for r in rows
        ]

    def get_metrics(self, run_id: str, keys: list[str] | None = None) -> dict[str, list[tuple[int, float]]]:
        if keys:
            placeholders = ",".join("?" for _ in keys)
            rows = self._conn.execute(
                f"SELECT step, key, value FROM metrics WHERE run_id=? AND key IN ({placeholders}) ORDER BY step",
                (run_id, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT step, key, value FROM metrics WHERE run_id=? ORDER BY step",
                (run_id,),
            ).fetchall()
        result: dict[str, list[tuple[int, float]]] = {}
        for step, key, value in rows:
            result.setdefault(key, []).append((step, value))
        return result

    def close(self):
        self._conn.close()
