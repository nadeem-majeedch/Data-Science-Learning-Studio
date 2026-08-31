"""
Experiment tracker for the Data Science Learning Studio.

Stores ML experiments in a local SQLite database with full metadata:
dataset, model, hyperparameters, metrics, preprocessing steps, and
generated code.

Usage::

    tracker = ExperimentTracker()                       # default path
    exp_id = tracker.save_experiment(
        dataset_name="iris.csv",
        task_type="classification",
        target="species",
        features=["sepal_length", "sepal_width", ...],
        model="RandomForestClassifier",
        params={"n_estimators": 100},
        metrics={"accuracy": 0.96, "f1": 0.95},
    )
    df = tracker.list_experiments()
    tracker.delete_experiment(exp_id)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Default database path ──────────────────────────────────────────

_DEFAULT_DB = Path("experiments.db")


# ── Data class ──────────────────────────────────────────────────────

@dataclass
class Experiment:
    """Single ML experiment record."""
    experiment_id: int = 0
    created_at: str = ""
    dataset_name: str = ""
    task_type: str = ""  # classification | regression | clustering
    target: str = ""
    features: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    model: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    generated_code: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serialisable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "target": self.target,
            "features": self.features,
            "preprocessing_steps": self.preprocessing_steps,
            "model": self.model,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "generated_code": self.generated_code,
            "notes": self.notes,
        }


# ── Tracker ─────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Lightweight SQLite-backed experiment tracker.

    Each experiment stores full reproducibility metadata.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    # ── Public API ──────────────────────────────────────────────────

    def save_experiment(
        self,
        dataset_name: str,
        task_type: str,
        target: str,
        features: list[str],
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        preprocessing_steps: list[str] | None = None,
        generated_code: str = "",
        notes: str = "",
    ) -> int:
        """
        Save a new experiment. Returns the experiment ID.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO experiments
               (created_at, dataset_name, task_type, target, features,
                preprocessing_steps, model, hyperparameters, metrics,
                generated_code, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                dataset_name,
                task_type,
                target,
                json.dumps(features),
                json.dumps(preprocessing_steps or []),
                model,
                json.dumps(hyperparameters or {}),
                json.dumps(metrics or {}),
                generated_code,
                notes,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_experiment(self, experiment_id: int) -> Experiment | None:
        """Retrieve a single experiment by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def list_experiments(
        self,
        task_type: str | None = None,
        dataset_name: str | None = None,
        order_by: str = "created_at DESC",
    ) -> list[Experiment]:
        """
        List experiments with optional filters.

        Parameters
        ----------
        task_type : filter by 'classification', 'regression', or 'clustering'.
        dataset_name : filter by exact dataset name.
        order_by : SQL ORDER BY clause (default newest first).
        """
        query = "SELECT * FROM experiments WHERE 1=1"
        params: list[Any] = []

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        if dataset_name:
            query += " AND dataset_name = ?"
            params.append(dataset_name)

        query += f" ORDER BY {order_by}"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment by ID. Returns True if it existed."""
        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def compare_experiments(self, experiment_ids: list[int]) -> list[Experiment]:
        """Retrieve multiple experiments for side-by-side comparison."""
        experiments = []
        for eid in experiment_ids:
            exp = self.get_experiment(eid)
            if exp:
                experiments.append(exp)
        return experiments

    def export_csv(self, path: str | Path) -> Path:
        """Export all experiments to a CSV file."""
        import csv

        experiments = self.list_experiments()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "experiment_id", "created_at", "dataset_name", "task_type",
            "target", "features", "model", "hyperparameters", "metrics",
            "notes",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for exp in experiments:
                writer.writerow({
                    "experiment_id": exp.experiment_id,
                    "created_at": exp.created_at,
                    "dataset_name": exp.dataset_name,
                    "task_type": exp.task_type,
                    "target": exp.target,
                    "features": json.dumps(exp.features),
                    "model": exp.model,
                    "hyperparameters": json.dumps(exp.hyperparameters),
                    "metrics": json.dumps(exp.metrics),
                    "notes": exp.notes,
                })

        return path

    def export_json(self, path: str | Path) -> Path:
        """Export all experiments to a JSON file."""
        experiments = self.list_experiments()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [exp.to_dict() for exp in experiments]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return path

    def count(self, task_type: str | None = None) -> int:
        """Count total experiments, optionally filtered by task type."""
        if task_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE task_type = ?", (task_type,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        return row[0]  # type: ignore[index]

    def clear(self) -> int:
        """Delete all experiments. Returns the number deleted."""
        cursor = self._conn.execute("DELETE FROM experiments")
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Internal ────────────────────────────────────────────────────

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                target TEXT NOT NULL,
                features TEXT NOT NULL,
                preprocessing_steps TEXT NOT NULL,
                model TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                metrics TEXT NOT NULL,
                generated_code TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            experiment_id=row["experiment_id"],
            created_at=row["created_at"],
            dataset_name=row["dataset_name"],
            task_type=row["task_type"],
            target=row["target"],
            features=json.loads(row["features"]),
            preprocessing_steps=json.loads(row["preprocessing_steps"]),
            model=row["model"],
            hyperparameters=json.loads(row["hyperparameters"]),
            metrics=json.loads(row["metrics"]),
            generated_code=row["generated_code"],
            notes=row["notes"],
        )
