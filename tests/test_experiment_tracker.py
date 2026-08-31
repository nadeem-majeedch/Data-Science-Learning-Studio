"""
Tests for utils/experiment_tracker.py
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from utils.experiment_tracker import Experiment, ExperimentTracker


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path):
    """Create a temporary tracker for testing."""
    db_path = tmp_path / "test_experiments.db"
    t = ExperimentTracker(db_path=str(db_path))
    yield t
    t.close()


@pytest.fixture
def sample_experiment():
    """Default experiment kwargs for saving."""
    return {
        "dataset_name": "iris.csv",
        "task_type": "classification",
        "target": "species",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "model": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 100, "max_depth": 5},
        "metrics": {"accuracy": 0.96, "f1": 0.95, "precision": 0.97},
        "preprocessing_steps": ["StandardScaler", "OneHotEncoder"],
        "generated_code": "from sklearn.ensemble import RandomForestClassifier\n...",
        "notes": "Baseline experiment",
    }


@pytest.fixture
def populated_tracker(tracker, sample_experiment):
    """Tracker with several saved experiments."""
    tracker.save_experiment(**sample_experiment)
    tracker.save_experiment(
        dataset_name="iris.csv",
        task_type="classification",
        target="species",
        features=["sepal_length", "sepal_width"],
        model="LogisticRegression",
        hyperparameters={"C": 1.0},
        metrics={"accuracy": 0.92, "f1": 0.91},
    )
    tracker.save_experiment(
        dataset_name="housing.csv",
        task_type="regression",
        target="price",
        features=["area", "bedrooms", "age"],
        model="Ridge",
        hyperparameters={"alpha": 1.0},
        metrics={"r2": 0.85, "mae": 25000},
    )
    return tracker


# ── CRUD operations ─────────────────────────────────────────────────

class TestSaveExperiment:
    def test_returns_id(self, tracker, sample_experiment):
        exp_id = tracker.save_experiment(**sample_experiment)
        assert isinstance(exp_id, int)
        assert exp_id > 0

    def test_incrementing_ids(self, tracker, sample_experiment):
        id1 = tracker.save_experiment(**sample_experiment)
        id2 = tracker.save_experiment(**sample_experiment)
        assert id2 == id1 + 1

    def test_count_after_save(self, tracker, sample_experiment):
        tracker.save_experiment(**sample_experiment)
        assert tracker.count() == 1

    def test_save_with_defaults(self, tracker):
        exp_id = tracker.save_experiment(
            dataset_name="test.csv",
            task_type="classification",
            target="y",
            features=["x1", "x2"],
            model="DummyClassifier",
        )
        assert exp_id > 0


class TestGetExperiment:
    def test_get_existing(self, populated_tracker):
        exp = populated_tracker.get_experiment(1)
        assert exp is not None
        assert exp.experiment_id == 1
        assert exp.dataset_name == "iris.csv"
        assert exp.model == "RandomForestClassifier"

    def test_get_nonexistent(self, populated_tracker):
        exp = populated_tracker.get_experiment(9999)
        assert exp is None

    def test_fields_populated(self, populated_tracker, sample_experiment):
        exp = populated_tracker.get_experiment(1)
        assert exp.target == sample_experiment["target"]
        assert exp.features == sample_experiment["features"]
        assert exp.hyperparameters == sample_experiment["hyperparameters"]
        assert exp.metrics == sample_experiment["metrics"]
        assert exp.preprocessing_steps == sample_experiment["preprocessing_steps"]


class TestDeleteExperiment:
    def test_delete_existing(self, populated_tracker):
        assert populated_tracker.delete_experiment(1) is True
        assert populated_tracker.count() == 2

    def test_delete_nonexistent(self, populated_tracker):
        assert populated_tracker.delete_experiment(9999) is False

    def test_delete_reduces_count(self, populated_tracker):
        initial = populated_tracker.count()
        populated_tracker.delete_experiment(2)
        assert populated_tracker.count() == initial - 1


# ── Listing and filtering ───────────────────────────────────────────

class TestListExperiments:
    def test_list_all(self, populated_tracker):
        experiments = populated_tracker.list_experiments()
        assert len(experiments) == 3

    def test_list_by_task_type(self, populated_tracker):
        cls_exps = populated_tracker.list_experiments(task_type="classification")
        assert len(cls_exps) == 2
        reg_exps = populated_tracker.list_experiments(task_type="regression")
        assert len(reg_exps) == 1

    def test_list_by_dataset(self, populated_tracker):
        iris_exps = populated_tracker.list_experiments(dataset_name="iris.csv")
        assert len(iris_exps) == 2

    def test_list_by_both_filters(self, populated_tracker):
        result = populated_tracker.list_experiments(task_type="classification", dataset_name="housing.csv")
        assert len(result) == 0

    def test_list_returns_experiment_objects(self, populated_tracker):
        experiments = populated_tracker.list_experiments()
        for exp in experiments:
            assert isinstance(exp, Experiment)


# ── Comparison ──────────────────────────────────────────────────────

class TestCompareExperiments:
    def test_compare_two(self, populated_tracker):
        results = populated_tracker.compare_experiments([1, 2])
        assert len(results) == 2
        assert results[0].model == "RandomForestClassifier"
        assert results[1].model == "LogisticRegression"

    def test_compare_with_missing(self, populated_tracker):
        results = populated_tracker.compare_experiments([1, 9999, 2])
        assert len(results) == 2

    def test_compare_empty_list(self, populated_tracker):
        results = populated_tracker.compare_experiments([])
        assert len(results) == 0


# ── Export ──────────────────────────────────────────────────────────

class TestExport:
    def test_export_csv(self, populated_tracker, tmp_path):
        csv_path = tmp_path / "experiments.csv"
        result = populated_tracker.export_csv(csv_path)
        assert result.exists()

        content = csv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "experiment_id" in lines[0]
        # Default order is created_at DESC — all 3 models should appear somewhere
        body = " ".join(lines[1:])
        assert "RandomForestClassifier" in body
        assert "LogisticRegression" in body
        assert "Ridge" in body

    def test_export_json(self, populated_tracker, tmp_path):
        json_path = tmp_path / "experiments.json"
        result = populated_tracker.export_json(json_path)
        assert result.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data) == 3
        models = {d["model"] for d in data}
        assert "RandomForestClassifier" in models
        assert "LogisticRegression" in models
        assert "Ridge" in models

    def test_export_csv_creates_directory(self, populated_tracker, tmp_path):
        nested = tmp_path / "reports" / "exports" / "experiments.csv"
        populated_tracker.export_csv(nested)
        assert nested.exists()

    def test_export_json_creates_directory(self, populated_tracker, tmp_path):
        nested = tmp_path / "reports" / "exports" / "experiments.json"
        populated_tracker.export_json(nested)
        assert nested.exists()


# ── Count and clear ─────────────────────────────────────────────────

class TestCountAndClear:
    def test_count_all(self, populated_tracker):
        assert populated_tracker.count() == 3

    def test_count_by_type(self, populated_tracker):
        assert populated_tracker.count(task_type="classification") == 2
        assert populated_tracker.count(task_type="regression") == 1

    def test_count_empty(self, tracker):
        assert tracker.count() == 0

    def test_clear(self, populated_tracker):
        deleted = populated_tracker.clear()
        assert deleted == 3
        assert populated_tracker.count() == 0

    def test_count_after_save(self, tracker, sample_experiment):
        tracker.save_experiment(**sample_experiment)
        tracker.save_experiment(**sample_experiment)
        assert tracker.count() == 2


# ── Data integrity ──────────────────────────────────────────────────

class TestDataIntegrity:
    def test_list_features_as_list(self, populated_tracker):
        exp = populated_tracker.get_experiment(1)
        assert isinstance(exp.features, list)
        assert len(exp.features) == 4

    def test_metrics_as_dict(self, populated_tracker):
        exp = populated_tracker.get_experiment(1)
        assert isinstance(exp.metrics, dict)
        assert "accuracy" in exp.metrics

    def test_hyperparameters_as_dict(self, populated_tracker):
        exp = populated_tracker.get_experiment(1)
        assert isinstance(exp.hyperparameters, dict)
        assert exp.hyperparameters["n_estimators"] == 100

    def test_created_at_not_empty(self, populated_tracker):
        exp = populated_tracker.get_experiment(1)
        assert exp.created_at  # should be a non-empty ISO timestamp


# ── Experiment data class ───────────────────────────────────────────

class TestExperimentDataclass:
    def test_to_dict(self):
        exp = Experiment(
            experiment_id=1,
            dataset_name="test.csv",
            task_type="classification",
            target="y",
            features=["x1"],
            model="Dummy",
            metrics={"acc": 0.5},
        )
        d = exp.to_dict()
        assert d["experiment_id"] == 1
        assert d["features"] == ["x1"]
        assert d["metrics"] == {"acc": 0.5}

    def test_default_values(self):
        exp = Experiment()
        assert exp.experiment_id == 0
        assert exp.features == []
        assert exp.metrics == {}
        assert exp.hyperparameters == {}


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_features(self, tracker):
        exp_id = tracker.save_experiment(
            dataset_name="test.csv",
            task_type="classification",
            target="y",
            features=[],
            model="Dummy",
            metrics={},
        )
        exp = tracker.get_experiment(exp_id)
        assert exp.features == []

    def test_special_characters_in_notes(self, tracker):
        notes = "Special chars: <>&\"' and unicode: café ñ"
        exp_id = tracker.save_experiment(
            dataset_name="test.csv",
            task_type="classification",
            target="y",
            features=["x"],
            model="Dummy",
            notes=notes,
        )
        exp = tracker.get_experiment(exp_id)
        assert exp.notes == notes

    def test_large_metrics_dict(self, tracker):
        big_metrics = {f"metric_{i}": float(i) / 100 for i in range(50)}
        exp_id = tracker.save_experiment(
            dataset_name="test.csv",
            task_type="classification",
            target="y",
            features=["x"],
            model="Dummy",
            metrics=big_metrics,
        )
        exp = tracker.get_experiment(exp_id)
        assert len(exp.metrics) == 50

    def test_persistence_across_instances(self, tmp_path):
        db_path = tmp_path / "persist.db"
        t1 = ExperimentTracker(db_path=str(db_path))
        t1.save_experiment(
            dataset_name="test.csv", task_type="classification",
            target="y", features=["x"], model="Dummy",
        )
        t1.close()

        t2 = ExperimentTracker(db_path=str(db_path))
        assert t2.count() == 1
        exp = t2.get_experiment(1)
        assert exp is not None
        assert exp.model == "Dummy"
        t2.close()
