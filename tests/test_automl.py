"""
Tests for utils/automl.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine, make_regression

from utils.automl import (
    AutoMLReport,
    detect_task_type,
    explain_best_model,
    run_automl,
    validate_dataset,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def iris_df():
    """Classic Iris classification dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.fixture
def regression_df():
    """Synthetic regression dataset."""
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    df["target"] = y
    return df


@pytest.fixture
def messy_df():
    """Dataset with missing values and duplicates."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "x1": rng.randn(100),
        "x2": rng.randn(100),
        "cat": rng.choice(["A", "B", "C"], 100),
        "label": rng.choice([0, 1], 100),
    })
    df.loc[5, "x1"] = np.nan
    df.loc[10, "x2"] = np.nan
    # Add some duplicates
    dup = df.iloc[:5].copy()
    df = pd.concat([df, dup], ignore_index=True)
    return df


@pytest.fixture
def wine_df():
    """Wine classification dataset."""
    data = load_wine()
    df = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(data.data.shape[1])])
    df["target"] = data.target
    return df


# ── Task detection ──────────────────────────────────────────────────

class TestTaskDetection:
    def test_detects_classification_from_integers(self, iris_df):
        task, reason = detect_task_type(iris_df, "target")
        assert task == "classification"
        assert "unique" in reason.lower() or "class" in reason.lower()

    def test_detects_regression_from_continuous(self, regression_df):
        task, reason = detect_task_type(regression_df, "target")
        assert task == "regression"

    def test_detects_string_classes(self):
        df = pd.DataFrame({"x": range(20), "y": ["cat", "dog"] * 10})
        task, _ = detect_task_type(df, "y")
        assert task == "classification"

    def test_detects_boolean(self):
        df = pd.DataFrame({"x": range(20), "y": [True, False] * 10})
        task, _ = detect_task_type(df, "y")
        assert task == "classification"


# ── Validation ──────────────────────────────────────────────────────

class TestValidation:
    def test_valid_dataset(self, iris_df):
        result = validate_dataset(iris_df, "target")
        assert result.valid
        assert result.n_rows == 150
        assert result.n_numeric == 4  # 4 features (excl target)

    def test_missing_target(self, iris_df):
        result = validate_dataset(iris_df, "nonexistent")
        assert not result.valid
        assert any("not found" in e for e in result.errors)

    def test_warnings_for_large_dataset(self):
        df = pd.DataFrame({
            "x": range(600_000),
            "y": range(600_000),
        })
        result = validate_dataset(df, "y")
        assert any("rows" in w.lower() for w in result.warnings)

    def test_warnings_for_duplicates(self, messy_df):
        # Check that duplicates are counted correctly
        result = validate_dataset(messy_df, "label")
        assert result.n_duplicates > 0

    def test_no_features_error(self):
        df = pd.DataFrame({"target": [1, 2, 3]})
        result = validate_dataset(df, "target")
        # With only target and no features, n_numeric and n_categorical should be 0
        assert result.n_numeric == 0 and result.n_categorical == 0


# ── AutoML run — classification ─────────────────────────────────────

class TestAutoMLClassification:
    def test_runs_on_iris(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=3,
        )
        assert isinstance(report, AutoMLReport)
        assert report.task == "classification"
        assert len(report.model_results) == 3

    def test_best_model_exists(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=3,
        )
        assert report.best_model is not None
        assert report.best_model.name
        assert "F1" in report.best_model.metrics

    def test_comparison_table(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=3,
        )
        assert "Model" in report.comparison_table.columns
        assert "F1" in report.comparison_table.columns
        assert len(report.comparison_table) == 3

    def test_code_generated(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=2,
        )
        assert isinstance(report.code, str)
        assert "train_test_split" in report.code
        assert len(report.code) > 200

    def test_timing_recorded(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=2,
        )
        assert report.total_time > 0
        for mr in report.model_results:
            assert mr.train_time >= 0

    def test_progress_callback_called(self, iris_df):
        calls = []
        run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=2,
            progress_callback=lambda c, t, m: calls.append((c, t, m)),
        )
        assert len(calls) > 0

    def test_multiclass(self, wine_df):
        report = run_automl(
            wine_df, target="target", task="classification",
            dataset_name="wine", max_models=3,
        )
        assert len(report.model_results) == 3
        assert all("Accuracy" in mr.metrics for mr in report.model_results)

    def test_max_models_cap(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=2,
        )
        assert len(report.model_results) <= 2


# ── AutoML run — regression ─────────────────────────────────────────

class TestAutoMLRegression:
    def test_runs_on_regression(self, regression_df):
        report = run_automl(
            regression_df, target="target", task="regression",
            dataset_name="synthetic", max_models=3,
        )
        assert report.task == "regression"
        assert len(report.model_results) == 3

    def test_regression_metrics(self, regression_df):
        report = run_automl(
            regression_df, target="target", task="regression",
            dataset_name="synthetic", max_models=3,
        )
        for mr in report.model_results:
            assert "R²" in mr.metrics
            assert "MAE" in mr.metrics
            assert "RMSE" in mr.metrics

    def test_best_model_by_r2(self, regression_df):
        report = run_automl(
            regression_df, target="target", task="regression",
            dataset_name="synthetic", max_models=3,
        )
        assert report.primary_metric == "R²"


# ── Explanation ─────────────────────────────────────────────────────

class TestExplanation:
    def test_explanation_returns_string(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=3,
        )
        explanation = explain_best_model(report)
        assert isinstance(explanation, str)
        assert "Best Model" in explanation
        assert report.best_model.name in explanation

    def test_explanation_mentions_caveats(self, regression_df):
        report = run_automl(
            regression_df, target="target", task="regression",
            dataset_name="synthetic", max_models=3,
        )
        explanation = explain_best_model(report)
        assert "depends" in explanation.lower() or "⚠️" in explanation


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_target_raises(self, iris_df):
        with pytest.raises(ValueError, match="not found"):
            run_automl(
                iris_df, target="nonexistent", task="classification",
                dataset_name="iris", max_models=2,
            )

    def test_two_model_min(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="iris", max_models=2,
        )
        assert len(report.model_results) >= 2

    def test_report_has_dataset_name(self, iris_df):
        report = run_automl(
            iris_df, target="target", task="classification",
            dataset_name="my_custom_iris", max_models=2,
        )
        assert report.dataset_name == "my_custom_iris"
