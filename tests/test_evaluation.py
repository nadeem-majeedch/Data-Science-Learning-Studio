"""
Tests for utils.evaluation and utils.model_comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.pipeline import Pipeline

from utils.evaluation import (
    ClassificationMetrics,
    CrossValResult,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
    fig_confusion_matrix,
    fig_cross_val_bars,
    fig_residual_plot,
    fig_roc_curve,
    run_cross_validation,
)
from utils.model_comparison import (
    ComparisonResult,
    compare_classifiers,
    compare_regressors,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def iris_df():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = 3 * x1 + 2 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


# ── Classification metrics ──────────────────────────────────────────


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        m = compute_classification_metrics(y, y)
        assert m.accuracy == 1.0
        assert m.f1 == 1.0

    def test_imperfect(self):
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        m = compute_classification_metrics(y_test, y_pred)
        assert m.accuracy == 0.75
        assert m.confusion.shape == (2, 2)

    def test_with_probabilities(self):
        y_test = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.3])
        m = compute_classification_metrics(y_test, y_pred, y_prob)
        assert m.roc_auc is not None
        assert 0 <= m.roc_auc <= 1

    def test_multiclass(self):
        y_test = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 0])
        m = compute_classification_metrics(y_test, y_pred)
        assert m.accuracy > 0
        assert len(m.classes) == 3


# ── Regression metrics ──────────────────────────────────────────────


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_regression_metrics(y, y)
        assert m.r2 == 1.0
        assert m.mae == 0.0
        assert m.rmse == 0.0

    def test_imperfect(self):
        y_test = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.0])
        m = compute_regression_metrics(y_test, y_pred)
        assert m.r2 < 1.0
        assert m.mae > 0
        assert len(m.residuals) == 3


# ── Cross-validation ────────────────────────────────────────────────


class TestCrossValidation:
    def test_returns_result(self, iris_df):
        from sklearn.linear_model import LogisticRegression
        from utils.model_training import build_preprocessor, detect_feature_types

        target = "target"
        num_cols, cat_cols = detect_feature_types(iris_df, target)
        pp = build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("preprocessor", pp), ("classifier", LogisticRegression(max_iter=1000))])

        X = iris_df.drop(columns=[target])
        y = iris_df[target]
        result = run_cross_validation(pipe, X, y, task="classification", cv=3)
        assert isinstance(result, CrossValResult)
        assert "accuracy" in result.means
        assert 0 <= result.means["accuracy"] <= 1


# ── Plotly figures ──────────────────────────────────────────────────


class TestFigures:
    def test_confusion_matrix(self):
        cm = np.array([[10, 2], [3, 15]])
        fig = fig_confusion_matrix(cm, np.array([0, 1]))
        assert isinstance(fig, go.Figure)

    def test_roc_curve(self):
        y_test = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        fig = fig_roc_curve(y_test, y_prob)
        assert isinstance(fig, go.Figure)

    def test_residual_plot(self):
        y_test = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 1.8, 3.3, 3.7])
        fig = fig_residual_plot(y_test, y_pred)
        assert isinstance(fig, go.Figure)

    def test_cross_val_bars(self):
        fig = fig_cross_val_bars({"acc": 0.9, "f1": 0.85}, {"acc": 0.02, "f1": 0.03})
        assert isinstance(fig, go.Figure)


# ── Model comparison ────────────────────────────────────────────────


class TestClassificationComparison:
    def test_compare_all(self, iris_df):
        result = compare_classifiers(iris_df, "target")
        assert isinstance(result, ComparisonResult)
        assert result.task == "classification"
        assert len(result.rows) == 7
        assert "Accuracy" in result.table.columns
        assert "F1" in result.table.columns

    def test_compare_subset(self, iris_df):
        result = compare_classifiers(
            iris_df, "target",
            model_keys=["logistic_regression", "random_forest"],
        )
        assert len(result.rows) == 2

    def test_table_sorted_by_f1(self, iris_df):
        result = compare_classifiers(iris_df, "target")
        f1_values = result.table["F1"].tolist()
        assert f1_values == sorted(f1_values, reverse=True)

    def test_code_generated(self, iris_df):
        result = compare_classifiers(iris_df, "target", model_keys=["logistic_regression"])
        assert result.code
        assert "fit" in result.code


class TestRegressionComparison:
    def test_compare_all(self, reg_df):
        result = compare_regressors(reg_df, "target")
        assert isinstance(result, ComparisonResult)
        assert result.task == "regression"
        assert len(result.rows) == 7
        assert "R²" in result.table.columns
        assert "MAE" in result.table.columns

    def test_compare_subset(self, reg_df):
        result = compare_regressors(
            reg_df, "target",
            model_keys=["linear", "ridge"],
        )
        assert len(result.rows) == 2

    def test_table_sorted_by_r2(self, reg_df):
        result = compare_regressors(reg_df, "target")
        r2_values = result.table["R²"].tolist()
        assert r2_values == sorted(r2_values, reverse=True)
