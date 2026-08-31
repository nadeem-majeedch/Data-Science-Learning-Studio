"""
Tests for utils.models and utils.model_training — model registry and training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from utils.models import CLASSIFIERS, ModelInfo, get_classifier, get_classifier_names, key_from_name
from sklearn.compose import ColumnTransformer

from utils.model_training import (
    TrainResult,
    build_preprocessor,
    detect_feature_types,
    extract_feature_importance,
    train_classifier,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Iris-like dataset for testing."""
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


@pytest.fixture
def binary_df() -> pd.DataFrame:
    """Simple binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "feature_c": rng.uniform(0, 10, n),
        "cat_feature": rng.choice(["X", "Y", "Z"], n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Dataset with missing values and mixed types."""
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "label": rng.integers(0, 3, n),
    })
    # Inject missing values
    df.loc[rng.choice(n, 5), "num1"] = np.nan
    df.loc[rng.choice(n, 3), "cat1"] = np.nan
    return df


# ── Model registry tests ────────────────────────────────────────────


class TestModelRegistry:
    def test_all_classifiers_registered(self):
        assert len(CLASSIFIERS) == 7

    def test_each_has_sklearn_class(self):
        for key, info in CLASSIFIERS.items():
            assert hasattr(info.sklearn_class, "fit")
            assert hasattr(info.sklearn_class, "predict")

    def test_get_classifier(self):
        info = get_classifier("logistic_regression")
        assert isinstance(info, ModelInfo)
        assert info.name == "Logistic Regression"

    def test_get_classifier_names(self):
        names = get_classifier_names()
        assert len(names) == 7
        assert "Random Forest" in names

    def test_key_from_name(self):
        assert key_from_name("Random Forest") == "random_forest"

    def test_key_from_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            key_from_name("Nonexistent Model")

    def test_all_have_descriptions(self):
        for info in CLASSIFIERS.values():
            assert info.description
            assert info.advantages
            assert info.limitations
            assert info.when_to_use


# ── Preprocessing builder tests ─────────────────────────────────────


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        pp = build_preprocessor(["a", "b"], ["c"])
        assert isinstance(pp, ColumnTransformer)

    def test_no_scaler(self):
        pp = build_preprocessor(["a"], [], scaler=None)
        assert isinstance(pp, ColumnTransformer)

    def test_empty_columns(self):
        pp = build_preprocessor([], [])
        assert isinstance(pp, ColumnTransformer)


# ── Feature type detection ──────────────────────────────────────────


class TestDetectFeatureTypes:
    def test_basic(self, binary_df):
        num, cat = detect_feature_types(binary_df, "target")
        assert "feature_a" in num
        assert "cat_feature" in cat
        assert "target" not in num
        assert "target" not in cat

    def test_all_numeric(self, iris_df):
        num, cat = detect_feature_types(iris_df, "target")
        assert len(cat) == 0
        assert len(num) == 4


# ── Training tests ──────────────────────────────────────────────────


class TestTrainClassifier:
    def test_logistic_regression(self, binary_df):
        result = train_classifier(binary_df, "target", "logistic_regression")
        assert isinstance(result, TrainResult)
        assert 0 <= result.accuracy <= 1
        assert result.model_name == "Logistic Regression"

    def test_random_forest(self, binary_df):
        result = train_classifier(binary_df, "target", "random_forest")
        assert isinstance(result.pipeline, Pipeline)

    def test_decision_tree(self, iris_df):
        result = train_classifier(iris_df, "target", "decision_tree")
        assert result.accuracy > 0.5

    def test_naive_bayes(self, iris_df):
        result = train_classifier(iris_df, "target", "naive_bayes")
        assert result.accuracy > 0.5

    def test_knn(self, iris_df):
        result = train_classifier(iris_df, "target", "knn")
        assert 0 <= result.f1 <= 1

    def test_svm(self, iris_df):
        result = train_classifier(iris_df, "target", "svm")
        assert result.report_text  # non-empty

    def test_gradient_boosting(self, iris_df):
        result = train_classifier(iris_df, "target", "gradient_boosting")
        assert result.accuracy > 0.5

    def test_multiclass_detection(self, iris_df):
        result = train_classifier(iris_df, "target", "logistic_regression")
        assert not result.is_binary

    def test_binary_detection(self, binary_df):
        result = train_classifier(binary_df, "target", "logistic_regression")
        assert result.is_binary

    def test_messy_data(self, messy_df):
        result = train_classifier(messy_df, "label", "random_forest")
        assert result.accuracy > 0

    def test_custom_params(self, iris_df):
        result = train_classifier(
            iris_df, "target", "random_forest",
            model_params={"n_estimators": 10},
        )
        assert isinstance(result.pipeline, Pipeline)

    def test_with_minmax(self, iris_df):
        result = train_classifier(iris_df, "target", "logistic_regression", scaler="minmax")
        assert result.accuracy > 0.5

    def test_no_scaler(self, iris_df):
        result = train_classifier(iris_df, "target", "random_forest", scaler=None)
        assert result.accuracy > 0.5


# ── Feature importance ──────────────────────────────────────────────


class TestFeatureImportance:
    def test_random_forest(self, binary_df):
        result = train_classifier(binary_df, "target", "random_forest")
        imp = extract_feature_importance(result)
        assert imp is not None
        assert "importance" in imp.columns
        assert len(imp) > 0

    def test_logistic_regression(self, binary_df):
        result = train_classifier(binary_df, "target", "logistic_regression")
        imp = extract_feature_importance(result)
        assert imp is not None

    def test_naive_bayes_returns_none(self, iris_df):
        result = train_classifier(iris_df, "target", "naive_bayes")
        imp = extract_feature_importance(result)
        assert imp is None  # GaussianNB doesn't have coef_ or feature_importances_


# ── Code generation ─────────────────────────────────────────────────


class TestCodeGeneration:
    def test_code_is_nonempty(self, iris_df):
        result = train_classifier(iris_df, "target", "logistic_regression")
        assert result.code
        assert "import" in result.code
        assert "fit" in result.code

    def test_code_mentions_target(self, iris_df):
        result = train_classifier(iris_df, "target", "random_forest")
        assert "target" in result.code
