"""
Tests for utils.regression_models and utils.regression_training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils.regression_models import (
    REGRESSORS,
    RegressorInfo,
    get_regressor,
    get_regressor_names,
    key_from_name,
)
from utils.regression_training import (
    RegressionResult,
    build_preprocessor,
    detect_feature_types,
    extract_feature_importance,
    train_regressor,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def boston_like() -> pd.DataFrame:
    """A synthetic regression dataset."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    x3 = rng.uniform(0, 10, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = 3 * x1 + 2 * x2 + 0.5 * x3 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "target": y})


@pytest.fixture
def messy_reg() -> pd.DataFrame:
    """Regression dataset with missing values."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "cat": rng.choice(["X", "Y"], n),
        "y": rng.normal(10, 3, n),
    })
    df.loc[rng.choice(n, 5), "a"] = np.nan
    df.loc[rng.choice(n, 3), "cat"] = np.nan
    return df


# ── Registry tests ──────────────────────────────────────────────────


class TestRegressorRegistry:
    def test_all_7_registered(self):
        assert len(REGRESSORS) == 7

    def test_each_has_sklearn_class(self):
        for info in REGRESSORS.values():
            assert hasattr(info.sklearn_class, "fit")
            assert hasattr(info.sklearn_class, "predict")

    def test_get_regressor(self):
        info = get_regressor("linear")
        assert isinstance(info, RegressorInfo)
        assert info.name == "Linear Regression"

    def test_get_regressor_names(self):
        names = get_regressor_names()
        assert len(names) == 7
        assert "Ridge Regression" in names

    def test_key_from_name(self):
        assert key_from_name("Random Forest Regressor") == "random_forest_reg"

    def test_key_from_name_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            key_from_name("Nonexistent")

    def test_all_have_descriptions(self):
        for info in REGRESSORS.values():
            assert info.description
            assert info.advantages
            assert info.limitations
            assert info.when_to_use


# ── Preprocessing tests ─────────────────────────────────────────────


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        pp = build_preprocessor(["a", "b"], ["c"])
        assert isinstance(pp, ColumnTransformer)

    def test_no_scaler(self):
        pp = build_preprocessor(["a"], [], scaler=None)
        assert isinstance(pp, ColumnTransformer)

    def test_minmax(self):
        pp = build_preprocessor(["a"], [], scaler="minmax")
        assert isinstance(pp, ColumnTransformer)


class TestDetectFeatureTypes:
    def test_basic(self, boston_like):
        num, cat = detect_feature_types(boston_like, "target")
        assert "x1" in num
        assert "cat" in cat
        assert "target" not in num
        assert "target" not in cat


# ── Training tests ──────────────────────────────────────────────────


class TestTrainRegressor:
    def test_linear(self, boston_like):
        result = train_regressor(boston_like, "target", "linear")
        assert isinstance(result, RegressionResult)
        assert result.r2 > 0.5

    def test_ridge(self, boston_like):
        result = train_regressor(boston_like, "target", "ridge")
        assert isinstance(result.pipeline, Pipeline)
        assert result.r2 > 0.5

    def test_lasso(self, boston_like):
        result = train_regressor(boston_like, "target", "lasso")
        assert result.mae >= 0

    def test_decision_tree(self, boston_like):
        result = train_regressor(boston_like, "target", "decision_tree_reg")
        assert result.r2 > 0

    def test_random_forest(self, boston_like):
        result = train_regressor(boston_like, "target", "random_forest_reg")
        assert result.rmse > 0

    def test_gradient_boosting(self, boston_like):
        result = train_regressor(boston_like, "target", "gradient_boosting_reg")
        assert result.r2 > 0.5

    def test_knn(self, boston_like):
        result = train_regressor(boston_like, "target", "knn_reg")
        assert 0 <= result.mse

    def test_messy_data(self, messy_reg):
        result = train_regressor(messy_reg, "y", "random_forest_reg")
        assert isinstance(result, RegressionResult)

    def test_custom_params(self, boston_like):
        result = train_regressor(
            boston_like, "target", "random_forest_reg",
            model_params={"n_estimators": 10},
        )
        assert isinstance(result.pipeline, Pipeline)

    def test_no_scaler(self, boston_like):
        result = train_regressor(boston_like, "target", "ridge", scaler=None)
        assert result.r2 > 0.5

    def test_minmax(self, boston_like):
        result = train_regressor(boston_like, "target", "linear", scaler="minmax")
        assert result.r2 > 0.5


# ── Feature importance ──────────────────────────────────────────────


class TestFeatureImportance:
    def test_linear(self, boston_like):
        result = train_regressor(boston_like, "target", "linear")
        imp = extract_feature_importance(result)
        assert imp is not None
        assert "importance" in imp.columns

    def test_random_forest(self, boston_like):
        result = train_regressor(boston_like, "target", "random_forest_reg")
        imp = extract_feature_importance(result)
        assert imp is not None
        assert len(imp) > 0

    def test_knn_returns_none(self, boston_like):
        result = train_regressor(boston_like, "target", "knn_reg")
        imp = extract_feature_importance(result)
        assert imp is None


# ── Code generation ─────────────────────────────────────────────────


class TestCodeGeneration:
    def test_code_is_nonempty(self, boston_like):
        result = train_regressor(boston_like, "target", "linear")
        assert result.code
        assert "fit" in result.code
        assert "r2_score" in result.code

    def test_code_mentions_target(self, boston_like):
        result = train_regressor(boston_like, "target", "ridge")
        assert "target" in result.code
