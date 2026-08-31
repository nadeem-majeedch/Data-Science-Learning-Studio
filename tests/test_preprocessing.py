"""
Tests for utils.preprocessing — preprocessing functions and Pipeline builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from utils.preprocessing import (
    PreprocessingStep,
    build_sklearn_pipeline,
    compare_before_after,
    detect_duplicates,
    detect_outliers_iqr,
    handle_missing_values,
    label_encode,
    one_hot_encode,
    remove_duplicates,
    remove_outliers,
    scale_features,
    split_data,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """DataFrame with missing values, duplicates, and mixed types."""
    return pd.DataFrame({
        "age": [25, 30, None, 35, 30, 25],
        "salary": [50000, 60000, 70000, None, 60000, 50000],
        "city": ["NYC", "LA", "NYC", "SF", "LA", "NYC"],
        "grade": ["A", "B", "A", "C", "B", "A"],
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Clean numerical DataFrame for scaling tests."""
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "label": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })


# ── Missing values ──────────────────────────────────────────────────


class TestHandleMissingValues:
    def test_drop_rows(self, messy_df):
        df, step = handle_missing_values(messy_df, "drop_rows")
        assert df.isna().sum().sum() == 0
        assert step.name == "Missing values (drop_rows)"
        assert isinstance(step.code, str)

    def test_drop_columns(self, messy_df):
        df, step = handle_missing_values(messy_df, "drop_columns", columns=["age"])
        assert "age" not in df.columns
        assert "age" in step.columns_affected

    def test_mean(self, messy_df):
        df, step = handle_missing_values(messy_df, "mean")
        assert df["age"].isna().sum() == 0
        assert "mean" in step.description

    def test_median(self, messy_df):
        df, step = handle_missing_values(messy_df, "median")
        assert df["salary"].isna().sum() == 0

    def test_mode(self, messy_df):
        df, step = handle_missing_values(messy_df, "mode")
        assert "mode" in step.description

    def test_constant(self, messy_df):
        df, step = handle_missing_values(messy_df, "constant", fill_value=-1)
        assert df["age"].isna().sum() == 0

    def test_unknown_strategy_raises(self, messy_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing_values(messy_df, "invalid")


# ── Duplicates ──────────────────────────────────────────────────────


class TestDuplicates:
    def test_detect(self, messy_df):
        n, code = detect_duplicates(messy_df)
        # Rows 1&4 and rows 0&5 are duplicates (keep='first')
        assert n == 2
        assert isinstance(code, str)

    def test_remove(self, messy_df):
        df, step = remove_duplicates(messy_df)
        assert len(df) == 4  # 6 rows - 2 duplicates
        assert step.name == "Remove duplicates"

    def test_no_duplicates(self, clean_df):
        n, _ = detect_duplicates(clean_df)
        assert n == 0


# ── Encoding ────────────────────────────────────────────────────────


class TestEncoding:
    def test_one_hot(self, messy_df):
        df, step = one_hot_encode(messy_df, ["city"])
        assert "city" not in df.columns
        assert any("city_" in c for c in df.columns)
        assert step.name == "One-hot encoding"

    def test_one_hot_no_drop(self, messy_df):
        df, step = one_hot_encode(messy_df, ["city"], drop_first=False)
        n_city_cols = sum(1 for c in df.columns if c.startswith("city_"))
        assert n_city_cols == messy_df["city"].nunique()

    def test_label_encode(self, messy_df):
        df, step, mappings = label_encode(messy_df, ["grade"])
        assert df["grade"].dtype in [np.float64, np.int64, float, int]
        assert "grade" in mappings
        assert isinstance(mappings["grade"], dict)


# ── Scaling ─────────────────────────────────────────────────────────


class TestScaling:
    def test_standard(self, clean_df):
        df, step = scale_features(clean_df, ["x", "y"], method="standard")
        assert abs(df["x"].mean()) < 0.1
        assert abs(df["x"].std() - 1.0) < 0.1

    def test_minmax(self, clean_df):
        df, step = scale_features(clean_df, ["x", "y"], method="minmax")
        assert df["x"].min() >= 0
        assert df["x"].max() <= 1

    def test_robust(self, clean_df):
        df, step = scale_features(clean_df, ["x", "y"], method="robust")
        assert isinstance(step.code, str)

    def test_unknown_raises(self, clean_df):
        with pytest.raises(ValueError, match="Unknown scaler"):
            scale_features(clean_df, ["x"], method="unknown")


# ── Outliers ────────────────────────────────────────────────────────


class TestOutliers:
    def test_detect(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        mask, summary = detect_outliers_iqr(df, "x")
        assert summary["n_outliers"] >= 1
        assert summary["lower_bound"] < summary["upper_bound"]

    def test_remove(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        df_out, step = remove_outliers(df, ["x"])
        assert len(df_out) < len(df)
        assert "Remove outliers" in step.name


# ── Train/test split ────────────────────────────────────────────────


class TestSplitData:
    def test_basic_split(self, messy_df):
        X_train, X_test, y_train, y_test, step = split_data(
            messy_df, target="age", test_size=0.5, random_state=42
        )
        assert len(X_train) + len(X_test) == len(messy_df)
        assert "age" not in X_train.columns
        assert y_train is not None

    def test_split_no_target(self, clean_df):
        X_train, X_test, y_train, y_test, step = split_data(
            clean_df.drop(columns=["label"]), target=None, test_size=0.3
        )
        assert y_train is None
        assert y_test is None
        assert len(X_train) + len(X_test) == len(clean_df)

    def test_stratified(self, clean_df):
        X_train, X_test, y_train, y_test, step = split_data(
            clean_df, target="label", test_size=0.4, stratify=True
        )
        assert len(X_train) + len(X_test) == len(clean_df)
        assert "stratification" in step.description


# ── Before / After ──────────────────────────────────────────────────


class TestCompareBeforeAfter:
    def test_returns_dataframe(self):
        before = pd.DataFrame({"a": [1, 2, 3]})
        after = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        result = compare_before_after(before, after)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["Rows"]) == [3, 2]


# ── sklearn Pipeline builder ────────────────────────────────────────


class TestBuildSklearnPipeline:
    def test_returns_pipeline(self):
        pipe = build_sklearn_pipeline(
            numeric_columns=["x", "y"],
            categorical_columns=["label"],
        )
        assert isinstance(pipe, Pipeline)

    def test_with_scaler(self):
        pipe = build_sklearn_pipeline(
            numeric_columns=["x"],
            categorical_columns=[],
            scaler="standard",
        )
        assert isinstance(pipe, Pipeline)

    def test_label_encoder(self):
        pipe = build_sklearn_pipeline(
            numeric_columns=[],
            categorical_columns=["cat"],
            cat_encoder="ordinal",
        )
        assert isinstance(pipe, Pipeline)

    def test_fit_transform(self, clean_df):
        pipe = build_sklearn_pipeline(
            numeric_columns=["x", "y"],
            categorical_columns=["label"],
        )
        result = pipe.fit_transform(clean_df)
        assert result.shape[0] == len(clean_df)
