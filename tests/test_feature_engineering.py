"""
Tests for utils.feature_engineering — transformation and selection functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.feature_engineering import (
    FeatureStep,
    apply_math_transform,
    bin_numerical,
    create_interaction,
    create_polynomial,
    extract_date_features,
    extract_text_features,
    get_feature_importance,
    variance_threshold_select,
    correlation_select,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "name": ["alice", "bob", "charlie", "diana", "eve",
                 "frank", "grace", "heidi", "ivan", "judy"],
        "date": pd.date_range("2024-01-01", periods=10, freq="MS"),
    })


# ── Math transforms ─────────────────────────────────────────────────


class TestMathTransform:
    def test_log(self, sample_df):
        df, step = apply_math_transform(sample_df, "price", "log")
        assert "price_log" in df.columns
        assert df["price_log"].notna().all()
        assert step.name == "Math transform (log)"

    def test_log_with_negatives(self):
        df = pd.DataFrame({"x": [-5, 0, 3, 7]})
        df_out, _ = apply_math_transform(df, "x", "log")
        assert "x_log" in df_out.columns
        assert df_out["x_log"].notna().all()

    def test_sqrt(self, sample_df):
        df, step = apply_math_transform(sample_df, "quantity", "sqrt")
        assert "quantity_sqrt" in df.columns
        assert df["quantity_sqrt"].iloc[0] == pytest.approx(1.0)

    def test_square(self, sample_df):
        df, step = apply_math_transform(sample_df, "price", "square")
        assert "price_square" in df.columns
        assert df["price_square"].iloc[0] == pytest.approx(100.0)

    def test_unknown_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_math_transform(sample_df, "price", "cubic")


# ── Binning ─────────────────────────────────────────────────────────


class TestBinning:
    def test_equal_width(self, sample_df):
        df, step = bin_numerical(sample_df, "price", n_bins=5, method="equal_width")
        assert "price_binned" in df.columns
        assert df["price_binned"].notna().all()

    def test_equal_freq(self, sample_df):
        df, step = bin_numerical(sample_df, "price", n_bins=5, method="equal_freq")
        assert "price_binned" in df.columns

    def test_fewer_bins(self, sample_df):
        df, _ = bin_numerical(sample_df, "price", n_bins=2)
        assert df["price_binned"].nunique() <= 3  # <= due to possible NaN edges


# ── Date features ───────────────────────────────────────────────────


class TestDateFeatures:
    def test_basic_extraction(self, sample_df):
        df, step = extract_date_features(sample_df, "date", features=["year", "month"])
        assert "date_year" in df.columns
        assert "date_month" in df.columns
        assert (df["date_year"] == 2024).all()

    def test_all_features(self, sample_df):
        df, step = extract_date_features(
            sample_df, "date",
            features=["year", "month", "day", "weekday", "hour", "quarter"],
        )
        assert "date_hour" in df.columns
        assert "date_quarter" in df.columns

    def test_string_dates(self):
        df = pd.DataFrame({"d": ["2024-03-15", "2024-06-20", "2024-09-01"]})
        df_out, step = extract_date_features(df, "d", features=["month"])
        assert "d_month" in df_out.columns


# ── Text features ───────────────────────────────────────────────────


class TestTextFeatures:
    def test_length(self, sample_df):
        df, step = extract_text_features(sample_df, "name", features=["length"])
        assert "name_len" in df.columns
        assert df["name_len"].iloc[0] == 5  # "alice"

    def test_word_count(self, sample_df):
        df, _ = extract_text_features(sample_df, "name", features=["word_count"])
        assert "name_words" in df.columns
        assert (df["name_words"] == 1).all()  # single words

    def test_uppercase_ratio(self, sample_df):
        df, _ = extract_text_features(sample_df, "name", features=["uppercase_ratio"])
        assert "name_upper_ratio" in df.columns


# ── Interactions ────────────────────────────────────────────────────


class TestInteraction:
    def test_multiply(self, sample_df):
        df, step = create_interaction(sample_df, "price", "quantity", operation="multiply")
        assert "price_multiply_quantity" in df.columns
        assert df["price_multiply_quantity"].iloc[0] == pytest.approx(10.0)

    def test_divide(self, sample_df):
        df, _ = create_interaction(sample_df, "price", "quantity", operation="divide")
        assert "price_divide_quantity" in df.columns

    def test_add(self, sample_df):
        df, _ = create_interaction(sample_df, "price", "quantity", operation="add")
        assert df["price_add_quantity"].iloc[0] == pytest.approx(11.0)

    def test_subtract(self, sample_df):
        df, _ = create_interaction(sample_df, "price", "quantity", operation="subtract")
        assert df["price_subtract_quantity"].iloc[0] == pytest.approx(9.0)


# ── Polynomial ──────────────────────────────────────────────────────


class TestPolynomial:
    def test_degree_2(self, sample_df):
        df, step = create_polynomial(sample_df, "price", degree=2)
        assert "price_pow2" in df.columns
        assert df["price_pow2"].iloc[0] == pytest.approx(100.0)

    def test_degree_3(self, sample_df):
        df, _ = create_polynomial(sample_df, "price", degree=3)
        assert "price_pow2" in df.columns
        assert "price_pow3" in df.columns
        assert df["price_pow3"].iloc[0] == pytest.approx(1000.0)


# ── Variance threshold ──────────────────────────────────────────────


class TestVarianceThreshold:
    def test_drops_constant(self):
        df = pd.DataFrame({
            "const": [5, 5, 5, 5, 5],
            "varying": [1, 2, 3, 4, 5],
        })
        df_out, step, dropped = variance_threshold_select(df, threshold=0.01)
        assert "const" in dropped
        assert "varying" not in dropped

    def test_keeps_all_when_high_threshold(self, sample_df):
        df_out, step, dropped = variance_threshold_select(sample_df, threshold=0.0)
        assert len(dropped) == 0


# ── Correlation selection ───────────────────────────────────────────


class TestCorrelationSelect:
    def test_drops_highly_correlated(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.001, 2.001, 3.001, 4.001, 5.001],  # nearly identical to a
            "c": [10, 20, 30, 40, 50],
        })
        df_out, step, dropped = correlation_select(df, threshold=0.99)
        assert len(dropped) >= 1
        assert "a" in df_out.columns or "b" in df_out.columns  # one kept

    def test_keeps_uncorrelated(self, sample_df):
        df_out, step, dropped = correlation_select(sample_df, threshold=0.99)
        # price and quantity are not highly correlated
        assert "price" in df_out.columns


# ── Feature importance ──────────────────────────────────────────────


class TestFeatureImportance:
    def test_returns_dataframe(self, sample_df):
        imp = get_feature_importance(sample_df, "label")
        assert imp is not None
        assert "importance" in imp.columns
        assert len(imp) > 0

    def test_no_numeric_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        assert get_feature_importance(df, "a") is None

    def test_target_not_in_df(self, sample_df):
        assert get_feature_importance(sample_df, "nonexistent") is None
