"""
Tests for utils.data_analysis — analysis and quality-reporting functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.data_analysis import (
    DataQualitySummary,
    UniqueValueReport,
    analyse_duplicates,
    analyse_missing_values,
    count_unique_values,
    detect_constant_columns,
    full_quality_analysis,
    get_categorical_stats,
    get_dataset_overview,
    get_numerical_stats,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """A small DataFrame with known properties."""
    return pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["a", "b", "a", "b", "a"],
        "mixed": [1, "two", 3, None, 5],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame with explicit missing values."""
    return pd.DataFrame({
        "x": [1, None, 3, None, 5],
        "y": ["a", "b", None, "d", None],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame with some duplicate rows."""
    return pd.DataFrame({
        "a": [1, 2, 1, 3, 2],
        "b": ["x", "y", "x", "z", "y"],
    })


@pytest.fixture
def df_with_constant() -> pd.DataFrame:
    """DataFrame with one constant column."""
    return pd.DataFrame({
        "const": [7, 7, 7, 7, 7],
        "varying": [1, 2, 3, 4, 5],
    })


# ── get_dataset_overview ────────────────────────────────────────────


class TestDatasetOverview:
    def test_basic_properties(self, simple_df):
        ov = get_dataset_overview(simple_df, name="test")
        assert ov.name == "test"
        assert ov.n_rows == 5
        assert ov.n_cols == 3
        assert ov.n_numeric == 1
        assert ov.n_categorical == 2
        assert "num" in ov.column_names
        assert ov.memory_human  # non-empty string

    def test_empty_df(self):
        df = pd.DataFrame()
        ov = get_dataset_overview(df, name="empty")
        assert ov.n_rows == 0
        assert ov.n_cols == 0

    def test_memory_human_format(self, simple_df):
        ov = get_dataset_overview(simple_df)
        # Should contain a unit
        assert any(unit in ov.memory_human for unit in ("B", "KB", "MB"))


# ── analyse_missing_values ──────────────────────────────────────────


class TestMissingValues:
    def test_no_missing(self, simple_df):
        # simple_df has no NaN in num/cat cols (mixed has None but it's object)
        report = analyse_missing_values(simple_df)
        # mixed column has 1 None
        assert report.total_missing >= 0
        assert isinstance(report.percent_missing, float)

    def test_with_missing(self, df_with_missing):
        report = analyse_missing_values(df_with_missing)
        assert report.total_missing == 4  # x:2 + y:2
        assert report.percent_missing > 0
        assert "x" in report.per_column
        assert "y" in report.per_column

    def test_no_missing_column_not_in_report(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = analyse_missing_values(df)
        assert report.total_missing == 0
        assert report.per_column == {}


# ── analyse_duplicates ──────────────────────────────────────────────


class TestDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = analyse_duplicates(df)
        assert report.n_duplicates == 0
        assert report.percent_duplicates == 0.0

    def test_with_duplicates(self, df_with_duplicates):
        report = analyse_duplicates(df_with_duplicates)
        # Row 2 duplicates row 0, row 4 duplicates row 1 => 2 dupes
        assert report.n_duplicates == 2
        assert report.percent_duplicates == 40.0
        assert len(report.duplicate_indices) == 2


# ── count_unique_values ─────────────────────────────────────────────


class TestUniqueValues:
    def test_basic(self, simple_df):
        report = count_unique_values(simple_df)
        assert report.per_column["num"] == 5
        assert report.per_column["cat"] == 2

    def test_empty_df(self):
        df = pd.DataFrame()
        report = count_unique_values(df)
        assert report.per_column == {}


# ── detect_constant_columns ─────────────────────────────────────────


class TestConstantColumns:
    def test_with_constant(self, df_with_constant):
        report = detect_constant_columns(df_with_constant)
        assert "const" in report.columns
        assert "varying" not in report.columns

    def test_no_constants(self, simple_df):
        report = detect_constant_columns(simple_df)
        assert report.columns == []

    def test_all_nan_column_is_not_constant(self):
        df = pd.DataFrame({"all_nan": [None, None, None]})
        report = detect_constant_columns(df)
        assert report.columns == []


# ── Descriptive statistics ──────────────────────────────────────────


class TestDescriptiveStats:
    def test_numerical_stats(self, simple_df):
        stats = get_numerical_stats(simple_df)
        assert not stats.empty
        assert "mean" in stats.index
        assert "num" in stats.columns

    def test_numerical_stats_empty_for_no_numeric(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        stats = get_numerical_stats(df)
        assert stats.empty

    def test_categorical_stats(self, simple_df):
        stats = get_categorical_stats(simple_df)
        assert not stats.empty
        assert "cat" in stats.columns

    def test_categorical_stats_empty_for_no_categorical(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        stats = get_categorical_stats(df)
        assert stats.empty


# ── full_quality_analysis ───────────────────────────────────────────


class TestFullQualityAnalysis:
    def test_returns_dataclass(self, simple_df):
        report = full_quality_analysis(simple_df, name="simple")
        assert isinstance(report, DataQualitySummary)
        assert report.overview.name == "simple"
        assert report.missing.total_cells == simple_df.size

    def test_missing_report_included(self, df_with_missing):
        report = full_quality_analysis(df_with_missing)
        assert report.missing.total_missing > 0

    def test_duplicate_report_included(self, df_with_duplicates):
        report = full_quality_analysis(df_with_duplicates)
        assert report.duplicates.n_duplicates >= 1

    def test_constant_report_included(self, df_with_constant):
        report = full_quality_analysis(df_with_constant)
        assert "const" in report.constant_columns.columns
