"""
Tests for utils.visualization — Plotly chart functions.

Verifies that every ``fig_*`` function returns a valid Plotly figure
object and that educational hint functions return non-empty strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from utils.visualization import (
    fig_bar_chart,
    fig_box_plot,
    fig_correlation_heatmap,
    fig_density,
    fig_frequency_distribution,
    fig_histogram,
    fig_missing_matrix,
    fig_missing_values,
    fig_pairwise_scatter,
    fig_scatter,
    interpret_correlation,
    interpret_skewness,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def num_df() -> pd.DataFrame:
    """DataFrame with numerical and categorical columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "y": rng.normal(5, 2, 100),
        "z": rng.uniform(0, 10, 100),
        "cat": rng.choice(["A", "B", "C"], 100),
    })


@pytest.fixture
def missing_df() -> pd.DataFrame:
    """DataFrame with some missing values."""
    return pd.DataFrame({
        "a": [1, None, 3, None, 5],
        "b": ["x", "y", None, "z", None],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# ── Numerical charts ────────────────────────────────────────────────


class TestHistogram:
    def test_returns_figure(self, num_df):
        fig = fig_histogram(num_df, "x")
        assert isinstance(fig, go.Figure)

    def test_custom_bins(self, num_df):
        fig = fig_histogram(num_df, "x", nbins=10)
        assert isinstance(fig, go.Figure)

    def test_with_color(self, num_df):
        fig = fig_histogram(num_df, "x", color="cat")
        assert isinstance(fig, go.Figure)


class TestBoxPlot:
    def test_returns_figure(self, num_df):
        fig = fig_box_plot(num_df, "x")
        assert isinstance(fig, go.Figure)

    def test_grouped(self, num_df):
        fig = fig_box_plot(num_df, "x", by="cat")
        assert isinstance(fig, go.Figure)


class TestDensity:
    def test_single_column(self, num_df):
        fig = fig_density(num_df, ["x"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_multiple_columns(self, num_df):
        fig = fig_density(num_df, ["x", "y", "z"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3


class TestScatter:
    def test_returns_figure(self, num_df):
        fig = fig_scatter(num_df, "x", "y")
        assert isinstance(fig, go.Figure)

    def test_with_trendline(self, num_df):
        fig = fig_scatter(num_df, "x", "y", trendline=True)
        assert isinstance(fig, go.Figure)

    def test_with_color(self, num_df):
        fig = fig_scatter(num_df, "x", "y", color="cat")
        assert isinstance(fig, go.Figure)


# ── Correlation ─────────────────────────────────────────────────────


class TestCorrelationHeatmap:
    def test_returns_figure(self, num_df):
        fig = fig_correlation_heatmap(num_df)
        assert isinstance(fig, go.Figure)

    def test_spearman(self, num_df):
        fig = fig_correlation_heatmap(num_df, method="spearman")
        assert isinstance(fig, go.Figure)

    def test_too_few_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        fig = fig_correlation_heatmap(df)
        assert isinstance(fig, go.Figure)


# ── Categorical charts ──────────────────────────────────────────────


class TestBarChart:
    def test_returns_figure(self, num_df):
        fig = fig_bar_chart(num_df, "cat")
        assert isinstance(fig, go.Figure)

    def test_top_n(self, num_df):
        fig = fig_bar_chart(num_df, "cat", top_n=2)
        assert isinstance(fig, go.Figure)


class TestFrequencyDistribution:
    def test_returns_figure(self, num_df):
        fig = fig_frequency_distribution(num_df, "cat")
        assert isinstance(fig, go.Figure)


# ── Missing value charts ────────────────────────────────────────────


class TestMissingValues:
    def test_returns_figure_with_missing(self, missing_df):
        fig = fig_missing_values(missing_df)
        assert isinstance(fig, go.Figure)

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fig = fig_missing_values(df)
        assert isinstance(fig, go.Figure)


class TestMissingMatrix:
    def test_returns_figure(self, missing_df):
        fig = fig_missing_matrix(missing_df)
        assert isinstance(fig, go.Figure)


# ── Pairwise ────────────────────────────────────────────────────────


class TestPairwiseScatter:
    def test_returns_figure(self, num_df):
        fig = fig_pairwise_scatter(num_df, ["x", "y"])
        assert isinstance(fig, go.Figure)

    def test_with_color(self, num_df):
        fig = fig_pairwise_scatter(num_df, ["x", "y", "z"], color="cat")
        assert isinstance(fig, go.Figure)


# ── Educational hints ───────────────────────────────────────────────


class TestInterpretCorrelation:
    def test_positive(self):
        result = interpret_correlation(0.8)
        assert "positive" in result
        assert "very strong" in result

    def test_negative(self):
        result = interpret_correlation(-0.2)
        assert "negative" in result
        assert "weak" in result

    def test_zero(self):
        result = interpret_correlation(0.0)
        assert "negligible" in result


class TestInterpretSkewness:
    def test_symmetric(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = interpret_skewness(s)
        assert "symmetric" in result

    def test_right_skewed(self):
        s = pd.Series([1, 1, 1, 2, 3, 10, 20])
        result = interpret_skewness(s)
        assert "right-skewed" in result

    def test_left_skewed(self):
        s = pd.Series([1, 2, 7, 8, 9, 9, 9])
        result = interpret_skewness(s)
        assert "left-skewed" in result
