"""
Reusable visualisation helpers for Data Science Lab.

Every public function takes a ``pandas.DataFrame`` (and optional column
names) and returns a ``plotly.graph_objects.Figure`` or
``plotly.express`` figure that can be rendered by Streamlit via
``st.plotly_chart``.

Naming convention: ``fig_<chart_type>`` keeps the namespace tidy and
makes imports explicit.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette ──────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Set2


# ═════════════════════════════════════════════════════════════════════
#  NUMERICAL CHARTS
# ═════════════════════════════════════════════════════════════════════


def fig_histogram(
    df: pd.DataFrame,
    col: str,
    *,
    nbins: int = 30,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive histogram with optional KDE overlay.

    Parameters
    ----------
    df : DataFrame
    col : column name to plot
    nbins : number of bins
    color : optional column for colour grouping
    title : optional custom title
    """
    fig = px.histogram(
        df,
        x=col,
        color=color,
        nbins=nbins,
        opacity=0.75,
        marginal="rug",
        title=title or f"Distribution of {col}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        template="plotly_white",
    )
    return fig


def fig_box_plot(
    df: pd.DataFrame,
    col: str,
    *,
    by: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Box plot for a single numerical column, optionally grouped."""
    fig = px.box(
        df,
        y=col,
        x=by,
        points="outliers",
        title=title or (f"{col} by {by}" if by else f"Box Plot of {col}"),
        color=by,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white")
    return fig


def fig_density(
    df: pd.DataFrame,
    cols: list[str],
    *,
    title: str | None = None,
) -> go.Figure:
    """Overlaid kernel-density estimate (KDE) for one or more columns."""
    fig = go.Figure()
    for i, col in enumerate(cols):
        data = df[col].dropna()
        fig.add_trace(
            go.Violin(
                y=data,
                name=col,
                box_visible=True,
                meanline_visible=True,
                opacity=0.6,
                line_color=PALETTE[i % len(PALETTE)],
            )
        )
    fig.update_layout(
        title=title or "Density / Violin Plot",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=len(cols) > 1,
    )
    return fig


def fig_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    color: str | None = None,
    trendline: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Scatter plot with optional trendline and colour grouping."""
    trendline_arg = None
    if trendline:
        try:
            import statsmodels.api  # noqa: F401

            trendline_arg = "ols"
        except ImportError:
            pass  # statsmodels not installed — skip trendline silently

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        opacity=0.7,
        trendline=trendline_arg,
        title=title or f"{y_col} vs {x_col}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white")
    return fig


# ═════════════════════════════════════════════════════════════════════
#  CORRELATION
# ═════════════════════════════════════════════════════════════════════


def fig_correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: str | None = None,
) -> go.Figure:
    """Correlation heatmap for all numerical columns."""
    num_df = df.select_dtypes("number")
    if num_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Need at least 2 numerical columns for correlation",
            template="plotly_white",
        )
        return fig

    corr = num_df.corr(method=method).round(3)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 11},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title or f"Correlation Matrix ({method})",
        template="plotly_white",
        width=700,
        height=600,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
#  CATEGORICAL CHARTS
# ═════════════════════════════════════════════════════════════════════


def fig_bar_chart(
    df: pd.DataFrame,
    col: str,
    *,
    top_n: int = 20,
    title: str | None = None,
) -> go.Figure:
    """Bar chart of value counts for a categorical column."""
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]

    fig = px.bar(
        vc,
        x=col,
        y="count",
        title=title or f"Value Counts: {col}",
        color=col,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def fig_frequency_distribution(
    df: pd.DataFrame,
    col: str,
    *,
    title: str | None = None,
) -> go.Figure:
    """Horizontal bar chart showing frequency distribution."""
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    vc["percent"] = (vc["count"] / vc["count"].sum() * 100).round(1)

    fig = px.bar(
        vc,
        y=col,
        x="count",
        orientation="h",
        title=title or f"Frequency Distribution: {col}",
        text=vc.apply(lambda r: f"{r['count']} ({r['percent']}%)", axis=1),
        color=col,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis_title="Count",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_traces(textposition="outside")
    return fig


# ═════════════════════════════════════════════════════════════════════
#  MISSING VALUES
# ═════════════════════════════════════════════════════════════════════


def fig_missing_values(
    df: pd.DataFrame,
    *,
    title: str | None = None,
) -> go.Figure:
    """Horizontal bar chart of missing-value counts per column."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No missing values detected 🎉",
            template="plotly_white",
        )
        return fig

    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        title=title or "Missing Values per Column",
        labels={"x": "Missing Count", "y": "Column"},
        color=missing.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def fig_missing_matrix(
    df: pd.DataFrame,
    *,
    max_rows: int = 50,
    title: str | None = None,
) -> go.Figure:
    """Heatmap-style missing-value matrix (present=1, missing=0)."""
    subset = df.head(max_rows)
    missing_matrix = subset.notna().astype(int)

    fig = go.Figure(
        data=go.Heatmap(
            z=missing_matrix.values.T,
            x=[str(i) for i in missing_matrix.index],
            y=missing_matrix.columns.tolist(),
            colorscale=[[0, "#e74c3c"], [1, "#2ecc71"]],
            showscale=False,
            hovertemplate="Row: %{x}<br>Column: %{y}<br>Present: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or f"Missing Value Matrix (first {max_rows} rows)",
        xaxis_title="Row Index",
        template="plotly_white",
        height=max(300, len(df.columns) * 25),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
#  PAIRWISE / MULTI-VIEW
# ═════════════════════════════════════════════════════════════════════


def fig_pairwise_scatter(
    df: pd.DataFrame,
    cols: list[str],
    *,
    color: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Scatter matrix (pair plot) for selected numerical columns."""
    fig = px.scatter_matrix(
        df,
        dimensions=cols,
        color=color,
        opacity=0.6,
        title=title or "Pairwise Scatter Matrix",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(diagonal_visible=True, marker=dict(size=3))
    fig.update_layout(template="plotly_white")
    return fig


# ═════════════════════════════════════════════════════════════════════
#  EDUCATIONAL HINTS
# ═════════════════════════════════════════════════════════════════════


def interpret_correlation(corr_value: float) -> str:
    """Return a short educational hint about a correlation coefficient.

    These are labelled as *educational hints*, not conclusions.
    """
    strength = abs(corr_value)
    direction = "positive" if corr_value > 0 else "negative"

    if strength < 0.1:
        desc = "negligible"
    elif strength < 0.3:
        desc = "weak"
    elif strength < 0.5:
        desc = "moderate"
    elif strength < 0.7:
        desc = "strong"
    else:
        desc = "very strong"

    return (
        f"📚 **Educational hint:** The {direction} correlation is {desc} "
        f"(r = {corr_value:.3f}). This does **not** imply causation — "
        "further analysis is needed to understand the relationship."
    )


def interpret_skewness(series: pd.Series) -> str:
    """Return a hint about the skewness of a distribution."""
    skew = series.dropna().skew()
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    elif skew > 0:
        shape = "right-skewed (tail extends to the right)"
    else:
        shape = "left-skewed (tail extends to the left)"

    return (
        f"📚 **Educational hint:** The distribution is {shape} "
        f"(skewness = {skew:.2f}). "
    )
