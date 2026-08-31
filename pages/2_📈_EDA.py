"""
Exploratory Data Analysis — Statistics, distributions, correlations.

Students can:
- View an automatic EDA summary
- Choose columns and chart types interactively
- Explore numerical and categorical visualisations
- Study correlation relationships
- See missing-value patterns
- Read educational interpretation hints for every chart
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_analysis import (
    analyse_missing_values,
    get_categorical_stats,
    get_numerical_stats,
    get_dataset_overview,
)
from utils.ui import build_sidebar, page_header
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

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
build_sidebar()
page_header("eda")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

num_cols = df.select_dtypes("number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

st.success(
    f"📊 Exploring dataset: **{name}** "
    f"({df.shape[0]:,} rows × {df.shape[1]} columns — "
    f"{len(num_cols)} numerical, {len(cat_cols)} categorical)"
)

# ═════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════
(
    tab_auto,
    tab_numerical,
    tab_categorical,
    tab_correlation,
    tab_missing,
    tab_pairwise,
) = st.tabs([
    "🤖 Auto Summary",
    "📊 Numerical",
    "🏷️ Categorical",
    "🔗 Correlation",
    "⚠️ Missing Values",
    "🔬 Pairwise",
])


# ── TAB: Auto Summary ──────────────────────────────────────────────
with tab_auto:
    st.markdown("### 🤖 Automatic EDA Summary")
    st.caption(
        "A quick snapshot of the dataset's shape, types, and key statistics. "
        "This is a starting point — always dig deeper in the other tabs."
    )

    ov = get_dataset_overview(df, name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov.n_rows:,}")
    c2.metric("Columns", f"{ov.n_cols:,}")
    c3.metric("Numeric", ov.n_numeric)
    c4.metric("Categorical", ov.n_categorical)

    miss = analyse_missing_values(df)
    c5, c6 = st.columns(2)
    c5.metric("Missing Cells", f"{miss.total_missing:,}", f"{miss.percent_missing}%")
    c6.metric("Memory", ov.memory_human)

    st.markdown("---")
    st.markdown("#### Column Types")
    type_summary = pd.DataFrame({
        "Column": ov.column_names,
        "Type": [ov.dtypes[c] for c in ov.column_names],
        "Non-Null": [df[c].notna().sum() for c in ov.column_names],
        "Unique": [df[c].nunique() for c in ov.column_names],
    })
    st.dataframe(type_summary, use_container_width=True, hide_index=True)

    if num_cols:
        st.markdown("---")
        st.markdown("#### Numerical Statistics")
        st.dataframe(get_numerical_stats(df).T, use_container_width=True)

    if cat_cols:
        st.markdown("---")
        st.markdown("#### Categorical Statistics")
        st.dataframe(get_categorical_stats(df), use_container_width=True)


# ── TAB: Numerical ─────────────────────────────────────────────────
with tab_numerical:
    st.markdown("### 📊 Numerical Feature Exploration")
    st.caption(
        "Choose a column and a chart type to explore the distribution "
        "of numerical features."
    )

    if not num_cols:
        st.info("No numerical columns in this dataset.")
    else:
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            num_col = st.selectbox("Select numerical column", num_cols, key="num_col")
        with col_chart2:
            chart_type = st.selectbox(
                "Chart type",
                ["Histogram", "Box Plot", "Density (Violin)"],
                key="num_chart_type",
            )

        n_bins = st.slider("Number of bins (histogram)", 5, 100, 30, key="num_bins")

        if chart_type == "Histogram":
            fig = fig_histogram(df, num_col, nbins=n_bins)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(interpret_skewness(df[num_col]))

        elif chart_type == "Box Plot":
            group_by = None
            if cat_cols:
                group_by = st.selectbox(
                    "Group by (optional)", ["— None —"] + cat_cols, key="box_group"
                )
                if group_by == "— None —":
                    group_by = None
            fig = fig_box_plot(df, num_col, by=group_by)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "📚 **Educational hint:** A box plot shows the median (line), "
                "interquartile range (box), and outliers (points). "
                "Large boxes indicate high variability; many outliers "
                "may suggest data quality issues or a heavy-tailed distribution."
            )

        elif chart_type == "Density (Violin)":
            fig = fig_density(df, [num_col])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "📚 **Educational hint:** A violin plot combines a box plot with a "
                "kernel density estimate. The wider the shape, the more data points "
                "concentrate at that value."
            )

        # Multi-column overlay
        if len(num_cols) > 1:
            st.markdown("---")
            st.markdown("#### Compare Multiple Columns")
            multi_cols = st.multiselect(
                "Select columns to overlay",
                num_cols,
                default=[num_cols[0]],
                key="multi_num_cols",
            )
            if multi_cols:
                fig = fig_density(df, multi_cols)
                st.plotly_chart(fig, use_container_width=True)


# ── TAB: Categorical ────────────────────────────────────────────────
with tab_categorical:
    st.markdown("### 🏷️ Categorical Feature Exploration")
    st.caption(
        "Explore the frequency and distribution of categorical (text) features. "
        "Class imbalance in the target column can bias classification models."
    )

    if not cat_cols:
        st.info("No categorical columns in this dataset.")
    else:
        cat_col = st.selectbox("Select categorical column", cat_cols, key="cat_col")
        cat_chart = st.selectbox(
            "Chart type",
            ["Bar Chart", "Frequency Distribution"],
            key="cat_chart_type",
        )

        top_n = st.slider("Show top N categories", 5, 50, 20, key="cat_top_n")

        if cat_chart == "Bar Chart":
            fig = fig_bar_chart(df, cat_col, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = fig_frequency_distribution(df, cat_col)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "📚 **Educational hint:** Look for heavily imbalanced categories — "
            "if one category dominates, models may learn to always predict it. "
            "This is especially important for classification targets."
        )

        # Cross-tabulation if another categorical exists
        if len(cat_cols) > 1:
            st.markdown("---")
            st.markdown("#### Cross-tabulation")
            other_cat = st.selectbox(
                "Second categorical column",
                [c for c in cat_cols if c != cat_col],
                key="cross_cat",
            )
            ct = pd.crosstab(df[cat_col], df[other_cat])
            st.dataframe(ct, use_container_width=True)


# ── TAB: Correlation ────────────────────────────────────────────────
with tab_correlation:
    st.markdown("### 🔗 Correlation Analysis")
    st.caption(
        "Correlation measures the linear relationship between two numerical "
        "variables. Values range from -1 (perfect negative) to +1 (perfect positive)."
    )

    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns for correlation analysis.")
    else:
        method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            key="corr_method",
            help="Pearson: linear. Spearman: monotonic. Kendall: ordinal.",
        )

        fig = fig_correlation_heatmap(df, method=method)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Interpret a Correlation")
        col_a, col_b = st.columns(2)
        with col_a:
            x_corr = st.selectbox("Column A", num_cols, key="corr_x")
        with col_b:
            y_corr = st.selectbox(
                "Column B", [c for c in num_cols if c != x_corr], key="corr_y"
            )

        corr_val = df[[x_corr, y_corr]].corr(method=method).iloc[0, 1]
        st.metric(f"{method.title()} correlation", f"{corr_val:.4f}")
        st.markdown(interpret_correlation(corr_val))

        # Scatter for the two selected columns
        st.markdown("---")
        st.markdown(f"#### Scatter: {y_corr} vs {x_corr}")
        color_col = None
        if cat_cols:
            color_choice = st.selectbox(
                "Colour by (optional)", ["— None —"] + cat_cols, key="scatter_color"
            )
            if color_choice != "— None —":
                color_col = color_choice
        fig = fig_scatter(df, x_corr, y_corr, color=color_col, trendline=True)
        st.plotly_chart(fig, use_container_width=True)


# ── TAB: Missing Values ────────────────────────────────────────────
with tab_missing:
    st.markdown("### ⚠️ Missing Value Analysis")
    st.caption(
        "Understanding where and how much data is missing helps choose "
        "the right imputation strategy in the Preprocessing module."
    )

    if miss.total_missing == 0:
        st.success("🎉 No missing values in this dataset!")
    else:
        fig = fig_missing_values(df)
        st.plotly_chart(fig, use_container_width=True)

        fig_matrix = fig_missing_matrix(df)
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown(
            "📚 **Educational hint:** The matrix shows missing (red) vs present (green) "
            "values. Patterns like entire rows or columns being missing suggest "
            "systematic issues that simple imputation may not address."
        )


# ── TAB: Pairwise ──────────────────────────────────────────────────
with tab_pairwise:
    st.markdown("### 🔬 Pairwise Relationships")
    st.caption(
        "A scatter matrix reveals relationships between multiple numerical "
        "features at once. Look for linear patterns, clusters, and outliers."
    )

    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns for pairwise analysis.")
    else:
        default_pair = num_cols[: min(4, len(num_cols))]
        pair_cols = st.multiselect(
            "Select columns (2–6 recommended for readability)",
            num_cols,
            default=default_pair,
            key="pair_cols",
        )

        if len(pair_cols) < 2:
            st.warning("Select at least 2 columns.")
        else:
            pair_color = None
            if cat_cols:
                pc = st.selectbox(
                    "Colour by (optional)", ["— None —"] + cat_cols, key="pair_color"
                )
                if pc != "— None —":
                    pair_color = pc

            fig = fig_pairwise_scatter(df, pair_cols[:6], color=pair_color)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "📚 **Educational hint:** Diagonal plots show individual distributions. "
                "Off-diagonal scatter plots show pairwise relationships. "
                "Look for linear patterns (correlation), clusters, and outliers."
            )
