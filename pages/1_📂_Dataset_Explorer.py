"""
Dataset Explorer — Upload, load, preview, and analyse datasets.

This module provides:
- CSV / XLSX file upload with graceful error handling
- Built-in sample datasets from datasets/
- Dataset overview (rows, cols, memory, dtypes)
- Interactive table with head / tail / random sample
- Numerical and categorical descriptive statistics
- Data-quality report (missing values, duplicates, unique values, constant columns)
- Download of the displayed dataset
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from utils.data_analysis import (
    full_quality_analysis,
    get_categorical_stats,
    get_numerical_stats,
)
from utils.data_loader import (
    list_sample_datasets,
    load_sample_dataset,
    load_uploaded_file,
)
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Dataset Explorer", page_icon="📂", layout="wide")
build_sidebar()
page_header("dataset_explorer")

# ── Upload or select a dataset ──────────────────────────────────────
st.markdown("### 📁 Load a Dataset")

col_upload, col_sample = st.columns(2)

with col_upload:
    st.markdown("#### Upload your own file")
    uploaded_file = st.file_uploader(
        "Drag and drop a CSV or Excel file",
        type=["csv", "tsv", "xlsx", "xls"],
        help="Supported formats: CSV, TSV, XLSX, XLS",
        label_visibility="collapsed",
    )

with col_sample:
    st.markdown("#### Or choose a sample dataset")
    samples = list_sample_datasets()
    sample_labels = ["— Select —"] + [f"{s['name']} — {s['description']}" for s in samples]
    sample_choice = st.selectbox("Sample datasets", sample_labels, label_visibility="collapsed")

# ── Load the chosen data ────────────────────────────────────────────
df: pd.DataFrame | None = None
dataset_name: str = ""

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file, uploaded_file.name)
        dataset_name = uploaded_file.name
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unexpected error loading file: {exc}")
        st.stop()

elif sample_choice != "— Select —":
    idx = sample_labels.index(sample_choice) - 1
    sample_info = samples[idx]
    try:
        df = load_sample_dataset(sample_info["name"])
        dataset_name = sample_info["name"]
    except FileNotFoundError as exc:
        st.error(f"❌ {exc}")
        st.stop()

# ── Main content: only if we have data ──────────────────────────────
if df is None:
    st.info("⬆️ Upload a file or select a sample dataset above to begin.")
    st.stop()

# Store in session state for other modules
st.session_state["current_dataset"] = df
st.session_state["current_dataset_name"] = dataset_name

st.success(f"✅ Loaded **{dataset_name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Run the full quality analysis ───────────────────────────────────
report = full_quality_analysis(df, name=dataset_name)

# ── Tab layout ──────────────────────────────────────────────────────
(
    tab_overview,
    tab_preview,
    tab_numerical,
    tab_categorical,
    tab_quality,
) = st.tabs([
    "📊 Overview",
    "📋 Data Preview",
    "🔢 Numerical Stats",
    "🏷️ Categorical Stats",
    "✅ Data Quality",
])

# ── Tab 1: Overview ─────────────────────────────────────────────────
with tab_overview:
    st.markdown("#### Dataset Overview")
    st.caption(
        "A high-level summary of the dataset's shape, size, and column types."
    )

    ov = report.overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov.n_rows:,}")
    c2.metric("Columns", f"{ov.n_cols:,}")
    c3.metric("Memory", ov.memory_human)
    c4.metric("Missing cells", f"{report.missing.total_missing:,}")

    st.markdown("---")
    st.markdown("##### Column Types")
    type_df = pd.DataFrame({
        "Column": ov.column_names,
        "Data Type": [ov.dtypes[c] for c in ov.column_names],
        "Unique Values": [report.unique_values.per_column.get(c, 0) for c in ov.column_names],
        "Missing": [df[c].isna().sum() for c in ov.column_names],
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)

    st.markdown("##### Type Distribution")
    type_counts = pd.Series({
        "Numeric": ov.n_numeric,
        "Categorical": ov.n_categorical,
        "DateTime": ov.n_datetime,
        "Boolean": ov.n_boolean,
    })
    type_counts = type_counts[type_counts > 0]
    if not type_counts.empty:
        st.bar_chart(type_counts)

# ── Tab 2: Data Preview ─────────────────────────────────────────────
with tab_preview:
    st.markdown("#### Interactive Data Preview")
    st.caption(
        "Browse the raw data. Choose head, tail, or a random sample."
    )

    view_mode = st.radio(
        "View mode",
        ["📋 First rows (head)", "📋 Last rows (tail)", "🎲 Random sample"],
        horizontal=True,
    )

    n_rows = st.slider("Number of rows to display", 5, min(100, len(df)), 20)

    if "head" in view_mode:
        preview_df = df.head(n_rows)
    elif "tail" in view_mode:
        preview_df = df.tail(n_rows)
    else:
        preview_df = df.sample(n=min(n_rows, len(df)), random_state=42)

    st.dataframe(preview_df, use_container_width=True, height=400)

    # Download button
    csv_buffer = io.StringIO()
    preview_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download displayed data as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{dataset_name}_preview.csv",
        mime="text/csv",
    )

# ── Tab 3: Numerical statistics ─────────────────────────────────────
with tab_numerical:
    st.markdown("#### Numerical Descriptive Statistics")
    st.caption(
        "Standard descriptive statistics (count, mean, std, min, quartiles, max) "
        "for all numerical columns. These help you understand the central tendency "
        "and spread of each feature."
    )

    num_stats = get_numerical_stats(df)
    if num_stats.empty:
        st.info("No numerical columns in this dataset.")
    else:
        st.dataframe(num_stats, use_container_width=True)

        st.markdown("---")
        st.markdown("##### What do these statistics tell you?")
        with st.expander("📖 Guide to interpreting numerical statistics", expanded=False):
            st.markdown(
                """
                | Statistic | Meaning |
                |---|---|
                | **count** | Number of non-null values |
                | **mean** | Average — sensitive to outliers |
                | **std** | Standard deviation — measures spread |
                | **min / max** | Smallest and largest values — check for impossible numbers |
                | **25% / 50% / 75%** | Quartiles — the median (50%) is the middle value |
                """
            )
            st.markdown(
                "**Tip:** If *mean* and *median* (50%) differ greatly, the data "
                "may be skewed. A large *std* relative to the *mean* indicates "
                "high variability."
            )

# ── Tab 4: Categorical statistics ───────────────────────────────────
with tab_categorical:
    st.markdown("#### Categorical Descriptive Statistics")
    st.caption(
        "Frequency and distribution of categorical (text) columns. "
        "Understanding category balances helps identify class imbalance "
        "in classification tasks."
    )

    cat_stats = get_categorical_stats(df)
    if cat_stats.empty:
        st.info("No categorical columns in this dataset.")
    else:
        st.dataframe(cat_stats, use_container_width=True)

        # Per-column value counts
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.markdown("---")
            selected_cat = st.selectbox("Explore a categorical column", cat_cols)
            vc = df[selected_cat].value_counts()
            st.bar_chart(vc)
            st.dataframe(vc.to_frame("Count"), use_container_width=True)

# ── Tab 5: Data quality ─────────────────────────────────────────────
with tab_quality:
    st.markdown("#### Data Quality Report")
    st.caption(
        "Automated checks for common data-quality issues: missing values, "
        "duplicate rows, constant columns, and more."
    )

    # Overall score
    total_issues = (
        report.missing.total_missing
        + report.duplicates.n_duplicates
        + len(report.constant_columns.columns)
    )
    if total_issues == 0:
        st.success("🎉 No data-quality issues detected!")
    else:
        st.warning(f"⚠️ Found **{total_issues}** potential issue(s) to review.")

    # Missing values
    with st.expander("1️⃣ Missing Values", expanded=report.missing.total_missing > 0):
        if report.missing.total_missing == 0:
            st.success("No missing values.")
        else:
            st.info(
                f"**{report.missing.total_missing:,}** missing cells "
                f"({report.missing.percent_missing}% of total)"
            )
            missing_df = pd.DataFrame({
                "Column": list(report.missing.per_column.keys()),
                "Missing Count": list(report.missing.per_column.values()),
            }).sort_values("Missing Count", ascending=False)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            st.bar_chart(missing_df.set_index("Column")["Missing Count"])

    # Duplicates
    with st.expander("2️⃣ Duplicate Rows", expanded=report.duplicates.n_duplicates > 0):
        if report.duplicates.n_duplicates == 0:
            st.success("No duplicate rows.")
        else:
            st.info(
                f"**{report.duplicates.n_duplicates:,}** duplicate rows "
                f"({report.duplicates.percent_duplicates}% of total)"
            )
            if st.checkbox("Show duplicate rows"):
                dupes = df.loc[df.duplicated(keep="first")]
                st.dataframe(dupes, use_container_width=True)

    # Unique values
    with st.expander("3️⃣ Unique Values per Column"):
        unique_df = pd.DataFrame({
            "Column": list(report.unique_values.per_column.keys()),
            "Unique Values": list(report.unique_values.per_column.values()),
        }).sort_values("Unique Values", ascending=False)
        st.dataframe(unique_df, use_container_width=True, hide_index=True)

    # Constant columns
    with st.expander("4️⃣ Constant Columns"):
        if not report.constant_columns.columns:
            st.success("No constant columns detected.")
        else:
            st.warning(
                f"**{len(report.constant_columns.columns)}** column(s) have only one "
                "distinct value and provide no information for modelling:"
            )
            st.write(report.constant_columns.columns)

    # Educational explanation
    with st.expander("📖 Why data quality matters", expanded=False):
        st.markdown(
            """
            **Data quality directly impacts model performance.**

            - **Missing values** can bias results or cause algorithms to fail.
            - **Duplicate rows** inflate dataset size and overfit models.
            - **Constant columns** add noise with zero predictive power.
            - **Skewed distributions** may require transformations before modelling.

            Always clean your data **before** training any model.
            """
        )

# ── Download full processed dataset ─────────────────────────────────
st.markdown("---")
csv_full = io.StringIO()
df.to_csv(csv_full, index=False)
st.download_button(
    label="⬇️ Download full dataset as CSV",
    data=csv_full.getvalue(),
    file_name=f"{dataset_name}_full.csv",
    mime="text/csv",
    type="primary",
)
