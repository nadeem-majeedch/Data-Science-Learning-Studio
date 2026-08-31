"""
Feature Engineering — Create, transform, and select features.

Students learn and perform:
- Mathematical transformations (log, sqrt, square)
- Binning numerical variables
- Date/time feature extraction
- String/text feature extraction
- Interaction features
- Polynomial features
- Variance-threshold and correlation-based feature selection
- Feature importance from a trained model
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

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
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")
build_sidebar()
page_header("feature_engineering")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

# Session-state management
if "fe_original" not in st.session_state:
    st.session_state["fe_original"] = df.copy()
if "fe_current" not in st.session_state:
    st.session_state["fe_current"] = df.copy()
if "fe_steps" not in st.session_state:
    st.session_state["fe_steps"] = []
if st.session_state.get("fe_dataset_name") != name:
    st.session_state["fe_original"] = df.copy()
    st.session_state["fe_current"] = df.copy()
    st.session_state["fe_steps"] = []
    st.session_state["fe_dataset_name"] = name

current: pd.DataFrame = st.session_state["fe_current"]
fe_steps: list[FeatureStep] = st.session_state["fe_steps"]

st.success(
    f"⚙️ Engineering features: **{name}** "
    f"({current.shape[0]:,} rows × {current.shape[1]:,} cols)"
)

# Reset button
col_r, _ = st.columns([1, 5])
with col_r:
    if st.button("🔄 Reset to original"):
        st.session_state["fe_current"] = st.session_state["fe_original"].copy()
        st.session_state["fe_steps"] = []
        st.rerun()

# Identify column types
num_cols = current.select_dtypes("number").columns.tolist()
cat_cols = current.select_dtypes(include=["object", "category", "string"]).columns.tolist()
date_cols = current.select_dtypes(include=["datetime"]).columns.tolist()
# Also detect columns that look like dates (strings with common date patterns)
potential_date_cols = [
    c for c in cat_cols
    if current[c].dropna().astype(str).str.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.5
]

# ═════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════
(
    tab_math,
    tab_bin,
    tab_date,
    tab_text,
    tab_interact,
    tab_poly,
    tab_select,
    tab_importance,
    tab_code,
) = st.tabs([
    "🔢 Math Transforms",
    "📦 Binning",
    "📅 Date Features",
    "📝 Text Features",
    "✖️ Interactions",
    "📈 Polynomial",
    "🎯 Selection",
    "🏆 Importance",
    "📜 Code",
])


# ── TAB: Math Transforms ───────────────────────────────────────────
with tab_math:
    st.markdown("#### Mathematical Transformations")
    st.caption(
        "Transform a numerical column to reduce skewness, linearise "
        "relationships, or bring features to a similar scale."
    )

    if not num_cols:
        st.info("No numerical columns available.")
    else:
        col = st.selectbox("Select column", num_cols, key="math_col")
        transform = st.radio(
            "Transform",
            ["log", "sqrt", "square"],
            horizontal=True,
            key="math_type",
        )

        if transform == "log":
            st.markdown(
                "📚 **Hint:** Log transforms reduce right skew and compress large values. "
                "Useful for features like income, population, or price."
            )
        elif transform == "sqrt":
            st.markdown(
                "📚 **Hint:** Square root is milder than log. Good for count data "
                "and moderate right skew."
            )
        else:
            st.markdown(
                "📚 **Hint:** Squaring amplifies large values and creates a "
                "non-linear relationship. Useful for polynomial features."
            )

        if st.button("Apply transform", key="apply_math"):
            current, step = apply_math_transform(current, col, transform)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Binning ───────────────────────────────────────────────────
with tab_bin:
    st.markdown("#### Binning Numerical Variables")
    st.caption(
        "Convert continuous values into discrete bins. This can help "
        "linear models capture non-linear relationships and reduce the "
        "effect of outliers."
    )

    if not num_cols:
        st.info("No numerical columns available.")
    else:
        col = st.selectbox("Select column", num_cols, key="bin_col")
        n_bins = st.slider("Number of bins", 2, 10, 5, key="bin_n")
        method = st.radio(
            "Method",
            ["equal_width", "equal_freq"],
            format_func=lambda x: "Equal Width (uniform intervals)" if x == "equal_width" else "Equal Frequency (quantile)",
            horizontal=True,
            key="bin_method",
        )

        if method == "equal_width":
            st.markdown(
                "📚 **Hint:** Equal width divides the range into equally spaced intervals. "
                "Sensitive to outliers — a few extreme values can make bins very uneven."
            )
        else:
            st.markdown(
                "📚 **Hint:** Equal frequency (quantile) puts roughly the same number of "
                "rows in each bin. Handles outliers better but may group very different values."
            )

        if st.button("Apply binning", key="apply_bin"):
            current, step = bin_numerical(current, col, n_bins=n_bins, method=method)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Date Features ─────────────────────────────────────────────
with tab_date:
    st.markdown("#### Date/Time Feature Extraction")
    st.caption(
        "Extract meaningful components from datetime columns. "
        "Models can't read raw datetimes — they need numeric proxies."
    )

    all_date_cols = date_cols + potential_date_cols
    if not all_date_cols:
        st.info(
            "No datetime columns detected. If you have date strings, "
            "load the dataset and they will be auto-detected."
        )
    else:
        col = st.selectbox("Select datetime column", all_date_cols, key="date_col")
        features = st.multiselect(
            "Components to extract",
            ["year", "month", "day", "weekday", "hour", "quarter"],
            default=["year", "month", "day", "weekday"],
            key="date_feats",
        )

        st.markdown(
            "📚 **Hint:** `weekday` (0=Monday, 6=Sunday) captures weekly patterns. "
            "`month` captures seasonality. `quarter` captures fiscal patterns."
        )

        if features and st.button("Extract date features", key="apply_date"):
            current, step = extract_date_features(current, col, features=features)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Text Features ─────────────────────────────────────────────
with tab_text:
    st.markdown("#### String/Text Feature Extraction")
    st.caption(
        "Extract basic numerical features from text columns. "
        "These simple features can capture signal before NLP."
    )

    if not cat_cols:
        st.info("No text/categorical columns available.")
    else:
        col = st.selectbox("Select text column", cat_cols, key="text_col")
        features = st.multiselect(
            "Features to extract",
            ["length", "word_count", "uppercase_ratio"],
            default=["length", "word_count"],
            key="text_feats",
        )

        st.markdown(
            "📚 **Hint:** `length` captures response verbosity. "
            "`word_count` is similar but splits on spaces. "
            "`uppercase_ratio` can flag emphasis or anger in text data."
        )

        if features and st.button("Extract text features", key="apply_text"):
            current, step = extract_text_features(current, col, features=features)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Interactions ──────────────────────────────────────────────
with tab_interact:
    st.markdown("#### Interaction Features")
    st.caption(
        "Combine two features with an arithmetic operation. "
        "Interactions capture relationships that neither feature shows alone."
    )

    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns.")
    else:
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", num_cols, key="int_a")
        col_b = c2.selectbox("Column B", [c for c in num_cols if c != col_a], key="int_b")
        op = st.radio(
            "Operation",
            ["multiply", "add", "subtract", "divide"],
            format_func=str.title,
            horizontal=True,
            key="int_op",
        )

        st.markdown(
            "📚 **Hint:** `a × b` captures combined effect. "
            "`a / b` creates a ratio (e.g. price per unit). "
            "`a - b` captures the difference (e.g. profit = revenue − cost)."
        )

        if st.button("Create interaction", key="apply_int"):
            current, step = create_interaction(current, col_a, col_b, operation=op)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Polynomial ────────────────────────────────────────────────
with tab_poly:
    st.markdown("#### Polynomial Features")
    st.caption(
        "Create higher-order powers of a single feature. "
        "Useful for capturing non-linear trends in linear models."
    )

    if not num_cols:
        st.info("No numerical columns available.")
    else:
        col = st.selectbox("Select column", num_cols, key="poly_col")
        degree = st.slider("Degree", 2, 5, 2, key="poly_deg")

        st.markdown(
            "📚 **Hint:** A degree-2 polynomial adds x², degree-3 adds x² and x³, etc. "
            "Higher degrees risk overfitting — start with 2 and increase only if needed."
        )

        if st.button("Create polynomial features", key="apply_poly"):
            current, step = create_polynomial(current, col, degree=degree)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()


# ── TAB: Feature Selection ─────────────────────────────────────────
with tab_select:
    st.markdown("#### Feature Selection")
    st.caption(
        "Remove features that add little or redundant information. "
        "Fewer features can improve model performance and reduce overfitting."
    )

    sel_method = st.radio(
        "Selection method",
        ["Variance Threshold", "Correlation-Based"],
        horizontal=True,
        key="sel_method",
    )

    if sel_method == "Variance Threshold":
        threshold = st.slider(
            "Variance threshold", 0.0, 1.0, 0.01, 0.01, key="var_thresh",
            help="Columns with variance below this are dropped."
        )
        st.markdown(
            "📚 **Hint:** Near-zero variance means a feature is almost constant — "
            "it carries almost no information for the model."
        )
        if st.button("Apply variance threshold", key="apply_var"):
            current, step, dropped = variance_threshold_select(current, threshold=threshold)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            if dropped:
                st.success(f"✅ {step.description}")
                st.write(f"Dropped: {dropped}")
            else:
                st.info("No columns below the variance threshold.")
            st.rerun()

    else:
        threshold = st.slider(
            "Correlation threshold", 0.5, 1.0, 0.95, 0.05, key="corr_thresh",
            help="One column from each pair above this is dropped."
        )
        st.markdown(
            "📚 **Hint:** Highly correlated features carry redundant information. "
            "Keeping one of each pair simplifies the model without losing signal."
        )
        if st.button("Apply correlation selection", key="apply_corr"):
            current, step, dropped = correlation_select(current, threshold=threshold)
            st.session_state["fe_current"] = current
            fe_steps.append(step)
            if dropped:
                st.success(f"✅ {step.description}")
                st.write(f"Dropped: {dropped}")
            else:
                st.info("No columns above the correlation threshold.")
            st.rerun()


# ── TAB: Feature Importance ────────────────────────────────────────
with tab_importance:
    st.markdown("#### Feature Importance")
    st.caption(
        "Train a quick Random Forest to rank features by importance. "
        "This gives an initial signal about which features matter most."
    )

    if not num_cols:
        st.info("No numerical columns available.")
    else:
        target = st.selectbox("Target column", ["— None —"] + num_cols, key="imp_target")

        if target != "— None —":
            imp_df = get_feature_importance(current, target)
            if imp_df is not None:
                st.dataframe(imp_df, use_container_width=True)
                st.bar_chart(imp_df)
                st.markdown(
                    "📚 **Hint:** Feature importance from a Random Forest measures "
                    "how much each feature contributes to reducing impurity. "
                    "This is a **rough guide** — not a causal ranking."
                )
            else:
                st.warning("Could not compute importance for this target.")


# ── TAB: Generated Code ────────────────────────────────────────────
with tab_code:
    st.markdown("#### 📜 Generated Python Code")
    st.caption("Reproduce all feature engineering steps in a notebook.")

    if fe_steps:
        code_lines = ["import pandas as pd", "import numpy as np", ""]
        for i, step in enumerate(fe_steps, 1):
            code_lines.append(f"# Step {i}: {step.name}")
            code_lines.append(step.code)
            code_lines.append("")
        st.code("\n".join(code_lines), language="python")
    else:
        st.info("No feature engineering steps applied yet.")


# ── Before / After & Download ───────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Before / After")
original = st.session_state["fe_original"]
comp = pd.DataFrame({
    "Metric": ["Before", "After"],
    "Rows": [original.shape[0], current.shape[0]],
    "Columns": [original.shape[1], current.shape[1]],
    "Numeric": [original.select_dtypes("number").shape[1], current.select_dtypes("number").shape[1]],
})
st.dataframe(comp, use_container_width=True, hide_index=True)

csv_buf = io.StringIO()
current.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Download engineered dataset as CSV",
    csv_buf.getvalue(),
    file_name=f"{name}_engineered.csv",
    mime="text/csv",
    type="primary",
)
