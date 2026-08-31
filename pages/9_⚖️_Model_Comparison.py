"""
Model Comparison — Train and benchmark multiple models side by side.

Students learn:
- Train all supported models on the same dataset
- Compare metrics in a single table
- Visualise model performance
- Understand that the "best" metric depends on the problem
- Download comparison results
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from utils.model_comparison import ComparisonResult, compare_classifiers, compare_regressors
from utils.models import CLASSIFIERS, get_classifier_names
from utils.regression_models import REGRESSORS, get_regressor_names
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
build_sidebar()
page_header("model_comparison")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

st.success(f"⚖️ Comparing models on: **{name}** ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")

# ── Task selection ──────────────────────────────────────────────────
task = st.radio("Task type", ["Classification", "Regression"], horizontal=True, key="comp_task")

if task == "Classification":
    num_cols_all = df.select_dtypes("number").columns.tolist()
    potential_targets = [c for c in df.columns if df[c].nunique() <= 50]
    if not potential_targets:
        st.error("No suitable target column found.")
        st.stop()

    target = st.selectbox("🎯 Target column", potential_targets, key="comp_target")
    clf_names = get_classifier_names()
    model_keys = st.multiselect(
        "Select models to compare",
        list(CLASSIFIERS.keys()),
        default=list(CLASSIFIERS.keys()),
        format_func=lambda k: CLASSIFIERS[k].name,
        key="comp_models",
    )
else:
    num_cols_all = df.select_dtypes("number").columns.tolist()
    if len(num_cols_all) < 2:
        st.error("Need at least 2 numerical columns.")
        st.stop()
    target = st.selectbox("🎯 Target column (continuous)", num_cols_all, key="comp_target")
    reg_names = get_regressor_names()
    model_keys = st.multiselect(
        "Select models to compare",
        list(REGRESSORS.keys()),
        default=list(REGRESSORS.keys()),
        format_func=lambda k: REGRESSORS[k].name,
        key="comp_models",
    )

test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="comp_test")
random_state = st.number_input("Random state", value=42, key="comp_rs")

# ── Run comparison ──────────────────────────────────────────────────
if st.button("🚀 Run Comparison", type="primary", key="run_comp") and model_keys:
    with st.spinner(f"Training {len(model_keys)} models…"):
        if task == "Classification":
            comp: ComparisonResult = compare_classifiers(
                df, target, model_keys=model_keys,
                test_size=test_size, random_state=random_state,
            )
        else:
            comp = compare_regressors(
                df, target, model_keys=model_keys,
                test_size=test_size, random_state=random_state,
            )
    st.session_state["comp_result"] = comp
    st.rerun()

# ── Display results ─────────────────────────────────────────────────
comp: ComparisonResult | None = st.session_state.get("comp_result")

if comp is not None:
    st.markdown("---")
    st.markdown(f"### 📊 {comp.task.title()} Model Comparison")

    st.dataframe(comp.table, use_container_width=True, hide_index=True)

    # Visual comparison
    st.markdown("#### Visual Comparison")
    numeric_cols = [c for c in comp.table.columns if c != "Model"]
    st.bar_chart(comp.table.set_index("Model")[numeric_cols])

    # Best model
    st.markdown("---")
    st.markdown("### 🏆 Best Model")
    if comp.task == "classification":
        best_idx = comp.table["F1"].idxmax()
        best = comp.table.loc[best_idx]
        st.info(
            f"**{best['Model']}** — F1: {best['F1']:.4f}, "
            f"Accuracy: {best['Accuracy']:.4f}, AUC: {best.get('AUC', 'N/A')}"
        )
    else:
        best_idx = comp.table["R²"].idxmax()
        best = comp.table.loc[best_idx]
        st.info(
            f"**{best['Model']}** — R²: {best['R²']:.4f}, "
            f"MAE: {best['MAE']:.4f}, RMSE: {best['RMSE']:.4f}"
        )

    st.warning(
        "⚠️ **Important:** The model with the highest metric on this split is not "
        "necessarily the best real-world model. Consider:\n"
        "- Cross-validation for more reliable estimates\n"
        "- Inference speed and memory requirements\n"
        "- Interpretability needs\n"
        "- How the model handles unseen data"
    )

    # Download
    st.markdown("---")
    csv_buf = io.StringIO()
    comp.table.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download comparison table as CSV",
        csv_buf.getvalue(),
        file_name=f"{name}_{comp.task}_comparison.csv",
        mime="text/csv",
    )

    # Code
    with st.expander("📜 Generated Python Code", expanded=False):
        st.code(comp.code, language="python")

# ── Educational section ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 How to Choose the Right Model", expanded=False):
    st.markdown(
        """
        1. **Start simple** — Linear/Logistic Regression as a baseline
        2. **Try tree-based** — Random Forest is a strong default
        3. **Push accuracy** — Gradient Boosting often wins on tabular data
        4. **Check cross-validation** — Don't trust a single train/test split
        5. **Consider constraints** — Speed, interpretability, memory
        6. **Domain matters** — A slightly lower metric with an interpretable model may be better in practice
        """
    )
