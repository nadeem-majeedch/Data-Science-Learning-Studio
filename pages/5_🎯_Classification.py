"""
Classification Lab — Train and evaluate classification models.

Students learn and perform:
- Select target column and features
- Automatic preprocessing (imputation, encoding, scaling) via Pipeline
- Train/test split with stratification
- Train 7 different classifiers
- Evaluate with accuracy, precision, recall, F1, confusion matrix
- View feature importance
- Generate reproducible Python code
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_training import (
    TrainResult,
    detect_feature_types,
    extract_feature_importance,
    train_classifier,
)
from utils.models import CLASSIFIERS, get_classifier, get_classifier_names, key_from_name
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Classification", page_icon="🎯", layout="wide")
build_sidebar()
page_header("classification")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

st.success(f"🎯 Classification on: **{name}** ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")

# ═════════════════════════════════════════════════════════════════════
#  SIDEBAR: Configuration
# ═════════════════════════════════════════════════════════════════════

# Target selection
potential_targets = [c for c in df.columns if df[c].nunique() <= 50]
if not potential_targets:
    st.error("No suitable target column found (need a column with ≤50 unique values).")
    st.stop()

target = st.selectbox("🎯 Target column", potential_targets, key="clf_target")
n_classes = df[target].nunique()
st.caption(f"Detected **{n_classes}** classes" + (" (binary)" if n_classes == 2 else ""))

# Feature type detection
num_cols, cat_cols = detect_feature_types(df, target)
st.info(f"Features: **{len(num_cols)}** numerical, **{len(cat_cols)}** categorical")

# Model selection
model_names = get_classifier_names()
model_name = st.selectbox("🤖 Algorithm", model_names, key="clf_model")
model_info = get_classifier(key_from_name(model_name))

# Split configuration
test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="clf_test")
random_state = st.number_input("Random state", value=42, key="clf_rs")

# Preprocessing
scaler = st.selectbox(
    "Numerical scaler",
    ["standard", "minmax", None],
    format_func=lambda x: {"standard": "StandardScaler", "minmax": "MinMaxScaler", None: "None (imputer only)"}[x],
    key="clf_scaler",
)

# ═════════════════════════════════════════════════════════════════════
#  TRAIN
# ═════════════════════════════════════════════════════════════════════

if st.button("🚀 Train Model", type="primary", key="train_btn"):
    with st.spinner("Training model…"):
        result = train_classifier(
            df, target, key_from_name(model_name),
            test_size=test_size,
            random_state=random_state,
            scaler=scaler,
        )
    st.session_state["clf_result"] = result
    st.session_state["clf_result_name"] = name
    st.rerun()

# ═════════════════════════════════════════════════════════════════════
#  RESULTS
# ═════════════════════════════════════════════════════════════════════

result: TrainResult | None = st.session_state.get("clf_result")

if result is not None:
    st.markdown("---")
    st.markdown(f"### ✅ Results: {result.model_name}")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{result.accuracy:.4f}")
    c2.metric("Precision", f"{result.precision:.4f}")
    c3.metric("Recall", f"{result.recall:.4f}")
    c4.metric("F1 Score", f"{result.f1:.4f}")

    st.markdown("---")

    (
        tab_report,
        tab_cm,
        tab_importance,
        tab_info,
        tab_code,
    ) = st.tabs(["📋 Classification Report", "📊 Confusion Matrix", "🏆 Feature Importance", "📖 About Algorithm", "📜 Code"])

    with tab_report:
        st.code(result.report_text, language="text")

    with tab_cm:
        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(
            result.confusion,
            index=[f"Actual: {c}" for c in result.classes],
            columns=[f"Pred: {c}" for c in result.classes],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown(
            "📚 **How to read:** Each row is an actual class; each column is a "
            "predicted class. The diagonal shows correct predictions. Off-diagonal "
            "values are errors."
        )

    with tab_importance:
        imp_df = extract_feature_importance(result)
        if imp_df is not None:
            st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)
            st.bar_chart(imp_df.head(15).set_index("feature")["importance"])
            st.markdown(
                "📚 **Hint:** Feature importance shows how much each feature "
                "contributed to the model's decisions. High importance ≠ causation."
            )
        else:
            st.info(f"{result.model_name} does not expose feature importances directly.")

    with tab_info:
        info = get_classifier(key_from_name(result.model_name))
        st.markdown(f"#### {info.name}")
        st.markdown(info.description)
        st.markdown("**Why use it?**")
        st.markdown(info.why_use)

        col_adv, col_lim = st.columns(2)
        with col_adv:
            st.markdown("**Advantages**")
            for a in info.advantages:
                st.markdown(f"- ✅ {a}")
        with col_lim:
            st.markdown("**Limitations**")
            for l in info.limitations:
                st.markdown(f"- ⚠️ {l}")

        st.markdown("**When to use:**")
        st.info(info.when_to_use)

        st.markdown("**Key parameters:**")
        for param, desc in info.important_params.items():
            st.markdown(f"- `{param}`: {desc}")

    with tab_code:
        st.markdown("#### 📜 Generated Python Code")
        st.code(result.code, language="python")

# ═════════════════════════════════════════════════════════════════════
#  EDUCATIONAL EXPANDERS (always visible)
# ═════════════════════════════════════════════════════════════════════

st.markdown("---")

with st.expander("📖 Understanding Classification Metrics", expanded=False):
    st.markdown(
        """
        | Metric | What it measures | When it matters most |
        |---|---|---|
        | **Accuracy** | Overall correctness | Balanced classes |
        | **Precision** | Of predicted positives, how many are correct | Minimising false positives (spam filter) |
        | **Recall** | Of actual positives, how many were caught | Minimising false negatives (disease detection) |
        | **F1 Score** | Harmonic mean of precision and recall | Imbalanced classes |

        ⚠️ **Accuracy can be misleading** with imbalanced classes. A model that
        always predicts the majority class can achieve high accuracy but be useless.
        """
    )

with st.expander("📖 Data Leakage Warning", expanded=False):
    st.markdown(
        """
        **Data leakage** occurs when information from the test set leaks into
        training. This module prevents leakage by:

        1. Splitting data **before** preprocessing
        2. Fitting all transformers on **training data only**
        3. Using a single `Pipeline` that chains preprocessing → model

        Never fit a scaler, encoder, or imputer on the full dataset before
        splitting — this is the most common source of leakage.
        """
    )
