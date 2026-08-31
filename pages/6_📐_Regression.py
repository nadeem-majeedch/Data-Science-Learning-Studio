"""
Regression Lab — Train and evaluate regression models.

Students learn and perform:
- Select a continuous target column
- Automatic preprocessing via sklearn Pipeline
- Train/test split
- Train 7 regression algorithms
- Evaluate with R², MAE, MSE, RMSE
- View actual vs predicted and residual plots
- Feature importance, code generation, educational explanations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.regression_models import REGRESSORS, get_regressor, get_regressor_names, key_from_name
from utils.regression_training import (
    RegressionResult,
    detect_feature_types,
    extract_feature_importance,
    train_regressor,
)
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Regression", page_icon="📐", layout="wide")
build_sidebar()
page_header("regression")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

st.success(f"📐 Regression on: **{name}** ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")

# ═════════════════════════════════════════════════════════════════════
#  SIDEBAR: Configuration
# ═════════════════════════════════════════════════════════════════════

num_cols_all = df.select_dtypes("number").columns.tolist()
if len(num_cols_all) < 2:
    st.error("Need at least 2 numerical columns for regression.")
    st.stop()

target = st.selectbox("🎯 Target column (continuous)", num_cols_all, key="reg_target")
num_cols, cat_cols = detect_feature_types(df, target)
st.info(f"Features: **{len(num_cols)}** numerical, **{len(cat_cols)}** categorical")

reg_names = get_regressor_names()
model_name = st.selectbox("🤖 Algorithm", reg_names, key="reg_model")
test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="reg_test")
random_state = st.number_input("Random state", value=42, key="reg_rs")
scaler = st.selectbox(
    "Numerical scaler",
    ["standard", "minmax", None],
    format_func=lambda x: {"standard": "StandardScaler", "minmax": "MinMaxScaler", None: "None"}[x],
    key="reg_scaler",
)

# ═════════════════════════════════════════════════════════════════════
#  TRAIN
# ═════════════════════════════════════════════════════════════════════

if st.button("🚀 Train Model", type="primary", key="reg_train"):
    with st.spinner("Training model…"):
        result = train_regressor(
            df, target, key_from_name(model_name),
            test_size=test_size, random_state=random_state, scaler=scaler,
        )
    st.session_state["reg_result"] = result
    st.session_state["reg_result_name"] = name
    st.rerun()

# ═════════════════════════════════════════════════════════════════════
#  RESULTS
# ═════════════════════════════════════════════════════════════════════

result: RegressionResult | None = st.session_state.get("reg_result")

if result is not None:
    st.markdown("---")
    st.markdown(f"### ✅ Results: {result.model_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{result.r2:.4f}", help="Coefficient of determination — 1.0 is perfect")
    c2.metric("MAE", f"{result.mae:.4f}", help="Mean absolute error")
    c3.metric("MSE", f"{result.mse:.4f}", help="Mean squared error")
    c4.metric("RMSE", f"{result.rmse:.4f}", help="Root mean squared error")

    st.markdown("---")

    (
        tab_actual,
        tab_residual,
        tab_table,
        tab_importance,
        tab_info,
        tab_code,
    ) = st.tabs([
        "📊 Actual vs Predicted",
        "📉 Residuals",
        "📋 Prediction Table",
        "🏆 Feature Importance",
        "📖 About Algorithm",
        "📜 Code",
    ])

    with tab_actual:
        st.markdown("#### Actual vs Predicted")
        comp = pd.DataFrame({
            "Index": range(len(result.y_test)),
            "Actual": result.y_test,
            "Predicted": result.y_pred,
        })
        st.line_chart(comp.set_index("Index")[["Actual", "Predicted"]])
        st.markdown(
            "📚 **Hint:** The closer the two lines track each other, the better the model. "
            "Divergences indicate regions where the model struggles."
        )

    with tab_residual:
        st.markdown("#### Residual Plot")
        residuals = result.y_test - result.y_pred
        res_df = pd.DataFrame({
            "Predicted": result.y_pred,
            "Residual": residuals,
        })
        st.scatter_chart(res_df, x="Predicted", y="Residual")
        st.markdown(
            "📚 **Hint:** A good model shows residuals scattered randomly around zero "
            "with no pattern. Patterns (curves, funnels) suggest the model is "
            "missing something."
        )

        col_m, col_s = st.columns(2)
        col_m.metric("Mean Residual", f"{residuals.mean():.4f}")
        col_s.metric("Std Residual", f"{residuals.std():.4f}")

    with tab_table:
        st.markdown("#### Prediction Table")
        pred_df = pd.DataFrame({
            "Actual": result.y_test,
            "Predicted": result.y_pred.round(4),
            "Error": (result.y_test - result.y_pred).round(4),
            "|Error|": np.abs(result.y_test - result.y_pred).round(4),
        })
        st.dataframe(pred_df, use_container_width=True, height=400)

    with tab_importance:
        st.markdown("#### Feature Importance")
        imp_df = extract_feature_importance(result)
        if imp_df is not None:
            st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)
            st.bar_chart(imp_df.head(15).set_index("feature")["importance"])
            st.markdown(
                "📚 **Hint:** Feature importance shows contribution to predictions. "
                "High importance ≠ causation."
            )
        else:
            st.info(f"{result.model_name} does not expose feature importances.")

    with tab_info:
        info = get_regressor(key_from_name(result.model_name))
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

# ── Always-visible educational sections ─────────────────────────────
st.markdown("---")

with st.expander("📖 Understanding Regression Metrics", expanded=False):
    st.markdown(
        """
        | Metric | Formula | Interpretation |
        |---|---|---|
        | **R²** | 1 − SS_res / SS_tot | 1.0 = perfect, 0.0 = predicts mean |
        | **MAE** | mean(|y − ŷ|) | Average error in original units |
        | **MSE** | mean((y − ŷ)²) | Penalises large errors more |
        | **RMSE** | √MSE | Same units as target, interpretable |

        ⚠️ R² can be negative if the model is worse than simply predicting the mean.
        """
    )

with st.expander("📖 Data Leakage Warning", expanded=False):
    st.markdown(
        """
        This module prevents leakage by:
        1. Splitting data **before** preprocessing
        2. Fitting imputers and scalers on **training data only**
        3. Using a single `Pipeline` for the entire flow

        Never fit any transformer on the full dataset before splitting.
        """
    )
