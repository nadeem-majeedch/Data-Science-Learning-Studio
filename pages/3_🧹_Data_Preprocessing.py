"""
Data Preprocessing — Clean, transform, and prepare data for modelling.

Students learn and perform:
- Missing value handling (drop, mean, median, mode, constant)
- Duplicate detection and removal
- Categorical encoding (one-hot, label)
- Numerical scaling (Standard, MinMax, Robust)
- Outlier detection and removal (IQR)
- Train/test split with stratification
- Before/after comparison
- Equivalent Python code for every step
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from utils.data_analysis import analyse_missing_values, get_dataset_overview
from utils.preprocessing import (
    compare_before_after,
    detect_duplicates,
    detect_outliers_iqr,
    handle_missing_values,
    label_encode,
    one_hot_encode,
    remove_duplicates,
    remove_outliers,
    scale_features,
    split_data,
)
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")
build_sidebar()
page_header("preprocessing")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning(
        "⚠️ No dataset loaded. Go to **Dataset Explorer** and upload or select a dataset first."
    )
    st.stop()

# Work on a copy; original stays in session state for reset
if "preprocess_original" not in st.session_state:
    st.session_state["preprocess_original"] = df.copy()
if "preprocess_current" not in st.session_state:
    st.session_state["preprocess_current"] = df.copy()
if "preprocess_steps" not in st.session_state:
    st.session_state["preprocess_steps"] = []

# Reset if dataset changed
if st.session_state.get("preprocess_dataset_name") != name:
    st.session_state["preprocess_original"] = df.copy()
    st.session_state["preprocess_current"] = df.copy()
    st.session_state["preprocess_steps"] = []
    st.session_state["preprocess_dataset_name"] = name

current = st.session_state["preprocess_current"]
steps: list = st.session_state["preprocess_steps"]

st.success(
    f"🧹 Preprocessing: **{name}** "
    f"({current.shape[0]:,} rows × {current.shape[1]:,} cols)"
)

# ── Reset button ────────────────────────────────────────────────────
col_reset, col_spacer = st.columns([1, 5])
with col_reset:
    if st.button("🔄 Reset to original"):
        st.session_state["preprocess_current"] = st.session_state["preprocess_original"].copy()
        st.session_state["preprocess_steps"] = []
        st.rerun()

# ── Pipeline steps ──────────────────────────────────────────────────
# 1️⃣ Missing values  2️⃣ Duplicates  3️⃣ Encoding
# 4️⃣ Scaling  5️⃣ Outliers  6️⃣ Split

(
    tab_missing,
    tab_dupes,
    tab_encode,
    tab_scale,
    tab_outliers,
    tab_split,
    tab_code,
) = st.tabs([
    "1️⃣ Missing Values",
    "2️⃣ Duplicates",
    "3️⃣ Encoding",
    "4️⃣ Scaling",
    "5️⃣ Outliers",
    "6️⃣ Train/Test Split",
    "📜 Generated Code",
])


# ── TAB 1: Missing Values ──────────────────────────────────────────
with tab_missing:
    st.markdown("#### Handle Missing Values")
    miss = analyse_missing_values(current)

    if miss.total_missing == 0:
        st.success("🎉 No missing values in the current dataset.")
    else:
        st.info(
            f"**{miss.total_missing:,}** missing cells ({miss.percent_missing}% of total)"
        )
        miss_df = pd.DataFrame({
            "Column": list(miss.per_column.keys()),
            "Missing": list(miss.per_column.values()),
        }).sort_values("Missing", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        strategy = st.radio(
            "Strategy",
            ["Drop rows", "Drop columns", "Mean", "Median", "Mode", "Constant"],
            horizontal=True,
            key="miss_strategy",
        )

        affected_cols = None
        fill_val = 0
        if strategy in ("Drop columns", "Constant"):
            affected_cols = st.multiselect(
                "Select columns",
                list(miss.per_column.keys()),
                key="miss_cols",
            )
        if strategy == "Constant":
            fill_val = st.text_input("Fill value", value="0", key="miss_fill")

        if st.button("Apply missing value handling", key="apply_missing"):
            strat_map = {
                "Drop rows": "drop_rows",
                "Drop columns": "drop_columns",
                "Mean": "mean",
                "Median": "median",
                "Mode": "mode",
                "Constant": "constant",
            }
            try:
                numeric_fill = float(fill_val) if strategy == "Constant" else 0
            except ValueError:
                numeric_fill = fill_val

            current, step = handle_missing_values(
                current,
                strat_map[strategy],
                columns=affected_cols,
                fill_value=numeric_fill,
            )
            st.session_state["preprocess_current"] = current
            steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()

        with st.expander("📖 Why handle missing values?", expanded=False):
            st.markdown(
                """
                **Missing values** can cause errors in many ML algorithms. Common strategies:

                | Strategy | When to use |
                |---|---|
                | **Drop rows** | Few rows missing; data is MCAR (missing completely at random) |
                | **Drop columns** | A column has too many missing values to be useful |
                | **Mean/Median** | Numerical columns; median is robust to outliers |
                | **Mode** | Categorical columns |
                | **Constant** | When a meaningful default exists (e.g. 0 for "amount") |

                ⚠️ **Data leakage warning:** Always fit imputers on training data only.
                In production, use `sklearn.impute.SimpleImputer` inside a Pipeline.
                """
            )


# ── TAB 2: Duplicates ──────────────────────────────────────────────
with tab_dupes:
    st.markdown("#### Handle Duplicate Rows")

    n_dupes, dup_code = detect_duplicates(current)

    if n_dupes == 0:
        st.success("🎉 No duplicate rows found.")
    else:
        st.warning(f"Found **{n_dupes}** duplicate row(s) ({n_dupes / len(current) * 100:.1f}%)")
        if st.checkbox("Show duplicates"):
            st.dataframe(current[current.duplicated(keep="first")], use_container_width=True)

        if st.button("Remove duplicates", key="apply_dupes"):
            current, step = remove_duplicates(current)
            st.session_state["preprocess_current"] = current
            steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()

    with st.expander("📖 Why remove duplicates?", expanded=False):
        st.markdown(
            """
            Duplicate rows **inflate** the dataset and can cause models to
            overfit by seeing the same example multiple times. They also
            bias statistics like mean and standard deviation.

            Not all duplicates are errors — sometimes they represent real
            frequency. Think critically about *why* they exist before removing.
            """
        )


# ── TAB 3: Encoding ────────────────────────────────────────────────
with tab_encode:
    st.markdown("#### Encode Categorical Columns")

    cat_cols = current.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if not cat_cols:
        st.info("No categorical columns to encode.")
    else:
        encode_method = st.radio(
            "Encoding method",
            ["One-Hot Encoding", "Label Encoding"],
            horizontal=True,
            key="enc_method",
        )

        encode_cols = st.multiselect("Select columns to encode", cat_cols, key="enc_cols")

        if encode_cols and st.button("Apply encoding", key="apply_encode"):
            if encode_method == "One-Hot Encoding":
                current, step = one_hot_encode(current, encode_cols)
            else:
                current, step, mappings = label_encode(current, encode_cols)
                st.markdown("**Label mappings:**")
                for col, mapping in mappings.items():
                    st.write(f"`{col}`: {mapping}")

            st.session_state["preprocess_current"] = current
            steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()

        with st.expander("📖 One-Hot vs Label Encoding", expanded=False):
            st.markdown(
                """
                | Method | Pros | Cons |
                |---|---|---|
                | **One-Hot** | No false ordering; works for nominal data | Creates many columns; multicollinearity |
                | **Label** | Compact; preserves ordinal relationships | Imposes false ordering on nominal data |

                **When to use which:**
                - **One-Hot** for nominal categories (e.g. colours, countries)
                - **Label** for ordinal categories (e.g. low/medium/high)

                ⚠️ **Data leakage warning:** Always `fit` the encoder on training data.
                Use `sklearn.preprocessing.OneHotEncoder` inside a Pipeline.
                """
            )


# ── TAB 4: Scaling ──────────────────────────────────────────────────
with tab_scale:
    st.markdown("#### Scale Numerical Features")

    num_cols = current.select_dtypes("number").columns.tolist()

    if not num_cols:
        st.info("No numerical columns to scale.")
    else:
        scaler_method = st.radio(
            "Scaler",
            ["StandardScaler (z-score)", "MinMaxScaler (0–1)", "RobustScaler (median/IQR)"],
            horizontal=True,
            key="scaler_method",
        )

        scale_cols = st.multiselect("Select columns to scale", num_cols, key="scale_cols")

        if scale_cols and st.button("Apply scaling", key="apply_scale"):
            method = "standard" if "Standard" in scaler_method else (
                "minmax" if "MinMax" in scaler_method else "robust"
            )
            current, step = scale_features(current, scale_cols, method=method)
            st.session_state["preprocess_current"] = current
            steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()

        with st.expander("📖 When to scale?", expanded=False):
            st.markdown(
                """
                Scaling is **required** for algorithms that use distance
                (KNN, SVM, K-Means) and gradient descent (Linear/Logistic
                Regression, Neural Networks). Tree-based models (Random Forest,
                XGBoost) do **not** require scaling.

                | Scaler | Formula | Robust to outliers? |
                |---|---|---|
                | **StandardScaler** | z = (x − μ) / σ | No |
                | **MinMaxScaler** | x′ = (x − min) / (max − min) | No |
                | **RobustScaler** | x′ = (x − median) / IQR | **Yes** |
                """
            )


# ── TAB 5: Outliers ────────────────────────────────────────────────
with tab_outliers:
    st.markdown("#### Detect and Remove Outliers")
    st.caption("Uses the IQR (Interquartile Range) method.")

    outlier_cols = st.multiselect(
        "Select numerical columns to check",
        num_cols,
        key="outlier_cols",
    )

    multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_mult",
                           help="Lower = more aggressive outlier detection")

    if outlier_cols:
        for col in outlier_cols:
            mask, summary = detect_outliers_iqr(current, col, multiplier=multiplier)
            st.markdown(
                f"**{col}**: {summary['n_outliers']} outliers "
                f"({summary['pct_outliers']}%) — "
                f"bounds: [{summary['lower_bound']}, {summary['upper_bound']}]"
            )

        total_outliers = sum(
            detect_outliers_iqr(current, c, multiplier=multiplier)[1]["n_outliers"]
            for c in outlier_cols
        )

        if total_outliers > 0 and st.button("Remove outliers from selected columns", key="apply_outliers"):
            current, step = remove_outliers(current, outlier_cols, multiplier=multiplier)
            st.session_state["preprocess_current"] = current
            steps.append(step)
            st.success(f"✅ {step.description}")
            st.rerun()

    with st.expander("📖 What are outliers?", expanded=False):
        st.markdown(
            """
            **Outliers** are data points that lie far from the majority.
            They can distort model training, especially for linear models
            and distance-based algorithms.

            **IQR method:** A point is an outlier if it falls below
            Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.

            ⚠️ Not all outliers are errors — some may represent rare but
            genuine events. Always investigate before removing.
            """
        )


# ── TAB 6: Train/Test Split ────────────────────────────────────────
with tab_split:
    st.markdown("#### Train / Test Split")
    st.caption(
        "Split your preprocessed data into training and test sets. "
        "The test set is held out to evaluate model generalisation."
    )

    target = st.selectbox(
        "Target column (optional — leave blank for unsupervised tasks)",
        ["— None —"] + current.columns.tolist(),
        key="split_target",
    )

    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="test_size")
    random_state = st.number_input("Random state", value=42, key="random_state")

    stratify = False
    if target != "— None —":
        target_vals = current[target].nunique()
        if target_vals <= 20:
            stratify = st.checkbox(
                f"Stratify by target ({target_vals} unique values)",
                value=True,
                key="stratify",
                help="Stratification preserves class proportions in train/test sets.",
            )

    if st.button("Split data", key="apply_split"):
        t = target if target != "— None —" else None
        X_train, X_test, y_train, y_test, step = split_data(
            current, target=t,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        st.session_state["preprocess_current"] = current
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        steps.append(step)
        st.success(f"✅ {step.description}")

        c1, c2 = st.columns(2)
        c1.metric("Training set", f"{len(X_train):,} rows")
        c2.metric("Test set", f"{len(X_test):,} rows")

    with st.expander("📖 Why split data?", expanded=False):
        st.markdown(
            """
            ⚠️ **Data leakage** occurs when test-set information "leaks"
            into the training process. This makes your model appear more
            accurate than it really is.

            **Golden rule:** Never fit any scaler, encoder, or imputer on
            the full dataset before splitting. Always split first, then
            `fit` on training data and `transform` both sets.

            ```
            # CORRECT (no leakage):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            scaler.fit(X_train)                  # fit on train only
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # use train stats

            # WRONG (leakage):
            scaler.fit(X)  # sees test data statistics!
            ```
            """
        )


# ── TAB 7: Generated Code ──────────────────────────────────────────
with tab_code:
    st.markdown("#### 📜 Equivalent Python Code")
    st.caption(
        "The code below reproduces every preprocessing step applied above. "
        "Copy it into a notebook or script for reproducible results."
    )

    if steps:
        code_lines = [
            "import pandas as pd",
            "from sklearn.model_selection import train_test_split",
            "",
            f"# Load dataset",
            f"df = pd.read_csv('{name}.csv')  # or your data source",
            "",
        ]
        for i, step in enumerate(steps, 1):
            code_lines.append(f"# Step {i}: {step.name}")
            code_lines.append(step.code)
            code_lines.append("")

        code_lines.append("# Preview result")
        code_lines.append("print(df.head())")
        code_lines.append("print(df.shape)")

        st.code("\n".join(code_lines), language="python")
    else:
        st.info("No preprocessing steps applied yet. Apply steps in the other tabs.")


# ── Before / After comparison ───────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Before / After Comparison")

original = st.session_state["preprocess_original"]
comp = compare_before_after(original, current)
st.dataframe(comp, use_container_width=True, hide_index=True)

# ── Download ────────────────────────────────────────────────────────
st.markdown("---")
csv_buffer = io.StringIO()
current.to_csv(csv_buffer, index=False)
st.download_button(
    label="⬇️ Download preprocessed dataset as CSV",
    data=csv_buffer.getvalue(),
    file_name=f"{name}_preprocessed.csv",
    mime="text/csv",
    type="primary",
)
