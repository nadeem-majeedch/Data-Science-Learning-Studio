"""
AutoML — Automate model selection, tuning, and pipeline construction.

Educational AutoML that walks students through the full workflow
from raw data to a ranked model comparison, using only algorithms
already implemented in the application.
"""

import pandas as pd
import streamlit as st

from utils.automl import (
    AutoMLReport,
    detect_task_type,
    explain_best_model,
    run_automl,
    validate_dataset,
)
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="AutoML", page_icon="🤖", layout="wide")
build_sidebar()
page_header("automl")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning("⚠️ No dataset loaded. Go to **Dataset Explorer** and upload a dataset first.")
    st.stop()

st.success(f"🤖 AutoML on: **{name}** ({df.shape[0]:,} rows × {df.shape[1]} cols)")

# ── Step 1: Problem type detection ──────────────────────────────────
st.markdown("### 📋 Step 1: Problem Type")

auto_task, reason = detect_task_type(df, target=df.columns[-1])
st.info(f"💡 **Auto-detected:** {auto_task.title()} — {reason}")

task_options = ["classification", "regression"]
task = st.radio(
    "Confirm or change the problem type:",
    task_options,
    index=task_options.index(auto_task),
    horizontal=True,
    help="AutoML used heuristics to guess the task type. Override if needed.",
)

# ── Step 2: Target selection ────────────────────────────────────────
st.markdown("### 🎯 Step 2: Target Column")

# Suggest the last column, or the one that best matches the detected task
all_cols = df.columns.tolist()
target = st.selectbox(
    "Select the target column",
    all_cols,
    index=len(all_cols) - 1,
    help="The column you want to predict.",
)

# Feature selection
feature_cols = [c for c in all_cols if c != target]
selected_features = st.multiselect(
    "Feature columns (leave empty for all)",
    feature_cols,
    default=[],
    help="Optionally restrict which features are used. Leave empty for all.",
)

if not selected_features:
    selected_features = feature_cols

# ── Step 3: Data validation ────────────────────────────────────────
st.markdown("### ✅ Step 3: Data Validation")

validation = validate_dataset(df, target)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{validation.n_rows:,}")
col_b.metric("Features", f"{validation.n_numeric} numeric, {validation.n_categorical} categorical")
col_c.metric("Missing", f"{validation.n_missing:,}")
col_d.metric("Duplicates", f"{validation.n_duplicates:,}")

if validation.warnings:
    for w in validation.warnings:
        st.warning(w)

if not validation.valid:
    for e in validation.errors:
        st.error(e)
    st.stop()

# ── Step 4: Configuration ───────────────────────────────────────────
st.markdown("### ⚙️ Step 4: Configuration")

with st.expander("Configuration options", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random state", value=42, min_value=0, max_value=9999)
    with col3:
        scaler_opt = st.selectbox("Scaler", ["standard", "minmax", "none"], index=0)
        scaler = scaler_opt if scaler_opt != "none" else None

    max_models = st.slider("Maximum models to train", 3, 10, 7, help="Caps the number of models for faster execution.")

# ── Step 5: Run AutoML ─────────────────────────────────────────────
st.markdown("### 🚀 Step 5: Run AutoML")

if st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True):
    # Build the subset dataframe
    automl_df = df[selected_features + [target]].copy()

    # Progress bar
    progress_bar = st.progress(0, text="Starting AutoML...")
    status_text = st.empty()

    def update_progress(current: int, total: int, msg: str) -> None:
        progress_bar.progress(current / total, text=msg)
        status_text.caption(msg)

    with st.spinner("Running AutoML pipeline..."):
        try:
            report = run_automl(
                df=automl_df,
                target=target,
                task=task,
                dataset_name=name,
                test_size=test_size,
                random_state=random_state,
                scaler=scaler,
                max_models=max_models,
                progress_callback=update_progress,
            )
            st.session_state["automl_report"] = report
        except ValueError as e:
            st.error(f"❌ AutoML failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    progress_bar.progress(1.0, text="✅ Complete!")
    st.success(f"✅ AutoML completed in {report.total_time:.1f}s — {len(report.model_results)} models trained.")

# ── Display results ─────────────────────────────────────────────────
report: AutoMLReport | None = st.session_state.get("automl_report")

if report is None:
    st.info("👆 Configure and click **Run AutoML Pipeline** to start.")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Task", report.task.title())
col2.metric("Best Model", report.best_model.name)
col3.metric(report.primary_metric, f"{report.best_model.metrics.get(report.primary_metric, 0):.4f}")
col4.metric("Total Time", f"{report.total_time:.1f}s")

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────
tab_rank, tab_explain, tab_code, tab_report, tab_experiment = st.tabs([
    "📊 Model Ranking",
    "🏆 Best Model",
    "💻 Code",
    "📄 Report",
    "📓 Save Experiment",
])

# ── Tab 1: Model Ranking ───────────────────────────────────────────
with tab_rank:
    st.markdown("### Model Comparison")

    st.dataframe(
        report.comparison_table.style.highlight_max(
            subset=[c for c in report.comparison_table.columns if c not in ("Model", "Time (s)")],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart of primary metric
    chart_data = report.comparison_table[["Model", report.primary_metric]].set_index("Model")
    st.bar_chart(chart_data)

    # Timing chart
    time_data = report.comparison_table[["Model", "Time (s)"]].set_index("Model")
    st.markdown("#### ⏱️ Training Time")
    st.bar_chart(time_data)

    with st.expander("📚 How to read these results"):
        st.markdown(f"""
        **Primary metric:** {report.primary_metric}
        - Models are ranked by this metric (highest = best).

        **Key caveats:**
        - Highest {report.primary_metric} ≠ best real-world model.
        - Consider: inference speed, interpretability, maintenance cost.
        - A simpler model with slightly lower {report.primary_metric} may be preferable.
        - Always validate on data that was not used for selection.
        """)

# ── Tab 2: Best Model Explanation ──────────────────────────────────
with tab_explain:
    explanation = explain_best_model(report)
    st.markdown(explanation)

    # Show full metrics for the best model
    st.markdown("#### Detailed Metrics")
    best_metrics_df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in report.best_model.metrics.items()]
    )
    st.dataframe(best_metrics_df, hide_index=True, use_container_width=True)

    # All model metrics
    with st.expander("📊 All model metrics side by side"):
        all_rows = []
        for mr in report.model_results:
            row = {"Model": mr.name}
            row.update(mr.metrics)
            all_rows.append(row)
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, use_container_width=True)

# ── Tab 3: Code ────────────────────────────────────────────────────
with tab_code:
    st.markdown("### Generated Python Code")
    st.markdown("Copy this code to reproduce the AutoML workflow in a Jupyter notebook.")

    st.code(report.code, language="python")

    st.download_button(
        "📥 Download code as .py file",
        data=report.code,
        file_name="automl_workflow.py",
        mime="text/x-python",
    )

    with st.expander("📚 Code explanation"):
        st.markdown("""
        The generated code:
        1. **Loads data** — replace `'your_dataset.csv'` with your file.
        2. **Splits** — stratified train/test split for reproducibility.
        3. **Preprocesses** — ColumnTransformer with imputation, scaling, and encoding.
        4. **Trains multiple models** — all classifiers/regressors in the registry.
        5. **Evaluates** — full classification or regression metrics.
        """)

# ── Tab 4: Downloadable Report ─────────────────────────────────────
with tab_report:
    st.markdown("### AutoML Report")

    report_text = f"""# AutoML Report
## Dataset: {report.dataset_name}
- **Rows:** {report.validation.n_rows:,}
- **Features:** {report.validation.n_numeric} numeric, {report.validation.n_categorical} categorical
- **Missing values:** {report.validation.n_missing:,}
- **Target:** {report.target}
- **Task:** {report.task}

## Configuration
- **Test size:** {report.comparison_table.get('test_size', 'N/A')}
- **Scaler:** {report.comparison_table.get('scaler', 'N/A')}
- **Models trained:** {len(report.model_results)}

## Results

| Model | {report.primary_metric} | Time (s) |
|-------|---------|----------|
"""
    for _, row in report.comparison_table.iterrows():
        report_text += f"| {row['Model']} | {row[report.primary_metric]:.4f} | {row['Time (s)']:.3f} |\n"

    report_text += f"""
## Best Model: {report.best_model.name}
- **{report.primary_metric}:** {report.best_model.metrics.get(report.primary_metric, 0):.4f}

## All Metrics for Best Model
"""
    for k, v in report.best_model.metrics.items():
        report_text += f"- **{k}:** {v}\n"

    report_text += f"""
## Timing
- **Total time:** {report.total_time:.1f}s
"""

    st.code(report_text, language="markdown")

    st.download_button(
        "📥 Download report as Markdown",
        data=report_text,
        file_name=f"automl_report_{report.dataset_name}.md",
        mime="text/markdown",
    )

# ── Tab 5: Save Experiment ─────────────────────────────────────────
with tab_experiment:
    st.markdown("### 📓 Save Experiment")

    notes = st.text_area(
        "Notes (optional)",
        placeholder="e.g. Baseline run on Iris dataset, all default params",
        height=100,
    )

    if st.button("💾 Save this AutoML run", type="secondary"):
        from utils.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()
        exp_id = tracker.save_experiment(
            dataset_name=report.dataset_name,
            task_type=report.task,
            target=report.target,
            features=selected_features,
            model=f"AutoML: {report.best_model.name}",
            hyperparameters=report.best_model.metrics,
            metrics=report.best_model.metrics,
            preprocessing_steps=[f"scaler={scaler}", f"test_size={test_size}"],
            generated_code=report.code,
            notes=notes or f"AutoML run — {len(report.model_results)} models, best={report.best_model.name}",
        )
        tracker.close()
        st.success(f"✅ Experiment #{exp_id} saved!")
        st.caption("View it in the Experiment Tracker or the experiments.db database.")

    # Show existing experiments
    from utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()
    exp_count = tracker.count()
    if exp_count > 0:
        st.markdown(f"#### 📋 Your saved experiments ({exp_count})")
        experiments = tracker.list_experiments()
        exp_df = pd.DataFrame([
            {
                "ID": e.experiment_id,
                "Date": e.created_at[:10],
                "Dataset": e.dataset_name,
                "Task": e.task_type,
                "Model": e.model,
                "Notes": e.notes[:50] if e.notes else "",
            }
            for e in experiments[:20]
        ])
        st.dataframe(exp_df, hide_index=True, use_container_width=True)

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export all as CSV"):
                csv_path = tracker.export_csv("reports/experiments_export.csv")
                st.success(f"✅ Exported to {csv_path}")
        with col2:
            if st.button("📥 Export all as JSON"):
                json_path = tracker.export_json("reports/experiments_export.json")
                st.success(f"✅ Exported to {json_path}")

    tracker.close()

# ── Educational footer ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Understanding AutoML", expanded=False):
    st.markdown("""
    ### What is AutoML?

    **AutoML** (Automated Machine Learning) automates repetitive ML tasks:
    1. **Task detection** — Is this classification or regression?
    2. **Data validation** — Are there issues that might break training?
    3. **Preprocessing** — Imputation, scaling, encoding.
    4. **Model selection** — Which algorithm works best on this data?
    5. **Evaluation** — How well does each model perform?

    ### Educational AutoML vs Production AutoML

    | Aspect | Educational (this module) | Production |
    |--------|--------------------------|------------|
    | **Goal** | Learn the workflow | Best possible performance |
    | **Scope** | Simple hyperparameters | Extensive tuning |
    | **Speed** | Fast, interactive | Hours/days |
    | **Algorithms** | All from our registry | Includes XGBoost, LightGBM, etc. |
    | **Validation** | Single train/test split | Cross-validation, holdout |

    ### Why not just use AutoML libraries?
    - This module teaches you **what AutoML does internally**.
    - Understanding the pipeline helps you debug and improve models.
    - AutoML libraries like Auto-sklearn, H2O, or TPOT are great for production,
      but the learning happens when you understand each step.
    """)
