"""
Educational AutoML engine for the Data Science Lab.

Orchestrates a complete machine-learning workflow — from raw data
to a ranked model comparison — using only algorithms already
implemented in the application.  Every step is explainable,
reproducible, and generates Python code.

Workflow::

    Dataset → Detect task → Validate → Preprocess → Split
    → Train multiple models → Evaluate → Rank → Report
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.model_training import build_preprocessor, detect_feature_types


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of data validation checks."""
    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_missing: int = 0
    n_duplicates: int = 0
    n_numeric: int = 0
    n_categorical: int = 0


@dataclass
class AutoMLModelResult:
    """Result of training a single model inside AutoML."""
    name: str
    key: str
    metrics: dict[str, float]
    train_time: float
    pipeline: Pipeline | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None


@dataclass
class AutoMLReport:
    """Complete AutoML run report."""
    task: str
    dataset_name: str
    target: str
    validation: ValidationResult
    model_results: list[AutoMLModelResult]
    comparison_table: pd.DataFrame
    best_model: AutoMLModelResult
    primary_metric: str
    code: str
    total_time: float


# ── Task detection ──────────────────────────────────────────────────

def detect_task_type(
    df: pd.DataFrame,
    target: str,
    max_unique_ratio: float = 0.05,
    max_unique_classes: int = 30,
) -> tuple[str, str]:
    """
    Heuristically detect whether a target column suggests
    classification or regression.

    Returns (task_type, reason).
    """
    y = df[target]

    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(y):
        return "classification", "Target column is text/categorical."

    if pd.api.types.is_bool_dtype(y):
        return "classification", "Target column is boolean."

    n_unique = y.nunique()
    n_total = len(y)

    if n_unique <= max_unique_classes:
        ratio = n_unique / n_total
        if ratio <= max_unique_ratio:
            return (
                "classification",
                f"Target has {n_unique} unique values "
                f"({ratio:.1%} of rows) — likely discrete classes.",
            )

    return "regression", f"Target has {n_unique} unique continuous values."


# ── Data validation ─────────────────────────────────────────────────

MAX_ROWS = 500_000  # safeguard against freezing
MAX_COLS = 500


def validate_dataset(
    df: pd.DataFrame,
    target: str,
    max_rows: int = MAX_ROWS,
    max_cols: int = MAX_COLS,
) -> ValidationResult:
    """Run validation checks and return warnings/errors."""
    result = ValidationResult(valid=True)
    result.n_rows = df.shape[0]
    result.n_cols = df.shape[1]
    result.n_missing = int(df.isnull().sum().sum())
    result.n_duplicates = int(df.duplicated().sum())
    cols_excl_target = [c for c in df.columns if c != target]
    df_features = df[cols_excl_target]
    result.n_numeric = len(df_features.select_dtypes("number").columns)
    result.n_categorical = len(df_features.select_dtypes(include=["object", "category", "string"]).columns)

    if df.shape[0] > max_rows:
        result.warnings.append(
            f"⚠️ Dataset has {df.shape[0]:,} rows (limit: {max_rows:,}). "
            "Large datasets may cause slow execution. Consider sampling."
        )

    if df.shape[1] > max_cols:
        result.warnings.append(
            f"⚠️ Dataset has {df.shape[1]} columns. Consider selecting key features."
        )

    if target not in df.columns:
        result.errors.append(f"❌ Target column '{target}' not found in the dataset.")
        result.valid = False
        return result

    if df[target].isnull().all():
        result.errors.append(f"❌ Target column '{target}' is entirely missing.")
        result.valid = False
        return result

    missing_pct = df.isnull().mean().mean()
    if missing_pct > 0.3:
        result.warnings.append(
            f"⚠️ {missing_pct:.1%} of values are missing. "
            "Results may be unreliable."
        )

    dup_pct = result.n_duplicates / max(df.shape[0], 1)
    if dup_pct > 0.1:
        result.warnings.append(
            f"⚠️ {dup_pct:.1%} of rows are duplicates. "
            "Consider removing them."
        )

    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if not num_cols and not cat_cols:
        result.errors.append("❌ No usable features found (all non-numeric).")
        result.valid = False

    return result


# ── Model selection ─────────────────────────────────────────────────

def _get_classifier_keys() -> list[str]:
    from utils.models import CLASSIFIERS
    return list(CLASSIFIERS.keys())


def _get_regressor_keys() -> list[str]:
    from utils.regression_models import REGRESSORS
    return list(REGRESSORS.keys())


# ── Core AutoML runner ──────────────────────────────────────────────

def run_automl(
    df: pd.DataFrame,
    target: str,
    task: str,
    dataset_name: str = "dataset",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = "standard",
    model_keys: list[str] | None = None,
    max_models: int = 7,
    progress_callback: Any = None,
) -> AutoMLReport:
    """
    Run the full AutoML workflow.

    Parameters
    ----------
    df : full DataFrame (including target)
    target : target column name
    task : "classification" or "regression"
    dataset_name : for the report header
    test_size : test split fraction
    random_state : reproducibility seed
    scaler : preprocessing scaler
    model_keys : subset of models to try (None = all)
    max_models : hard cap on number of models
    progress_callback : callable(current, total, message) for progress updates

    Returns
    -------
    AutoMLReport
    """
    t0 = time.time()

    def _progress(current: int, total: int, msg: str) -> None:
        if progress_callback:
            progress_callback(current, total, msg)

    # ── Validate ────────────────────────────────────────────────────
    _progress(0, 6, "Validating dataset...")
    validation = validate_dataset(df, target)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))

    # ── Feature detection ───────────────────────────────────────────
    _progress(1, 6, "Detecting features...")
    num_cols, cat_cols = detect_feature_types(df, target)

    y = df[target]
    X = df.drop(columns=[target])

    # ── Split ───────────────────────────────────────────────────────
    _progress(2, 6, "Splitting data...")
    stratify = y if (task == "classification" and y.nunique() <= 50) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    # ── Preprocessor ────────────────────────────────────────────────
    _progress(3, 6, "Building preprocessing pipeline...")
    preprocessor = build_preprocessor(num_cols, cat_cols, scaler=scaler)

    # ── Model selection ─────────────────────────────────────────────
    if task == "classification":
        from utils.models import CLASSIFIERS
        registry = CLASSIFIERS
        keys = model_keys or _get_classifier_keys()
        step_name = "classifier"
    else:
        from utils.regression_models import REGRESSORS
        registry = REGRESSORS
        keys = model_keys or _get_regressor_keys()
        step_name = "regressor"

    keys = keys[:max_models]

    # ── Train & evaluate ────────────────────────────────────────────
    model_results: list[AutoMLModelResult] = []
    total_models = len(keys)

    for i, key in enumerate(keys):
        if key not in registry:
            continue

        info = registry[key]
        _progress(4, 6, f"Training model {i + 1}/{total_models}: {info.name}")

        params = {**info.default_params}

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (step_name, info.sklearn_class(**params)),
        ])

        t_model = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        train_time = round(time.time() - t_model, 3)

        # Compute metrics
        metrics = _compute_metrics(y_test, y_pred, task)

        model_results.append(AutoMLModelResult(
            name=info.name,
            key=key,
            metrics=metrics,
            train_time=train_time,
            pipeline=pipeline,
            y_test=y_test.values,
            y_pred=y_pred,
        ))

    # ── Rank ────────────────────────────────────────────────────────
    _progress(5, 6, "Ranking models...")

    if task == "classification":
        primary = "F1"
    else:
        primary = "R²"

    # Build comparison table
    rows = []
    for mr in sorted(model_results, key=lambda m: m.metrics.get(primary, 0), reverse=True):
        row = {"Model": mr.name, **mr.metrics, "Time (s)": mr.train_time}
        rows.append(row)

    table = pd.DataFrame(rows)

    best = model_results[0] if model_results else None
    # Re-find best by primary metric
    if model_results:
        best = max(model_results, key=lambda m: m.metrics.get(primary, 0))

    # ── Generate code ───────────────────────────────────────────────
    _progress(6, 6, "Generating code...")
    code = _generate_automl_code(
        task, num_cols, cat_cols, target, test_size, random_state,
        scaler, keys, registry,
    )

    total_time = round(time.time() - t0, 2)

    return AutoMLReport(
        task=task,
        dataset_name=dataset_name,
        target=target,
        validation=validation,
        model_results=model_results,
        comparison_table=table,
        best_model=best,  # type: ignore[arg-type]
        primary_metric=primary,
        code=code,
        total_time=total_time,
    )


# ── Metric computation ──────────────────────────────────────────────

def _compute_metrics(y_test: pd.Series, y_pred: np.ndarray, task: str) -> dict[str, float]:
    """Compute task-appropriate metrics."""
    if task == "classification":
        n_classes = y_test.nunique()
        avg = "binary" if n_classes == 2 else "weighted"
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }
    else:
        y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
        return {
            "R²": round(r2_score(y_test_arr, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test_arr, y_pred), 4),
            "MSE": round(mean_squared_error(y_test_arr, y_pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test_arr, y_pred))), 4),
        }


# ── Best model explanation ──────────────────────────────────────────

def explain_best_model(report: AutoMLReport) -> str:
    """Generate a human-readable explanation of why the best model ranked highly."""
    best = report.best_model
    primary = report.primary_metric
    score = best.metrics.get(primary, 0)

    lines = [
        f"### 🏆 Best Model: {best.name}",
        f"**Primary metric ({primary}):** {score}",
        "",
    ]

    # Timing context
    avg_time = np.mean([mr.train_time for mr in report.model_results])
    if best.train_time < avg_time:
        lines.append(f"✅ Also one of the faster models ({best.train_time:.2f}s vs avg {avg_time:.2f}s).")
    else:
        lines.append(f"⏱️ Trained in {best.train_time:.2f}s (avg: {avg_time:.2f}s).")

    # Compare to second-best
    others = [m for m in report.model_results if m.key != best.key]
    if others:
        second = max(others, key=lambda m: m.metrics.get(primary, 0))
        diff = score - second.metrics.get(primary, 0)
        if diff > 0.05:
            lines.append(f"📈 Significantly ahead of {second.name} (+{diff:.3f} on {primary}).")
        elif diff > 0.01:
            lines.append(f"📊 Narrowly ahead of {second.name} (+{diff:.3f} on {primary}).")
        else:
            lines.append(f"⚖️ Very close to {second.name} ({diff:+.3f} on {primary}). Consider both.")

    # Caveats
    lines.extend([
        "",
        "> **⚠️ Important:** The 'best' model depends on the metric, dataset, and task. ",
        "> A model that ranks highest on F1 may not be best for production. ",
        "> Always consider interpretability, speed, and domain requirements.",
    ])

    return "\n".join(lines)


# ── Code generation ─────────────────────────────────────────────────

def _generate_automl_code(
    task: str,
    num_cols: list[str],
    cat_cols: list[str],
    target: str,
    test_size: float,
    random_state: int,
    scaler: str | None,
    model_keys: list[str],
    registry: dict,
) -> str:
    """Generate a complete AutoML Python script."""
    lines = [
        '"""',
        "AutoML Workflow — Generated by Data Science Lab",
        '"""',
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "",
    ]

    # Import models
    for key in model_keys:
        if key in registry:
            info = registry[key]
            mod = info.sklearn_class.__module__
            cls_name = info.sklearn_class.__name__
            lines.append(f"from {mod} import {cls_name}")

    lines.extend([
        "",
        "# ── Load data ──",
        "df = pd.read_csv('your_dataset.csv')  # replace with your file",
        "",
        f"# Target: {target}",
        f"X = df.drop(columns=['{target}'])",
        f"y = df['{target}']",
        "",
        "# ── Split ──",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={test_size}, random_state={random_state}",
        f")",
        "",
        "# ── Preprocessing ──",
        f"num_cols = {num_cols!r}",
        f"cat_cols = {cat_cols!r}",
        "",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([",
        "        ('imputer', SimpleImputer(strategy='median')),",
    ])

    if scaler == "standard":
        lines.append("        ('scaler', StandardScaler()),")
    lines.extend([
        "    ]), num_cols),",
        "    ('cat', Pipeline([",
        "        ('imputer', SimpleImputer(strategy='most_frequent')),",
        "        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),",
        "    ]), cat_cols),",
        "])",
        "",
        "# ── Train & compare models ──",
        "models = {}",
        "",
    ])

    for key in model_keys:
        if key in registry:
            info = registry[key]
            cls_name = info.sklearn_class.__name__
            param_str = ", ".join(f"{k}={v!r}" for k, v in info.default_params.items())
            step = "classifier" if task == "classification" else "regressor"
            lines.extend([
                f"# {info.name}",
                f"pipe_{key} = Pipeline([",
                f"    ('preprocessor', preprocessor),",
                f"    ('{step}', {cls_name}({param_str}))",
                f"])",
                f"pipe_{key}.fit(X_train, y_train)",
                f"models['{info.name}'] = pipe_{key}",
                "",
            ])

    if task == "classification":
        lines.extend([
            "# ── Evaluate ──",
            "from sklearn.metrics import classification_report",
            "",
            "for name, pipe in models.items():",
            "    y_pred = pipe.predict(X_test)",
            "    print(f'\\n=== {name} ===')",
            "    print(classification_report(y_test, y_pred))",
        ])
    else:
        lines.extend([
            "# ── Evaluate ──",
            "from sklearn.metrics import r2_score, mean_absolute_error",
            "",
            "for name, pipe in models.items():",
            "    y_pred = pipe.predict(X_test)",
            "    print(f'{name}: R²={r2_score(y_test, y_pred):.4f}, MAE={mean_absolute_error(y_test, y_pred):.4f}')",
        ])

    return "\n".join(lines)
