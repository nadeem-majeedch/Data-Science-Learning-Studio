"""
Model comparison utilities for Data Science Lab.

Trains multiple models on the same dataset using the same preprocessing
pipeline and consistent train/test splits, then produces comparison
tables and code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.model_training import build_preprocessor, detect_feature_types
from utils.evaluation import compute_classification_metrics, compute_regression_metrics


@dataclass
class ModelRow:
    """One row of a comparison table."""

    name: str
    metrics: dict[str, float]
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None


@dataclass
class ComparisonResult:
    """Complete comparison across multiple models."""

    task: str  # "classification" or "regression"
    rows: list[ModelRow]
    table: pd.DataFrame
    code: str


# ═════════════════════════════════════════════════════════════════════
#  CLASSIFICATION COMPARISON
# ═════════════════════════════════════════════════════════════════════


def compare_classifiers(
    df: pd.DataFrame,
    target: str,
    *,
    model_keys: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = "standard",
) -> ComparisonResult:
    """Train all (or selected) classifiers and compare metrics."""
    from utils.models import CLASSIFIERS

    if model_keys is None:
        model_keys = list(CLASSIFIERS.keys())

    y = df[target]
    X = df.drop(columns=[target])
    num_cols, cat_cols = detect_feature_types(df, target)

    stratify = y if y.nunique() <= 50 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols, scaler=scaler)
    preprocessor.fit(X_train)

    rows: list[ModelRow] = []
    import_code_lines: list[str] = []

    for key in model_keys:
        if key not in CLASSIFIERS:
            continue
        info = CLASSIFIERS[key]
        params = {**info.default_params}

        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", info.sklearn_class(**params)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Try to get probabilities
        y_prob = None
        try:
            y_prob = pipeline.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
        except Exception:
            pass

        m = compute_classification_metrics(y_test.values, y_pred, y_prob)
        metric_dict = {
            "Accuracy": m.accuracy,
            "Precision": m.precision,
            "Recall": m.recall,
            "F1": m.f1,
        }
        if m.roc_auc is not None:
            metric_dict["AUC"] = m.roc_auc

        rows.append(ModelRow(name=info.name, metrics=metric_dict, y_test=y_test.values, y_pred=y_pred))

        import_path = f"sklearn.{info.sklearn_class.__module__.split('.')[-2]}.{info.sklearn_class.__name__}"
        import_code_lines.append(f"from {import_path} import {info.sklearn_class.__name__}")

    table = pd.DataFrame([{"Model": r.name, **r.metrics} for r in rows])
    table = table.sort_values("F1", ascending=False).reset_index(drop=True)

    code = _comparison_code_class(import_code_lines, num_cols, cat_cols, target, test_size, random_state)

    return ComparisonResult(task="classification", rows=rows, table=table, code=code)


# ═════════════════════════════════════════════════════════════════════
#  REGRESSION COMPARISON
# ═════════════════════════════════════════════════════════════════════


def compare_regressors(
    df: pd.DataFrame,
    target: str,
    *,
    model_keys: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = "standard",
) -> ComparisonResult:
    """Train all (or selected) regressors and compare metrics."""
    from utils.regression_models import REGRESSORS

    if model_keys is None:
        model_keys = list(REGRESSORS.keys())

    y = df[target]
    X = df.drop(columns=[target])
    num_cols, cat_cols = detect_feature_types(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols, scaler=scaler)
    preprocessor.fit(X_train)

    rows: list[ModelRow] = []
    import_code_lines: list[str] = []

    for key in model_keys:
        if key not in REGRESSORS:
            continue
        info = REGRESSORS[key]
        params = {**info.default_params}

        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", info.sklearn_class(**params)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        m = compute_regression_metrics(y_test.values, y_pred)
        metric_dict = {
            "R²": m.r2,
            "MAE": m.mae,
            "RMSE": m.rmse,
        }

        rows.append(ModelRow(name=info.name, metrics=metric_dict, y_test=y_test.values, y_pred=y_pred))

        import_path = f"sklearn.{info.sklearn_class.__module__.split('.')[-2]}.{info.sklearn_class.__name__}"
        import_code_lines.append(f"from {import_path} import {info.sklearn_class.__name__}")

    table = pd.DataFrame([{"Model": r.name, **r.metrics} for r in rows])
    table = table.sort_values("R²", ascending=False).reset_index(drop=True)

    code = _comparison_code_reg(import_code_lines, num_cols, cat_cols, target, test_size, random_state)

    return ComparisonResult(task="regression", rows=rows, table=table, code=code)


# ── Code generation ─────────────────────────────────────────────────


def _comparison_code_class(imports, num_cols, cat_cols, target, test_size, rs) -> str:
    lines = [
        "import pandas as pd",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.metrics import classification_report",
        "",
    ]
    lines.extend(imports)
    lines += [
        "",
        f"df = pd.read_csv('your_dataset.csv')",
        f"X = df.drop(columns=['{target}'])",
        f"y = df['{target}']",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={test_size}, random_state={rs}, stratify=y)",
        "",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),",
        "                      ('scaler', StandardScaler())]), num_cols),",
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),",
        "                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)",
        "])",
        "",
        "results = {}",
        "for name, model_cls, params in models:",
        "    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model_cls(**params))])",
        "    pipe.fit(X_train, y_train)",
        "    y_pred = pipe.predict(X_test)",
        "    print(f'{name}: {classification_report(y_test, y_pred)}')",
    ]
    return "\n".join(lines)


def _comparison_code_reg(imports, num_cols, cat_cols, target, test_size, rs) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error",
        "",
    ]
    lines.extend(imports)
    lines += [
        "",
        f"df = pd.read_csv('your_dataset.csv')",
        f"X = df.drop(columns=['{target}'])",
        f"y = df['{target}']",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={test_size}, random_state={rs})",
        "",
        "# Train and compare multiple regressors",
        "results = {}",
        "for name, model_cls, params in models:",
        "    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model_cls(**params))])",
        "    pipe.fit(X_train, y_train)",
        "    y_pred = pipe.predict(X_test)",
        "    print(f'{name}: R2={r2_score(y_test, y_pred):.4f}, MAE={mean_absolute_error(y_test, y_pred):.4f}')",
    ]
    return "\n".join(lines)
