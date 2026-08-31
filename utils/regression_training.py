"""
Regression training utilities for Data Science Lab.

Builds sklearn ``Pipeline`` objects that chain preprocessing
(ColumnTransformer) with a regressor, preventing data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class RegressionResult:
    """All outputs of a regression training run."""

    model_name: str
    r2: float
    mae: float
    mse: float
    rmse: float
    y_test: np.ndarray
    y_pred: np.ndarray
    feature_names: list[str]
    pipeline: Pipeline
    code: str


# ── Preprocessing ───────────────────────────────────────────────────


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scaler: str | None = "standard",
) -> ColumnTransformer:
    """Build a leakage-safe ColumnTransformer for regression."""
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))

    cat_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", Pipeline(num_steps), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", Pipeline(cat_steps), categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def detect_feature_types(
    df: pd.DataFrame, target: str
) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) excluding the target."""
    df2 = df.drop(columns=[target], errors="ignore")
    num_cols = df2.select_dtypes("number").columns.tolist()
    cat_cols = df2.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return num_cols, cat_cols


# ── Training ────────────────────────────────────────────────────────


def train_regressor(
    df: pd.DataFrame,
    target: str,
    model_key: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = "standard",
    model_params: dict | None = None,
) -> RegressionResult:
    """Train a regressor inside a Pipeline (preprocessing → model)."""
    from utils.regression_models import REGRESSORS

    info = REGRESSORS[model_key]
    params = {**info.default_params}
    if model_params:
        params.update(model_params)

    y = df[target]
    X = df.drop(columns=[target])
    num_cols, cat_cols = detect_feature_types(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols, scaler=scaler)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", info.sklearn_class(**params)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    feature_names = _get_feature_names(pipeline, num_cols, cat_cols)
    code = _generate_code(info, num_cols, cat_cols, target, test_size, random_state, scaler, params)

    return RegressionResult(
        model_name=info.name,
        r2=round(r2, 4),
        mae=round(mae, 4),
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        y_test=y_test.values,
        y_pred=y_pred,
        feature_names=feature_names,
        pipeline=pipeline,
        code=code,
    )


# ── Feature importance ──────────────────────────────────────────────


def extract_feature_importance(result: RegressionResult) -> pd.DataFrame | None:
    """Extract feature importances or coefficients from the trained model."""
    reg = result.pipeline.named_steps["regressor"]

    if hasattr(reg, "feature_importances_"):
        imp = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        coef = reg.coef_
        if coef.ndim > 1:
            imp = np.mean(np.abs(coef), axis=0)
        else:
            imp = np.abs(coef).flatten()
    else:
        return None

    n = min(len(imp), len(result.feature_names))
    imp = imp[:n]
    names = result.feature_names[:n]

    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ── Code generation ─────────────────────────────────────────────────


def _generate_code(info, num_cols, cat_cols, target, test_size, random_state, scaler, params):
    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        f"from sklearn.{info.sklearn_class.__module__.split('.')[-2]}."
        f"{info.sklearn_class.__name__} import {info.sklearn_class.__name__}",
        "",
        "# Load data",
        "df = pd.read_csv('your_dataset.csv')",
        "",
        f"X = df.drop(columns=['{target}'])",
        f"y = df['{target}']",
        "",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={test_size}, random_state={random_state}",
        f")",
        "",
        "# Preprocessing",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),"
        + (f" ('scaler', StandardScaler())])" if scaler else "]))"),
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),",
        "                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]))",
        "])",
        "",
        "# Full pipeline",
        "pipeline = Pipeline([",
        "    ('preprocessor', preprocessor),",
        f"    ('regressor', {info.sklearn_class.__name__}({param_str}))",
        "])",
        "",
        "pipeline.fit(X_train, y_train)",
        "y_pred = pipeline.predict(X_test)",
        "",
        "# Evaluate",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "print(f'R²:   {r2_score(y_test, y_pred):.4f}')",
        "print(f'MAE:  {mean_absolute_error(y_test, y_pred):.4f}')",
        "print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')",
    ]
    return "\n".join(lines)


def _get_feature_names(pipeline, num_cols, cat_cols):
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            return preprocessor.get_feature_names_out().tolist()
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                names.extend(cols if isinstance(cols, list) else num_cols + cat_cols)
            else:
                names.extend(cols if isinstance(cols, list) else list(cols))
        return names
    except Exception:
        return num_cols + cat_cols
