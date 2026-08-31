"""
Model training utilities for Data Science Lab.

Builds sklearn ``Pipeline`` objects that chain preprocessing
(ColumnTransformer) with a classifier, preventing data leakage.
All randomness is controlled via ``random_state`` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class TrainResult:
    """Holds all outputs of a training run."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    report_text: str
    confusion: np.ndarray
    classes: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    feature_names: list[str]
    pipeline: Pipeline
    code: str
    is_binary: bool


# ── Preprocessing builder ──────────────────────────────────────────


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scaler: str | None = "standard",
) -> ColumnTransformer:
    """Build a leakage-safe ColumnTransformer.

    Parameters
    ----------
    scaler : ``None`` | ``"standard"`` | ``"minmax"``
    """
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

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )


# ── Feature column detection ────────────────────────────────────────


def detect_feature_types(
    df: pd.DataFrame,
    target: str,
) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) excluding the target."""
    df = df.drop(columns=[target], errors="ignore")
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return num_cols, cat_cols


# ── Training ────────────────────────────────────────────────────────


def train_classifier(
    df: pd.DataFrame,
    target: str,
    model_key: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = "standard",
    model_params: dict | None = None,
) -> TrainResult:
    """Train a classifier inside a Pipeline (preprocessing → model).

    Parameters
    ----------
    df : full DataFrame (including target)
    target : name of the target column
    model_key : key from ``utils.models.CLASSIFIERS``
    test_size : fraction for the test set
    random_state : for reproducibility
    scaler : preprocessing scaler
    model_params : override default hyperparameters

    Returns
    -------
    TrainResult
    """
    from utils.models import CLASSIFIERS

    info = CLASSIFIERS[model_key]
    params = {**info.default_params}
    if model_params:
        params.update(model_params)

    # Split features / target
    y = df[target]
    X = df.drop(columns=[target])

    num_cols, cat_cols = detect_feature_types(df, target)

    # Train / test split (stratified)
    stratify = y if y.nunique() <= 50 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Build pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols, scaler=scaler)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", info.sklearn_class(**params)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    avg = "binary" if y.nunique() == 2 else "weighted"
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    classes = np.array(sorted(y.unique()))

    # Feature names after preprocessing
    feature_names = _get_feature_names(pipeline, num_cols, cat_cols)

    # Code generation
    code = _generate_code(
        info, num_cols, cat_cols, target, test_size, random_state, scaler, params
    )

    return TrainResult(
        model_name=info.name,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        report_text=report,
        confusion=cm,
        classes=classes,
        y_test=y_test.values,
        y_pred=y_pred,
        feature_names=feature_names,
        pipeline=pipeline,
        code=code,
        is_binary=(y.nunique() == 2),
    )


# ── Feature importance ──────────────────────────────────────────────


def extract_feature_importance(result: TrainResult) -> pd.DataFrame | None:
    """Extract feature importances or coefficients from the trained model.

    Returns None if the model does not support it.
    """
    clf = result.pipeline.named_steps["classifier"]

    # Try feature_importances_ (tree-based)
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    # Try coef_ (linear models)
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        if coef.ndim > 1:
            imp = np.mean(np.abs(coef), axis=0)
        else:
            imp = np.abs(coef).flatten()
    else:
        return None

    n_features = min(len(imp), len(result.feature_names))
    imp = imp[:n_features]
    names = result.feature_names[:n_features]

    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ── Code generation ─────────────────────────────────────────────────


def _generate_code(
    info, num_cols, cat_cols, target, test_size, random_state, scaler, params
) -> str:
    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    lines = [
        "import pandas as pd",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        f"from sklearn.{info.sklearn_class.__module__.split('.')[-2]}."
        f"{info.sklearn_class.__name__} import {info.sklearn_class.__name__}",
        "",
        "# Load data",
        "df = pd.read_csv('your_dataset.csv')  # replace with your data",
        "",
        f"X = df.drop(columns=['{target}'])",
        f"y = df['{target}']",
        "",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={test_size}, random_state={random_state}, stratify=y",
        f")",
        "",
        "# Preprocessing",
        f"num_cols = {num_cols!r}",
        f"cat_cols = {cat_cols!r}",
        "",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),"
        + (f" ('scaler', StandardScaler())])" if scaler else "]))"),
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),",
        "                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]))",
        "])",
        "",
        "# Full pipeline",
        f"pipeline = Pipeline([",
        f"    ('preprocessor', preprocessor),",
        f"    ('classifier', {info.sklearn_class.__name__}({param_str}))",
        f"])",
        "",
        "pipeline.fit(X_train, y_train)",
        "y_pred = pipeline.predict(X_test)",
        "",
        "# Evaluate",
        f"from sklearn.metrics import accuracy_score, classification_report",
        f"print(f'Accuracy: {{accuracy_score(y_test, y_pred):.4f}}')",
        f"print(classification_report(y_test, y_pred))",
    ]
    return "\n".join(lines)


def _get_feature_names(
    pipeline: Pipeline, num_cols: list[str], cat_cols: list[str]
) -> list[str]:
    """Extract feature names from the fitted ColumnTransformer."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        names: list[str] = []

        if hasattr(preprocessor, "get_feature_names_out"):
            all_names = preprocessor.get_feature_names_out()
            return all_names.tolist()

        # Fallback
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                if isinstance(cols, list):
                    names.extend(cols)
                else:
                    names.extend(num_cols + cat_cols)
            else:
                names.extend(cols if isinstance(cols, list) else list(cols))

        return names
    except Exception:
        return num_cols + cat_cols
