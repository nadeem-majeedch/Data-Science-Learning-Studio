"""
Preprocessing utilities for Data Science Lab.

Provides reusable functions that build sklearn ``Pipeline`` and
``ColumnTransformer`` objects so that preprocessing steps can be
safely reused during model training without data leakage.

Every public function also returns a human-readable Python code
string that documents exactly what was done — useful for students
learning the equivalent scikit-learn calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class PreprocessingStep:
    """Records a single preprocessing action for display and code gen."""

    name: str
    description: str
    code: str
    columns_affected: list[str]


@dataclass
class PreprocessingResult:
    """Full result of a preprocessing session."""

    df: pd.DataFrame
    steps: list[PreprocessingStep] = field(default_factory=list)
    pipeline: Pipeline | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    @property
    def code_summary(self) -> str:
        """Concatenate all step code strings into a runnable script."""
        lines = ["import pandas as pd", "from sklearn.pipeline import Pipeline", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"# Step {i}: {step.name}")
            lines.append(step.code)
            lines.append("")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  1. MISSING VALUES
# ═════════════════════════════════════════════════════════════════════


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    *,
    columns: list[str] | None = None,
    fill_value: float | str = 0,
) -> tuple[pd.DataFrame, PreprocessingStep]:
    """Handle missing values using the specified strategy.

    Parameters
    ----------
    df : DataFrame
    strategy : one of
        ``"drop_rows"``, ``"drop_columns"``, ``"mean"``, ``"median"``,
        ``"mode"``, ``"constant"``
    columns : explicit column list; if ``None`` applies to all eligible cols
    fill_value : used when strategy is ``"constant"``

    Returns
    -------
    (DataFrame, PreprocessingStep)
    """
    df = df.copy()
    affected: list[str] = []

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna()
        code = f"df = df.dropna()  # removed {before - len(df)} rows"
        desc = f"Dropped {before - len(df)} rows with missing values."

    elif strategy == "drop_columns":
        if columns is None:
            columns = [c for c in df.columns if df[c].isna().any()]
        df = df.drop(columns=columns)
        affected = columns
        code = f"df = df.drop(columns={columns!r})"
        desc = f"Dropped {len(columns)} column(s) with missing values."

    elif strategy in ("mean", "median"):
        num_cols = columns or df.select_dtypes("number").columns.tolist()
        num_cols = [c for c in num_cols if df[c].isna().any()]
        if num_cols:
            filler = df[num_cols].mean() if strategy == "mean" else df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(filler)
        affected = num_cols
        fn = "mean" if strategy == "mean" else "median"
        code = f"df[{num_cols!r}] = df[{num_cols!r}].fillna(df[{num_cols!r}].{fn}())"
        desc = f"Filled {len(num_cols)} column(s) with {fn}."

    elif strategy == "mode":
        cols = columns or df.columns.tolist()
        cols = [c for c in cols if df[c].isna().any()]
        for c in cols:
            mode_val = df[c].mode()
            if not mode_val.empty:
                df[c] = df[c].fillna(mode_val.iloc[0])
        affected = cols
        code = (
            "for col in columns:\n"
            "    mode_val = df[col].mode()\n"
            "    if not mode_val.empty:\n"
            "        df[col] = df[col].fillna(mode_val.iloc[0])"
        )
        desc = f"Filled {len(cols)} column(s) with mode."

    elif strategy == "constant":
        cols = columns or df.columns.tolist()
        cols = [c for c in cols if df[c].isna().any()]
        df[cols] = df[cols].fillna(fill_value)
        affected = cols
        code = f"df[{cols!r}] = df[{cols!r}].fillna({fill_value!r})"
        desc = f"Filled {len(cols)} column(s) with constant {fill_value!r}."

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df, PreprocessingStep(
        name=f"Missing values ({strategy})",
        description=desc,
        code=code,
        columns_affected=affected,
    )


# ═════════════════════════════════════════════════════════════════════
#  2. DUPLICATES
# ═════════════════════════════════════════════════════════════════════


def detect_duplicates(df: pd.DataFrame) -> tuple[int, str]:
    """Return (count, code_string) of duplicate rows."""
    n = int(df.duplicated().sum())
    code = f"n_duplicates = df.duplicated().sum()  # {n}"
    return n, code


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingStep]:
    """Drop exact duplicate rows, keeping the first occurrence."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = before - len(df)
    return df, PreprocessingStep(
        name="Remove duplicates",
        description=f"Removed {n_removed} duplicate row(s).",
        code=f"df = df.drop_duplicates().reset_index(drop=True)",
        columns_affected=[],
    )


# ═════════════════════════════════════════════════════════════════════
#  3. CATEGORICAL ENCODING
# ═════════════════════════════════════════════════════════════════════


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    *,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, PreprocessingStep]:
    """One-hot encode the given categorical columns.

    Parameters
    ----------
    drop_first : if ``True`` drops the first category to avoid collinearity
        (recommended for linear models).
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    new_cols = [c for c in df.columns if any(c.startswith(f"{col}_") for col in columns)]
    return df, PreprocessingStep(
        name="One-hot encoding",
        description=f"Encoded {len(columns)} column(s) → {len(new_cols)} new binary columns.",
        code=(
            f"df = pd.get_dummies(df, columns={columns!r}, drop_first={drop_first})"
        ),
        columns_affected=columns,
    )


def label_encode(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, PreprocessingStep, dict[str, dict]]:
    """Label-encode categorical columns (ordinal mapping).

    Returns the mapping dict so students can see what each integer represents.
    """
    df = df.copy()
    mappings: dict[str, dict] = {}
    for col in columns:
        uniques = df[col].dropna().unique()
        mapping = {v: i for i, v in enumerate(sorted(uniques, key=str))}
        mappings[col] = mapping
        df[col] = df[col].map(mapping)

    return df, PreprocessingStep(
        name="Label encoding",
        description=f"Label-encoded {len(columns)} column(s) to integers.",
        code=(
            "from sklearn.preprocessing import OrdinalEncoder\n"
            f"encoder = OrdinalEncoder()\n"
            f"df[{columns!r}] = encoder.fit_transform(df[{columns!r}])"
        ),
        columns_affected=columns,
    ), mappings


# ═════════════════════════════════════════════════════════════════════
#  4. NUMERICAL SCALING
# ═════════════════════════════════════════════════════════════════════


_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_features(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, PreprocessingStep]:
    """Scale numerical columns using the chosen scaler.

    Parameters
    ----------
    method : ``"standard"`` | ``"minmax"`` | ``"robust"``
    """
    df = df.copy()
    cls = _SCALER_MAP.get(method)
    if cls is None:
        raise ValueError(f"Unknown scaler: {method}")

    scaler = cls()
    df[columns] = scaler.fit_transform(df[columns])

    return df, PreprocessingStep(
        name=f"Scaling ({method})",
        description=f"Scaled {len(columns)} column(s) using {cls.__name__}.",
        code=(
            f"from sklearn.preprocessing import {cls.__name__}\n"
            f"scaler = {cls.__name__}()\n"
            f"df[{columns!r}] = scaler.fit_transform(df[{columns!r}])"
        ),
        columns_affected=columns,
    )


# ═════════════════════════════════════════════════════════════════════
#  5. OUTLIER HANDLING
# ═════════════════════════════════════════════════════════════════════


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    *,
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Detect outliers using the IQR method.

    Returns a boolean mask DataFrame and a summary dict.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (df[column] < lower) | (df[column] > upper)

    summary = {
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(mask.sum() / len(df) * 100, 2) if len(df) else 0,
    }
    return mask, summary


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    *,
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, PreprocessingStep]:
    """Remove rows that are outliers (IQR method) in ANY of the given columns."""
    df = df.copy()
    before = len(df)
    combined_mask = pd.Series(False, index=df.index)

    for col in columns:
        mask, _ = detect_outliers_iqr(df, col, multiplier=multiplier)
        combined_mask = combined_mask | mask

    df = df[~combined_mask].reset_index(drop=True)
    n_removed = before - len(df)

    return df, PreprocessingStep(
        name="Remove outliers (IQR)",
        description=(
            f"Removed {n_removed} row(s) that were outliers in "
            f"{len(columns)} column(s) (IQR multiplier={multiplier})."
        ),
        code=(
            f"from utils.preprocessing import detect_outliers_iqr\n"
            f"for col in {columns!r}:\n"
            f"    mask, _ = detect_outliers_iqr(df, col, multiplier={multiplier})\n"
            f"    df = df[~mask].reset_index(drop=True)"
        ),
        columns_affected=columns,
    )


# ═════════════════════════════════════════════════════════════════════
#  6. TRAIN / TEST SPLIT
# ═════════════════════════════════════════════════════════════════════


def split_data(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, PreprocessingStep]:
    """Split into train/test sets.

    Returns (X_train, X_test, y_train, y_test, step).
    """
    strat_col = None
    if stratify and target and target in df.columns:
        strat_col = df[target]

    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df.copy()
        y = None

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_col,
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state,
        )
        y_train, y_test = None, None

    strat_info = " with stratification" if stratify else ""
    code = (
        f"from sklearn.model_selection import train_test_split\n"
        f"X = df.drop(columns=[{target!r}])\n"
        f"y = df[{target!r}]\n"
        f"X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X, y, test_size={test_size}, random_state={random_state},"
        f"{' stratify=y' if stratify else ''}\n"
        f")"
    ) if target else (
        f"X_train, X_test = train_test_split(\n"
        f"    df, test_size={test_size}, random_state={random_state}\n"
        f")"
    )

    return X_train, X_test, y_train, y_test, PreprocessingStep(
        name="Train/test split",
        description=(
            f"Split into train ({len(X_train)} rows) and test ({len(X_test)} rows) "
            f"with test_size={test_size}{strat_info}."
        ),
        code=code,
        columns_affected=[],
    )


# ═════════════════════════════════════════════════════════════════════
#  7. BEFORE / AFTER STATISTICS
# ═════════════════════════════════════════════════════════════════════


def compare_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> pd.DataFrame:
    """Return a summary DataFrame comparing shape, missing, dtypes."""
    rows = []
    for label, df in [("Before", df_before), ("After", df_after)]:
        rows.append({
            "Metric": label,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing Cells": int(df.isna().sum().sum()),
            "Memory (KB)": round(df.memory_usage(deep=True).sum() / 1024, 1),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
#  8. BUILD A FULL SKLEARN PIPELINE
# ═════════════════════════════════════════════════════════════════════


def build_sklearn_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    num_strategy: str = "median",
    cat_encoder: str = "onehot",
    scaler: str | None = None,
) -> Pipeline:
    """Build an sklearn ``Pipeline`` with a ``ColumnTransformer``.

    This can later be ``fit`` on training data and ``transform`` on
    test data without leakage.

    Parameters
    ----------
    num_strategy : ``"mean"`` | ``"median"`` | ``"most_frequent"``
    cat_encoder : ``"onehot"`` | ``"ordinal"``
    scaler : ``None`` | ``"standard"`` | ``"minmax"`` | ``"robust"``
    """
    from sklearn.impute import SimpleImputer

    # Numerical sub-pipeline
    num_steps: list[tuple[str, object]] = []
    if num_strategy in ("mean", "median", "most_frequent"):
        num_steps.append(("imputer", SimpleImputer(strategy=num_strategy)))
    if scaler and scaler in _SCALER_MAP:
        num_steps.append(("scaler", _SCALER_MAP[scaler]()))
    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    # Categorical sub-pipeline
    cat_steps: list[tuple[str, object]] = []
    cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    if cat_encoder == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    else:
        cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    cat_pipeline = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_columns),
            ("cat", cat_pipeline, categorical_columns),
        ],
        remainder="passthrough",
    )

    return Pipeline([("preprocessor", preprocessor)])
