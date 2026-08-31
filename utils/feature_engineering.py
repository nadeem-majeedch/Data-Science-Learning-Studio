"""
Feature engineering utilities for Data Science Lab.

Pure functions that create, transform, and select features.  Every
function returns ``(pd.DataFrame, FeatureStep)`` so the page can
display before/after data and generated code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ── Data class ──────────────────────────────────────────────────────


@dataclass
class FeatureStep:
    """Records a single feature-engineering action."""

    name: str
    description: str
    code: str
    columns_created: list[str]
    columns_dropped: list[str] | None = None


# ═════════════════════════════════════════════════════════════════════
#  1. MATHEMATICAL TRANSFORMATIONS
# ═════════════════════════════════════════════════════════════════════


def apply_math_transform(
    df: pd.DataFrame,
    col: str,
    transform: Literal["log", "sqrt", "square"],
) -> tuple[pd.DataFrame, FeatureStep]:
    """Apply a mathematical transformation and add as a new column.

    Parameters
    ----------
    transform : ``"log"`` | ``"sqrt"`` | ``"square"``
    """
    df = df.copy()
    new_col = f"{col}_{transform}"

    if transform == "log":
        # Shift positive so log is defined
        offset = 0.0
        if df[col].min() <= 0:
            offset = abs(df[col].min()) + 1
        df[new_col] = np.log(df[col] + offset)
        code = (
            f"# Log transform (offset={offset:.2f} added to handle non-positive values)\n"
            f"df['{new_col}'] = np.log(df['{col}'] + {offset})"
        )
    elif transform == "sqrt":
        offset = 0.0
        if df[col].min() < 0:
            offset = abs(df[col].min())
        df[new_col] = np.sqrt(df[col] + offset)
        code = (
            f"# Square root transform (offset={offset:.2f} for non-negative input)\n"
            f"df['{new_col}'] = np.sqrt(df['{col}'] + {offset})"
        )
    elif transform == "square":
        df[new_col] = df[col] ** 2
        code = f"df['{new_col}'] = df['{col}'] ** 2"
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return df, FeatureStep(
        name=f"Math transform ({transform})",
        description=f"Created '{new_col}' by applying {transform} to '{col}'.",
        code=code,
        columns_created=[new_col],
    )


# ═════════════════════════════════════════════════════════════════════
#  2. BINNING
# ═════════════════════════════════════════════════════════════════════


def bin_numerical(
    df: pd.DataFrame,
    col: str,
    *,
    n_bins: int = 5,
    method: Literal["equal_width", "equal_freq"] = "equal_width",
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureStep]:
    """Bin a numerical column into discrete intervals.

    Parameters
    ----------
    n_bins : number of bins
    method : ``"equal_width"`` (uniform intervals) or ``"equal_freq"`` (quantile)
    """
    df = df.copy()
    new_col = f"{col}_binned"

    if method == "equal_freq":
        df[new_col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
    else:
        df[new_col] = pd.cut(df[col], bins=n_bins, labels=labels)

    return df, FeatureStep(
        name=f"Binning ({method})",
        description=f"Binned '{col}' into {n_bins} bins → '{new_col}'.",
        code=(
            f"df['{new_col}'] = pd.cut(df['{col}'], bins={n_bins})"
            if method == "equal_width"
            else f"df['{new_col}'] = pd.qcut(df['{col}'], q={n_bins}, duplicates='drop')"
        ),
        columns_created=[new_col],
    )


# ═════════════════════════════════════════════════════════════════════
#  3. DATE / TIME FEATURES
# ═════════════════════════════════════════════════════════════════════


def extract_date_features(
    df: pd.DataFrame,
    col: str,
    *,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureStep]:
    """Extract date/time components from a datetime column.

    Available features: ``year``, ``month``, ``day``, ``weekday``,
    ``hour``, ``quarter``.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if features is None:
        features = ["year", "month", "day", "weekday"]

    extractors = {
        "year": ("year", lambda s: s.dt.year),
        "month": ("month", lambda s: s.dt.month),
        "day": ("day", lambda s: s.dt.day),
        "weekday": ("weekday", lambda s: s.dt.weekday),
        "hour": ("hour", lambda s: s.dt.hour),
        "quarter": ("quarter", lambda s: s.dt.quarter),
    }

    created: list[str] = []
    code_lines = [f"df['{col}'] = pd.to_datetime(df['{col}'])"]
    for feat in features:
        if feat in extractors:
            suffix, fn = extractors[feat]
            new_col = f"{col}_{suffix}"
            df[new_col] = fn(df[col])
            created.append(new_col)
            code_lines.append(f"df['{new_col}'] = df['{col}'].dt.{suffix}")

    return df, FeatureStep(
        name="Date feature extraction",
        description=f"Extracted {len(created)} date features from '{col}'.",
        code="\n".join(code_lines),
        columns_created=created,
    )


# ═════════════════════════════════════════════════════════════════════
#  4. STRING / TEXT FEATURES
# ═════════════════════════════════════════════════════════════════════


def extract_text_features(
    df: pd.DataFrame,
    col: str,
    *,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureStep]:
    """Extract basic text features from a string column.

    Available features: ``length``, ``word_count``, ``uppercase_ratio``.
    """
    df = df.copy()
    series = df[col].astype(str)

    if features is None:
        features = ["length", "word_count"]

    extractors = {
        "length": ("len", lambda s: s.str.len()),
        "word_count": ("words", lambda s: s.str.split().str.len()),
        "uppercase_ratio": ("upper_ratio", lambda s: s.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )),
    }

    created: list[str] = []
    code_lines: list[str] = []
    for feat in features:
        if feat in extractors:
            suffix, fn = extractors[feat]
            new_col = f"{col}_{suffix}"
            df[new_col] = fn(series)
            created.append(new_col)
            code_lines.append(f"df['{new_col}'] = df['{col}'].astype(str).str.{suffix.replace('_', ' ')}")

    return df, FeatureStep(
        name="Text feature extraction",
        description=f"Extracted {len(created)} text features from '{col}'.",
        code="\n".join(code_lines) if code_lines else f"# text features from '{col}'",
        columns_created=created,
    )


# ═════════════════════════════════════════════════════════════════════
#  5. INTERACTION FEATURES
# ═════════════════════════════════════════════════════════════════════


def create_interaction(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    operation: Literal["multiply", "add", "subtract", "divide"] = "multiply",
) -> tuple[pd.DataFrame, FeatureStep]:
    """Create an interaction feature from two columns."""
    df = df.copy()
    ops = {
        "multiply": ("*", lambda a, b: a * b),
        "add": ("+", lambda a, b: a + b),
        "subtract": ("-", lambda a, b: a - b),
        "divide": ("/", lambda a, b: a / b.replace(0, np.nan)),
    }
    symbol, fn = ops[operation]
    new_col = f"{col_a}_{operation}_{col_b}"
    df[new_col] = fn(df[col_a].astype(float), df[col_b].astype(float))

    return df, FeatureStep(
        name=f"Interaction ({operation})",
        description=f"Created '{new_col}' = {col_a} {symbol} {col_b}.",
        code=f"df['{new_col}'] = df['{col_a}'] {symbol} df['{col_b}']",
        columns_created=[new_col],
    )


# ═════════════════════════════════════════════════════════════════════
#  6. POLYNOMIAL FEATURES
# ═════════════════════════════════════════════════════════════════════


def create_polynomial(
    df: pd.DataFrame,
    col: str,
    *,
    degree: int = 2,
) -> tuple[pd.DataFrame, FeatureStep]:
    """Create polynomial features (x², x³, …) for a single column."""
    df = df.copy()
    created: list[str] = []
    code_lines: list[str] = []

    for d in range(2, degree + 1):
        new_col = f"{col}_pow{d}"
        df[new_col] = df[col] ** d
        created.append(new_col)
        code_lines.append(f"df['{new_col}'] = df['{col}'] ** {d}")

    return df, FeatureStep(
        name=f"Polynomial (degree={degree})",
        description=f"Created {len(created)} polynomial feature(s) from '{col}'.",
        code="\n".join(code_lines),
        columns_created=created,
    )


# ═════════════════════════════════════════════════════════════════════
#  7. FEATURE SELECTION — VARIANCE THRESHOLD
# ═════════════════════════════════════════════════════════════════════


def variance_threshold_select(
    df: pd.DataFrame,
    *,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, FeatureStep, list[str]]:
    """Drop numerical columns with variance below the threshold.

    Returns (df, step, dropped_columns).
    """
    df = df.copy()
    num_df = df.select_dtypes("number")
    variances = num_df.var()
    low_var = variances[variances < threshold].index.tolist()

    df = df.drop(columns=low_var, errors="ignore")

    return df, FeatureStep(
        name=f"Variance threshold ({threshold})",
        description=f"Dropped {len(low_var)} low-variance column(s).",
        code=(
            f"from sklearn.feature_selection import VarianceThreshold\n"
            f"selector = VarianceThreshold(threshold={threshold})\n"
            f"selector.fit(df.select_dtypes('number'))\n"
            f"keep = selector.get_support()\n"
            f"df = df[keep_cols + non_numeric_cols]"
        ),
        columns_created=[],
        columns_dropped=low_var,
    ), low_var


# ═════════════════════════════════════════════════════════════════════
#  8. FEATURE SELECTION — CORRELATION-BASED
# ═════════════════════════════════════════════════════════════════════


def correlation_select(
    df: pd.DataFrame,
    *,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, FeatureStep, list[str]]:
    """Drop one of each pair of highly correlated numerical columns.

    Keeps the column that appears first in the DataFrame.
    """
    df = df.copy()
    num_df = df.select_dtypes("number")
    if num_df.shape[1] < 2:
        return df, FeatureStep(
            name="Correlation selection",
            description="Need ≥2 numerical columns.",
            code="# Not enough numerical columns",
            columns_created=[],
        ), []

    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop, errors="ignore")

    return df, FeatureStep(
        name=f"Correlation selection (>{threshold})",
        description=f"Dropped {len(to_drop)} highly correlated column(s).",
        code=(
            f"corr = df.select_dtypes('number').corr().abs()\n"
            f"upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\n"
            f"to_drop = [c for c in upper.columns if any(upper[c] > {threshold})]\n"
            f"df = df.drop(columns=to_drop)"
        ),
        columns_created=[],
        columns_dropped=to_drop,
    ), to_drop


# ═════════════════════════════════════════════════════════════════════
#  9. FEATURE IMPORTANCE (when model available)
# ═════════════════════════════════════════════════════════════════════


def get_feature_importance(
    df: pd.DataFrame,
    target: str,
    *,
    max_features: int = 20,
) -> pd.DataFrame | None:
    """Train a quick RandomForest and return feature importances.

    Returns None if no numeric features or target is missing.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    num_df = df.select_dtypes("number")
    if target not in num_df.columns or num_df.shape[1] < 2:
        return None

    X = num_df.drop(columns=[target])
    y = num_df[target]

    if X.empty:
        return None

    # Choose classifier vs regressor
    if y.nunique() <= 20 and y.dtype in [np.int64, np.float64, int, float]:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(max_features)

    return imp.to_frame("importance").round(4)
