"""
Data analysis utilities for Data Science Lab.

Pure functions that operate on pandas DataFrames and return structured
results.  These are consumed by the Dataset Explorer page and can be
reused by EDA, Preprocessing, and other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Data classes for structured results ─────────────────────────────


@dataclass
class DatasetOverview:
    """High-level summary of a DataFrame."""

    name: str
    n_rows: int
    n_cols: int
    memory_bytes: int
    memory_human: str
    column_names: list[str]
    dtypes: dict[str, str]
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_boolean: int


@dataclass
class MissingValueReport:
    """Missing-value analysis for every column."""

    total_cells: int
    total_missing: int
    percent_missing: float
    per_column: dict[str, int]  # column -> missing count


@dataclass
class DuplicateReport:
    """Duplicate-row analysis."""

    n_duplicates: int
    percent_duplicates: float
    duplicate_indices: list[int]


@dataclass
class UniqueValueReport:
    """Unique-value counts per column."""

    per_column: dict[str, int]


@dataclass
class ConstantColumnReport:
    """Columns that have only one distinct non-null value."""

    columns: list[str]


@dataclass
class DataQualitySummary:
    """Complete data-quality summary combining all checks."""

    overview: DatasetOverview
    missing: MissingValueReport
    duplicates: DuplicateReport
    unique_values: UniqueValueReport
    constant_columns: ConstantColumnReport


# ── Overview ────────────────────────────────────────────────────────


def get_dataset_overview(df: pd.DataFrame, name: str = "dataset") -> DatasetOverview:
    """Compute a high-level overview of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Human-readable name for the dataset.

    Returns
    -------
    DatasetOverview
    """
    memory_bytes = int(df.memory_usage(deep=True).sum())

    dtypes_map = df.dtypes.astype(str).to_dict()
    n_numeric = len(df.select_dtypes("number").columns)
    n_categorical = len(df.select_dtypes(include=["object", "category", "string"]).columns)
    n_datetime = len(df.select_dtypes("datetime").columns)
    n_boolean = len(df.select_dtypes("bool").columns)

    return DatasetOverview(
        name=name,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        memory_bytes=memory_bytes,
        memory_human=_human_bytes(memory_bytes),
        column_names=df.columns.tolist(),
        dtypes=dtypes_map,
        n_numeric=n_numeric,
        n_categorical=n_categorical,
        n_datetime=n_datetime,
        n_boolean=n_boolean,
    )


# ── Missing values ──────────────────────────────────────────────────


def analyse_missing_values(df: pd.DataFrame) -> MissingValueReport:
    """Count missing (NaN / None) values per column and overall.

    Returns
    -------
    MissingValueReport
    """
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    percent = (total_missing / total_cells * 100) if total_cells else 0.0

    per_col = df.isna().sum()
    per_column = {col: int(cnt) for col, cnt in per_col.items() if cnt > 0}

    return MissingValueReport(
        total_cells=total_cells,
        total_missing=total_missing,
        percent_missing=round(percent, 2),
        per_column=per_column,
    )


# ── Duplicates ──────────────────────────────────────────────────────


def analyse_duplicates(df: pd.DataFrame) -> DuplicateReport:
    """Identify duplicate rows.

    Returns
    -------
    DuplicateReport
    """
    mask = df.duplicated(keep="first")
    n_dupes = int(mask.sum())
    n_rows = len(df)
    percent = (n_dupes / n_rows * 100) if n_rows else 0.0
    indices = df.index[mask].tolist()

    return DuplicateReport(
        n_duplicates=n_dupes,
        percent_duplicates=round(percent, 2),
        duplicate_indices=indices[:100],  # cap for display
    )


# ── Unique values ───────────────────────────────────────────────────


def count_unique_values(df: pd.DataFrame) -> UniqueValueReport:
    """Return the number of unique (non-null) values per column.

    Returns
    -------
    UniqueValueReport
    """
    counts = df.nunique(dropna=True)
    per_column = {col: int(cnt) for col, cnt in counts.items()}
    return UniqueValueReport(per_column=per_column)


# ── Constant columns ────────────────────────────────────────────────


def detect_constant_columns(df: pd.DataFrame) -> ConstantColumnReport:
    """Find columns where every non-null value is the same.

    Returns
    -------
    ConstantColumnReport
    """
    constants: list[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            constants.append(col)

    return ConstantColumnReport(columns=constants)


# ── Descriptive statistics ──────────────────────────────────────────


def get_numerical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numerical columns.

    Returns a DataFrame with index = statistics and columns = numeric
    column names.
    """
    num_df = df.select_dtypes("number")
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe().round(3)


def get_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for categorical columns.

    Includes count, unique, top value, and frequency of the top value.
    """
    cat_df = df.select_dtypes(include=["object", "category", "string"])
    if cat_df.empty:
        return pd.DataFrame()
    return cat_df.describe()


# ── Convenience: run all analyses at once ────────────────────────────


def full_quality_analysis(
    df: pd.DataFrame, name: str = "dataset"
) -> DataQualitySummary:
    """Run every analysis and return a single structured report.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Human-readable dataset name.

    Returns
    -------
    DataQualitySummary
    """
    return DataQualitySummary(
        overview=get_dataset_overview(df, name=name),
        missing=analyse_missing_values(df),
        duplicates=analyse_duplicates(df),
        unique_values=count_unique_values(df),
        constant_columns=detect_constant_columns(df),
    )


# ── Private helpers ─────────────────────────────────────────────────


def _human_bytes(n: int) -> str:
    """Convert byte count to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"
