"""
Data loading utilities for Data Science Learning Studio.

Handles CSV, TSV, and XLSX file uploads, as well as loading built-in
sample datasets from the ``datasets/`` directory.  Every function
returns a ``pandas.DataFrame`` and raises clear errors on failure.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

# Absolute path to the datasets directory (one level up from utils/).
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


# ── Upload loaders ──────────────────────────────────────────────────


def load_csv(file: BinaryIO) -> pd.DataFrame:
    """Read a CSV file object into a DataFrame.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode (e.g. from
        ``st.file_uploader``).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file is empty or cannot be parsed.
    """
    content = file.read()
    if not content:
        raise ValueError("The uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc


def load_excel(file: BinaryIO) -> pd.DataFrame:
    """Read an Excel (.xlsx) file object into a DataFrame.

    Parameters
    ----------
    file : BinaryIO
        A file-like object opened in binary mode.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file is empty or cannot be parsed.
    """
    content = file.read()
    if not content:
        raise ValueError("The uploaded file is empty.")
    try:
        return pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not parse Excel file: {exc}") from exc


def load_uploaded_file(file: BinaryIO, filename: str) -> pd.DataFrame:
    """Dispatch to the correct loader based on file extension.

    Supported extensions: ``.csv``, ``.tsv``, ``.xlsx``.

    Parameters
    ----------
    file : BinaryIO
        The uploaded file object.
    filename : str
        The original filename (used to determine the format).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the extension is unsupported or the file cannot be parsed.
    """
    name_lower = filename.lower()
    if name_lower.endswith(".csv"):
        return load_csv(file)
    elif name_lower.endswith(".tsv"):
        file.seek(0)
        content = file.read()
        if not content:
            raise ValueError("The uploaded file is empty.")
        try:
            return pd.read_csv(io.BytesIO(content), sep="\t")
        except Exception as exc:
            raise ValueError(f"Could not parse TSV: {exc}") from exc
    elif name_lower.endswith((".xlsx", ".xls")):
        return load_excel(file)
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. "
            "Please upload a CSV, TSV, or XLSX file."
        )


# ── Sample dataset loaders ──────────────────────────────────────────


def list_sample_datasets() -> list[dict[str, str]]:
    """Return metadata for every CSV file in ``datasets/``.

    Returns
    -------
    list[dict]
        Each dict has keys ``name``, ``path``, and ``description``.
    """
    samples: list[dict[str, str]] = []
    if not DATASETS_DIR.exists():
        return samples

    for csv_path in sorted(DATASETS_DIR.glob("*.csv")):
        samples.append(
            {
                "name": csv_path.stem,
                "path": str(csv_path),
                "description": _dataset_description(csv_path.stem),
            }
        )
    return samples


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset by name (without extension).

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"iris"`` or ``"titanic"``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist in ``datasets/``.
    """
    csv_path = DATASETS_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample dataset '{name}' not found in {DATASETS_DIR}")
    return pd.read_csv(csv_path)


# ── Helpers ─────────────────────────────────────────────────────────


def _dataset_description(name: str) -> str:
    """Return a short human-readable description for a known sample."""
    descriptions = {
        "iris": "Classic Fisher's Iris dataset — 150 flowers across 3 species with 4 measurements.",
        "titanic": "Titanic passenger data — survival prediction based on demographics and ticket info.",
        "wine_quality": "Red wine quality ratings — physicochemical attributes and sensory scores.",
        "breast_cancer": "Wisconsin Breast Cancer dataset — 569 samples, 30 features, malignant vs benign.",
        "california_housing": "California Housing prices — 20k samples, 8 features for median house value prediction.",
    }
    return descriptions.get(name, "Sample dataset for practice.")
