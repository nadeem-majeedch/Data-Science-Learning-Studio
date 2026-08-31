"""
Tests for utils.data_loader — file loading and sample dataset functions.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from utils.data_loader import (
    DATASETS_DIR,
    list_sample_datasets,
    load_csv,
    load_excel,
    load_sample_dataset,
    load_uploaded_file,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_csv_bytes(content: str) -> io.BytesIO:
    """Wrap a CSV string in a BytesIO for upload simulation."""
    return io.BytesIO(content.encode("utf-8"))


# ── load_csv ────────────────────────────────────────────────────────


class TestLoadCSV:
    def test_load_simple_csv(self):
        csv = _make_csv_bytes("a,b,c\n1,2,3\n4,5,6")
        df = load_csv(csv)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["a", "b", "c"]

    def test_empty_file_raises(self):
        csv = io.BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            load_csv(csv)

    def test_invalid_csv_raises(self):
        # A binary blob that isn't valid CSV
        csv = io.BytesIO(b"\xff\xfe\x00\x01")
        with pytest.raises(ValueError, match="Could not parse CSV"):
            load_csv(csv)


# ── load_excel (basic) ──────────────────────────────────────────────


class TestLoadExcel:
    def test_empty_file_raises(self):
        xlsx = io.BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            load_excel(xlsx)


# ── load_uploaded_file ──────────────────────────────────────────────


class TestLoadUploadedFile:
    def test_csv_dispatch(self):
        csv = _make_csv_bytes("x,y\n1,2")
        df = load_uploaded_file(csv, "data.csv")
        assert list(df.columns) == ["x", "y"]

    def test_tsv_dispatch(self):
        tsv = _make_csv_bytes("x\ty\n1\t2")
        df = load_uploaded_file(tsv, "data.tsv")
        assert df.shape == (1, 2)

    def test_unsupported_extension_raises(self):
        f = io.BytesIO(b"anything")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_uploaded_file(f, "data.parquet")


# ── Sample datasets ─────────────────────────────────────────────────


class TestSampleDatasets:
    def test_list_samples_finds_iris(self):
        samples = list_sample_datasets()
        names = [s["name"] for s in samples]
        assert "iris" in names

    def test_load_iris(self):
        df = load_sample_dataset("iris")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150
        assert "species" in df.columns

    def test_load_titanic(self):
        df = load_sample_dataset("titanic")
        assert isinstance(df, pd.DataFrame)
        assert "survived" in df.columns

    def test_load_wine_quality(self):
        df = load_sample_dataset("wine_quality")
        assert isinstance(df, pd.DataFrame)
        assert "quality" in df.columns

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_sample_dataset("nonexistent_dataset_xyz")

    def test_datasets_dir_exists(self):
        assert DATASETS_DIR.exists()
        assert any(DATASETS_DIR.glob("*.csv"))
