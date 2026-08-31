"""
Tests for utils/clustering.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from utils.clustering import (
    ALGO_INFO,
    cluster_summary,
    compute_elbow_data,
    compute_silhouette_scores,
    pca_project,
    preprocess_features,
    run_agglomerative,
    run_dbscan,
    run_kmeans,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def blob_df():
    """3-blob dataset with 4 features."""
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    return pd.DataFrame(X, columns=["a", "b", "c", "d"])


@pytest.fixture
def df_with_nan():
    """Dataset with missing values."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3)
    X[5, 0] = np.nan
    X[10, 1] = np.nan
    return pd.DataFrame(X, columns=["x", "y", "z"])


@pytest.fixture
def df_with_categorical():
    """Dataset with both numerical and categorical columns."""
    return pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })


# ── Algorithm info ──────────────────────────────────────────────────

class TestAlgoInfo:
    def test_all_algorithms_have_info(self):
        assert "K-Means" in ALGO_INFO
        assert "DBSCAN" in ALGO_INFO
        assert "Agglomerative Clustering" in ALGO_INFO

    def test_info_structure(self):
        for name, info in ALGO_INFO.items():
            assert "description" in info, f"{name} missing description"
            assert "why_use" in info, f"{name} missing why_use"
            assert "advantages" in info, f"{name} missing advantages"
            assert "limitations" in info, f"{name} missing limitations"
            assert "when_to_use" in info, f"{name} missing when_to_use"
            assert "important_params" in info, f"{name} missing important_params"
            assert len(info["advantages"]) >= 2, f"{name} needs ≥2 advantages"
            assert len(info["limitations"]) >= 2, f"{name} needs ≥2 limitations"


# ── Preprocessing ──────────────────────────────────────────────────

class TestPreprocessFeatures:
    def test_selects_correct_columns(self, blob_df):
        X, scaler = preprocess_features(blob_df, ["a", "b"], scale=False)
        assert list(X.columns) == ["a", "b"]
        assert X.shape[0] == blob_df.shape[0]

    def test_drops_nan_rows(self, df_with_nan):
        X, _ = preprocess_features(df_with_nan, ["x", "y"], scale=False)
        assert X.shape[0] == df_with_nan.dropna().shape[0]

    def test_scaling(self, blob_df):
        X, scaler = preprocess_features(blob_df, ["a", "b"], scale=True)
        assert scaler is not None
        assert abs(X["a"].mean()) < 0.1
        assert abs(X["a"].std() - 1.0) < 0.1

    def test_no_scaling(self, blob_df):
        X, scaler = preprocess_features(blob_df, ["a", "b"], scale=False)
        assert scaler is None
        # Values should not be standardised
        assert not (abs(X["a"].mean()) < 0.1 and abs(X["a"].std() - 1.0) < 0.1)

    def test_keyword_arguments(self, blob_df):
        """Verify keyword-style call used by the Clustering page works."""
        X, scaler = preprocess_features(
            df=blob_df, selected_features=["a", "b"], scale=True,
        )
        assert list(X.columns) == ["a", "b"]
        assert scaler is not None


# ── K-Means ────────────────────────────────────────────────────────

class TestKMeans:
    def test_returns_correct_clusters(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=3)
        assert result.n_clusters == 3
        assert result.algorithm == "K-Means"
        assert len(result.labels) == blob_df.shape[0]
        assert result.n_noise == 0

    def test_silhouette_positive(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=3)
        assert result.silhouette is not None
        assert result.silhouette > 0.3  # well-separated blobs should be decent

    def test_cluster_sizes_sum_to_n(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=3)
        assert sum(result.cluster_sizes.values()) == blob_df.shape[0]

    def test_labels_are_integers(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=2)
        assert all(isinstance(int(l), int) for l in result.labels)

    def test_reproducible_with_random_state(self, blob_df):
        r1 = run_kmeans(blob_df, n_clusters=3, random_state=42)
        r2 = run_kmeans(blob_df, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_different_k_gives_different_results(self, blob_df):
        r2 = run_kmeans(blob_df, n_clusters=2)
        r4 = run_kmeans(blob_df, n_clusters=4)
        assert r2.n_clusters != r4.n_clusters

    def test_code_is_string(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=3)
        assert isinstance(result.code, str)
        assert "KMeans" in result.code

    def test_params_stored(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=5, n_init=20, max_iter=500)
        assert result.params["n_clusters"] == 5
        assert result.params["n_init"] == 20
        assert result.params["max_iter"] == 500


# ── DBSCAN ──────────────────────────────────────────────────────────

class TestDBSCAN:
    def test_runs_without_error(self, blob_df):
        result = run_dbscan(blob_df, eps=1.5, min_samples=5)
        assert result.algorithm == "DBSCAN"
        assert len(result.labels) == blob_df.shape[0]

    def test_finds_clusters_on_blob_data(self, blob_df):
        result = run_dbscan(blob_df, eps=2.0, min_samples=3)
        assert result.n_clusters >= 1  # at least 1 cluster on well-separated blobs

    def test_noise_points_detected(self, blob_df):
        result = run_dbscan(blob_df, eps=0.1, min_samples=10)
        assert result.n_noise > 0  # very strict params should produce noise

    def test_silhouette_for_valid_clusters(self, blob_df):
        result = run_dbscan(blob_df, eps=2.0, min_samples=3)
        if result.n_clusters >= 2:
            assert result.silhouette is not None

    def test_code_generation(self, blob_df):
        result = run_dbscan(blob_df, eps=1.5, min_samples=5)
        assert "DBSCAN" in result.code

    def test_params_stored(self, blob_df):
        result = run_dbscan(blob_df, eps=0.8, min_samples=7, metric="manhattan")
        assert result.params["eps"] == 0.8
        assert result.params["min_samples"] == 7
        assert result.params["metric"] == "manhattan"


# ── Agglomerative ───────────────────────────────────────────────────

class TestAgglomerative:
    def test_runs_without_error(self, blob_df):
        result = run_agglomerative(blob_df, n_clusters=3)
        assert result.algorithm == "Agglomerative Clustering"
        assert result.n_clusters == 3

    def test_silhouette_positive(self, blob_df):
        result = run_agglomerative(blob_df, n_clusters=3)
        assert result.silhouette is not None
        assert result.silhouette > 0.3

    def test_deterministic(self, blob_df):
        r1 = run_agglomerative(blob_df, n_clusters=3, linkage="ward")
        r2 = run_agglomerative(blob_df, n_clusters=3, linkage="ward")
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_linkage_types(self, blob_df):
        for linkage in ["ward", "complete", "average", "single"]:
            result = run_agglomerative(blob_df, n_clusters=3, linkage=linkage)
            assert result.n_clusters == 3

    def test_code_generation(self, blob_df):
        result = run_agglomerative(blob_df, n_clusters=3, linkage="ward")
        assert "AgglomerativeClustering" in result.code

    def test_params_stored(self, blob_df):
        result = run_agglomerative(blob_df, n_clusters=4, linkage="complete", metric="cosine")
        assert result.params["linkage"] == "complete"
        assert result.params["metric"] == "cosine"


# ── Elbow / Silhouette ─────────────────────────────────────────────

class TestElbowData:
    def test_returns_dataframe(self, blob_df):
        df = compute_elbow_data(blob_df, k_range=range(2, 6))
        assert isinstance(df, pd.DataFrame)
        assert "k" in df.columns
        assert "inertia" in df.columns
        assert len(df) == 4

    def test_inertia_decreases_with_k(self, blob_df):
        df = compute_elbow_data(blob_df, k_range=range(2, 8))
        # Inertia should be monotonically non-increasing
        assert df["inertia"].is_monotonic_decreasing

    def test_different_k_values(self, blob_df):
        df = compute_elbow_data(blob_df, k_range=[3, 5, 7])
        assert set(df["k"]) == {3, 5, 7}


class TestSilhouetteScores:
    def test_returns_dataframe(self, blob_df):
        df = compute_silhouette_scores(blob_df, k_range=range(2, 6))
        assert isinstance(df, pd.DataFrame)
        assert "k" in df.columns
        assert "silhouette" in df.columns

    def test_scores_between_neg1_and_1(self, blob_df):
        df = compute_silhouette_scores(blob_df, k_range=range(2, 8))
        assert df["silhouette"].between(-1, 1).all()

    def test_single_k(self, blob_df):
        df = compute_silhouette_scores(blob_df, k_range=[3])
        assert len(df) == 1


# ── PCA ─────────────────────────────────────────────────────────────

class TestPCAProject:
    def test_reduces_to_2d(self, blob_df):
        proj, pca = pca_project(blob_df, n_components=2)
        assert proj.shape == (blob_df.shape[0], 2)
        assert list(proj.columns) == ["PC1", "PC2"]

    def test_reduces_to_1d(self, blob_df):
        proj, pca = pca_project(blob_df, n_components=1)
        assert proj.shape == (blob_df.shape[0], 1)

    def test_preserves_index(self, blob_df):
        proj, _ = pca_project(blob_df, n_components=2)
        assert list(proj.index) == list(blob_df.index)

    def test_explained_variance(self, blob_df):
        _, pca = pca_project(blob_df, n_components=2)
        assert sum(pca.explained_variance_ratio_) <= 1.0
        assert sum(pca.explained_variance_ratio_) > 0.0


# ── Cluster summary ─────────────────────────────────────────────────

class TestClusterSummary:
    def test_grouped_by_cluster(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=3)
        summary = cluster_summary(blob_df, result.labels, ["a", "b"])
        assert len(summary) == 3
        assert list(summary.columns) == ["a", "b"]

    def test_includes_all_features(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=2)
        summary = cluster_summary(blob_df, result.labels, ["a", "b", "c", "d"])
        assert list(summary.columns) == ["a", "b", "c", "d"]


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_two_points_two_clusters(self):
        X = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        result = run_kmeans(X, n_clusters=2)
        assert result.n_clusters == 2
        assert len(result.labels) == 2

    def test_single_feature_fails_preprocess(self):
        X = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises((KeyError, ValueError)):
            preprocess_features(X, ["x", "y"], scale=False)

    def test_all_nan_column(self):
        X = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1, 2, 3]})
        result_X, _ = preprocess_features(X, ["x", "y"], scale=False)
        # All rows with NaN in x should be dropped
        assert result_X.shape[0] == 0

    def test_kmeans_n_clusters_1(self, blob_df):
        result = run_kmeans(blob_df, n_clusters=1)
        assert result.n_clusters == 1
        assert result.silhouette is None  # silhouette needs ≥2 clusters

    def test_dbscan_all_noise(self, blob_df):
        # Very strict: eps tiny, min_samples huge → everything is noise
        result = run_dbscan(blob_df, eps=0.001, min_samples=100)
        assert result.n_noise == blob_df.shape[0]
        assert result.n_clusters == 0
        assert result.silhouette is None
