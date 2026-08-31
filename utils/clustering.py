"""
Clustering utilities for the Data Science Lab.

Provides reusable functions for K-Means, DBSCAN, and Agglomerative
Clustering, including preprocessing, evaluation, and code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class ClusteringStep:
    """Single clustering step with metadata for code generation."""
    operation: str
    params: dict[str, Any]
    code: str


@dataclass
class ClusteringResult:
    """Complete result of a clustering run."""
    labels: np.ndarray
    algorithm: str
    params: dict[str, Any]
    n_clusters: int
    n_noise: int  # noise points for DBSCAN
    silhouette: float | None
    cluster_sizes: dict[int, int]
    code: str
    steps: list[ClusteringStep] = field(default_factory=list)


# ── Algorithm information ──────────────────────────────────────────

ALGO_INFO: dict[str, dict[str, Any]] = {
    "K-Means": {
        "description": (
            "K-Means partitions data into *k* clusters by minimising the within-cluster "
            "sum of squared distances. Each point is assigned to the nearest centroid, "
            "and centroids are iteratively updated."
        ),
        "why_use": (
            "Fast and easy to understand. Works well when clusters are spherical, "
            "roughly equal in size, and well-separated."
        ),
        "advantages": [
            "Computationally efficient — O(n·k·d) per iteration.",
            "Scales well to large datasets.",
            "Results are easy to interpret.",
        ],
        "limitations": [
            "Must specify *k* in advance.",
            "Assumes clusters are convex and isotropic.",
            "Sensitive to initial centroid placement (mitigated by n_init).",
            "Struggles with clusters of varying density or non-globular shapes.",
        ],
        "when_to_use": (
            "Use K-Means when you have a rough idea of the number of clusters, "
            "your data is numerical, and clusters are expected to be roughly "
            "spherical. Great as a first pass."
        ),
        "important_params": {
            "n_clusters": "Number of clusters to form (the *k*).",
            "n_init": "Number of times K-Means runs with different seeds (default 10). Higher = more stable.",
            "max_iter": "Maximum iterations per run.",
            "random_state": "Seed for reproducibility.",
        },
    },
    "DBSCAN": {
        "description": (
            "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups "
            "points that are closely packed and marks outliers in low-density regions as noise."
        ),
        "why_use": (
            "Does not require specifying the number of clusters. Can find arbitrarily "
            "shaped clusters and automatically detects noise."
        ),
        "advantages": [
            "No need to pre-specify the number of clusters.",
            "Can find arbitrarily shaped clusters.",
            "Robust to outliers — labels them as noise (-1).",
        ],
        "limitations": [
            "Struggles with clusters of varying density.",
            "Sensitive to the choice of *eps* and *min_samples*.",
            "Not ideal for high-dimensional data without dimensionality reduction.",
        ],
        "when_to_use": (
            "Use DBSCAN when clusters are non-globular, you suspect outliers, "
            "or you don't know the number of clusters. Best for low-to-medium "
            "dimensional data."
        ),
        "important_params": {
            "eps": "Maximum distance between two points for them to be considered neighbours.",
            "min_samples": "Minimum number of points required to form a dense region.",
        },
    },
    "Agglomerative Clustering": {
        "description": (
            "Agglomerative (hierarchical) clustering builds a tree of clusters by "
            "starting with each point as its own cluster and successively merging "
            "the two closest clusters."
        ),
        "why_use": (
            "Produces a hierarchy that can be visualised as a dendrogram. Useful when "
            "the number of clusters is not known and a hierarchy is informative."
        ),
        "advantages": [
            "Does not require pre-specifying the number of clusters (but can set it).",
            "Produces a dendrogram for visual inspection.",
            "Deterministic — no random initialisation.",
        ],
        "limitations": [
            "Computationally expensive: O(n³) with standard linkage.",
            "Does not scale well to large datasets.",
            "Sensitive to noise and outliers.",
        ],
        "when_to_use": (
            "Use Agglomerative Clustering on smaller datasets where hierarchy matters, "
            "or when you want a dendrogram to explore possible numbers of clusters."
        ),
        "important_params": {
            "n_clusters": "Number of clusters to find.",
            "linkage": "'ward', 'complete', 'average', or 'single'. Ward minimises variance within clusters.",
            "affinity": "Distance metric: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'.",
        },
    },
}


# ── Preprocessing ──────────────────────────────────────────────────

def preprocess_features(
    df: pd.DataFrame,
    selected_features: list[str],
    scale: bool = True,
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Select and optionally scale features for clustering.

    Returns (processed_df, scaler). scaler is None when scale=False.
    """
    X = df[selected_features].copy()

    # Drop rows with any NaN in selected features
    X = X.dropna()

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)

    return X, scaler


# ── Clustering runners ─────────────────────────────────────────────

def run_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> ClusteringResult:
    """Run K-Means and return structured results."""
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    sil = _safe_silhouette(X, labels, n_clusters)

    sizes = _cluster_sizes(labels)
    code = _generate_code("kmeans", X.columns.tolist(), {
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": random_state,
    })

    return ClusteringResult(
        labels=labels,
        algorithm="K-Means",
        params={"n_clusters": n_clusters, "n_init": n_init, "max_iter": max_iter, "random_state": random_state},
        n_clusters=n_clusters,
        n_noise=0,
        silhouette=sil,
        cluster_sizes=sizes,
        code=code,
    )


def run_dbscan(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> ClusteringResult:
    """Run DBSCAN and return structured results."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)

    unique = set(labels)
    n_clusters = len(unique - {-1})
    n_noise = int((labels == -1).sum())

    sil = None
    if n_clusters >= 2:
        # Silhouette only for non-noise points belonging to a cluster
        mask = labels != -1
        if mask.sum() > n_clusters:
            sil = _safe_silhouette(X.loc[mask], labels[mask], n_clusters)

    sizes = _cluster_sizes(labels)
    code = _generate_code("dbscan", X.columns.tolist(), {
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
    })

    return ClusteringResult(
        labels=labels,
        algorithm="DBSCAN",
        params={"eps": eps, "min_samples": min_samples, "metric": metric},
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=sil,
        cluster_sizes=sizes,
        code=code,
    )


def run_agglomerative(
    X: pd.DataFrame,
    n_clusters: int = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> ClusteringResult:
    """Run Agglomerative Clustering and return structured results."""
    # Ward linkage only supports euclidean
    actual_metric = "euclidean" if linkage == "ward" else metric

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=actual_metric,
    )
    labels = model.fit_predict(X)
    sil = _safe_silhouette(X, labels, n_clusters)

    sizes = _cluster_sizes(labels)
    code = _generate_code("agglomerative", X.columns.tolist(), {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "metric": actual_metric,
    })

    return ClusteringResult(
        labels=labels,
        algorithm="Agglomerative Clustering",
        params={"n_clusters": n_clusters, "linkage": linkage, "metric": actual_metric},
        n_clusters=n_clusters,
        n_noise=0,
        silhouette=sil,
        cluster_sizes=sizes,
        code=code,
    )


# ── Evaluation helpers ─────────────────────────────────────────────

def compute_elbow_data(
    X: pd.DataFrame,
    k_range: range | list[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute inertia (within-cluster sum of squares) for a range of k values.
    Returns a DataFrame with columns ['k', 'inertia'].
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        model.fit(X)
        rows.append({"k": k, "inertia": model.inertia_})
    return pd.DataFrame(rows)


def compute_silhouette_scores(
    X: pd.DataFrame,
    k_range: range | list[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute silhouette score for K-Means with varying k.
    Returns a DataFrame with columns ['k', 'silhouette'].
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        score = _safe_silhouette(X, labels, k)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)


def pca_project(X: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, PCA]:
    """
    Project features to n_components dimensions via PCA.
    Returns (projected_df, fitted_pca).
    """
    pca = PCA(n_components=n_components, random_state=42)
    arr = pca.fit_transform(X)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(arr, columns=cols, index=X.index), pca


def cluster_summary(df: pd.DataFrame, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Compute per-cluster mean for each feature."""
    tmp = df[features].copy()
    tmp["Cluster"] = labels
    return tmp.groupby("Cluster").mean().round(4)


# ── Internal helpers ───────────────────────────────────────────────

def _safe_silhouette(X: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> float | None:
    """Compute silhouette score, returning None if not applicable."""
    if n_clusters < 2:
        return None
    # Need at least 2 clusters and more samples than clusters
    unique_labels = set(labels)
    if len(unique_labels - {-1}) < 2:
        return None
    if len(labels) <= len(unique_labels):
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def _cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Count samples per cluster label."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


# ── Code generation ────────────────────────────────────────────────

def _generate_code(algo: str, features: list[str], params: dict[str, Any]) -> str:
    """Generate reproducible Python code for a clustering run."""
    lines = [
        "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.decomposition import PCA",
        "",
        "# Select features",
        f"features = {features!r}",
        "X = df[features].dropna()",
        "",
        "# Scale",
        "scaler = StandardScaler()",
        "X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)",
        "",
    ]

    if algo == "kmeans":
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"model = KMeans({params_str})")
    elif algo == "dbscan":
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"model = DBSCAN({params_str})")
    elif algo == "agglomerative":
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"model = AgglomerativeClustering({params_str})")

    lines.extend([
        "",
        "labels = model.fit_predict(X_scaled)",
        'df["Cluster"] = labels',
        "",
        "# PCA projection for visualisation",
        "pca = PCA(n_components=2)",
        "X_pca = pca.fit_transform(X_scaled)",
        "",
        "# Elbow / silhouette analysis (for K-Means)",
        "inertias = []",
        "sil_scores = []",
        "for k in range(2, 11):",
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)",
        "    km.fit(X_scaled)",
        "    inertias.append(km.inertia_)",
        "    sil_scores.append(silhouette_score(X_scaled, km.labels_))",
    ])

    return "\n".join(lines)
