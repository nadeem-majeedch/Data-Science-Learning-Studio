"""
Clustering — Discover natural groupings in unlabeled data.

Implements K-Means, DBSCAN, and Agglomerative Clustering with
interactive parameter tuning, visualisation, and educational context.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.clustering import (
    ALGO_INFO,
    ClusteringResult,
    cluster_summary,
    compute_elbow_data,
    compute_silhouette_scores,
    pca_project,
    preprocess_features,
    run_agglomerative,
    run_dbscan,
    run_kmeans,
)
from utils.ui import build_sidebar, page_header

st.set_page_config(page_title="Clustering", page_icon="🔮", layout="wide")
build_sidebar()
page_header("clustering")

# ── Load dataset ────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("current_dataset")
name: str = st.session_state.get("current_dataset_name", "")

if df is None:
    st.warning("⚠️ No dataset loaded. Go to **Dataset Explorer** and upload a dataset first.")
    st.stop()

num_df = df.select_dtypes("number")
if num_df.shape[1] < 2:
    st.warning("⚠️ Clustering requires at least 2 numerical columns. This dataset only has "
               f"{num_df.shape[1]}. Try the **Feature Engineering** module to create more features.")
    st.stop()

st.success(f"🔮 Clustering on: **{name}** ({df.shape[0]:,} rows, {num_df.shape[1]} numerical features)")

# ── Educational sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧠 Clustering Overview")
    st.markdown(
        "**Clustering** is an **unsupervised learning** technique — it finds "
        "patterns in data *without* labels.\n\n"
        "**Supervised vs Unsupervised:**\n"
        "- *Supervised* (Classification/Regression): data has known labels.\n"
        "- *Unsupervised* (Clustering): data has *no* labels — we discover structure."
    )
    st.markdown(
        "**When to use clustering:**\n"
        "- Customer segmentation\n"
        "- Anomaly/outlier detection\n"
        "- Document grouping\n"
        "- Image segmentation\n"
        "- Exploratory analysis"
    )

# ── Configuration ───────────────────────────────────────────────────
with st.expander("⚙️ Configuration", expanded=True):
    selected_features = st.multiselect(
        "Feature columns for clustering",
        num_df.columns.tolist(),
        default=num_df.columns.tolist()[:min(4, num_df.shape[1])],
        help="Select 2 or more numerical features.",
    )

    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features.")
        st.stop()

    scale_data = st.checkbox(
        "Standardise features (recommended)",
        value=True,
        help="Scales each feature to zero mean and unit variance. "
             "Strongly recommended for distance-based algorithms.",
    )

# ── Algorithm selection ─────────────────────────────────────────────
algo = st.selectbox(
    "Clustering algorithm",
    ["K-Means", "DBSCAN", "Agglomerative Clustering"],
    help="Choose the algorithm to discover clusters.",
)

# Algorithm-specific parameters
params: dict = {}

if algo == "K-Means":
    with st.expander("🔧 K-Means Parameters", expanded=True):
        n_clusters = st.slider("Number of clusters (k)", 2, 15, 3, help="The number of groups to find.")
        n_init = st.slider("n_init", 1, 30, 10, help="Number of restarts with different seeds. Higher = more stable.")
        max_iter = st.slider("Max iterations", 10, 1000, 300, step=10)
        params = {"n_clusters": n_clusters, "n_init": n_init, "max_iter": max_iter}

elif algo == "DBSCAN":
    with st.expander("🔧 DBSCAN Parameters", expanded=True):
        eps = st.slider(
            "Epsilon (eps)", 0.1, 10.0, 0.5, 0.1,
            help="Max distance between two points to be considered neighbours.",
        )
        min_samples = st.slider(
            "Min samples", 2, 30, 5,
            help="Minimum points to form a dense region.",
        )
        metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"], index=0)
        params = {"eps": eps, "min_samples": min_samples, "metric": metric}

elif algo == "Agglomerative Clustering":
    with st.expander("🔧 Agglomerative Parameters", expanded=True):
        n_clusters_agg = st.slider("Number of clusters", 2, 15, 3)
        linkage = st.selectbox(
            "Linkage criterion",
            ["ward", "complete", "average", "single"],
            help="How to measure distance between clusters.",
        )
        if linkage == "ward":
            st.info("ℹ️ Ward linkage only supports Euclidean distance.")
        params = {"n_clusters": n_clusters_agg, "linkage": linkage}

# ── Educational info ────────────────────────────────────────────────
with st.expander(f"📖 About {algo}", expanded=False):
    info = ALGO_INFO[algo]
    st.markdown(info["description"])
    st.markdown(f"**Why use it:** {info['why_use']}")
    st.markdown("**Advantages:**")
    for adv in info["advantages"]:
        st.markdown(f"- ✅ {adv}")
    st.markdown("**Limitations:**")
    for lim in info["limitations"]:
        st.markdown(f"- ⚠️ {lim}")
    st.markdown(f"**When to use:** {info['when_to_use']}")
    st.markdown("**Key Parameters:**")
    for pname, pdesc in info["important_params"].items():
        st.markdown(f"- **{pname}**: {pdesc}")

st.markdown("---")

# ── Run clustering ──────────────────────────────────────────────────
if st.button("🚀 Run Clustering", type="primary"):
    with st.spinner("Preprocessing and clustering..."):
        X, scaler = preprocess_features(selected_features=selected_features, df=num_df, scale=scale_data)

        if X.shape[0] < 2:
            st.error("❌ Not enough data points after dropping NaN rows. Check your selected features.")
            st.stop()

        if algo == "K-Means":
            result = run_kmeans(X, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
        elif algo == "DBSCAN":
            result = run_dbscan(X, eps=eps, min_samples=min_samples, metric=metric)
        else:
            result = run_agglomerative(X, n_clusters=n_clusters_agg, linkage=linkage)

    st.session_state["clustering_result"] = result
    st.session_state["clustering_X"] = X
    st.session_state["clustering_features"] = selected_features
    st.session_state["clustering_df"] = num_df

# ── Display results ─────────────────────────────────────────────────
result: ClusteringResult | None = st.session_state.get("clustering_result")

if result is None:
    st.info("👆 Configure parameters and click **Run Clustering** to see results.")
    st.stop()

# Result summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Algorithm", result.algorithm)
col2.metric("Clusters found", result.n_clusters)
col3.metric("Noise points", result.n_noise)
if result.silhouette is not None:
    col4.metric("Silhouette Score", f"{result.silhouette:.3f}", help="Ranges from -1 to 1. Higher = better defined clusters.")
else:
    col4.metric("Silhouette Score", "N/A", help="Needs ≥2 clusters with ≥3 points each.")

st.markdown("---")

# ── Visualisation tabs ──────────────────────────────────────────────
X = st.session_state["clustering_X"]
features = st.session_state["clustering_features"]
df_full = st.session_state["clustering_df"]

tab_viz, tab_elbow, tab_summary, tab_download, tab_code = st.tabs([
    "📊 Cluster Visualization",
    "📈 Elbow / Silhouette",
    "📋 Cluster Summary",
    "📥 Download",
    "💻 Code",
])

# ── Tab 1: Cluster visualisation ───────────────────────────────────
with tab_viz:
    st.markdown("### Cluster Visualisation")

    # Decide whether PCA is needed
    need_pca = X.shape[1] > 2
    if need_pca:
        st.info(f"ℹ️ Your data has {X.shape[1]} dimensions. Showing PCA projection to 2D.")
        X_proj, pca_model = pca_project(X, n_components=2)
        x_col, y_col = "PC1", "PC2"
        st.caption(
            f"PCA explained variance: {pca_model.explained_variance_ratio_[0]:.1%} + "
            f"{pca_model.explained_variance_ratio_[1]:.1%} = "
            f"{sum(pca_model.explained_variance_ratio_):.1%}"
        )
    else:
        X_proj = X.copy()
        x_col, y_col = features[0], features[1]

    viz_df = X_proj.copy()
    viz_df["Cluster"] = result.labels.astype(str)

    # Color by cluster
    fig = px.scatter(
        viz_df,
        x=x_col,
        y=y_col,
        color="Cluster",
        title=f"{result.algorithm} — {result.n_clusters} clusters",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(legend_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # If PCA was used, also show the original features colored by cluster
    if need_pca:
        st.markdown("#### Original Features by Cluster")
        orig_viz = df_full[features].copy()
        orig_viz["Cluster"] = result.labels.astype(str)
        fig2 = px.scatter(
            orig_viz,
            x=features[0],
            y=features[1],
            color="Cluster",
            title="Original feature space (first two selected features)",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Elbow / Silhouette ─────────────────────────────────────
with tab_elbow:
    st.markdown("### Finding the Optimal Number of Clusters")
    st.markdown(
        "These charts help you decide how many clusters to use with K-Means. "
        "They are **exploratory tools**, not definitive answers."
    )

    k_range = range(2, min(16, X.shape[0]))

    with st.spinner("Computing elbow data..."):
        elbow_df = compute_elbow_data(X, k_range=k_range)
        sil_df = compute_silhouette_scores(X, k_range=k_range)

    col_e, col_s = st.columns(2)

    with col_e:
        st.markdown("#### Elbow Method")
        st.markdown(
            "Plot of within-cluster sum of squares (inertia) vs k. "
            "Look for the 'elbow' — the point where adding more clusters "
            "gives diminishing returns."
        )
        fig_elbow = px.line(
            elbow_df, x="k", y="inertia", markers=True,
            title="Elbow Plot",
        )
        fig_elbow.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Inertia (WCSS)")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_s:
        st.markdown("#### Silhouette Score vs k")
        st.markdown(
            "Silhouette measures how similar each point is to its own cluster "
            "versus the nearest other cluster. Higher is better."
        )
        fig_sil = px.line(
            sil_df, x="k", y="silhouette", markers=True,
            title="Silhouette Score",
        )
        fig_sil.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Silhouette Score")
        st.plotly_chart(fig_sil, use_container_width=True)

    with st.expander("📚 How to read these charts"):
        st.markdown("""
        **Elbow method:**
        - Plot inertia against k.
        - The 'elbow' (sharp bend) suggests the optimal k.
        - No clear elbow may mean clusters are not well-separated.

        **Silhouette score:**
        - Ranges from -1 to 1.
        - Close to 1: points are well-matched to their cluster.
        - Close to 0: points are on the border between clusters.
        - Negative: points may be in the wrong cluster.
        - Pick the k with the highest score.
        """)

# ── Tab 3: Cluster summary ─────────────────────────────────────────
with tab_summary:
    st.markdown("### Cluster Profiles")

    summary = cluster_summary(df_full, result.labels, features)
    st.dataframe(summary, use_container_width=True)

    # Cluster sizes
    st.markdown("#### Cluster Sizes")
    sizes_df = pd.DataFrame(
        [{"Cluster": k, "Count": v, "Percentage": f"{v / len(result.labels) * 100:.1f}%"}
         for k, v in sorted(result.cluster_sizes.items())]
    )
    st.dataframe(sizes_df, use_container_width=True, hide_index=True)

    # Bar chart of cluster sizes
    fig_sizes = px.bar(
        sizes_df, x="Cluster", y="Count",
        title="Samples per Cluster",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

# ── Tab 4: Download ────────────────────────────────────────────────
with tab_download:
    st.markdown("### Download Results")

    download_df = df_full[features].copy()
    download_df["Cluster"] = result.labels

    csv = download_df.to_csv(index=False)
    st.download_button(
        "📥 Download dataset with cluster labels (CSV)",
        data=csv,
        file_name=f"{name}_clustered.csv" if name else "clustered_data.csv",
        mime="text/csv",
    )

    st.dataframe(download_df.head(50), use_container_width=True)

# ── Tab 5: Generated code ──────────────────────────────────────────
with tab_code:
    st.markdown("### Generated Python Code")
    st.markdown("Copy this code to reproduce the clustering in a Jupyter notebook.")
    st.code(result.code, language="python")

    with st.expander("📚 Code explanation"):
        st.markdown("""
        The generated code:
        1. **Selects features** — the same columns you chose in the UI.
        2. **Scales** — StandardScaler normalises each feature (if you enabled scaling).
        3. **Runs the algorithm** — with the exact parameters you specified.
        4. **PCA projection** — optional 2D visualisation of high-dimensional clusters.
        5. **Elbow / Silhouette** — loops over k=2..10 to help find the optimal number.
        """)

# ── Educational footer ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Clustering Fundamentals", expanded=False):
    st.markdown("""
    ### Supervised vs Unsupervised Learning

    | Aspect | Supervised | Unsupervised |
    |--------|-----------|-------------|
    | **Labels** | Known (y) | None |
    | **Goal** | Predict y from X | Find structure in X |
    | **Examples** | Classification, Regression | Clustering, Dimensionality Reduction |
    | **Evaluation** | Accuracy, F1, R² | Silhouette, visual inspection |

    ### Choosing an Algorithm

    | Algorithm | Best for | Avoid when |
    |-----------|----------|------------|
    | **K-Means** | Spherical clusters, known k | Non-globular shapes, varying density |
    | **DBSCAN** | Arbitrary shapes, outlier detection | Varying density clusters |
    | **Agglomerative** | Hierarchical structure, small datasets | Large datasets (>10k rows) |

    ### Common Pitfalls
    - **Not scaling features**: Distance-based algorithms are sensitive to feature scales.
    - **Too many/few clusters**: Always validate with elbow + silhouette + domain knowledge.
    - **Ignoring outliers**: DBSCAN handles noise; K-Means doesn't.
    """)
