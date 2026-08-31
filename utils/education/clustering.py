"""
Clustering — learning topics for the Clustering module.
"""
from utils.education.base import T

TOPICS = {
    "what_is_unsupervised_learning": T(
        title="What Is Unsupervised Learning",
        module="clustering",
        what=(
            "Unsupervised learning finds hidden patterns in data "
            "without labelled targets. Unlike supervised learning, "
            "there is no 'correct answer' — the algorithm discovers "
            "structure on its own."
        ),
        why=(
            "Most real-world data is unlabelled. Unsupervised learning "
            "reveals structure you didn't know existed: customer "
            "segments, anomaly patterns, topic clusters."
        ),
        when=(
            "When you don't have labels, or want to discover hidden "
            "groups in data. Use before supervised learning to "
            "understand data structure."
        ),
        example="Customer segmentation: group customers by purchasing behaviour without predefined groups.",
        mistakes=[
            "Expecting unsupervised results to have clear 'right' answers.",
            "Not scaling features before clustering.",
            "Assuming clusters correspond to meaningful real-world categories.",
        ],
        interpretation=(
            "Unsupervised results require domain expertise to interpret. "
            "The algorithm finds mathematical structure; humans assign meaning."
        ),
        think_about_it=(
            "You run K-Means and get 3 clusters. Does this mean "
            "there are 3 'real' groups? What else could explain this?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "print(f'Cluster sizes: {np.bincount(labels)}')\n"
            "```"
        ),
        keywords=["unsupervised", "no labels", "hidden patterns", "discovery"],
    ),

    "supervised_vs_unsupervised": T(
        title="Supervised vs Unsupervised Learning",
        module="clustering",
        what=(
            "Supervised learning uses labelled data (X → y) to learn "
            "a mapping. Unsupervised learning finds structure in "
            "unlabelled data (X only, no y)."
        ),
        why=(
            "The choice between supervised and unsupervised depends on "
            "whether you have labels. Understanding both is essential "
            "for a complete data science toolkit."
        ),
        when=(
            "Have labels? → Supervised (classification/regression). "
            "No labels? → Unsupervised (clustering/dimensionality reduction)."
        ),
        example="Supervised: predict if an email is spam (labels: spam/not spam). Unsupervised: group emails by topic (no labels).",
        mistakes=[
            "Using supervised methods when no labels exist.",
            "Forcing unsupervised clusters to match expected labels.",
            "Not trying unsupervised methods to discover new patterns even when labels exist.",
        ],
        interpretation=(
            "Supervised: you know what to predict. "
            "Unsupervised: you want to discover what's in the data."
        ),
        think_about_it="You have patient data with a disease diagnosis column. When might unsupervised learning still be useful?",
        code_link=(
            "```python\n"
            "# Supervised\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X_train, y_train)  # uses labels\n"
            "\n"
            "# Unsupervised\n"
            "from sklearn.cluster import KMeans\n"
            "kmeans = KMeans(n_clusters=3)\n"
            "labels = kmeans.fit_predict(X)  # no labels\n"
            "```"
        ),
        keywords=["supervised", "unsupervised", "labels", "classification", "clustering"],
    ),

    "what_is_clustering": T(
        title="What Is Clustering",
        module="clustering",
        what=(
            "Clustering groups similar data points together so that "
            "points within a group are more similar to each other "
            "than to points in other groups."
        ),
        why=(
            "Clustering reveals natural groupings in data: customer "
            "segments, document topics, image regions, gene expressions. "
            "It's one of the most widely used unsupervised techniques."
        ),
        when=(
            "Use when you want to: segment customers, detect anomalies, "
            "organise documents, or explore data structure before "
            "building supervised models."
        ),
        example="Segmenting customers into 'budget', 'regular', and 'premium' groups based on purchasing behaviour.",
        mistakes=[
            "Choosing the number of clusters arbitrarily.",
            "Not interpreting clusters with domain knowledge.",
            "Assuming clusters are permanent — they depend on the features used.",
        ],
        interpretation=(
            "Cluster labels are arbitrary (0, 1, 2 have no inherent "
            "order). The quality depends on the distance metric, "
            "number of clusters, and feature scaling."
        ),
        think_about_it="You cluster customers and get Group A (high income, low spending) and Group B (low income, high spending). What business actions might you take?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "```"
        ),
        keywords=["clustering", "grouping", "similarity", "segments", "unsupervised"],
    ),

    "applications_of_clustering": T(
        title="Applications of Clustering",
        module="clustering",
        what=(
            "Clustering is used across industries: customer "
            "segmentation (marketing), anomaly detection (fraud), "
            "image segmentation (computer vision), document grouping "
            "(NLP), and biological taxonomy."
        ),
        why=(
            "Understanding real-world applications motivates learning "
            "clustering techniques and helps you apply them to "
            "your own problems."
        ),
        when=(
            "Whenever you need to find natural groups without labels. "
            "Common use cases: segmentation, anomaly detection, "
            "data exploration, feature engineering."
        ),
        example=(
            "Marketing: segment customers by purchase frequency and value.\n"
            "Healthcare: group patients by symptoms for treatment planning.\n"
            "Retail: organise products into categories automatically."
        ),
        mistakes=[
            "Over-engineering clustering when simple rules would work.",
            "Not validating clusters with business metrics.",
            "Creating too many segments that aren't actionable.",
        ],
        interpretation=(
            "Good clustering produces groups that are: internally "
            "cohesive (similar within), externally separated (different "
            "between), and actionable (business can use them)."
        ),
        think_about_it="An e-commerce company wants to personalise recommendations. How could clustering help?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "import pandas as pd\n"
            "\n"
            "# Customer segmentation\n"
            "features = df[['total_purchases', 'avg_order_value', 'days_since_last']]\n"
            "kmeans = KMeans(n_clusters=4, random_state=42)\n"
            "df['segment'] = kmeans.fit_predict(features)\n"
            "print(df.groupby('segment').mean())\n"
            "```"
        ),
        keywords=["applications", "segmentation", "anomaly", "marketing", "use cases"],
    ),

    "similarity_distance": T(
        title="Similarity and Distance Measures",
        module="clustering",
        what=(
            "Distance measures quantify how different two data points are. "
            "Common measures: Euclidean (straight-line), Manhattan "
            "(city-block), and Cosine (angle between vectors)."
        ),
        why=(
            "The choice of distance metric determines what 'similar' "
            "means. Euclidean favours magnitude, Cosine favours direction. "
            "Wrong metric = meaningless clusters."
        ),
        when=(
            "Always consider the distance metric before clustering. "
            "Euclidean for continuous data, Cosine for text, "
            "Manhattan for sparse data."
        ),
        example="Euclidean: sqrt((x1-x2)² + (y1-y2)²). Cosine: 1 - cos(θ). Manhattan: |x1-x2| + |y1-y2|.",
        mistakes=[
            "Using Euclidean distance without scaling features.",
            "Ignoring that high-dimensional distances become less meaningful.",
            "Not considering the data type when choosing a metric.",
        ],
        interpretation=(
            "In high dimensions, all points become equidistant "
            "(curse of dimensionality). This makes distance-based "
            "clustering less effective."
        ),
        think_about_it="Two documents have the same word frequencies but one is twice as long. Which distance metric would treat them as similar?",
        code_link=(
            "```python\n"
            "from sklearn.metrics import pairwise_distances\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "# Euclidean (default for K-Means)\n"
            "from sklearn.cluster import KMeans\n"
            "kmeans = KMeans(metric='l2')\n"
            "\n"
            "# Cosine (for text)\n"
            "from sklearn.cluster import DBSCAN\n"
            "dbscan = DBSCAN(metric='cosine')\n"
            "```"
        ),
        keywords=["distance", "euclidean", "cosine", "manhattan", "similarity", "metric"],
    ),

    "kmeans": T(
        title="K-Means Clustering",
        module="clustering",
        what=(
            "K-Means partitions data into K clusters by minimising "
            "the within-cluster sum of squares (WCSS). Each point "
            "is assigned to the nearest centroid."
        ),
        why=(
            "K-Means is the most widely used clustering algorithm. "
            "It's fast, simple, and works well when clusters are "
            "spherical and similarly sized."
        ),
        when=(
            "Use when: you know or can estimate K, data is numerical, "
            "clusters are roughly spherical. Not suitable for: "
            "non-spherical clusters, varying densities, outliers."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "print(f'WCSS: {kmeans.inertia_:.2f}')\n"
            "```"
        ),
        mistakes=[
            "Not choosing K carefully — use elbow method or silhouette.",
            "Not scaling features before K-Means.",
            "Running with default n_init=10 on large data (slow).",
        ],
        interpretation=(
            "kmeans.inertia_ = total within-cluster sum of squares. "
            "Lower is better, but always decreases with more K. "
            "Use the elbow to find the optimal K."
        ),
        think_about_it="K-Means finds 3 clusters in customer data. One cluster has 90% of customers. Is this useful?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "```"
        ),
        keywords=["k-means", "kmeans", "centroids", "partition", "wcss"],
    ),

    "centroids": T(
        title="Centroids",
        module="clustering",
        what=(
            "A centroid is the center of a cluster — the mean of all "
            "points assigned to that cluster. K-Means assigns each "
            "point to its nearest centroid."
        ),
        why=(
            "Centroids represent clusters. They define cluster boundaries "
            "and can be used to classify new data points."
        ),
        when=(
            "After K-Means training, use cluster_centers_ to examine "
            "what each cluster represents."
        ),
        example=(
            "```python\n"
            "kmeans = KMeans(n_clusters=3).fit(X_scaled)\n"
            "centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)\n"
            "print(centroids)\n"
            "# Cluster 0: high income, low spending\n"
            "# Cluster 1: low income, high spending\n"
            "# Cluster 2: medium income, medium spending\n"
            "```"
        ),
        mistakes=[
            "Interpreting centroids without knowing what features mean.",
            "Assuming centroids are actual data points (they're averages).",
            "Using centroids for non-spherical clusters (K-Means assumes spheres).",
        ],
        interpretation=(
            "Each centroid shows the 'average' member of that cluster. "
            "Compare centroids to understand what differentiates clusters."
        ),
        think_about_it="If you add a new feature to the clustering, how do the centroids and cluster assignments change?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "centroids = pd.DataFrame(\n"
            "    kmeans.cluster_centers_,\n"
            "    columns=feature_names\n"
            ")\n"
            "print(centroids.round(3))\n"
            "```"
        ),
        keywords=["centroid", "center", "mean", "cluster center", "representative"],
    ),

    "kmeans_algorithm_steps": T(
        title="K-Means Algorithm Steps",
        module="clustering",
        what=(
            "K-Means iterates: (1) Initialise K centroids randomly, "
            "(2) Assign each point to nearest centroid, (3) Update "
            "centroids as cluster means, (4) Repeat until convergence."
        ),
        why=(
            "Understanding the algorithm helps you know its limitations: "
            "it can get stuck in local optima (run multiple times), "
            "assumes spherical clusters, and is sensitive to initialisation."
        ),
        when=(
            "K-Means converges when centroids stop moving. The final "
            "result depends on initialisation — that's why n_init>1."
        ),
        example=(
            "```python\n"
            "# K-Means steps (conceptual)\n"
            "# 1. Random init: K centroids placed randomly\n"
            "# 2. Assignment: each point → nearest centroid\n"
            "# 3. Update: new centroids = mean of assigned points\n"
            "# 4. Repeat 2-3 until convergence\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)\n"
            "kmeans.fit(X_scaled)\n"
            "print(f'Converged in {kmeans.n_iter_} iterations')\n"
            "```"
        ),
        mistakes=[
            "Using n_init=1 — may converge to poor local optimum.",
            "Not understanding that results vary with different random seeds.",
        ],
        interpretation=(
            "n_init=10 means K-Means runs 10 times with different "
            "initialisations and picks the best result. This reduces "
            "sensitivity to initialisation."
        ),
        think_about_it="Why does K-Means sometimes give different results on the same data?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "\n"
            "kmeans = KMeans(\n"
            "    n_clusters=3,\n"
            "    n_init=10,       # run 10 times, pick best\n"
            "    max_iter=300,    # max iterations per run\n"
            "    random_state=42\n"
            ")\n"
            "kmeans.fit(X_scaled)\n"
            "print(f'Labels: {kmeans.labels_}')\n"
            "print(f'Inertia: {kmeans.inertia_:.2f}')\n"
            "```"
        ),
        keywords=["algorithm", "steps", "iteration", "convergence", "initialisation"],
    ),

    "choosing_k": T(
        title="Choosing the Number of Clusters (K)",
        module="clustering",
        what=(
            "Choosing K is the hardest part of K-Means. Too few clusters "
            "misses structure; too many creates artificial splits. "
            "Use the elbow method and silhouette score."
        ),
        why=(
            "The 'right' K depends on the data and business context. "
            "Automated methods suggest candidates; domain knowledge "
            "makes the final choice."
        ),
        when=(
            "Before running K-Means. Try K=2 through K=10 and evaluate "
            "each using elbow and silhouette methods."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "inertias = []\n"
            "for k in range(2, 11):\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    km.fit(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "\n"
            "plt.plot(range(2, 11), inertias, 'bo-')\n"
            "plt.xlabel('K')\n"
            "plt.ylabel('Inertia')\n"
            "plt.title('Elbow Method')\n"
            "```"
        ),
        mistakes=[
            "Choosing K=2 by default without investigation.",
            "Only using the elbow method (combine with silhouette).",
            "Ignoring domain knowledge in K selection.",
        ],
        interpretation=(
            "The 'elbow' is where adding more K gives diminishing returns. "
            "Look for the point where the curve bends sharply."
        ),
        think_about_it="Your elbow plot shows a gradual curve with no clear elbow. What does this suggest about your data?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import silhouette_score\n"
            "\n"
            "# Elbow + Silhouette\n"
            "for k in range(2, 11):\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    sil = silhouette_score(X_scaled, labels)\n"
            "    print(f'K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.3f}')\n"
            "```"
        ),
        keywords=["choosing", "k", "elbow", "silhouette", "optimal"],
    ),

    "elbow_method": T(
        title="Elbow Method",
        module="clustering",
        what=(
            "The elbow method plots within-cluster sum of squares "
            "(WCSS/inertia) against K. The optimal K is where the "
            "curve bends like an elbow — adding more K doesn't reduce "
            "WCSS significantly."
        ),
        why=(
            "WCSS always decreases with more K (trivially: more "
            "clusters = smaller within-cluster distances). The elbow "
            "identifies the point of diminishing returns."
        ),
        when=(
            "Run K-Means for K=2 to K=10, plot inertia, and look for "
            "the elbow. Works best when clusters are well-separated."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "import plotly.graph_objects as go\n"
            "\n"
            "inertias = []\n"
            "K_range = range(2, 11)\n"
            "for k in K_range:\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    km.fit(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "\n"
            "fig = go.Figure()\n"
            "fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))\n"
            "fig.update_layout(title='Elbow Method', xaxis_title='K', yaxis_title='Inertia')\n"
            "```"
        ),
        mistakes=[
            "Expecting a sharp elbow — real data often has a gradual curve.",
            "Using elbow alone without silhouette score.",
            "Not scaling data before computing distances.",
        ],
        interpretation=(
            "Sharp elbow → clear K. Gradual curve → clusters aren't "
            "well-separated, consider domain knowledge."
        ),
        think_about_it="The elbow is at K=4 but the business team wants exactly 3 segments. What do you do?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "inertias = []\n"
            "for k in range(2, 11):\n"
            "    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "plt.plot(range(2, 11), inertias, 'bo-')\n"
            "plt.xlabel('K'); plt.ylabel('Inertia')\n"
            "plt.title('Elbow Method')\n"
            "```"
        ),
        keywords=["elbow", "wcss", "inertia", "method", "optimal k"],
    ),

    "silhouette_score": T(
        title="Silhouette Score",
        module="clustering",
        what=(
            "Silhouette score measures how similar a point is to its "
            "own cluster vs the nearest other cluster. Range: -1 to 1. "
            "1 = well-clustered, 0 = on boundary, -1 = wrong cluster."
        ),
        why=(
            "Silhouette score provides a more objective measure than "
            "the elbow method. It evaluates both cohesion (within-cluster "
            "similarity) and separation (between-cluster distance)."
        ),
        when=(
            "Use alongside the elbow method to choose K. "
            "Higher silhouette = better-defined clusters."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import silhouette_score\n"
            "\n"
            "for k in range(2, 11):\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    score = silhouette_score(X_scaled, labels)\n"
            "    print(f'K={k}: Silhouette={score:.3f}')\n"
            "# K=3: Silhouette=0.52  ← highest\n"
            "```"
        ),
        mistakes=[
            "Using silhouette on very large datasets (computationally expensive).",
            "Not scaling features before computing silhouette.",
            "Expecting silhouette=1.0 — real data rarely achieves this.",
        ],
        interpretation=(
            "Silhouette > 0.5 → reasonable clustering. "
            "> 0.7 → strong structure. "
            "< 0.25 → probably no meaningful clusters."
        ),
        think_about_it="K=3 gives silhouette=0.45 and K=5 gives 0.48. Is the difference significant enough to choose K=5?",
        code_link=(
            "```python\n"
            "from sklearn.metrics import silhouette_score, silhouette_samples\n"
            "\n"
            "score = silhouette_score(X_scaled, labels)\n"
            "print(f'Average silhouette: {score:.3f}')\n"
            "\n"
            "# Per-sample silhouette\n"
            "sample_scores = silhouette_samples(X_scaled, labels)\n"
            "```"
        ),
        keywords=["silhouette", "score", "cohesion", "separation", "quality"],
    ),

    "dbscan": T(
        title="DBSCAN",
        module="clustering",
        what=(
            "DBSCAN (Density-Based Spatial Clustering of Applications "
            "with Noise) groups points that are closely packed and "
            "marks outliers as noise. It finds arbitrary-shaped clusters."
        ),
        why=(
            "Unlike K-Means, DBSCAN doesn't assume spherical clusters. "
            "It handles noise/outliers naturally and can find clusters "
            "of any shape."
        ),
        when=(
            "Use when: clusters have irregular shapes, data has noise, "
            "or you don't know the number of clusters. "
            "Parameters: eps (neighborhood radius), min_samples (minimum "
            "points to form a dense region)."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n"
            "\n"
            "dbscan = DBSCAN(eps=0.5, min_samples=5)\n"
            "labels = dbscan.fit_predict(X_scaled)\n"
            "n_clusters = len(set(labels)) - (1 if -1 in labels else 0)\n"
            "n_noise = (labels == -1).sum()\n"
            "print(f'Clusters: {n_clusters}, Noise points: {n_noise}')\n"
            "```"
        ),
        mistakes=[
            "Choosing eps without visualising the k-distance graph.",
            "Using on high-dimensional data without dimensionality reduction.",
            "Expecting all points to be assigned to a cluster.",
        ],
        interpretation=(
            "Label -1 = noise point (not in any cluster). "
            "High noise count → eps too small or min_samples too large."
        ),
        think_about_it="DBSCAN finds 2 clusters and labels 40% of points as noise. What should you do?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n"
            "from sklearn.neighbors import NearestNeighbors\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Find optimal eps using k-distance graph\n"
            "nn = NearestNeighbors(n_neighbors=5)\n"
            "nn.fit(X_scaled)\n"
            "distances, _ = nn.kneighbors(X_scaled)\n"
            "k_distances = sorted(distances[:, -1])\n"
            "plt.plot(k_distances)\n"
            "plt.ylabel('5th nearest neighbor distance')\n"
            "plt.title('K-Distance Graph for eps selection')\n"
            "```"
        ),
        keywords=["dbscan", "density", "noise", "arbitrary shape", "eps"],
    ),

    "core_border_noise_points": T(
        title="Core, Border and Noise Points in DBSCAN",
        module="clustering",
        what=(
            "DBSCAN classifies points as: Core (≥min_samples in eps "
            "radius), Border (in eps radius of a core but <min_samples "
            "itself), and Noise (not reachable from any core)."
        ),
        why=(
            "Understanding these point types explains how DBSCAN forms "
            "clusters and handles outliers. Core points are the cluster "
            "backbone; border points connect; noise points are outliers."
        ),
        when=(
            "After running DBSCAN, examine the ratio of core/border/noise "
            "points to assess cluster quality."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n"
            "\n"
            "dbscan = DBSCAN(eps=0.5, min_samples=5)\n"
            "labels = dbscan.fit_predict(X_scaled)\n"
            "\n"
            "n_core = len(dbscan.core_sample_indices_)\n"
            "n_noise = (labels == -1).sum()\n"
            "print(f'Core: {n_core}, Noise: {n_noise}')\n"
            "```"
        ),
        mistakes=[
            "Confusing border points with noise points.",
            "Setting min_samples too high — too many noise points.",
        ],
        interpretation=(
            "Many core points → dense, well-defined clusters. "
            "Many noise points → either wrong parameters or genuinely "
            "noisy data."
        ),
        think_about_it="What happens to cluster assignments if you increase eps slightly?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n"
            "\n"
            "dbscan = DBSCAN(eps=0.5, min_samples=5)\n"
            "labels = dbscan.fit_predict(X_scaled)\n"
            "\n"
            "core_mask = np.zeros(len(X_scaled), dtype=bool)\n"
            "core_mask[dbscan.core_sample_indices_] = True\n"
            "noise_mask = labels == -1\n"
            "border_mask = (~core_mask) & (~noise_mask)\n"
            "print(f'Core: {core_mask.sum()}, Border: {border_mask.sum()}, Noise: {noise_mask.sum()}')\n"
            "```"
        ),
        keywords=["core", "border", "noise", "dbscan", "density", "reachability"],
    ),

    "agglomerative_clustering": T(
        title="Agglomerative Clustering",
        module="clustering",
        what=(
            "Agglomerative clustering is a bottom-up approach: start "
            "with each point as its own cluster, then repeatedly merge "
            "the two closest clusters until K clusters remain."
        ),
        why=(
            "Agglomerative clustering produces a hierarchy (dendrogram) "
            "that lets you choose K after seeing the data structure. "
            "It's intuitive and doesn't require specifying K upfront."
        ),
        when=(
            "Use when you want hierarchical clustering or when K is "
            "unknown. Linkage determines how inter-cluster distance "
            "is measured: single, complete, average, ward."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import AgglomerativeClustering\n"
            "\n"
            "agg = AgglomerativeClustering(n_clusters=3, linkage='ward')\n"
            "labels = agg.fit_predict(X_scaled)\n"
            "```"
        ),
        mistakes=[
            "Not understanding linkage methods (ward is usually best).",
            "Using on large datasets — O(n²) memory complexity.",
            "Choosing K after looking at the dendrogram without domain justification.",
        ],
        interpretation=(
            "Ward linkage minimises within-cluster variance. "
            "Complete linkage uses maximum distance (compact clusters). "
            "Single linkage uses minimum distance (chain-like clusters)."
        ),
        think_about_it="When would you prefer agglomerative clustering over K-Means?",
        code_link=(
            "```python\n"
            "from sklearn.cluster import AgglomerativeClustering\n"
            "\n"
            "agg = AgglomerativeClustering(n_clusters=3, linkage='ward')\n"
            "labels = agg.fit_predict(X_scaled)\n"
            "```"
        ),
        keywords=["agglomerative", "hierarchical", "bottom-up", "merge", "linkage"],
    ),

    "hierarchical_clustering": T(
        title="Hierarchical Clustering",
        module="clustering",
        what=(
            "Hierarchical clustering builds a tree of clusters "
            "(dendrogram). You can 'cut' the tree at different levels "
            "to get different numbers of clusters."
        ),
        why=(
            "Dendrograms visualise the full clustering hierarchy. "
            "They help you see how many natural groups exist and how "
            "clusters relate to each other."
        ),
        when=(
            "Use when you want to explore multiple values of K "
            "simultaneously. Visualise the dendrogram to make "
            "an informed K choice."
        ),
        example=(
            "```python\n"
            "from scipy.cluster.hierarchy import dendrogram, linkage\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "Z = linkage(X_scaled, method='ward')\n"
            "dendrogram(Z, truncate_mode='lastp', p=30)\n"
            "plt.title('Dendrogram')\n"
            "plt.xlabel('Samples')\n"
            "plt.ylabel('Distance')\n"
            "plt.show()\n"
            "```"
        ),
        mistakes=[
            "Using on large datasets — dendrograms become unreadable.",
            "Not truncating the dendrogram for large datasets.",
        ],
        interpretation=(
            "Tall vertical lines in the dendrogram indicate natural "
            "cluster boundaries. Cut where vertical lines are longest."
        ),
        think_about_it="Your dendrogram has two main branches, each with three sub-branches. How many clusters would you choose?",
        code_link=(
            "```python\n"
            "from scipy.cluster.hierarchy import dendrogram, linkage\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "Z = linkage(X_scaled, method='ward')\n"
            "plt.figure(figsize=(10, 5))\n"
            "dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90)\n"
            "plt.title('Dendrogram (truncated)')\n"
            "plt.show()\n"
            "```"
        ),
        keywords=["hierarchical", "dendrogram", "tree", "cut", "level"],
    ),

    "dendrograms": T(
        title="Reading Dendrograms",
        module="clustering",
        what=(
            "A dendrogram is a tree diagram showing how clusters merge "
            "during hierarchical clustering. The height of a merge "
            "indicates the distance between the merged clusters."
        ),
        why=(
            "Dendrograms help you determine the optimal number of "
            "clusters by showing the distance at which clusters form. "
            "Large jumps indicate natural cluster boundaries."
        ),
        when=(
            "Always visualise the dendrogram before choosing K in "
            "hierarchical clustering. Use truncate_mode for large datasets."
        ),
        example=(
            "```python\n"
            "from scipy.cluster.hierarchy import dendrogram, linkage\n"
            "\n"
            "Z = linkage(X_scaled, method='ward')\n"
            "# Look for the largest vertical gap — cut there\n"
            "dendrogram(Z)\n"
            "plt.axhline(y=15, color='r', linestyle='--')  # cutting line\n"
            "```"
        ),
        mistakes=[
            "Reading the x-axis order as meaningful — it's just layout.",
            "Not truncating for large datasets.",
        ],
        interpretation=(
            "Long vertical lines = clusters that are well-separated. "
            "Cut the dendrogram where you see the longest vertical gap."
        ),
        think_about_it="Two clusters merge at height 0.5 and another pair merges at height 5.0. What does this tell you?",
        code_link=(
            "```python\n"
            "from scipy.cluster.hierarchy import dendrogram, linkage, fcluster\n"
            "\n"
            "Z = linkage(X_scaled, method='ward')\n"
            "labels = fcluster(Z, t=3, criterion='maxclust')  # 3 clusters\n"
            "print(f'Cluster labels: {np.unique(labels)}')\n"
            "```"
        ),
        keywords=["dendrogram", "tree", "height", "merge", "cut"],
    ),

    "scaling_for_clustering": T(
        title="Scaling Before Clustering",
        module="clustering",
        what=(
            "Clustering algorithms use distance measures, so features "
            "with larger scales dominate. Scaling ensures all features "
            "contribute equally."
        ),
        why=(
            "Without scaling, income (0-200K) dominates age (0-100) "
            "in distance calculations. Income would determine clusters "
            "almost exclusively."
        ),
        when=(
            "ALWAYS scale before clustering (K-Means, DBSCAN, "
            "Agglomerative). Use StandardScaler or MinMaxScaler."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "```"
        ),
        mistakes=[
            "Clustering without scaling — most common mistake.",
            "Scaling after clustering.",
            "Not using the same scaler for new data.",
        ],
        interpretation=(
            "After scaling, all features have mean=0 and std=1. "
            "Each feature contributes equally to the distance calculation."
        ),
        think_about_it="You cluster data without scaling and get clusters dominated by one feature. How would you verify this is the problem?",
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "\n"
            "# Verify scaling\n"
            "print(f'Mean: {X_scaled.mean(axis=0).round(4)}')\n"
            "print(f'Std:  {X_scaled.std(axis=0).round(4)}')\n"
            "```"
        ),
        keywords=["scaling", "standard", "normalize", "distance", "essential"],
    ),

    "pca_for_visualization": T(
        title="PCA for Cluster Visualization",
        module="clustering",
        what=(
            "PCA reduces high-dimensional data to 2D for visualising "
            "clusters. It preserves the maximum variance, showing "
            "cluster structure in a plot."
        ),
        why=(
            "You can't visualise data with >2 features directly. PCA "
            "projects data to 2D while preserving as much structure "
            "as possible."
        ),
        when=(
            "After clustering with >2 features, use PCA to visualise "
            "the clusters in 2D scatter plots."
        ),
        example=(
            "```python\n"
            "from sklearn.decomposition import PCA\n"
            "import plotly.express as px\n"
            "\n"
            "pca = PCA(n_components=2)\n"
            "X_2d = pca.fit_transform(X_scaled)\n"
            "\n"
            "df_plot = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])\n"
            "df_plot['cluster'] = labels.astype(str)\n"
            "fig = px.scatter(df_plot, x='PC1', y='PC2', color='cluster')\n"
            "fig.show()\n"
            "```"
        ),
        mistakes=[
            "Interpreting PCA components as original features.",
            "Forgetting that PCA is a projection — structure may be lost.",
        ],
        interpretation=(
            "If clusters are well-separated in 2D PCA, they're likely "
            "meaningful. If they overlap heavily, the clusters may not "
            "be well-defined."
        ),
        think_about_it="Clusters look good in 2D PCA but overlap in 3D. What does this tell you?",
        code_link=(
            "```python\n"
            "from sklearn.decomposition import PCA\n"
            "import plotly.express as px\n"
            "\n"
            "pca = PCA(n_components=2)\n"
            "X_2d = pca.fit_transform(X_scaled)\n"
            "print(f'Variance explained: {pca.explained_variance_ratio_.sum():.2%}')\n"
            "\n"
            "fig = px.scatter(x=X_2d[:,0], y=X_2d[:,1], color=labels.astype(str))\n"
            "fig.update_layout(title='Clusters (PCA 2D)')\n"
            "fig.show()\n"
            "```"
        ),
        keywords=["pca", "visualization", "2d", "dimensionality reduction", "scatter"],
    ),

    "cluster_interpretation": T(
        title="Interpreting Clustering Results",
        module="clustering",
        what=(
            "After clustering, examine cluster profiles to understand "
            "what each group represents. Compare cluster means/medians "
            "across features."
        ),
        why=(
            "Cluster labels (0, 1, 2) are meaningless without "
            "interpretation. Domain expertise turns mathematical "
            "clusters into actionable insights."
        ),
        when=(
            "Always after clustering. Use groupby().mean() to profile "
            "clusters and assign meaningful names."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "df['cluster'] = labels\n"
            "profiles = df.groupby('cluster').mean()\n"
            "print(profiles)\n"
            "# Cluster 0: high income, low spending → 'Savers'\n"
            "# Cluster 1: low income, high spending → 'Spenders'\n"
            "```"
        ),
        mistakes=[
            "Not examining cluster profiles after clustering.",
            "Using only visualisation without statistical comparison.",
            "Assigning arbitrary names without data support.",
        ],
        interpretation=(
            "Look for: which features differ most between clusters, "
            "cluster sizes (balanced vs dominated), and whether "
            "clusters are actionable."
        ),
        think_about_it="One cluster has only 2% of customers. Is this a useful segment or an artifact?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "df['cluster'] = labels\n"
            "cluster_profile = df.groupby('cluster').agg(['mean', 'median', 'count'])\n"
            "print(cluster_profile)\n"
            "\n"
            "# Cluster sizes\n"
            "print(df['cluster'].value_counts())\n"
            "```"
        ),
        keywords=["interpret", "profile", "mean", "description", "meaning"],
    ),

    "clustering_limitations": T(
        title="Limitations of Clustering",
        module="clustering",
        what=(
            "Clustering has important limitations: no ground truth "
            "for evaluation, sensitive to parameters, assumes certain "
            "cluster shapes, and results can be unstable."
        ),
        why=(
            "Understanding limitations prevents over-interpreting "
            "results and helps you choose the right algorithm "
            "and parameters."
        ),
        when=(
            "Know these limitations before presenting clustering "
            "results to stakeholders."
        ),
        example=(
            "Limitations:\n"
            "- K-Means: assumes spherical clusters of equal size\n"
            "- DBSCAN: struggles with varying densities\n"
            "- Agglomerative: O(n²) memory, sensitive to noise\n"
            "- All: no 'correct' answer without ground truth"
        ),
        mistakes=[
            "Presenting clustering results as definitive truth.",
            "Not mentioning sensitivity to parameters.",
            "Using clustering for causal claims (clusters ≠ causes).",
        ],
        interpretation=(
            "Clustering reveals structure in data, but that structure "
            "depends on: features used, distance metric, algorithm, "
            "and parameters. Different choices → different clusters."
        ),
        think_about_it="You get very different clusters with K=3 vs K=5. How would you explain this to a non-technical stakeholder?",
        code_link=(
            "```python\n"
            "# Compare different K values\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import silhouette_score\n"
            "\n"
            "for k in [3, 5, 7]:\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    sil = silhouette_score(X_scaled, labels)\n"
            "    print(f'K={k}: Silhouette={sil:.3f}, Sizes={np.bincount(labels)}')\n"
            "```"
        ),
        keywords=["limitations", "weaknesses", "sensitivity", "assumptions", "caution"],
    ),

    "clustering_case_study": T(
        title="Clustering Case Study",
        module="clustering",
        what=(
            "A complete clustering workflow: scaling, K-Means, "
            "elbow method, silhouette, DBSCAN, and cluster interpretation."
        ),
        why=(
            "Seeing the full workflow connects individual concepts "
            "into a practical clustering pipeline."
        ),
        when="Use as a reference for any clustering project.",
        example="Complete customer segmentation workflow.",
        mistakes=[
            "Only using one algorithm.",
            "Not interpreting clusters with business context.",
        ],
        interpretation=(
            "Good clustering produces segments that are statistically "
            "distinct and practically useful."
        ),
        think_about_it="After clustering, how would you validate that the segments are actually useful for the business?",
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans, DBSCAN\n"
            "from sklearn.metrics import silhouette_score\n"
            "from sklearn.decomposition import PCA\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# 1. Scale\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "\n"
            "# 2. Elbow method\n"
            "for k in range(2, 8):\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    print(f'K={k}: Silhouette={silhouette_score(X_scaled, labels):.3f}')\n"
            "\n"
            "# 3. Final clustering\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "\n"
            "# 4. PCA visualization\n"
            "pca = PCA(n_components=2)\n"
            "X_2d = pca.fit_transform(X_scaled)\n"
            "plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')\n"
            "plt.title('Customer Segments')\n"
            "plt.show()\n"
            "\n"
            "# 5. Interpret\n"
            "df['segment'] = labels\n"
            "print(df.groupby('segment').mean())\n"
            "```"
        ),
        keywords=["case study", "workflow", "end-to-end", "segmentation", "complete"],
    ),
}
