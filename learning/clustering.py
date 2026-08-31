"""Clustering curriculum — 20 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="clus_01", title="What is Unsupervised Learning?",
        section="clustering", order=1, difficulty="beginner",
        objectives=[
            "Define unsupervised learning and its goal",
            "Distinguish unsupervised from supervised learning",
            "Understand that no labels are available",
        ],
        concept=(
            "Unsupervised learning finds patterns in data without labels. There is no "
            "'correct answer' to learn from — the algorithm discovers structure on its own. "
            "Unlike supervised learning, there is no y to predict; the model explores the "
            "structure of X alone."
        ),
        why_matters=(
            "Most real-world data is unlabelled. Labelling is expensive and sometimes impossible. "
            "Unsupervised learning reveals hidden structure: customer segments, gene groups, "
            "document topics, anomaly patterns."
        ),
        simple_explanation=(
            "Imagine a teacher giving you a box of mixed Lego bricks without instructions. "
            "You group similar pieces together by colour, shape, or type — that's unsupervised learning."
        ),
        example=(
            "A retailer has 100,000 customers with purchase history but no labels. "
            "Unsupervised learning groups them into segments like 'bargain hunters', "
            "'premium shoppers', and 'occasional buyers' — without being told these groups exist."
        ),
        common_mistakes=[
            "Expecting supervised-learning-style accuracy metrics (there is no ground truth)",
            "Not evaluating results with multiple methods (silhouette, visual inspection, domain knowledge)",
            "Assuming discovered clusters are 'real' — they are mathematical groupings, not necessarily meaningful",
        ],
        practice_exercise=(
            "Open the Clustering Lab and load a dataset. Before running any algorithm, "
            "look at two numerical features in a scatter plot. Can you visually see groups? "
            "How many? This is the intuition unsupervised learning tries to formalise."
        ),
        quiz=[
            QuizQuestion(
                question="Which of the following is an unsupervised learning task?",
                options=[
                    "Predicting house prices from features",
                    "Classifying emails as spam or not spam",
                    "Grouping customers by purchasing behaviour without predefined labels",
                    "Predicting whether a patient has a disease",
                ],
                correct_index=2,
                explanation=(
                    "Grouping customers without predefined labels is clustering — unsupervised. "
                    "All other options have a target variable (price, spam/not-spam, disease) "
                    "making them supervised learning tasks."
                ),
            ),
        ],
        takeaways=[
            "Unsupervised = no labels, no ground truth",
            "The algorithm discovers structure, it doesn't predict",
            "Results always require human interpretation and domain validation",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_02", title="Supervised vs Unsupervised Learning",
        section="clustering", order=2, difficulty="beginner",
        objectives=[
            "Compare supervised and unsupervised paradigms",
            "Know when to use each approach",
            "Understand semi-supervised learning",
        ],
        concept=(
            "Supervised learning: learn from labelled data — map X → y (classification, "
            "regression). Unsupervised learning: discover structure in unlabelled data "
            "(clustering, dimensionality reduction). Semi-supervised: small number of labels "
            "plus large unlabelled data."
        ),
        why_matters=(
            "Choosing the wrong paradigm wastes time. If you have labels, use them (supervised). "
            "If you don't, unsupervised methods can still extract useful information. "
            "Semi-supervised is valuable when labelling is expensive."
        ),
        example=(
            "Medical imaging: Labelling every X-ray costs expert time.\n"
            "• Supervised: train on labelled X-rays → classify disease\n"
            "• Unsupervised: find natural groupings in X-ray features\n"
            "• Semi-supervised: label 100 X-rays, let the model learn from 10,000 unlabelled ones"
        ),
        common_mistakes=[
            "Trying to use supervised metrics (accuracy, F1) on unsupervised results",
            "Not considering semi-supervised approaches when some labels exist",
            "Assuming unsupervised results are less valuable than supervised ones",
        ],
        practice_exercise=(
            "For each scenario, decide: supervised, unsupervised, or semi-supervised?\n"
            "1. You have 50 labelled customer reviews and 5000 unlabelled ones.\n"
            "2. You want to detect unusual network traffic patterns.\n"
            "3. You have 10,000 images with labels for 10 object categories."
        ),
        quiz=[
            QuizQuestion(
                question="When is semi-supervised learning most appropriate?",
                options=[
                    "When you have a large labelled dataset",
                    "When you have a small labelled dataset and a large unlabelled one",
                    "When you have no data at all",
                    "When you need real-time predictions",
                ],
                correct_index=1,
                explanation=(
                    "Semi-supervised learning shines when labels are scarce but unlabelled data "
                    "is abundant. It leverages the structure in unlabelled data to improve "
                    "the small set of labels."
                ),
            ),
        ],
        takeaways=[
            "Supervised = labels exist; Unsupervised = no labels",
            "Different goals, different evaluation methods",
            "Semi-supervised bridges the gap when labels are scarce",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_03", title="What is Clustering?",
        section="clustering", order=3, difficulty="beginner",
        objectives=[
            "Define clustering precisely",
            "Identify clustering applications",
            "Understand what makes a 'good' cluster",
        ],
        concept=(
            "Clustering groups similar data points so that points within the same group "
            "(cluster) are more similar to each other than to points in other groups. "
            "Similarity is defined by a distance metric (usually Euclidean distance)."
        ),
        why_matters=(
            "Clustering is used in customer segmentation (marketing), document grouping "
            "(search engines), anomaly detection (fraud), image segmentation (computer vision), "
            "and gene expression analysis (bioinformatics)."
        ),
        simple_explanation=(
            "Clustering is like sorting a mixed bag of sweets into groups by type — "
            "without being told what types exist. You use visual similarity to group them."
        ),
        example=(
            "A shopping mall groups visitors by spending patterns:\n"
            "• Cluster 1: frequent, low-spend → 'regular browsers'\n"
            "• Cluster 2: rare, high-spend → 'luxury shoppers'\n"
            "• Cluster 3: frequent, medium-spend → 'loyal customers'\n"
            "The mall tailors marketing to each group."
        ),
        common_mistakes=[
            "Assuming clusters must be spherical (K-Means assumes this, but DBSCAN doesn't)",
            "Not scaling features before clustering (distance-based algorithms are scale-sensitive)",
            "Choosing the number of clusters arbitrarily without using data-driven methods",
        ],
        practice_exercise=(
            "Load the Iris dataset in the Clustering Lab. Plot sepal_length vs petal_length. "
            "How many groups do you see? Now run K-Means with k=3. Does it match your visual impression?"
        ),
        quiz=[
            QuizQuestion(
                question="What defines a 'good' cluster?",
                options=[
                    "All clusters must have the same number of points",
                    "Points within a cluster are more similar to each other than to points in other clusters",
                    "Clusters must be perfectly separated",
                    "Every cluster must have at least 100 points",
                ],
                correct_index=1,
                explanation=(
                    "The fundamental criterion for clustering is intra-cluster similarity: "
                    "points within the same cluster should be more similar to each other "
                    "than to points in other clusters. Clusters don't need to be equal-sized "
                    "or perfectly separated."
                ),
            ),
        ],
        takeaways=[
            "Clustering = grouping similar points together",
            "No ground truth needed — the algorithm discovers groups",
            "Quality depends on the definition of 'similar' (distance metric)",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_04", title="Applications of Clustering",
        section="clustering", order=4, difficulty="beginner",
        objectives=[
            "Identify real-world clustering applications",
            "Connect clustering to business problems",
            "Understand domain-specific clustering needs",
        ],
        concept=(
            "Clustering applications span many domains:\n"
            "• Customer segmentation (marketing): group buyers by behaviour\n"
            "• Document clustering (NLP): organise articles by topic\n"
            "• Anomaly detection (security): find unusual patterns\n"
            "• Image segmentation (CV): separate objects in images\n"
            "• Gene clustering (biology): find co-expressed genes\n"
            "• Social network analysis: detect communities"
        ),
        why_matters=(
            "Clustering solves real problems across industries. Understanding applications "
            "helps you choose the right algorithm and parameters for your specific domain."
        ),
        example=(
            "A bank uses clustering on transaction data:\n"
            "• Normal transaction cluster: typical amounts, familiar merchants\n"
            "• Unusual cluster: large amounts, foreign merchants → potential fraud\n"
            "• New customer cluster: few transactions, small amounts → nurture campaign"
        ),
        common_mistakes=[
            "Using clustering without a clear use case or business question",
            "Not validating clusters against domain knowledge",
            "Assuming more clusters always means better insight",
        ],
        practice_exercise=(
            "Think of a clustering application in your field of study. "
            "1. What data would you use?\n"
            "2. What features would matter?\n"
            "3. How many clusters might you expect?\n"
            "4. How would you validate the results?"
        ),
        quiz=[
            QuizQuestion(
                question="A hospital wants to group patients by treatment response patterns. Which approach is most appropriate?",
                options=[
                    "Classification — predict which treatment works",
                    "Clustering — discover natural response groups",
                    "Regression — predict recovery time",
                    "None — this is not a machine learning problem",
                ],
                correct_index=1,
                explanation=(
                    "Without predefined response categories, clustering discovers natural "
                    "groups in how patients respond to treatment. This could reveal that some "
                    "patients respond well to treatment A while others need treatment B."
                ),
            ),
        ],
        takeaways=[
            "Clustering solves real problems across industries",
            "Always validate clusters with domain experts",
            "Different applications need different cluster numbers and algorithms",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_05", title="K-Means",
        section="clustering", order=5, difficulty="intermediate",
        objectives=[
            "Understand the K-Means algorithm steps",
            "Apply K-Means with sklearn",
            "Interpret K-Means results",
        ],
        concept=(
            "K-Means algorithm:\n"
            "1. Initialise k centroids randomly\n"
            "2. Assign each point to the nearest centroid\n"
            "3. Update each centroid to the mean of assigned points\n"
            "4. Repeat steps 2-3 until centroids stop moving (convergence)\n\n"
            "The algorithm minimises within-cluster sum of squares (inertia)."
        ),
        why_matters=(
            "K-Means is the most widely used clustering algorithm. It's fast, simple, "
            "and works well when clusters are roughly spherical and equally sized. "
            "Understanding K-Means is essential before learning more advanced methods."
        ),
        example=(
            "A retail chain has customer data with annual_spend and visit_frequency. "
            "K-Means with k=3 identifies:\n"
            "• Cluster 0: low spend, low frequency → 'occasional shoppers'\n"
            "• Cluster 1: high spend, high frequency → 'loyal customers'\n"
            "• Cluster 2: high spend, low frequency → 'big spenders'"
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "import pandas as pd\n\n"
            "# Always scale first!\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n\n"
            "# Fit K-Means\n"
            "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n"
            "clusters = kmeans.fit_predict(X_scaled)\n\n"
            "print(f'Cluster sizes: {pd.Series(clusters).value_counts().sort_index().tolist()}')\n"
            "print(f'Inertia (within-cluster sum of squares): {kmeans.inertia_:.1f}')\n"
            "\n"
            "# Centroid positions (in scaled space)\n"
            "centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)\n"
            "print(centroids.round(2))\n"
            "```"
        ),
        interpretation=(
            "Inertia: lower is better (tighter clusters). Cluster sizes should be reasonable — "
            "a cluster with 2 points out of 1000 suggests k is too high. Centroid positions "
            "reveal what each cluster 'looks like' in feature space."
        ),
        common_mistakes=[
            "Not scaling features — K-Means is distance-based, features with larger ranges dominate",
            "Assuming the first initialisation is best (n_init=10 runs it 10 times and picks the best)",
            "Using K-Means on non-spherical clusters (use DBSCAN for arbitrary shapes)",
        ],
        practice_exercise=(
            "Run K-Means on California Housing with k=2, 3, 4, 5. "
            "1. How does inertia change?\n"
            "2. Are cluster sizes balanced?\n"
            "3. What characterises each cluster (check centroid values)?"
        ),
        quiz=[
            QuizQuestion(
                question="What happens if you run K-Means without scaling features where one feature ranges 0-1000 and another ranges 0-1?",
                options=[
                    "Both features contribute equally to distance",
                    "The feature with range 0-1000 dominates the clustering completely",
                    "K-Means automatically normalises features",
                    "It crashes with an error",
                ],
                correct_index=1,
                explanation=(
                    "K-Means uses Euclidean distance. A difference of 500 in the first feature "
                    "dwarfs any difference in the second feature (max range 1). The clustering "
                    "will be determined almost entirely by the large-range feature."
                ),
            ),
        ],
        takeaways=[
            "K-Means needs k specified in advance",
            "Always scale features before K-Means",
            "Assumes spherical, equally-sized clusters",
            "Use n_init=10+ to avoid poor initialisations",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_06", title="Centroids",
        section="clustering", order=6, difficulty="intermediate",
        objectives=[
            "Understand the role of centroids in K-Means",
            "Interpret centroid positions to profile clusters",
            "Use centroids for cluster analysis",
        ],
        concept=(
            "Centroids are the 'centres' of clusters in K-Means. Each centroid represents "
            "the average position of all points assigned to that cluster. K-Means iteratively "
            "adjusts centroids to minimise the average distance from each point to its centroid."
        ),
        why_matters=(
            "Centroid positions reveal cluster characteristics. By examining centroid coordinates, "
            "you can profile what each cluster 'looks like' — high spending vs low spending, "
            "young vs old, etc."
        ),
        example=(
            "Customer segmentation centroids:\n"
            "• Cluster 0 centroid: annual_spend=£2,100, frequency=3/year → 'occasional'\n"
            "• Cluster 1 centroid: annual_spend=£12,500, frequency=45/year → 'loyal'\n"
            "• Cluster 2 centroid: annual_spend=£8,000, frequency=5/year → 'big spender'\n"
            "Comparing centroids tells the story of each segment."
        ),
        python_example=(
            "```python\n"
            "import pandas as pd\n\n"
            "# Centroid positions reveal cluster profiles\n"
            "centroids = pd.DataFrame(\n"
            "    kmeans.cluster_centers_, columns=feature_names\n"
            ")\n"
            "print('Cluster Centroids (scaled):')\n"
            "print(centroids.round(2))\n\n"
            "# Inverse transform to original scale for interpretation\n"
            "centroids_original = pd.DataFrame(\n"
            "    scaler.inverse_transform(kmeans.cluster_centers_),\n"
            "    columns=feature_names\n"
            ")\n"
            "print('\\nCluster Centroids (original scale):')\n"
            "print(centroids_original.round(2))\n"
            "```"
        ),
        common_mistakes=[
            "Assuming centroids are actual data points (they are averages, often not real observations)",
            "Not interpreting centroid positions in the context of the original feature values",
            "Ignoring that centroid comparison in scaled space is misleading without inverse transforming",
        ],
        practice_exercise=(
            "After running K-Means, print the centroid values in original scale. "
            "1. What characterises each cluster?\n"
            "2. Give each cluster a meaningful name.\n"
            "3. Which cluster would you target for a marketing campaign?"
        ),
        quiz=[
            QuizQuestion(
                question="A K-Means centroid has values {age: 25, income: £18000, spending: 3}. What does this represent?",
                options=[
                    "The oldest person in that cluster",
                    "The average values of all points assigned to that cluster",
                    "The most common point in that cluster",
                    "The first point assigned to that cluster",
                ],
                correct_index=1,
                explanation=(
                    "A centroid is the mean (average) position of all points in the cluster. "
                    "It represents the 'typical' profile of that cluster's members. "
                    "It may not correspond to any actual data point."
                ),
            ),
        ],
        takeaways=[
            "Centroids = cluster centres (averages of assigned points)",
            "Compare centroid positions to profile clusters",
            "Inverse transform centroids to interpret in original scale",
            "Centroids may not correspond to real data points",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_07", title="Distance Measures",
        section="clustering", order=7, difficulty="intermediate",
        objectives=[
            "Understand Euclidean, Manhattan, and Cosine distance",
            "Choose the appropriate distance metric",
            "Know when Euclidean distance fails",
        ],
        concept=(
            "Distance metrics define 'similarity':\n"
            "• Euclidean (L2): straight-line distance √Σ(xᵢ-yᵢ)² — the default\n"
            "• Manhattan (L1): city-block distance Σ|xᵢ-yᵢ| — more robust to outliers\n"
            "• Cosine: angle between vectors — measures direction, not magnitude\n"
            "Different metrics suit different data types and problems."
        ),
        why_matters=(
            "The choice of distance metric fundamentally affects clustering results. "
            "For text data, cosine distance captures topic similarity regardless of document "
            "length. For spatial data, Euclidean distance is natural."
        ),
        example=(
            "Document clustering:\n"
            "• Euclidean: 'the cat sat' and 'a cat sat on a mat' are far apart (different lengths)\n"
            "• Cosine: same direction in term-frequency space → similar topic, close distance\n"
            "Cosine is better for text because it ignores document length."
        ),
        common_mistakes=[
            "Always using Euclidean without considering alternatives",
            "Not scaling features — features with larger ranges dominate distance calculations",
            "Using Euclidean on very high-dimensional data (distance becomes meaningless — curse of dimensionality)",
        ],
        practice_exercise=(
            "Using two features from a dataset:\n"
            "1. Compute Euclidean distance between two points manually\n"
            "2. Compute Manhattan distance between the same points\n"
            "3. Which is larger? Why?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is cosine distance often preferred over Euclidean for text clustering?",
                options=[
                    "Cosine is faster to compute",
                    "Cosine measures the angle (topic direction) rather than magnitude (document length)",
                    "Euclidean cannot handle sparse data",
                    "Cosine always produces more clusters",
                ],
                correct_index=1,
                explanation=(
                    "Cosine distance measures the angle between vectors, ignoring their magnitude. "
                    "A short document about cats and a long document about cats point in the same "
                    "direction in term-frequency space. Euclidean would consider them very different "
                    "because of the length difference."
                ),
            ),
        ],
        takeaways=[
            "Euclidean: default, for continuous numerical features",
            "Manhattan: more robust to outliers",
            "Cosine: best for text and high-dimensional data",
            "Always scale features before using Euclidean/Manhattan distance",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_08", title="Choosing K",
        section="clustering", order=8, difficulty="intermediate",
        objectives=[
            "Apply the elbow method and silhouette analysis",
            "Use domain knowledge to select k",
            "Understand that there is no single 'correct' k",
        ],
        concept=(
            "Choosing the number of clusters k is the hardest part of K-Means. Methods:\n"
            "1. Elbow method: plot inertia vs k, look for the 'elbow'\n"
            "2. Silhouette score: measure cluster quality for each k\n"
            "3. Domain knowledge: how many groups make practical sense?\n"
            "4. Gap statistic: compare with random uniform data\n"
            "No method is definitive — use multiple approaches together."
        ),
        why_matters=(
            "Choosing the wrong k produces meaningless results. Too few clusters merge "
            "distinct groups; too many create artificial subdivisions. The optimal k depends "
            "on the data structure and the practical requirements of the application."
        ),
        example=(
            "Customer segmentation:\n"
            "• k=2: 'high value' vs 'low value' — too coarse\n"
            "• k=3: 'budget', 'regular', 'premium' — makes business sense\n"
            "• k=7: too many segments to act on meaningfully\n"
            "The elbow method suggests k=3-4; domain knowledge confirms k=3."
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import silhouette_score\n"
            "import matplotlib.pyplot as plt\n\n"
            "K_range = range(2, 11)\n"
            "inertias = []\n"
            "silhouettes = []\n\n"
            "for k in K_range:\n"
            "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "    silhouettes.append(silhouette_score(X_scaled, labels))\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "axes[0].plot(K_range, inertias, 'bo-')\n"
            "axes[0].set_title('Elbow Method')\n"
            "axes[0].set_xlabel('k')\n"
            "axes[0].set_ylabel('Inertia')\n"
            "axes[1].plot(K_range, silhouettes, 'ro-')\n"
            "axes[1].set_title('Silhouette Score')\n"
            "axes[1].set_xlabel('k')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Choosing k=2 by default without testing alternatives",
            "Using only one method — elbow can be ambiguous, silhouette can mislead",
            "Not considering whether the resulting number of clusters is actionable",
        ],
        practice_exercise=(
            "Run the elbow method and silhouette analysis on the Iris dataset with k=2..10. "
            "1. Where is the 'elbow' in the inertia plot?\n"
            "2. Which k maximises silhouette score?\n"
            "3. Given that Iris has 3 species, does the optimal k match?"
        ),
        quiz=[
            QuizQuestion(
                question="The elbow method shows a smooth curve with no clear 'elbow'. What should you do?",
                options=[
                    "Pick k=2 as a safe default",
                    "Combine with silhouette analysis and domain knowledge to choose k",
                    "The data has no clusters — clustering is not appropriate",
                    "Use k equal to the number of features",
                ],
                correct_index=1,
                explanation=(
                    "A smooth elbow curve is common and doesn't mean clustering is impossible. "
                    "Use silhouette scores, domain knowledge, and try a few values of k to see "
                    "which produces the most interpretable results."
                ),
            ),
        ],
        takeaways=[
            "Use elbow method + silhouette score + domain knowledge together",
            "Higher silhouette = better-defined clusters",
            "No single 'right' k — it depends on the application",
            "Test k from 2 to a reasonable maximum (e.g., √n or 10)",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_09", title="Elbow Method",
        section="clustering", order=9, difficulty="intermediate",
        objectives=[
            "Apply the elbow method to select k",
            "Interpret the elbow plot correctly",
            "Understand the method's limitations",
        ],
        concept=(
            "The elbow method plots inertia (within-cluster sum of squares) against k. "
            "As k increases, inertia always decreases (more clusters = tighter groups). "
            "The 'elbow' is where the rate of decrease sharply changes — adding more clusters "
            "beyond this point gives diminishing returns."
        ),
        why_matters=(
            "The elbow method provides a data-driven way to choose k instead of guessing. "
            "It formalises the intuition that the 'right' k balances cluster tightness "
            "with simplicity."
        ),
        example=(
            "Inertia for customer data: k=2 → 10,000, k=3 → 5,500, k=4 → 4,800, k=5 → 4,500. "
            "Big drop from k=2 to k=3 (4,500 reduction). Smaller drop from k=3 to k=4 (700). "
            "The elbow is at k=3 — beyond that, diminishing returns."
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "import matplotlib.pyplot as plt\n\n"
            "inertias = []\n"
            "K_range = range(2, 11)\n"
            "for k in K_range:\n"
            "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
            "    km.fit(X_scaled)\n"
            "    inertias.append(km.inertia_)\n\n"
            "plt.figure(figsize=(8, 5))\n"
            "plt.plot(K_range, inertias, 'bo-', linewidth=2)\n"
            "plt.xlabel('Number of Clusters (k)')\n"
            "plt.ylabel('Inertia')\n"
            "plt.title('Elbow Method')\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Expecting a clear elbow (often the curve is smooth — that's normal)",
            "Choosing k too low (misses meaningful structure) or too high (artificial subdivisions)",
            "Not combining with silhouette analysis for confirmation",
        ],
        practice_exercise=(
            "Plot the elbow curve for k=2..15 on a dataset. "
            "1. Is there a clear elbow?\n"
            "2. Try k values around the elbow. Which gives the most interpretable clusters?\n"
            "3. What happens to cluster sizes as k increases?"
        ),
        quiz=[
            QuizQuestion(
                question="Inertia always decreases as k increases. Why doesn't this mean k=n (one point per cluster) is optimal?",
                options=[
                    "Because inertia doesn't measure cluster quality",
                    "Because k=n gives inertia=0 but creates meaningless 'clusters' of single points",
                    "Because K-Means cannot handle large k values",
                    "Because sklearn limits k to 10",
                ],
                correct_index=1,
                explanation=(
                    "At k=n, each point is its own cluster with zero inertia. But this is useless — "
                    "there is no grouping. The elbow method looks for where adding clusters stops "
                    "providing meaningful improvement, not where inertia reaches zero."
                ),
            ),
        ],
        takeaways=[
            "Elbow = where inertia decrease sharply slows",
            "Look for diminishing returns in the plot",
            "Often ambiguous — combine with silhouette analysis",
            "Domain knowledge helps break ties",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_10", title="Silhouette Score",
        section="clustering", order=10, difficulty="intermediate",
        objectives=[
            "Calculate and interpret silhouette scores",
            "Use per-cluster silhouette analysis",
            "Compare k values using silhouette scores",
        ],
        concept=(
            "For each point, the silhouette score is: s = (b - a) / max(a, b)\n"
            "• a = mean distance to other points in the same cluster (cohesion)\n"
            "• b = mean distance to points in the nearest other cluster (separation)\n"
            "Range: -1 to 1. Score > 0.5 is reasonable; > 0.7 is good."
        ),
        why_matters=(
            "The silhouette score measures both how tightly points are grouped within clusters "
            "and how well-separated clusters are from each other. It provides a more complete "
            "picture than inertia alone."
        ),
        example=(
            "K-Means with k=3: overall silhouette = 0.55. Per-cluster: Cluster 0 = 0.72 "
            "(well-defined), Cluster 1 = 0.41 (poorly defined), Cluster 2 = 0.52 (okay). "
            "This reveals that Cluster 1 is fuzzy and may not be a natural group."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import silhouette_score, silhouette_samples\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "# Overall score\n"
            "score = silhouette_score(X_scaled, clusters)\n"
            "print(f'Overall Silhouette: {score:.3f}')\n\n"
            "# Per-cluster analysis\n"
            "sample_scores = silhouette_samples(X_scaled, clusters)\n"
            "for i in range(k):\n"
            "    cluster_scores = sample_scores[clusters == i]\n"
            "    print(f'Cluster {i}: mean={cluster_scores.mean():.3f}, '\n"
            "          f'min={cluster_scores.min():.3f}, '\n"
            "          f'size={len(cluster_scores)}')\n"
            "```"
        ),
        interpretation=(
            "Score near +1: point is well-matched to its cluster. "
            "Score near 0: point is on the boundary between two clusters. "
            "Score near -1: point is assigned to the wrong cluster. "
            "Overall average > 0.5 is reasonable; per-cluster scores reveal which clusters are fuzzy."
        ),
        common_mistakes=[
            "Only reporting overall silhouette — always check per-cluster scores",
            "Not visualising silhouette plots to see score distributions",
            "Ignoring points with negative silhouette (they may be misclassified)",
        ],
        practice_exercise=(
            "Run K-Means with k=2, 3, 4, 5 and compute the overall silhouette score for each. "
            "1. Which k gives the highest score?\n"
            "2. For the best k, plot the per-cluster silhouette scores.\n"
            "3. Are there any points with negative scores?"
        ),
        quiz=[
            QuizQuestion(
                question="A point has a silhouette score of -0.3. What does this mean?",
                options=[
                    "The point is well-clustered",
                    "The point is on the cluster boundary",
                    "The point is likely assigned to the wrong cluster",
                    "The point is an outlier",
                ],
                correct_index=2,
                explanation=(
                    "A negative silhouette score means the point is closer to a neighbouring "
                    "cluster than to its assigned cluster (b < a). This suggests the point "
                    "may be misclassified. Investigate these points — they may indicate the "
                    "wrong k or a non-spherical cluster structure."
                ),
            ),
        ],
        takeaways=[
            "Silhouette: -1 to 1; higher is better",
            "Check both overall and per-cluster scores",
            "Negative scores indicate potential misclassification",
            "Combine with elbow method for robust k selection",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_11", title="DBSCAN",
        section="clustering", order=11, difficulty="intermediate",
        objectives=[
            "Apply DBSCAN clustering",
            "Tune eps and min_samples parameters",
            "Identify and interpret noise points",
        ],
        concept=(
            "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups "
            "densely packed points and marks sparse points as noise. Parameters:\n"
            "• eps: maximum distance between two points to be considered neighbours\n"
            "• min_samples: minimum points in a neighbourhood to form a dense region\n"
            "Key advantage: finds arbitrarily-shaped clusters and detects outliers."
        ),
        why_matters=(
            "Real-world clusters are rarely spherical. Customer segments may be elongated, "
            "network traffic clusters may be irregularly shaped. DBSCAN handles these naturally "
            "and automatically identifies outliers as noise."
        ),
        example=(
            "Geographic data: houses along a river form a long, narrow cluster. "
            "K-Means (spherical assumption) splits this into two artificial groups. "
            "DBSCAN follows the river shape and keeps it as one cluster. "
            "Isolated houses far from any group are marked as noise."
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n\n"
            "dbscan = DBSCAN(eps=0.5, min_samples=5)\n"
            "clusters = dbscan.fit_predict(X_scaled)\n\n"
            "n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)\n"
            "n_noise = list(clusters).count(-1)\n"
            "print(f'Clusters found: {n_clusters}')\n"
            "print(f'Noise points: {n_noise}')\n"
            "print(f'Cluster sizes: {pd.Series(clusters[clusters != -1]).value_counts().sort_index().tolist()}')\n"
            "```"
        ),
        common_mistakes=[
            "Not scaling features before DBSCAN (it uses distance too)",
            "Choosing eps without visualising the k-distance plot",
            "Expecting all points to be clustered — noise is a feature, not a bug",
        ],
        practice_exercise=(
            "Run DBSCAN on a dataset. Try eps values [0.3, 0.5, 0.7, 1.0] with min_samples=5. "
            "1. How does the number of clusters change?\n"
            "2. How does the number of noise points change?\n"
            "3. Which eps gives the most interpretable result?"
        ),
        quiz=[
            QuizQuestion(
                question="What happens when you increase the eps parameter in DBSCAN?",
                options=[
                    "More clusters are found",
                    "Points become more spread out",
                    "Neighbourhoods expand, potentially merging clusters and reducing noise points",
                    "The algorithm runs faster",
                ],
                correct_index=2,
                explanation=(
                    "Larger eps means each point has a larger neighbourhood. More points become "
                    "reachable from each other, so small clusters merge and some noise points "
                    "get absorbed into clusters. Very large eps puts everything in one cluster."
                ),
            ),
        ],
        takeaways=[
            "DBSCAN finds arbitrarily-shaped clusters",
            "Noise points (labelled -1) are a feature, not a bug",
            "No need to specify k — eps and min_samples control clustering",
            "Use the k-distance plot to choose eps",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_12", title="Core, Border and Noise Points",
        section="clustering", order=12, difficulty="intermediate",
        objectives=[
            "Distinguish core, border, and noise points",
            "Understand how DBSCAN classifies each point",
            "Use point types for cluster interpretation",
        ],
        concept=(
            "DBSCAN classifies every point as one of three types:\n"
            "• Core point: has at least min_samples neighbours within eps (dense region)\n"
            "• Border point: within eps of a core point but not dense enough to be core itself\n"
            "• Noise point: neither core nor border; doesn't belong to any cluster (label = -1)"
        ),
        why_matters=(
            "Understanding these three point types is essential for interpreting DBSCAN results. "
            "Core points form the backbone of clusters, border points fill the edges, and "
            "noise points may indicate anomalies worth investigating."
        ),
        example=(
            "DBSCAN on sensor data (min_samples=3, eps=0.5):\n"
            "• Core points: sensors in busy areas with many nearby readings\n"
            "• Border points: sensors at the edge of busy areas\n"
            "• Noise points: isolated sensor readings — potential equipment failure"
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import DBSCAN\n"
            "import numpy as np\n\n"
            "dbscan = DBSCAN(eps=0.5, min_samples=5)\n"
            "labels = dbscan.fit_predict(X_scaled)\n\n"
            "# Identify point types\n"
            "core_indices = set(dbscan.core_sample_indices_)\n"
            "n_core = len(core_indices)\n"
            "n_noise = (labels == -1).sum()\n"
            "n_border = len(labels) - n_core - n_noise\n\n"
            "print(f'Core points: {n_core}')\n"
            "print(f'Border points: {n_border}')\n"
            "print(f'Noise points: {n_noise}')\n"
            "```"
        ),
        common_mistakes=[
            "Setting min_samples too high (too few core points → everything becomes noise)",
            "Automatically removing all noise points without investigating them",
            "Confusing border points with noise points",
        ],
        practice_exercise=(
            "Run DBSCAN with min_samples in [3, 5, 10, 20]. "
            "1. How do core, border, and noise counts change?\n"
            "2. With min_samples=20, are there any clusters left?\n"
            "3. What is a reasonable min_samples for your dataset size?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the difference between a border point and a noise point in DBSCAN?",
                options=[
                    "Border points are in clusters; noise points are not",
                    "Border points are within eps of a core point but don't have enough neighbours; noise points are far from any core point",
                    "Border points are always on the edge of the dataset",
                    "There is no difference — both are outliers",
                ],
                correct_index=1,
                explanation=(
                    "A border point is within eps of at least one core point, so it's part of a "
                    "cluster but doesn't meet the min_samples threshold to be core itself. "
                    "A noise point has no core point within eps — it's too isolated for any cluster."
                ),
            ),
        ],
        takeaways=[
            "Core points: dense backbone of clusters",
            "Border points: cluster edges, assigned to the nearest core",
            "Noise points: isolated outliers, potential anomalies worth investigating",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_15", title="Agglomerative Clustering",
        section="clustering", order=15, difficulty="intermediate",
        objectives=[
            "Apply agglomerative clustering with sklearn",
            "Understand linkage methods (ward, complete, average, single)",
            "Interpret the hierarchical structure",
        ],
        concept=(
            "Agglomerative clustering is a bottom-up approach:\n"
            "1. Start with each point as its own cluster\n"
            "2. Find the two closest clusters and merge them\n"
            "3. Repeat until k clusters remain\n\n"
            "Linkage defines 'closest':\n"
            "• Ward: merge clusters that minimise increase in total within-cluster variance (default)\n"
            "• Complete: merge clusters with smallest maximum distance\n"
            "• Average: merge clusters with smallest average distance\n"
            "• Single: merge clusters with smallest minimum distance"
        ),
        why_matters=(
            "Agglomerative clustering doesn't assume spherical clusters. Ward linkage produces "
            "compact, equally-sized clusters (similar to K-Means). Single linkage can find "
            "elongated clusters but is sensitive to noise."
        ),
        example=(
            "Gene expression data: Ward linkage groups genes with similar expression patterns "
            "into tight clusters. Complete linkage ensures all genes in a cluster are similar "
            "to each other. Single linkage can chain together loosely related genes."
        ),
        python_example=(
            "```python\n"
            "from sklearn.cluster import AgglomerativeClustering\n\n"
            "# Ward linkage (default) — minimises variance\n"
            "agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')\n"
            "labels_ward = agg_ward.fit_predict(X_scaled)\n\n"
            "# Complete linkage — compact clusters\n"
            "agg_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')\n"
            "labels_complete = agg_complete.fit_predict(X_scaled)\n\n"
            "# Average linkage\n"
            "agg_avg = AgglomerativeClustering(n_clusters=3, linkage='average')\n"
            "labels_avg = agg_avg.fit_predict(X_scaled)\n"
            "```"
        ),
        common_mistakes=[
            "Not specifying the linkage method (defaults may not suit your data)",
            "Using single linkage on noisy data (chain effect merges distant points)",
            "Not scaling features before clustering",
        ],
        practice_exercise=(
            "Run agglomerative clustering on the same dataset with all four linkage methods "
            "(k=3). Compare the cluster sizes and visualise with PCA. "
            "1. Which linkage produces the most balanced clusters?\n"
            "2. Which produces the most compact clusters?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the main advantage of Ward linkage over single linkage?",
                options=[
                    "Ward is faster to compute",
                    "Ward produces compact, equally-sized clusters by minimising variance increase",
                    "Ward can find any-shaped clusters",
                    "Ward doesn't require specifying k",
                ],
                correct_index=1,
                explanation=(
                    "Ward linkage merges the pair of clusters that causes the smallest increase "
                    "in total within-cluster variance. This produces compact, spherical clusters "
                    "similar to K-Means. Single linkage uses the minimum distance, which can "
                    "create long chains of loosely connected points."
                ),
            ),
        ],
        takeaways=[
            "Agglomerative: bottom-up merging of clusters",
            "Ward linkage is the best default (compact clusters)",
            "Single linkage is sensitive to noise (chain effect)",
            "Can be combined with dendrograms for visual analysis",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_16", title="Hierarchical Clustering",
        section="clustering", order=16, difficulty="intermediate",
        objectives=[
            "Understand hierarchical cluster structure",
            "Choose between agglomerative and divisive approaches",
            "Know when hierarchy adds value",
        ],
        concept=(
            "Hierarchical clustering creates a tree of cluster merges (agglomerative, "
            "bottom-up) or splits (divisive, top-down). The tree can be cut at any level "
            "to obtain different numbers of clusters, without re-running the algorithm."
        ),
        why_matters=(
            "Hierarchy reveals the multi-scale structure of data. Sometimes you need 3 broad "
            "groups; other times you need 8 detailed subgroups. A hierarchical approach lets "
            "you explore both without re-clustering."
        ),
        example=(
            "A biology department groups species:\n"
            "• Cut at top level: animals vs plants (2 clusters)\n"
            "• Cut mid-level: mammals, birds, reptiles, flowering plants (4 clusters)\n"
            "• Cut low level: lions, tigers, eagles, sparrows, roses, ferns (6 clusters)\n"
            "One clustering, many possible granularity levels."
        ),
        common_mistakes=[
            "Computing full hierarchy on very large datasets (>10K points) — extremely slow",
            "Not considering whether hierarchical structure is meaningful for the data",
            "Cutting the dendrogram at an arbitrary height without justification",
        ],
        practice_exercise=(
            "Run agglomerative clustering on a small dataset (first 200 rows). "
            "Use scipy to compute the full dendrogram. "
            "1. At what height would you cut to get 3 clusters?\n"
            "2. At what height for 5 clusters?\n"
            "3. Are the merges at similar heights?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the main disadvantage of hierarchical clustering compared to K-Means?",
                options=[
                    "Hierarchical clustering is less accurate",
                    "Hierarchical clustering scales poorly to large datasets (O(n³) time and memory)",
                    "Hierarchical clustering cannot handle numerical data",
                    "Hierarchical clustering always produces exactly 2 clusters",
                ],
                correct_index=1,
                explanation=(
                    "Hierarchical clustering requires computing and storing the distance matrix "
                    "(O(n²) memory) and the merge process is O(n³). For datasets with more "
                    "than ~10,000 points, this becomes impractical. K-Means is O(n×k×iterations)."
                ),
            ),
        ],
        takeaways=[
            "Hierarchical = tree of clusters (multi-scale structure)",
            "Agglomerative (bottom-up) is most common",
            "Can cut at any level for different k without re-clustering",
            "Doesn't scale to large datasets (>10K points)",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_17", title="Dendrograms",
        section="clustering", order=17, difficulty="intermediate",
        objectives=[
            "Read and interpret dendrograms",
            "Use dendrograms to choose k",
            "Understand linkage visually",
        ],
        concept=(
            "A dendrogram is a tree diagram showing the hierarchical merge process. "
            "Height = distance at which clusters merged. Long vertical lines between merges "
            "suggest natural cluster boundaries. Cutting the dendrogram at a specific height "
            "gives a specific number of clusters."
        ),
        why_matters=(
            "Dendrograms provide a visual way to choose k and understand the cluster structure. "
            "They reveal which clusters are closely related and where natural boundaries exist."
        ),
        example=(
            "Dendrogram of customer data: two main branches merge at height 15. "
            "The left branch splits into 3 sub-clusters at heights 4, 5, and 6. "
            "Cutting below height 15 gives 2 clusters; below height 7 gives 5 clusters."
        ),
        python_example=(
            "```python\n"
            "from scipy.cluster.hierarchy import dendrogram, linkage\n"
            "import matplotlib.pyplot as plt\n\n"
            "# Compute linkage matrix\n"
            "Z = linkage(X_scaled, method='ward')\n\n"
            "# Plot dendrogram (truncated for readability)\n"
            "plt.figure(figsize=(14, 5))\n"
            "dendrogram(\n"
            "    Z, truncate_mode='lastp', p=20,\n"
            "    leaf_rotation=90, leaf_font_size=10\n"
            ")\n"
            "plt.title('Dendrogram (Ward Linkage)')\n"
            "plt.xlabel('Cluster Size')\n"
            "plt.ylabel('Distance')\n"
            "plt.axhline(y=10, color='r', linestyle='--', label='Cut for k=3')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Not truncating the dendrogram (too many leaves to read)",
            "Choosing k without considering the merge distances",
            "Ignoring that different linkage methods produce very different dendrograms",
        ],
        practice_exercise=(
            "Compute a dendrogram on a small dataset. "
            "1. How many main branches are there?\n"
            "2. Where would you draw a horizontal line to get 3 clusters?\n"
            "3. What is the distance at the last merge?"
        ),
        quiz=[
            QuizQuestion(
                question="In a dendrogram, what does a long vertical line between two merges indicate?",
                options=[
                    "The clusters are very similar",
                    "There is a large distance between clusters — a natural cluster boundary",
                    "The algorithm made an error",
                    "The clusters should be merged",
                ],
                correct_index=1,
                explanation=(
                    "A long vertical line means a large increase in distance is needed to merge "
                    "the next pair of clusters. This indicates a natural boundary — the clusters "
                    "below the line are well-separated from each other."
                ),
            ),
        ],
        takeaways=[
            "Dendrogram shows the cluster hierarchy visually",
            "Long vertical lines = natural cluster boundaries",
            "Cut the tree at a specific height for the desired number of clusters",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_18", title="Scaling for Clustering",
        section="clustering", order=18, difficulty="intermediate",
        objectives=[
            "Understand why scaling is critical for clustering",
            "Apply StandardScaler before clustering",
            "Know which algorithms are affected by unscaled features",
        ],
        concept=(
            "Clustering algorithms use distance measures. Without scaling, features with "
            "larger numerical ranges dominate the distance calculation. For example, income "
            "(£50,000) overwhelms age (30) in Euclidean distance: a difference of £1,000 "
            "in income contributes more than a difference of 30 years in age."
        ),
        why_matters=(
            "Unscaled features produce misleading clusters where only the largest-scale "
            "features determine groupings. This defeats the purpose of using multiple features "
            "to define similarity."
        ),
        example=(
            "Customer data: income (10,000-100,000), age (18-80), frequency (1-50). "
            "Without scaling: clustering is determined almost entirely by income. "
            "With StandardScaler: all features contribute equally to distance."
        ),
        python_example=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans\n\n"
            "# WRONG: clustering on raw features\n"
            "kmeans_bad = KMeans(n_clusters=3, random_state=42)\n"
            "clusters_bad = kmeans_bad.fit_predict(X)  # income dominates!\n\n"
            "# CORRECT: scale first\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "kmeans_good = KMeans(n_clusters=3, random_state=42)\n"
            "clusters_good = kmeans_good.fit_predict(X_scaled)\n"
            "```"
        ),
        common_mistakes=[
            "Not scaling before K-Means (distance-based, completely dominated by large-range features)",
            "Not scaling before DBSCAN (also distance-based)",
            "Scaling after clustering (too late — must scale before fitting)",
        ],
        practice_exercise=(
            "Run K-Means on a dataset with and without scaling. "
            "1. Are the cluster assignments different?\n"
            "2. Which result makes more sense?\n"
            "3. Plot two features to see the difference visually."
        ),
        quiz=[
            QuizQuestion(
                question="Which clustering algorithms REQUIRE feature scaling?",
                options=[
                    "Only K-Means",
                    "Only DBSCAN",
                    "All distance-based algorithms: K-Means, DBSCAN, Agglomerative",
                    "No clustering algorithm needs scaling",
                ],
                correct_index=2,
                explanation=(
                    "Any algorithm that uses distance measures is affected by feature scales. "
                    "K-Means, DBSCAN, and Agglomerative Clustering all compute distances. "
                    "Tree-based methods (like in Random Forest) don't use distance, but they "
                    "aren't traditional clustering algorithms."
                ),
            ),
        ],
        takeaways=[
            "Always scale before distance-based clustering",
            "StandardScaler is the default choice",
            "Features with larger ranges completely dominate unscaled distance calculations",
            "Exception: DBSCAN with a precomputed distance matrix",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_19", title="PCA for Visualization",
        section="clustering", order=19, difficulty="intermediate",
        objectives=[
            "Use PCA to reduce dimensions for visualisation",
            "Create 2D scatter plots of clusters",
            "Interpret PCA plots and explained variance",
        ],
        concept=(
            "PCA (Principal Component Analysis) reduces high-dimensional data to 2 or 3 "
            "dimensions for visualisation. It projects data onto the directions of maximum "
            "variance. The 2D plot shows the most important structure in the data."
        ),
        why_matters=(
            "We can't visualise data with more than 3 features. PCA gives us a 2D 'shadow' "
            "that preserves as much of the original structure as possible. This lets us "
            "visually assess whether clusters make sense."
        ),
        example=(
            "Iris has 4 features. PCA reduces to 2D, showing three clearly separated groups. "
            "This confirms that the K-Means clusters match the natural structure. "
            "The first two principal components capture 95% of the variance."
        ),
        python_example=(
            "```python\n"
            "from sklearn.decomposition import PCA\n"
            "import matplotlib.pyplot as plt\n\n"
            "pca = PCA(n_components=2)\n"
            "X_pca = pca.fit_transform(X_scaled)\n\n"
            "plt.figure(figsize=(10, 7))\n"
            "scatter = plt.scatter(\n"
            "    X_pca[:, 0], X_pca[:, 1],\n"
            "    c=clusters, cmap='viridis', alpha=0.6, s=30\n"
            ")\n"
            "plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')\n"
            "plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')\n"
            "plt.title(f'Clusters in PCA Space ({pca.explained_variance_ratio_.sum():.1%} total variance)')\n"
            "plt.colorbar(scatter, label='Cluster')\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "If clusters are clearly separated in PCA space, the clustering is likely meaningful. "
            "If clusters overlap heavily, the chosen features may not distinguish groups well. "
            "Check explained variance: if PC1+PC2 captures <50% of variance, the 2D plot is "
            "a poor representation."
        ),
        common_mistakes=[
            "Not checking how much variance PCA preserves",
            "Interpreting PCA axes as original features (they are linear combinations)",
            "Assuming good PCA separation means good clustering (and vice versa)",
        ],
        practice_exercise=(
            "Run K-Means on a dataset with 5+ features. Use PCA to plot clusters in 2D. "
            "1. How much variance do PC1 and PC2 capture?\n"
            "2. Are clusters well-separated in the PCA plot?\n"
            "3. Does the visual match the numerical metrics (silhouette)?"
        ),
        quiz=[
            QuizQuestion(
                question="If PCA only captures 40% of variance in 2D, what does this mean for cluster visualisation?",
                options=[
                    "The clusters are invalid",
                    "The 2D plot shows less than half the information — clusters may look overlapping in 2D but be well-separated in the full feature space",
                    "PCA is broken",
                    "You should use t-SNE instead",
                ],
                correct_index=1,
                explanation=(
                    "Low explained variance means the 2D projection loses a lot of information. "
                    "Clusters that appear overlapping in 2D may actually be well-separated along "
                    "the dropped dimensions. The visualisation is unreliable, not the clustering."
                ),
            ),
        ],
        takeaways=[
            "PCA reduces to 2D/3D for visualisation",
            "Always check explained variance ratio",
            "If < 50% variance captured, the 2D plot is unreliable",
            "PCA is for visualisation, not for the clustering itself",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_20", title="Cluster Interpretation",
        section="clustering", order=20, difficulty="intermediate",
        objectives=[
            "Profile clusters by computing statistics per cluster",
            "Give clusters meaningful names",
            "Connect cluster findings to business insights",
        ],
        concept=(
            "After clustering, the essential next step is profiling: computing mean, median, "
            "and distribution of features per cluster to understand what each group represents. "
            "Give clusters meaningful names based on their characteristics."
        ),
        why_matters=(
            "Clusters without interpretation are just numbers. Meaningful names and profiles "
            "turn mathematical groupings into actionable insights. Business stakeholders need "
            "to understand what each cluster represents."
        ),
        example=(
            "Customer segmentation results:\n"
            "• Cluster 0 (n=2,340): avg_age=25, avg_spend=£500, freq=2/year → 'Young Budget Shoppers'\n"
            "• Cluster 1 (n=890): avg_age=45, avg_spend=£5,000, freq=30/year → 'Loyal Premium Customers'\n"
            "• Cluster 2 (n=1,200): avg_age=35, avg_spend=£2,000, freq=8/year → 'Regular Mid-Range Shoppers'\n"
            "Marketing can now target each segment with tailored campaigns."
        ),
        python_example=(
            "```python\n"
            "import pandas as pd\n\n"
            "# Create clustered dataframe\n"
            "df_clustered = pd.DataFrame(X, columns=feature_names)\n"
            "df_clustered['Cluster'] = clusters\n\n"
            "# Profile: mean per cluster\n"
            "profiles = df_clustered.groupby('Cluster').mean()\n"
            "print('Cluster Profiles:')\n"
            "print(profiles.round(2))\n\n"
            "# Cluster sizes\n"
            "print('\\nCluster Sizes:')\n"
            "print(df_clustered['Cluster'].value_counts().sort_index())\n"
            "```"
        ),
        common_mistakes=[
            "Not profiling clusters after running the algorithm",
            "Giving clusters arbitrary numbers instead of meaningful names",
            "Not validating cluster profiles with domain experts",
        ],
        practice_exercise=(
            "After running K-Means, profile each cluster:\n"
            "1. What are the average values of each feature per cluster?\n"
            "2. Give each cluster a meaningful name.\n"
            "3. Which cluster would you target for a specific business action?"
        ),
        quiz=[
            QuizQuestion(
                question="After clustering customers, you find Cluster 3 has very high average values for all features. What should you check?",
                options=[
                    "Nothing — high values are always good",
                    "Whether Cluster 3 is actually well-defined or if it's a catch-all for outliers",
                    "Whether the scaling was applied correctly",
                    "Whether k was too high",
                ],
                correct_index=1,
                explanation=(
                    "A cluster where all features are high might be a catch-all for outliers "
                    "rather than a meaningful segment. Check the cluster size, the silhouette "
                    "score for those points, and whether the profile makes business sense."
                ),
            ),
        ],
        takeaways=[
            "Profile clusters by computing statistics per cluster",
            "Give clusters meaningful, descriptive names",
            "Validate profiles with domain experts",
            "Turn cluster findings into actionable business insights",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_21", title="Limitations of Clustering",
        section="clustering", order=21, difficulty="intermediate",
        objectives=[
            "Understand clustering limitations",
            "Know when clustering fails or misleads",
            "Avoid common clustering pitfalls",
        ],
        concept=(
            "Clustering has important limitations:\n"
            "1. Results depend heavily on parameters (k, eps, min_samples)\n"
            "2. Different algorithms give different results on the same data\n"
            "3. No ground truth to validate against (unsupervised!)\n"
            "4. Assumes 'similarity' = distance (may not capture true relationships)\n"
            "5. Struggles with mixed data types (numerical + categorical)\n"
            "6. Sensitive to feature scaling and feature selection"
        ),
        why_matters=(
            "Understanding limitations prevents overconfidence in results. Clustering is a "
            "tool for exploration, not a definitive analysis. Results must always be validated "
            "with domain knowledge."
        ),
        example=(
            "A marketing team segments customers with K-Means (k=5) and finds 5 'clear' segments. "
            "But when they try DBSCAN, they get 3 clusters + noise. When they try hierarchical "
            "clustering, the dendrogram suggests k=4. The 'clear' segmentation was actually "
            "an artefact of the chosen algorithm and parameters."
        ),
        common_mistakes=[
            "Treating clustering results as ground truth",
            "Only using one algorithm — always try multiple methods",
            "Ignoring that results depend heavily on parameters",
        ],
        practice_exercise=(
            "Run K-Means, DBSCAN, and Agglomerative on the same dataset. "
            "1. Do they produce similar clusters?\n"
            "2. What are the differences?\n"
            "3. Which result seems most interpretable?"
        ),
        quiz=[
            QuizQuestion(
                question="Why can't we use accuracy to evaluate clustering results?",
                options=[
                    "Accuracy is too slow for clustering",
                    "Clustering has no ground truth labels — accuracy requires knowing the correct answers",
                    "Accuracy only works for regression",
                    "Accuracy doesn't work with Python",
                ],
                correct_index=1,
                explanation=(
                    "Accuracy measures how many predictions match ground truth labels. "
                    "In unsupervised learning, there are no labels to compare against. "
                    "We evaluate clustering using internal metrics (silhouette) or domain validation."
                ),
            ),
        ],
        takeaways=[
            "Clustering results are algorithm-dependent — different methods give different results",
            "No 'correct' answer without ground truth",
            "Always try multiple algorithms and validate with domain knowledge",
            "Clustering is exploratory, not definitive",
        ],
        lab_module="clustering",
    ),
    Topic(
        id="clus_22", title="Clustering Case Study",
        section="clustering", order=22, difficulty="advanced",
        objectives=[
            "Apply the complete clustering workflow end-to-end",
            "Compare algorithms on the same dataset",
            "Document findings and present actionable insights",
        ],
        concept=(
            "Complete clustering workflow:\n"
            "1. Select relevant features\n"
            "2. Handle missing values\n"
            "3. Scale features (StandardScaler)\n"
            "4. Determine optimal k (elbow + silhouette)\n"
            "5. Run K-Means, DBSCAN, Agglomerative\n"
            "6. Compare results visually (PCA) and numerically (silhouette)\n"
            "7. Profile and name clusters\n"
            "8. Validate with domain knowledge"
        ),
        why_matters=(
            "Real clustering projects require the full workflow. Understanding how all pieces "
            "fit together — from data preparation to cluster interpretation — is essential "
            "for independent work."
        ),
        example=(
            "Customer segmentation case study:\n"
            "1. Features: annual_spend, visit_frequency, recency\n"
            "2. Scaled with StandardScaler\n"
            "3. Elbow suggests k=3, silhouette confirms k=3 (score=0.58)\n"
            "4. K-Means: 3 clusters. DBSCAN: 2 clusters + outliers. Hierarchical: also 3.\n"
            "5. Consensus: 3 main segments\n"
            "6. Profiles: Budget (n=4,500), Regular (n=2,800), Premium (n=700)\n"
            "7. Action: premium segment gets VIP offers; budget gets discount campaigns"
        ),
        common_mistakes=[
            "Not validating clusters with business context or domain experts",
            "Only using one algorithm — multiple methods increase confidence",
            "Not presenting actionable insights — clusters are useless if nobody acts on them",
        ],
        practice_exercise=(
            "Complete a clustering case study:\n"
            "1. Select 3-5 features from a dataset\n"
            "2. Scale and determine optimal k\n"
            "3. Run K-Means and one other algorithm\n"
            "4. Profile and name the clusters\n"
            "5. Write a one-paragraph summary of findings"
        ),
        quiz=[
            QuizQuestion(
                question="In a case study, K-Means gives k=3 and DBSCAN gives k=2 + noise. What is the best approach?",
                options=[
                    "Always trust K-Means over DBSCAN",
                    "Use the result from the more popular algorithm",
                    "Investigate both: check cluster profiles, silhouette scores, and domain sense",
                    "Run K-Means 100 times and take the majority vote",
                ],
                correct_index=2,
                explanation=(
                    "Different algorithms reveal different structure. K-Means forces k clusters; "
                    "DBSCAN discovers the natural number. Investigate both: profile the clusters, "
                    "check silhouette scores, and consult domain knowledge to determine which "
                    "result is more meaningful."
                ),
            ),
        ],
        takeaways=[
            "Follow the complete workflow: select → scale → choose k → cluster → profile → validate",
            "Always try multiple algorithms for robustness",
            "Present actionable insights, not just numbers",
            "Validate findings with domain experts",
        ],
        lab_module="clustering",
    ),
]
