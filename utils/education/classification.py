"""
Classification — learning topics for the Classification module.
"""
from utils.education.base import T

TOPICS = {
    # ── 1 ──────────────────────────────────────────────────────────
    "what_is_classification": T(
        title="What Is Classification",
        module="classification",
        what=(
            "Classification is a supervised learning task where the goal "
            "is to predict a discrete category (label) for each observation. "
            "Examples: spam/not spam, disease/no disease, customer churn."
        ),
        why=(
            "Classification is one of the most common ML tasks. Email "
            "filtering, medical diagnosis, fraud detection, and image "
            "recognition are all classification problems."
        ),
        when=(
            "Use classification when your target variable is categorical "
            "(finite number of classes). If the target is continuous "
            "(e.g., price), use regression instead."
        ),
        example=(
            "Titanic survival prediction: given features (Age, Fare, "
            "Sex, Pclass), predict whether a passenger survived (0 or 1). "
            "This is binary classification with two classes."
        ),
        mistakes=[
            "Using regression for classification problems.",
            "Ignoring class imbalance (90% 'No' vs 10% 'Yes').",
            "Not evaluating beyond accuracy.",
        ],
        interpretation=(
            "A good classifier assigns correct labels to most test "
            "samples. But 'good' depends on context: 95% accuracy "
            "may be great for spam detection but terrible for cancer "
            "screening."
        ),
        think_about_it=(
            "A model predicts 'No fraud' for every transaction and "
            "achieves 99.9% accuracy. Is this a good classifier?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)\n"
            "```"
        ),
        keywords=["classification", "supervised", "categorical", "target", "label"],
    ),

    # ── 2 ──────────────────────────────────────────────────────────
    "classification_vs_regression": T(
        title="Classification vs Regression",
        module="classification",
        what=(
            "Classification predicts categories (discrete labels). "
            "Regression predicts continuous values (numbers). The choice "
            "depends on your target variable."
        ),
        why=(
            "Using the wrong task type leads to meaningless results. "
            "Classification on continuous data won't work; regression "
            "on categories won't either."
        ),
        when=(
            "Check your target: if it has finite categories → "
            "classification. If it's continuous numbers → regression. "
            "Some cases are ambiguous (ordinal targets with few values)."
        ),
        example=(
            "Classification: Will it rain tomorrow? (Yes/No)\n"
            "Regression: What will tomorrow's temperature be? (23.5°C)\n"
            "Ambiguous: Rating 1-5 stars — could be either."
        ),
        mistakes=[
            "Using regression when the target is truly categorical.",
            "Using classification when the target is continuous.",
            "Treating an ordinal target (1-5) as purely numerical.",
        ],
        interpretation=(
            "When in doubt, ask: 'Does the exact number matter, or "
            "just the category?' If you care about the number (price), "
            "use regression. If you care about the group (spam/not), "
            "use classification."
        ),
        think_about_it=(
            "Predicting house prices: $200K, $350K, $1.2M. This is "
            "regression. But what if you group them into 'cheap', "
            "'medium', 'expensive'? Now it's classification."
        ),
        code_link=(
            "```python\n"
            "# Classification\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression()\n"
            "\n"
            "# Regression\n"
            "from sklearn.linear_model import LinearRegression\n"
            "model = LinearRegression()\n"
            "```"
        ),
        keywords=["classification", "regression", "continuous", "discrete", "target"],
    ),

    # ── 3 ──────────────────────────────────────────────────────────
    "binary_classification": T(
        title="Binary Classification",
        module="classification",
        what=(
            "Binary classification has exactly two classes: positive/negative, "
            "yes/no, 0/1. It's the simplest form of classification."
        ),
        why=(
            "Many real-world problems are binary: spam detection, "
            "disease diagnosis, fraud detection, customer churn. "
            "Understanding binary metrics is foundational."
        ),
        when=(
            "When the target has exactly two unique values. Binary "
            "classification has specific metrics (precision, recall, "
            "ROC-AUC) that don't apply to multiclass."
        ),
        example=(
            "Titanic survival: Survived (1) vs Did Not Survive (0). "
            "The model learns patterns in features that distinguish "
            "the two outcomes."
        ),
        mistakes=[
            "Not using stratify when splitting — class proportions may differ.",
            "Ignoring which class is 'positive' — it affects precision/recall.",
            "Reporting only accuracy with imbalanced classes.",
        ],
        interpretation=(
            "In binary classification, one class is typically the "
            "'positive' class (the event of interest). Precision and "
            "recall are always reported for the positive class."
        ),
        think_about_it=(
            "In cancer detection, which is the 'positive' class: "
            "cancer or no cancer? How does this choice affect "
            "precision and recall?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "model = LogisticRegression()\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)      # class labels (0 or 1)\n"
            "y_prob = model.predict_proba(X_test)[:, 1]  # probabilities\n"
            "```"
        ),
        keywords=["binary", "two classes", "positive", "negative", "0/1"],
    ),

    # ── 4 ──────────────────────────────────────────────────────────
    "multiclass_classification": T(
        title="Multiclass Classification",
        module="classification",
        what=(
            "Multiclass classification has more than two classes: "
            "species of iris (setosa/versicolor/virginica), digit "
            "recognition (0-9), or disease type."
        ),
        why=(
            "Many problems have more than two outcomes. Multiclass "
            "extends binary classification with different averaging "
            "strategies and evaluation approaches."
        ),
        when=(
            "When the target has 3+ unique values. Most sklearn "
            "classifiers handle multiclass natively or via strategies "
            "like One-vs-Rest."
        ),
        example=(
            "Iris dataset: 3 classes (setosa, versicolor, virginica). "
            "The model learns to distinguish all three simultaneously."
        ),
        mistakes=[
            "Using binary metrics without specifying average='weighted'.",
            "Not checking per-class performance — overall accuracy can hide failures.",
            "Assuming all classes have equal importance.",
        ],
        interpretation=(
            "For multiclass, use weighted average for precision/recall/F1 "
            "to account for class imbalance. Per-class metrics reveal "
            "which classes the model struggles with."
        ),
        think_about_it=(
            "A model achieves 95% accuracy on a 10-class problem. "
            "But one class has only 60% recall. Why might overall "
            "accuracy still be high?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import classification_report\n"
            "\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Use 'weighted' for imbalanced multiclass\n"
            "print(classification_report(y_test, y_pred, average='weighted'))\n"
            "```"
        ),
        keywords=["multiclass", "multiple classes", "multi-class", "n-classes"],
    ),

    # ── 5 ──────────────────────────────────────────────────────────
    "logistic_regression": T(
        title="Logistic Regression",
        module="classification",
        what=(
            "Despite its name, Logistic Regression is a classification "
            "algorithm. It models the probability of the positive class "
            "using the sigmoid function: P(y=1) = 1 / (1 + e^(-z))."
        ),
        why=(
            "It's the simplest classification algorithm, fast, interpretable, "
            "and provides probabilities. It's the standard baseline for "
            "binary classification."
        ),
        when=(
            "Use as a first baseline. Works well when the decision "
            "boundary is approximately linear. Good for understanding "
            "feature importance via coefficients."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "model = LogisticRegression(max_iter=1000)\n"
            "model.fit(X_train, y_train)\n"
            "print(f'Accuracy: {model.score(X_test, y_test):.4f}')\n"
            "\n"
            "# Feature importance via coefficients\n"
            "for name, coef in zip(feature_names, model.coef_[0]):\n"
            "    print(f'{name}: {coef:.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not scaling features — logistic regression is sensitive to scale.",
            "Using without increasing max_iter for convergence warnings.",
            "Expecting it to capture non-linear relationships.",
        ],
        interpretation=(
            "Positive coefficient → feature increases probability of "
            "positive class. Negative → decreases. Magnitude shows "
            "strength. Always scale features first."
        ),
        think_about_it=(
            "Logistic Regression has coef_ = [2.5, -1.2, 0.1]. "
            "What does this tell you about the most important feature?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', LogisticRegression(max_iter=1000))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'Score: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["logistic", "regression", "sigmoid", "linear", "baseline"],
    ),

    # ── 6 ──────────────────────────────────────────────────────────
    "knn": T(
        title="K-Nearest Neighbors (KNN)",
        module="classification",
        what=(
            "KNN classifies a data point by majority vote of its K "
            "nearest neighbors. It's a lazy learner — no training, "
            "all computation happens at prediction time."
        ),
        why=(
            "KNN is simple, intuitive, and requires no training. "
            "It works well for small datasets with clear class "
            "separation."
        ),
        when=(
            "Use for small datasets (<10K samples). Always scale features "
            "first (distance-based). Choose K: small K = noisy, "
            "large K = smooth."
        ),
        example=(
            "```python\n"
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "X_test_s = scaler.transform(X_test)\n"
            "\n"
            "knn = KNeighborsClassifier(n_neighbors=5)\n"
            "knn.fit(X_train_s, y_train)\n"
            "print(f'Accuracy: {knn.score(X_test_s, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not scaling features — distance is dominated by large-magnitude features.",
            "Choosing K=1 — too sensitive to noise.",
            "Using on large datasets — prediction is slow (O(n) per point).",
        ],
        interpretation=(
            "K=1: very flexible but noisy. K=n: always predicts the "
            "majority class. Start with K=5 and try odd numbers to "
            "avoid ties."
        ),
        think_about_it=(
            "With K=1, your training accuracy is 100% but test accuracy "
            "is 65%. What's happening, and how would you fix it?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('knn', KNeighborsClassifier(n_neighbors=5))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'Accuracy: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["knn", "k-nearest", "neighbors", "lazy", "distance"],
    ),

    # ── 7 ──────────────────────────────────────────────────────────
    "decision_trees": T(
        title="Decision Trees",
        module="classification",
        what=(
            "A decision tree splits data into branches using feature "
            "thresholds. Each leaf node predicts a class. It's like "
            "a flowchart of if-else rules."
        ),
        why=(
            "Decision trees are intuitive, handle mixed data types, "
            "and don't require scaling. They're the building block "
            "for Random Forest and Gradient Boosting."
        ),
        when=(
            "Use for explainable models or as a baseline. Tune "
            "max_depth to control overfitting. Deep trees overfit, "
            "shallow trees underfit."
        ),
        example=(
            "```python\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "\n"
            "tree = DecisionTreeClassifier(max_depth=4, random_state=42)\n"
            "tree.fit(X_train, y_train)\n"
            "print(f'Train: {tree.score(X_train, y_train):.4f}')\n"
            "print(f'Test:  {tree.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not limiting max_depth — unpruned trees overfit completely.",
            "Using decision trees alone for production — they're unstable.",
            "Not checking feature importances after training.",
        ],
        interpretation=(
            "Train accuracy = 1.0 and test = 0.80 means overfitting. "
            "Reduce max_depth or use pruning. The tree structure itself "
            "shows which features and thresholds matter most."
        ),
        think_about_it=(
            "A decision tree achieves 100% training accuracy. Is this "
            "good? What does it tell you about the model?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "\n"
            "tree = DecisionTreeClassifier(\n"
            "    max_depth=4,\n"
            "    min_samples_split=10,\n"
            "    random_state=42\n"
            ")\n"
            "tree.fit(X_train, y_train)\n"
            "print(f'Feature importances: {tree.feature_importances_}')\n"
            "```"
        ),
        keywords=["decision", "tree", "split", "depth", "pruning", "explainable"],
    ),

    # ── 8 ──────────────────────────────────────────────────────────
    "random_forest": T(
        title="Random Forest",
        module="classification",
        what=(
            "Random Forest trains many decision trees on random subsets "
            "of data and features, then aggregates their predictions "
            "by majority vote. It reduces overfitting of single trees."
        ),
        why=(
            "Random Forest is one of the most reliable general-purpose "
            "classifiers. It handles non-linearity, feature interactions, "
            "and missing values well with minimal tuning."
        ),
        when=(
            "Use as a strong default classifier. Works well out-of-the-box "
            "for most tabular data. Tune n_estimators and max_depth "
            "for better performance."
        ),
        example=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "rf = RandomForestClassifier(\n"
            "    n_estimators=100,\n"
            "    max_depth=10,\n"
            "    random_state=42\n"
            ")\n"
            "rf.fit(X_train, y_train)\n"
            "print(f'Accuracy: {rf.score(X_test, y_test):.4f}')\n"
            "\n"
            "# Feature importance\n"
            "import pandas as pd\n"
            "imp = pd.Series(rf.feature_importances_, index=feature_names)\n"
            "imp.sort_values(ascending=False).plot(kind='bar')\n"
            "```"
        ),
        mistakes=[
            "Using too few trees (<50) — more trees improve stability.",
            "Not tuning max_depth — default 'None' lets trees overfit.",
            "Ignoring training time — 1000 trees are slower to predict.",
        ],
        interpretation=(
            "Feature importance shows which features the forest relies "
            "on most. High importance means the feature helps split "
            "data correctly across many trees."
        ),
        think_about_it=(
            "Random Forest has 95% train accuracy and 93% test accuracy. "
            "Decision Tree has 100% train and 82% test. Why is RF "
            "better?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "rf = RandomForestClassifier(\n"
            "    n_estimators=200,\n"
            "    max_depth=10,\n"
            "    min_samples_leaf=5,\n"
            "    random_state=42,\n"
            "    n_jobs=-1\n"
            ")\n"
            "rf.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["random", "forest", "ensemble", "trees", "bagging"],
    ),

    # ── 9 ──────────────────────────────────────────────────────────
    "naive_bayes": T(
        title="Naive Bayes",
        module="classification",
        what=(
            "Naive Bayes applies Bayes' theorem with the 'naive' "
            "assumption that features are independent given the class. "
            "Despite this simplification, it works surprisingly well "
            "for text classification."
        ),
        why=(
            "Naive Bayes is extremely fast, works well with small data, "
            "and excels at text classification (spam, sentiment analysis). "
            "It's a good baseline for NLP tasks."
        ),
        when=(
            "Use for text classification and when features are roughly "
            "independent. MultinomialNB for text counts, GaussianNB "
            "for continuous features, BernoulliNB for binary features."
        ),
        example=(
            "```python\n"
            "from sklearn.naive_bayes import GaussianNB\n"
            "\n"
            "nb = GaussianNB()\n"
            "nb.fit(X_train, y_train)\n"
            "print(f'Accuracy: {nb.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Assuming features are truly independent — they rarely are.",
            "Using GaussianNB on count data (use MultinomialNB).",
            "Expecting it to outperform ensemble methods on tabular data.",
        ],
        interpretation=(
            "Despite the 'naive' independence assumption, Naive Bayes "
            "often performs well because it only needs the correct "
            "ranking of class probabilities, not exact probabilities."
        ),
        think_about_it=(
            "In email spam detection, the words 'free' and 'money' "
            "appear together. Naive Bayes assumes they're independent. "
            "Why does it still work?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.naive_bayes import GaussianNB, MultinomialNB\n"
            "\n"
            "# For continuous features\n"
            "gnb = GaussianNB()\n"
            "gnb.fit(X_train, y_train)\n"
            "\n"
            "# For text/count data\n"
            "mnb = MultinomialNB()\n"
            "mnb.fit(X_train_counts, y_train)\n"
            "```"
        ),
        keywords=["naive", "bayes", "probability", "text", "fast"],
    ),

    # ── 10 ─────────────────────────────────────────────────────────
    "svm": T(
        title="Support Vector Machines",
        module="classification",
        what=(
            "SVM finds the optimal hyperplane that maximises the margin "
            "between classes. With kernel trick, it can learn non-linear "
            "decision boundaries."
        ),
        why=(
            "SVMs work well in high-dimensional spaces and with clear "
            "margin of separation. The kernel trick allows non-linear "
            "classification without explicit feature transformation."
        ),
        when=(
            "Use for small-to-medium datasets with clear class separation. "
            "Scale features first. RBF kernel is the default and most "
            "versatile."
        ),
        example=(
            "```python\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('svm', SVC(kernel='rbf', C=1.0))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'Accuracy: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not scaling features — SVM is very sensitive to feature scale.",
            "Using SVM on large datasets (>10K rows) — training is slow.",
            "Not tuning C and gamma — defaults may not be optimal.",
        ],
        interpretation=(
            "C controls margin width: large C = narrow margin (fewer "
            "errors on train), small C = wide margin (more generalisation). "
            "Gamma controls the influence of each training point."
        ),
        think_about_it=(
            "SVM with default parameters gives 70% accuracy on scaled "
            "data. Random Forest gives 90% without scaling. Which is "
            "the better model for this problem?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["svm", "support vector", "kernel", "margin", "hyperplane"],
    ),

    # ── 11 ─────────────────────────────────────────────────────────
    "gradient_boosting": T(
        title="Gradient Boosting",
        module="classification",
        what=(
            "Gradient Boosting builds trees sequentially, where each "
            "new tree corrects errors of the previous ones. It's an "
            "ensemble method that often achieves top accuracy."
        ),
        why=(
            "Gradient Boosting is among the highest-performing algorithms "
            "for tabular data. It wins Kaggle competitions and is used "
            "in production systems worldwide."
        ),
        when=(
            "Use when you need maximum accuracy and can afford longer "
            "training time. Tune n_estimators, learning_rate, and "
            "max_depth carefully."
        ),
        example=(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "\n"
            "gb = GradientBoostingClassifier(\n"
            "    n_estimators=100,\n"
            "    learning_rate=0.1,\n"
            "    max_depth=3,\n"
            "    random_state=42\n"
            ")\n"
            "gb.fit(X_train, y_train)\n"
            "print(f'Accuracy: {gb.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using too many trees with high learning_rate — overfits.",
            "Not using early stopping for large n_estimators.",
            "Comparing training time with simpler models unfairly.",
        ],
        interpretation=(
            "learning_rate shrinks each tree's contribution. Lower "
            "rates need more trees but generalise better. "
            "n_estimators x learning_rate controls total model complexity."
        ),
        think_about_it=(
            "Gradient Boosting with learning_rate=0.3 and 100 trees "
            "gives the same performance as learning_rate=0.03 and "
            "1000 trees. Which configuration is more robust?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "\n"
            "gb = GradientBoostingClassifier(\n"
            "    n_estimators=200,\n"
            "    learning_rate=0.05,\n"
            "    max_depth=4,\n"
            "    subsample=0.8,\n"
            "    random_state=42\n"
            ")\n"
            "gb.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["gradient", "boosting", "ensemble", "sequential", "accurate"],
    ),

    # ── 12 ─────────────────────────────────────────────────────────
    "training_a_classifier": T(
        title="Training a Classifier",
        module="classification",
        what=(
            "Training a classifier means fitting the model to labelled "
            "training data. The model learns the relationship between "
            "features (X) and labels (y)."
        ),
        why=(
            "Training is the core step of supervised learning. "
            "Understanding what happens during training helps you "
            "diagnose problems like overfitting and underfitting."
        ),
        when=(
            "After preprocessing and splitting. Use model.fit(X_train, y_train). "
            "The model learns patterns in training data that it will "
            "apply to new data."
        ),
        example=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "model = RandomForestClassifier(n_estimators=100)\n"
            "model.fit(X_train, y_train)  # training\n"
            "\n"
            "# Check: does the model memorise or learn?\n"
            "print(f'Train: {model.score(X_train, y_train):.4f}')\n"
            "print(f'Test:  {model.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Training on the entire dataset including test data.",
            "Not checking train vs test scores after training.",
            "Using default hyperparameters without any tuning.",
        ],
        interpretation=(
            "After training, the model has internal parameters "
            "(e.g., tree splits, coefficients) that encode the "
            "learned patterns. These are stored in the model object."
        ),
        think_about_it=(
            "You train a model and it achieves 99% training accuracy "
            "but only 70% test accuracy. What is this called, and "
            "what are three ways to fix it?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
            "\n"
            "# Single split\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Cross-validation (more reliable)\n"
            "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "print(f'CV: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["training", "fit", "learn", "model", "fitting"],
    ),

    # ── 13 ─────────────────────────────────────────────────────────
    "making_predictions": T(
        title="Making Predictions",
        module="classification",
        what=(
            "After training, use model.predict() to get class labels "
            "and model.predict_proba() to get class probabilities "
            "for new data."
        ),
        why=(
            "Predictions are the output of your model. Understanding "
            "the difference between class labels and probabilities "
            "is crucial for threshold tuning and evaluation."
        ),
        when=(
            "After training on the test set or new data. Always "
            "preprocess new data using the same pipeline."
        ),
        example=(
            "```python\n"
            "y_pred = model.predict(X_test)         # class labels\n"
            "y_prob = model.predict_proba(X_test)   # probabilities\n"
            "\n"
            "print(y_pred[:5])      # [1, 0, 1, 0, 1]\n"
            "print(y_prob[:5])      # [[0.2, 0.8], [0.9, 0.1], ...]\n"
            "```"
        ),
        mistakes=[
            "Using predict() when you need probabilities for ROC/AUC.",
            "Applying different preprocessing to prediction data.",
            "Forgetting that predict_proba() returns probabilities for ALL classes.",
        ],
        interpretation=(
            "predict() returns the most likely class. predict_proba() "
            "returns [P(class=0), P(class=1)]. For binary classification, "
            "y_prob[:, 1] gives the positive class probability."
        ),
        think_about_it=(
            "Model gives predict_proba() = [0.51, 0.49]. The prediction "
            "is class 0, but with very low confidence. Should you trust "
            "this prediction?"
        ),
        code_link=(
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "y_prob = model.predict_proba(X_test)[:, 1]  # positive class prob\n"
            "\n"
            "# Custom threshold\n"
            "threshold = 0.3\n"
            "y_pred_custom = (y_prob >= threshold).astype(int)"
        ),
        keywords=["predict", "prediction", "probability", "predict_proba", "output"],
    ),

    # ── 14 ─────────────────────────────────────────────────────────
    "prediction_probabilities": T(
        title="Prediction Probabilities",
        module="classification",
        what=(
            "Predict_proba() returns the model's estimated probability "
            "for each class. Unlike predict(), it gives confidence levels, "
            "not just the final label."
        ),
        why=(
            "Probabilities enable threshold tuning, ROC curves, and "
            "better decision-making. A model saying 99% vs 51% confident "
            "makes a big difference in practice."
        ),
        when=(
            "Use predict_proba() when you need to: tune classification "
            "thresholds, compute ROC-AUC, rank predictions by confidence, "
            "or make cost-sensitive decisions."
        ),
        example=(
            "```python\n"
            "probs = model.predict_proba(X_test)\n"
            "# [[0.2, 0.8], [0.9, 0.1], ...]\n"
            "# Each row sums to 1.0\n"
            "\n"
            "positive_probs = probs[:, 1]  # probability of class 1\n"
            "```"
        ),
        mistakes=[
            "Using predict_proba() with models that don't support it.",
            "Not checking that probabilities sum to 1.",
            "Treating probabilities as exact — they're estimates.",
        ],
        interpretation=(
            "Calibrated probabilities: well-calibrated means P(=0.7) "
            "actually happens 70% of the time. Random Forest probabilities "
            "are often uncalibrated. Use CalibratedClassifierCV if needed."
        ),
        think_about_it=(
            "A medical model gives 60% probability of disease. The "
            "doctor treats the patient. Was this the right decision? "
            "What additional information matters?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "\n"
            "# Rank by confidence\n"
            "ranking = np.argsort(y_prob)[::-1]\n"
            "\n"
            "# Custom threshold\n"
            "y_pred = np.where(y_prob > 0.3, 1, 0)  # lower threshold\n"
            "```"
        ),
        keywords=["probability", "confidence", "calibrated", "threshold", "proba"],
    ),

    # ── 15 ─────────────────────────────────────────────────────────
    "decision_boundaries": T(
        title="Decision Boundaries",
        module="classification",
        what=(
            "A decision boundary is the surface that separates classes "
            "in feature space. Different algorithms create different "
            "shapes: linear (Logistic Regression), axis-aligned "
            "(Decision Trees), curved (SVM with RBF kernel)."
        ),
        why=(
            "Visualising decision boundaries helps understand how "
            "an algorithm works and why it succeeds or fails on a "
            "particular dataset."
        ),
        when=(
            "For 2D or 3D data, or after PCA reduction. Visualise "
            "decision boundaries to compare algorithms."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.decomposition import PCA\n"
            "\n"
            "# Reduce to 2D for visualisation\n"
            "pca = PCA(n_components=2)\n"
            "X_2d = pca.fit_transform(X_train)\n"
            "\n"
            "model.fit(X_2d, y_train)\n"
            "# Plot boundary using contour plot\n"
            "```"
        ),
        mistakes=[
            "Only looking at accuracy — boundary shape reveals WHY a model works.",
            "Forgetting that 2D boundaries are projections of higher-dimensional boundaries.",
        ],
        interpretation=(
            "Linear boundaries: Logistic Regression, Linear SVM. "
            "Non-linear boundaries: RBF SVM, Random Forest, KNN. "
            "Axis-aligned splits: Decision Trees."
        ),
        think_about_it=(
            "Two classes overlap in 2D but are separable in 3D. "
            "Which algorithm would handle this better: Logistic "
            "Regression or SVM with RBF kernel?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.decomposition import PCA\n"
            "\n"
            "pca = PCA(n_components=2)\n"
            "X_2d = pca.fit_transform(X_train)\n"
            "model.fit(X_2d, y_train)\n"
            "\n"
            "xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))\n"
            "Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)\n"
            "plt.contourf(xx, yy, Z, alpha=0.3)\n"
            "plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, s=10)\n"
            "```"
        ),
        keywords=["boundary", "decision", "separator", "boundary", "visualisation"],
    ),

    # ── 16 ─────────────────────────────────────────────────────────
    "classification_threshold": T(
        title="Classification Threshold",
        module="classification",
        what=(
            "The classification threshold converts probabilities into "
            "class labels. Default is 0.5: P(positive) >= 0.5 → class 1. "
            "Changing the threshold shifts the precision/recall trade-off."
        ),
        why=(
            "The default 0.5 threshold isn't always optimal. Lowering "
            "to 0.3 catches more positives (higher recall) but adds "
            "false positives (lower precision)."
        ),
        when=(
            "Use when the cost of false positives and false negatives "
            "differs. Lower threshold when missing positives is costly "
            "(cancer detection). Raise it when false alarms are costly "
            "(spam filtering)."
        ),
        example=(
            "```python\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "\n"
            "# Default threshold (0.5)\n"
            "y_pred_default = (y_prob >= 0.5).astype(int)\n"
            "\n"
            "# Lower threshold → more recall\n"
            "y_pred_low = (y_prob >= 0.3).astype(int)\n"
            "\n"
            "# Higher threshold → more precision\n"
            "y_pred_high = (y_prob >= 0.7).astype(int)\n"
            "```"
        ),
        mistakes=[
            "Always using 0.5 without considering the problem context.",
            "Not evaluating at multiple thresholds.",
            "Changing threshold on test data instead of validation data.",
        ],
        interpretation=(
            "At threshold 0.3: more predictions are positive → higher "
            "recall (catch more actual positives) but lower precision "
            "(more false alarms). At 0.7: fewer positive predictions → "
            "higher precision, lower recall."
        ),
        think_about_it=(
            "A cancer screening model has threshold 0.5 and catches "
            "80% of cancers. If you lower the threshold to 0.3, "
            "recall increases to 95% but precision drops to 60%. "
            "Should you change the threshold?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import precision_recall_curve\n"
            "\n"
            "precision, recall, thresholds = precision_recall_curve(y_test, y_prob)\n"
            "\n"
            "# Find threshold for desired recall\n"
            "target_recall = 0.9\n"
            "idx = np.argmin(np.abs(recall - target_recall))\n"
            "print(f'Threshold for 90% recall: {thresholds[idx]:.3f}')\n"
            "```"
        ),
        keywords=["threshold", "cutoff", "decision", "precision", "recall"],
    ),

    # ── 17 ─────────────────────────────────────────────────────────
    "class_imbalance": T(
        title="Class Imbalance",
        module="classification",
        what=(
            "Class imbalance occurs when one class has significantly "
            "more samples than another. E.g., 95% 'No' and 5% 'Yes'. "
            "Models tend to be biased toward the majority class."
        ),
        why=(
            "Imbalanced data is the norm, not the exception. Fraud "
            "detection, disease diagnosis, and defect detection all "
            "have rare positive cases. Standard models will predict "
            "the majority class and achieve high accuracy while "
            "missing all minority cases."
        ),
        when=(
            "Check class balance with value_counts(normalize=True). "
            "If one class has <30% of the data, apply techniques "
            "like class weighting or resampling."
        ),
        example=(
            "```python\n"
            "print(y_train.value_counts(normalize=True))\n"
            "# No     0.616  (549)\n"
            "# Yes    0.384  (342)\n"
            "\n"
            "# Handle with class weights\n"
            "rf = RandomForestClassifier(class_weight='balanced')\n"
            "rf.fit(X_train, y_train)\n"
            "```"
        ),
        mistakes=[
            "Reporting accuracy on imbalanced data — it's misleading.",
            "Not using stratify in train_test_split.",
            "Oversampling without checking for data leakage.",
        ],
        interpretation=(
            "With 95% majority class, a model predicting 'majority' "
            "achieves 95% accuracy but 0% recall on the minority class. "
            "Use F1, precision, recall, and confusion matrix instead."
        ),
        think_about_it=(
            "A fraud detection model has 99.5% accuracy. The confusion "
            "matrix shows it detected 0 out of 50 fraud cases. "
            "What metric should you optimise?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from imblearn.over_sampling import SMOTE\n"
            "\n"
            "# Option 1: class weights\n"
            "rf = RandomForestClassifier(class_weight='balanced')\n"
            "\n"
            "# Option 2: SMOTE oversampling\n"
            "smote = SMOTE(random_state=42)\n"
            "X_res, y_res = smote.fit_resample(X_train, y_train)\n"
            "```"
        ),
        keywords=["imbalance", "balanced", "class weight", "minority", "majority"],
    ),

    # ── 18 ─────────────────────────────────────────────────────────
    "overfitting_classification": T(
        title="Overfitting in Classification",
        module="classification",
        what=(
            "Overfitting occurs when a classifier memorises training "
            "data instead of learning general patterns. It performs "
            "great on training data but poorly on test data."
        ),
        why=(
            "Overfitting is the most common classification failure. "
            "Understanding its causes (too complex model, too little "
            "data, too many features) is essential for building "
            "reliable models."
        ),
        when=(
            "Always check by comparing train vs test scores. A large "
            "gap indicates overfitting."
        ),
        example=(
            "```\n"
            "Model              Train Acc   Test Acc   Diagnosis\n"
            "Decision Tree(d=20)   1.00       0.78     Overfitting\n"
            "Random Forest        0.95       0.93     Good fit\n"
            "Logistic Reg         0.80       0.79     Underfitting\n"
            "```"
        ),
        mistakes=[
            "Not checking train vs test scores.",
            "Using deep decision trees without pruning.",
            "Having more features than samples.",
        ],
        interpretation=(
            "Train ≫ Test → Overfit. Solutions: simplify the model "
            "(reduce depth, add regularization), get more data, "
            "reduce features, or use cross-validation."
        ),
        think_about_it=(
            "A Random Forest with 1000 trees achieves 98% train and "
            "85% test accuracy. What would you try to close the gap?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "train_scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "test_scores = cross_val_score(model, X_test, y_test, cv=5)\n"
            "print(f'Train CV: {train_scores.mean():.4f}')\n"
            "print(f'Test CV:  {test_scores.mean():.4f}')\n"
            "```"
        ),
        keywords=["overfit", "overfitting", "memorise", "generalise", "gap"],
    ),

    # ── 19 ─────────────────────────────────────────────────────────
    "underfitting_classification": T(
        title="Underfitting in Classification",
        module="classification",
        what=(
            "Underfitting occurs when a classifier is too simple to "
            "capture the underlying patterns. Both training and test "
            "scores are low."
        ),
        why=(
            "Underfitting means the model isn't powerful enough. "
            "No amount of data or preprocessing can fix it — you "
            "need a more complex model or better features."
        ),
        when=(
            "When both train and test accuracy are low. Common with "
            "linear models on non-linear data."
        ),
        example=(
            "```python\n"
            "# Logistic Regression on non-linear data\n"
            "lr = LogisticRegression()\n"
            "lr.fit(X_train, y_train)\n"
            "print(f'Train: {lr.score(X_train, y_train):.4f}')  # 0.62\n"
            "print(f'Test:  {lr.score(X_test, y_test):.4f}')    # 0.60\n"
            "# Both low → underfitting\n"
            "```"
        ),
        mistakes=[
            "Trying to fix underfitting by adding more data.",
            "Not considering that the model may be too simple.",
            "Ignoring feature engineering as a solution.",
        ],
        interpretation=(
            "Solutions: use a more complex model (Random Forest, "
            "Gradient Boosting), add polynomial features, engineer "
            "better features, or reduce regularization."
        ),
        think_about_it=(
            "A linear model gives 65% accuracy on a dataset with "
            "clear non-linear patterns. Is the problem the data "
            "or the model?"
        ),
        code_link=(
            "```python\n"
            "# Instead of simple logistic regression, try:\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "\n"
            "# Option 1: more complex model\n"
            "rf = RandomForestClassifier(n_estimators=100)\n"
            "\n"
            "# Option 2: add polynomial features\n"
            "poly = PolynomialFeatures(degree=2)\n"
            "X_poly = poly.fit_transform(X_train)\n"
            "```"
        ),
        keywords=["underfit", "underfitting", "simple", "complexity", "bias"],
    ),

    # ── 20 ─────────────────────────────────────────────────────────
    "classification_case_study": T(
        title="Classification Case Study",
        module="classification",
        what=(
            "Putting it all together: a complete classification workflow "
            "on the Titanic dataset, from loading to evaluation."
        ),
        why=(
            "Seeing the full workflow helps you connect individual "
            "concepts into a practical data science process."
        ),
        when=(
            "This is a reference workflow. Follow these steps for any "
            "classification project."
        ),
        example=(
            "Complete Titanic survival prediction workflow:"
        ),
        mistakes=[
            "Skipping EDA and jumping to modelling.",
            "Not comparing multiple models.",
            "Reporting only one metric.",
        ],
        interpretation=(
            "The best model balances accuracy, interpretability, "
            "training time, and business requirements. Always compare "
            "at least 3 algorithms."
        ),
        think_about_it=(
            "After completing the workflow, you get 82% accuracy. "
            "What next steps would you try to improve it?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split, cross_val_score\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "# 1. Load\n"
            "df = pd.read_csv('titanic.csv')\n"
            "\n"
            "# 2. Select features\n"
            "X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]\n"
            "y = df['Survived']\n"
            "\n"
            "# 3. Preprocessing\n"
            "num_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])\n"
            "cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "                      ('encoder', OneHotEncoder(handle_unknown='ignore'))])\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', num_pipe, ['Age', 'SibSp', 'Parch', 'Fare']),\n"
            "    ('cat', cat_pipe, ['Sex', 'Pclass']),\n"
            "])\n"
            "\n"
            "# 4. Split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)\n"
            "\n"
            "# 5. Train & compare\n"
            "for name, model in [('RF', RandomForestClassifier()), ('LR', LogisticRegression(max_iter=1000))]:\n"
            "    pipe = Pipeline([('prep', preprocessor), ('model', model)])\n"
            "    scores = cross_val_score(pipe, X_train, y_train, cv=5)\n"
            "    print(f'{name}: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["case study", "workflow", "end-to-end", "titanic", "complete"],
    ),
}
