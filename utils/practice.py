"""
Practice mode for Data Science Lab.

Provides structured, academically-focused challenges where students
apply what they've learned.  Each challenge:

1. Presents a dataset/scenario
2. Asks a conceptual or applied question
3. Evaluates the student's answer
4. Provides detailed feedback and correct reasoning

No gamification — pure academic practice with immediate feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class ChallengeOption:
    """A single answer option."""
    label: str
    text: str
    is_correct: bool = False


@dataclass
class Challenge:
    """A practice challenge for a data science topic."""
    id: str
    topic: str
    title: str
    scenario: str
    question: str
    options: list[ChallengeOption]
    correct_index: int
    explanation: str
    follow_up: str  # deeper reasoning after answering
    code_hint: str = ""  # optional code snippet hint
    difficulty: str = "intermediate"  # beginner | intermediate | advanced


# ══════════════════════════════════════════════════════════════════════
#  CHALLENGE BANK
# ══════════════════════════════════════════════════════════════════════

CHALLENGES: list[Challenge] = [

    # ── Missing Values ──────────────────────────────────────────────

    Challenge(
        id="mv_01",
        topic="missing_values",
        title="Choosing an Imputation Strategy",
        scenario=(
            "You have a dataset with 10,000 rows. The 'Income' column "
            "has 3,000 missing values (30%). The distribution of Income "
            "is heavily right-skewed with several extreme outliers "
            "(e.g., $500K when most people earn $30K–$80K)."
        ),
        question="What is the BEST imputation strategy for the 'Income' column?",
        options=[
            ChallengeOption("A", "Fill with mean", False),
            ChallengeOption("B", "Fill with median", True),
            ChallengeOption("C", "Fill with zero", False),
            ChallengeOption("D", "Drop all rows with missing Income", False),
        ],
        correct_index=1,
        explanation=(
            "**Median** is best here because:\n"
            "1. The data is **skewed** — the mean is pulled up by outliers ($500K).\n"
            "2. The median is **robust to outliers** — it represents the 'typical' value.\n"
            "3. Dropping 30% of rows loses too much information from other columns.\n"
            "4. Filling with zero would create a false 'low income' cluster."
        ),
        follow_up=(
            "With 30% missing, also consider: is the data Missing Completely "
            "At Random (MCAR), Missing At Random (MAR), or Missing Not At "
            "Random (MNAR)? If people with very high income are hiding their "
            "earnings (MNAR), no simple imputation is fully correct."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "imputer = SimpleImputer(strategy='median')\n"
            "df['Income'] = imputer.fit_transform(df[['Income']])\n"
            "```"
        ),
        difficulty="beginner",
    ),

    Challenge(
        id="mv_02",
        topic="missing_values",
        title="Data Leakage in Imputation",
        scenario=(
            "You are building a pipeline to predict house prices. "
            "You decide to fill missing 'LotArea' values with the "
            "mean of the entire dataset before splitting into "
            "train/test sets."
        ),
        question="What is wrong with this approach?",
        options=[
            ChallengeOption("A", "Nothing — mean imputation is always fine", False),
            ChallengeOption("B", "The test set's information leaks into training via the global mean", True),
            ChallengeOption("C", "Mean imputation doesn't work for numerical data", False),
            ChallengeOption("D", "You should use median instead of mean", False),
        ],
        correct_index=1,
        explanation=(
            "This is **data leakage**. The mean is computed on ALL data, "
            "including test samples. This means the model indirectly "
            "'sees' test data during training.\n\n"
            "**Correct approach:** Split first, then fit the imputer on "
            "training data only, and transform both train and test."
        ),
        follow_up=(
            "In a sklearn Pipeline, this is handled automatically — "
            "the preprocessor is fitted only on training data during "
            "cross-validation. This is why using Pipeline is a best practice."
        ),
        code_hint=(
            "```python\n"
            "# WRONG: fitting on all data\n"
            "df['LotArea'] = df['LotArea'].fillna(df['LotArea'].mean())\n"
            "X_train, X_test = train_test_split(df)\n"
            "\n"
            "# RIGHT: split first, then impute\n"
            "X_train, X_test = train_test_split(df)\n"
            "mean = X_train['LotArea'].mean()\n"
            "X_train['LotArea'] = X_train['LotArea'].fillna(mean)\n"
            "X_test['LotArea'] = X_test['LotArea'].fillna(mean)\n"
            "```"
        ),
        difficulty="intermediate",
    ),

    # ── Feature Scaling ─────────────────────────────────────────────

    Challenge(
        id="fs_01",
        topic="feature_scaling",
        title="Which Algorithms Need Scaling?",
        scenario=(
            "Your dataset has three features:\n"
            "- Age: 18–65 (range 47)\n"
            "- Income: $20,000–$200,000 (range $180,000)\n"
            "- Years_of_Education: 8–20 (range 12)\n\n"
            "You want to train a KNN classifier."
        ),
        question="Why is feature scaling essential for KNN in this case?",
        options=[
            ChallengeOption("A", "KNN uses distance calculations, so Income dominates due to its larger scale", True),
            ChallengeOption("B", "KNN requires all features to be positive", False),
            ChallengeOption("C", "Scaling makes the model train faster", False),
            ChallengeOption("D", "KNN cannot handle features with different ranges", False),
        ],
        correct_index=0,
        explanation=(
            "KNN classifies by finding the **k nearest neighbors** using "
            "distance (typically Euclidean). Income ($20K–$200K) has a "
            "range 3,800× larger than Age (47).\n\n"
            "Without scaling, Income **completely dominates** the distance "
            "calculation — Age and Education become irrelevant."
        ),
        follow_up=(
            "After scaling, try training KNN with and without scaling "
            "and compare accuracy. You'll likely see a significant "
            "improvement with scaling."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)  # use train stats!\n"
            "```"
        ),
        difficulty="beginner",
    ),

    Challenge(
        id="fs_02",
        topic="feature_scaling",
        title="Scaling with Outliers",
        scenario=(
            "Your 'Payment_Amount' column has values mostly between "
            "$10–$100, but 5 rows have values of $50,000+ due to "
            "data entry errors. You need to scale this feature."
        ),
        question="Which scaler is most appropriate?",
        options=[
            ChallengeOption("A", "StandardScaler — it's the default", False),
            ChallengeOption("B", "MinMaxScaler — it bounds to [0,1]", False),
            ChallengeOption("C", "RobustScaler — uses median and IQR, resistant to outliers", True),
            ChallengeOption("D", "No scaler needed — tree models don't require it", False),
        ],
        correct_index=2,
        explanation=(
            "**RobustScaler** uses the median and interquartile range (IQR) "
            "instead of mean and standard deviation. Outliers have minimal "
            "effect on these statistics.\n\n"
            "StandardScaler computes mean and std, which are pulled by outliers. "
            "MinMaxScaler is even worse — one $50K value squashes everything "
            "else into a tiny range."
        ),
        follow_up=(
            "Even better: first identify and handle the outliers "
            "(e.g., cap at 99th percentile), then use StandardScaler. "
            "Preprocessing decisions should be based on domain knowledge."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "scaler = RobustScaler()  # uses median, IQR\n"
            "X_scaled = scaler.fit_transform(X_train[['Payment_Amount']])\n"
            "```"
        ),
        difficulty="intermediate",
    ),

    # ── Classification Metrics ──────────────────────────────────────

    Challenge(
        id="cm_01",
        topic="classification_metrics",
        title="Choosing the Right Metric",
        scenario=(
            "You're building a model to detect fraudulent transactions. "
            "Only 0.5% of transactions are fraudulent. Your model "
            "predicts 'not fraud' for everything and achieves 99.5% "
            "accuracy."
        ),
        question="Why is accuracy the wrong metric here?",
        options=[
            ChallengeOption("A", "Accuracy is always unreliable", False),
            ChallengeOption("B", "The model catches zero frauds despite high accuracy — it's a useless model", True),
            ChallengeOption("C", "Accuracy can only be used for regression", False),
            ChallengeOption("D", "99.5% accuracy is too high to be meaningful", False),
        ],
        correct_index=1,
        explanation=(
            "With 99.5% non-fraud, a model that always says 'not fraud' "
            "gets 99.5% accuracy — but **catches zero frauds**.\n\n"
            "**Recall** (sensitivity) is critical here: what percentage of "
            "actual frauds did we catch? With this 'dumb' model, recall = 0%.\n\n"
            "For imbalanced data, use: Precision, Recall, F1, or PR-AUC."
        ),
        follow_up=(
            "In fraud detection, false negatives (missed fraud) are very "
            "costly. You'd prioritise **recall** — catching as many frauds "
            "as possible, even at the cost of some false alarms."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.metrics import classification_report\n"
            "print(classification_report(y_test, y_pred))\n"
            "# Focus on the 'fraud' class: recall and f1-score\n"
            "```"
        ),
        difficulty="beginner",
    ),

    Challenge(
        id="cm_02",
        topic="classification_metrics",
        title="Precision vs Recall Trade-off",
        scenario=(
            "A cancer screening model has:\n"
            "- Precision = 0.40 (40% of positive predictions are correct)\n"
            "- Recall = 0.95 (it catches 95% of actual cancer cases)\n\n"
            "The hospital board wants to improve precision to 0.80."
        ),
        question="What will likely happen to recall if precision is increased?",
        options=[
            ChallengeOption("A", "Recall will stay the same — they're independent", False),
            ChallengeOption("B", "Recall will increase too — both measure correctness", False),
            ChallengeOption("C", "Recall will decrease — increasing precision means raising the classification threshold, catching fewer cases", True),
            ChallengeOption("D", "Both metrics will improve with more training data", False),
        ],
        correct_index=2,
        explanation=(
            "Precision and recall are **inversely related** through the "
            "classification threshold.\n\n"
            "To increase precision (fewer false positives), you raise the "
            "threshold → fewer positive predictions → you miss more actual "
            "cases → recall drops.\n\n"
            "This is the fundamental **precision-recall trade-off**."
        ),
        follow_up=(
            "In cancer screening, missing a case (false negative) is "
            "much worse than a false alarm (false positive). So the "
            "current model with high recall (0.95) may actually be "
            "appropriate despite low precision."
        ),
        code_hint=(
            "```python\n"
            "# Adjust threshold\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "threshold = 0.7  # higher = more precision, less recall\n"
            "y_pred = (y_prob >= threshold).astype(int)\n"
            "```"
        ),
        difficulty="advanced",
    ),

    # ── Overfitting ─────────────────────────────────────────────────

    Challenge(
        id="of_01",
        topic="overfitting",
        title="Diagnosing Model Problems",
        scenario=(
            "You train two models on the same dataset:\n\n"
            "**Model A (Decision Tree, max_depth=None):**\n"
            "Train accuracy: 1.00 | Test accuracy: 0.78\n\n"
            "**Model B (Logistic Regression):**\n"
            "Train accuracy: 0.82 | Test accuracy: 0.81"
        ),
        question="Which model has a better fit, and why?",
        options=[
            ChallengeOption("A", "Model A — it has perfect training accuracy", False),
            ChallengeOption("B", "Model B — small train/test gap means it generalises well", True),
            ChallengeOption("C", "Both are equally good", False),
            ChallengeOption("D", "Neither — both have low test accuracy", False),
        ],
        correct_index=1,
        explanation=(
            "Model A is **overfitting**: perfect training score but much "
            "lower test score (gap = 0.22). It memorised the training data.\n\n"
            "Model B has a tiny gap (0.01) between train and test — it "
            "**generalises** well. It learned patterns, not noise.\n\n"
            "Always compare train vs test performance."
        ),
        follow_up=(
            "To fix Model A, try: limit max_depth, increase min_samples_split, "
            "use pruning, or switch to Random Forest (which averages many trees)."
        ),
        code_hint=(
            "```python\n"
            "# Compare train vs test\n"
            "train_acc = model.score(X_train, y_train)\n"
            "test_acc = model.score(X_test, y_test)\n"
            "gap = train_acc - test_acc\n"
            "if gap > 0.1:\n"
            "    print('Warning: possible overfitting')\n"
            "```"
        ),
        difficulty="beginner",
    ),

    # ── Cross-Validation ────────────────────────────────────────────

    Challenge(
        id="cv_01",
        topic="cross_validation",
        title="Why Cross-Validation?",
        scenario=(
            "You split your data 80/20 and get 92% accuracy on the "
            "test set. You're confident the model is good. Your "
            "professor says to use cross-validation instead."
        ),
        question="Why might a single train/test split be unreliable?",
        options=[
            ChallengeOption("A", "It's always unreliable — never use train/test split", False),
            ChallengeOption("B", "The test set might be 'lucky' — a different split could give a very different score", True),
            ChallengeOption("C", "Cross-validation is only for neural networks", False),
            ChallengeOption("D", "Train/test split takes too long to compute", False),
        ],
        correct_index=1,
        explanation=(
            "A single split is a **single sample** of performance. "
            "Depending on which rows end up in the test set, the score "
            "can vary significantly.\n\n"
            "Cross-validation trains and tests k times on different "
            "splits, giving you a **mean and standard deviation** — "
            "a much more reliable estimate."
        ),
        follow_up=(
            "Try it: run train_test_split 10 times with different "
            "random_state values and record the test accuracy. You'll "
            "see it varies by several percentage points."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# 5-fold CV gives a mean ± std\n"
            "scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n"
            "print(f'{scores.mean():.3f} ± {scores.std():.3f}')\n"
            "# e.g., 0.934 ± 0.021 (much more informative than a single number)\n"
            "```"
        ),
        difficulty="beginner",
    ),

    # ── Clustering ──────────────────────────────────────────────────

    Challenge(
        id="cl_01",
        topic="clustering_basics",
        title="Choosing K for K-Means",
        scenario=(
            "You run K-Means with different values of k and get:\n\n"
            "k=2: Inertia=5200, Silhouette=0.45\n"
            "k=3: Inertia=3100, Silhouette=0.58\n"
            "k=4: Inertia=2800, Silhouette=0.42\n"
            "k=5: Inertia=2600, Silhouette=0.35"
        ),
        question="Based on these results, which k is best?",
        options=[
            ChallengeOption("A", "k=5 — lowest inertia", False),
            ChallengeOption("B", "k=3 — highest silhouette score", True),
            ChallengeOption("C", "k=2 — simplest model", False),
            ChallengeOption("D", "k=4 — good balance", False),
        ],
        correct_index=1,
        explanation=(
            "**k=3** has the highest silhouette score (0.58), meaning "
            "clusters are most well-separated.\n\n"
            "While inertia always decreases with k (more clusters = "
            "less within-cluster variance), silhouette measures how "
            "well each point fits its cluster vs others.\n\n"
            "The 'elbow' in inertia is between k=2 and k=3, confirming "
            "k=3 is the natural choice."
        ),
        follow_up=(
            "Silhouette doesn't always agree with the elbow method. "
            "When they disagree, visualise the clusters (via PCA) and "
            "consult domain knowledge."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import silhouette_score\n"
            "\n"
            "for k in range(2, 7):\n"
            "    km = KMeans(n_clusters=k, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    sil = silhouette_score(X_scaled, labels)\n"
            "    print(f'k={k}: silhouette={sil:.3f}, inertia={km.inertia_:.0f}')\n"
            "```"
        ),
        difficulty="intermediate",
    ),

    # ── Feature Engineering ─────────────────────────────────────────

    Challenge(
        id="fe_01",
        topic="feature_engineering_basics",
        title="Log Transform for Skewed Data",
        scenario=(
            "You have a 'Salary' column with values: "
            "[30000, 35000, 40000, 45000, 50000, 55000, 60000, 200000]. "
            "The distribution is heavily right-skewed."
        ),
        question="Why would a log transform help?",
        options=[
            ChallengeOption("A", "It makes the data fit a normal distribution, which some models assume", True),
            ChallengeOption("B", "It removes missing values", False),
            ChallengeOption("C", "It converts categorical data to numerical", False),
            ChallengeOption("D", "It increases the variance of the feature", False),
        ],
        correct_index=0,
        explanation=(
            "A log transform compresses the right tail of a skewed "
            "distribution, making it more **symmetric and bell-shaped**.\n\n"
            "Many models (Linear Regression, Logistic Regression) assume "
            "roughly normal features. Log transform also reduces the "
            "influence of extreme values (like the $200K salary)."
        ),
        follow_up=(
            "Use np.log1p() instead of np.log() — log1p(x) = log(1+x) "
            "handles zero values gracefully. After transforming, always "
            "re-check the distribution with a histogram."
        ),
        code_hint=(
            "```python\n"
            "import numpy as np\n"
            "df['log_salary'] = np.log1p(df['Salary'])\n"
            "# Verify the transform\n"
            "df['log_salary'].hist()\n"
            "```"
        ),
        difficulty="beginner",
    ),

    # ── Model Selection ─────────────────────────────────────────────

    Challenge(
        id="ms_01",
        topic="model_selection",
        title="No Free Lunch",
        scenario=(
            "You compare three models on two different datasets:\n\n"
            "**Dataset A (Iris — small, clean, linear):**\n"
            "Logistic Regression: 96% | Random Forest: 95% | KNN: 97%\n\n"
            "**Dataset B (Customer Churn — large, noisy, non-linear):**\n"
            "Logistic Regression: 71% | Random Forest: 84% | KNN: 68%"
        ),
        question="What does this demonstrate about model selection?",
        options=[
            ChallengeOption("A", "Random Forest is always the best model", False),
            ChallengeOption("B", "KNN is always the worst model", False),
            ChallengeOption("C", "No single model is universally best — performance depends on the data", True),
            ChallengeOption("D", "Logistic Regression should never be used", False),
        ],
        correct_index=2,
        explanation=(
            "This illustrates the **No Free Lunch theorem**: no algorithm "
            "works best on every problem.\n\n"
            "On Iris (linear, clean), simpler models do great. On Churn "
            "(non-linear, noisy), ensemble methods (Random Forest) win.\n\n"
            "Always **try multiple models** and compare."
        ),
        follow_up=(
            "This is why we always start with a simple baseline "
            "(Logistic/Linear Regression) — it tells you if the problem "
            "is linear. If it underperforms, move to non-linear models."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "models = {'LR': LogisticRegression(), 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier()}\n"
            "for name, m in models.items():\n"
            "    scores = cross_val_score(m, X, y, cv=5)\n"
            "    print(f'{name}: {scores.mean():.3f} +/- {scores.std():.3f}')\n"
            "```"
        ),
        difficulty="intermediate",
    ),

    # ── Encoding ────────────────────────────────────────────────────

    Challenge(
        id="en_01",
        topic="categorical_encoding",
        title="One-Hot vs Label Encoding",
        scenario=(
            "You have a 'Color' column with values: "
            "Red, Blue, Green, Yellow. You need to encode it for "
            "a Logistic Regression model."
        ),
        question="Which encoding should you use and why?",
        options=[
            ChallengeOption("A", "Label encoding: Red=0, Blue=1, Green=2, Yellow=3 — it's simpler", False),
            ChallengeOption("B", "One-hot encoding — colors have no natural order, so label encoding would imply Red < Blue < Green", True),
            ChallengeOption("C", "Neither — Logistic Regression handles strings", False),
            ChallengeOption("D", "Either works the same — the model doesn't care", False),
        ],
        correct_index=1,
        explanation=(
            "Label encoding assigns integers that imply **order**: "
            "Red(0) < Blue(1) < Green(2) < Yellow(3). But colors "
            "have no inherent ranking.\n\n"
            "**One-hot encoding** creates separate binary columns "
            "(is_Red, is_Blue, is_Green, is_Yellow) with no false "
            "ordering. Use it for **nominal** categories."
        ),
        follow_up=(
            "Exception: Tree-based models can sometimes handle label "
            "encoding because they split on individual values. But for "
            "linear models and distance-based models, always use one-hot."
        ),
        code_hint=(
            "```python\n"
            "# One-hot for nominal (no order)\n"
            "df = pd.get_dummies(df, columns=['Color'])\n"
            "\n"
            "# Label for ordinal (has order)\n"
            "# e.g., Education: High School < Bachelor < Master < PhD\n"
            "from sklearn.preprocessing import OrdinalEncoder\n"
            "```"
        ),
        difficulty="beginner",
    ),

    # ── Train/Test Split ────────────────────────────────────────────

    Challenge(
        id="tt_01",
        topic="train_test_split",
        title="Stratification Matters",
        scenario=(
            "You have a dataset with 1000 samples: 950 'No' and "
            "50 'Yes' (5% positive class). You do a random 80/20 "
            "split without stratification."
        ),
        question="What problem might occur?",
        options=[
            ChallengeOption("A", "The test set might have very few or zero 'Yes' samples, making evaluation unreliable", True),
            ChallengeOption("B", "The model will overfit", False),
            ChallengeOption("C", "Training will be slower", False),
            ChallengeOption("D", "The data will be corrupted", False),
        ],
        correct_index=0,
        explanation=(
            "Without stratification, the random split might put most "
            "or all 'Yes' samples in the training set. The test set "
            "could have only 1-2 positive cases (or none), making "
            "recall/F1 evaluation meaningless.\n\n"
            "**Stratification** ensures both sets have the same class "
            "proportions as the original data."
        ),
        follow_up=(
            "With very small minority classes (<50 samples), consider "
            "stratified k-fold cross-validation instead of a single "
            "split for more reliable evaluation."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Without stratification (risky for imbalanced data)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "\n"
            "# With stratification (preserves class balance)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y\n"
            ")\n"
            "```"
        ),
        difficulty="beginner",
    ),

    # ── Regression Metrics ──────────────────────────────────────────

    Challenge(
        id="rg_01",
        topic="regression_metrics",
        title="Interpreting R²",
        scenario=(
            "You train a regression model to predict house prices. "
            "The test set R² = 0.85, MAE = $15,000, RMSE = $25,000."
        ),
        question="What does R² = 0.85 actually mean?",
        options=[
            ChallengeOption("A", "The model is 85% accurate", False),
            ChallengeOption("B", "The model explains 85% of the variance in house prices", True),
            ChallengeOption("C", "85% of predictions are within $15K of the actual price", False),
            ChallengeOption("D", "The model is correct 85% of the time", False),
        ],
        correct_index=1,
        explanation=(
            "R² = 0.85 means the model **explains 85% of the variance** "
            "in the target variable. The remaining 15% is unexplained "
            "(noise, missing features, etc.).\n\n"
            "It does NOT mean 85% accuracy or 85% correct predictions. "
            "It's about how much variation the model captures."
        ),
        follow_up=(
            "RMSE ($25K) > MAE ($15K) suggests there are some large "
            "errors pulling RMSE up. Check the residual plot for "
            "outlier predictions."
        ),
        code_hint=(
            "```python\n"
            "from sklearn.metrics import r2_score, mean_absolute_error\n"
            "import numpy as np\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "print(f'R²:   {r2_score(y_test, y_pred):.4f}')\n"
            "print(f'MAE:  ${mean_absolute_error(y_test, y_pred):,.0f}')\n"
            "print(f'RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}')\n"
            "```"
        ),
        difficulty="beginner",
    ),
]


# ── Public API ──────────────────────────────────────────────────────

def get_challenges_by_topic(topic: str) -> list[Challenge]:
    """Return all challenges for a given topic."""
    return [c for c in CHALLENGES if c.topic == topic]


def get_challenge(challenge_id: str) -> Challenge | None:
    """Return a single challenge by ID."""
    for c in CHALLENGES:
        if c.id == challenge_id:
            return c
    return None


def list_challenge_topics() -> list[str]:
    """Return unique topic keys that have challenges."""
    return list(dict.fromkeys(c.topic for c in CHALLENGES))


def check_answer(challenge: Challenge, selected_index: int) -> dict:
    """
    Check a student's answer and return structured feedback.

    Returns
    -------
    dict with keys:
        correct: bool
        selected: int
        correct_option: ChallengeOption
        explanation: str
        follow_up: str
        code_hint: str
    """
    is_correct = selected_index == challenge.correct_index
    return {
        "correct": is_correct,
        "selected": selected_index,
        "correct_option": challenge.options[challenge.correct_index],
        "explanation": challenge.explanation,
        "follow_up": challenge.follow_up,
        "code_hint": challenge.code_hint,
    }
