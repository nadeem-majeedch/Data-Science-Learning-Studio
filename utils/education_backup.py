"""
Educational content library for Data Science Learning Studio.

Provides structured, academically-focused content for every major
data science topic.  Each topic entry includes:

- What is it?
- Why is it important?
- When should it be used?
- Simple example
- Common mistakes
- Interpretation of results
- "Think About It" question
- Link to the underlying Python/sklearn code

Used by the Learning Mode and Practice Mode pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicContent:
    """Structured educational content for a single data science topic."""
    title: str
    module: str  # which module page this belongs to
    what: str
    why: str
    when: str
    example: str
    common_mistakes: list[str]
    interpretation: str
    think_about_it: str
    code_link: str  # snippet showing the Python/sklearn code
    keywords: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
#  TOPIC CONTENT
# ══════════════════════════════════════════════════════════════════════

TOPICS: dict[str, TopicContent] = {

    # ── Data Loading ────────────────────────────────────────────────

    "data_loading": TopicContent(
        title="Loading Data",
        module="dataset_explorer",
        what=(
            "Loading data means reading a file (CSV, Excel, etc.) into a "
            "pandas DataFrame — a table-like structure you can inspect, "
            "filter, and transform."
        ),
        why=(
            "Every data science project starts with data. If loading fails "
            "or the data is read incorrectly, all downstream analysis will "
            "be wrong."
        ),
        when=(
            "Always the first step. Use CSV for plain-text tabular data, "
            "Excel (.xlsx) for spreadsheets with formatting, and Parquet "
            "for large efficient storage."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_csv('iris.csv')\n"
            "print(df.shape)  # (150, 5)\n"
            "print(df.head())\n"
            "```"
        ),
        common_mistakes=[
            "Not checking if the file path is correct (relative vs absolute).",
            "Forgetting to specify a separator for TSV files (use sep='\\t').",
            "Not checking column types after loading — numbers may be read as strings.",
            "Loading a file with encoding issues (try encoding='latin-1' or 'cp1252').",
        ],
        interpretation=(
            "After loading, always check: shape (rows × columns), dtypes "
            "(are numbers actually numeric?), head() (does the data look right?), "
            "and isnull().sum() (how much is missing?)."
        ),
        think_about_it=(
            "If a column like 'Age' is read as type 'object' instead of "
            "'float64', what might be causing that, and how would you fix it?"
        ),
        code_link=(
            "```python\n"
            "df = pd.read_csv('data.csv')           # CSV\n"
            "df = pd.read_excel('data.xlsx')         # Excel\n"
            "df = pd.read_csv('data.tsv', sep='\\t') # TSV\n"
            "```"
        ),
        keywords=["load", "csv", "read", "import", "file", "pandas", "dataframe"],
    ),

    # ── Data Types ──────────────────────────────────────────────────

    "data_types": TopicContent(
        title="Understanding Data Types",
        module="dataset_explorer",
        what=(
            "Data types define what kind of values a column holds: "
            "numerical (int, float), categorical (strings/categories), "
            "boolean (True/False), or datetime."
        ),
        why=(
            "Different algorithms expect different data types. "
            "Numerical features can be scaled; categorical features must "
            "be encoded. Using the wrong type silently produces wrong results."
        ),
        when=(
            "Check data types immediately after loading. The dtypes attribute "
            "and df.info() give you a quick overview."
        ),
        example=(
            "```python\n"
            "print(df.dtypes)\n"
            "# sepal_length    float64\n"
            "# species          object  ← this is categorical\n"
            "\n"
            "# Convert string to category\n"
            "df['species'] = df['species'].astype('category')\n"
            "```"
        ),
        common_mistakes=[
            "Treating a categorical column as numerical (e.g., zip codes).",
            "Not converting strings that represent numbers (e.g., '$10.50').",
            "Ignoring datetime columns that could be split into year/month/day.",
        ],
        interpretation=(
            "float64/int64 = numerical (can do math on them). "
            "object/category = categorical (need encoding before modelling). "
            "bool = binary (often the target for classification)."
        ),
        think_about_it=(
            "A column 'Gender' has values 'M', 'F', 'Other'. Should this "
            "be treated as categorical or numerical? Why?"
        ),
        code_link=(
            "```python\n"
            "df.dtypes                          # check types\n"
            "df.info()                          # summary with non-null counts\n"
            "df.select_dtypes('number')         # numerical columns only\n"
            "df.select_dtypes('object')         # string columns only\n"
            "```"
        ),
        keywords=["dtype", "type", "numerical", "categorical", "object", "category"],
    ),

    # ── Missing Values ──────────────────────────────────────────────

    "missing_values": TopicContent(
        title="Handling Missing Values",
        module="preprocessing",
        what=(
            "Missing values (NaN, None, blank cells) occur when data wasn't "
            "recorded, merged, or available. They break most sklearn models."
        ),
        why=(
            "Most machine learning algorithms cannot handle missing data. "
            "Even if a model runs, missing values introduce bias and reduce "
            "accuracy. You must decide: drop or impute."
        ),
        when=(
            "Before training any model. Check with df.isnull().sum(). "
            "Drop rows only when missing data is minimal (<5%). "
            "Otherwise, impute with mean, median, or mode."
        ),
        example=(
            "```python\n"
            "df.isnull().sum()                    # count missing per column\n"
            "\n"
            "# Drop rows with any missing values\n"
            "df_clean = df.dropna()\n"
            "\n"
            "# Fill with median (robust to outliers)\n"
            "df['age'] = df['age'].fillna(df['age'].median())\n"
            "\n"
            "# Fill with mode (for categorical)\n"
            "df['city'] = df['city'].fillna(df['city'].mode()[0])\n"
            "```"
        ),
        common_mistakes=[
            "Dropping too many rows (>20% of data) — you lose information.",
            "Filling numerical columns with mean when outliers are present (use median).",
            "Fitting the imputer on the entire dataset before splitting — data leakage!",
            "Ignoring missing values and hoping the model handles them.",
        ],
        interpretation=(
            "After imputation, verify: re-check isnull().sum() and compare "
            "df.describe() before/after. If the mean/median shifted dramatically, "
            "the imputation may have distorted the distribution."
        ),
        think_about_it=(
            "If 40% of a column is missing, should you impute or drop the "
            "column entirely? What factors influence this decision?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "\n"
            "# Median imputation (numerical)\n"
            "imputer = SimpleImputer(strategy='median')\n"
            "df[num_cols] = imputer.fit_transform(df[num_cols])\n"
            "\n"
            "# Most frequent (categorical)\n"
            "imputer = SimpleImputer(strategy='most_frequent')\n"
            "df[cat_cols] = imputer.fit_transform(df[cat_cols])\n"
            "```"
        ),
        keywords=["missing", "nan", "null", "impute", "dropna", "fillna", "imputer"],
    ),

    # ── Encoding ────────────────────────────────────────────────────

    "categorical_encoding": TopicContent(
        title="Categorical Encoding",
        module="preprocessing",
        what=(
            "Encoding converts text categories into numbers that models "
            "can process. One-hot encoding creates binary columns; "
            "label encoding assigns integers."
        ),
        why=(
            "Algorithms like Logistic Regression, SVM, and KNN require "
            "numerical input. Without encoding, you get errors or "
            "nonsensical results."
        ),
        when=(
            "Whenever you have categorical (string) features. Use one-hot "
            "for nominal categories (no order: red/blue/green). Use label "
            "encoding for ordinal categories (low/medium/high)."
        ),
        example=(
            "```python\n"
            "# One-hot encoding\n"
            "df = pd.get_dummies(df, columns=['color'])\n"
            "# Creates: color_red, color_blue, color_green (0 or 1)\n"
            "\n"
            "# Label encoding\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "le = LabelEncoder()\n"
            "df['size_encoded'] = le.fit_transform(df['size'])\n"
            "# low=0, medium=1, high=2\n"
            "```"
        ),
        common_mistakes=[
            "One-hot encoding the target variable (use LabelEncoder for that).",
            "Label encoding nominal categories (red=1, blue=2 implies red < blue).",
            "Fitting the encoder on test data — always fit on train only!",
            "Creating too many dummy columns (high cardinality → sparse features).",
        ],
        interpretation=(
            "After one-hot encoding, check the new columns. Each should "
            "be 0 or 1. If a category has very few 1s (<1% of rows), it "
            "may be too rare to learn from."
        ),
        think_about_it=(
            "A column 'City' has 500 unique values. One-hot encoding would "
            "create 500 new columns. What alternative approaches exist?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import OneHotEncoder, LabelEncoder\n"
            "\n"
            "# One-hot (for nominal features)\n"
            "encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
            "encoded = encoder.fit_transform(df[['color']])\n"
            "\n"
            "# Label (for ordinal features)\n"
            "le = LabelEncoder()\n"
            "df['size'] = le.fit_transform(df['size'])\n"
            "```"
        ),
        keywords=["encode", "one-hot", "label", "dummy", "categorical", "get_dummies"],
    ),

    # ── Feature Scaling ─────────────────────────────────────────────

    "feature_scaling": TopicContent(
        title="Feature Scaling",
        module="preprocessing",
        what=(
            "Scaling transforms features to a common range. StandardScaler "
            "gives mean=0, std=1. MinMaxScaler squashes to [0, 1]. "
            "RobustScaler uses median and IQR (resistant to outliers)."
        ),
        why=(
            "Distance-based algorithms (KNN, SVM, K-Means) are biased "
            "toward features with larger scales. A feature measured in "
            "thousands (income) will dominate one measured in single digits "
            "(age) without scaling."
        ),
        when=(
            "Always scale for KNN, SVM, Logistic Regression, and K-Means. "
            "Tree-based models (Random Forest, Decision Tree) don't need "
            "scaling. When in doubt, scale."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)  # fit + transform\n"
            "X_test_scaled = scaler.transform(X_test)        # transform only!\n"
            "```"
        ),
        common_mistakes=[
            "Fitting the scaler on test data — this leaks test information into training.",
            "Scaling the target variable (usually not needed for classification).",
            "Scaling features before handling missing values (NaN breaks the scaler).",
            "Using StandardScaler when outliers are extreme (use RobustScaler instead).",
        ],
        interpretation=(
            "After scaling, all features should have mean ≈ 0 and std ≈ 1 "
            "(for StandardScaler). Check with pd.DataFrame(X_scaled).describe()."
        ),
        think_about_it=(
            "Why do tree-based models like Random Forest not need feature "
            "scaling, while KNN does?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\n"
            "\n"
            "scaler = StandardScaler()   # mean=0, std=1\n"
            "scaler = MinMaxScaler()     # range [0, 1]\n"
            "scaler = RobustScaler()     # uses median, robust to outliers\n"
            "\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test = scaler.transform(X_test)  # NEVER fit on test!\n"
            "```"
        ),
        keywords=["scale", "standard", "minmax", "robust", "normalise", "standardise"],
    ),

    # ── Train/Test Split ────────────────────────────────────────────

    "train_test_split": TopicContent(
        title="Train/Test Split",
        module="preprocessing",
        what=(
            "Splitting separates your data into a training set (to teach "
            "the model) and a test set (to evaluate it on unseen data). "
            "Typical split: 80% train, 20% test."
        ),
        why=(
            "If you evaluate a model on data it has already seen, you get "
            "an overly optimistic score. The test set simulates 'future' "
            "unseen data."
        ),
        when=(
            "Before any modelling. Always. For classification, use "
            "stratify=y to preserve class proportions in both sets."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train: {len(X_train)}, Test: {len(X_test)}')\n"
            "```"
        ),
        common_mistakes=[
            "Fitting preprocessing (scaling, imputation) before splitting — data leakage!",
            "Not using stratify for imbalanced classification problems.",
            "Using a test size that's too small (<5%) — unreliable evaluation.",
            "Not setting random_state — results change every run.",
        ],
        interpretation=(
            "The train and test sets should have similar distributions. "
            "Check by comparing y_train.value_counts() and y_test.value_counts(). "
            "Big differences mean the split went wrong."
        ),
        think_about_it=(
            "If you have only 100 samples, is a 80/20 split reliable? "
            "What technique would give more stable estimates?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=0.2,        # 80/20 split\n"
            "    random_state=42,      # reproducible\n"
            "    stratify=y            # preserve class balance\n"
            ")\n"
            "```"
        ),
        keywords=["split", "train", "test", "validation", "stratify", "holdout"],
    ),

    # ── Classification Metrics ──────────────────────────────────────

    "classification_metrics": TopicContent(
        title="Classification Metrics",
        module="classification",
        what=(
            "Metrics that measure how well a classifier predicts categories. "
            "Key metrics: Accuracy (overall correctness), Precision (of "
            "positives predicted, how many are correct), Recall (of actual "
            "positives, how many found), F1 (harmonic mean of both)."
        ),
        why=(
            "Accuracy alone is misleading, especially with imbalanced classes. "
            "A model that always predicts 'No' achieves 99% accuracy on a "
            "dataset with 1% fraud — but catches zero fraud."
        ),
        when=(
            "After training any classifier. Choose the metric based on "
            "the cost of errors: use Precision when false positives are "
            "costly, Recall when false negatives are costly, F1 when "
            "both matter equally."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import classification_report\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "print(classification_report(y_test, y_pred))\n"
            "#              precision  recall  f1-score  support\n"
            "#     No         0.95     0.98     0.96       80\n"
            "#     Yes        0.93     0.85     0.89       20\n"
            "#  accuracy                         0.95      100\n"
            "```"
        ),
        common_mistakes=[
            "Using accuracy on imbalanced datasets — it hides poor minority-class performance.",
            "Confusing precision and recall — Precision = quality of positives, Recall = coverage.",
            "Evaluating on training data instead of test data.",
            "Not looking at per-class metrics — overall F1 can mask a failed class.",
        ],
        interpretation=(
            "Precision 0.93 means 93% of predicted positives are correct. "
            "Recall 0.85 means the model found 85% of all actual positives. "
            "F1 balances both. For imbalanced data, focus on the minority class metrics."
        ),
        think_about_it=(
            "In medical diagnosis, is it worse to miss a disease (low recall) "
            "or falsely diagnose a healthy person (low precision)?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, precision_score, recall_score,\n"
            "    f1_score, classification_report, confusion_matrix\n"
            ")\n"
            "\n"
            "accuracy = accuracy_score(y_test, y_pred)\n"
            "precision = precision_score(y_test, y_pred, average='weighted')\n"
            "recall = recall_score(y_test, y_pred, average='weighted')\n"
            "f1 = f1_score(y_test, y_pred, average='weighted')\n"
            "print(classification_report(y_test, y_pred))\n"
            "```"
        ),
        keywords=["accuracy", "precision", "recall", "f1", "confusion", "classification", "report"],
    ),

    # ── Confusion Matrix ────────────────────────────────────────────

    "confusion_matrix": TopicContent(
        title="Confusion Matrix",
        module="classification",
        what=(
            "A confusion matrix is a table showing True Positives, True "
            "Negatives, False Positives, and False Negatives — the four "
            "possible outcomes of a binary classifier."
        ),
        why=(
            "It reveals *where* the model makes errors. Accuracy tells you "
            "the rate; the confusion matrix tells you the type."
        ),
        when=(
            "After training any classifier. Especially useful for "
            "multiclass problems where per-class errors matter."
        ),
        example=(
            "```\n"
            "              Predicted\n"
            "              No    Yes\n"
            "Actual No  [ 78     2 ]   ← 2 false positives\n"
            "Actual Yes [  3    17 ]   ← 3 false negatives\n"
            "\n"
            "Precision = 17/(17+2) = 89.5%\n"
            "Recall    = 17/(17+3) = 85.0%\n"
            "```"
        ),
        common_mistakes=[
            "Reading the matrix rows/columns backwards — check axis labels.",
            "Ignoring the matrix when accuracy looks good — there may be class-specific failures.",
            "Not normalising for imbalanced classes (use normalize='true').",
        ],
        interpretation=(
            "Diagonal = correct predictions. Off-diagonal = errors. "
            "High values on the diagonal and near-zero elsewhere = good model. "
            "Look for which classes are confused with each other."
        ),
        think_about_it=(
            "If your confusion matrix shows 0 false negatives but many "
            "false positives, what kind of problem might this model be "
            "suited for?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import confusion_matrix\n"
            "import seaborn as sns\n"
            "\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n"
            "```"
        ),
        keywords=["confusion", "matrix", "tp", "tn", "fp", "fn", "true positive"],
    ),

    # ── ROC Curve ───────────────────────────────────────────────────

    "roc_curve": TopicContent(
        title="ROC Curve and AUC",
        module="classification",
        what=(
            "The ROC curve plots True Positive Rate vs False Positive Rate "
            "at different classification thresholds. AUC (Area Under the "
            "Curve) summarises the curve into a single number: 1.0 = "
            "perfect, 0.5 = random guessing."
        ),
        why=(
            "ROC-AUC measures how well the model separates classes across "
            "all thresholds, not just the default 0.5. It's threshold-independent."
        ),
        when=(
            "When you need to choose a threshold (e.g., spam filter where "
            "you can tune sensitivity). Less useful when classes are "
            "very imbalanced — use Precision-Recall AUC instead."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "fpr, tpr, thresholds = roc_curve(y_test, y_prob)\n"
            "roc_auc = auc(fpr, tpr)\n"
            "print(f'AUC: {roc_auc:.3f}')  # 0.98 = excellent\n"
            "```"
        ),
        common_mistakes=[
            "Using ROC-AUC on highly imbalanced data — it can be misleadingly high.",
            "Forgetting to use predict_proba() instead of predict() for probabilities.",
            "Interpreting AUC as accuracy — AUC is about ranking, not classification.",
        ],
        interpretation=(
            "AUC = 0.9+ is excellent, 0.8–0.9 is good, 0.7–0.8 is fair, "
            "<0.7 is poor. A curve hugging the top-left corner is best. "
            "The diagonal line represents random guessing (AUC = 0.5)."
        ),
        think_about_it=(
            "Can two models have the same AUC but very different "
            "performance at a specific threshold? Why might that matter?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "fpr, tpr, _ = roc_curve(y_test, y_prob)\n"
            "plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')\n"
            "plt.plot([0,1], [0,1], 'k--')  # random baseline\n"
            "plt.xlabel('False Positive Rate')\n"
            "plt.ylabel('True Positive Rate')\n"
            "```"
        ),
        keywords=["roc", "auc", "curve", "threshold", "probability", "tpr", "fpr"],
    ),

    # ── Regression Metrics ──────────────────────────────────────────

    "regression_metrics": TopicContent(
        title="Regression Metrics",
        module="regression",
        what=(
            "Metrics for evaluating regression models: R² (proportion of "
            "variance explained), MAE (average absolute error), MSE (mean "
            "squared error), RMSE (root MSE, in original units)."
        ),
        why=(
            "R² tells you how much of the target's variation the model "
            "captures. MAE/RMSE tell you the typical prediction error in "
            "the target's units."
        ),
        when=(
            "After training any regressor. R² for overall fit quality. "
            "MAE for interpretable average error. RMSE penalises large "
            "errors more heavily."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import r2_score, mean_absolute_error\n"
            "import numpy as np\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "print(f'R²:   {r2_score(y_test, y_pred):.4f}')\n"
            "print(f'MAE:  {mean_absolute_error(y_test, y_pred):.4f}')\n"
            "print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n"
            "```"
        ),
        common_mistakes=[
            "Only looking at R² — a high R² doesn't mean predictions are useful.",
            "Comparing MAE across datasets with different scales.",
            "Ignoring residual plots — they reveal systematic errors.",
            "Evaluating on training data — always use the test set.",
        ],
        interpretation=(
            "R² = 0.85 means the model explains 85% of variance. "
            "MAE = 500 means predictions are off by $500 on average. "
            "RMSE > MAE suggests some large outlier errors."
        ),
        think_about_it=(
            "If RMSE is much larger than MAE, what does that tell you "
            "about the distribution of errors?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n"
            "import numpy as np\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "print(f'R²:   {r2_score(y_test, y_pred):.4f}')\n"
            "print(f'MAE:  {mean_absolute_error(y_test, y_pred):.4f}')\n"
            "print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n"
            "```"
        ),
        keywords=["r2", "mae", "mse", "rmse", "regression", "error", "residual"],
    ),

    # ── Overfitting & Underfitting ──────────────────────────────────

    "overfitting": TopicContent(
        title="Overfitting and Underfitting",
        module="model_evaluation",
        what=(
            "Overfitting: the model memorises training data (low train "
            "error, high test error). Underfitting: the model is too "
            "simple to capture patterns (high error on both)."
        ),
        why=(
            "The goal is a model that generalises — performs well on "
            "unseen data. Overfitting and underfitting both fail at this."
        ),
        when=(
            "Always check. Compare training score vs test score. A large "
            "gap signals overfitting. Low scores on both signal underfitting."
        ),
        example=(
            "```\n"
            "Model            Train Accuracy    Test Accuracy    Diagnosis\n"
            "Decision Tree         1.00              0.82        Overfitting\n"
            "Linear Reg            0.75              0.74        Underfitting\n"
            "Random Forest         0.96              0.94        Good fit ✓\n"
            "```"
        ),
        common_mistakes=[
            "Only reporting test accuracy without comparing to training accuracy.",
            "Using a very deep decision tree — it will overfit almost always.",
            "Not using cross-validation — a single split can be misleading.",
            "Adding more features without checking if they're informative.",
        ],
        interpretation=(
            "Train ≫ Test → Overfit (reduce complexity, add regularization, "
            "get more data). Train ≈ Test but both low → Underfit (add features, "
            "use a more complex model). Train ≈ Test and both high → Good fit."
        ),
        think_about_it=(
            "You have a model with 99% training accuracy and 72% test "
            "accuracy. Name three things you could try to fix this."
        ),
        code_link=(
            "```python\n"
            "# Compare train vs test\n"
            "train_acc = model.score(X_train, y_train)\n"
            "test_acc = model.score(X_test, y_test)\n"
            "print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')\n"
            "\n"
            "# Cross-validation for more stable estimate\n"
            "from sklearn.model_selection import cross_val_score\n"
            "scores = cross_val_score(model, X, y, cv=5)\n"
            "print(f'CV: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["overfit", "underfit", "bias", "variance", "generalise", "cross-validation"],
    ),

    # ── Cross-Validation ────────────────────────────────────────────

    "cross_validation": TopicContent(
        title="Cross-Validation",
        module="model_evaluation",
        what=(
            "Cross-validation splits data into k folds, trains on k-1, "
            "tests on 1, and repeats k times. Each fold serves as the "
            "test set once."
        ),
        why=(
            "A single train/test split can be lucky or unlucky. "
            "Cross-validation gives a more reliable estimate of model "
            "performance by using every data point for both training "
            "and testing."
        ),
        when=(
            "When you have limited data, or want a robust performance "
            "estimate. Always use CV for model selection and comparison."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# 5-fold cross-validation\n"
            "scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n"
            "print(f'Scores: {scores}')\n"
            "print(f'Mean: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        common_mistakes=[
            "Using cross-validation on time-series data without respecting temporal order.",
            "Reporting only the mean without the standard deviation.",
            "Using too few folds (2-3) — high variance in estimates.",
            "Fitting preprocessing inside CV — must use Pipeline to prevent leakage.",
        ],
        interpretation=(
            "Mean CV score ≈ expected test performance. Std shows stability. "
            "High std means the model is sensitive to which data it trains on. "
            "Use this to compare models fairly."
        ),
        think_about_it=(
            "If Model A gets CV accuracy 0.92 ± 0.03 and Model B gets "
            "0.91 ± 0.01, which model would you prefer and why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, KFold\n"
            "\n"
            "# Simple 5-fold\n"
            "scores = cross_val_score(pipeline, X, y, cv=5)\n"
            "\n"
            "# With custom CV splitter\n"
            "kf = KFold(n_splits=10, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(pipeline, X, y, cv=kf)\n"
            "```"
        ),
        keywords=["cross-validation", "cv", "fold", "k-fold", "generalise"],
    ),

    # ── Clustering ──────────────────────────────────────────────────

    "clustering_basics": TopicContent(
        title="Clustering Fundamentals",
        module="clustering",
        what=(
            "Clustering groups similar data points together without "
            "labels. Unlike classification (supervised), clustering "
            "discovers structure on its own (unsupervised)."
        ),
        why=(
            "Many real-world problems have no labels: customer segmentation, "
            "anomaly detection, document grouping. Clustering finds patterns "
            "you didn't know existed."
        ),
        when=(
            "When you don't have labels, or want to discover hidden structure. "
            "Use K-Means for spherical clusters, DBSCAN for arbitrary shapes, "
            "Agglomerative for hierarchical relationships."
        ),
        example=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "df['cluster'] = labels\n"
            "```"
        ),
        common_mistakes=[
            "Not scaling features before clustering — distance-based algorithms are scale-sensitive.",
            "Assuming the number of k in K-Means is obvious — use elbow/silhouette methods.",
            "Interpreting clusters as 'classes' — clusters have no inherent meaning.",
            "Using clustering evaluation metrics on the full dataset including noise.",
        ],
        interpretation=(
            "Silhouette score ranges from -1 to 1. Close to 1 = well-defined "
            "clusters. Near 0 = overlapping clusters. Negative = possibly "
            "wrong assignment. Always visualise clusters in 2D (via PCA)."
        ),
        think_about_it=(
            "K-Means found 5 clusters in your data. Does that mean there "
            "are 5 'real' groups? What else could explain this?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.metrics import silhouette_score\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "\n"
            "kmeans = KMeans(n_clusters=3, random_state=42)\n"
            "labels = kmeans.fit_predict(X_scaled)\n"
            "print(f'Silhouette: {silhouette_score(X_scaled, labels):.3f}')\n"
            "```"
        ),
        keywords=["clustering", "k-means", "dbscan", "hierarchical", "unsupervised", "cluster"],
    ),

    # ── Feature Engineering ─────────────────────────────────────────

    "feature_engineering_basics": TopicContent(
        title="Feature Engineering",
        module="feature_engineering",
        what=(
            "Feature engineering creates new input variables from existing "
            "data to improve model performance. It includes mathematical "
            "transforms, binning, interactions, and dimensionality reduction."
        ),
        why=(
            "Better features often matter more than a better algorithm. "
            "A well-engineered feature can make a simple model outperform "
            "a complex one."
        ),
        when=(
            "After preprocessing, before modelling. Try: log transforms "
            "for skewed data, binning for continuous variables, interactions "
            "to capture combined effects, PCA to reduce dimensions."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "# Log transform for skewed data\n"
            "df['log_income'] = np.log1p(df['income'])\n"
            "\n"
            "# Interaction feature\n"
            "df['price_per_sqft'] = df['price'] / df['sqft']\n"
            "\n"
            "# Binning\n"
            "df['age_group'] = pd.cut(df['age'], bins=5, labels=False)\n"
            "```"
        ),
        common_mistakes=[
            "Creating features from test data — always engineer on train only.",
            "Adding polynomial features without checking for overfitting.",
            "Not removing redundant/constant features after engineering.",
            "Engineering features that leak the target (e.g., using future data).",
        ],
        interpretation=(
            "After engineering, check correlation with the target. New "
            "features should be more informative than the originals. "
            "Use feature importance to verify they actually help."
        ),
        think_about_it=(
            "You have 'date_of_birth' and the current date. What new "
            "features could you create, and which might be most useful "
            "for predicting income?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "\n"
            "# Log transform\n"
            "df['log_feature'] = np.log1p(df['skewed_feature'])\n"
            "\n"
            "# Polynomial features\n"
            "poly = PolynomialFeatures(degree=2, interaction_only=True)\n"
            "X_poly = poly.fit_transform(X)\n"
            "\n"
            "# Feature selection\n"
            "from sklearn.feature_selection import VarianceThreshold\n"
            "selector = VarianceThreshold(threshold=0.01)\n"
            "X_selected = selector.fit_transform(X)\n"
            "```"
        ),
        keywords=["feature", "engineering", "transform", "polynomial", "interaction", "binning"],
    ),

    # ── Model Selection ─────────────────────────────────────────────

    "model_selection": TopicContent(
        title="Choosing the Right Model",
        module="model_comparison",
        what=(
            "Model selection is the process of comparing multiple algorithms "
            "on the same data to find the best fit. No single model wins "
            "everywhere — this is the 'No Free Lunch' theorem."
        ),
        why=(
            "Different models have different strengths. Linear models are "
            "fast and interpretable. Trees capture non-linear patterns. "
            "Ensembles combine multiple models for better performance."
        ),
        when=(
            "After preprocessing. Start simple (Logistic/Linear Regression), "
            "then try tree-based (Random Forest, Gradient Boosting), "
            "then compare."
        ),
        example=(
            "```\n"
            "Model                Accuracy    Time    Interpretable\n"
            "Logistic Regression    0.89       0.1s      Yes\n"
            "Random Forest          0.94       0.8s      Somewhat\n"
            "Gradient Boosting      0.95       1.2s      No\n"
            "\n"
            "→ Gradient Boosting wins on accuracy, but RF is faster\n"
            "  and still interpretable. Choose based on your needs.\n"
            "```"
        ),
        common_mistakes=[
            "Picking the 'best' model without considering deployment constraints.",
            "Not using the same preprocessing for all models — unfair comparison.",
            "Training on the full dataset instead of train/test split.",
            "Ignoring model interpretability when it matters (e.g., healthcare).",
        ],
        interpretation=(
            "Look at multiple metrics, not just one. Consider: accuracy, "
            "training time, interpretability, and maintenance cost. "
            "The 'best' model depends on your specific requirements."
        ),
        think_about_it=(
            "Gradient Boosting has 0.96 accuracy and Random Forest has "
            "0.95 accuracy. When might you still choose Random Forest?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "\n"
            "models = {\n"
            "    'LR': LogisticRegression(max_iter=1000),\n"
            "    'RF': RandomForestClassifier(n_estimators=100),\n"
            "    'GB': GradientBoostingClassifier(n_estimators=100),\n"
            "}\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "    print(f'{name}: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["model", "selection", "compare", "algorithm", "choose", "baseline"],
    ),
}


# ── Public API ──────────────────────────────────────────────────────

def get_topic(topic_key: str) -> TopicContent | None:
    """Return educational content for a topic, or None if not found."""
    return TOPICS.get(topic_key)


def get_topics_by_module(module: str) -> list[TopicContent]:
    """Return all topics belonging to a module, in order."""
    return [t for t in TOPICS.values() if t.module == module]


def list_topic_keys() -> list[str]:
    """Return all available topic keys."""
    return list(TOPICS.keys())


def render_topic(topic_key: str, expanded: bool = True) -> None:
    """
    Render a single topic's educational content using Streamlit widgets.

    Call this from any module page to display the full educational
    content block (What → Why → When → Example → Mistakes →
    Interpretation → Think About It → Code Link).
    """
    import streamlit as st

    topic = TOPICS.get(topic_key)
    if topic is None:
        return

    with st.expander(f"📚 {topic.title}", expanded=expanded):
        st.markdown(f"### What is it?")
        st.markdown(topic.what)

        st.markdown(f"### Why is it important?")
        st.markdown(topic.why)

        st.markdown(f"### When should it be used?")
        st.markdown(topic.when)

        st.markdown(f"### Simple example")
        st.markdown(topic.example)

        st.markdown(f"### Common mistakes")
        for mistake in topic.common_mistakes:
            st.markdown(f"- ❌ {mistake}")

        st.markdown(f"### How to interpret results")
        st.markdown(topic.interpretation)

        st.markdown(f"### 💭 Think About It")
        st.info(topic.think_about_it)

        st.markdown(f"### 🔗 Underlying Python/sklearn code")
        st.code(topic.code_link, language="python")
