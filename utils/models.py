"""
Model registry for Data Science Lab.

Defines every supported classifier with its sklearn class, default
hyperparameters, and educational metadata (description, advantages,
limitations, when-to-use).  The Classification page reads from this
registry to populate widgets and show explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelInfo:
    """Metadata for a single classification algorithm."""

    name: str
    key: str
    sklearn_class: type
    default_params: dict = field(default_factory=dict)
    description: str = ""
    why_use: str = ""
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    when_to_use: str = ""
    important_params: dict[str, str] = field(default_factory=dict)
    has_feature_importance: bool = False


# ── Registry ────────────────────────────────────────────────────────

CLASSIFIERS: dict[str, ModelInfo] = {
    "logistic_regression": ModelInfo(
        name="Logistic Regression",
        key="logistic_regression",
        sklearn_class=LogisticRegression,
        default_params={"max_iter": 1000, "random_state": 42},
        description=(
            "A linear model that estimates probabilities using the logistic "
            "(sigmoid) function. Despite its name, it is used for classification."
        ),
        why_use=(
            "It is the simplest probabilistic classifier — fast to train, easy "
            "to interpret, and a strong baseline for binary and multiclass tasks."
        ),
        advantages=[
            "Fast training and prediction",
            "Outputs calibrated probabilities",
            "Coefficients are interpretable (feature importance)",
            "Works well when classes are linearly separable",
            "Low risk of overfitting with regularisation (C parameter)",
        ],
        limitations=[
            "Assumes a linear decision boundary",
            "Cannot capture complex non-linear relationships",
            "Sensitive to outliers and correlated features",
            "Requires features to be on similar scales",
        ],
        when_to_use=(
            "Start here as a baseline. Use when you need interpretable results, "
            "calibrated probabilities, or a quick prototype. Move to tree-based "
            "models if the relationship is clearly non-linear."
        ),
        important_params={
            "C": "Inverse regularisation strength (smaller = stronger regularisation)",
            "max_iter": "Maximum iterations for the solver to converge",
            "solver": "Optimisation algorithm (lbfgs, liblinear, saga)",
        },
    ),
    "knn": ModelInfo(
        name="K-Nearest Neighbors",
        key="knn",
        sklearn_class=KNeighborsClassifier,
        default_params={"n_neighbors": 5},
        description=(
            "A non-parametric algorithm that classifies a point based on the "
            "majority class among its k closest training examples."
        ),
        why_use=(
            "It requires no training phase and naturally handles multi-class "
            "problems. Good for understanding the 'lazy learning' paradigm."
        ),
        advantages=[
            "No training phase — predictions are fast for small datasets",
            "Naturally handles multi-class problems",
            "Non-parametric — makes no assumptions about data distribution",
            "Intuitive and easy to understand",
        ],
        limitations=[
            "Slow prediction on large datasets (computes distances to all points)",
            "Sensitive to irrelevant features and feature scaling",
            "Struggles with high-dimensional data (curse of dimensionality)",
            "Performance degrades with imbalanced classes",
        ],
        when_to_use=(
            "Use for small-to-medium datasets where you want a non-parametric "
            "baseline. Always scale features first. Tune k to balance "
            "bias-variance."
        ),
        important_params={
            "n_neighbors": "Number of neighbors to consider (odd values avoid ties)",
            "weights": "'uniform' (equal) or 'distance' (closer = more weight)",
            "metric": "Distance function (euclidean, manhattan, minkowski)",
        },
    ),
    "decision_tree": ModelInfo(
        name="Decision Tree",
        key="decision_tree",
        sklearn_class=DecisionTreeClassifier,
        default_params={"random_state": 42},
        description=(
            "A tree-structured model that splits data on feature thresholds, "
            "creating an interpretable flowchart of if-else decisions."
        ),
        why_use=(
            "Decision trees are highly interpretable and require no feature "
            "scaling. They are the building blocks of ensemble methods like "
            "Random Forest and Gradient Boosting."
        ),
        advantages=[
            "Highly interpretable — visualise the decision process",
            "No feature scaling required",
            "Handles both numerical and categorical features",
            "Captures non-linear relationships and interactions",
            "Fast to train on small-to-medium datasets",
        ],
        limitations=[
            "Prone to overfitting (low bias, high variance)",
            "Unstable — small data changes can produce very different trees",
            "Biased toward features with more levels",
            "Greedy optimisation does not guarantee a globally optimal tree",
        ],
        when_to_use=(
            "Use when interpretability is critical (e.g. explaining decisions "
            "to stakeholders). For better generalisation, use Random Forest or "
            "Gradient Boosting instead."
        ),
        important_params={
            "max_depth": "Maximum tree depth (controls overfitting)",
            "min_samples_split": "Minimum samples required to split a node",
            "min_samples_leaf": "Minimum samples in a leaf node",
            "criterion": "Split quality measure (gini, entropy, log_loss)",
        },
        has_feature_importance=True,
    ),
    "random_forest": ModelInfo(
        name="Random Forest",
        key="random_forest",
        sklearn_class=RandomForestClassifier,
        default_params={"n_estimators": 100, "random_state": 42},
        description=(
            "An ensemble of decision trees trained on random subsets of data "
            "and features, aggregated by majority vote."
        ),
        why_use=(
            "Combines many weak learners (trees) into a strong learner. "
            "Reduces overfitting while maintaining high accuracy."
        ),
        advantages=[
            "High accuracy with minimal tuning",
            "Robust to overfitting (averages many trees)",
            "Handles high-dimensional data well",
            "Provides feature importance rankings",
            "Works with missing values and mixed data types",
        ],
        limitations=[
            "Less interpretable than a single tree",
            "Slower to train and predict than linear models",
            "Can overfit on very noisy datasets",
            "Memory-intensive for very large forests",
        ],
        when_to_use=(
            "A strong default choice for tabular data. Use when you need "
            "good accuracy without extensive hyperparameter tuning. Check "
            "feature importance to understand what drives predictions."
        ),
        important_params={
            "n_estimators": "Number of trees (more = better but slower)",
            "max_depth": "Maximum depth of each tree",
            "min_samples_split": "Minimum samples to split a node",
            "max_features": "Number of features per split ('sqrt' is common)",
        },
        has_feature_importance=True,
    ),
    "naive_bayes": ModelInfo(
        name="Naive Bayes (Gaussian)",
        key="naive_bayes",
        sklearn_class=GaussianNB,
        default_params={},
        description=(
            "A probabilistic classifier based on Bayes' theorem with the "
            "'naive' assumption that features are conditionally independent "
            "given the class label."
        ),
        why_use=(
            "Extremely fast, works well with limited training data, and is "
            "a classic baseline for text classification and spam detection."
        ),
        advantages=[
            "Very fast training and prediction",
            "Works well with small datasets",
            "Handles multi-class natively",
            "Good baseline for text classification",
            "No hyperparameter tuning needed",
        ],
        limitations=[
            "Independence assumption is rarely true in practice",
            "Can be outperformed by more complex models",
            "Struggles with correlated features",
            "Gaussian assumption may not fit all numerical distributions",
        ],
        when_to_use=(
            "Use as a quick baseline, especially for text classification "
            "(with TF-IDF) or when training data is very limited. Compare "
            "against more complex models."
        ),
        important_params={
            "var_smoothing": "Portion of largest variance to add for stability",
        },
    ),
    "svm": ModelInfo(
        name="Support Vector Machine",
        key="svm",
        sklearn_class=SVC,
        default_params={"random_state": 42},
        description=(
            "Finds the optimal hyperplane that maximises the margin between "
            "classes. Uses the kernel trick to handle non-linear boundaries."
        ),
        why_use=(
            "SVMs are effective in high-dimensional spaces and when the number "
            "of features exceeds the number of samples. The kernel trick "
            "captures complex boundaries."
        ),
        advantages=[
            "Effective in high-dimensional spaces",
            "Memory-efficient (uses support vectors only)",
            "Versatile kernel options (linear, RBF, polynomial)",
            "Strong theoretical foundation",
        ],
        limitations=[
            "Slow on large datasets (O(n²) to O(n³) training time)",
            "Does not natively output probabilities",
            "Sensitive to feature scaling",
            "Hard to interpret — black-box model",
            "Kernel and C/gamma tuning is critical",
        ],
        when_to_use=(
            "Use for small-to-medium datasets with many features. Always "
            "scale features. Tune C and gamma carefully. Consider LinearSVC "
            "for large datasets."
        ),
        important_params={
            "C": "Regularisation parameter (smaller = wider margin)",
            "kernel": "Kernel type (linear, rbf, poly, sigmoid)",
            "gamma": "Kernel coefficient ('scale' or 'auto')",
        },
    ),
    "gradient_boosting": ModelInfo(
        name="Gradient Boosting",
        key="gradient_boosting",
        sklearn_class=GradientBoostingClassifier,
        default_params={"n_estimators": 100, "random_state": 42},
        description=(
            "An ensemble that sequentially builds trees, each correcting "
            "the errors of the previous one by fitting to the gradient "
            "of the loss function."
        ),
        why_use=(
            "Often achieves the highest accuracy on tabular data. Each new "
            "tree focuses on the hard-to-classify examples."
        ),
        advantages=[
            "Often top-performing on tabular data",
            "Handles missing values and mixed types",
            "Sequential learning captures complex patterns",
            "Provides feature importance",
            "Flexible loss functions",
        ],
        limitations=[
            "Slower to train than Random Forest (sequential nature)",
            "More prone to overfitting without careful tuning",
            "Sensitive to noisy data and outliers",
            "Many hyperparameters to tune",
        ],
        when_to_use=(
            "Use when you need the best possible accuracy on tabular data. "
            "Tune learning_rate, n_estimators, and max_depth carefully. "
            "Consider XGBoost or LightGBM for production use."
        ),
        important_params={
            "n_estimators": "Number of boosting rounds",
            "learning_rate": "Shrinkage — lower = more robust but slower",
            "max_depth": "Depth of each tree (usually shallow, 3-8)",
            "subsample": "Fraction of data per tree (adds randomness)",
        },
        has_feature_importance=True,
    ),
}


def get_classifier(key: str) -> ModelInfo:
    """Return ModelInfo by key, raising KeyError if not found."""
    return CLASSIFIERS[key]


def get_classifier_names() -> list[str]:
    """Return display names in registry order."""
    return [info.name for info in CLASSIFIERS.values()]


def key_from_name(name: str) -> str:
    """Map a display name back to its registry key."""
    for info in CLASSIFIERS.values():
        if info.name == name:
            return info.key
    raise ValueError(f"Unknown model name: {name}")
