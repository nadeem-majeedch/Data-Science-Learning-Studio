"""
Regression model registry for Data-Science-Learning-Studio.

Defines every supported regressor with its sklearn class, default
hyperparameters, and educational metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class RegressorInfo:
    """Metadata for a single regression algorithm."""

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


REGRESSORS: dict[str, RegressorInfo] = {
    "linear": RegressorInfo(
        name="Linear Regression",
        key="linear",
        sklearn_class=LinearRegression,
        default_params={},
        description=(
            "Fits a linear relationship between features and the target by "
            "minimising the sum of squared residuals."
        ),
        why_use=(
            "The simplest and most interpretable regression model. Great "
            "baseline — coefficients directly show the effect of each feature."
        ),
        advantages=[
            "Fast to train and predict",
            "Highly interpretable — coefficients show feature effects",
            "No hyperparameters to tune",
            "Works well when relationships are truly linear",
        ],
        limitations=[
            "Assumes a linear relationship between features and target",
            "Sensitive to outliers",
            "Cannot capture non-linear patterns",
            "Performs poorly with multicollinear features",
        ],
        when_to_use=(
            "Start here as a baseline. If R² is high, the relationship is "
            "likely linear and you may not need a complex model."
        ),
        important_params={
            "fit_intercept": "Whether to fit an intercept (default True)",
        },
        has_feature_importance=True,
    ),
    "ridge": RegressorInfo(
        name="Ridge Regression",
        key="ridge",
        sklearn_class=Ridge,
        default_params={"alpha": 1.0},
        description=(
            "Linear regression with L2 regularisation that penalises large "
            "coefficients, preventing overfitting."
        ),
        why_use=(
            "Adds regularisation to linear regression — shrinks coefficients "
            "towards zero without removing features. Handles multicollinearity."
        ),
        advantages=[
            "Reduces overfitting compared to plain linear regression",
            "Handles multicollinear features gracefully",
            "All features are retained (no feature elimination)",
            "Stable with many correlated predictors",
        ],
        limitations=[
            "Still assumes a linear relationship",
            "Does not perform feature selection (all coefficients > 0)",
            "Requires tuning of alpha (regularisation strength)",
        ],
        when_to_use=(
            "Use when linear regression overfits or when features are "
            "highly correlated. Tune alpha with cross-validation."
        ),
        important_params={
            "alpha": "Regularisation strength (larger = more shrinkage)",
        },
        has_feature_importance=True,
    ),
    "lasso": RegressorInfo(
        name="Lasso Regression",
        key="lasso",
        sklearn_class=Lasso,
        default_params={"alpha": 1.0, "max_iter": 10000},
        description=(
            "Linear regression with L1 regularisation that can shrink some "
            "coefficients exactly to zero, performing automatic feature selection."
        ),
        why_use=(
            "Combines regression with feature selection — drives irrelevant "
            "feature weights to zero. Useful for high-dimensional data."
        ),
        advantages=[
            "Automatic feature selection (some coefficients become zero)",
            "Prevents overfitting",
            "Produces sparse, interpretable models",
            "Handles high-dimensional data well",
        ],
        limitations=[
            "Can be unstable when features are highly correlated",
            "Tends to select only one feature from a correlated group",
            "May over-regularise if alpha is too high",
        ],
        when_to_use=(
            "Use when you suspect many features are irrelevant and want "
            "automatic feature selection. Tune alpha carefully."
        ),
        important_params={
            "alpha": "Regularisation strength (larger = more features dropped)",
            "max_iter": "Maximum iterations for convergence",
        },
        has_feature_importance=True,
    ),
    "decision_tree_reg": RegressorInfo(
        name="Decision Tree Regressor",
        key="decision_tree_reg",
        sklearn_class=DecisionTreeRegressor,
        default_params={"random_state": 42},
        description=(
            "Splits data into regions with constant predictions using "
            "recursive binary splits on feature thresholds."
        ),
        why_use=(
            "Highly interpretable, captures non-linear relationships, "
            "and requires no feature scaling."
        ),
        advantages=[
            "Easy to visualise and interpret",
            "Captures non-linear relationships",
            "No feature scaling required",
            "Fast training",
        ],
        limitations=[
            "Prone to overfitting",
            "Unstable — small changes cause different trees",
            "Creates axis-aligned splits only",
            "Greedy algorithm — no global optimum guarantee",
        ],
        when_to_use=(
            "Use for interpretability. For better generalisation, prefer "
            "Random Forest or Gradient Boosting."
        ),
        important_params={
            "max_depth": "Maximum tree depth (controls overfitting)",
            "min_samples_split": "Minimum samples to split a node",
        },
        has_feature_importance=True,
    ),
    "random_forest_reg": RegressorInfo(
        name="Random Forest Regressor",
        key="random_forest_reg",
        sklearn_class=RandomForestRegressor,
        default_params={"n_estimators": 100, "random_state": 42},
        description=(
            "Ensemble of decision trees trained on random data subsets, "
            "predictions averaged for variance reduction."
        ),
        why_use=(
            "Strong default — high accuracy with minimal tuning. Averaging "
            "many trees reduces overfitting of individual trees."
        ),
        advantages=[
            "High accuracy out of the box",
            "Robust to overfitting",
            "Provides feature importance",
            "Handles missing values and mixed types",
            "Low hyperparameter sensitivity",
        ],
        limitations=[
            "Less interpretable than a single tree",
            "Slower than linear models",
            "Can overfit on very noisy data",
            "Memory-intensive for large forests",
        ],
        when_to_use=(
            "A strong default for tabular regression. Check feature "
            "importance to understand predictions."
        ),
        important_params={
            "n_estimators": "Number of trees",
            "max_depth": "Maximum depth per tree",
            "max_features": "Features per split ('sqrt' or 1.0)",
        },
        has_feature_importance=True,
    ),
    "gradient_boosting_reg": RegressorInfo(
        name="Gradient Boosting Regressor",
        key="gradient_boosting_reg",
        sklearn_class=GradientBoostingRegressor,
        default_params={"n_estimators": 100, "random_state": 42},
        description=(
            "Sequentially builds trees, each correcting the residual errors "
            "of the previous ensemble via gradient descent on the loss."
        ),
        why_use=(
            "Often achieves the highest accuracy on tabular data by "
            "focusing training on the hardest-to-predict examples."
        ),
        advantages=[
            "Often top-performing on tabular data",
            "Handles missing values and mixed types",
            "Provides feature importance",
            "Flexible loss functions",
        ],
        limitations=[
            "Slower than Random Forest (sequential)",
            "More prone to overfitting without tuning",
            "Many hyperparameters to tune",
            "Sensitive to noisy data",
        ],
        when_to_use=(
            "Use when you need maximum accuracy. Tune learning_rate and "
            "n_estimators carefully. Consider XGBoost/LightGBM for production."
        ),
        important_params={
            "n_estimators": "Number of boosting rounds",
            "learning_rate": "Shrinkage — lower = more robust",
            "max_depth": "Depth of each tree (usually 3–8)",
            "subsample": "Fraction of data per tree",
        },
        has_feature_importance=True,
    ),
    "knn_reg": RegressorInfo(
        name="KNN Regressor",
        key="knn_reg",
        sklearn_class=KNeighborsRegressor,
        default_params={"n_neighbors": 5},
        description=(
            "Predicts the target as the average of the k nearest training "
            "examples, based on feature distance."
        ),
        why_use=(
            "Non-parametric baseline — makes no assumptions about the "
            "underlying data distribution. Simple and intuitive."
        ),
        advantages=[
            "No training phase",
            "Non-parametric — flexible",
            "Easy to understand",
        ],
        limitations=[
            "Slow prediction on large datasets",
            "Sensitive to irrelevant features and scaling",
            "Struggles with high dimensions",
            "Performance degrades with uneven density",
        ],
        when_to_use=(
            "Use for small-to-medium datasets as a non-parametric baseline. "
            "Always scale features first."
        ),
        important_params={
            "n_neighbors": "Number of neighbors (higher = smoother predictions)",
            "weights": "'uniform' or 'distance' (closer = more weight)",
        },
    ),
}


def get_regressor(key: str) -> RegressorInfo:
    return REGRESSORS[key]


def get_regressor_names() -> list[str]:
    return [info.name for info in REGRESSORS.values()]


def key_from_name(name: str) -> str:
    for info in REGRESSORS.values():
        if info.name == name:
            return info.key
    raise ValueError(f"Unknown regressor name: {name}")
