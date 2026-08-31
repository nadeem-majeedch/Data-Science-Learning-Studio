"""
Model Selection — learning topics for the Model Selection / Model Comparison module.
"""
from utils.education.base import T

TOPICS = {
    "what_is_model_selection": T(
        title="What Is Model Selection",
        module="model_selection",
        what=(
            "Model selection is the process of choosing the best algorithm and "
            "hyperparameters for a given dataset and problem. It balances accuracy, "
            "interpretability, training time, and generalisation."
        ),
        why=(
            "No single algorithm works best on every dataset (the 'No Free Lunch' "
            "theorem). Systematic model selection ensures you pick a model that fits "
            "your data, your constraints, and your goals — not just the one you "
            "happened to learn first."
        ),
        when=(
            "After preprocessing and feature engineering, before committing to a "
            "final model. You should revisit model selection whenever your data or "
            "requirements change."
        ),
        example=(
            "On a small tabular dataset with 500 rows, a Decision Tree may "
            "outperform a deep neural network because it overfits less with "
            "limited data and is easier to interpret."
        ),
        mistakes=[
            "Jumping to a complex model without trying a simple baseline first.",
            "Selecting a model based on training accuracy alone.",
            "Ignoring interpretability requirements in regulated domains.",
        ],
        interpretation=(
            "The goal is not to find the 'best' model universally, but the "
            "best model for your specific data, problem, and constraints."
        ),
        think_about_it=(
            "If two models have the same accuracy but one is a Decision Tree "
            "and the other is a Gradient Boosting ensemble, which would you "
            "choose and why?"
        ),
        code_link=(
            "# Compare models side by side\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "models = {\n"
            "    'LogisticRegression': LogisticRegression(max_iter=1000),\n"
            "    'RandomForest': RandomForestClassifier(n_estimators=100),\n"
            "}\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    y_pred = model.predict(X_test)\n"
            "    print(f'{name}: {accuracy_score(y_test, y_pred):.4f}')"
        ),
        keywords=["model selection", "algorithm choice", "no free lunch"],
    ),
    "baseline_model": T(
        title="Baseline Model",
        module="model_selection",
        what=(
            "A baseline model is the simplest possible model used as a reference "
            "point. It sets the minimum performance that any real model must beat."
        ),
        why=(
            "Without a baseline you cannot tell whether a complex model is actually "
            "adding value. A baseline for classification might always predict the "
            "majority class; for regression, it might predict the mean."
        ),
        when=(
            "Always start here before trying sophisticated algorithms. If your "
            "complex model barely beats the baseline, you may be overcomplicating "
            "things."
        ),
        example=(
            "For a dataset where 90 % of customers do not churn, a model that "
            "always predicts 'no churn' achieves 90 % accuracy. A real model "
            "must beat 90 % to be useful."
        ),
        mistakes=[
            "Skipping the baseline and going straight to XGBoost.",
            "Using a baseline that is too strong, masking real improvements.",
            "Forgetting that accuracy can be misleading with imbalanced classes.",
        ],
        interpretation=(
            "If your trained model's accuracy is close to the baseline's accuracy, "
            "the model has not learned meaningful patterns."
        ),
        think_about_it=(
            "What baseline would you choose for a regression problem where the "
            "target ranges from $50k to $500k?"
        ),
        code_link=(
            "import numpy as np\n"
            "from sklearn.metrics import accuracy_score, mean_absolute_error\n\n"
            "# Classification baseline: majority class\n"
            "majority_class = y_train.mode()[0]\n"
            "baseline_pred = np.full_like(y_test, fill_value=majority_class)\n"
            "print('Baseline accuracy:', accuracy_score(y_test, baseline_pred))\n\n"
            "# Regression baseline: mean target\n"
            "mean_target = y_train.mean()\n"
            "baseline_pred = np.full_like(y_test, fill_value=mean_target)\n"
            "print('Baseline MAE:', mean_absolute_error(y_test, baseline_pred))"
        ),
        keywords=["baseline", "reference model", "majority class"],
    ),
    "choosing_classification_algorithms": T(
        title="Choosing Classification Algorithms",
        module="model_selection",
        what=(
            "Different classification algorithms have different strengths. Logistic "
            "Regression is fast and interpretable; Random Forest handles non-linear "
            "relationships; SVMs work well with clear margins of separation."
        ),
        why=(
            "Choosing the right algorithm avoids wasted effort. A linear model on "
            "linearly separable data is faster and more interpretable than a neural "
            "network that achieves the same accuracy."
        ),
        when=(
            "After understanding your data's characteristics: number of features, "
            "sample size, linearity, class balance, and interpretability needs."
        ),
        example=(
            "Text classification with high-dimensional sparse features: Logistic "
            "Regression or Naive Bayes often outperform SVMs while being much "
            "faster to train."
        ),
        mistakes=[
            "Using SVM on a dataset with 10 million rows — it will be extremely slow.",
            "Choosing a black-box model when the stakeholder needs to explain decisions.",
            "Ignoring class imbalance — most algorithms assume balanced classes.",
        ],
        interpretation=(
            "Start with Logistic Regression (fast, interpretable). Move to Random "
            "Forest or Gradient Boosting if you need more accuracy. Use SVM only "
            "when the decision boundary is clearly non-linear and the dataset is "
            "moderate in size."
        ),
        think_about_it=(
            "You have 500 samples and 200 features. Which algorithms are likely "
            "to overfit, and which are more robust?"
        ),
        code_link=(
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.naive_bayes import GaussianNB\n\n"
            "# Quick comparison\n"
            "for Cls in [LogisticRegression, RandomForestClassifier, SVC, GaussianNB]:\n"
            "    m = Cls().fit(X_train, y_train)\n"
            "    print(Cls.__name__, m.score(X_test, y_test))"
        ),
        keywords=["classification", "algorithm choice", "logistic regression", "random forest"],
    ),
    "choosing_regression_algorithms": T(
        title="Choosing Regression Algorithms",
        module="model_selection",
        what=(
            "Regression algorithms predict continuous values. Linear Regression "
            "is the starting point; tree-based ensembles (Random Forest, Gradient "
            "Boosting) handle non-linear relationships."
        ),
        why=(
            "Different regression problems have different shapes. A linear model "
            "works well when the relationship is truly linear; tree-based models "
            "capture non-linear interactions without feature engineering."
        ),
        when=(
            "After exploring the data (scatter plots, residuals) to understand "
            "whether the relationship is linear or non-linear."
        ),
        example=(
            "House prices vs. square footage: roughly linear — Linear Regression "
            "is a good start. House prices vs. location, size, age, condition: "
            "complex non-linear — Random Forest or Gradient Boosting."
        ),
        mistakes=[
            "Assuming Linear Regression is always the simplest option — Ridge or "
            "Lasso may perform better with many correlated features.",
            "Using Gradient Boosting on a tiny dataset — it will overfit.",
            "Not checking residual plots — they reveal whether linear assumptions hold.",
        ],
        interpretation=(
            "If residual plots show a clear pattern, a linear model is insufficient. "
            "If residuals are randomly scattered around zero, a linear model is "
            "appropriate."
        ),
        think_about_it=(
            "Your dataset has 50 features and 200 samples. Which regression "
            "algorithm should you start with?"
        ),
        code_link=(
            "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n"
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n\n"
            "for Reg in [LinearRegression, Ridge, Lasso,\n"
            "            RandomForestRegressor, GradientBoostingRegressor]:\n"
            "    m = Reg().fit(X_train, y_train)\n"
            "    print(Reg.__name__, m.score(X_test, y_test))"
        ),
        keywords=["regression", "algorithm choice", "linear regression", "random forest"],
    ),
    "dataset_size_and_algorithm": T(
        title="Dataset Size and Algorithm Choice",
        module="model_selection",
        what=(
            "The amount of data you have strongly influences which algorithms "
            "will work well. Small datasets favour simple models; large datasets "
            "can support complex ones."
        ),
        why=(
            "Complex models (deep neural networks, large ensembles) need lots of "
            "data to learn generalisable patterns. On small datasets, they memorise "
            "noise instead."
        ),
        when=(
            "Before choosing an algorithm — always check your sample size relative "
            "to the number of features."
        ),
        example=(
            "100 samples, 10 features: Logistic Regression or a small Decision "
            "Tree. 100,000 samples, 100 features: Gradient Boosting or a neural "
            "network."
        ),
        mistakes=[
            "Using a deep learning model on 500 rows — it will overfit.",
            "Ignoring the effective sample size per feature (rule of thumb: ≥10 per feature).",
            "Forgetting that very large datasets may require approximate algorithms.",
        ],
        interpretation=(
            "A useful heuristic: start with the simplest model that can handle "
            "your data size. Complexity should increase only if it demonstrably "
            "improves test performance."
        ),
        think_about_it=(
            "You have 1,000 samples and 500 features. What are your options?"
        ),
        code_link=(
            "import pandas as pd\n"
            "\n"
            "print(f'Training samples: {X_train.shape[0]}')\n"
            "print(f'Features: {X_train.shape[1]}')\n"
            "print(f'Ratio: {X_train.shape[0] / X_train.shape[1]:.1f} samples per feature')\n"
            "# If ratio < 10, prefer simpler models or feature selection"
        ),
        keywords=["dataset size", "sample size", "complexity"],
    ),
    "feature_types_and_algorithm": T(
        title="Feature Types and Algorithm Choice",
        module="model_selection",
        what=(
            "The types of features in your dataset — numerical, categorical, text, "
            "or mixed — affect which algorithms are suitable and what preprocessing "
            "is needed."
        ),
        why=(
            "Some algorithms (e.g., linear models) require all features to be "
            "numerical. Others (e.g., tree-based models) can handle categorical "
            "variables natively or with minimal encoding."
        ),
        when=(
            "After EDA, when you know the types and distribution of your features."
        ),
        example=(
            "A dataset with 50 one-hot-encoded binary columns is very wide and "
            "sparse — a linear model or Naive Bayes may outperform a Decision Tree."
        ),
        mistakes=[
            "Passing string columns directly to sklearn — it will raise a TypeError.",
            "One-hot encoding high-cardinality features (e.g., ZIP code) — creates thousands of columns.",
            "Ignoring ordinal relationships in categorical features.",
        ],
        interpretation=(
            "Match your algorithm to your feature space. Use algorithms that "
            "naturally handle your feature types, or preprocess accordingly."
        ),
        think_about_it=(
            "Your dataset has 10 numerical features and 5 categorical features "
            "with 20 unique values each. How would you handle this?"
        ),
        code_link=(
            "import pandas as pd\n"
            "\n"
            "# Detect feature types automatically\n"
            "numerical = X.select_dtypes(include='number').columns.tolist()\n"
            "categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()\n"
            "print(f'Numerical: {len(numerical)}, Categorical: {len(categorical)}')"
        ),
        keywords=["feature types", "numerical", "categorical", "encoding"],
    ),
    "interpretability": T(
        title="Model Interpretability",
        module="model_selection",
        what=(
            "Interpretability is the degree to which a human can understand the "
            "reasoning behind a model's predictions. Linear models are highly "
            "interpretable; ensemble methods are less so."
        ),
        why=(
            "In regulated industries (healthcare, finance, law), you must explain "
            "why a model made a particular prediction. Even in unconstrained "
            "domains, interpretability builds trust and helps debug models."
        ),
        when=(
            "When stakeholders require explanations, when debugging errors, or "
            "when operating in a regulated domain."
        ),
        example=(
            "A bank uses Logistic Regression to predict loan defaults because "
            "regulators require a clear explanation of each decision. A Gradient "
            "Boosting model might be more accurate but cannot be easily explained."
        ),
        mistakes=[
            "Choosing a black-box model without considering interpretability needs.",
            "Assuming feature importance from tree-based models is the same as causal interpretation.",
            "Ignoring SHAP or LIME when you need local explanations from complex models.",
        ],
        interpretation=(
            "Simple models are transparent: you can see exactly how each feature "
            "contributes. Complex models require tools (SHAP, LIME) to extract "
            "feature-level explanations."
        ),
        think_about_it=(
            "Would you trust a medical diagnosis model that is 95 % accurate "
            "but cannot explain its decisions? What about one that is 90 % "
            "accurate but shows the reasoning?"
        ),
        code_link=(
            "from sklearn.linear_model import LogisticRegression\n"
            "import pandas as pd\n\n"
            "model = LogisticRegression().fit(X_train, y_train)\n"
            "coefficients = pd.Series(model.coef_[0], index=X_train.columns)\n"
            "print('Feature importance (coefficients):')\n"
            "print(coefficients.sort_values(ascending=False))"
        ),
        keywords=["interpretability", "explainability", "shap", "lime"],
    ),
    "model_complexity": T(
        title="Model Complexity",
        module="model_selection",
        what=(
            "Model complexity refers to the capacity of a model to fit intricate "
            "patterns. More parameters and deeper structures increase complexity."
        ),
        why=(
            "High complexity can capture subtle patterns but risks overfitting. "
            "Low complexity may underfit. The goal is to match complexity to the "
            "amount and quality of data available."
        ),
        when=(
            "Whenever you compare models or tune hyperparameters. More complex "
            "is not always better."
        ),
        example=(
            "A Decision Tree with max_depth=2 is simple and underfits complex data. "
            "A tree with max_depth=20 on a small dataset will overfit. max_depth=5 "
            "might be the sweet spot."
        ),
        mistakes=[
            "Always choosing the most complex model available.",
            "Not regularising complex models — most have parameters to control complexity.",
            "Assuming more features always means a better model.",
        ],
        interpretation=(
            "Plot validation accuracy against model complexity (e.g., max_depth, "
            "n_estimators). Accuracy rises then falls — the peak is your optimal "
            "complexity."
        ),
        think_about_it=(
            "You are comparing a Logistic Regression with 5 features to a Random "
            "Forest with 500 trees. Which is more complex? Is more complex always "
            "better?"
        ),
        code_link=(
            "# Compare validation accuracy across tree depths\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "for depth in [2, 5, 10, 20, None]:\n"
            "    m = DecisionTreeClassifier(max_depth=depth)\n"
            "    scores = cross_val_score(m, X_train, y_train, cv=5)\n"
            "    print(f'depth={depth}: {scores.mean():.4f} ± {scores.std():.4f}')"
        ),
        keywords=["complexity", "model capacity", "overfitting"],
    ),
    "training_time": T(
        title="Training Time",
        module="model_selection",
        what=(
            "Training time is how long a model takes to learn from the training "
            "data. It varies hugely across algorithms and dataset sizes."
        ),
        why=(
            "In production systems with frequent retraining, a model that takes "
            "hours to train may be impractical. Quick models enable faster "
            "iteration and experimentation."
        ),
        when=(
            "When you need rapid prototyping, real-time learning, or have "
            "strict time budgets for model updates."
        ),
        example=(
            "Logistic Regression on 10,000 rows: < 1 second. SVM with RBF "
            "kernel on the same data: 30 seconds. A neural network with "
            "GPU: 5 minutes."
        ),
        mistakes=[
            "Ignoring training time until deployment — then realising the model "
            "cannot be retrained within the update window.",
            "Comparing only accuracy without considering the cost of training.",
        ],
        interpretation=(
            "If two models have similar accuracy but one trains 10× faster, "
            "choose the faster one unless there is a specific reason not to."
        ),
        think_about_it=(
            "You need to retrain a model every hour on new data arriving "
            "from a streaming source. Which algorithms would you consider?"
        ),
        code_link=(
            "import time\n\n"
            "start = time.time()\n"
            "model.fit(X_train, y_train)\n"
            "elapsed = time.time() - start\n"
            "print(f'Training time: {elapsed:.2f}s')"
        ),
        keywords=["training time", "efficiency", "scalability"],
    ),
    "prediction_time": T(
        title="Prediction Time",
        module="model_selection",
        what=(
            "Prediction time (latency) is how long a model takes to produce an "
            "output for a single input or a batch. It is critical for real-time "
            "applications."
        ),
        why=(
            "A model that is accurate but slow at prediction may be unusable "
            "for real-time systems (e.g., fraud detection, recommendation "
            "engines, autonomous driving)."
        ),
        when=(
            "When deploying a model to production, especially for real-time or "
            "low-latency systems."
        ),
        example=(
            "A complex ensemble with 1000 trees might predict in 50ms per "
            "sample. A logistic regression model predicts in <1ms. For a "
            "system handling 10,000 requests per second, this matters."
        ),
        mistakes=[
            "Optimising only for accuracy and ignoring inference cost.",
            "Not benchmarking prediction latency before deployment.",
            "Using a heavy model for batch predictions when a lightweight model is sufficient.",
        ],
        interpretation=(
            "Always measure prediction time on realistic data. A model's "
            "complexity directly impacts its prediction latency."
        ),
        think_about_it=(
            "You have a fraud detection system that must flag suspicious "
            "transactions within 100ms. Would you use a deep neural network "
            "or a Gradient Boosted Tree?"
        ),
        code_link=(
            "import time\n\n"
            "start = time.time()\n"
            "preds = model.predict(X_test)\n"
            "elapsed = time.time() - start\n"
            "print(f'Prediction time for {len(X_test)} samples: {elapsed:.3f}s')\n"
            "print(f'Average per sample: {elapsed / len(X_test) * 1000:.2f}ms')"
        ),
        keywords=["prediction time", "latency", "inference"],
    ),
    "bias_and_variance": T(
        title="Bias and Variance",
        module="model_selection",
        what=(
            "Bias is the error from overly simplistic assumptions (underfitting). "
            "Variance is the error from sensitivity to small fluctuations in "
            "training data (overfitting). The total error = Bias² + Variance + "
            "Irreducible Noise."
        ),
        why=(
            "Understanding bias and variance helps diagnose why a model performs "
            "poorly and guides the right fix: simplify the model to reduce "
            "variance, or complexify to reduce bias."
        ),
        when=(
            "Whenever your model performs poorly. High training error → high bias. "
            "Low training error but high test error → high variance."
        ),
        example=(
            "Linear Regression on a non-linear dataset: high bias (underfitting). "
            "Decision Tree with no depth limit on a small dataset: high variance "
            "(overfitting)."
        ),
        mistakes=[
            "Trying to fix underfitting by collecting more data — it won't help.",
            "Trying to fix overfitting by making the model more complex.",
            "Ignoring the bias-variance tradeoff when tuning hyperparameters.",
        ],
        interpretation=(
            "The sweet spot is where total error is minimised — neither too simple "
            "nor too complex. Cross-validation helps find this balance."
        ),
        think_about_it=(
            "Your training accuracy is 60 % and test accuracy is 58 %. What does "
            "this tell you about bias and variance?"
        ),
        code_link=(
            "# Bias-variance diagnostic\n"
            "train_acc = model.score(X_train, y_train)\n"
            "test_acc = model.score(X_test, y_test)\n"
            "print(f'Training accuracy: {train_acc:.4f}')\n"
            "print(f'Test accuracy: {test_acc:.4f}')\n"
            "print(f'Gap: {train_acc - test_acc:.4f} (high = overfitting)')"
        ),
        keywords=["bias", "variance", "bias-variance tradeoff"],
    ),
    "underfitting": T(
        title="Underfitting",
        module="model_selection",
        what=(
            "Underfitting occurs when a model is too simple to capture the "
            "underlying pattern in the data. Both training and test performance "
            "are poor."
        ),
        why=(
            "An underfitted model wastes computational resources and produces "
            "unreliable predictions. It signals that the model or features need "
            "to be more expressive."
        ),
        when=(
            "When both training and validation accuracy are low, or when residuals "
            "show a clear systematic pattern."
        ),
        example=(
            "Using a straight line (Linear Regression) to fit data that follows "
            "a quadratic curve. The model cannot capture the curve regardless of "
            "how much data it sees."
        ),
        mistakes=[
            "Adding more data when the model is underfitting — it will not help.",
            "Reducing regularisation when the problem is insufficient model capacity.",
            "Ignoring feature engineering that could make the problem linearly separable.",
        ],
        interpretation=(
            "If training error is high and similar to validation error, the model "
            "is underfitting. Solutions: use a more complex model, add polynomial "
            "features, reduce regularisation."
        ),
        think_about_it=(
            "Your Linear Regression has an R² of 0.3 on both train and test. "
            "What steps would you take?"
        ),
        code_link=(
            "# Detect underfitting\n"
            "print(f'Train R²: {model.score(X_train, y_train):.4f}')\n"
            "print(f'Test R²:  {model.score(X_test, y_test):.4f}')\n"
            "# Both low → underfitting\n"
            "# Fix: try PolynomialFeatures, reduce alpha in Ridge, use RandomForest"
        ),
        keywords=["underfitting", "high bias", "model simplicity"],
    ),
    "overfitting": T(
        title="Overfitting",
        module="model_selection",
        what=(
            "Overfitting occurs when a model memorises noise in the training data "
            "instead of learning the true pattern. Training performance is high "
            "but test performance is poor."
        ),
        why=(
            "Overfitting is the most common practical problem in machine learning. "
            "A model that overfits is unreliable on new, unseen data."
        ),
        when=(
            "When training accuracy is much higher than test accuracy, or when "
            "validation performance degrades while training performance improves."
        ),
        example=(
            "A Decision Tree with no depth limit achieves 100 % training accuracy "
            "but only 70 % test accuracy on Titanic survival prediction."
        ),
        mistakes=[
            "Not using cross-validation to detect overfitting.",
            "Adding more features without regularisation.",
            "Training for too many epochs (in neural networks) without early stopping.",
        ],
        interpretation=(
            "The gap between training and test accuracy indicates overfitting "
            "severity. Solutions: regularisation, simpler model, more data, "
            "feature selection, cross-validation."
        ),
        think_about_it=(
            "Your Random Forest achieves 99 % training accuracy and 75 % test "
            "accuracy. How would you diagnose and fix this?"
        ),
        code_link=(
            "# Detect overfitting\n"
            "train_acc = model.score(X_train, y_train)\n"
            "test_acc = model.score(X_test, y_test)\n"
            "print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}, Gap: {train_acc - test_acc:.4f}')\n"
            "# Large gap → overfitting → try: max_depth, min_samples_leaf, n_estimators"
        ),
        keywords=["overfitting", "high variance", "regularisation"],
    ),
    "cross_validation_model_selection": T(
        title="Cross-Validation for Model Selection",
        module="model_selection",
        what=(
            "Cross-validation splits data into k folds, trains on k-1, and "
            "validates on the remaining fold — rotating through all folds. It "
            "gives a more reliable performance estimate than a single train/test "
            "split."
        ),
        why=(
            "A single split may be lucky or unlucky. Cross-validation averages "
            "over multiple splits, giving a stable estimate of how a model "
            "generalises."
        ),
        when=(
            "Whenever comparing models, tuning hyperparameters, or working with "
            "small datasets where a single split is unreliable."
        ),
        example=(
            "5-fold CV on Titanic: Logistic Regression 0.79±0.03, Random Forest "
            "0.81±0.02, SVM 0.78±0.04. Random Forest is slightly better and more "
            "stable."
        ),
        mistakes=[
            "Using cross-validation scores as the final test score — you still need a held-out test set.",
            "Performing feature selection before cross-validation — this leaks information.",
            "Using stratified k-fold for regression — use regular k-fold instead.",
        ],
        interpretation=(
            "Look at both the mean and standard deviation of CV scores. A high "
            "mean with low std is ideal. High std means the model is sensitive "
            "to which data it trains on."
        ),
        think_about_it=(
            "Model A has CV accuracy 0.82±0.01 and Model B has 0.83±0.05. "
            "Which would you choose?"
        ),
        code_link=(
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n"
            "print(f'CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')"
        ),
        keywords=["cross-validation", "k-fold", "model evaluation"],
    ),
    "hyperparameters": T(
        title="Hyperparameters",
        module="model_selection",
        what=(
            "Hyperparameters are settings you choose before training a model — "
            "they are not learned from data. Examples: learning rate, tree depth, "
            "number of estimators, regularisation strength."
        ),
        why=(
            "Hyperparameters control model behaviour. Poor choices lead to "
            "underfitting or overfitting. Systematic tuning often yields large "
            "performance gains."
        ),
        when=(
            "After selecting a candidate algorithm, before final evaluation. "
            "Always tune hyperparameters on a validation set or via cross-validation."
        ),
        example=(
            "Random Forest with n_estimators=10 gives 75 % accuracy. With "
            "n_estimators=200 it reaches 82 %. With max_depth=5 it reaches 84 %."
        ),
        mistakes=[
            "Using default hyperparameters without trying to tune them.",
            "Tuning on the test set — this causes information leakage.",
            "Tuning too many hyperparameters at once on a small dataset.",
        ],
        interpretation=(
            "The effect of a hyperparameter depends on the dataset. Always validate "
            "tuning choices with cross-validation, never with the test set."
        ),
        think_about_it=(
            "Which hyperparameters would you tune first for a Random Forest "
            "classifier? Why?"
        ),
        code_link=(
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "# Key hyperparameters for Random Forest\n"
            "model = RandomForestClassifier(\n"
            "    n_estimators=200,      # number of trees\n"
            "    max_depth=10,          # tree depth limit\n"
            "    min_samples_split=5,   # minimum samples to split a node\n"
            "    min_samples_leaf=2,    # minimum samples in a leaf\n"
            "    random_state=42\n"
            ")"
        ),
        keywords=["hyperparameters", "model configuration", "tuning"],
    ),
    "grid_search": T(
        title="Grid Search",
        module="model_selection",
        what=(
            "Grid Search systematically evaluates every combination of specified "
            "hyperparameter values. It guarantees finding the best combination "
            "within the defined grid."
        ),
        why=(
            "Manual tuning is slow and biased. Grid Search automates the process "
            "and ensures no combination is missed."
        ),
        when=(
            "When you have few hyperparameters and a small grid. For high-dimensional "
            "hyperparameter spaces, Random Search is often more efficient."
        ),
        example=(
            "Tuning a Decision Tree with max_depth=[3,5,10] and "
            "min_samples_split=[2,5,10]: Grid Search tries all 9 combinations "
            "with 5-fold CV = 45 model fits."
        ),
        mistakes=[
            "Creating a grid that is too large — 1000 combinations × 5 folds = 5000 fits.",
            "Forgetting to set refit=True so the best model is retrained on all data.",
            "Not scaling features when using distance-based models in the grid.",
        ],
        interpretation=(
            "Grid Search returns the best parameters and the corresponding CV "
            "score. Use the refitted model for final predictions."
        ),
        think_about_it=(
            "You want to tune 4 hyperparameters with 10 values each. How many "
            "combinations will Grid Search evaluate? Is this practical?"
        ),
        code_link=(
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "param_grid = {\n"
            "    'n_estimators': [50, 100, 200],\n"
            "    'max_depth': [5, 10, 20, None],\n"
            "    'min_samples_split': [2, 5, 10]\n"
            "}\n"
            "grid = GridSearchCV(RandomForestClassifier(random_state=42),\n"
            "                    param_grid, cv=5, scoring='accuracy')\n"
            "grid.fit(X_train, y_train)\n"
            "print('Best params:', grid.best_params_)\n"
            "print('Best score:', grid.best_score_)"
        ),
        keywords=["grid search", "hyperparameter tuning", "exhaustive search"],
    ),
    "random_search": T(
        title="Random Search",
        module="model_selection",
        what=(
            "Random Search samples hyperparameter combinations randomly from "
            "specified distributions. It is often more efficient than Grid Search "
            "for large hyperparameter spaces."
        ),
        why=(
            "Not all hyperparameters are equally important. Random Search explores "
            "more of the space per unit time, finding good combinations faster "
            "than exhaustive Grid Search."
        ),
        when=(
            "When the hyperparameter space is large or when you have a limited "
            "time budget for tuning."
        ),
        example=(
            "With 100 iterations of Random Search, you explore 100 combinations. "
            "Grid Search with the same budget might only try 10 values per "
            "parameter across 2 parameters."
        ),
        mistakes=[
            "Setting n_iter too low — you may miss good combinations.",
            "Not defining合理的 distributions for continuous hyperparameters.",
            "Using random search on a very small grid where Grid Search is fast enough.",
        ],
        interpretation=(
            "Random Search is especially effective when only a few hyperparameters "
            "strongly affect performance. It finds near-optimal settings faster "
            "than Grid Search in most practical scenarios."
        ),
        think_about_it=(
            "You have 5 hyperparameters, each with a continuous range. Would "
            "Grid Search or Random Search be more practical?"
        ),
        code_link=(
            "from sklearn.model_selection import RandomizedSearchCV\n"
            "from scipy.stats import randint, uniform\n\n"
            "param_dist = {\n"
            "    'n_estimators': randint(50, 300),\n"
            "    'max_depth': randint(3, 30),\n"
            "    'min_samples_split': randint(2, 20),\n"
            "    'min_samples_leaf': randint(1, 10)\n"
            "}\n"
            "search = RandomizedSearchCV(\n"
            "    RandomForestClassifier(random_state=42),\n"
            "    param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42\n"
            ")\n"
            "search.fit(X_train, y_train)\n"
            "print('Best params:', search.best_params_)"
        ),
        keywords=["random search", "stochastic tuning", "efficiency"],
    ),
    "hyperparameter_tuning": T(
        title="Hyperparameter Tuning Strategy",
        module="model_selection",
        what=(
            "A systematic approach to finding optimal hyperparameters: start with "
            "defaults, tune the most impactful parameters first, then refine."
        ),
        why=(
            "Random tuning wastes time. A strategic approach — starting broad, "
            "then narrowing — finds good settings efficiently."
        ),
        when=(
            "After selecting a candidate algorithm and before final evaluation."
        ),
        example=(
            "Step 1: Run with defaults → 80 % accuracy.\n"
            "Step 2: Tune n_estimators and max_depth → 84 % accuracy.\n"
            "Step 3: Tune min_samples_split and min_samples_leaf → 85 % accuracy.\n"
            "Step 4: Final evaluation on held-out test set → 84 % accuracy."
        ),
        mistakes=[
            "Tuning every hyperparameter at once — wasteful and slow.",
            "Not using cross-validation during tuning.",
            "Over-tuning: optimising so precisely that the model overfits the validation set.",
        ],
        interpretation=(
            "Diminishing returns are normal. If the first tuning round improves "
            "accuracy by 4 % and the second by 1 %, the second round may not "
            "be worth the compute."
        ),
        think_about_it=(
            "You have limited compute budget. Which hyperparameters would you "
            "tune first for a Gradient Boosting model?"
        ),
        code_link=(
            "# Two-phase tuning example\n"
            "from sklearn.model_selection import GridSearchCV\n\n"
            "# Phase 1: coarse search\n"
            "param_grid_1 = {'n_estimators': [50, 200], 'max_depth': [3, 10]}\n"
            "g1 = GridSearchCV(model, param_grid_1, cv=5)\n"
            "g1.fit(X_train, y_train)\n\n"
            "# Phase 2: fine-tune around best values\n"
            "best = g1.best_params_\n"
            "param_grid_2 = {\n"
            "    'n_estimators': [best['n_estimators']-20, best['n_estimators'], best['n_estimators']+20],\n"
            "    'max_depth': [max(1, best['max_depth']-1), best['max_depth'], best['max_depth']+1]\n"
            "}\n"
            "g2 = GridSearchCV(model, param_grid_2, cv=5)\n"
            "g2.fit(X_train, y_train)"
        ),
        keywords=["tuning strategy", "two-phase tuning", "efficiency"],
    ),
    "model_comparison_selection": T(
        title="Model Comparison",
        module="model_selection",
        what=(
            "Model comparison evaluates multiple algorithms on the same data with "
            "the same preprocessing and evaluation protocol. It reveals which "
            "algorithm is most suitable for the problem."
        ),
        why=(
            "Choosing a model without comparison is guesswork. Systematic comparison "
            "ensures the selected model is genuinely the best for your data and "
            "constraints."
        ),
        when=(
            "After preprocessing, before final model selection. Always compare "
            "at least 3-4 algorithms."
        ),
        example=(
            "Titanic dataset: Logistic Regression 0.79, Random Forest 0.82, "
            "Gradient Boosting 0.83, SVM 0.78. Gradient Boosting wins on "
            "accuracy, but Random Forest is faster and more interpretable."
        ),
        mistakes=[
            "Comparing models trained on different train/test splits.",
            "Using different preprocessing for different models.",
            "Comparing only one metric — always compare multiple.",
        ],
        interpretation=(
            "Look at the full picture: accuracy, training time, interpretability, "
            "and robustness. The 'best' model depends on your priorities."
        ),
        think_about_it=(
            "If Model A is 1 % more accurate than Model B but takes 10× longer "
            "to train, which would you choose for a real-time system?"
        ),
        code_link=(
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n\n"
            "models = {\n"
            "    'LogisticRegression': LogisticRegression(max_iter=1000),\n"
            "    'RandomForest': RandomForestClassifier(n_estimators=100),\n"
            "    'GradientBoosting': GradientBoostingClassifier(n_estimators=100)\n"
            "}\n"
            "for name, m in models.items():\n"
            "    scores = cross_val_score(m, X, y, cv=5)\n"
            "    print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')"
        ),
        keywords=["model comparison", "algorithm comparison", "benchmarking"],
    ),
    "metric_selection": T(
        title="Selecting Evaluation Metrics",
        module="model_selection",
        what=(
            "The evaluation metric you choose determines what 'best' means. "
            "Accuracy, F1, AUC, RMSE, and R² each emphasise different aspects "
            "of model performance."
        ),
        why=(
            "Wrong metrics lead to wrong conclusions. Accuracy on an imbalanced "
            "dataset is misleading. RMSE penalises large errors more than MAE. "
            "Your metric must match your business goal."
        ),
        when=(
            "Before model training — decide your metric first so you optimise "
            "for the right thing."
        ),
        example=(
            "Cancer detection (1 % positive rate): accuracy of 99 % is trivial "
            "(always predict negative). Recall matters more — catching every "
            "cancer case is the priority."
        ),
        mistakes=[
            "Defaulting to accuracy without considering class balance.",
            "Optimising for F1 when the cost of false positives is very high.",
            "Using R² as the sole regression metric — it hides the scale of errors.",
        ],
        interpretation=(
            "Match your metric to your goal:\n"
            "- Minimise false negatives → Recall\n"
            "- Minimise false positives → Precision\n"
            "- Balance both → F1\n"
            "- Minimise large errors → RMSE\n"
            "- Explain variance → R²"
        ),
        think_about_it=(
            "You are building a spam filter. Is precision or recall more important? "
            "Why?"
        ),
        code_link=(
            "# Metric comparison\n"
            "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n\n"
            "y_pred = model.predict(X_test)\n"
            "y_proba = model.predict_proba(X_test)[:, 1]\n\n"
            "print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\n"
            "print(f'F1 Score:  {f1_score(y_test, y_pred):.4f}')\n"
            "print(f'AUC:       {roc_auc_score(y_test, y_proba):.4f}')"
        ),
        keywords=["metrics", "accuracy", "f1", "auc", "evaluation"],
    ),
    "choosing_final_model": T(
        title="Choosing the Final Model",
        module="model_selection",
        what=(
            "The final model is selected after comparing candidates, tuning "
            "hyperparameters, and evaluating on a held-out test set. It is the "
            "model that will be deployed or presented."
        ),
        why=(
            "The final model must generalise to unseen data. It should be the "
            "best balance of accuracy, interpretability, speed, and simplicity "
            "for your specific use case."
        ),
        when=(
            "After all experiments are complete. The test set is used exactly "
            "once for this final evaluation."
        ),
        example=(
            "After testing 5 algorithms, Gradient Boosting with tuned "
            "hyperparameters achieves the best cross-validation score. You "
            "retrain it on all training data and evaluate on the test set: "
            "84 % accuracy. This is your final model."
        ),
        mistakes=[
            "Tweaking the model after seeing test results — this is data leakage.",
            "Choosing the most complex model without considering deployment constraints.",
            "Not saving the final model and its preprocessing pipeline together.",
        ],
        interpretation=(
            "The final test score is your honest estimate of real-world performance. "
            "If it is significantly lower than your CV score, your model may be "
            "overfitting to the validation data."
        ),
        think_about_it=(
            "Your Gradient Boosting model gets 88 % CV accuracy and 82 % test "
            "accuracy. What might explain the gap, and what would you do?"
        ),
        code_link=(
            "# Save the final model and pipeline\n"
            "import joblib\n\n"
            "final_model = GradientBoostingClassifier(n_estimators=200, max_depth=5)\n"
            "final_model.fit(X_train, y_train)\n\n"
            "# Save for deployment\n"
            "joblib.dump(final_model, 'final_model.pkl')\n"
            "print(f'Final test accuracy: {final_model.score(X_test, y_test):.4f}')"
        ),
        keywords=["final model", "deployment", "model saving"],
    ),
    "model_selection_case_study": T(
        title="Model Selection Case Study",
        module="model_selection",
        what=(
            "A practical walkthrough of selecting the best model for the Titanic "
            "survival prediction task, from baseline to final model."
        ),
        why=(
            "Seeing the complete model selection process end-to-end helps you "
            "apply the same structured approach to any new problem."
        ),
        when=(
            "Use this as a template for every classification project."
        ),
        example=(
            "Step 1: Baseline → always predict majority class → 62 % accuracy.\n"
            "Step 2: Logistic Regression → 78 % accuracy.\n"
            "Step 3: Random Forest → 81 % accuracy.\n"
            "Step 4: Gradient Boosting (tuned) → 83 % accuracy.\n"
            "Step 5: Final model → Gradient Boosting with 83 % test accuracy."
        ),
        mistakes=[
            "Skipping steps 1-2 and jumping straight to a complex model.",
            "Not recording experiments — you cannot reproduce or compare results.",
            "Declaring the model 'done' without checking for overfitting.",
        ],
        interpretation=(
            "Model selection is systematic, not magical. Each step provides "
            "information that guides the next decision."
        ),
        think_about_it=(
            "If you had to justify your model choice to a non-technical "
            "stakeholder, how would you explain why you chose Gradient Boosting "
            "over Logistic Regression?"
        ),
        code_link=(
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n\n"
            "# Systematic comparison\n"
            "models = [\n"
            "    ('Baseline (majority)', None),\n"
            "    ('Logistic Regression', LogisticRegression(max_iter=1000)),\n"
            "    ('Random Forest', RandomForestClassifier(n_estimators=100)),\n"
            "    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100))\n"
            "]\n"
            "for name, model in models:\n"
            "    if model is None:\n"
            "        from sklearn.dummy import DummyClassifier\n"
            "        model = DummyClassifier(strategy='most_frequent')\n"
            "    scores = cross_val_score(model, X, y, cv=5)\n"
            "    print(f'{name}: {scores.mean():.4f} ± {scores.std():.4f}')"
        ),
        keywords=["case study", "titanic", "end-to-end", "systematic"],
    ),
    "which_model_should_i_try": T(
        title="Which Model Should I Try? — Decision Guide",
        module="model_selection",
        what=(
            "A practical decision tree for choosing algorithms based on your "
            "dataset characteristics, problem type, and constraints."
        ),
        why=(
            "Knowing where to start saves hours of experimentation. This guide "
            "provides a structured starting point based on data characteristics."
        ),
        when=(
            "At the beginning of any modelling task, after EDA and preprocessing."
        ),
        example=(
            "Small dataset (n < 1000), few features, need interpretability → "
            "Logistic Regression / Linear Regression.\n"
            "Medium dataset, mixed features, need accuracy → Random Forest / "
            "Gradient Boosting.\n"
            "Large dataset (n > 100,000), need speed → Linear model with SGD, "
            "or a small neural network."
        ),
        mistakes=[
            "Always using the same algorithm for every problem.",
            "Not considering the constraints (time, interpretability, deployment).",
            "Ignoring the 'simplest sufficient model' principle.",
        ],
        interpretation=(
            "The best model depends on your specific context. Use this guide "
            "as a starting point, then validate with cross-validation."
        ),
        think_about_it=(
            "You have a dataset with 50,000 rows, 30 features (mix of numerical "
            "and categorical), and the stakeholder needs to explain each decision "
            "to a regulator. Which model would you try first?"
        ),
        code_link=(
            "import pandas as pd\n"
            "\n"
            "# Quick model recommendation logic\n"
            "n_samples, n_features = X.shape\n"
            "print(f'Samples: {n_samples}, Features: {n_features}')\n"
            "print(f'Ratio: {n_samples/n_features:.1f}')\n\n"
            "if n_samples < 1000:\n"
            "    print('→ Try: Logistic/Linear Regression, small Decision Tree')\n"
            "elif n_samples < 50000:\n"
            "    print('→ Try: Random Forest, Gradient Boosting, SVM')\n"
            "else:\n"
            "    print('→ Try: Linear models (SGD), LightGBM, small neural net')"
        ),
        keywords=["decision guide", "algorithm selection", "practical guide"],
    ),
}
