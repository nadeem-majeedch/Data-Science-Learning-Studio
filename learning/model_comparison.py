"""Model Comparison curriculum — 15 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="cmp_01", title="Why Compare Models?",
        section="model_comparison", order=1, difficulty="beginner",
        objectives=[
            "Understand the need for systematic comparison",
            "Know that no single model is universally best",
            "Appreciate fair, reproducible comparison",
        ],
        concept=(
            "Different algorithms make different assumptions and excel in different situations. "
            "Comparing models objectively reveals which works best for YOUR specific data "
            "and problem. Without comparison, you might settle for a suboptimal model."
        ),
        why_matters=(
            "Without comparison, you might choose a model based on reputation rather than "
            "performance. A 2% improvement in F1 can mean hundreds of correctly classified "
            "cases in production. But the improvement must be real, not an artefact of "
            "unfair comparison."
        ),
        example=(
            "A data scientist builds only a Random Forest and reports 82% accuracy. "
            "A fair comparison shows Logistic Regression achieves 81% (10x faster, "
            "interpretable) and Gradient Boosting achieves 84% (2x slower). "
            "The 'best' model depends on whether accuracy, speed, or interpretability matters more."
        ),
        practice_exercise=(
            "Load a classification dataset and compare Logistic Regression and Random Forest "
            "using 5-fold cross-validation. "
            "1. Which has higher F1?\n"
            "2. What is the standard deviation?\n"
            "3. Are they statistically different?"
        ),
        quiz=[
            QuizQuestion(
                question="Why can't you just pick the model with the highest reported accuracy?",
                options=[
                    "Because accuracy is always wrong",
                    "Because reported accuracy may come from an unfair comparison (different splits, metrics, or data)",
                    "Because all models have the same accuracy",
                    "Because accuracy cannot be computed for most models",
                ],
                correct_index=1,
                explanation=(
                    "Without controlled comparison (same data, split, metric, and CV), "
                    "accuracy numbers are not comparable. Model A's 85% on one split may be "
                    "worse than Model B's 83% on a harder split."
                ),
            ),
        ],
        common_mistakes=[
            "Using only one model",
            "Choosing based on reputation instead of data",
            "Not comparing fairly",
        ],
        takeaways=[
            "Always compare multiple models systematically",
            "Let data decide, not assumptions or reputation",
            "Fair comparison requires consistent methodology",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_02", title="Fair Model Comparison",
        section="model_comparison", order=2, difficulty="beginner",
        objectives=[
            "Set up fair comparisons",
            "Control confounding variables",
            "Ensure reproducibility",
        ],
        concept=(
            "Fair comparison requires:\n"
            "• Same dataset (features, rows, target)\n"
            "• Same train/test split or CV folds\n"
            "• Same preprocessing pipeline\n"
            "• Same evaluation metric\n"
            "• Same random_state for reproducibility\n"
            "Change ONLY the algorithm."
        ),
        why_matters=(
            "If model A is tested on an easy split and model B on a hard split, "
            "the comparison is meaningless. Controlling all variables except the algorithm "
            "ensures the difference in performance is due to the algorithm, not the setup."
        ),
        example=(
            "Unfair comparison: RF trained on 80% of data, LR trained on 70%. "
            "RF appears better but it had more training data.\n"
            "Fair comparison: Both use exactly the same 80/20 split with random_state=42. "
            "Any performance difference is due to the algorithm."
        ),
        practice_exercise=(
            "Set up a fair comparison:\n"
            "1. Create a single train_test_split with random_state=42\n"
            "2. Preprocess training data and apply the same preprocessing to test data\n"
            "3. Train both models on the same preprocessed training data\n"
            "4. Evaluate on the same test data"
        ),
        quiz=[
            QuizQuestion(
                question="What must stay the same for a fair model comparison?",
                options=[
                    "Only the training data",
                    "Dataset, split, preprocessing, metric, and CV folds — only the algorithm changes",
                    "Nothing — just compare accuracy",
                    "The hyperparameters must be identical",
                ],
                correct_index=1,
                explanation=(
                    "Every factor that affects performance must be identical. If you change "
                    "the preprocessing, you might favour one model. If you change the metric, "
                    "you might favour another. Only the algorithm should differ."
                ),
            ),
        ],
        common_mistakes=[
            "Comparing on different splits",
            "Different preprocessing for different models",
            "Different metrics for different models",
        ],
        takeaways=[
            "Same data, same split, same preprocessing, same metric, same CV folds",
            "Change only the algorithm",
            "Use the same random_state everywhere for reproducibility",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_03", title="Same Dataset",
        section="model_comparison", order=3, difficulty="beginner",
        objectives=[
            "Use identical data for all models",
            "Handle missing values consistently",
            "Apply the same feature set",
        ],
        concept=(
            "Every model must see the same features, the same rows, and the same target "
            "variable. Any difference in the data invalidates the comparison — you'd be "
            "comparing apples to oranges."
        ),
        why_matters=(
            "If model A is trained on 50 features and model B on 30 features, the comparison "
            "is unfair. Model B might perform worse simply because it has less information, "
            "not because the algorithm is inferior."
        ),
        example=(
            "Dataset with 500 rows, 15 features:\n"
            "• Fair: both models use all 500 rows and 15 features\n"
            "• Unfair: model A drops rows with missing values (400 rows left), "
            "model B imputes missing values (500 rows). Model B has more data."
        ),
        common_mistakes=[
            "Dropping different rows for different models",
            "Engineering different features per model",
            "Not using the same target variable",
        ],
        practice_exercise=(
            "Load a dataset with missing values. "
            "1. If you drop rows with NaN, how many samples remain?\n"
            "2. If you impute, all samples remain.\n"
            "3. Which approach should you use for a fair comparison?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A is trained on 50 features and Model B on 40 features (10 were removed). Is the comparison fair?",
                options=[
                    "Yes — fewer features is simpler",
                    "No — different feature sets mean they're solving different problems",
                    "Yes — as long as both use the same algorithm",
                    "It depends on which features were removed",
                ],
                correct_index=1,
                explanation=(
                    "Different feature sets change the problem itself. Model B has less "
                    "information available. Any performance difference could be due to the "
                    "missing features, not the algorithm. Keep features identical."
                ),
            ),
        ],
        takeaways=[
            "All models must see identical data: same rows, same features, same target",
            "Preprocessing must be consistent",
            "Document any data differences if unavoidable",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_04", title="Same Data Split",
        section="model_comparison", order=4, difficulty="beginner",
        objectives=[
            "Apply consistent train/test splits",
            "Use cross-validation with the same folds",
            "Understand why split consistency matters",
        ],
        concept=(
            "Using different splits for different models means they're evaluated on different "
            "test data. Always use the same random_state and the same CV folds. "
            "Stratified k-fold ensures class distribution is preserved in each fold."
        ),
        why_matters=(
            "A test set might contain 'easy' or 'hard' examples. If model A happens to get "
            "an easier test set, it appears better. Using identical folds eliminates this "
            "source of unfairness."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n\n"
            "# SAME CV object for all models\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')\n"
            "    print(f'{name}: {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Not setting random_state in CV (different folds each run)",
            "Using different numbers of folds for different models",
            "Not shuffling data before splitting (temporal bias)",
        ],
        practice_exercise=(
            "Compare two models using the same StratifiedKFold with random_state=42. "
            "1. Create the CV object once and reuse it for both models.\n"
            "2. Is the comparison fair? Why?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is StratifiedKFold preferred over regular KFold for classification?",
                options=[
                    "StratifiedKFold is faster",
                    "StratifiedKFold ensures each fold has the same class proportions as the full dataset",
                    "StratifiedKFold uses more data",
                    "It doesn't matter — both give the same results",
                ],
                correct_index=1,
                explanation=(
                    "StratifiedKFold maintains the same class distribution in every fold. "
                    "Regular KFold might create a fold where a rare class is entirely absent, "
                    "producing misleading scores. This ensures fair evaluation across all folds."
                ),
            ),
        ],
        takeaways=[
            "Same random_state = same splits for all models",
            "Same number of folds for all models",
            "Use StratifiedKFold for classification",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_05", title="Same Preprocessing",
        section="model_comparison", order=5, difficulty="beginner",
        objectives=[
            "Apply consistent preprocessing for all models",
            "Use Pipeline to enforce consistency",
            "Handle model-specific needs within the pipeline",
        ],
        concept=(
            "Preprocessing must be identical for all models: same imputation strategy, "
            "same scaling, same encoding. Model-specific requirements (like scaling for KNN) "
            "are handled within each model's pipeline."
        ),
        why_matters=(
            "Different preprocessing can favour different models. Scaling helps KNN but doesn't "
            "affect Random Forest. If you scale for one model but not another, the comparison "
            "is unfair. Use Pipeline to make preprocessing part of each model."
        ),
        python_example=(
            "```python\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.impute import SimpleImputer\n\n"
            "# Each model includes its own preprocessing\n"
            "lr_pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', LogisticRegression())\n"
            "])\n\n"
            "rf_pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('model', RandomForestClassifier(random_state=42))\n"
            "])\n"
            "# Note: RF doesn't need scaling, but imputation is the same for fairness\n"
            "```"
        ),
        common_mistakes=[
            "Different scaling strategies for different models",
            "Fitting preprocessing on different data subsets",
            "Not using Pipeline (prone to data leakage)",
        ],
        practice_exercise=(
            "Build pipelines for LR (with scaling) and RF (without scaling) using the same "
            "imputation strategy. "
            "1. Are the pipelines equivalent in terms of data handling?\n"
            "2. Why does RF not need scaling in the pipeline?"
        ),
        quiz=[
            QuizQuestion(
                question="Why should preprocessing be part of a Pipeline rather than applied separately?",
                options=[
                    "Pipelines are faster",
                    "Pipelines ensure preprocessing is applied consistently and prevent data leakage",
                    "sklearn requires it",
                    "Pipelines use less memory",
                ],
                correct_index=1,
                explanation=(
                    "Pipelines ensure preprocessing is fitted on training data only and "
                    "applied consistently during cross-validation and prediction. Separate "
                    "preprocessing risks data leakage (fitting on test data) and inconsistencies."
                ),
            ),
        ],
        takeaways=[
            "Same preprocessing for all models (imputation, encoding, scaling)",
            "Use Pipeline to enforce consistency and prevent data leakage",
            "Model-specific needs (like scaling for KNN) are handled within the pipeline",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_06", title="Same Evaluation Metrics",
        section="model_comparison", order=6, difficulty="beginner",
        objectives=[
            "Use consistent metrics for comparison",
            "Choose metrics appropriate for the problem",
            "Report multiple metrics for completeness",
        ],
        concept=(
            "Compare all models on the same primary metric (e.g., F1 for imbalanced classification). "
            "Also report supplementary metrics for a complete picture. "
            "Never cherry-pick the metric that favours one model."
        ),
        why_matters=(
            "Different metrics tell different stories. Model A might have higher accuracy but "
            "lower F1. If you only report accuracy, Model A looks better. Fair comparison "
            "requires consistent metrics."
        ),
        example=(
            "Fraud detection comparison (99% non-fraud, 1% fraud):\n"
            "• Model A: accuracy=99.1%, F1=0.45 (good at predicting non-fraud)\n"
            "• Model B: accuracy=97.5%, F1=0.72 (better at catching fraud)\n"
            "Reporting only accuracy makes A look better. F1 reveals B is superior."
        ),
        practice_exercise=(
            "Compare two models using accuracy, F1, precision, and recall. "
            "1. Do the models rank differently depending on the metric?\n"
            "2. Which metric is most appropriate for this problem?"
        ),
        common_mistakes=[
            "Comparing accuracy for some models, F1 for others",
            "Not reporting multiple metrics",
            "Choosing the metric that favours one model",
        ],
        quiz=[
            QuizQuestion(
                question="A model comparison report shows Model A with accuracy=0.85 and Model B with F1=0.78. Is this a fair comparison?",
                options=[
                    "Yes — both metrics are standard",
                    "No — different metrics make the comparison meaningless",
                    "Yes — accuracy and F1 are the same thing",
                    "It depends on the dataset size",
                ],
                correct_index=1,
                explanation=(
                    "Comparing accuracy for one model and F1 for another is meaningless. "
                    "Both models must be evaluated with the same metric. A might have F1=0.60 "
                    "and B might have accuracy=0.92 — you can't tell without consistent metrics."
                ),
            ),
        ],
        takeaways=[
            "One primary metric for comparison, plus supplementary metrics",
            "Don't cherry-pick metrics that favour one model",
            "Report all results honestly",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_07", title="Cross-Validation Comparison",
        section="model_comparison", order=7, difficulty="intermediate",
        objectives=[
            "Use cross-validation for reliable model comparison",
            "Report and interpret mean ± std",
            "Assess whether differences are meaningful",
        ],
        concept=(
            "Cross-validation provides more reliable comparison than a single holdout. "
            "Report mean ± std. If the mean difference between models is smaller than the "
            "standard deviation, the models may be statistically equivalent."
        ),
        why_matters=(
            "A single test set evaluation depends on which specific samples are included. "
            "Cross-validation averages over multiple splits, giving a more reliable estimate. "
            "The standard deviation reveals how stable the performance is."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "import numpy as np\n\n"
            "results = {}\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
            "    results[name] = {'mean': scores.mean(), 'std': scores.std()}\n\n"
            "# Rank by mean F1\n"
            "print(f'{\"Model\":25s} {\"F1\":>15s}')\n"
            "print('-' * 42)\n"
            "for name, r in sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True):\n"
            "    print(f'{name:25s} {r[\"mean\"]:.3f} ± {r[\"std\"]:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Only reporting mean (ignoring std — critical information)",
            "Claiming a 0.5% difference is meaningful without checking std",
            "Not using paired statistical tests for formal significance",
        ],
        practice_exercise=(
            "Run 5-fold CV on 3 models and create a results table with mean and std. "
            "1. Is the difference between the top two models larger than the standard deviation?\n"
            "2. If not, can you confidently say one is better?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A: F1 = 0.80 ± 0.06. Model B: F1 = 0.79 ± 0.02. Can you confidently say Model A is better?",
                options=[
                    "Yes — 0.80 > 0.79",
                    "No — the difference (0.01) is smaller than Model A's variance (0.06), so they may be equivalent",
                    "Yes — higher mean always means better",
                    "Cannot determine without seeing the test set",
                ],
                correct_index=1,
                explanation=(
                    "The difference (0.01) is much smaller than Model A's standard deviation (0.06). "
                    "On different folds, Model A could easily score 0.74 while Model B scores 0.81. "
                    "They are likely statistically equivalent."
                ),
            ),
        ],
        takeaways=[
            "CV comparison: report mean ± std",
            "If std > difference, the models are statistically similar",
            "Use paired statistical tests for formal significance",
            "More folds give more reliable estimates but take longer",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_08", title="Classification Model Comparison",
        section="model_comparison", order=8, difficulty="intermediate",
        objectives=[
            "Compare classifiers with multiple metrics",
            "Create comparison tables and visualisations",
            "Interpret disagreements between metrics",
        ],
        concept=(
            "For classification: compare accuracy, precision, recall, F1, and AUC across "
            "all models. Use bar charts and comparison tables. If models rank differently "
            "on different metrics, investigate which metric matters most for the problem."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n"
            "                             f1_score, roc_auc_score)\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "comparison = []\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    y_pred = model.predict(X_test)\n"
            "    y_prob = model.predict_proba(X_test)[:, 1]\n"
            "    comparison.append({\n"
            "        'Model': name,\n"
            "        'Accuracy': accuracy_score(y_test, y_pred),\n"
            "        'Precision': precision_score(y_test, y_pred),\n"
            "        'Recall': recall_score(y_test, y_pred),\n"
            "        'F1': f1_score(y_test, y_pred),\n"
            "        'AUC': roc_auc_score(y_test, y_prob)\n"
            "    })\n\n"
            "df = pd.DataFrame(comparison).set_index('Model')\n"
            "print(df.round(3))\n"
            "df.plot(kind='bar', figsize=(12, 5))\n"
            "plt.title('Classification Model Comparison')\n"
            "plt.ylabel('Score')\n"
            "plt.ylim(0, 1)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Comparing only accuracy (misses class-specific performance)",
            "Not visualising the comparison (tables alone are hard to read)",
            "Ignoring per-class differences in multiclass problems",
        ],
        practice_exercise=(
            "Compare 4 classifiers with all 5 metrics. Create a bar chart. "
            "1. Do all models rank the same across all metrics?\n"
            "2. If not, which metric best represents this problem?\n"
            "3. Which model would you choose and why?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A has higher accuracy but lower recall than Model B. What does this mean?",
                options=[
                    "Model A is better overall",
                    "Model B catches more positive cases but makes more false predictions",
                    "Model A is always preferable",
                    "Both models are equivalent",
                ],
                correct_index=1,
                explanation=(
                    "Higher accuracy with lower recall means Model A correctly classifies more "
                    "samples overall but misses more positive cases. Model B catches more positives "
                    "(higher recall) but may have more false positives. The right choice depends "
                    "on whether catching positives or overall accuracy matters more."
                ),
            ),
        ],
        takeaways=[
            "Compare all relevant metrics (accuracy, precision, recall, F1, AUC)",
            "Visualise with bar charts or radar plots",
            "Check per-class performance for multiclass problems",
            "If metrics disagree, choose based on business requirements",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_09", title="Regression Model Comparison",
        section="model_comparison", order=9, difficulty="intermediate",
        objectives=[
            "Compare regressors with R², MAE, RMSE",
            "Create comparison visualisations",
            "Analyse residual patterns across models",
        ],
        concept=(
            "For regression: compare R² (explained variance), MAE (average error), and "
            "RMSE (outlier-sensitive error). Also compare residual patterns — a model with "
            "higher R² but systematic residual bias may be less reliable."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "comparison = []\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    y_pred = model.predict(X_test)\n"
            "    comparison.append({\n"
            "        'Model': name,\n"
            "        'R²': r2_score(y_test, y_pred),\n"
            "        'MAE': mean_absolute_error(y_test, y_pred),\n"
            "        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "    })\n\n"
            "df = pd.DataFrame(comparison).set_index('Model')\n"
            "print(df.round(3))\n"
            "```"
        ),
        common_mistakes=[
            "Only reporting R² (misses the magnitude of errors)",
            "Not comparing residual distributions (a model with systematic bias is unreliable)",
            "Ignoring model complexity in the comparison",
        ],
        practice_exercise=(
            "Compare Linear Regression and Random Forest on a regression dataset. "
            "1. Which has higher R²? Lower MAE?\n"
            "2. Plot residuals for both. Which model has more random residuals?\n"
            "3. What does the residual pattern tell you?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A: R²=0.85, MAE=£15,000. Model B: R²=0.82, MAE=£12,000. Which is better?",
                options=[
                    "Model A — higher R²",
                    "Model B — lower MAE means smaller typical errors",
                    "It depends on whether large errors or typical errors matter more",
                    "Both are equivalent",
                ],
                correct_index=2,
                explanation=(
                    "Model A explains more variance (higher R²) but has larger typical errors "
                    "(higher MAE). Model B has smaller typical errors but less explained variance. "
                    "RMSE would reveal if Model A has a few very large errors. The choice depends "
                    "on whether outliers or typical performance matter more."
                ),
            ),
        ],
        takeaways=[
            "Compare R², MAE, RMSE — they capture different aspects of performance",
            "Check residual patterns across models",
            "Consider model complexity alongside accuracy",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_10", title="Accuracy vs F1",
        section="model_comparison", order=10, difficulty="intermediate",
        objectives=[
            "Understand when accuracy and F1 diverge",
            "Choose the right metric for class balance",
            "Interpret metric disagreements",
        ],
        concept=(
            "Accuracy and F1 agree on balanced datasets but diverge on imbalanced data. "
            "A model that always predicts the majority class can have high accuracy but F1=0 "
            "on the minority class. The divergence reveals class-specific performance issues."
        ),
        why_matters=(
            "For imbalanced data (fraud detection, disease screening), accuracy is dangerously "
            "misleading. A model with 99% accuracy may catch 0% of the minority class. "
            "F1 reveals this failure."
        ),
        example=(
            "Fraud detection (1% fraud):\n"
            "• Model A: accuracy=99.0%, F1=0.00 (always predicts 'not fraud')\n"
            "• Model B: accuracy=96.5%, F1=0.65 (catches some fraud)\n"
            "Model A's accuracy is misleading — it's useless. Model B is far better."
        ),
        practice_exercise=(
            "Create a dataset with 95% class 0 and 5% class 1. Train a model. "
            "1. What is accuracy? What is F1?\n"
            "2. Is accuracy misleading here?\n"
            "3. What does F1 tell you that accuracy doesn't?"
        ),
        common_mistakes=[
            "Not checking both metrics",
            "Ignoring the class imbalance",
            "Choosing accuracy for imbalanced problems",
        ],
        quiz=[
            QuizQuestion(
                question="Accuracy is 98% but F1 is 0.30. What does this tell you?",
                options=[
                    "The model is excellent — 98% accuracy",
                    "The model is good at predicting the majority class but poor at predicting the minority class",
                    "The model is broken",
                    "F1 is always lower than accuracy",
                ],
                correct_index=1,
                explanation=(
                    "High accuracy with low F1 means the model correctly classifies most majority "
                    "class samples (boosting accuracy) but fails on the minority class (low F1). "
                    "This is typical for imbalanced datasets where accuracy is misleading."
                ),
            ),
        ],
        takeaways=[
            "Accuracy: overall correctness across all classes",
            "F1: balance of precision and recall on the positive class",
            "For imbalanced data, F1 is far more informative than accuracy",
            "If accuracy >> F1, the model is ignoring the minority class",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_11", title="MAE vs RMSE",
        section="model_comparison", order=11, difficulty="intermediate",
        objectives=[
            "Compare RMSE and MAE meaningfully",
            "Use their ratio to detect outlier influence",
            "Choose based on error tolerance",
        ],
        concept=(
            "RMSE > MAE always. The RMSE/MAE ratio reveals the error distribution:\n"
            "• Ratio ≈ 1.0: errors are evenly distributed (no dominant outliers)\n"
            "• Ratio >> 1.0: a few large errors dominate (outlier influence)\n"
            "Use MAE when you want the typical error. Use RMSE when large errors are costly."
        ),
        why_matters=(
            "Different models have different error patterns. One model might have many small "
            "errors (low MAE) while another has few but large errors (low RMSE). "
            "Reporting only one metric hides this information."
        ),
        example=(
            "House price prediction:\n"
            "• Model A: MAE=£18K, RMSE=£22K (ratio=1.22 — few large errors)\n"
            "• Model B: MAE=£16K, RMSE=£35K (ratio=2.19 — many large errors)\n"
            "Model A has higher MAE but is more consistent. Model B's average error is smaller "
            "but it occasionally makes very wrong predictions."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error\n"
            "import numpy as np\n\n"
            "for name, model in models.items():\n"
            "    y_pred = model.predict(X_test)\n"
            "    mae = mean_absolute_error(y_test, y_pred)\n"
            "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "    ratio = rmse / mae\n"
            "    print(f'{name:25s} MAE={mae:,.0f}  RMSE={rmse:,.0f}  Ratio={ratio:.2f}')\n"
            "```"
        ),
        common_mistakes=[
            "Reporting only one metric (RMSE or MAE)",
            "Not comparing the RMSE/MAE ratio",
            "Choosing RMSE without understanding its outlier sensitivity",
        ],
        practice_exercise=(
            "Calculate MAE and RMSE for two regression models. "
            "1. What is the RMSE/MAE ratio for each?\n"
            "2. Which model has more outlier predictions?\n"
            "3. If large errors are unacceptable, which model is better?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A: RMSE/MAE = 1.1. Model B: RMSE/MAE = 2.5. What does this tell you?",
                options=[
                    "Model B is more accurate",
                    "Model B has some very large errors that are inflating RMSE",
                    "Model A has larger errors overall",
                    "The ratio is meaningless",
                ],
                correct_index=1,
                explanation=(
                    "A high RMSE/MAE ratio means a few very large errors are inflating RMSE. "
                    "Model B's errors are unevenly distributed — mostly small with a few huge ones. "
                    "Model A's errors are more consistent."
                ),
            ),
        ],
        takeaways=[
            "RMSE/MAE ratio reveals outlier influence",
            "Ratio ≈ 1: consistent errors; Ratio >> 1: few very large errors",
            "MAE for typical error, RMSE when large errors are costly",
            "Always report both for a complete picture",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_12", title="Comparing Training Time",
        section="model_comparison", order=12, difficulty="intermediate",
        objectives=[
            "Measure and compare training and prediction times",
            "Consider time as a model selection factor",
            "Balance speed with accuracy",
        ],
        concept=(
            "Training time varies enormously: LR (ms) → RF (s) → GB (min) → SVM (hours). "
            "For prototyping, fast models help iterate. For production, both training and "
            "prediction time matter. Time is a real cost."
        ),
        python_example=(
            "```python\n"
            "import time\n\n"
            "print(f'{\"Model\":25s} {\"Train\":>10s} {\"Predict\":>10s}')\n"
            "print('-' * 47)\n\n"
            "for name, model in models.items():\n"
            "    start = time.time()\n"
            "    model.fit(X_train, y_train)\n"
            "    train_time = time.time() - start\n"
            "\n"
            "    start = time.time()\n"
            "    for _ in range(100):\n"
            "        model.predict(X_test)\n"
            "    pred_time = (time.time() - start) / 100\n"
            "\n"
            "    print(f'{name:25s} {train_time:8.3f}s {pred_time*1000:8.2f}ms')\n"
            "```"
        ),
        common_mistakes=[
            "Ignoring time requirements until deployment",
            "Only measuring training time (prediction time matters for real-time)",
            "Not considering deployment constraints (memory, CPU)",
        ],
        practice_exercise=(
            "Measure training and prediction time for 3 models. "
            "1. Which trains fastest? Which predicts fastest?\n"
            "2. If you need real-time prediction (<10ms), which models are viable?\n"
            "3. If you retrain daily, which training time is acceptable?"
        ),
        quiz=[
            QuizQuestion(
                question="A web API must predict in <5ms. KNN predicts in 150ms. What should you do?",
                options=[
                    "Use KNN anyway — it's more accurate",
                    "Use a faster model (LR, RF, GB) that meets the latency requirement",
                    "Increase the timeout to 200ms",
                    "Reduce the dataset size",
                ],
                correct_index=1,
                explanation=(
                    "Production requirements are constraints, not suggestions. If prediction "
                    "must be under 5ms, KNN is disqualified regardless of accuracy. Choose a "
                    "model that meets the speed requirement."
                ),
            ),
        ],
        takeaways=[
            "Fast training helps prototyping (LR, DT)",
            "Fast prediction matters for production (LR, RF, GB are fast; KNN is slow)",
            "Always measure both training and prediction time",
            "Production constraints can override accuracy preferences",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_13", title="Comparing Complexity",
        section="model_comparison", order=13, difficulty="intermediate",
        objectives=[
            "Assess model complexity across multiple dimensions",
            "Understand the accuracy-complexity tradeoff",
            "Justify complexity with meaningful performance gains",
        ],
        concept=(
            "Model complexity includes:\n"
            "• Number of parameters (LR: n_features; RF: thousands of trees × nodes)\n"
            "• Training time and prediction time\n"
            "• Interpretability (LR: full; RF: partial; GB: low)\n"
            "• Deployment difficulty (memory, dependencies)\n"
            "Simpler models are easier to maintain, debug, and explain."
        ),
        why_matters=(
            "Complexity has real costs: slower training, harder debugging, less "
            "interpretability, more maintenance. A slightly less accurate but simpler model "
            "is often preferable in production."
        ),
        example=(
            "Model comparison:\n"
            "• Logistic Regression: F1=0.77, 0.1s train, fully interpretable, 10KB\n"
            "• Random Forest: F1=0.81, 5s train, partially interpretable, 50MB\n"
            "• Gradient Boosting: F1=0.83, 30s train, not interpretable, 100MB\n"
            "The 6% improvement from LR to GB costs 300x more training time and 10,000x more memory."
        ),
        common_mistakes=[
            "Choosing the most complex model by default",
            "Not considering maintenance costs (updates, debugging, monitoring)",
            "Ignoring that small accuracy gains may not justify complexity",
        ],
        practice_exercise=(
            "Compare models on complexity: parameters, training time, memory, interpretability. "
            "1. Create a complexity comparison table.\n"
            "2. If two models have similar accuracy, which would you prefer?\n"
            "3. What is the smallest accuracy gain that justifies a more complex model?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A (LR) achieves F1=0.80. Model B (GB) achieves F1=0.82. Model B takes 50x longer to train and is not interpretable. Should you always choose Model B?",
                options=[
                    "Yes — higher F1 is always better",
                    "No — the 2% gain may not justify the costs (training time, complexity, lack of interpretability)",
                    "Yes — accuracy is the most important factor",
                    "Neither — use a different model",
                ],
                correct_index=1,
                explanation=(
                    "A 2% F1 gain with 50x training time and loss of interpretability is "
                    "rarely justified unless accuracy is critical. The simpler model may be "
                    "better for deployment, maintenance, and explainability."
                ),
            ),
        ],
        takeaways=[
            "Simpler models are preferred when accuracy is similar",
            "Complexity has real costs: maintenance, debugging, explainability",
            "Justify complexity with meaningful performance gains",
            "Consider the full lifecycle, not just accuracy",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_14", title="Final Model Selection",
        section="model_comparison", order=14, difficulty="advanced",
        objectives=[
            "Make the final model selection decision",
            "Balance all factors: performance, speed, interpretability, maintainability",
            "Document and justify the choice",
        ],
        concept=(
            "Final model selection balances:\n"
            "1. Performance: CV scores (mean ± std)\n"
            "2. Stability: low variance across CV folds\n"
            "3. Speed: training and prediction time\n"
            "4. Interpretability: can you explain decisions?\n"
            "5. Maintainability: how complex to deploy and update?\n"
            "6. Stakeholder requirements: what do they actually need?"
        ),
        why_matters=(
            "The model with the highest test score isn't always the best production model. "
            "A model that's 1% more accurate but 10x slower, impossible to explain, and "
            "hard to deploy is usually the wrong choice."
        ),
        python_example=(
            "```python\n"
            "# Complete comparison framework\n"
            "import time\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "results = []\n"
            "for name, model in models.items():\n"
            "    # Cross-validation scores\n"
            "    scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
            "\n"
            "    # Training time\n"
            "    start = time.time()\n"
            "    model.fit(X_train, y_train)\n"
            "    train_time = time.time() - start\n"
            "\n"
            "    results.append({\n"
            "        'Model': name,\n"
            "        'F1': f'{scores.mean():.3f} ± {scores.std():.3f}',\n"
            "        'Train Time': f'{train_time:.2f}s'\n"
            "    })\n"
            "\n"
            "df = pd.DataFrame(results)\n"
            "print(df.to_string(index=False))\n"
            "```"
        ),
        common_mistakes=[
            "Chasing the last 0.1% accuracy at the cost of everything else",
            "Not considering production constraints (speed, memory, latency)",
            "Not documenting the decision process",
        ],
        practice_exercise=(
            "Complete a model selection exercise:\n"
            "1. Compare 4 models on all relevant metrics and time.\n"
            "2. Create a summary table.\n"
            "3. Write a one-paragraph justification for your final choice.\n"
            "4. What would change your mind?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A: F1=0.84, 10ms prediction, interpretable. Model B: F1=0.85, 200ms prediction, black box. Production needs <50ms latency and regulatory explainability. Which do you choose?",
                options=[
                    "Model B — higher F1",
                    "Model A — it meets both speed and interpretability requirements",
                    "A compromise — train a simpler version of Model B",
                    "Neither — retrain both with different parameters",
                ],
                correct_index=1,
                explanation=(
                    "Model B fails both requirements: 200ms > 50ms latency, and it's not "
                    "interpretable. Model A meets both constraints while performing nearly as "
                    "well. Requirements are non-negotiable; accuracy differences must be "
                    "evaluated within constraints."
                ),
            ),
        ],
        takeaways=[
            "Balance performance, speed, interpretability, and maintainability",
            "Requirements are constraints, not suggestions",
            "Document why you chose the model — future you will thank present you",
            "Small accuracy gains rarely justify major complexity increases",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="cmp_15", title="Model Complexity",
        section="model_comparison", order=15, difficulty="intermediate",
        objectives=[
            "Assess model complexity across dimensions",
            "Understand the accuracy-complexity tradeoff",
            "Choose the simplest adequate model",
        ],
        concept=(
            "Model complexity encompasses:\n"
            "• Number of parameters: LR (n+1) vs RF (hundreds of thousands)\n"
            "• Training time: LR (ms) vs GB (minutes)\n"
            "• Interpretability: LR (coefficients) vs GB (not directly interpretable)\n"
            "• Deployment: LR (tiny model file) vs GB (large model file)\n"
            "The goal is the simplest model that performs well enough."
        ),
        why_matters=(
            "Complexity has real costs: slower training, harder debugging, less "
            "interpretability, more maintenance burden. A model with 100,000 parameters "
            "for a 1000-sample problem is overkill."
        ),
        example=(
            "Occam's Razor in ML:\n"
            "• Linear Regression: R²=0.78, 11 parameters, 0.01s training\n"
            "• Random Forest: R²=0.81, 50,000+ parameters, 10s training\n"
            "• Gradient Boosting: R²=0.83, 200,000+ parameters, 60s training\n"
            "If 78% accuracy is sufficient, the simpler model is preferable."
        ),
        common_mistakes=[
            "Using complex models on small datasets (overfitting)",
            "Assuming more parameters always means better performance",
            "Ignoring maintenance and deployment complexity",
        ],
        practice_exercise=(
            "Estimate the number of 'parameters' in Linear Regression, Decision Tree "
            "(depth=5), and Random Forest (100 trees, depth=10). "
            "1. How do they compare?\n"
            "2. Does more complexity guarantee better performance?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the principle of parsimony (Occam's Razor) in model selection?",
                options=[
                    "Always choose the most complex model",
                    "Among models with similar performance, prefer the simpler one",
                    "Simpler models are always more accurate",
                    "Complexity doesn't matter",
                ],
                correct_index=1,
                explanation=(
                    "Occam's Razor says: when models perform similarly, prefer the simpler one. "
                    "It's easier to understand, debug, deploy, and maintain. Complexity should "
                    "be justified by meaningful performance gains."
                ),
            ),
        ],
        takeaways=[
            "The goal is the simplest model that performs well enough",
            "Complexity has real costs: maintenance, debugging, explainability",
            "Occam's Razor: prefer simplicity when performance is similar",
            "Justify complexity with meaningful, reliable performance gains",
        ],
        lab_module="model_comparison",
    ),
]
