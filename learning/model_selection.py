"""Model Selection curriculum — 23 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="sel_01", title="What is Model Selection?",
        section="model_selection", order=1, difficulty="beginner",
        objectives=[
            "Define model selection and its purpose",
            "Understand the 'No Free Lunch' theorem",
            "Know why no single model works best everywhere",
        ],
        concept=(
            "Model selection is choosing the best algorithm and hyperparameters for your "
            "specific problem. The 'No Free Lunch' theorem proves no model works best on "
            "all problems. Every algorithm makes assumptions — the right model matches the "
            "assumptions to your data."
        ),
        why_matters=(
            "Different models have different strengths. A decision tree is interpretable but "
            "may underperform; gradient boosting is accurate but complex. Model selection "
            "finds the right trade-off for your specific requirements."
        ),
        simple_explanation=(
            "Model selection is choosing the right tool for the job. A hammer isn't always "
            "better than a screwdriver — it depends on whether you're hitting nails or turning screws."
        ),
        example=(
            "Same dataset, different results:\n"
            "• Linear Regression: R² = 0.60 (fast, interpretable, misses non-linear patterns)\n"
            "• Random Forest: R² = 0.80 (slower, less interpretable, captures interactions)\n"
            "• Gradient Boosting: R² = 0.83 (slowest, least interpretable, best accuracy)\n"
            "Which is 'best' depends on whether you prioritise speed, interpretability, or accuracy."
        ),
        practice_exercise=(
            "Load a dataset and train three different models (Linear Regression, Random Forest, "
            "Gradient Boosting). Record R², MAE, and training time for each. "
            "1. Which model has the best R²?\n"
            "2. Which trains fastest?\n"
            "3. Which would you choose and why?"
        ),
        quiz=[
            QuizQuestion(
                question="The 'No Free Lunch' theorem states that:",
                options=[
                    "Neural networks are always the best model",
                    "No single algorithm works best on every problem",
                    "More data always improves model performance",
                    "Simpler models are always better",
                ],
                correct_index=1,
                explanation=(
                    "The No Free Lunch theorem proves that no algorithm can outperform all "
                    "others across all possible problems. Every algorithm has strengths and "
                    "weaknesses — the 'best' model depends on the specific data and task."
                ),
            ),
        ],
        common_mistakes=[
            "Using the same model for every problem",
            "Not comparing models objectively",
            "Choosing the most complex model by default",
        ],
        takeaways=[
            "No Free Lunch: no model wins everywhere",
            "Always compare multiple models",
            "Choose based on data characteristics and requirements, not reputation",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_02", title="Baseline Models",
        section="model_selection", order=2, difficulty="beginner",
        objectives=[
            "Establish a performance baseline",
            "Use DummyClassifier/DummyRegressor as benchmarks",
            "Understand the minimum performance bar",
        ],
        concept=(
            "A baseline is the simplest reasonable model. For classification: always predict "
            "the majority class. For regression: always predict the mean. Every model you "
            "build must beat the baseline — otherwise the complex model adds no value."
        ),
        why_matters=(
            "Without a baseline, you don't know if your complex model is actually useful. "
            "A model with 80% accuracy sounds good until you learn the baseline is 78% — "
            "your complex model only gained 2%."
        ),
        example=(
            "Titanic survival prediction:\n"
            "• Baseline (majority class): accuracy = 61.6% (always predict 'died')\n"
            "• Logistic Regression: accuracy = 80.2% (+18.6%)\n"
            "• Random Forest: accuracy = 82.1% (+20.5%)\n"
            "Without the baseline, you wouldn't know how much improvement you've achieved."
        ),
        python_example=(
            "```python\n"
            "from sklearn.dummy import DummyClassifier, DummyRegressor\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "# Classification baseline: always predict majority class\n"
            "clf_baseline = DummyClassifier(strategy='most_frequent', random_state=42)\n"
            "scores = cross_val_score(clf_baseline, X, y, cv=5, scoring='accuracy')\n"
            "print(f'Classification baseline: {scores.mean():.3f}')\n\n"
            "# Regression baseline: always predict the mean\n"
            "reg_baseline = DummyRegressor(strategy='mean')\n"
            "scores = cross_val_score(reg_baseline, X, y, cv=5, scoring='r2')\n"
            "print(f'Regression baseline R²: {scores.mean():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Not establishing a baseline before building complex models",
            "Being satisfied with small improvements over baseline (2% may not justify complexity)",
            "Using a complex model as the baseline (defeats the purpose)",
        ],
        practice_exercise=(
            "Create a baseline for a classification dataset. "
            "1. What accuracy does the majority-class baseline achieve?\n"
            "2. Now train Logistic Regression. How much better is it?\n"
            "3. Is the improvement worth the added complexity?"
        ),
        quiz=[
            QuizQuestion(
                question="A regression dataset has target values with mean £250,000. What is the baseline R²?",
                options=["1.0", "0.5", "0.0", "-1.0"],
                correct_index=2,
                explanation=(
                    "A model that always predicts the mean has R² = 0. This is because "
                    "SS_residuals = SS_total when you always predict the mean, so R² = 1 - SS_res/SS_tot = 0."
                ),
            ),
        ],
        takeaways=[
            "Always start with a simple baseline",
            "Every model must beat the baseline to justify its complexity",
            "Baseline = majority class (classification) or mean (regression)",
            "Use sklearn's DummyClassifier/DummyRegressor",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_03", title="Classification Algorithm Selection",
        section="model_selection", order=3, difficulty="intermediate",
        objectives=[
            "Match classification algorithms to problem characteristics",
            "Understand when to use each algorithm",
            "Make informed algorithm choices",
        ],
        concept=(
            "Algorithm selection depends on multiple factors:\n"
            "• Dataset size: small → LR, KNN; large → GB, RF\n"
            "• Feature types: mixed → trees; numerical → LR, SVM\n"
            "• Interpretability: needed → LR, DT; not needed → RF, GB\n"
            "• Training time: limited → LR; abundant → SVM, GB\n"
            "• Accuracy: maximum → GB, RF; acceptable → LR"
        ),
        why_matters=(
            "Choosing the right algorithm saves time and produces better results. Running "
            "all seven classifiers on every problem is thorough but wasteful. Understanding "
            "which algorithms suit your data lets you focus on the most promising candidates."
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "classifiers = {\n"
            "    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),\n"
            "    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n"
            "    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),\n"
            "    'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),\n"
            "}\n\n"
            "for name, clf in classifiers.items():\n"
            "    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')\n"
            "    print(f'{name:25s}: F1 = {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Always using Random Forest — sometimes simpler is better",
            "Not considering interpretability requirements",
            "Ignoring training time constraints for real-time applications",
        ],
        practice_exercise=(
            "Compare Logistic Regression, KNN, Random Forest, and Gradient Boosting on a "
            "classification dataset using 5-fold cross-validation. "
            "1. Which has the highest F1?\n"
            "2. Which trains fastest?\n"
            "3. If interpretability matters, which would you choose?"
        ),
        quiz=[
            QuizQuestion(
                question="You have a small dataset (200 samples, 10 features) and need an interpretable model. Which is most appropriate?",
                options=[
                    "Gradient Boosting with 500 trees",
                    "SVM with RBF kernel",
                    "Logistic Regression",
                    "Random Forest with max_depth=None",
                ],
                correct_index=2,
                explanation=(
                    "Logistic Regression is ideal: it works well on small datasets, is fully "
                    "interpretable (coefficients show feature importance), and trains in milliseconds. "
                    "Complex models like GB and RF overfit small datasets and are hard to interpret."
                ),
            ),
        ],
        takeaways=[
            "Start simple (LR), try ensembles (RF, GB), compare with cross-validation",
            "Match algorithm to data size, feature types, and requirements",
            "Consider interpretability, training time, and prediction time",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_04", title="Regression Algorithm Selection",
        section="model_selection", order=4, difficulty="intermediate",
        objectives=[
            "Match regression algorithms to problem characteristics",
            "Understand when linear vs non-linear models are appropriate",
            "Choose based on interpretability and accuracy needs",
        ],
        concept=(
            "Regression algorithm selection:\n"
            "• Linear Regression: baseline, linear relationships, interpretable\n"
            "• Ridge/Lasso: multicollinearity, feature selection (Lasso)\n"
            "• Decision Tree: non-linear, interpretable, overfits easily\n"
            "• Random Forest: non-linear, robust, good default\n"
            "• Gradient Boosting: highest accuracy, most complex\n"
            "• KNN: local patterns, simple baseline"
        ),
        why_matters=(
            "Different regression problems have different structures. Linear relationships "
            "are best served by linear models; complex non-linear patterns need tree-based "
            "methods. The choice affects both accuracy and interpretability."
        ),
        example=(
            "California Housing prediction:\n"
            "• Linear Regression: R²=0.60 (misses non-linear geography effects)\n"
            "• Ridge: R²=0.62 (handles multicollinearity)\n"
            "• Random Forest: R²=0.80 (captures non-linear patterns)\n"
            "• Gradient Boosting: R²=0.83 (best accuracy)\n"
            "If you need to explain WHY a prediction was made, choose Ridge."
        ),
        common_mistakes=[
            "Using complex models when linear regression already works well",
            "Not checking if relationships are linear first (plot scatter plots)",
            "Ignoring interpretability needs in regulated industries",
        ],
        practice_exercise=(
            "Fit Linear Regression, Ridge, Random Forest, and Gradient Boosting on a "
            "regression dataset. Compare R², MAE, RMSE. "
            "1. How much does accuracy improve from Linear to Gradient Boosting?\n"
            "2. Is the improvement worth the loss of interpretability?"
        ),
        quiz=[
            QuizQuestion(
                question="When should you prefer Lasso over Ridge regression?",
                options=[
                    "When the dataset is very large",
                    "When you suspect many features are irrelevant and want automatic feature selection",
                    "When the target is categorical",
                    "When the data is not scaled",
                ],
                correct_index=1,
                explanation=(
                    "Lasso drives some coefficients exactly to zero, effectively removing "
                    "features. This makes it ideal when you suspect many features are noise. "
                    "Ridge shrinks all coefficients but never removes any."
                ),
            ),
        ],
        takeaways=[
            "Linear regression first as a baseline",
            "Try Ridge/Lasso when multicollinearity or feature selection is needed",
            "Random Forest/Gradient Boosting for maximum accuracy",
            "Balance accuracy with interpretability",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_05", title="Dataset Size and Algorithm Selection",
        section="model_selection", order=5, difficulty="intermediate",
        objectives=[
            "Match algorithms to dataset sizes",
            "Understand sample complexity",
            "Choose appropriate models for data availability",
        ],
        concept=(
            "Dataset size strongly affects which algorithms work well:\n"
            "• Small (<1K samples): simple models (LR, KNN) — complex models overfit\n"
            "• Medium (1K-100K): tree ensembles (RF, GB) — sweet spot for most algorithms\n"
            "• Large (>100K): GB, neural networks — enough data for complex patterns\n"
            "The feature-to-sample ratio also matters: 100 features with 500 samples is risky."
        ),
        why_matters=(
            "Complex models need more data to learn effectively. Using gradient boosting "
            "with 50 samples overfits immediately; using linear regression with 1 million "
            "samples may miss complex patterns. Matching algorithm to data size is crucial."
        ),
        example=(
            "Three scenarios:\n"
            "1. Medical study: 80 patients, 50 features → Logistic Regression (simple, less overfitting)\n"
            "2. E-commerce: 50,000 customers, 20 features → Random Forest (captures interactions)\n"
            "3. Web-scale: 10M records, 200 features → Gradient Boosting (maximum accuracy)"
        ),
        common_mistakes=[
            "Using deep learning on small datasets (<1K samples)",
            "Using linear models on very large datasets with complex patterns",
            "Not considering the feature-to-sample ratio",
        ],
        practice_exercise=(
            "Train Random Forest with varying dataset sizes (100, 500, 1000, 5000, all). "
            "1. How does R² change with more data?\n"
            "2. At what size does the model stabilise?\n"
            "3. What happens with only 100 samples?"
        ),
        quiz=[
            QuizQuestion(
                question="You have 150 samples and 200 features. What is the biggest risk?",
                options=[
                    "The model will be too slow",
                    "The model will overfit because there are more features than samples",
                    "The model will underfit",
                    "The model will crash",
                ],
                correct_index=1,
                explanation=(
                    "More features than samples (p > n) means the model has too many degrees of "
                    "freedom. It can perfectly fit the training data by memorising noise, "
                    "leading to severe overfitting. Use feature selection or regularisation."
                ),
            ),
        ],
        takeaways=[
            "Small data → simple models, large data → complex models",
            "Feature-to-sample ratio matters as much as absolute size",
            "Start simple, add complexity only if justified by cross-validation",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_06", title="Numerical vs Categorical Features",
        section="model_selection", order=6, difficulty="intermediate",
        objectives=[
            "Choose algorithms based on feature types",
            "Understand algorithm requirements for encoding/scaling",
            "Prepare data appropriately for each algorithm",
        ],
        concept=(
            "Different algorithms handle feature types differently:\n"
            "• Trees (RF, GB, DT): handle mixed types natively, no scaling needed\n"
            "• Linear models (LR, Ridge): need numerical input, scaling helps\n"
            "• Distance-based (KNN, SVM): need scaling AND numerical encoding\n"
            "Mixed types need ColumnTransformer to apply different preprocessing per feature type."
        ),
        why_matters=(
            "Using the wrong preprocessing for an algorithm produces poor results. "
            "One-hot encoding with linear models is fine; one-hot encoding with Random Forest "
            "wastes memory. Not scaling for KNN makes distance calculations meaningless."
        ),
        example=(
            "Dataset with age (numerical), income (numerical), gender (categorical):\n"
            "• KNN: StandardScaler on age+income, OneHotEncode gender\n"
            "• Random Forest: pass raw features (handles categoricals natively)\n"
            "• Linear Regression: StandardScaler on all, OneHotEncode gender"
        ),
        common_mistakes=[
            "One-hot encoding for tree-based models (not needed, wastes memory)",
            "Not scaling for distance-based models (KNN, SVM)",
            "Using label encoding for nominal data with linear models (implies false ordering)",
        ],
        practice_exercise=(
            "Load a dataset with mixed numerical and categorical features. "
            "1. Which features are which type?\n"
            "2. What preprocessing does KNN need?\n"
            "3. What preprocessing does Random Forest need?"
        ),
        quiz=[
            QuizQuestion(
                question="Why doesn't Random Forest need feature scaling?",
                options=[
                    "Random Forest uses distance-based calculations",
                    "Random Forest makes splits based on feature thresholds, not distances",
                    "Random Forest automatically scales features",
                    "Random Forest only works with categorical features",
                ],
                correct_index=1,
                explanation=(
                    "Random Forest splits on feature thresholds (e.g., age > 30), not distances. "
                    "Whether age ranges from 0-100 or 0-1000 doesn't affect the split quality — "
                    "only the threshold value changes. Distance-based models like KNN are affected."
                ),
            ),
        ],
        takeaways=[
            "Trees handle mixed types natively, no scaling needed",
            "Distance-based models (KNN, SVM) require scaling and numerical encoding",
            "Linear models need numerical input with encoding for categoricals",
            "Use ColumnTransformer for mixed preprocessing strategies",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_07", title="Interpretability",
        section="model_selection", order=7, difficulty="intermediate",
        objectives=[
            "Understand the accuracy-interpretability tradeoff",
            "Choose interpretable models when required",
            "Apply explainability techniques for black-box models",
        ],
        concept=(
            "Model interpretability spectrum:\n"
            "• Fully interpretable: Linear Regression (coefficients), Decision Trees (rules)\n"
            "• Partially interpretable: Random Forest (feature importance)\n"
            "• Black box: Gradient Boosting, SVM, Neural Networks\n\n"
            "In healthcare, finance, and law, interpretability may be legally required."
        ),
        why_matters=(
            "A doctor needs to know WHY a model predicts cancer, not just that it does. "
            "A bank must explain why a loan was denied. An interpretable model with 82% accuracy "
            "may be preferable to a black-box model with 85% if explanations are required."
        ),
        example=(
            "Credit scoring:\n"
            "• Logistic Regression: 'Loan denied because income=£18K (coefficient=-0.3) and "
            "debt_ratio=0.85 (coefficient=-0.2)' — fully explainable\n"
            "• Gradient Boosting: 'Loan denied' — no inherent explanation\n"
            "• Use SHAP values with GB: 'Loan denied mainly because debt_ratio is high' "
            "— post-hoc explanation, but less trustworthy"
        ),
        common_mistakes=[
            "Choosing the most accurate model without considering interpretability",
            "Not using SHAP/LIME for black-box explainability",
            "Assuming accuracy is always the priority",
        ],
        practice_exercise=(
            "Train both Logistic Regression and Random Forest on a classification task. "
            "1. Extract and interpret the LR coefficients — which features matter?\n"
            "2. Compare with RF feature importance — are they similar?\n"
            "3. Which model would you choose for a regulated industry?"
        ),
        quiz=[
            QuizQuestion(
                question="A bank must explain every loan rejection to customers. Which model is most appropriate?",
                options=[
                    "Deep neural network (highest accuracy)",
                    "Logistic Regression (coefficients explain each decision)",
                    "SVM with RBF kernel (best for tabular data)",
                    "Ensemble of 10 models (most robust)",
                ],
                correct_index=1,
                explanation=(
                    "Logistic Regression coefficients directly show how each feature influences "
                    "the decision. A coefficient of -0.3 for income means higher income increases "
                    "approval probability. This is directly explainable to customers."
                ),
            ),
        ],
        takeaways=[
            "Interpretability matters in regulated industries (banking, healthcare, law)",
            "LR and DT are naturally interpretable",
            "Use SHAP/LIME for black-box model explainability",
            "Accuracy and interpretability are often inversely related",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_08", title="Training Time",
        section="model_selection", order=8, difficulty="intermediate",
        objectives=[
            "Consider training time in model selection",
            "Understand algorithm complexity",
            "Choose models that fit time constraints",
        ],
        concept=(
            "Training time varies enormously across algorithms:\n"
            "• Logistic Regression: milliseconds\n"
            "• Decision Tree: milliseconds\n"
            "• KNN: seconds (no training phase — lazy learner)\n"
            "• Random Forest: seconds to minutes\n"
            "• Gradient Boosting: minutes\n"
            "• SVM on large data: minutes to hours\n"
            "For prototyping, fast models help you iterate quickly."
        ),
        why_matters=(
            "In a 2-hour hackathon, you can't afford to wait 30 minutes per model training. "
            "In production, retraining nightly requires algorithms that finish within the window. "
            "Training time is a practical constraint that affects model choice."
        ),
        example=(
            "A data scientist testing 20 hyperparameter combinations:\n"
            "• Logistic Regression (0.1s each): 2 seconds total\n"
            "• Gradient Boosting (10s each): 200 seconds total\n"
            "• SVM (60s each): 20 minutes total\n"
            "LR lets you test quickly, then focus GB tuning on the most promising region."
        ),
        common_mistakes=[
            "Ignoring training time until deployment",
            "Using SVM on large datasets without checking training time",
            "Not considering prediction time separately from training time",
        ],
        practice_exercise=(
            "Time the training of Logistic Regression, Random Forest, and Gradient Boosting "
            "on a dataset. "
            "1. What is the time ratio between fastest and slowest?\n"
            "2. If you had 10x the data, how would training time change?"
        ),
        quiz=[
            QuizQuestion(
                question="For a real-time prediction system (must respond in <10ms), which training consideration matters most?",
                options=[
                    "Training time must be under 1 minute",
                    "Prediction (inference) time must be under 10ms",
                    "The model must have the highest accuracy",
                    "The model must be trained on GPU",
                ],
                correct_index=1,
                explanation=(
                    "For real-time systems, prediction (inference) time matters, not training time. "
                    "A model that takes hours to train but predicts in 0.1ms is fine. "
                    "KNN is slow at prediction (computes distances); LR and RF are fast."
                ),
            ),
        ],
        takeaways=[
            "Fast models for prototyping (LR, DT); thorough models for production (RF, GB)",
            "Consider both training AND prediction time",
            "For real-time: LR and RF have fastest predictions",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_09", title="Prediction Time",
        section="model_selection", order=9, difficulty="intermediate",
        objectives=[
            "Consider prediction latency requirements",
            "Match models to production constraints",
            "Understand model size implications",
        ],
        concept=(
            "Prediction time (inference speed) varies by algorithm:\n"
            "• Fast (<1ms): Logistic Regression, Decision Tree, Random Forest\n"
            "• Medium (1-10ms): Gradient Boosting, SVM (small dataset)\n"
            "• Slow (>10ms): KNN (computes distances to all training points)\n"
            "For web APIs serving millions of requests, even 1ms matters."
        ),
        why_matters=(
            "A model deployed in a mobile app must predict in milliseconds. A fraud detection "
            "system must score transactions before they complete. Prediction speed is often "
            "the binding constraint, not accuracy."
        ),
        example=(
            "Real-time fraud detection (must predict in <50ms):\n"
            "• Gradient Boosting: 2ms ✓\n"
            "• Random Forest: 1ms ✓\n"
            "• KNN (100K training samples): 150ms ✗ (too slow)\n"
            "• Deep Neural Network: 5ms ✓\n"
            "KNN is eliminated purely on speed, regardless of accuracy."
        ),
        common_mistakes=[
            "Only optimising accuracy without considering prediction speed",
            "Not measuring prediction time on realistic data sizes",
            "Using KNN for real-time applications (slow at prediction time)",
        ],
        practice_exercise=(
            "Measure prediction time for each model on a test set of 1000 samples. "
            "1. Which model predicts fastest?\n"
            "2. Which is 10x slower than the fastest?\n"
            "3. Would the slowest model work for a real-time web API?"
        ),
        quiz=[
            QuizQuestion(
                question="KNN trains instantly but is slow at prediction. Why?",
                options=[
                    "KNN uses a complex neural network for prediction",
                    "KNN stores the entire training set and computes distances to ALL points for each prediction",
                    "KNN's predictions require GPU acceleration",
                    "KNN re-trains the model for each prediction",
                ],
                correct_index=1,
                explanation=(
                    "KNN is a 'lazy learner' — it stores all training data and defers computation "
                    "to prediction time. For each new point, it must compute distances to all "
                    "training points, find the k nearest, and average their targets."
                ),
            ),
        ],
        takeaways=[
            "LR and RF have fastest prediction times",
            "KNN is slow at prediction (computes distances to all training points)",
            "Always measure prediction time on realistic data sizes",
            "Match model speed to production requirements",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_10", title="Accuracy vs Interpretability",
        section="model_selection", order=10, difficulty="intermediate",
        objectives=[
            "Navigate the accuracy-interpretability tradeoff",
            "Know when to sacrifice accuracy for interpretability",
            "Justify model choices to stakeholders",
        ],
        concept=(
            "More complex models usually achieve higher accuracy but are harder to interpret. "
            "The optimal choice depends on whether stakeholders need to understand 'why'. "
            "A 2% accuracy gain may not justify switching from an explainable to a black-box model."
        ),
        why_matters=(
            "Accuracy is not the only criterion. In regulated industries, you must explain "
            "decisions. In debugging, you need to understand failures. In deployment, simpler "
            "models are easier to maintain. The accuracy-interpretability trade-off is real."
        ),
        example=(
            "Hospital readmission prediction:\n"
            "• Logistic Regression: 81% accuracy, fully interpretable\n"
            "• Gradient Boosting: 84% accuracy, black box\n"
            "• Decision: LR chosen — doctors need to understand the model's reasoning, "
            "and 3% accuracy loss is acceptable for trust and regulatory compliance."
        ),
        common_mistakes=[
            "Always maximising accuracy without considering interpretability",
            "Not communicating trade-offs to stakeholders",
            "Using a black-box model when interpretability is legally required",
        ],
        practice_exercise=(
            "Train LR and GB on a classification task. Extract LR coefficients and GB "
            "feature importances. "
            "1. Do they agree on which features matter?\n"
            "2. Can you explain a single prediction from each model?\n"
            "3. Which explanation would you present to a non-technical stakeholder?"
        ),
        quiz=[
            QuizQuestion(
                question="Your model achieves 95% accuracy but is a black box. A simpler model achieves 93% and is fully interpretable. Which do you choose?",
                options=[
                    "Always choose the highest accuracy",
                    "It depends — if stakeholders need explanations, choose the simpler model",
                    "Always choose the simpler model",
                    "Train both and let the algorithm decide",
                ],
                correct_index=1,
                explanation=(
                    "The choice depends on requirements. If explanations are needed (healthcare, "
                    "finance), the 93% interpretable model is better. If accuracy is everything "
                    "(competition, ad targeting), the 95% model wins. Always ask stakeholders."
                ),
            ),
        ],
        takeaways=[
            "More complex ≠ always better",
            "Accuracy and interpretability are often inversely related",
            "Know your stakeholders' needs before choosing",
            "A 2% accuracy gain may not justify losing interpretability",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_11", title="Bias and Variance",
        section="model_selection", order=11, difficulty="intermediate",
        objectives=[
            "Understand the bias-variance tradeoff",
            "Diagnose bias vs variance problems",
            "Apply appropriate remedies",
        ],
        concept=(
            "Total prediction error = Bias² + Variance + Irreducible Noise\n"
            "• Bias: error from wrong assumptions (underfitting). The model is too simple.\n"
            "• Variance: error from sensitivity to training data (overfitting). The model "
            "is too complex and learns noise.\n"
            "• Noise: inherent randomness in the data — no model can eliminate this."
        ),
        why_matters=(
            "Understanding whether your model suffers from high bias or high variance tells "
            "you exactly what to do next. High bias → make the model more complex. "
            "High variance → simplify or add more data."
        ),
        example=(
            "Linear Regression on non-linear data:\n"
            "• Train error: high, Test error: high → High bias (underfitting)\n"
            "Remedy: add polynomial features or use a non-linear model.\n\n"
            "Decision Tree with unlimited depth:\n"
            "• Train error: 0, Test error: high → High variance (overfitting)\n"
            "Remedy: limit depth, add regularisation, or get more data."
        ),
        common_mistakes=[
            "Not diagnosing whether the problem is bias or variance before trying fixes",
            "Adding complexity when the problem is overfitting (makes it worse)",
            "Simplifying when the problem is underfitting (makes it worse)",
        ],
        practice_exercise=(
            "Train a model and compute train and test scores. "
            "1. If train=0.95 and test=0.60, is it bias or variance?\n"
            "2. If train=0.65 and test=0.63, is it bias or variance?\n"
            "3. What remedy would you apply in each case?"
        ),
        quiz=[
            QuizQuestion(
                question="You have high training accuracy but low test accuracy. What is the diagnosis?",
                options=[
                    "High bias — the model is too simple",
                    "High variance — the model is overfitting to training data",
                    "High noise — the data is too noisy",
                    "The model is well-calibrated",
                ],
                correct_index=1,
                explanation=(
                    "High training accuracy means the model fits the training data well. "
                    "Low test accuracy means it doesn't generalise. This is the classic "
                    "signature of high variance (overfitting) — the model memorised the "
                    "training data instead of learning patterns."
                ),
            ),
        ],
        takeaways=[
            "High bias = underfitting; High variance = overfitting",
            "More data helps variance, not bias",
            "More complexity helps bias, increases variance",
            "Diagnose first, then apply the right remedy",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_12", title="Underfitting",
        section="model_selection", order=12, difficulty="beginner",
        objectives=[
            "Identify underfitting from train/test scores",
            "Understand causes and remedies",
            "Apply fixes systematically",
        ],
        concept=(
            "Underfitting: model is too simple to capture the data's pattern. "
            "Signs: low training score AND low test score (both are poor). "
            "High bias. The model makes strong, incorrect assumptions."
        ),
        why_matters=(
            "An underfitted model wastes your data. No amount of data will help if the model "
            "is fundamentally too simple. You need to increase model complexity."
        ),
        example=(
            "Fitting a straight line to quadratic data: the line misses the curve everywhere. "
            "Train R² = 0.35, Test R² = 0.30. Both are low because the model cannot "
            "capture the non-linear relationship."
        ),
        common_mistakes=[
            "Adding more data (doesn't help underfitting — the model is too simple)",
            "Not recognising underfitting because both scores are similar (both are low!)",
            "Over-regularising (very large alpha in Ridge/Lasso)",
        ],
        practice_exercise=(
            "Fit a linear model on non-linear data. "
            "1. Are train and test scores similar? Both low?\n"
            "2. Add polynomial features (degree 2). How do scores change?\n"
            "3. What does this tell you about model complexity?"
        ),
        quiz=[
            QuizQuestion(
                question="Train R² = 0.40, Test R² = 0.38. Is this underfitting or overfitting?",
                options=[
                    "Overfitting — the model is too complex",
                    "Underfitting — both scores are low",
                    "Neither — the model is well-calibrated",
                    "Cannot determine without more information",
                ],
                correct_index=1,
                explanation=(
                    "Both train and test R² are low (and close together). This means the model "
                    "can't capture the pattern even in training data — it's too simple. "
                    "Overfitting would show high train, low test."
                ),
            ),
        ],
        takeaways=[
            "Underfitting: low train score + low test score",
            "Remedies: add features, use a more complex model, reduce regularisation",
            "Adding more data won't fix underfitting",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_13", title="Overfitting",
        section="model_selection", order=13, difficulty="beginner",
        objectives=[
            "Identify overfitting from train/test scores",
            "Understand causes and remedies",
            "Apply regularisation and cross-validation",
        ],
        concept=(
            "Overfitting: model is too complex and memorises training data noise. "
            "Signs: high training score, low test score. High variance. "
            "The model performs perfectly on data it has seen but fails on new data."
        ),
        why_matters=(
            "Overfitted models are useless in production. They look perfect during development "
            "but fail when deployed. Detecting and preventing overfitting is the most "
            "important skill in machine learning."
        ),
        example=(
            "Decision Tree with unlimited depth on Titanic: train accuracy = 100%, "
            "test accuracy = 72%. The tree memorised every training passenger but "
            "can't generalise. Limiting depth to 5: train = 84%, test = 81%."
        ),
        common_mistakes=[
            "Not noticing the gap between train and test scores",
            "Adding more features when overfitting (makes it worse)",
            "Not using cross-validation to get reliable test estimates",
        ],
        practice_exercise=(
            "Train a Decision Tree with max_depth in [1, 3, 5, 10, None]. "
            "Plot train and test accuracy vs depth. "
            "1. At what depth does overfitting start?\n"
            "2. What is the optimal depth?"
        ),
        quiz=[
            QuizQuestion(
                question="Which of the following does NOT help prevent overfitting?",
                options=[
                    "Using cross-validation",
                    "Adding more features to the model",
                    "Regularisation (Ridge, Lasso)",
                    "Getting more training data",
                ],
                correct_index=1,
                explanation=(
                    "Adding more features increases model complexity, which increases the risk "
                    "of overfitting. Cross-validation, regularisation, and more data all help "
                    "prevent overfitting by constraining the model or providing better estimates."
                ),
            ),
        ],
        takeaways=[
            "Overfitting: high train score + low test score",
            "Remedies: simpler model, regularisation, more data, cross-validation",
            "Use learning curves to diagnose",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_14", title="Cross-Validation",
        section="model_selection", order=14, difficulty="intermediate",
        objectives=[
            "Use cross-validation for reliable model selection",
            "Compare models fairly with the same CV folds",
            "Avoid data leakage in the selection process",
        ],
        concept=(
            "Cross-validation provides reliable performance estimates for model comparison. "
            "Use the SAME CV folds for all models to ensure fair comparison. "
            "Stratified k-fold preserves class distribution in each fold."
        ),
        why_matters=(
            "A single train/test split can mislead — a lucky split might favour one model. "
            "Cross-validation averages over multiple splits, giving a more reliable estimate. "
            "Using the same folds ensures models are compared on identical data."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "# SAME CV folds for all models\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\n"
            "models = {\n"
            "    'LR': LogisticRegression(max_iter=1000),\n"
            "    'RF': RandomForestClassifier(n_estimators=100, random_state=42),\n"
            "}\n\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')\n"
            "    print(f'{name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Using different CV splits for different models (unfair comparison)",
            "Not setting random_state in CV (non-reproducible results)",
            "Comparing models on different metrics",
        ],
        practice_exercise=(
            "Compare 3 models using 5-fold cross-validation with random_state=42. "
            "1. Which model has the highest mean F1?\n"
            "2. What is the standard deviation?\n"
            "3. Are any two models statistically indistinguishable?"
        ),
        quiz=[
            QuizQuestion(
                question="Why must you use the SAME cross-validation folds for all models?",
                options=[
                    "It's faster",
                    "Different folds contain different data — using different folds makes comparison unfair",
                    "sklearn requires it",
                    "It prevents data leakage",
                ],
                correct_index=1,
                explanation=(
                    "If model A is evaluated on folds {1,3,5} and model B on {2,4}, they see "
                    "different test data. A might appear better just because it got easier folds. "
                    "Same folds = same data = fair comparison."
                ),
            ),
        ],
        takeaways=[
            "Use the same CV folds for all models",
            "Compare on the same metric",
            "Report mean ± std of CV scores",
            "Set random_state for reproducibility",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_15", title="Hyperparameters",
        section="model_selection", order=15, difficulty="intermediate",
        objectives=[
            "Understand what hyperparameters are",
            "Know key hyperparameters for common algorithms",
            "Apply systematic tuning strategies",
        ],
        concept=(
            "Hyperparameters are settings you choose BEFORE training. They control model "
            "complexity and behaviour. Examples:\n"
            "• Random Forest: n_estimators, max_depth, min_samples_split\n"
            "• Gradient Boosting: n_estimators, learning_rate, max_depth\n"
            "• KNN: n_neighbors\n"
            "• Ridge/Lasso: alpha (regularisation strength)"
        ),
        why_matters=(
            "Default hyperparameters are rarely optimal. Tuning the right hyperparameters "
            "can improve performance by 5-15%. But tuning too many wastes time — focus on "
            "the most impactful ones first."
        ),
        python_example=(
            "```python\n"
            "# Key hyperparameters to tune:\n"
            "# Random Forest: n_estimators=100, max_depth=10, min_samples_split=5\n"
            "# GB: n_estimators=200, learning_rate=0.1, max_depth=3\n"
            "# Ridge/Lasso: alpha=1.0\n"
            "# KNN: n_neighbors=5\n\n"
            "# Start with defaults, then tune the most impactful:\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "from sklearn.model_selection import GridSearchCV\n\n"
            "param_grid = {\n"
            "    'n_estimators': [100, 200, 300],\n"
            "    'learning_rate': [0.01, 0.1, 0.2],\n"
            "    'max_depth': [3, 5, 7]\n"
            "}\n"
            "grid = GridSearchCV(\n"
            "    GradientBoostingClassifier(random_state=42),\n"
            "    param_grid, cv=5, scoring='f1', n_jobs=-1\n"
            ")\n"
            "grid.fit(X_train, y_train)\n"
            "print(f'Best: {grid.best_params_}, F1: {grid.best_score_:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Using defaults for all models without testing alternatives",
            "Tuning too many hyperparameters at once (exponential combinations)",
            "Tuning on test data instead of using cross-validation",
        ],
        practice_exercise=(
            "Tune a Random Forest's max_depth from 1 to 20. "
            "1. Plot test accuracy vs max_depth.\n"
            "2. What is the optimal max_depth?\n"
            "3. What happens with max_depth=None?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the difference between a hyperparameter and a model parameter?",
                options=[
                    "They are the same thing",
                    "Hyperparameters are set before training; parameters are learned from data during training",
                    "Hyperparameters are for classification; parameters are for regression",
                    "Parameters are more important than hyperparameters",
                ],
                correct_index=1,
                explanation=(
                    "Hyperparameters are settings you choose before training (max_depth, learning_rate). "
                    "Parameters are learned from data during training (coefficients in LR, "
                    "split thresholds in Decision Trees). You tune hyperparameters; the model "
                    "learns parameters."
                ),
            ),
        ],
        takeaways=[
            "Hyperparameters are set before training; parameters are learned from data",
            "Start with defaults, then tune the most impactful hyperparameters",
            "Always use cross-validation for tuning (never tune on test data)",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_16", title="Grid Search",
        section="model_selection", order=16, difficulty="intermediate",
        objectives=[
            "Apply GridSearchCV for hyperparameter tuning",
            "Define effective parameter grids",
            "Interpret grid search results",
        ],
        concept=(
            "Grid search tries EVERY combination of specified hyperparameters. "
            "It's exhaustive but slow. Returns the combination with the best "
            "cross-validation score. Works well for small parameter spaces."
        ),
        why_matters=(
            "Grid search automates hyperparameter tuning. Instead of manually testing "
            "combinations, you define the search space and let the algorithm find the best. "
            "It ensures you don't miss good combinations."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "param_grid = {\n"
            "    'n_estimators': [50, 100, 200],\n"
            "    'max_depth': [3, 5, 10, None],\n"
            "    'min_samples_split': [2, 5, 10]\n"
            "}\n"
            "# Total combinations: 3 × 4 × 3 = 36 models × 5 folds = 180 fits\n\n"
            "grid = GridSearchCV(\n"
            "    RandomForestClassifier(random_state=42),\n"
            "    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1\n"
            ")\n"
            "grid.fit(X_train, y_train)\n\n"
            "print(f'Best params: {grid.best_params_}')\n"
            "print(f'Best CV F1: {grid.best_score_:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Too many parameters (exponential combinations → too slow)",
            "Not using n_jobs=-1 (wastes CPU cores)",
            "Tuning on test data instead of using cross-validation",
        ],
        practice_exercise=(
            "Run GridSearchCV on a dataset with the Random Forest parameter grid above. "
            "1. How many total model fits were performed?\n"
            "2. What are the best parameters?\n"
            "3. How much better is the tuned model vs defaults?"
        ),
        quiz=[
            QuizQuestion(
                question="Grid search has 3 parameters with 5 values each. How many model fits are performed with 5-fold CV?",
                options=["15", "125", "625", "3125"],
                correct_index=2,
                explanation=(
                    "Total combinations: 5 × 5 × 5 = 125. Each combination evaluated with "
                    "5-fold CV = 125 × 5 = 625 model fits. This is why grid search is slow "
                    "for large parameter spaces."
                ),
            ),
        ],
        takeaways=[
            "Grid search: exhaustive but slow (all combinations)",
            "Best for small parameter grids (3-4 parameters)",
            "Use n_jobs=-1 for parallel processing",
            "Never tune on test data — use cross-validation",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_17", title="Random Search",
        section="model_selection", order=17, difficulty="intermediate",
        objectives=[
            "Apply RandomizedSearchCV for hyperparameter tuning",
            "Compare random search with grid search",
            "Choose the right search strategy",
        ],
        concept=(
            "Random search tries RANDOM combinations of hyperparameters. It's faster than "
            "grid search and often finds comparable results. Research shows random search "
            "is more efficient because it explores the space more broadly."
        ),
        why_matters=(
            "For large parameter spaces, grid search wastes time on unimportant parameters. "
            "Random search allocates more trials to the most promising regions. "
            "It's the preferred method when you have many hyperparameters."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import RandomizedSearchCV\n"
            "from scipy.stats import randint, uniform\n\n"
            "param_distributions = {\n"
            "    'n_estimators': randint(50, 300),\n"
            "    'max_depth': randint(3, 15),\n"
            "    'min_samples_split': randint(2, 20),\n"
            "    'min_samples_leaf': randint(1, 10)\n"
            "}\n\n"
            "random_search = RandomizedSearchCV(\n"
            "    RandomForestClassifier(random_state=42),\n"
            "    param_distributions, n_iter=50, cv=5,\n"
            "    scoring='f1', random_state=42, n_jobs=-1\n"
            ")\n"
            "random_search.fit(X_train, y_train)\n"
            "print(f'Best: {random_search.best_score_:.3f}')\n"
            "print(f'Params: {random_search.best_params_}')\n"
            "```"
        ),
        common_mistakes=[
            "Too few iterations (n_iter=10 misses good combinations)",
            "Not setting random_state (non-reproducible results)",
            "Using random search for tiny parameter spaces (grid is better for 2-3 params)",
        ],
        practice_exercise=(
            "Compare grid search and random search on the same parameter space. "
            "1. Which finds the better combination in less time?\n"
            "2. Try random search with n_iter=20 vs n_iter=100. How much better is 100?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is random search often more efficient than grid search?",
                options=[
                    "Random search uses faster algorithms",
                    "Random search explores more of the parameter space with fewer iterations, especially when some parameters matter more than others",
                    "Grid search doesn't work with cross-validation",
                    "Random search can handle more parameters",
                ],
                correct_index=1,
                explanation=(
                    "Grid search wastes combinations on unimportant parameters. If learning_rate "
                    "matters more than subsample, random search explores more learning_rate values, "
                    "while grid search is forced to try all subsample values."
                ),
            ),
        ],
        takeaways=[
            "Random search: faster, often as good as grid search",
            "Better for large parameter spaces (>3 parameters)",
            "Use n_iter=50-100 for good coverage",
            "Use scipy.stats distributions for continuous parameters",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_18", title="Hyperparameter Tuning",
        section="model_selection", order=18, difficulty="intermediate",
        objectives=[
            "Plan a systematic tuning strategy",
            "Prioritise the most impactful parameters",
            "Avoid overfitting to validation data",
        ],
        concept=(
            "Systematic tuning strategy:\n"
            "1. Start with default hyperparameters\n"
            "2. Establish a baseline score\n"
            "3. Tune the most impactful parameter first (e.g., max_depth, learning_rate)\n"
            "4. Tune secondary parameters with the best primary value fixed\n"
            "5. Use random search for large spaces, grid search for small spaces\n"
            "6. Never tune on test data"
        ),
        why_matters=(
            "Random tuning wastes time. A systematic approach focuses effort on parameters "
            "that matter most. Documenting what you tried prevents repeating failed approaches."
        ),
        python_example=(
            "```python\n"
            "# Step 1: Baseline with defaults\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "baseline = GradientBoostingClassifier(random_state=42)\n"
            "print(f'Default: {cross_val_score(baseline, X, y, cv=5).mean():.3f}')\n\n"
            "# Step 2: Tune learning_rate (most impactful for GB)\n"
            "from sklearn.model_selection import RandomizedSearchCV\n"
            "from scipy.stats import uniform\n\n"
            "param_grid = {'learning_rate': uniform(0.01, 0.3)}\n"
            "search = RandomizedSearchCV(\n"
            "    GradientBoostingClassifier(random_state=42),\n"
            "    param_grid, n_iter=20, cv=5, random_state=42\n"
            ")\n"
            "search.fit(X_train, y_train)\n"
            "print(f'Best lr: {search.best_params_[\"learning_rate\"]:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Tuning everything at once (too many combinations)",
            "Over-tuning: too many iterations on the same validation data can overfit",
            "Not tracking what you tried (can't reproduce or learn from experiments)",
        ],
        practice_exercise=(
            "Tune a Gradient Boosting model systematically:\n"
            "1. First, tune learning_rate (try [0.01, 0.05, 0.1, 0.2]).\n"
            "2. Fix the best learning_rate and tune n_estimators.\n"
            "3. Fix both and tune max_depth.\n"
            "4. How much did each step improve the score?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is it important to tune one or two parameters at a time instead of all at once?",
                options=[
                    "It's faster to run",
                    "You can see which parameters have the most impact, and avoid overfitting to the validation set",
                    "sklearn doesn't support tuning multiple parameters",
                    "It always produces better results",
                ],
                correct_index=1,
                explanation=(
                    "Tuning one parameter at a time lets you understand its impact. Tuning all "
                    "at once finds a good combination but doesn't tell you which parameters "
                    "matter. It also requires far fewer model fits."
                ),
            ),
        ],
        takeaways=[
            "Start simple (defaults), add complexity systematically",
            "Tune the most impactful parameter first",
            "Document all tuning attempts",
            "Never tune on test data",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_19", title="Model Comparison",
        section="model_selection", order=19, difficulty="intermediate",
        objectives=[
            "Compare models systematically",
            "Use consistent evaluation methodology",
            "Account for variability in results",
        ],
        concept=(
            "Fair model comparison requires:\n"
            "• Same dataset and preprocessing\n"
            "• Same train/test split or CV folds\n"
            "• Same evaluation metric\n"
            "• Report mean ± std to account for variability\n"
            "• Consider statistical significance of differences"
        ),
        why_matters=(
            "Unfair comparisons lead to wrong conclusions. Comparing model A on one split "
            "and model B on another makes the comparison meaningless. Consistency is essential."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n"
            "import pandas as pd\n\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "results = []\n\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')\n"
            "    results.append({'Model': name, 'F1_mean': scores.mean(), 'F1_std': scores.std()})\n\n"
            "df = pd.DataFrame(results).sort_values('F1_mean', ascending=False)\n"
            "print(df.to_string(index=False))\n"
            "```"
        ),
        common_mistakes=[
            "Comparing models on different datasets or splits",
            "Only reporting the mean (ignoring standard deviation)",
            "Choosing the model with the highest single-run score",
        ],
        practice_exercise=(
            "Compare 4 models with 5-fold CV. Create a comparison table with mean and std. "
            "1. Which model is best by mean F1?\n"
            "2. If model A has F1=0.75±0.08 and model B has F1=0.74±0.03, which is better?\n"
            "3. Why does standard deviation matter?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A: F1 = 0.78 ± 0.06. Model B: F1 = 0.77 ± 0.02. Which model is more reliable?",
                options=[
                    "Model A — higher mean",
                    "Model B — lower variance means more consistent performance",
                    "They are identical",
                    "Need to see the training scores to decide",
                ],
                correct_index=1,
                explanation=(
                    "Model B is more reliable because its lower standard deviation means "
                    "performance is consistent across folds. Model A's higher variance means "
                    "it sometimes performs well and sometimes poorly — less predictable."
                ),
            ),
        ],
        takeaways=[
            "Use same CV, same metric, same preprocessing for all models",
            "Report mean ± std",
            "Lower variance = more reliable",
            "Statistical tests for formal significance",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_20", title="Selecting Evaluation Metrics",
        section="model_selection", order=20, difficulty="intermediate",
        objectives=[
            "Match metrics to business requirements",
            "Understand metric implications",
            "Use multiple metrics for a complete picture",
        ],
        concept=(
            "Metrics must match the problem:\n"
            "• Balanced classification: accuracy\n"
            "• Imbalanced classification: F1, AUC, precision, recall\n"
            "• Regression: RMSE (outlier-sensitive), MAE (robust), R² (explained variance)\n"
            "Ask stakeholders: what's the cost of different types of errors?"
        ),
        why_matters=(
            "Choosing the wrong metric leads to optimising for the wrong thing. "
            "Maximising accuracy on fraud detection produces a useless model. "
            "The metric determines what the model optimises."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "# Different metrics for the same model\n"
            "for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:\n"
            "    scores = cross_val_score(model, X, y, cv=5, scoring=metric)\n"
            "    print(f'{metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Defaulting to accuracy without checking class balance",
            "Using metrics that don't match business requirements",
            "Not asking stakeholders what type of error matters most",
        ],
        practice_exercise=(
            "Train a model and evaluate with accuracy, F1, precision, and recall. "
            "1. Are all metrics similar? If not, why?\n"
            "2. For this problem, which metric matters most?\n"
            "3. What would a stakeholder care about?"
        ),
        quiz=[
            QuizQuestion(
                question="In a medical screening test, which metric is most important: accuracy, precision, or recall?",
                options=[
                    "Accuracy — overall correctness",
                    "Precision — few false alarms",
                    "Recall — catching all disease cases (minimising missed diagnoses)",
                    "They are all equally important",
                ],
                correct_index=2,
                explanation=(
                    "For medical screening, missing a disease (false negative) is dangerous. "
                    "Recall measures how many actual disease cases are caught. Maximising recall "
                    "ensures fewer missed diagnoses, even if it means more false alarms."
                ),
            ),
        ],
        takeaways=[
            "Metrics depend on the problem and business requirements",
            "Ask about the cost of different error types",
            "Use multiple metrics for a complete picture",
            "Never default to accuracy without checking class balance",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_21", title="Choosing a Final Model",
        section="model_selection", order=21, difficulty="advanced",
        objectives=[
            "Make the final model selection decision",
            "Balance performance, speed, and interpretability",
            "Document and justify the choice",
        ],
        concept=(
            "Final model selection considers five factors:\n"
            "1. Performance: CV scores (mean ± std)\n"
            "2. Speed: training and prediction time\n"
            "3. Interpretability: can you explain decisions?\n"
            "4. Maintenance: how complex to deploy and update?\n"
            "5. Stakeholder requirements: what do they need?"
        ),
        why_matters=(
            "The 'best' model on paper may not be the best in production. A model that "
            "requires 10GB of RAM, takes 30 minutes to retrain, and is impossible to explain "
            "may be worse than a slightly less accurate model that runs on a laptop."
        ),
        example=(
            "Final model comparison:\n"
            "• Gradient Boosting: F1=0.84, 5min training, 10ms prediction, black box\n"
            "• Random Forest: F1=0.81, 30s training, 2ms prediction, feature importance\n"
            "• Logistic Regression: F1=0.77, 0.1s training, 0.1ms prediction, fully interpretable\n"
            "Decision: Random Forest — best balance of accuracy, speed, and explainability."
        ),
        common_mistakes=[
            "Choosing solely on accuracy",
            "Not considering deployment complexity and maintenance",
            "Not documenting the decision process for future reference",
        ],
        practice_exercise=(
            "Compare 3 models on all five factors: accuracy, training time, prediction time, "
            "interpretability, and deployment complexity. Create a comparison table and "
            "write a one-paragraph justification for your choice."
        ),
        quiz=[
            QuizQuestion(
                question="Which factor should ALWAYS be considered when choosing a final model?",
                options=[
                    "Training accuracy must be the highest possible",
                    "The model must be the most complex available",
                    "The model must meet the deployment and business requirements",
                    "The model must have the most hyperparameters",
                ],
                correct_index=2,
                explanation=(
                    "A model that doesn't meet deployment requirements (too slow, too large, "
                    "not interpretable when required) is useless regardless of accuracy. "
                    "Business and deployment requirements are non-negotiable constraints."
                ),
            ),
        ],
        takeaways=[
            "Balance performance, speed, interpretability, and maintainability",
            "Document why you chose the model",
            "Consider the full production lifecycle",
            "Small accuracy gains may not justify complexity",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_22", title="Model Selection Case Study",
        section="model_selection", order=22, difficulty="advanced",
        objectives=[
            "Apply the complete model selection process",
            "Compare algorithms systematically",
            "Document and justify the final choice",
        ],
        concept=(
            "Complete case study workflow:\n"
            "1. Establish baseline (majority class / mean)\n"
            "2. Train 4-5 candidate models\n"
            "3. Compare with cross-validation\n"
            "4. Tune the top 2-3 models\n"
            "5. Final comparison on tuned models\n"
            "6. Choose based on all factors (accuracy, speed, interpretability)\n"
            "7. Document the process"
        ),
        why_matters=(
            "A case study ties everything together. It demonstrates the systematic approach "
            "that real data science projects require. Every decision should be justified."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n"
            "import time\n\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "results = []\n\n"
            "for name, model in models.items():\n"
            "    start = time.time()\n"
            "    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')\n"
            "    train_time = time.time() - start\n"
            "    results.append({\n"
            "        'Model': name,\n"
            "        'F1': f'{scores.mean():.3f} ± {scores.std():.3f}',\n"
            "        'Time': f'{train_time:.1f}s'\n"
            "    })\n"
            "```"
        ),
        common_mistakes=[
            "Chasing 1% improvement at the cost of interpretability and speed",
            "Not documenting the selection process",
            "Choosing the most complex model by default",
        ],
        practice_exercise=(
            "Complete a model selection case study:\n"
            "1. Establish a baseline\n"
            "2. Compare 4 models with CV\n"
            "3. Tune the best 2\n"
            "4. Make a final choice with justification\n"
            "5. Write a one-paragraph summary of your decision"
        ),
        quiz=[
            QuizQuestion(
                question="After a complete model selection process, Model A has F1=0.83 and Model B has F1=0.81. Model A is a Gradient Boosting and B is Logistic Regression. The business requires interpretable predictions. What do you recommend?",
                options=[
                    "Model A — highest F1",
                    "Model B — Logistic Regression is interpretable and the 2% accuracy loss is acceptable",
                    "Train a new model that combines both",
                    "Ask the stakeholders to accept the less interpretable model",
                ],
                correct_index=1,
                explanation=(
                    "When interpretability is a business requirement, the 2% F1 difference "
                    "does not justify switching to a black-box model. Model B meets the "
                    "requirement while performing nearly as well."
                ),
            ),
        ],
        takeaways=[
            "Always start with a baseline and compare systematically",
            "Use cross-validation for reliable estimates",
            "Balance performance with practical requirements",
            "Document every decision and justify the final choice",
        ],
        lab_module="model_comparison",
    ),
    Topic(
        id="sel_23", title="Model Complexity",
        section="model_selection", order=23, difficulty="intermediate",
        objectives=[
            "Understand what makes a model complex",
            "Balance complexity with data size",
            "Know when complexity helps vs hurts",
        ],
        concept=(
            "Model complexity is the capacity to fit complex patterns. Simple models "
            "(linear regression, depth-2 trees) have low complexity. Complex models "
            "(deep trees, gradient boosting with many estimators, neural networks) have high "
            "complexity. Too little complexity → underfitting. Too much → overfitting."
        ),
        why_matters=(
            "Choosing the right complexity for your data is the central challenge of machine "
            "learning. The optimal complexity depends on data size, feature quality, and "
            "noise level. Regularisation controls effective complexity."
        ),
        example=(
            "Decision Tree max_depth vs performance:\n"
            "• depth=1: underfitting (too simple, misses patterns)\n"
            "• depth=5: good fit (captures key patterns, generalises)\n"
            "• depth=50: overfitting (memorises training data, fails on test)\n"
            "The optimal depth is the simplest that captures the pattern."
        ),
        common_mistakes=[
            "Using complex models on small datasets (overfitting)",
            "Not regularising complex models",
            "Assuming more complex always means better",
        ],
        practice_exercise=(
            "Train models of increasing complexity on a dataset. "
            "1. Plot training and test performance vs model complexity.\n"
            "2. At what complexity does overfitting start?\n"
            "3. What is the sweet spot?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the 'sweet spot' of model complexity?",
                options=[
                    "Maximum possible complexity",
                    "Minimum possible complexity",
                    "The simplest model that captures the underlying pattern without overfitting",
                    "The model with the most parameters",
                ],
                correct_index=2,
                explanation=(
                    "The sweet spot is where the model is complex enough to capture the "
                    "data's pattern (low bias) but simple enough to generalise to new data "
                    "(low variance). This is the bias-variance tradeoff in practice."
                ),
            ),
        ],
        takeaways=[
            "Complexity must match data size and quality",
            "Regularisation controls effective complexity",
            "Start simple, increase only if justified by cross-validation",
            "The sweet spot balances bias (underfitting) and variance (overfitting)",
        ],
        lab_module="model_comparison",
    ),
]
