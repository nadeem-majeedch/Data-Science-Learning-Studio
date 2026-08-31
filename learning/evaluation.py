"""Model Evaluation curriculum — 23 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="eval_01", title="Why Model Evaluation Matters", section="evaluation", order=1,
        difficulty="beginner",
        objectives=[
            "Understand why evaluation is essential",
            "Recognise that training performance ≠ real performance",
            "Appreciate the cost of poor evaluation",
        ],
        concept=(
            "Model evaluation measures how well your model generalises to unseen data. "
            "Without proper evaluation, you cannot know if your model is useful, harmful, "
            "or simply memorising the training data."
        ),
        why_matters=(
            "An unevaluated model is like an untested medicine — it might work, it might harm, "
            "and you have no way to know. Evaluation provides the evidence that your model "
            "does what it claims on data it has never seen."
        ),
        simple_explanation=(
            "Would you trust a student who aces practice exams but fails the real exam? "
            "That's what happens when you evaluate on training data."
        ),
        example=(
            "A spam filter trained on 10,000 emails achieves 99% accuracy on training data. "
            "Deployed to real email, it marks 30% of legitimate emails as spam. "
            "The training accuracy was a lie because the model memorised the specific emails."
        ),
        common_mistakes=[
            "Evaluating on training data (always overestimates performance)",
            "Not holding out a test set before any analysis",
            "Choosing metrics that don't match the business problem",
        ],
        practice_exercise=(
            "Train a Decision Tree with unlimited depth on any dataset. "
            "Calculate accuracy on the training set vs the test set. "
            "What do you observe? Is this a trustworthy model?"
        ),
        quiz=[
            QuizQuestion(
                question="Why can't you evaluate a model's performance on the same data it was trained on?",
                options=[
                    "Because the model takes too long to predict on training data",
                    "Because the model has already seen those examples and may have memorised them",
                    "Because sklearn doesn't allow it",
                    "Because training data is always too small",
                ],
                correct_index=1,
                explanation=(
                    "Models can memorise training data, especially complex ones. Evaluating on "
                    "training data gives an optimistic estimate. You need unseen test data to "
                    "measure true generalisation ability."
                ),
            ),
        ],
        takeaways=[
            "Always evaluate on unseen data — never on training data",
            "Training score ≠ test score",
            "Choose metrics that match the real-world problem and costs",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_02", title="Training vs Testing", section="evaluation", order=2,
        difficulty="beginner",
        objectives=[
            "Understand the train/test split concept",
            "Know standard split ratios",
            "Apply random splitting with train_test_split",
        ],
        concept=(
            "Data is split into training (model learns patterns) and testing (model is evaluated). "
            "The test set must be completely invisible during training — it simulates the 'real world' "
            "where the model encounters new data it has never seen."
        ),
        why_matters=(
            "If the model sees test data during training (even indirectly through preprocessing), "
            "the evaluation is dishonest and the model will fail in production. "
            "The train/test split is the foundation of trustworthy evaluation."
        ),
        simple_explanation=(
            "Training = studying for the exam. Testing = taking the actual exam. "
            "You can't study for an exam you've already seen."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "print(f'Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)')\n"
            "print(f'Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)')\n"
            "```"
        ),
        common_mistakes=[
            "Not setting random_state (non-reproducible splits each time)",
            "Fitting preprocessing on the full dataset before splitting (data leakage)",
            "Test size too small (<10%) gives unreliable performance estimates",
        ],
        practice_exercise=(
            "Load Titanic dataset. Split with test_size=0.2 and random_state=42. "
            "1. How many samples in train and test?\n"
            "2. Try splitting 5 times without random_state. Do you get different splits?\n"
            "3. Why does reproducibility matter?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the correct order of operations for a machine learning project?",
                options=[
                    "Train model → preprocess data → split data → evaluate",
                    "Split data → preprocess training data → train model → evaluate on test",
                    "Preprocess all data → train model → split → evaluate",
                    "Split data → train model → preprocess test data → evaluate",
                ],
                correct_index=1,
                explanation=(
                    "Split first, then preprocess using ONLY the training data (fit the scaler "
                    "on training, transform both). This prevents data leakage where information "
                    "from the test set influences the model."
                ),
            ),
        ],
        takeaways=[
            "Split before any preprocessing",
            "Standard ratios: 80/20 or 70/30 (train/test)",
            "Set random_state for reproducibility",
            "Preprocessing must be fitted on training data only",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_03", title="Validation Dataset", section="evaluation", order=3,
        difficulty="beginner",
        objectives=[
            "Understand the three-way split (train/validation/test)",
            "Know when to use a validation set",
            "Apply train/validation/test splitting",
        ],
        concept=(
            "A validation set is a third partition used for model selection and hyperparameter "
            "tuning. The test set should only be used for the FINAL evaluation — never for "
            "making decisions about the model during development."
        ),
        why_matters=(
            "If you tune hyperparameters on the test set, you're optimising for that specific "
            "test set, not for generalisation. The validation set provides an honest estimate "
            "during development. Cross-validation can replace a fixed validation set."
        ),
        simple_explanation=(
            "Training = study material. Validation = practice exam (choose your strategy). "
            "Test = final exam (only shows your true performance)."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n\n"
            "# First split: train+val vs test\n"
            "X_trainval, X_test, y_trainval, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "# Second split: train vs validation\n"
            "X_train, X_val, y_train, y_val = train_test_split(\n"
            "    X_trainval, y_trainval, test_size=0.25, random_state=42\n"
            ")\n"
            "# Result: 60% train, 20% validation, 20% test\n"
            "```"
        ),
        common_mistakes=[
            "Tuning hyperparameters on the test set",
            "Making the validation set too small (unreliable estimates)",
            "Not keeping the test set completely separate until the very end",
        ],
        practice_exercise=(
            "Split a dataset into 60% train, 20% validation, 20% test. "
            "1. Train a model on training data only.\n"
            "2. Evaluate on validation — tune a hyperparameter.\n"
            "3. Only then evaluate on test. Is the test score different from validation?"
        ),
        quiz=[
            QuizQuestion(
                question="When should you use the test set?",
                options=[
                    "During hyperparameter tuning",
                    "To decide which model architecture to use",
                    "Only for the final evaluation after all decisions are made",
                    "To preprocess the data",
                ],
                correct_index=2,
                explanation=(
                    "The test set is held out until the very end. All model selection, "
                    "hyperparameter tuning, and feature engineering decisions should be made "
                    "using training and validation data only. The test set gives your final, "
                    "unbiased performance estimate."
                ),
            ),
        ],
        takeaways=[
            "Train: learn patterns, Validation: tune model, Test: final evaluation",
            "Never touch the test set until all decisions are made",
            "Cross-validation can replace a fixed validation set",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_04", title="Holdout Validation", section="evaluation", order=4,
        difficulty="beginner",
        objectives=[
            "Apply holdout validation (single train/test split)",
            "Understand its advantages and limitations",
            "Know when cross-validation is preferable",
        ],
        concept=(
            "Holdout validation splits data once into training and test sets. It's the simplest "
            "validation method — fast and straightforward. However, the evaluation depends on "
            "which specific split you get."
        ),
        why_matters=(
            "Holdout is the starting point for all validation. Understanding its limitations — "
            "instability from a single split, wasted data — motivates the need for "
            "cross-validation in smaller datasets."
        ),
        example=(
            "A dataset of 200 samples split 80/20: test set has 40 samples. "
            "A few lucky/unlucky samples in the test set can swing accuracy by 5-10%. "
            "With 10,000 samples, the test set has 2,000 samples — much more stable."
        ),
        common_mistakes=[
            "Relying on a single holdout split as the final word",
            "Using different random states for different experiments (inconsistent comparison)",
            "Not checking if the split preserves class distribution (stratified split)",
        ],
        practice_exercise=(
            "Split a dataset with random_state=42, then with random_state=123. "
            "Train the same model on both splits. How different are the test scores? "
            "What does this tell you about holdout stability?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the main limitation of holdout validation?",
                options=[
                    "It's too slow for large datasets",
                    "The evaluation depends on which specific split you get",
                    "It requires too much code",
                    "It can't be used with sklearn",
                ],
                correct_index=1,
                explanation=(
                    "A single holdout split may not be representative. Different splits "
                    "produce different test scores, making evaluation unstable. Cross-validation "
                    "solves this by using multiple splits."
                ),
            ),
        ],
        takeaways=[
            "Holdout: fast and simple, but unstable with small datasets",
            "Use when the dataset is very large (>100K samples)",
            "For smaller datasets, cross-validation provides more reliable estimates",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_05", title="Cross-Validation", section="evaluation", order=5,
        difficulty="intermediate",
        objectives=[
            "Apply k-fold cross-validation",
            "Understand stratified k-fold for classification",
            "Interpret cross-validation scores",
        ],
        concept=(
            "K-fold cross-validation splits data into k folds. For each fold, train on k-1 folds "
            "and test on the remaining fold. After k iterations, average the scores. "
            "Stratified k-fold preserves the class distribution in each fold."
        ),
        why_matters=(
            "Cross-validation gives a more reliable performance estimate than a single holdout. "
            "It uses every sample for both training and testing (at different times), "
            "reducing the variance of the estimate."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "# Classification: use StratifiedKFold\n"
            "model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')\n"
            "print(f'CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "print(f'Per-fold: {[f\"{s:.3f}\" for s in scores]}')\n\n"
            "# Regression: use KFold\n"
            "from sklearn.model_selection import KFold\n"
            "cv = KFold(n_splits=5, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv, scoring='r2')\n"
            "print(f'CV R²: {scores.mean():.3f} ± {scores.std():.3f}')\n"
            "```"
        ),
        interpretation=(
            "Mean CV score = expected performance. Std = stability. A high std (>0.05 for accuracy) "
            "means the model is sensitive to which data it sees — possibly overfitting or the "
            "dataset is too small. Per-fold scores reveal if certain data subsets are harder."
        ),
        common_mistakes=[
            "Not shuffling data before splitting (temporal bias in sequential data)",
            "Using regular KFold for classification (may create imbalanced folds)",
            "Reporting only the best fold score instead of mean ± std",
        ],
        practice_exercise=(
            "Apply 5-fold cross-validation on Titanic dataset with Random Forest. "
            "1. What is the mean and std of accuracy?\n"
            "2. Are any folds significantly worse than others?\n"
            "3. Now try 10-fold CV. Does the estimate change?"
        ),
        quiz=[
            QuizQuestion(
                question="Why should you use StratifiedKFold instead of regular KFold for classification?",
                options=[
                    "StratifiedKFold is faster",
                    "StratifiedKFold ensures each fold has the same class distribution as the full dataset",
                    "StratifiedKFold uses more data",
                    "It doesn't matter — both give the same results",
                ],
                correct_index=1,
                explanation=(
                    "StratifiedKFold maintains the same proportion of each class in every fold. "
                    "Regular KFold might create folds where a rare class is entirely absent, "
                    "producing misleading accuracy scores."
                ),
            ),
        ],
        takeaways=[
            "K-fold CV: more reliable than holdout, uses all data for training and testing",
            "Use StratifiedKFold for classification, KFold for regression",
            "Report mean ± std of cross-validation scores",
            "5-fold is standard; 10-fold for small datasets",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_06", title="Accuracy", section="evaluation", order=6,
        difficulty="beginner",
        objectives=[
            "Calculate accuracy",
            "Understand accuracy's limitations",
            "Know when accuracy is misleading",
        ],
        concept=(
            "Accuracy = (TP + TN) / Total. The proportion of all predictions that are correct. "
            "It's the most intuitive metric but dangerously misleading for imbalanced datasets."
        ),
        why_matters=(
            "Accuracy is the most commonly reported metric but also the most misused. "
            "For imbalanced data (e.g., 99% non-fraudulent), a model that always predicts "
            "'not fraud' achieves 99% accuracy while being completely useless."
        ),
        example=(
            "Credit card fraud detection: 99.9% of transactions are legitimate. "
            "A model that always predicts 'legitimate' has 99.9% accuracy but catches "
            "0% of fraud. It's a perfect accuracy score with zero practical value."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "accuracy = accuracy_score(y_test, y_pred)\n"
            "print(f'Accuracy: {accuracy:.3f}')\n\n"
            "# Manual calculation\n"
            "accuracy_manual = (y_test == y_pred).mean()\n"
            "print(f'Manual: {accuracy_manual:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Using accuracy for imbalanced datasets (misleadingly high)",
            "Reporting accuracy without checking class distribution",
            "Not reporting precision/recall alongside accuracy",
        ],
        practice_exercise=(
            "Create a dataset with 95% class 0 and 5% class 1. "
            "Train a model that always predicts class 0. "
            "What is its accuracy? Is it useful?"
        ),
        quiz=[
            QuizQuestion(
                question="A model for cancer detection has 95% accuracy. The dataset is 95% healthy patients. Is accuracy a good metric here?",
                options=[
                    "Yes — 95% is a good score",
                    "No — the model could just predict 'healthy' for everyone and get 95%",
                    "Yes — accuracy is always the best metric",
                    "It depends on the model's training time",
                ],
                correct_index=1,
                explanation=(
                    "With 95% healthy patients, a model that always predicts 'healthy' achieves "
                    "95% accuracy while catching 0% of cancers. For imbalanced data, precision, "
                    "recall, and F1 are far more informative."
                ),
            ),
        ],
        takeaways=[
            "Accuracy = correct predictions / total predictions",
            "Misleading for imbalanced datasets",
            "Always report with precision, recall, F1 for classification",
            "A model predicting the majority class can have high accuracy while being useless",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_07", title="Precision", section="evaluation", order=7,
        difficulty="beginner",
        objectives=[
            "Calculate precision",
            "Interpret precision in practical context",
            "Understand the precision-recall tradeoff",
        ],
        concept=(
            "Precision = TP / (TP + FP). Of all instances predicted as positive, how many are "
            "actually positive? High precision = few false alarms. It measures the quality of "
            "positive predictions."
        ),
        why_matters=(
            "Precision matters when false positives are costly. In email spam detection, a "
            "legitimate email marked as spam (FP) is worse than letting some spam through (FN). "
            "In search engines, irrelevant results (FP) frustrate users."
        ),
        example=(
            "An email spam filter: 100 emails predicted as spam, 90 are actually spam.\n"
            "Precision = 90/100 = 0.90. 10% of flagged emails were legitimate (false positives).\n"
            "Users would be annoyed by 10 legitimate emails going to spam."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import precision_score\n\n"
            "precision = precision_score(y_test, y_pred)\n"
            "print(f'Precision: {precision:.3f}')\n\n"
            "# Per-class precision\n"
            "precision_per_class = precision_score(y_test, y_pred, average=None)\n"
            "for i, p in enumerate(precision_per_class):\n"
            "    print(f'  Class {i}: {p:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Reporting precision without recall (tells only half the story)",
            "Confusing precision with accuracy",
            "Not specifying which class for multiclass problems",
        ],
        practice_exercise=(
            "Train a classifier on Titanic. Calculate precision for 'Survived' class. "
            "1. What does this precision value tell you?\n"
            "2. Would you prefer high precision or high recall for this problem?"
        ),
        quiz=[
            QuizQuestion(
                question="When is high precision more important than high recall?",
                options=[
                    "Cancer screening — missing cancer is dangerous",
                    "Spam detection — marking legitimate emails as spam is unacceptable",
                    "Disease surveillance — catching all cases is critical",
                    "Fraud detection — all fraud must be caught",
                ],
                correct_index=1,
                explanation=(
                    "In spam detection, a false positive means a legitimate email goes to spam — "
                    "very disruptive. High precision minimises these false alarms. Cancer screening "
                    "and disease surveillance prioritise recall (catching all cases)."
                ),
            ),
        ],
        takeaways=[
            "Precision = quality of positive predictions (few false alarms)",
            "High precision: when false positives are costly",
            "Precision = TP / (TP + FP)",
            "Always report with recall — they tell different parts of the story",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_08", title="Recall", section="evaluation", order=8,
        difficulty="beginner",
        objectives=[
            "Calculate recall",
            "Interpret recall in practical context",
            "Understand the precision-recall tradeoff",
        ],
        concept=(
            "Recall = TP / (TP + FN). Of all actual positive instances, how many did the model "
            "catch? High recall = few missed cases. It measures the completeness of positive "
            "detection."
        ),
        why_matters=(
            "Recall matters when false negatives are costly. In cancer screening, missing a "
            "cancer patient (FN) is far worse than a false alarm (FP). In security, missing "
            "a threat could be catastrophic."
        ),
        example=(
            "Cancer screening: 50 patients have cancer. The model detects 45 of them.\n"
            "Recall = 45/50 = 0.90. Five cancer cases were missed (FN).\n"
            "Increasing recall to 0.98 would catch 49 of 50, missing only 1."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import recall_score\n\n"
            "recall = recall_score(y_test, y_pred)\n"
            "print(f'Recall: {recall:.3f}')\n\n"
            "# F2 score weights recall higher than precision\n"
            "from sklearn.metrics import fbeta_score\n"
            "f2 = fbeta_score(y_test, y_pred, beta=2)\n"
            "print(f'F2 (recall-weighted): {f2:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Reporting recall without precision (incomplete story)",
            "Confusing recall with sensitivity (they are the same thing)",
            "Not understanding that increasing recall typically decreases precision",
        ],
        practice_exercise=(
            "Train a classifier on Titanic. Calculate recall for survivors. "
            "1. What fraction of actual survivors did the model catch?\n"
            "2. If you lower the classification threshold, does recall increase or decrease?"
        ),
        quiz=[
            QuizQuestion(
                question="What does a recall of 0.70 mean for a disease detection model?",
                options=[
                    "The model correctly identifies 70% of all patients",
                    "The model catches 70% of actual disease cases, missing 30%",
                    "70% of the model's predictions are correct",
                    "The model has a 70% false positive rate",
                ],
                correct_index=1,
                explanation=(
                    "Recall measures the fraction of actual positives that are caught. "
                    "70% recall means 30% of actual disease cases are missed (false negatives). "
                    "In medical screening, this means 3 in 10 sick patients go undetected."
                ),
            ),
        ],
        takeaways=[
            "Recall = coverage of actual positives (few missed cases)",
            "High recall: when false negatives are costly",
            "Recall = TP / (TP + FN)",
            "Increasing recall typically decreases precision — there is always a tradeoff",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_09", title="F1 Score", section="evaluation", order=9,
        difficulty="beginner",
        objectives=[
            "Calculate F1 score as harmonic mean of precision and recall",
            "Interpret F1 in context",
            "Know when F1 is appropriate vs when accuracy suffices",
        ],
        concept=(
            "F1 = 2 × (Precision × Recall) / (Precision + Recall). The harmonic mean of "
            "precision and recall. A model with perfect precision (1.0) but zero recall (0.0) "
            "gets F1 = 0. It requires both metrics to be high."
        ),
        why_matters=(
            "F1 is the standard metric for imbalanced classification. It penalises models that "
            "sacrifice one metric entirely for the other. A model that predicts only one positive "
            "correctly (precision=1, recall=0.01) gets F1 ≈ 0.02, reflecting its uselessness."
        ),
        example=(
            "Model A: precision=0.90, recall=0.90 → F1 = 0.90\n"
            "Model B: precision=0.99, recall=0.30 → F1 = 0.46\n"
            "Despite B's near-perfect precision, F1 correctly identifies A as the better model."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import f1_score, fbeta_score\n\n"
            "f1 = f1_score(y_test, y_pred)\n"
            "print(f'F1: {f1:.3f}')\n\n"
            "# F-beta: beta > 1 favours recall, beta < 1 favours precision\n"
            "f2 = fbeta_score(y_test, y_pred, beta=2)  # favours recall\n"
            "f05 = fbeta_score(y_test, y_pred, beta=0.5)  # favours precision\n"
            "print(f'F2: {f2:.3f}, F0.5: {f05:.3f}')\n"
            "```"
        ),
        interpretation=(
            "F1 = 1.0 → perfect precision and recall. F1 = 0.5 → moderate. "
            "F1 = 0 → model fails completely. F1 balances the precision-recall tradeoff "
            "into a single number. Use F-beta when you want to weight one metric higher."
        ),
        common_mistakes=[
            "Using F1 when classes are balanced (accuracy may suffice)",
            "Not knowing about F-beta for custom precision/recall weighting",
            "Assuming F1 is always the right metric",
        ],
        practice_exercise=(
            "Train a classifier on Titanic. Calculate precision, recall, and F1. "
            "1. Are precision and recall similar or very different?\n"
            "2. Calculate F2 (weights recall higher). Does it change much?\n"
            "3. Which metric best represents this model's performance?"
        ),
        quiz=[
            QuizQuestion(
                question="A model has precision=0.95 and recall=0.40. What is its approximate F1 score?",
                options=["0.675", "0.562", "0.95", "0.40"],
                correct_index=1,
                explanation=(
                    "F1 = 2 × (0.95 × 0.40) / (0.95 + 0.40) = 2 × 0.38 / 1.35 = 0.563. "
                    "Despite high precision, the low recall drags F1 down significantly."
                ),
            ),
        ],
        takeaways=[
            "F1 = harmonic mean of precision and recall",
            "Best for imbalanced classification — requires both precision and recall to be high",
            "F-beta allows custom weighting: F2 favours recall, F0.5 favours precision",
            "Use accuracy when classes are balanced and all errors are equally costly",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_10", title="Confusion Matrix", section="evaluation", order=10,
        difficulty="beginner",
        objectives=[
            "Read and interpret a confusion matrix",
            "Calculate TP, TN, FP, FN from the matrix",
            "Identify systematic model errors",
        ],
        concept=(
            "The confusion matrix is a table showing predictions vs actuals:\n"
            "• True Positive (TP): correctly predicted positive\n"
            "• True Negative (TN): correctly predicted negative\n"
            "• False Positive (FP): predicted positive, actually negative (Type I error)\n"
            "• False Negative (FN): predicted negative, actually positive (Type II error)"
        ),
        why_matters=(
            "The confusion matrix reveals exactly WHERE the model makes errors. It shows "
            "which classes are confused with which — information that aggregate metrics like "
            "accuracy completely hide."
        ),
        example=(
            "Titanic survival matrix:\n"
            "• Predicted survived, actually survived: 120 (TP)\n"
            "• Predicted died, actually died: 200 (TN)\n"
            "• Predicted survived, actually died: 30 (FP — model is too optimistic)\n"
            "• Predicted died, actually survived: 50 (FN — model misses survivors)"
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import confusion_matrix\n"
            "import matplotlib.pyplot as plt\n\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "print('Confusion Matrix:')\n"
            "print(cm)\n"
            "# [[TN, FP]\n"
            "#  [FN, TP]]\n\n"
            "# Annotated heatmap\n"
            "import seaborn as sns\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
            "            xticklabels=['Predicted 0', 'Predicted 1'],\n"
            "            yticklabels=['Actual 0', 'Actual 1'])\n"
            "plt.title('Confusion Matrix')\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "Diagonal values = correct predictions. Off-diagonal = errors. "
            "Look for which off-diagonal value is largest: FP means the model is too trigger-happy; "
            "FN means it's too conservative."
        ),
        common_mistakes=[
            "Not normalising the matrix (counts are misleading for imbalanced data)",
            "Confusing rows and columns (rows = actuals, columns = predictions in sklearn)",
            "Relying on the matrix alone without computing derived metrics",
        ],
        practice_exercise=(
            "Generate a confusion matrix for a Titanic classifier. "
            "1. How many survivors were missed (FN)?\n"
            "2. How many non-survivors were incorrectly predicted as survivors (FP)?\n"
            "3. Which type of error is more costly for this problem?"
        ),
        quiz=[
            QuizQuestion(
                question="In a confusion matrix, what does a high number of False Negatives mean for a cancer detection model?",
                options=[
                    "The model generates too many false alarms",
                    "The model misses many actual cancer cases",
                    "The model is perfectly calibrated",
                    "The model predicts all cases as negative",
                ],
                correct_index=1,
                explanation=(
                    "False Negatives are actual positive cases that the model incorrectly predicts "
                    "as negative. For cancer detection, high FN means many cancer patients are told "
                    "they're healthy — a dangerous and potentially fatal error."
                ),
            ),
        ],
        takeaways=[
            "Confusion matrix: 2×2 table of TP, TN, FP, FN",
            "Diagonal = correct predictions, off-diagonal = errors",
            "Most informative single evaluation tool",
            "Normalise for imbalanced datasets to see proportions",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_11", title="TP, TN, FP, FN", section="evaluation", order=11,
        difficulty="beginner",
        objectives=[
            "Define the four confusion matrix cells precisely",
            "Understand Type I and Type II errors",
            "Calculate each from predictions and actuals",
        ],
        concept=(
            "True Positive (TP): predicted positive, actually positive. "
            "True Negative (TN): predicted negative, actually negative. "
            "False Positive (FP): predicted positive, actually negative — Type I error (false alarm). "
            "False Negative (FN): predicted negative, actually positive — Type II error (missed detection)."
        ),
        why_matters=(
            "These four values are the foundation of every classification metric. Accuracy, "
            "precision, recall, F1, specificity, and all others are computed from TP, TN, FP, FN. "
            "Understanding them is essential for interpreting any classification result."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import confusion_matrix\n\n"
            "y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]\n"
            "y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]\n\n"
            "TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()\n"
            "print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')\n"
            "# Output: TP=4, TN=4, FP=1, FN=1\n"
            "```"
        ),
        common_mistakes=[
            "Confusing FP with FN (they are fundamentally different error types)",
            "Ignoring FN costs in high-stakes applications (medical, security)",
            "Not understanding that changing the classification threshold changes TP/FP/FN counts",
        ],
        practice_exercise=(
            "Given these predictions and actuals:\n"
            "Actual:    [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]\n"
            "Predicted: [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]\n"
            "Calculate TP, TN, FP, FN. What is the accuracy?"
        ),
        quiz=[
            QuizQuestion(
                question="Which is a Type I error?",
                options=[
                    "Predicting negative when the actual is positive (FN)",
                    "Predicting positive when the actual is negative (FP)",
                    "Predicting correctly (TP or TN)",
                    "Missing a data point during evaluation",
                ],
                correct_index=1,
                explanation=(
                    "Type I error (false positive) = predicting positive when the actual is negative. "
                    "It's like a fire alarm going off when there's no fire. "
                    "Type II error (false negative) = missing a real positive case."
                ),
            ),
        ],
        takeaways=[
            "TP/TN = correct predictions; FP/FN = errors",
            "FP = false alarm (Type I error); FN = missed detection (Type II error)",
            "All classification metrics derive from these four values",
            "Changing the classification threshold shifts the TP/FP/FN balance",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_15", title="ROC Curve", section="evaluation", order=15,
        difficulty="intermediate",
        objectives=[
            "Plot and interpret the ROC curve",
            "Understand TPR vs FPR tradeoff across thresholds",
            "Compare models using ROC curves",
        ],
        concept=(
            "The ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR = Recall) "
            "against False Positive Rate (FPR = FP / (FP + TN)) at every possible classification "
            "threshold. Each point on the curve corresponds to a different threshold."
        ),
        why_matters=(
            "The ROC curve shows the full tradeoff between catching positives (TPR) and generating "
            "false alarms (FPR) across ALL thresholds simultaneously. It's threshold-independent, "
            "making it ideal for comparing models."
        ),
        example=(
            "Threshold = 0.3: high TPR (catches most positives) but high FPR (many false alarms).\n"
            "Threshold = 0.8: low TPR (misses many positives) but low FPR (few false alarms).\n"
            "The ROC curve traces this tradeoff. A good model hugs the top-left corner."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "import matplotlib.pyplot as plt\n\n"
            "# Need probability predictions for ROC\n"
            "probs = model.predict_proba(X_test)[:, 1]\n"
            "fpr, tpr, thresholds = roc_curve(y_test, probs)\n"
            "roc_auc = auc(fpr, tpr)\n\n"
            "plt.figure(figsize=(8, 6))\n"
            "plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})', linewidth=2)\n"
            "plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')\n"
            "plt.xlabel('False Positive Rate')\n"
            "plt.ylabel('True Positive Rate (Recall)')\n"
            "plt.title('ROC Curve')\n"
            "plt.legend()\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "Top-left corner = perfect model. Diagonal = random guessing (no skill). "
            "Curve above diagonal = better than random. The higher the curve, the better "
            "the model discriminates between classes."
        ),
        common_mistakes=[
            "Using ROC for highly imbalanced data (Precision-Recall curve is better)",
            "Confusing AUC with accuracy",
            "Not comparing multiple models on the same plot",
        ],
        practice_exercise=(
            "Train Logistic Regression and Random Forest on Titanic. Plot both ROC curves "
            "on the same figure. Which model has higher AUC? "
            "What does this mean about their discrimination ability?"
        ),
        quiz=[
            QuizQuestion(
                question="What does a point on the ROC curve represent?",
                options=[
                    "A specific train/test split",
                    "Model performance at a specific classification threshold",
                    "A specific feature's importance",
                    "A specific fold in cross-validation",
                ],
                correct_index=1,
                explanation=(
                    "Each point on the ROC curve corresponds to a different classification threshold. "
                    "Moving along the curve means changing the threshold, which changes the TPR/FPR "
                    "balance."
                ),
            ),
        ],
        takeaways=[
            "ROC: TPR vs FPR across all possible thresholds",
            "Higher curve = better discrimination between classes",
            "Useful for comparing models regardless of threshold",
            "Prefer Precision-Recall curve for highly imbalanced data",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_16", title="AUC (Area Under Curve)", section="evaluation", order=16,
        difficulty="intermediate",
        objectives=[
            "Calculate and interpret AUC",
            "Understand AUC as probability of correct ranking",
            "Know AUC limitations",
        ],
        concept=(
            "AUC is the area under the ROC curve. It equals the probability that the model ranks "
            "a randomly chosen positive instance higher than a randomly chosen negative instance. "
            "AUC = 0.5 = random, AUC = 1.0 = perfect, AUC < 0.5 = worse than random."
        ),
        why_matters=(
            "AUC summarises the model's discrimination ability in a single threshold-independent "
            "number. It is the standard metric for comparing binary classifiers because it "
            "accounts for all possible thresholds simultaneously."
        ),
        example=(
            "AUC = 0.92 means: if you randomly pick one positive and one negative instance, "
            "the model ranks the positive higher 92% of the time. A random model ranks them "
            "correctly only 50% of the time."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import roc_auc_score\n\n"
            "probs = model.predict_proba(X_test)[:, 1]\n"
            "auc_score = roc_auc_score(y_test, probs)\n"
            "print(f'AUC: {auc_score:.3f}')\n\n"
            "# Interpretation guide:\n"
            "# AUC 0.9-1.0: Excellent discrimination\n"
            "# AUC 0.8-0.9: Good\n"
            "# AUC 0.7-0.8: Fair\n"
            "# AUC 0.6-0.7: Poor\n"
            "# AUC 0.5-0.6: Fail (no better than random)\n"
            "```"
        ),
        common_mistakes=[
            "Using AUC for multiclass without averaging strategy (macro/micro/weighted)",
            "Assuming AUC is the only metric that matters",
            "Ignoring that AUC doesn't tell you the optimal threshold",
        ],
        practice_exercise=(
            "Calculate AUC for Logistic Regression on Titanic. "
            "Then change the classification threshold from 0.5 to 0.3. "
            "How do precision and recall change? Does AUC change?"
        ),
        quiz=[
            QuizQuestion(
                question="What does AUC = 0.5 mean?",
                options=[
                    "The model is 50% accurate",
                    "The model discriminates no better than random guessing",
                    "The model has a 50% error rate",
                    "The model is perfectly calibrated",
                ],
                correct_index=1,
                explanation=(
                    "AUC = 0.5 means the model's ROC curve follows the diagonal — it ranks "
                    "positives and negatives equally well as random chance. The model has no "
                    "discrimination ability whatsoever."
                ),
            ),
        ],
        takeaways=[
            "AUC = probability that the model ranks a random positive higher than a random negative",
            "0.5 = random (no skill), 1.0 = perfect",
            "Standard metric for comparing binary classifiers",
            "AUC is threshold-independent but doesn't tell you the optimal threshold",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_17", title="Classification Report", section="evaluation", order=17,
        difficulty="beginner",
        objectives=[
            "Read and interpret a classification report",
            "Understand per-class metrics and support",
            "Distinguish macro vs weighted averages",
        ],
        concept=(
            "The classification report shows precision, recall, and F1 for each class, plus "
            "support (number of actual instances per class). It also shows macro average "
            "(unweighted mean across classes) and weighted average (weighted by support)."
        ),
        why_matters=(
            "Aggregate metrics hide per-class problems. A classification report reveals that the "
            "model is excellent for Class A (F1=0.95) but terrible for Class B (F1=0.45), "
            "which a single accuracy number would hide."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import classification_report\n\n"
            "print(classification_report(\n"
            "    y_test, y_pred, target_names=['Not Survived', 'Survived']\n"
            "))\n"
            "#               precision  recall  f1-score  support\n"
            "# Not Survived      0.82     0.91     0.86      210\n"
            "# Survived          0.78     0.63     0.70      100\n"
            "# accuracy                              0.81      310\n"
            "# macro avg         0.80     0.77     0.78      310\n"
            "# weighted avg      0.81     0.81     0.80      310\n"
            "```"
        ),
        common_mistakes=[
            "Only looking at the accuracy line — ignore per-class details",
            "Not checking support — small classes have unreliable metrics",
            "Ignoring that macro avg treats all classes equally (penalises performance on small classes)",
        ],
        practice_exercise=(
            "Generate a classification report for Titanic. "
            "1. Which class has better recall?\n"
            "2. Is macro avg or weighted avg higher? Why?\n"
            "3. Which class does the model struggle with?"
        ),
        quiz=[
            QuizQuestion(
                question="In a classification report, macro average treats all classes equally regardless of size. What problem can this cause?",
                options=[
                    "It always overestimates performance",
                    "A rare class with poor performance is given the same weight as a common class",
                    "It requires all classes to have the same number of samples",
                    "It cannot be calculated for binary classification",
                ],
                correct_index=1,
                explanation=(
                    "Macro average gives equal weight to each class. If a rare class (50 samples) "
                    "has F1=0.3 and a common class (500 samples) has F1=0.9, macro avg = 0.6 — "
                    "hiding the poor performance on the rare class. Weighted avg accounts for this."
                ),
            ),
        ],
        takeaways=[
            "Classification report: precision, recall, F1 per class + support",
            "Check per-class performance — don't just look at overall accuracy",
            "Macro avg = unweighted (treats classes equally), weighted avg = weighted by support",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_18", title="MAE (Regression)", section="evaluation", order=18,
        difficulty="beginner",
        objectives=[
            "Calculate MAE for regression evaluation",
            "Interpret MAE in the target's units",
            "Compare MAE with RMSE",
        ],
        concept=(
            "MAE = mean(|actual - predicted|). The average absolute prediction error. "
            "Directly interpretable in the target's original units. Robust to outliers "
            "because every error is weighted equally regardless of magnitude."
        ),
        why_matters=(
            "MAE is the most intuitive regression metric. If MAE is £15,000 for house prices, "
            "you know exactly what it means: predictions are off by £15K on average. "
            "Stakeholders understand it immediately."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: £{mae:,.0f}')\n\n"
            "# Per-sample absolute errors for analysis\n"
            "abs_errors = np.abs(y_test - y_pred)\n"
            "print(f'Median AE: {np.median(abs_errors):,.0f}')\n"
            "print(f'Max AE: {abs_errors.max():,.0f}')\n"
            "```"
        ),
        common_mistakes=[
            "Reporting MSE instead of MAE (squared units are uninterpretable)",
            "Not comparing MAE with RMSE to check for outlier influence",
            "Comparing MAE across datasets with different scales",
        ],
        practice_exercise=(
            "Calculate MAE for Linear Regression and Random Forest on California Housing. "
            "1. Which model has lower MAE?\n"
            "2. Is the difference practically significant?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is MAE more robust to outliers than RMSE?",
                options=[
                    "MAE uses absolute values, so a large error of 100 contributes only 100, while RMSE contributes 10,000",
                    "MAE automatically removes outliers before calculating",
                    "MAE uses the median instead of the mean",
                    "MAE only counts errors above a threshold",
                ],
                correct_index=0,
                explanation=(
                    "MAE treats all errors equally in absolute terms. An error of 100 contributes "
                    "100. RMSE squares it, contributing 10,000. So a single large outlier can "
                    "dominate RMSE but has proportional influence in MAE."
                ),
            ),
        ],
        takeaways=[
            "MAE = average absolute error in original units",
            "Easy to interpret: 'off by £X on average'",
            "Robust to outliers (equal weight to all errors)",
            "Use MAE when you want to understand typical prediction error",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_19", title="MSE (Regression)", section="evaluation", order=19,
        difficulty="beginner",
        objectives=[
            "Calculate MSE",
            "Understand why MSE is used as a loss function",
            "Know when to report RMSE instead",
        ],
        concept=(
            "MSE = mean((actual - predicted)²). Penalises large errors more than MAE. "
            "Useful as a loss function for optimisation (smooth, differentiable) but hard to "
            "interpret directly because the units are squared."
        ),
        why_matters=(
            "MSE is the default loss function for most regression algorithms (OLS, neural networks, "
            "gradient boosting). Its mathematical properties make it ideal for gradient-based "
            "optimisation. Always report RMSE instead for interpretability."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "print(f'MSE: {mse:,.0f}')  # Squared units — hard to interpret\n\n"
            "# Always convert to RMSE for reporting\n"
            "rmse = np.sqrt(mse)\n"
            "print(f'RMSE: £{rmse:,.0f}')  # Same units as target\n"
            "```"
        ),
        common_mistakes=[
            "Reporting MSE to stakeholders (squared units are meaningless)",
            "Not converting MSE to RMSE for interpretability",
            "Forgetting that MSE penalises large errors more than MAE",
        ],
        practice_exercise=(
            "Calculate MSE for a regression model on California Housing. "
            "1. What are the units of MSE?\n"
            "2. Convert to RMSE. What are the units now?\n"
            "3. Which number is more useful to report?"
        ),
        quiz=[
            QuizQuestion(
                question="The MSE of a house price model is 2,500,000,000. What is the RMSE?",
                options=["£50,000,000", "£50,000", "£2,500", "£5,000,000"],
                correct_index=1,
                explanation=(
                    "RMSE = √MSE = √2,500,000,000 = £50,000. "
                    "The RMSE is in the same units as the target (pounds), making it directly "
                    "interpretable as the typical prediction error."
                ),
            ),
        ],
        takeaways=[
            "MSE = mean squared error — units are squared (uninterpretable)",
            "Penalises large errors more heavily than MAE",
            "Always report RMSE for interpretable units",
            "MSE is used internally as a loss function; RMSE for reporting",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_20", title="RMSE (Regression)", section="evaluation", order=20,
        difficulty="beginner",
        objectives=[
            "Calculate RMSE",
            "Interpret RMSE in the target's units",
            "Compare RMSE and MAE to detect outlier influence",
        ],
        concept=(
            "RMSE = √MSE. The most commonly reported regression metric because it's in the same "
            "units as the target. RMSE is always ≥ MAE; the difference indicates how much "
            "outliers influence the model's errors."
        ),
        why_matters=(
            "RMSE is the standard metric for regression evaluation. It's used in competitions, "
            "papers, and production. Understanding its relationship with MAE reveals the "
            "error distribution shape."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
            "import numpy as np\n\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "mae = mean_absolute_error(y_test, y_pred)\n\n"
            "print(f'RMSE: £{rmse:,.0f}')\n"
            "print(f'MAE:  £{mae:,.0f}')\n"
            "print(f'RMSE/MAE ratio: {rmse/mae:.2f}')\n"
            "# Ratio ~1.0: errors are evenly distributed\n"
            "# Ratio >1.5: a few large errors (outliers) dominate\n"
            "```"
        ),
        common_mistakes=[
            "Reporting MSE instead of RMSE (uninterpretable squared units)",
            "Not comparing RMSE with MAE to understand error distribution",
            "Assuming RMSE is always better than MAE",
        ],
        practice_exercise=(
            "Calculate RMSE and MAE for a model on California Housing. "
            "1. What is the RMSE/MAE ratio?\n"
            "2. What does it tell you about the error distribution?\n"
            "3. Remove the top 5% largest errors. How much does RMSE change vs MAE?"
        ),
        quiz=[
            QuizQuestion(
                question="RMSE/MAE ratio is approximately 1.0. What does this tell you?",
                options=[
                    "The model is very accurate",
                    "Errors are roughly uniformly distributed (no dominant outliers)",
                    "The model has no errors",
                    "The model is overfitting",
                ],
                correct_index=1,
                explanation=(
                    "RMSE and MAE converge when all errors are similar in magnitude. "
                    "A ratio close to 1.0 means there are no extreme outliers dominating RMSE. "
                    "A high ratio (e.g., 2.0) indicates a few very large errors."
                ),
            ),
        ],
        takeaways=[
            "RMSE = square root of MSE, same units as target",
            "RMSE ≥ MAE always",
            "Large RMSE/MAE gap → outlier influence",
            "Standard metric for reporting regression performance",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_21", title="R² (Regression)", section="evaluation", order=21,
        difficulty="beginner",
        objectives=[
            "Calculate and interpret R²",
            "Understand explained variance",
            "Know the limitations of R²",
        ],
        concept=(
            "R² = 1 - (SS_residuals / SS_total). The proportion of variance in the target "
            "that the model explains. R² = 0.85 means 85% of the target's variation is "
            "captured by the model. R² = 0 means the model is no better than predicting the mean."
        ),
        why_matters=(
            "R² provides a scale-independent measure of model quality. Unlike RMSE, R² = 0.85 "
            "means the same thing whether you're predicting house prices or temperatures. "
            "It's the standard metric for comparing regression models."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import r2_score\n\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.3f}')\n"
            "# 0.85 → model explains 85% of price variation\n\n"
            "# Adjusted R² (penalises for extra features)\n"
            "n = len(y_test)\n"
            "p = X_test.shape[1]  # number of features\n"
            "adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)\n"
            "print(f'Adjusted R²: {adj_r2:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "High R² doesn't mean the model is correct (could overfit or miss non-linearity)",
            "R² always increases with more features (use adjusted R² to penalise)",
            "Comparing R² across datasets with different target variance",
        ],
        practice_exercise=(
            "Calculate R² for Linear Regression and Random Forest on California Housing. "
            "1. Which is higher?\n"
            "2. Calculate adjusted R². Does the comparison change?\n"
            "3. What does 'explained variance' actually mean in this context?"
        ),
        quiz=[
            QuizQuestion(
                question="Model A has R² = 0.75, Model B has R² = 0.85. Can we conclude Model B is better?",
                options=[
                    "Yes — higher R² always means a better model",
                    "Not necessarily — Model B might be overfitting or have more features",
                    "No — R² cannot compare models",
                    "Yes — but only if they use the same dataset",
                ],
                correct_index=1,
                explanation=(
                    "Higher R² suggests better fit, but Model B might use more features "
                    "(check adjusted R²), might be overfitting (check train/test gap), or "
                    "might have violated assumptions. R² is necessary but not sufficient "
                    "for model comparison."
                ),
            ),
        ],
        takeaways=[
            "R² = proportion of variance explained (0 to 1, higher is better)",
            "R² = 0 → model predicts no better than the mean",
            "Use adjusted R² when comparing models with different numbers of features",
            "High R² doesn't guarantee a correct model — check residuals",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_22", title="Regression Residuals", section="evaluation", order=22,
        difficulty="intermediate",
        objectives=[
            "Analyse regression residuals for model diagnostics",
            "Identify non-linearity, heteroscedasticity, and outliers",
            "Use residual plots to decide if the model needs improvement",
        ],
        concept=(
            "Residual analysis checks if the model's assumptions are met: random scatter "
            "(linearity OK), constant variance (homoscedasticity OK), normal distribution. "
            "Patterns in residuals reveal problems that R² and RMSE completely miss."
        ),
        why_matters=(
            "Metrics like R² show how well the model fits. Residual analysis shows WHY it fits "
            "or doesn't, and what to fix. A model with good metrics but bad residuals is unreliable."
        ),
        python_example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from scipy import stats\n\n"
            "residuals = y_test - model.predict(X_test)\n"
            "y_pred = model.predict(X_test)\n\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n\n"
            "# 1. Residuals vs Predicted\n"
            "axes[0, 0].scatter(y_pred, residuals, alpha=0.5)\n"
            "axes[0, 0].axhline(0, color='r', linestyle='--')\n"
            "axes[0, 0].set_title('Residuals vs Predicted')\n\n"
            "# 2. Histogram\n"
            "axes[0, 1].hist(residuals, bins=30, edgecolor='black')\n"
            "axes[0, 1].set_title('Residual Distribution')\n\n"
            "# 3. Q-Q Plot\n"
            "stats.probplot(residuals, dist='norm', plot=axes[1, 0])\n"
            "axes[1, 0].set_title('Q-Q Plot')\n\n"
            "# 4. Scale-Location\n"
            "axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)\n"
            "axes[1, 1].set_title('Scale-Location')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Skipping residual analysis because metrics look good",
            "Ignoring funnel shapes (heteroscedasticity) in residual plots",
            "Not checking the Q-Q plot for normality in the tails",
        ],
        practice_exercise=(
            "Fit a linear regression on California Housing. Generate all four residual plots. "
            "For each plot, state whether the assumption is met and what remedial action to take."
        ),
        quiz=[
            QuizQuestion(
                question="A funnel shape (increasing spread) in the residuals-vs-predicted plot indicates:",
                options=[
                    "Linearity is violated",
                    "Homoscedasticity is violated (heteroscedasticity)",
                    "Normality is violated",
                    "The model is overfitting",
                ],
                correct_index=1,
                explanation=(
                    "A funnel shape where residuals spread out as predicted values increase means "
                    "the error variance is not constant (heteroscedasticity). The model is less "
                    "reliable for higher predictions. Remedies: log-transform target or use "
                    "weighted regression."
                ),
            ),
        ],
        takeaways=[
            "Residuals vs Predicted: random scatter = good, patterns = problems",
            "Q-Q plot: points on diagonal = normal residuals",
            "Funnel shape = heteroscedasticity",
            "Always check residuals — they reveal what metrics hide",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_23", title="Choosing the Right Metric", section="evaluation", order=23,
        difficulty="intermediate",
        objectives=[
            "Match metrics to problem types and business requirements",
            "Understand metric tradeoffs",
            "Justify metric selection",
        ],
        concept=(
            "No single metric is universally best. The right metric depends on: problem type "
            "(classification vs regression), class balance, the relative cost of different "
            "errors, and business requirements. Always ask: 'What type of error is most costly?'"
        ),
        why_matters=(
            "Choosing the wrong metric leads to optimising for the wrong thing. Maximising "
            "accuracy on fraud detection (0.1% fraud) produces a useless model. Optimising "
            "RMSE when MAE better represents the typical error can mislead stakeholders."
        ),
        example=(
            "Problem: Predict house prices.\n"
            "• Stakeholder wants 'typical error' → MAE\n"
            "• Large errors are very costly → RMSE (penalises them more)\n"
            "• Want to explain 'how much variance we explain' → R²\n"
            "• Comparing models on the same scale → R²\n"
            "Different metrics tell different stories."
        ),
        common_mistakes=[
            "Defaulting to accuracy for all classification problems",
            "Using RMSE when MAE better represents the typical error",
            "Not asking stakeholders which type of error matters most",
            "Reporting a single metric without context",
        ],
        practice_exercise=(
            "For each scenario, choose the most appropriate metric and justify your choice:\n"
            "1. Cancer detection (missing cancer is deadly)\n"
            "2. Spam detection (flagging legitimate email is annoying)\n"
            "3. House price prediction (typical error matters)\n"
            "4. Comparing models for a Kaggle competition"
        ),
        quiz=[
            QuizQuestion(
                question="For fraud detection where missing fraud (FN) is very costly, which metrics should you prioritise?",
                options=[
                    "Accuracy — it's the simplest metric",
                    "Recall and F1 — to minimise missed fraud cases",
                    "Precision — to minimise false accusations",
                    "R² — to explain the variance",
                ],
                correct_index=1,
                explanation=(
                    "For fraud detection, missing actual fraud (FN) is very costly. Recall measures "
                    "how many fraud cases are caught. F1 balances recall with precision. Accuracy "
                    "is misleading for this highly imbalanced problem."
                ),
            ),
        ],
        takeaways=[
            "Classification: accuracy (balanced), F1 (imbalanced), AUC (threshold-independent)",
            "Regression: RMSE (outlier-sensitive), MAE (robust), R² (explained variance)",
            "Always ask: what's the cost of false positives vs false negatives?",
            "Report multiple metrics to give a complete picture",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_24", title="Evaluation with Imbalanced Data", section="evaluation", order=24,
        difficulty="intermediate",
        objectives=[
            "Evaluate models on imbalanced datasets",
            "Use precision, recall, F1, and PR-AUC instead of accuracy",
            "Apply class weighting strategies",
        ],
        concept=(
            "For imbalanced data: accuracy is misleading (predicting majority class gives high "
            "accuracy). Use precision/recall/F1 instead. The confusion matrix shows the full "
            "picture. Precision-Recall curve is more informative than ROC for imbalanced data."
        ),
        why_matters=(
            "Most real-world classification problems are imbalanced: fraud detection (0.1%), "
            "disease diagnosis (5%), customer churn (2%). A model with 99% accuracy but 0% "
            "recall on the minority class is useless."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import classification_report, average_precision_score\n\n"
            "# Classification report — shows per-class performance\n"
            "print(classification_report(y_test, y_pred))\n\n"
            "# Average Precision (PR-AUC) — better than ROC-AUC for imbalance\n"
            "probs = model.predict_proba(X_test)[:, 1]\n"
            "ap = average_precision_score(y_test, probs)\n"
            "print(f'Average Precision: {ap:.3f}')\n\n"
            "# Handle imbalance with class weights\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "model = RandomForestClassifier(class_weight='balanced', random_state=42)\n"
            "```"
        ),
        common_mistakes=[
            "Reporting only accuracy for imbalanced problems",
            "Using ROC-AUC when PR-AUC (Average Precision) is more informative",
            "Oversampling the test set (never modify the test set)",
        ],
        practice_exercise=(
            "Create an imbalanced dataset (90% class 0, 10% class 1). "
            "1. Train a model and report accuracy. Is it misleading?\n"
            "2. Report precision, recall, F1, and Average Precision.\n"
            "3. Retrain with class_weight='balanced'. How do the metrics change?"
        ),
        quiz=[
            QuizQuestion(
                question="Why is the Precision-Recall curve more informative than ROC for imbalanced data?",
                options=[
                    "PR curves are easier to plot",
                    "PR curves focus on the minority class performance, while ROC can look good even when recall on the minority is poor",
                    "ROC curves don't work with imbalanced data",
                    "PR curves always show higher scores",
                ],
                correct_index=1,
                explanation=(
                    "In imbalanced data, the large number of TNs makes FPR low, so the ROC curve "
                    "can look good even when recall on the minority class is terrible. "
                    "The PR curve focuses on positive class performance and reveals the truth."
                ),
            ),
        ],
        takeaways=[
            "Never use accuracy alone for imbalanced data",
            "Use F1, precision, recall, and Average Precision (PR-AUC)",
            "class_weight='balanced' is the easiest fix for imbalance",
            "PR curve is more informative than ROC for imbalanced data",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_25", title="Threshold Selection", section="evaluation", order=25,
        difficulty="advanced",
        objectives=[
            "Choose the optimal classification threshold",
            "Use precision-recall curves for threshold selection",
            "Balance FP and FN costs based on application requirements",
        ],
        concept=(
            "The default 0.5 threshold isn't always optimal. Different thresholds create "
            "different precision/recall tradeoffs. Lowering the threshold increases recall "
            "(more positives detected) but decreases precision (more false alarms)."
        ),
        why_matters=(
            "Threshold selection directly impacts real-world outcomes. For cancer screening: "
            "lowering the threshold catches more cancers (higher recall) but has more false "
            "alarms (lower precision). The optimal threshold depends on the relative cost "
            "of false positives vs false negatives."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import precision_recall_curve\n"
            "import numpy as np\n\n"
            "probs = model.predict_proba(X_test)[:, 1]\n"
            "precisions, recalls, thresholds = precision_recall_curve(y_test, probs)\n\n"
            "# Find threshold that maximises F1\n"
            "f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)\n"
            "optimal_idx = np.argmax(f1_scores)\n"
            "optimal_threshold = thresholds[optimal_idx]\n"
            "print(f'Optimal threshold: {optimal_threshold:.3f}')\n"
            "print(f'At this threshold: P={precisions[optimal_idx]:.3f}, R={recalls[optimal_idx]:.3f}')\n\n"
            "# Apply custom threshold\n"
            "y_pred_custom = (probs >= 0.3).astype(int)  # lower threshold → more positives\n"
            "```"
        ),
        common_mistakes=[
            "Always using the default 0.5 threshold",
            "Optimising the threshold on test data (use validation set instead)",
            "Not accounting for class distribution changes in production",
        ],
        practice_exercise=(
            "Train a classifier on Titanic. Compute predictions with thresholds "
            "[0.3, 0.4, 0.5, 0.6, 0.7]. For each, calculate precision and recall. "
            "1. Plot precision and recall vs threshold.\n"
            "2. Which threshold maximises F1?\n"
            "3. Which threshold would you choose for this problem?"
        ),
        quiz=[
            QuizQuestion(
                question="You lower the classification threshold from 0.5 to 0.3. What happens to precision and recall?",
                options=[
                    "Both increase",
                    "Precision increases, recall decreases",
                    "Precision decreases, recall increases",
                    "Both decrease",
                ],
                correct_index=2,
                explanation=(
                    "Lowering the threshold means more instances are classified as positive. "
                    "This catches more true positives (higher recall) but also classifies more "
                    "negatives as positive (lower precision). There's always a tradeoff."
                ),
            ),
        ],
        takeaways=[
            "Default 0.5 threshold is rarely optimal",
            "Lower threshold → more positives → higher recall, lower precision",
            "Choose threshold based on the relative cost of FP vs FN",
            "Use the precision-recall curve to find the optimal threshold",
        ],
        lab_module="evaluation",
    ),
    Topic(
        id="eval_26", title="Model Evaluation Case Study", section="evaluation", order=26,
        difficulty="advanced",
        objectives=[
            "Apply the complete evaluation workflow end-to-end",
            "Compare models with multiple metrics",
            "Document and justify evaluation decisions",
        ],
        concept=(
            "A complete evaluation workflow: split data → train model → compute classification "
            "or regression metrics → visualise → cross-validate → compare with baseline → "
            "document. Evaluation is not just computing one number — it's building evidence."
        ),
        why_matters=(
            "Evaluation is not just computing one number. A thorough evaluation provides the "
            "evidence needed to select and justify a model choice. This evidence is what "
            "stakeholders and examiners need to trust your results."
        ),
        example=(
            "Titanic survival prediction — complete evaluation:\n"
            "1. Split: 80/20 stratified\n"
            "2. Metrics: Accuracy=0.81, F1=0.74, AUC=0.85\n"
            "3. Confusion matrix: 15 missed survivors (FN)\n"
            "4. 5-fold CV: accuracy=0.80 ± 0.03\n"
            "5. Comparison: RF=0.81, LR=0.79, GB=0.83\n"
            "6. Decision: Gradient Boosting with best F1 and AUC"
        ),
        common_mistakes=[
            "Reporting a single metric (tells an incomplete story)",
            "Not using cross-validation for reliable estimates",
            "Not documenting which metrics were used and why",
        ],
        practice_exercise=(
            "Complete an evaluation case study:\n"
            "1. Train two different classifiers on Titanic\n"
            "2. Report accuracy, precision, recall, F1, confusion matrix, and AUC for both\n"
            "3. Perform 5-fold cross-validation\n"
            "4. Compare and justify your model choice"
        ),
        quiz=[
            QuizQuestion(
                question="What is the most complete way to evaluate a classification model?",
                options=[
                    "Report accuracy only",
                    "Report accuracy and R²",
                    "Report multiple metrics (accuracy, precision, recall, F1, AUC, confusion matrix) with cross-validation",
                    "Just look at the training score",
                ],
                correct_index=2,
                explanation=(
                    "A thorough evaluation uses multiple metrics that capture different aspects "
                    "of model performance, cross-validation for reliability, and a confusion matrix "
                    "for error analysis. No single metric tells the complete story."
                ),
            ),
        ],
        takeaways=[
            "Evaluate with multiple metrics, not just one",
            "Use cross-validation for reliable performance estimates",
            "Visualise results (confusion matrix, ROC curve)",
            "Document all evaluation decisions and results",
        ],
        lab_module="evaluation",
    ),
]
