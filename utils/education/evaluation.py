"""
Evaluation — learning topics for the Model Evaluation module.
"""
from utils.education.base import T

TOPICS = {
    # ── 1 ──────────────────────────────────────────────────────────
    "why_evaluation_matters": T(
        title="Why Model Evaluation Matters",
        module="model_evaluation",
        what=(
            "Model evaluation measures how well a trained model performs "
            "on unseen data. Without evaluation, you have no idea if your "
            "model is useful or just memorising training data."
        ),
        why=(
            "An unevaluated model is an unvalidated model. Evaluation "
            "proves that your model generalises to new data and identifies "
            "weaknesses before deployment."
        ),
        when=(
            "After every training step. Evaluate on the test set exactly "
            "once — at the very end. Use cross-validation during "
            "development."
        ),
        example=(
            "Two models: Model A has 95% training accuracy but 70% test. "
            "Model B has 88% training and 87% test. Model B is better "
            "because it generalises."
        ),
        mistakes=[
            "Evaluating on training data — gives misleadingly high scores.",
            "Changing the model based on test set results — that IS using test data.",
            "Using only one metric — different metrics reveal different weaknesses.",
        ],
        interpretation=(
            "Good evaluation answers: How accurate is the model? Where "
            "does it fail? Is it overfitting? Would it work in production?"
        ),
        think_about_it=(
            "A model gets 99% accuracy on the test set. Does this mean "
            "it's ready for deployment? What else should you check?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# Cross-validation (recommended)\n"
            "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "print(f'CV: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "\n"
            "# Final evaluation on test set (do once!)\n"
            "test_score = model.score(X_test, y_test)\n"
            "print(f'Test: {test_score:.4f}')\n"
            "```"
        ),
        keywords=["evaluation", "performance", "generalise", "unseen", "metrics"],
    ),

    # ── 2 ──────────────────────────────────────────────────────────
    "training_set": T(
        title="The Training Set",
        module="model_evaluation",
        what=(
            "The training set is the portion of data used to teach the "
            "model. The model learns patterns, coefficients, and "
            "decision rules from this data."
        ),
        why=(
            "The training set determines what the model learns. "
            "Its size and quality directly affect model performance. "
            "Too small → underfitting. Too biased → biased model."
        ),
        when=(
            "Always split data before training. Typical split: "
            "80% training, 20% test. For small datasets, use "
            "cross-validation instead."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "print(f'Training: {len(X_train)} samples')\n"
            "print(f'Testing:  {len(X_test)} samples')\n"
            "```"
        ),
        mistakes=[
            "Training on the entire dataset including test data.",
            "Not checking that train and test distributions match.",
            "Using too small a training set (<60% of data).",
        ],
        interpretation=(
            "The training set should be representative of the full "
            "dataset. Use stratify for classification to ensure "
            "class proportions match."
        ),
        think_about_it=(
            "If you have 1000 samples and use 90% for training, "
            "is this always better than using 80%? Why or why not?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Classification: use stratify\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y, random_state=42\n"
            ")\n"
            "\n"
            "# Check distribution matches\n"
            "print(y_train.value_counts(normalize=True))\n"
            "print(y_test.value_counts(normalize=True))\n"
            "```"
        ),
        keywords=["training", "train", "split", "fit", "learn"],
    ),

    # ── 3 ──────────────────────────────────────────────────────────
    "test_set": T(
        title="The Test Set",
        module="model_evaluation",
        what=(
            "The test set is held-out data never seen during training. "
            "It simulates real-world performance and gives an unbiased "
            "estimate of model quality."
        ),
        why=(
            "Without a test set, you can't know if your model "
            "generalises. Training accuracy is always optimistic. "
            "The test set provides the real performance number."
        ),
        when=(
            "Touch the test set exactly once: at the very end, after "
            "all development is complete. Every peek at the test set "
            "biases your results."
        ),
        example=(
            "```python\n"
            "# After all development is done:\n"
            "final_score = best_model.score(X_test, y_test)\n"
            "print(f'Final test accuracy: {final_score:.4f}')\n"
            "# Report this number in your paper/report\n"
            "```"
        ),
        mistakes=[
            "Peeking at test performance and changing the model — that IS using test data.",
            "Not keeping the test set completely separate.",
            "Evaluating multiple models on test and picking the best — use validation for that.",
        ],
        interpretation=(
            "The test score is your model's expected performance on "
            "new, unseen data. It's the number you report to stakeholders."
        ),
        think_about_it=(
            "You try 10 different models and pick the one with the "
            "highest test accuracy. Is this test accuracy still unbiased? "
            "What should you have done instead?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Split once, at the beginning\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "# X_test is sacred — don't touch until the very end!\n"
            "```"
        ),
        keywords=["test", "holdout", "unseen", "final", "evaluation"],
    ),

    # ── 4 ──────────────────────────────────────────────────────────
    "validation_set": T(
        title="Validation Set",
        module="model_evaluation",
        what=(
            "A validation set is a third split used during development "
            "for model selection and hyperparameter tuning. It's separate "
            "from both training and test sets."
        ),
        why=(
            "Using the test set for model selection leaks information. "
            "The validation set provides a 'practice test' during "
            "development, keeping the test set for the final exam."
        ),
        when=(
            "When you need to compare models or tune hyperparameters. "
            "Typical split: 60% train, 20% validation, 20% test. "
            "Cross-validation replaces a fixed validation set."
        ),
        example=(
            "```python\n"
            "# Three-way split\n"
            "X_trainval, X_test, y_trainval, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "X_train, X_val, y_train, y_val = train_test_split(\n"
            "    X_trainval, y_trainval, test_size=0.25, random_state=42\n"
            ")\n"
            "# Train on X_train, tune on X_val, final test on X_test\n"
            "```"
        ),
        mistakes=[
            "Using test set for hyperparameter tuning — data leakage!",
            "Making the validation set too small (<10% of data).",
            "Not using the same preprocessing pipeline on all splits.",
        ],
        interpretation=(
            "The validation set acts as a proxy for the test set during "
            "development. Cross-validation is preferred because it uses "
            "all data for both training and validation across folds."
        ),
        think_about_it=(
            "Cross-validation vs fixed validation set: what are the "
            "trade-offs for a dataset of 500 samples?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# Cross-validation replaces a fixed validation set\n"
            "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "print(f'CV mean: {scores.mean():.4f}')\n"
            "```"
        ),
        keywords=["validation", "tuning", "selection", "hyperparameter", "cv"],
    ),

    # ── 5 ──────────────────────────────────────────────────────────
    "holdout_validation": T(
        title="Holdout Validation",
        module="model_evaluation",
        what=(
            "Holdout validation splits data into train/test once. "
            "It's the simplest evaluation method. The test set is "
            "held out and used only for final evaluation."
        ),
        why=(
            "Holdout is fast and simple. For large datasets (>10K "
            "samples), it provides reliable estimates. For small "
            "datasets, cross-validation is better."
        ),
        when=(
            "Use for large datasets where a single split gives enough "
            "data in each set. Always set random_state for reproducibility."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "model.fit(X_train, y_train)\n"
            "print(f'Holdout score: {model.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using holdout on small datasets — high variance in estimates.",
            "Not setting random_state — different splits give different scores.",
            "Changing test_size without documenting it.",
        ],
        interpretation=(
            "Holdout gives one performance estimate. Cross-validation "
            "gives a distribution of estimates (mean + std), which is "
            "more informative."
        ),
        think_about_it=(
            "Holdout gives 85% accuracy. Cross-validation gives "
            "83% ± 3%. Which is more trustworthy and why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "```"
        ),
        keywords=["holdout", "split", "simple", "train-test", "evaluation"],
    ),

    # ── 6 ──────────────────────────────────────────────────────────
    "cross_validation": T(
        title="Cross-Validation",
        module="model_evaluation",
        what=(
            "Cross-validation splits data into k folds, trains on k-1, "
            "tests on 1, and repeats k times. Every sample is used for "
            "both training and testing across folds."
        ),
        why=(
            "A single split can be lucky or unlucky. Cross-validation "
            "gives a more reliable estimate by averaging over multiple "
            "splits. It's the gold standard for model evaluation."
        ),
        when=(
            "Always prefer CV over holdout for model selection. Use "
            "k=5 or k=10. For time series, use TimeSeriesSplit."
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
        mistakes=[
            "Using CV on time-series data without respecting temporal order.",
            "Reporting only mean without standard deviation.",
            "Using too few folds (2-3) — high variance.",
        ],
        interpretation=(
            "Mean CV score ≈ expected test performance. Std shows "
            "stability. High std means the model is sensitive to "
            "which data it trains on."
        ),
        think_about_it=(
            "Model A: CV accuracy 0.92 ± 0.03. Model B: 0.91 ± 0.01. "
            "Which would you prefer and why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, KFold\n"
            "\n"
            "# Standard 5-fold\n"
            "scores = cross_val_score(model, X, y, cv=5)\n"
            "\n"
            "# Custom CV splitter\n"
            "kf = KFold(n_splits=10, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=kf)\n"
            "```"
        ),
        keywords=["cross-validation", "cv", "fold", "k-fold", "reliable"],
    ),

    # ── 7 ──────────────────────────────────────────────────────────
    "accuracy": T(
        title="Accuracy",
        module="model_evaluation",
        what=(
            "Accuracy is the proportion of correct predictions: "
            "Accuracy = (TP + TN) / (TP + TN + FP + FN). It's the "
            "most intuitive but often misleading metric."
        ),
        why=(
            "Accuracy is easy to understand but dangerous with "
            "imbalanced classes. A model predicting 'No' for every "
            "transaction achieves 99.9% accuracy on fraud data — "
            "while catching zero fraud."
        ),
        when=(
            "Use only when classes are roughly balanced (40-60% split). "
            "For imbalanced data, use F1, precision, recall, or AUC."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import accuracy_score\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "acc = accuracy_score(y_test, y_pred)\n"
            "print(f'Accuracy: {acc:.4f}')  # 0.95\n"
            "```"
        ),
        mistakes=[
            "Using accuracy on imbalanced datasets — misleading.",
            "Reporting accuracy without context (95% may be terrible).",
            "Comparing accuracy across datasets with different class distributions.",
        ],
        interpretation=(
            "95% accuracy means 95 out of 100 predictions are correct. "
            "But if 95% of the data is one class, a model predicting "
            "that class always achieves 95% accuracy with zero learning."
        ),
        think_about_it=(
            "A spam classifier has 99% accuracy. Is this necessarily good? "
            "What if only 1% of emails are spam?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import accuracy_score\n"
            "\n"
            "acc = accuracy_score(y_test, y_pred)\n"
            "print(f'Accuracy: {acc:.4f}')\n"
            "\n"
            "# Equivalent to model.score()\n"
            "print(f'Accuracy: {model.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["accuracy", "correct", "proportion", "simple", "misleading"],
    ),

    # ── 8 ──────────────────────────────────────────────────────────
    "precision": T(
        title="Precision",
        module="model_evaluation",
        what=(
            "Precision measures the quality of positive predictions: "
            "Precision = TP / (TP + FP). Of all predicted positives, "
            "how many are actually positive?"
        ),
        why=(
            "Precision matters when false positives are costly: "
            "flagging legitimate emails as spam, diagnosing healthy "
            "patients as sick, approving unqualified loan applicants."
        ),
        when=(
            "Use when FP cost is high. Email spam: high precision "
            "means fewer legitimate emails go to spam. "
            "Use precision_score(y_true, y_pred)."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import precision_score\n"
            "\n"
            "precision = precision_score(y_test, y_pred)\n"
            "print(f'Precision: {precision:.4f}')  # 0.89\n"
            "# Of all predicted 'spam', 89% are actually spam\n"
            "```"
        ),
        mistakes=[
            "Confusing precision with accuracy.",
            "Using precision alone without recall.",
            "Not specifying average='weighted' for multiclass.",
        ],
        interpretation=(
            "Precision=0.89 means 89% of positive predictions are "
            "correct. The remaining 11% are false positives."
        ),
        think_about_it=(
            "A medical test has precision=0.70. This means 30% of "
            "positive diagnoses are wrong (healthy people told they're "
            "sick). Is this acceptable?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import precision_score\n"
            "\n"
            "# Binary\n"
            "precision = precision_score(y_test, y_pred)\n"
            "\n"
            "# Multiclass\n"
            "precision = precision_score(y_test, y_pred, average='weighted')\n"
            "print(f'Precision: {precision:.4f}')\n"
            "```"
        ),
        keywords=["precision", "positive", "prediction", "quality", "false positive"],
    ),

    # ── 9 ──────────────────────────────────────────────────────────
    "recall": T(
        title="Recall",
        module="model_evaluation",
        what=(
            "Recall measures the ability to find all positive samples: "
            "Recall = TP / (TP + FN). Of all actual positives, how many "
            "did the model find?"
        ),
        why=(
            "Recall matters when false negatives are costly: missing "
            "cancer diagnosis, failing to detect fraud, not catching "
            "a safety defect."
        ),
        when=(
            "Use when FN cost is high. Medical screening: high recall "
            "means few diseases are missed. "
            "Use recall_score(y_true, y_pred)."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import recall_score\n"
            "\n"
            "recall = recall_score(y_test, y_pred)\n"
            "print(f'Recall: {recall:.4f}')  # 0.85\n"
            "# Of all actual 'cancer' cases, model found 85%\n"
            "```"
        ),
        mistakes=[
            "Confusing recall with precision.",
            "Using recall alone without precision.",
            "Ignoring that increasing recall typically decreases precision.",
        ],
        interpretation=(
            "Recall=0.85 means 85% of actual positive cases are "
            "detected. The remaining 15% are false negatives (missed)."
        ),
        think_about_it=(
            "A cancer screening has recall=0.95 but precision=0.40. "
            "This catches 95% of cancers but 60% of positive results "
            "are false alarms. Is this acceptable?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import recall_score\n"
            "\n"
            "recall = recall_score(y_test, y_pred)\n"
            "print(f'Recall: {recall:.4f}')\n"
            "```"
        ),
        keywords=["recall", "sensitivity", "true positive rate", "coverage", "find"],
    ),

    # ── 10 ─────────────────────────────────────────────────────────
    "f1_score": T(
        title="F1 Score",
        module="model_evaluation",
        what=(
            "F1 is the harmonic mean of precision and recall: "
            "F1 = 2 × (Precision × Recall) / (Precision + Recall). "
            "It balances both metrics into a single number."
        ),
        why=(
            "F1 is preferred when you need a balance between precision "
            "and recall. It penalises extreme imbalances (e.g., perfect "
            "precision but zero recall)."
        ),
        when=(
            "Use for imbalanced datasets where accuracy is misleading. "
            "F1 is a better single-number summary than accuracy."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import f1_score\n"
            "\n"
            "f1 = f1_score(y_test, y_pred)\n"
            "print(f'F1: {f1:.4f}')  # 0.87\n"
            "\n"
            "# Multiclass\n"
            "f1_weighted = f1_score(y_test, y_pred, average='weighted')\n"
            "```"
        ),
        mistakes=[
            "Using F1 without understanding the precision-recall trade-off.",
            "Not specifying average for multiclass problems.",
            "Using F1 when both false positives and negatives have equal cost — use accuracy.",
        ],
        interpretation=(
            "F1=0.87 means the model balances precision and recall well. "
            "F1 ranges 0-1; higher is better. F1 is always ≤ min(precision, recall)."
        ),
        think_about_it=(
            "Model A: Precision=0.95, Recall=0.60 → F1=0.74. "
            "Model B: Precision=0.80, Recall=0.80 → F1=0.80. "
            "Which model is better for a balanced problem?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import f1_score\n"
            "\n"
            "f1 = f1_score(y_test, y_pred)\n"
            "print(f'F1: {f1:.4f}')\n"
            "\n"
            "# Compare precision, recall, and F1\n"
            "from sklearn.metrics import precision_score, recall_score\n"
            "print(f'P={precision_score(y_test, y_pred):.3f}, '\n"
            "      f'R={recall_score(y_test, y_pred):.3f}, '\n"
            "      f'F1={f1_score(y_test, y_pred):.3f}')\n"
            "```"
        ),
        keywords=["f1", "harmonic mean", "balance", "precision", "recall"],
    ),

    # ── 11 ─────────────────────────────────────────────────────────
    "confusion_matrix": T(
        title="Confusion Matrix",
        module="model_evaluation",
        what=(
            "A confusion matrix shows the four outcomes of a binary "
            "classifier: True Positives, True Negatives, False "
            "Positives, and False Negatives in a table."
        ),
        why=(
            "The confusion matrix reveals WHERE the model makes errors. "
            "Accuracy tells you the rate; the matrix tells you the type."
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
            "```"
        ),
        mistakes=[
            "Reading the matrix rows/columns backwards — check axis labels.",
            "Ignoring the matrix when accuracy looks good.",
            "Not normalising for imbalanced classes.",
        ],
        interpretation=(
            "Diagonal = correct predictions. Off-diagonal = errors. "
            "Look for which classes are confused with each other."
        ),
        think_about_it=(
            "Confusion matrix shows 0 false negatives but many false "
            "positives. What kind of problem is this model suited for?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import confusion_matrix\n"
            "import plotly.graph_objects as go\n"
            "\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "fig = go.Figure(data=go.Heatmap(z=cm, colorscale='Blues'))\n"
            "fig.update_layout(title='Confusion Matrix')\n"
            "fig.show()\n"
            "```"
        ),
        keywords=["confusion", "matrix", "tp", "tn", "fp", "fn", "true positive"],
    ),

    # ── 12 ─────────────────────────────────────────────────────────
    "true_positive": T(
        title="True Positive, True Negative, False Positive, False Negative",
        module="model_evaluation",
        what=(
            "The four building blocks of classification evaluation: "
            "TP = correctly predicted positive, TN = correctly predicted "
            "negative, FP = incorrectly predicted positive (Type I error), "
            "FN = incorrectly predicted negative (Type II error)."
        ),
        why=(
            "Understanding TP/TN/FP/FN is essential for computing "
            "precision, recall, F1, and accuracy. Each type of error "
            "has different real-world consequences."
        ),
        when=(
            "After making predictions. Compute confusion matrix to get "
            "all four values."
        ),
        example=(
            "```\n"
            "Cancer diagnosis:\n"
            "TP = Cancer detected, patient has cancer (correct)\n"
            "TN = No cancer detected, patient is healthy (correct)\n"
            "FP = Cancer detected, patient is healthy (false alarm)\n"
            "FN = No cancer detected, patient has cancer (MISSED!)\n"
            "```"
        ),
        mistakes=[
            "Confusing FP and FN — FP = predicted positive, FN = missed positive.",
            "Treating all errors as equal — FP and FN have different costs.",
            "Not naming the positive class explicitly.",
        ],
        interpretation=(
            "Accuracy = (TP+TN)/(TP+TN+FP+FN). "
            "Precision = TP/(TP+FP). "
            "Recall = TP/(TP+FN). "
            "The cost of FP vs FN determines which metric matters most."
        ),
        think_about_it=(
            "In airport security screening, is a false positive or "
            "false negative more dangerous? How does this affect "
            "the choice of metric?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import confusion_matrix\n"
            "\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "tn, fp, fn, tp = cm.ravel()\n"
            "print(f'TP={tp}, TN={tn}, FP={fp}, FN={fn}')\n"
            "```"
        ),
        keywords=["true positive", "true negative", "false positive", "false negative", "tp", "tn"],
    ),

    # ── 13 ─────────────────────────────────────────────────────────
    "roc_curve": T(
        title="ROC Curve",
        module="model_evaluation",
        what=(
            "The ROC curve plots True Positive Rate (recall) vs False "
            "Positive Rate at different classification thresholds. "
            "It shows the trade-off between sensitivity and specificity."
        ),
        why=(
            "ROC visualises classifier performance across ALL thresholds, "
            "not just 0.5. It helps choose the optimal threshold and "
            "compare models independently of threshold choice."
        ),
        when=(
            "Use when comparing classifiers, choosing thresholds, or "
            "when the cost of FP vs FN is unknown. Less useful for "
            "highly imbalanced data."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "fpr, tpr, thresholds = roc_curve(y_test, y_prob)\n"
            "roc_auc = auc(fpr, tpr)\n"
            "print(f'AUC: {roc_auc:.3f}')\n"
            "```"
        ),
        mistakes=[
            "Using ROC on highly imbalanced data — PR curve is better.",
            "Forgetting to use predict_proba() not predict().",
            "Interpreting the curve at a single point — it shows all thresholds.",
        ],
        interpretation=(
            "Top-left corner = perfect classifier. Diagonal = random "
            "guessing. A curve close to the top-left indicates strong "
            "performance."
        ),
        think_about_it=(
            "Two models have the same AUC but different ROC curves. "
            "At low FPR, Model A is better; at high FPR, Model B is "
            "better. Which should you choose?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "import plotly.graph_objects as go\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "fpr, tpr, _ = roc_curve(y_test, y_prob)\n"
            "fig = go.Figure()\n"
            "fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC={auc(fpr, tpr):.3f}'))\n"
            "fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random'))\n"
            "fig.update_layout(xaxis_title='FPR', yaxis_title='TPR')\n"
            "```"
        ),
        keywords=["roc", "curve", "threshold", "fpr", "tpr", "trade-off"],
    ),

    # ── 14 ─────────────────────────────────────────────────────────
    "auc": T(
        title="AUC (Area Under the ROC Curve)",
        module="model_evaluation",
        what=(
            "AUC summarises the ROC curve into a single number: "
            "probability that a random positive is ranked higher "
            "than a random negative. 1.0 = perfect, 0.5 = random."
        ),
        why=(
            "AUC is threshold-independent — it measures ranking quality "
            "regardless of where you set the decision threshold. "
            "It's the standard metric for comparing classifiers."
        ),
        when=(
            "Use for binary classification comparison. AUC=0.9+ is "
            "excellent, 0.8-0.9 is good, 0.7-0.8 is fair, <0.7 is poor."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import roc_auc_score\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "auc = roc_auc_score(y_test, y_prob)\n"
            "print(f'AUC: {auc:.4f}')  # 0.95 = excellent\n"
            "```"
        ),
        mistakes=[
            "Using AUC with highly imbalanced data — can be misleadingly high.",
            "Confusing AUC with accuracy.",
            "Not using probabilities (predict_proba) for AUC.",
        ],
        interpretation=(
            "AUC=0.95 means: if you pick one positive and one negative "
            "sample randomly, the model ranks the positive higher 95% "
            "of the time."
        ),
        think_about_it=(
            "Model A: AUC=0.90, Accuracy=0.85. Model B: AUC=0.85, "
            "Accuracy=0.90. How can this happen, and which is better?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import roc_auc_score\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "print(f'AUC: {roc_auc_score(y_test, y_prob):.4f}')\n"
            "```"
        ),
        keywords=["auc", "area", "under", "curve", "ranking", "probability"],
    ),

    # ── 15 ─────────────────────────────────────────────────────────
    "classification_report": T(
        title="Classification Report",
        module="model_evaluation",
        what=(
            "classification_report from sklearn produces precision, "
            "recall, F1-score, and support for each class in one call. "
            "It's the most comprehensive classification summary."
        ),
        why=(
            "Instead of computing metrics one by one, the classification "
            "report gives you everything in a readable table. "
            "It's essential for multiclass problems."
        ),
        when=(
            "Always generate a classification report after training. "
            "It reveals per-class performance that aggregate metrics hide."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import classification_report\n"
            "\n"
            "print(classification_report(y_test, y_pred))\n"
            "#              precision  recall  f1-score  support\n"
            "#     No         0.95     0.98     0.96       80\n"
            "#     Yes        0.93     0.85     0.89       20\n"
            "#  accuracy                         0.95      100\n"
            "# macro avg      0.94     0.91     0.92      100\n"
            "# weighted avg   0.95     0.95     0.94      100\n"
            "```"
        ),
        mistakes=[
            "Ignoring per-class metrics — overall F1 can mask a failed class.",
            "Not using weighted average for imbalanced multiclass.",
            "Not looking at support (sample count per class).",
        ],
        interpretation=(
            "support = number of samples per class. macro avg treats "
            "all classes equally. weighted avg weights by support. "
            "Use weighted for imbalanced datasets."
        ),
        think_about_it=(
            "A multiclass report shows class 'Cancer' has recall=0.60 "
            "while other classes have >0.90. What should you do?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import classification_report\n"
            "\n"
            "print(classification_report(y_test, y_pred))\n"
            "print(classification_report(y_test, y_pred, digits=4))  # more precision\n"
            "```"
        ),
        keywords=["report", "classification", "precision", "recall", "f1", "support"],
    ),

    # ── 16 ─────────────────────────────────────────────────────────
    "regression_mae_metric": T(
        title="MAE for Regression",
        module="model_evaluation",
        what=(
            "Mean Absolute Error (MAE) measures average absolute "
            "prediction error: MAE = mean(|y - ŷ|). It's in original "
            "units and robust to outliers."
        ),
        why=(
            "MAE tells you the typical prediction error in real units. "
            "If MAE=$25K for house prices, you're off by $25K on average."
        ),
        when=(
            "Use for interpretable error measurement. Report alongside "
            "RMSE to understand error distribution."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: ${mae:,.0f}')\n"
            "```"
        ),
        mistakes=[
            "Comparing MAE across datasets with different scales.",
            "Ignoring MAE when RMSE is much larger (outlier indicator).",
        ],
        interpretation=(
            "MAE is the median-like error measure. It doesn't over-weight "
            "large errors like RMSE does."
        ),
        think_about_it=(
            "Model A: MAE=$20K, RMSE=$40K. Model B: MAE=$25K, RMSE=$30K. "
            "What does the RMSE/MAE ratio tell you about each model?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: {mae:.2f}')\n"
            "```"
        ),
        keywords=["mae", "mean absolute", "regression", "error", "interpret"],
    ),

    # ── 17 ─────────────────────────────────────────────────────────
    "regression_mse_metric": T(
        title="MSE for Regression",
        module="model_evaluation",
        what=(
            "Mean Squared Error (MSE) = mean((y - ŷ)²). It penalises "
            "large errors more than small ones due to squaring."
        ),
        why=(
            "MSE is the mathematical foundation for many regression "
            "algorithms (it's what they minimise). It amplifies large "
            "errors."
        ),
        when=(
            "Use when large errors are disproportionately bad. "
            "Always convert to RMSE for interpretability."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "rmse = np.sqrt(mse)\n"
            "print(f'MSE: {mse:,.0f}, RMSE: {rmse:,.0f}')\n"
            "```"
        ),
        mistakes=[
            "Reporting MSE directly — units are squared, hard to interpret.",
            "Not using RMSE alongside MSE.",
        ],
        interpretation=(
            "MSE > MAE² always. The gap indicates outlier influence."
        ),
        think_about_it=(
            "Why is RMSE more commonly reported than MSE?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "print(f'MSE: {mse:.2f}')\n"
            "```"
        ),
        keywords=["mse", "mean squared", "regression", "squared", "penalty"],
    ),

    # ── 18 ─────────────────────────────────────────────────────────
    "regression_rmse_metric": T(
        title="RMSE for Regression",
        module="model_evaluation",
        what=(
            "Root Mean Squared Error = √MSE. It returns error to "
            "original units while retaining MSE's sensitivity to "
            "large errors."
        ),
        why=(
            "RMSE is the most reported regression metric because "
            "it's interpretable (original units) and penalises "
            "large errors."
        ),
        when=(
            "Use as the primary regression metric. Always report "
            "RMSE alongside MAE."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.metrics import mean_squared_error\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "print(f'RMSE: ${rmse:,.0f}')  # Average error ~$25,000\n"
            "```"
        ),
        mistakes=[
            "Reporting RMSE without context (relative to target range).",
            "Ignoring RMSE/MAE ratio as an outlier diagnostic.",
        ],
        interpretation=(
            "RMSE/MAE > 1.5 suggests significant outlier errors."
        ),
        think_about_it=(
            "Your model has RMSE=$50K on houses ranging $100K-$500K. "
            "Is this good? How would you contextualise it?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.metrics import mean_squared_error\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "print(f'RMSE: {rmse:.2f}')\n"
            "```"
        ),
        keywords=["rmse", "root", "mean squared", "regression", "interpretable"],
    ),

    # ── 19 ─────────────────────────────────────────────────────────
    "regression_r2_metric": T(
        title="R² for Regression",
        module="model_evaluation",
        what=(
            "R² measures the proportion of variance explained: "
            "R² = 1 - SS_res/SS_tot. Range: -∞ to 1. "
            "R²=1 is perfect, R²=0 is predicting the mean."
        ),
        why=(
            "R² is scale-free and allows comparison across datasets. "
            "It answers: 'How much of the target variation does the "
            "model explain?'"
        ),
        when=(
            "Use as the primary regression comparison metric. "
            "Combine with RMSE for a complete picture."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import r2_score\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.4f}')  # 0.85 = explains 85% of variance\n"
            "```"
        ),
        mistakes=[
            "R² can be negative — worse than predicting the mean.",
            "Not using adjusted R² when comparing models with different features.",
            "Assuming high R² means good predictions.",
        ],
        interpretation=(
            "R²=0.85: model captures 85% of price variation. "
            "15% remains unexplained."
        ),
        think_about_it=(
            "You have R²=0.99 with 50 features and R²=0.95 with "
            "3 features. Which model would you prefer for production?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import r2_score\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.4f}')\n"
            "```"
        ),
        keywords=["r2", "r-squared", "variance", "explained", "regression"],
    ),

    # ── 20 ─────────────────────────────────────────────────────────
    "regression_residuals_metric": T(
        title="Regression Residual Analysis",
        module="model_evaluation",
        what=(
            "Residual analysis examines prediction errors to diagnose "
            "model problems. Residual = actual - predicted. Patterns "
            "in residuals indicate specific issues."
        ),
        why=(
            "Residual plots reveal whether the model assumptions hold: "
            "linearity, homoscedasticity, normality. Without this "
            "diagnostic, you're flying blind."
        ),
        when=(
            "After training any regression model. Always check "
            "residual plots before reporting results."
        ),
        example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "residuals = y_test - model.predict(X_test)\n"
            "plt.scatter(model.predict(X_test), residuals, alpha=0.5)\n"
            "plt.axhline(y=0, color='r', linestyle='--')\n"
            "plt.xlabel('Predicted')\n"
            "plt.ylabel('Residuals')\n"
            "plt.show()\n"
            "```"
        ),
        mistakes=[
            "Skipping residual analysis — the most common diagnostic omission.",
            "Only looking at aggregate metrics without visual inspection.",
        ],
        interpretation=(
            "Random scatter around 0 → good. Curve → non-linearity. "
            "Funnel → heteroscedasticity. Outliers → influential points."
        ),
        think_about_it=(
            "Your residual plot shows residuals getting wider as "
            "predicted values increase. What is this called and "
            "what does it mean?"
        ),
        code_link=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy import stats\n"
            "\n"
            "residuals = y_test - model.predict(X_test)\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "axes[0].scatter(model.predict(X_test), residuals, alpha=0.5)\n"
            "axes[0].axhline(y=0, color='r')\n"
            "axes[0].set_title('Residuals vs Predicted')\n"
            "stats.probplot(residuals, dist='norm', plot=axes[1])\n"
            "axes[1].set_title('Q-Q Plot')\n"
            "axes[2].hist(residuals, bins=30)\n"
            "plt.tight_layout()\n"
            "```"
        ),
        keywords=["residual", "diagnostic", "pattern", "normal", "heteroscedasticity"],
    ),

    # ── 21 ─────────────────────────────────────────────────────────
    "choosing_right_metric": T(
        title="Choosing the Right Metric",
        module="model_evaluation",
        what=(
            "The right metric depends on your problem: classification "
            "vs regression, balanced vs imbalanced classes, and the "
            "cost of different error types."
        ),
        why=(
            "Using the wrong metric leads to optimising the wrong thing. "
            "99% accuracy on fraud data means zero fraud caught."
        ),
        when=(
            "Before starting modelling. Define your metric based on "
            "business requirements, then optimise for it."
        ),
        example=(
            "Classification:\n"
            "- Balanced classes → Accuracy or F1\n"
            "- Imbalanced → F1, Precision, Recall, or AUC\n"
            "- FP costly → Precision\n"
            "- FN costly → Recall\n"
            "\n"
            "Regression:\n"
            "- Standard → RMSE\n"
            "- Outlier-resistant → MAE\n"
            "- Scale-free → R²"
        ),
        mistakes=[
            "Defaulting to accuracy for all classification problems.",
            "Not discussing metric choice with stakeholders.",
            "Optimising one metric while ignoring others.",
        ],
        interpretation=(
            "Always report at least 2-3 metrics. No single metric "
            "tells the full story."
        ),
        think_about_it=(
            "You're building a loan default predictor. Which metric "
            "matters more: precision or recall? Why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, f1_score, precision_score, recall_score,\n"
            "    roc_auc_score, classification_report\n"
            ")\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "\n"
            "print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\n"
            "print(f'Precision: {precision_score(y_test, y_pred):.4f}')\n"
            "print(f'Recall:    {recall_score(y_test, y_pred):.4f}')\n"
            "print(f'F1:        {f1_score(y_test, y_pred):.4f}')\n"
            "print(f'AUC:       {roc_auc_score(y_test, y_prob):.4f}')\n"
            "```"
        ),
        keywords=["metric", "choice", "accuracy", "f1", "rmse", "problem"],
    ),

    # ── 22 ─────────────────────────────────────────────────────────
    "imbalanced_evaluation": T(
        title="Evaluation with Imbalanced Data",
        module="model_evaluation",
        what=(
            "Standard accuracy is misleading with imbalanced classes. "
            "When one class dominates, a model can achieve high accuracy "
            "by always predicting the majority class."
        ),
        why=(
            "Real-world problems are often imbalanced: fraud (1%), "
            "disease (5%), churn (5%). Standard metrics hide these "
            "rare but important events."
        ),
        when=(
            "When class distribution is skewed. Use F1, precision, "
            "recall, AUC, or confusion matrix instead of accuracy."
        ),
        example=(
            "```python\n"
            "# 95% majority, 5% minority\n"
            "y_pred_all_no = [0] * len(y_test)  # predict 'no' for everything\n"
            "print(f'Accuracy: {accuracy_score(y_test, y_pred_all_no):.2f}')  # 0.95!\n"
            "print(f'F1: {f1_score(y_test, y_pred_all_no):.2f}')              # 0.00\n"
            "```"
        ),
        mistakes=[
            "Reporting accuracy on imbalanced data — always misleading.",
            "Using macro average without understanding per-class performance.",
            "Not looking at the confusion matrix.",
        ],
        interpretation=(
            "For imbalanced data: look at the minority class metrics "
            "(precision, recall, F1 for class 1). The confusion matrix "
            "reveals the full picture."
        ),
        think_about_it=(
            "Your model has 95% accuracy but catches only 2 out of "
            "50 fraud cases. What metric best describes this failure?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import classification_report, confusion_matrix\n"
            "\n"
            "# Always check the report for per-class metrics\n"
            "print(classification_report(y_test, y_pred))\n"
            "\n"
            "# And the confusion matrix\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "print(f'TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')\n"
            "```"
        ),
        keywords=["imbalanced", "minority", "rare", "accuracy trap", "fraud"],
    ),

    # ── 23 ─────────────────────────────────────────────────────────
    "threshold_selection": T(
        title="Classification Threshold Selection",
        module="model_evaluation",
        what=(
            "The default 0.5 threshold may not be optimal. "
            "Adjusting the threshold shifts the precision-recall "
            "trade-off to match your specific cost structure."
        ),
        why=(
            "Different applications have different costs for FP vs FN. "
            "Medical screening: lower threshold to catch more cases "
            "(higher recall). Spam filtering: raise threshold to "
            "avoid blocking legitimate email (higher precision)."
        ),
        when=(
            "When the cost of false positives differs from false "
            "negatives. Use precision-recall curves or ROC curves "
            "to find the optimal threshold."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import precision_recall_curve\n"
            "import numpy as np\n"
            "\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "precision, recall, thresholds = precision_recall_curve(y_test, y_prob)\n"
            "\n"
            "# Find threshold for 90% recall\n"
            "idx = np.argmin(np.abs(recall - 0.90))\n"
            "print(f'Threshold for 90% recall: {thresholds[idx]:.3f}')\n"
            "```"
        ),
        mistakes=[
            "Always using 0.5 without considering the problem.",
            "Tuning threshold on test data instead of validation data.",
        ],
        interpretation=(
            "Lower threshold → more positive predictions → higher "
            "recall, lower precision. Higher threshold → fewer "
            "positive predictions → higher precision, lower recall."
        ),
        think_about_it=(
            "A COVID test has recall=0.99 at threshold=0.3. Should "
            "public health authorities use this threshold?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import precision_recall_curve\n"
            "import numpy as np\n"
            "\n"
            "precision, recall, thresholds = precision_recall_curve(y_test, y_prob)\n"
            "# Choose threshold where precision and recall cross\n"
            "f1_scores = 2 * precision * recall / (precision + recall + 1e-8)\n"
            "best_idx = np.argmax(f1_scores)\n"
            "print(f'Best threshold: {thresholds[best_idx]:.3f}')\n"
            "```"
        ),
        keywords=["threshold", "cutoff", "precision-recall", "trade-off", "tuning"],
    ),

    # ── 24 ─────────────────────────────────────────────────────────
    "evaluation_case_study": T(
        title="Model Evaluation Case Study",
        module="model_evaluation",
        what=(
            "A complete evaluation workflow: train-test split, "
            "cross-validation, multiple metrics, confusion matrix, "
            "ROC curve, and residual analysis."
        ),
        why=(
            "Seeing all evaluation tools used together helps you "
            "build a comprehensive evaluation practice."
        ),
        when=(
            "Use as a reference for evaluating any model."
        ),
        example="Complete Titanic evaluation workflow.",
        mistakes=[
            "Evaluating with only one metric.",
            "Not using cross-validation.",
            "Ignoring the confusion matrix.",
        ],
        interpretation=(
            "Good evaluation uses multiple metrics and visualisations "
            "to build a complete picture of model performance."
        ),
        think_about_it=(
            "After comprehensive evaluation, you find the model has "
            "good accuracy but poor recall on the minority class. "
            "What would you try next?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import cross_val_score, train_test_split\n"
            "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# CV score\n"
            "cv_scores = cross_val_score(model, X_train, y_train, cv=5)\n"
            "print(f'CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')\n"
            "\n"
            "# Test metrics\n"
            "y_pred = model.predict(X_test)\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "print(classification_report(y_test, y_pred))\n"
            "print(f'AUC: {roc_auc_score(y_test, y_prob):.4f}')\n"
            "print(f'Confusion Matrix:\\n{confusion_matrix(y_test, y_pred)}')\n"
            "```"
        ),
        keywords=["case study", "workflow", "comprehensive", "complete", "evaluation"],
    ),

    # ── 25 ─────────────────────────────────────────────────────────
    "common_misconceptions": T(
        title="Common Evaluation Misconceptions",
        module="model_evaluation",
        what=(
            "Several widespread misconceptions lead to incorrect "
            "model evaluation. Understanding them prevents costly "
            "mistakes in practice."
        ),
        why=(
            "Misconceptions about evaluation lead to overconfident "
            "models that fail in production. Correcting them is "
            "essential for every data scientist."
        ),
        when=(
            "Know these misconceptions before you evaluate any model."
        ),
        example=(
            "Misconceptions:\n"
            "1. 'High accuracy = good model' (not with imbalanced data)\n"
            "2. 'Cross-validation = no need for test set' (still need holdout)\n"
            "3. 'R²=0.9 means the model is excellent' (depends on context)\n"
            "4. 'More training data always helps' (diminishing returns)\n"
            "5. 'Lower training error = better model' (overfitting)"
        ),
        mistakes=[
            "Reporting only one metric.",
            "Not using a holdout test set.",
            "Claiming causation from correlation metrics.",
        ],
        interpretation=(
            "Good evaluation practice: use multiple metrics, "
            "cross-validation, confusion matrix, and residual analysis. "
            "Never rely on a single number."
        ),
        think_about_it=(
            "A colleague says their model has 99% accuracy. "
            "What questions would you ask before believing this "
            "is good performance?"
        ),
        code_link=(
            "```python\n"
            "# Comprehensive evaluation checklist\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, f1_score, precision_score, recall_score,\n"
            "    confusion_matrix, classification_report, roc_auc_score\n"
            ")\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "y_prob = model.predict_proba(X_test)[:, 1]\n"
            "\n"
            "print('=== Evaluation Checklist ===')\n"
            "print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\n"
            "print(f'Precision: {precision_score(y_test, y_pred):.4f}')\n"
            "print(f'Recall:    {recall_score(y_test, y_pred):.4f}')\n"
            "print(f'F1:        {f1_score(y_test, y_pred):.4f}')\n"
            "print(f'AUC:       {roc_auc_score(y_test, y_prob):.4f}')\n"
            "print(classification_report(y_test, y_pred))\n"
            "```"
        ),
        keywords=["misconceptions", "pitfalls", "accuracy trap", "best practices", "checklist"],
    ),
}
