"""Generate all remaining learning section files."""
import os

sections = {
    'feature_engineering': [
        ('fe_01', 'What is Feature Engineering?', 'beginner', 'Feature engineering creates new input variables from existing data to improve model performance.'),
        ('fe_02', 'Why Features Matter', 'beginner', 'Features determine what the model can learn. Better features often beat better algorithms.'),
        ('fe_03', 'Feature Creation', 'beginner', 'Create new columns from existing ones: ratios, differences, aggregations.'),
        ('fe_04', 'Log Transformation', 'intermediate', 'np.log1p(x) compresses right-skewed data. Use for income, prices, counts.'),
        ('fe_05', 'Polynomial Features', 'intermediate', 'PolynomialFeatures creates x^2, x^3, and interactions.'),
        ('fe_06', 'Interaction Features', 'intermediate', 'Multiply two features: height * width = area.'),
        ('fe_07', 'Binning', 'intermediate', 'pd.cut() groups continuous values into discrete bins.'),
        ('fe_08', 'Date/Time Features', 'intermediate', 'Extract year, month, day, weekday from datetime columns.'),
        ('fe_09', 'Text Features', 'intermediate', 'Length, word count, uppercase ratio from text columns.'),
        ('fe_10', 'Feature Selection', 'intermediate', 'Remove useless features. Correlation-based, variance-based, or model-based.'),
        ('fe_11', 'Multicollinearity', 'advanced', 'Highly correlated features confuse models. Check VIF.'),
        ('fe_12', 'Curse of Dimensionality', 'advanced', 'As features increase, data becomes sparse. Reduce with PCA.'),
        ('fe_13', 'Feature Engineering Case Study', 'advanced', 'Titanic: FamilySize, IsAlone, Title from Name.'),
    ],
    'classification': [
        ('clf_01', 'What is Classification?', 'beginner', 'Classification predicts discrete class labels: spam/not spam, species A/B/C.'),
        ('clf_02', 'Binary vs Multiclass', 'beginner', 'Binary: two classes. Multiclass: three or more.'),
        ('clf_03', 'Logistic Regression', 'beginner', 'Uses sigmoid to output probabilities. Fast, interpretable.'),
        ('clf_04', 'K-Nearest Neighbors', 'beginner', 'Classifies by majority vote of k closest examples. Requires scaling.'),
        ('clf_05', 'Decision Trees', 'beginner', 'Splits on feature thresholds. Interpretable. Prone to overfitting.'),
        ('clf_06', 'Random Forest', 'intermediate', 'Ensemble of trees. High accuracy, robust, feature importance.'),
        ('clf_07', 'Gradient Boosting', 'intermediate', 'Sequential trees correcting errors. Often highest accuracy.'),
        ('clf_08', 'Support Vector Machines', 'intermediate', 'Finds optimal hyperplane. Effective in high dimensions.'),
        ('clf_09', 'Class Imbalance', 'intermediate', 'When one class dominates, accuracy is misleading. Use F1.'),
        ('clf_10', 'Classification Threshold', 'intermediate', 'Default 0.5. Lower threshold = more positives = higher recall.'),
        ('clf_11', 'Overfitting in Classification', 'intermediate', 'High train accuracy, low test accuracy. Fix: simplify, regularize.'),
        ('clf_12', 'Classification Case Study', 'advanced', 'Titanic: EDA → preprocess → LR, RF, GB → best model ~82%.'),
    ],
    'regression': [
        ('reg_01', 'What is Regression?', 'beginner', 'Regression predicts continuous values: house price, temperature, salary.'),
        ('reg_02', 'Simple Linear Regression', 'beginner', 'Fits a line: y = mx + b.'),
        ('reg_03', 'Multiple Linear Regression', 'beginner', 'y = b0 + b1*x1 + b2*x2 + ...'),
        ('reg_04', 'Residuals', 'beginner', 'Residual = actual - predicted. Random scatter = good model.'),
        ('reg_05', 'MAE, MSE, RMSE', 'beginner', 'MAE: average absolute error. RMSE: sqrt of average squared error.'),
        ('reg_06', 'R-squared', 'beginner', 'Proportion of variance explained. 0.85 = 85% captured.'),
        ('reg_07', 'Ridge Regression', 'intermediate', 'Linear regression + L2 penalty. Shrinks coefficients.'),
        ('reg_08', 'Lasso Regression', 'intermediate', 'Linear regression + L1 penalty. Feature selection.'),
        ('reg_09', 'Polynomial Regression', 'intermediate', 'Fits polynomial curve. Captures non-linear relationships.'),
        ('reg_10', 'Random Forest Regression', 'intermediate', 'Ensemble of trees. High accuracy, robust.'),
        ('reg_11', 'Gradient Boosting Regression', 'intermediate', 'Sequential trees correcting residuals.'),
        ('reg_12', 'Residual Analysis', 'advanced', 'Plot residuals vs predicted. Patterns = missing structure.'),
        ('reg_13', 'Regression Case Study', 'advanced', 'California Housing: linear → ridge → RF → GB. Best: GB R2~0.85.'),
    ],
    'evaluation': [
        ('eval_01', 'Why Model Evaluation Matters', 'beginner', 'Without evaluation, you have no idea if your model works.'),
        ('eval_02', 'Train vs Test', 'beginner', 'Train teaches. Test evaluates. Never evaluate on training data.'),
        ('eval_03', 'Cross-Validation', 'intermediate', 'K-fold CV: split into k parts, train on k-1, test on 1, repeat.'),
        ('eval_04', 'Accuracy', 'beginner', '(TP+TN)/(Total). Misleading with imbalanced classes.'),
        ('eval_05', 'Precision and Recall', 'intermediate', 'Precision: quality of positives. Recall: coverage of positives.'),
        ('eval_06', 'F1 Score', 'intermediate', 'Harmonic mean of precision and recall. Best for imbalanced data.'),
        ('eval_07', 'Confusion Matrix', 'beginner', 'Table of TP, TN, FP, FN. Shows where errors occur.'),
        ('eval_08', 'ROC Curve and AUC', 'intermediate', 'ROC: TPR vs FPR. AUC: 1.0=perfect, 0.5=random.'),
        ('eval_09', 'MAE, MSE, RMSE for Regression', 'beginner', 'MAE: interpretable. RMSE: penalises large errors.'),
        ('eval_10', 'R2 for Regression', 'beginner', 'Proportion of variance explained. Higher is better.'),
        ('eval_11', 'Choosing the Right Metric', 'intermediate', 'Accuracy for balanced. F1 for imbalance. R2 for regression.'),
        ('eval_12', 'Threshold Selection', 'advanced', 'Default 0.5 may not be optimal. Adjust based on costs.'),
        ('eval_13', 'Evaluation Case Study', 'advanced', 'Titanic: compare accuracy, F1, AUC across models.'),
    ],
    'model_selection': [
        ('sel_01', 'What is Model Selection?', 'beginner', 'Choosing algorithm and hyperparameters. No Free Lunch theorem.'),
        ('sel_02', 'Baseline Model', 'beginner', 'Start simple. Logistic Regression sets the minimum bar.'),
        ('sel_03', 'Choosing Classification Algorithms', 'intermediate', 'Start simple (LR), try tree-based (RF, GB), compare.'),
        ('sel_04', 'Choosing Regression Algorithms', 'intermediate', 'Start with Linear Regression. Try Ridge, RF, GB if R2 is low.'),
        ('sel_05', 'Dataset Size and Algorithm', 'intermediate', 'Small: LR, KNN. Medium: RF, GB. Large: GB, neural nets.'),
        ('sel_06', 'Interpretability', 'intermediate', 'Healthcare/finance need explainable models.'),
        ('sel_07', 'Bias and Variance', 'intermediate', 'Bias = underfitting. Variance = overfitting.'),
        ('sel_08', 'Hyperparameters', 'intermediate', 'Settings before training: n_estimators, max_depth.'),
        ('sel_09', 'Grid Search vs Random Search', 'intermediate', 'Grid: exhaustive, slow. Random: faster, often good enough.'),
        ('sel_10', 'Model Comparison', 'intermediate', 'Compare using cross-validation. Same preprocessing, same metrics.'),
        ('sel_11', 'Choosing a Final Model', 'advanced', 'Balance performance, speed, interpretability.'),
        ('sel_12', 'Model Selection Case Study', 'advanced', 'Iris: LR=96%, KNN=97%, RF=95%.'),
    ],
    'clustering': [
        ('clus_01', 'What is Unsupervised Learning?', 'beginner', 'Finding patterns WITHOUT labels.'),
        ('clus_02', 'What is Clustering?', 'beginner', 'Grouping similar data points together.'),
        ('clus_03', 'K-Means Algorithm', 'intermediate', 'Assigns points to nearest centroid, updates, repeats.'),
        ('clus_04', 'Choosing K', 'intermediate', 'Elbow method, Silhouette score, domain knowledge.'),
        ('clus_05', 'Elbow Method', 'intermediate', 'Plot inertia vs k. Look for the elbow.'),
        ('clus_06', 'Silhouette Score', 'intermediate', 'How well each point fits its cluster. -1 to 1.'),
        ('clus_07', 'DBSCAN', 'intermediate', 'Density-based. Groups packed points. Marks noise.'),
        ('clus_08', 'Agglomerative Clustering', 'intermediate', 'Bottom-up: merges closest pairs.'),
        ('clus_09', 'Dendrograms', 'intermediate', 'Tree showing cluster merges. Cut for k clusters.'),
        ('clus_10', 'Scaling Before Clustering', 'intermediate', 'Distance algorithms need scaled features.'),
        ('clus_11', 'PCA for Visualisation', 'intermediate', 'Reduce to 2D for plotting clusters.'),
        ('clus_12', 'Clustering Case Study', 'advanced', 'Customer segmentation: Budget, Premium, Occasional.'),
    ],
    'model_comparison': [
        ('cmp_01', 'Why Compare Models?', 'beginner', 'No single model wins everywhere.'),
        ('cmp_02', 'Fair Model Comparison', 'intermediate', 'Same dataset, preprocessing, split, metrics.'),
        ('cmp_03', 'Cross-Validation Comparison', 'intermediate', 'Use k-fold CV. Compare mean ± std.'),
        ('cmp_04', 'Classification Comparison', 'intermediate', 'Compare accuracy, precision, recall, F1, AUC.'),
        ('cmp_05', 'Regression Comparison', 'intermediate', 'Compare R2, MAE, RMSE.'),
        ('cmp_06', 'Accuracy vs F1', 'intermediate', 'Accuracy misleading with imbalance. F1 balances.'),
        ('cmp_07', 'Comparing Speed and Complexity', 'intermediate', 'Slightly less accurate but 10x faster may be better.'),
        ('cmp_08', 'Choosing the Final Model', 'advanced', 'Balance performance, speed, interpretability.'),
    ],
    'automl': [
        ('auto_01', 'What is AutoML?', 'beginner', 'Automates ML pipeline: preprocessing, selection, tuning.'),
        ('auto_02', 'Why AutoML?', 'beginner', 'Reduces time to baseline. Good for prototyping.'),
        ('auto_03', 'AutoML Workflow', 'intermediate', 'Load → Validate → Detect → Preprocess → Split → Train → Evaluate → Rank.'),
        ('auto_04', 'Automated Preprocessing', 'intermediate', 'Handles missing values, encoding, scaling automatically.'),
        ('auto_05', 'Automated Model Selection', 'intermediate', 'Trains multiple algorithms and ranks them.'),
        ('auto_06', 'Advantages and Limitations', 'intermediate', 'Fast baseline, but cannot replace domain knowledge.'),
        ('auto_07', 'Human vs AutoML', 'intermediate', 'AutoML optimises metrics. Humans consider ethics, cost.'),
        ('auto_08', 'AutoML Case Study', 'advanced', 'Iris: tries LR, RF, GB, KNN, ranks KNN highest.'),
    ],
}

TEMPLATE = '''"""%(title)s curriculum."""

from learning import QuizQuestion, Topic

TOPICS = [
%(topics)s
]
'''

TOPIC_TEMPLATE = '''    Topic(id="%(id)s", title="%(title)s", section="%(section)s", order=%(order)d, difficulty="%(diff)s",
        objectives=["Understand %(title)s", "Apply in practice"],
        concept="""%(concept)s""",
        why_matters="This concept is fundamental to data science.",
        common_mistakes=["Not understanding deeply enough", "Skipping this step"],
        takeaways=["%(title)s is core to data science", "Practice with real datasets"],
        lab_module="%(section)s",
    ),'''

for section_id, topics in sections.items():
    topic_lines = []
    for i, (tid, title, diff, concept) in enumerate(topics, 1):
        topic_lines.append(TOPIC_TEMPLATE % {
            'id': tid, 'title': title, 'section': section_id,
            'order': i, 'diff': diff, 'concept': concept
        })
    
    content = TEMPLATE % {
        'title': section_id.replace('_', ' ').title(),
        'topics': '\n'.join(topic_lines)
    }
    
    filepath = os.path.join('learning', f'{section_id}.py')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Created {filepath} ({len(topics)} topics)')
