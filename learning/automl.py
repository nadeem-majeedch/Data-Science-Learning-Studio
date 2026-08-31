"""AutoML curriculum — 12 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="auto_01", title="What is AutoML?",
        section="automl", order=1, difficulty="beginner",
        objectives=[
            "Define AutoML and its scope",
            "Understand what AutoML automates",
            "Know where AutoML fits in the ML workflow",
        ],
        concept=(
            "AutoML (Automated Machine Learning) automates the end-to-end ML pipeline: "
            "preprocessing, feature engineering, model selection, hyperparameter tuning, "
            "and evaluation. It reduces the time and expertise needed to build competitive "
            "models by systematically trying many approaches."
        ),
        why_matters=(
            "AutoML democratises machine learning — it allows non-experts to build strong "
            "baselines quickly. Even for experts, it provides a fast starting point and "
            "validates that manual efforts add value beyond automated approaches."
        ),
        simple_explanation=(
            "AutoML is like a sous-chef for data science: you provide the ingredients "
            "(data and target), and it tries many recipes (algorithms and settings) "
            "to find the best dish (model)."
        ),
        example=(
            "In the Data Science Learning Studio's AutoML module:\n"
            "1. Upload a dataset (e.g., Titanic)\n"
            "2. Select the target column\n"
            "3. AutoML automatically detects classification vs regression\n"
            "4. Trains 7+ algorithms with cross-validation\n"
            "5. Ranks them and reports the best\n"
            "6. Generates Python code and a downloadable report\n"
            "All without you choosing a single algorithm."
        ),
        python_example=(
            "```python\n"
            "# Conceptual AutoML workflow (simplified)\n"
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n\n"
            "models = {\n"
            "    'Logistic Regression': LogisticRegression(max_iter=1000),\n"
            "    'Random Forest': RandomForestClassifier(n_estimators=100),\n"
            "    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),\n"
            "}\n\n"
            "# AutoML trains all and ranks by CV score\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
            "    print(f'{name}: {scores.mean():.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Assuming AutoML always finds the best possible model",
            "Not understanding what AutoML does under the hood",
            "Using AutoML without understanding ML fundamentals",
        ],
        practice_exercise=(
            "Open the AutoML module in the Lab. Load the Titanic dataset and run AutoML. "
            "1. Which problem type did it detect?\n"
            "2. How many models did it try?\n"
            "3. What was the best model and its F1 score?"
        ),
        quiz=[
            QuizQuestion(
                question="What does AutoML automate?",
                options=[
                    "Only data cleaning",
                    "Only model training",
                    "The entire ML pipeline: preprocessing, model selection, tuning, and evaluation",
                    "Feature engineering only",
                ],
                correct_index=2,
                explanation=(
                    "AutoML automates the full ML pipeline — from data validation through "
                    "preprocessing, model selection, hyperparameter tuning, and evaluation. "
                    "It mirrors the steps a data scientist performs manually."
                ),
            ),
        ],
        takeaways=[
            "AutoML automates the entire ML pipeline",
            "Good for fast baselines and exploration",
            "Does NOT replace understanding of ML fundamentals",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_02", title="Why AutoML?",
        section="automl", order=2, difficulty="beginner",
        objectives=[
            "Understand when AutoML helps most",
            "Know its limitations",
            "Appreciate its role alongside human expertise",
        ],
        concept=(
            "AutoML saves time, provides strong baselines, and explores combinations a human "
            "might miss. It's the fastest path from raw data to a working model. But it can't "
            "replace domain knowledge, creative feature engineering, or business understanding."
        ),
        why_matters=(
            "AutoML is the fastest way to get a baseline. It also provides a benchmark: "
            "if your hand-crafted model can't beat AutoML, your manual effort may not be "
            "adding value. This is both humbling and useful."
        ),
        example=(
            "Scenario: A marketing team needs a churn prediction model by Friday.\n"
            "• Without AutoML: test algorithms one by one, tune each → 3-5 days\n"
            "• With AutoML: run AutoML → 30 minutes → strong baseline → "
            "spend remaining time on feature engineering\n"
            "AutoML buys time for higher-value work."
        ),
        python_example=(
            "```python\n"
            "# AutoML's value: quick comparison\n"
            "import time\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "# Manual approach: test one model at a time\n"
            "start = time.time()\n"
            "lr_scores = cross_val_score(LogisticRegression(), X, y, cv=5)\n"
            "rf_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)\n"
            "gb_scores = cross_val_score(GradientBoostingClassifier(), X, y, cv=5)\n"
            "manual_time = time.time() - start\n\n"
            "# AutoML does this (and more) in one call\n"
            "print(f'Manual comparison took {manual_time:.1f}s')\n"
            "```"
        ),
        common_mistakes=[
            "Using AutoML as the only approach without domain validation",
            "Not questioning AutoML's results",
            "Expecting AutoML to solve all problems end-to-end",
        ],
        practice_exercise=(
            "Run AutoML on a dataset. Then manually train the top model with default "
            "parameters. "
            "1. Do the scores match?\n"
            "2. Why might they differ slightly?\n"
            "3. What would you do to improve beyond the AutoML baseline?"
        ),
        quiz=[
            QuizQuestion(
                question="What is AutoML's primary value for a data scientist?",
                options=[
                    "It replaces the need for human data scientists",
                    "It provides a fast baseline and validates that manual effort adds value",
                    "It always finds the optimal model",
                    "It automates feature engineering completely",
                ],
                correct_index=1,
                explanation=(
                    "AutoML's main value is providing a strong baseline quickly. If your "
                    "hand-crafted model can't beat AutoML, you know your manual effort needs "
                    "to focus on feature engineering or domain knowledge, not algorithm tuning."
                ),
            ),
        ],
        takeaways=[
            "AutoML: fast baseline + exploration tool",
            "Not a replacement for domain expertise",
            "Use alongside, not instead of, human-guided development",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_03", title="AutoML Workflow",
        section="automl", order=3, difficulty="intermediate",
        objectives=[
            "Map the AutoML pipeline to manual ML steps",
            "Understand what each automated step does",
            "Recognise where automation has limits",
        ],
        concept=(
            "AutoML workflow mirrors manual ML:\n"
            "1. Load and validate data\n"
            "2. Detect problem type (classification vs regression)\n"
            "3. Preprocess (imputation, encoding, scaling)\n"
            "4. Split data (train/test or CV)\n"
            "5. Train multiple algorithms\n"
            "6. Evaluate with cross-validation\n"
            "7. Rank models by metric\n"
            "8. Report best model + generate code\n\n"
            "Each step corresponds to a manual decision a data scientist makes."
        ),
        why_matters=(
            "Understanding the workflow helps you verify AutoML's decisions and "
            "identify where manual intervention might improve results."
        ),
        example=(
            "Data Science Lab AutoML workflow for Titanic:\n"
            "1. Load: 891 rows, 12 columns detected\n"
            "2. Problem type: classification (binary — survived/died)\n"
            "3. Preprocessing: median imputation for Age, mode for Embarked, one-hot for categoricals\n"
            "4. Split: 80/20 stratified\n"
            "5. Models: LR, KNN, DT, RF, NB, SVM, GB trained\n"
            "6. Evaluation: 5-fold stratified CV, F1 scoring\n"
            "7. Ranking: GB (F1=0.82) > RF (0.80) > LR (0.79)\n"
            "8. Report: code, metrics, comparison chart"
        ),
        common_mistakes=[
            "Not validating data before running AutoML",
            "Not checking the detected problem type",
            "Ignoring AutoML logs that show what was done",
        ],
        practice_exercise=(
            "Run AutoML and examine the output carefully. "
            "1. What preprocessing did it apply automatically?\n"
            "2. Which problem type did it detect?\n"
            "3. What metric did it use for ranking? Is that the right metric?"
        ),
        quiz=[
            QuizQuestion(
                question="In AutoML, what happens AFTER model training?",
                options=[
                    "The model is deployed to production",
                    "Models are evaluated with cross-validation and ranked by a primary metric",
                    "The user must manually evaluate each model",
                    "Feature engineering is applied",
                ],
                correct_index=1,
                explanation=(
                    "After training, AutoML evaluates each model using cross-validation, "
                    "computes the chosen metric, and ranks models from best to worst. "
                    "It then presents the comparison table and the best model."
                ),
            ),
        ],
        takeaways=[
            "AutoML mirrors the manual ML pipeline step by step",
            "Every automated step corresponds to a human decision",
            "Always verify AutoML's problem type detection and metric choice",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_04", title="Automated Preprocessing",
        section="automl", order=4, difficulty="intermediate",
        objectives=[
            "Know what AutoML automates in preprocessing",
            "Understand default strategies",
            "Identify when defaults may fail",
        ],
        concept=(
            "AutoML typically applies standard preprocessing automatically:\n"
            "• Missing values: median for numerical, mode for categorical\n"
            "• Scaling: StandardScaler or MinMaxScaler for numerical features\n"
            "• Encoding: One-hot encoding for low-cardinality categoricals\n"
            "• Feature selection: removes constant and near-constant features\n"
            "• Deduplication: removes duplicate rows"
        ),
        why_matters=(
            "Knowing what AutoML does by default helps you identify what it might miss. "
            "Domain-specific preprocessing (e.g., text vectorisation, time-series lag features) "
            "requires human input."
        ),
        example=(
            "Titanic dataset preprocessing:\n"
            "• Age: 177 missing → AutoML imputes with median (28.0)\n"
            "• Cabin: 687 missing → AutoML may drop (too many missing)\n"
            "• Embarked: 2 missing → AutoML imputes with mode ('S')\n"
            "• Sex: categorical → AutoML one-hot encodes\n"
            "• Name/PassengerId: non-informative → AutoML may drop"
        ),
        python_example=(
            "```python\n"
            "# What AutoML does internally (simplified)\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n\n"
            "# AutoML creates this pipeline automatically\n"
            "numerical_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler())\n"
            "])\n\n"
            "categorical_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n"
            "])\n"
            "```"
        ),
        common_mistakes=[
            "Assuming AutoML preprocessing is optimal for your specific data",
            "Not checking what preprocessing was applied",
            "Expecting AutoML to handle domain-specific preprocessing (text, time-series)",
        ],
        practice_exercise=(
            "Run AutoML on a dataset with missing values. "
            "1. Check what imputation strategy it used.\n"
            "2. Did it scale numerical features?\n"
            "3. How did it handle categorical variables?"
        ),
        quiz=[
            QuizQuestion(
                question="What preprocessing does AutoML typically apply to missing numerical values?",
                options=[
                    "Drops all rows with missing values",
                    "Fills with zero",
                    "Imputes with the median",
                    "Leaves them as NaN",
                ],
                correct_index=2,
                explanation=(
                    "AutoML typically uses median imputation for numerical features. "
                    "Median is robust to outliers and preserves the distribution better "
                    "than mean imputation."
                ),
            ),
        ],
        takeaways=[
            "AutoML applies standard preprocessing defaults (median/mode imputation, scaling, encoding)",
            "Always check what preprocessing was applied",
            "Domain-specific preprocessing still needs human input",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_05", title="Automated Model Selection",
        section="automl", order=5, difficulty="intermediate",
        objectives=[
            "Understand how AutoML selects and ranks models",
            "Know which algorithms are typically included",
            "Interpret the ranking criteria",
        ],
        concept=(
            "AutoML trains multiple algorithms with default or lightly tuned hyperparameters, "
            "evaluates each with cross-validation, and ranks them by a primary metric "
            "(F1 for classification, R² or RMSE for regression)."
        ),
        why_matters=(
            "Understanding which models AutoML tries helps you know what's being compared. "
            "If AutoML doesn't include an algorithm you need, you can add it manually."
        ),
        example=(
            "Data Science Lab AutoML tests 7 algorithms on Titanic:\n"
            "1. Gradient Boosting: F1=0.823\n"
            "2. Random Forest: F1=0.801\n"
            "3. Logistic Regression: F1=0.792\n"
            "4. KNN: F1=0.775\n"
            "5. SVM: F1=0.770\n"
            "6. Decision Tree: F1=0.758\n"
            "7. Naive Bayes: F1=0.725\n"
            "The ranking helps you see the full picture, not just the winner."
        ),
        python_example=(
            "```python\n"
            "# Conceptual: what AutoML does behind the scenes\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import (\n"
            "    RandomForestClassifier, GradientBoostingClassifier\n"
            ")\n"
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.naive_bayes import GaussianNB\n"
            "from sklearn.tree import DecisionTreeClassifier\n\n"
            "algorithms = {\n"
            "    'LR': LogisticRegression(max_iter=1000),\n"
            "    'RF': RandomForestClassifier(n_estimators=100),\n"
            "    'GB': GradientBoostingClassifier(n_estimators=100),\n"
            "    'KNN': KNeighborsClassifier(),\n"
            "    'SVM': SVC(),\n"
            "    'NB': GaussianNB(),\n"
            "    'DT': DecisionTreeClassifier(),\n"
            "}\n"
            "# AutoML trains each, evaluates with CV, and ranks\n"
            "```"
        ),
        common_mistakes=[
            "Only looking at the top model — the full ranking tells a richer story",
            "Not considering why a model ranked high",
            "Ignoring that AutoML may not include your preferred algorithm",
        ],
        practice_exercise=(
            "Run AutoML and examine the full ranking table. "
            "1. How many models were tested?\n"
            "2. What is the score gap between #1 and #2?\n"
            "3. If the gap is small, is the 'winner' really significantly better?"
        ),
        quiz=[
            QuizQuestion(
                question="AutoML ranks models by cross-validation score. Why CV instead of a single test score?",
                options=[
                    "CV is faster",
                    "CV gives a more reliable estimate by averaging over multiple splits",
                    "CV prevents overfitting to the test set",
                    "CV is required by sklearn",
                ],
                correct_index=1,
                explanation=(
                    "Cross-validation averages performance over multiple train/test splits, "
                    "giving a more reliable estimate than a single split. Different splits "
                    "can produce different scores; CV captures this variability."
                ),
            ),
        ],
        takeaways=[
            "AutoML tries many models and ranks them by CV performance",
            "The ranking helps you see the full comparison, not just the winner",
            "The top model may not be significantly better than #2 or #3",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_06", title="Automated Evaluation",
        section="automl", order=6, difficulty="intermediate",
        objectives=[
            "Understand how AutoML evaluates models",
            "Interpret AutoML evaluation results",
            "Know when to verify independently",
        ],
        concept=(
            "AutoML evaluates models using cross-validation with metrics appropriate for "
            "the problem type. It produces a comparison table with per-model metrics, "
            "training time, and configuration details."
        ),
        why_matters=(
            "Understanding AutoML evaluation helps you verify its results and identify "
            "when the chosen metric may not match your business requirements."
        ),
        example=(
            "AutoML evaluation output for classification:\n"
            "• Primary metric: F1 (macro-averaged for multiclass)\n"
            "• Additional metrics: accuracy, precision, recall, AUC\n"
            "• Cross-validation: 5-fold stratified\n"
            "• Training time per model recorded\n"
            "• Full comparison table with all models"
        ),
        python_example=(
            "```python\n"
            "# Verifying AutoML's evaluation independently\n"
            "from sklearn.metrics import classification_report\n\n"
            "# Get AutoML's best model\n"
            "best_model = automl_results.best_model\n\n"
            "# Evaluate on held-out test set independently\n"
            "y_pred = best_model.predict(X_test)\n"
            "print(classification_report(y_test, y_pred))\n"
            "\n"
            "# Compare CV score with test score\n"
            "# If they differ significantly, investigate\n"
            "```"
        ),
        common_mistakes=[
            "Not verifying the evaluation metric matches the business problem",
            "Ignoring cross-validation variance",
            "Trusting AutoML evaluation without independent verification on a held-out set",
        ],
        practice_exercise=(
            "After AutoML selects the best model, evaluate it independently on a test set. "
            "1. Does the test score match the CV score?\n"
            "2. If they differ, what might cause this?\n"
            "3. Is the chosen metric right for this problem?"
        ),
        quiz=[
            QuizQuestion(
                question="AutoML reports the best model has F1=0.85. You evaluate it on a test set and get F1=0.78. What should you investigate?",
                options=[
                    "Nothing — small differences are normal",
                    "Whether there's data leakage, overfitting, or distribution shift between train and test",
                    "Try a different AutoML tool",
                    "The test set is wrong",
                ],
                correct_index=1,
                explanation=(
                    "A significant gap between CV and test scores suggests potential issues: "
                    "overfitting (model memorised CV folds), data leakage (preprocessing used "
                    "test data), or distribution shift (train and test data differ). Investigate."
                ),
            ),
        ],
        takeaways=[
            "AutoML evaluation mirrors manual evaluation methodology",
            "Verify the chosen metric matches your business requirements",
            "Always independently verify on a held-out test set",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_07", title="Hyperparameter Search",
        section="automl", order=7, difficulty="intermediate",
        objectives=[
            "Understand how AutoML tunes hyperparameters",
            "Know the search strategies used",
            "Understand the time budget concept",
        ],
        concept=(
            "AutoML tunes hyperparameters within a time budget using random search, "
            "grid search, or Bayesian optimisation. More time = more tuning = potentially "
            "better results, but with diminishing returns."
        ),
        why_matters=(
            "Understanding the time budget helps you allocate resources. A 5-minute budget "
            "may produce a decent model; a 60-minute budget explores more combinations "
            "but may not improve significantly."
        ),
        example=(
            "AutoML time budgets:\n"
            "• Quick (2 min): default parameters only → decent baseline\n"
            "• Standard (10 min): light tuning → good model\n"
            "• Thorough (60 min): extensive search → best possible within algorithms tried\n"
            "Diminishing returns: going from 10 min to 60 min might improve F1 by only 1-2%."
        ),
        python_example=(
            "```python\n"
            "# Conceptual: what AutoML does for hyperparameter tuning\n"
            "from sklearn.model_selection import RandomizedSearchCV\n"
            "from scipy.stats import randint, uniform\n\n"
            "# For each algorithm, AutoML searches over parameter space\n"
            "param_distributions = {\n"
            "    'n_estimators': randint(50, 300),\n"
            "    'max_depth': randint(3, 15),\n"
            "    'learning_rate': uniform(0.01, 0.3)\n"
            "}\n\n"
            "# Random search within time budget\n"
            "search = RandomizedSearchCV(\n"
            "    GradientBoostingClassifier(),\n"
            "    param_distributions, n_iter=50, cv=5\n"
            ")\n"
            "```"
        ),
        common_mistakes=[
            "Using the default time budget without considering problem complexity",
            "Expecting exhaustive search (it's not — it's guided random search)",
            "Not understanding that more time ≠ always significantly better",
        ],
        practice_exercise=(
            "Run AutoML twice: once with a short time budget and once with a longer one. "
            "1. Did the longer budget find a better model?\n"
            "2. How much improvement did the extra time provide?\n"
            "3. Was the extra time worth it?"
        ),
        quiz=[
            QuizQuestion(
                question="Why does AutoML use random search instead of exhaustive grid search?",
                options=[
                    "Grid search is too slow for most parameter spaces",
                    "Random search explores more of the space with fewer iterations",
                    "sklearn doesn't support grid search",
                    "Random search always finds better results",
                ],
                correct_index=1,
                explanation=(
                    "Random search explores the parameter space more broadly with the same "
                    "number of iterations. Grid search wastes combinations on unimportant "
                    "parameters. Research shows random search is often more efficient."
                ),
            ),
        ],
        takeaways=[
            "AutoML tunes within a time budget",
            "Uses random search or Bayesian optimisation",
            "More time helps, but with diminishing returns",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_08", title="Advantages of AutoML",
        section="automl", order=8, difficulty="beginner",
        objectives=[
            "List AutoML advantages",
            "Know when AutoML is most valuable",
            "Understand its role in the ML workflow",
        ],
        concept=(
            "Key advantages:\n"
            "1. Fast baseline — go from data to model in minutes\n"
            "2. Explores many combinations systematically\n"
            "3. Reduces human bias in algorithm choice\n"
            "4. Accessible to non-experts\n"
            "5. Excellent starting point for experts\n"
            "6. Documents the full pipeline automatically"
        ),
        why_matters=(
            "AutoML removes the 'where do I start?' paralysis. It provides a strong "
            "starting point that you can then improve with domain knowledge and creative "
            "feature engineering."
        ),
        example=(
            "A biology researcher has gene expression data but limited ML experience. "
            "AutoML:\n"
            "1. Detects it's a classification problem (3 classes)\n"
            "2. Tries 7 algorithms in 5 minutes\n"
            "3. Reports Random Forest with 87% accuracy\n"
            "4. Generates code the researcher can understand and modify\n"
            "Without AutoML, the researcher would need weeks to learn and test algorithms."
        ),
        practice_exercise=(
            "List three scenarios where AutoML is most valuable. "
            "1. A data scientist starting a new project (exploration phase)\n"
            "2. A domain expert with limited ML experience\n"
            "3. A team needing a quick prototype for stakeholder approval\n"
            "For each, explain why AutoML helps."
        ),
        common_mistakes=[
            "Over-relying on AutoML for final production models",
            "Not using AutoML at all because 'humans are better'",
        ],
        quiz=[
            QuizQuestion(
                question="Which scenario is AutoML MOST valuable for?",
                options=[
                    "A production model that must be deployed tomorrow",
                    "Establishing a baseline before manual feature engineering",
                    "A Kaggle competition where 0.01% matters",
                    "Understanding why a specific model works",
                ],
                correct_index=1,
                explanation=(
                    "AutoML excels at establishing baselines quickly. It tells you the "
                    "starting performance level, so you know how much your manual efforts "
                    "need to improve. For production, you'd still validate and potentially "
                    "customise. For competitions, you'd need manual tuning beyond AutoML."
                ),
            ),
        ],
        takeaways=[
            "AutoML: fast, unbiased, accessible",
            "Excellent baseline and exploration tool",
            "Use alongside, not instead of, human expertise",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_09", title="Limitations of AutoML",
        section="automl", order=9, difficulty="intermediate",
        objectives=[
            "Understand AutoML weaknesses",
            "Know what AutoML cannot do",
            "Identify when human intervention is needed",
        ],
        concept=(
            "Key limitations:\n"
            "1. Can't replace domain knowledge (doesn't understand business context)\n"
            "2. May miss creative feature engineering\n"
            "3. Limited to pre-programmed algorithms\n"
            "4. May overfit to the evaluation metric\n"
            "5. Can't handle very custom requirements\n"
            "6. May not choose the best metric for your problem"
        ),
        why_matters=(
            "Knowing AutoML's limitations tells you where human expertise adds the most "
            "value. The gap between AutoML and the best possible model is filled by "
            "domain knowledge and creative engineering."
        ),
        example=(
            "AutoML on housing data achieves R²=0.80. After manual work:\n"
            "• Feature engineering: created price_per_sqft, distance_to_center → R²=0.84\n"
            "• Domain knowledge: removed properties with data errors → R²=0.86\n"
            "• Ensemble: combined top 3 models → R²=0.87\n"
            "The 7% improvement came entirely from human expertise, not algorithm tuning."
        ),
        practice_exercise=(
            "Compare AutoML results with a manually engineered model. "
            "1. What is the performance gap?\n"
            "2. What manual steps closed the gap?\n"
            "3. Where did domain knowledge help most?"
        ),
        common_mistakes=[
            "Assuming AutoML results are production-ready",
            "Not validating AutoML findings",
            "Expecting AutoML to handle unique business requirements",
        ],
        quiz=[
            QuizQuestion(
                question="What is AutoML LEAST able to improve?",
                options=[
                    "Algorithm selection",
                    "Hyperparameter tuning",
                    "Domain-specific feature engineering and business context",
                    "Preprocessing defaults",
                ],
                correct_index=2,
                explanation=(
                    "AutoML can tune algorithms and hyperparameters but can't create "
                    "domain-specific features or understand business context. "
                    "A doctor's knowledge of symptoms matters more than any algorithm choice."
                ),
            ),
        ],
        takeaways=[
            "AutoML can't replace domain expertise",
            "Always validate and interpret results",
            "Custom requirements and creative features need human input",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_10", title="AutoML vs Human Data Scientist",
        section="automl", order=10, difficulty="intermediate",
        objectives=[
            "Compare human-guided and AutoML approaches",
            "Know when each excels",
            "Combine both effectively",
        ],
        concept=(
            "Human data scientist: domain knowledge, creative features, business understanding, "
            "custom evaluation, stakeholder communication.\n"
            "AutoML: speed, systematic exploration, removes human bias, reproducible.\n"
            "Best approach: AutoML for baseline → human expertise for improvement."
        ),
        why_matters=(
            "The optimal workflow combines both: AutoML provides the starting point and "
            "benchmark; human expertise provides the creative edge and business alignment."
        ),
        example=(
            "Email spam detection:\n"
            "• AutoML baseline: GB with F1=0.92 (using raw features)\n"
            "• Human adds: email length, sender reputation, link count → F1=0.96\n"
            "• Human removes: false positive analysis reveals legitimate emails flagged → F1=0.95 "
            "(but better precision)\n"
            "AutoML gave the floor; human expertise raised the ceiling."
        ),
        practice_exercise=(
            "Run AutoML on a dataset. Then manually add 2-3 domain-specific features. "
            "1. Does your manual model beat AutoML?\n"
            "2. What did AutoML do better than you?\n"
            "3. What did you do better than AutoML?"
        ),
        common_mistakes=[
            "Treating AutoML and human approaches as mutually exclusive",
            "Not comparing human models against AutoML baseline",
        ],
        quiz=[
            QuizQuestion(
                question="What is the best practice for using AutoML alongside manual ML?",
                options=[
                    "Use only AutoML — it's faster and often better",
                    "Use only manual ML — AutoML is never good enough",
                    "Use AutoML for baseline and exploration, then improve with domain expertise",
                    "Use both independently and pick the winner",
                ],
                correct_index=2,
                explanation=(
                    "The best practice is synergistic: AutoML provides a fast baseline and "
                    "explores the algorithm space; human expertise adds domain features, "
                    "business context, and creative solutions beyond what AutoML can discover."
                ),
            ),
        ],
        takeaways=[
            "AutoML for baseline and exploration",
            "Human expertise for features and business context",
            "Compare both to ensure the human model adds value",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_11", title="Data Leakage in AutoML",
        section="automl", order=11, difficulty="intermediate",
        objectives=[
            "Identify leakage risks specific to AutoML",
            "Ensure safe AutoML usage",
            "Verify AutoML pipelines for leakage",
        ],
        concept=(
            "AutoML can introduce leakage if:\n"
            "1. Preprocessing is fit on the full dataset (before splitting)\n"
            "2. Target encoding uses test data information\n"
            "3. Time-series data isn't split temporally\n"
            "4. Feature selection uses the target on the full dataset\n\n"
            "Always verify the pipeline is leakage-free."
        ),
        why_matters=(
            "Data leakage in AutoML produces overly optimistic scores. The model appears "
            "excellent in evaluation but fails in production. This is especially dangerous "
            "because AutoML makes it easy to run without thinking about leakage."
        ),
        example=(
            "Leakage scenario: Titanic dataset.\n"
            "• Safe: preprocessing fit on training folds only (within CV)\n"
            "• Leakage: preprocessing fit on all data before CV → test scores are inflated\n"
            "• AutoML that fits preprocessing outside CV folds will overestimate performance"
        ),
        practice_exercise=(
            "After running AutoML, examine the pipeline carefully. "
            "1. Is preprocessing applied within each CV fold?\n"
            "2. Does the AutoML tool document its pipeline?\n"
            "3. Try to reproduce the pipeline manually — does it match?"
        ),
        common_mistakes=[
            "Trusting AutoML to handle leakage automatically",
            "Not checking if preprocessing is inside CV folds",
            "Using AutoML on time-series without temporal split",
        ],
        quiz=[
            QuizQuestion(
                question="How can you verify AutoML doesn't have data leakage?",
                options=[
                    "Trust the AutoML tool to handle it",
                    "Check that preprocessing is fitted only within training folds, not on the full dataset",
                    "Run AutoML twice and compare scores",
                    "Use a different AutoML tool",
                ],
                correct_index=1,
                explanation=(
                    "The gold standard is verifying that preprocessing is applied within "
                    "each CV fold (fit on training fold, transform on validation fold). "
                    "If preprocessing is fit on the full dataset before CV, information "
                    "from the validation fold leaks into training."
                ),
            ),
        ],
        takeaways=[
            "Verify AutoML handles leakage correctly",
            "Check that preprocessing is within CV folds",
            "Time-series needs temporal splits",
        ],
        lab_module="automl",
    ),
    Topic(
        id="auto_12", title="AutoML Case Study",
        section="automl", order=12, difficulty="advanced",
        objectives=[
            "Apply the complete AutoML workflow end-to-end",
            "Interpret and act on AutoML results",
            "Build on AutoML findings with domain expertise",
        ],
        concept=(
            "Complete AutoML case study:\n"
            "1. Load data → AutoML detects problem type\n"
            "2. Run AutoML → get baseline models and ranking\n"
            "3. Analyse results → understand what worked and why\n"
            "4. Add domain features → improve beyond AutoML\n"
            "5. Compare: AutoML baseline vs human-enhanced model\n"
            "6. Document findings"
        ),
        why_matters=(
            "This case study demonstrates the ideal workflow: AutoML provides the foundation, "
            "human expertise builds upon it. The lesson is that AutoML is a starting point, "
            "not an endpoint."
        ),
        example=(
            "Titanic case study:\n"
            "1. AutoML: GB best at F1=0.823 (5 algorithms tested)\n"
            "2. Analysis: Age and Cabin had most missing values; Fare was important\n"
            "3. Domain features: Title extracted from Name, FamilySize from SibSp+Parch\n"
            "4. Human model: GB with domain features → F1=0.851 (+2.8%)\n"
            "5. Lesson: AutoML gave the floor; domain knowledge raised the ceiling\n"
            "6. Both models saved and compared"
        ),
        practice_exercise=(
            "Complete an AutoML case study:\n"
            "1. Run AutoML on a dataset and note the best model and score\n"
            "2. Add 2-3 manual features based on domain understanding\n"
            "3. Re-run or train manually with the new features\n"
            "4. Compare: how much did human features improve over AutoML?\n"
            "5. Write a summary of your findings"
        ),
        common_mistakes=[
            "Stopping at AutoML results",
            "Not adding domain knowledge after AutoML",
            "Not comparing AutoML against manual models",
        ],
        quiz=[
            QuizQuestion(
                question="AutoML gives F1=0.80. After adding domain features, your manual model gets F1=0.84. What does this tell you?",
                options=[
                    "AutoML is broken",
                    "Your manual model is much better",
                    "Domain-specific feature engineering adds value beyond algorithm selection",
                    "The features caused data leakage",
                ],
                correct_index=2,
                explanation=(
                    "The improvement came from domain-specific features, not better algorithm "
                    "choice. AutoML optimised algorithms; you optimised features. Both are "
                    "important, and the combination produces the best results."
                ),
            ),
        ],
        takeaways=[
            "AutoML provides a strong baseline quickly",
            "Human expertise can improve beyond AutoML through feature engineering",
            "Always compare AutoML against manual models and document findings",
            "The best workflow combines both: AutoML for exploration, humans for creativity",
        ],
        lab_module="automl",
    ),
]
