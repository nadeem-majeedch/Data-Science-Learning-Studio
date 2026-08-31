"""Practical exercises for each curriculum section."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Exercise:
    """A single practice exercise."""
    id: str
    section: str
    title: str
    difficulty: str  # beginner | intermediate | advanced
    description: str
    steps: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    lab_module: str = ""


EXERCISES: dict[str, list[Exercise]] = {
    "data_loading": [
        Exercise(
            id="ex_dl_b1", section="data_loading",
            title="Load and Inspect Titanic",
            difficulty="beginner",
            description="Load the Titanic dataset and perform initial inspection.",
            steps=[
                "Load 'datasets/titanic.csv' using pd.read_csv()",
                "Print the shape (rows × columns)",
                "Print column names and data types with df.info()",
                "Show the first 5 rows with df.head()",
                "Show the last 3 rows with df.tail()",
            ],
            hints=["Use df.shape, df.info(), df.head(), df.tail()"],
            expected_outcome="You should see 50 rows, 8 columns, with mixed dtypes (int64, float64, object).",
            lab_module="dataset_explorer",
        ),
        Exercise(
            id="ex_dl_b2", section="data_loading",
            title="Identify Data Types",
            difficulty="beginner",
            description="Separate Titanic columns into numerical and categorical.",
            steps=[
                "Use df.dtypes to check all column types",
                "Use df.select_dtypes('number').columns for numerical",
                "Use df.select_dtypes(include=['object']).columns for categorical",
                "Count how many of each type",
            ],
            hints=["Use select_dtypes with 'number' and 'object'"],
            expected_outcome="You should find 3 numerical and 5 categorical columns (or similar, depending on encoding).",
            lab_module="dataset_explorer",
        ),
        Exercise(
            id="ex_dl_i1", section="data_loading",
            title="Data Quality Audit",
            difficulty="intermediate",
            description="Perform a complete data quality check on the Breast Cancer dataset.",
            steps=[
                "Load 'datasets/breast_cancer.csv'",
                "Check shape and dtypes",
                "Count missing values per column",
                "Count duplicates",
                "Compute basic statistics with describe()",
                "Identify any columns with zero or near-zero variance",
                "Write a summary of findings",
            ],
            hints=["Use df.isnull().sum(), df.duplicated().sum(), df.describe()"],
            expected_outcome="A written report covering shape, types, missing values, duplicates, and any anomalies found.",
            lab_module="dataset_explorer",
        ),
        Exercise(
            id="ex_dl_a1", section="data_loading",
            title="Load Multiple Formats",
            difficulty="advanced",
            description="Load data from different sources and combine them.",
            steps=[
                "Load 'datasets/iris.csv'",
                "Create a second dataset by selecting only 2 columns",
                "Save it as a new CSV",
                "Load both datasets and merge them on a common column",
                "Verify the merged dataset has the correct shape",
            ],
            hints=["Use pd.merge() with on parameter"],
            expected_outcome="A merged dataset combining information from both sources.",
            lab_module="dataset_explorer",
        ),
    ],
    "eda": [
        Exercise(
            id="ex_eda_b1", section="eda",
            title="Univariate Analysis of Iris",
            difficulty="beginner",
            description="Perform univariate analysis on the Iris dataset.",
            steps=[
                "Load the Iris dataset",
                "Compute df.describe() for all numerical columns",
                "Create histograms for each feature",
                "Compute skewness for each feature",
                "Identify which feature is most normally distributed",
            ],
            hints=["Use df.hist(), df.skew()"],
            expected_outcome="Four histograms showing the distribution of each iris feature. Identification of the most symmetric distribution.",
            lab_module="eda",
        ),
        Exercise(
            id="ex_eda_b2", section="eda",
            title="Correlation Discovery",
            difficulty="beginner",
            description="Find correlations in the California Housing dataset.",
            steps=[
                "Load 'datasets/california_housing.csv'",
                "Compute the correlation matrix with df.corr()",
                "Find the feature most correlated with median_house_value",
                "Find the pair of features most correlated with each other",
            ],
            hints=["Use df.corr()['median_house_value'].sort_values(ascending=False)"],
            expected_outcome="Identification of the most predictive feature and any multicollinear pairs.",
            lab_module="eda",
        ),
        Exercise(
            id="ex_eda_i1", section="eda",
            title="Titanic EDA Deep Dive",
            difficulty="intermediate",
            description="Perform comprehensive EDA on Titanic.",
            steps=[
                "Load Titanic and check for missing values",
                "Visualise survival rate by gender (bar chart)",
                "Create a box plot of Age grouped by Survival",
                "Compute correlation between numerical features",
                "Create a scatter plot of Age vs Fare, coloured by Survival",
                "Write 3 findings from your analysis",
            ],
            hints=["Use pd.crosstab(), groupby(), plt.scatter()"],
            expected_outcome="Multiple visualisations and a written summary of 3 key findings about survival patterns.",
            lab_module="eda",
        ),
        Exercise(
            id="ex_eda_a1", section="eda",
            title="Multi-Dataset EDA Report",
            difficulty="advanced",
            description="Create an EDA comparison report for two datasets.",
            steps=[
                "Load Breast Cancer and Wine Quality datasets",
                "For each: shape, types, missing values, statistics",
                "Compare feature distributions",
                "Identify the most important features in each",
                "Write a one-page comparison report",
            ],
            hints=["Use the same analysis steps for both, then compare"],
            expected_outcome="A structured comparison report with findings for both datasets.",
            lab_module="eda",
        ),
    ],
    "preprocessing": [
        Exercise(
            id="ex_pp_b1", section="preprocessing",
            title="Handle Missing Values",
            difficulty="beginner",
            description="Apply different missing value strategies on Titanic.",
            steps=[
                "Load Titanic and count missing values in Age",
                "Strategy 1: Drop rows with missing Age",
                "Strategy 2: Fill with mean",
                "Strategy 3: Fill with median",
                "Compare the distribution before and after each strategy",
            ],
            hints=["Use df.dropna(), df.fillna(), compare with df['Age'].hist()"],
            expected_outcome="Three different versions of the Age column, with understanding of how each strategy affects the distribution.",
            lab_module="preprocessing",
        ),
        Exercise(
            id="ex_pp_i1", section="preprocessing",
            title="Complete Preprocessing Pipeline",
            difficulty="intermediate",
            description="Build a complete preprocessing pipeline for classification.",
            steps=[
                "Load Titanic, select features and target",
                "Split data 80/20 with stratify",
                "Build ColumnTransformer: StandardScaler for numeric, OneHotEncoder for categorical",
                "Wrap in a Pipeline with LogisticRegression",
                "Train and evaluate with cross_val_score",
            ],
            hints=["Use sklearn Pipeline and ColumnTransformer"],
            expected_outcome="A complete pipeline with cross-validation score.",
            lab_module="preprocessing",
        ),
        Exercise(
            id="ex_pp_a1", section="preprocessing",
            title="Preprocessing Experiment",
            difficulty="advanced",
            description="Compare different preprocessing strategies on model performance.",
            steps=[
                "Load a dataset and select features/target",
                "Strategy A: Drop missing, no scaling",
                "Strategy B: Median imputation + StandardScaler",
                "Strategy C: Median imputation + MinMaxScaler",
                "Strategy D: Median imputation + RobustScaler",
                "Compare accuracy of each on the same model",
                "Document which strategy works best and why",
            ],
            hints=["Use Pipeline to ensure consistent preprocessing"],
            expected_outcome="A comparison table showing how preprocessing choices affect model performance.",
            lab_module="preprocessing",
        ),
    ],
    "feature_engineering": [
        Exercise(
            id="ex_fe_b1", section="feature_engineering",
            title="Create Titanic Features",
            difficulty="beginner",
            description="Engineer new features from the Titanic dataset.",
            steps=[
                "Create FamilySize = SibSp + Parch + 1",
                "Create IsAlone = (FamilySize == 1)",
                "Extract Title from Name using str.extract()",
                "Create FarePerPerson = Fare / FamilySize",
                "Check if these new features correlate with Survival",
            ],
            hints=["Use df['Name'].str.extract(r' ([A-Za-z]+)\\.')"],
            expected_outcome="Four new features with correlation analysis showing their predictive value.",
            lab_module="feature_engineering",
        ),
        Exercise(
            id="ex_fe_i1", section="feature_engineering",
            title="Mathematical Transformations",
            difficulty="intermediate",
            description="Apply and compare transformations on California Housing.",
            steps=[
                "Load the dataset and check the distribution of median_income",
                "Apply log transformation with np.log1p()",
                "Apply square root transformation",
                "Compare distributions before and after",
                "Compute skewness for original and transformed versions",
            ],
            hints=["Use np.log1p(), np.sqrt(), df.skew()"],
            expected_outcome="Comparison of three distributions with skewness values showing which transformation works best.",
            lab_module="feature_engineering",
        ),
        Exercise(
            id="ex_fe_a1", section="feature_engineering",
            title="Feature Selection Pipeline",
            difficulty="advanced",
            description="Build a feature selection pipeline and measure impact.",
            steps=[
                "Load a classification dataset",
                "Train a model with all features, record F1",
                "Apply VarianceThreshold (remove near-zero variance)",
                "Apply correlation-based selection (remove |r| > 0.9 pairs)",
                "Apply SelectKBest (keep top 10)",
                "Compare F1 scores at each step",
                "Document which features were selected and why",
            ],
            hints=["Use VarianceThreshold, SelectKBest, and correlation matrix"],
            expected_outcome="A comparison showing model performance at each selection step.",
            lab_module="feature_engineering",
        ),
    ],
    "classification": [
        Exercise(
            id="ex_clf_b1", section="classification",
            title="First Classification Model",
            difficulty="beginner",
            description="Train a Logistic Regression classifier on Iris.",
            steps=[
                "Load Iris dataset",
                "Split 80/20 with random_state=42",
                "Train LogisticRegression with StandardScaler",
                "Print accuracy and classification report",
            ],
            hints=["Use Pipeline with StandardScaler and LogisticRegression"],
            expected_outcome="A trained model with accuracy around 96%+.",
            lab_module="classification",
        ),
        Exercise(
            id="ex_clf_i1", section="classification",
            title="Compare Classifiers",
            difficulty="intermediate",
            description="Compare 4 classifiers on the same dataset.",
            steps=[
                "Load a classification dataset",
                "Build pipelines for: LR, KNN, Random Forest, Gradient Boosting",
                "Use the same train/test split for all",
                "Train each and compute accuracy, F1, and AUC",
                "Create a comparison table and bar chart",
                "Identify the best model",
            ],
            hints=["Use cross_val_score with the same CV for all models"],
            expected_outcome="A comparison table and chart showing performance of all 4 models.",
            lab_module="classification",
        ),
        Exercise(
            id="ex_clf_a1", section="classification",
            title="End-to-End Titanic Classification",
            difficulty="advanced",
            description="Build a complete classification pipeline for Titanic survival.",
            steps=[
                "Load Titanic, perform EDA",
                "Engineer features: Title, FamilySize, IsAlone",
                "Build preprocessing pipeline with ColumnTransformer",
                "Train 3+ models with cross-validation",
                "Compare metrics (accuracy, F1, AUC)",
                "Select best model and explain why",
                "Generate the Python code for the complete workflow",
            ],
            hints=["Use Pipeline, ColumnTransformer, cross_val_score"],
            expected_outcome="A complete, reproducible classification pipeline with model comparison and justification.",
            lab_module="classification",
        ),
    ],
    "regression": [
        Exercise(
            id="ex_reg_b1", section="regression",
            title="Linear Regression Baseline",
            difficulty="beginner",
            description="Train a linear regression on California Housing.",
            steps=[
                "Load California Housing dataset",
                "Split 80/20",
                "Train LinearRegression",
                "Print R², MAE, RMSE",
                "Plot actual vs predicted",
            ],
            hints=["Use sklearn LinearRegression"],
            expected_outcome="A linear regression model with R² around 0.60.",
            lab_module="regression",
        ),
        Exercise(
            id="ex_reg_i1", section="regression",
            title="Compare Regression Models",
            difficulty="intermediate",
            description="Compare multiple regression algorithms.",
            steps=[
                "Load California Housing",
                "Train: LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor",
                "Evaluate each with R², MAE, RMSE",
                "Create a comparison table",
                "Plot residual distributions for each model",
            ],
            hints=["Use Pipeline with scaling for linear models"],
            expected_outcome="A comparison table showing all models and their metrics.",
            lab_module="regression",
        ),
        Exercise(
            id="ex_reg_a1", section="regression",
            title="Residual Analysis",
            difficulty="advanced",
            description="Perform complete residual analysis on the best regression model.",
            steps=[
                "Train a GradientBoostingRegressor on California Housing",
                "Create residuals = y_test - y_pred",
                "Plot residuals vs predicted values",
                "Create a Q-Q plot",
                "Check for heteroscedasticity",
                "Identify outliers and investigate",
            ],
            hints=["Use scipy.stats.probplot for Q-Q plot"],
            expected_outcome="Four diagnostic plots and a written analysis of model assumptions.",
            lab_module="regression",
        ),
    ],
    "evaluation": [
        Exercise(
            id="ex_eval_b1", section="evaluation",
            title="Confusion Matrix Workshop",
            difficulty="beginner",
            description="Build and interpret a confusion matrix.",
            steps=[
                "Train a classifier on any dataset",
                "Generate predictions on the test set",
                "Compute the confusion matrix",
                "Identify TP, TN, FP, FN",
                "Calculate accuracy, precision, recall, F1 manually",
            ],
            hints=["Use confusion_matrix from sklearn.metrics"],
            expected_outcome="A confusion matrix with all four components identified and metrics calculated.",
            lab_module="evaluation",
        ),
        Exercise(
            id="ex_eval_i1", section="evaluation",
            title="Metric Selection Challenge",
            difficulty="intermediate",
            description="Choose the right metric for different scenarios.",
            steps=[
                "Scenario 1: Spam detection (FP = real email blocked). Which metric?",
                "Scenario 2: Cancer screening (FN = missed cancer). Which metric?",
                "Scenario 3: Balanced binary classification. Which metric?",
                "Scenario 4: House price prediction. Which metrics?",
                "Implement and verify each scenario",
            ],
            hints=["Think about the cost of each type of error"],
            expected_outcome="Justified metric choices for each scenario with implementation.",
            lab_module="evaluation",
        ),
    ],
    "model_selection": [
        Exercise(
            id="ex_sel_i1", section="model_selection",
            title="Baseline vs Complex Models",
            difficulty="intermediate",
            description="Establish a baseline and compare with complex models.",
            steps=[
                "Load a classification dataset",
                "Train DummyClassifier (majority class) as baseline",
                "Train LogisticRegression",
                "Train RandomForestClassifier",
                "Train GradientBoostingClassifier",
                "Compare all using the same CV folds",
                "Document the improvement over baseline",
            ],
            hints=["Use DummyClassifier for baseline"],
            expected_outcome="A comparison showing the improvement from baseline to the best model.",
            lab_module="model_comparison",
        ),
        Exercise(
            id="ex_sel_a1", section="model_selection",
            title="Hyperparameter Tuning Challenge",
            difficulty="advanced",
            description="Tune a Random Forest using grid search and random search.",
            steps=[
                "Load a classification dataset",
                "Define parameter grid for RandomForest",
                "Run GridSearchCV with 3 parameters",
                "Run RandomizedSearchCV with 5 parameters",
                "Compare results and timing",
                "Document which found the better model and why",
            ],
            hints=["Use GridSearchCV and RandomizedSearchCV"],
            expected_outcome="Comparison of grid vs random search with timing and performance.",
            lab_module="model_comparison",
        ),
    ],
    "clustering": [
        Exercise(
            id="ex_clus_b1", section="clustering",
            title="K-Means on Iris",
            difficulty="beginner",
            description="Apply K-Means to the Iris dataset.",
            steps=[
                "Load Iris (ignore the species column)",
                "Scale the features with StandardScaler",
                "Apply K-Means with k=3",
                "Visualise clusters using PCA (2D)",
                "Compare clusters with actual species",
            ],
            hints=["Use PCA(n_components=2) for visualisation"],
            expected_outcome="A 2D scatter plot showing 3 clusters, with comparison to actual species.",
            lab_module="clustering",
        ),
        Exercise(
            id="ex_clus_i1", section="clustering",
            title="Find Optimal K",
            difficulty="intermediate",
            description="Use elbow method and silhouette to find optimal k.",
            steps=[
                "Scale the features",
                "Run K-Means for k=2 to k=10",
                "Plot inertia vs k (elbow method)",
                "Plot silhouette scores vs k",
                "Choose the optimal k and justify",
            ],
            hints=["Use inertia_ and silhouette_score"],
            expected_outcome="Two plots (elbow and silhouette) with a justified choice of k.",
            lab_module="clustering",
        ),
        Exercise(
            id="ex_clus_a1", section="clustering",
            title="Customer Segmentation",
            difficulty="advanced",
            description="Segment customers using multiple clustering algorithms.",
            steps=[
                "Create a synthetic customer dataset (spending, frequency, recency)",
                "Scale the features",
                "Apply K-Means, DBSCAN, and Agglomerative Clustering",
                "Compare results visually (PCA 2D)",
                "Profile each cluster for each algorithm",
                "Recommend the best approach with justification",
            ],
            hints=["Use StandardScaler, PCA, and cluster profiling"],
            expected_outcome="Comparison of three clustering approaches with profiles and recommendation.",
            lab_module="clustering",
        ),
    ],
    "model_comparison": [
        Exercise(
            id="ex_cmp_i1", section="model_comparison",
            title="Fair Model Comparison",
            difficulty="intermediate",
            description="Compare models with completely consistent methodology.",
            steps=[
                "Load a dataset",
                "Define preprocessing pipeline",
                "Train 4 models: LR, KNN, RF, GB",
                "Use identical CV, preprocessing, and metric",
                "Create a comparison table (accuracy, F1, AUC)",
                "Create a bar chart visualisation",
                "Select and justify the best model",
            ],
            hints=["Use cross_val_score with same CV for all"],
            expected_outcome="A fair comparison table and chart with justified model selection.",
            lab_module="model_comparison",
        ),
    ],
    "automl": [
        Exercise(
            id="ex_auto_i1", section="automl",
            title="AutoML Exploration",
            difficulty="intermediate",
            description="Use the AutoML module and interpret results.",
            steps=[
                "Navigate to the AutoML page",
                "Select a classification dataset",
                "Let AutoML run with default settings",
                "Review the comparison table",
                "Identify the best model and its metrics",
                "Generate the Python code",
                "Reproduce the result in a notebook",
            ],
            hints=["Follow the AutoML workflow step by step"],
            expected_outcome="Understanding of AutoML workflow and ability to reproduce results.",
            lab_module="automl",
        ),
        Exercise(
            id="ex_auto_a1", section="automl",
            title="AutoML vs Manual",
            difficulty="advanced",
            description="Compare AutoML results with a manually engineered model.",
            steps=[
                "Run AutoML on a dataset, record the best model and score",
                "Manually perform EDA and feature engineering on the same data",
                "Train the same algorithm that AutoML chose",
                "Compare manual model vs AutoML model",
                "Document where manual engineering helped (or didn't)",
            ],
            hints=["Use the same train/test split for both"],
            expected_outcome="A comparison showing whether manual feature engineering improves on AutoML.",
            lab_module="automl",
        ),
    ],
}


def get_exercises_for_section(section_id: str) -> list[Exercise]:
    """Return exercises for a specific section."""
    return EXERCISES.get(section_id, [])


def get_all_exercises() -> dict[str, list[Exercise]]:
    """Return all exercises grouped by section."""
    return EXERCISES


def get_exercise_by_id(exercise_id: str) -> Exercise | None:
    """Find a specific exercise by ID."""
    for exercises in EXERCISES.values():
        for ex in exercises:
            if ex.id == exercise_id:
                return ex
    return None


def get_total_exercises() -> int:
    """Count total exercises across all sections."""
    return sum(len(exs) for exs in EXERCISES.values())
