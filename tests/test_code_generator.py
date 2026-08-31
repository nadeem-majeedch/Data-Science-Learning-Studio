"""
Tests for utils/code_generator.py
"""

import pytest

from utils.code_generator import CodeGenerator


# ── Basic construction ──────────────────────────────────────────────

class TestBasicConstruction:
    def test_empty_generator(self):
        gen = CodeGenerator()
        code = gen.build()
        assert isinstance(code, str)
        assert "Generated Python Script" in code

    def test_single_loading(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        code = gen.build()
        assert "import pandas as pd" in code
        assert "pd.read_csv('data.csv')" in code

    def test_excel_loading(self):
        gen = CodeGenerator()
        gen.add_loading("data.xlsx", file_type="xlsx")
        code = gen.build()
        assert "pd.read_excel('data.xlsx')" in code

    def test_parquet_loading(self):
        gen = CodeGenerator()
        gen.add_loading("data.parquet", file_type="parquet")
        code = gen.build()
        assert "pd.read_parquet('data.parquet')" in code


# ── EDA ─────────────────────────────────────────────────────────────

class TestEDA:
    def test_eda_basics(self):
        gen = CodeGenerator()
        gen.add_eda_basics()
        code = gen.build()
        assert "Exploratory Data Analysis" in code
        assert "df.dtypes" in code
        assert "df.describe()" in code

    def test_eda_visualizations_default(self):
        gen = CodeGenerator()
        gen.add_eda_visualizations()
        code = gen.build()
        assert "histograms" in code.lower() or "hist(" in code
        assert "corr" in code.lower() or "correlation" in code.lower()

    def test_eda_visualizations_specific(self):
        gen = CodeGenerator()
        gen.add_eda_visualizations(chart_types=["histogram"])
        code = gen.build()
        assert "hist(" in code

    def test_eda_visualizations_boxplot(self):
        gen = CodeGenerator()
        gen.add_eda_visualizations(chart_types=["boxplot"])
        code = gen.build()
        assert "boxplot" in code


# ── Preprocessing ───────────────────────────────────────────────────

class TestPreprocessing:
    def test_basic_preprocessing(self):
        gen = CodeGenerator()
        gen.add_preprocessing()
        code = gen.build()
        assert "ColumnTransformer" in code
        assert "Pipeline" in code
        assert "SimpleImputer" in code

    def test_scaled_preprocessing(self):
        gen = CodeGenerator()
        gen.add_preprocessing(scale=True, scaler="standard")
        code = gen.build()
        assert "StandardScaler" in code

    def test_minmax_scaler(self):
        gen = CodeGenerator()
        gen.add_preprocessing(scale=True, scaler="minmax")
        code = gen.build()
        assert "MinMaxScaler" in code

    def test_robust_scaler(self):
        gen = CodeGenerator()
        gen.add_preprocessing(scale=True, scaler="robust")
        code = gen.build()
        assert "RobustScaler" in code

    def test_no_encode_categorical(self):
        gen = CodeGenerator()
        gen.add_preprocessing(encode_categorical=False)
        code = gen.build()
        assert "OneHotEncoder" not in code

    def test_imputer_strategy_mean(self):
        gen = CodeGenerator()
        gen.add_preprocessing(strategy="mean")
        code = gen.build()
        assert "strategy='mean'" in code

    def test_imputer_strategy_mode(self):
        gen = CodeGenerator()
        gen.add_preprocessing(strategy="mode")
        code = gen.build()
        assert "strategy='most_frequent'" in code


# ── Feature engineering ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_default_feature_engineering(self):
        gen = CodeGenerator()
        gen.add_feature_engineering()
        code = gen.build()
        assert "Feature Engineering" in code
        assert "log" in code.lower() or "binning" in code.lower()

    def test_math_transform_log(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "math_transform", "column": "age", "function": "log"}])
        code = gen.build()
        assert "log1p" in code

    def test_math_transform_sqrt(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "math_transform", "column": "income", "function": "sqrt"}])
        code = gen.build()
        assert "sqrt" in code

    def test_binning(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "binning", "column": "age", "n_bins": 5, "method": "equal_width"}])
        code = gen.build()
        assert "pd.cut" in code

    def test_quantile_binning(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "binning", "column": "age", "n_bins": 4, "method": "quantile"}])
        code = gen.build()
        assert "pd.qcut" in code

    def test_interaction_multiply(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "interaction", "column_a": "a", "column_b": "b", "operation": "multiply"}])
        code = gen.build()
        assert "df['a'] * df['b']" in code

    def test_interaction_divide(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "interaction", "column_a": "x", "column_b": "y", "operation": "divide"}])
        code = gen.build()
        assert "replace(0, np.nan)" in code

    def test_polynomial(self):
        gen = CodeGenerator()
        gen.add_feature_engineering([{"type": "polynomial", "column": "x", "degree": 3}])
        code = gen.build()
        assert "df['x'] ** 3" in code


# ── Train/test split ────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_basic_split(self):
        gen = CodeGenerator()
        gen.add_train_test_split(target="species")
        code = gen.build()
        assert "train_test_split" in code
        assert "y = df['species']" in code
        assert "X = df.drop(columns=['species'])" in code

    def test_custom_split(self):
        gen = CodeGenerator()
        gen.add_train_test_split(target="target", test_size=0.3, random_state=123)
        code = gen.build()
        assert "test_size=0.3" in code
        assert "random_state=123" in code

    def test_no_stratify(self):
        gen = CodeGenerator()
        gen.add_train_test_split(target="y", stratify=False)
        code = gen.build()
        assert "stratify" not in code

    def test_with_stratify(self):
        gen = CodeGenerator()
        gen.add_train_test_split(target="y", stratify=True)
        code = gen.build()
        assert "stratify=y" in code


# ── Training ────────────────────────────────────────────────────────

class TestTraining:
    def test_classification_training(self):
        gen = CodeGenerator()
        gen.add_training("RandomForestClassifier", params={"n_estimators": 100}, task="classification")
        code = gen.build()
        assert "RandomForestClassifier(n_estimators=100)" in code
        assert "'classifier'" in code
        assert "pipeline.fit(X_train, y_train)" in code

    def test_regression_training(self):
        gen = CodeGenerator()
        gen.add_training("Ridge", params={"alpha": 1.0}, task="regression")
        code = gen.build()
        assert "Ridge(alpha=1.0)" in code
        assert "'regressor'" in code

    def test_no_params(self):
        gen = CodeGenerator()
        gen.add_training("DecisionTreeClassifier")
        code = gen.build()
        assert "DecisionTreeClassifier()" in code


# ── Evaluation ──────────────────────────────────────────────────────

class TestEvaluation:
    def test_classification_evaluation(self):
        gen = CodeGenerator()
        gen.add_evaluation(task="classification")
        code = gen.build()
        assert "accuracy_score" in code
        assert "classification_report" in code
        assert "confusion_matrix" in code

    def test_regression_evaluation(self):
        gen = CodeGenerator()
        gen.add_evaluation(task="regression")
        code = gen.build()
        assert "r2_score" in code
        assert "mean_absolute_error" in code
        assert "RMSE" in code

    def test_auto_task_detection(self):
        gen = CodeGenerator()
        gen.add_training("RandomForestClassifier", task="classification")
        gen.add_evaluation()  # should auto-detect classification
        code = gen.build()
        assert "accuracy_score" in code


# ── Prediction ──────────────────────────────────────────────────────

class TestPrediction:
    def test_prediction_with_sample(self):
        gen = CodeGenerator()
        gen.add_prediction(sample_data={"age": [25], "income": [50000]})
        code = gen.build()
        assert "pd.DataFrame" in code
        assert "pipeline.predict" in code

    def test_prediction_without_sample(self):
        gen = CodeGenerator()
        gen.add_prediction()
        code = gen.build()
        assert "new_data.csv" in code or "pipeline.predict" in code


# ── Cross validation ────────────────────────────────────────────────

class TestCrossValidation:
    def test_cv_classification(self):
        gen = CodeGenerator()
        gen.add_cross_validation(cv=5, task="classification")
        code = gen.build()
        assert "cross_val_score" in code
        assert "cv=5" in code
        assert "scoring='accuracy'" in code

    def test_cv_regression(self):
        gen = CodeGenerator()
        gen.add_cross_validation(cv=10, task="regression")
        code = gen.build()
        assert "scoring='r2'" in code
        assert "cv=10" in code


# ── Clustering ──────────────────────────────────────────────────────

class TestClustering:
    def test_kmeans_clustering(self):
        gen = CodeGenerator()
        gen.add_clustering(algorithm="KMeans", features=["a", "b"], params={"n_clusters": 3})
        code = gen.build()
        assert "KMeans(n_clusters=3)" in code
        assert "StandardScaler" in code
        assert "PCA" in code

    def test_dbscan_clustering(self):
        gen = CodeGenerator()
        gen.add_clustering(algorithm="DBSCAN", params={"eps": 0.5, "min_samples": 5})
        code = gen.build()
        assert "DBSCAN(eps=0.5, min_samples=5)" in code


# ── Comparison ──────────────────────────────────────────────────────

class TestComparison:
    def test_comparison_classification(self):
        gen = CodeGenerator()
        gen.add_comparison(["LogisticRegression", "RandomForestClassifier"], task="classification")
        code = gen.build()
        assert "LogisticRegression" in code
        assert "RandomForestClassifier" in code
        assert "accuracy_score" in code

    def test_comparison_regression(self):
        gen = CodeGenerator()
        gen.add_comparison(["LinearRegression", "Ridge"], task="regression")
        code = gen.build()
        assert "r2_score" in code


# ── Full pipeline ───────────────────────────────────────────────────

class TestFullPipeline:
    def test_complete_classification_pipeline(self):
        gen = CodeGenerator()
        gen.add_loading("iris.csv")
        gen.add_eda_basics()
        gen.add_preprocessing(scale=True)
        gen.add_train_test_split(target="species", test_size=0.2)
        gen.add_training("RandomForestClassifier", params={"n_estimators": 100})
        gen.add_evaluation(task="classification")
        gen.add_prediction()
        code = gen.build()

        assert "pd.read_csv('iris.csv')" in code
        assert "ColumnTransformer" in code
        assert "train_test_split" in code
        assert "RandomForestClassifier" in code
        assert "accuracy_score" in code
        assert "pipeline.predict" in code

    def test_complete_regression_pipeline(self):
        gen = CodeGenerator()
        gen.add_loading("housing.csv")
        gen.add_preprocessing(scale=True, scaler="robust")
        gen.add_train_test_split(target="price", stratify=False)
        gen.add_training("GradientBoostingRegressor")
        gen.add_evaluation(task="regression")
        code = gen.build()

        assert "RobustScaler" in code
        assert "GradientBoostingRegressor" in code
        assert "r2_score" in code


# ── Blocks and clear ────────────────────────────────────────────────

class TestBlockManagement:
    def test_get_blocks(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        gen.add_eda_basics()
        blocks = gen.get_blocks()
        assert len(blocks) == 2
        assert blocks[0]["section"] == "Loading"
        assert blocks[1]["section"] == "EDA"

    def test_clear(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        gen.add_eda_basics()
        gen.clear()
        blocks = gen.get_blocks()
        assert len(blocks) == 0

    def test_ordering(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        gen.add_preprocessing()
        gen.add_training("RandomForestClassifier")
        blocks = gen.get_blocks()
        assert blocks[0]["section"] == "Loading"
        assert blocks[1]["section"] == "Preprocessing"
        assert blocks[2]["section"] == "Training"


# ── Code quality ────────────────────────────────────────────────────

class TestCodeQuality:
    def test_header_present(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        code = gen.build()
        assert '"""' in code
        assert "Generated Python Script" in code

    def test_no_duplicate_imports(self):
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        gen.add_eda_basics()
        gen.add_preprocessing()
        code = gen.build()
        # pandas import should appear only once
        assert code.count("import pandas as pd") == 1

    def test_runnable_syntax(self):
        """Generated code should at least parse as valid Python."""
        gen = CodeGenerator()
        gen.add_loading("data.csv")
        gen.add_preprocessing(scale=True)
        gen.add_train_test_split(target="y")
        gen.add_training("RandomForestClassifier")
        gen.add_evaluation(task="classification")
        code = gen.build()

        # Try to compile — will raise SyntaxError if invalid
        compile(code, "<generated>", "exec")
