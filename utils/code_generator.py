"""
Centralized Python code generator for the Data Science Learning Studio.

Builds complete, runnable, well-commented Python scripts from operations
tracked across the entire data science pipeline — loading, EDA,
preprocessing, feature engineering, training, evaluation, and prediction.

Usage::

    gen = CodeGenerator()
    gen.add_loading("iris.csv")
    gen.add_preprocessing(strategy="median", scale=True)
    gen.add_train_test_split(test_size=0.2, target="species")
    gen.add_training("RandomForestClassifier", n_estimators=100)
    gen.add_evaluation(task="classification")
    print(gen.build())
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Import maps ─────────────────────────────────────────────────────

_IMPORT_MAP: dict[str, str] = {
    "pandas": "import pandas as pd",
    "numpy": "import numpy as np",
    "sklearn.compose": "from sklearn.compose import ColumnTransformer",
    "sklearn.pipeline": "from sklearn.pipeline import Pipeline",
    "sklearn.impute": "from sklearn.impute import SimpleImputer",
    "sklearn.preprocessing.scaler": "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
    "sklearn.preprocessing.encoder": "from sklearn.preprocessing import OneHotEncoder, LabelEncoder",
    "sklearn.model_selection": "from sklearn.model_selection import train_test_split",
    "sklearn.metrics.classification": (
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_score, recall_score, f1_score,\n"
        "    classification_report, confusion_matrix, roc_auc_score\n"
        ")"
    ),
    "sklearn.metrics.regression": (
        "from sklearn.metrics import (\n"
        "    mean_absolute_error, mean_squared_error, r2_score\n"
        ")"
    ),
    "sklearn.cluster": "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering",
    "sklearn.decomposition": "from sklearn.decomposition import PCA",
    "matplotlib": "import matplotlib.pyplot as plt",
    "seaborn": "import sns  # import seaborn as sns",
    "plotly.express": "import plotly.express as px",
    "sklearn.feature_selection": "from sklearn.feature_selection import VarianceThreshold",
    "sklearn.preprocessing.polynomial": "from sklearn.preprocessing import PolynomialFeatures",
}

_SCALER_MAP: dict[str, str] = {
    "standard": "StandardScaler()",
    "minmax": "MinMaxScaler()",
    "robust": "RobustScaler()",
}

_IMPUTER_MAP: dict[str, str] = {
    "mean": "SimpleImputer(strategy='mean')",
    "median": "SimpleImputer(strategy='median')",
    "mode": "SimpleImputer(strategy='most_frequent')",
    "constant": "SimpleImputer(strategy='constant', fill_value=0)",
}


# ── Code block data class ──────────────────────────────────────────

@dataclass
class CodeBlock:
    """A single block of generated code with metadata."""
    section: str
    title: str
    code: str
    imports: list[str] = field(default_factory=list)
    order: int = 0


# ── Main generator ─────────────────────────────────────────────────

class CodeGenerator:
    """
    Accumulates data-science operations and produces a single,
    runnable, well-commented Python script.

    Operations are added in pipeline order and the final script is
    built with ``build()``.
    """

    def __init__(self) -> None:
        self._blocks: list[CodeBlock] = []
        self._imports: set[str] = set()
        self._order = 0
        self._target: str = ""
        self._task: str = ""
        self._dataset_name: str = ""

    # ── Public API: add operations ──────────────────────────────────

    def add_loading(self, source: str, file_type: str = "csv") -> None:
        """Add data loading code."""
        self._dataset_name = source
        self._add_import("pandas")

        if file_type == "csv":
            code = f"df = pd.read_csv('{source}')"
        elif file_type in ("xlsx", "excel"):
            code = f"df = pd.read_excel('{source}')"
        elif file_type == "parquet":
            code = f"df = pd.read_parquet('{source}')"
        else:
            code = f"df = pd.read_csv('{source}')  # adjust reader for {file_type}"

        self._add_block("Loading", "Load Data", code)

    def add_eda_basics(self) -> None:
        """Add basic EDA code (shape, dtypes, describe, missing)."""
        self._add_import("pandas")
        code = textwrap.dedent("""\
            # ── Exploratory Data Analysis ──
            print(f"Shape: {df.shape}")
            print(f"\\nData types:\\n{df.dtypes}")
            print(f"\\nMissing values:\\n{df.isnull().sum()}")
            print(f"\\nDescriptive statistics:\\n{df.describe()}")
        """)
        self._add_block("EDA", "Exploratory Data Analysis", code)

    def add_eda_visualizations(self, chart_types: list[str] | None = None) -> None:
        """Add EDA visualization code."""
        charts = chart_types or ["histogram", "correlation", "boxplot"]
        self._add_import("pandas")
        self._add_import("matplotlib")

        lines = ["# ── Visualisations ──"]

        if "histogram" in charts:
            self._add_import("matplotlib")
            lines.extend([
                "",
                "# Histograms of all numerical columns",
                "df.select_dtypes('number').hist(figsize=(12, 8), bins=20)",
                "plt.tight_layout()",
                "plt.show()",
            ])

        if "correlation" in charts:
            lines.extend([
                "",
                "# Correlation heatmap",
                "import numpy as np",
                "corr = df.select_dtypes('number').corr()",
                "plt.figure(figsize=(10, 8))",
                "plt.imshow(corr, cmap='coolwarm', aspect='auto')",
                "plt.colorbar()",
                "plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')",
                "plt.yticks(range(len(corr)), corr.columns)",
                "plt.title('Correlation Matrix')",
                "plt.tight_layout()",
                "plt.show()",
            ])

        if "boxplot" in charts:
            self._add_import("matplotlib")
            num_cols_code = "df.select_dtypes('number').columns"
            lines.extend([
                "",
                "# Box plots",
                f"num_cols = {num_cols_code}",
                "df[num_cols].boxplot(figsize=(12, 6))",
                "plt.title('Box Plots of Numerical Features')",
                "plt.xticks(rotation=45, ha='right')",
                "plt.tight_layout()",
                "plt.show()",
            ])

        self._add_block("EDA", "EDA Visualisations", "\n".join(lines))

    def add_preprocessing(
        self,
        strategy: str = "median",
        scale: bool = False,
        scaler: str = "standard",
        encode_categorical: bool = True,
        handle_missing: str | None = None,
        missing_strategy: str = "median",
    ) -> None:
        """Add preprocessing pipeline code."""
        self._add_import("sklearn.compose")
        self._add_import("sklearn.pipeline")
        self._add_import("sklearn.impute")

        lines = ["# ── Preprocessing ──"]

        if scale:
            self._add_import("sklearn.preprocessing.scaler")
            scaler_str = _SCALER_MAP.get(scaler, "StandardScaler()")
        else:
            scaler_str = None

        imputer_str = _IMPUTER_MAP.get(strategy, "SimpleImputer(strategy='median')")

        # Auto-detect columns
        lines.extend([
            "",
            "# Auto-detect column types",
            "num_cols = df.select_dtypes(include='number').columns.tolist()",
            "cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()",
            "",
            "preprocessor = ColumnTransformer(transformers=[",
            f"    ('num', Pipeline([",
            f"        ('imputer', {imputer_str}),",
        ])

        if scaler_str:
            lines.append(f"        ('scaler', {scaler_str}),")
        lines.extend([
            f"    ]), num_cols),",
        ])

        if encode_categorical:
            self._add_import("sklearn.preprocessing.encoder")
            lines.extend([
                f"    ('cat', Pipeline([",
                f"        ('imputer', SimpleImputer(strategy='most_frequent')),",
                f"        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),",
                f"    ]), cat_cols),",
            ])

        lines.append("])")
        self._add_block("Preprocessing", "Build Preprocessing Pipeline", "\n".join(lines))

    def add_feature_engineering(self, operations: list[dict[str, Any]] | None = None) -> None:
        """Add feature engineering code."""
        lines = ["# ── Feature Engineering ──"]

        if not operations:
            lines.extend([
                "",
                "# Example: create interaction features",
                "# df['feature_interaction'] = df['col_a'] * df['col_b']",
                "",
                "# Example: binning",
                "# df['age_bin'] = pd.cut(df['age'], bins=5, labels=False)",
                "",
                "# Example: log transform",
                "# import numpy as np",
                "# df['log_feature'] = np.log1p(df['skewed_feature'])",
            ])
        else:
            for op in operations:
                op_type = op.get("type", "")
                if op_type == "math_transform":
                    col = op.get("column", "feature")
                    func = op.get("function", "log")
                    if func == "log":
                        lines.append(f"df['{col}_log'] = np.log1p(df['{col}'])  # log(1+x) handles zeros")
                    elif func == "sqrt":
                        lines.append(f"df['{col}_sqrt'] = np.sqrt(df['{col}'].clip(lower=0))")
                    elif func == "square":
                        lines.append(f"df['{col}_sq'] = df['{col}'] ** 2")
                elif op_type == "binning":
                    col = op.get("column", "feature")
                    n_bins = op.get("n_bins", 5)
                    method = op.get("method", "equal_width")
                    if method == "equal_width":
                        lines.append(f"df['{col}_bin'] = pd.cut(df['{col}'], bins={n_bins}, labels=False)")
                    else:
                        lines.append(f"df['{col}_bin'] = pd.qcut(df['{col}'], q={n_bins}, labels=False, duplicates='drop')")
                elif op_type == "interaction":
                    col_a = op.get("column_a", "a")
                    col_b = op.get("column_b", "b")
                    op_name = op.get("operation", "multiply")
                    if op_name == "multiply":
                        lines.append(f"df['{col_a}_x_{col_b}'] = df['{col_a}'] * df['{col_b}']")
                    elif op_name == "divide":
                        lines.append(f"df['{col_a}_div_{col_b}'] = df['{col_a}'] / df['{col_b}'].replace(0, np.nan)")
                    elif op_name == "add":
                        lines.append(f"df['{col_a}_plus_{col_b}'] = df['{col_a}'] + df['{col_b}']")
                    elif op_name == "subtract":
                        lines.append(f"df['{col_a}_minus_{col_b}'] = df['{col_a}'] - df['{col_b}']")
                elif op_type == "polynomial":
                    col = op.get("column", "feature")
                    degree = op.get("degree", 2)
                    lines.append(f"df['{col}_pow{degree}'] = df['{col}'] ** {degree}")

        self._add_block("Feature Engineering", "Feature Engineering", "\n".join(lines))

    def add_train_test_split(
        self,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        """Add train/test split code."""
        self._target = target
        self._add_import("sklearn.model_selection")

        lines = [
            "# ── Train / Test Split ──",
            f"X = df.drop(columns=['{target}'])",
            f"y = df['{target}']",
            "",
        ]

        if stratify:
            lines.append(
                f"X_train, X_test, y_train, y_test = train_test_split(\n"
                f"    X, y, test_size={test_size}, random_state={random_state}, stratify=y\n"
                f")"
            )
        else:
            lines.append(
                f"X_train, X_test, y_train, y_test = train_test_split(\n"
                f"    X, y, test_size={test_size}, random_state={random_state}\n"
                f")"
            )

        lines.extend([
            "",
            f"print(f'Train: {{X_train.shape[0]}} samples, Test: {{X_test.shape[0]}} samples')",
        ])

        self._add_block("Split", "Train / Test Split", "\n".join(lines))

    def add_training(
        self,
        model_class: str,
        params: dict[str, Any] | None = None,
        task: str = "classification",
    ) -> None:
        """Add model training code."""
        self._task = task
        param_str = ""
        if params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())

        step_name = "classifier" if task == "classification" else "regressor"

        lines = [
            "# ── Model Training ──",
            f"pipeline = Pipeline([\n"
            f"    ('preprocessor', preprocessor),\n"
            f"    ('{step_name}', {model_class}({param_str}))\n"
            f"])",
            "",
            "pipeline.fit(X_train, y_train)",
            f"y_pred = pipeline.predict(X_test)",
            "",
        ]

        self._add_block("Training", f"Train {model_class}", "\n".join(lines))

    def add_evaluation(self, task: str | None = None) -> None:
        """Add evaluation code based on task type."""
        task = task or self._task or "classification"

        if task == "classification":
            self._add_import("sklearn.metrics.classification")
            code = textwrap.dedent("""\
                # ── Evaluation ──
                y_pred = pipeline.predict(X_test)

                print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
                print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
                print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
                print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
                print()
                print("Classification Report:")
                print(classification_report(y_test, y_pred))

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, cmap='Blues')
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.show()
            """)
        else:
            self._add_import("sklearn.metrics.regression")
            code = textwrap.dedent("""\
                # ── Evaluation ──
                y_pred = pipeline.predict(X_test)

                print(f"R²:   {r2_score(y_test, y_pred):.4f}")
                print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
                print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
                print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

                # Actual vs Predicted
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted')
                plt.tight_layout()
                plt.show()

                # Residuals
                residuals = y_test - y_pred
                plt.figure(figsize=(8, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted')
                plt.ylabel('Residual')
                plt.title('Residual Plot')
                plt.tight_layout()
                plt.show()
            """)
        self._add_block("Evaluation", "Model Evaluation", code)

    def add_prediction(self, sample_data: dict[str, Any] | None = None) -> None:
        """Add prediction code."""
        lines = [
            "# ── Prediction ──",
            "",
        ]

        if sample_data:
            items = ", ".join(f"'{k}': [{v}]" for k, v in sample_data.items())
            lines.extend([
                f"sample = pd.DataFrame({{{items}}})",
                "predictions = pipeline.predict(sample)",
                "print(f'Predictions: {predictions}')",
            ])
        else:
            lines.extend([
                "# Predict on new data",
                "# sample = pd.read_csv('new_data.csv')",
                "# predictions = pipeline.predict(sample)",
                "# print(f'Predictions: {predictions}')",
            ])

        self._add_block("Prediction", "Make Predictions", "\n".join(lines))

    def add_cross_validation(self, cv: int = 5, task: str | None = None) -> None:
        """Add cross-validation code."""
        task = task or self._task or "classification"
        self._add_import("sklearn.model_selection")

        scoring = "accuracy" if task == "classification" else "r2"

        code = textwrap.dedent(f"""\
            # ── Cross-Validation ──
            from sklearn.model_selection import cross_val_score

            cv_scores = cross_val_score(pipeline, X, y, cv={cv}, scoring='{scoring}')
            print(f"CV Scores: {{cv_scores}}")
            print(f"Mean {scoring}: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std():.4f}})")
        """)
        self._add_block("Validation", "Cross-Validation", code)

    def add_clustering(
        self,
        algorithm: str = "KMeans",
        features: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Add clustering code."""
        self._add_import("sklearn.cluster")
        self._add_import("sklearn.preprocessing.scaler")
        self._add_import("sklearn.decomposition")

        param_str = ""
        if params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())

        feat_str = repr(features) if features else "df.select_dtypes('number').columns.tolist()"

        code = textwrap.dedent(f"""\
            # ── Clustering ──
            features = {feat_str}
            X = df[features].dropna()

            # Scale
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

            # Cluster
            model = {algorithm}({param_str})
            labels = model.fit_predict(X_scaled)
            df['Cluster'] = labels

            # PCA for visualisation
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f'PC1 ({{pca.explained_variance_ratio_[0]:.1%}})')
            plt.ylabel(f'PC2 ({{pca.explained_variance_ratio_[1]:.1%}})')
            plt.title(f'{algorithm} Clustering — {{len(set(labels))}} clusters')
            plt.tight_layout()
            plt.show()
        """)
        self._add_block("Clustering", "Clustering Analysis", code)

    def add_comparison(self, models: list[str], task: str = "classification") -> None:
        """Add model comparison code."""
        lines = [
            "# ── Model Comparison ──",
            "",
        ]

        for model_name in models:
            lines.extend([
                f"# {model_name}",
                f"pipe_{model_name.lower().replace(' ', '_')} = Pipeline([\n"
                f"    ('preprocessor', preprocessor),\n"
                f"    ('model', {model_name}())\n"
                f"])",
                "",
            ])

        lines.extend([
            "results = {}",
            "",
        ])

        for model_name in models:
            var = f"pipe_{model_name.lower().replace(' ', '_')}"
            if task == "classification":
                lines.extend([
                    f"{var}.fit(X_train, y_train)",
                    f"y_pred_{model_name.lower().replace(' ', '_')} = {var}.predict(X_test)",
                    f"results['{model_name}'] = accuracy_score(y_test, y_pred_{model_name.lower().replace(' ', '_')})",
                    "",
                ])
            else:
                lines.extend([
                    f"{var}.fit(X_train, y_train)",
                    f"y_pred_{model_name.lower().replace(' ', '_')} = {var}.predict(X_test)",
                    f"results['{model_name}'] = r2_score(y_test, y_pred_{model_name.lower().replace(' ', '_')})",
                    "",
                ])

        metric = "Accuracy" if task == "classification" else "R²"
        lines.extend([
            "# Summary",
            "results_df = pd.DataFrame(list(results.items()), columns=['Model', '" + metric + "'])",
            "results_df = results_df.sort_values('" + metric + "', ascending=False)",
            "print(results_df.to_string(index=False))",
        ])

        self._add_block("Comparison", "Model Comparison", "\n".join(lines))

    # ── Build ───────────────────────────────────────────────────────

    def build(self) -> str:
        """Build the complete, runnable Python script."""
        # Collect all needed imports
        import_lines = []
        seen = set()
        for block in self._blocks:
            for imp in block.imports:
                if imp not in seen:
                    import_lines.append(_IMPORT_MAP.get(imp, imp))
                    seen.add(imp)

        # Always include matplotlib for plots
        if "matplotlib" not in seen and any(b.section in ("EDA", "Evaluation", "Clustering") for b in self._blocks):
            import_lines.append(_IMPORT_MAP["matplotlib"])

        # Header
        header = (
            f'"""\n'
            f"Generated Python Script — Data Science Lab\n"
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f'Dataset: {self._dataset_name}\n'
            f'"""\n'
        )

        parts = [header]

        if import_lines:
            parts.append("\n".join(import_lines))
            parts.append("")

        for block in sorted(self._blocks, key=lambda b: b.order):
            parts.append(f"# {'=' * 60}")
            parts.append(f"#  {block.title}")
            parts.append(f"# {'=' * 60}")
            parts.append(block.code.rstrip())
            parts.append("")

        return "\n".join(parts)

    def get_blocks(self) -> list[dict[str, str]]:
        """Return block metadata for UI display."""
        return [
            {"section": b.section, "title": b.title, "code": b.code}
            for b in sorted(self._blocks, key=lambda b: b.order)
        ]

    def clear(self) -> None:
        """Reset the generator."""
        self._blocks.clear()
        self._imports.clear()
        self._order = 0
        self._target = ""
        self._task = ""
        self._dataset_name = ""

    # ── Internal helpers ────────────────────────────────────────────

    def _add_block(self, section: str, title: str, code: str) -> None:
        block = CodeBlock(
            section=section,
            title=title,
            code=code,
            imports=list(self._imports),
            order=self._order,
        )
        self._blocks.append(block)
        self._order += 1

    def _add_import(self, key: str) -> None:
        self._imports.add(key)
