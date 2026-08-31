"""
Reusable UI helper functions for Data Science Lab.

Provides consistent page headers, sidebar navigation, module cards,
and shared layout components used across every page.
"""

from __future__ import annotations

import streamlit as st

# ── Module registry ─────────────────────────────────────────────────
# Central source of truth for every module: icon, label, short
# description, and the learning objectives shown on each page.

MODULES: dict[str, dict[str, str]] = {
    "dataset_explorer": {
        "icon": "📂",
        "label": "Dataset Explorer",
        "title": "📂 Dataset Explorer",
        "subtitle": "Upload, load, and preview datasets from multiple sources.",
        "description": (
            "The Dataset Explorer is your starting point in any data science project. "
            "Upload your own CSV / TSV / Excel files or use built-in sample datasets "
            "to quickly load data into the lab environment."
        ),
        "objectives": [
            "Understand common file formats used in data science (CSV, TSV, Parquet).",
            "Preview raw data to assess its shape, types, and quality.",
            "Identify initial data issues before deeper analysis.",
        ],
        "help_text": (
            "Upload a file using the widget above, or select a sample dataset "
            "from the sidebar to get started."
        ),
    },
    "eda": {
        "icon": "📈",
        "label": "EDA",
        "title": "📈 Exploratory Data Analysis",
        "subtitle": "Generate summary statistics, distributions, and correlation insights.",
        "description": (
            "Exploratory Data Analysis (EDA) is the process of visually and statistically "
            "examining your data to uncover patterns, spot anomalies, and form hypotheses — "
            "before any modeling begins."
        ),
        "objectives": [
            "Compute summary statistics (mean, median, std, quartiles) for numerical features.",
            "Visualize distributions with histograms and box plots.",
            "Discover relationships with scatter plots and correlation heatmaps.",
            "Assess data quality by analysing missing values.",
        ],
        "help_text": (
            "Load a dataset in the **Dataset Explorer** first, then return here to explore it."
        ),
    },
    "preprocessing": {
        "icon": "🧹",
        "label": "Data Preprocessing",
        "title": "🧹 Data Preprocessing",
        "subtitle": "Clean, transform, and prepare data for modelling.",
        "description": (
            "Real-world data is rarely ready for modelling. Preprocessing handles missing "
            "values, encodes categorical variables, scales numerical features, and splits "
            "your data into training and test sets."
        ),
        "objectives": [
            "Identify and impute or remove missing values.",
            "Encode categorical variables (one-hot, label, ordinal encoding).",
            "Scale and normalise numerical features.",
            "Split data into train / test sets with appropriate ratios.",
        ],
        "help_text": (
            "Load and explore a dataset first, then use this module to clean and prepare it."
        ),
    },
    "feature_engineering": {
        "icon": "⚙️",
        "label": "Feature Engineering",
        "title": "⚙️ Feature Engineering",
        "subtitle": "Create, select, and transform features for better models.",
        "description": (
            "Feature engineering transforms raw data into informative inputs that help "
            "models learn more effectively. Good features can outperform complex algorithms."
        ),
        "objectives": [
            "Create new features through polynomial, interaction, or binning techniques.",
            "Select the most informative features using statistical methods.",
            "Reduce dimensionality with PCA or t-SNE for visualisation and modelling.",
            "Rank features by importance to guide model selection.",
        ],
        "help_text": (
            "Preprocess your data first, then use this module to engineer better features."
        ),
    },
    "classification": {
        "icon": "🎯",
        "label": "Classification",
        "title": "🎯 Classification",
        "subtitle": "Train and evaluate models that predict categorical labels.",
        "description": (
            "Classification is a supervised learning task where the goal is to predict a "
            "discrete class label. This module lets you train, tune, and evaluate multiple "
            "classification algorithms interactively."
        ),
        "objectives": [
            "Train classifiers such as Logistic Regression, Decision Trees, and Random Forests.",
            "Tune hyperparameters with interactive controls.",
            "Evaluate models using confusion matrices, precision, recall, and F1 score.",
            "Plot ROC curves and compute AUC for model comparison.",
        ],
        "help_text": (
            "Preprocess a labelled dataset first, then return here to train classifiers."
        ),
    },
    "regression": {
        "icon": "📐",
        "label": "Regression",
        "title": "📐 Regression",
        "subtitle": "Train and evaluate models that predict continuous values.",
        "description": (
            "Regression is a supervised learning task that predicts a continuous numeric "
            "outcome. This module provides interactive tools to train, visualise, and "
            "evaluate regression models."
        ),
        "objectives": [
            "Train regressors such as Linear Regression, Ridge, Lasso, and Random Forest.",
            "Analyse residuals to diagnose model fit.",
            "Compare predicted vs actual values with scatter plots.",
            "Evaluate models using R², MAE, MSE, and RMSE metrics.",
        ],
        "help_text": (
            "Preprocess a labelled dataset first, then return here to train regressors."
        ),
    },
    "model_evaluation": {
        "icon": "✅",
        "label": "Model Evaluation",
        "title": "✅ Model Evaluation",
        "subtitle": "Assess model performance with robust evaluation techniques.",
        "description": (
            "Model evaluation goes beyond simple accuracy. This module teaches cross-validation, "
            "learning curves, and bias-variance analysis to build a deep understanding of how "
            "well a model generalises to unseen data."
        ),
        "objectives": [
            "Perform k-fold cross-validation to get reliable performance estimates.",
            "Plot learning curves to diagnose under- and over-fitting.",
            "Analyse the bias-variance tradeoff of your models.",
            "Interpret feature importance and model decisions.",
        ],
        "help_text": (
            "Train at least one model in Classification or Regression, then evaluate it here."
        ),
    },
    "clustering": {
        "icon": "🔮",
        "label": "Clustering",
        "title": "🔮 Clustering",
        "subtitle": "Discover natural groupings in unlabeled data.",
        "description": (
            "Clustering is an unsupervised learning technique that groups similar data points "
            "together without labels. It is useful for customer segmentation, anomaly detection, "
            "and exploratory pattern discovery."
        ),
        "objectives": [
            "Apply K-Means, DBSCAN, and Hierarchical clustering algorithms.",
            "Determine the optimal number of clusters with Elbow and Silhouette methods.",
            "Visualise clusters in 2D and 3D projections.",
            "Profile and characterise each discovered cluster.",
        ],
        "help_text": (
            "Load and preprocess an unlabeled dataset, then return here to discover clusters."
        ),
    },
    "model_comparison": {
        "icon": "⚖️",
        "label": "Model Comparison",
        "title": "⚖️ Model Comparison",
        "subtitle": "Benchmark and compare multiple models side by side.",
        "description": (
            "No single model is best for every problem. This module lets you compare multiple "
            "algorithms on the same dataset, using consistent metrics, to select the model "
            "that best fits your requirements."
        ),
        "objectives": [
            "Compare accuracy, precision, recall, F1, R², and other metrics across models.",
            "Visualise comparisons with bar charts and radar plots.",
            "Rank models in a sortable table to find the best performer.",
            "Export comparison reports for documentation.",
        ],
        "help_text": (
            "Train multiple models in Classification or Regression first, then compare them here."
        ),
    },
    "automl": {
        "icon": "🤖",
        "label": "AutoML",
        "title": "🤖 AutoML",
        "subtitle": "Automate model selection, tuning, and pipeline construction.",
        "description": (
            "AutoML (Automated Machine Learning) automates repetitive tasks in the ML pipeline: "
            "algorithm selection, hyperparameter tuning, and feature preprocessing. It helps "
            "you quickly establish a strong baseline model."
        ),
        "objectives": [
            "Understand how AutoML selects and ranks algorithms automatically.",
            "Explore automated hyperparameter optimisation strategies.",
            "Generate complete ML pipelines from raw data to predictions.",
            "Benchmark automated results against manually tuned models.",
        ],
        "help_text": (
            "Load a dataset and specify the task type (classification or regression) to begin."
        ),
    },
}

# Ordered list used by the sidebar and pipeline visualisation.
MODULE_ORDER: list[str] = [
    "dataset_explorer",
    "eda",
    "preprocessing",
    "feature_engineering",
    "classification",
    "regression",
    "model_evaluation",
    "clustering",
    "model_comparison",
    "automl",
]

# Page file paths (relative to project root) for st.page_link.
_PAGE_PATHS: dict[str, str] = {
    "dataset_explorer": "pages/1_📂_Dataset_Explorer.py",
    "eda": "pages/2_📈_EDA.py",
    "preprocessing": "pages/3_🧹_Data_Preprocessing.py",
    "feature_engineering": "pages/4_⚙️_Feature_Engineering.py",
    "classification": "pages/5_🎯_Classification.py",
    "regression": "pages/6_📐_Regression.py",
    "model_evaluation": "pages/7_✅_Model_Evaluation.py",
    "clustering": "pages/8_🔮_Clustering.py",
    "model_comparison": "pages/9_⚖️_Model_Comparison.py",
    "automl": "pages/10_🤖_AutoML.py",
    "learning_mode": "pages/11_📚_Learning_Mode.py",
    "practice_mode": "pages/12_🧪_Practice_Mode.py",
}

# ── Sidebar ─────────────────────────────────────────────────────────


def build_sidebar() -> None:
    """Render the shared sidebar with logo, navigation links, and footer."""
    with st.sidebar:
        st.markdown("## 🔬 Data Science Lab")
        st.caption("Interactive Data Science Learning Platform")
        st.markdown("---")

        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.markdown("#### 📊 Modules")

        for key in MODULE_ORDER:
            mod = MODULES[key]
            st.page_link(
                _PAGE_PATHS[key],
                label=mod["label"],
                icon=mod["icon"],
            )

        st.markdown("---")
        st.markdown(
            "**Pipeline order**  \n"
            "1. Load data  \n"
            "2. Explore  \n"
            "3. Preprocess  \n"
            "4. Engineer features  \n"
            "5. Train & evaluate  \n"
            "6. Compare & deploy"
        )
        st.markdown("---")
        st.markdown("#### 🎓 Study")
        st.page_link(
            "pages/13_📖_Curriculum.py",
            label="Curriculum",
            icon="📖",
        )
        st.page_link(
            "pages/11_📚_Learning_Mode.py",
            label="Learning Mode",
            icon="📚",
        )
        st.page_link(
            "pages/12_🧪_Practice_Mode.py",
            label="Practice Mode",
            icon="🧪",
        )
        st.caption("Built with ❤️ for Data Science students")
        st.caption("v1.0.0")
        st.markdown("---")
        st.markdown(
            "**👨‍🏫 Course Instructor**  \n"
            "Engr. Dr. Muhammad Nadeem Majeed  \n"
            "Professor  \n"
            "Department of Data Science  \n"
            "Faculty of Computing and IT  \n"
            "University of the Punjab, Lahore"
        )


# ── Page header ─────────────────────────────────────────────────────


def page_header(module_key: str) -> None:
    """
    Render a consistent page header: title, subtitle, and
    educational description with learning objectives.

    Call this at the top of every module page, right after
    ``st.set_page_config``.
    """
    mod = MODULES[module_key]

    st.title(mod["title"])
    st.caption(mod["subtitle"])

    with st.expander("📖 About this module", expanded=False):
        st.markdown(mod["description"])
        st.markdown("**Learning Objectives**")
        for obj in mod["objectives"]:
            st.markdown(f"- {obj}")

    st.info(mod["help_text"])
    st.markdown("---")


# ── Utility cards ───────────────────────────────────────────────────


def module_card(module_key: str) -> None:
    """
    Render a compact card for a single module on the home page.
    """
    mod = MODULES[module_key]
    page_path = _PAGE_PATHS[module_key]
    st.markdown(
        f"**{mod['icon']} {mod['label']}**  \n"
        f"{mod['subtitle']}"
    )
    st.page_link(page_path, label=f"Open {mod['label']}", icon="➡️")


def metric_card(label: str, value: str | int | float, help_text: str = "") -> None:
    """
    Display a single metric in a compact container.
    """
    st.metric(label=label, value=value, help=help_text or None)


def pipeline_visualisation(active: str | None = None) -> None:
    """
    Render a simple horizontal pipeline showing all module steps.
    The *active* step (if any) is highlighted with bold text.
    """
    icons = [MODULES[k]["icon"] for k in MODULE_ORDER]
    labels = [MODULES[k]["label"].split()[0] for k in MODULE_ORDER]

    parts: list[str] = []
    for icon, label, key in zip(icons, labels, MODULE_ORDER):
        if key == active:
            parts.append(f"**{icon} {label}**")
        else:
            parts.append(f"{icon} {label}")

    st.markdown(" → ".join(parts))
