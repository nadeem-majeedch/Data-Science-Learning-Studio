#  Data Science Learning Studio

An interactive, open-source learning platform built with **Python** and **Streamlit** for BS Data Science students to explore the full data science pipeline — from raw data to deployed models.

---

## 🎯 Purpose

Data Science Lab provides a hands-on, browser-based environment where students can learn and practice every stage of the data science workflow through an intuitive graphical interface. No command-line expertise required.

---

## ✨ Features

| Module | Description |
|---|---|
| **📂 Dataset Explorer** | Upload CSVs, load sample datasets, preview data |
| **📈 EDA** | Summary statistics, distributions, correlations |
| **🧹 Data Preprocessing** | Handle missing values, encode, scale, split |
| **⚙️ Feature Engineering** | Create, select, and transform features |
| **🎯 Classification** | Train and evaluate classification models |
| **📐 Regression** | Train and evaluate regression models |
| **✅ Model Evaluation** | Cross-validation, learning curves, metrics |
| **🔮 Clustering** | Unsupervised pattern discovery |
| **⚖️ Model Comparison** | Side-by-side model benchmarking |
| **🤖 AutoML** | Automated model selection and tuning |

### Cross-Cutting Features

| Feature | Description |
|---|---|
| **🐍 Python Code Generator** | Auto-generates complete, runnable Python scripts for every operation |
| **📓 Experiment Tracker** | Save, view, compare, delete, and export ML experiments via local SQLite |
| **📚 Learning Mode** | Structured educational content: What, Why, When, Examples, Mistakes, Interpretation, Think About It, Code |
| **🧪 Practice Mode** | Interactive challenges with scenario-based questions, answer checking, and detailed feedback |

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/nadeem-majeedch/Data-Science-Learning-Studio.git
cd Data-Science-Learning-Studio

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

For a faster getting-started experience, see **[QUICKSTART.md](QUICKSTART.md)**.

---

## ▶️ Local Execution

```bash
streamlit run app.py
```

The application will open in your browser at **http://localhost:8501**.

---

## 📁 Project Structure

```
Data-Science-Learning-Studio/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── app.py                 # Main application & landing page
├── pages/                 # Streamlit multi-page modules
│   ├── 1_📂_Dataset_Explorer.py
│   ├── 2_📈_EDA.py
│   ├── 3_🧹_Data_Preprocessing.py
│   ├── 4_⚙️_Feature_Engineering.py
│   ├── 5_🎯_Classification.py
│   ├── 6_📐_Regression.py
│   ├── 7_✅_Model_Evaluation.py
│   ├── 8_🔮_Clustering.py
│   ├── 9_⚖️_Model_Comparison.py
│   ├── 10_🤖_AutoML.py
│   ├── 11_📚_Learning_Mode.py
│   ├── 12_🧪_Practice_Mode.py
│   └── 13_📖_Curriculum.py
├── learning/              # Structured curriculum content (219 topics)
│   ├── data_loading.py         # 16 topics
│   ├── eda.py                  # 20 topics
│   ├── preprocessing.py        # 20 topics
│   ├── feature_engineering.py  # 22 topics
│   ├── classification.py       # 20 topics
│   ├── regression.py           # 25 topics
│   ├── evaluation.py           # 26 topics
│   ├── model_selection.py      # 22 topics
│   ├── clustering.py           # 22 topics
│   ├── model_comparison.py     # 14 topics
│   ├── automl.py               # 12 topics
│   └── exercises.py            # 30 practice exercises
├── utils/                 # Shared utility functions
│   ├── education.py           # Educational content library
│   ├── practice.py            # Practice challenges with feedback
│   ├── content_loader.py      # Curriculum content loader
│   ├── ui_components.py       # Reusable learning UI components
│   ├── code_generator.py      # Centralised code generation
│   ├── experiment_tracker.py  # SQLite-backed experiment log
│   ├── clustering.py          # Clustering algorithms
│   ├── evaluation.py          # Model evaluation metrics
│   ├── feature_engineering.py # Feature transformations
│   ├── model_training.py      # Classification training
│   ├── models.py              # Classifier registry
│   ├── preprocessing.py       # Data preprocessing
│   ├── regression_models.py   # Regressor registry
│   ├── regression_training.py # Regression training
│   ├── data_analysis.py       # Data quality analysis
│   ├── data_loader.py         # File loading utilities
│   └── visualization.py       # Plotly chart functions
├── QUICKSTART.md          # Quick start guide for students
├── datasets/              # Sample datasets
│   ├── iris.csv               # 150 rows, 3-class classification
│   ├── breast_cancer.csv      # 569 rows, binary classification
│   ├── titanic.csv            # 50 rows, survival prediction
│   ├── wine_quality.csv       # 50 rows, regression
│   └── california_housing.csv # 20k rows, regression
├── notebooks/             # Jupyter notebooks for reference
├── reports/               # Generated reports and plots
└── tests/                 # Test suite
```

---

## 👥 Intended Audience

- **BS Data Science students** learning the end-to-end data science pipeline
- **Instructors** looking for a visual tool to demonstrate ML concepts
- **Self-learners** who prefer an interactive interface over code-only workflows

---

## 🧪 Testing

```bash
# Run the full test suite (410+ tests)
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_clustering.py -v
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is open source under the [MIT License](LICENSE).
