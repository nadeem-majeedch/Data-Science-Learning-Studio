# 🚀 Quick Start Guide — Data Science Lab

Get up and running in **under 5 minutes**.

---

## 1. Install

```bash
# Clone
git clone https://github.com/your-username/Data-Science-Lab.git
cd Data-Science-Lab

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Launch

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## 3. Your First Workflow

Follow these steps in order:

### Step 1 — Load Data
Open **📂 Dataset Explorer** in the sidebar. Pick a sample dataset (try **Iris** or **Breast Cancer**) or upload your own CSV.

### Step 2 — Explore
Open **📈 EDA**. Check distributions, correlations, and missing values. Notice patterns before modelling.

### Step 3 — Preprocess
Open **🧹 Data Preprocessing**. Handle missing values, encode any categorical columns, scale features, and split into train/test.

### Step 4 — Train a Model
Open **🎯 Classification** (or **📐 Regression** if your target is continuous). Pick an algorithm and click **Train**.

### Step 5 — Evaluate
Open **✅ Model Evaluation**. Check the confusion matrix, ROC curve, and cross-validation scores.

### Step 6 — Compare
Open **⚖️ Model Comparison**. Train all algorithms on the same data and see which performs best.

### Step 7 — AutoML (Optional)
Open **🤖 AutoML** for an automated walkthrough — it detects the task, validates data, trains multiple models, and ranks them.

---

## Sample Datasets Included

| Dataset | Rows | Type | Target | Good for |
|---------|------|------|--------|----------|
| **Iris** | 150 | Classification | species (3 classes) | First-time users |
| **Breast Cancer** | 569 | Classification | diagnosis (2 classes) | Binary classification |
| **Wine Quality** | 50 | Regression | quality score | Small regression |
| **Titanic** | 50 | Classification | survival (0/1) | Mixed features |
| **California Housing** | 20,640 | Regression | median house value | Large regression |

---

## Module Overview

| # | Module | What you learn |
|---|--------|----------------|
| 1 | 📂 Dataset Explorer | Loading, previewing, data quality |
| 2 | 📈 EDA | Statistics, distributions, correlations |
| 3 | 🧹 Preprocessing | Missing values, encoding, scaling, splits |
| 4 | ⚙️ Feature Engineering | Transforms, binning, interactions, selection |
| 5 | 🎯 Classification | 7 classifiers, confusion matrix, ROC |
| 6 | 📐 Regression | 7 regressors, residuals, actual vs predicted |
| 7 | ✅ Model Evaluation | Cross-validation, metrics, bias-variance |
| 8 | 🔮 Clustering | K-Means, DBSCAN, Agglomerative, PCA |
| 9 | ⚖️ Model Comparison | Side-by-side benchmarking |
| 10 | 🤖 AutoML | Automated model selection & ranking |

---

## Tips for Students

1. **Always explore before modelling** — EDA reveals problems that preprocessing fixes.
2. **Prevent data leakage** — The app handles this, but understand *why* fit() is only on training data.
3. **No free lunch** — No model is universally best. Try several and compare.
4. **Read the educational hints** — Every module includes 📚 expanders with explanations.
5. **Generate code** — Every operation has a "Generated Code" tab for Jupyter notebooks.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| App is slow | Reduce dataset size or number of models in AutoML |
| Blank page | Check terminal for errors; restart the app |

---

## Next Steps

- Try uploading your own dataset in the Dataset Explorer.
- Use Feature Engineering to create new features and see if models improve.
- Save experiments with the Experiment Tracker to build a record of your work.
- Copy generated code into Jupyter notebooks for further analysis.
