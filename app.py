"""
Data-Science-Learning-Studio — Main Application
Polished landing page and sidebar for the educational platform.
"""

import streamlit as st

from utils.ui import (
    MODULES,
    MODULE_ORDER,
    build_sidebar,
    module_card,
    pipeline_visualisation,
)

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Data Science Learning Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar ──────────────────────────────────────────────────
build_sidebar()

# ── Hero section ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .hero-title { font-size: 4rem !important; font-weight: 700; text-align: center; padding: 0.5rem 0 0.2rem; }
        .hero-sub   { font-size: 1.2rem; text-align: center; color: #888; padding-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">📊 Data Science Learning Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">An interactive, hands-on platform for learning the full data science pipeline.</p>',
    unsafe_allow_html=True,
)

# ── Pipeline overview ───────────────────────────────────────────────
st.markdown("### 🧭 Data Science Pipeline")
pipeline_visualisation()
st.markdown("---")

# ── Module cards ────────────────────────────────────────────────────
st.markdown("### 📚 Modules")

# Row 1
c1, c2, c3 = st.columns(3)
with c1:
    module_card("dataset_explorer")
with c2:
    module_card("eda")
with c3:
    module_card("preprocessing")

# Row 2
c4, c5, c6 = st.columns(3)
with c4:
    module_card("feature_engineering")
with c5:
    module_card("classification")
with c6:
    module_card("regression")

# Row 3
c7, c8, c9, c10 = st.columns(4)
with c7:
    module_card("model_evaluation")
with c8:
    module_card("clustering")
with c9:
    module_card("model_comparison")
with c10:
    module_card("automl")

st.markdown("---")

# ── Getting Started ─────────────────────────────────────────────────
st.markdown("### 🚀 Getting Started")
st.markdown(
    """
    1. **Upload a dataset** in the **Dataset Explorer** or choose a sample.
    2. **Explore** the data with the EDA module — look for distributions, outliers, and correlations.
    3. **Preprocess** — handle missing values, encode categories, scale features.
    4. **Engineer features** to improve signal before modelling.
    5. **Train models** using Classification, Regression, or Clustering.
    6. **Evaluate** with cross-validation and learning curves.
    7. **Compare** models side by side and pick the best one.
    """
)

st.markdown("---")

# ── About section ───────────────────────────────────────────────────
with st.expander("ℹ️ About Data Science Learning Studio", expanded=False):
    st.markdown(
        """
        **Data Science Learning Studio** is an open-source educational tool built with
        [Streamlit](https://streamlit.io/) for BS Data Science students.

        It provides a guided, visual interface for every stage of the data
        science lifecycle — from loading raw CSV files to comparing
        production-ready models — so you can focus on learning concepts
        rather than wrestling with boilerplate code.

        **Key principles:**
        - *Learn by doing* — every module includes explanations and objectives.
        - *Consistent workflow* — the pipeline sidebar keeps you on track.
        - *No setup friction* — runs locally with just Python and Streamlit.
        """
    )

st.markdown("---")
st.markdown(
    "### 👨‍🏫 Course Instructor\n\n"
    "**Engr. Dr. Muhammad Nadeem Majeed**  \n"
    "Professor  \n"
    "Department of Data Science  \n"
    "Faculty of Computing and Information Technology  \n"
    "University of the Punjab, Lahore"
)
st.markdown("---")
st.caption("Data Science Learning Studio • v1.0.0 • Built with Streamlit")
