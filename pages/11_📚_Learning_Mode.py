"""
📚 Learning Mode — Guided educational walkthroughs.

Provides structured explanations for every data science concept
with definitions, examples, common mistakes, and code links.
"""

import streamlit as st

from utils.education import (
    TOPICS,
    TopicContent,
    get_topics_by_module,
    get_topics_by_module_with_keys,
    list_topic_keys,
    render_topic,
)
from utils.ui import build_sidebar

st.set_page_config(page_title="Learning Mode", page_icon="📚", layout="wide")
build_sidebar()

# ── Header ──────────────────────────────────────────────────────────
st.title("📚 Learning Mode")
st.caption("Structured explanations for every data science concept")
st.info(
    "Browse topics by category. Each section covers: **What** it is, **Why** it matters, "
    "**When** to use it, a **simple example**, **common mistakes**, **how to interpret** "
    "results, a **Think About It** question, and the **underlying Python/sklearn code**."
)
st.markdown("---")

# ── Instructor ──────────────────────────────────────────────────────
st.markdown(
    "**Course Instructor**  \n"
    "Engr. Dr. Muhammad Nadeem Majeed  \n"
    "Professor  \n"
    "Department of Data Science  \n"
    "Faculty of Computing and Information Technology  \n"
    "University of the Punjab, Lahore"
)
st.markdown("---")

# ── Module tabs ─────────────────────────────────────────────────────
MODULE_LABELS = {
    "dataset_explorer": "📂 Data Loading",
    "preprocessing": "🧹 Preprocessing",
    "classification": "🎯 Classification",
    "regression": "📐 Regression",
    "model_evaluation": "✅ Evaluation",
    "clustering": "🔮 Clustering",
    "feature_engineering": "⚙️ Feature Eng.",
    "model_selection": "⚖️ Model Selection",
}

module_keys = list(MODULE_LABELS.keys())
module_tabs = st.tabs([MODULE_LABELS[k] for k in module_keys])

for tab, module_key in zip(module_tabs, module_keys):
    with tab:
        topic_pairs = get_topics_by_module_with_keys(module_key)
        if not topic_pairs:
            st.info("No learning topics available for this module yet.")
            continue

        st.markdown(f"### {MODULE_LABELS[module_key]}")
        st.markdown(
            f"**{len(topic_pairs)} topic{'s' if len(topic_pairs) != 1 else ''}** available. "
            "Click each to expand the full educational content."
        )
        st.markdown("")

        for topic_key, topic in topic_pairs:
            render_topic(topic_key, expanded=False)

# ── Quick reference at bottom ───────────────────────────────────────
st.markdown("---")
with st.expander("📋 All Topics — Quick Reference", expanded=False):
    st.markdown("### Complete topic index")
    all_topics = list_topic_keys()
    for i, key in enumerate(all_topics, 1):
        topic = TOPICS[key]
        module_label = MODULE_LABELS.get(topic.module, topic.module)
        st.markdown(f"**{i}. {topic.title}** ({module_label})")
        st.caption(topic.what[:120] + "...")
        st.markdown("")

# ── Study tips ──────────────────────────────────────────────────────
with st.expander("🎓 Study Tips", expanded=False):
    st.markdown("""
    ### How to use Learning Mode effectively

    1. **Read actively** — don't just skim. Pause at "Think About It" questions.
    2. **Try the code** — copy examples into a Jupyter notebook and experiment.
    3. **Make mistakes** — understanding *why* something is wrong teaches more than getting it right.
    4. **Connect concepts** — missing values → imputation → data leakage → cross-validation.
    5. **Practice after learning** — use the 🧪 Practice Mode to test your understanding.

    ### Recommended learning order

    1. Data Loading → Data Types → Missing Values
    2. Categorical Encoding → Feature Scaling → Train/Test Split
    3. Classification Metrics → Confusion Matrix → ROC Curve
    4. Regression Metrics → Overfitting → Cross-Validation
    5. Feature Engineering → Model Selection → Clustering
    """)
