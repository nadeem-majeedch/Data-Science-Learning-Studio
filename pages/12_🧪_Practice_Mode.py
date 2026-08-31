"""
🧪 Practice Mode — Test your understanding with structured challenges.

Students answer conceptual questions about data science topics
and receive detailed feedback explaining the correct reasoning.
"""

import streamlit as st

from utils.practice import (
    CHALLENGES,
    Challenge,
    check_answer,
    get_challenge,
    get_challenges_by_topic,
    list_challenge_topics,
)
from utils.ui import build_sidebar

st.set_page_config(page_title="Practice Mode", page_icon="🧪", layout="wide")
build_sidebar()

# ── Header ──────────────────────────────────────────────────────────
st.title("🧪 Practice Mode")
st.caption("Test your understanding with structured challenges")
st.info(
    "Each challenge presents a **real scenario**, asks a **conceptual question**, "
    "and gives you **detailed feedback** after you answer. "
    "No grades — just learning."
)
st.markdown("---")

# ── Track progress in session state ─────────────────────────────────
if "practice_results" not in st.session_state:
    st.session_state["practice_results"] = {}

# ── Topic selection ─────────────────────────────────────────────────
st.markdown("### Choose a Topic")

topic_labels = {
    "missing_values": "🧹 Missing Values",
    "feature_scaling": "📐 Feature Scaling",
    "categorical_encoding": "🏷️ Categorical Encoding",
    "train_test_split": "✂️ Train/Test Split",
    "classification_metrics": "🎯 Classification Metrics",
    "confusion_matrix": "📊 Confusion Matrix",
    "roc_curve": "📈 ROC Curve & AUC",
    "regression_metrics": "📐 Regression Metrics",
    "overfitting": "⚠️ Overfitting & Underfitting",
    "cross_validation": "🔄 Cross-Validation",
    "clustering_basics": "🔮 Clustering",
    "feature_engineering_basics": "⚙️ Feature Engineering",
    "model_selection": "⚖️ Model Selection",
}

available_topics = list_challenge_topics()
selected_topic = st.selectbox(
    "Select a topic to practice",
    available_topics,
    format_func=lambda x: topic_labels.get(x, x),
)

# ── Get challenges for this topic ───────────────────────────────────
challenges = get_challenges_by_topic(selected_topic)

if not challenges:
    st.warning("No challenges available for this topic yet.")
    st.stop()

st.markdown(f"### {topic_labels.get(selected_topic, selected_topic)}")
st.markdown(f"**{len(challenges)} challenge{'s' if len(challenges) != 1 else ''}** available")

# ── Render each challenge ───────────────────────────────────────────
for challenge in challenges:
    st.markdown("---")
    st.markdown(f"#### Challenge: {challenge.title}")

    # Difficulty badge
    diff_colors = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
    st.caption(f"{diff_colors.get(challenge.difficulty, '⚪')} {challenge.difficulty.title()}")

    # Scenario
    with st.expander("📋 Scenario", expanded=True):
        st.markdown(challenge.scenario)

    # Question
    st.markdown(f"**{challenge.question}**")

    # Answer options
    challenge_key = f"challenge_{challenge.id}"

    if challenge_key not in st.session_state:
        st.session_state[challenge_key] = None

    option_labels = [f"{opt.label}. {opt.text}" for opt in challenge.options]
    selected = st.radio(
        "Your answer:",
        option_labels,
        key=f"radio_{challenge.id}",
        index=None,
    )

    # Submit button
    if st.button(f"✅ Check Answer", key=f"btn_{challenge.id}", type="primary"):
        if selected is None:
            st.warning("Please select an answer first.")
        else:
            selected_index = option_labels.index(selected)
            result = check_answer(challenge, selected_index)
            st.session_state[challenge_key] = result

    # Show result if answered
    result = st.session_state.get(challenge_key)
    if result is not None:
        if result["correct"]:
            st.success("✅ **Correct!** Well done.")
        else:
            correct = result["correct_option"]
            st.error(
                f"❌ **Not quite.** The correct answer is "
                f"**{correct.label}. {correct.text}**."
            )

        # Explanation
        st.markdown("### 📖 Explanation")
        st.markdown(result["explanation"])

        # Follow-up
        st.markdown("### 💭 Deeper Thinking")
        st.markdown(result["follow_up"])

        # Code hint
        if result["code_hint"]:
            st.markdown("### 🔗 Code Reference")
            st.code(result["code_hint"], language="python")

# ── Progress summary ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Your Progress This Session")

answered = sum(1 for k in st.session_state if k.startswith("challenge_"))
correct = sum(
    1 for k, v in st.session_state.items()
    if k.startswith("challenge_") and isinstance(v, dict) and v.get("correct")
)

col1, col2, col3 = st.columns(3)
col1.metric("Challenges Attempted", answered)
col2.metric("Correct", correct)
col3.metric("Accuracy", f"{correct / answered * 100:.0f}%" if answered > 0 else "N/A")

if answered > 0:
    st.progress(correct / answered, text=f"{correct}/{answered} correct")
