"""
Reusable UI components for the learning curriculum.

Provides Streamlit components for rendering topics, quizzes,
exercises, and progress indicators.
"""

from __future__ import annotations

import streamlit as st

from learning import QuizQuestion, Topic


def render_topic_card(topic: Topic, expanded: bool = True) -> None:
    """Render a single learning topic with all sections."""
    with st.expander(f"📖 {topic.title}", expanded=expanded):
        # Learning Objectives
        if topic.objectives:
            st.markdown("**🎯 Learning Objectives**")
            for obj in topic.objectives:
                st.markdown(f"- {obj}")

        # Concept
        if topic.concept:
            st.markdown("**📚 Concept**")
            st.markdown(topic.concept)

        # Why it matters
        if topic.why_matters:
            st.markdown("**💡 Why It Matters**")
            st.markdown(topic.why_matters)

        # Simple explanation
        if topic.simple_explanation:
            st.markdown("**🔤 Simple Explanation**")
            st.markdown(topic.simple_explanation)

        # Example
        if topic.example:
            st.markdown("**📝 Example**")
            st.markdown(topic.example)

        # Python example
        if topic.python_example:
            st.markdown("**🐍 Python Example**")
            st.code(topic.python_example, language="python")

        # Interpretation
        if topic.interpretation:
            st.markdown("**🔍 What the Result Means**")
            st.markdown(topic.interpretation)

        # Common mistakes
        if topic.common_mistakes:
            st.markdown("**⚠️ Common Mistakes**")
            for mistake in topic.common_mistakes:
                st.markdown(f"- ❌ {mistake}")

        # Key takeaways
        if topic.takeaways:
            st.markdown("**✅ Key Takeaways**")
            for takeaway in topic.takeaways:
                st.markdown(f"- {takeaway}")

        # Lab link
        if topic.lab_module:
            MODULE_PAGES = {
                "dataset_explorer": "pages/1_📂_Dataset_Explorer.py",
                "eda": "pages/2_📈_EDA.py",
                "preprocessing": "pages/3_🧹_Data_Preprocessing.py",
                "feature_engineering": "pages/4_⚙️_Feature_Engineering.py",
                "classification": "pages/5_🎯_Classification.py",
                "regression": "pages/6_📐_Regression.py",
                "evaluation": "pages/7_✅_Model_Evaluation.py",
                "clustering": "pages/8_🔮_Clustering.py",
                "model_comparison": "pages/9_⚖️_Model_Comparison.py",
                "automl": "pages/10_🤖_AutoML.py",
            }
            page = MODULE_PAGES.get(topic.lab_module)
            if page:
                st.page_link(page, label=f"🔬 Try this in the Lab →", icon="🔬")


def render_quiz(questions: list[QuizQuestion], topic_id: str) -> None:
    """Render quiz questions with answer checking."""
    if not questions:
        return

    st.markdown("### 🧪 Quiz")

    for i, q in enumerate(questions):
        q_key = f"quiz_{topic_id}_{i}"

        st.markdown(f"**Q{i+1}: {q.question}**")

        # Options
        option_labels = [f"{chr(65+j)}. {opt}" for j, opt in enumerate(q.options)]
        selected = st.radio(
            "Select your answer:",
            option_labels,
            key=f"radio_{q_key}",
            index=None,
        )

        # Check button
        if st.button(f"Check Answer", key=f"btn_{q_key}"):
            if selected is None:
                st.warning("Please select an answer.")
            else:
                selected_idx = option_labels.index(selected)
                if selected_idx == q.correct_index:
                    st.success(f"✅ **Correct!** {q.explanation}")
                else:
                    correct_text = option_labels[q.correct_index]
                    st.error(f"❌ **Incorrect.** The correct answer is **{correct_text}**.\n\n{q.explanation}")

        st.markdown("---")


def render_exercise(exercise) -> None:
    """Render a single practice exercise with steps and hints."""
    diff_badge = render_difficulty_badge(exercise.difficulty)
    with st.expander(f"🛠️ {exercise.title} — {diff_badge}", expanded=False):
        st.markdown(exercise.description)

        if exercise.steps:
            st.markdown("**📋 Steps:**")
            for i, step in enumerate(exercise.steps, 1):
                st.markdown(f"{i}. {step}")

        if exercise.hints:
            with st.expander("💡 Hints", expanded=False):
                for hint in exercise.hints:
                    st.markdown(f"- {hint}")

        if exercise.expected_outcome:
            st.markdown(f"**🎯 Expected Outcome:** {exercise.expected_outcome}")

        if exercise.lab_module:
            MODULE_PAGES = {
                "dataset_explorer": "pages/1_📂_Dataset_Explorer.py",
                "eda": "pages/2_📈_EDA.py",
                "preprocessing": "pages/3_🧹_Data_Preprocessing.py",
                "feature_engineering": "pages/4_⚙️_Feature_Engineering.py",
                "classification": "pages/5_🎯_Classification.py",
                "regression": "pages/6_📐_Regression.py",
                "evaluation": "pages/7_✅_Model_Evaluation.py",
                "clustering": "pages/8_🔮_Clustering.py",
                "model_comparison": "pages/9_⚖️_Model_Comparison.py",
                "automl": "pages/10_🤖_AutoML.py",
            }
            page = MODULE_PAGES.get(exercise.lab_module)
            if page:
                st.page_link(page, label="🔬 Try this in the Lab →", icon="🔬")


def render_progress_bar(completed: int, total: int, label: str = "") -> None:
    """Render a progress bar with label."""
    if total > 0:
        pct = completed / total
        st.progress(pct, text=f"{label}: {completed}/{total} ({pct*100:.0f}%)")


def render_section_header(section: dict) -> None:
    """Render a section header with icon, title, and topic count."""
    st.markdown(f"## {section['icon']} {section['title']}")
    st.caption(f"{section['topic_count']} topics — {section['description']}")


def render_difficulty_badge(difficulty: str) -> str:
    """Return a difficulty badge string."""
    badges = {
        "beginner": "🟢 Beginner",
        "intermediate": "🟡 Intermediate",
        "advanced": "🔴 Advanced",
    }
    return badges.get(difficulty, f"⚪ {difficulty}")
