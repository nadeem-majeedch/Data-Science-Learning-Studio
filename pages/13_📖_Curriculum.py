"""
📖 Curriculum — Complete BS Data Science Learning System.

Provides structured learning paths, quizzes, and practice exercises
for every topic in the data science pipeline.
"""

import streamlit as st

from utils.content_loader import (
    get_all_sections,
    get_section,
    get_topic_by_id,
    get_total_topics,
    search_topics,
)
from learning.exercises import get_exercises_for_section, get_total_exercises
from utils.ui_components import (
    render_difficulty_badge,
    render_exercise,
    render_progress_bar,
    render_quiz,
    render_section_header,
    render_topic_card,
)
from utils.ui import build_sidebar

st.set_page_config(page_title="Curriculum", page_icon="📖", layout="wide")
build_sidebar()

# ── Load curriculum FIRST (needed for stats) ────────────────────────
sections = get_all_sections()
total_topics = get_total_topics()
total_exercises = get_total_exercises()

# ── Header ──────────────────────────────────────────────────────────
st.title("📖 Data Science Curriculum")
st.caption("Complete learning path for BS Data Science students")
st.info(
    f"**{total_topics} learning topics** with concepts, examples, common mistakes, quizzes, and lab links. "
    f"**{total_exercises} practice exercises** across all sections. Progress through each section sequentially."
)
st.markdown("---")

# Session state for progress tracking
if "curriculum_completed" not in st.session_state:
    st.session_state["curriculum_completed"] = set()

# ── Progress overview ───────────────────────────────────────────────
completed = len(st.session_state["curriculum_completed"])
st.markdown("### 📊 Your Progress")
render_progress_bar(completed, total_topics, "Overall")

col1, col2, col3 = st.columns(3)
col1.metric("Total Topics", total_topics)
col2.metric("Completed", completed)
col3.metric("Remaining", total_topics - completed)
st.markdown("---")

# ── Section navigation ──────────────────────────────────────────────
section_titles = [f"{s['icon']} {s['title']} ({s['topic_count']} topics)" for s in sections]
selected_section_idx = st.selectbox(
    "Select a section to study:",
    range(len(sections)),
    format_func=lambda i: section_titles[i],
)

section = sections[selected_section_idx]
render_section_header(section)

# ── Render topics ───────────────────────────────────────────────────
for topic in section["topics"]:
    col_content, col_action = st.columns([5, 1])

    with col_content:
        render_topic_card(topic, expanded=False)

    with col_action:
        topic_key = f"done_{topic.id}"
        if topic.id in st.session_state["curriculum_completed"]:
            st.success("✅ Done")
        else:
            if st.button("✅ Mark Complete", key=f"mark_{topic.id}"):
                st.session_state["curriculum_completed"].add(topic.id)
                st.rerun()

    # Quiz for this topic
    if topic.quiz:
        render_quiz(topic.quiz, topic.id)

# ── Section quiz summary ────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🎯 Section Complete?")
section_topics = section["topics"]
section_completed = sum(1 for t in section_topics if t.id in st.session_state["curriculum_completed"])
render_progress_bar(section_completed, len(section_topics), f"{section['title']} Progress")

if section_completed == len(section_topics) and len(section_topics) > 0:
    st.success(f"🎉 You've completed all {len(section_topics)} topics in **{section['title']}**!")
elif section_completed > 0:
    remaining = len(section_topics) - section_completed
    st.info(f"📚 {remaining} topic{'s' if remaining != 1 else ''} remaining in this section.")

# ── Practice Exercises ──────────────────────────────────────────────
exercises = get_exercises_for_section(section["id"])
if exercises:
    st.markdown("---")
    st.markdown(f"### 🛠️ Practice Exercises — {section['title']}")
    st.caption("Apply what you learned. Each exercise has beginner, intermediate, and advanced levels.")
    for exercise in exercises:
        render_exercise(exercise)

# ── Study tips ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🎓 How to Use This Curriculum", expanded=False):
    st.markdown("""
    ### Learning Progression

    1. **Read the concept** — understand what it is and why it matters.
    2. **Study the example** — see how it works in practice.
    3. **Try the Python code** — copy it into a notebook and experiment.
    4. **Check common mistakes** — avoid the pitfalls.
    5. **Take the quiz** — test your understanding.
    6. **Mark complete** — track your progress.
    7. **Try in the Lab** — apply what you learned in the interactive modules.

    ### Tips
    - **Don't rush** — understanding matters more than speed.
    - **Revisit topics** — learning is iterative, not linear.
    - **Use the lab links** — practice makes permanent.
    - **Teach someone else** — the best way to learn is to explain.
    """)
