"""
Educational content library for Data Science Lab.

Provides structured, academically-focused content for every major
data science topic.  Each topic entry includes:

- What is it?
- Why is it important?
- When should it be used?
- Simple example
- Common mistakes
- Interpretation of results
- "Think About It" question
- Link to the underlying Python/sklearn code

Used by the Learning Mode and Practice Mode pages.
"""

from __future__ import annotations

from utils.education.base import TopicContent, T

# Import all topic modules
from utils.education.data_loading import TOPICS as _DL_TOPICS
from utils.education.preprocessing import TOPICS as _PP_TOPICS
from utils.education.classification import TOPICS as _CL_TOPICS
from utils.education.regression import TOPICS as _RG_TOPICS
from utils.education.evaluation import TOPICS as _EV_TOPICS
from utils.education.clustering import TOPICS as _CU_TOPICS
from utils.education.feature_engineering import TOPICS as _FE_TOPICS
from utils.education.model_selection import TOPICS as _MS_TOPICS

# Merge all into one TOPICS dict
TOPICS: dict[str, TopicContent] = {}
TOPICS.update(_DL_TOPICS)
TOPICS.update(_PP_TOPICS)
TOPICS.update(_CL_TOPICS)
TOPICS.update(_RG_TOPICS)
TOPICS.update(_EV_TOPICS)
TOPICS.update(_CU_TOPICS)
TOPICS.update(_FE_TOPICS)
TOPICS.update(_MS_TOPICS)


# ── Public API ──────────────────────────────────────────────────────

def get_topic(topic_key: str) -> TopicContent | None:
    """Return educational content for a topic, or None if not found."""
    return TOPICS.get(topic_key)


def get_topics_by_module(module: str) -> list[TopicContent]:
    """Return all topics belonging to a module, in order."""
    return [t for t in TOPICS.values() if t.module == module]


def get_topics_by_module_with_keys(module: str) -> list[tuple[str, TopicContent]]:
    """Return (key, topic) pairs for all topics in a module, in order."""
    return [(k, t) for k, t in TOPICS.items() if t.module == module]


def list_topic_keys() -> list[str]:
    """Return all available topic keys."""
    return list(TOPICS.keys())


def render_topic(topic_key: str, expanded: bool = True) -> None:
    """
    Render a single topic's educational content using Streamlit widgets.

    Call this from any module page to display the full educational
    content block (What → Why → When → Example → Mistakes →
    Interpretation → Think About It → Code Link).
    """
    import streamlit as st

    topic = TOPICS.get(topic_key)
    if topic is None:
        return

    with st.expander(f"📚 {topic.title}", expanded=expanded):
        st.markdown("### What is it?")
        st.markdown(topic.what)

        st.markdown("### Why is it important?")
        st.markdown(topic.why)

        st.markdown("### When should it be used?")
        st.markdown(topic.when)

        st.markdown("### Simple example")
        st.markdown(topic.example)

        st.markdown("### Common mistakes")
        for mistake in topic.common_mistakes:
            st.markdown(f"- ❌ {mistake}")

        st.markdown("### How to interpret results")
        st.markdown(topic.interpretation)

        st.markdown("### 💭 Think About It")
        st.info(topic.think_about_it)

        st.markdown("### 🔗 Underlying Python/sklearn code")
        st.code(topic.code_link, language="python")
