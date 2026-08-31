"""
Content loader for the learning curriculum.

Discovers and loads all section modules, provides search and
lookup functions, and manages curriculum state.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from learning import Section, Topic

# All section IDs in curriculum order
SECTION_ORDER = [
    "data_loading", "eda", "preprocessing", "feature_engineering",
    "classification", "regression", "evaluation", "model_selection",
    "clustering", "model_comparison", "automl",
]

SECTION_META = {
    "data_loading": {"icon": "📂", "title": "Data Loading", "description": "Loading, inspecting, and understanding datasets."},
    "eda": {"icon": "📈", "title": "Exploratory Data Analysis", "description": "Statistical and visual examination of data."},
    "preprocessing": {"icon": "🧹", "title": "Data Preprocessing", "description": "Cleaning, encoding, and preparing data for modelling."},
    "feature_engineering": {"icon": "⚙️", "title": "Feature Engineering", "description": "Creating and selecting informative features."},
    "classification": {"icon": "🎯", "title": "Classification", "description": "Predicting discrete class labels."},
    "regression": {"icon": "📐", "title": "Regression", "description": "Predicting continuous numerical values."},
    "evaluation": {"icon": "✅", "title": "Model Evaluation", "description": "Measuring model performance and generalisation."},
    "model_selection": {"icon": "⚖️", "title": "Model Selection", "description": "Choosing the right algorithm and hyperparameters."},
    "clustering": {"icon": "🔮", "title": "Clustering", "description": "Unsupervised pattern discovery."},
    "model_comparison": {"icon": "📊", "title": "Model Comparison", "description": "Benchmarking multiple models side by side."},
    "automl": {"icon": "🤖", "title": "AutoML", "description": "Automated machine learning pipelines."},
}

_cache: dict[str, list[Topic]] = {}


def _load_section(section_id: str) -> list[Topic]:
    """Load topics for a section from its module."""
    if section_id in _cache:
        return _cache[section_id]

    try:
        module = importlib.import_module(f"learning.{section_id}")
        topics = getattr(module, "TOPICS", [])
        _cache[section_id] = topics
        return topics
    except (ImportError, ModuleNotFoundError):
        return []


def get_all_sections() -> list[dict]:
    """Return all sections with their topics, in curriculum order."""
    sections = []
    for sid in SECTION_ORDER:
        topics = _load_section(sid)
        meta = SECTION_META.get(sid, {})
        sections.append({
            "id": sid,
            "icon": meta.get("icon", "📘"),
            "title": meta.get("title", sid.replace("_", " ").title()),
            "description": meta.get("description", ""),
            "topics": sorted(topics, key=lambda t: t.order),
            "topic_count": len(topics),
        })
    return sections


def get_section(section_id: str) -> dict | None:
    """Return a single section with its topics."""
    topics = _load_section(section_id)
    meta = SECTION_META.get(section_id, {})
    if not meta:
        return None
    return {
        "id": section_id,
        "icon": meta.get("icon", "📘"),
        "title": meta.get("title", section_id.replace("_", " ").title()),
        "description": meta.get("description", ""),
        "topics": sorted(topics, key=lambda t: t.order),
        "topic_count": len(topics),
    }


def get_topic_by_id(topic_id: str) -> Topic | None:
    """Find a topic by its ID across all sections."""
    for sid in SECTION_ORDER:
        topics = _load_section(sid)
        for t in topics:
            if t.id == topic_id:
                return t
    return None


def search_topics(query: str) -> list[Topic]:
    """Search topics by title or concept."""
    query_lower = query.lower()
    results = []
    for sid in SECTION_ORDER:
        topics = _load_section(sid)
        for t in topics:
            if (query_lower in t.title.lower() or
                query_lower in t.concept.lower() or
                any(query_lower in kw.lower() for kw in getattr(t, 'keywords', []))):
                results.append(t)
    return results


def get_total_topics() -> int:
    """Return total number of topics across all sections."""
    return sum(len(_load_section(sid)) for sid in SECTION_ORDER)


def get_section_icon(section_id: str) -> str:
    """Return the icon for a section."""
    return SECTION_META.get(section_id, {}).get("icon", "📘")
