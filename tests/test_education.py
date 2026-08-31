"""
Tests for utils/education.py
"""

import pytest

from utils.education import (
    TOPICS,
    TopicContent,
    get_topic,
    get_topics_by_module,
    list_topic_keys,
)


# ── Topic content structure ─────────────────────────────────────────

class TestTopicStructure:
    def test_all_topics_have_required_fields(self):
        for key, topic in TOPICS.items():
            assert topic.title, f"{key} missing title"
            assert topic.what, f"{key} missing what"
            assert topic.why, f"{key} missing why"
            assert topic.when, f"{key} missing when"
            assert topic.example, f"{key} missing example"
            assert topic.common_mistakes, f"{key} missing common_mistakes"
            assert len(topic.common_mistakes) >= 2, f"{key} needs ≥2 mistakes"
            assert topic.interpretation, f"{key} missing interpretation"
            assert topic.think_about_it, f"{key} missing think_about_it"
            assert topic.code_link, f"{key} missing code_link"

    def test_topics_have_valid_modules(self):
        valid_modules = {
            "dataset_explorer", "eda", "preprocessing", "feature_engineering",
            "classification", "regression", "model_evaluation", "clustering",
            "model_comparison", "model_selection", "automl",
        }
        for key, topic in TOPICS.items():
            assert topic.module in valid_modules, f"{key} has invalid module: {topic.module}"

    def test_topics_have_keywords(self):
        for key, topic in TOPICS.items():
            assert topic.keywords, f"{key} missing keywords"
            assert len(topic.keywords) >= 2, f"{key} needs ≥2 keywords"

    def test_examples_contain_code(self):
        """Every topic must have code somewhere — either in the example or code_link."""
        for key, topic in TOPICS.items():
            has_example_code = ("```" in topic.example or ">>>" in topic.example
                                or "print(" in topic.example or "df." in topic.example)
            has_code_link = ("import" in topic.code_link or "from" in topic.code_link
                             or "print(" in topic.code_link or "pd." in topic.code_link
                             or "df." in topic.code_link)
            assert has_example_code or has_code_link, (
                f"{key} must have code in example or code_link"
            )


# ── Lookup functions ────────────────────────────────────────────────

class TestGetTopic:
    def test_existing_topic(self):
        topic = get_topic("missing_data")
        assert topic is not None
        assert "Missing" in topic.title

    def test_nonexistent_topic(self):
        topic = get_topic("nonexistent_topic_xyz")
        assert topic is None

    def test_all_keys_accessible(self):
        for key in TOPICS:
            topic = get_topic(key)
            assert topic is not None, f"get_topic('{key}') returned None"


class TestGetTopicsByModule:
    def test_preprocessing_topics(self):
        topics = get_topics_by_module("preprocessing")
        assert len(topics) >= 3  # missing, encoding, scaling, split
        topic_keys = {t.title for t in topics}
        assert any("Missing" in k for k in topic_keys), f"Expected a missing-values topic, got: {topic_keys}"

    def test_classification_topics(self):
        topics = get_topics_by_module("classification")
        assert len(topics) >= 2

    def test_empty_module(self):
        topics = get_topics_by_module("nonexistent_module")
        assert len(topics) == 0


class TestListTopicKeys:
    def test_returns_all_keys(self):
        keys = list_topic_keys()
        assert len(keys) == len(TOPICS)
        assert set(keys) == set(TOPICS.keys())


# ── Content quality ─────────────────────────────────────────────────

class TestContentQuality:
    def test_explanations_not_empty(self):
        for key, topic in TOPICS.items():
            assert len(topic.what) > 30, f"{key} 'what' too short"
            assert len(topic.why) > 30, f"{key} 'why' too short"

    def test_mistakes_are_strings(self):
        for key, topic in TOPICS.items():
            for mistake in topic.common_mistakes:
                assert isinstance(mistake, str)
                assert len(mistake) > 10

    def test_code_links_contain_imports(self):
        """Code links should reference actual Python libraries or sklearn usage."""
        for key, topic in TOPICS.items():
            has_import = ("import" in topic.code_link or "from" in topic.code_link
                          or "pd." in topic.code_link or "df." in topic.code_link
                          or "model." in topic.code_link or "cross_val_score" in topic.code_link)
            assert has_import, f"{key} code_link should contain imports or usage"

    def test_keywords_are_lowercase(self):
        for key, topic in TOPICS.items():
            for kw in topic.keywords:
                assert kw == kw.lower(), f"{key} keyword '{kw}' should be lowercase"
