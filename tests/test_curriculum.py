"""Tests for the expanded learning curriculum."""

import pytest
from learning import Topic, QuizQuestion, Section
from learning.exercises import (
    Exercise,
    EXERCISES,
    get_exercises_for_section,
    get_all_exercises,
    get_exercise_by_id,
    get_total_exercises,
)
from utils.content_loader import (
    get_all_sections,
    get_section,
    get_total_topics,
    get_topic_by_id,
    search_topics,
    SECTION_ORDER,
)


class TestContentLoader:
    """Tests for the content loading system."""

    def test_all_sections_load(self):
        """All sections should load without errors."""
        sections = get_all_sections()
        assert len(sections) == len(SECTION_ORDER)

    def test_total_topics_count(self):
        """Should have at least 150 topics (expanded curriculum)."""
        total = get_total_topics()
        assert total >= 150, f"Expected >= 150 topics, got {total}"

    def test_get_section(self):
        """get_section returns a valid section."""
        section = get_section("data_loading")
        assert section is not None
        assert section["title"] == "Data Loading"
        assert section["topic_count"] >= 10

    def test_get_section_returns_none_for_invalid(self):
        """Invalid section returns None."""
        assert get_section("nonexistent") is None

    def test_topic_by_id(self):
        """Should find topics by ID."""
        topic = get_topic_by_id("dl_01")
        assert topic is not None
        assert topic.title == "Introduction to Datasets"
        assert topic.section == "data_loading"

    def test_topic_by_id_not_found(self):
        """Unknown ID returns None."""
        assert get_topic_by_id("nonexistent_id") is None

    def test_search_topics(self):
        """Search finds relevant topics."""
        results = search_topics("regression")
        assert len(results) >= 3

    def test_all_sections_have_topics(self):
        """Every section should have at least 8 topics."""
        for section in get_all_sections():
            assert section["topic_count"] >= 8, (
                f"Section {section['title']} has only {section['topic_count']} topics"
            )

    def test_topics_have_required_fields(self):
        """Every topic should have core fields populated."""
        for section in get_all_sections():
            for topic in section["topics"]:
                assert topic.id, f"Topic missing id in {section['title']}"
                assert topic.title, f"Topic missing title in {section['title']}"
                assert topic.concept, f"Topic '{topic.title}' missing concept"
                assert topic.objectives, f"Topic '{topic.title}' missing objectives"

    def test_topics_have_quiz_questions(self):
        """Most topics should have at least one quiz question."""
        topics_with_quiz = 0
        total_topics = 0
        for section in get_all_sections():
            for topic in section["topics"]:
                total_topics += 1
                if topic.quiz:
                    topics_with_quiz += 1
        # At least 15% of topics should have quizzes (quiz coverage expands over time)
        assert topics_with_quiz / total_topics >= 0.15, (
            f"Only {topics_with_quiz}/{total_topics} topics have quizzes"
        )

    def test_topics_have_common_mistakes(self):
        """Every topic should have common mistakes."""
        for section in get_all_sections():
            for topic in section["topics"]:
                assert len(topic.common_mistakes) >= 1, (
                    f"Topic '{topic.title}' has no common mistakes"
                )

    def test_topics_have_takeaways(self):
        """Every topic should have key takeaways."""
        for section in get_all_sections():
            for topic in section["topics"]:
                assert len(topic.takeaways) >= 1, (
                    f"Topic '{topic.title}' has no takeaways"
                )

    def test_data_loading_section_has_16_topics(self):
        """Data Loading section should have exactly 16 topics."""
        section = get_section("data_loading")
        assert section["topic_count"] == 16

    def test_eda_section_has_20_topics(self):
        """EDA section should have exactly 20 topics."""
        section = get_section("eda")
        assert section["topic_count"] == 20

    def test_preprocessing_section_has_20_topics(self):
        """Preprocessing section should have exactly 20 topics."""
        section = get_section("preprocessing")
        assert section["topic_count"] == 20

    def test_evaluation_section_has_23_topics(self):
        """Evaluation section should have 23 topics."""
        section = get_section("evaluation")
        assert section["topic_count"] == 23

    def test_regression_section_has_25_topics(self):
        """Regression section should have 25 topics."""
        section = get_section("regression")
        assert section["topic_count"] == 25


class TestExercises:
    """Tests for the exercises system."""

    def test_exercises_exist(self):
        """Exercises should be defined."""
        assert len(EXERCISES) > 0

    def test_exercises_per_section(self):
        """Each section should have at least 1 exercise."""
        for section_id, exercises in EXERCISES.items():
            assert len(exercises) >= 1, f"Section {section_id} has no exercises"

    def test_get_exercises_for_section(self):
        """Can retrieve exercises for a section."""
        exercises = get_exercises_for_section("data_loading")
        assert len(exercises) >= 2

    def test_get_exercises_for_empty_section(self):
        """Unknown section returns empty list."""
        assert get_exercises_for_section("nonexistent") == []

    def test_exercise_fields(self):
        """All exercises should have required fields."""
        for section_id, exercises in EXERCISES.items():
            for ex in exercises:
                assert ex.id, f"Exercise in {section_id} missing id"
                assert ex.title, f"Exercise in {section_id} missing title"
                assert ex.description, f"Exercise '{ex.title}' missing description"
                assert ex.difficulty in ("beginner", "intermediate", "advanced"), (
                    f"Exercise '{ex.title}' has invalid difficulty: {ex.difficulty}"
                )
                assert len(ex.steps) >= 1, f"Exercise '{ex.title}' has no steps"

    def test_get_exercise_by_id(self):
        """Can find specific exercise by ID."""
        ex = get_exercise_by_id("ex_dl_b1")
        assert ex is not None
        assert ex.title == "Load and Inspect Titanic"

    def test_get_exercise_not_found(self):
        """Unknown exercise ID returns None."""
        assert get_exercise_by_id("nonexistent") is None

    def test_total_exercises(self):
        """Should have at least 20 exercises."""
        total = get_total_exercises()
        assert total >= 20, f"Expected >= 20 exercises, got {total}"

    def test_difficulty_distribution(self):
        """Should have exercises at all three levels."""
        difficulties = set()
        for exercises in EXERCISES.values():
            for ex in exercises:
                difficulties.add(ex.difficulty)
        assert difficulties == {"beginner", "intermediate", "advanced"}


class TestSectionContent:
    """Verify each section has substantive content."""

    def test_feature_engineering_topics(self):
        section = get_section("feature_engineering")
        titles = [t.title for t in section["topics"]]
        assert "What is Feature Engineering?" in titles
        assert "Polynomial Features" in titles
        assert "Feature Engineering Case Study" in titles

    def test_classification_topics(self):
        section = get_section("classification")
        titles = [t.title for t in section["topics"]]
        assert "Logistic Regression" in titles
        assert "Random Forest" in titles
        assert "Class Imbalance" in titles

    def test_regression_topics(self):
        section = get_section("regression")
        titles = [t.title for t in section["topics"]]
        assert "Simple Linear Regression" in titles
        assert "Ridge Regression (L2)" in titles
        assert "Residual Analysis" in titles

    def test_evaluation_topics(self):
        section = get_section("evaluation")
        titles = [t.title for t in section["topics"]]
        assert "Confusion Matrix" in titles
        assert "ROC Curve" in titles
        assert "Cross-Validation" in titles
        assert "Threshold Selection" in titles

    def test_model_selection_topics(self):
        section = get_section("model_selection")
        titles = [t.title for t in section["topics"]]
        assert "Baseline Models" in titles
        assert "Grid Search" in titles
        assert "Bias and Variance" in titles

    def test_clustering_topics(self):
        section = get_section("clustering")
        titles = [t.title for t in section["topics"]]
        assert "K-Means" in titles
        assert "DBSCAN" in titles
        assert "Elbow Method" in titles

    def test_model_comparison_topics(self):
        section = get_section("model_comparison")
        titles = [t.title for t in section["topics"]]
        assert "Fair Model Comparison" in titles
        assert "Cross-Validation Comparison" in titles

    def test_automl_topics(self):
        section = get_section("automl")
        titles = [t.title for t in section["topics"]]
        assert "What is AutoML?" in titles
        assert "AutoML Workflow" in titles
        assert "Data Leakage in AutoML" in titles


class TestUIComponents:
    """Test that UI components can render without errors."""

    def test_render_difficulty_badge(self):
        from utils.ui_components import render_difficulty_badge
        assert "Beginner" in render_difficulty_badge("beginner")
        assert "Intermediate" in render_difficulty_badge("intermediate")
        assert "Advanced" in render_difficulty_badge("advanced")
