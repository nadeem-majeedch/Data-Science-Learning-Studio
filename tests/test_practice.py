"""
Tests for utils/practice.py
"""

import pytest

from utils.practice import (
    CHALLENGES,
    Challenge,
    ChallengeOption,
    check_answer,
    get_challenge,
    get_challenges_by_topic,
    list_challenge_topics,
)


# ── Challenge structure ─────────────────────────────────────────────

class TestChallengeStructure:
    def test_all_challenges_have_required_fields(self):
        for c in CHALLENGES:
            assert c.id, f"Challenge missing id"
            assert c.topic, f"Challenge {c.id} missing topic"
            assert c.title, f"Challenge {c.id} missing title"
            assert c.scenario, f"Challenge {c.id} missing scenario"
            assert c.question, f"Challenge {c.id} missing question"
            assert c.explanation, f"Challenge {c.id} missing explanation"
            assert c.follow_up, f"Challenge {c.id} missing follow_up"

    def test_all_challenges_have_options(self):
        for c in CHALLENGES:
            assert len(c.options) >= 2, f"Challenge {c.id} needs ≥2 options"
            assert len(c.options) <= 5, f"Challenge {c.id} has too many options"

    def test_correct_index_in_range(self):
        for c in CHALLENGES:
            assert 0 <= c.correct_index < len(c.options), (
                f"Challenge {c.id}: correct_index {c.correct_index} "
                f"out of range for {len(c.options)} options"
            )

    def test_exactly_one_correct_option(self):
        for c in CHALLENGES:
            correct_count = sum(1 for opt in c.options if opt.is_correct)
            assert correct_count == 1, (
                f"Challenge {c.id}: expected exactly 1 correct option, got {correct_count}"
            )

    def test_correct_index_matches_is_correct(self):
        for c in CHALLENGES:
            assert c.options[c.correct_index].is_correct, (
                f"Challenge {c.id}: option at correct_index {c.correct_index} "
                f"is not marked as correct"
            )

    def test_option_labels_unique(self):
        for c in CHALLENGES:
            labels = [opt.label for opt in c.options]
            assert len(labels) == len(set(labels)), (
                f"Challenge {c.id}: duplicate option labels"
            )

    def test_difficulty_valid(self):
        valid = {"beginner", "intermediate", "advanced"}
        for c in CHALLENGES:
            assert c.difficulty in valid, (
                f"Challenge {c.id}: invalid difficulty '{c.difficulty}'"
            )


# ── Answer checking ─────────────────────────────────────────────────

class TestCheckAnswer:
    def test_correct_answer(self):
        c = CHALLENGES[0]
        result = check_answer(c, c.correct_index)
        assert result["correct"] is True

    def test_wrong_answer(self):
        c = CHALLENGES[0]
        wrong_index = (c.correct_index + 1) % len(c.options)
        result = check_answer(c, wrong_index)
        assert result["correct"] is False

    def test_result_has_required_keys(self):
        c = CHALLENGES[0]
        result = check_answer(c, 0)
        assert "correct" in result
        assert "selected" in result
        assert "correct_option" in result
        assert "explanation" in result
        assert "follow_up" in result
        assert "code_hint" in result

    def test_correct_option_is_returned(self):
        c = CHALLENGES[0]
        result = check_answer(c, 0)
        assert isinstance(result["correct_option"], ChallengeOption)
        assert result["correct_option"].is_correct

    def test_explanation_is_string(self):
        for c in CHALLENGES:
            result = check_answer(c, c.correct_index)
            assert isinstance(result["explanation"], str)
            assert len(result["explanation"]) > 20


# ── Public API ──────────────────────────────────────────────────────

class TestGetChallengesByTopic:
    def test_existing_topic(self):
        challenges = get_challenges_by_topic("missing_values")
        assert len(challenges) >= 2
        for c in challenges:
            assert c.topic == "missing_values"

    def test_nonexistent_topic(self):
        challenges = get_challenges_by_topic("nonexistent")
        assert len(challenges) == 0

    def test_all_topics_have_challenges(self):
        for topic in list_challenge_topics():
            challenges = get_challenges_by_topic(topic)
            assert len(challenges) >= 1, f"Topic '{topic}' has no challenges"


class TestGetChallenge:
    def test_existing_challenge(self):
        c = get_challenge("mv_01")
        assert c is not None
        assert c.id == "mv_01"

    def test_nonexistent_challenge(self):
        c = get_challenge("nonexistent_id")
        assert c is None

    def test_all_challenge_ids_accessible(self):
        for c in CHALLENGES:
            found = get_challenge(c.id)
            assert found is not None, f"Challenge {c.id} not found by get_challenge()"


class TestListChallengeTopics:
    def test_returns_unique_topics(self):
        topics = list_challenge_topics()
        assert len(topics) == len(set(topics))

    def test_all_topics_have_challenges(self):
        topics = list_challenge_topics()
        for topic in topics:
            challenges = get_challenges_by_topic(topic)
            assert len(challenges) > 0


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_duplicate_challenge_ids(self):
        ids = [c.id for c in CHALLENGES]
        assert len(ids) == len(set(ids))

    def test_all_topics_covered(self):
        """Every topic with challenges should have at least one challenge."""
        for c in CHALLENGES:
            assert c.topic in list_challenge_topics()

    def test_challenges_across_difficulties(self):
        difficulties = {c.difficulty for c in CHALLENGES}
        assert "beginner" in difficulties
        assert "intermediate" in difficulties or "advanced" in difficulties
