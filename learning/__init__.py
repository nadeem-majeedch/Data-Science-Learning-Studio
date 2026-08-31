"""
Student Learning Curriculum for Data Science Lab.

Structured educational content for BS Data Science students,
organised by topic area with quizzes, exercises, and progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuizQuestion:
    """A single quiz question."""
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    question_type: str = "multiple_choice"  # multiple_choice | true_false | scenario


@dataclass
class Topic:
    """A single learning topic with full educational content."""
    id: str
    title: str
    section: str  # which curriculum section
    order: int  # ordering within section

    # Learning Objectives
    objectives: list[str] = field(default_factory=list)

    # Concept explanation
    concept: str = ""

    # Why it matters
    why_matters: str = ""

    # Simple explanation
    simple_explanation: str = ""

    # Examples (markdown)
    example: str = ""

    # Python/Pandas example (code block)
    python_example: str = ""

    # What the result means
    interpretation: str = ""

    # Common mistakes
    common_mistakes: list[str] = field(default_factory=list)

    # Practice exercise
    practice_exercise: str = ""

    # Quiz questions for this topic
    quiz: list[QuizQuestion] = field(default_factory=list)

    # Key takeaways
    takeaways: list[str] = field(default_factory=list)

    # Link to lab module ("Try this in the Lab →")
    lab_module: str = ""  # module key from utils/ui.py

    # Difficulty: beginner | intermediate | advanced
    difficulty: str = "beginner"

    # Prerequisites (topic IDs)
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class Section:
    """A curriculum section containing multiple topics."""
    id: str
    title: str
    icon: str
    description: str
    order: int
    topics: list[Topic] = field(default_factory=list)


# Registry of all sections — populated by section modules
SECTIONS: dict[str, Section] = {}
