"""
Base dataclass for educational content and helper function to build topics quickly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicContent:
    """Structured educational content for a single data science topic."""
    title: str
    module: str  # which module page this belongs to
    what: str
    why: str
    when: str
    example: str
    common_mistakes: list[str]
    interpretation: str
    think_about_it: str
    code_link: str  # snippet showing the Python/sklearn code
    keywords: list[str] = field(default_factory=list)


def T(
    title: str,
    module: str,
    what: str,
    why: str,
    when: str,
    example: str,
    mistakes: list[str],
    interpretation: str,
    think_about_it: str,
    code_link: str,
    keywords: list[str] | None = None,
) -> TopicContent:
    """Shorthand factory for TopicContent to reduce boilerplate."""
    return TopicContent(
        title=title,
        module=module,
        what=what,
        why=why,
        when=when,
        example=example,
        common_mistakes=mistakes,
        interpretation=interpretation,
        think_about_it=think_about_it,
        code_link=code_link,
        keywords=keywords or [],
    )
