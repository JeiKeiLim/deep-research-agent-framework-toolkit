"""Output type hints for the critic agent.

This module provides type hints for the critic agent's output,
allowing for better type checking and code completion in IDEs
and OpenAI agents framework.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

from pydantic import BaseModel


class CriticScore(BaseModel):
    """Type hints for the score provided by the critic agent.

    This class represents the score structure that the critic agent
    will use to evaluate the answer.
    """

    factual_accuracy: int
    """Score for factual accuracy of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    completeness: int
    """Score for completeness of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    coherence: int
    """Score for coherence of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    citations: int
    """Score for citations of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    overall: int
    """Overall score for the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""


class CriticFeedback(BaseModel):
    """Type hints for feedback provided by the critic agent.

    This class represents the feedback structure that the critic agent
    will use to evaluate and provide insights on the synthesizer agent's answer.
    """

    issues: list[str]
    """A list of identified issues found."""

    suggestions: list[str]
    """A list of suggestions for improvement."""

    scores: CriticScore
    """Scores for the answer based on various criteria such as
    factual accuracy, completeness, etc."""

    reasoning: str
    """A detailed reasoning for the scores and feedback provided."""

    needs_revision: bool
    """Indicates whether the answer needs revision with more information search.
    True if revision is needed, False otherwise."""
