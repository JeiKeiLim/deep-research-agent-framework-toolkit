"""Output type hints for the critic agent.

This module provides type hints for the critic agent's output,
allowing for better type checking and code completion in IDEs
and OpenAI agents framework.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

from pydantic import BaseModel


class ScoreFactualAccuracy(BaseModel):
    """Type hints for the factual accuracy score.

    This class represents the factual accuracy score structure
    that the critic agent will use to evaluate the answer.
    """

    score: int
    """Score for factual accuracy of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    reasoning: str
    """Reasoning for the factual accuracy score."""


class ScoreCompleteness(BaseModel):
    """Type hints for the completeness score.

    This class represents the completeness score structure
    that the critic agent will use to evaluate the answer.
    """

    score: int
    """Score for completeness of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    reasoning: str
    """Reasoning for the completeness score."""


class ScoreCoherence(BaseModel):
    """Type hints for the coherence score.

    This class represents the coherence score structure
    that the critic agent will use to evaluate the answer.
    """

    score: int
    """Score for coherence of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    reasoning: str
    """Reasoning for the coherence score."""


class ScoreCitations(BaseModel):
    """Type hints for the citations score.

    This class represents the citations score structure
    that the critic agent will use to evaluate the answer.
    """

    score: int
    """Score for citations of the answer.
    1-5 scale, where 1 is poor and 5 is excellent."""

    reasoning: str
    """Reasoning for the citations score."""


class EvaluatorFeedback(BaseModel):
    """Type hints for feedback provided by the critic agent.

    This class represents the feedback structure that the critic agent
    will use to evaluate and provide insights on the synthesizer agent's answer.
    """

    score_factual_accuracy: ScoreFactualAccuracy
    """Score for factual accuracy of the answer."""

    score_completeness: ScoreCompleteness
    """Score for completeness of the answer."""

    score_coherence: ScoreCoherence
    """Score for coherence of the answer."""

    score_citations: ScoreCitations
    """Score for citations of the answer."""
