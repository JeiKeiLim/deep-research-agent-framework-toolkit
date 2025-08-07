"""Output type hints for the planner agent.

This module provides type hints for the planner agent's output,
allowing for better type checking and code completion in IDEs
and OpenAI agents framework.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

from pydantic import BaseModel


class SearchItem(BaseModel):
    """Type hints for a search item in the search results."""

    search_term: str
    """The search term to be used in the knowledge base search"""

    reasoning: str
    """A description of the search term and its relevance to the query"""
    
    tool_name: str
    """The name of the tool to use for this search. (e.g., "weaviate", "perplexity")"""


class SearchPlan(BaseModel):
    """Type hints for a search plan in the planner agent."""

    search_steps: list[SearchItem]
    """A list of search items to be used in the search"""

    def __str__(self) -> str:
        """Return a string representation of the search plan."""
        return "\n".join(
            f"Search Term: {item.search_term}\nReasoning: {item.reasoning}"
            for item in self.search_steps
        )
