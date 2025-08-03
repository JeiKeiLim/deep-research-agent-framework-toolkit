from src.draft_agents.output_types.critic import CriticFeedback
from src.draft_agents.output_types.planner import SearchItem, SearchPlan


output_types = {"Planner": SearchPlan, "Critic": CriticFeedback}

__all__ = [
    "output_types",
]
