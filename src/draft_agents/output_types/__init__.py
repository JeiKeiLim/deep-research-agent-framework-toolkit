from src.draft_agents.output_types.critic import CriticFeedback
from src.draft_agents.output_types.evaluator import EvaluatorFeedback
from src.draft_agents.output_types.planner import SearchItem, SearchPlan


output_types = {
    "Planner": SearchPlan,
    "Critic": CriticFeedback,
    "Evaluator": EvaluatorFeedback,
}

__all__ = [
    "output_types",
]
