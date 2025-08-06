"""Evaluate the agent's response using LLM-as-a-judge."""

import asyncio
from typing import Tuple

import agents
import hydra
import pydantic
from agents import Agent
from langfuse._client.datasets import DatasetItemClient
from omegaconf import DictConfig
from openai import AsyncOpenAI
from rich.progress import track

from src.draft_agents.agent import DeepResearchAgent, agent_config_to_agent
from src.draft_agents.output_types.evaluator import EvaluatorFeedback
from src.utils import gather_with_progress, setup_langfuse_tracer
from src.utils.langfuse.shared_client import langfuse_client


class LangFuseTracedResponse(pydantic.BaseModel):
    """Agent Response and LangFuse Trace info."""

    answer: str | None
    trace_id: str | None


EVALUATOR_TEMPLATE = """\
# Question

{question}

# Ground Truth

{ground_truth}

# Proposed Answer

{proposed_response}

"""


class EvaluatorQuery(pydantic.BaseModel):
    """Query to the evaluator agent."""

    question: str
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        """Obtain query string to the evaluator agent."""
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


async def run_agent_with_trace(
    agent: DeepResearchAgent, query: str
) -> LangFuseTracedResponse:
    """Run OpenAI Agent on query, returning response and trace_id.

    Returns None if agent exceeds max_turn limit.
    """
    try:
        result = await agent.query(query)
        if "|" in result.final_output:
            answer = result.final_output.split("|")[-1].strip()
        else:
            answer = result.final_output

    except agents.MaxTurnsExceeded:
        answer = None

    return LangFuseTracedResponse(
        answer=answer, trace_id=langfuse_client.get_current_trace_id()
    )


async def run_evaluation_coroutine(
    agent: DeepResearchAgent,
    evaluator_agent: Agent,
    lf_dataset_item: DatasetItemClient,
    run_name: str,
) -> Tuple[LangFuseTracedResponse, EvaluatorFeedback | None]:
    """Run the evaluation coroutine."""
    ground_truth = lf_dataset_item.expected_output
    assert ground_truth is not None

    with lf_dataset_item.run(run_name=run_name) as root_span:
        root_span.update(input=lf_dataset_item.input["text"])

        # Run the main agent to get the response
        traced_response = await run_agent_with_trace(
            agent, query=lf_dataset_item.input["text"]
        )
        root_span.update(output=traced_response.answer)

    answer = traced_response.answer
    if answer is None:
        return traced_response, None

    evaluator_result = await agents.Runner.run(
        evaluator_agent,
        input=EvaluatorQuery(
            question=lf_dataset_item.input["text"],
            ground_truth=ground_truth["text"],
            proposed_response=answer,
        ).get_query(),
    )
    evaluator_response = evaluator_result.final_output_as(EvaluatorFeedback)

    return traced_response, evaluator_response


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_evaluation(cfg: DictConfig) -> None:
    """Run the evaluation of the Deep Research Agent."""
    # Initialize the agent with the provided configuration
    eval_cfg = cfg.evaluation_configs
    setup_langfuse_tracer()
    agent = DeepResearchAgent(cfg.agents, **cfg.agent_configs)

    async_openai_client = AsyncOpenAI()
    evaluator_agent = agent_config_to_agent(cfg.evaluator, async_openai_client)

    lf_dataset_items = langfuse_client.get_dataset(eval_cfg.langfuse_dataset_name).items

    coroutines = [
        run_evaluation_coroutine(
            agent, evaluator_agent, lf_dataset_item, eval_cfg.run_name
        )
        for lf_dataset_item in lf_dataset_items
    ]
    results = asyncio.run(
        gather_with_progress(coroutines, description="Running agent and evaluating")
    )

    for _traced_response, _eval_output in track(
        results, total=len(results), description="Uploading scores"
    ):
        # Link the trace to the dataset item for analysis
        if _eval_output is not None:
            langfuse_client.create_score(
                name="factual_accuracy",
                value=_eval_output.score_factual_accuracy.score,
                comment=_eval_output.score_factual_accuracy.reasoning,
                trace_id=_traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="completeness",
                value=_eval_output.score_completeness.score,
                comment=_eval_output.score_completeness.reasoning,
                trace_id=_traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="coherence",
                value=_eval_output.score_coherence.score,
                comment=_eval_output.score_coherence.reasoning,
                trace_id=_traced_response.trace_id,
            )
            langfuse_client.create_score(
                name="citations",
                value=_eval_output.score_citations.score,
                comment=_eval_output.score_citations.reasoning,
                trace_id=_traced_response.trace_id,
            )


if __name__ == "__main__":
    run_evaluation()
