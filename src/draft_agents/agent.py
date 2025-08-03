"""Deep Research Agent script.

This script defines the Deep Research Agent class, which initializes agents based on
the provided configuration. It includes methods for converting configuration
dictionaries to agent instances and processing queries asynchronously.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import asyncio
import os
from typing import Callable, List

import agents
from agents.result import RunResult
from omegaconf import DictConfig
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.draft_agents.function_tools import function_tools
from src.draft_agents.output_types import (
    CriticFeedback,
    SearchItem,
    SearchPlan,
    output_types,
)
from src.utils.langfuse.shared_client import langfuse_client


def agent_config_to_agent(
    config: DictConfig, openai_client: AsyncOpenAI
) -> agents.Agent:
    """Convert a DictConfig to an Agent instance.

    This function takes a configuration dictionary and converts it into
    an Agent instance,
    allowing for dynamic agent creation based on the provided configuration.

    Args:
        config (DictConfig): The configuration dictionary for the agent.

    Returns
    -------
        agents.Agent: An instance of the Agent class configured with the provided
        settings.
    """
    prompt = ""
    if "prompt" in config.configs:
        prompt = config.configs.prompt

    if os.path.exists(prompt):
        with open(prompt, "r") as file:
            prompt = file.read()

    if "prompt_args" in config.configs:
        for key, value in config.configs.prompt_args.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

    model_settings = agents.ModelSettings()
    # TODO: Check if model_settings exist in config.configs,
    # and if so, set the model settings accordingly

    tools = []
    if "function_tools" in config.configs:
        tools: List[agents.Tool] = [
            agents.function_tool(function_tools[tool_name], name_override=tool_name)
            for tool_name in config.configs.function_tools
        ]

    if tools:
        model_settings.tool_choice = "required"

    return agents.Agent(
        name=f"{config.name} Agent",
        instructions=prompt,
        model=agents.OpenAIChatCompletionsModel(
            model=config.configs.model, openai_client=openai_client
        ),
        tools=tools,
        output_type=output_types.get(config.name, None),
        model_settings=model_settings,
    )


class DeepResearchProgress(BaseModel):
    """Progress of the Deep Research Agent.

    This class is used to track the progress of the Deep Research Agent during its
    operations.
    """

    progress_text: str
    """Text description of the current progress step."""

    progress_percentage: float
    """Percentage of completion for the current progress step,
    ranging from 0.0 to 1.0."""


class DeepResearchAgent:
    """Deep Research Agent."""

    def __init__(self, agents_config: DictConfig, max_revision: int = 3) -> None:
        """Initialize the Deep Research Agent with the provided configuration.

        Args:
            agents_config: Configuration for the agents.
            max_revision: Maximum number of revisions for the agent's output.
        """
        self._async_openai_client = AsyncOpenAI()
        self.agents = {
            key: agent_config_to_agent(value, self._async_openai_client)
            for key, value in agents_config.items()
        }
        self.progress_callbacks = []
        self.max_revision = max_revision

    def add_progress_callback(
        self, callback: Callable[[DeepResearchProgress], None]
    ) -> None:
        """Add a progress callback to the agent.

        This method allows external code to register a callback function that will be
        called with the current progress of the agent.

        Args:
            callback: A callable that takes a DeepResearchProgress object as an
            argument.
        """
        self.progress_callbacks.append(callback)

    def remove_progress_callback(
        self, callback: Callable[[DeepResearchProgress], None]
    ) -> None:
        """Remove a progress callback from the agent.

        This method allows external code to unregister a previously registered
        callback function.

        Args:
            callback: A callable that was previously registered as a progress callback.
        """
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)

    def _notify_progress(self, p_percentage: float, p_text: str) -> None:
        """Notify all registered progress callbacks with the current progress.

        This method iterates through all registered callbacks and calls each one with
        the provided progress information.

        Args:
            p_percentage: The current progress percentage, ranging from 0.0 to 1.0.
            p_text: A text description of the current progress step.
        """
        progress = DeepResearchProgress(
            progress_text=p_text,
            progress_percentage=p_percentage,
        )
        for callback in self.progress_callbacks:
            callback(progress)

    async def _run_search(
        self, search_plan: SearchPlan, revision_header_str: str
    ) -> List[str]:
        with langfuse_client.start_as_current_span(
            name="DeepResearchAgentFrameworkToolkit._run_search", input=search_plan
        ) as search_span:
            total_steps = len(search_plan.search_steps)
            local_search_results = []

            async def run_search(
                search_item: SearchItem,
            ) -> tuple[SearchItem, RunResult | None]:
                with search_span.start_as_current_span(
                    name="DeepResearchAgentFrameworkToolkit._run_search.run_search",
                    input=search_item.search_term,
                ) as search_item_span:
                    try:
                        response = await agents.Runner.run(
                            self.agents["Search"], input=search_item.search_term
                        )
                        search_result = response.final_output_as(str)
                        # Extracting tools used in the search
                        tools_used = []
                        for m_res in response.raw_responses:
                            for res_out in m_res.output:
                                if hasattr(res_out, "call_id") and hasattr(
                                    res_out, "name"
                                ):
                                    tools_used.append(res_out.name)

                        search_item_span.update(
                            output=search_result,
                            metadata={"tools_used": tools_used},
                        )
                    except Exception as e:
                        print(
                            f"Error during search for '{search_item.search_term}': {str(e)}"
                        )
                        return search_item, None
                    return search_item, response

            tasks = [
                run_search(search_item) for search_item in search_plan.search_steps
            ]

            for i, future in enumerate(asyncio.as_completed(tasks)):
                search_item, search_response = await future
                if search_response is None:
                    self._notify_progress(
                        0.1 + ((i + 1) / total_steps) * 0.6,
                        f"{revision_header_str} Search for '{search_item.search_term}' failed",
                    )
                    continue

                self._notify_progress(
                    0.1 + ((i + 1) / total_steps) * 0.6,
                    f"{revision_header_str} Searching for '{search_item.search_term}' ({i + 1}/{total_steps})",
                )
                local_search_results.append(search_response.final_output_as(str))

            self._notify_progress(
                0.8,
                f"{revision_header_str} Search completed: {len(local_search_results)} results found",
            )

            search_span.update(output=local_search_results)

            return local_search_results

    async def query(self, query: str) -> RunResult:
        """Process a query using the configured agents.

        This method runs asynchronously and processes the provided query

        Args:
            query: The query to process.

        Returns
        -------
            Coroutine[Any, Any, RunResult]: A coroutine that processes the query and
            returns the result.
        """
        revision_count = 0
        previous_answer = "Initial state, no answer yet"
        critic_feedback_str = "Initial state, no feedback yet"
        search_results = []

        synthesizer_response: RunResult | None = None

        with langfuse_client.start_as_current_span(
            name="DeepResearchAgentFrameworkToolkit.query", input=query
        ) as agent_span:
            while revision_count < self.max_revision:
                revision_header_str = f"R[{revision_count + 1:d}/{self.max_revision:d}]"
                self._notify_progress(
                    0.0, f"{revision_header_str} Planning search steps"
                )
                revision_count += 1

                with langfuse_client.start_as_current_span(
                    name=f"DeepResearchAgentFrameworkToolkit.query.{revision_header_str}",
                ) as revision_span:
                    planner_input = f"""Question: {query}

                    Previous Answer: {previous_answer}

                    Critic Feedback: {critic_feedback_str}
                    """

                    with langfuse_client.start_as_current_span(
                        name="DeepResearchAgentFrameworkToolkit.query.planner",
                        input=planner_input,
                    ) as planner_span:
                        response = await agents.Runner.run(
                            self.agents["Planner"],
                            input=planner_input,
                        )
                        search_plan = response.final_output_as(SearchPlan)
                        self._notify_progress(
                            0.1,
                            f"{revision_header_str} Planning search steps completed: {len(search_plan.search_steps)} steps",
                        )
                        planner_span.update(output=search_plan)

                    local_search_results = await self._run_search(
                        search_plan, revision_header_str
                    )

                    search_results.extend(local_search_results)

                    evidences = "\n".join(
                        f"{i + 1:d}. {result}"
                        for i, result in enumerate(search_results)
                    )
                    synthesizer_input = f"""Question:
                    {query}

                    Evidence:
                    {evidences}

                    Previous Answer:
                    {previous_answer}

                    Critic Feedback:
                    {critic_feedback_str}
                    """

                    self._notify_progress(
                        0.9, f"{revision_header_str} Synthesizing final answer"
                    )

                    with langfuse_client.start_as_current_span(
                        name="DeepResearchAgentFrameworkToolkit.query.synthesizer",
                        input=synthesizer_input,
                    ) as synthesizer_span:
                        synthesizer_response = await agents.Runner.run(
                            self.agents["Synthesizer"],
                            input=synthesizer_input,
                        )
                        synthesized_answer = synthesizer_response.final_output_as(str)

                        synthesizer_span.update(output=synthesized_answer)

                    revision_span.update(output=synthesized_answer)
                    critic_input = f"""Query**: {query}

                    **Answer**: {synthesized_answer}

                    **Evidence**: {evidences}
                    """

                    with langfuse_client.start_as_current_span(
                        name="DeepResearchAgentFrameworkToolkit.query.critic",
                        input=critic_input,
                    ) as critic_span:
                        critic_response = await agents.Runner.run(
                            self.agents["Critic"],
                            input=critic_input,
                        )
                        critic_feedback = critic_response.final_output_as(
                            CriticFeedback
                        )  # noqa
                        critic_span.update(output=critic_feedback)

                    # TODO: Make a loop to improve the answer based on critic feedback

                    if not critic_feedback.needs_revision:
                        agent_span.update(output=synthesized_answer)
                        return synthesizer_response

                    previous_answer = synthesized_answer
                    critic_feedback_str = f"""Previous answer has following issues:
                    {critic_feedback.issues}

                    Suggestions for improvement:
                    {critic_feedback.suggestions}
                    """

                    if revision_count >= self.max_revision:
                        agent_span.update(output=synthesized_answer)
                        return synthesizer_response

            if synthesizer_response is None:
                raise ValueError(
                    "Synthesized answer is None after maximum revisions reached."
                )

            agent_span.update(output=synthesizer_response.final_output_as(str))
            return synthesizer_response
