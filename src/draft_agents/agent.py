"""Deep Research Agent script.

This script defines the Deep Research Agent class, which initializes agents based on
the provided configuration. It includes methods for converting configuration
dictionaries to agent instances and processing queries asynchronously.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import os
from typing import Callable

import agents
from agents.result import RunResult
from omegaconf import DictConfig
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.draft_agents.function_tools import function_tools
from src.draft_agents.output_types import SearchPlan, output_types


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
    # TODO: Check if tools exist in config.configs,
    # and if so, set the tools accordingly
    if "function_tools" in config.configs:
        tools = [
            agents.function_tool(function_tools[tool_name])
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

    def __init__(self, agents_config: DictConfig) -> None:
        """Initialize the Deep Research Agent with the provided configuration.

        Args:
            agents_config: Configuration for the agents.
        """
        self._async_openai_client = AsyncOpenAI()
        self.agents = {
            key: agent_config_to_agent(value, self._async_openai_client)
            for key, value in agents_config.items()
        }
        self.progress_callbacks = []

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

    def _notify_progress(self, progress: DeepResearchProgress) -> None:
        """Notify all registered progress callbacks with the current progress.

        This method iterates through all registered callbacks and calls each one with
        the provided progress information.

        Args:
            progress: The current progress of the agent.
        """
        for callback in self.progress_callbacks:
            callback(progress)

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
        self._notify_progress(
            DeepResearchProgress(
                progress_text="Planning search steps",
                progress_percentage=0.0,
            )
        )
        response = await agents.Runner.run(
            self.agents["Planner"],
            input=query,
        )
        search_plan = response.final_output_as(SearchPlan)
        self._notify_progress(
            DeepResearchProgress(
                progress_text=f"Planning search steps completed: {len(search_plan.search_steps)} steps",
                progress_percentage=0.1,
            )
        )

        total_steps = len(search_plan.search_steps)

        search_results = []
        # TODO: Use threads or asyncio.gather to parallelize search requests
        for search_item in search_plan.search_steps:
            self._notify_progress(
                DeepResearchProgress(
                    progress_text=f"Searching for '{search_item.search_term}' ({len(search_results) + 1}/{total_steps})",
                    progress_percentage=0.1 + (len(search_results) / total_steps) * 0.8,
                )
            )

            search_response = await agents.Runner.run(
                self.agents["Search"],
                input=search_item.search_term,
            )
            search_results.append(search_response.final_output_as(str))

        evidences = "\n".join(
            f"{i + 1:d}. {result}" for i, result in enumerate(search_results)
        )
        synthesizer_input = f"Question: {query}\n\nEvidence:\n{evidences}"

        self._notify_progress(
            DeepResearchProgress(
                progress_text="Synthesizing final answer",
                progress_percentage=0.9,
            )
        )

        return await agents.Runner.run(
            self.agents["Synthesizer"],
            input=synthesizer_input,
        )
