"""Deep Research Agent script.

This script defines the Deep Research Agent class, which initializes agents based on
the provided configuration. It includes methods for converting configuration
dictionaries to agent instances and processing queries asynchronously.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import os

import agents
from agents.result import RunResult
from omegaconf import DictConfig
from openai import AsyncOpenAI

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
        response = await agents.Runner.run(
            self.agents["Planner"],
            input=query,
        )
        search_plan = response.final_output_as(SearchPlan)

        search_results = []
        # TODO: Use threads or asyncio.gather to parallelize search requests
        for search_item in search_plan.search_steps:
            search_response = await agents.Runner.run(
                self.agents["Search"],
                input=search_item.search_term,
            )
            search_results.append(search_response.final_output_as(str))

        evidences = "\n".join(
            f"{i + 1:d}. {result}" for i, result in enumerate(search_results)
        )
        synthesizer_input = f"Question: {query}\n\nEvidence:\n{evidences}"

        return await agents.Runner.run(
            self.agents["Synthesizer"],
            input=synthesizer_input,
        )
