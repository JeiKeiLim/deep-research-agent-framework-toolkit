"""Deep Research Agent script.

This script defines the Deep Research Agent class, which initializes agents based on
the provided configuration. It includes methods for converting configuration
dictionaries to agent instances and processing queries asynchronously.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import os
from typing import Any, Coroutine

import agents
from agents.result import RunResult
from omegaconf import DictConfig
from openai import AsyncOpenAI


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

    tools = []
    model_settings = agents.ModelSettings()
    if tools:
        model_settings.tool_choice = "required"

    return agents.Agent(
        name=f"{config.name} Agent",
        instructions=prompt,
        model=agents.OpenAIChatCompletionsModel(
            model=config.configs.model, openai_client=openai_client
        ),
        tools=tools,
        output_type=None,  # TODO: Define the output type if needed
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

    def query(self, query: str) -> Coroutine[Any, Any, RunResult]:
        """Process a query using the configured agents.

        This method runs asynchronously and processes the provided query

        Args:
            query: The query to process.

        Returns
        -------
            Coroutine[Any, Any, RunResult]: A coroutine that processes the query and
            returns the result.
        """
        return agents.Runner.run(
            self.agents["Main"],
            input=query,
        )
