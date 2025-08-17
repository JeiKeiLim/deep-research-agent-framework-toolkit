"""Deep Research Agent script.

This script defines the Deep Research Agent class, which initializes agents based on
the provided configuration. It includes methods for converting configuration
dictionaries to agent instances and processing queries asynchronously.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import agents
from agents.mcp import MCPServer, MCPServerStdio
from agents.result import RunResult, RunResultStreaming
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
from src.utils.conversation_history import Conversation, ConversationHistory
from src.utils.gradio.messages import oai_agent_stream_to_str_list
from src.utils.langfuse.shared_client import langfuse_client


def agent_config_to_agent(
    config: DictConfig,
    openai_client: AsyncOpenAI,
) -> Tuple[agents.Agent, List[MCPServer]]:
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

    sub_mcp_servers: List[MCPServer] = []
    if "sub_agents" in config:
        for sub_agent_name, sub_agent_config in config.sub_agents.items():
            sub_agent, sub_sub_mcp_servers = agent_config_to_agent(
                sub_agent_config, openai_client
            )
            sub_mcp_servers.extend(sub_sub_mcp_servers)
            tools.append(
                sub_agent.as_tool(
                    tool_name=sub_agent_name,
                    tool_description=sub_agent_config.description,
                )
            )

    # TODO: Add support for HostedMCPTool
    if tools:
        model_settings.tool_choice = "required"
        # TODO: Find a way to limit the number of parallel tool calls
        model_settings.parallel_tool_calls = False

    mcp_servers: List[MCPServerStdio] = []
    if "mcp_servers" in config.configs:
        for mcp_server_name, mcp_server_config in config.configs.mcp_servers.items():
            if (
                "params" in mcp_server_config.kwargs
                and "args" in mcp_server_config.kwargs.params
            ):
                for i in range(len(mcp_server_config.kwargs.params.args)):
                    mcp_server_config.kwargs.params.args[i] = (
                        mcp_server_config.kwargs.params.args[i].format(**os.environ)
                    )

            mcp_servers.append(
                MCPServerStdio(name=mcp_server_name, **mcp_server_config.kwargs)
            )

    return (
        agents.Agent(
            name=f"{config.name} Agent",
            instructions=prompt,
            model=agents.OpenAIChatCompletionsModel(
                model=config.configs.model, openai_client=openai_client
            ),
            tools=tools,
            output_type=output_types.get(config.name, None),
            model_settings=model_settings,
            mcp_servers=mcp_servers,
        ),
        mcp_servers + sub_mcp_servers,
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

    def __init__(
        self,
        agents_config: DictConfig,
        max_revision: int = 3,
        orchestration_mode: Literal["agent", "sequential"] = "agent",
        enable_history: bool = True,
        history_storage_dir: str = "conversation_history",
        max_messages: int = 5,
    ) -> None:
        """Initialize the Deep Research Agent with the provided configuration.

        Args:
            agents_config: Configuration for the agents.
            max_revision: Maximum number of revisions for the agent's output.
            orchestration_mode: Mode of orchestration ("agent" or "sequential").
            enable_history: Whether to enable conversation history.
            history_storage_dir: Directory to store conversation history.
            max_messages: Maximum number of recent messages to include in context.
        """
        self.orchestration_mode = orchestration_mode
        self._async_openai_client = AsyncOpenAI()
        self._mcp_servers: List[MCPServer] = []
        self.agents: dict[str, agents.Agent] = {}

        for key, value in agents_config.items():
            agent, mcp_servers = agent_config_to_agent(value, self._async_openai_client)
            self.agents[key] = agent
            self._mcp_servers.extend(mcp_servers)

        self.progress_callbacks = []
        self.max_revision = max_revision

        # History management
        self.enable_history = enable_history
        self.max_messages = max_messages
        self.title_update_callbacks = []  # 제목 업데이트 콜백 추가
        if self.enable_history:
            self.history = ConversationHistory(history_storage_dir)
            self.current_conversation_id: Optional[str] = None
        else:
            self.history = None
            self.current_conversation_id = None

    def add_progress_callback(
        self, callback: Callable[[DeepResearchProgress], None]
    ) -> None:
        """Add a progress callback to the agent.

        This method allows external code to register a callback function that will be
        called with the current progress of the agent.

        Args:
            callback:
                A callable previously registered as a title-update
                callback.
        """
        self.progress_callbacks.append(callback)

    def add_title_update_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add a title update callback to the agent.

        This method allows external code to register a callback function that will be
        called when a conversation title is updated.

        Args:
            callback: A callable that takes (conversation_id, new_title) as arguments.
        """
        self.title_update_callbacks.append(callback)

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

    def remove_title_update_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Remove a title update callback from the agent.

        This method allows external code to unregister a previously registered
        title update callback function.

        Args:
            callback: A callable that was previously registered as a
                     title update callback.
        """
        if callback in self.title_update_callbacks:
            self.title_update_callbacks.remove(callback)

    # History management methods
    def start_new_conversation(self, title: str = None) -> str:
        """Start a new conversation session.

        Args:
            title: Optional title for the conversation. If None, will use timestamp.

        Returns
        -------
            Conversation ID
        """
        if not self.enable_history or not self.history:
            return None

        if title is None:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.current_conversation_id = self.history.create_conversation(title)
        return self.current_conversation_id

    def get_conversation_context(self, max_messages: int = None) -> str:
        """Get current conversation context for agent reference.

        Args:
            max_messages: Maximum number of recent messages to include.
                         If None, uses the configured default value.

        Returns
        -------
            Context string for the agent
        """
        if (
            not self.enable_history
            or not self.history
            or not self.current_conversation_id
        ):
            return "No conversation context available."

        # Use configured default if max_messages is not specified
        if max_messages is None:
            max_messages = self.max_messages

        return self.history.get_conversation_context(
            self.current_conversation_id, max_messages
        )

    def add_message_to_history(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message to the current conversation history.

        Args:
            role: Role of the message sender ("user" or "assistant")
            content: Message content
            metadata: Optional metadata

        Returns
        -------
            True if successful, False if history is disabled or no current conversation
        """
        if (
            not self.enable_history
            or not self.history
            or not self.current_conversation_id
        ):
            return False

        return self.history.add_message(
            self.current_conversation_id, role, content, metadata
        )

    def search_history(self, query: str, max_results: int = 10) -> List[Conversation]:
        """Search conversation history.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns
        -------
            List of matching conversations
        """
        if not self.enable_history or not self.history:
            return []

        return self.history.search_conversations(query, max_results)

    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations.

        Returns
        -------
            List of all conversations
        """
        if not self.enable_history or not self.history:
            return []

        return self.history.get_all_conversations()

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of the conversation to switch to

        Returns
        -------
            True if successful, False if conversation not found
        """
        if not self.enable_history or not self.history:
            return False

        if conversation_id in self.history.conversations:
            self.current_conversation_id = conversation_id
            return True
        return False

    def get_conversation(self, conversation_id: str):
        """Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation

        Returns
        -------
            Conversation object or None if not found
        """
        if not self.enable_history or not self.history:
            return None

        return self.history.get_conversation(conversation_id)

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
                            self.agents["Search"],
                            input=search_item.model_dump_json(indent=2),
                        )
                        search_result = response.final_output_as(str)
                        # Extracting tools used in the search
                        tools_used = []
                        tool_args = []
                        for m_res in response.raw_responses:
                            for res_out in m_res.output:
                                if hasattr(res_out, "call_id") and hasattr(
                                    res_out, "name"
                                ):
                                    tools_used.append(res_out.name)

                                    if hasattr(res_out, "arguments"):
                                        tool_args.append(res_out.arguments)

                        search_item_span.update(
                            output=search_result,
                            metadata={"tools_used": tools_used, "tool_args": tool_args},
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
                    f"{revision_header_str} Searching for '{search_item.search_term}' ({i + 1}/{total_steps})"
                    f"\n{search_response.final_output_as(str)}...",
                )
                local_search_results.append(search_response.final_output_as(str))

            self._notify_progress(
                0.7,
                f"{revision_header_str} Search completed: {len(local_search_results)} results found"
                f"\n{local_search_results}",
            )

            search_span.update(output=local_search_results)

            return local_search_results

    async def query(self, query: str) -> RunResult | RunResultStreaming:  # noqa: PLR0912
        """Process a query using the configured agents.

        This method runs asynchronously and processes the provided query

        Args:
            query: The query to process.

        Returns
        -------
            Coroutine[Any, Any, RunResult]: A coroutine that processes the query and
            returns the result.
        """
        # Start new conversation if none exists
        if self.enable_history and not self.current_conversation_id:
            self.start_new_conversation()

        # Get conversation context for agent reference
        conversation_context = ""
        if self.enable_history:
            conversation_context = self.get_conversation_context(max_messages=10)

            # Check if this is the first user message (set as title)
            current_conv = self.get_conversation(self.current_conversation_id)
            if current_conv and len(current_conv.messages) == 0:
                # First message - set as conversation title
                self.history.update_conversation_title(
                    self.current_conversation_id, query
                )
                print(f"Set conversation title to: {query}")

                # Notify title update callbacks
                for callback in self.title_update_callbacks:
                    try:
                        callback(self.current_conversation_id, query)
                    except Exception as e:
                        print(f"Error in title update callback: {e}")

            # Add user message to history
            self.add_message_to_history("user", query)

        print(len(self._mcp_servers), "MCP servers to connect")
        for mcp_server in self._mcp_servers:
            try:
                await mcp_server.cleanup()
            except Exception as e:
                print(f"Error cleaning up MCP server {mcp_server.name}: {e}")

        for mcp_server in self._mcp_servers:
            try:
                await mcp_server.connect()
            except Exception as e:
                print(f"Error connecting to MCP server {mcp_server.name}: {e}")

        # Enhance query with conversation context
        enhanced_query = query
        if (
            conversation_context
            and conversation_context != "No conversation context available."
        ):
            enhanced_query = f"""Previous conversation context:
            {conversation_context}

            Current question: {query}

            Please answer by referring to the previous conversation above, and maintain continuity with it in your response."""

        if self.orchestration_mode == "agent":
            response = await self._query_agent(enhanced_query)
        else:
            response = await self._query_sequential(enhanced_query)

        # Add assistant response to history
        if self.enable_history:
            response_content = response.final_output_as(str)
            self.add_message_to_history("assistant", response_content)

        return response

    async def _query_agent(self, query: str) -> RunResultStreaming:
        """Process a query using the configured agents.

        This method runs asynchronously and processes the provided query using the
        main agent, allowing for a more streamlined and efficient query handling.
        It utilizes the `agents.Runner` to execute the main agent with the given query.

        Args:
            query: The query to process.

        Returns
        -------
            Coroutine[Any, Any, RunResult]: A coroutine that processes the query and
            returns the result.
        """
        progress_percentage = 0.0
        response_stream: RunResultStreaming | None = None
        with langfuse_client.start_as_current_span(
            name="DeepResearchAgentFrameworkToolkit.query_agent", input=query
        ) as agent_span:
            try:
                response_stream = agents.Runner.run_streamed(
                    self.agents["Main"],
                    input=query,
                    max_turns=999,
                )
                async for _item in response_stream.stream_events():
                    intermediate_messages = oai_agent_stream_to_str_list(_item)

                    for intermediate_message in intermediate_messages:
                        progress_percentage += 0.01
                        # TODO: Find a way to calculate pseudo progress_percentage
                        self._notify_progress(
                            progress_percentage,
                            f"{intermediate_message}",
                        )

                agent_span.update(output=response_stream.final_output_as(str))
                return response_stream
            except Exception as e:
                error_message = f"The agent encountered an unrecoverable error: {str(e)}\nThis can happen if the agent tries to call a sub-agent that does not exist or with incorrect parameters. The operation will now stop."
                self._notify_progress(1.0, error_message)
                assert response_stream is not None, (
                    "Response stream should not be None if an exception occurs."
                )
                return response_stream

    async def _query_sequential(self, query: str) -> RunResult:
        """Process a query using the configured agents.

        This method runs asynchronously and processes the provided query in a
        sequential manner, allowing for multiple revisions and feedback from the
        critic agent.

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
            name="DeepResearchAgentFrameworkToolkit.query_sequential", input=query
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
                    planner_input = f"""Question:
                    {query}

                    Previous Answer:
                    {previous_answer}

                    Critic Feedback:
                    {critic_feedback_str}
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
                            f"{revision_header_str} Planning search steps completed: {len(search_plan.search_steps)} steps"
                            f"\n{search_plan}",
                        )
                        planner_span.update(output=search_plan)

                    local_search_results = await self._run_search(
                        search_plan, revision_header_str
                    )

                    search_results.extend(local_search_results)

                    evidences = "\n".join(
                        f"## Evidence {i + 1:d}\n{result}"
                        for i, result in enumerate(search_results)
                    )
                    synthesizer_input = f"""Question:
                    {query}

                    # Evidences:
                    {evidences}

                    Previous Answer:
                    {previous_answer}

                    Critic Feedback:
                    {critic_feedback_str}
                    """

                    self._notify_progress(
                        0.8, f"{revision_header_str} Synthesizing final answer"
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

                    self._notify_progress(
                        0.9,
                        f"{revision_header_str} Synthesized answer: {synthesized_answer}",
                    )

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

                    self._notify_progress(
                        1.0,
                        f"{revision_header_str} Critic feedback received: {critic_feedback}",
                    )

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
