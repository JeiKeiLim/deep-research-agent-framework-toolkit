"""Deep Research Agent CLI.

This module provides a command-line interface for interacting with the Deep
Research Agent framework. It supports both a one-time query mode and an
interactive chat mode.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import asyncio
import sys
from collections.abc import Awaitable

import hydra
from omegaconf import DictConfig

from src.draft_agents.agent import DeepResearchAgent, DeepResearchProgress
from src.utils import setup_langfuse_tracer


class DeepResearchAgentCLI:
    """A class for the Deep Research Agent CLI."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the CLI app."""
        self.agent = DeepResearchAgent(cfg.agents, **cfg.agent_configs)
        self.progress_queue: asyncio.Queue[DeepResearchProgress] = asyncio.Queue()
        self.agent.add_progress_callback(self.progress_queue.put_nowait)

    async def _run_query_with_progress(self, query: str) -> None:
        """Run a single query and display live progress."""
        # Drain any stale progress from previous runs
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        agent_task = asyncio.create_task(self.agent.query(query))

        intermediate_steps: list[str] = []
        spinner = ["|", "/", "-", "\\"]
        spinner_idx = 0
        latest_step = "Starting research..."

        print()  # Newline after user input
        while not agent_task.done():
            try:
                progress = await asyncio.wait_for(
                    self.progress_queue.get(), timeout=0.1
                )
                if (
                    not intermediate_steps
                    or intermediate_steps[-1] != progress.progress_text
                ):
                    intermediate_steps.append(progress.progress_text)
                latest_step = progress.progress_text

            except TimeoutError:
                pass

            # Render spinner and latest step
            spinner_char = spinner[spinner_idx % len(spinner)]
            spinner_idx += 1
            line = f"\r{spinner_char} {latest_step}"
            # Truncate to fit in one line
            line = line[:79] if len(line) > 79 else line
            sys.stdout.write(line.ljust(80))
            sys.stdout.flush()

        # Clear the progress line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        result = await agent_task

        print("✅ Research complete.\n")
        if intermediate_steps:
            print("--- Intermediate steps ---")
            for i, step in enumerate(intermediate_steps, 1):
                print(f"{i}. {step}")
            print("--------------------------\n")

        print("--- Final Answer ---")
        print(result.final_output_as(str))

    async def run_one_time(self, query: str) -> None:
        """Run the agent for a single query and exit."""
        print(f"Running query: {query}")
        await self._run_query_with_progress(query)

    async def run_interactive(self) -> None:
        """Run the agent in an interactive chat session."""
        print("Starting interactive chat session.")
        print("Type your query and press Enter. Type 'exit' or 'quit' to end.")
        if (
            hasattr(self.agent, "conversation_manager")
            and self.agent.conversation_manager
        ):
            self.agent.conversation_manager.start_new_conversation()
            print("Started a new conversation.")

        while True:
            try:
                query = await asyncio.to_thread(input, "You: ")
                if query.lower() in ["exit", "quit"]:
                    print("Exiting chat.")
                    break
                if not query.strip():
                    continue

                await self._run_query_with_progress(query)
                print("\n" + "=" * 40 + "\n")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for the CLI application.

    To run in one-time mode, provide the query as a Hydra override:
    $ python apps/cli.py +query="What is RASA?"

    To run in interactive mode, run without the query argument:
    $ python apps/cli.py
    """
    query = cfg.get("query", None)

    setup_langfuse_tracer()
    cli_app = DeepResearchAgentCLI(cfg)

    runner: Awaitable[None] = (
        cli_app.run_one_time(query) if query else cli_app.run_interactive()
    )

    try:
        asyncio.run(runner)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")


if __name__ == "__main__":
    main()
