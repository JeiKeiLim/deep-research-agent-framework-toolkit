"""Deep Research Agent GUI environment.

This module sets up a Gradio interface for the Deep Research Agent GUI,
allowing users to interact with the agent framework through a web interface.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

import asyncio
from typing import AsyncGenerator

import gradio as gr
import hydra
from gradio.components.chatbot import ChatMessage
from omegaconf import DictConfig

from src.draft_agents.agent import DeepResearchAgent, DeepResearchProgress
from src.utils import setup_langfuse_tracer


class DeepResearchAgentGUI:
    """A class representing the Deep Research Agent GUI.

    This class initializes the Gradio interface for the Deep Research Agent GUI,
    allowing users to interact with the agent framework through a web interface.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the Deep Research Agent GUI.

        Args:
            cfg: The configuration dictionary for the agent.
        """
        self.app = self._create_gui()
        self.agent = DeepResearchAgent(cfg.agents, **cfg.agent_configs)
        self.progress_queue = asyncio.Queue()
        self.agent.add_progress_callback(
            lambda progress: self.progress_queue.put_nowait(progress)
        )

        self._chatbot: gr.Chatbot
        self._chat_message: gr.Textbox
        self._progress_details: gr.Markdown

    def start(self) -> None:
        """Start the Gradio app."""
        self.app.launch(
            server_name="0.0.0.0", server_port=7860, share=False, debug=True
        )

    async def _process_query(
        self,
        query: str,
        gr_messages: list[ChatMessage],
    ) -> AsyncGenerator[dict, None]:
        """Process the user's question and return the updated chat messages.

        This function simulates an asynchronous query processing, appending
        the user's question to the chat history and yielding the updated messages.

        Args:
            query: The user's question to be processed.
            gr_messages: The current chat messages in the Gradio interface.

        Returns
        -------
            AsyncGenerator[dict, None]: An asynchronous generator yielding
            dictionaries to update Gradio components.
        """
        gr_messages.append(ChatMessage(role="user", content=query))

        # Yield the first update to clear the input and show "Thinking..."
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                value="", placeholder="Thinking...", interactive=False
            ),
        }

        agent_task = asyncio.create_task(self.agent.query2(query))

        while not agent_task.done():
            try:
                progress: DeepResearchProgress = await asyncio.wait_for(
                    self.progress_queue.get(), timeout=0.1
                )
                progress_text = (
                    f"Progress: {progress.progress_percentage:.1%} - "
                    f"{progress.progress_text}"
                )
                yield {
                    self._chat_message: gr.update(placeholder=progress_text),
                }
            except asyncio.TimeoutError:
                pass

        agent_result = await agent_task
        gr_messages.append(
            ChatMessage(role="assistant", content=agent_result.final_output_as(str))
        )

        # Yield the final result
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                placeholder="Ask a question", interactive=True
            ),
        }

    def _create_gui(self) -> gr.Blocks:
        """Create and return the Gradio app."""
        with gr.Blocks(title="DRAFT - Deep Research Agent Framework Toolkit") as app:
            gr.HTML(  # Make progress text darker
                """
                <style>
                    #chat-message::placeholder {
                        color: #4A5568; /* Darker gray for placeholder text */
                        opacity: 1;
                    }
                </style>
                """
            )
            gr.Markdown("# Deep Research Agent GUI")
            gr.Markdown(
                "This is a simple Gradio interface for the Deep Research Agent GUI."
            )
            self._chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
            self._chat_message = gr.Textbox(
                elem_id="chat-message",
                placeholder="Ask a question",
                show_label=False,
                lines=1,
                submit_btn=True,
            )

            self._chat_message.submit(
                self._process_query,
                [self._chat_message, self._chatbot],
                [self._chatbot, self._chat_message],
            )

            return app


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load the configuration using Hydra."""
    setup_langfuse_tracer()
    gui_app = DeepResearchAgentGUI(cfg)
    gui_app.start()


if __name__ == "__main__":
    main()
