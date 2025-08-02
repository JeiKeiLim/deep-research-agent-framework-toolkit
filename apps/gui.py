"""Deep Research Agent GUI environment.

This module sets up a Gradio interface for the Deep Research Agent GUI,
allowing users to interact with the agent framework through a web interface.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

from typing import AsyncGenerator, List

import gradio as gr
import hydra
from gradio.components.chatbot import ChatMessage
from omegaconf import DictConfig

from src.draft_agents.agent import DeepResearchAgent


class DeepResearchAgentGUI:
    """A class representing the Deep Research Agent GUI.

    This class initializes the Gradio interface for the Deep Research Agent GUI,
    allowing users to interact with the agent framework through a web interface.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.app = self._create_gui()
        self.agent = DeepResearchAgent(cfg.agents)

    def start(self) -> None:
        """Start the Gradio app."""
        self.app.launch(
            server_name="0.0.0.0", server_port=7860, share=False, debug=True
        )

    async def _process_query(
        self, query: str, gr_messages: list[ChatMessage]
    ) -> AsyncGenerator[List[ChatMessage], None]:
        """Process the user's question and return the updated chat messages.

        This function simulates an asynchronous query processing, appending
        the user's question to the chat history and yielding the updated messages.

        Args:
            query: The user's question to be processed.
            gr_messages: The current chat messages in the Gradio interface.

        Returns
        -------
            AsyncGenerator[List[ChatMessage], None]: An asynchronous generator yielding
            the updated chat messages.
        """
        gr_messages.append(
            ChatMessage(
                role="user",
                content=query,
            )
        )
        yield gr_messages

        agent_result = await self.agent.query(query)
        gr_messages.append(
            ChatMessage(role="assistant", content=agent_result.final_output_as(str))
        )
        yield gr_messages

    def _create_gui(self) -> gr.Blocks:
        """Create and return the Gradio app."""
        with gr.Blocks(title="OAI Agent SDK - Multi-agent") as app:
            gr.Markdown("# Deep Research Agent GUI")
            gr.Markdown(
                "This is a simple Gradio interface for the Deep Research Agent GUI."
            )
            chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
            chat_message = gr.Textbox(lines=1, label="Ask a question", submit_btn=True)
            chat_message.submit(self._process_query, [chat_message, chatbot], [chatbot])

            return app


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load the configuration using Hydra."""
    gui_app = DeepResearchAgentGUI(cfg)
    gui_app.start()


if __name__ == "__main__":
    main()
