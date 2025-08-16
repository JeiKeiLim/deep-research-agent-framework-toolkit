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


SPINNER_HTML = """
<span class="lds-spinner">
  <div></div><div></div><div></div><div></div><div></div><div></div>
  <div></div><div></div><div></div><div></div><div></div><div></div>
</span>
<style>
.lds-spinner {
  display:inline-block;
  position:relative;
  width:20px;
  height:20px;
  vertical-align:middle;
  margin-right:6px;
}
.lds-spinner div {
  transform-origin:10px 10px;
  animation: lds-spinner 1.2s linear infinite;
}
.lds-spinner div:after {
  content:" ";
  display:block;
  position:absolute;
  top:1px;
  left:9px;
  width:2px;
  height:5px;
  border-radius:20%;
  background:#333;
}
.lds-spinner div:nth-child(1)  { transform: rotate(0deg);   animation-delay:-1.1s; }
.lds-spinner div:nth-child(2)  { transform: rotate(30deg);  animation-delay:-1.0s; }
.lds-spinner div:nth-child(3)  { transform: rotate(60deg);  animation-delay:-0.9s; }
.lds-spinner div:nth-child(4)  { transform: rotate(90deg);  animation-delay:-0.8s; }
.lds-spinner div:nth-child(5)  { transform: rotate(120deg); animation-delay:-0.7s; }
.lds-spinner div:nth-child(6)  { transform: rotate(150deg); animation-delay:-0.6s; }
.lds-spinner div:nth-child(7)  { transform: rotate(180deg); animation-delay:-0.5s; }
.lds-spinner div:nth-child(8)  { transform: rotate(210deg); animation-delay:-0.4s; }
.lds-spinner div:nth-child(9)  { transform: rotate(240deg); animation-delay:-0.3s; }
.lds-spinner div:nth-child(10) { transform: rotate(270deg); animation-delay:-0.2s; }
.lds-spinner div:nth-child(11) { transform: rotate(300deg); animation-delay:-0.1s; }
.lds-spinner div:nth-child(12) { transform: rotate(330deg); animation-delay: 0s; }
@keyframes lds-spinner { 0% {opacity:1;} 100% {opacity:0;} }
</style>
"""


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
        # No global intermediate panel; per-query details live inside Chatbot messages

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

        The method appends the user's message, creates a temporary assistant
        message that aggregates intermediate progress in a collapsible section,
        and updates it live until the final answer replaces the draft.

        Args:
            query: The user's question to be processed.
            gr_messages: The current chat messages in the Gradio interface.

        Returns
        -------
            AsyncGenerator[dict, None]: An asynchronous generator yielding
            dictionaries to update Gradio components.
        """
        # Append the user's message
        gr_messages.append(ChatMessage(role="user", content=query))

        # Initial UI update: clear the input and show disabled state
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                value="", placeholder="Thinking...", interactive=False
            ),
        }

        # Drain any stale progress in the queue from previous runs
        while not self.progress_queue.empty():
            try:
                _ = self.progress_queue.get_nowait()
            except Exception:
                break

        # Start the agent query
        agent_task = asyncio.create_task(self.agent.query(query))

        # Prepare intermediate aggregation and insert a temporary assistant message
        intermediate_steps: list[str] = []
        temp_assistant_index = len(gr_messages)
        gr_messages.append(
            ChatMessage(
                role="assistant",
                content=self._compose_progress_message(
                    intermediate_steps=intermediate_steps,
                    latest_step=None,
                    progress_label="Starting...",
                ),
            )
        )

        # Yield to render the temporary assistant message
        yield {self._chatbot: gr_messages}

        # Stream and aggregate progress updates
        while not agent_task.done():
            try:
                progress: DeepResearchProgress = await asyncio.wait_for(
                    self.progress_queue.get(), timeout=0.1
                )
                # Stack unique steps; always keep the latest for drafting message
                if (
                    not intermediate_steps
                    or intermediate_steps[-1] != progress.progress_text
                ):
                    intermediate_steps.append(progress.progress_text)

                progress_label = f"{progress.progress_percentage:.1%}"

                # Update the temporary assistant message content with
                # latest step + stacked details
                latest_step = intermediate_steps[-1] if intermediate_steps else None
                gr_messages[temp_assistant_index] = ChatMessage(
                    role="assistant",
                    content=self._compose_progress_message(
                        intermediate_steps=intermediate_steps,
                        latest_step=latest_step,
                        progress_label=progress_label,
                    ),
                )

                # Yield the updated chat only
                # (per-message intermediate lives in Chatbot)
                yield {self._chatbot: gr_messages}
            except asyncio.TimeoutError:
                # No progress right now; continue polling
                pass

        # Finalize: replace the temporary assistant message with the final answer
        agent_result = await agent_task
        gr_messages[temp_assistant_index] = ChatMessage(
            role="assistant",
            content=self._compose_progress_message(
                intermediate_steps=intermediate_steps,
                label_text="🎉🎊🥳 Research complete.",
                latest_step=None,  # No latest step in final answer
                progress_label="100%",
            ),
        )

        final_answer_text = "---\n" + agent_result.final_output_as(str)
        # Post final answer as a new assistant message (separate from drafting/progress)
        gr_messages.append(ChatMessage(role="assistant", content=final_answer_text))

        # Yield the final result and re-enable input
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                placeholder="Ask a question", interactive=True
            ),
        }

    def _compose_progress_message(
        self,
        *,
        intermediate_steps: list[str],
        label_text: str = f"{SPINNER_HTML} Drafting answer...",
        latest_step: str | None,
        progress_label: str | None,
    ) -> str:
        """Compose per-query progress content embedded in the Chatbot message.

        Renders a folded details block stacking all steps and a 'Drafting answer...'
        section that shows only the latest step.
        """
        title = "Intermediate steps"
        if progress_label:
            title = f"{title} — {progress_label}"

        steps_md = self._format_intermediate_steps(intermediate_steps)
        # Default folded; do not set 'open' so folded state persists (stays folded)
        details_block = (
            f"<details>\n"
            f"  <summary><strong>{title}</strong> (click to expand)</summary>\n\n"
            f"{steps_md}\n"
            f"</details>"
        )

        drafting = label_text
        if latest_step:
            safe_step = latest_step.replace("```", "\u0060\u0060\u0060")
            drafting = f"{drafting}\n\n{safe_step}"

        return f"{details_block}\n\n{drafting}"

    def _format_intermediate_steps(self, steps: list[str]) -> str:
        """Return Markdown list formatting for intermediate steps.

        Falls back to a placeholder when there are no steps yet.
        """
        if not steps:
            return "No intermediate steps yet."

        # Number the steps for readability
        lines = []
        for index, step in enumerate(steps, start=1):
            # Escape triple backticks from steps to avoid breaking Markdown
            safe_step = step.replace("```", "\u0060\u0060\u0060")
            lines.append(f"{index}. {safe_step}")
        return "\n".join(lines)

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
            self._chatbot = gr.Chatbot(type="messages", label="Agent", height=800)
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
