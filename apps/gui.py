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
from src.utils.conversation_history import Conversation


SPINNER_HTML = """
<span class="lds-spinner">
  <div></div><div></div><div></div><div></div><div></div><div></div>
  <div></div><div></div><div></div><div></div><div></div><div></div>
</span>
<style>
.lds-spinner { display:inline-block; position:relative; width:20px; height:20px; vertical-align:middle; margin-right:6px; }
.lds-spinner div { transform-origin:10px 10px; animation: lds-spinner 1.2s linear infinite; }
.lds-spinner div:after { content:" "; display:block; position:absolute; top:1px; left:9px; width:2px; height:5px; border-radius:20%; background:#333; }
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
    """A class representing the Deep Research Agent GUI."""

    def __init__(self, cfg: DictConfig) -> None:
        self.app = self._create_gui()
        self.agent = DeepResearchAgent(cfg.agents, **cfg.agent_configs)
        self.progress_queue = asyncio.Queue()
        self.agent.add_progress_callback(
            lambda progress: self.progress_queue.put_nowait(progress)
        )

        # 제목 업데이트 콜백 등록
        if hasattr(self.agent, 'conversation_manager') and self.agent.conversation_manager:
            self.agent.conversation_manager.add_title_update_callback(self._on_title_updated)

        # UI handles (assigned in _create_gui)
        self._chatbot: gr.Chatbot
        self._chat_message: gr.Textbox
        self._conv_dropdown: gr.Dropdown
        self._search_query: gr.Textbox
        self._search_btn: gr.Button
        self._search_dropdown: gr.Dropdown
        self._new_btn: gr.Button
        self._delete_btn: gr.Button
        self._current_info: gr.Markdown

        # Internal mapping: label <-> id (for reference).
        # Dropdown choices use (label, id) tuples; the value is the id.
        self._label_to_id: dict[str, str] = {}
        self._id_to_label: dict[str, str] = {}

        # Initialize after GUI is created
        self._initialize_conversation_history()

    # ---------- Helpers (mapping / labels) ----------

    def _build_label(self, conv: Conversation) -> str:
        title = conv.title if conv.title and conv.title.strip() else "No title"
        return f"{title} | {conv.created_at.strftime('%Y-%m-%d %H:%M')} | {len(conv.messages)}개"

    def _get_default_title(self) -> str:
        return "New chat"

    def _create_chat_messages(self, conversation) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=("user" if m.role == "user" else "assistant"), content=m.content
            )
            for m in conversation.messages
        ]

    def _create_info_text(self, conversation, title_override: str = None) -> str:
        title = title_override or conversation.title or self._get_default_title()
        return (
            f"**Current Chat:** {title}\n"
            f"**Created at:** {conversation.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            f"**Message count:** {len(conversation.messages)}"
        )

    def _rebuild_maps(self) -> list[tuple[str, str]]:
        """Rebuild label/id maps and return Dropdown choices as (label, value=id)."""
        if not hasattr(self.agent, "conversation_manager") or not self.agent.conversation_manager:
            return []
        convs = self.agent.conversation_manager.get_all_conversations() or []
        convs.sort(key=lambda x: x.updated_at, reverse=True)

        self._label_to_id.clear()
        self._id_to_label.clear()

        choices: list[tuple[str, str]] = []
        for c in convs:
            label = self._build_label(c)
            choices.append((label, c.id))
            self._label_to_id[label] = c.id
            self._id_to_label[c.id] = label
        return choices

    # ---------- Initialization ----------

    def _initialize_conversation_history(self) -> None:
        print("Initializing conversation history...")
        if getattr(self.agent, "enable_history", False) and hasattr(self.agent, "conversation_manager") and self.agent.conversation_manager:
            if not self.agent.conversation_manager.current_conversation_id:
                print("No current conversation, creating one...")
                self.agent.conversation_manager.start_new_conversation()

            # Ensure at least one conversation exists
            if not self.agent.conversation_manager.get_all_conversations():
                self.agent.conversation_manager.start_new_conversation()

            choices = self._rebuild_maps()  # [(label, id)]
            print(f"Initial conv count: {len(choices)}")
            if hasattr(self, "_conv_dropdown"):
                self._conv_dropdown.choices = choices
                cur_id = self.agent.conversation_manager.current_conversation_id
                self._conv_dropdown.value = (
                    cur_id if cur_id else (choices[0][1] if choices else None)
                )

            if self.agent.conversation_manager.current_conversation_id:
                cur = self.agent.conversation_manager.get_conversation(self.agent.conversation_manager.current_conversation_id)
                if cur:
                    self._update_current_conversation_info(cur)
        else:
            print("History not enabled")

    # ---------- App ----------

    def start(self) -> None:
        """Launch the Gradio app."""
        self.app.launch(
            server_name="0.0.0.0", server_port=7860, share=False, debug=True
        )

    async def _process_query(
        self, query: str, gr_messages: list[ChatMessage]
    ) -> AsyncGenerator[dict, None]:
        """Process the user's question and return the updated chat messages."""
        # Add user message
        gr_messages.append(ChatMessage(role="user", content=query))
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                value="", placeholder="Thinking...", interactive=False
            ),
        }

        # Drain stale progress
        while not self.progress_queue.empty():
            try:
                _ = self.progress_queue.get_nowait()
            except Exception:
                break

        # Run agent
        task = asyncio.create_task(self.agent.query(query))

        # Temp assistant message
        steps: list[str] = []
        idx = len(gr_messages)
        gr_messages.append(
            ChatMessage(
                role="assistant",
                content=self._compose_progress_message(
                    intermediate_steps=steps,
                    latest_step=None,
                    progress_label="Starting...",
                ),
            )
        )
        yield {self._chatbot: gr_messages}

        # Stream progress
        while not task.done():
            try:
                prog: DeepResearchProgress = await asyncio.wait_for(
                    self.progress_queue.get(), timeout=0.1
                )
                if not steps or steps[-1] != prog.progress_text:
                    steps.append(prog.progress_text)
                label = f"{prog.progress_percentage:.1%}"
                latest = steps[-1] if steps else None
                gr_messages[idx] = ChatMessage(
                    role="assistant",
                    content=self._compose_progress_message(
                        intermediate_steps=steps,
                        latest_step=latest,
                        progress_label=label,
                    ),
                )
                yield {self._chatbot: gr_messages}
            except asyncio.TimeoutError:
                pass

        # Finalize
        result = await task
        gr_messages[idx] = ChatMessage(
            role="assistant",
            content=self._compose_progress_message(
                intermediate_steps=steps,
                label_text="🎉🎊🥳 Research complete.",
                latest_step=None,
                progress_label="100%",
            ),
        )
        gr_messages.append(
            ChatMessage(role="assistant", content="---\n" + result.final_output_as(str))
        )
        yield {
            self._chatbot: gr_messages,
            self._chat_message: gr.update(
                placeholder="Ask a question", interactive=True
            ),
        }

    # ---------- UI composition helpers ----------

    def _compose_progress_message(
        self,
        *,
        intermediate_steps: list[str],
        label_text: str = f"{SPINNER_HTML} Drafting answer...",
        latest_step: str | None,
        progress_label: str | None,
    ) -> str:
        title = "Intermediate steps" + (
            f" — {progress_label}" if progress_label else ""
        )
        steps_md = self._format_intermediate_steps(intermediate_steps)
        details = (
            f"<details>\n"
            f"  <summary><strong>{title}</strong> (click to expand)</summary>\n\n"
            f"{steps_md}\n"
            f"</details>"
        )
        drafting = label_text
        if latest_step:
            safe = latest_step.replace("```", "\u0060\u0060\u0060")
            drafting = f"{drafting}\n\n{safe}"
        return f"{details}\n\n{drafting}"

    def _format_intermediate_steps(self, steps: list[str]) -> str:
        if not steps:
            return "No intermediate steps yet."
        return "\n".join(
            f"{i}. {s.replace('```', '\u0060\u0060\u0060')}"
            for i, s in enumerate(steps, 1)
        )

    # ---------- GUI ----------

    def _create_gui(self) -> gr.Blocks:
        with gr.Blocks(title="DRAFT - Deep Research Agent Framework Toolkit") as app:
            gr.HTML(
                """
                <style>
                  #chat-message::placeholder { color:#4A5568; opacity:1; }
                  .sidebar button { width:100% !important; margin-bottom:12px; padding:8px 16px; font-size:14px; border-radius:6px; }
                  .sidebar .gr-form { width:100%; }
                  .sidebar .gr-markdown { margin-bottom:16px; margin-top:20px; }
                  .sidebar .gr-textbox { width:100% !important; margin-bottom:12px; }
                  .sidebar > div { margin-bottom:20px; }
                  .sidebar button[data-variant="primary"] { background-color:#3b82f6; border-color:#3b82f6; }
                  .sidebar button[data-variant="secondary"] { background-color:#6b7280; border-color:#6b7280; }
                  .sidebar button[data-variant="stop"] { background-color:#ef4444; border-color:#ef4444; }
                </style>
                """
            )
            gr.Markdown("# Deep Research Agent")
            gr.Markdown(
                "A multi-agent research framework that uses AI agents to plan, search, synthesize, and critique to produce high-quality research answers with live progress and observability."
            )

            with gr.Row():
                # ---------- Sidebar only ----------
                with gr.Column(scale=1, min_width=340, elem_classes=["sidebar"]):
                    self._new_btn = gr.Button("Start New Chat", variant="primary")

                    gr.Markdown("### 📚 Chat List")
                    self._conv_dropdown = gr.Dropdown(
                        label="Select a chat",
                        choices=[],  # [(label, id)]
                        value=None,  # id
                        interactive=True,
                    )

                    self._delete_btn = gr.Button("Delete Selected Chat", variant="stop")

                    gr.Markdown("### 🔍 Search Chat")
                    self._search_query = gr.Textbox(
                        placeholder="Search chat", lines=1, show_label=False
                    )
                    self._search_btn = gr.Button("Search", variant="secondary")
                    self._search_dropdown = gr.Dropdown(
                        label="Select from search results",
                        choices=[],  # [(label, id)]
                        value=None,  # id
                        interactive=True,
                        visible=False,
                    )

                    self._current_info = gr.Markdown("**Current Chat:** New Chat")

                # ---------- Main chat ----------
                with gr.Column(scale=3):
                    self._chatbot = gr.Chatbot(
                        type="messages", label="Agent", height=700
                    )
                    self._chat_message = gr.Textbox(
                        elem_id="chat-message",
                        placeholder="Enter your question",
                        show_label=False,
                        lines=1,
                        submit_btn=True,
                    )

            # ---------- Events ----------
            # Send message
            self._chat_message.submit(
                self._process_query,
                [self._chat_message, self._chatbot],
                [self._chatbot, self._chat_message],
            )

            # New chat
            self._new_btn.click(
                self._create_new_conversation,
                outputs=[
                    self._chatbot,
                    self._chat_message,
                    self._current_info,
                    self._conv_dropdown,
                ],
            )

            # Select chat (Dropdown -> value=id)
            self._conv_dropdown.change(
                self._switch_conversation,  # def _switch_conversation(self, conv_id: str)
                inputs=[self._conv_dropdown],  # value=id
                outputs=[self._chatbot, self._chat_message, self._current_info],
            )

            # Search
            self._search_btn.click(
                self._search_conversation_history,
                inputs=[self._search_query],
                outputs=[self._search_dropdown, self._search_query],
            )

            # Select search result → switch + sync main dropdown
            self._search_dropdown.change(
                self._switch_conversation_and_sync,  # def _switch_conversation_and_sync(self, conv_id: str)
                inputs=[self._search_dropdown],
                outputs=[
                    self._chatbot,
                    self._chat_message,
                    self._current_info,
                    self._conv_dropdown,
                ],
            )

            # Delete
            self._delete_btn.click(
                self._delete_current_conversation,
                outputs=[
                    self._chatbot,
                    self._chat_message,
                    self._current_info,
                    self._conv_dropdown,
                ],
            )

            return app

    # ---------- Sidebar actions ----------
    def _search_conversation_history(self, query: str) -> tuple[gr.update, gr.update]:
        """Search conversation history and update the search dropdown.

        Returns
        -------
        tuple[gr.update, gr.update]
            First: update for search dropdown (choices as (label, id),
            clears value, sets visible).
            Second: update for search textbox (clears its value).
        """
        if not hasattr(self.agent, "search_history") or not query.strip():
            return gr.update(visible=False), gr.update()

        results = self.agent.conversation_manager.search_history(query.strip()) or []
        if not results:
            return gr.update(visible=False), gr.update()

        choices = [(self._build_label(c), c.id) for c in results]

        return gr.update(choices=choices, value=None, visible=True), gr.update(value="")

    def _update_conv_dropdown(self, select_id: str | None = None) -> gr.update:
        choices = self._rebuild_maps()  # [(label, id)]
        if select_id:
            value = select_id
        elif self.agent.current_conversation_id:
            value = self.agent.current_conversation_id
        else:
            value = choices[0][1] if choices else None
        return gr.update(choices=choices, value=value)

    def _create_new_conversation(self) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        if not hasattr(self.agent, "conversation_manager") or not self.agent.conversation_manager:
            return gr.update(), gr.update(), gr.update(), gr.update()
        print("Creating new conversation...")
        new_id = self.agent.conversation_manager.start_new_conversation()
        print("Created:", new_id)

        info = self._create_info_text(
            self.agent.conversation_manager.get_conversation(new_id),
            title_override=self._get_default_title(),
        )

        return (
            gr.update(value=[]),  # chatbot clear
            gr.update(value=""),  # textbox clear
            gr.update(value=info),  # info
            self._update_conv_dropdown(select_id=new_id),  # dropdown updated & selected
        )

    def _switch_conversation_and_sync(self, conv_id: str) -> tuple:
        """Switch from search dropdown then sync main dropdown selection."""
        updates = self._switch_conversation(conv_id)  # 3 outputs
        dd_update = self._update_conv_dropdown(select_id=conv_id)
        return (*updates, dd_update)

    def _delete_current_conversation(self) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        if not hasattr(self.agent, "conversation_manager") or not self.agent.conversation_manager:
            return gr.update(), gr.update(), gr.update(), gr.update()
        cur_id = self.agent.conversation_manager.current_conversation_id
        if not cur_id:
            return gr.update(), gr.update(), gr.update(), gr.update()

        current = self.agent.conversation_manager.get_conversation(cur_id)
        if not current:
            return gr.update(), gr.update(), gr.update(), gr.update()

        print("Deleting conversation:", current.title)
        if hasattr(self.agent, "history") and self.agent.history:
            _ = self.agent.history.delete_conversation(cur_id)

        # pick next
        convs = self.agent.conversation_manager.get_all_conversations() or []
        remaining = [c for c in convs if c.id != cur_id]
        if remaining:
            remaining.sort(key=lambda x: x.updated_at, reverse=True)
            self.agent.conversation_manager.switch_conversation(remaining[0].id)
            new_current = remaining[0]
        else:
            self.agent.conversation_manager.start_new_conversation()
            new_current = self.agent.conversation_manager.get_conversation(
                self.agent.conversation_manager.current_conversation_id
            )

        if new_current:
            chats = self._create_chat_messages(new_current)
            info = self._create_info_text(new_current)
            new_id = new_current.id
        else:
            chats = []
            info = self._create_info_text(
                None, title_override=self._get_default_title()
            )
            new_id = getattr(new_current, "id", None)

        return (
            gr.update(value=chats),
            gr.update(value=""),
            gr.update(value=info),
            self._update_conv_dropdown(select_id=new_id),
        )

    # ---------- Core switching ----------

    def _switch_conversation(self, conv_id: str) -> tuple[gr.update, gr.update, gr.update]  :
        if not hasattr(self.agent, "conversation_manager") or not self.agent.conversation_manager or not conv_id:
            return gr.update(), gr.update(), gr.update()

        print("Switching to conversation:", conv_id)
        ok = self.agent.conversation_manager.switch_conversation(conv_id)
        if not ok:
            return gr.update(), gr.update(), gr.update()

        conv = self.agent.conversation_manager.get_conversation(conv_id)
        if not conv:
            return gr.update(), gr.update(), gr.update()

        chats = self._create_chat_messages(conv)
        self._update_current_conversation_info(conv)

        info = self._create_info_text(conv)
        return gr.update(value=chats), gr.update(value=""), gr.update(value=info)

    def _update_current_conversation_info(self, conversation) -> None:
        if hasattr(self, "_current_info") and conversation:
            info = self._create_info_text(conversation)
            self._current_info.value = info

    def _on_title_updated(self, conversation_id: str, new_title: str) -> None:
        """Handle title update event for a conversation."""
        print(f"Title updated for conversation {conversation_id}: {new_title}")

        # 대화 목록 새로고침
        self._rebuild_maps()

        # 현재 대화 정보 업데이트
        if self.agent.conversation_manager.current_conversation_id == conversation_id:
            current_conv = self.agent.conversation_manager.get_conversation(conversation_id)
            if current_conv:
                self._update_current_conversation_info(current_conv)

        # 드롭다운 업데이트
        if hasattr(self, "_conv_dropdown"):
            self._conv_dropdown.choices = self._rebuild_maps()
            if self.agent.conversation_manager.current_conversation_id:
                self._conv_dropdown.value = self.agent.conversation_manager.current_conversation_id


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for the GUI application."""
    setup_langfuse_tracer()
    gui_app = DeepResearchAgentGUI(cfg)
    gui_app.start()


if __name__ == "__main__":
    main()
