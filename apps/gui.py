"""Deep Research Agent GUI environment.

This module sets up a Gradio interface for the Deep Research Agent GUI,
allowing users to interact with the agent framework through a web interface.

Author: Jongkuk Lim
Contact: lim.jeikei@gmail.com
"""

from typing import AsyncGenerator, List

import agents
import gradio as gr
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI


async_openai_client = AsyncOpenAI()

main_agent = agents.Agent(
    name="Main Agent",
    instructions="You are a helpful assistant that can answer questions and provide information.",
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
    output_type=None,
)


async def process_query(
    query: str, gr_messages: list[ChatMessage]
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

    # TODO: Implement the actual query processing logic here.
    main_result = await agents.Runner.run(main_agent, input=query)
    gr_messages.append(
        ChatMessage(role="assistant", content=main_result.final_output_as(str))
    )
    yield gr_messages


def app():
    """Create and return the Gradio app."""
    with gr.Blocks(title="OAI Agent SDK - Multi-agent") as app:
        gr.Markdown("# Deep Research Agent GUI")
        gr.Markdown(
            "This is a simple Gradio interface for the Deep Research Agent GUI."
        )
        chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
        chat_message = gr.Textbox(lines=1, label="Ask a question", submit_btn=True)
        chat_message.submit(process_query, [chat_message, chatbot], [chatbot])

        return app


if __name__ == "__main__":
    demo = app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
