# Architecture

The DRAFT framework is designed to be modular and extensible, allowing for the easy composition of AI agents to perform complex research tasks. The architecture is centered around a multi-agent system where each agent has a specialized role.

## Multi-Agent System

The core of DRAFT is its multi-agent system, which consists of the following specialized agents:

- **Planner**: This agent is responsible for creating a structured research plan based on the user's query. It breaks down the research task into smaller, manageable steps.
- **Search**: The search agent executes the research plan by querying various knowledge sources and tools. It can interact with knowledge bases like Weaviate, and web search services like Perplexity and Tavily.
- **Synthesizer**: This agent takes the evidence gathered by the search agent and synthesizes it into a comprehensive and coherent answer.
- **Critic**: The critic agent evaluates the quality of the synthesized answer and provides feedback for improvement. This iterative process ensures the final output is of high quality.

## Key Components

DRAFT is built upon a set of key components that provide the foundation for its functionality:

- **Agent Framework**: We use the OpenAI Agents SDK as the backbone of our agent framework, with custom output types to handle structured data.
- **Knowledge Base**: DRAFT uses Weaviate as its default vector database for knowledge storage and retrieval.
- **Web Search**: The framework integrates with Perplexity API and Tavily for real-time web search capabilities.
- **MCP Integration**: DRAFT supports external capability servers through the Multi-Agent Capability Protocol (MCP). This allows for the integration of external tools and services.
- **Observability**: We use Langfuse for tracing and observability, providing real-time insights into the agent's operations.
- **GUI**: A Gradio-based web interface is provided for easy interaction with the framework.

## Agent Orchestration

DRAFT supports two modes of agent orchestration:

- **`agent` mode (default)**: In this mode, a single orchestrator agent manages the entire research process with streaming updates.
- **`sequential` mode**: This mode follows a more explicit workflow of plan → search → synthesize → critique, with a configurable number of revision iterations. (We are planning to deprecate this mode in the future.)

This modular architecture allows for the flexible composition of agents and tools, making it easy to extend and adapt the framework to different research needs.
