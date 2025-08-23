# Welcome to DRAFT

DRAFT (Deep Research Agent Framework Toolkit) is a multi-agent research framework that uses AI agents to plan, search, synthesize, and critique to produce high-quality research answers with live progress and observability.

This documentation provides a comprehensive guide to understanding, using, and extending the DRAFT framework.

## Key Features

- **Modular Multi-Agent System**: Specialized agents for planning, searching, synthesizing, and critiquing.
- **Agent Builder Web Application**: Create and manage custom agents with an intuitive web interface.
- **Extensible and Composable**: Easily create new agents and tools to extend the framework's capabilities.
- **Live Progress and Observability**: Monitor the research process in real-time with Langfuse tracing.
- **Multi-Source Knowledge Retrieval**: Integrates with various knowledge sources like Weaviate, Perplexity, and Tavily.
- **Conversation History**: Persistently stores and retrieves past conversations for context-aware research.

## Getting Started

To get started with DRAFT, check out the following sections:

- **[Architecture](architecture.md)**: Learn about the modular design of the framework.
- **[Agent Builder](agent-builder.md)**: Complete guide to the web-based agent creation interface.
- **[Creating Your Own Agent](creating-agents.md)**: A step-by-step guide to creating your own agents using simple YAML configuration or the Agent Builder web interface.
- **[Contributing](contributing.md)**: Find out how you can contribute to the DRAFT project.

### Quick Start

1. **Run the Agent Builder**: Start the web-based agent creation tool at `http://localhost:7861`
2. **Create Your First Agent**: Use the intuitive interface to design your custom agent
3. **Run Research**: Use your new agent with the research GUI at `http://localhost:7860`
