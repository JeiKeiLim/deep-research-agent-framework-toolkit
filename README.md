# DRAFT (Deep Research Agent Framework Toolkit)

A multi-agent research framework that uses AI agents to perform comprehensive research tasks through planning, searching, synthesizing, and critiquing.

## Overview

DRAFT is an educational project that demonstrates how to build a sophisticated AI agent system for deep research tasks. The framework uses multiple specialized agents that work together to:

1. **Plan** research strategies
2. **Search** multiple knowledge sources
3. **Synthesize** information into comprehensive answers
4. **Criticize** and improve the results through iterative refinement

## Architecture

### Multi-Agent System

The framework consists of four main agents:

- **Planner Agent**: Breaks down research queries into executable search steps
- **Search Agent**: Performs searches using multiple tools (Weaviate knowledge base, Perplexity API)
- **Synthesizer Agent**: Combines evidence into comprehensive, well-structured answers
- **Critic Agent**: Evaluates answers and provides feedback for improvement

### Key Components

- **Agent Framework**: Built on OpenAI's agents framework with custom output types
- **Knowledge Base**: Weaviate vector database with Wikipedia dump (cutoff: May 2025)
- **Web Search**: Perplexity API integration for current information
- **Observability**: Langfuse integration for tracing and monitoring
- **GUI**: Gradio-based web interface for interactive research

## Features

- **Iterative Refinement**: Agents can revise answers based on critic feedback (up to 3 revisions)
- **Multi-Source Search**: Combines knowledge base and web search capabilities
- **Progress Tracking**: Real-time progress updates during research
- **Structured Output**: Type-safe agent outputs using Pydantic models
- **Async Processing**: Full async support for concurrent operations
- **Comprehensive Logging**: Detailed tracing with Langfuse

## Installation

### Prerequisites

- Python 3.12+
- UV package manager
- Environment variables configured (see Configuration section)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit.git
cd deep-research-agent-framework-toolkit
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables (see Configuration section)

## Configuration

The project requires several environment variables to be configured:

### Required Environment Variables
- Make sure to copy `.env.example` to `.env` and modify accordingly.

```bash
# Embeddings API
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=your_embedding_base_url

# Weaviate
WEAVIATE_HTTP_HOST=your_weaviate_host
WEAVIATE_GRPC_HOST=your_weaviate_host
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_HTTP_PORT=443
WEAVIATE_GRPC_PORT=443
WEAVIATE_HTTP_SECURE=true
WEAVIATE_GRPC_SECURE=true

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Perplexity
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## Usage

### Running the GUI

Start the Gradio web interface:

```bash
PYTHONPATH=. uv run --env-file .env apps/gui.py
```

The GUI will be available at `http://localhost:7860`

## Project Structure

```
deep-research-agent-framework-toolkit/
├── apps/
│   └── gui.py                 # Gradio web interface
├── configs/
│   ├── agents/                # Agent configurations
│   │   ├── critic.yaml
│   │   ├── main.yaml
│   │   ├── planner.yaml
│   │   ├── search.yaml
│   │   └── synthesizer.yaml
│   └── config.yaml           # Main configuration
├── outputs/                   # Generated outputs
├── prompts/                   # Agent prompt templates
│   ├── critic.txt
│   ├── planner.txt
│   ├── search.txt
│   └── synthesizer.txt
├── src/
│   ├── draft_agents/         # Core agent implementation
│   │   ├── agent.py         # Main agent class
│   │   ├── function_tools/   # Search and knowledge tools
│   │   └── output_types/    # Pydantic models for outputs
│   └── utils/               # Utility modules
│       ├── async_utils.py
│       ├── env_vars.py
│       ├── gradio/
│       ├── langfuse/
│       └── logging.py
└── tests/                   # Test suite
    └── tool_tests/
```

## Development

### Code Quality

- **Formatting**: `uv run ruff format .`
- **Linting**: `uv run ruff check .`
- **Testing**: `uv run pytest`
- **Pre-commit hooks**: `uv run pre-commit run --all-files`

### Testing

Run the test suite:

```bash
PYTHONPATH=. uv run --env-file .env pytest
```

Run integration tests:

```bash
PYTHONPATH=. uv run --env-file .env pytest tests/tool_tests/test_integration.py -v
```

### Adding New Agents

1. Create agent configuration in `configs/agents/`
2. Add prompt template in `prompts/`
3. Define output types in `src/draft_agents/output_types/`
    - If desired output type is `str`, skip this.
4. Update the main agent configuration

### Adding New Tools

1. Implement tool in `src/draft_agents/function_tools/`
2. Register in `src/draft_agents/function_tools/__init__.py`
3. Update agent configurations to use the new tool

## Evaluation
### Upload dataset
```bash
PYTHONPATH=. uv run --env-file .env evals/upload_data.py --source_dataset ./evals/datasets/test_dataset.jsonl --langfuse_dataset_name test_questions
```

## Technologies Used

- **AI Framework**: OpenAI Agents Framework
- **LLM**: Gemini 2.5 Flash models
- **Vector Database**: Weaviate
- **Web Search**: Perplexity API
- **Embeddings**: Cloudflare AI embeddings
- **Observability**: Langfuse
- **Web Interface**: Gradio
- **Configuration**: Hydra + OmegaConf
- **Async Support**: asyncio, httpx
- **Type Safety**: Pydantic
- **Testing**: pytest, pytest-asyncio

## Contributing

This is an educational project for learning about AI agents. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Disclaimer

This is an educational project designed to learn about AI agents and multi-agent systems. The name "DRAFT" stands for "Deep Research Agent Framework Toolkit" and is intentionally verbose for educational clarity.
