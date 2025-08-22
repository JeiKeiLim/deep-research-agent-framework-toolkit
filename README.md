# DRAFT (Deep Research Agent Framework Toolkit)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

A multi-agent research framework that uses AI agents to plan, search, synthesize, and critique to produce high-quality research answers with live progress and observability.

## Overview

DRAFT demonstrates how to build a sophisticated, composable AI agent system for deep research tasks. Specialized agents collaborate to:

1. Plan research strategies
2. Search multiple knowledge sources and tools
3. Synthesize evidence into comprehensive answers
4. Critique and iteratively improve results

## Architecture

### Multi-Agent System

- **Planner**: Produces a structured search plan
- **Search**: Executes searches via function tools and MCP servers
- **Synthesizer**: Writes the final answer from evidence
- **Critic**: Evaluates answer quality and suggests revisions

### Key Components

- **Agent framework**: OpenAI Agents SDK with custom output types
- **Knowledge base**: Weaviate vector DB (default collection: `enwiki_20250520`)
- **Web search**: Perplexity API and Tavily
- **MCP integration**: External servers (e.g., Tavily via `mcp-remote`, ArXiv MCP server)
- **Observability**: Langfuse tracing (OpenTelemetry + Logfire instrumentation)
- **GUI**: Gradio-based web interface

## Features

- **Agent orchestration modes**:
  - `agent` (default): single orchestrator agent with streaming updates
  - `sequential`: explicit plan → search → synthesize → critique with up to `max_revision` iterations
- **Multi-source retrieval**: Weaviate + Perplexity + Tavily
- **MCP support**: Connect to external capability servers via stdio (see MCP section)
- **Structured outputs**: Pydantic models; safe type conversions
- **Async-ready**: Concurrent tool calls with rate limiting
- **Tracing**: Langfuse traces linked to evaluations
- **Conversation History**: Persistent storage and retrieval of past conversations for context-aware research

## Installation

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- Node.js + `npx` (required to run `mcp-remote` for some MCP servers)

### Setup

1. Clone and install:

```bash
git clone https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit.git
cd deep-research-agent-framework-toolkit
uv sync
```

2. Configure environment variables by copying `.env.example`:

```bash
cp .env.example .env
# Fill in values in .env
```

## Configuration

Set the following environment variables (see `.env.example`):

- **OPENAI-compatible LLM**:
  - `OPENAI_BASE_URL` (e.g., OpenAI-compatible endpoint)
  - `OPENAI_API_KEY`
- **Embeddings** (Cloudflare Workers AI or compatible):
  - `EMBEDDING_BASE_URL`
  - `EMBEDDING_API_KEY`
- **Langfuse**:
  - `LANGFUSE_PUBLIC_KEY` (must start with `pk-lf-`)
  - `LANGFUSE_SECRET_KEY` (must start with `sk-lf-`)
  - `LANGFUSE_HOST` (default `https://us.cloud.langfuse.com`)
- **Weaviate**:
  - `WEAVIATE_HTTP_HOST`, `WEAVIATE_GRPC_HOST`, `WEAVIATE_API_KEY`
  - `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_PORT` (443 for cloud; 8080/50051 locally)
  - `WEAVIATE_HTTP_SECURE`, `WEAVIATE_GRPC_SECURE` (true/false)
- **Perplexity**: `PERPLEXITY_API_KEY`
- **Tavily**: `TAVILY_API_KEY`

Hydra config lives in `configs/config.yaml` and composes agent configs from `configs/agents/*.yaml`. Notable toggles:

- `agent_configs.max_revision`: number of sequential-mode revisions (default: 3)
- `agent_configs.orchestration_mode`: `agent` or `sequential`
- `agent_configs.enable_history`: enable conversation history (default: true)
- `agent_configs.history_storage_dir`: directory for storing conversation files (default: "conversation_history")

You can also override Hydra values via CLI flags if desired.

## Usage

### Run the GUI

```bash
PYTHONPATH=. uv run --env-file .env apps/gui.py
```

Visit `http://localhost:7860`.

### Run the CLI

The Deep Research Agent can also be interacted with via a command-line interface.

#### Interactive Mode

To start an interactive chat session:

```bash
PYTHONPATH=. uv run --env-file .env apps/cli.py
```

Type your query and press Enter. Type `exit` or `quit` to end the session.

#### One-Time Query Mode

To run a single query and get the answer directly, use the `+query` argument:

```bash
PYTHONPATH=. uv run --env-file .env apps/cli.py +query="Your research question here""
```

**Note:** The `+` prefix before `query` is important. It tells Hydra (the configuration manager) to add `query` to the configuration, as it's not defined in the default configuration files.

#### Overriding Configurations

You can override any configuration parameter defined in the `configs/` directory directly from the command line. For example, to enable conversation history:

```bash
PYTHONPATH=. uv run --env-file .env apps/cli.py +query="Your question" agent_configs.enable_history=true
```

### Conversation History Features

The GUI now includes conversation history management:

- **New Conversation**: Start a fresh research session
- **Conversation Selection**: Switch between different conversation threads
- **History Search**: Search through past conversations by content
- **Context Awareness**: Agents can reference previous conversations for better continuity

### Switch orchestration mode

Edit `configs/config.yaml` or pass a Hydra override, e.g.:

```bash
PYTHONPATH=. uv run --env-file .env apps/gui.py agent_configs.orchestration_mode=sequential
```

## MCP servers

The Search agent can connect to MCP servers defined in `configs/mcp_servers/*.yaml`.

- Included configs:
  - `tavily_search.yaml` → launches Tavily via `npx mcp-remote https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}`
  - `arxiv_search.yaml` → launches `arxiv-mcp-server` via `uv tool run`

Requirements:

- Node.js + `npx` for `mcp-remote`
- Install the ArXiv MCP server (optional pre-install to avoid on-demand install):

```bash
uv tool install arxiv-mcp-server
```

## Project structure

```
deep-research-agent-framework-toolkit/
├── apps/
│   └── gui.py                      # Gradio web interface
├── configs/
│   ├── agents/                     # Agent configs (Main, Planner, Search, Synthesizer, Critic, Evaluator)
│   ├── mcp_servers/                # MCP server definitions (Tavily, ArXiv)
│   └── config.yaml                 # Root Hydra config
├── evals/
│   ├── datasets/test_dataset.jsonl # Example dataset
│   ├── run_evaluation.py           # LLM-as-a-judge evaluation pipeline
│   └── upload_data.py              # Upload dataset to Langfuse
├── prompts/                        # Prompts (main, planner, search, synthesizer, critic, evaluator)
├── src/
│   ├── draft_agents/
│   │   ├── agent.py                # Agent orchestration + MCP wiring
│   │   └── function_tools/         # Tooling: Weaviate, Perplexity, Tavily
│   └── utils/                      # Env, tracing, gradio helpers, conversation history, etc.
├── tests/                          # Unit/integration tests
└── pyproject.toml                  # Dependencies and tooling config
```

## Evaluation

Upload dataset to Langfuse:

```bash
PYTHONPATH=. uv run --env-file .env evals/upload_data.py \
  --source_dataset ./evals/datasets/test_dataset.jsonl \
  --langfuse_dataset_name test_questions
```

Run evaluation with LLM-as-a-judge and record scores back to Langfuse:

```bash
PYTHONPATH=. uv run --env-file .env evals/run_evaluation.py
```

## Development

- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`
- **Test**: `PYTHONPATH=. uv run --env-file .env pytest`
- **Pre-commit**: `uv run pre-commit run --all-files`

### Adding new agents

1. Create config in `configs/agents/`
2. Add prompt in `prompts/`
3. Add output types in `src/draft_agents/output_types/` (skip if `str`)
4. Reference from `configs/agents/main.yaml` if part of Main

### Adding new tools

1. Implement in `src/draft_agents/function_tools/`
2. Register in `src/draft_agents/function_tools/__init__.py`
3. Add to agent config (e.g., `Search`)

## Technologies used

- **Agent framework**: OpenAI Agents SDK
- **Models**: Gemini 2.5 (via OpenAI-compatible endpoint)
- **Vector DB**: Weaviate
- **Search**: Perplexity API, Tavily
- **Embeddings**: Cloudflare AI (e.g., `@cf/baai/bge-m3`)
- **Observability**: Langfuse (OTLP via OpenTelemetry)
- **UI**: Gradio
- **Config**: Hydra + OmegaConf
- **Typing/validation**: Pydantic

## Troubleshooting

- Langfuse keys must start with `pk-lf-`/`sk-lf-` or startup will fail.
- Weaviate connection requires matching host/port/secure flags; local dev often uses `8080`/`50051` and `false` for secure flags.
- Ensure `OPENAI_BASE_URL`/`OPENAI_API_KEY` are set; the Agents SDK relies on them.
- MCP servers require Node.js (`npx`) and network access to fetch `mcp-remote`.

## Contributing

Contributions are welcome. Please open an issue or pull request.

## Disclaimer

This is an educational project for learning about AI agents and multi-agent systems. The name "DRAFT" stands for "Deep Research Agent Framework Toolkit".

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://limjk.ai"><img src="https://avatars.githubusercontent.com/u/10356193?v=4?s=100" width="100px;" alt="Jongkuk Lim"/><br /><sub><b>Jongkuk Lim</b></sub></a><br /><a href="https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit/commits?author=JeiKeiLim" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RijuKim"><img src="https://avatars.githubusercontent.com/u/93368654?v=4?s=100" width="100px;" alt="Rijoo"/><br /><sub><b>Rijoo</b></sub></a><br /><a href="https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit/commits?author=RijuKim" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
