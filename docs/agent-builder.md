# Agent Builder Web Application

The Agent Builder is a modern, web-based interface for creating and managing custom agents in the DRAFT framework. It provides an intuitive way to design multi-agent systems without manually editing YAML configuration files.

## Overview

The Agent Builder is a Flask-based web application that offers a visual interface for agent creation and management. It features a modern, responsive design built with Tailwind CSS and provides real-time validation and feedback.

## Getting Started

### Starting the Agent Builder

```bash
PYTHONPATH=. uv run --env-file .env apps/agent-builder/app.py
```

The application will be available at `http://localhost:7861`.

### Interface Overview

The Agent Builder consists of four main sections:

1. **Browse Agents**: View and manage existing agents
2. **Create Agent**: Build new agents from scratch
3. **Edit Agent**: Modify existing agent configurations
4. **Delete Agent**: Remove agents from the system

## Creating Agents

### Step 1: Basic Information

When creating a new agent, you'll need to provide:

- **Agent Name**: A unique identifier (used for file names and internal references)
- **Display Name**: A human-readable name for the agent
- **Description**: A brief description of the agent's purpose
- **Model**: Select from available LLM providers

### Step 2: Main Prompt

The main prompt defines the agent's behavior and instructions. This is where you specify:

- The agent's role and responsibilities
- How it should interact with users
- What tasks it should perform
- Any specific guidelines or constraints

### Step 3: Tools and Capabilities

Configure the agent's available tools:

#### Function Tools
- **kb_weaviate**: Knowledge base search using Weaviate
- **perplexity_search**: Web search via Perplexity API
- **tavily_search**: Web search via Tavily API

#### MCP Servers
- **tavily_search**: Tavily search via MCP
- **arxiv_search**: ArXiv paper search via MCP

### Step 4: Sub-Agents (Optional)

For complex workflows, you can add sub-agents that work together:

1. **Add Sub-Agent**: Click to add a new sub-agent
2. **Configure**: Set name, description, model, and prompt
3. **Nest**: Sub-agents can have their own sub-agents (up to 5 levels deep)

## Managing Agents

### Browse Agents

The browse page shows all existing agents with:
- Agent name and display name
- Description
- Model information
- Available tools
- Quick actions (edit/delete)

### Edit Agents

Editing an agent allows you to:
- Modify basic information
- Update the main prompt
- Add/remove tools and MCP servers
- Manage sub-agents
- Save changes without recreating the agent

### Delete Agents

Safely remove agents and their associated files:
- Configuration files
- Prompt files
- Directory structures

## Advanced Features

### Real-time Validation

The Agent Builder provides instant feedback on:
- Required field validation
- Configuration format checking
- Tool availability verification
- Model compatibility

### Automatic File Management

The application automatically handles:
- Creating proper directory structures in `configs/agents/`
- Generating YAML configuration files
- Organizing prompt files in `prompts/`
- Setting up sub-agent configurations
- Maintaining file consistency

### Configuration Structure

The Agent Builder creates the following structure for each agent:

```
configs/agents/{agent_name}/
├── main.yaml              # Main agent configuration
├── sub_agent1.yaml        # Sub-agent configurations
├── sub_agent2.yaml
└── ...

prompts/{agent_name}/
├── main.txt               # Main agent prompt
├── sub_agent1.txt         # Sub-agent prompts
├── sub_agent2.txt
└── ...
```

## Best Practices

### Agent Design

1. **Clear Purpose**: Define a specific, focused role for each agent
2. **Descriptive Names**: Use meaningful names that indicate the agent's function
3. **Comprehensive Prompts**: Write detailed prompts that cover edge cases
4. **Tool Selection**: Only include tools that the agent actually needs

### Sub-Agent Organization

1. **Hierarchical Structure**: Organize sub-agents in logical hierarchies
2. **Separation of Concerns**: Each sub-agent should have a distinct responsibility
3. **Communication Flow**: Consider how agents will communicate and coordinate
4. **Depth Limits**: Avoid nesting more than 3-4 levels deep

The Agent Builder simplifies the process of creating custom research agents while maintaining the full power and flexibility of the underlying DRAFT framework.
