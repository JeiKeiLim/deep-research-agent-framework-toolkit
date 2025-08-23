# Creating Your Own Agent

One of the key features of the DRAFT framework is the ability to easily create and integrate your own custom agents. This guide will walk you through the process of creating a new agent using a simple YAML configuration.

## 1. Create a Configuration File

The first step is to create a new YAML configuration file for your agent in the `configs/agents/{your_agent_name}` directory. For example, let's create a new agent called `my_agent`:

```yaml
# configs/agents/my_agent/main.yaml

defaults:
  - ../mcp_servers/tavily_search@configs.mcp_servers  # Optional, specify the MCP server to use
  - _self_

name: "Main"
description: "This is my custom agent."
configs:
  model: "gemini-1.5-pro-latest"
  prompt: prompts/my_agent/main.txt  # This can be both a file path or a string
  function_tools:  # Optional. Function tools are defined at src/draft_agents/function_tools/__init__.py
    - perplexity_search
```

In this configuration, you define the main agent's name, description, LLM settings and prompt file.
You can also specify any sub-agents by adding additional configuration files in the same directory (e.g., `configs/agents/my_agent/my_sub_agent.yaml`).

## 2. Organize Prompt Files

Next, create a new directory for your main agent's prompt files under `prompts/` (e.g., `prompts/my_agent/`). This directory will contain the prompt files for your main agent and any sub-agents within that configuration.

For our example, we'll create `prompts/my_agent/main.txt`:

```
You are the main orchestrator for the 'My Agent' configuration. Your task is to...
```

This file contains the prompt that will be used to instruct your main agent on its task. Similarly, you would create prompt files for any sub-agents (e.g., `prompts/my_agent/my_sub_agent.txt`).

## 3. Define an Output Type (Optional)

The output type of an agent is determined automatically based on the agent's `name` field in its configuration. The framework looks for a Pydantic model in `src/draft_agents/output_types/` with a class name that matches the agent's `name`.

**Important:** The `name` field in your agent's YAML configuration **must exactly match** the PascalCase class name of the Pydantic model defined in `src/draft_agents/output_types/__init__.py`. If a matching Pydantic model is not found, or if the names do not match, the agent's output type will default to `str`.

To define a custom output type, you need to:

1.  **Create a Pydantic model** in a new Python file in `src/draft_agents/output_types/`. For example, if your agent's `name` is `MyAgent`, create `src/draft_agents/output_types/my_agent.py`:

    ```python
    from pydantic import BaseModel, Field

    class MyAgentOutput(BaseModel):
        result: str = Field(description="The result of the agent's task.")
    ```

2.  **Import the new model** in `src/draft_agents/output_types/__init__.py`:

    ```python
    # src/draft_agents/output_types/__init__.py

    from src.draft_agents.output_types.critic import CriticFeedback
    from src.draft_agents.output_types.evaluator import EvaluatorFeedback
    from src.draft_agents.output_types.planner import SearchItem, SearchPlan
    from src.draft_agents.output_types.my_agent import MyAgentOutput # Add this line

    output_types = {
        "Planner": SearchPlan,
        "Critic": CriticFeedback,
        "Evaluator": EvaluatorFeedback,
        "MyAgent": MyAgentOutput,  # Add your new agent here, ensuring name matches
    }
    ```

Now, when you run your agent, the framework will automatically use the specified Pydantic model as the output type, provided the names match.

## 4. Activate Your Main Agent Configuration

To use your newly created main agent configuration, you need to activate it in the `configs/config.yaml` file. This file acts as the central configuration for the entire framework.

Locate the `defaults` section in `configs/config.yaml` and modify the entry for the main agent to point to your new main agent's `main.yaml` file. For example:

```yaml
# configs/config.yaml

defaults:
  # - agents/draft_agent/main@agents.Main # Comment out or remove the previous main agent
  - agents/my_agent/main@agents.Main # Uncomment or add your new main agent here
  # ... other defaults ...
```

By changing this line, you tell the framework to load and use the `main.yaml` from your `my_agent` directory as the primary orchestrator. You can easily switch between different main agent configurations by modifying this single line.

And that's it! You have successfully created and integrated your own custom main agent configuration into the DRAFT framework.
