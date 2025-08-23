# Creating Your Own Agent

One of the key features of the DRAFT framework is the ability to easily create and integrate your own custom agents. This guide will walk you through the process of creating a new agent using a simple YAML configuration.

## 1. Create a Configuration File

The first step is to create a new YAML configuration file for your agent in the `configs/agents/` directory. For example, let's create a new agent called `my_agent.yaml`:

```yaml
# configs/agents/my_agent.yaml

defaults:
  - ../mcp_servers/tavily_search@configs.mcp_servers  # Optional, specify the MCP server to use
  - _self_

name: "My Agent"
description: "This is my custom agent."
configs:
  model: "gemini-1.5-pro-latest"
  prompt: prompts/my_agent.txt  # This can be both a file path or a string
  function_tools:  # Optional. Function tools are defined at src/draft_agents/function_tools/__init__.py
    - perplexity_search
```

In this configuration, you define the agent's name, description, LLM settings and prompt file.

## 2. Create a Prompt File

Next, create a new prompt file in the `prompts/` directory with the same name as specified in your agent's configuration. For our example, we'll create `prompts/my_agent.txt`:

```
You are a helpful assistant. Your task is to...
```

This file contains the prompt that will be used to instruct your agent on its task.

## 3. Define an Output Type (Optional)

The output type of an agent is determined automatically based on the agent's name. The framework looks for a Pydantic model in `src/draft_agents/output_types/` with a class name that matches the agent's name in PascalCase. For example, if your agent's name is `my_agent`, the framework will look for a class named `MyAgent` in `src/draft_agents/output_types/__init__.py`.

If a matching Pydantic model is not found, the output type defaults to `str`.

To define a custom output type, you need to:

1.  **Create a Pydantic model** in a new Python file in `src/draft_agents/output_types/`. For example, `src/draft_agents/output_types/my_agent.py`:

    ```python
    from pydantic import BaseModel, Field

    class MyAgent(BaseModel):
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
        "MyAgent": MyAgentOutput,  # Add your new agent here
    }
    ```

Now, when you run your agent, the framework will automatically use the `MyAgent` Pydantic model as the output type.

## 4. Integrate Your Agent

Finally, you need to integrate your new agent into the framework. This is done in two steps:

### Add to `configs/agents/main.yaml`

If you want your new agent to be a sub-agent of the `Main` agent, you need to add it to the `defaults` list in `configs/agents/main.yaml`:

```yaml
# configs/agents/main.yaml

defaults:
  - planner@sub_agents.Planner
  - synthesizer@sub_agents.Synthesizer
  - search@sub_agents.Search
  - critic@sub_agents.Critic
  - my_agent@sub_agents.MyAgent  # Add your new agent here
  - _self_

# ...
```

And that's it! You have successfully created and integrated your own custom agent into the DRAFT framework.
