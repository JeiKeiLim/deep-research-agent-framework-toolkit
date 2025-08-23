#!/usr/bin/env python3
"""Agent Builder Flask Application for DRAFT Framework.

A modern, professional web application for creating and managing
multi-agent configurations for the DRAFT research framework.
"""

import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, jsonify, redirect, render_template, request, url_for


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Change this in production

# Configuration
app.config["CONFIG_DIR"] = project_root / "configs"
app.config["PROMPTS_DIR"] = project_root / "prompts"


@dataclass
class SubAgent:
    """Data class for sub-agent configuration."""

    name: str
    description: str
    model: str
    prompt: str
    function_tools: list[str]
    mcp_servers: list[str]
    has_nested: bool = False
    sub_agents: list["SubAgent"] | None = None

    def __post_init__(self) -> None:
        """Initialize sub_agents if None."""
        if self.sub_agents is None:
            self.sub_agents = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["sub_agents"] = [agent.to_dict() for agent in self.sub_agents]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubAgent":
        """Create from dictionary representation."""
        sub_agents_data = data.pop("sub_agents", [])
        sub_agents = [cls.from_dict(sa) for sa in sub_agents_data]
        data["sub_agents"] = sub_agents
        return cls(**data)


@dataclass
class Agent:
    """Data class for agent configuration."""

    name: str
    display_name: str
    description: str
    model: str
    main_prompt: str
    function_tools: list[str]
    mcp_servers: list[str]
    sub_agents: list[SubAgent]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["sub_agents"] = [agent.to_dict() for agent in self.sub_agents]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """Create from dictionary representation."""
        sub_agents_data = data.pop("sub_agents", [])
        sub_agents = [SubAgent.from_dict(sa) for sa in sub_agents_data]
        data["sub_agents"] = sub_agents
        return cls(**data)


class AgentManager:
    """Manages agent operations and configurations."""

    def __init__(self, config_dir: Path, prompts_dir: Path) -> None:
        """Initialize the agent manager."""
        self.config_dir = config_dir
        self.prompts_dir = prompts_dir
        self.agents_dir = config_dir / "agents"
        self.agents_dir.mkdir(exist_ok=True)

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
        ]

    def get_available_tools(self) -> list[str]:
        """Get list of available function tools."""
        # Read from actual function_tools module
        return ["kb_weaviate", "perplexity_search", "tavily_search"]

    def get_available_mcp_servers(self) -> list[str]:
        """Get list of available MCP servers."""
        # Read from actual MCP server configs
        mcp_servers = []
        mcp_dir = self.config_dir.parent / "configs" / "mcp_servers"
        if mcp_dir.exists():
            for config_file in mcp_dir.glob("*.yaml"):
                server_name = config_file.stem
                mcp_servers.append(server_name)
        return mcp_servers if mcp_servers else ["tavily_search", "arxiv_search"]

    def get_existing_agents(self) -> list[dict[str, str]]:
        """Get list of existing agents."""
        agents = []
        for agent_dir in self.agents_dir.iterdir():
            if agent_dir.is_dir():
                config_file = agent_dir / "main.yaml"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = yaml.safe_load(f)
                            agents.append(
                                {
                                    "name": agent_dir.name,
                                    "display_name": config.get(
                                        "display_name", agent_dir.name
                                    ),
                                }
                            )
                    except Exception:
                        continue
        return agents

    def get_agent_details(self, agent_name: str) -> dict[str, Any] | None:
        """Get detailed information about an agent."""
        agent_dir = self.agents_dir / agent_name
        if not agent_dir.exists():
            return None

        try:
            # Load main configuration
            config_file = agent_dir / "main.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Load main prompt
            prompt_file = self.prompts_dir / agent_name / "main.txt"
            main_prompt = ""
            if prompt_file.exists():
                with open(prompt_file) as f:
                    main_prompt = f.read()

            # Load sub-agents
            sub_agents = self._load_sub_agents_recursive(
                agent_dir, self.prompts_dir / agent_name
            )

            return {
                "name": agent_name,
                "display_name": config.get("name", agent_name),
                "description": config.get("description", ""),
                "model": config.get("configs", {}).get("model", ""),
                "main_prompt": main_prompt,
                "function_tools": config.get("configs", {}).get("function_tools", []),
                "mcp_servers": self._extract_mcp_servers(config),
                "sub_agents": sub_agents,
            }

        except Exception as e:
            print(f"Error loading agent {agent_name}: {e}")
            return None

    def create_agent(self, agent: Agent) -> bool:
        """Create a new agent configuration."""
        try:
            agent_dir = self.agents_dir / agent.name
            agent_dir.mkdir(exist_ok=True)

            # Create main configuration
            config = {
                "name": agent.display_name,
                "description": agent.description,
                "configs": {
                    "model": agent.model,
                    "prompt": f"prompts/{agent.name}/main.txt",
                },
            }

            # Add function tools if they exist
            if agent.function_tools:
                config["configs"]["function_tools"] = agent.function_tools

            # Initialize defaults list
            config["defaults"] = []

            # Add MCP servers to defaults if they exist
            if agent.mcp_servers:
                for mcp_server in agent.mcp_servers:
                    config["defaults"].append(
                        f"../../mcp_servers/{mcp_server}@configs.mcp_servers"
                    )

            # Add sub-agents to defaults if they exist
            if agent.sub_agents:
                for sub_agent in agent.sub_agents:
                    config["defaults"].append(
                        f"{sub_agent.name.lower()}@sub_agents.{sub_agent.name.lower()}"
                    )

            # Always add _self_ at the end
            config["defaults"].append("_self_")

            config_file = agent_dir / "main.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            # Create main prompt
            prompt_dir = self.prompts_dir / agent.name
            prompt_dir.mkdir(exist_ok=True)

            prompt_file = prompt_dir / "main.txt"
            with open(prompt_file, "w") as f:
                f.write(agent.main_prompt)

            # Create sub-agents
            self._create_sub_agents_recursive(agent.sub_agents, agent_dir, prompt_dir)

            return True

        except Exception as e:
            print(f"Error creating agent {agent.name}: {e}")
            return False

    def update_agent(self, agent: Agent) -> bool:
        """Update an existing agent configuration."""
        return self.create_agent(agent)  # Same logic for now

    def delete_agent(self, agent_name: str) -> bool:
        """Delete an agent configuration."""
        try:
            # Remove agent directory
            agent_dir = self.agents_dir / agent_name
            if agent_dir.exists():
                shutil.rmtree(agent_dir)

            # Remove prompt directory
            prompt_dir = self.prompts_dir / agent_name
            if prompt_dir.exists():
                shutil.rmtree(prompt_dir)

            return True

        except Exception as e:
            print(f"Error deleting agent {agent_name}: {e}")
            return False

    def _load_sub_agents_recursive(
        self, agent_dir: Path, prompt_dir: Path, depth: int = 0
    ) -> list[dict[str, Any]]:
        """Recursively load sub-agents."""
        if depth > 5:  # Prevent infinite recursion
            return []

        sub_agents = []
        for config_file in agent_dir.glob("*.yaml"):
            if config_file.name != "main.yaml":
                try:
                    with open(config_file) as f:
                        config = yaml.safe_load(f)

                    # Load prompt
                    sub_agent_name = config_file.stem
                    prompt_file = prompt_dir / f"{sub_agent_name}.txt"
                    prompt = ""
                    if prompt_file.exists():
                        with open(prompt_file) as f:
                            prompt = f.read()

                    sub_agents.append(
                        {
                            "name": config.get("name", sub_agent_name),
                            "description": config.get("description", ""),
                            "model": config.get("configs", {}).get("model", ""),
                            "prompt": prompt,
                            "function_tools": config.get("configs", {}).get(
                                "function_tools", []
                            ),
                            "mcp_servers": self._extract_mcp_servers(config),
                            "has_nested": False,  # No nested sub-agents in this structure
                            "sub_agents": [],
                        }
                    )

                except Exception as e:
                    print(f"Error loading sub-agent {config_file.name}: {e}")
                    continue

        return sub_agents

    def _create_sub_agents_recursive(
        self, sub_agents: list[SubAgent], agent_dir: Path, prompt_dir: Path
    ) -> None:
        """Recursively create sub-agents."""
        for sub_agent in sub_agents:
            # Create sub-agent configuration (in same directory as main.yaml)
            config = {
                "name": sub_agent.name,
                "description": sub_agent.description,
                "configs": {
                    "model": sub_agent.model,
                    "prompt": f"prompts/{agent_dir.name}/{sub_agent.name.lower()}.txt",
                },
            }

            # Add function tools if they exist
            if sub_agent.function_tools:
                config["configs"]["function_tools"] = sub_agent.function_tools

            # Add MCP servers to defaults if they exist
            if sub_agent.mcp_servers:
                config["defaults"] = []
                for mcp_server in sub_agent.mcp_servers:
                    config["defaults"].append(
                        f"../../mcp_servers/{mcp_server}@configs.mcp_servers"
                    )
                config["defaults"].append("_self_")

            config_file = agent_dir / f"{sub_agent.name.lower()}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            # Create sub-agent prompt (in same directory as main.txt)
            prompt_file = prompt_dir / f"{sub_agent.name.lower()}.txt"
            with open(prompt_file, "w") as f:
                f.write(sub_agent.prompt)

    def _extract_mcp_servers(self, config: dict[str, Any]) -> list[str]:
        """Extract MCP servers from configuration."""
        defaults = config.get("defaults", [])
        if isinstance(defaults, list):
            # Extract MCP server references from the defaults list
            mcp_servers = []
            for item in defaults:
                if isinstance(item, str) and "mcp_servers" in item:
                    # Extract server name from path like
                    # "../../mcp_servers/arxiv_search@configs.mcp_servers"
                    server_name = item.split("/")[-1].split("@")[0]
                    mcp_servers.append(server_name)
            return mcp_servers
        if isinstance(defaults, dict):
            return defaults.get("mcp_servers", [])
        return []


# Initialize agent manager
agent_manager = AgentManager(app.config["CONFIG_DIR"], app.config["PROMPTS_DIR"])


@app.route("/")
def index() -> str:
    """Main page."""
    return redirect(url_for("browse_agents"))


@app.route("/browse")
def browse_agents() -> str:
    """Browse agents page."""
    agents = agent_manager.get_existing_agents()
    return render_template("browse.html", agents=agents)


@app.route("/create")
def create_agent() -> str:
    """Create agent page."""
    models = agent_manager.get_available_models()
    tools = agent_manager.get_available_tools()
    mcp_servers = agent_manager.get_available_mcp_servers()
    return render_template(
        "create.html", models=models, tools=tools, mcp_servers=mcp_servers
    )


@app.route("/edit")
def edit_agent() -> str:
    """Edit agent page."""
    agents = agent_manager.get_existing_agents()
    models = agent_manager.get_available_models()
    tools = agent_manager.get_available_tools()
    mcp_servers = agent_manager.get_available_mcp_servers()
    return render_template(
        "edit.html", agents=agents, models=models, tools=tools, mcp_servers=mcp_servers
    )


@app.route("/delete")
def delete_agent() -> str:
    """Delete agent page."""
    agents = agent_manager.get_existing_agents()
    return render_template("delete.html", agents=agents)


# API Routes
@app.route("/api/agents")
def api_get_agents() -> str:
    """Get all agents."""
    agents = agent_manager.get_existing_agents()
    return jsonify(agents)


@app.route("/api/agents/<agent_name>")
def api_get_agent(agent_name: str) -> str:
    """Get agent details."""
    details = agent_manager.get_agent_details(agent_name)
    if details:
        return jsonify(details)
    return jsonify({"error": "Agent not found"}), 404


@app.route("/api/agents", methods=["POST"])
def api_create_agent() -> str:
    """Create a new agent."""
    try:
        data = request.get_json()

        # Convert sub-agents data
        sub_agents_data = data.get("sub_agents", [])
        sub_agents = [SubAgent.from_dict(sa) for sa in sub_agents_data]

        # Create agent
        agent = Agent(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            model=data["model"],
            main_prompt=data["main_prompt"],
            function_tools=data.get("function_tools", []),
            mcp_servers=data.get("mcp_servers", []),
            sub_agents=sub_agents,
        )

        success = agent_manager.create_agent(agent)

        if success:
            return jsonify({"message": f"Agent '{agent.name}' created successfully"})
        return jsonify({"error": f"Failed to create agent '{agent.name}'"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/agents/<agent_name>", methods=["PUT"])
def api_update_agent(agent_name: str) -> str:
    """Update an agent."""
    try:
        # Check if agent exists
        existing_agent = agent_manager.get_agent_details(agent_name)
        if not existing_agent:
            return jsonify({"error": f"Agent '{agent_name}' not found"}), 404

        data = request.get_json()

        # Convert sub-agents data
        sub_agents_data = data.get("sub_agents", [])
        sub_agents = [SubAgent.from_dict(sa) for sa in sub_agents_data]

        # Create agent
        agent = Agent(
            name=agent_name,
            display_name=data["display_name"],
            description=data["description"],
            model=data["model"],
            main_prompt=data["main_prompt"],
            function_tools=data.get("function_tools", []),
            mcp_servers=data.get("mcp_servers", []),
            sub_agents=sub_agents,
        )

        success = agent_manager.update_agent(agent)

        if success:
            return jsonify({"message": f"Agent '{agent.name}' updated successfully"})
        return jsonify({"error": f"Failed to update agent '{agent.name}'"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/agents/<agent_name>", methods=["DELETE"])
def api_delete_agent(agent_name: str) -> str:
    """Delete an agent."""
    try:
        # Check if agent exists
        existing_agent = agent_manager.get_agent_details(agent_name)
        if not existing_agent:
            return jsonify({"error": f"Agent '{agent_name}' not found"}), 404

        success = agent_manager.delete_agent(agent_name)

        if success:
            return jsonify({"message": f"Agent '{agent_name}' deleted successfully"})
        return jsonify({"error": f"Failed to delete agent '{agent_name}'"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/options")
def api_get_options() -> str:
    """Get available options."""
    return jsonify(
        {
            "models": agent_manager.get_available_models(),
            "tools": agent_manager.get_available_tools(),
            "mcp_servers": agent_manager.get_available_mcp_servers(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7861)
