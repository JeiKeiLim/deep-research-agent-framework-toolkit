# Contributing

We welcome contributions to the DRAFT framework! Whether you want to fix a bug, add a new feature, or improve the documentation, your help is greatly appreciated.

## How to Contribute

To contribute to the project, please follow these steps:

1. **Fork the repository**: Create your own fork of the [DRAFT repository](https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit) on GitHub.
2. **Create a new branch**: Create a new branch in your forked repository for your changes.
3. **Make your changes**: Make your desired changes to the codebase.
4. **Run the tests**: Ensure that all tests pass by running `PYTHONPATH=. uv run --env-file .env pytest`.
5. **Submit a pull request**: Open a pull request from your branch to the `main` branch of the DRAFT repository.

## Development Setup

For a detailed guide on setting up your development environment, please refer to the [Installation](https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit/blob/main/README.md#installation) section in the `README.md` file.

### Formatting and Linting

We use `ruff` for formatting and linting the code. Before submitting your changes, please make sure to run the following commands:

- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`

### Pre-commit Hooks

We use pre-commit hooks to ensure that the code is properly formatted and linted before each commit. To install the pre-commit hooks, run:

```bash
uv run pre-commit install
```

## Adding New Agents and Tools

For instructions on how to add new agents and tools to the framework, please refer to the following guides:

- **[Creating Your Own Agent](creating-agents.md)**
- **[Adding New Tools](https://github.com/JeiKeiLim/deep-research-agent-framework-toolkit/blob/main/README.md#adding-new-tools)**

Thank you for your contributions!
