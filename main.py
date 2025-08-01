"""Main entry point for the Deep Research Agent Framework Toolkit.

This script initializes the application, parses command line arguments,
and loads configurations from environment variables.
"""

import argparse

from src.utils import Configs


def parse_args():
    """Parse command line arguments.

    For now, no arguments are defined, but this can be extended later.
    """
    parser = argparse.ArgumentParser(
        description="Deep Research Agent Framework Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()


def main():
    """Run the main application.

    This function initializes the application, parses command line arguments,
    and loads configurations from environment variables.
    """
    args = parse_args()

    configs = Configs.from_env_var()
    print("Arguments parsed successfully:", args)
    print("Configs loaded successfully:", configs)
    print("Hello from deep-research-agent-framework-toolkit!")


if __name__ == "__main__":
    main()
