"""Script for uploading test data to Langfuse.

See:
langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents
"""

import argparse

import pandas as pd
from dotenv import load_dotenv
from rich.progress import track

from src.utils.data import get_dataset_url_hash
from src.utils.env_vars import Configs
from src.utils.langfuse.otlp_env_setup import set_up_langfuse_otlp_env_vars
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)


parser = argparse.ArgumentParser()
parser.add_argument("--source_dataset", required=True)
parser.add_argument("--langfuse_dataset_name", required=True)
parser.add_argument("--allow-duplicates", action="store_true", default=False)


if __name__ == "__main__":
    args = parser.parse_args()
    configs = Configs.from_env_var()
    set_up_langfuse_otlp_env_vars()
    dataset_url_hash = get_dataset_url_hash(args.source_dataset)

    # Create a dataset in Langfuse
    assert langfuse_client.auth_check()
    langfuse_client.create_dataset(
        name=args.langfuse_dataset_name,
        description=f"[{dataset_url_hash}] Data from {args.source_dataset}",
        metadata={
            "url_hash": dataset_url_hash,
            "source": args.source_dataset,
            "type": "benchmark",
        },
    )

    df = pd.read_json(args.source_dataset, lines=True)

    for idx, row in track(
        df.iterrows(),
        total=len(df),
        description="Uploading to Langfuse",
    ):
        langfuse_client.create_dataset_item(
            dataset_name=args.langfuse_dataset_name,
            input={"text": row["question"]},
            expected_output={"text": row["ground_truth"]},
            # unique id to enable upsert without creating duplicates
            id=None if args.allow_duplicates else f"{dataset_url_hash}-{idx:05}",
        )
