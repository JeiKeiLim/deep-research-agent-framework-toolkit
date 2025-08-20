"""Test cases for Weaviate integration."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from src.draft_agents.function_tools import (
    AsyncWeaviateKnowledgeBase,
    get_weaviate_async_client,
)
from src.utils import (
    Configs,
    pretty_print,
)


load_dotenv(verbose=True)


@pytest.fixture()
def configs() -> Configs:
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def weaviate_kb(
    configs: Configs,
) -> AsyncGenerator[AsyncWeaviateKnowledgeBase, None]:
    """Weaviate knowledgebase for testing."""
    async_client = get_weaviate_async_client(
        http_host=configs.weaviate_http_host,
        http_port=configs.weaviate_http_port,
        http_secure=configs.weaviate_http_secure,
        grpc_host=configs.weaviate_grpc_host,
        grpc_port=configs.weaviate_grpc_port,
        grpc_secure=configs.weaviate_grpc_secure,
        api_key=configs.weaviate_api_key,
    )

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name="enwiki_20250520"
    )

    await async_client.close()


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase) -> None:
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)


if __name__ == "__main__":
    configs2 = Configs.from_env_var()
    result = test_weaviate_kb(
        weaviate_kb=AsyncWeaviateKnowledgeBase(
            async_client=get_weaviate_async_client(
                http_host=configs2.weaviate_http_host,
                http_port=configs2.weaviate_http_port,
                http_secure=configs2.weaviate_http_secure,
                grpc_host=configs2.weaviate_grpc_host,
                grpc_port=configs2.weaviate_grpc_port,
                grpc_secure=configs2.weaviate_grpc_secure,
                api_key=configs2.weaviate_api_key,
            ),
            collection_name="enwiki_20250520",
        )
    )

    import asyncio

    asyncio.run(result)
