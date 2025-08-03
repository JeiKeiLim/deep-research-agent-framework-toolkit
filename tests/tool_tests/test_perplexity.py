"""Test cases for Perplexity API integration."""

import asyncio

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from src.draft_agents.function_tools import (
    AsyncPerplexitySearch,
    get_perplexity_async_client,
)
from src.utils import (
    Configs,
    pretty_print,
)


load_dotenv(verbose=True)


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def perplexity_search(configs):
    """Perplexity search client for testing."""
    async_client = get_perplexity_async_client(
        api_key=configs.perplexity_api_key,
    )

    yield AsyncPerplexitySearch(
        async_client=async_client,
        model="sonar",
    )

    await async_client.aclose()


def test_perplexity_client_creation(configs):
    """Test Perplexity client creation."""
    client = get_perplexity_async_client(
        api_key=configs.perplexity_api_key,
    )
    assert client is not None
    assert client.base_url == "https://api.perplexity.ai"


@pytest.mark.asyncio
async def test_perplexity_search_basic(perplexity_search: AsyncPerplexitySearch):
    """Test basic Perplexity search functionality."""
    query = "What is the latest news about artificial intelligence?"
    result = await perplexity_search.search(query)

    assert result is not None
    assert hasattr(result, "content")
    assert hasattr(result, "urls")
    assert isinstance(result.content, str)
    assert isinstance(result.urls, list)
    assert len(result.content) > 0

    print(f"Search query: {query}")
    print(f"Content length: {len(result.content)} characters")
    print(f"URLs found: {len(result.urls)}")
    pretty_print(result)


@pytest.mark.asyncio
async def test_perplexity_search_technical_query(
    perplexity_search: AsyncPerplexitySearch,
):
    """Test Perplexity search with technical query."""
    query = "What are the latest developments in quantum computing research?"
    result = await perplexity_search.search(query)

    assert result is not None
    assert len(result.content) > 0
    assert "quantum" in result.content.lower() or "computing" in result.content.lower()

    print(f"Technical query: {query}")
    print(f"Content preview: {result.content[:200]}...")
    print(f"URLs: {result.urls}")


@pytest.mark.asyncio
async def test_perplexity_search_current_events(
    perplexity_search: AsyncPerplexitySearch,
):
    """Test Perplexity search with current events query."""
    query = "What are the major technology trends in 2024?"
    result = await perplexity_search.search(query)

    assert result is not None
    assert len(result.content) > 0
    assert len(result.urls) > 0

    print(f"Current events query: {query}")
    print(f"Number of URLs: {len(result.urls)}")
    print(f"First few URLs: {result.urls[:3]}")


@pytest.mark.asyncio
async def test_perplexity_search_error_handling(
    perplexity_search: AsyncPerplexitySearch,
):
    """Test Perplexity search error handling with empty query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await perplexity_search.search("")


@pytest.mark.asyncio
async def test_perplexity_search_model_configuration():
    """Test Perplexity search with different model configurations."""
    configs = Configs.from_env_var()
    async_client = get_perplexity_async_client(
        api_key=configs.perplexity_api_key,
    )

    # Test with different model
    search_tool = AsyncPerplexitySearch(
        async_client=async_client,
        model="sonar",
        max_tokens=500,
        temperature=0.1,
    )

    result = await search_tool.search("What is machine learning?")
    assert result is not None
    assert len(result.content) > 0

    await async_client.aclose()


@pytest.mark.asyncio
async def test_perplexity_search_concurrent_requests(
    perplexity_search: AsyncPerplexitySearch,
):
    """Test Perplexity search with concurrent requests."""
    queries = [
        "What is Python programming?",
        "What is JavaScript?",
        "What is machine learning?",
    ]

    async def search_single(query: str):
        return await perplexity_search.search(query)

    # Run concurrent searches
    tasks = [search_single(query) for query in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for result in results:
        assert result is not None
        assert len(result.content) > 0

    print(f"Concurrent searches completed: {len(results)} results")
    for i, result in enumerate(results):
        print(f"Query {i + 1}: {len(result.content)} chars, {len(result.urls)} URLs")


@pytest.mark.asyncio
async def test_perplexity_search_url_extraction(
    perplexity_search: AsyncPerplexitySearch,
):
    """Test URL extraction from Perplexity search results."""
    query = "What will be the best programming languages to learn in 2026?"
    result = await perplexity_search.search(query)

    assert result is not None
    assert len(result.urls) >= 0  # URLs might be empty in some cases

    # Check if URLs are valid
    for url in result.urls:
        assert url.startswith(("http://", "https://"))
        assert len(url) > 10  # Basic URL length check

    print("URL extraction test:")
    print(f"Query: {query}")
    print(f"URLs found: {len(result.urls)}")
    if result.urls:
        print(f"Sample URLs: {result.urls[:3]}")


if __name__ == "__main__":
    configs2 = Configs.from_env_var()

    async def run_tests():
        """Run sample test."""
        async_client = get_perplexity_async_client(
            api_key=configs2.perplexity_api_key,
        )

        search_tool = AsyncPerplexitySearch(
            async_client=async_client,
            model="sonar",
        )

        # Run basic test
        _ = await test_perplexity_search_url_extraction(search_tool)
        print("Basic test completed successfully")

        await async_client.aclose()

    import asyncio

    asyncio.run(run_tests())
