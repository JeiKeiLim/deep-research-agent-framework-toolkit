"""Implements a tool for searching with the Perplexity API."""

import asyncio
import logging
import os
from typing import Any

import backoff
import httpx
import pydantic

from src.utils.async_utils import rate_limited


class _Message(pydantic.BaseModel):
    """Type hints for the "message" field in Perplexity API response."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


class _Choice(pydantic.BaseModel):
    """Type hints for the "choices" field in Perplexity API response."""

    message: _Message

    class Config:
        """Pydantic config."""

        extra = "allow"


class _PerplexityResponse(pydantic.BaseModel):
    """Type hints for the Perplexity API response."""

    citations: list[str]
    """Citation URLs from the choices."""

    choices: list[_Choice]
    """List of choices returned by the model."""

    usage: dict[str, Any] | None = None


class PerplexitySearchResult(pydantic.BaseModel):
    """Type hints for the structured search result."""

    content: str
    urls: list[str]
    raw_response: _PerplexityResponse | None = None

    def __repr__(self) -> str:
        """Return a JSON representation of the model.

        Returns:
        -------
        str
            The model dumped as a JSON string.
        """
        return self.model_dump_json(indent=2)


class AsyncPerplexitySearch:
    """Configurable search tool for the Perplexity API."""

    def __init__(
        self,
        async_client: httpx.AsyncClient,
        model: str = "sonar",
        max_concurrency: int = 3,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> None:
        """Initialize the AsyncPerplexitySearch tool.

        Parameters
        ----------
        async_client : httpx.AsyncClient
            An asynchronous httpx client.
        model : str, optional
            The model to use for the search, by default "sonar-medium-online".
        max_concurrency : int, optional
            The maximum number of concurrent requests, by default 3.
        max_tokens : int, optional
            The maximum number of tokens to generate, by default 1000.
        temperature : float, optional
            The temperature for sampling, by default 0.1.
        """
        self.async_client = async_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    @backoff.on_exception(backoff.expo, httpx.HTTPStatusError, max_tries=5)
    async def search(self, query: str) -> PerplexitySearchResult:
        """Search the web using Perplexity API.

        Parameters
        ----------
        query : str
            The search query.

        Returns:
        -------
        PerplexitySearchResult
            A structured search result containing the content and extracted URLs.

        Raises:
        ------
        ValueError
            If the query is empty.
        httpx.HTTPStatusError
            If the API returns an error status code after retries.
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = await rate_limited(
            lambda: self.async_client.post("/chat/completions", json=payload),
            semaphore=self.semaphore,
        )
        response.raise_for_status()

        raw_data = response.json()
        parsed_response = _PerplexityResponse.model_validate(raw_data)

        content = ""
        if parsed_response.choices:
            content = parsed_response.choices[0].message.content

        if os.getenv("DEBUG_PERPLEXITY", "false").lower() == "true":
            self.logger.debug(f"DEBUG: Content preview: {content[:200]}...")

        return PerplexitySearchResult(
            content=content,
            urls=parsed_response.citations,
            raw_response=parsed_response,
        )


def get_perplexity_async_client(
    api_key: str | None = None,
    base_url: str = "https://api.perplexity.ai",
    timeout: int = 60,
) -> httpx.AsyncClient:
    """Get an async httpx client for Perplexity AI.

    If no api_key is provided, it will attempt to use the `PERPLEXITY_API_KEY`
    environment variable.

    Parameters
    ----------
    api_key : str, optional
        The Perplexity API key.
    base_url : str, optional
        The base URL for the Perplexity API.
    timeout : int, optional
        The timeout for requests in seconds.

    Returns:
    -------
    httpx.AsyncClient
        An asynchronous httpx client.

    Raises:
    ------
    ValueError
        If the API key is not provided and not found in environment variables.
    """
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY not provided and not found in environment variables."
        )

    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
