"""Implements a tool for searching with the Perplexity API."""

import asyncio
import json
import logging
import os
import re
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

    choices: list[_Choice]
    usage: dict[str, Any] | None = None


class PerplexitySearchResult(pydantic.BaseModel):
    """Type hints for the structured search result."""

    content: str
    urls: list[str]
    raw_response: _PerplexityResponse | None = None

    def __repr__(self) -> str:
        """Return a JSON representation of the model.

        Returns
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

        Returns
        -------
        PerplexitySearchResult
            A structured search result containing the content and extracted URLs.

        Raises
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

        urls = self._extract_urls(parsed_response, content)

        if os.getenv("DEBUG_PERPLEXITY", "false").lower() == "true":
            self.logger.debug(f"DEBUG: Extracted {len(urls)} URLs from content")
            self.logger.debug(f"DEBUG: URLs: {urls}")
            self.logger.debug(f"DEBUG: Content preview: {content[:200]}...")

        return PerplexitySearchResult(
            content=content,
            urls=urls,
            raw_response=parsed_response,
        )

    def _extract_urls(self, response: _PerplexityResponse, content: str) -> list[str]:
        """Extract URLs from Perplexity API response.

        Parameters
        ----------
        response : _PerplexityResponse
            The parsed Perplexity API response.
        content : str
            The content from the response.

        Returns
        -------
        list[str]
            A list of unique URLs found in the response.
        """
        urls = []

        # Method 1: Extract URLs from content text
        content_urls = re.findall(r'https?://[^\s<>"\']+', content)
        urls.extend(content_urls)

        if not response.choices:
            return list(set(urls))

        choice = response.choices[0]

        # Method 2: Check for tool_calls in response
        if choice.message and choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if (
                    tool_call.get("type") == "function"
                    and tool_call.get("function", {}).get("name") == "search"
                ):
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        if "urls" in args:
                            urls.extend(args["urls"])
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Method 3: Check for usage field which might contain source URLs
        if response.usage:
            # Some APIs include source URLs in usage metadata
            pass

        # Method 4: Check for additional fields that might contain URLs
        def find_urls_in_dict(d: dict[str, Any]) -> list[str]:
            found_urls = []
            for _, v in d.items():
                if isinstance(v, str) and "http" in v:
                    found_urls.extend(re.findall(r'https?://[^\s<>"\']+', v))
                elif isinstance(v, dict):
                    found_urls.extend(find_urls_in_dict(v))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            found_urls.extend(find_urls_in_dict(item))
            return found_urls

        # The choice object is a pydantic model, so we dump it to a dict
        choice_dict = choice.model_dump()
        for key, value in choice_dict.items():
            if key not in ["message", "index", "finish_reason"]:
                if isinstance(value, str) and "http" in value:
                    additional_urls = re.findall(r'https?://[^\s<>"\']+', value)
                    urls.extend(additional_urls)
                elif isinstance(value, dict):
                    urls.extend(find_urls_in_dict(value))

        # Remove duplicates and clean URLs
        unique_urls = list(set(urls))
        return [url.rstrip(".,;:!?") for url in unique_urls]


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

    Returns
    -------
    httpx.AsyncClient
        An asynchronous httpx client.

    Raises
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
