"""Tavily API client for accessing various APIs like weather, news, and more."""

import os

import pydantic
from tavily import TavilyClient


class _TavilySearchItem(pydantic.BaseModel):
    """Type hints for the Tavily search item."""

    title: str
    """The title of the search item."""

    url: str
    """The URL of the search item."""

    content: str
    """The content of the search item."""

    raw_content: str | None = None
    """Raw content of the search item, if available."""

    def __repr__(self) -> str:
        """Return a JSON representation of the model.

        Returns
        -------
        str
            The model dumped as a JSON string.
        """
        return self.model_dump_json(indent=2)


class TavilySearchResult(pydantic.BaseModel):
    """Type hints for the Tavily search result."""

    query: str
    contents: list[_TavilySearchItem]


def get_tavily_client(api_key: str | None = None) -> TavilyClient:
    """Get a Tavily client.

    If no api_key is provided, it will attempt to use the `TAVILY_API_KEY`
    environment variable.

    Parameters
    ----------
    api_key : str, optional
        The Tavily API key.

    Returns
    -------
    TavilyClient
        An instance of TavilyClient configured with the provided API key.

    Raises
    ------
    ValueError
        If the API key is not provided and not found in environment variables.
    """
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not provided and not found in environment variables."
        )

    return TavilyClient(api_key=api_key)


class AsyncTavilyClient:
    """Asynchronous Tavily client for performing searches."""

    def __init__(self, client: TavilyClient) -> None:
        self.client = client

    async def search(self, query: str) -> TavilySearchResult:
        """Perform a search using the Tavily API.

        Parameters
        ----------
        query : str
            The search query.
        api_key : str, optional
            The Tavily API key.

        Returns
        -------
        dict
            The search results from the Tavily API.
        """
        tavily_result = self.client.search(query=query)

        if not tavily_result or "results" not in tavily_result:
            return TavilySearchResult(query=query, contents=[])

        contents = [
            _TavilySearchItem.model_validate(item) for item in tavily_result["results"]
        ]
        return TavilySearchResult(query=query, contents=contents)


if __name__ == "__main__":
    # Example usage
    async_tavily_client = AsyncTavilyClient(get_tavily_client())
    results = async_tavily_client.search("What is the weather like today?")
    print(results)
