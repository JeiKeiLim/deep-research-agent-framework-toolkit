from src.utils.env_vars import Configs

from .kb_weaviate import AsyncWeaviateKnowledgeBase, get_weaviate_async_client
from .perplexity_api import AsyncPerplexitySearch, get_perplexity_async_client
from .tavily_api import AsyncTavilyClient, get_tavily_client


_env_configs = Configs.from_env_var()
_weaviate_async_client = get_weaviate_async_client(
    http_host=_env_configs.weaviate_http_host,
    http_port=_env_configs.weaviate_http_port,
    http_secure=_env_configs.weaviate_http_secure,
    grpc_host=_env_configs.weaviate_grpc_host,
    grpc_port=_env_configs.weaviate_grpc_port,
    grpc_secure=_env_configs.weaviate_grpc_secure,
    api_key=_env_configs.weaviate_api_key,
)
_weaviate_kb = AsyncWeaviateKnowledgeBase(
    async_client=_weaviate_async_client,
    collection_name="enwiki_20250520",
    # TODO: Get rid of hardcoded collection name
)

_perplexity_async_client = get_perplexity_async_client(
    api_key=_env_configs.perplexity_api_key,
)
_perplexity_search = AsyncPerplexitySearch(
    async_client=_perplexity_async_client,
    model="sonar",
)

_tavily_async_client = AsyncTavilyClient(
    client=get_tavily_client(
        api_key=_env_configs.tavily_api_key,
    )
)

function_tools = {
    "kb_weaviate": _weaviate_kb.search_knowledgebase,
    "perplexity_search": _perplexity_search.search,
    "tavily_search": _tavily_async_client.search,
}

__all__ = [
    "function_tools",
]
