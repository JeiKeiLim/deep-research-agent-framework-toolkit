from src.utils.env_vars import Configs

from .kb_weaviate import AsyncWeaviateKnowledgeBase, get_weaviate_async_client


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

function_tools = {
    "kb_weaviate": _weaviate_kb.search_knowledgebase,
}

__all__ = [
    "function_tools",
]
