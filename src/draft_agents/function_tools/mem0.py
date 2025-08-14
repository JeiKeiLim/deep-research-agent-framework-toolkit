"""Implements a tool for memory management using the Mem0 API."""

import asyncio
import logging

import httpx


class AsyncMem0Client:
    """Async client for Mem0 API operations."""

    def __init__(self, api_key: str, base_url: str = "https://api.mem0.ai/v1"):
        """Initialize the Mem0 client.
        
        Args:
            api_key: Mem0 API key
            base_url: Base URL for Mem0 API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }

    async def search_memories(
        self,
        query,
        user_id=None,
        limit=10,
    ):
        """Search for memories using the Mem0 API."""
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "query": query,
                    "limit": limit,
                    "user_id": user_id or "default_user"  # 필수 파라미터
                }
                
                response = await client.post(
                    f"{self.base_url}/memories/search/",
                    headers=self.headers,
                    json=params,  # POST 방식으로 JSON 전송
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logging.error(f"Mem0 search error: {str(e)}")
            return {"memories": [], "error": str(e)}

    async def add_memory(
        self,
        content,
        user_id=None,
        metadata=None
    ):
        """Add a new memory using the Mem0 API."""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "messages": [{"role": "user", "content": content}],  # MEM0 API 형식
                    "user_id": user_id or "default_user"  # 필수 파라미터
                }
                    
                if metadata:
                    payload["metadata"] = metadata
                
                response = await client.post(
                    f"{self.base_url}/memories/",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logging.error(f"Mem0 add memory error: {str(e)}")
            return {"error": str(e)}

    async def get_all_memories(
        self,
        user_id=None,
        limit=100,
    ):
        """Get all memories for a user."""
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "limit": limit,
                    "user_id": user_id or "default_user"  # 필수 파라미터
                }
                
                response = await client.get(
                    f"{self.base_url}/memories/",
                    headers=self.headers,
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                # MEM0 API 응답이 리스트 형태일 수 있음
                if isinstance(data, list):
                    return data
                return data.get("memories", [])
                
        except Exception as e:
            logging.error(f"Mem0 get memories error: {str(e)}")
            return []


class AsyncMem0Memory:
    """Async wrapper for Mem0 memory operations with agent-friendly methods."""

    def __init__(self, async_client):
        """Initialize the Mem0 memory manager."""
        self.async_client = async_client

    async def search(
        self,
        query,
        user_id=None,
        limit=5
    ):
        """Search memories and return formatted results for agent use."""
        try:
            result = await self.async_client.search_memories(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            if "error" in result:
                return f"Error searching memories: {result['error']}"
            
            memories = result.get("memories", [])
            if not memories:
                return f"No memories found for query: '{query}'"
            
            formatted_results = [f"Memory Search Results for: '{query}'\n"]
            
            for i, memory in enumerate(memories, 1):
                content = memory.get("content", "No content")
                memory_id = memory.get("id", "Unknown ID")
                created_at = memory.get("created_at", "Unknown date")
                
                formatted_results.append(
                    f"{i}. [{memory_id}] {content}\n   Created: {created_at}"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    async def add(
        self,
        content,
        user_id=None,
        metadata=None
    ):
        """Add a memory and return confirmation for agent use."""
        try:
            result = await self.async_client.add_memory(
                content=content,
                user_id=user_id,
                metadata=metadata
            )
            
            if "error" in result:
                return f"❌ Error adding memory: {result['error']}"
            
            memory_id = result.get("id", "Unknown")
            return f"✅ Memory added successfully!\nID: {memory_id}\nContent: {content}"
            
        except Exception as e:
            return f"❌ Error adding memory: {str(e)}"

    async def get_all(
        self,
        user_id=None,
        limit=10
    ):
        """Get all memories and return formatted results for agent use."""
        try:
            memories = await self.async_client.get_all_memories(
                user_id=user_id,
                limit=limit
            )
            
            if not memories:
                return "No memories found."
            
            formatted_results = [f"All Memories ({len(memories)} found):\n"]
            
            for i, memory in enumerate(memories, 1):
                content = memory.get("content", "No content")
                memory_id = memory.get("id", "Unknown ID")
                created_at = memory.get("created_at", "Unknown date")
                
                formatted_results.append(
                    f"{i}. [{memory_id}] {content}\n   Created: {created_at}"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"


def get_mem0_client(api_key):
    """Create and return a Mem0 async client."""
    return AsyncMem0Client(api_key=api_key)
