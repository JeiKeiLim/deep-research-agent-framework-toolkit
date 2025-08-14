#!/usr/bin/env python3
"""
MEM0 API 테스트 도구

이 스크립트는 MEM0 API와의 연동을 테스트하고 UI에서 확인할 수 있도록 도와줍니다.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import httpx
from dotenv import load_dotenv


class Mem0Tester:
    """MEM0 API 테스트 클래스"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.mem0.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        print(f"🔧 MEM0 Tester 초기화")
        print(f"   Base URL: {base_url}")
        print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
        print()

    async def test_connection(self):
        """API 연결 테스트"""
        print("🔗 API 연결 테스트 중...")
        try:
            async with httpx.AsyncClient() as client:
                # 간단한 GET 요청으로 연결 테스트
                response = await client.get(
                    f"{self.base_url}/memories/",
                    headers=self.headers,
                    params={"limit": 1, "user_id": "test_user"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ API 연결 성공!")
                    return True
                else:
                    print(f"❌ API 연결 실패 - Status: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ API 연결 에러: {str(e)}")
            return False

    async def add_test_memory(self, content: str, user_id: str = "test_user"):
        """테스트 메모리 추가"""
        print(f"💾 메모리 추가 테스트: '{content}'")
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "messages": [{"role": "user", "content": content}],
                    "user_id": user_id,
                    "metadata": {
                        "source": "mem0_tester",
                        "timestamp": datetime.now().isoformat(),
                        "test": True
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/memories/",
                    headers=self.headers,
                    json=payload,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    memory_id = data.get("id", "Unknown")
                    print(f"✅ 메모리 추가 성공!")
                    print(f"   Memory ID: {memory_id}")
                    print(f"   Content: {content}")
                    return memory_id
                else:
                    print(f"❌ 메모리 추가 실패 - Status: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ 메모리 추가 에러: {str(e)}")
            return None

    async def search_memories(self, query: str, user_id: str = "test_user", limit: int = 5):
        """메모리 검색 테스트"""
        print(f"🔍 메모리 검색 테스트: '{query}'")
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "query": query,
                    "user_id": user_id,
                    "limit": limit
                }
                
                response = await client.post(
                    f"{self.base_url}/memories/search/",
                    headers=self.headers,
                    json=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    memories = data.get("memories", [])
                    print(f"✅ 메모리 검색 성공! ({len(memories)}개 결과)")
                    
                    for i, memory in enumerate(memories, 1):
                        content = memory.get("content", "No content")
                        memory_id = memory.get("id", "Unknown ID")
                        score = memory.get("score", "No score")
                        print(f"   {i}. [{memory_id}] {content} (Score: {score})")
                    
                    return memories
                else:
                    print(f"❌ 메모리 검색 실패 - Status: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return []
                    
        except Exception as e:
            print(f"❌ 메모리 검색 에러: {str(e)}")
            return []

    async def get_all_memories(self, user_id: str = "test_user", limit: int = 10):
        """모든 메모리 조회 테스트"""
        print(f"📋 모든 메모리 조회 테스트 (limit: {limit})")
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "user_id": user_id,
                    "limit": limit
                }
                
                response = await client.get(
                    f"{self.base_url}/memories/",
                    headers=self.headers,
                    params=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    memories = data.get("memories", [])
                    print(f"✅ 메모리 조회 성공! ({len(memories)}개 메모리)")
                    
                    for i, memory in enumerate(memories, 1):
                        content = memory.get("content", "No content")
                        memory_id = memory.get("id", "Unknown ID")
                        created_at = memory.get("created_at", "Unknown date")
                        print(f"   {i}. [{memory_id}] {content}")
                        print(f"      Created: {created_at}")
                    
                    return memories
                else:
                    print(f"❌ 메모리 조회 실패 - Status: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return []
                    
        except Exception as e:
            print(f"❌ 메모리 조회 에러: {str(e)}")
            return []

    async def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 MEM0 API 전체 테스트 시작")
        print("=" * 50)
        
        # 1. 연결 테스트
        if not await self.test_connection():
            print("❌ 연결 테스트 실패로 테스트 중단")
            return False
        
        print()
        
        # 2. 기존 메모리 조회
        await self.get_all_memories()
        print()
        
        # 3. 테스트 메모리 추가
        test_memories = [
            "사용자는 Python 프로그래밍을 좋아한다",
            "MEM0는 메모리 관리 시스템이다",
            "Deep Research Agent는 AI 연구 도구이다"
        ]
        
        added_ids = []
        for content in test_memories:
            memory_id = await self.add_test_memory(content)
            if memory_id:
                added_ids.append(memory_id)
            print()
        
        # 4. 검색 테스트
        search_queries = [
            "Python",
            "MEM0",
            "AI 도구"
        ]
        
        for query in search_queries:
            await self.search_memories(query)
            print()
        
        # 5. 최종 메모리 상태 확인
        print("📊 최종 메모리 상태:")
        await self.get_all_memories()
        
        print()
        print("🎉 전체 테스트 완료!")
        print(f"   추가된 메모리: {len(added_ids)}개")
        print("   MEM0 UI에서 결과를 확인해보세요!")
        
        return True


async def main():
    """메인 함수"""
    print("🧪 MEM0 API 테스트 도구")
    print("=" * 30)
    
    # 환경 변수 로드
    load_dotenv()
    
    # API 키 확인
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key or api_key == "test-mem0-api-key":
        print("❌ 유효한 MEM0_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에서 실제 API 키를 설정해주세요.")
        return
    
    # 테스터 생성 및 실행
    tester = Mem0Tester(api_key)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "connect":
            await tester.test_connection()
        elif command == "add":
            content = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "테스트 메모리"
            await tester.add_test_memory(content)
        elif command == "search":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "테스트"
            await tester.search_memories(query)
        elif command == "list":
            await tester.get_all_memories()
        else:
            print("사용법:")
            print("  python tool_mem0.py connect    # 연결 테스트")
            print("  python tool_mem0.py add [내용]  # 메모리 추가")
            print("  python tool_mem0.py search [검색어]  # 메모리 검색")
            print("  python tool_mem0.py list       # 모든 메모리 조회")
            print("  python tool_mem0.py            # 전체 테스트")
    else:
        # 전체 테스트 실행
        await tester.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())
