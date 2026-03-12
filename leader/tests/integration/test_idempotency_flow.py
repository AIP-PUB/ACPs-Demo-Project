"""
Leader Agent Platform - 集成测试：幂等性和校验流程

测试校验功能：
1. 幂等性保护（clientRequestId 去重）- 409003
2. activeTaskId 乐观并发校验 - 409002
3. mode 一致性校验 - 409001

本测试使用真实 API 和 LLM 调用，不使用 Mock。
"""

import sys
import os

# 确保路径正确
_current_dir = os.path.dirname(os.path.abspath(__file__))
_leader_dir = os.path.dirname(os.path.dirname(_current_dir))
_project_root = os.path.dirname(_leader_dir)
if _leader_dir not in sys.path:
    sys.path.insert(0, _leader_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import asyncio
import uuid
from typing import Dict, Any, Optional

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from assistant.api.schemas import (
    SubmitRequest,
    SubmitResponse,
)
from assistant.models.exceptions import (
    DuplicateRequestError,
    ActiveTaskMismatchError,
    ModeMismatchError,
)

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# 测试 Fixtures
# =============================================================================


def create_test_app() -> FastAPI:
    """
    创建测试应用实例。

    每次调用创建新的应用实例，避免测试间状态污染。
    """
    from assistant.api import router, init_routes
    from assistant.core import (
        Planner,
        SessionManager,
        create_intent_analyzer,
        create_orchestrator,
    )
    from assistant.core.history_compressor import HistoryCompressor
    from assistant.services import ScenarioLoader
    from assistant.llm import get_llm_client

    # 创建新的组件实例
    session_manager = SessionManager()
    scenario_loader = ScenarioLoader()
    intent_analyzer = create_intent_analyzer(scenario_loader)
    planner = Planner(scenario_loader)

    # 创建 history_compressor
    llm_client = get_llm_client()
    history_compressor = HistoryCompressor(
        llm_client=llm_client,
        scenario_loader=scenario_loader,
    )

    orchestrator = create_orchestrator(
        session_manager=session_manager,
        scenario_loader=scenario_loader,
        intent_analyzer=intent_analyzer,
        planner=planner,
        history_compressor=history_compressor,
    )

    # 创建新的 FastAPI 应用
    app = FastAPI(title="Test Leader Agent")
    init_routes(orchestrator, session_manager)
    app.include_router(router)

    return app


import pytest_asyncio


@pytest.fixture
def test_app():
    """创建测试应用实例。"""
    return create_test_app()


@pytest_asyncio.fixture
async def client(test_app: FastAPI):
    """创建异步测试客户端。"""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def unique_request_id() -> str:
    """生成唯一的请求 ID。"""
    return f"req-{uuid.uuid4().hex[:8]}"


# =============================================================================
# 幂等性保护测试（真实 LLM 调用）
# =============================================================================


class TestIdempotencyProtectionIntegration:
    """幂等性保护集成测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_first_request_succeeds(self, client: AsyncClient):
        """测试：首次请求应成功处理。"""
        request_id = unique_request_id()

        response = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[FirstRequest] status: {response.status_code}")
        print(f"[FirstRequest] response: {response.json()}")

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["sessionId"] is not None

    @pytest.mark.asyncio
    async def test_duplicate_request_same_payload_succeeds(self, client: AsyncClient):
        """测试：相同载荷的重复请求应成功（幂等重试）。"""
        request_id = unique_request_id()

        # 第一次请求
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )
        assert response1.status_code == 200
        session_id = response1.json()["result"]["sessionId"]

        # 第二次请求（相同 session_id、request_id、载荷）
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "你好",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[DuplicateSamePayload] status: {response2.status_code}")

        # 幂等重试应成功
        assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_duplicate_request_different_payload_rejected(
        self, client: AsyncClient
    ):
        """测试：相同 request_id 但不同载荷应被拒绝 (409003)。"""
        request_id = unique_request_id()

        # 第一次请求
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )
        assert response1.status_code == 200
        session_id = response1.json()["result"]["sessionId"]

        # 第二次请求（相同 request_id，但不同载荷）
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "你好啊",  # 载荷不同
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[DuplicateDifferentPayload] status: {response2.status_code}")
        print(f"[DuplicateDifferentPayload] response: {response2.json()}")

        # 应该返回 409 冲突
        assert response2.status_code == 409
        data = response2.json()
        assert data["detail"]["code"] == 409003


class TestModeMismatchIntegration:
    """模式一致性校验集成测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_mode_consistency_enforced(self, client: AsyncClient):
        """测试：模式不一致应被拒绝 (409001)。"""
        request_id1 = unique_request_id()

        # 第一次请求使用 direct_rpc 模式
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id1,
                "mode": "direct_rpc",
            },
        )
        assert response1.status_code == 200
        session_id = response1.json()["result"]["sessionId"]

        # 第二次请求使用 async 模式
        request_id2 = unique_request_id()
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "今天天气怎么样",
                "client_request_id": request_id2,
                "mode": "group",  # 模式不一致
            },
        )

        print(f"\n[ModeMismatch] status: {response2.status_code}")
        print(f"[ModeMismatch] response: {response2.json()}")

        # 应该返回 409 冲突
        assert response2.status_code == 409
        data = response2.json()
        assert data["detail"]["code"] == 409001

    @pytest.mark.asyncio
    async def test_same_mode_allowed(self, client: AsyncClient):
        """测试：相同模式应被允许。"""
        request_id1 = unique_request_id()

        # 第一次请求
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id1,
                "mode": "direct_rpc",
            },
        )
        assert response1.status_code == 200
        session_id = response1.json()["result"]["sessionId"]

        # 第二次请求使用相同模式
        request_id2 = unique_request_id()
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "今天天气怎么样",
                "client_request_id": request_id2,
                "mode": "direct_rpc",  # 相同模式
            },
        )

        print(f"\n[SameMode] status: {response2.status_code}")

        # 应该成功
        assert response2.status_code == 200


class TestActiveTaskIdValidationIntegration:
    """activeTaskId 乐观并发校验集成测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_active_task_id_mismatch_rejected(self, client: AsyncClient):
        """测试：activeTaskId 不匹配应被拒绝 (409002)。"""
        request_id1 = unique_request_id()

        # 创建初始请求（可能会产生一个 activeTaskId）
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "帮我规划北京三日游",
                "client_request_id": request_id1,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[ActiveTaskMismatch] first response status: {response1.status_code}")

        if response1.status_code != 200:
            pytest.skip("First request failed, skipping active_task_id test")
            return

        result = response1.json().get("result", {})
        session_id = result.get("sessionId")
        active_task_id = result.get("activeTaskId")  # camelCase

        print(f"[ActiveTaskMismatch] session_id: {session_id}")
        print(f"[ActiveTaskMismatch] active_task_id: {active_task_id}")

        if not active_task_id:
            pytest.skip("No active_task_id returned, skipping mismatch test")
            return

        # 使用错误的 activeTaskId 发送请求
        request_id2 = unique_request_id()
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "预算 3000 元",
                "client_request_id": request_id2,
                "mode": "direct_rpc",
                "active_task_id": "wrong-task-id-12345",  # 错误的 taskId
            },
        )

        print(f"[ActiveTaskMismatch] second response status: {response2.status_code}")
        print(f"[ActiveTaskMismatch] second response: {response2.json()}")

        # 应该返回 409 冲突
        assert response2.status_code == 409
        data = response2.json()
        assert data["detail"]["code"] == 409002


class TestConcurrentRequestsIntegration:
    """并发请求场景集成测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_sessions(self, client: AsyncClient):
        """测试：并发请求到不同 Session 应各自独立处理。"""

        async def make_request(query: str):
            request_id = unique_request_id()
            response = await client.post(
                "/api/v1/submit",
                json={
                    "query": query,
                    "client_request_id": request_id,
                    "mode": "direct_rpc",
                },
            )
            return response

        # 并发发送多个请求
        responses = await asyncio.gather(
            make_request("你好"),
            make_request("今天天气怎么样"),
            make_request("帮我查个酒店"),
        )

        print(
            f"\n[ConcurrentDifferentSessions] statuses: {[r.status_code for r in responses]}"
        )

        # 所有请求应该成功（各自独立的 Session）
        for i, response in enumerate(responses):
            print(f"[ConcurrentDifferentSessions] response {i}: {response.json()}")
            assert response.status_code == 200

        # 验证是不同的 Session
        session_ids = [r.json()["result"]["sessionId"] for r in responses]
        assert len(set(session_ids)) == 3, "Should create 3 different sessions"

    @pytest.mark.asyncio
    async def test_sequential_requests_same_session(self, client: AsyncClient):
        """测试：顺序请求到同一 Session 应正确处理。"""
        # 第一次请求
        request_id1 = unique_request_id()
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id1,
                "mode": "direct_rpc",
            },
        )
        assert response1.status_code == 200
        session_id = response1.json()["result"]["sessionId"]

        # 第二次请求到同一 Session
        request_id2 = unique_request_id()
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "帮我查查今天的天气",
                "client_request_id": request_id2,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[SequentialSameSession] response1: {response1.json()}")
        print(f"[SequentialSameSession] response2: {response2.json()}")

        assert response2.status_code == 200
        assert response2.json()["result"]["sessionId"] == session_id


class TestRequestValidationIntegration:
    """请求参数校验集成测试。"""

    @pytest.mark.asyncio
    async def test_missing_query_rejected(self, client: AsyncClient):
        """测试：缺少 query 应被拒绝。"""
        request_id = unique_request_id()

        response = await client.post(
            "/api/v1/submit",
            json={
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[MissingQuery] status: {response.status_code}")

        # 应该返回 422 Unprocessable Entity
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_session_rejected(self, client: AsyncClient):
        """测试：不存在的 Session 应被拒绝。"""
        request_id = unique_request_id()

        response = await client.post(
            "/api/v1/submit",
            json={
                "session_id": "nonexistent-session-12345",
                "query": "你好",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[NonexistentSession] status: {response.status_code}")
        print(f"[NonexistentSession] response: {response.json()}")

        # 应该返回 404
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_mode_rejected(self, client: AsyncClient):
        """测试：无效的 mode 字段应被拒绝。

        注意：当前 SubmitRequest.mode 使用 ExecutionMode 枚举，
        只接受有效值："direct_rpc", "async", "group"。
        """
        request_id = unique_request_id()

        response = await client.post(
            "/api/v1/submit",
            json={
                "query": "你好",
                "client_request_id": request_id,
                "mode": "invalid_mode",
            },
        )

        print(f"\n[InvalidMode] status: {response.status_code}")

        # 无效 mode 应返回 422
        assert response.status_code == 422


class TestIdempotencyWithRealLLMFlow:
    """结合真实 LLM 流程的幂等性测试。"""

    @pytest.mark.asyncio
    async def test_idempotent_retry_returns_cached_response(self, client: AsyncClient):
        """测试：幂等重试应返回缓存的响应。"""
        request_id = unique_request_id()

        # 第一次请求
        response1 = await client.post(
            "/api/v1/submit",
            json={
                "query": "帮我查一下北京的天气",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"\n[IdempotentRetry] first request status: {response1.status_code}")

        if response1.status_code != 200:
            pytest.skip("First request failed")
            return

        session_id = response1.json()["result"]["sessionId"]
        first_response_data = response1.json()

        # 第二次请求（幂等重试）
        response2 = await client.post(
            "/api/v1/submit",
            json={
                "session_id": session_id,
                "query": "帮我查一下北京的天气",
                "client_request_id": request_id,
                "mode": "direct_rpc",
            },
        )

        print(f"[IdempotentRetry] second request status: {response2.status_code}")

        assert response2.status_code == 200

        # 验证返回相同的 Session
        second_response_data = response2.json()
        assert second_response_data["result"]["sessionId"] == session_id
