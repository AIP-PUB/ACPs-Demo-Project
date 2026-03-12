"""
Leader Agent Platform - 集成测试：Session 创建与状态验证

测试通过 /submit API 创建新 Session，验证：
1. API 响应正确（使用 CommonResponse 结构）
2. Session 被正确创建
3. Session 初始状态正确
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from .conftest import (
    build_submit_request,
    extract_session_id,
    is_success_response,
)

pytest_plugins = ("pytest_asyncio",)


class TestSessionCreation:
    """Session 创建集成测试。"""

    @pytest.mark.asyncio
    async def test_first_submit_creates_session(
        self, client: AsyncClient, session_manager
    ):
        """测试：首次提交创建新 Session。"""
        # 发送请求（不带 session_id）
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )

        # 1. 验证 API 响应
        assert response.status_code == 200
        data = response.json()

        # 验证 CommonResponse 结构
        assert is_success_response(data), f"Expected success response, got: {data}"
        session_id = extract_session_id(data)
        assert session_id.startswith("sess_")  # session ID 格式

        print(f"\n[API Response] session_id={session_id}")
        print(f"[API Response] result={data.get('result')}")

        # 2. 验证 Session 被创建
        session = session_manager.get_session(session_id)
        assert session is not None, "Session should be created"

        print(f"\n[Session] base_scenario.kind={session.base_scenario.kind}")
        print(f"[Session] closed={session.closed}")

        # 3. 验证 Session 初始状态
        assert session.base_scenario.kind == "base"
        # Session 未关闭表示活跃状态
        assert session.closed is None or session.closed is False

    @pytest.mark.asyncio
    async def test_session_has_dialog_context(
        self, client: AsyncClient, session_manager
    ):
        """测试：Session 创建后 dialog_context 被初始化。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证 dialog_context 结构（新字段名）
        ctx = session.dialog_context
        assert ctx is not None
        assert hasattr(ctx, "recent_turns")
        assert hasattr(ctx, "history_summary")

        print(f"\n[DialogContext] recent_turns count: {len(ctx.recent_turns)}")
        print(f"[DialogContext] history_summary: {ctx.history_summary}")

    @pytest.mark.asyncio
    async def test_session_has_event_log(self, client: AsyncClient, session_manager):
        """测试：Session 创建后 event_log 被初始化。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证 event_log 存在
        assert session.event_log is not None
        assert isinstance(session.event_log, list)

        print(f"\n[EventLog] entries count: {len(session.event_log)}")
        for i, entry in enumerate(session.event_log):
            # 新字段名：type, payload (不是 event_type, event_data)
            print(f"[EventLog][{i}] type={entry.type}, payload={entry.payload}")

    @pytest.mark.asyncio
    async def test_reuse_existing_session(self, client: AsyncClient, session_manager):
        """测试：使用已有 session_id 复用 Session。"""
        # 首次请求创建 Session
        response1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        data1 = response1.json()
        session_id = extract_session_id(data1)

        # 获取初始 session 状态
        session_before = session_manager.get_session(session_id)
        turns_before = len(session_before.dialog_context.recent_turns)

        print(f"\n[Before] session_id={session_id}")
        print(f"[Before] recent_turns count={turns_before}")

        # 第二次请求使用相同 session_id
        response2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="今天天气怎么样？",
                session_id=session_id,
            ),
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert extract_session_id(data2) == session_id

        # 验证 Session 被复用（对话轮次增加）
        session_after = session_manager.get_session(session_id)
        turns_after = len(session_after.dialog_context.recent_turns)

        print(f"[After] recent_turns count={turns_after}")

        # 应该增加了轮次
        assert turns_after > turns_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
