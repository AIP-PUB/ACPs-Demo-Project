"""
Leader Agent Platform - 集成测试：CHIT_CHAT 意图流程

测试闲聊意图的完整处理流程，验证：
1. 意图被正确识别为 CHIT_CHAT
2. dialog_context 正确记录对话
3. event_log 正确记录事件
4. Session 状态保持不变
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


class TestChitChatFlow:
    """CHIT_CHAT 意图完整流程测试。"""

    @pytest.mark.asyncio
    async def test_greeting_full_flow(self, client: AsyncClient, session_manager):
        """测试：问候语的完整流程验证。"""
        # 发送问候
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )

        assert response.status_code == 200
        data = response.json()

        # 验证 CommonResponse 结构
        assert is_success_response(data), f"Expected success response, got: {data}"
        session_id = extract_session_id(data)

        # === 验证 API 响应 ===
        result = data["result"]
        assert result["sessionId"].startswith("sess_")
        # 新结构不再包含 intentType/responseType/responseText

        print(f"\n[Response] session_id={session_id}")

        # === 验证 Session 状态 ===
        session = session_manager.get_session(session_id)

        # 场景应保持为 base
        assert session.base_scenario.kind == "base"
        assert session.expert_scenario is None

        # === 验证 dialog_context ===
        ctx = session.dialog_context
        assert len(ctx.recent_turns) == 1  # 每轮一条 DialogTurn

        # 验证 DialogTurn
        turn = ctx.recent_turns[0]
        assert turn.user_query == "你好"
        assert turn.intent_type.value == "CHIT_CHAT"
        assert turn.response_summary  # 非空

        print(f"\n[DialogContext] user_query: {turn.user_query}")
        print(f"[DialogContext] response_summary: {turn.response_summary[:50]}...")

        # === 验证 event_log ===
        events = session.event_log
        assert len(events) >= 2

        # 验证事件类型（使用新字段名）
        event_types = [e.type.value for e in events]
        assert "user_submit" in event_types
        assert "intent_decision" in event_types

        # 验证 intent_decision 事件
        intent_event = next(e for e in events if e.type.value == "intent_decision")
        assert intent_event.payload["intent_type"] == "CHIT_CHAT"
        assert intent_event.payload.get("target_scenario") is None

        print(f"\n[EventLog] Events: {event_types}")

    @pytest.mark.asyncio
    async def test_weather_question_no_scenario_switch(
        self, client: AsyncClient, session_manager
    ):
        """测试：天气问题不触发场景切换。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="今天北京天气怎么样？"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证意图（从 event_log 获取）
        intent_event = next(
            e for e in session.event_log if e.type.value == "intent_decision"
        )
        assert intent_event.payload["intent_type"] == "CHIT_CHAT"

        # 验证场景未切换
        assert session.base_scenario.kind == "base"
        assert session.expert_scenario is None

        # 验证有响应
        assert len(session.dialog_context.recent_turns) > 0
        print(
            f"\n[Weather Response] {session.dialog_context.recent_turns[0].response_summary}"
        )

    @pytest.mark.asyncio
    async def test_out_of_scope_request(self, client: AsyncClient, session_manager):
        """测试：超出能力范围的请求返回 CHIT_CHAT。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我写一段Python代码"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证意图（从 event_log 获取）
        intent_event = next(
            e for e in session.event_log if e.type.value == "intent_decision"
        )
        assert intent_event.payload["intent_type"] == "CHIT_CHAT"

        # 验证场景未切换
        assert session.base_scenario.kind == "base"

        print(
            f"\n[OutOfScope Response] {session.dialog_context.recent_turns[0].response_summary}"
        )

    @pytest.mark.asyncio
    async def test_consecutive_chitchat(self, client: AsyncClient, session_manager):
        """测试：连续闲聊的状态累积。"""
        # 第一轮
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        session_id = extract_session_id(r1.json())

        # 第二轮（继续闲聊）
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你叫什么名字？", session_id=session_id),
        )

        # 第三轮（继续闲聊）
        r3 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你能做什么？", session_id=session_id),
        )

        # 验证 Session 状态
        session = session_manager.get_session(session_id)

        # 验证对话轮次累积（每轮一条 DialogTurn）
        assert len(session.dialog_context.recent_turns) == 3

        # 验证事件日志累积
        intent_events = [
            e for e in session.event_log if e.type.value == "intent_decision"
        ]
        assert len(intent_events) == 3  # 3 次意图分析

        # 所有意图都应该是 CHIT_CHAT
        for event in intent_events:
            assert event.payload["intent_type"] == "CHIT_CHAT"

        print(f"\n[Turns] Total: {len(session.dialog_context.recent_turns)}")
        print(f"[Events] Intent decision: {len(intent_events)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
