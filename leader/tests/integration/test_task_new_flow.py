"""
Leader Agent Platform - 集成测试：TASK_NEW 意图流程

测试新任务意图的完整处理流程，验证：
1. 意图被正确识别为 TASK_NEW
2. 场景正确切换到 expert 场景
3. dialog_context 正确记录对话
4. event_log 正确记录场景切换事件
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


class TestTaskNewFlow:
    """TASK_NEW 意图完整流程测试。"""

    @pytest.mark.asyncio
    async def test_travel_plan_triggers_scenario_switch(
        self, client: AsyncClient, session_manager
    ):
        """测试：旅游规划请求触发场景切换。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划一个北京三日游"),
        )

        assert response.status_code == 200
        data = response.json()
        session_id = extract_session_id(data)

        # === 验证 API 响应 ===
        assert is_success_response(data)
        result = data["result"]

        print(f"\n[Response] sessionId={result['sessionId'][:8]}...")

        # === 验证场景切换 ===
        session = session_manager.get_session(session_id)

        # 场景应切换到 tour
        assert session.expert_scenario.kind == "expert"
        assert session.expert_scenario.id == "tour"

        print(f"\n[Session] scenario_kind={session.expert_scenario.kind}")
        print(f"[Session] scenario_id={session.expert_scenario.id}")

        # === 验证 dialog_context 意图 ===
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"

        # === 验证 event_log 包含意图决策 ===
        events = session.event_log
        event_types = [e.type.value for e in events]

        assert "intent_decision" in event_types

        # 检查 intent_decision 事件
        intent_event = next(e for e in events if e.type.value == "intent_decision")
        assert (
            intent_event.payload.get("intent_type") == "TASK_NEW"
            or "target_scenario" in intent_event.payload
        )

        print(f"\n[EventLog] {event_types}")

    @pytest.mark.asyncio
    async def test_hotel_recommendation_scenario_switch(
        self, client: AsyncClient, session_manager
    ):
        """测试：酒店推荐请求触发场景切换。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="推荐北京朝阳区的五星级酒店"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证响应成功和场景
        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.kind == "expert"
        assert session.expert_scenario.id == "tour"

        print(f"\n[Hotel] scenario switched to: {session.expert_scenario.id}")

    @pytest.mark.asyncio
    async def test_dialog_context_records_task_intent(
        self, client: AsyncClient, session_manager
    ):
        """测试：dialog_context 正确记录 TASK_NEW 意图。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我找一家北京的特色餐厅"),
        )

        session_id = extract_session_id(response.json())
        session = session_manager.get_session(session_id)

        # 验证 dialog_context
        ctx = session.dialog_context
        assert len(ctx.recent_turns) >= 1

        # 验证对话轮次
        turn = ctx.recent_turns[0]
        assert turn.user_query is not None
        assert "餐厅" in turn.user_query
        assert turn.intent_type.value == "TASK_NEW"

        print(f"\n[DialogContext] userQuery: {turn.user_query}")
        print(f"[DialogContext] intentType: {turn.intent_type.value}")

    @pytest.mark.asyncio
    async def test_scenario_persists_across_requests(
        self, client: AsyncClient, session_manager
    ):
        """测试：场景在后续请求中保持。"""
        # 第一次请求：触发场景切换
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京旅游"),
        )
        session_id = extract_session_id(r1.json())

        session1 = session_manager.get_session(session_id)
        assert session1.expert_scenario.id == "tour"

        # 第二次请求：继续在 tour 场景
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="推荐一些景点", session_id=session_id),
        )

        session2 = session_manager.get_session(session_id)

        # 场景应保持
        assert session2.expert_scenario.id == "tour"
        assert session2.expert_scenario.kind == "expert"

        print(
            f"\n[Persistence] Scenario after 2nd request: {session2.expert_scenario.id}"
        )

    @pytest.mark.asyncio
    async def test_complex_travel_request(self, client: AsyncClient, session_manager):
        """测试：复杂旅游请求的完整状态验证。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="我想带家人去北京玩5天，预算1万元，需要酒店和景点推荐"
            ),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证响应成功
        assert is_success_response(data)

        # 验证意图
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"

        # 验证场景
        assert session.expert_scenario.id == "tour"

        # 验证事件日志
        intent_event = next(
            e for e in session.event_log if e.type.value == "intent_decision"
        )
        assert (
            "intent_type" in intent_event.payload
            or "target_scenario" in intent_event.payload
        )

        print(f"\n[ComplexRequest] Intent: {recent_turns[0].intent_type.value}")
        print(f"[ComplexRequest] Scenario: {session.expert_scenario.id}")
        print(f"[ComplexRequest] Turns: {len(session.dialog_context.recent_turns)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
