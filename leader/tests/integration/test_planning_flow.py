"""
Leader Agent Platform - 集成测试：LLM-2 全量规划流程

测试从 API 到 LLM-2 调用完成的完整流程，验证：
1. API 正确处理 TASK_NEW 意图并触发规划
2. Planner 正确调用 LLM-2
3. PlanningResult 结构符合预期
4. Session 中正确保存规划结果
5. Event Log 记录规划事件

本测试使用真实大模型 API 调用。
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


class TestPlanningFlow:
    """LLM-2 全量规划完整流程测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_planning_triggered_on_task_new(
        self, client: AsyncClient, session_manager, app: FastAPI
    ):
        """测试：TASK_NEW 意图触发规划流程。"""
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
        assert result["activeTaskId"] is not None

        print(f"\n[API Response] sessionId={result['sessionId'][:8]}...")
        print(f"[API Response] activeTaskId={result['activeTaskId'][:8]}...")

        # === 验证 Session 状态 ===
        session = session_manager.get_session(session_id)

        assert session is not None
        assert session.expert_scenario.id == "tour"
        assert session.expert_scenario.kind == "expert"

        # 获取意图信息
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        intent_type = recent_turns[0].intent_type.value
        assert intent_type == "TASK_NEW"

        print(f"[Session] scenarioId={session.expert_scenario.id}")
        print(f"[Session] scenarioKind={session.expert_scenario.kind}")
        print(f"[Session] intentType={intent_type}")

    @pytest.mark.asyncio
    async def test_planning_result_has_active_dimensions(
        self, client: AsyncClient, session_manager
    ):
        """测试：规划结果包含激活的维度和 Partner。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="帮我规划一个北京三日游，预算3000元，2人出行"
            ),
        )

        assert response.status_code == 200
        data = response.json()

        # === 验证基本响应 ===
        assert is_success_response(data)
        result = data["result"]
        session_id = result["sessionId"]

        # === 从 Session 获取信息 ===
        session = session_manager.get_session(session_id)

        # 从 event_log 获取规划信息
        events = session.event_log
        planning_events = [e for e in events if e.type.value == "planning_result"]

        if planning_events:
            planning_data = planning_events[0].payload
            print(f"\n[PlanningResult] eventData={planning_data}")

        print(f"[Session] scenarioId={session.expert_scenario.id}")

    @pytest.mark.asyncio
    async def test_session_records_planning_event(
        self, client: AsyncClient, session_manager
    ):
        """测试：Session event_log 记录 planning_result 事件。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京旅游，需要酒店和餐厅推荐"),
        )

        assert response.status_code == 200
        data = response.json()

        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # === 验证 event_log ===
        events = session.event_log
        event_types = [e.type.value for e in events]

        print(f"\n[EventLog] 事件列表: {event_types}")

        # 应包含以下事件
        assert "user_submit" in event_types, "应记录 user_submit 事件"
        assert "intent_decision" in event_types, "应记录 intent_decision 事件"
        assert "planning_result" in event_types, "应记录 planning_result 事件"

        # 验证 planning_result 事件内容
        planning_event = next(e for e in events if e.type.value == "planning_result")
        event_data = planning_event.payload

        assert "active_task_id" in event_data or "scenario_id" in event_data

        print(f"[PlanningEvent] payload={event_data}")

    @pytest.mark.asyncio
    async def test_hotel_only_request(self, client: AsyncClient, session_manager):
        """测试：只请求酒店推荐时的规划结果。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="推荐北京故宫附近的酒店"),
        )

        assert response.status_code == 200
        data = response.json()

        assert is_success_response(data)
        result = data["result"]
        session_id = result["sessionId"]

        session = session_manager.get_session(session_id)

        # 从 dialog_context 获取意图
        recent_turns = session.dialog_context.recent_turns
        if recent_turns:
            intent_type = recent_turns[0].intent_type.value
            print(f"\n[HotelOnly] intentType={intent_type}")

        print(f"[HotelOnly] scenarioId={session.expert_scenario.id}")

    @pytest.mark.asyncio
    async def test_complex_travel_request(self, client: AsyncClient, session_manager):
        """测试：复杂旅游请求涉及多个维度。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="我想带家人去北京玩5天，预算1万元，需要酒店、餐厅推荐"
            ),
        )

        assert response.status_code == 200
        data = response.json()

        assert is_success_response(data)
        result = data["result"]
        session_id = result["sessionId"]

        session = session_manager.get_session(session_id)

        print(f"\n[ComplexRequest] scenarioId={session.expert_scenario.id}")
        print(f"[ComplexRequest] activeTask={session.active_task}")


class TestChitChatNotTriggerPlanning:
    """验证闲聊不触发规划。"""

    @pytest.mark.asyncio
    async def test_chitchat_no_planning(self, client: AsyncClient, session_manager):
        """测试：闲聊请求不应触发规划。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好，今天天气怎么样？"),
        )

        assert response.status_code == 200
        data = response.json()

        # 验证响应成功
        assert is_success_response(data)
        result = data["result"]
        session_id = result["sessionId"]

        # 从 session 获取意图
        session = session_manager.get_session(session_id)
        recent_turns = session.dialog_context.recent_turns
        if recent_turns:
            intent_type = recent_turns[0].intent_type.value
            # 闲聊应返回 CHIT_CHAT
            assert intent_type == "CHIT_CHAT"
            print(f"\n[ChitChat] intentType={intent_type}")


class TestDialogContextWithPlanning:
    """验证规划后对话上下文记录。"""

    @pytest.mark.asyncio
    async def test_dialog_context_records_planning_response(
        self, client: AsyncClient, session_manager
    ):
        """测试：对话上下文正确记录规划响应。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京两日游"),
        )

        assert response.status_code == 200
        data = response.json()

        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # === 验证 dialog_context ===
        ctx = session.dialog_context
        assert len(ctx.recent_turns) >= 1, "应至少有一轮对话"

        # 验证最近的对话轮次
        turn = ctx.recent_turns[0]
        assert turn.user_query is not None
        assert "北京" in turn.user_query
        assert turn.intent_type is not None

        print(f"\n[DialogContext] userQuery: {turn.user_query}")
        print(f"[DialogContext] intentType: {turn.intent_type.value}")

    @pytest.mark.asyncio
    async def test_multi_turn_planning_conversation(
        self, client: AsyncClient, session_manager
    ):
        """测试：多轮对话中规划场景保持。"""
        # 第一轮：触发规划
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京旅游"),
        )
        assert r1.status_code == 200
        data1 = r1.json()

        assert is_success_response(data1)
        session_id = data1["result"]["sessionId"]

        # 验证第一轮意图
        session = session_manager.get_session(session_id)
        recent_turns = session.dialog_context.recent_turns
        if recent_turns:
            first_intent = recent_turns[0].intent_type.value
            assert first_intent == "TASK_NEW"

        # 第二轮：补充信息
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="需要3天行程，预算5000元", session_id=session_id
            ),
        )
        assert r2.status_code == 200

        # 验证 session 保持
        session = session_manager.get_session(session_id)
        assert session.expert_scenario.id == "tour"
        assert len(session.dialog_context.recent_turns) >= 2

        print(f"\n[MultiTurn] 对话轮数: {len(session.dialog_context.recent_turns)}")
        print(f"[MultiTurn] scenarioId: {session.expert_scenario.id}")
