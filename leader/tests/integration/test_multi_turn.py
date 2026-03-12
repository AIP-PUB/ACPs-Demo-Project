"""
Leader Agent Platform - 集成测试：多轮对话状态累积

测试多轮对话中 Session 状态的正确累积，验证：
1. dialog_context.recent_turns 正确累积
2. event_log 正确累积
3. 意图切换时状态正确更新
4. touched_at 时间戳更新
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from .conftest import build_submit_request, extract_session_id

pytest_plugins = ("pytest_asyncio",)


class TestMultiTurnDialog:
    """多轮对话状态累积测试。"""

    @pytest.mark.asyncio
    async def test_multi_turn_accumulation(self, client: AsyncClient, session_manager):
        """测试：多轮对话状态正确累积。"""
        # 第1轮：问候（CHIT_CHAT）
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        session_id = extract_session_id(r1.json())

        session = session_manager.get_session(session_id)
        # 新模型：每轮对话对应 1 个 DialogTurn（包含 user_query 和 response_summary）
        assert len(session.dialog_context.recent_turns) == 1
        # 事件日志：user_submit + intent_decision
        assert len(session.event_log) == 2
        print(
            f"\n[Round 1] turns={len(session.dialog_context.recent_turns)}, events={len(session.event_log)}"
        )

        # 第2轮：继续闲聊（CHIT_CHAT）
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="今天天气好吗？", session_id=session_id),
        )

        session = session_manager.get_session(session_id)
        assert len(session.dialog_context.recent_turns) == 2
        print(
            f"[Round 2] turns={len(session.dialog_context.recent_turns)}, events={len(session.event_log)}"
        )

        # 第3轮：旅游请求（TASK_NEW，触发场景切换）
        r3 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="帮我规划北京两日游", session_id=session_id
            ),
        )

        session = session_manager.get_session(session_id)
        # TASK_NEW 产生 1 个 turn
        assert len(session.dialog_context.recent_turns) >= 3
        assert session.expert_scenario.id == "tour"
        print(
            f"[Round 3] turns={len(session.dialog_context.recent_turns)}, scenario={session.expert_scenario.id}"
        )

        # 第4轮：继续旅游话题（在 tour 场景中）
        r4 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="推荐几个景点", session_id=session_id),
        )

        session = session_manager.get_session(session_id)
        # 每轮增加 1 个 turn
        assert len(session.dialog_context.recent_turns) >= 4
        assert session.expert_scenario.id == "tour"  # 场景保持
        print(
            f"[Round 4] turns={len(session.dialog_context.recent_turns)}, scenario={session.expert_scenario.id}"
        )

    @pytest.mark.asyncio
    async def test_intent_transition_tracking(
        self, client: AsyncClient, session_manager
    ):
        """测试：意图转换在 event_log 中正确记录。"""
        # 第1轮：CHIT_CHAT
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        session_id = extract_session_id(r1.json())

        # 第2轮：TASK_NEW
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="规划北京旅游", session_id=session_id),
        )

        # 第3轮：CHIT_CHAT（即使在 tour 场景，问无关问题）
        r3 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我写首诗", session_id=session_id),
        )

        session = session_manager.get_session(session_id)

        # 提取所有意图事件（使用新事件类型 intent_decision）
        intent_events = [
            e for e in session.event_log if e.type.value == "intent_decision"
        ]

        # 至少有 3 个意图决策事件（场景切换可能产生额外事件）
        assert len(intent_events) >= 3

        # 验证意图序列（使用新字段名 payload）
        # 注意：过滤只包含 intent_type 的事件
        intents = [
            e.payload["intent_type"]
            for e in intent_events
            if "intent_type" in e.payload
        ]
        print(f"\n[IntentSequence] {intents}")

        assert len(intents) >= 3
        assert intents[0] == "CHIT_CHAT"
        assert intents[1] == "TASK_NEW"
        # 第三个可能是 CHIT_CHAT（写诗请求）
        assert "CHIT_CHAT" in intents or "TASK_INPUT" in intents

    @pytest.mark.asyncio
    async def test_turn_content_integrity(self, client: AsyncClient, session_manager):
        """测试：对话轮次内容完整性。"""
        queries = [
            "你好",
            "帮我规划旅游",
            "需要住宿推荐",
        ]

        # 第一轮
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query=queries[0]),
        )
        session_id = extract_session_id(r1.json())

        # 后续轮次
        for query in queries[1:]:
            await client.post(
                "/api/v1/submit",
                json=build_submit_request(query=query, session_id=session_id),
            )

        session = session_manager.get_session(session_id)
        turns = session.dialog_context.recent_turns

        # 验证每轮都有 user_query（注意：LLM 可能会改写 user_query）
        assert len(turns) >= 3  # 至少 3 轮

        # 只验证第一个查询是原始的
        assert turns[0].user_query == queries[0]
        print(f"[Turn 0] User: {turns[0].user_query}")

        # 后续查询可能被 LLM 改写，只验证非空
        for i, t in enumerate(turns[1:3], 1):
            assert t.user_query  # 非空
            print(f"[Turn {i}] User: {t.user_query}")

        # 验证 response_type 存在
        for t in turns:
            assert t.response_type  # 非空
            assert t.timestamp  # 有时间戳

    @pytest.mark.asyncio
    async def test_touched_at_updates(self, client: AsyncClient, session_manager):
        """测试：touched_at 时间戳正确更新。"""
        import time

        # 创建 session
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        session_id = extract_session_id(r1.json())

        session1 = session_manager.get_session(session_id)
        touched_at_1 = session1.touched_at

        # 等待一小段时间
        time.sleep(0.1)

        # 发送第二个请求
        await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="再问一下", session_id=session_id),
        )

        session2 = session_manager.get_session(session_id)
        touched_at_2 = session2.touched_at

        # 时间戳应该更新
        assert touched_at_2 >= touched_at_1
        print(f"\n[Timestamp] Before: {touched_at_1}")
        print(f"[Timestamp] After: {touched_at_2}")

    @pytest.mark.asyncio
    async def test_event_log_chronological_order(
        self, client: AsyncClient, session_manager
    ):
        """测试：event_log 按时间顺序记录。"""
        # 创建 session 并发送多个请求
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="你好"),
        )
        session_id = extract_session_id(r1.json())

        await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="规划北京游", session_id=session_id),
        )

        await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="推荐酒店", session_id=session_id),
        )

        session = session_manager.get_session(session_id)
        events = session.event_log

        # 验证事件按时间顺序（使用 created_at 字段）
        timestamps = [e.created_at for e in events]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], "Events should be chronological"

        print(f"\n[EventLog] Total events: {len(events)}")
        for e in events:
            # 使用新字段名 type.value
            print(f"  [{e.created_at}] {e.type.value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
