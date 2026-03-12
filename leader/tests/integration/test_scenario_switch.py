"""
Leader Agent Platform - 集成测试：场景切换功能

测试不同场景之间的切换功能：
1. 从 base 切换到 tour 场景
2. 从 base 切换到 divination 场景
3. 场景内的请求保持在当前场景
4. 跨场景切换（从一个 expert 场景切换到另一个）
5. 不匹配的请求保持 CHIT_CHAT
"""

import pytest
from httpx import AsyncClient

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


class TestScenarioSwitch:
    """场景切换功能测试。"""

    # =========================================================================
    # 切换到 tour 场景
    # =========================================================================

    @pytest.mark.asyncio
    async def test_switch_to_tour(self, client: AsyncClient, session_manager):
        """测试：从 base 切换到 tour 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京三日游"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 验证场景切换
        assert is_success_response(data)
        assert session.expert_scenario.kind == "expert"
        assert session.expert_scenario.id == "tour"

        # 从 dialog_context 验证意图
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"

        print(f"\n[Switch] base -> tour")
        print(f"[Session] scenario_id={session.expert_scenario.id}")

    @pytest.mark.asyncio
    async def test_tour_with_landmark(self, client: AsyncClient, session_manager):
        """测试：提及北京地标时切换到 tour。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="故宫门票多少钱？怎么预约？"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.id == "tour"

        print(f"\n[Landmark] 故宫 -> tour")

    # =========================================================================
    # 切换到 divination 场景
    # =========================================================================

    @pytest.mark.asyncio
    async def test_switch_to_divination_bazi(
        self, client: AsyncClient, session_manager
    ):
        """测试：八字算命请求切换到 divination 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="帮我算算八字，我是1990年5月15日早上8点出生的"
            ),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.kind == "expert"
        assert session.expert_scenario.id == "divination"

        print(f"\n[Switch] base -> divination (八字)")
        print(f"[Session] scenario_id={session.expert_scenario.id}")

    @pytest.mark.asyncio
    async def test_switch_to_divination_tarot(
        self, client: AsyncClient, session_manager
    ):
        """测试：塔罗牌请求切换到 divination 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我抽一张塔罗牌，看看我最近的感情运势"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.id == "divination"

        print(f"\n[Switch] base -> divination (塔罗)")

    @pytest.mark.asyncio
    async def test_switch_to_divination_astrology(
        self, client: AsyncClient, session_manager
    ):
        """测试：星座运势请求切换到 divination 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="我是天蝎座的，帮我看看这个月的运势"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.id == "divination"

        print(f"\n[Switch] base -> divination (星座)")

    @pytest.mark.asyncio
    async def test_switch_to_divination_yijing(
        self, client: AsyncClient, session_manager
    ):
        """测试：易经占卜请求切换到 divination 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="帮我用易经算一卦，看看我这个项目能不能成功"
            ),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.id == "divination"

        print(f"\n[Switch] base -> divination (易经)")

    # =========================================================================
    # 场景内请求保持
    # =========================================================================

    @pytest.mark.asyncio
    async def test_stay_in_tour(self, client: AsyncClient, session_manager):
        """测试：在 tour 场景内继续相关请求。"""
        # 第一轮：切换到 tour
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京旅游"),
        )
        session_id = extract_session_id(r1.json())

        # 第二轮：继续北京相关请求
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="推荐几个好吃的餐厅", session_id=session_id
            ),
        )

        session = session_manager.get_session(session_id)

        # 应保持在 tour 场景
        assert session.expert_scenario.id == "tour"
        print(f"\n[Stay] tour 场景内持续")

    @pytest.mark.asyncio
    async def test_stay_in_divination(self, client: AsyncClient, session_manager):
        """测试：在 divination 场景内继续相关请求。"""
        # 第一轮：切换到 divination
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我算个命"),
        )
        session_id = extract_session_id(r1.json())

        # 第二轮：继续占卜相关请求
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="那我的财运怎么样？", session_id=session_id
            ),
        )

        session = session_manager.get_session(session_id)

        # 应保持在 divination 场景
        assert session.expert_scenario.id == "divination"
        print(f"\n[Stay] divination 场景内持续")

    # =========================================================================
    # 跨场景切换
    # =========================================================================

    @pytest.mark.asyncio
    async def test_switch_between_expert_scenarios(
        self, client: AsyncClient, session_manager
    ):
        """测试：从一个 expert 场景切换到另一个。"""
        # 第一轮：进入 tour
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划北京旅游"),
        )
        session_id = extract_session_id(r1.json())

        session1 = session_manager.get_session(session_id)
        assert session1.expert_scenario.id == "tour"
        print(f"\n[Step 1] 进入 tour")

        # 第二轮：切换到 divination
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="算了，先帮我算算命，看看最近适不适合出行", session_id=session_id
            ),
        )

        session2 = session_manager.get_session(session_id)
        data2 = r2.json()

        # 验证场景切换
        assert is_success_response(data2)
        recent_turns = session2.dialog_context.recent_turns
        if len(recent_turns) > 0:
            latest_intent = recent_turns[-1].intent_type.value
            print(f"[Step 2] latest intent: {latest_intent}")

        # 验证切换到 divination
        assert session2.expert_scenario.id == "divination"
        print(f"[Step 2] 切换到 divination")

    @pytest.mark.asyncio
    async def test_switch_from_divination_to_tour(
        self, client: AsyncClient, session_manager
    ):
        """测试：从 divination 切换到 tour。"""
        # 第一轮：进入 divination
        r1 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我算算今年运势"),
        )
        session_id = extract_session_id(r1.json())

        session1 = session_manager.get_session(session_id)
        assert session1.expert_scenario.id == "divination"
        print(f"\n[Step 1] 进入 divination")

        # 第二轮：切换到 tour
        r2 = await client.post(
            "/api/v1/submit",
            json=build_submit_request(
                query="好的运势不错，那帮我规划一下北京三日游吧", session_id=session_id
            ),
        )

        session2 = session_manager.get_session(session_id)

        assert session2.expert_scenario.id == "tour"
        print(f"[Step 2] 切换到 tour")

    # =========================================================================
    # 不匹配的请求保持 CHIT_CHAT
    # =========================================================================

    @pytest.mark.asyncio
    async def test_unrelated_request_stays_chitchat(
        self, client: AsyncClient, session_manager
    ):
        """测试：不匹配任何场景的请求保持 CHIT_CHAT。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我写一段Python代码"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "CHIT_CHAT"

        # expert_scenario 可能是 None 或者 base
        if session.expert_scenario:
            assert session.expert_scenario.kind == "base"
            assert session.expert_scenario.id is None

        print(f"\n[CHIT_CHAT] 不匹配的请求保持在 base")

    @pytest.mark.asyncio
    async def test_other_city_travel_triggers_tour(
        self, client: AsyncClient, session_manager
    ):
        """测试：其他城市旅游请求也能触发通用 tour 场景。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我规划上海三日游"),
        )

        data = response.json()
        session_id = extract_session_id(data)
        session = session_manager.get_session(session_id)

        # 通用 tour 场景应该能处理任意城市的旅游请求
        assert is_success_response(data)
        recent_turns = session.dialog_context.recent_turns
        assert len(recent_turns) > 0
        assert recent_turns[0].intent_type.value == "TASK_NEW"
        assert session.expert_scenario.id == "tour"

        print(f"\n[Universal] 上海旅游触发 tour 场景")

    # =========================================================================
    # 场景切换事件日志验证
    # =========================================================================

    @pytest.mark.asyncio
    async def test_scenario_switch_event_log(
        self, client: AsyncClient, session_manager
    ):
        """测试：场景切换时 event_log 正确记录。"""
        response = await client.post(
            "/api/v1/submit",
            json=build_submit_request(query="帮我算算八字"),
        )

        session_id = extract_session_id(response.json())
        session = session_manager.get_session(session_id)

        # 验证事件日志有记录
        events = session.event_log
        event_types = [e.type.value for e in events]

        print(f"\n[EventLog] 事件列表: {event_types}")

        # 应该有基本的事件记录
        assert len(events) > 0, "应有事件记录"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
