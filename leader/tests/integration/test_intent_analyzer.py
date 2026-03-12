"""
LLM-1 Intent Analyzer 集成测试

使用真实 LLM 配置进行测试，验证意图分析功能是否正常。

运行方式：
    cd leader
    source ../venv/bin/activate
    pytest tests/integration/test_intent_analyzer.py -v -s
"""

import pytest
import sys
from pathlib import Path

# 配置 pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# 确保可以导入 assistant 模块
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assistant.services.scenario_loader import ScenarioLoader
from assistant.core.intent_analyzer import IntentAnalyzer
from assistant.core.session_manager import SessionManager
from assistant.llm.client import LLMClient
from assistant.models import IntentType, ExecutionMode


class TestLLM1IntentAnalyzer:
    """LLM-1 意图分析器集成测试。"""

    @pytest.fixture(scope="class")
    def scenario_loader(self):
        """创建场景加载器。"""
        loader = ScenarioLoader()
        print(f"\n[Setup] Scenario briefs: {[b.id for b in loader.scenario_briefs]}")
        return loader

    @pytest.fixture(scope="class")
    def llm_client(self):
        """创建 LLM 客户端。"""
        client = LLMClient()
        print(f"\n[Setup] LLM profiles: {list(client._profiles.keys())}")
        return client

    @pytest.fixture(scope="class")
    def intent_analyzer(self, scenario_loader, llm_client):
        """创建意图分析器。"""
        return IntentAnalyzer(
            scenario_loader=scenario_loader,
            llm_client=llm_client,
        )

    @pytest.fixture(scope="class")
    def session_manager(self):
        """创建 Session 管理器。"""
        return SessionManager()

    @pytest.fixture
    def base_session(self, session_manager, scenario_loader):
        """创建一个基础场景的 Session。"""
        return session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=scenario_loader.base_scenario,
        )

    @pytest.fixture
    def tour_session(self, session_manager, scenario_loader):
        """创建一个旅游专业场景的 Session。"""
        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=scenario_loader.base_scenario,
        )
        # 设置 expert_scenario
        expert_scenario = scenario_loader.get_expert_scenario("tour")
        if expert_scenario:
            session.expert_scenario = expert_scenario
        return session

    # =========================================================================
    # CHIT_CHAT 测试用例
    # =========================================================================

    @pytest.mark.asyncio
    async def test_chit_chat_greeting(self, intent_analyzer, base_session):
        """测试闲聊：打招呼。"""
        result = await intent_analyzer.analyze(
            user_query="你好",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT
        assert result.task_instruction is None
        assert result.response_guide is not None

    @pytest.mark.asyncio
    async def test_chit_chat_weather(self, intent_analyzer, base_session):
        """测试闲聊：询问天气（超出平台能力）。"""
        result = await intent_analyzer.analyze(
            user_query="今天北京天气怎么样？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_chit_chat_general_question(self, intent_analyzer, base_session):
        """测试闲聊：通用知识问答。"""
        result = await intent_analyzer.analyze(
            user_query="Python 是什么编程语言？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT

    # =========================================================================
    # TASK_NEW 测试用例
    # =========================================================================

    @pytest.mark.asyncio
    async def test_task_new_travel_plan(self, intent_analyzer, base_session):
        """测试新任务：旅游规划请求。"""
        result = await intent_analyzer.analyze(
            user_query="帮我规划一个北京三日游",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.task_instruction is not None
        # 应该路由到 tour 场景
        assert result.target_scenario == "tour"

    @pytest.mark.asyncio
    async def test_task_new_hotel_recommendation(self, intent_analyzer, base_session):
        """测试新任务：酒店推荐请求。"""
        result = await intent_analyzer.analyze(
            user_query="推荐几家北京的五星级酒店",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.task_instruction is not None

    @pytest.mark.asyncio
    async def test_task_new_food_recommendation(self, intent_analyzer, base_session):
        """测试新任务：美食推荐请求。"""
        result = await intent_analyzer.analyze(
            user_query="北京有什么好吃的特色餐厅？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW

    @pytest.mark.asyncio
    async def test_task_new_in_expert_session(self, intent_analyzer, tour_session):
        """测试新任务：在专业场景 Session 中提出新请求。"""
        result = await intent_analyzer.analyze(
            user_query="再帮我推荐几个景点",
            session=tour_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        # 已在 tour 场景中，不需要再切换
        # target_scenario 应该是 None 或 "tour"

    # =========================================================================
    # 边界情况测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_ambiguous_request(self, intent_analyzer, base_session):
        """测试边界情况：模糊请求。"""
        result = await intent_analyzer.analyze(
            user_query="我想去玩",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")
        print(f"[Result] response_guide={result.response_guide}")

        # 这个请求可能被判定为 TASK_NEW 或 CHIT_CHAT
        # 主要验证不会报错
        assert result.intent_type in [IntentType.TASK_NEW, IntentType.CHIT_CHAT]

    @pytest.mark.asyncio
    async def test_mixed_intent(self, intent_analyzer, base_session):
        """测试边界情况：混合意图。"""
        result = await intent_analyzer.analyze(
            user_query="你好，能帮我规划一下上海两日游吗？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")

        # 虽然包含问候，但主要意图是任务
        assert result.intent_type == IntentType.TASK_NEW

    # =========================================================================
    # 复杂场景测试用例
    # =========================================================================

    @pytest.mark.asyncio
    async def test_multi_city_travel(self, intent_analyzer, base_session):
        """测试复杂场景：多城市行程规划。"""
        result = await intent_analyzer.analyze(
            user_query="我想规划一个从北京出发，经过西安，最后到成都的7天自驾游",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.target_scenario == "tour"
        assert result.task_instruction is not None

    @pytest.mark.asyncio
    async def test_detailed_requirements(self, intent_analyzer, base_session):
        """测试复杂场景：详细需求描述。"""
        result = await intent_analyzer.analyze(
            user_query="帮我找一家北京朝阳区的酒店，要五星级，有游泳池，离国贸近，预算每晚1500以内",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.task_instruction is not None

    @pytest.mark.asyncio
    async def test_family_travel_with_constraints(self, intent_analyzer, base_session):
        """测试复杂场景：带约束的家庭游。"""
        result = await intent_analyzer.analyze(
            user_query="下周末想带孩子去北京玩两天，孩子5岁，希望行程轻松一点，有适合儿童的景点",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.task_instruction is not None

    @pytest.mark.asyncio
    async def test_budget_sensitive_request(self, intent_analyzer, base_session):
        """测试复杂场景：预算敏感型请求。"""
        result = await intent_analyzer.analyze(
            user_query="穷游北京，3天2晚，总预算2000块，怎么安排最划算？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW

    @pytest.mark.asyncio
    async def test_seasonal_travel(self, intent_analyzer, base_session):
        """测试复杂场景：季节性旅游请求。"""
        result = await intent_analyzer.analyze(
            user_query="春节期间北京有什么好玩的？想看庙会和灯会",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW

    # =========================================================================
    # 非旅游场景边界测试（应为 CHIT_CHAT）
    # =========================================================================

    @pytest.mark.asyncio
    async def test_medical_question(self, intent_analyzer, base_session):
        """测试非旅游场景：医疗咨询（超出能力）。"""
        result = await intent_analyzer.analyze(
            user_query="我最近头疼，应该吃什么药？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        # 医疗咨询不在已注册场景范围内
        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_finance_question(self, intent_analyzer, base_session):
        """测试非旅游场景：金融咨询（超出能力）。"""
        result = await intent_analyzer.analyze(
            user_query="帮我分析一下茅台股票，现在能买吗？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_code_generation(self, intent_analyzer, base_session):
        """测试非旅游场景：代码生成请求。"""
        result = await intent_analyzer.analyze(
            user_query="帮我写一个Python爬虫程序",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_emotional_support(self, intent_analyzer, base_session):
        """测试非旅游场景：情感支持请求。"""
        result = await intent_analyzer.analyze(
            user_query="今天心情不好，工作压力太大了",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        assert result.intent_type == IntentType.CHIT_CHAT

    # =========================================================================
    # 场景切换测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_scenario_switch_from_expert(self, intent_analyzer, tour_session):
        """测试场景切换：从专业场景发起不相关请求。"""
        result = await intent_analyzer.analyze(
            user_query="帮我写一首诗",
            session=tour_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")

        # 在 tour 场景中请求写诗，应为 CHIT_CHAT
        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_continue_in_expert_session(self, intent_analyzer, tour_session):
        """测试场景延续：在专业场景中继续相关请求。"""
        result = await intent_analyzer.analyze(
            user_query="那住宿方面有什么推荐？",
            session=tour_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        # 已在 tour 场景，继续旅游相关请求
        assert result.intent_type == IntentType.TASK_NEW
        # 不需要切换场景
        assert result.target_scenario is None or result.target_scenario == "tour"

    # =========================================================================
    # 特殊输入测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_very_short_input(self, intent_analyzer, base_session):
        """测试特殊输入：极短输入。"""
        result = await intent_analyzer.analyze(
            user_query="？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        # 无意义输入应为 CHIT_CHAT
        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_emoji_only(self, intent_analyzer, base_session):
        """测试特殊输入：纯表情符号。"""
        result = await intent_analyzer.analyze(
            user_query="😊👍",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")

        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_long_detailed_request(self, intent_analyzer, base_session):
        """测试特殊输入：超长详细请求。"""
        result = await intent_analyzer.analyze(
            user_query="""我计划下个月15号到20号和家人一起去北京旅游。
            我们一共4个人，两个大人两个小孩（分别是8岁和12岁）。
            我们住宿预算大概每晚800-1200元，希望住在市中心交通方便的地方。
            行程方面，必去故宫和长城，其他景点可以你来推荐。
            另外孩子们想去环球影城，不知道时间够不够。
            餐饮方面希望能体验正宗的北京烤鸭和涮羊肉。
            请帮我规划一下整个行程，包括交通、住宿、景点和餐饮。""",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW
        assert result.target_scenario == "tour"
        assert result.task_instruction is not None

    @pytest.mark.asyncio
    async def test_question_about_platform(self, intent_analyzer, base_session):
        """测试特殊输入：询问平台能力。"""
        result = await intent_analyzer.analyze(
            user_query="你能帮我做什么？有哪些功能？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] response_guide={result.response_guide}")

        # 询问平台能力应为 CHIT_CHAT
        assert result.intent_type == IntentType.CHIT_CHAT

    @pytest.mark.asyncio
    async def test_negation_request(self, intent_analyzer, base_session):
        """测试特殊输入：否定式请求。"""
        result = await intent_analyzer.analyze(
            user_query="我不想去热门景点，有没有小众一点的北京游玩路线？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW

    @pytest.mark.asyncio
    async def test_comparison_request(self, intent_analyzer, base_session):
        """测试特殊输入：比较式请求。"""
        result = await intent_analyzer.analyze(
            user_query="北京和上海哪个更适合带老人旅游？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        # 旅游相关的比较请求
        assert result.intent_type == IntentType.TASK_NEW

    # =========================================================================
    # 多语言/方言测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_english_request(self, intent_analyzer, base_session):
        """测试多语言：英文请求。"""
        result = await intent_analyzer.analyze(
            user_query="Can you help me plan a 3-day trip to Beijing?",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")

        assert result.intent_type == IntentType.TASK_NEW

    @pytest.mark.asyncio
    async def test_mixed_language(self, intent_analyzer, base_session):
        """测试多语言：中英混合。"""
        result = await intent_analyzer.analyze(
            user_query="帮我plan一个Beijing的trip，大概three days",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")

        assert result.intent_type == IntentType.TASK_NEW

    # =========================================================================
    # 隐式意图测试
    # =========================================================================

    @pytest.mark.asyncio
    async def test_implicit_travel_intent(self, intent_analyzer, base_session):
        """测试隐式意图：间接表达旅游需求。"""
        result = await intent_analyzer.analyze(
            user_query="听说故宫很值得去，有什么注意事项吗？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] target_scenario={result.target_scenario}")

        # 虽然是问题形式，但与旅游场景相关
        assert result.intent_type in [IntentType.TASK_NEW, IntentType.CHIT_CHAT]

    @pytest.mark.asyncio
    async def test_hypothetical_travel(self, intent_analyzer, base_session):
        """测试隐式意图：假设性旅游需求。"""
        result = await intent_analyzer.analyze(
            user_query="如果我有一周假期，去北京玩的话怎么安排比较好？",
            session=base_session,
            active_task=None,
        )

        print(f"\n[Result] intent_type={result.intent_type}")
        print(f"[Result] task_instruction={result.task_instruction}")

        assert result.intent_type == IntentType.TASK_NEW


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
