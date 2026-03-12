"""
Leader Agent Platform - IntentAnalyzer 单元测试 (LLM-1)

测试内容：
1. 输入上下文构建 (_build_input_context)
2. 场景信息构建 (_build_scenario_section)
3. 任务信息构建 (_build_task_section)
4. 会话信息构建 (_build_session_section)
5. 系统 Prompt 获取 (_get_system_prompt)
6. 结果验证与规范化 (_validate_and_normalize)
7. analyze 方法的核心逻辑
"""

import sys
from pathlib import Path

_current_dir = Path(__file__).parent
_leader_dir = _current_dir.parent.parent
_project_root = _leader_dir.parent

if str(_leader_dir) not in sys.path:
    sys.path.insert(0, str(_leader_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from assistant.core.intent_analyzer import IntentAnalyzer
from assistant.models import (
    IntentDecision,
    IntentType,
    TaskInstruction,
    Session,
    ActiveTask,
    ScenarioBrief,
    DialogTurn,
    DialogContext,
    ScenarioRuntime,
    UserResult,
    UserResultType,
    ExecutionMode,
    ResponseType,
)
from assistant.models.task import PartnerTask
from assistant.models.base import now_iso
from assistant.models.exceptions import LLMCallError, LLMParseError


@pytest.fixture
def mock_llm_client():
    """创建 Mock LLM 客户端。"""
    client = MagicMock()
    # 默认返回一个正确的 IntentDecision
    client.call_structured = MagicMock(
        return_value=IntentDecision(
            intent_type=IntentType.TASK_NEW,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="我想吃北京烤鸭"),
            response_guide=None,
        )
    )
    return client


@pytest.fixture
def mock_scenario_loader():
    """创建 Mock 场景加载器。"""
    loader = MagicMock()
    # 返回测试场景简介
    loader.scenario_briefs = [
        ScenarioBrief(
            id="beijing_food",
            name="北京美食",
            description="北京美食推荐",
            keywords=["美食", "北京", "烤鸭"],
        ),
        ScenarioBrief(
            id="beijing_hotel",
            name="北京酒店",
            description="北京酒店预订",
            keywords=["酒店", "住宿"],
        ),
    ]
    loader.get_prompt = MagicMock(return_value="你是意图分析助手。")
    loader.get_persona_system = MagicMock(return_value="你是一个旅游助手。")
    loader.get_llm_profile = MagicMock(return_value="gpt-4o")
    loader.get_expert_scenario = MagicMock(return_value=None)
    return loader


@pytest.fixture
def analyzer(mock_llm_client, mock_scenario_loader):
    """创建 IntentAnalyzer 实例。"""
    return IntentAnalyzer(
        scenario_loader=mock_scenario_loader,
        llm_client=mock_llm_client,
    )


@pytest.fixture
def base_session():
    """创建基本测试 Session。"""
    now = now_iso()
    return Session(
        session_id="test-session-001",
        mode=ExecutionMode.DIRECT_RPC,
        created_at=now,
        updated_at=now,
        touched_at=now,
        ttl_seconds=3600,
        expires_at=now,
        base_scenario=ScenarioRuntime(
            id="base", kind="base", version="1.0.0", loaded_at=now
        ),
        user_result=UserResult(
            type=UserResultType.PENDING,
            data_items=[],
            updated_at=now,
        ),
        user_context={"location": "北京"},
    )


@pytest.fixture
def session_with_dialog(base_session):
    """创建带对话上下文的 Session。"""
    now = now_iso()
    base_session.dialog_context = DialogContext(
        session_id=base_session.session_id,
        updated_at=now,
        recent_turns=[
            DialogTurn(
                user_query="你好",
                intent_type=IntentType.CHIT_CHAT,
                response_type=ResponseType.CHAT,
                response_summary="问候",
                timestamp=now,
            ),
            DialogTurn(
                user_query="我想找餐厅",
                intent_type=IntentType.TASK_NEW,
                response_type=ResponseType.PENDING,
                response_summary="开始找餐厅",
                timestamp=now,
            ),
            DialogTurn(
                user_query="北京烤鸭",
                intent_type=IntentType.TASK_INPUT,
                response_type=ResponseType.PENDING,
                response_summary="搜索北京烤鸭",
                timestamp=now,
            ),
        ],
        history_summary="用户在寻找餐厅推荐",
    )
    return base_session


@pytest.fixture
def expert_session():
    """创建专业场景 Session。"""
    now = now_iso()
    return Session(
        session_id="test-session-002",
        mode=ExecutionMode.DIRECT_RPC,
        created_at=now,
        updated_at=now,
        touched_at=now,
        ttl_seconds=3600,
        expires_at=now,
        base_scenario=ScenarioRuntime(
            id="base", kind="base", version="1.0.0", loaded_at=now
        ),
        expert_scenario=ScenarioRuntime(
            id="beijing_food",
            kind="expert",
            version="1.0.0",
            loaded_at=now,
            domain_meta={
                "meta": {
                    "id": "beijing_food",
                    "name": "北京美食",
                    "description": "北京特色美食推荐",
                }
            },
        ),
        user_result=UserResult(
            type=UserResultType.PENDING,
            data_items=[],
            updated_at=now,
        ),
    )


@pytest.fixture
def active_task():
    """创建测试活跃任务。"""
    return ActiveTask(
        active_task_id="task-001",
        created_at=datetime.now().isoformat(),
        external_status="RUNNING",
        partner_tasks={
            "partner-food": PartnerTask(
                partner_aic="partner-food",
                aip_task_id="aip-task-001",
                dimensions=["food"],
                state="working",
            ),
        },
    )


# =============================================================================
# 测试 _build_scenario_section
# =============================================================================


class TestBuildScenarioSection:
    """测试场景信息构建。"""

    def test_base_scenario(self, analyzer, base_session):
        """测试基础场景模式。"""
        result = analyzer._build_scenario_section(base_session)

        assert "scenarioBriefs" in result
        assert len(result["scenarioBriefs"]) == 2
        assert result["expertScenario"] is None

    def test_expert_scenario(self, analyzer, expert_session, mock_scenario_loader):
        """测试专业场景模式。"""
        # 设置返回专业场景信息
        mock_scenario_loader.get_expert_scenario.return_value = MagicMock(
            domain_meta={
                "meta": {
                    "id": "beijing_food",
                    "name": "北京美食",
                    "description": "北京特色美食推荐",
                }
            }
        )

        result = analyzer._build_scenario_section(expert_session)

        assert result["expertScenario"] is not None
        assert result["expertScenario"]["id"] == "beijing_food"
        assert result["expertScenario"]["name"] == "北京美食"

    def test_scenario_briefs_content(self, analyzer, base_session):
        """测试场景简介内容。"""
        result = analyzer._build_scenario_section(base_session)

        briefs = result["scenarioBriefs"]
        assert any(b["id"] == "beijing_food" for b in briefs)
        assert any(b["id"] == "beijing_hotel" for b in briefs)


# =============================================================================
# 测试 _build_task_section
# =============================================================================


class TestBuildTaskSection:
    """测试任务信息构建。"""

    def test_no_active_task(self, analyzer):
        """测试无活跃任务。"""
        result = analyzer._build_task_section(None)

        assert result["activeTask"] is None

    def test_with_active_task(self, analyzer, active_task):
        """测试有活跃任务。"""
        result = analyzer._build_task_section(active_task)

        assert result["activeTask"] is not None
        assert result["activeTask"]["activeTaskId"] == "task-001"
        assert len(result["activeTask"]["partnerTaskSummaries"]) == 1

    def test_partner_task_summary(self, analyzer, active_task):
        """测试 Partner 任务摘要。"""
        result = analyzer._build_task_section(active_task)

        summaries = result["activeTask"]["partnerTaskSummaries"]
        assert summaries[0]["partnerAic"] == "partner-food"
        # partnerName 会 fallback 到 partner_aic
        assert summaries[0]["partnerName"] == "partner-food"


# =============================================================================
# 测试 _build_session_section
# =============================================================================


class TestBuildSessionSection:
    """测试会话信息构建。"""

    def test_with_metadata(self, analyzer, base_session):
        """测试用户上下文元数据。"""
        result = analyzer._build_session_section(base_session)

        assert "userContext" in result
        assert result["userContext"]["location"] == "北京"

    def test_with_dialog_context(self, analyzer, session_with_dialog):
        """测试对话上下文。"""
        result = analyzer._build_session_section(session_with_dialog)

        assert "dialogContext" in result
        assert len(result["dialogContext"]["recentTurns"]) == 3
        assert result["dialogContext"]["historySummary"] == "用户在寻找餐厅推荐"

    def test_with_clarification_user_result(self, analyzer, base_session):
        """测试 clarification 类型的 user_result。"""
        from assistant.models import UserResult, UserResultType, TextDataItem
        from assistant.models.base import now_iso

        base_session.user_result = UserResult(
            type=UserResultType.CLARIFICATION,
            data_items=[
                TextDataItem(text="请问您想吃什么类型的菜？"),
            ],
            updated_at=now_iso(),
        )

        result = analyzer._build_session_section(base_session)

        assert result["userResult"]["type"] == "clarification"
        assert len(result["userResult"]["dataItems"]) == 1
        assert result["userResult"]["dataItems"][0]["type"] == "text"

    def test_pending_user_result(self, analyzer, base_session):
        """测试 pending 类型的 user_result。"""
        result = analyzer._build_session_section(base_session)

        assert result["userResult"]["type"] == "pending"


# =============================================================================
# 测试 _get_system_prompt
# =============================================================================


class TestGetSystemPrompt:
    """测试系统 Prompt 获取。"""

    def test_base_scenario_prompt(self, analyzer, mock_scenario_loader):
        """测试基础场景 Prompt。"""
        result = analyzer._get_system_prompt(None)

        assert "你是一个旅游助手" in result
        assert "你是意图分析助手" in result

    def test_no_prompt_raises_error(self, analyzer, mock_scenario_loader):
        """测试无 Prompt 抛出错误。"""
        mock_scenario_loader.get_prompt.return_value = None

        with pytest.raises(LLMCallError) as exc_info:
            analyzer._get_system_prompt(None)

        assert "prompt not found" in str(exc_info.value)


# =============================================================================
# 测试 _validate_and_normalize
# =============================================================================


class TestValidateAndNormalize:
    """测试结果验证与规范化。"""

    def test_task_input_without_active_task(self, analyzer, base_session):
        """测试 TASK_INPUT 无活跃任务时降级为 TASK_NEW。"""
        result = IntentDecision(
            intent_type=IntentType.TASK_INPUT,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="我想要辣的"),
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.intent_type == IntentType.TASK_NEW

    def test_task_input_with_active_task(self, analyzer, base_session, active_task):
        """测试 TASK_INPUT 有活跃任务时保持不变。"""
        result = IntentDecision(
            intent_type=IntentType.TASK_INPUT,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="我想要辣的"),
        )

        normalized = analyzer._validate_and_normalize(result, base_session, active_task)

        assert normalized.intent_type == IntentType.TASK_INPUT

    def test_chit_chat_clears_task_instruction(self, analyzer, base_session):
        """测试 CHIT_CHAT 清除任务指令。"""
        result = IntentDecision(
            intent_type=IntentType.CHIT_CHAT,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="错误的指令"),
            response_guide="你好！",
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.intent_type == IntentType.CHIT_CHAT
        assert normalized.task_instruction is None
        assert normalized.target_scenario is None
        assert normalized.response_guide is not None

    def test_chit_chat_default_response_guide(self, analyzer, base_session):
        """测试 CHIT_CHAT 默认 response_guide。"""
        result = IntentDecision(
            intent_type=IntentType.CHIT_CHAT,
            task_instruction=TaskInstruction(text="错误"),
            response_guide=None,
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.response_guide is not None
        assert "友好对话" in normalized.response_guide

    def test_task_new_without_instruction(self, analyzer, base_session):
        """测试 TASK_NEW 无指令时自动填充。"""
        result = IntentDecision(
            intent_type=IntentType.TASK_NEW,
            target_scenario="beijing_food",
            task_instruction=None,
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.task_instruction is not None
        assert "待补充详情" in normalized.task_instruction.text

    def test_unknown_scenario_cleared(self, analyzer, base_session):
        """测试未知场景被清除。"""
        result = IntentDecision(
            intent_type=IntentType.TASK_NEW,
            target_scenario="unknown_scenario",
            task_instruction=TaskInstruction(text="测试"),
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.target_scenario is None

    def test_valid_scenario_kept(self, analyzer, base_session):
        """测试有效场景保留。"""
        result = IntentDecision(
            intent_type=IntentType.TASK_NEW,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="测试"),
        )

        normalized = analyzer._validate_and_normalize(result, base_session, None)

        assert normalized.target_scenario == "beijing_food"


# =============================================================================
# 测试 analyze 方法
# =============================================================================


class TestAnalyze:
    """测试 analyze 方法。"""

    @pytest.mark.asyncio
    async def test_analyze_task_new(self, analyzer, base_session, mock_llm_client):
        """测试分析新任务请求。"""
        result = await analyzer.analyze(
            user_query="我想吃北京烤鸭",
            session=base_session,
            active_task=None,
        )

        # 验证 LLM 被调用
        mock_llm_client.call_structured.assert_called_once()
        assert result.intent_type == IntentType.TASK_NEW

    @pytest.mark.asyncio
    async def test_analyze_chit_chat(self, analyzer, base_session, mock_llm_client):
        """测试分析闲聊请求。"""
        mock_llm_client.call_structured.return_value = IntentDecision(
            intent_type=IntentType.CHIT_CHAT,
            response_guide="你好！很高兴为您服务。",
        )

        result = await analyzer.analyze(
            user_query="你好",
            session=base_session,
            active_task=None,
        )

        assert result.intent_type == IntentType.CHIT_CHAT
        assert result.response_guide is not None

    @pytest.mark.asyncio
    async def test_analyze_task_input(
        self, analyzer, base_session, active_task, mock_llm_client
    ):
        """测试分析任务输入。"""
        mock_llm_client.call_structured.return_value = IntentDecision(
            intent_type=IntentType.TASK_INPUT,
            target_scenario="beijing_food",
            task_instruction=TaskInstruction(text="两个人"),
        )

        result = await analyzer.analyze(
            user_query="两个人",
            session=base_session,
            active_task=active_task,
        )

        assert result.intent_type == IntentType.TASK_INPUT

    @pytest.mark.asyncio
    async def test_analyze_llm_error(self, analyzer, base_session, mock_llm_client):
        """测试 LLM 调用失败。"""
        mock_llm_client.call_structured.side_effect = Exception("LLM unavailable")

        with pytest.raises(LLMCallError) as exc_info:
            await analyzer.analyze(
                user_query="测试",
                session=base_session,
                active_task=None,
            )

        assert "failed" in str(exc_info.value).lower()


# =============================================================================
# 测试 _build_input_context
# =============================================================================


class TestBuildInputContext:
    """测试输入上下文构建。"""

    def test_basic_input_context(self, analyzer, base_session):
        """测试基本输入上下文。"""
        context = analyzer._build_input_context(
            user_query="我想找餐厅",
            session=base_session,
            active_task=None,
        )

        # 检查 userQuery 使用 alias
        dumped = context.model_dump(by_alias=True, exclude_none=True)
        assert dumped.get("userQuery") == "我想找餐厅"
        assert "scenario" in dumped
        assert "session" in dumped

    def test_input_context_with_active_task(self, analyzer, base_session, active_task):
        """测试有活跃任务的输入上下文。"""
        context = analyzer._build_input_context(
            user_query="我要两份",
            session=base_session,
            active_task=active_task,
        )

        dumped = context.model_dump(by_alias=True, exclude_none=True)
        assert "task" in dumped
        assert dumped["task"]["activeTask"] is not None


# =============================================================================
# 边界测试
# =============================================================================


class TestEdgeCases:
    """测试边界情况。"""

    def test_empty_user_query(self, analyzer, base_session):
        """测试空用户输入。"""
        context = analyzer._build_input_context(
            user_query="",
            session=base_session,
            active_task=None,
        )

        dumped = context.model_dump(by_alias=True, exclude_none=True)
        assert dumped.get("userQuery") == ""

    def test_very_long_user_query(self, analyzer, base_session):
        """测试超长用户输入。"""
        long_query = "测试" * 1000
        context = analyzer._build_input_context(
            user_query=long_query,
            session=base_session,
            active_task=None,
        )

        dumped = context.model_dump(by_alias=True, exclude_none=True)
        assert dumped.get("userQuery") == long_query

    def test_session_no_metadata(self, analyzer):
        """测试无元数据的会话。"""
        now = now_iso()
        session = Session(
            session_id="test",
            mode=ExecutionMode.DIRECT_RPC,
            created_at=now,
            updated_at=now,
            touched_at=now,
            ttl_seconds=3600,
            expires_at=now,
            base_scenario=ScenarioRuntime(
                id="base", kind="base", version="1.0.0", loaded_at=now
            ),
            user_result=UserResult(
                type=UserResultType.PENDING,
                data_items=[],
                updated_at=now,
            ),
            user_context={},  # 空字典
        )

        result = analyzer._build_session_section(session)

        # 不应包含 userContext
        assert "userContext" not in result or result.get("userContext") == {}

    def test_empty_scenario_briefs(self, analyzer, base_session, mock_scenario_loader):
        """测试空场景列表。"""
        mock_scenario_loader.scenario_briefs = []

        result = analyzer._build_scenario_section(base_session)

        assert result["scenarioBriefs"] == []
