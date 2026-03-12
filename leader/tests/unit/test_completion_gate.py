"""
Leader Agent Platform - CompletionGate 单元测试 (LLM-5)

测试内容：
1. 数据模型 (ConflictInfo, FollowupDirective, AwaitingCompletionDecision, etc.)
2. 输入 JSON 构建 (_build_input_json)
3. LLM 响应解析 (_parse_llm_response)
4. 降级处理 (_fallback_result)
5. Prompt 配置加载 (_load_prompt_config)
6. 一致性规则加载 (_load_consistency_rules)
7. evaluate 方法的核心逻辑
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
import json
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from assistant.core.completion_gate import (
    CompletionGate,
    ConflictInfo,
    FollowupDirective,
    AwaitingCompletionDecision,
    AwaitingCompletionGateResult,
    PartnerProductSummary,
)
from acps_sdk.aip.aip_base_model import TaskState


@pytest.fixture
def mock_llm_client():
    """创建 Mock LLM 客户端。"""
    client = MagicMock()
    client.call = MagicMock(
        return_value=json.dumps(
            {
                "decidedAt": "2024-01-01T00:00:00Z",
                "decisions": [
                    {
                        "partnerAic": "partner-food",
                        "aipTaskId": "task-001",
                        "nextAction": "complete",
                        "conflicts": [],
                    }
                ],
            }
        )
    )
    return client


@pytest.fixture
def mock_scenario_loader():
    """创建 Mock 场景加载器。"""
    loader = MagicMock()
    loader.get_expert_scenario = MagicMock(
        return_value=MagicMock(
            domain=MagicMock(consistency_rules={"time_conflict": "检查时间冲突"})
        )
    )
    loader.get_merged_prompts = MagicMock(
        return_value={
            "completion_gate.system": "你是完成闸门助手",
            "completion_gate.llm_profile": "gpt-4o",
        }
    )
    return loader


@pytest.fixture
def gate(mock_llm_client, mock_scenario_loader):
    """创建 CompletionGate 实例。"""
    return CompletionGate(
        llm_client=mock_llm_client,
        scenario_loader=mock_scenario_loader,
    )


@pytest.fixture
def awaiting_partner():
    """创建等待完成的 Partner 摘要。"""
    return PartnerProductSummary(
        partner_aic="partner-food",
        aip_task_id="task-001",
        dimension_id="food",
        state=TaskState.AwaitingCompletion.value,
        data_items=[{"type": "text", "text": "推荐北京烤鸭"}],
        products=[],
    )


@pytest.fixture
def working_partner():
    """创建工作中的 Partner 摘要。"""
    return PartnerProductSummary(
        partner_aic="partner-hotel",
        aip_task_id="task-002",
        dimension_id="hotel",
        state=TaskState.Working.value,
        data_items=[],
        products=[],
    )


# =============================================================================
# 测试数据模型
# =============================================================================


class TestConflictInfo:
    """测试 ConflictInfo 数据模型。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        conflict = ConflictInfo(
            code="TIME_CONFLICT",
            message="时间冲突",
        )
        assert conflict.code == "TIME_CONFLICT"
        assert conflict.message == "时间冲突"
        assert conflict.details is None

    def test_with_details(self):
        """测试带详情创建。"""
        conflict = ConflictInfo(
            code="BUDGET_EXCEED",
            message="超出预算",
            details={"budget": 1000, "actual": 1500},
        )
        assert conflict.details is not None
        assert conflict.details["budget"] == 1000


class TestFollowupDirective:
    """测试 FollowupDirective 数据模型。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        followup = FollowupDirective(
            text="请提供更便宜的选项",
        )
        assert followup.text == "请提供更便宜的选项"
        assert followup.data is None

    def test_with_data(self):
        """测试带数据创建。"""
        followup = FollowupDirective(
            text="调整预算",
            data={"max_budget": 800},
        )
        assert followup.data["max_budget"] == 800


class TestAwaitingCompletionDecision:
    """测试 AwaitingCompletionDecision 数据模型。"""

    def test_complete_decision(self):
        """测试 complete 决策。"""
        decision = AwaitingCompletionDecision(
            partner_aic="partner-001",
            aip_task_id="task-001",
            next_action="complete",
        )
        assert decision.next_action == "complete"
        assert decision.followup is None
        assert decision.conflicts == []

    def test_continue_decision(self):
        """测试 continue 决策。"""
        decision = AwaitingCompletionDecision(
            partner_aic="partner-001",
            aip_task_id="task-001",
            next_action="continue",
            followup=FollowupDirective(text="请调整"),
            conflicts=[ConflictInfo(code="ERR", message="错误")],
        )
        assert decision.next_action == "continue"
        assert decision.followup is not None
        assert len(decision.conflicts) == 1

    def test_alias_serialization(self):
        """测试别名序列化。"""
        decision = AwaitingCompletionDecision(
            partner_aic="partner-001",
            aip_task_id="task-001",
            next_action="complete",
        )
        dumped = decision.model_dump(by_alias=True)
        assert "partnerAic" in dumped
        assert "aipTaskId" in dumped
        assert "nextAction" in dumped


class TestAwaitingCompletionGateResult:
    """测试 AwaitingCompletionGateResult 数据模型。"""

    def test_empty_decisions(self):
        """测试空决策列表。"""
        result = AwaitingCompletionGateResult(
            decided_at="2024-01-01T00:00:00Z",
            decisions=[],
        )
        assert result.decisions == []

    def test_with_decisions(self):
        """测试带决策列表。"""
        decision = AwaitingCompletionDecision(
            partner_aic="p1",
            aip_task_id="t1",
            next_action="complete",
        )
        result = AwaitingCompletionGateResult(
            decided_at="2024-01-01T00:00:00Z",
            decisions=[decision],
        )
        assert len(result.decisions) == 1


class TestPartnerProductSummary:
    """测试 PartnerProductSummary 数据模型。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        summary = PartnerProductSummary(
            partner_aic="partner-001",
            aip_task_id="task-001",
            dimension_id="food",
            state="AwaitingCompletion",
        )
        assert summary.partner_aic == "partner-001"
        assert summary.data_items == []
        assert summary.products == []

    def test_with_data(self):
        """测试带数据创建。"""
        summary = PartnerProductSummary(
            partner_aic="partner-001",
            aip_task_id="task-001",
            dimension_id="food",
            state="AwaitingCompletion",
            data_items=[{"type": "text", "text": "推荐"}],
            products=[{"name": "产品1"}],
        )
        assert len(summary.data_items) == 1
        assert len(summary.products) == 1


# =============================================================================
# 测试 _build_input_json
# =============================================================================


class TestBuildInputJson:
    """测试输入 JSON 构建。"""

    def test_basic_build(self, gate, awaiting_partner):
        """测试基本构建。"""
        result = gate._build_input_json(
            awaiting_partners=[awaiting_partner],
            user_constraints=None,
            consistency_rules={},
        )

        assert "partnerProducts" in result
        assert len(result["partnerProducts"]) == 1
        assert result["partnerProducts"][0]["partnerAic"] == "partner-food"
        assert result["userConstraints"] == {}
        assert result["consistencyRules"] == {}

    def test_with_constraints(self, gate, awaiting_partner):
        """测试带约束构建。"""
        constraints = {"max_budget": 1000, "cuisine": "中餐"}

        result = gate._build_input_json(
            awaiting_partners=[awaiting_partner],
            user_constraints=constraints,
            consistency_rules={"rule1": "规则1"},
        )

        assert result["userConstraints"]["max_budget"] == 1000
        assert result["consistencyRules"]["rule1"] == "规则1"

    def test_multiple_partners(self, gate, awaiting_partner):
        """测试多个 Partner。"""
        partner2 = PartnerProductSummary(
            partner_aic="partner-hotel",
            aip_task_id="task-002",
            dimension_id="hotel",
            state=TaskState.AwaitingCompletion.value,
            data_items=[{"type": "text", "text": "推荐酒店"}],
        )

        result = gate._build_input_json(
            awaiting_partners=[awaiting_partner, partner2],
            user_constraints=None,
            consistency_rules={},
        )

        assert len(result["partnerProducts"]) == 2


# =============================================================================
# 测试 _parse_llm_response
# =============================================================================


class TestParseLlmResponse:
    """测试 LLM 响应解析。"""

    def test_parse_valid_response(self, gate, awaiting_partner):
        """测试解析有效响应。"""
        response = json.dumps(
            {
                "decidedAt": "2024-01-01T00:00:00Z",
                "decisions": [
                    {
                        "partnerAic": "partner-food",
                        "aipTaskId": "task-001",
                        "nextAction": "complete",
                        "conflicts": [],
                    }
                ],
            }
        )

        result = gate._parse_llm_response(response, [awaiting_partner])

        assert result.decided_at == "2024-01-01T00:00:00Z"
        assert len(result.decisions) == 1
        assert result.decisions[0].next_action == "complete"

    def test_parse_with_markdown(self, gate, awaiting_partner):
        """测试解析带 markdown 标记的响应。"""
        response = """```json
{
    "decidedAt": "2024-01-01T00:00:00Z",
    "decisions": [
        {
            "partnerAic": "partner-food",
            "aipTaskId": "task-001",
            "nextAction": "complete",
            "conflicts": []
        }
    ]
}
```"""

        result = gate._parse_llm_response(response, [awaiting_partner])

        assert len(result.decisions) == 1

    def test_parse_continue_decision(self, gate, awaiting_partner):
        """测试解析 continue 决策。"""
        response = json.dumps(
            {
                "decidedAt": "2024-01-01T00:00:00Z",
                "decisions": [
                    {
                        "partnerAic": "partner-food",
                        "aipTaskId": "task-001",
                        "nextAction": "continue",
                        "followup": {
                            "text": "请提供更便宜的选项",
                            "data": {"max_price": 100},
                        },
                        "conflicts": [
                            {
                                "code": "BUDGET_EXCEED",
                                "message": "超出预算",
                            }
                        ],
                    }
                ],
            }
        )

        result = gate._parse_llm_response(response, [awaiting_partner])

        decision = result.decisions[0]
        assert decision.next_action == "continue"
        assert decision.followup is not None
        assert decision.followup.text == "请提供更便宜的选项"
        assert len(decision.conflicts) == 1

    def test_parse_invalid_json(self, gate, awaiting_partner):
        """测试解析无效 JSON 时降级。"""
        response = "这不是有效的 JSON"

        result = gate._parse_llm_response(response, [awaiting_partner])

        # 应该触发降级
        assert len(result.decisions) == 1
        assert result.decisions[0].next_action == "complete"
        assert any(c.code == "FALLBACK" for c in result.decisions[0].conflicts)


# =============================================================================
# 测试 _fallback_result
# =============================================================================


class TestFallbackResult:
    """测试降级结果。"""

    def test_single_partner_fallback(self, gate, awaiting_partner):
        """测试单个 Partner 降级。"""
        result = gate._fallback_result([awaiting_partner])

        assert len(result.decisions) == 1
        assert result.decisions[0].partner_aic == "partner-food"
        assert result.decisions[0].next_action == "complete"
        assert any(c.code == "FALLBACK" for c in result.decisions[0].conflicts)

    def test_multiple_partners_fallback(self, gate, awaiting_partner):
        """测试多个 Partner 降级。"""
        partner2 = PartnerProductSummary(
            partner_aic="partner-hotel",
            aip_task_id="task-002",
            dimension_id="hotel",
            state=TaskState.AwaitingCompletion.value,
        )

        result = gate._fallback_result([awaiting_partner, partner2])

        assert len(result.decisions) == 2
        partner_aics = {d.partner_aic for d in result.decisions}
        assert "partner-food" in partner_aics
        assert "partner-hotel" in partner_aics


# =============================================================================
# 测试 _load_prompt_config
# =============================================================================


class TestLoadPromptConfig:
    """测试 Prompt 配置加载。"""

    def test_load_from_scenario(self, gate, mock_scenario_loader):
        """测试从场景加载。"""
        result = gate._load_prompt_config("beijing_food")

        assert result["system"] == "你是完成闸门助手"
        assert result["llm_profile"] == "gpt-4o"

    def test_load_default_on_error(self, gate, mock_scenario_loader):
        """测试加载失败时使用默认配置。"""
        mock_scenario_loader.get_merged_prompts.side_effect = Exception("加载失败")

        result = gate._load_prompt_config("unknown_scenario")

        assert "system" in result
        assert "llm_profile" in result
        assert result["llm_profile"] == "llm.fast"  # CompletionGate 使用 fast profile

    def test_load_default_for_none_scenario(self, gate, mock_scenario_loader):
        """测试无场景时使用默认配置。"""
        mock_scenario_loader.get_merged_prompts.return_value = {}

        result = gate._load_prompt_config(None)

        assert "system" in result


# =============================================================================
# 测试 _load_consistency_rules
# =============================================================================


class TestLoadConsistencyRules:
    """测试一致性规则加载。"""

    def test_load_rules(self, gate, mock_scenario_loader):
        """测试加载规则。"""
        result = gate._load_consistency_rules("beijing_food")

        assert "time_conflict" in result

    def test_no_scenario(self, gate):
        """测试无场景时返回空。"""
        result = gate._load_consistency_rules(None)

        assert result == {}

    def test_load_error(self, gate, mock_scenario_loader):
        """测试加载错误时返回空。"""
        mock_scenario_loader.get_expert_scenario.side_effect = Exception("错误")

        result = gate._load_consistency_rules("beijing_food")

        assert result == {}


# =============================================================================
# 测试 evaluate 方法
# =============================================================================


class TestEvaluate:
    """测试 evaluate 方法。"""

    @pytest.mark.asyncio
    async def test_evaluate_empty_list(self, gate):
        """测试空列表。"""
        result = await gate.evaluate(
            partner_summaries=[],
            user_constraints=None,
            scenario_id=None,
        )

        assert result.decisions == []

    @pytest.mark.asyncio
    async def test_evaluate_no_awaiting_partners(self, gate, working_partner):
        """测试无等待完成的 Partner。"""
        result = await gate.evaluate(
            partner_summaries=[working_partner],
            user_constraints=None,
            scenario_id=None,
        )

        assert result.decisions == []

    @pytest.mark.asyncio
    async def test_evaluate_with_awaiting_partner(
        self, gate, awaiting_partner, mock_llm_client
    ):
        """测试有等待完成的 Partner。"""
        result = await gate.evaluate(
            partner_summaries=[awaiting_partner],
            user_constraints=None,
            scenario_id="beijing_food",
        )

        mock_llm_client.call.assert_called_once()
        assert len(result.decisions) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_constraints(
        self, gate, awaiting_partner, mock_llm_client
    ):
        """测试带约束的评估。"""
        constraints = {"max_budget": 500}

        result = await gate.evaluate(
            partner_summaries=[awaiting_partner],
            user_constraints=constraints,
            scenario_id="beijing_food",
        )

        # 验证 LLM 被调用
        mock_llm_client.call.assert_called_once()
        # 验证返回了决策结果
        assert result is not None
        assert len(result.decisions) == 1


# =============================================================================
# 边界测试
# =============================================================================


class TestEdgeCases:
    """测试边界情况。"""

    def test_parse_empty_decisions(self, gate, awaiting_partner):
        """测试解析空决策列表。"""
        response = json.dumps(
            {
                "decidedAt": "2024-01-01T00:00:00Z",
                "decisions": [],
            }
        )

        result = gate._parse_llm_response(response, [awaiting_partner])

        assert result.decisions == []

    def test_parse_missing_fields(self, gate, awaiting_partner):
        """测试解析缺少字段的响应。"""
        response = json.dumps(
            {
                "decidedAt": "2024-01-01T00:00:00Z",
                "decisions": [
                    {
                        "partnerAic": "partner-food",
                        "aipTaskId": "task-001",
                        "nextAction": "complete",
                        # 缺少 conflicts 和 followup
                    }
                ],
            }
        )

        result = gate._parse_llm_response(response, [awaiting_partner])

        assert len(result.decisions) == 1
        assert result.decisions[0].conflicts == []
        assert result.decisions[0].followup is None

    def test_default_prompt_config(self, gate):
        """测试默认 Prompt 配置。"""
        config = gate._default_prompt_config()

        assert "system" in config
        assert "llm_profile" in config
        assert "{{input_json}}" in config["system"]
