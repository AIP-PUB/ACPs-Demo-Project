"""
LLM-6 Aggregator 单元测试

测试内容：
1. _build_input_json 方法
2. _parse_llm_response 方法
3. _fallback_aggregate 方法
4. 空输入处理
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from leader.assistant.core.aggregator import (
    Aggregator,
    AggregationResult,
    AggregationInput,
    PartnerOutput,
    DegradationInfo,
    get_aggregator,
)
from acps_sdk.aip.aip_base_model import TaskState


class TestPartnerOutputModel:
    """测试 PartnerOutput 数据模型"""

    def test_basic_creation(self):
        """测试基本创建"""
        po = PartnerOutput(
            partner_aic="partner-001",
            dimension_id="hotel",
            state="completed",
            data_items=[{"text": "推荐酒店A"}],
        )
        assert po.partner_aic == "partner-001"
        assert po.dimension_id == "hotel"
        assert po.state == "completed"
        assert len(po.data_items) == 1

    def test_with_alias(self):
        """测试 alias 字段"""
        po = PartnerOutput(
            partnerAic="partner-001",
            dimensionId="food",
            state="completed",
            dataItems=[{"text": "美食推荐"}],
        )
        assert po.partner_aic == "partner-001"
        assert po.dimension_id == "food"

    def test_with_optional_fields(self):
        """测试可选字段"""
        po = PartnerOutput(
            partner_aic="partner-001",
            dimension_id="hotel",
            state="failed",
            partner_name="酒店助手",
            error="连接超时",
        )
        assert po.partner_name == "酒店助手"
        assert po.error == "连接超时"


class TestDegradationInfoModel:
    """测试 DegradationInfo 数据模型"""

    def test_basic_creation(self):
        """测试基本创建"""
        deg = DegradationInfo(
            dimension_id="transport",
            reason="服务不可用",
        )
        assert deg.dimension_id == "transport"
        assert deg.reason == "服务不可用"
        assert deg.suggestion is None

    def test_with_suggestion(self):
        """测试包含建议"""
        deg = DegradationInfo(
            dimension_id="transport",
            reason="服务不可用",
            suggestion="请稍后重试",
        )
        assert deg.suggestion == "请稍后重试"


class TestAggregationResultModel:
    """测试 AggregationResult 数据模型"""

    def test_basic_creation(self):
        """测试基本创建"""
        result = AggregationResult(
            type="final",
            text="这是整合后的结果",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        assert result.type == "final"
        assert result.text == "这是整合后的结果"

    def test_with_structured_data(self):
        """测试结构化数据"""
        result = AggregationResult(
            type="final",
            text="结果",
            structured={"hotels": [{"name": "酒店A"}]},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        assert result.structured is not None
        assert "hotels" in result.structured


class TestAggregatorBuildInputJson:
    """测试 _build_input_json 方法"""

    @pytest.fixture
    def mock_aggregator(self):
        """创建 mock aggregator"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()
        return Aggregator(llm_client=mock_llm, scenario_loader=mock_loader)

    def test_basic_input(self, mock_aggregator):
        """测试基本输入构建"""
        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state="completed",
                data_items=[{"text": "酒店推荐"}],
            )
        ]

        result = mock_aggregator._build_input_json(
            partner_outputs=partner_outputs,
            degradations=[],
            user_query="我想预订酒店",
            dialog_summary=None,
            user_constraints=None,
        )

        assert result["userQuery"] == "我想预订酒店"
        assert len(result["partnerOutputs"]) == 1
        assert result["partnerOutputs"][0]["partnerAic"] == "partner-001"
        assert result["partnerOutputs"][0]["dimensionId"] == "hotel"

    def test_with_multiple_partners(self, mock_aggregator):
        """测试多个 Partner 输出"""
        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state="completed",
                data_items=[{"text": "酒店A"}],
            ),
            PartnerOutput(
                partner_aic="partner-002",
                dimension_id="food",
                state="completed",
                data_items=[{"text": "餐厅B"}],
            ),
        ]

        result = mock_aggregator._build_input_json(
            partner_outputs=partner_outputs,
            degradations=[],
            user_query="查询行程",
            dialog_summary=None,
            user_constraints=None,
        )

        assert len(result["partnerOutputs"]) == 2

    def test_with_degradations(self, mock_aggregator):
        """测试包含降级信息"""
        degradations = [
            DegradationInfo(
                dimension_id="transport",
                reason="服务不可用",
                suggestion="请稍后重试",
            )
        ]

        result = mock_aggregator._build_input_json(
            partner_outputs=[],
            degradations=degradations,
            user_query="查询行程",
            dialog_summary=None,
            user_constraints=None,
        )

        assert len(result["degradations"]) == 1
        assert result["degradations"][0]["dimensionId"] == "transport"
        assert result["degradations"][0]["reason"] == "服务不可用"

    def test_with_dialog_summary(self, mock_aggregator):
        """测试包含对话摘要"""
        result = mock_aggregator._build_input_json(
            partner_outputs=[],
            degradations=[],
            user_query="查询行程",
            dialog_summary="用户之前询问了酒店价格...",
            user_constraints=None,
        )

        assert result["dialogSummary"] == "用户之前询问了酒店价格..."

    def test_with_user_constraints(self, mock_aggregator):
        """测试包含用户约束"""
        constraints = {"budget": "1000元以内", "date": "明天"}

        result = mock_aggregator._build_input_json(
            partner_outputs=[],
            degradations=[],
            user_query="查询行程",
            dialog_summary=None,
            user_constraints=constraints,
        )

        assert result["userConstraints"] == constraints


class TestAggregatorParseLlmResponse:
    """测试 _parse_llm_response 方法

    注意：当前实现已改为直接输出 Markdown 文本，不再解析 JSON。
    LLM 直接返回可读的 Markdown 格式文本。
    """

    @pytest.fixture
    def mock_aggregator(self):
        """创建 mock aggregator"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()
        return Aggregator(llm_client=mock_llm, scenario_loader=mock_loader)

    def test_plain_text_response(self, mock_aggregator):
        """测试纯文本响应"""
        llm_response = "这是整合后的结果"

        result = mock_aggregator._parse_llm_response(llm_response, [])

        assert result.type == "final"
        assert result.text == "这是整合后的结果"

    def test_markdown_code_block_stripped(self, mock_aggregator):
        """测试 Markdown 代码块被正确移除"""
        llm_response = """```markdown
这是整合后的结果
```"""

        result = mock_aggregator._parse_llm_response(llm_response, [])

        assert result.type == "final"
        assert result.text == "这是整合后的结果"

    def test_plain_code_block_stripped(self, mock_aggregator):
        """测试普通代码块被正确移除"""
        llm_response = """```
整合结果内容
```"""

        result = mock_aggregator._parse_llm_response(llm_response, [])

        assert result.type == "final"
        assert result.text == "整合结果内容"

    def test_text_without_code_block(self, mock_aggregator):
        """测试没有代码块的普通文本"""
        llm_response = "这不是有效的 JSON，只是普通文本回复"

        result = mock_aggregator._parse_llm_response(llm_response, [])

        # 应该将整个响应作为文本
        assert result.type == "final"
        assert result.text == "这不是有效的 JSON，只是普通文本回复"

    def test_whitespace_trimmed(self, mock_aggregator):
        """测试空白字符被正确处理"""
        llm_response = "   只有 text   "

        result = mock_aggregator._parse_llm_response(llm_response, [])

        assert result.type == "final"
        assert result.text == "只有 text"


class TestAggregatorFallbackAggregate:
    """测试 _fallback_aggregate 降级整合方法"""

    @pytest.fixture
    def mock_aggregator(self):
        """创建 mock aggregator"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()
        return Aggregator(llm_client=mock_llm, scenario_loader=mock_loader)

    def test_basic_fallback(self, mock_aggregator):
        """测试基本降级整合"""
        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[{"text": "酒店推荐内容"}],
            )
        ]

        result = mock_aggregator._fallback_aggregate(
            partner_outputs=partner_outputs,
            degradations=[],
            user_query="查询酒店",
        )

        assert result.type == "final"
        assert "查询结果" in result.text
        assert "酒店推荐内容" in result.text
        assert result.structured["fallback"] is True

    def test_fallback_with_failures(self, mock_aggregator):
        """测试包含失败的降级整合"""
        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state=TaskState.Failed.value,
                data_items=[],
                error="连接超时",
            )
        ]

        result = mock_aggregator._fallback_aggregate(
            partner_outputs=partner_outputs,
            degradations=[],
            user_query="查询酒店",
        )

        assert "失败" in result.text or "连接超时" in result.text

    def test_fallback_with_degradations(self, mock_aggregator):
        """测试包含降级说明"""
        degradations = [
            DegradationInfo(
                dimension_id="transport",
                reason="服务维护中",
                suggestion="请明天再试",
            )
        ]

        result = mock_aggregator._fallback_aggregate(
            partner_outputs=[],
            degradations=degradations,
            user_query="查询行程",
        )

        assert "注意事项" in result.text
        assert "服务维护中" in result.text
        assert "请明天再试" in result.text

    def test_empty_inputs(self, mock_aggregator):
        """测试空输入"""
        result = mock_aggregator._fallback_aggregate(
            partner_outputs=[],
            degradations=[],
            user_query="",
        )

        assert result.type == "final"
        assert result.text  # 不应为空


class TestAggregatorAggregate:
    """测试 aggregate 主方法"""

    @pytest.fixture
    def mock_aggregator(self):
        """创建 mock aggregator"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()
        mock_loader.get_merged_prompts.return_value = {
            "aggregation.system": "你是结果整合器...",
            "aggregation.llm_profile": "llm.default",
        }
        return Aggregator(llm_client=mock_llm, scenario_loader=mock_loader)

    @pytest.mark.asyncio
    async def test_empty_inputs_returns_empty_result(self, mock_aggregator):
        """测试空输入返回空结果"""
        result = await mock_aggregator.aggregate(
            partner_outputs=[],
            degradations=[],
            user_query="",
        )

        assert result.type == "final"
        assert "暂无" in result.text

    @pytest.mark.asyncio
    async def test_successful_aggregation(self, mock_aggregator):
        """测试成功的聚合

        当前实现 LLM 直接返回 Markdown 文本，不再返回 JSON。
        """
        # 模拟 LLM 返回 Markdown 文本
        mock_aggregator._llm_client.call.return_value = "整合后的结果"

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state="completed",
                data_items=[{"text": "酒店信息"}],
            )
        ]

        result = await mock_aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="查询酒店",
        )

        assert result.type == "final"
        assert result.text == "整合后的结果"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, mock_aggregator):
        """测试 LLM 失败时降级"""
        mock_aggregator._llm_client.call.side_effect = Exception("LLM 调用失败")

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[{"text": "酒店信息"}],
            )
        ]

        result = await mock_aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="查询酒店",
        )

        # 应该使用降级逻辑
        assert result.type == "final"
        assert result.structured.get("fallback") is True


class TestGetDimensionDisplayName:
    """测试维度名称映射"""

    @pytest.fixture
    def mock_aggregator(self):
        """创建 mock aggregator"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()
        return Aggregator(llm_client=mock_llm, scenario_loader=mock_loader)

    def test_known_dimensions(self, mock_aggregator):
        """测试已知维度"""
        assert mock_aggregator._get_dimension_display_name("hotel") == "酒店住宿"
        assert mock_aggregator._get_dimension_display_name("transport") == "交通出行"
        assert mock_aggregator._get_dimension_display_name("food") == "餐饮美食"
        assert mock_aggregator._get_dimension_display_name("scenic") == "景点游览"

    def test_unknown_dimension(self, mock_aggregator):
        """测试未知维度返回原始 ID"""
        assert mock_aggregator._get_dimension_display_name("unknown") == "unknown"
        assert mock_aggregator._get_dimension_display_name("custom_dim") == "custom_dim"


class TestGetAggregator:
    """测试工厂函数"""

    def test_creates_with_defaults(self):
        """测试使用默认参数创建"""
        # 由于依赖外部服务，可能需要 mock
        with patch("assistant.llm.client.get_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()
            aggregator = get_aggregator()
            assert aggregator is not None
            assert isinstance(aggregator, Aggregator)

    def test_creates_with_injected_dependencies(self):
        """测试注入依赖创建"""
        mock_llm = MagicMock()
        mock_loader = MagicMock()

        aggregator = get_aggregator(
            llm_client=mock_llm,
            scenario_loader=mock_loader,
        )

        assert aggregator is not None
        assert aggregator._llm_client is mock_llm
        assert aggregator._scenario_loader is mock_loader
