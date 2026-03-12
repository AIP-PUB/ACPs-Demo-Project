"""
Leader Agent Platform - 集成测试：LLM-6 结果整合流程

测试从 Partner 产出到最终用户结果的完整整合流程，验证：
1. Aggregator 正确调用 LLM-6
2. 多 Partner 结果正确整合
3. 降级信息正确处理
4. 失败场景的优雅降级
5. 场景配置正确加载

本测试使用真实大模型 API 调用。
"""

import sys
import os

# 确保 leader 目录在 path 中（与 conftest.py 保持一致）
leader_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if leader_dir not in sys.path:
    sys.path.insert(0, leader_dir)

# 确保项目根目录在 path 中（用于导入 acps_sdk.aip）
project_root = os.path.dirname(leader_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from unittest.mock import MagicMock, patch

from assistant.core.aggregator import (
    Aggregator,
    AggregationResult,
    PartnerOutput,
    DegradationInfo,
    get_aggregator,
)
from assistant.services import ScenarioLoader
from acps_sdk.aip.aip_base_model import TaskState

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


class TestAggregatorLLMIntegration:
    """LLM-6 结果整合完整流程测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_basic_aggregation_with_llm(self, app: FastAPI):
        """测试：基本 LLM 整合调用。"""
        aggregator = get_aggregator()

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                partner_name="酒店服务",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "推荐酒店：北京国贸大酒店，价格：800元/晚，评分：4.8"},
                    {"text": "推荐酒店：王府井希尔顿，价格：1200元/晚，评分：4.9"},
                ],
            )
        ]

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="帮我推荐北京的酒店",
        )

        # === 验证结果结构 ===
        assert result.type == "final"
        assert result.text is not None
        assert len(result.text) > 20

        print(f"\n[Aggregation] type={result.type}")
        print(f"[Aggregation] text length={len(result.text)} chars")
        print(f"[Aggregation] text preview={result.text[:100]}...")

    @pytest.mark.asyncio
    async def test_multi_partner_aggregation(self, app: FastAPI):
        """测试：多 Partner 结果整合。"""
        aggregator = get_aggregator()

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                partner_name="酒店服务",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "酒店推荐：北京国贸大酒店，800元/晚"},
                ],
            ),
            PartnerOutput(
                partner_aic="partner-food-001",
                partner_name="餐饮服务",
                dimension_id="food",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "餐厅推荐：全聚德烤鸭店，人均消费200元"},
                ],
            ),
        ]

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="帮我安排北京一日游",
        )

        # === 验证结果 ===
        assert result.type == "final"
        assert result.text is not None
        assert len(result.text) > 50

        print(f"\n[MultiPartner] text length={len(result.text)} chars")
        print(f"[MultiPartner] text preview={result.text[:150]}...")

    @pytest.mark.asyncio
    async def test_aggregation_with_degradation(self, app: FastAPI):
        """测试：包含降级信息的整合。"""
        aggregator = get_aggregator()

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "酒店推荐：北京国贸大酒店"},
                ],
            ),
        ]

        degradations = [
            DegradationInfo(
                dimension_id="transport",
                reason="交通服务暂时不可用",
                suggestion="您可以稍后查询或使用其他交通方式",
            ),
        ]

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            degradations=degradations,
            user_query="帮我安排北京一日游，包括交通",
        )

        # === 验证结果 ===
        assert result.type == "final"
        assert result.text is not None

        print(f"\n[WithDegradation] text length={len(result.text)} chars")
        print(f"[WithDegradation] text preview={result.text[:150]}...")

    @pytest.mark.asyncio
    async def test_aggregation_with_dialog_summary(self, app: FastAPI):
        """测试：包含对话摘要的整合。"""
        aggregator = get_aggregator()

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "推荐酒店：北京国贸大酒店，价格：800元/晚"},
                ],
            ),
        ]

        dialog_summary = """
        用户之前询问了以下内容：
        - 用户预算：500-1000元/晚
        - 偏好：安静、交通便利
        - 入住日期：明天
        """

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="帮我推荐北京的酒店",
            dialog_summary=dialog_summary,
        )

        # === 验证结果 ===
        assert result.type == "final"
        assert result.text is not None

        print(f"\n[WithDialogSummary] text length={len(result.text)} chars")

    @pytest.mark.asyncio
    async def test_aggregation_with_failed_partner(self, app: FastAPI):
        """测试：处理失败的 Partner。"""
        aggregator = get_aggregator()

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[
                    {"text": "酒店推荐：北京国贸大酒店"},
                ],
            ),
            PartnerOutput(
                partner_aic="partner-transport-001",
                dimension_id="transport",
                state=TaskState.Failed.value,
                data_items=[],
                error="网络连接超时",
            ),
        ]

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="帮我安排北京一日游",
        )

        # === 验证结果 ===
        assert result.type == "final"
        assert result.text is not None

        print(f"\n[WithFailedPartner] text length={len(result.text)} chars")
        print(f"[WithFailedPartner] text preview={result.text[:150]}...")


class TestAggregatorScenarioConfig:
    """Aggregator 场景配置测试。"""

    @pytest.mark.asyncio
    async def test_scenario_prompts_loading(self, scenario_loader: ScenarioLoader):
        """测试：场景 prompt 配置正确加载。"""
        # 获取合并后的 prompts
        merged_prompts = scenario_loader.get_merged_prompts("tour")

        # === 验证 aggregation 配置 ===
        has_aggregation_system = "aggregation.system" in merged_prompts
        has_aggregation_profile = "aggregation.llm_profile" in merged_prompts

        print(f"\n[ScenarioConfig] aggregation.system exists: {has_aggregation_system}")
        print(
            f"[ScenarioConfig] aggregation.llm_profile exists: {has_aggregation_profile}"
        )

        assert has_aggregation_system, "应包含 aggregation.system 配置"
        assert has_aggregation_profile, "应包含 aggregation.llm_profile 配置"

        if has_aggregation_system:
            system_prompt = merged_prompts["aggregation.system"]
            print(
                f"[ScenarioConfig] aggregation.system length: {len(system_prompt)} chars"
            )
            assert len(system_prompt) > 50, "system prompt 不应为空"

    @pytest.mark.asyncio
    async def test_default_prompt_fallback(self, app: FastAPI):
        """测试：无场景配置时使用默认 prompt。"""
        from assistant.llm.client import get_llm_client

        llm_client = get_llm_client()
        mock_loader = MagicMock()
        mock_loader.get_merged_prompts.side_effect = Exception("场景不存在")

        aggregator = Aggregator(
            llm_client=llm_client,
            scenario_loader=mock_loader,
        )

        # 应该使用默认配置而不是崩溃
        config = aggregator._load_prompt_config("non_existent_scenario")

        assert "system" in config
        assert "llm_profile" in config
        # 默认值为 llm.pro（紧急兜底配置）
        assert config["llm_profile"] == "llm.pro"

        print(f"\n[DefaultFallback] llm_profile={config['llm_profile']}")
        print(f"[DefaultFallback] system prompt length={len(config['system'])} chars")


class TestAggregatorFallback:
    """Aggregator 降级逻辑测试。"""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, app: FastAPI):
        """测试：LLM 调用失败时的降级整合。"""
        # 创建一个会失败的 mock LLM client
        mock_llm = MagicMock()
        mock_llm.call.side_effect = Exception("LLM 服务不可用")

        mock_loader = MagicMock()
        mock_loader.get_merged_prompts.return_value = {
            "aggregation.system": "test system",
            "aggregation.llm_profile": "llm.default",
        }

        aggregator = Aggregator(
            llm_client=mock_llm,
            scenario_loader=mock_loader,
        )

        partner_outputs = [
            PartnerOutput(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                state=TaskState.Completed.value,
                data_items=[{"text": "酒店推荐内容"}],
            ),
        ]

        result = await aggregator.aggregate(
            partner_outputs=partner_outputs,
            user_query="查询酒店",
        )

        # === 验证降级结果 ===
        assert result.type == "final"
        assert result.text is not None
        assert result.structured is not None
        assert result.structured.get("fallback") is True

        print(f"\n[Fallback] type={result.type}")
        print(f"[Fallback] fallback={result.structured.get('fallback')}")
        print(f"[Fallback] text contains content: {'酒店推荐内容' in result.text}")

    @pytest.mark.asyncio
    async def test_empty_inputs_handling(self, app: FastAPI):
        """测试：空输入处理。"""
        aggregator = get_aggregator()

        result = await aggregator.aggregate(
            partner_outputs=[],
            degradations=[],
            user_query="",
        )

        # === 验证空结果处理 ===
        assert result.type == "final"
        assert result.text is not None
        assert "暂无" in result.text

        print(f"\n[EmptyInputs] text={result.text}")


class TestAggregatorDataModels:
    """Aggregator 数据模型测试。"""

    def test_partner_output_model(self):
        """测试：PartnerOutput 数据模型。"""
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

    def test_degradation_info_model(self):
        """测试：DegradationInfo 数据模型。"""
        deg = DegradationInfo(
            dimension_id="transport",
            reason="服务不可用",
            suggestion="请稍后重试",
        )

        assert deg.dimension_id == "transport"
        assert deg.reason == "服务不可用"
        assert deg.suggestion == "请稍后重试"

    def test_aggregation_result_model(self):
        """测试：AggregationResult 数据模型。"""
        from datetime import datetime, timezone

        result = AggregationResult(
            type="final",
            text="整合结果",
            structured={"key": "value"},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        assert result.type == "final"
        assert result.text == "整合结果"
        assert result.structured == {"key": "value"}
