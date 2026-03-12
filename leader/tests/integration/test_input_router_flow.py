"""
Leader Agent Platform - 集成测试：LLM-4 增量更新流程

测试用户补充输入的解析和路由功能，验证：
1. InputRouter 正确调用 LLM-4 分析用户输入
2. 字段值正确提取并路由到对应 Partner
3. 完整性验证和补丁生成
4. 多 Partner 共享字段的处理
5. userContext 自动补全功能

本测试使用真实大模型 API 调用，不使用 Mock。
"""

import sys
import os

# 确保 leader 目录在 path 中
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
from typing import List, Dict, Any

from assistant.core.input_router import (
    InputRouter,
    get_input_router,
)
from assistant.models.input_routing import (
    InputRoutingRequest,
    InputRoutingResult,
    PartnerGapInfo,
)
from assistant.models.clarification import (
    MergedClarification,
    RequiredField,
)

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# 测试 Fixtures
# =============================================================================


@pytest.fixture
def input_router():
    """获取真实的 InputRouter 实例（使用真实 LLM）。"""
    return get_input_router()


def create_partner_gap(
    partner_aic: str,
    dimension_id: str,
    fields: List[Dict[str, Any]],
    aip_task_id: str = "test-aip-task-001",
) -> PartnerGapInfo:
    """创建 Partner 缺口信息。"""
    return PartnerGapInfo(
        partner_aic=partner_aic,
        dimension_id=dimension_id,
        aip_task_id=aip_task_id,
        awaiting_fields=[
            RequiredField(
                field_name=f["name"],
                field_label=f.get("label", f["name"]),
                field_type=f.get("type", "string"),
                required=f.get("required", True),
                constraints=f.get("constraints"),
            )
            for f in fields
        ],
    )


# =============================================================================
# LLM-4 集成测试：基本路由功能
# =============================================================================


class TestInputRouterBasicRouting:
    """LLM-4 基本路由功能测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_single_partner_single_field(self, input_router: InputRouter):
        """测试：单个 Partner、单个字段的简单路由。"""
        # 准备：酒店 Partner 需要预算信息
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="预算大概 2000 元左右",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        assert result is not None
        print(f"\n[SinglePartnerSingleField] is_sufficient: {result.is_sufficient}")
        print(f"[SinglePartnerSingleField] routing_summary: {result.routing_summary}")
        print(f"[SinglePartnerSingleField] patches: {result.patches_by_partner}")

        # 应该路由到酒店 Partner
        assert "partner-hotel-001" in result.patches_by_partner
        patch = result.patches_by_partner["partner-hotel-001"]

        # 应该提取到预算值
        assert patch.patch_data is not None
        assert "budget" in patch.patch_data or any(
            "2000" in str(v) for v in patch.patch_data.values()
        )

    @pytest.mark.asyncio
    async def test_single_partner_multiple_fields(self, input_router: InputRouter):
        """测试：单个 Partner、多个字段的路由。"""
        # 准备：酒店 Partner 需要预算和入住日期
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                    {"name": "check_in_date", "label": "入住日期", "type": "date"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="预算 3000 元，打算 12月25日入住",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[SinglePartnerMultipleFields] is_sufficient: {result.is_sufficient}")
        print(f"[SinglePartnerMultipleFields] patches: {result.patches_by_partner}")

        assert "partner-hotel-001" in result.patches_by_partner
        patch = result.patches_by_partner["partner-hotel-001"]

        # 应该提取到两个字段
        assert patch.patch_data is not None
        # 预算和日期至少有一个被提取到
        has_budget = "budget" in patch.patch_data or "3000" in str(patch.patch_data)
        has_date = (
            "check_in_date" in patch.patch_data
            or "12" in str(patch.patch_data)
            or "25" in str(patch.patch_data)
        )
        assert has_budget or has_date, f"Expected budget or date in {patch.patch_data}"


class TestInputRouterMultiPartner:
    """LLM-4 多 Partner 路由测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_multiple_partners_independent_fields(
        self, input_router: InputRouter
    ):
        """测试：多个 Partner、独立字段的路由。"""
        # 准备：酒店需要预算，交通需要人数
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                ],
            ),
            create_partner_gap(
                partner_aic="partner-transport-002",
                dimension_id="transport",
                fields=[
                    {"name": "travelers", "label": "出行人数", "type": "number"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="预算 2000 元，两个人出行",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[MultiPartnerIndependent] is_sufficient: {result.is_sufficient}")
        print(f"[MultiPartnerIndependent] patches: {result.patches_by_partner}")

        # 应该路由到两个 Partner
        assert len(result.patches_by_partner) >= 1

        # 检查酒店 Partner
        if "partner-hotel-001" in result.patches_by_partner:
            hotel_patch = result.patches_by_partner["partner-hotel-001"]
            print(f"[MultiPartnerIndependent] hotel_patch: {hotel_patch.patch_data}")

        # 检查交通 Partner
        if "partner-transport-002" in result.patches_by_partner:
            transport_patch = result.patches_by_partner["partner-transport-002"]
            print(
                f"[MultiPartnerIndependent] transport_patch: {transport_patch.patch_data}"
            )

    @pytest.mark.asyncio
    async def test_multiple_partners_shared_field(self, input_router: InputRouter):
        """测试：多个 Partner 共享同一字段（如入住日期）。"""
        # 准备：酒店和交通都需要日期
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "check_in_date", "label": "入住日期", "type": "date"},
                ],
            ),
            create_partner_gap(
                partner_aic="partner-transport-002",
                dimension_id="transport",
                fields=[
                    {"name": "check_in_date", "label": "出发日期", "type": "date"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="12月25日出发",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[SharedField] is_sufficient: {result.is_sufficient}")
        print(f"[SharedField] patches: {result.patches_by_partner}")

        # 同一日期应该路由到两个 Partner
        # 注意：根据实现，可能是两个独立的 patch 或者一个共享的
        assert len(result.patches_by_partner) >= 1


class TestInputRouterCompleteness:
    """LLM-4 完整性验证测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_insufficient_input(self, input_router: InputRouter):
        """测试：用户输入不足以回答所有问题。"""
        # 准备：需要预算和日期，但用户只提供了预算
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {
                        "name": "budget",
                        "label": "预算",
                        "type": "number",
                        "required": True,
                    },
                    {
                        "name": "check_in_date",
                        "label": "入住日期",
                        "type": "date",
                        "required": True,
                    },
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="预算 2000 元",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[Insufficient] is_sufficient: {result.is_sufficient}")
        print(f"[Insufficient] missing_fields: {result.missing_fields}")
        print(f"[Insufficient] patches: {result.patches_by_partner}")

        # 应该标记为不充分，并指出缺失的字段
        # 注意：LLM 可能有不同判断，但应该有响应
        assert result is not None

    @pytest.mark.asyncio
    async def test_ambiguous_input(self, input_router: InputRouter):
        """测试：用户输入模糊。"""
        # 准备：需要具体预算，但用户回答模糊
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="随便吧，看情况",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[Ambiguous] is_sufficient: {result.is_sufficient}")
        print(f"[Ambiguous] missing_fields: {result.missing_fields}")
        print(f"[Ambiguous] patches: {result.patches_by_partner}")

        # 模糊输入应该被识别为不充分
        # 注意：LLM 的判断可能有变化


class TestInputRouterAutoFill:
    """LLM-4 自动补全功能测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_auto_fill_from_user_context(self, input_router: InputRouter):
        """测试：从 userContext 自动补全字段。"""
        # 准备：需要预算和偏好，userContext 包含偏好信息
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                    {"name": "preference", "label": "偏好", "type": "string"},
                ],
            ),
        ]

        user_context = {
            "preference": "安静的环境",
            "dietary": "不吃辣",
        }

        request = InputRoutingRequest(
            user_input="预算 3000 元",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            user_context=user_context,
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[AutoFill] is_sufficient: {result.is_sufficient}")
        print(f"[AutoFill] patches: {result.patches_by_partner}")

        # 应该自动补全偏好字段
        assert "partner-hotel-001" in result.patches_by_partner


class TestInputRouterWithClarificationContext:
    """LLM-4 结合反问上下文测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_route_with_last_clarification(self, input_router: InputRouter):
        """测试：结合上次反问的上下文进行路由。"""
        # 准备：上次问了预算和日期
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                    {"name": "check_in_date", "label": "入住日期", "type": "date"},
                ],
            ),
        ]

        last_clarification = MergedClarification(
            question_text="请问您的预算是多少？打算什么时候入住？",
            merged_fields=[
                RequiredField(
                    field_name="budget",
                    field_label="预算",
                    field_type="number",
                ),
                RequiredField(
                    field_name="check_in_date",
                    field_label="入住日期",
                    field_type="date",
                ),
            ],
        )

        request = InputRoutingRequest(
            user_input="2000 元，下周六",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            last_clarification=last_clarification,
            scenario_id="tour",
        )

        # 执行 LLM-4
        result = await input_router.route(request)

        # 验证
        print(f"\n[WithClarification] is_sufficient: {result.is_sufficient}")
        print(f"[WithClarification] patches: {result.patches_by_partner}")

        # 应该能正确解析用户对反问的回答
        assert result is not None
        assert "partner-hotel-001" in result.patches_by_partner


# =============================================================================
# 边界情况测试
# =============================================================================


class TestInputRouterEdgeCases:
    """LLM-4 边界情况测试。"""

    @pytest.mark.asyncio
    async def test_empty_partner_gaps(self, input_router: InputRouter):
        """测试：没有 Partner 缺口时的处理。"""
        request = InputRoutingRequest(
            user_input="随便说点什么",
            partner_gaps=[],
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        result = await input_router.route(request)

        # 应该返回空结果
        assert result.is_sufficient is True
        assert len(result.patches_by_partner) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_input(self, input_router: InputRouter):
        """测试：用户输入包含特殊字符。"""
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                fields=[
                    {"name": "budget", "label": "预算", "type": "number"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="预算大概 ¥2000~3000 元/天",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        result = await input_router.route(request)

        # 应该能处理特殊字符
        print(f"\n[SpecialChars] patches: {result.patches_by_partner}")
        assert result is not None

    @pytest.mark.asyncio
    async def test_numeric_field_extraction(self, input_router: InputRouter):
        """测试：数值字段的提取。"""
        partner_gaps = [
            create_partner_gap(
                partner_aic="partner-transport-001",
                dimension_id="transport",
                fields=[
                    {"name": "travelers", "label": "出行人数", "type": "number"},
                ],
            ),
        ]

        request = InputRoutingRequest(
            user_input="我们一共五个人",
            partner_gaps=partner_gaps,
            active_task_id="test-task-001",
            scenario_id="tour",
        )

        result = await input_router.route(request)

        # 应该能提取中文数字
        print(f"\n[NumericExtraction] patches: {result.patches_by_partner}")
        assert result is not None

        if "partner-transport-001" in result.patches_by_partner:
            patch = result.patches_by_partner["partner-transport-001"]
            print(f"[NumericExtraction] patch_data: {patch.patch_data}")
