"""
Leader Agent Platform - 集成测试：LLM-3 反问合并流程

测试从 Partner AwaitingInput 状态到用户反问的完整流程，验证：
1. ClarificationMerger 正确调用 LLM-3
2. 多 Partner 反问需求正确合并
3. 字段去重和约束合并
4. 自动填充功能
5. 降级场景的处理

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

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from assistant.core.clarification_merger import (
    ClarificationMerger,
    get_clarification_merger,
)
from assistant.core.orchestrator import Orchestrator
from assistant.core.executor import (
    ExecutionResult,
    ExecutionPhase,
    PartnerExecutionResult,
)
from assistant.models.clarification import (
    ClarificationMergeInput,
    MergedClarification,
    PartnerClarificationItem,
    RequiredField,
    extract_clarification_from_task_status,
)
from acps_sdk.aip.aip_base_model import TaskState, TaskResult, TaskStatus

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# 测试辅助函数
# =============================================================================


def create_mock_task(
    task_id: str,
    state: TaskState,
    data_items: List[Dict[str, Any]],
) -> TaskResult:
    """创建模拟的 AIP TaskResult 对象。"""
    return TaskResult(
        id=f"result-{task_id}",
        sentAt="2024-01-01T00:00:00Z",
        senderRole="partner",
        senderId="test-partner",
        taskId=task_id,
        sessionId="test-session",
        status=TaskStatus(
            state=state,
            stateChangedAt="2024-01-01T00:00:00Z",
            dataItems=data_items,
        ),
    )


def create_mock_partner_result(
    partner_aic: str,
    dimension_id: str,
    state: TaskState,
    data_items: List[Dict[str, Any]] = None,
) -> PartnerExecutionResult:
    """创建模拟的 Partner 执行结果。"""
    task = create_mock_task(
        task_id=f"task-{partner_aic[:8]}",
        state=state,
        data_items=data_items or [],
    )
    return PartnerExecutionResult(
        partner_aic=partner_aic,
        dimension_id=dimension_id,
        state=state,
        task=task,
        data_items=data_items or [],
    )


# =============================================================================
# LLM-3 集成测试
# =============================================================================


class TestClarificationMergerLLMIntegration:
    """LLM-3 反问合并完整流程测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_basic_merge_with_llm(self):
        """测试：基本 LLM 合并调用。"""
        merger = get_clarification_merger()

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                partner_name="酒店服务",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                question_text=None,
                required_fields=[
                    RequiredField(
                        field_name="check_in_date",
                        field_label="入住日期",
                        field_type="date",
                        description="请提供您的入住日期",
                    ),
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                        constraints={"min": 200, "max": 2000},
                    ),
                ],
            )
        ]

        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我预订北京的酒店",
        )

        result = await merger.merge(merge_input)

        # === 验证结果结构 ===
        assert result.question_text is not None
        assert len(result.question_text) > 10
        assert len(result.merged_fields) == 2
        assert "partner-hotel-001" in result.source_partners

        print(f"\n[BasicMerge] question_text: {result.question_text}")
        print(
            f"[BasicMerge] merged_fields: {[f.field_name for f in result.merged_fields]}"
        )

    @pytest.mark.asyncio
    async def test_multi_partner_merge_with_llm(self):
        """测试：多 Partner 反问合并（相同字段去重）。"""
        merger = get_clarification_merger()

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                partner_name="酒店服务",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                        constraints={"min": 200},
                    ),
                ],
            ),
            PartnerClarificationItem(
                partner_aic="partner-food-001",
                partner_name="餐饮服务",
                dimension_id="food",
                aip_task_id="task-food-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="餐饮预算",
                        field_type="number",
                        constraints={"max": 500},
                    ),
                ],
            ),
        ]

        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我安排北京一日游",
        )

        result = await merger.merge(merge_input)

        # === 验证去重 ===
        assert result.question_text is not None
        # budget 字段应该被去重合并
        budget_fields = [f for f in result.merged_fields if f.field_name == "budget"]
        assert len(budget_fields) == 1

        # 约束应该被合并：min=200, max=500
        if budget_fields[0].constraints:
            constraints = budget_fields[0].constraints
            assert constraints.get("min") == 200
            assert constraints.get("max") == 500

        # 两个 Partner 都应该在 field_to_partners 中
        assert "budget" in result.field_to_partners
        assert "partner-hotel-001" in result.field_to_partners["budget"]
        assert "partner-food-001" in result.field_to_partners["budget"]

        print(f"\n[MultiPartner] question_text: {result.question_text}")
        print(f"[MultiPartner] field_to_partners: {result.field_to_partners}")

    @pytest.mark.asyncio
    async def test_auto_fill_from_context(self):
        """测试：从用户上下文自动填充字段。"""
        merger = get_clarification_merger()

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    ),
                    RequiredField(
                        field_name="city",
                        field_label="城市",
                        field_type="string",
                    ),
                ],
            ),
        ]

        # 用户上下文中已有 budget 信息
        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我预订酒店",
            user_context={
                "budget": 800,  # 已知预算
            },
        )

        result = await merger.merge(merge_input)

        # === 验证自动填充 ===
        # budget 应该被过滤掉（已从上下文填充）
        remaining_fields = [f.field_name for f in result.merged_fields]
        assert "budget" not in remaining_fields
        assert "city" in remaining_fields

        print(f"\n[AutoFill] remaining fields: {remaining_fields}")
        print(f"[AutoFill] question_text: {result.question_text}")

    @pytest.mark.asyncio
    async def test_text_only_question(self):
        """测试：仅文本问题（无结构化字段）。"""
        merger = get_clarification_merger()

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                partner_name="酒店服务",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                question_text="请问您更喜欢哪种风格的酒店？（商务/度假/民宿）",
                required_fields=[],
            ),
        ]

        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我预订北京的酒店",
        )

        result = await merger.merge(merge_input)

        # === 验证结果 ===
        assert result.question_text is not None
        # 应该生成一个泛化字段来收集答案
        assert len(result.merged_fields) >= 1

        print(f"\n[TextOnly] question_text: {result.question_text}")
        print(
            f"[TextOnly] merged_fields: {[f.field_name for f in result.merged_fields]}"
        )

    @pytest.mark.asyncio
    async def test_enum_constraint_intersection(self):
        """测试：枚举约束交集合并。"""
        merger = get_clarification_merger()

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                required_fields=[
                    RequiredField(
                        field_name="room_type",
                        field_label="房型",
                        field_type="enum",
                        constraints={
                            "enum_values": ["standard", "deluxe", "suite"],
                        },
                    ),
                ],
            ),
            PartnerClarificationItem(
                partner_aic="partner-hotel-002",
                dimension_id="hotel",
                aip_task_id="task-hotel-002",
                required_fields=[
                    RequiredField(
                        field_name="room_type",
                        field_label="房型选择",
                        field_type="enum",
                        constraints={
                            "enum_values": ["deluxe", "suite", "presidential"],
                        },
                    ),
                ],
            ),
        ]

        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我预订酒店",
        )

        result = await merger.merge(merge_input)

        # === 验证枚举交集 ===
        room_type_field = next(
            (f for f in result.merged_fields if f.field_name == "room_type"),
            None,
        )
        assert room_type_field is not None

        if room_type_field.constraints:
            enum_values = room_type_field.constraints.get("enum_values", [])
            # 交集应该是 ["deluxe", "suite"]
            assert set(enum_values) == {"deluxe", "suite"}

        print(f"\n[EnumIntersection] enum_values: {enum_values}")


# =============================================================================
# 从 AIP Task 提取反问测试
# =============================================================================


class TestExtractClarificationFromTaskStatus:
    """测试从 AIP Task 状态提取反问需求。"""

    def test_extract_text_question(self):
        """测试：提取文本类型问题。"""
        data_items = [
            {"type": "text", "text": "请问您的预算是多少？"},
        ]

        result = extract_clarification_from_task_status(
            partner_aic="partner-001",
            partner_name="测试服务",
            dimension_id="test",
            aip_task_id="task-001",
            data_items=data_items,
        )

        assert result.question_text == "请问您的预算是多少？"
        assert result.partner_aic == "partner-001"

    def test_extract_structured_fields(self):
        """测试：提取结构化字段需求。"""
        data_items = [
            {
                "type": "data",
                "data": {
                    "requiredFields": [
                        {
                            "name": "budget",
                            "label": "预算",
                            "type": "number",
                            "constraints": {"min": 100},
                        },
                        {
                            "name": "date",
                            "label": "日期",
                            "type": "date",
                        },
                    ],
                },
            },
        ]

        result = extract_clarification_from_task_status(
            partner_aic="partner-001",
            partner_name="测试服务",
            dimension_id="test",
            aip_task_id="task-001",
            data_items=data_items,
        )

        assert len(result.required_fields) == 2
        assert result.required_fields[0].field_name == "budget"
        assert result.required_fields[1].field_name == "date"

    def test_extract_single_field_format(self):
        """测试：提取单字段格式。"""
        data_items = [
            {
                "type": "data",
                "data": {
                    "fieldName": "budget",
                    "fieldLabel": "预算",
                    "fieldType": "number",
                },
            },
        ]

        result = extract_clarification_from_task_status(
            partner_aic="partner-001",
            partner_name="测试服务",
            dimension_id="test",
            aip_task_id="task-001",
            data_items=data_items,
        )

        assert len(result.required_fields) == 1
        assert result.required_fields[0].field_name == "budget"


# =============================================================================
# Orchestrator 集成测试
# =============================================================================


class TestOrchestratorClarificationIntegration:
    """测试 Orchestrator 与 ClarificationMerger 的集成。"""

    @pytest.mark.asyncio
    async def test_handle_awaiting_input_creates_clarification(self, app):
        """测试：Orchestrator 正确处理 AwaitingInput 状态。"""
        from assistant.core import SessionManager
        from assistant.services import ScenarioLoader

        # 创建模拟组件
        session_manager = SessionManager()
        scenario_loader = ScenarioLoader()
        clarification_merger = get_clarification_merger()

        # 创建 Orchestrator，直接注入 ClarificationMerger
        orchestrator = Orchestrator(
            session_manager=session_manager,
            scenario_loader=scenario_loader,
            clarification_merger=clarification_merger,
        )

        # 创建测试 Session - 使用正确的 API
        from assistant.models import ExecutionMode

        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=scenario_loader.base_scenario,
        )

        # 创建模拟的 ExecutionResult
        execution_result = ExecutionResult(
            phase=ExecutionPhase.AWAITING_INPUT,
            partner_results={
                "partner-hotel-001": create_mock_partner_result(
                    partner_aic="partner-hotel-001",
                    dimension_id="hotel",
                    state=TaskState.AwaitingInput,
                    data_items=[
                        {
                            "type": "data",
                            "data": {
                                "requiredFields": [
                                    {
                                        "name": "budget",
                                        "label": "预算",
                                        "type": "number",
                                    },
                                ],
                            },
                        },
                    ],
                ),
            },
            awaiting_input_partners=["partner-hotel-001"],
        )

        # 创建模拟的 planning_result
        @dataclass
        class MockPlanningResult:
            selected_partners: Dict[str, List[Any]] = field(default_factory=dict)

        planning_result = MockPlanningResult(
            selected_partners={"hotel": []},
        )

        # 调用 _handle_awaiting_input
        result = await orchestrator._handle_awaiting_input(
            session=session,
            execution_result=execution_result,
            planning_result=planning_result,
            user_query="帮我预订北京的酒店",
        )

        # === 验证结果 ===
        assert isinstance(result, MergedClarification)
        assert result.question_text is not None
        assert len(result.question_text) > 0
        assert "partner-hotel-001" in result.source_partners

        print(f"\n[OrchestratorIntegration] question_text: {result.question_text}")
        print(f"[OrchestratorIntegration] source_partners: {result.source_partners}")


# =============================================================================
# 降级场景测试
# =============================================================================


class TestClarificationFallback:
    """测试反问合并的降级场景。"""

    @pytest.mark.asyncio
    async def test_fallback_on_empty_items(self):
        """测试：空输入时的降级处理。"""
        merger = get_clarification_merger()

        merge_input = ClarificationMergeInput(
            partner_items=[],
            user_query="帮我预订酒店",
        )

        result = await merger.merge(merge_input)

        # 空输入应该返回友好的默认提示
        assert result.question_text == "请问有什么可以帮助您的？"
        assert len(result.merged_fields) == 0

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """测试：LLM 调用失败时的降级处理。"""
        from assistant.llm.client import LLMClient
        from assistant.services import ScenarioLoader

        # 创建一个会失败的 mock LLM client
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke = MagicMock(side_effect=Exception("LLM service unavailable"))

        # 使用真实的 ScenarioLoader
        scenario_loader = ScenarioLoader()

        merger = ClarificationMerger(
            llm_client=mock_llm,
            scenario_loader=scenario_loader,
        )

        partner_items = [
            PartnerClarificationItem(
                partner_aic="partner-hotel-001",
                dimension_id="hotel",
                aip_task_id="task-hotel-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    ),
                ],
            ),
        ]

        merge_input = ClarificationMergeInput(
            partner_items=partner_items,
            user_query="帮我预订酒店",
        )

        result = await merger.merge(merge_input)

        # 应该使用降级方案生成问题
        assert result.question_text is not None
        assert len(result.question_text) > 0
        assert "预算" in result.question_text or "budget" in result.question_text

        print(f"\n[Fallback] question_text: {result.question_text}")


# =============================================================================
# 端到端流程测试
# =============================================================================


class TestClarificationE2EFlow:
    """端到端反问流程测试。"""

    @pytest.mark.asyncio
    async def test_full_clarification_flow(self):
        """测试：完整的反问流程。"""
        # 1. 从 Task 状态提取反问需求
        data_items = [
            {"type": "text", "text": "请问您的入住日期是什么时候？"},
            {
                "type": "data",
                "data": {
                    "requiredFields": [
                        {
                            "name": "check_in_date",
                            "label": "入住日期",
                            "type": "date",
                        },
                        {
                            "name": "nights",
                            "label": "入住天数",
                            "type": "number",
                            "constraints": {"min": 1, "max": 30},
                        },
                    ],
                },
            },
        ]

        clarification_item = extract_clarification_from_task_status(
            partner_aic="partner-hotel-001",
            partner_name="酒店服务",
            dimension_id="hotel",
            aip_task_id="task-hotel-001",
            data_items=data_items,
        )

        # 验证提取结果
        assert clarification_item.question_text == "请问您的入住日期是什么时候？"
        assert len(clarification_item.required_fields) == 2

        # 2. 调用 LLM-3 合并
        merger = get_clarification_merger()
        merge_input = ClarificationMergeInput(
            partner_items=[clarification_item],
            user_query="帮我预订北京的酒店",
        )

        result = await merger.merge(merge_input)

        # 验证合并结果
        assert result.question_text is not None
        assert len(result.merged_fields) == 2
        assert "partner-hotel-001" in result.source_partners

        print("\n=== E2E 反问流程测试 ===")
        print(f"[Extract] question_text: {clarification_item.question_text}")
        print(
            f"[Extract] fields: {[f.field_name for f in clarification_item.required_fields]}"
        )
        print(f"[Merge] final question: {result.question_text}")
        print(f"[Merge] merged_fields: {[f.field_name for f in result.merged_fields]}")
