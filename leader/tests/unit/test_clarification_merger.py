"""
Leader Agent Platform - LLM-3 反问合并器单元测试

测试 ClarificationMerger 的核心功能：
1. 字段合并与去重
2. 约束合并
3. 自动补全
4. LLM 调用与降级
"""

import sys
import os

# 确保路径正确
_current_dir = os.path.dirname(os.path.abspath(__file__))
_leader_dir = os.path.dirname(os.path.dirname(_current_dir))
_project_root = os.path.dirname(_leader_dir)
if _leader_dir not in sys.path:
    sys.path.insert(0, _leader_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from assistant.core.clarification_merger import ClarificationMerger
from assistant.models.clarification import (
    ClarificationMergeInput,
    PartnerClarificationItem,
    RequiredField,
    MergedClarification,
)

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock()
    client.call = MagicMock(
        return_value="为了更好地为您服务，请告诉我您的预算和出行日期。"
    )
    return client


@pytest.fixture
def mock_scenario_loader():
    """Mock 场景加载器"""
    loader = MagicMock()
    loader.load_prompts = MagicMock(
        return_value={
            "clarification": {
                "system_prompt": "你是澄清问题生成器。",
                "user_prompt": "{input_json}",
            }
        }
    )
    return loader


@pytest.fixture
def merger(mock_llm_client, mock_scenario_loader):
    """创建 ClarificationMerger 实例"""
    return ClarificationMerger(
        llm_client=mock_llm_client,
        scenario_loader=mock_scenario_loader,
    )


# =============================================================================
# 字段合并测试
# =============================================================================


class TestFieldMerging:
    """测试字段合并与去重"""

    def test_single_partner_single_field(self, merger: ClarificationMerger):
        """单 Partner 单字段"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            )
        ]

        merged_fields, field_to_partners = merger._merge_fields(items)

        assert len(merged_fields) == 1
        assert merged_fields[0].field_name == "budget"
        assert "budget" in field_to_partners
        assert field_to_partners["budget"] == ["partner-001"]

    def test_multiple_partners_same_field(self, merger: ClarificationMerger):
        """多 Partner 相同字段 - 应去重"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="check_in_date",
                        field_label="入住日期",
                        field_type="date",
                    )
                ],
            ),
            PartnerClarificationItem(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                required_fields=[
                    RequiredField(
                        field_name="check_in_date",
                        field_label="出发日期",
                        field_type="date",
                    )
                ],
            ),
        ]

        merged_fields, field_to_partners = merger._merge_fields(items)

        # 字段应该去重
        assert len(merged_fields) == 1
        assert merged_fields[0].field_name == "check_in_date"
        # 两个 Partner 都需要这个字段
        assert len(field_to_partners["check_in_date"]) == 2
        assert "partner-001" in field_to_partners["check_in_date"]
        assert "partner-002" in field_to_partners["check_in_date"]

    def test_multiple_partners_different_fields(self, merger: ClarificationMerger):
        """多 Partner 不同字段"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            ),
            PartnerClarificationItem(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                required_fields=[
                    RequiredField(
                        field_name="departure_city",
                        field_label="出发城市",
                        field_type="string",
                    )
                ],
            ),
        ]

        merged_fields, field_to_partners = merger._merge_fields(items)

        assert len(merged_fields) == 2
        field_names = {f.field_name for f in merged_fields}
        assert "budget" in field_names
        assert "departure_city" in field_names

    def test_text_only_question(self, merger: ClarificationMerger):
        """只有文本问题，没有结构化字段 - 应生成通用字段"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="food",
                aip_task_id="task-001",
                question_text="您喜欢什么口味的餐厅？",
                required_fields=[],
            )
        ]

        merged_fields, field_to_partners = merger._merge_fields(items)

        assert len(merged_fields) == 1
        assert merged_fields[0].field_name == "answer_food"
        assert "您喜欢什么口味的餐厅？" in merged_fields[0].description


# =============================================================================
# 约束合并测试
# =============================================================================


class TestConstraintMerging:
    """测试字段约束合并"""

    def test_merge_required_or(self, merger: ClarificationMerger):
        """required 取 OR：有一个 True 就是 True"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            required=False,
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            required=True,
        )

        merged = merger._merge_field_constraints(field_a, field_b)

        assert merged.required is True

    def test_merge_min_constraint(self, merger: ClarificationMerger):
        """数值约束 min 取更大值"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            constraints={"min": 100},
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            constraints={"min": 200},
        )

        merged = merger._merge_field_constraints(field_a, field_b)

        assert merged.constraints["min"] == 200

    def test_merge_max_constraint(self, merger: ClarificationMerger):
        """数值约束 max 取更小值"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            constraints={"max": 1000},
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            field_type="number",
            constraints={"max": 500},
        )

        merged = merger._merge_field_constraints(field_a, field_b)

        assert merged.constraints["max"] == 500

    def test_merge_enum_intersection(self, merger: ClarificationMerger):
        """枚举约束取交集"""
        field_a = RequiredField(
            field_name="room_type",
            field_label="房型",
            field_type="enum",
            constraints={"enum": ["标准间", "大床房", "套房"]},
        )
        field_b = RequiredField(
            field_name="room_type",
            field_label="房型",
            field_type="enum",
            constraints={"enum": ["大床房", "套房", "豪华套房"]},
        )

        merged = merger._merge_field_constraints(field_a, field_b)

        # 交集
        enum_values = set(merged.constraints["enum"])
        assert enum_values == {"大床房", "套房"}


# =============================================================================
# 自动补全测试
# =============================================================================


class TestAutoFill:
    """测试从 userContext 自动补全"""

    def test_direct_field_match(self, merger: ClarificationMerger):
        """直接字段名匹配"""
        fields = [
            RequiredField(
                field_name="budget",
                field_label="预算",
                field_type="number",
            ),
            RequiredField(
                field_name="travelers",
                field_label="人数",
                field_type="number",
            ),
        ]
        field_to_partners = {
            "budget": ["partner-001"],
            "travelers": ["partner-002"],
        }
        user_context = {"budget": 1000}

        remaining, updated_mapping = merger._auto_fill_from_context(
            fields, field_to_partners, user_context
        )

        # budget 被补全，只剩 travelers
        assert len(remaining) == 1
        assert remaining[0].field_name == "travelers"
        assert "budget" not in updated_mapping

    def test_alias_field_match(self, merger: ClarificationMerger):
        """别名字段匹配"""
        fields = [
            RequiredField(
                field_name="destination",
                field_label="目的地",
                field_type="string",
            ),
        ]
        field_to_partners = {"destination": ["partner-001"]}
        user_context = {"city": "北京"}  # city 是 destination 的别名

        remaining, updated_mapping = merger._auto_fill_from_context(
            fields, field_to_partners, user_context
        )

        # destination 通过别名 city 被补全
        assert len(remaining) == 0

    def test_all_fields_filled(self, merger: ClarificationMerger):
        """所有字段都被补全"""
        fields = [
            RequiredField(
                field_name="budget",
                field_label="预算",
                field_type="number",
            ),
        ]
        field_to_partners = {"budget": ["partner-001"]}
        user_context = {"budget": 500}

        remaining, updated_mapping = merger._auto_fill_from_context(
            fields, field_to_partners, user_context
        )

        assert len(remaining) == 0
        assert len(updated_mapping) == 0


# =============================================================================
# LLM 调用与降级测试
# =============================================================================


class TestLLMIntegration:
    """测试 LLM 调用与降级"""

    @pytest.mark.asyncio
    async def test_merge_calls_llm(self, merger: ClarificationMerger):
        """正常调用 LLM"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            )
        ]

        result = await merger.merge(
            ClarificationMergeInput(
                partner_items=items,
                user_query="帮我订酒店",
            )
        )

        assert result.question_text != ""
        assert len(result.source_partners) == 1
        assert "partner-001" in result.source_partners

    @pytest.mark.asyncio
    async def test_merge_empty_items(self, merger: ClarificationMerger):
        """空输入"""
        result = await merger.merge(ClarificationMergeInput(partner_items=[]))

        assert result.question_text == "请问有什么可以帮助您的？"
        assert len(result.merged_fields) == 0

    @pytest.mark.asyncio
    async def test_merge_with_auto_fill(self, merger: ClarificationMerger):
        """带自动补全的合并"""
        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            )
        ]

        result = await merger.merge(
            ClarificationMergeInput(
                partner_items=items,
                user_query="帮我订酒店",
                user_context={"budget": 1000},  # 已有 budget
            )
        )

        # 所有字段都被补全，question_text 为空
        assert result.question_text == ""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, mock_llm_client, mock_scenario_loader):
        """LLM 调用失败时降级"""
        mock_llm_client.call = MagicMock(side_effect=Exception("LLM error"))
        merger = ClarificationMerger(
            llm_client=mock_llm_client,
            scenario_loader=mock_scenario_loader,
        )

        items = [
            PartnerClarificationItem(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                required_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    ),
                    RequiredField(
                        field_name="date",
                        field_label="日期",
                        field_type="date",
                    ),
                ],
            )
        ]

        result = await merger.merge(ClarificationMergeInput(partner_items=items))

        # 应该使用降级逻辑生成问询
        assert "预算" in result.question_text or "日期" in result.question_text


# =============================================================================
# 响应解析测试
# =============================================================================


class TestResponseParsing:
    """测试 LLM 响应解析"""

    def test_parse_plain_text(self, merger: ClarificationMerger):
        """解析纯文本"""
        response = "请告诉我您的预算范围。"
        result = merger._parse_llm_response(response)
        assert result == "请告诉我您的预算范围。"

    def test_parse_json(self, merger: ClarificationMerger):
        """解析 JSON"""
        response = '{"question_text": "请告诉我您的预算范围。"}'
        result = merger._parse_llm_response(response)
        assert result == "请告诉我您的预算范围。"

    def test_parse_json_code_block(self, merger: ClarificationMerger):
        """解析 Markdown 代码块中的 JSON"""
        response = '```json\n{"question_text": "请告诉我您的预算范围。"}\n```'
        result = merger._parse_llm_response(response)
        assert result == "请告诉我您的预算范围。"

    def test_parse_json_text_field(self, merger: ClarificationMerger):
        """解析 JSON 中的 text 字段"""
        response = '{"text": "请告诉我您的预算范围。"}'
        result = merger._parse_llm_response(response)
        assert result == "请告诉我您的预算范围。"


# =============================================================================
# 降级问询生成测试
# =============================================================================


class TestFallbackQuestion:
    """测试降级问询生成"""

    def test_single_field(self, merger: ClarificationMerger):
        """单字段"""
        fields = [
            RequiredField(
                field_name="budget",
                field_label="预算",
                field_type="number",
            )
        ]

        result = merger._fallback_question_text(fields)

        assert "预算" in result

    def test_single_field_with_description(self, merger: ClarificationMerger):
        """单字段带描述"""
        fields = [
            RequiredField(
                field_name="budget",
                field_label="预算",
                field_type="number",
                description="请告诉我您每晚的预算范围（元）",
            )
        ]

        result = merger._fallback_question_text(fields)

        assert "每晚的预算范围" in result

    def test_multiple_fields(self, merger: ClarificationMerger):
        """多字段"""
        fields = [
            RequiredField(
                field_name="budget",
                field_label="预算",
                field_type="number",
            ),
            RequiredField(
                field_name="date",
                field_label="日期",
                field_type="date",
            ),
            RequiredField(
                field_name="travelers",
                field_label="人数",
                field_type="number",
            ),
        ]

        result = merger._fallback_question_text(fields)

        assert "预算" in result
        assert "日期" in result
        assert "人数" in result

    def test_empty_fields(self, merger: ClarificationMerger):
        """空字段列表"""
        result = merger._fallback_question_text([])
        assert "补充" in result or "帮助" in result
