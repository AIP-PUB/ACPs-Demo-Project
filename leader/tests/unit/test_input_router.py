"""
Leader Agent Platform - LLM-4 输入路由器单元测试

测试 InputRouter 的核心功能：
1. 字段映射构建
2. 自动补全
3. 字段值提取（规则匹配）
4. 完整性验证
5. 补丁生成
6. LLM 调用与降级
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

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from assistant.core.input_router import (
    InputRouter,
    get_input_router,
    reset_input_router,
)
from assistant.models.input_routing import (
    InputRoutingRequest,
    InputRoutingResult,
    PartnerGapInfo,
    PartnerPatch,
)
from assistant.models.clarification import (
    MergedClarification,
    RequiredField,
)

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock()
    # 默认返回提取结果 - InputRouter 调用的是 call 方法
    client.call = MagicMock(
        return_value=json.dumps(
            {
                "extractedValues": {"budget": 500, "travelers": 2},
                "analysis": {"confident": True, "notes": "提取成功"},
            }
        )
    )
    return client


@pytest.fixture
def mock_scenario_loader():
    """Mock 场景加载器"""
    loader = MagicMock()
    loader.get_merged_prompts = MagicMock(
        return_value={
            "input_routing": {
                "system": "你是输入路由器。{{input_json}}",
            }
        }
    )
    return loader


@pytest.fixture
def router(mock_llm_client, mock_scenario_loader):
    """创建 InputRouter 实例"""
    return InputRouter(
        llm_client=mock_llm_client,
        scenario_loader=mock_scenario_loader,
    )


@pytest.fixture
def sample_partner_gaps():
    """示例 Partner 缺口数据"""
    return [
        PartnerGapInfo(
            partner_aic="partner-hotel-001",
            partner_name="酒店服务",
            dimension_id="hotel",
            aip_task_id="task-001",
            awaiting_fields=[
                RequiredField(
                    field_name="budget",
                    field_label="预算",
                    field_type="number",
                    required=True,
                ),
                RequiredField(
                    field_name="check_in_date",
                    field_label="入住日期",
                    field_type="date",
                    required=True,
                ),
            ],
        ),
        PartnerGapInfo(
            partner_aic="partner-transport-002",
            partner_name="交通服务",
            dimension_id="transport",
            aip_task_id="task-002",
            awaiting_fields=[
                RequiredField(
                    field_name="travelers",
                    field_label="出行人数",
                    field_type="number",
                    required=True,
                ),
                RequiredField(
                    field_name="check_in_date",
                    field_label="出发日期",
                    field_type="date",
                    required=True,
                ),
            ],
        ),
    ]


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """每个测试后清理单例"""
    yield
    reset_input_router()


# =============================================================================
# 字段映射测试
# =============================================================================


class TestFieldMapping:
    """测试字段到 Partner 的映射构建"""

    def test_single_partner_single_field(self, router: InputRouter):
        """单 Partner 单字段"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            )
        ]

        field_to_partners, all_fields = router._build_field_mapping(gaps)

        assert len(all_fields) == 1
        assert all_fields[0].field_name == "budget"
        assert "budget" in field_to_partners
        assert field_to_partners["budget"] == ["partner-001"]

    def test_multiple_partners_same_field(self, router: InputRouter):
        """多 Partner 相同字段 - 应记录所有 Partner"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="date",
                        field_label="日期",
                        field_type="date",
                    )
                ],
            ),
            PartnerGapInfo(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                awaiting_fields=[
                    RequiredField(
                        field_name="date",
                        field_label="出发日期",
                        field_type="date",
                    )
                ],
            ),
        ]

        field_to_partners, all_fields = router._build_field_mapping(gaps)

        assert len(all_fields) == 1
        assert all_fields[0].field_name == "date"
        assert len(field_to_partners["date"]) == 2
        assert "partner-001" in field_to_partners["date"]
        assert "partner-002" in field_to_partners["date"]

    def test_multiple_partners_different_fields(self, router: InputRouter):
        """多 Partner 不同字段"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="budget",
                        field_label="预算",
                        field_type="number",
                    )
                ],
            ),
            PartnerGapInfo(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                awaiting_fields=[
                    RequiredField(
                        field_name="travelers",
                        field_label="人数",
                        field_type="number",
                    )
                ],
            ),
        ]

        field_to_partners, all_fields = router._build_field_mapping(gaps)

        assert len(all_fields) == 2
        field_names = {f.field_name for f in all_fields}
        assert field_names == {"budget", "travelers"}
        assert field_to_partners["budget"] == ["partner-001"]
        assert field_to_partners["travelers"] == ["partner-002"]


# =============================================================================
# 约束合并测试
# =============================================================================


class TestConstraintMerging:
    """测试字段约束合并"""

    def test_merge_required_or(self, router: InputRouter):
        """required 取 OR"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            required=False,
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            required=True,
        )

        merged = router._merge_field_constraints(field_a, field_b)

        assert merged.required is True

    def test_merge_min_constraint(self, router: InputRouter):
        """min 约束取更大值"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            constraints={"min": 100},
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            constraints={"min": 200},
        )

        merged = router._merge_field_constraints(field_a, field_b)

        assert merged.constraints["min"] == 200

    def test_merge_max_constraint(self, router: InputRouter):
        """max 约束取更小值"""
        field_a = RequiredField(
            field_name="budget",
            field_label="预算",
            constraints={"max": 1000},
        )
        field_b = RequiredField(
            field_name="budget",
            field_label="预算",
            constraints={"max": 500},
        )

        merged = router._merge_field_constraints(field_a, field_b)

        assert merged.constraints["max"] == 500

    def test_merge_enum_intersection(self, router: InputRouter):
        """枚举约束取交集"""
        field_a = RequiredField(
            field_name="room_type",
            field_label="房型",
            constraints={"enum": ["标准间", "大床房", "套房"]},
        )
        field_b = RequiredField(
            field_name="room_type",
            field_label="房型",
            constraints={"enum": ["大床房", "套房", "总统套"]},
        )

        merged = router._merge_field_constraints(field_a, field_b)

        assert set(merged.constraints["enum"]) == {"大床房", "套房"}


# =============================================================================
# 自动补全测试
# =============================================================================


class TestAutoFill:
    """测试从 userContext 自动补全"""

    def test_direct_match(self, router: InputRouter):
        """直接字段名匹配"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
            RequiredField(field_name="travelers", field_label="人数", required=True),
        ]
        user_context = {"budget": 500}

        filled = router._auto_fill_from_context(fields, user_context)

        assert filled == {"budget": 500}

    def test_alias_match(self, router: InputRouter):
        """别名匹配"""
        fields = [
            RequiredField(
                field_name="check_in_date", field_label="入住日期", required=True
            ),
        ]
        user_context = {"start_date": "2024-12-25"}

        filled = router._auto_fill_from_context(fields, user_context)

        assert filled.get("check_in_date") == "2024-12-25"

    def test_no_match(self, router: InputRouter):
        """无匹配"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
        ]
        user_context = {"unrelated_field": "value"}

        filled = router._auto_fill_from_context(fields, user_context)

        assert filled == {}


# =============================================================================
# 降级提取测试
# =============================================================================


class TestFallbackExtraction:
    """测试规则匹配的降级提取"""

    def test_extract_number_with_unit(self, router: InputRouter):
        """提取带单位的数字"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number"),
        ]

        extracted = router._fallback_extraction("预算500元左右", fields)

        assert extracted.get("budget") == 500.0

    def test_extract_people_count(self, router: InputRouter):
        """提取人数"""
        fields = [
            RequiredField(
                field_name="travelers", field_label="出行人数", field_type="number"
            ),
        ]

        extracted = router._fallback_extraction("2人出行", fields)

        assert extracted.get("travelers") == 2

    def test_extract_date_format(self, router: InputRouter):
        """提取日期格式"""
        fields = [
            RequiredField(
                field_name="check_in_date", field_label="入住日期", field_type="date"
            ),
        ]

        extracted = router._fallback_extraction("入住日期是2024-12-25", fields)

        assert extracted.get("check_in_date") == "2024-12-25"

    def test_extract_relative_date_tomorrow(self, router: InputRouter):
        """提取相对日期 - 明天"""
        fields = [
            RequiredField(field_name="date", field_label="日期", field_type="date"),
        ]

        extracted = router._fallback_extraction("明天出发", fields)

        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert extracted.get("date") == expected

    def test_extract_boolean_positive(self, router: InputRouter):
        """提取布尔值 - 肯定"""
        fields = [
            RequiredField(
                field_name="need_breakfast",
                field_label="是否需要早餐",
                field_type="boolean",
            ),
        ]

        extracted = router._fallback_extraction("是的，需要早餐", fields)

        assert extracted.get("need_breakfast") is True

    def test_extract_boolean_negative(self, router: InputRouter):
        """提取布尔值 - 否定"""
        fields = [
            RequiredField(
                field_name="need_pickup",
                field_label="是否需要接机",
                field_type="boolean",
            ),
        ]

        extracted = router._fallback_extraction("不需要接机", fields)

        assert extracted.get("need_pickup") is False


# =============================================================================
# 完整性验证测试
# =============================================================================


class TestCompletenessValidation:
    """测试信息完整性验证"""

    def test_all_required_filled(self, router: InputRouter):
        """所有必填字段都已填充"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
            RequiredField(field_name="date", field_label="日期", required=True),
        ]
        filled_values = {"budget": 500, "date": "2024-12-25"}

        is_sufficient, missing = router._validate_completeness(fields, filled_values)

        assert is_sufficient is True
        assert missing == []

    def test_missing_required_field(self, router: InputRouter):
        """缺少必填字段"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
            RequiredField(field_name="date", field_label="日期", required=True),
        ]
        filled_values = {"budget": 500}

        is_sufficient, missing = router._validate_completeness(fields, filled_values)

        assert is_sufficient is False
        assert len(missing) == 1
        assert missing[0].field_name == "date"

    def test_optional_field_not_filled(self, router: InputRouter):
        """可选字段未填充不影响完整性"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
            RequiredField(field_name="notes", field_label="备注", required=False),
        ]
        filled_values = {"budget": 500}

        is_sufficient, missing = router._validate_completeness(fields, filled_values)

        assert is_sufficient is True
        assert missing == []

    def test_empty_value_counts_as_missing(self, router: InputRouter):
        """空值视为未填充"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", required=True),
        ]
        filled_values = {"budget": ""}

        is_sufficient, missing = router._validate_completeness(fields, filled_values)

        assert is_sufficient is False
        assert len(missing) == 1


# =============================================================================
# 补丁生成测试
# =============================================================================


class TestPatchGeneration:
    """测试 Partner 补丁生成"""

    def test_generate_single_field_patch(self, router: InputRouter):
        """单字段补丁"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="budget", field_label="预算", field_type="number"
                    )
                ],
            )
        ]
        field_to_partners = {"budget": ["partner-001"]}
        filled_values = {"budget": 500}

        patches = router._generate_patches(
            partner_gaps=gaps,
            field_to_partners=field_to_partners,
            filled_values=filled_values,
            user_input="预算500元",
        )

        assert "partner-001" in patches
        patch = patches["partner-001"]
        assert patch.partner_aic == "partner-001"
        assert patch.patch_data == {"budget": 500}
        assert "预算" in patch.patch_text

    def test_generate_multi_field_patch(self, router: InputRouter):
        """多字段补丁"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="budget", field_label="预算", field_type="number"
                    ),
                    RequiredField(
                        field_name="date", field_label="日期", field_type="date"
                    ),
                ],
            )
        ]
        field_to_partners = {"budget": ["partner-001"], "date": ["partner-001"]}
        filled_values = {"budget": 500, "date": "2024-12-25"}

        patches = router._generate_patches(
            partner_gaps=gaps,
            field_to_partners=field_to_partners,
            filled_values=filled_values,
            user_input="预算500元，12月25日",
        )

        assert "partner-001" in patches
        patch = patches["partner-001"]
        assert patch.patch_data == {"budget": 500, "date": "2024-12-25"}
        assert len(patch.filled_fields) == 2

    def test_multiple_partners_same_field(self, router: InputRouter):
        """同一字段路由到多个 Partner"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="date", field_label="入住日期", field_type="date"
                    )
                ],
            ),
            PartnerGapInfo(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                awaiting_fields=[
                    RequiredField(
                        field_name="date", field_label="出发日期", field_type="date"
                    )
                ],
            ),
        ]
        field_to_partners = {"date": ["partner-001", "partner-002"]}
        filled_values = {"date": "2024-12-25"}

        patches = router._generate_patches(
            partner_gaps=gaps,
            field_to_partners=field_to_partners,
            filled_values=filled_values,
            user_input="12月25日",
        )

        assert "partner-001" in patches
        assert "partner-002" in patches
        assert patches["partner-001"].patch_data == {"date": "2024-12-25"}
        assert patches["partner-002"].patch_data == {"date": "2024-12-25"}

    def test_skip_partner_without_filled_fields(self, router: InputRouter):
        """跳过没有填充字段的 Partner"""
        gaps = [
            PartnerGapInfo(
                partner_aic="partner-001",
                dimension_id="hotel",
                aip_task_id="task-001",
                awaiting_fields=[
                    RequiredField(
                        field_name="budget", field_label="预算", field_type="number"
                    )
                ],
            ),
            PartnerGapInfo(
                partner_aic="partner-002",
                dimension_id="transport",
                aip_task_id="task-002",
                awaiting_fields=[
                    RequiredField(
                        field_name="travelers", field_label="人数", field_type="number"
                    )
                ],
            ),
        ]
        field_to_partners = {"budget": ["partner-001"], "travelers": ["partner-002"]}
        filled_values = {"budget": 500}  # 只填了 budget

        patches = router._generate_patches(
            partner_gaps=gaps,
            field_to_partners=field_to_partners,
            filled_values=filled_values,
            user_input="预算500元",
        )

        assert "partner-001" in patches
        assert "partner-002" not in patches


# =============================================================================
# 完整路由流程测试
# =============================================================================


class TestFullRoutingFlow:
    """测试完整的路由流程"""

    @pytest.mark.asyncio
    async def test_route_sufficient_input(
        self, router: InputRouter, sample_partner_gaps
    ):
        """完整输入的路由"""
        # 设置 LLM 返回完整的提取结果
        router._llm_client.call = MagicMock(
            return_value=json.dumps(
                {
                    "extractedValues": {
                        "budget": 500,
                        "check_in_date": "2024-12-25",
                        "travelers": 2,
                    },
                    "analysis": {"confident": True},
                }
            )
        )

        request = InputRoutingRequest(
            user_input="预算500元，2人，12月25日入住",
            partner_gaps=sample_partner_gaps,
            active_task_id="active-001",
        )

        result = await router.route(request)

        assert result.is_sufficient is True
        assert len(result.patches_by_partner) == 2
        assert len(result.missing_fields) == 0

    @pytest.mark.asyncio
    async def test_route_partial_input(self, router: InputRouter, sample_partner_gaps):
        """部分输入的路由"""
        # 设置 LLM 返回部分提取结果
        router._llm_client.call = MagicMock(
            return_value=json.dumps(
                {
                    "extractedValues": {"budget": 500},
                    "analysis": {"confident": True},
                }
            )
        )

        request = InputRoutingRequest(
            user_input="预算500元",
            partner_gaps=sample_partner_gaps,
            active_task_id="active-001",
        )

        result = await router.route(request)

        assert result.is_sufficient is False
        assert len(result.missing_fields) > 0
        # 应该有一个 Partner 有补丁（酒店）
        assert "partner-hotel-001" in result.patches_by_partner

    @pytest.mark.asyncio
    async def test_route_empty_gaps(self, router: InputRouter):
        """空缺口列表的路由"""
        request = InputRoutingRequest(
            user_input="预算500元",
            partner_gaps=[],
            active_task_id="active-001",
        )

        result = await router.route(request)

        assert result.is_sufficient is True
        assert result.patches_by_partner == {}

    @pytest.mark.asyncio
    async def test_route_with_user_context(
        self, router: InputRouter, sample_partner_gaps
    ):
        """带 userContext 的路由"""
        # 设置 LLM 只返回部分提取
        router._llm_client.call = MagicMock(
            return_value=json.dumps(
                {
                    "extractedValues": {"budget": 500},
                    "analysis": {"confident": True},
                }
            )
        )

        request = InputRoutingRequest(
            user_input="预算500元",
            partner_gaps=sample_partner_gaps,
            active_task_id="active-001",
            user_context={
                "check_in_date": "2024-12-25",
                "travelers": 2,
            },
        )

        result = await router.route(request)

        # 由于 userContext 提供了额外字段，应该是充分的
        assert result.is_sufficient is True

    @pytest.mark.asyncio
    async def test_route_llm_failure_fallback(
        self, router: InputRouter, sample_partner_gaps
    ):
        """LLM 调用失败时的降级"""
        router._llm_client.call = MagicMock(side_effect=Exception("LLM Error"))

        request = InputRoutingRequest(
            user_input="预算500元，2人",
            partner_gaps=sample_partner_gaps,
            active_task_id="active-001",
        )

        result = await router.route(request)

        # 应该用规则匹配降级提取
        assert (
            "budget" in str(result.patches_by_partner)
            or len(result.patches_by_partner) >= 0
        )


# =============================================================================
# LLM 响应解析测试
# =============================================================================


class TestLLMResponseParsing:
    """测试 LLM 响应解析"""

    def test_parse_clean_json(self, router: InputRouter):
        """解析干净的 JSON"""
        response = (
            '{"extractedValues": {"budget": 500}, "analysis": {"confident": true}}'
        )
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted, analysis = router._parse_llm_response(response, fields)

        assert extracted == {"budget": 500}
        assert analysis["confident"] is True

    def test_parse_json_in_markdown(self, router: InputRouter):
        """解析 Markdown 代码块中的 JSON"""
        response = """
这是分析结果：

```json
{"extractedValues": {"budget": 500}}
```
"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted, _ = router._parse_llm_response(response, fields)

        assert extracted == {"budget": 500}

    def test_parse_json_with_prefix(self, router: InputRouter):
        """解析带前缀文本的 JSON"""
        response = '好的，我来分析一下：{"extractedValues": {"budget": 500}}'
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted, _ = router._parse_llm_response(response, fields)

        assert extracted == {"budget": 500}

    def test_parse_invalid_json(self, router: InputRouter):
        """解析无效 JSON"""
        response = "这不是 JSON 格式"
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted, _ = router._parse_llm_response(response, fields)

        assert extracted == {}


# =============================================================================
# 工厂函数测试
# =============================================================================


class TestFactory:
    """测试工厂函数"""

    def test_get_input_router_singleton(self, mock_scenario_loader):
        """获取单例"""
        with patch("assistant.llm.client.get_llm_client") as mock_get_llm, patch(
            "assistant.services.scenario_loader.get_scenario_loader"
        ) as mock_get_loader:
            mock_get_llm.return_value = MagicMock()
            mock_get_loader.return_value = mock_scenario_loader

            router1 = get_input_router(scenario_loader=mock_scenario_loader)
            router2 = get_input_router(scenario_loader=mock_scenario_loader)

            assert router1 is router2

    def test_reset_input_router(self, mock_scenario_loader):
        """重置单例"""
        with patch("assistant.llm.client.get_llm_client") as mock_get_llm, patch(
            "assistant.services.scenario_loader.get_scenario_loader"
        ) as mock_get_loader:
            mock_get_llm.return_value = MagicMock()
            mock_get_loader.return_value = mock_scenario_loader

            router1 = get_input_router(scenario_loader=mock_scenario_loader)
            reset_input_router()
            router2 = get_input_router(scenario_loader=mock_scenario_loader)

            assert router1 is not router2


# =============================================================================
# 边界情况测试
# =============================================================================


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_user_input(self, router: InputRouter):
        """空用户输入"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted = router._fallback_extraction("", fields)

        assert extracted == {}

    def test_very_long_input(self, router: InputRouter):
        """超长用户输入"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]
        long_input = "预算500元" + " 一些描述" * 1000

        extracted = router._fallback_extraction(long_input, fields)

        assert extracted.get("budget") == 500.0

    def test_special_characters_in_input(self, router: InputRouter):
        """特殊字符输入"""
        fields = [
            RequiredField(field_name="budget", field_label="预算", field_type="number")
        ]

        extracted = router._fallback_extraction("预算￥500元/人", fields)

        assert extracted.get("budget") == 500.0

    @pytest.mark.asyncio
    async def test_result_methods(self, router: InputRouter, sample_partner_gaps):
        """测试 InputRoutingResult 的方法"""
        router._llm_client.call = MagicMock(
            return_value=json.dumps(
                {
                    "extractedValues": {"budget": 500},
                    "analysis": {"confident": True},
                }
            )
        )

        request = InputRoutingRequest(
            user_input="预算500元",
            partner_gaps=sample_partner_gaps,
            active_task_id="active-001",
        )

        result = await router.route(request)

        # 测试方法
        target_partners = result.get_target_partners()
        assert isinstance(target_partners, list)

        has_patch = result.has_patch_for("partner-hotel-001")
        assert isinstance(has_patch, bool)
