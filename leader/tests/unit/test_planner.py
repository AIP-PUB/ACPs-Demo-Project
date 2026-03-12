"""
Leader Agent Platform - Planner 单元测试 (LLM-2)

测试内容：
1. 维度提取 (_extract_dimensions)
2. 维度Partner映射构建 (_build_dimension_partner_map)
3. ACS 加载与解析 (_load_partner_acs_by_filename, _parse_acs_to_candidate)
4. 输入上下文构建 (_build_input_context)
5. 结果验证与规范化 (_validate_and_normalize)
6. plan 方法的核心逻辑
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
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from tempfile import TemporaryDirectory

from assistant.core.planner import (
    Planner,
    DimensionDef,
    PartnerCandidate,
    PartnerSkillInfo,
)
from assistant.models import (
    IntentDecision,
    IntentType,
    TaskInstruction,
    Session,
    PlanningResult,
    PartnerSelection,
    DimensionNote,
    LLMPlanningOutput,
    ScenarioRuntime,
    ExecutionMode,
    UserResult,
    UserResultType,
)
from assistant.models.base import now_iso
from assistant.models.exceptions import LLMCallError


@pytest.fixture
def mock_llm_client():
    """创建 Mock LLM 客户端。"""
    client = MagicMock()
    client.call_structured = MagicMock(
        return_value=LLMPlanningOutput(
            selected_partners={
                "food": [
                    PartnerSelection(
                        partner_aic="partner-food",
                        skill_id="search_food",
                        reason="用户想找餐厅",
                        instruction_text="帮用户查找餐厅推荐",
                    )
                ],
                "transport": [],
            },
            dimension_notes={
                "transport": DimensionNote(status="inactive", reason="用户未提及交通"),
            },
        )
    )
    return client


@pytest.fixture
def mock_scenario_loader():
    """创建 Mock 场景加载器。"""
    loader = MagicMock()
    loader.scenario_root = Path("/fake/scenario")
    loader.get_prompt = MagicMock(return_value="规划助手 Prompt {{input_json}}")
    loader.get_llm_profile = MagicMock(return_value="gpt-4o")
    loader.get_expert_scenario = MagicMock(
        return_value=MagicMock(
            domain_meta={
                "meta": {
                    "id": "beijing_food",
                    "name": "北京美食",
                    "description": "北京美食推荐",
                },
                "dimensions": {
                    "food": {
                        "name": "美食",
                        "description": "餐厅推荐",
                        "required_fields": ["cuisine_type", "location"],
                        "optional_fields": ["budget"],
                    },
                    "transport": {
                        "name": "交通",
                        "description": "交通安排",
                        "required_fields": ["destination"],
                    },
                },
                "partners": {
                    "mapping": {
                        "food": "beijing_food.json",
                        "transport": "china_transport.json",
                    }
                },
            }
        )
    )
    return loader


@pytest.fixture
def planner(mock_llm_client, mock_scenario_loader):
    """创建 Planner 实例。"""
    return Planner(
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
        expert_scenario=ScenarioRuntime(
            id="beijing_food", kind="expert", version="1.0.0", loaded_at=now
        ),
        user_result=UserResult(
            type=UserResultType.PENDING,
            data_items=[],
            updated_at=now,
        ),
    )


@pytest.fixture
def intent_decision():
    """创建测试意图决策。"""
    return IntentDecision(
        intent_type=IntentType.TASK_NEW,
        target_scenario="beijing_food",
        task_instruction=TaskInstruction(text="我想找一家北京烤鸭店"),
    )


# =============================================================================
# 测试数据模型
# =============================================================================


class TestDimensionDef:
    """测试 DimensionDef 数据结构。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        dim = DimensionDef(
            id="food",
            name="美食",
            description="餐厅推荐",
        )
        assert dim.id == "food"
        assert dim.name == "美食"
        assert dim.fields == []

    def test_with_fields(self):
        """测试带字段创建。"""
        dim = DimensionDef(
            id="food",
            name="美食",
            description="餐厅推荐",
            fields=[{"name": "cuisine", "type": "string", "required": True}],
        )
        assert len(dim.fields) == 1
        assert dim.fields[0]["name"] == "cuisine"

    def test_to_dict(self):
        """测试转换为字典。"""
        dim = DimensionDef(
            id="food",
            name="美食",
            description="餐厅推荐",
        )
        result = dim.to_dict()
        assert result["id"] == "food"
        assert result["name"] == "美食"


class TestPartnerSkillInfo:
    """测试 PartnerSkillInfo 数据结构。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        skill = PartnerSkillInfo(
            id="search_food",
            name="搜索美食",
            description="搜索附近餐厅",
        )
        assert skill.id == "search_food"
        assert skill.tags == []

    def test_with_tags(self):
        """测试带标签创建。"""
        skill = PartnerSkillInfo(
            id="search_food",
            name="搜索美食",
            description="搜索餐厅",
            tags=["美食", "推荐"],
        )
        assert skill.tags == ["美食", "推荐"]

    def test_to_dict(self):
        """测试转换为字典。"""
        skill = PartnerSkillInfo(
            id="search_food",
            name="搜索美食",
            description="搜索餐厅",
            tags=["美食"],
        )
        result = skill.to_dict()
        assert result["tags"] == ["美食"]


class TestPartnerCandidate:
    """测试 PartnerCandidate 数据结构。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        candidate = PartnerCandidate(
            partner_aic="partner-001",
            partner_name="美食助手",
            description="提供美食推荐",
            source="static",
            skills=[],
        )
        assert candidate.partner_aic == "partner-001"
        assert candidate.source == "static"

    def test_to_dict(self):
        """测试转换为字典。"""
        skill = PartnerSkillInfo(id="s1", name="技能1", description="描述")
        candidate = PartnerCandidate(
            partner_aic="partner-001",
            partner_name="助手",
            description="描述",
            source="static",
            skills=[skill],
        )
        result = candidate.to_dict()
        assert result["partnerAic"] == "partner-001"
        assert len(result["skills"]) == 1


# =============================================================================
# 测试 _extract_dimensions
# =============================================================================


class TestExtractDimensions:
    """测试维度提取。"""

    def test_extract_basic(self, planner, mock_scenario_loader):
        """测试基本维度提取。"""
        domain_meta = mock_scenario_loader.get_expert_scenario().domain_meta

        result = planner._extract_dimensions(domain_meta)

        assert len(result) == 2
        dim_ids = {d.id for d in result}
        assert "food" in dim_ids
        assert "transport" in dim_ids

    def test_extract_with_fields(self, planner, mock_scenario_loader):
        """测试带字段的维度提取。"""
        domain_meta = mock_scenario_loader.get_expert_scenario().domain_meta

        result = planner._extract_dimensions(domain_meta)

        food_dim = next(d for d in result if d.id == "food")
        assert len(food_dim.fields) == 3  # 2 required + 1 optional

        # 检查 required 标记
        required_fields = [f for f in food_dim.fields if f["required"]]
        assert len(required_fields) == 2

    def test_extract_empty_dimensions(self, planner):
        """测试空维度配置。"""
        domain_meta = {"dimensions": {}}

        result = planner._extract_dimensions(domain_meta)

        assert result == []

    def test_extract_no_dimensions_key(self, planner):
        """测试无 dimensions 键。"""
        domain_meta = {}

        result = planner._extract_dimensions(domain_meta)

        assert result == []


# =============================================================================
# 测试 _parse_acs_to_candidate
# =============================================================================


class TestParseAcsToCandidate:
    """测试 ACS 解析。"""

    def test_basic_parse(self, planner):
        """测试基本 ACS 解析。"""
        acs_data = {
            "aic": "partner-food",
            "name": "美食助手",
            "description": "提供美食推荐",
            "skills": [
                {
                    "id": "search",
                    "name": "搜索",
                    "description": "搜索餐厅",
                    "tags": ["美食"],
                }
            ],
        }

        result = planner._parse_acs_to_candidate(acs_data, "static")

        assert result.partner_aic == "partner-food"
        assert result.partner_name == "美食助手"
        assert result.source == "static"
        assert len(result.skills) == 1

    def test_parse_no_skills(self, planner):
        """测试无技能的 ACS。"""
        acs_data = {
            "aic": "partner-001",
            "name": "助手",
            "description": "描述",
        }

        result = planner._parse_acs_to_candidate(acs_data, "dynamic")

        assert result.skills == []

    def test_parse_missing_fields(self, planner):
        """测试缺少字段的 ACS。"""
        acs_data = {}

        result = planner._parse_acs_to_candidate(acs_data, "static")

        assert result.partner_aic == ""
        assert result.partner_name == ""


# =============================================================================
# 测试 _build_input_context
# =============================================================================


class TestBuildInputContext:
    """测试输入上下文构建。"""

    def test_basic_context(
        self, planner, base_session, intent_decision, mock_scenario_loader
    ):
        """测试基本输入上下文。"""
        scenario_runtime = mock_scenario_loader.get_expert_scenario()
        dimensions = [DimensionDef(id="food", name="美食", description="描述")]
        dim_map = {"food": []}

        result = planner._build_input_context(
            user_query="我想吃烤鸭",
            intent=intent_decision,
            scenario_id="beijing_food",
            scenario_runtime=scenario_runtime,
            dimensions=dimensions,
            dimension_partner_map=dim_map,
            session=base_session,
        )

        assert result["userQuery"] == "我想吃烤鸭"
        assert result["taskInstruction"]["scenarioId"] == "beijing_food"
        assert "scenario" in result
        assert "dimensionPartnerMap" in result

    def test_context_with_partner_candidates(
        self, planner, base_session, intent_decision, mock_scenario_loader
    ):
        """测试包含 Partner 候选的上下文。"""
        scenario_runtime = mock_scenario_loader.get_expert_scenario()
        dimensions = [DimensionDef(id="food", name="美食", description="描述")]

        skill = PartnerSkillInfo(id="s1", name="技能", description="描述")
        candidate = PartnerCandidate(
            partner_aic="partner-001",
            partner_name="助手",
            description="描述",
            source="static",
            skills=[skill],
        )
        dim_map = {"food": [candidate]}

        result = planner._build_input_context(
            user_query="测试",
            intent=intent_decision,
            scenario_id="beijing_food",
            scenario_runtime=scenario_runtime,
            dimensions=dimensions,
            dimension_partner_map=dim_map,
            session=base_session,
        )

        assert len(result["dimensionPartnerMap"]["food"]) == 1
        assert result["dimensionPartnerMap"]["food"][0]["partnerAic"] == "partner-001"


# =============================================================================
# 测试 _validate_and_normalize
# =============================================================================


class TestValidateAndNormalize:
    """测试结果验证与规范化。"""

    def test_add_missing_dimensions(self, planner):
        """测试补充缺失维度。"""
        dimensions = [
            DimensionDef(id="food", name="美食", description=""),
            DimensionDef(id="hotel", name="酒店", description=""),
        ]
        dim_map = {"food": [], "hotel": []}

        result = PlanningResult(
            created_at="",
            scenario_id="test",
            selected_partners={
                "food": [
                    PartnerSelection(
                        partner_aic="p1",
                        skill_id="s1",
                        reason="test",
                        instruction_text="test",
                    )
                ],
            },
        )

        normalized = planner._validate_and_normalize(result, dimensions, dim_map)

        assert "hotel" in normalized.selected_partners
        assert normalized.selected_partners["hotel"] == []
        assert "hotel" in normalized.dimension_notes
        assert normalized.dimension_notes["hotel"].status == "inactive"

    def test_set_created_at(self, planner):
        """测试设置创建时间。"""
        dimensions = [DimensionDef(id="food", name="美食", description="")]
        dim_map = {"food": []}

        result = PlanningResult(
            created_at="",
            scenario_id="test",
            selected_partners={"food": []},
        )

        normalized = planner._validate_and_normalize(result, dimensions, dim_map)

        assert normalized.created_at != ""
        assert normalized.created_at is not None

    def test_validate_partner_aic(self, planner):
        """测试验证 Partner AIC。"""
        dimensions = [DimensionDef(id="food", name="美食", description="")]
        candidate = PartnerCandidate(
            partner_aic="valid-partner",
            partner_name="助手",
            description="",
            source="static",
            skills=[PartnerSkillInfo(id="skill1", name="技能", description="")],
        )
        dim_map = {"food": [candidate]}

        result = PlanningResult(
            created_at="2024-01-01T00:00:00Z",
            scenario_id="test",
            selected_partners={
                "food": [
                    PartnerSelection(
                        partner_aic="invalid-partner",  # 无效
                        skill_id="skill1",
                        reason="test",
                        instruction_text="test",
                    ),
                ],
            },
        )

        # 应该记录警告但不抛出异常
        normalized = planner._validate_and_normalize(result, dimensions, dim_map)
        assert normalized is not None


# =============================================================================
# 测试 plan 方法
# =============================================================================


class TestPlan:
    """测试 plan 方法。"""

    @pytest.mark.asyncio
    async def test_plan_success(
        self, planner, base_session, intent_decision, mock_llm_client
    ):
        """测试成功规划。"""
        # Mock ACS 文件加载
        with patch.object(planner, "_load_partner_acs_by_filename") as mock_load:
            mock_load.return_value = PartnerCandidate(
                partner_aic="partner-food",
                partner_name="美食助手",
                description="",
                source="static",
                skills=[
                    PartnerSkillInfo(id="search_food", name="搜索", description="")
                ],
            )

            result = await planner.plan(
                user_query="我想吃烤鸭",
                session=base_session,
                intent=intent_decision,
            )

            assert result.scenario_id == "beijing_food"
            mock_llm_client.call_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_no_scenario(self, planner, intent_decision):
        """测试无场景时抛出错误。"""
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
        )
        intent_decision.target_scenario = None

        with pytest.raises(LLMCallError) as exc_info:
            await planner.plan(
                user_query="测试",
                session=session,
                intent=intent_decision,
            )

        assert "No scenario" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_plan_scenario_not_found(
        self, planner, base_session, intent_decision, mock_scenario_loader
    ):
        """测试场景未找到。"""
        mock_scenario_loader.get_expert_scenario.return_value = None

        with pytest.raises(LLMCallError) as exc_info:
            await planner.plan(
                user_query="测试",
                session=base_session,
                intent=intent_decision,
            )

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_plan_prompt_not_found(
        self, planner, base_session, intent_decision, mock_scenario_loader
    ):
        """测试 Prompt 未找到。"""
        mock_scenario_loader.get_prompt.return_value = None

        with patch.object(planner, "_load_partner_acs_by_filename"):
            with pytest.raises(LLMCallError) as exc_info:
                await planner.plan(
                    user_query="测试",
                    session=base_session,
                    intent=intent_decision,
                )

            assert "prompt not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_plan_llm_error(
        self, planner, base_session, intent_decision, mock_llm_client
    ):
        """测试 LLM 调用失败。"""
        mock_llm_client.call_structured.side_effect = Exception("LLM unavailable")

        with patch.object(planner, "_load_partner_acs_by_filename") as mock_load:
            mock_load.return_value = None

            with pytest.raises(LLMCallError) as exc_info:
                await planner.plan(
                    user_query="测试",
                    session=base_session,
                    intent=intent_decision,
                )

            assert "failed" in str(exc_info.value).lower()


# =============================================================================
# 边界测试
# =============================================================================


class TestEdgeCases:
    """测试边界情况。"""

    def test_dimension_with_no_fields(self, planner):
        """测试无字段的维度。"""
        domain_meta = {
            "dimensions": {
                "simple": {
                    "name": "简单维度",
                    "description": "描述",
                }
            }
        }

        result = planner._extract_dimensions(domain_meta)

        assert len(result) == 1
        assert result[0].fields == []

    def test_acs_cache_hit(self, planner):
        """测试 ACS 缓存命中。"""
        cache_key = "/fake/path:test.json"
        planner._acs_cache[cache_key] = {
            "aic": "cached-partner",
            "name": "缓存助手",
            "description": "",
            "skills": [],
        }

        result = planner._parse_acs_to_candidate(
            planner._acs_cache[cache_key], "static"
        )

        assert result.partner_aic == "cached-partner"

    def test_empty_selected_partners(self, planner):
        """测试空选择结果。"""
        dimensions = [DimensionDef(id="food", name="美食", description="")]
        dim_map = {"food": []}

        result = PlanningResult(
            created_at="2024-01-01T00:00:00Z",
            scenario_id="test",
            selected_partners={},
        )

        normalized = planner._validate_and_normalize(result, dimensions, dim_map)

        assert "food" in normalized.selected_partners


# =============================================================================
# 测试动态发现集成（_build_dimension_partner_map + _query_discovery_server）
# =============================================================================


class TestDynamicDiscovery:
    """测试动态发现服务集成。"""

    @pytest.fixture
    def mock_discovery_client(self):
        """创建 Mock 发现客户端。"""
        from unittest.mock import AsyncMock, PropertyMock

        client = MagicMock()
        type(client).is_configured = PropertyMock(return_value=True)
        client.discover_for_dimension = AsyncMock()
        return client

    @pytest.fixture
    def planner_with_discovery(
        self, mock_llm_client, mock_scenario_loader, mock_discovery_client
    ):
        """创建带发现客户端的 Planner（prefer_static=True）。"""
        p = Planner(
            scenario_loader=mock_scenario_loader,
            llm_client=mock_llm_client,
            discovery_client=mock_discovery_client,
        )
        p._prefer_static = True
        return p

    @pytest.fixture
    def planner_dynamic_only(
        self, mock_llm_client, mock_scenario_loader, mock_discovery_client
    ):
        """创建 prefer_static=False 的 Planner。"""
        p = Planner(
            scenario_loader=mock_scenario_loader,
            llm_client=mock_llm_client,
            discovery_client=mock_discovery_client,
        )
        p._prefer_static = False
        return p

    def _make_discovery_response(self, aic="AIC-DYN-001", name="动态Agent"):
        """构造用于测试的 DiscoveryResponse。"""
        from acps_sdk.adp import (
            DiscoveryResponse,
            DiscoveryResult,
            DiscoveryAgentGroup,
            DiscoveryAgentSkill,
        )

        return DiscoveryResponse.success(
            result=DiscoveryResult(
                acs_map={
                    aic: {
                        "name": name,
                        "description": "动态发现的智能体",
                        "skills": [
                            {
                                "id": "skill-dyn-001",
                                "name": "动态技能",
                                "description": "描述",
                                "tags": ["test"],
                            }
                        ],
                    }
                },
                agents=[
                    DiscoveryAgentGroup(
                        group="test",
                        agent_skills=[
                            DiscoveryAgentSkill(
                                aic=aic,
                                skill_id="skill-dyn-001",
                                ranking=1,
                            )
                        ],
                    )
                ],
            )
        )

    @pytest.mark.asyncio
    async def test_prefer_static_uses_static_when_configured(
        self, planner_with_discovery, mock_discovery_client
    ):
        """prefer_static=True 且维度有静态配置时，使用静态，不查询发现服务。"""
        dimensions = [DimensionDef(id="food", name="美食", description="餐厅推荐")]
        domain_meta = {
            "partners": {"mapping": {"food": "beijing_food.json"}},
        }

        with patch.object(
            planner_with_discovery, "_load_partner_acs_by_filename"
        ) as mock_load:
            mock_load.return_value = PartnerCandidate(
                partner_aic="static-partner",
                partner_name="静态助手",
                description="",
                source="static",
                skills=[],
            )

            result = await planner_with_discovery._build_dimension_partner_map(
                scenario_id="tour",
                dimensions=dimensions,
                domain_meta=domain_meta,
            )

        assert len(result["food"]) == 1
        assert result["food"][0].partner_aic == "static-partner"
        assert result["food"][0].source == "static"
        mock_discovery_client.discover_for_dimension.assert_not_called()

    @pytest.mark.asyncio
    async def test_prefer_static_falls_back_to_dynamic(
        self, planner_with_discovery, mock_discovery_client
    ):
        """prefer_static=True 但维度无静态配置时，自动动态查询。"""
        dimensions = [
            DimensionDef(id="food", name="美食", description="餐厅推荐"),
            DimensionDef(id="weather", name="天气", description="天气预报"),
        ]
        domain_meta = {
            "partners": {"mapping": {"food": "beijing_food.json"}},
        }

        mock_discovery_client.discover_for_dimension.return_value = (
            self._make_discovery_response()
        )

        with patch.object(
            planner_with_discovery, "_load_partner_acs_by_filename"
        ) as mock_load:
            mock_load.return_value = PartnerCandidate(
                partner_aic="static-partner",
                partner_name="静态助手",
                description="",
                source="static",
                skills=[],
            )

            result = await planner_with_discovery._build_dimension_partner_map(
                scenario_id="tour",
                dimensions=dimensions,
                domain_meta=domain_meta,
            )

        # food 使用静态
        assert result["food"][0].source == "static"
        # weather 无静态配置，走动态
        assert len(result["weather"]) == 1
        assert result["weather"][0].source == "dynamic"
        mock_discovery_client.discover_for_dimension.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_prefer_static_always_dynamic(
        self, planner_dynamic_only, mock_discovery_client
    ):
        """prefer_static=False 时，所有维度都走动态查询，不加载静态。"""
        dimensions = [DimensionDef(id="food", name="美食", description="餐厅推荐")]
        domain_meta = {
            "partners": {"mapping": {"food": "beijing_food.json"}},
        }

        mock_discovery_client.discover_for_dimension.return_value = (
            self._make_discovery_response()
        )

        result = await planner_dynamic_only._build_dimension_partner_map(
            scenario_id="tour",
            dimensions=dimensions,
            domain_meta=domain_meta,
        )

        assert len(result["food"]) == 1
        assert result["food"][0].source == "dynamic"
        mock_discovery_client.discover_for_dimension.assert_called_once()

    @pytest.mark.asyncio
    async def test_dynamic_query_failure_returns_empty(
        self, planner_dynamic_only, mock_discovery_client
    ):
        """动态查询失败时返回空列表（不抛异常）。"""
        from assistant.services.discovery_client import DiscoveryClientError

        dimensions = [DimensionDef(id="food", name="美食", description="餐厅推荐")]
        domain_meta = {"partners": {"mapping": {}}}

        mock_discovery_client.discover_for_dimension.side_effect = DiscoveryClientError(
            "连接失败"
        )

        result = await planner_dynamic_only._build_dimension_partner_map(
            scenario_id="tour",
            dimensions=dimensions,
            domain_meta=domain_meta,
        )

        assert result["food"] == []

    @pytest.mark.asyncio
    async def test_dynamic_discovery_not_configured(
        self, mock_llm_client, mock_scenario_loader
    ):
        """发现服务未配置时跳过动态查询，返回空列表。"""
        unconfigured_client = MagicMock()
        from unittest.mock import PropertyMock

        type(unconfigured_client).is_configured = PropertyMock(return_value=False)

        p = Planner(
            scenario_loader=mock_scenario_loader,
            llm_client=mock_llm_client,
            discovery_client=unconfigured_client,
        )
        p._prefer_static = False

        dimensions = [DimensionDef(id="food", name="美食", description="餐厅推荐")]
        domain_meta = {"partners": {"mapping": {}}}

        result = await p._build_dimension_partner_map(
            scenario_id="tour",
            dimensions=dimensions,
            domain_meta=domain_meta,
        )

        assert result["food"] == []

    def test_parse_discovery_response(self, planner_with_discovery):
        """测试 _parse_discovery_response 转换。"""
        response = self._make_discovery_response(aic="AIC-PARSE-001", name="解析测试")

        candidates = planner_with_discovery._parse_discovery_response(response)

        assert len(candidates) == 1
        assert candidates[0].partner_aic == "AIC-PARSE-001"
        assert candidates[0].source == "dynamic"
        assert candidates[0].partner_name == "解析测试"

    def test_parse_discovery_response_dedup(self, planner_with_discovery):
        """测试 _parse_discovery_response 去重。"""
        from acps_sdk.adp import (
            DiscoveryResponse,
            DiscoveryResult,
            DiscoveryAgentGroup,
            DiscoveryAgentSkill,
        )

        response = DiscoveryResponse.success(
            result=DiscoveryResult(
                acs_map={
                    "SAME-AIC": {
                        "name": "Agent",
                        "description": "",
                        "skills": [],
                    }
                },
                agents=[
                    DiscoveryAgentGroup(
                        group="g1",
                        agent_skills=[
                            DiscoveryAgentSkill(
                                aic="SAME-AIC", skill_id="s1", ranking=1
                            ),
                            DiscoveryAgentSkill(
                                aic="SAME-AIC", skill_id="s2", ranking=2
                            ),
                        ],
                    )
                ],
            )
        )

        candidates = planner_with_discovery._parse_discovery_response(response)

        # 同一 AIC 应去重
        assert len(candidates) == 1

    def test_parse_discovery_response_empty(self, planner_with_discovery):
        """测试空发现响应的解析。"""
        from acps_sdk.adp import DiscoveryResponse, DiscoveryResult

        response = DiscoveryResponse.success(result=DiscoveryResult(agents=[]))

        candidates = planner_with_discovery._parse_discovery_response(response)

        assert candidates == []
