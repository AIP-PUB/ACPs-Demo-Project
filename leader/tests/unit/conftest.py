"""
Leader Agent Platform - 单元测试 Fixtures

本模块提供单元测试所需的公共 fixtures。
单元测试特点：
- 不依赖真实 LLM 调用（使用 mock）
- 不依赖外部服务
- 测试单个组件的独立功能
"""

import sys
import os
from pathlib import Path

# 确保路径正确
_current_dir = Path(__file__).parent
_leader_dir = _current_dir.parent.parent
_project_root = _leader_dir.parent

if str(_leader_dir) not in sys.path:
    sys.path.insert(0, str(_leader_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端。"""
    client = MagicMock()
    client.chat = AsyncMock()
    return client


@pytest.fixture
def mock_scenario_loader():
    """Mock 场景加载器。"""
    from assistant.services.scenario_loader import ScenarioLoader
    from assistant.models import ScenarioBrief, ScenarioRuntime
    from assistant.models.base import now_iso

    loader = MagicMock(spec=ScenarioLoader)
    loader.scenario_briefs = [
        ScenarioBrief(
            id="tour",
            name="旅游助手",
            description="提供旅游规划服务",
            keywords=["旅游", "行程", "酒店", "景点"],
        )
    ]
    loader.get_prompt.return_value = {
        "system": "你是一个智能助手。",
        "llm_profile": "llm.default",
    }
    loader.get_expert_scenario.return_value = MagicMock()

    # 提供真实的 base_scenario
    loader.base_scenario = ScenarioRuntime(
        id="base",
        kind="base",
        version="1.0.0",
        loaded_at=now_iso(),
    )
    return loader


@pytest.fixture
def sample_session():
    """创建示例 Session。"""
    from assistant.models import (
        Session,
        ScenarioRuntime,
        DialogContext,
        ExecutionMode,
        UserResult,
        UserResultType,
    )
    from assistant.models.base import now_iso
    from datetime import datetime, timezone, timedelta

    now = now_iso()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    base_scenario = ScenarioRuntime(
        id="base",
        kind="base",
        version="1.0.0",
        loaded_at=now,
    )

    return Session(
        session_id="test-session-001",
        mode=ExecutionMode.DIRECT_RPC,
        created_at=now,
        updated_at=now,
        touched_at=now,
        ttl_seconds=3600,
        expires_at=expires_at,
        base_scenario=base_scenario,
        dialog_context=DialogContext(
            dialog_summary="",
            history_turns=[],
            slot_memory={},
        ),
        event_log=[],
        user_result=UserResult(
            type=UserResultType.PENDING,
            data_items=[],
            updated_at=now,
        ),
    )


@pytest.fixture
def sample_active_task():
    """创建示例活跃任务。"""
    from assistant.models import ActiveTask, ActiveTaskStatus
    from assistant.models.base import now_iso

    return ActiveTask(
        active_task_id="task-001",
        scenario_id="tour",
        created_at=now_iso(),
        external_status=ActiveTaskStatus.RUNNING,
        partner_tasks={},
        dimension_map={},
    )


@pytest.fixture
def session_manager():
    """创建 SessionManager 实例。"""
    from assistant.core import SessionManager

    return SessionManager()


@pytest.fixture
def orchestrator(mock_scenario_loader, session_manager):
    """创建 Orchestrator 实例（使用 Mock 组件）。"""
    from assistant.core import Orchestrator, Planner, create_intent_analyzer
    from assistant.core.history_compressor import HistoryCompressor

    intent_analyzer = create_intent_analyzer(mock_scenario_loader)
    planner = Planner(mock_scenario_loader)

    # Mock LLM client for history compressor
    mock_llm = MagicMock()
    history_compressor = HistoryCompressor(
        llm_client=mock_llm,
        scenario_loader=mock_scenario_loader,
    )

    return Orchestrator(
        session_manager=session_manager,
        scenario_loader=mock_scenario_loader,
        intent_analyzer=intent_analyzer,
        planner=planner,
        history_compressor=history_compressor,
    )
