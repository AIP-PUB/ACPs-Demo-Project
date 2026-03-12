"""
Leader Agent Platform - TaskExecutor 单元测试

测试内容：
1. 配置 (ExecutorConfig)
2. 数据模型 (ExecutionPhase, PartnerExecutionResult, ExecutionResult)
3. Partner 端点提取 (extract_partner_endpoint)
4. Partner 任务构建 (_build_partner_tasks)
5. 收敛检查 (_check_convergence)
6. 结果分类 (_classify_results)
7. execute 方法的核心逻辑
8. continue_partner 和 complete_partner 方法
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

from assistant.core.executor import (
    TaskExecutor,
    ExecutorConfig,
    ExecutionPhase,
    ExecutionResult,
    PartnerExecutionResult,
    extract_partner_endpoint,
)
from assistant.models.task import (
    PlanningResult,
    PartnerSelection,
)
from acps_sdk.aip.aip_base_model import TaskResult, TaskState, TaskStatus


@pytest.fixture
def executor_config():
    """创建执行器配置。"""
    return ExecutorConfig(
        poll_interval_ms=100,  # 测试用短间隔
        max_poll_retries=2,
        start_timeout_ms=5000,
        max_execution_rounds=5,
        convergence_timeout_s=10,
    )


@pytest.fixture
def executor(executor_config):
    """创建 TaskExecutor 实例。"""
    return TaskExecutor(
        leader_aic="test-leader",
        config=executor_config,
    )


@pytest.fixture
def planning_result():
    """创建测试规划结果。"""
    return PlanningResult(
        created_at="2024-01-01T00:00:00Z",
        scenario_id="beijing_food",
        user_query="我想吃烤鸭",
        selected_partners={
            "food": [
                PartnerSelection(
                    partner_aic="partner-food",
                    skill_id="search_food",
                    reason="用户想找餐厅",
                    instruction_text="搜索北京烤鸭餐厅",
                ),
            ],
            "transport": [],
        },
    )


@pytest.fixture
def mock_task_working():
    """创建Working状态的 TaskResult。"""
    return TaskResult(
        id="result-task-001",
        sentAt=datetime.now().isoformat(),
        senderRole="partner",
        senderId="test-partner",
        taskId="task-001",
        sessionId="session-001",
        status=TaskStatus(
            state=TaskState.Working,
            stateChangedAt=datetime.now().isoformat(),
            dataItems=[],
        ),
    )


@pytest.fixture
def mock_task_completed():
    """创建已完成的 TaskResult。"""
    return TaskResult(
        id="result-task-001",
        sentAt=datetime.now().isoformat(),
        senderRole="partner",
        senderId="test-partner",
        taskId="task-001",
        sessionId="session-001",
        status=TaskStatus(
            state=TaskState.Completed,
            stateChangedAt=datetime.now().isoformat(),
            dataItems=[{"type": "text", "text": "推荐全聚德"}],
        ),
    )


@pytest.fixture
def mock_task_awaiting_input():
    """创建等待输入的 TaskResult。"""
    return TaskResult(
        id="result-task-001",
        sentAt=datetime.now().isoformat(),
        senderRole="partner",
        senderId="test-partner",
        taskId="task-001",
        sessionId="session-001",
        status=TaskStatus(
            state=TaskState.AwaitingInput,
            stateChangedAt=datetime.now().isoformat(),
            dataItems=[{"type": "text", "text": "请问您预算多少？"}],
        ),
    )


# =============================================================================
# 测试 ExecutorConfig
# =============================================================================


class TestExecutorConfig:
    """测试执行器配置。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = ExecutorConfig()
        assert config.poll_interval_ms == 2000
        assert config.max_poll_retries == 3
        assert config.max_execution_rounds == 50

    def test_custom_config(self):
        """测试自定义配置。"""
        config = ExecutorConfig(
            poll_interval_ms=500,
            max_execution_rounds=10,
        )
        assert config.poll_interval_ms == 500
        assert config.max_execution_rounds == 10


# =============================================================================
# 测试数据模型
# =============================================================================


class TestExecutionPhase:
    """测试执行阶段枚举。"""

    def test_all_phases(self):
        """测试所有阶段。"""
        assert ExecutionPhase.STARTING == "starting"
        assert ExecutionPhase.POLLING == "polling"
        assert ExecutionPhase.AWAITING_INPUT == "awaiting_input"
        assert ExecutionPhase.AWAITING_COMPLETION == "awaiting_completion"
        assert ExecutionPhase.COMPLETED == "completed"
        assert ExecutionPhase.FAILED == "failed"
        assert ExecutionPhase.TIMEOUT == "timeout"


class TestPartnerExecutionResult:
    """测试 Partner 执行结果。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        result = PartnerExecutionResult(
            partner_aic="partner-001",
            dimension_id="food",
            state=TaskState.Working,
        )
        assert result.partner_aic == "partner-001"
        assert result.state == TaskState.Working
        assert result.task is None
        assert result.error is None

    def test_with_task(self, mock_task_completed):
        """测试带 Task 创建。"""
        result = PartnerExecutionResult(
            partner_aic="partner-001",
            dimension_id="food",
            state=TaskState.Completed,
            task=mock_task_completed,
            data_items=mock_task_completed.status.dataItems,
        )
        assert result.task is not None
        assert len(result.data_items) > 0

    def test_with_error(self):
        """测试带错误创建。"""
        result = PartnerExecutionResult(
            partner_aic="partner-001",
            dimension_id="food",
            state=TaskState.Failed,
            error="连接超时",
        )
        assert result.error == "连接超时"


class TestExecutionResult:
    """测试执行结果。"""

    def test_initial_state(self):
        """测试初始状态。"""
        result = ExecutionResult(phase=ExecutionPhase.STARTING)
        assert result.phase == ExecutionPhase.STARTING
        assert result.partner_results == {}
        assert result.awaiting_input_partners == []
        assert result.completed_partners == []

    def test_with_results(self):
        """测试带结果创建。"""
        result = ExecutionResult(
            phase=ExecutionPhase.COMPLETED,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.Completed,
                ),
            },
            completed_partners=["p1"],
        )
        assert len(result.partner_results) == 1
        assert "p1" in result.completed_partners


# =============================================================================
# 测试 extract_partner_endpoint
# =============================================================================


class TestExtractPartnerEndpoint:
    """测试端点提取。"""

    def test_http_endpoint(self):
        """测试 HTTP 端点提取。"""
        acs_data = {
            "endPoints": [
                {"url": "http://localhost:8011/rpc", "transport": "HTTP"},
            ]
        }
        result = extract_partner_endpoint(acs_data)
        assert result == "http://localhost:8011/rpc"

    def test_jsonrpc_endpoint(self):
        """测试 JSONRPC 端点提取。"""
        acs_data = {
            "endPoints": [
                {"url": "http://localhost:8011/jsonrpc", "transport": "JSONRPC"},
            ]
        }
        result = extract_partner_endpoint(acs_data)
        assert result == "http://localhost:8011/jsonrpc"

    def test_multiple_endpoints(self):
        """测试多端点时优先 HTTP。"""
        acs_data = {
            "endPoints": [
                {"url": "grpc://localhost:8012", "transport": "GRPC"},
                {"url": "http://localhost:8011/rpc", "transport": "HTTP"},
            ]
        }
        result = extract_partner_endpoint(acs_data)
        assert result == "http://localhost:8011/rpc"

    def test_fallback_to_first(self):
        """测试降级到第一个端点。"""
        acs_data = {
            "endPoints": [
                {"url": "grpc://localhost:8012", "transport": "GRPC"},
            ]
        }
        result = extract_partner_endpoint(acs_data)
        assert result == "grpc://localhost:8012"

    def test_no_endpoints(self):
        """测试无端点返回 None。"""
        acs_data = {"endPoints": []}
        result = extract_partner_endpoint(acs_data)
        assert result is None

    def test_empty_acs(self):
        """测试空 ACS 数据。"""
        result = extract_partner_endpoint({})
        assert result is None


# =============================================================================
# 测试 _build_partner_tasks
# =============================================================================


class TestBuildPartnerTasks:
    """测试 Partner 任务构建。"""

    def test_build_with_selection(self, executor, planning_result):
        """测试构建 Partner 任务。"""
        result = executor._build_partner_tasks(
            session_id="session-001",
            active_task_id="task-001",
            planning_result=planning_result,
        )

        assert "partner-food" in result
        assert result["partner-food"]["dimension_id"] == "food"
        assert "aip_task_id" in result["partner-food"]

    def test_build_empty_selections(self, executor):
        """测试空选择列表。"""
        planning_result = PlanningResult(
            created_at="2024-01-01T00:00:00Z",
            scenario_id="test",
            selected_partners={"food": []},
        )

        result = executor._build_partner_tasks(
            session_id="session-001",
            active_task_id="task-001",
            planning_result=planning_result,
        )

        assert result == {}

    def test_build_with_acs_cache(self, executor, planning_result):
        """测试使用 ACS 缓存。"""
        executor.acs_cache = {
            "partner-food": {
                "endPoints": [
                    {"url": "http://custom:9000/rpc", "transport": "HTTP"},
                ]
            }
        }

        result = executor._build_partner_tasks(
            session_id="session-001",
            active_task_id="task-001",
            planning_result=planning_result,
        )

        assert result["partner-food"]["endpoint"] == "http://custom:9000/rpc"


# =============================================================================
# 测试 _check_convergence
# =============================================================================


class TestCheckConvergence:
    """测试收敛检查。"""

    def test_awaiting_input(self, executor):
        """测试 AwaitingInput 状态收敛。"""
        result = ExecutionResult(
            phase=ExecutionPhase.POLLING,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.AwaitingInput,
                ),
            },
        )

        converged, phase = executor._check_convergence(result)

        assert converged is True
        assert phase == ExecutionPhase.AWAITING_INPUT

    def test_awaiting_completion(self, executor):
        """测试 AwaitingCompletion 状态收敛。"""
        result = ExecutionResult(
            phase=ExecutionPhase.POLLING,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.AwaitingCompletion,
                ),
            },
        )

        converged, phase = executor._check_convergence(result)

        assert converged is True
        assert phase == ExecutionPhase.AWAITING_COMPLETION

    def test_all_completed(self, executor):
        """测试全部完成收敛。"""
        result = ExecutionResult(
            phase=ExecutionPhase.POLLING,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.Completed,
                ),
                "p2": PartnerExecutionResult(
                    partner_aic="p2",
                    dimension_id="hotel",
                    state=TaskState.Completed,
                ),
            },
        )

        converged, phase = executor._check_convergence(result)

        assert converged is True
        assert phase == ExecutionPhase.COMPLETED

    def test_some_failed(self, executor):
        """测试部分失败收敛。"""
        result = ExecutionResult(
            phase=ExecutionPhase.POLLING,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.Completed,
                ),
                "p2": PartnerExecutionResult(
                    partner_aic="p2",
                    dimension_id="hotel",
                    state=TaskState.Failed,
                ),
            },
        )

        converged, phase = executor._check_convergence(result)

        assert converged is True
        assert phase == ExecutionPhase.FAILED

    def test_still_working(self, executor):
        """测试仍在工作中不收敛。"""
        result = ExecutionResult(
            phase=ExecutionPhase.POLLING,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.Working,
                ),
                "p2": PartnerExecutionResult(
                    partner_aic="p2",
                    dimension_id="hotel",
                    state=TaskState.Completed,
                ),
            },
        )

        converged, phase = executor._check_convergence(result)

        assert converged is False
        assert phase == ExecutionPhase.POLLING


# =============================================================================
# 测试 _classify_results
# =============================================================================


class TestClassifyResults:
    """测试结果分类。"""

    def test_classify_mixed_results(self, executor, mock_task_awaiting_input):
        """测试混合结果分类。"""
        result = ExecutionResult(
            phase=ExecutionPhase.AWAITING_INPUT,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.AwaitingInput,
                    task=mock_task_awaiting_input,
                    data_items=mock_task_awaiting_input.status.dataItems,
                ),
                "p2": PartnerExecutionResult(
                    partner_aic="p2",
                    dimension_id="hotel",
                    state=TaskState.Completed,
                ),
                "p3": PartnerExecutionResult(
                    partner_aic="p3",
                    dimension_id="transport",
                    state=TaskState.Failed,
                    error="连接失败",
                ),
            },
        )

        executor._classify_results(result)

        assert "p1" in result.awaiting_input_partners
        assert "p2" in result.completed_partners
        assert "p3" in result.failed_partners
        assert len(result.questions_for_user) > 0

    def test_classify_awaiting_completion(self, executor, mock_task_completed):
        """测试 AwaitingCompletion 分类。"""
        result = ExecutionResult(
            phase=ExecutionPhase.AWAITING_COMPLETION,
            partner_results={
                "p1": PartnerExecutionResult(
                    partner_aic="p1",
                    dimension_id="food",
                    state=TaskState.AwaitingCompletion,
                    data_items=[{"type": "text", "text": "产出物"}],
                ),
            },
        )

        executor._classify_results(result)

        assert "p1" in result.awaiting_completion_partners
        assert "p1" in result.products
        assert len(result.products["p1"]) > 0


# =============================================================================
# 测试 execute 方法
# =============================================================================


class TestExecute:
    """测试 execute 方法。"""

    @pytest.mark.asyncio
    async def test_execute_empty_partners(self, executor):
        """测试无 Partner 执行。"""
        planning_result = PlanningResult(
            created_at="2024-01-01T00:00:00Z",
            scenario_id="test",
            selected_partners={},
        )

        result = await executor.execute(
            session_id="session-001",
            active_task_id="task-001",
            planning_result=planning_result,
        )

        assert result.phase == ExecutionPhase.COMPLETED
        assert result.partner_results == {}

    @pytest.mark.asyncio
    async def test_execute_with_mock_rpc(
        self, executor, planning_result, mock_task_completed
    ):
        """测试带 Mock RPC 的执行。"""
        # Mock RPC 客户端
        mock_client = MagicMock()
        mock_client.start_task = AsyncMock(return_value=mock_task_completed)
        mock_client.get_task = AsyncMock(return_value=mock_task_completed)
        mock_client.close = AsyncMock()

        with patch.object(executor, "_get_or_create_client", return_value=mock_client):
            result = await executor.execute(
                session_id="session-001",
                active_task_id="task-001",
                planning_result=planning_result,
            )

        assert result.phase == ExecutionPhase.COMPLETED
        assert "partner-food" in result.partner_results


# =============================================================================
# 测试 continue_partner 和 complete_partner
# =============================================================================


class TestContinueAndComplete:
    """测试继续和完成 Partner。"""

    @pytest.mark.asyncio
    async def test_continue_partner_success(self, executor, mock_task_working):
        """测试成功继续 Partner。"""
        mock_client = MagicMock()
        mock_client.continue_task = AsyncMock(return_value=mock_task_working)

        with patch.object(executor, "_get_or_create_client", return_value=mock_client):
            task, error = await executor.continue_partner(
                session_id="session-001",
                partner_aic="partner-food",
                aip_task_id="task-001",
                endpoint="http://localhost:8011/rpc",
                user_input="500元",
            )

        assert task is not None
        assert error is None

    @pytest.mark.asyncio
    async def test_continue_partner_error(self, executor):
        """测试继续 Partner 失败。"""
        mock_client = MagicMock()
        mock_client.continue_task = AsyncMock(side_effect=Exception("连接失败"))

        with patch.object(executor, "_get_or_create_client", return_value=mock_client):
            task, error = await executor.continue_partner(
                session_id="session-001",
                partner_aic="partner-food",
                aip_task_id="task-001",
                endpoint="http://localhost:8011/rpc",
                user_input="500元",
            )

        assert task is None
        assert "连接失败" in error

    @pytest.mark.asyncio
    async def test_complete_partner_success(self, executor, mock_task_completed):
        """测试成功完成 Partner。"""
        mock_client = MagicMock()
        mock_client.complete_task = AsyncMock(return_value=mock_task_completed)

        with patch.object(executor, "_get_or_create_client", return_value=mock_client):
            task, error = await executor.complete_partner(
                session_id="session-001",
                partner_aic="partner-food",
                aip_task_id="task-001",
                endpoint="http://localhost:8011/rpc",
            )

        assert task is not None
        assert error is None


# =============================================================================
# 测试清理
# =============================================================================


class TestCleanup:
    """测试清理功能。"""

    @pytest.mark.asyncio
    async def test_cleanup_clients(self, executor):
        """测试清理 RPC 客户端。"""
        # 模拟有客户端
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        executor._rpc_clients = {"p1": mock_client}

        await executor._cleanup_clients()

        mock_client.close.assert_called_once()
        assert executor._rpc_clients == {}

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self, executor):
        """测试清理时出错不抛异常。"""
        mock_client = MagicMock()
        mock_client.close = AsyncMock(side_effect=Exception("关闭失败"))
        executor._rpc_clients = {"p1": mock_client}

        # 不应抛出异常
        await executor._cleanup_clients()

        assert executor._rpc_clients == {}
