"""
TaskExecutionManager 单元测试

测试内容：
1. 任务创建与查询
2. 状态更新
3. 过期清理
4. 索引正确性
"""

import sys
from pathlib import Path

# 确保路径正确
_current_dir = Path(__file__).parent
_leader_dir = _current_dir.parent.parent
_project_root = _leader_dir.parent

if str(_leader_dir) not in sys.path:
    sys.path.insert(0, str(_leader_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# 直接导入模块，避免 __init__.py 的依赖链问题
from assistant.core.task_execution_manager import (
    TaskExecutionManager,
    DEFAULT_TASK_RETENTION_SECONDS,
    get_task_execution_manager,
    reset_task_execution_manager,
)
from assistant.models.task_execution import (
    TaskExecution,
    TaskExecutionStatus,
    TaskExecutionPhase,
    TaskExecutionProgress,
)


@pytest.fixture
def manager():
    """创建测试用的 TaskExecutionManager 实例。"""
    return TaskExecutionManager(retention_seconds=60)


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试后重置单例。"""
    yield
    reset_task_execution_manager()


class TestTaskCreation:
    """任务创建测试。"""

    def test_create_task_basic(self, manager):
        """测试基本任务创建。"""
        task = manager.create_task(
            task_id="task-001",
            session_id="session-001",
        )

        assert task is not None
        assert task.task_id == "task-001"
        assert task.session_id == "session-001"
        assert task.status == TaskExecutionStatus.PENDING
        assert task.created_at is not None
        assert task.updated_at is not None

    def test_create_task_with_planning_result(self, manager):
        """测试带规划结果的任务创建。"""
        planning_result = {
            "scenario_id": "beijing_food",
            "selected_partners": {"dim1": ["partner-a"]},
        }

        task = manager.create_task(
            task_id="task-002",
            session_id="session-001",
            planning_result=planning_result,
        )

        assert task.planning_result == planning_result

    def test_create_task_with_metadata(self, manager):
        """测试带元数据的任务创建。"""
        metadata = {"user_query": "帮我推荐北京美食"}

        task = manager.create_task(
            task_id="task-003",
            session_id="session-001",
            metadata=metadata,
        )

        assert task.metadata == metadata

    def test_create_duplicate_task(self, manager):
        """测试重复创建任务返回已有任务。"""
        task1 = manager.create_task(
            task_id="task-004",
            session_id="session-001",
        )
        task2 = manager.create_task(
            task_id="task-004",
            session_id="session-002",  # 不同 session
        )

        # 应该返回第一次创建的任务
        assert task1.session_id == task2.session_id == "session-001"


class TestTaskQuery:
    """任务查询测试。"""

    def test_get_task(self, manager):
        """测试获取单个任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        task = manager.get_task("task-001")
        assert task is not None
        assert task.task_id == "task-001"

    def test_get_nonexistent_task(self, manager):
        """测试获取不存在的任务。"""
        task = manager.get_task("nonexistent")
        assert task is None

    def test_get_tasks_by_session(self, manager):
        """测试按会话查询任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")
        manager.create_task(task_id="task-003", session_id="session-002")

        tasks = manager.get_tasks_by_session("session-001")
        assert len(tasks) == 2
        task_ids = {t.task_id for t in tasks}
        assert task_ids == {"task-001", "task-002"}

    def test_get_tasks_by_session_exclude_terminal(self, manager):
        """测试排除终态任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")
        manager.mark_task_completed(task_id="task-001")

        tasks = manager.get_tasks_by_session(
            "session-001",
            include_terminal=False,
        )
        assert len(tasks) == 1
        assert tasks[0].task_id == "task-002"

    def test_get_active_task_for_session(self, manager):
        """测试获取会话的活跃任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")
        manager.mark_task_completed(task_id="task-001")

        task = manager.get_active_task_for_session("session-001")
        assert task is not None
        assert task.task_id == "task-002"

    def test_get_latest_task_for_session(self, manager):
        """测试获取会话最新任务（含已完成）。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")
        manager.mark_task_completed(task_id="task-002")

        task = manager.get_latest_task_for_session("session-001")
        assert task is not None
        assert task.task_id == "task-002"


class TestStatusUpdates:
    """状态更新测试。"""

    def test_mark_task_running(self, manager):
        """测试标记为运行中。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        task = manager.mark_task_running("task-001")

        assert task.status == TaskExecutionStatus.RUNNING
        assert task.started_at is not None

    def test_mark_task_awaiting_input(self, manager):
        """测试标记为等待输入。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        clarification = "请提供您的预算范围"
        task = manager.mark_task_awaiting_input("task-001", clarification)

        assert task.status == TaskExecutionStatus.AWAITING_INPUT
        assert task.clarification_text == clarification

    def test_mark_task_completed(self, manager):
        """测试标记为完成。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        execution_result = {"phase": "completed"}
        aggregation_result = {"summary": "推荐结果"}
        response_text = "为您推荐以下美食..."

        task = manager.mark_task_completed(
            task_id="task-001",
            execution_result=execution_result,
            aggregation_result=aggregation_result,
            response_text=response_text,
        )

        assert task.status == TaskExecutionStatus.COMPLETED
        assert task.completed_at is not None
        assert task.execution_result == execution_result
        assert task.aggregation_result == aggregation_result
        assert task.response_text == response_text

    def test_mark_task_failed(self, manager):
        """测试标记为失败。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        error_message = "Partner 超时"
        error_details = {"partner": "beijing_food", "timeout_ms": 30000}

        task = manager.mark_task_failed(
            task_id="task-001",
            error_message=error_message,
            error_details=error_details,
        )

        assert task.status == TaskExecutionStatus.FAILED
        assert task.completed_at is not None
        assert task.error_message == error_message
        assert task.error_details == error_details

    def test_mark_task_cancelled(self, manager):
        """测试标记为已取消。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        task = manager.mark_task_cancelled("task-001")

        assert task.status == TaskExecutionStatus.CANCELLED
        assert task.completed_at is not None

    def test_mark_nonexistent_task(self, manager):
        """测试更新不存在的任务。"""
        task = manager.mark_task_running("nonexistent")
        assert task is None


class TestProgressUpdates:
    """进度更新测试。"""

    def test_update_task_progress(self, manager):
        """测试更新任务进度。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        task = manager.update_task_progress(
            task_id="task-001",
            phase=TaskExecutionPhase.EXECUTION_POLLING,
            total_partners=3,
            completed_partners=1,
        )

        assert task.progress is not None
        assert task.progress.current_phase == TaskExecutionPhase.EXECUTION_POLLING
        assert task.progress.total_partners == 3
        assert task.progress.completed_partners == 1

    def test_progress_percent_calculation(self, manager):
        """测试进度百分比计算。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.update_task_progress(task_id="task-001", total_partners=4)

        task = manager.update_task_progress(
            task_id="task-001",
            completed_partners=2,
            failed_partners=1,
        )

        # (2+1)/4 = 75%
        assert task.progress.progress_percent == 75.0


class TestTaskDeletion:
    """任务删除测试。"""

    def test_delete_task(self, manager):
        """测试删除任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")

        result = manager.delete_task("task-001")
        assert result is True

        task = manager.get_task("task-001")
        assert task is None

    def test_delete_nonexistent_task(self, manager):
        """测试删除不存在的任务。"""
        result = manager.delete_task("nonexistent")
        assert result is False

    def test_delete_updates_session_index(self, manager):
        """测试删除时更新会话索引。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")

        manager.delete_task("task-001")

        tasks = manager.get_tasks_by_session("session-001")
        assert len(tasks) == 1
        assert tasks[0].task_id == "task-002"


class TestCleanup:
    """过期清理测试。"""

    def test_cleanup_expired_tasks(self, manager):
        """测试清理过期任务。"""
        # 创建并完成任务
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.mark_task_completed(task_id="task-001")

        # 手动设置完成时间为过去
        task = manager.get_task("task-001")
        past_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        task.completed_at = past_time

        # 清理
        count = manager.cleanup_expired_tasks()

        assert count == 1
        assert manager.get_task("task-001") is None

    def test_cleanup_keeps_recent_tasks(self, manager):
        """测试保留未过期任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.mark_task_completed(task_id="task-001")

        # 清理（刚完成的任务不应被清理）
        count = manager.cleanup_expired_tasks()

        assert count == 0
        assert manager.get_task("task-001") is not None

    def test_cleanup_keeps_running_tasks(self, manager):
        """测试保留运行中的任务。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.mark_task_running("task-001")

        count = manager.cleanup_expired_tasks()

        assert count == 0
        assert manager.get_task("task-001") is not None


class TestStats:
    """统计信息测试。"""

    def test_get_stats(self, manager):
        """测试获取统计信息。"""
        manager.create_task(task_id="task-001", session_id="session-001")
        manager.create_task(task_id="task-002", session_id="session-001")
        manager.create_task(task_id="task-003", session_id="session-002")

        manager.mark_task_running("task-001")
        manager.mark_task_completed(task_id="task-002")

        stats = manager.get_stats()

        assert stats["total_tasks"] == 3
        assert stats["total_sessions"] == 2
        assert stats["pending"] == 1
        assert stats["running"] == 1
        assert stats["completed"] == 1


class TestLifecycle:
    """生命周期测试。"""

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """测试启动和停止。"""
        await manager.start()
        assert manager._cleanup_task is not None

        await manager.stop()
        assert manager._cleanup_task is None


class TestSingleton:
    """单例模式测试。"""

    def test_get_task_execution_manager(self):
        """测试获取单例。"""
        manager1 = get_task_execution_manager()
        manager2 = get_task_execution_manager()

        assert manager1 is manager2

    def test_reset_singleton(self):
        """测试重置单例。"""
        manager1 = get_task_execution_manager()
        reset_task_execution_manager()
        manager2 = get_task_execution_manager()

        assert manager1 is not manager2


class TestConcurrency:
    """并发安全测试。"""

    def test_concurrent_create(self, manager):
        """测试并发创建任务。"""
        import threading

        errors = []

        def create_task(task_id):
            try:
                manager.create_task(
                    task_id=task_id,
                    session_id="session-001",
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_task, args=(f"task-{i}",)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manager.get_tasks_by_session("session-001")) == 10
