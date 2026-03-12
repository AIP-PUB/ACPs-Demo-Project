"""
Leader Agent Platform - SessionManager 单元测试

测试 SessionManager 的核心功能：
1. Session 创建与存储
2. Session 获取与更新
3. TTL 过期检测
4. 容量管理（LRU 淘汰）
5. 对话历史管理
6. 事件日志管理
7. 幂等性缓存
"""

import sys
import os
from pathlib import Path

_current_dir = Path(__file__).parent
_leader_dir = _current_dir.parent.parent
_project_root = _leader_dir.parent

if str(_leader_dir) not in sys.path:
    sys.path.insert(0, str(_leader_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from assistant.core.session_manager import SessionManager
from assistant.models import (
    ExecutionMode,
    ScenarioRuntime,
    Session,
    DialogContext,
    UserResult,
    UserResultType,
    IntentType,
    ResponseType,
    EventLogType,
)
from assistant.models.base import now_iso


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_scenario():
    """创建基础场景。"""
    return ScenarioRuntime(
        id="base",
        kind="base",
        version="1.0.0",
        loaded_at=now_iso(),
    )


@pytest.fixture
def manager():
    """创建 SessionManager 实例。"""
    return SessionManager(ttl_minutes=60)


# =============================================================================
# 测试 Session 创建
# =============================================================================


class TestSessionCreation:
    """测试 Session 创建功能。"""

    def test_create_session_basic(self, manager, base_scenario):
        """测试基本的 Session 创建。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert session is not None
        assert session.session_id.startswith("sess_")
        assert session.mode == ExecutionMode.DIRECT_RPC

    def test_create_session_with_group_mode(self, manager, base_scenario):
        """测试带 group mode 的 Session 创建。"""
        session = manager.create_session(
            mode=ExecutionMode.GROUP,
            base_scenario=base_scenario,
        )

        assert session.mode == ExecutionMode.GROUP

    def test_session_has_base_scenario(self, manager, base_scenario):
        """测试新 Session 有基础场景。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert session.base_scenario is not None
        assert session.base_scenario.id == "base"

    def test_session_has_user_result(self, manager, base_scenario):
        """测试新 Session 有 user_result。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert session.user_result is not None
        assert session.user_result.type == UserResultType.PENDING

    def test_session_has_empty_event_log(self, manager, base_scenario):
        """测试新 Session 有空的事件日志。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert len(session.event_log) == 0


# =============================================================================
# 测试 Session 获取
# =============================================================================


class TestSessionRetrieval:
    """测试 Session 获取功能。"""

    def test_get_existing_session(self, manager, base_scenario):
        """测试获取存在的 Session。"""
        created = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        retrieved = manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self, manager):
        """测试获取不存在的 Session。"""
        retrieved = manager.get_session("nonexistent-session-id")

        assert retrieved is None

    def test_get_expired_session_returns_none(self, base_scenario):
        """测试获取过期的 Session 返回 None。"""
        manager = SessionManager(ttl_minutes=1)  # 1 分钟 TTL
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        # 直接修改内部存储中的 touched_at 为过去时间
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        manager._sessions[session.session_id].touched_at = past_time

        retrieved = manager.get_session(session.session_id)

        assert retrieved is None


# =============================================================================
# 测试 Session 更新
# =============================================================================


class TestSessionUpdate:
    """测试 Session 更新功能。"""

    def test_update_session_refreshes_touched_at(self, manager, base_scenario):
        """测试更新 Session 会刷新 touched_at。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        original_time = session.touched_at

        import time

        time.sleep(0.01)

        manager.update_session(session)

        retrieved = manager.get_session(session.session_id)
        assert retrieved.touched_at >= original_time

    def test_update_nonexistent_session(self, base_scenario):
        """测试更新不存在的 Session 不会报错。"""
        manager = SessionManager(ttl_minutes=60)

        now = now_iso()
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        fake_session = Session(
            session_id="fake-session",
            mode=ExecutionMode.DIRECT_RPC,
            created_at=now,
            updated_at=now,
            touched_at=now,
            ttl_seconds=3600,
            expires_at=expires_at,
            base_scenario=base_scenario,
            user_result=UserResult(
                type=UserResultType.PENDING,
                data_items=[],
                updated_at=now,
            ),
        )

        # 不应抛出异常
        manager.update_session(fake_session)


# =============================================================================
# 测试 Session 删除
# =============================================================================


class TestSessionDeletion:
    """测试 Session 删除功能。"""

    def test_delete_existing_session(self, manager, base_scenario):
        """测试删除存在的 Session。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        result = manager.delete_session(session.session_id)

        assert result is True
        assert manager.get_session(session.session_id) is None

    def test_delete_nonexistent_session(self, manager):
        """测试删除不存在的 Session。"""
        result = manager.delete_session("nonexistent-session-id")

        assert result is False


# =============================================================================
# 测试容量管理
# =============================================================================


class TestCapacityManagement:
    """测试容量管理功能。"""

    def test_lru_eviction_when_full(self, base_scenario):
        """测试达到容量上限时的 LRU 淘汰。"""
        manager = SessionManager(ttl_minutes=60, max_sessions=3)

        # 创建 3 个 session
        session1 = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        session2 = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        session3 = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert manager.get_session_count() == 3

        # 创建第 4 个，应该淘汰最旧的
        session4 = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        assert manager.get_session_count() == 3
        assert manager.get_session(session1.session_id) is None  # 被淘汰
        assert manager.get_session(session4.session_id) is not None


# =============================================================================
# 测试对话历史
# =============================================================================


class TestDialogHistory:
    """测试对话历史管理功能。"""

    def test_add_dialog_turn(self, manager, base_scenario):
        """测试添加对话轮次。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        result = manager.add_dialog_turn(
            session_id=session.session_id,
            user_query="你好",
            intent_type=IntentType.CHIT_CHAT,
            response_type=ResponseType.CHAT,
        )

        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert retrieved.dialog_context is not None
        assert len(retrieved.dialog_context.recent_turns) == 1
        assert retrieved.dialog_context.recent_turns[0].user_query == "你好"

    def test_add_dialog_turn_with_response_summary(self, manager, base_scenario):
        """测试添加带响应摘要的对话轮次。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        manager.add_dialog_turn(
            session_id=session.session_id,
            user_query="帮我规划北京之旅",
            intent_type=IntentType.TASK_NEW,
            response_type=ResponseType.PENDING,
            response_summary="开始规划旅程",
        )

        retrieved = manager.get_session(session.session_id)
        assert (
            retrieved.dialog_context.recent_turns[0].response_summary == "开始规划旅程"
        )

    def test_add_dialog_turn_to_nonexistent_session(self, manager):
        """测试向不存在的 Session 添加对话轮次。"""
        result = manager.add_dialog_turn(
            session_id="nonexistent",
            user_query="你好",
            intent_type=IntentType.CHIT_CHAT,
            response_type=ResponseType.CHAT,
        )

        assert result is False


# =============================================================================
# 测试事件日志
# =============================================================================


class TestEventLog:
    """测试事件日志功能。"""

    def test_add_event_log(self, manager, base_scenario):
        """测试添加事件日志。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        result = manager.add_event_log(
            session_id=session.session_id,
            event_type=EventLogType.USER_SUBMIT,
            payload={"query": "你好"},
        )

        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.event_log) == 1
        assert retrieved.event_log[0].type == EventLogType.USER_SUBMIT

    def test_add_event_log_to_nonexistent_session(self, manager):
        """测试向不存在的 Session 添加事件日志。"""
        result = manager.add_event_log(
            session_id="nonexistent",
            event_type=EventLogType.USER_SUBMIT,
            payload={"query": "你好"},
        )

        assert result is False


# =============================================================================
# 测试幂等性缓存
# =============================================================================


class TestIdempotencyCache:
    """测试幂等性缓存功能。"""

    def test_cache_request(self, manager, base_scenario):
        """测试缓存请求。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        result = manager.cache_request(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        assert result is True

    def test_check_idempotency_new_request(self, manager, base_scenario):
        """测试检查新请求的幂等性。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        is_dup, cached = manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        assert is_dup is False
        assert cached is None

    def test_check_idempotency_duplicate_request(self, manager, base_scenario):
        """测试检查重复请求的幂等性。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        # 先缓存
        manager.cache_request(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        # 再检查
        is_dup, cached = manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        assert is_dup is True
        assert cached == "hash-001"

    def test_check_idempotency_different_hash(self, manager, base_scenario):
        """测试检查不同哈希的重复请求。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        # 先缓存
        manager.cache_request(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        # 用不同的哈希检查
        is_dup, cached = manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-002",
        )

        assert is_dup is True
        assert cached == "hash-001"  # 返回原始缓存的哈希


# =============================================================================
# 测试 touch_session
# =============================================================================


class TestTouchSession:
    """测试 touch_session 功能。"""

    def test_touch_session(self, manager, base_scenario):
        """测试 touch session 刷新时间。"""
        session = manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        import time

        time.sleep(0.01)

        result = manager.touch_session(session.session_id)

        assert result is True

    def test_touch_nonexistent_session(self, manager):
        """测试 touch 不存在的 session。"""
        result = manager.touch_session("nonexistent")

        assert result is False


# =============================================================================
# 测试列出 Session
# =============================================================================


class TestListSessions:
    """测试列出 Session 功能。"""

    def test_list_sessions(self, manager, base_scenario):
        """测试列出所有 Session。"""
        for _ in range(3):
            manager.create_session(
                mode=ExecutionMode.DIRECT_RPC,
                base_scenario=base_scenario,
            )

        sessions = manager.list_sessions()

        assert len(sessions) == 3

    def test_list_sessions_with_limit(self, manager, base_scenario):
        """测试限制返回数量。"""
        for _ in range(5):
            manager.create_session(
                mode=ExecutionMode.DIRECT_RPC,
                base_scenario=base_scenario,
            )

        sessions = manager.list_sessions(limit=2)

        assert len(sessions) == 2

    def test_list_sessions_with_filter(self, manager, base_scenario):
        """测试过滤 Session。"""
        manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        manager.create_session(
            mode=ExecutionMode.GROUP,
            base_scenario=base_scenario,
        )
        manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        sessions = manager.list_sessions(
            filter_fn=lambda s: s.mode == ExecutionMode.DIRECT_RPC
        )

        assert len(sessions) == 2
