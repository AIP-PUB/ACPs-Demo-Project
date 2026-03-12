"""
Leader Agent Platform - /submit 校验功能集成测试

本模块测试校验功能：
1. 幂等性保护（clientRequestId 去重）- 409003
2. activeTaskId 乐观并发校验 - 409002
3. mode 一致性校验 - 409001
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
from unittest.mock import AsyncMock, MagicMock, patch

# 使用与 conftest.py 一致的模块路径
from assistant.models.exceptions import (
    DuplicateRequestError,
    ActiveTaskMismatchError,
    ModeMismatchError,
)
from assistant.models import ExecutionMode


class TestIdempotencyProtection:
    """测试幂等性保护功能（409003）。"""

    @pytest.mark.asyncio
    async def test_first_request_should_succeed(self, orchestrator, session_manager):
        """首次请求应成功处理。"""
        from assistant.api.schemas import SubmitRequest

        request = SubmitRequest(
            query="你好",
            client_request_id="req-001",
            mode=ExecutionMode.DIRECT_RPC,
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            from assistant.models import IntentType, IntentDecision

            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！有什么可以帮助你的？",
            )

            response = await orchestrator.handle_submit(request)
            assert response.result is not None
            assert response.result.session_id is not None

    @pytest.mark.asyncio
    async def test_duplicate_request_with_same_payload_should_succeed(
        self, orchestrator, session_manager
    ):
        """相同载荷的重复请求应成功（幂等重试）。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 第一次请求
        request1 = SubmitRequest(
            query="你好",
            client_request_id="req-002",
            mode=ExecutionMode.DIRECT_RPC,
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id

        # 第二次请求（相同 session_id 和 client_request_id，相同载荷）
        request2 = SubmitRequest(
            session_id=session_id,
            query="你好",
            client_request_id="req-002",
            mode=ExecutionMode.DIRECT_RPC,
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response2 = await orchestrator.handle_submit(request2)
            assert response2.result is not None

    @pytest.mark.asyncio
    async def test_duplicate_request_with_different_payload_should_fail(
        self, orchestrator, session_manager
    ):
        """不同载荷的重复请求应返回 409003。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 第一次请求
        request1 = SubmitRequest(
            query="你好",
            client_request_id="req-003",
            mode=ExecutionMode.DIRECT_RPC,
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id

        # 第二次请求（相同 client_request_id，但不同 query）
        request2 = SubmitRequest(
            session_id=session_id,
            query="再见",  # 不同的 query
            client_request_id="req-003",
            mode=ExecutionMode.DIRECT_RPC,
        )

        with pytest.raises(DuplicateRequestError) as exc_info:
            await orchestrator.handle_submit(request2)

        assert exc_info.value.code == 409003
        assert "client_request_id" in str(exc_info.value.details)


class TestActiveTaskIdValidation:
    """测试 activeTaskId 乐观并发校验功能（409002）。"""

    @pytest.mark.asyncio
    async def test_correct_active_task_id_should_succeed(
        self, orchestrator, session_manager
    ):
        """正确的 activeTaskId 应成功。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 创建 session（CHIT_CHAT 不创建 active_task）
        request1 = SubmitRequest(
            query="你好",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-task-001",
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id
            # CHIT_CHAT 不创建真实的 active_task，所以不应该传 active_task_id

        # 后续请求不带 activeTaskId（因为 CHIT_CHAT 没有创建活跃任务）
        request2 = SubmitRequest(
            session_id=session_id,
            query="继续",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-task-002",
            # active_task_id=None，不传 active_task_id
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="继续聊天",
            )
            response2 = await orchestrator.handle_submit(request2)
            assert response2.result is not None

    @pytest.mark.asyncio
    async def test_wrong_active_task_id_should_fail(
        self, orchestrator, session_manager
    ):
        """错误的 activeTaskId 应返回 409002。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 创建 session
        request1 = SubmitRequest(
            query="你好",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-wrong-001",
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id

        # 后续请求带错误的 activeTaskId
        request2 = SubmitRequest(
            session_id=session_id,
            query="继续",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-wrong-002",
            active_task_id="wrong-task-id-12345",  # 错误的 task_id
        )

        with pytest.raises(ActiveTaskMismatchError) as exc_info:
            await orchestrator.handle_submit(request2)

        assert exc_info.value.code == 409002
        assert "expected_task_id" in str(exc_info.value.details)


class TestModeConsistencyValidation:
    """测试 mode 一致性校验功能（409001）。"""

    @pytest.mark.asyncio
    async def test_consistent_mode_should_succeed(self, orchestrator, session_manager):
        """一致的 mode 应成功。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 第一次请求（创建 session，mode=direct_rpc）
        request1 = SubmitRequest(
            query="你好",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-mode-001",
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id

        # 第二次请求（相同 mode）
        request2 = SubmitRequest(
            session_id=session_id,
            query="继续",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-mode-002",
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="继续聊天",
            )
            response2 = await orchestrator.handle_submit(request2)
            assert response2.result is not None

    @pytest.mark.asyncio
    async def test_inconsistent_mode_should_fail(self, orchestrator, session_manager):
        """不一致的 mode 应返回 409001。"""
        from assistant.api.schemas import SubmitRequest
        from assistant.models import IntentType, IntentDecision

        # 第一次请求（创建 session，mode=direct_rpc）
        request1 = SubmitRequest(
            query="你好",
            mode=ExecutionMode.DIRECT_RPC,
            client_request_id="req-inconsistent-001",
        )

        with patch.object(
            orchestrator._intent_analyzer,
            "analyze",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = IntentDecision(
                intent_type=IntentType.CHIT_CHAT,
                response_guide="你好！",
            )
            response1 = await orchestrator.handle_submit(request1)
            session_id = response1.result.session_id

        # 第二次请求（不同 mode）
        request2 = SubmitRequest(
            session_id=session_id,
            query="继续",
            mode=ExecutionMode.GROUP,  # 不同的 mode
            client_request_id="req-inconsistent-002",
        )

        with pytest.raises(ModeMismatchError) as exc_info:
            await orchestrator.handle_submit(request2)

        assert exc_info.value.code == 409001
        assert "expected_mode" in str(exc_info.value.details)
        assert "direct_rpc" in str(exc_info.value.details)


class TestSessionManagerRequestCache:
    """测试 SessionManager 的请求缓存功能。"""

    def _create_base_scenario(self):
        """创建基础场景。"""
        from assistant.models import ScenarioRuntime
        from assistant.models.base import now_iso

        return ScenarioRuntime(
            id="base",
            kind="base",
            version="1.0.0",
            loaded_at=now_iso(),
        )

    def test_check_idempotency_new_request(self, session_manager):
        """新请求应返回 (False, None)。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        is_duplicate, cached_hash = session_manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )
        assert is_duplicate is False
        assert cached_hash is None

    def test_cache_and_check_idempotency(self, session_manager):
        """缓存后再次检查应返回缓存的哈希。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        # 缓存请求
        result = session_manager.cache_request(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )
        assert result is True

        # 检查幂等性
        is_duplicate, cached_hash = session_manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )
        assert is_duplicate is True
        assert cached_hash == "hash-001"

    def test_check_idempotency_different_hash(self, session_manager):
        """相同 client_request_id 但不同哈希应能检测。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )

        # 缓存请求
        session_manager.cache_request(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-001",
        )

        # 检查幂等性（不同哈希）
        is_duplicate, cached_hash = session_manager.check_request_idempotency(
            session_id=session.session_id,
            client_request_id="req-001",
            request_hash="hash-002",  # 不同的哈希
        )
        assert is_duplicate is True
        assert cached_hash == "hash-001"  # 返回缓存的哈希


class TestSessionMode:
    """测试 Session 的 mode 字段。"""

    def _create_base_scenario(self):
        """创建基础场景。"""
        from assistant.models import ScenarioRuntime
        from assistant.models.base import now_iso

        return ScenarioRuntime(
            id="base",
            kind="base",
            version="1.0.0",
            loaded_at=now_iso(),
        )

    def test_create_session_with_mode(self, session_manager):
        """创建 Session 时应保存 mode。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.DIRECT_RPC,
            base_scenario=base_scenario,
        )
        assert session.mode == ExecutionMode.DIRECT_RPC

    def test_create_session_with_group_mode(self, session_manager):
        """创建 Session 时指定 GROUP 模式应保存。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.GROUP,
            base_scenario=base_scenario,
        )
        assert session.mode == ExecutionMode.GROUP

    def test_session_mode_persists(self, session_manager):
        """Session mode 应持久化。"""
        base_scenario = self._create_base_scenario()
        session = session_manager.create_session(
            mode=ExecutionMode.GROUP,
            base_scenario=base_scenario,
        )
        session_id = session.session_id

        # 重新获取 session
        retrieved = session_manager.get_session(session_id)
        assert retrieved.mode == ExecutionMode.GROUP
