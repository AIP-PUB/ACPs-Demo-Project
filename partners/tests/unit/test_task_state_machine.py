"""
任务状态机单元测试

测试 AIP 协议规定的任务状态流转：
- Accepted -> Working -> AwaitingInput -> Working -> AwaitingCompletion -> Completed
- 各种终态转换：Rejected, Canceled, Failed
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from acps_sdk.aip.aip_base_model import TaskState, TaskCommandType


BEIJING_TZ = timezone(timedelta(hours=8))


class TestTaskStateTransitions:
    """测试任务状态转换"""

    @pytest.mark.asyncio
    async def test_start_creates_accepted_state(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Start 命令创建 Accepted 状态的任务"""
        from partners.generic_runner import GenericRunner

        mock_client = mock_openai_client_factory(mock_llm_responses.decision_accept())

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("测试请求")
            task = await runner.on_start(message, None)

            # 初始状态应为 Accepted
            assert task.status.state == TaskState.Accepted

            # 状态历史应包含 Accepted
            assert len(task.statusHistory) > 0
            assert task.statusHistory[0].state == TaskState.Accepted

    @pytest.mark.asyncio
    async def test_accepted_to_working_transition(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Accepted -> Working 状态转换"""
        from partners.generic_runner import GenericRunner

        # 使用慢响应观察状态变化
        async def slow_decision(*args, **kwargs):
            await asyncio.sleep(0.1)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = mock_llm_responses.decision_accept()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = slow_decision

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("测试请求")
            task = await runner.on_start(message, None)

            # 初始为 Accepted
            assert task.status.state == TaskState.Accepted

            # 等待 Decision 完成
            await asyncio.sleep(0.3)

            ctx = runner.tasks.get(message.taskId)
            # Decision accept 后应进入 Working
            assert ctx.task.status.state == TaskState.Working

    @pytest.mark.asyncio
    async def test_working_to_awaiting_input_transition(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Working -> AwaitingInput 状态转换（信息缺失时）"""
        from partners.generic_runner import GenericRunner

        responses = [
            mock_llm_responses.decision_accept(),
            mock_llm_responses.analysis_missing_fields(["天数", "预算"]),
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            idx = min(call_count[0], len(responses) - 1)
            mock_resp.choices[0].message.content = responses[idx]
            call_count[0] += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=get_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("不完整请求")
            await runner.on_start(message, None)

            await asyncio.sleep(0.5)

            ctx = runner.tasks.get(message.taskId)
            assert ctx.task.status.state == TaskState.AwaitingInput

            # 验证状态历史
            states = [s.state for s in ctx.task.statusHistory]
            assert TaskState.Accepted in states
            assert TaskState.Working in states
            assert TaskState.AwaitingInput in states

    @pytest.mark.asyncio
    async def test_working_to_awaiting_completion_transition(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Working -> AwaitingCompletion 状态转换（生产完成时）"""
        from partners.generic_runner import GenericRunner

        responses = [
            mock_llm_responses.decision_accept(),
            mock_llm_responses.analysis_complete(),
            mock_llm_responses.production_output(),
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            idx = min(call_count[0], len(responses) - 1)
            mock_resp.choices[0].message.content = responses[idx]
            call_count[0] += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=get_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("完整请求")
            await runner.on_start(message, None)

            await asyncio.sleep(1.0)

            ctx = runner.tasks.get(message.taskId)
            assert ctx.task.status.state == TaskState.AwaitingCompletion

    @pytest.mark.asyncio
    async def test_awaiting_completion_to_completed_transition(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：AwaitingCompletion -> Completed 状态转换"""
        from partners.generic_runner import GenericRunner

        responses = [
            mock_llm_responses.decision_accept(),
            mock_llm_responses.analysis_complete(),
            mock_llm_responses.production_output(),
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            idx = min(call_count[0], len(responses) - 1)
            mock_resp.choices[0].message.content = responses[idx]
            call_count[0] += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=get_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("完整请求")
            await runner.on_start(message, None)
            task_id = message.taskId

            await asyncio.sleep(1.0)

            ctx = runner.tasks.get(task_id)
            assert ctx.task.status.state == TaskState.AwaitingCompletion

            # 发送 Complete 命令
            complete_message = message_factory(
                "确认完成", TaskCommandType.Complete, task_id
            )
            result = await runner.on_complete(complete_message, ctx.task)

            assert result.status.state == TaskState.Completed


class TestTerminalStates:
    """测试终态处理"""

    @pytest.mark.asyncio
    async def test_rejected_is_terminal(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Rejected 是终态，不能继续处理"""
        from partners.generic_runner import GenericRunner

        mock_client = mock_openai_client_factory(
            mock_llm_responses.decision_reject("超出服务范围")
        )

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("超出范围的请求")
            await runner.on_start(message, None)
            task_id = message.taskId

            await asyncio.sleep(0.5)

            ctx = runner.tasks.get(task_id)
            assert ctx.task.status.state == TaskState.Rejected

            # 尝试发送 Continue 命令
            continue_message = message_factory(
                "补充信息", TaskCommandType.Continue, task_id
            )
            result = await runner.on_continue(continue_message, ctx.task)

            # 状态应保持 Rejected
            assert result.status.state == TaskState.Rejected

    @pytest.mark.asyncio
    async def test_canceled_is_terminal(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Canceled 是终态"""
        from partners.generic_runner import GenericRunner

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = mock_llm_responses.decision_accept()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = slow_response

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("测试请求")
            await runner.on_start(message, None)
            task_id = message.taskId

            await asyncio.sleep(0.1)

            # 取消任务
            cancel_message = message_factory("", TaskCommandType.Cancel, task_id)
            ctx = runner.tasks.get(task_id)
            result = await runner.on_cancel(cancel_message, ctx.task)

            assert result.status.state == TaskState.Canceled

            # 再次取消应保持 Canceled
            result = await runner.on_cancel(cancel_message, result)
            assert result.status.state == TaskState.Canceled

    @pytest.mark.asyncio
    async def test_failed_is_terminal(self, test_agent_dir, message_factory):
        """测试：Failed 是终态"""
        from partners.generic_runner import GenericRunner

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM Error")
        )

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("测试请求")
            await runner.on_start(message, None)
            task_id = message.taskId

            await asyncio.sleep(0.5)

            ctx = runner.tasks.get(task_id)
            assert ctx.task.status.state == TaskState.Failed

    @pytest.mark.asyncio
    async def test_completed_is_terminal(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Completed 是终态"""
        from partners.generic_runner import GenericRunner

        responses = [
            mock_llm_responses.decision_accept(),
            mock_llm_responses.analysis_complete(),
            mock_llm_responses.production_output(),
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            idx = min(call_count[0], len(responses) - 1)
            mock_resp.choices[0].message.content = responses[idx]
            call_count[0] += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=get_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            message = message_factory("完整请求")
            await runner.on_start(message, None)
            task_id = message.taskId

            await asyncio.sleep(1.0)

            ctx = runner.tasks.get(task_id)
            complete_message = message_factory(
                "确认", TaskCommandType.Complete, task_id
            )
            result = await runner.on_complete(complete_message, ctx.task)

            assert result.status.state == TaskState.Completed

            # 再次 Complete 应保持 Completed
            result = await runner.on_complete(complete_message, result)
            assert result.status.state == TaskState.Completed


class TestCommandHistory:
    """测试命令历史记录"""

    @pytest.mark.asyncio
    async def test_start_command_in_history(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Start 命令被记录到历史"""
        from partners.generic_runner import GenericRunner

        mock_client = mock_openai_client_factory(mock_llm_responses.decision_accept())

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            command = message_factory("测试请求")
            task = await runner.on_start(command, None)

            assert task.commandHistory is not None
            assert len(task.commandHistory) == 1
            assert task.commandHistory[0].id == command.id

    @pytest.mark.asyncio
    async def test_continue_command_appended(
        self,
        test_agent_dir,
        mock_openai_client_factory,
        message_factory,
        mock_llm_responses,
    ):
        """测试：Continue 命令被追加到历史"""
        from partners.generic_runner import GenericRunner

        responses = [
            mock_llm_responses.decision_accept(),
            mock_llm_responses.analysis_missing_fields(["天数"]),
            mock_llm_responses.analysis_complete(),
            mock_llm_responses.production_output(),
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            idx = min(call_count[0], len(responses) - 1)
            mock_resp.choices[0].message.content = responses[idx]
            call_count[0] += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=get_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            command = message_factory("不完整请求")
            await runner.on_start(command, None)
            task_id = command.taskId

            await asyncio.sleep(0.5)

            # 发送 Continue
            continue_command = message_factory(
                "两天", TaskCommandType.Continue, task_id
            )
            ctx = runner.tasks.get(task_id)
            await runner.on_continue(continue_command, ctx.task)

            # 验证命令历史
            ctx = runner.tasks.get(task_id)
            assert len(ctx.task.commandHistory) == 2
            assert ctx.task.commandHistory[1].id == continue_command.id
