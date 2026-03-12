"""
Partner 端群组模式处理器

本模块提供 Partner 端群组模式的支持：
1. 管理群组连接生命周期（加入/退出）
2. 接收群组任务命令并分发给 GenericRunner 处理
3. 发送任务状态更新到群组

群组通信流程：
- Leader 通过 RPC 发送群组邀请 (RabbitMQRequest)
- Partner 接受邀请，连接 RabbitMQ，加入群组
- Leader 通过群组广播发送任务命令
- Partner 通过群组广播发送任务状态更新
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, TYPE_CHECKING

from acps_sdk.aip.aip_base_model import (
    TaskCommand,
    TaskResult,
    TaskState,
    TaskStatus,
    TaskCommandType,
    TextDataItem,
    Product,
)
from acps_sdk.aip.aip_group_model import (
    RabbitMQRequest,
    RabbitMQResponse,
    GroupMgmtCommand,
)
from acps_sdk.aip.aip_group_partner import (
    GroupPartnerMqClient,
    PartnerGroupState,
)
from acps_sdk.aip.aip_rpc_model import RpcRequest, RpcResponse, JSONRPCError

if TYPE_CHECKING:
    from partners.generic_runner import GenericRunner


logger = logging.getLogger("partners.group_handler")


class GroupHandler:
    """
    群组模式处理器

    管理 Partner 的群组连接和消息处理。
    每个 Partner Agent 可以同时参与多个群组（一个群组对应一个 session）。
    """

    def __init__(self, agent_name: str, runner: "GenericRunner"):
        """
        初始化群组处理器

        Args:
            agent_name: Partner 名称（用于构造 AIC）
            runner: 对应的 GenericRunner 实例（用于处理任务）
        """
        self.agent_name = agent_name
        self.runner = runner

        # AIC 标识（从 runner 的 ACS 获取或构造）
        acs = runner.acs
        self.partner_aic = acs.get("aic") or f"agent.{agent_name}"

        # 群组客户端缓存
        # group_id -> GroupPartnerMqClient
        self._group_clients: Dict[str, GroupPartnerMqClient] = {}

        # 任务到群组的映射
        # task_id -> group_id
        self._task_group_map: Dict[str, str] = {}

        # 设置状态变化回调，用于广播状态更新到群组
        self.runner.set_state_change_callback(self._on_runner_state_change)

        short_aic = (
            self.partner_aic[-12:] if len(self.partner_aic) > 12 else self.partner_aic
        )
        logger.info(
            "[GroupHandler:%s] Initialized: aic=...%s",
            agent_name,
            short_aic,
        )

    async def _on_runner_state_change(self, task_result: TaskResult) -> None:
        """
        GenericRunner 状态变化回调

        当任务状态变化时，广播到对应的群组
        """
        task_id = task_result.taskId
        if not task_id:
            return

        short_task = task_id[-12:] if len(task_id) > 12 else task_id
        state_name = (
            task_result.status.state.name
            if task_result.status and hasattr(task_result.status.state, "name")
            else str(task_result.status.state) if task_result.status else "unknown"
        )

        # 查找任务对应的群组
        group_id = self._task_group_map.get(task_id)
        if not group_id:
            logger.debug(
                "[GroupHandler:%s] State change for task ...%s but no group mapping, skipping",
                self.agent_name,
                short_task,
            )
            return

        short_group = group_id[-8:] if len(group_id) > 8 else group_id
        logger.info(
            "[GroupHandler:%s] State change callback: task=...%s state=%s -> broadcasting to group ...%s",
            self.agent_name,
            short_task,
            state_name,
            short_group,
        )

        await self._broadcast_task_update(task_result, group_id)

    @property
    def active_groups(self) -> Dict[str, GroupPartnerMqClient]:
        """获取活跃的群组连接"""
        return {
            gid: client
            for gid, client in self._group_clients.items()
            if client.is_joined
        }

    async def handle_group_rpc(self, request: RabbitMQRequest) -> RabbitMQResponse:
        """
        处理群组相关的 RPC 请求（joinGroup）

        RabbitMQRequest 的 method 字段固定为 "group"，
        实际代表 joinGroup 请求。

        Args:
            request: RabbitMQ 请求

        Returns:
            RabbitMQ 响应
        """
        method = request.method
        request_id = request.id or "unknown"
        logger.info(
            "[GroupHandler:%s] <<< Group RPC request: method=%s, id=%s",
            self.agent_name,
            method,
            request_id,
        )

        # RabbitMQRequest.method 固定为 "group"，代表 joinGroup 请求
        if method == "group":
            return await self._handle_join_group(request)
        else:
            logger.warning(
                "[GroupHandler:%s] Unknown method: %s",
                self.agent_name,
                method,
            )
            from acps_sdk.aip.aip_group_model import RabbitMQResponseError

            return RabbitMQResponse(
                id=request.id,
                error=RabbitMQResponseError(
                    code=-32601,
                    message=f"Method not found: {method}",
                ),
            )

    async def _handle_join_group(self, request: RabbitMQRequest) -> RabbitMQResponse:
        """
        处理群组加入请求

        Args:
            request: RabbitMQ 请求（已经是正确类型，无需转换）

        Returns:
            RabbitMQ 响应
        """
        from acps_sdk.aip.aip_group_model import (
            RabbitMQResponseResult,
            RabbitMQResponseError,
        )

        start_time = asyncio.get_event_loop().time()

        try:
            # request 已经是 RabbitMQRequest 类型，直接使用
            rabbitmq_request = request

            group_id = rabbitmq_request.params.group.groupId
            short_group = group_id[-8:] if len(group_id) > 8 else group_id
            # GroupInfo.leader 是 ACSObject 类型，包含 aic 字段
            leader_aic = (
                rabbitmq_request.params.group.leader.aic
                if rabbitmq_request.params.group.leader
                else "unknown"
            )
            short_leader = leader_aic[-8:] if len(leader_aic) > 8 else leader_aic

            logger.info(
                "[GroupHandler:%s] >>> joinGroup: group_id=...%s, leader=...%s",
                self.agent_name,
                short_group,
                short_leader,
            )

            server_params = rabbitmq_request.params.server
            amqp_params = rabbitmq_request.params.amqp
            logger.debug(
                "[GroupHandler:%s] RabbitMQ config: host=%s:%d, exchange=%s",
                self.agent_name,
                server_params.host,
                server_params.port,
                amqp_params.exchange,
            )

            # 检查是否已经加入该群组
            if group_id in self._group_clients:
                existing_client = self._group_clients[group_id]
                if existing_client.is_joined:
                    logger.warning(
                        "[GroupHandler:%s] Already joined group ...%s, returning existing info",
                        self.agent_name,
                        short_group,
                    )
                    # 返回现有连接信息
                    return RabbitMQResponse(
                        id=request.id,
                        result=RabbitMQResponseResult(
                            connectionName=existing_client._connection_name,
                            vhost=existing_client._vhost,
                            nodeName=existing_client._node_name,
                            queueName=existing_client.queue_name,
                            processId=f"pid-{__import__('os').getpid()}",
                        ),
                    )
                else:
                    # 清理旧客户端
                    logger.debug(
                        "[GroupHandler:%s] Cleaning up disconnected client for group ...%s",
                        self.agent_name,
                        short_group,
                    )
                    del self._group_clients[group_id]

            # 创建新的群组客户端
            logger.debug(
                "[GroupHandler:%s] Creating GroupPartnerMqClient...", self.agent_name
            )
            client = GroupPartnerMqClient(partner_aic=self.partner_aic)

            # 设置命令处理器
            client.set_command_handler(self._on_task_command)
            client.set_task_result_handler(self._on_task_result)
            client.set_mgmt_command_handler(self._on_mgmt_command)

            # 加入群组
            logger.debug(
                "[GroupHandler:%s] Joining group via MQ client...", self.agent_name
            )
            response = await client.join_group(rabbitmq_request)

            if response.error:
                elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.error(
                    "[GroupHandler:%s] <<< joinGroup FAILED: group=...%s, error=%s, elapsed=%.0fms",
                    self.agent_name,
                    short_group,
                    response.error.message,
                    elapsed_ms,
                )
                return RabbitMQResponse(
                    id=request.id,
                    error=RabbitMQResponseError(
                        code=response.error.code,
                        message=response.error.message,
                    ),
                )

            # 保存客户端
            self._group_clients[group_id] = client

            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(
                "[GroupHandler:%s] <<< joinGroup SUCCESS: group=...%s, queue=%s, elapsed=%.0fms",
                self.agent_name,
                short_group,
                client.queue_name,
                elapsed_ms,
            )

            # 返回 RabbitMQ 响应（直接返回，无需转换）
            return RabbitMQResponse(
                id=request.id,
                result=RabbitMQResponseResult(
                    connectionName=response.result.connectionName,
                    vhost=response.result.vhost,
                    nodeName=response.result.nodeName,
                    queueName=response.result.queueName,
                    processId=response.result.processId,
                ),
            )

        except Exception as e:
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(
                "[GroupHandler:%s] <<< joinGroup ERROR: %s, elapsed=%.0fms",
                self.agent_name,
                str(e)[:100],
                elapsed_ms,
                exc_info=True,
            )
            return RabbitMQResponse(
                id=request.id,
                error=RabbitMQResponseError(
                    code=-32603,
                    message=f"Internal error: {str(e)}",
                ),
            )

    async def _on_task_command(self, command: TaskCommand, is_mentioned: bool) -> None:
        """
        处理来自群组的任务命令

        Args:
            command: 任务命令
            is_mentioned: 是否被提及（即是否需要处理）
        """
        short_task = (
            command.taskId[-12:] if len(command.taskId) > 12 else command.taskId
        )
        short_sender = (
            command.senderId[-8:] if len(command.senderId) > 8 else command.senderId
        )

        logger.info(
            "[GroupHandler:%s] <<< TaskCommand: cmd=%s, task=...%s, sender=...%s, mentioned=%s",
            self.agent_name,
            (
                command.command.name
                if hasattr(command.command, "name")
                else command.command
            ),
            short_task,
            short_sender,
            is_mentioned,
        )

        # 如果没有被提及，不处理
        if not is_mentioned:
            logger.debug(
                "[GroupHandler:%s] Not mentioned, skipping command for task ...%s",
                self.agent_name,
                short_task,
            )
            return

        # 记录任务到群组的映射
        group_id = self._find_group_for_sender(command.senderId)
        short_group = group_id[-8:] if group_id and len(group_id) > 8 else group_id
        if group_id:
            self._task_group_map[command.taskId] = group_id
            logger.debug(
                "[GroupHandler:%s] Mapped task ...%s to group ...%s",
                self.agent_name,
                short_task,
                short_group,
            )

        # 获取现有任务（如果存在）
        task_ctx = self.runner.tasks.get(command.taskId)
        task = task_ctx.task if task_ctx else None

        # 将群组命令转换为 GenericRunner 的内部任务处理
        # 这里复用 GenericRunner 的现有逻辑
        start_time = asyncio.get_event_loop().time()

        try:
            cmd_type = command.command

            if cmd_type == TaskCommandType.Start:
                logger.debug(
                    "[GroupHandler:%s] Processing START command...", self.agent_name
                )
                # 创建任务
                task_result = await self.runner.on_start(command, task)
                # 发送状态更新到群组
                await self._broadcast_task_update(task_result, group_id)

            elif cmd_type == TaskCommandType.Continue:
                if not task:
                    logger.warning(
                        "[GroupHandler:%s] Task ...%s not found for CONTINUE",
                        self.agent_name,
                        short_task,
                    )
                    return
                logger.debug(
                    "[GroupHandler:%s] Processing CONTINUE command...", self.agent_name
                )
                # 继续任务
                task_result = await self.runner.on_continue(command, task)
                await self._broadcast_task_update(task_result, group_id)

            elif cmd_type == TaskCommandType.Complete:
                if not task:
                    logger.warning(
                        "[GroupHandler:%s] Task ...%s not found for COMPLETE",
                        self.agent_name,
                        short_task,
                    )
                    return
                logger.debug(
                    "[GroupHandler:%s] Processing COMPLETE command...", self.agent_name
                )
                # 完成任务
                task_result = await self.runner.on_complete(command, task)
                await self._broadcast_task_update(task_result, group_id)

            elif cmd_type == TaskCommandType.Cancel:
                if not task:
                    logger.warning(
                        "[GroupHandler:%s] Task ...%s not found for CANCEL",
                        self.agent_name,
                        short_task,
                    )
                    return
                logger.debug(
                    "[GroupHandler:%s] Processing CANCEL command...", self.agent_name
                )
                # 取消任务
                task_result = await self.runner.on_cancel(command, task)
                await self._broadcast_task_update(task_result, group_id)

            elif cmd_type == TaskCommandType.Get:
                if not task:
                    logger.warning(
                        "[GroupHandler:%s] Task ...%s not found for GET",
                        self.agent_name,
                        short_task,
                    )
                    return
                logger.debug(
                    "[GroupHandler:%s] Processing GET command...", self.agent_name
                )
                # 获取任务状态
                task_result = await self.runner.on_get(command, task)
                await self._broadcast_task_update(task_result, group_id)

            else:
                logger.warning(
                    "[GroupHandler:%s] Unknown command type: %s",
                    self.agent_name,
                    cmd_type,
                )

            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(
                "[GroupHandler:%s] Command processed: task=...%s, elapsed=%.0fms",
                self.agent_name,
                short_task,
                elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(
                "[GroupHandler:%s] Error processing command: task=...%s, error=%s, elapsed=%.0fms",
                self.agent_name,
                short_task,
                str(e)[:100],
                elapsed_ms,
                exc_info=True,
            )

    async def _on_task_result(self, task_result: TaskResult) -> None:
        """
        处理来自其他 Partner 的任务结果

        Args:
            task_result: 任务结果
        """
        # 通常 Partner 不需要处理其他 Partner 的任务结果
        # 但可以用于观察群组内的活动
        short_task = (
            task_result.id[-12:] if len(task_result.id) > 12 else task_result.id
        )
        state_name = (
            task_result.status.state.name
            if task_result.status and hasattr(task_result.status.state, "name")
            else str(task_result.status.state) if task_result.status else "unknown"
        )
        logger.debug(
            "[GroupHandler:%s] Received task result from other partner: task=...%s, state=%s",
            self.agent_name,
            short_task,
            state_name,
        )

    async def _on_mgmt_command(self, mgmt_cmd: GroupMgmtCommand) -> None:
        """
        处理群组管理命令

        Args:
            mgmt_cmd: 管理命令
        """
        short_sender = (
            mgmt_cmd.senderId[-8:] if len(mgmt_cmd.senderId) > 8 else mgmt_cmd.senderId
        )
        logger.info(
            "[GroupHandler:%s] Received mgmt command: cmd=%s, sender=...%s",
            self.agent_name,
            mgmt_cmd.command,
            short_sender,
        )
        # 大部分管理命令由 GroupPartnerMqClient 内部处理
        # 这里可以添加额外的业务逻辑

    async def _broadcast_task_update(
        self,
        task_result: TaskResult,
        group_id: Optional[str] = None,
    ) -> None:
        """
        广播任务状态更新到群组

        Args:
            task_result: 任务结果
            group_id: 群组 ID（如果为 None，尝试从任务映射查找）
        """
        short_task = (
            task_result.id[-12:] if len(task_result.id) > 12 else task_result.id
        )
        state_name = (
            task_result.status.state.name
            if task_result.status and hasattr(task_result.status.state, "name")
            else str(task_result.status.state) if task_result.status else "unknown"
        )

        if not group_id:
            group_id = self._task_group_map.get(task_result.id)

        if not group_id:
            logger.warning(
                "[GroupHandler:%s] Cannot find group for task ...%s, skipping broadcast",
                self.agent_name,
                short_task,
            )
            return

        short_group = group_id[-8:] if len(group_id) > 8 else group_id
        client = self._group_clients.get(group_id)
        if not client or not client.is_joined:
            logger.warning(
                "[GroupHandler:%s] Not connected to group ...%s, skipping broadcast",
                self.agent_name,
                short_group,
            )
            return

        try:
            logger.debug(
                "[GroupHandler:%s] >>> Broadcasting task update: task=...%s, state=%s, group=...%s",
                self.agent_name,
                short_task,
                state_name,
                short_group,
            )

            # 从 task_result 中提取必要的字段
            task_id = task_result.taskId or task_result.id
            session_id = task_result.sessionId or group_id.replace(
                "group-", ""
            )  # 从 group_id 推断
            state = (
                task_result.status.state if task_result.status else TaskState.Unknown
            )
            products = task_result.products
            status_data_items = (
                task_result.status.dataItems if task_result.status else None
            )

            await client.send_task_result(
                task_id=task_id,
                session_id=session_id,
                state=state,
                products=products,
                status_data_items=status_data_items,
            )

            logger.info(
                "[GroupHandler:%s] <<< Task update broadcasted: task=...%s, state=%s",
                self.agent_name,
                short_task,
                state_name,
            )
        except Exception as e:
            logger.error(
                "[GroupHandler:%s] Failed to broadcast task update: task=...%s, error=%s",
                self.agent_name,
                short_task,
                str(e)[:100],
            )

    def _find_group_for_sender(self, sender_id: str) -> Optional[str]:
        """
        根据发送者 ID 查找对应的群组

        Args:
            sender_id: 发送者 AIC

        Returns:
            群组 ID 或 None
        """
        short_sender = sender_id[-8:] if len(sender_id) > 8 else sender_id

        # 如果发送者是 Leader，查找其所属的群组
        for group_id, client in self._group_clients.items():
            if client.is_joined and client._group_info:
                # GroupInfo.leader 是 ACSObject 类型
                leader_aic = (
                    client._group_info.leader.aic if client._group_info.leader else None
                )
                if leader_aic == sender_id:
                    short_group = group_id[-8:] if len(group_id) > 8 else group_id
                    logger.debug(
                        "[GroupHandler:%s] Found group for sender ...%s: group=...%s",
                        self.agent_name,
                        short_sender,
                        short_group,
                    )
                    return group_id

        logger.debug(
            "[GroupHandler:%s] No group found for sender ...%s",
            self.agent_name,
            short_sender,
        )
        return None

    async def leave_group(self, group_id: str) -> bool:
        """
        离开群组

        Args:
            group_id: 群组 ID

        Returns:
            是否成功离开
        """
        short_group = group_id[-8:] if len(group_id) > 8 else group_id
        client = self._group_clients.get(group_id)
        if not client:
            logger.warning(
                "[GroupHandler:%s] Not in group ...%s",
                self.agent_name,
                short_group,
            )
            return False

        try:
            logger.info(
                "[GroupHandler:%s] Leaving group ...%s...",
                self.agent_name,
                short_group,
            )
            await client.leave_group()
            del self._group_clients[group_id]

            # 清理任务映射
            tasks_to_remove = [
                tid for tid, gid in self._task_group_map.items() if gid == group_id
            ]
            for tid in tasks_to_remove:
                del self._task_group_map[tid]

            logger.info(
                "[GroupHandler:%s] Left group ...%s, cleaned %d task mappings",
                self.agent_name,
                short_group,
                len(tasks_to_remove),
            )
            return True

        except Exception as e:
            logger.error(
                "[GroupHandler:%s] Failed to leave group ...%s: %s",
                self.agent_name,
                short_group,
                str(e)[:100],
            )
            return False

    async def leave_all_groups(self) -> None:
        """离开所有群组"""
        group_count = len(self._group_clients)
        if group_count == 0:
            logger.debug("[GroupHandler:%s] No groups to leave", self.agent_name)
            return

        logger.info(
            "[GroupHandler:%s] Leaving %d groups...", self.agent_name, group_count
        )
        group_ids = list(self._group_clients.keys())
        for group_id in group_ids:
            await self.leave_group(group_id)
        logger.info("[GroupHandler:%s] Left all groups", self.agent_name)

    async def shutdown(self) -> None:
        """关闭群组处理器"""
        logger.info("[GroupHandler:%s] Shutting down...", self.agent_name)
        await self.leave_all_groups()
        logger.info("[GroupHandler:%s] Shutdown complete", self.agent_name)
