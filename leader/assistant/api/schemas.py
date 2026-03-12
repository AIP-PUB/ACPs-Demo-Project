"""
Leader Agent Platform - API Schemas

本模块定义 Leader HTTP API 的请求/响应模型。

端点概览：
- POST /submit: 提交用户输入
- GET /result: 获取当前 Session 状态
- GET /log: 获取事件日志
- POST /cancel: 取消任务
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..models import (
    ActiveTask,
    ActiveTaskId,
    ActiveTaskStatus,
    AgentAic,
    ClientRequestId,
    DialogContext,
    EventLogEntry,
    ExecutionMode,
    IsoDateTimeString,
    PartnerRuntimeState,
    ScenarioBrief,
    SessionId,
    UserResult,
)


# =============================================================================
# 通用响应结构
# =============================================================================

T = TypeVar("T")


class CommonError(BaseModel):
    """
    通用错误结构。

    说明：
    - code 采用 6 位整数：HTTP 状态码(3位) + 业务错误码(3位)
    - data 推荐返回结构化细节，便于 UI/测试定位问题
    """

    code: int = Field(..., description="错误码（6位整数）")
    message: str = Field(..., description="错误消息（面向开发者）")
    data: Optional[Any] = Field(default=None, description="结构化错误详情")


class CommonResponse(BaseModel, Generic[T]):
    """
    通用响应结构。

    约束：result 与 error 互斥。
    """

    result: Optional[T] = Field(default=None, description="成功时的返回数据")
    error: Optional[CommonError] = Field(default=None, description="失败时的错误信息")


# =============================================================================
# /submit 端点
# =============================================================================


class SubmitRequest(BaseModel):
    """
    /submit 请求体。

    说明：
    - clientRequestId：用于幂等/重放保护（对应错误码 409003）
    - activeTaskId：用于"用户侧并发提交"的乐观校验（对应错误码 409002）
    - mode：创建新 Session 时指定执行模式，默认 direct_rpc；已有 Session 时忽略此字段
    """

    session_id: Optional[SessionId] = Field(
        default=None,
        alias="sessionId",
        description="会话 ID（不传则创建新会话）",
    )
    mode: ExecutionMode = Field(
        default=ExecutionMode.DIRECT_RPC,
        description="执行模式（创建新 Session 时指定，已有 Session 时忽略）",
    )
    client_request_id: ClientRequestId = Field(
        ...,
        alias="clientRequestId",
        description="客户端请求去重 ID",
    )
    query: str = Field(..., description="用户输入文本")
    active_task_id: Optional[ActiveTaskId] = Field(
        default=None,
        alias="activeTaskId",
        description="乐观校验用的任务 ID",
    )
    user_id: Optional[str] = Field(
        default=None,
        alias="userId",
        description="可选用户标识",
    )

    model_config = ConfigDict(populate_by_name=True)


class SubmitResult(BaseModel):
    """
    /submit 成功返回的最小结果。

    说明：submit 是异步受理，返回"本次提交落在了哪个 session/activeTask"。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="会话 ID",
    )
    mode: ExecutionMode = Field(..., description="执行模式")
    active_task_id: ActiveTaskId = Field(
        ...,
        alias="activeTaskId",
        description="活跃任务 ID",
    )
    accepted_at: IsoDateTimeString = Field(
        ...,
        alias="acceptedAt",
        description="受理时间",
    )
    external_status: ActiveTaskStatus = Field(
        ...,
        alias="externalStatus",
        description="activeTask 的对外状态",
    )

    model_config = ConfigDict(populate_by_name=True)


class SubmitResponse(CommonResponse[SubmitResult]):
    """
    /submit 响应体。
    """

    pass


# =============================================================================
# /result 端点
# =============================================================================


class ScenarioRuntimeView(BaseModel):
    """
    场景运行时视图（不包含 prompts 等内部配置）。
    """

    id: str = Field(..., description="场景 ID")
    kind: str = Field(..., description="场景类型：base / expert")
    version: Optional[str] = Field(default=None, description="场景版本")
    loaded_at: IsoDateTimeString = Field(
        ...,
        alias="loadedAt",
        description="场景加载时间",
    )
    source_path: Optional[str] = Field(
        default=None,
        alias="sourcePath",
        description="配置来源路径",
    )
    config_digest: Optional[str] = Field(
        default=None,
        alias="configDigest",
        description="配置内容摘要",
    )

    model_config = ConfigDict(populate_by_name=True)


class GroupRoutingInfoView(BaseModel):
    """
    Group 模式路由信息视图（脱敏版）。
    """

    group_id: str = Field(..., alias="groupId", description="Group ID")
    provider: str = Field(..., description="MQ 协议/实现标识")
    exchange: Optional[str] = Field(default=None, description="RabbitMQ exchange")
    routing_key: Optional[str] = Field(
        default=None,
        alias="routingKey",
        description="RabbitMQ routing key",
    )
    queue: Optional[str] = Field(default=None, description="RabbitMQ queue")
    topic: Optional[str] = Field(default=None, description="Kafka topic")

    model_config = ConfigDict(populate_by_name=True)


class LeaderResult(BaseModel):
    """
    /result 返回的 Session 视图（不包含 eventLog）。

    说明：
    - 前端交互主要依赖 activeTask/userResult/partner 状态
    - 配置原文（prompts/domainMeta）不在此暴露
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="会话 ID",
    )
    mode: ExecutionMode = Field(..., description="执行模式")
    user_id: Optional[str] = Field(
        default=None,
        alias="userId",
        description="用户标识",
    )
    created_at: IsoDateTimeString = Field(
        ...,
        alias="createdAt",
        description="创建时间",
    )
    updated_at: IsoDateTimeString = Field(
        ...,
        alias="updatedAt",
        description="更新时间",
    )
    touched_at: IsoDateTimeString = Field(
        ...,
        alias="touchedAt",
        description="最后触碰时间",
    )
    ttl_seconds: int = Field(
        ...,
        alias="ttlSeconds",
        description="TTL 秒数",
    )
    expires_at: IsoDateTimeString = Field(
        ...,
        alias="expiresAt",
        description="到期时间",
    )
    closed: Optional[bool] = Field(default=None, description="是否已关闭")
    closed_at: Optional[IsoDateTimeString] = Field(
        default=None,
        alias="closedAt",
        description="关闭时间",
    )
    closed_reason: Optional[str] = Field(
        default=None,
        alias="closedReason",
        description="关闭原因",
    )
    group_id: Optional[str] = Field(
        default=None,
        alias="groupId",
        description="Group 模式的 groupId",
    )
    group_routing: Optional[GroupRoutingInfoView] = Field(
        default=None,
        alias="groupRouting",
        description="Group 模式路由信息",
    )
    base_scenario: ScenarioRuntimeView = Field(
        ...,
        alias="baseScenario",
        description="基础场景",
    )
    expert_scenario: Optional[ScenarioRuntimeView] = Field(
        default=None,
        alias="expertScenario",
        description="当前专业场景",
    )
    scenario_briefs: List[ScenarioBrief] = Field(
        default_factory=list,
        alias="scenarioBriefs",
        description="已注册的专业场景列表",
    )
    active_task: Optional[ActiveTask] = Field(
        default=None,
        alias="activeTask",
        description="当前活跃任务",
    )
    partners: Dict[AgentAic, PartnerRuntimeState] = Field(
        default_factory=dict,
        description="Partner 运行时状态",
    )
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        alias="userContext",
        description="用户上下文",
    )
    dialog_context: Optional[DialogContext] = Field(
        default=None,
        alias="dialogContext",
        description="对话上下文",
    )
    user_result: UserResult = Field(
        ...,
        alias="userResult",
        description="用户可见输出",
    )

    model_config = ConfigDict(populate_by_name=True)


class ResultResponse(CommonResponse[LeaderResult]):
    """
    /result 响应体。
    """

    pass


# =============================================================================
# /log 端点
# =============================================================================


class LogRequest(BaseModel):
    """
    /log 请求参数（Query Parameters）。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="会话 ID",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="游标（不传表示从头开始）",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="返回条目数上限",
    )

    model_config = ConfigDict(populate_by_name=True)


class LogResult(BaseModel):
    """
    /log 成功返回的结果。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="会话 ID",
    )
    items: List[EventLogEntry] = Field(
        default_factory=list,
        description="事件日志条目列表",
    )
    next_cursor: Optional[str] = Field(
        default=None,
        alias="nextCursor",
        description="下一页游标",
    )
    has_more: Optional[bool] = Field(
        default=None,
        alias="hasMore",
        description="是否还有更多条目",
    )

    model_config = ConfigDict(populate_by_name=True)


class LogResponse(CommonResponse[LogResult]):
    """
    /log 响应体。
    """

    pass


# =============================================================================
# /cancel 端点
# =============================================================================


class CancelRequest(BaseModel):
    """
    /cancel 请求体。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="要取消的会话 ID",
    )
    active_task_id: ActiveTaskId = Field(
        ...,
        alias="activeTaskId",
        description="乐观校验用的 activeTaskId",
    )
    client_request_id: Optional[ClientRequestId] = Field(
        default=None,
        alias="clientRequestId",
        description="客户端请求去重 ID（推荐）",
    )

    model_config = ConfigDict(populate_by_name=True)


class CancelResult(BaseModel):
    """
    /cancel 成功返回的结果。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="被取消的会话 ID",
    )
    active_task_id: ActiveTaskId = Field(
        ...,
        alias="activeTaskId",
        description="被取消的任务 ID",
    )
    canceled_at: IsoDateTimeString = Field(
        ...,
        alias="canceledAt",
        description="取消时间",
    )

    model_config = ConfigDict(populate_by_name=True)


class CancelResponse(CommonResponse[CancelResult]):
    """
    /cancel 响应体。
    """

    pass
