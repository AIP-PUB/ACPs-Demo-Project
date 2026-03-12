"""
Leader Agent Platform - Session 聚合根

本模块定义 Session 及其相关的数据模型，包括场景配置、对话上下文、事件日志等。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import (
    ActiveTaskId,
    AgentAic,
    AipTaskId,
    EventLogType,
    ExecutionMode,
    IntentType,
    IsoDateTimeString,
    ResponseType,
    SessionClosedReason,
    SessionId,
)
from .partner import PartnerRuntimeState
from .task import ActiveTask, UserResult


# =============================================================================
# 场景相关
# =============================================================================


class ScenarioBrief(BaseModel):
    """
    场景简要信息，用于意图分析阶段做"是否切换/激活专业场景"的判断。
    """

    id: str = Field(..., description="场景 ID（建议与目录名一致）")
    name: str = Field(..., description="场景名称")
    description: str = Field(..., description="简要描述（用于 LLM-1 判断匹配度）")
    keywords: List[str] = Field(default_factory=list, description="路由关键词")


class ScenarioRuntime(BaseModel):
    """
    已加载的场景配置（运行时视图）。

    把"文件系统的配置目录"映射为内存对象。
    """

    id: str = Field(..., description="场景 ID：base 或某个 expert 场景")
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
    prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="提示词集合（来自 prompts.toml）",
    )
    domain_meta: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="domainMeta",
        description="领域元数据（来自 domain.toml）",
    )
    static_partners: Optional[Dict[AgentAic, Any]] = Field(
        default=None,
        alias="staticPartners",
        description="场景内静态 ACS 文件的解析结果",
    )

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# 对话上下文
# =============================================================================


class DialogTurn(BaseModel):
    """
    对话历史中的单轮交互记录。
    """

    user_query: str = Field(
        ...,
        alias="userQuery",
        description="用户输入",
    )
    intent_type: IntentType = Field(
        ...,
        alias="intentType",
        description="意图判定结果",
    )
    response_type: ResponseType = Field(
        ...,
        alias="responseType",
        description="响应类型",
    )
    response_summary: Optional[str] = Field(
        default=None,
        alias="responseSummary",
        description="响应摘要（非完整内容）",
    )
    timestamp: IsoDateTimeString = Field(
        ...,
        description="交互时间",
    )

    model_config = ConfigDict(populate_by_name=True)


class DialogContext(BaseModel):
    """
    对话上下文（用于 LLM 调用时提供历史信息）。

    设计原则：
    - recentTurns 保留最近 N 轮原始交互，支持指代消解
    - historySummary 存储更早历史的压缩摘要，由 LLM-7 生成
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="Session ID",
    )
    updated_at: IsoDateTimeString = Field(
        ...,
        alias="updatedAt",
        description="最后更新时间",
    )
    recent_turns: List[DialogTurn] = Field(
        default_factory=list,
        alias="recentTurns",
        description="最近 N 轮的原始交互记录",
    )
    history_summary: Optional[str] = Field(
        default=None,
        alias="historySummary",
        description="更早历史的压缩摘要（由 LLM-7 生成）",
    )
    history_turn_count: Optional[int] = Field(
        default=None,
        alias="historyTurnCount",
        description="historySummary 覆盖的交互轮数",
    )

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# 事件日志
# =============================================================================


class EventLogEntry(BaseModel):
    """
    Session 内的事件日志条目。

    用途：
    - 排障：能复盘意图分析/规划/路由决策/AIP 往返
    - 对话摘要：dialogSummary 可从 eventLog 中抽取关键信息

    约束：内存 FIFO + 最大长度，避免无限增长。
    """

    id: str = Field(..., description="事件唯一 ID")
    created_at: IsoDateTimeString = Field(
        ...,
        alias="createdAt",
        description="事件发生时间",
    )
    type: EventLogType = Field(..., description="事件类型")
    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="关联的 sessionId",
    )
    active_task_id: Optional[ActiveTaskId] = Field(
        default=None,
        alias="activeTaskId",
        description="关联的 activeTaskId",
    )
    partner_aic: Optional[AgentAic] = Field(
        default=None,
        alias="partnerAic",
        description="关联的 partnerAic",
    )
    aip_task_id: Optional[AipTaskId] = Field(
        default=None,
        alias="aipTaskId",
        description="关联的 aipTaskId",
    )
    payload: Any = Field(..., description="事件负载")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Group 模式路由
# =============================================================================


class GroupRoutingInfo(BaseModel):
    """
    Group 模式路由信息（最小可落地集）。
    """

    group_id: str = Field(
        ...,
        alias="groupId",
        description="绑定到 Session 的 groupId",
    )
    provider: str = Field(
        default="rabbitmq",
        description="MQ 协议/实现标识",
    )
    broker_url: Optional[str] = Field(
        default=None,
        alias="brokerUrl",
        description="Broker 连接串",
    )
    exchange: Optional[str] = Field(default=None, description="RabbitMQ exchange")
    routing_key: Optional[str] = Field(
        default=None,
        alias="routingKey",
        description="RabbitMQ routing key",
    )
    queue: Optional[str] = Field(default=None, description="RabbitMQ queue")
    topic: Optional[str] = Field(default=None, description="Kafka topic")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Session 聚合根
# =============================================================================


class Session(BaseModel):
    """
    Session 聚合根（Leader 内存态）。
    """

    session_id: SessionId = Field(
        ...,
        alias="sessionId",
        description="对外标识",
    )
    mode: ExecutionMode = Field(
        ...,
        description="执行模式（创建后不可变）",
    )
    user_id: Optional[str] = Field(
        default=None,
        alias="userId",
        description="可选用户标识",
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
        description="最后一次触碰时间",
    )
    ttl_seconds: int = Field(
        ...,
        alias="ttlSeconds",
        description="TTL 秒数",
    )
    revision: Optional[int] = Field(
        default=None,
        description="乐观并发版本号",
    )
    expires_at: IsoDateTimeString = Field(
        ...,
        alias="expiresAt",
        description="到期时间",
    )
    closed: Optional[bool] = Field(
        default=None,
        description="Session 是否已关闭",
    )
    closed_at: Optional[IsoDateTimeString] = Field(
        default=None,
        alias="closedAt",
        description="关闭时间",
    )
    closed_reason: Optional[SessionClosedReason] = Field(
        default=None,
        alias="closedReason",
        description="关闭原因",
    )
    group_id: Optional[str] = Field(
        default=None,
        alias="groupId",
        description="Group 模式关联的 groupId",
    )
    group_routing: Optional[GroupRoutingInfo] = Field(
        default=None,
        alias="groupRouting",
        description="Group 模式的路由信息",
    )
    base_scenario: ScenarioRuntime = Field(
        ...,
        alias="baseScenario",
        description="基础场景（始终存在）",
    )
    expert_scenario: Optional[ScenarioRuntime] = Field(
        default=None,
        alias="expertScenario",
        description="当前激活的专业场景",
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
        description="会话内可用 Partner 集合",
    )
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        alias="userContext",
        description="用户上下文（结构化偏好/约束）",
    )
    dialog_context: Optional[DialogContext] = Field(
        default=None,
        alias="dialogContext",
        description="对话上下文",
    )
    event_log: List[EventLogEntry] = Field(
        default_factory=list,
        alias="eventLog",
        description="事件日志（FIFO）",
    )
    user_result: UserResult = Field(
        ...,
        alias="userResult",
        description="对用户暴露的会话级输出",
    )

    model_config = ConfigDict(populate_by_name=True)
