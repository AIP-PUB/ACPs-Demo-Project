from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict, Any, Literal

from .aip_rpc_model import JSONRPCRequest, JSONRPCResponse, JSONRPCError
from .aip_base_model import Message


class ACSObject(BaseModel):
    aic: str
    skills: Optional[List[str]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class GroupInfo(BaseModel):
    groupId: str
    leader: ACSObject
    partners: List[ACSObject]


class RabbitMQServerConfig(BaseModel):
    host: str
    port: int
    vhost: str
    accessToken: str


class AMQPConfig(BaseModel):
    exchange: str
    exchangeType: str
    routingKey: str


class RabbitMQRequestParams(BaseModel):
    protocol: str  # MQProtocolVersion is not defined, using str
    group: GroupInfo
    server: RabbitMQServerConfig
    amqp: AMQPConfig


class RabbitMQRequest(JSONRPCRequest):
    method: Literal["group"] = "group"
    params: RabbitMQRequestParams


class RabbitMQResponseResult(BaseModel):
    connectionName: str
    vhost: str
    nodeName: str
    queueName: str
    processId: Optional[str] = None


class RabbitMQResponseErrorData(BaseModel):
    errorType: str
    details: Optional[Any] = None


class RabbitMQResponseError(JSONRPCError):
    data: Optional[RabbitMQResponseErrorData] = None


class RabbitMQResponse(JSONRPCResponse):
    result: Optional[RabbitMQResponseResult] = None
    error: Optional[RabbitMQResponseError] = None


class GroupMgmtCommand(str, Enum):
    GET_STATUS = "get-status"
    LEAVE_GROUP = "leave-group"
    MUTE = "mute"
    UNMUTE = "unmute"


class GroupMemberStatus(BaseModel):
    connected: bool
    muted: bool


class GroupMgmtMessage(Message):
    type: Literal["group-mgmt-message"] = "group-mgmt-message"
    groupMgmtCommand: Optional[GroupMgmtCommand] = None
    groupMemberStatus: Optional[GroupMemberStatus] = None
