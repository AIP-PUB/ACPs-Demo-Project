from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from .aip_base_model import TaskState, Message
from .aip_rpc_model import JSONRPCRequest, JSONRPCResponse


class NotificationConfig(BaseModel):
    id: Optional[str] = None
    url: str
    token: str
    taskId: str


class NotificationRequest(JSONRPCRequest):
    method: Literal["notification/set"] = "notification/set"
    params: NotificationConfig


class NotificationResponse(JSONRPCResponse):
    result: NotificationConfig


class NotificationIdParams(BaseModel):
    taskId: str
    notificationConfigId: Optional[str] = None


class NotificationIdRequest(JSONRPCRequest):
    method: Union[Literal["notification/delete"], Literal["notification/get"]]
    params: NotificationIdParams


class NotificationDeleteResult(BaseModel):
    success: bool


class NotificationDeleteResponse(JSONRPCResponse):
    result: Optional[NotificationDeleteResult] = None


class NotificationGetResponse(JSONRPCResponse):
    result: List[NotificationConfig]


class NotificationStartParams(BaseModel):
    notificationConfigId: str
    notifyOnStates: Optional[List[TaskState]] = None


class NotificationStartRequestParams(BaseModel):
    message: Message


class NotificationStartRequest(JSONRPCRequest):
    method: Literal["notification/start"] = "notification/start"
    params: NotificationStartRequestParams
