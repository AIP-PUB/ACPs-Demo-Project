from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from .aip_base_model import Task, Message, TaskStatus, Product
from .aip_rpc_model import JSONRPCRequest, JSONRPCResponse


class StreamRequestParams(BaseModel):
    message: Message


class StreamRequest(JSONRPCRequest):
    method: Literal["stream"] = "stream"
    params: StreamRequestParams


class TaskStatusUpdateEvent(BaseModel):
    type: Literal["status-update"] = "status-update"
    taskId: str
    status: TaskStatus
    sessionId: str


class ProductChunkEvent(BaseModel):
    type: Literal["product-chunk"] = "product-chunk"
    taskId: str
    product: Product
    append: bool
    lastChunk: bool
    sessionId: str


class StreamEventData(BaseModel):
    eventSeq: int
    eventData: Union[Task, Message, TaskStatusUpdateEvent, ProductChunkEvent]


class StreamResponse(JSONRPCResponse):
    result: StreamEventData


class ReStreamCommandParams(BaseModel):
    lastEventSeq: Optional[int] = None
