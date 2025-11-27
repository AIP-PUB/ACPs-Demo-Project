from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal


class TaskState(str, Enum):
    Accepted = "accepted"
    Working = "working"
    AwaitingInput = "awaiting-input"
    AwaitingCompletion = "awaiting-completion"
    Completed = "completed"
    Canceled = "canceled"
    Failed = "failed"
    Rejected = "rejected"


class DataItemBase(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class TextDataItem(DataItemBase):
    type: Literal["text"] = "text"
    text: str


class FileDataItem(DataItemBase):
    type: Literal["file"] = "file"
    name: Optional[str] = None
    mimeType: Optional[str] = None
    uri: Optional[str] = None
    bytes: Optional[str] = None


class StructuredDataItem(DataItemBase):
    type: Literal["data"] = "data"
    data: Dict[str, Any]


DataItem = Union[TextDataItem, FileDataItem, StructuredDataItem]


class TaskStatus(BaseModel):
    state: TaskState
    stateChangedAt: str
    dataItems: Optional[List[DataItem]] = None


class Product(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    dataItems: List[DataItem]


class TaskCommand(str, Enum):
    Get = "get"
    Start = "start"
    Continue = "continue"
    Cancel = "cancel"
    Complete = "complete"
    ReStream = "re-stream"


class GetCommandParams(BaseModel):
    lastMessageSentAt: Optional[str] = None
    lastStateChangedAt: Optional[str] = None


class StartCommandParams(BaseModel):
    timeout: Optional[int] = None
    maxProductsBytes: Optional[int] = None


class Message(BaseModel):
    type: Literal["message"] = "message"
    id: str
    sentAt: str
    senderRole: Literal["leader", "partner"]
    senderId: str
    mentions: Optional[Union[Literal["all"], List[str]]] = None
    command: Optional[TaskCommand] = None
    commandParams: Optional[Dict[str, Any]] = None
    dataItems: List[DataItem]
    taskId: Optional[str] = None
    groupId: Optional[str] = None
    sessionId: Optional[str] = None


class Task(BaseModel):
    type: Literal["task"] = "task"
    id: str
    senderId: Optional[str] = None
    status: TaskStatus
    products: Optional[List[Product]] = None
    messageHistory: Optional[List[Message]] = None
    statusHistory: Optional[List[TaskStatus]] = None
    groupId: Optional[str] = None
    sessionId: str
