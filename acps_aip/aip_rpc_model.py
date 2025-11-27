from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from .aip_base_model import Task, Message


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    id: Optional[Union[str, int]] = None
    params: Optional[Union[List[Any], Dict[str, Any]]] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


class RpcRequestParams(BaseModel):
    message: Message


class RpcRequest(JSONRPCRequest):
    method: Literal["rpc"] = "rpc"
    params: RpcRequestParams


class RpcResponse(JSONRPCResponse):
    result: Union[Task, Message]
