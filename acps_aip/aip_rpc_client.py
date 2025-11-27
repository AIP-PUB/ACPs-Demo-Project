import httpx
import uuid
import ssl
from datetime import datetime, timezone
from typing import Optional

from .aip_base_model import Message, Task, TaskCommand, TextDataItem
from .aip_rpc_model import RpcRequest, RpcRequestParams, RpcResponse


class AipRpcClient:
    """
    A client for interacting with an AIP-compliant Partner over RPC.
    Supports both HTTP and HTTPS with mTLS.
    """

    def __init__(
        self,
        partner_url: str,
        leader_id: str,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        """
        初始化AIP RPC客户端

        Args:
            partner_url: Partner的RPC端点URL
            leader_id: Leader Agent的ID
            ssl_context: 可选的SSL上下文，用于mTLS连接
        """
        self.partner_url = partner_url
        self.leader_id = leader_id

        # 如果提供了SSL上下文，创建支持mTLS的HTTP客户端
        if ssl_context:
            self.http_client = httpx.AsyncClient(verify=ssl_context)
        else:
            self.http_client = httpx.AsyncClient()

    async def _send_request(self, message: Message) -> RpcResponse:
        """
        Sends a message to the Partner and returns the response.
        """
        request_id = str(uuid.uuid4())
        rpc_request = RpcRequest(
            id=request_id,
            params=RpcRequestParams(message=message),
        )

        try:
            response = await self.http_client.post(
                self.partner_url,
                json=rpc_request.model_dump(exclude_none=True),
                headers={"Content-Type": "application/json"},
                timeout=30.0,  # Set a reasonable timeout
            )
            response.raise_for_status()

            # The response from the server should be a RpcResponse
            # Pydantic will automatically handle the validation and parsing
            # of the nested Task or Message object in the 'result' field.
            rpc_response = RpcResponse.model_validate(response.json())

            if rpc_response.error:
                raise Exception(
                    f"RPC Error: {rpc_response.error.code} - {rpc_response.error.message}"
                )

            if rpc_response.id != request_id:
                raise Exception("RPC Error: Response ID does not match request ID.")

            return rpc_response

        except httpx.HTTPStatusError as e:
            raise Exception(
                f"HTTP Error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            # Re-raise other exceptions like connection errors or parsing errors
            raise e

    def _create_message(
        self,
        command: TaskCommand,
        task_id: str,
        session_id: str,
        text_content: str | None = None,
    ) -> Message:
        """
        Helper to create a new Message object.
        """
        data_items = []
        if text_content:
            data_items.append(TextDataItem(text=text_content))

        return Message(
            id=f"msg-{uuid.uuid4()}",
            sentAt=datetime.now(timezone.utc).isoformat(),
            senderRole="leader",
            senderId=self.leader_id,
            command=command,
            dataItems=data_items,
            taskId=task_id,
            sessionId=session_id,
        )

    async def start_task(self, session_id: str, user_input: str) -> Task:
        """
        Starts a new task with the Partner.
        """
        task_id = f"task-{uuid.uuid4()}"
        message = self._create_message(
            command=TaskCommand.Start,
            task_id=task_id,
            session_id=session_id,
            text_content=user_input,
        )

        response = await self._send_request(message)
        if isinstance(response.result, Task):
            return response.result
        raise TypeError(f"Expected Task, got {type(response.result)}")

    async def continue_task(
        self, task_id: str, session_id: str, user_input: str
    ) -> Task:
        """
        Continues a task that is in a waiting state.
        """
        message = self._create_message(
            command=TaskCommand.Continue,
            task_id=task_id,
            session_id=session_id,
            text_content=user_input,
        )
        response = await self._send_request(message)
        if isinstance(response.result, Task):
            return response.result
        raise TypeError(f"Expected Task, got {type(response.result)}")

    async def complete_task(self, task_id: str, session_id: str) -> Task:
        """
        Marks a task as completed.
        """
        message = self._create_message(
            command=TaskCommand.Complete,
            task_id=task_id,
            session_id=session_id,
        )
        response = await self._send_request(message)
        if isinstance(response.result, Task):
            return response.result
        raise TypeError(f"Expected Task, got {type(response.result)}")

    async def get_task(self, task_id: str, session_id: str) -> Task:
        """
        Retrieves the current state of a task.
        """
        message = self._create_message(
            command=TaskCommand.Get,
            task_id=task_id,
            session_id=session_id,
        )
        response = await self._send_request(message)
        if isinstance(response.result, Task):
            return response.result
        raise TypeError(f"Expected Task, got {type(response.result)}")

    async def close(self):
        """
        Closes the underlying HTTP client.
        """
        await self.http_client.aclose()
