"""
Integration Test Fixtures - conftest.py

提供集成测试所需的 fixtures，包括：
- FastAPI TestClient
- 真实 Agent 访问
- 测试数据工厂
"""

import os
import sys
import pytest
import asyncio
import tomllib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from partners.main import create_agent_app
from partners.utils import discover_agents, ONLINE_DIR
from acps_sdk.aip.aip_base_model import (
    TaskResult,
    TaskStatus,
    TaskState,
    TaskCommand,
    TextDataItem,
    TaskCommandType,
)
from acps_sdk.aip.aip_rpc_model import RpcRequest, RpcRequestParams


# --- 时区常量 ---
BEIJING_TZ = timezone(timedelta(hours=8))


# --- LLM API 可用性检查 ---
def _check_llm_api_available():
    """
    检查 LLM API 是否可用

    优先检查任意一个 partner 的 config.toml 中是否配置了有效的 API Key
    """
    # 首先检查 partner 配置文件
    partners_dir = os.path.join(PROJECT_ROOT, "partners", "online")
    if os.path.exists(partners_dir):
        for agent_name in os.listdir(partners_dir):
            config_path = os.path.join(partners_dir, agent_name, "config.toml")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)
                    llm_config = config.get("llm", {})
                    for profile in llm_config.values():
                        if isinstance(profile, dict):
                            api_key = profile.get("api_key", "")
                            # 检查是否是有效的 API Key（非空且不是环境变量占位符）
                            if (
                                api_key
                                and not api_key.startswith("${")
                                and len(api_key) > 10
                            ):
                                return True
                except Exception:
                    continue

    # 回退：检查环境变量
    api_key_vars = [
        "OPENAI_API_KEY",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
    ]
    for var in api_key_vars:
        if os.getenv(var):
            return True

    return False


LLM_API_AVAILABLE = _check_llm_api_available()

# 用于跳过需要 LLM 的测试（当配置文件和环境变量都没有 API Key 时才跳过）
requires_llm = pytest.mark.skipif(
    not LLM_API_AVAILABLE,
    reason="LLM API key not available in config.toml or environment variables",
)


# --- 测试用 Agent 发现 ---
_test_agents = discover_agents()
_first_agent_name = next(iter(_test_agents)) if _test_agents else None
_first_agent_path = _test_agents.get(_first_agent_name, "") if _first_agent_name else ""


# --- FastAPI TestClient ---
@pytest.fixture(scope="module")
def client():
    """
    提供 FastAPI TestClient 实例。
    scope="module" 确保在每个测试模块中只创建一次。
    使用第一个 online agent 创建测试应用。
    """
    if not _first_agent_name:
        pytest.skip("No online agents available")
    test_app = create_agent_app(_first_agent_name, _first_agent_path)
    with TestClient(test_app) as c:
        yield c


@pytest.fixture(scope="module")
def async_client():
    """提供异步 HTTP 客户端"""
    if not _first_agent_name:
        pytest.skip("No online agents available")
    test_app = create_agent_app(_first_agent_name, _first_agent_path)
    return AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test")


# --- Agent 信息 ---
@pytest.fixture(scope="module")
def online_agents():
    """获取所有在线的 Agent 名称列表"""
    return list(_test_agents.keys())


@pytest.fixture
def get_agent_runner():
    """获取指定 Agent 的 GenericRunner 实例（测试环境下不可用）"""

    def _get(agent_name: str):
        return None

    return _get


# --- 命令和请求工厂 ---
@pytest.fixture
def message_factory():
    """创建测试 TaskCommand 的工厂函数"""

    def create_command(
        text: str,
        command_type: TaskCommandType = TaskCommandType.Start,
        task_id: str = None,
        session_id: str = None,
    ) -> TaskCommand:
        now = datetime.now(BEIJING_TZ)
        return TaskCommand(
            id=f"cmd-{now.timestamp()}",
            sentAt=now.isoformat(),
            senderRole="leader",
            senderId="test-leader-001",
            command=command_type,  # 使用 command 字段名
            taskId=task_id or f"task-{now.timestamp()}",
            sessionId=session_id or f"session-{now.timestamp()}",
            dataItems=[TextDataItem(text=text)],
        )

    return create_command


@pytest.fixture
def rpc_request_factory(message_factory):
    """创建测试 RPC 请求的工厂函数"""

    def create_request(
        text: str,
        command_type: TaskCommandType = TaskCommandType.Start,
        task_id: str = None,
        session_id: str = None,
    ) -> Dict[str, Any]:
        command = message_factory(text, command_type, task_id, session_id)
        request = RpcRequest(
            id=f"rpc-{datetime.now(BEIJING_TZ).timestamp()}",
            params=RpcRequestParams(command=command),
        )
        return request.model_dump()

    return create_request


@pytest.fixture
def rpc_call(client, rpc_request_factory):
    """
    执行 RPC 调用的辅助函数。
    返回 (response_json, task_id, session_id) 元组。
    """

    def _call(
        agent_name: str,
        text: str,
        command: TaskCommandType = TaskCommandType.Start,
        task_id: str = None,
        session_id: str = None,
    ):
        request_data = rpc_request_factory(text, command, task_id, session_id)
        response = client.post("/rpc", json=request_data)

        # 从请求中提取 task_id 和 session_id (AIP v2: params.command)
        cmd = request_data["params"]["command"]
        return response.json(), cmd["taskId"], cmd["sessionId"]

    return _call


# --- 任务生命周期辅助 ---
@pytest.fixture
def wait_for_state(client, rpc_request_factory):
    """
    等待任务达到指定状态。
    使用 Get 命令轮询任务状态。
    """
    import time

    def _wait(
        agent_name: str,
        task_id: str,
        session_id: str,
        target_states: List[TaskState],
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        start_time = time.time()
        target_state_values = [s.value for s in target_states]

        while time.time() - start_time < timeout:
            request_data = rpc_request_factory(
                text="",
                command_type=TaskCommandType.Get,
                task_id=task_id,
                session_id=session_id,
            )
            response = client.post("/rpc", json=request_data)
            result = response.json()

            if "result" in result and result["result"]:
                state = result["result"].get("status", {}).get("state")
                if state in target_state_values:
                    return result

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Task {task_id} did not reach state {target_states} within {timeout}s"
        )

    return _wait


@pytest.fixture
def continue_task(client, rpc_request_factory):
    """
    向处于 AwaitingInput 状态的任务发送 Continue 命令。
    用于补充信息继续任务流程。
    """

    def _continue(
        agent_name: str,
        task_id: str,
        session_id: str,
        text: str,
    ) -> Dict[str, Any]:
        request_data = rpc_request_factory(
            text=text,
            command_type=TaskCommandType.Continue,
            task_id=task_id,
            session_id=session_id,
        )
        response = client.post("/rpc", json=request_data)
        return response.json()

    return _continue


@pytest.fixture
def complete_task_flow(rpc_call, wait_for_state):
    """
    执行完整的任务流程：Start -> 等待状态 -> (可选 Continue) -> Complete。
    返回最终的任务结果。
    """

    def _flow(
        agent_name: str,
        initial_text: str,
        continue_texts: List[str] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        # Step 1: Start
        result, task_id, session_id = rpc_call(
            agent_name, initial_text, TaskCommandType.Start
        )

        # Step 2: 等待处理结果
        result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [
                TaskState.AwaitingInput,
                TaskState.AwaitingCompletion,
                TaskState.Rejected,
                TaskState.Failed,
            ],
            timeout=timeout,
        )

        # Step 3: 如果需要补充信息，发送 Continue
        if continue_texts:
            for text in continue_texts:
                state = result["result"]["status"]["state"]
                if state == TaskState.AwaitingInput.value:
                    result, _, _ = rpc_call(
                        agent_name, text, TaskCommandType.Continue, task_id, session_id
                    )
                    result = wait_for_state(
                        agent_name,
                        task_id,
                        session_id,
                        [
                            TaskState.AwaitingInput,
                            TaskState.AwaitingCompletion,
                            TaskState.Rejected,
                            TaskState.Failed,
                        ],
                        timeout=timeout,
                    )

        # Step 4: 如果到达 AwaitingCompletion，发送 Complete
        final_state = result["result"]["status"]["state"]
        if final_state == TaskState.AwaitingCompletion.value:
            result, _, _ = rpc_call(
                agent_name, "确认完成", TaskCommandType.Complete, task_id, session_id
            )

        return result

    return _flow


# --- 状态验证辅助 ---
@pytest.fixture
def assert_rpc_success():
    """验证 RPC 响应成功"""

    def _assert(response: Dict[str, Any], msg: str = ""):
        assert (
            "error" not in response or response["error"] is None
        ), f"RPC error: {response.get('error')}. {msg}"
        assert "result" in response, f"Missing result in response. {msg}"

    return _assert


@pytest.fixture
def assert_task_state():
    """验证任务状态"""

    def _assert(response: Dict[str, Any], expected_state: TaskState, msg: str = ""):
        actual_state = response.get("result", {}).get("status", {}).get("state")
        assert (
            actual_state == expected_state.value
        ), f"Expected state {expected_state.value}, got {actual_state}. {msg}"

    return _assert


@pytest.fixture
def assert_task_has_product():
    """验证任务有产出物"""

    def _assert(response: Dict[str, Any], msg: str = ""):
        products = response.get("result", {}).get("products", [])
        assert len(products) > 0, f"Task has no products. {msg}"

    return _assert


@pytest.fixture
def get_task_context(get_agent_runner):
    """获取任务的内部上下文（用于验证内部状态）"""

    def _get(agent_name: str, task_id: str):
        runner = get_agent_runner(agent_name)
        if runner and task_id in runner.tasks:
            return runner.tasks[task_id]
        return None

    return _get


# --- 测试数据 ---
@pytest.fixture
def test_scenarios():
    """提供各 Agent 的测试场景"""
    return {
        "beijing_urban": {
            "accept": [
                "我想去北京故宫和天坛玩两天",
                "请推荐一下北京城区的文化景点",
                "帮我规划一个北京城区亲子游行程",
            ],
            "reject": [
                "我想去上海迪士尼玩",
                "请推荐北京郊区的民宿",
                "帮我预订北京的酒店",
            ],
        },
        "beijing_rural": {
            "accept": [
                "我想去长城和十三陵玩",
                "请推荐北京郊区的自然风光景点",
                "帮我规划一个去慕田峪长城的行程",
            ],
            "reject": ["我想去北京故宫", "请推荐上海的景点"],
        },
        "beijing_food": {
            "accept": [
                "推荐北京好吃的烤鸭店",
                "我想吃正宗的北京小吃",
                "帮我找一家适合聚餐的餐厅",
            ],
            "reject": ["我想去故宫玩", "请推荐上海的美食"],
        },
        "china_hotel": {
            "accept": [
                "帮我在北京找一家五星级酒店",
                "我需要预订一间标准间",
                "推荐北京性价比高的酒店",
            ],
            "reject": ["我想去故宫玩", "帮我规划旅游路线"],
        },
        "china_transport": {
            "accept": [
                "帮我查一下北京到上海的高铁",
                "我想预订机票",
                "查询北京南站到上海虹桥的车次",
            ],
            "reject": ["我想去故宫玩", "推荐北京的美食"],
        },
    }


# --- 异步测试支持 ---
@pytest.fixture(scope="session")
def event_loop():
    """创建会话级别的事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
