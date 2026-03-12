"""
Partner E2E Tests - Fixtures and Utilities

端到端测试的公共 fixtures，通过 HTTP 直接调用真实运行的 Partner 服务。
所有测试都是完全的黑盒测试，不使用 TestClient，而是真正的 HTTP 请求。

注意：每个 Partner 运行在独立端口上，测试通过 agent_name 路由到对应端口。
"""

import os
import pytest
import httpx
import time
import uuid
import tomllib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# 配置
# =============================================================================

_PARTNERS_ONLINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "online"
)


def _discover_all_agents() -> Dict[str, str]:
    """扫描 online 目录，返回所有 agent 的 URL 映射 {name: url}。"""
    agent_urls: Dict[str, str] = {}
    if not os.path.isdir(_PARTNERS_ONLINE_DIR):
        return agent_urls
    for name in sorted(os.listdir(_PARTNERS_ONLINE_DIR)):
        config_path = os.path.join(_PARTNERS_ONLINE_DIR, name, "config.toml")
        if os.path.isfile(config_path):
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            port = cfg.get("server", {}).get("port")
            if port:
                agent_urls[name] = f"http://localhost:{port}"
    return agent_urls


# agent_name → base URL  映射
AGENT_URLS: Dict[str, str] = _discover_all_agents()

# 向后兼容：取第一个 agent 的 URL
PARTNER_URL = next(iter(AGENT_URLS.values()), "http://localhost:59221")

BEIJING_TZ = timezone(timedelta(hours=8))

# 轮询配置
POLL_INTERVAL = 0.5
MAX_POLL_TIME = 60
MAX_POLL_TIME_LONG = 120


def _get_agent_url(agent_name: str) -> str:
    """根据 agent_name 返回对应的 base URL。"""
    if agent_name in AGENT_URLS:
        return AGENT_URLS[agent_name]
    return PARTNER_URL


# =============================================================================
# 服务健康检查
# =============================================================================


def check_partner_service_health() -> Tuple[bool, str, List[str]]:
    """
    逐个检查各 Agent 服务的健康状态。

    Returns:
        (is_healthy, message, online_agents)
    """
    if not AGENT_URLS:
        return False, "No agents discovered in online directory", []

    online_agents = []
    try:
        with httpx.Client(timeout=10.0) as client:
            for name, url in AGENT_URLS.items():
                try:
                    resp = client.get(f"{url}/health")
                    if resp.status_code == 200:
                        online_agents.append(name)
                except httpx.ConnectError:
                    pass
    except Exception as e:
        return False, f"Service check failed: {e}", []

    if online_agents:
        return True, f"{len(online_agents)} agents healthy", online_agents
    return False, "No agents reachable", []


# 检查服务是否可用
_health_result = check_partner_service_health()
PARTNER_SERVICE_AVAILABLE = _health_result[0]
ONLINE_AGENTS = _health_result[2]

# 用于跳过需要 Partner 服务的测试
requires_partner_service = pytest.mark.skipif(
    not PARTNER_SERVICE_AVAILABLE,
    reason=f"Partner service not available: {_health_result[1]}",
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_task_command(
    text: str,
    command: str,
    task_id: str,
    session_id: str,
    command_id: Optional[str] = None,
) -> Dict[str, Any]:
    """创建 AIP v2 TaskCommand"""
    now = datetime.now(BEIJING_TZ)
    return {
        "type": "task-command",
        "id": command_id or f"cmd-{now.timestamp()}",
        "sentAt": now.isoformat(),
        "senderRole": "leader",
        "senderId": "test-leader-e2e",
        "command": command,
        "dataItems": [{"type": "text", "text": text}] if text else [],
        "taskId": task_id,
        "sessionId": session_id,
    }


def create_rpc_request(
    command: Dict[str, Any], request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建完整的 RPC 请求 (AIP v2 格式)"""
    return {
        "jsonrpc": "2.0",
        "id": request_id or f"rpc-{uuid.uuid4().hex[:12]}",
        "method": "rpc",
        "params": {"command": command},
    }


def send_rpc(
    client: httpx.Client,
    agent_name: str,
    text: str,
    command: str,
    task_id: str,
    session_id: str,
) -> Dict[str, Any]:
    """发送 RPC 请求到对应 agent 的独立端口。"""
    task_command = create_task_command(text, command, task_id, session_id)
    request = create_rpc_request(task_command)
    base_url = _get_agent_url(agent_name)

    response = client.post(
        f"{base_url}/rpc",
        json=request,
        timeout=30.0,
    )

    return {
        "status_code": response.status_code,
        "data": response.json() if response.status_code == 200 else None,
        "raw_response": response,
    }


def poll_task_state(
    client: httpx.Client,
    agent_name: str,
    task_id: str,
    session_id: str,
    target_states: List[str],
    max_time: float = MAX_POLL_TIME,
    poll_interval: float = POLL_INTERVAL,
) -> Dict[str, Any]:
    """
    轮询任务状态直到达到目标状态或超时。

    Returns:
        {
            "converged": bool,
            "final_state": str,
            "final_result": dict,
            "poll_count": int,
            "elapsed_time": float,
            "state_history": List[str],
        }
    """
    start_time = time.time()
    poll_count = 0
    state_history = []
    last_state = None

    while time.time() - start_time < max_time:
        poll_count += 1
        result = send_rpc(client, agent_name, "", "get", task_id, session_id)

        if result["status_code"] != 200 or not result["data"]:
            time.sleep(poll_interval)
            continue

        task_data = result["data"].get("result", {})
        current_state = task_data.get("status", {}).get("state")

        if current_state and current_state != last_state:
            state_history.append(current_state)
            last_state = current_state

        if current_state in target_states:
            return {
                "converged": True,
                "final_state": current_state,
                "final_result": result["data"],
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time,
                "state_history": state_history,
            }

        time.sleep(poll_interval)

    # 超时
    final_result = send_rpc(client, agent_name, "", "get", task_id, session_id)
    return {
        "converged": False,
        "final_state": last_state,
        "final_result": final_result.get("data"),
        "poll_count": poll_count,
        "elapsed_time": time.time() - start_time,
        "state_history": state_history,
    }


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def http_client():
    """提供 HTTP 客户端（模块级别）"""
    with httpx.Client(timeout=30.0) as client:
        yield client


@pytest.fixture(scope="module")
def available_agents():
    """返回可用的 Agent 列表"""
    return ONLINE_AGENTS


@pytest.fixture
def unique_ids():
    """生成唯一的 task_id 和 session_id"""
    timestamp = datetime.now(BEIJING_TZ).timestamp()
    return {
        "task_id": f"e2e-task-{timestamp}",
        "session_id": f"e2e-session-{timestamp}",
    }


@pytest.fixture
def rpc_helper(http_client):
    """提供 RPC 调用辅助函数"""

    class RpcHelper:
        def __init__(self, client):
            self.client = client

        def start(self, agent_name: str, text: str, task_id: str, session_id: str):
            """发送 START 命令"""
            return send_rpc(self.client, agent_name, text, "start", task_id, session_id)

        def get(self, agent_name: str, task_id: str, session_id: str):
            """发送 GET 命令"""
            return send_rpc(self.client, agent_name, "", "get", task_id, session_id)

        def continue_task(
            self, agent_name: str, text: str, task_id: str, session_id: str
        ):
            """发送 CONTINUE 命令"""
            return send_rpc(
                self.client, agent_name, text, "continue", task_id, session_id
            )

        def complete(self, agent_name: str, task_id: str, session_id: str):
            """发送 COMPLETE 命令"""
            return send_rpc(
                self.client, agent_name, "确认完成", "complete", task_id, session_id
            )

        def cancel(self, agent_name: str, task_id: str, session_id: str):
            """发送 CANCEL 命令"""
            return send_rpc(self.client, agent_name, "", "cancel", task_id, session_id)

        def poll_until(
            self,
            agent_name: str,
            task_id: str,
            session_id: str,
            target_states: List[str],
            max_time: float = MAX_POLL_TIME,
        ):
            """轮询直到达到目标状态"""
            return poll_task_state(
                self.client, agent_name, task_id, session_id, target_states, max_time
            )

    return RpcHelper(http_client)
