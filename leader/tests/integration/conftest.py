"""
Leader Agent Platform - 集成测试配置

本模块提供集成测试的 fixture，包括：
- FastAPI TestClient
- 测试应用工厂
- Session Manager 访问
- 辅助函数
"""

import pytest
import pytest_asyncio
import uuid
from typing import Generator, Any, Optional, Dict
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

# 确保使用 pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def anyio_backend():
    """指定 asyncio 作为异步后端。"""
    return "asyncio"


def create_test_app() -> FastAPI:
    """
    创建测试应用实例。

    每次调用创建新的应用实例，避免测试间状态污染。
    """
    import sys
    import os

    # 确保 leader 目录在 path 中
    leader_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if leader_dir not in sys.path:
        sys.path.insert(0, leader_dir)

    # 确保项目根目录在 path 中（用于导入 acps_sdk.aip）
    project_root = os.path.dirname(leader_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 重新导入以获取新实例
    from assistant.api import router, init_routes
    from assistant.core import (
        Planner,
        SessionManager,
        create_intent_analyzer,
        create_orchestrator,
    )
    from assistant.core.history_compressor import HistoryCompressor
    from assistant.services import ScenarioLoader
    from assistant.llm import get_llm_client

    # 创建新的组件实例
    session_manager = SessionManager()
    scenario_loader = ScenarioLoader()
    intent_analyzer = create_intent_analyzer(scenario_loader)
    planner = Planner(scenario_loader)

    # 创建 history_compressor
    llm_client = get_llm_client()
    history_compressor = HistoryCompressor(
        llm_client=llm_client,
        scenario_loader=scenario_loader,
    )

    orchestrator = create_orchestrator(
        session_manager=session_manager,
        scenario_loader=scenario_loader,
        intent_analyzer=intent_analyzer,
        planner=planner,
        history_compressor=history_compressor,
    )

    # 创建新的 FastAPI 应用
    app = FastAPI(title="Leader Agent Test")

    # 初始化路由
    init_routes(orchestrator, session_manager)
    app.include_router(router)

    # 保存组件引用到 app.state
    app.state.session_manager = session_manager
    app.state.scenario_loader = scenario_loader
    app.state.orchestrator = orchestrator

    return app


@pytest.fixture
def app() -> FastAPI:
    """获取测试应用实例。"""
    return create_test_app()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncClient:
    """
    获取异步测试客户端。

    使用 httpx.AsyncClient 发送 HTTP 请求。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def session_manager(app: FastAPI):
    """获取 Session Manager 引用。"""
    return app.state.session_manager


@pytest.fixture
def scenario_loader(app: FastAPI):
    """获取 Scenario Loader 引用。"""
    return app.state.scenario_loader


@pytest.fixture
def orchestrator(app: FastAPI):
    """获取 Orchestrator 引用。"""
    return app.state.orchestrator


# =============================================================================
# 辅助函数：构建标准 API 请求
# =============================================================================


def build_submit_request(
    query: str,
    session_id: Optional[str] = None,
    mode: str = "direct_rpc",
    client_request_id: Optional[str] = None,
    active_task_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    构建符合新 API 格式的 /submit 请求体。

    Args:
        query: 用户输入
        session_id: 会话 ID（可选）
        mode: 执行模式 (direct_rpc 或 async_callback)
        client_request_id: 客户端请求 ID（不传则自动生成）
        active_task_id: 乐观锁任务 ID（可选）
        user_id: 用户 ID（可选）

    Returns:
        符合 SubmitRequest 的字典
    """
    payload = {
        "query": query,
        "mode": mode,
        "clientRequestId": client_request_id or f"test-{uuid.uuid4().hex[:8]}",
    }
    if session_id:
        payload["sessionId"] = session_id
    if active_task_id:
        payload["activeTaskId"] = active_task_id
    if user_id:
        payload["userId"] = user_id
    return payload


def extract_session_id(response_data: dict) -> str:
    """
    从 API 响应中提取 session_id。

    新格式：response.result.sessionId
    """
    if "result" in response_data and response_data["result"]:
        return response_data["result"]["sessionId"]
    raise ValueError(f"Cannot extract session_id from response: {response_data}")


def extract_active_task_id(response_data: dict) -> Optional[str]:
    """
    从 API 响应中提取 active_task_id。

    新格式：response.result.activeTaskId
    """
    if "result" in response_data and response_data["result"]:
        return response_data["result"].get("activeTaskId")
    return None


def is_success_response(response_data: dict) -> bool:
    """检查 API 响应是否成功。"""
    return "result" in response_data and response_data["result"] is not None


def is_error_response(response_data: dict) -> bool:
    """检查 API 响应是否为错误。"""
    return "error" in response_data and response_data["error"] is not None


def get_error_code(response_data: dict) -> Optional[int]:
    """从错误响应中提取错误码。"""
    if is_error_response(response_data):
        return response_data["error"].get("code")
    return None


def get_error_message(response_data: dict) -> Optional[str]:
    """从错误响应中提取错误消息。"""
    if is_error_response(response_data):
        return response_data["error"].get("message")
    return None
