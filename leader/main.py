"""
Leader Agent Platform - Application Entry Point

本模块是 Leader Agent 的应用入口，负责：
- 初始化所有核心组件
- 配置 FastAPI 应用
- 启动/停止生命周期管理

使用方法：
    # 从 leader 目录运行：
    uvicorn main_v2:app --host 0.0.0.0 --port 59210 --reload

    # 或者从 demo-apps 目录运行：
    uvicorn leader.main_v2:app --host 0.0.0.0 --port 59210 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from assistant.api import init_routes, router
from assistant.config import settings
from assistant.core import (
    Orchestrator,
    SessionManager,
    create_intent_analyzer,
    create_orchestrator,
    get_session_manager,
    GroupManager,
    GroupTaskExecutor,
    create_group_manager,
    GroupConfig,
    RabbitMQConfig,
)
from assistant.models.exceptions import LeaderAgentError
from assistant.services import ScenarioLoader
from assistant.core.orchestrator import _build_client_ssl_context

# 配置日志
log_level = settings.get("logging", {}).get("level", "INFO")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 配置包级别日志
for package, level in settings.get("logging", {}).get("packages", {}).items():
    logging.getLogger(package).setLevel(level)

logger = logging.getLogger(__name__)

# 全局组件引用
_session_manager: Optional[SessionManager] = None
_scenario_loader: Optional[ScenarioLoader] = None
_orchestrator: Optional[Orchestrator] = None
_group_manager: Optional[GroupManager] = None


async def init_components() -> None:
    """初始化所有核心组件。"""
    global _session_manager, _scenario_loader, _orchestrator, _group_manager

    logger.info("Initializing Leader Agent components...")

    # 1. 初始化 Session Manager
    _session_manager = get_session_manager()
    await _session_manager.start()
    logger.info("Session Manager initialized")

    # 2. 初始化 Scenario Loader
    _scenario_loader = ScenarioLoader()
    scenario_count = len(_scenario_loader.scenario_briefs)
    logger.info(
        f"Scenario Loader initialized: {scenario_count} expert scenarios discovered"
    )

    # 3. 初始化 Intent Analyzer
    intent_analyzer = create_intent_analyzer(_scenario_loader)
    logger.info("Intent Analyzer (LLM-1) initialized")

    # 4. 初始化群组组件（如果启用）
    group_executor = None
    group_config = settings.get("group", {})
    if group_config.get("enabled", False):
        logger.info("Group Mode enabled, initializing group components...")

        # 获取 RabbitMQ 配置
        rabbitmq_config = settings.get("rabbitmq", {})
        rabbitmq_mgmt_config = settings.get("rabbitmq", {}).get("management", {})

        # 获取 leader_aic
        leader_aic = settings.get("app", {}).get("leader_aic", "unknown-leader")

        # 构建 SSL 上下文用于群组 invite 调用
        ssl_context = _build_client_ssl_context(settings)

        # 创建 GroupManager
        _group_manager = create_group_manager(
            leader_aic=leader_aic,
            rabbitmq_config=RabbitMQConfig(
                host=rabbitmq_config.get("host", "localhost"),
                port=rabbitmq_config.get("port", 5672),
                user=rabbitmq_config.get("user", "guest"),
                password=rabbitmq_config.get("password", "guest"),
                vhost=rabbitmq_config.get("vhost", "/"),
                management_host=rabbitmq_mgmt_config.get("host", "localhost"),
                management_port=rabbitmq_mgmt_config.get("port", 15672),
            ),
            group_config=GroupConfig(
                status_probe_interval=group_config.get("status_probe_interval", 30),
                max_wait_seconds=group_config.get("max_wait_seconds", 300),
                partner_join_timeout=group_config.get("partner_join_timeout", 60),
                max_retry_count=group_config.get("max_retry_count", 3),
            ),
            ssl_context=ssl_context,
        )
        await _group_manager.start()
        logger.info("GroupManager initialized and started")

        # 设置 SessionManager 的 GroupManager 引用
        _session_manager.set_group_manager(_group_manager)

        # 创建 GroupTaskExecutor
        group_executor = GroupTaskExecutor(
            leader_aic=leader_aic,
            group_manager=_group_manager,
        )
        logger.info("GroupTaskExecutor initialized")
    else:
        logger.info("Group Mode disabled")

    # 5. 初始化 Orchestrator
    _orchestrator = create_orchestrator(
        session_manager=_session_manager,
        scenario_loader=_scenario_loader,
        intent_analyzer=intent_analyzer,
        group_executor=group_executor,
        group_manager=_group_manager,
    )
    # 启动 Orchestrator，初始化所有懒加载组件（包括 HistoryCompressor）
    await _orchestrator.start()
    logger.info("Orchestrator initialized and started")

    # 6. 初始化路由
    init_routes(_orchestrator, _session_manager)
    logger.info("API routes initialized")

    logger.info("All Leader Agent components initialized successfully")


async def shutdown_components() -> None:
    """关闭所有核心组件。"""
    global _session_manager, _group_manager

    logger.info("Shutting down Leader Agent components...")

    if _session_manager:
        await _session_manager.stop()
        logger.info("Session Manager stopped")

    if _group_manager:
        await _group_manager.stop()
        logger.info("GroupManager stopped")

    logger.info("All Leader Agent components shut down")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。"""
    # 启动
    await init_components()
    yield
    # 关闭
    await shutdown_components()


# 创建 FastAPI 应用
app = FastAPI(
    title="Leader Agent Platform",
    description="智能协作平台的中枢调度服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=False,  # 与 allow_origins=["*"] 配合使用
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(LeaderAgentError)
async def leader_agent_exception_handler(
    request: Request,
    exc: LeaderAgentError,
) -> JSONResponse:
    """处理 Leader Agent 业务异常。"""
    # 从错误码提取 HTTP 状态码
    try:
        status_code = int(str(exc.code)[:3])
    except (ValueError, TypeError):
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


# 注册路由
app.include_router(router)


# 根路径
@app.get("/")
async def root():
    """根路径信息。"""
    return {
        "service": "Leader Agent Platform",
        "version": "1.0.0",
        "status": "running",
        "api_docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    host = settings.get("uvicorn", {}).get("host", "0.0.0.0")
    port = settings.get("uvicorn", {}).get("port", 59210)
    reload = settings.get("uvicorn", {}).get("reload", False)

    uvicorn.run(
        "leader.main:app",
        host=host,
        port=port,
        reload=reload,
    )
