"""
Leader Agent Platform - API Routes

本模块实现 HTTP API 端点：
- POST /submit: 提交用户输入
- GET /result/{session_id}: 获取任务结果（支持异步执行模式）
- GET /result/{session_id}/{task_id}: 获取指定任务的执行结果
- GET /log/{session_id}: 获取事件日志
- POST /cancel/{session_id}: 取消任务

异步执行模式：/submit 在任务编排之后就返回，后续的任务执行和结果返回是异步的。
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from .schemas import (
    CancelRequest,
    CancelResponse,
    CommonError,
    LeaderResult,
    LogResponse,
    ResultResponse,
    ScenarioRuntimeView,
    SubmitRequest,
    SubmitResponse,
)
from ..models import SessionId, SessionStatus, ActiveTaskId, SessionClosedReason
from ..models.exceptions import (
    LeaderAgentError,
    LLMCallError,
    LLMParseError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionClosedError,
    ModeMismatchError,
    ActiveTaskMismatchError,
    DuplicateRequestError,
)
from ..core.task_execution_manager import (
    TaskExecutionManager,
    get_task_execution_manager,
)
from ..models.task_execution import TaskExecutionStatus

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from ..core import Orchestrator, SessionManager

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["leader"])

# 全局组件引用（由 main.py 注入）
_orchestrator: Optional[Any] = None
_session_manager: Optional[Any] = None
_task_execution_manager: Optional[TaskExecutionManager] = None


def init_routes(
    orchestrator: "Orchestrator", session_manager: "SessionManager"
) -> None:
    """
    初始化路由组件。

    Args:
        orchestrator: 主编排器
        session_manager: Session 管理器
    """
    global _orchestrator, _session_manager, _task_execution_manager
    _orchestrator = orchestrator
    _session_manager = session_manager
    _task_execution_manager = get_task_execution_manager()


def _get_orchestrator() -> "Orchestrator":
    """获取编排器实例。"""
    if _orchestrator is None:
        raise RuntimeError("Routes not initialized. Call init_routes first.")
    return _orchestrator


def _get_session_manager() -> "SessionManager":
    """获取 Session 管理器实例。"""
    if _session_manager is None:
        raise RuntimeError("Routes not initialized. Call init_routes first.")
    return _session_manager


def _get_task_execution_manager() -> TaskExecutionManager:
    """获取任务执行管理器实例。"""
    global _task_execution_manager
    if _task_execution_manager is None:
        _task_execution_manager = get_task_execution_manager()
    return _task_execution_manager


def _session_to_leader_result(session: Any) -> LeaderResult:
    """
    将 Session 转换为 LeaderResult（API 响应视图）。

    LeaderResult 是 Session 的公开视图，不包含 eventLog 和 prompts 等内部配置。
    """
    # 转换 baseScenario
    base_scenario_view = ScenarioRuntimeView(
        id=session.base_scenario.id,
        kind=session.base_scenario.kind,
        version=session.base_scenario.version,
        loaded_at=session.base_scenario.loaded_at,
        source_path=session.base_scenario.source_path,
        config_digest=session.base_scenario.config_digest,
    )

    # 转换 expertScenario（如果存在）
    expert_scenario_view = None
    if session.expert_scenario:
        expert_scenario_view = ScenarioRuntimeView(
            id=session.expert_scenario.id,
            kind=session.expert_scenario.kind,
            version=session.expert_scenario.version,
            loaded_at=session.expert_scenario.loaded_at,
            source_path=session.expert_scenario.source_path,
            config_digest=session.expert_scenario.config_digest,
        )

    return LeaderResult(
        session_id=session.session_id,
        mode=session.mode,
        user_id=session.user_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        touched_at=session.touched_at,
        ttl_seconds=session.ttl_seconds,
        expires_at=session.expires_at,
        closed=session.closed,
        closed_at=session.closed_at,
        closed_reason=session.closed_reason.value if session.closed_reason else None,
        group_id=session.group_id,
        group_routing=None,  # 脱敏，不暴露内部路由信息
        base_scenario=base_scenario_view,
        expert_scenario=expert_scenario_view,
        scenario_briefs=session.scenario_briefs,
        active_task=session.active_task,
        partners=session.partners,
        user_context=session.user_context,
        dialog_context=session.dialog_context,
        user_result=session.user_result,
    )


# =============================================================================
# POST /submit - 提交用户输入
# =============================================================================


@router.post(
    "/submit",
    response_model=SubmitResponse,
    responses={
        400: {"model": CommonError, "description": "请求参数错误"},
        404: {"model": CommonError, "description": "Session 不存在"},
        500: {"model": CommonError, "description": "服务器内部错误"},
    },
    summary="提交用户输入",
    description="提交用户输入到 Leader Agent，触发意图分析和任务处理。",
)
async def submit(request: SubmitRequest) -> SubmitResponse:
    """
    提交用户输入。

    流程：
    1. 获取或创建 Session（含 mode/幂等性/activeTaskId 校验）
    2. 调用 LLM-1 分析意图
    3. 根据意图类型执行对应处理
    4. 返回响应
    """
    try:
        orchestrator = _get_orchestrator()
        response = await orchestrator.handle_submit(request)
        return response

    except SessionNotFoundError as e:
        logger.warning(f"Session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": 404001, "message": str(e)},
        )

    except SessionExpiredError as e:
        logger.warning(f"Session expired: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": 404001, "message": str(e)},
        )

    except SessionClosedError as e:
        logger.warning(f"Session closed: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.to_dict(),
        )

    except ModeMismatchError as e:
        logger.warning(f"Mode mismatch: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.to_dict(),
        )

    except ActiveTaskMismatchError as e:
        logger.warning(f"Active task mismatch: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.to_dict(),
        )

    except DuplicateRequestError as e:
        logger.warning(f"Duplicate request with different payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.to_dict(),
        )

    except LLMCallError as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "LLM_CALL_ERROR", "message": str(e)},
        )

    except LLMParseError as e:
        logger.error(f"LLM parse failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "LLM_PARSE_ERROR", "message": str(e)},
        )

    except LeaderAgentError as e:
        logger.error(f"Leader agent error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "LEADER_ERROR", "message": str(e)},
        )

    except Exception as e:
        logger.exception(f"Unexpected error in submit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "服务器内部错误"},
        )


# =============================================================================
# GET /result/{session_id} - 获取任务结果（异步执行模式）
# =============================================================================


@router.get(
    "/result/{session_id}",
    response_model=ResultResponse,
    responses={
        404: {"model": CommonError, "description": "Session 不存在"},
    },
    summary="获取任务结果",
    description="""
获取指定 Session 的当前任务结果。

**异步执行模式说明**：
- /submit 在任务编排之后就返回 pending 状态
- 客户端通过此 API 轮询获取任务执行状态和最终结果
- 支持通过 task_id 查询参数指定查询某个特定任务

**返回状态说明**：
- `pending`: 任务已创建，等待执行
- `running`: 任务正在执行中
- `awaiting_input`: 需要用户补充信息
- `completed`: 任务执行完成
- `failed`: 任务执行失败
- `cancelled`: 任务已取消
""",
)
async def get_result(
    session_id: SessionId,
    task_id: Optional[ActiveTaskId] = Query(
        default=None,
        alias="taskId",
        description="可选的任务 ID，不指定则返回最新任务的状态",
    ),
) -> ResultResponse:
    """
    获取任务结果（支持异步执行模式）。

    返回 LeaderResult 视图，包含当前 Session 的状态和结果。

    响应格式:
    {
        "result": LeaderResult | null,
        "error": CommonError | null
    }
    """
    try:
        session_manager = _get_session_manager()

        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )

        # 将 Session 转换为 LeaderResult 视图
        leader_result = _session_to_leader_result(session)

        return ResultResponse(result=leader_result)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error getting result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "服务器内部错误"},
        )


# =============================================================================
# GET /log/{session_id} - 获取事件日志
# =============================================================================


@router.get(
    "/log/{session_id}",
    response_model=LogResponse,
    responses={
        404: {"model": CommonError, "description": "Session 不存在"},
    },
    summary="获取事件日志",
    description="获取指定 Session 的事件日志。",
)
async def get_log(
    session_id: SessionId,
    limit: int = 100,
) -> LogResponse:
    """
    获取事件日志。

    返回 Session 的事件日志列表。
    """
    try:
        session_manager = _get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )

        # 获取事件日志
        events = session.event_log[-limit:] if limit > 0 else session.event_log

        return LogResponse(
            session_id=session_id,
            events=[
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "event_data": e.event_data,
                }
                for e in events
            ],
            total_count=len(session.event_log),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error getting log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "服务器内部错误"},
        )


# =============================================================================
# POST /cancel/{session_id} - 取消任务
# =============================================================================


@router.post(
    "/cancel/{session_id}",
    response_model=CancelResponse,
    responses={
        404: {"model": CommonError, "description": "Session 不存在"},
    },
    summary="取消任务",
    description="取消指定 Session 的当前活跃任务。",
)
async def cancel_task(
    session_id: SessionId,
    request: Optional[CancelRequest] = None,
) -> CancelResponse:
    """
    取消任务。

    取消当前活跃任务，并将 Session 设置为已关闭状态。
    /cancel 成功后 Session 将进入关闭态（closed=true）。
    """
    try:
        from datetime import datetime, timezone

        session_manager = _get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )

        # 取消活跃任务并尝试通知 Partner
        cancelled_tasks = []
        if session.active_task:
            cancelled_tasks.append(session.active_task.active_task_id)

            # 尽最大努力终止该 Session 关联的在途推进
            # 向各非终态 Partner 下发 AIP cancel
            try:
                orchestrator = _get_orchestrator()
                if orchestrator and hasattr(orchestrator, "_executor"):
                    from ..models.aip import TaskState

                    for (
                        partner_aic,
                        partner_task,
                    ) in session.active_task.partner_tasks.items():
                        # 只对非终态的 Partner 发送 cancel
                        terminal_states = {
                            TaskState.Completed,
                            TaskState.Failed,
                            TaskState.Rejected,
                            TaskState.Canceled,
                        }
                        if partner_task.state not in terminal_states:
                            # 获取端点
                            partner_runtime = session.partners.get(partner_aic)
                            if partner_runtime and partner_runtime.resolved_endpoint:
                                endpoint = partner_runtime.resolved_endpoint.url
                                try:
                                    await orchestrator._executor.cancel_partner(
                                        session_id=session_id,
                                        partner_aic=partner_aic,
                                        aip_task_id=partner_task.aip_task_id,
                                        endpoint=endpoint,
                                    )
                                    logger.info(f"Sent cancel to partner {partner_aic}")
                                except Exception as cancel_err:
                                    # 记录警告但不阻塞取消流程
                                    logger.warning(
                                        f"Failed to cancel partner {partner_aic}: {cancel_err}"
                                    )
            except Exception as e:
                logger.warning(f"Error during partner cancellation: {e}")

            session.active_task = None

        # /cancel 成功后 Session 进入关闭态
        now_iso_str = datetime.now(timezone.utc).isoformat()
        session.closed = True
        session.closed_at = now_iso_str
        session.closed_reason = SessionClosedReason.USER_CANCEL

        # 如果请求删除 Session
        if request and request.delete_session:
            session_manager.delete_session(session_id)
            session_deleted = True
        else:
            session_manager.update_session(session, refresh_ttl=False)
            session_deleted = False

        return CancelResponse(
            session_id=session_id,
            success=True,
            cancelled_tasks=cancelled_tasks,
            session_deleted=session_deleted,
            message=f"取消了 {len(cancelled_tasks)} 个任务，Session 已关闭",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error cancelling task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "服务器内部错误"},
        )


# =============================================================================
# 健康检查
# =============================================================================


@router.get(
    "/health",
    summary="健康检查",
    description="检查服务是否正常运行。",
)
async def health_check():
    """健康检查端点。"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "orchestrator": _orchestrator is not None,
            "session_manager": _session_manager is not None,
        },
    }
