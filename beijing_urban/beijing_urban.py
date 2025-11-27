import os
import asyncio
from typing import Callable, Awaitable
import json
from fastapi import FastAPI
from dotenv import load_dotenv
import openai
import uuid
from json import JSONDecodeError

import sys

_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base import (
    get_agent_logger,
    truncate,
    extract_text_from_message,
    load_capabilities_snippet_from_json,
    call_openai_chat,
)

from acps_aip.aip_rpc_server import (
    add_aip_rpc_router,
    TaskManager,
    CommandHandlers,
    DefaultHandlers,
)
from acps_aip.aip_base_model import (
    Message,
    Task,
    TaskState,
    Product,
    TextDataItem,
    TaskCommand,
)
from acps_aip.mtls_config import load_mtls_config_from_json

# 从 .env 文件加载环境变量
load_dotenv()

"""北京城区旅游规划 Agent (beijing_urban)

新增：结构化日志，帮助追踪任务全生命周期、判定决策分支与错误。
"""

# --- Agent 配置 ---
AGENT_ID = os.getenv("BEIJING_URBAN_AGENT_ID", "beijing_urban-planner-001")
AIP_ENDPOINT = os.getenv("BEIJING_URBAN_AIP_ENDPOINT", "/acps-aip-v1/rpc")

# --- 日志配置 ---
LOG_LEVEL = os.getenv("BEIJING_URBAN_LOG_LEVEL", "INFO").upper()
logger = get_agent_logger("agent.beijing_urban", "BEIJING_URBAN_LOG_LEVEL", LOG_LEVEL)

# --- OpenAI 配置 ---
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("OPENAI_MODEL", "Doubao-pro-32k")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="北京城区景点规划师 Agent",
    description="一个符合ACPs协议的，专为北京城区提供旅游规划的Agent.",
)


def _load_capabilities_snippet() -> str:
    json_path = os.path.join(os.path.dirname(__file__), "beijing_urban.json")
    fallback = (
        "职责：为北京城六区提供景点推荐、主题/多日/亲子行程规划、路线优化、文化体验建议；"
        "范围：仅限城六区；拒绝郊区/餐饮/跨城/专业预订。"
    )
    return load_capabilities_snippet_from_json(json_path, fallback)


CAPABILITIES_SNIPPET = _load_capabilities_snippet()

# --- 缺省超时/时长与产出限制（毫秒/字节） ---
# 建议覆盖第一个大模型调用的时间。可按环境调整。
DEFAULT_RESPONSE_TIMEOUT_MS = int(
    os.getenv("BEIJING_URBAN_RESPONSE_TIMEOUT_MS", "5000")
)
# 等待用户补充输入的超时（到期 -> Canceled）
DEFAULT_AWAITING_INPUT_TIMEOUT_MS = int(
    os.getenv("BEIJING_URBAN_AWAITING_INPUT_TIMEOUT_MS", "60000")
)
# 等待用户确认完成的超时（到期 -> Completed）
DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS = int(
    os.getenv("BEIJING_URBAN_AWAITING_COMPLETION_TIMEOUT_MS", "60000")
)
# 产出物总大小上限（字节），用于防止过大产品
DEFAULT_MAX_PRODUCTS_BYTES = int(
    os.getenv("BEIJING_URBAN_MAX_PRODUCTS_BYTES", "1048576")
)
# 工作阶段（LLM 等）超时（到期 -> Failed）。防止异步任务长时间挂起。
DEFAULT_WORK_TIMEOUT_MS = int(os.getenv("BEIJING_URBAN_WORK_TIMEOUT_MS", "10000"))

# --- LLM 提示词模板 ---
# 1) 接受/拒绝判定（用于 on_start）：
DECIDE_PROMPT = (
    "你是【北京城区旅游景点规划师 Agent】的请求门卫。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 只判断该请求是否属于北京城六区范围内，是否应当由本 Agent 处理。\n"
    "- 不要做需求结构化，也不要给出行程方案。\n\n"
    "[输出：严格 JSON，仅此一段]\n"
    "{\n"
    '  "decision": "accept" | "reject",\n'
    '  "reason": "string（decision=reject 必填，说明不在城六区范围或不合规）"\n'
    "}"
)

# 2) 异步阶段的需求分析，生成结构化需求数据
ANALYZE_PROMPT = (
    "你是【北京城区旅游景点规划师 Agent】的需求分析助手。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "1) 分析用户输入（可能是初始需求或补充需求），生成本 Agent 执行业务所需的结构化需求对象 requirements。\n"
    "2) 若补充信息超出城六区范围或与本 Agent 职责不符，请给出提示并将 decision 标记为 reject（用于引导继续补充）。\n\n"
    "[输出要求：必须是严格 JSON，仅此一段，无多余文本或注释]\n"
    "{\n"
    '    "decision": "accept" | "reject",\n'
    '    "reason": "string（decision=reject 必填，说明为何不在城六区范围或不合规）",\n'
    '    "requirements": {\n'
    '        "scope": "city-core-only",\n'
    '        "theme": "string|null",\n'
    '        "days": 1,\n'
    '        "preferences": ["轻体力", "亲子", "文化深度"],\n'
    '        "budgetLevel": "low|medium|high|null",\n'
    '        "mustSee": ["..."],\n'
    '        "avoid": ["..."],\n'
    '        "missingFields": ["若缺少必要信息，这里列出字段名"]\n'
    "    }\n"
    "}"
)

# 3) 通过结构化需求数据生成产出物
PRODUCE_PROMPT = (
    "你是【北京城区旅游景点规划师 Agent】的产出生成助手。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据给定的 requirements 生成最终的行程规划文本内容（纯文本，不含 JSON）。\n"
    "- 输出需结构清晰、主题明确、体力均衡、体现文化加值；严格限定在城六区范围内。\n"
)


async def decide_accept(user_text: str) -> dict:
    """仅用于 on_start 的门卫判断：accept / reject。"""
    raw = await call_openai_chat(
        [
            {"role": "system", "content": DECIDE_PROMPT},
            {"role": "user", "content": user_text or ""},
        ],
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=256,
    )
    try:
        obj = json.loads(raw)
    except JSONDecodeError:
        obj = {"decision": "accept"}
    if obj.get("decision") not in ("accept", "reject"):
        obj["decision"] = "accept"
    if obj.get("decision") == "reject" and not obj.get("reason"):
        obj["reason"] = "不满足北京城区范围或规范"
    return obj


async def analyze_requirements(
    user_text: str, previous_requirements: dict | None = None
) -> dict:
    payload = user_text
    if previous_requirements:
        payload = json.dumps(
            {"previous": previous_requirements, "supplement": user_text},
            ensure_ascii=False,
        )
    raw = await call_openai_chat(
        [
            {"role": "system", "content": ANALYZE_PROMPT},
            {"role": "user", "content": payload},
        ],
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=512,
    )
    try:
        obj = json.loads(raw)
    except JSONDecodeError:
        # 回退：若返回非 JSON，按接受且最小 requirements 处理
        obj = {
            "decision": "accept",
            "requirements": {"preferences": [], "missingFields": ["theme"]},
        }
    # 兜底字段
    if "decision" not in obj:
        obj["decision"] = "accept"
    if obj.get("decision") == "accept" and "requirements" not in obj:
        obj["requirements"] = {"preferences": [], "missingFields": ["theme"]}
    if obj.get("decision") == "reject" and "reason" not in obj:
        obj["reason"] = "不满足北京城区范围或规范"
    # 仅当 requirements 存在且为 dict 时再填充 missingFields，避免 reject 情况下 KeyError
    req = obj.get("requirements")
    if isinstance(req, dict) and "missingFields" not in req:
        req["missingFields"] = []
        obj["requirements"] = req
    return obj


async def produce_plan(requirements: dict) -> str:
    raw = await call_openai_chat(
        [
            {"role": "system", "content": PRODUCE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"requirements": requirements}, ensure_ascii=False
                ),
            },
        ],
        model=LLM_MODEL,
        temperature=0.6,
        max_tokens=1500,
    )
    return raw.strip()


# --- AIP CommandHandlers 实现 ---
async def on_start(message: Message, task: Task | None) -> Task:
    # 读取/合并 Start 参数与缺省值
    params = getattr(message, "commandParams", None) or {}
    response_timeout_ms = params.get("responseTimeout") or DEFAULT_RESPONSE_TIMEOUT_MS
    awaiting_input_timeout_ms = (
        params.get("awaitingInputTimeout") or DEFAULT_AWAITING_INPUT_TIMEOUT_MS
    )
    awaiting_completion_timeout_ms = (
        params.get("awaitingCompletionTimeout")
        or DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS
    )
    max_products_bytes = params.get("maxProductsBytes") or DEFAULT_MAX_PRODUCTS_BYTES
    work_timeout_ms = DEFAULT_WORK_TIMEOUT_MS

    user_text = extract_text_from_message(message)

    # 对 responseTimeout 进行预判（建议性）：若我们预估无法在该时间内完成 accept/reject 判定，则直接返回 Rejected
    estimated_first_llm_ms = 2000  # 经验值：首个 LLM 调用约 2s，可按需调优
    if response_timeout_ms is not None and response_timeout_ms < estimated_first_llm_ms:
        # 无法在该时间内完成决定，直接创建为 Rejected
        logger.info(
            "event=prejudge_reject task_id=%s reason=response_timeout_too_short timeout_ms=%s",
            message.taskId,
            response_timeout_ms,
        )
        rejected = TaskManager.create_task(
            message,
            initial_state=TaskState.Rejected,
            data_items=[TextDataItem(text="无法在指定 responseTimeout 内完成决策")],
        )
        return rejected

    #  做门卫判断，直接给出初态
    gate = await decide_accept(user_text)
    if gate.get("decision", "accept") == "reject":
        reason = gate.get("reason", "不满足北京城区范围或规范")
        logger.info(
            "event=state_init task_id=%s state=%s reason=gate_reject",
            message.taskId,
            TaskState.Rejected,
        )
        rejected = TaskManager.create_task(
            message,
            initial_state=TaskState.Rejected,
            data_items=[TextDataItem(text=reason)],
        )
        return rejected

    # 初态直接 Accepted，并持久化新的超时参数
    accepted = TaskManager.create_task(message, initial_state=TaskState.Accepted)
    setattr(accepted, "_aip_awaiting_input_timeout_ms", awaiting_input_timeout_ms)
    setattr(
        accepted, "_aip_awaiting_completion_timeout_ms", awaiting_completion_timeout_ms
    )
    setattr(accepted, "_aip_max_products_bytes", max_products_bytes)

    # 接受后，调度异步流水线（Working -> 分析 -> AwaitingInput/产出 -> AwaitingCompletion）
    logger.info(
        "event=job_schedule task_id=%s work_timeout_ms=%s",
        accepted.id,
        str(work_timeout_ms),
    )
    UrbanJobManager.start_job(
        accepted.id,
        lambda cancel_event: _run_urban_pipeline(
            accepted.id, user_text, cancel_event, work_timeout_ms
        ),
    )

    # 立即返回（保持 Accepted），后台接手
    return TaskManager.get_task(accepted.id)


async def on_cancel(message: Message, task: Task) -> Task:
    # 先尝试取消后台作业，再应用默认取消语义
    UrbanJobManager.cancel_job(task.id)
    return await DefaultHandlers.cancel(message, task)


async def on_continue(message: Message, task: Task) -> Task:
    # 统一追加消息到历史
    TaskManager.add_message_to_history(task.id, message)

    # 仅在 AwaitingInput / AwaitingCompletion 下处理；需要非空文本
    if task.status.state not in (TaskState.AwaitingInput, TaskState.AwaitingCompletion):
        logger.info(
            "event=continue_ignored task_id=%s state=%s", task.id, task.status.state
        )
        return task
    user_text = extract_text_from_message(message)
    if not user_text.strip():
        logger.info("event=continue_missing_text task_id=%s", task.id)
        return task

    # 调度异步流水线（Working -> 分析 -> AwaitingInput/产出 -> AwaitingCompletion）
    work_timeout_ms = DEFAULT_WORK_TIMEOUT_MS
    logger.info(
        "event=job_schedule task_id=%s work_timeout_ms=%s via=continue",
        task.id,
        str(work_timeout_ms),
    )
    UrbanJobManager.start_job(
        task.id,
        lambda cancel_event: _run_urban_pipeline(
            task.id, user_text, cancel_event, work_timeout_ms
        ),
    )
    return TaskManager.get_task(task.id)


# ---------------- Agent 内部：后台任务管理器（基于 asyncio） ----------------
class UrbanJobManager:
    _jobs: dict[str, dict] = {}
    _await_timers: dict[str, asyncio.Task] = {}

    @classmethod
    def start_job(
        cls,
        task_id: str,
        coro_factory: Callable[[asyncio.Event], Awaitable[None]],
    ) -> bool:
        """启动任务的后台协程（若同 task 已有在跑的作业，则不重复启动）。

        coro_factory: 接受一个 asyncio.Event（取消标记），返回要运行的协程对象。
        返回 True 表示已启动，False 表示已有任务在运行。
        """
        # 若已存在未完成的任务，直接返回 False（幂等保护）
        job = cls._jobs.get(task_id)
        if job and not job["task"].done():
            return False

        cancel_event = asyncio.Event()

        async def _runner():
            try:
                await coro_factory(cancel_event)
            except asyncio.CancelledError:
                # 被取消。若任务未在终态/取消态，则置为 Failed，避免停留在 Working
                logger.info("event=job_cancelled task_id=%s", task_id)
                current = TaskManager.get_task(task_id)
                if current and current.status.state not in (
                    TaskState.Canceled,
                    TaskState.Failed,
                    TaskState.Completed,
                    TaskState.Rejected,
                ):
                    TaskManager.update_task_status(
                        task_id,
                        TaskState.Failed,
                        data_items=[TextDataItem(text="后台执行被取消或超时")],
                    )
                # 不再向外抛出，正常结束
            except Exception as e:  # 兜底：将任务置为 Failed
                logger.exception("event=job_exception task_id=%s", task_id)
                TaskManager.update_task_status(
                    task_id,
                    TaskState.Failed,
                    data_items=[TextDataItem(text=f"后台执行异常: {str(e)}")],
                )
            finally:
                # 清理登记
                cls._jobs.pop(task_id, None)

        t = asyncio.create_task(_runner(), name=f"urban-job-{task_id}")
        cls._jobs[task_id] = {"task": t, "cancel_event": cancel_event}
        logger.info("event=job_started task_id=%s", task_id)
        return True

    @classmethod
    def cancel_job(cls, task_id: str) -> None:
        job = cls._jobs.get(task_id)
        if not job:
            return
        cancel_event: asyncio.Event = job["cancel_event"]
        cancel_event.set()
        task_obj: asyncio.Task = job["task"]
        if not task_obj.done():
            task_obj.cancel()
        logger.info("event=job_cancel_signal_sent task_id=%s", task_id)
        # 同时取消等待阶段定时器
        t = cls._await_timers.pop(task_id, None)
        if t and not t.done():
            t.cancel()

    @classmethod
    def schedule_await_timeout(
        cls, task_id: str, state: TaskState, timeout_ms: int | None
    ) -> None:
        """安排等待阶段的超时：
        - AwaitingInput 到期 -> Canceled (AIP 表 10)
        - AwaitingCompletion 到期 -> Completed (AIP 表 14)
        """
        # 先清理旧的
        prev = cls._await_timers.pop(task_id, None)
        if prev and not prev.done():
            prev.cancel()
        if not timeout_ms or timeout_ms <= 0:
            return

        async def _wait_then_transition():
            try:
                await asyncio.sleep(timeout_ms / 1000.0)
                task = TaskManager.get_task(task_id)
                if not task:
                    return
                # 仅当仍在该等待状态时才执行转移
                if (
                    state == TaskState.AwaitingInput
                    and task.status.state == TaskState.AwaitingInput
                ):
                    logger.info(
                        "event=await_timeout task_id=%s from=%s to=%s",
                        task_id,
                        TaskState.AwaitingInput,
                        TaskState.Canceled,
                    )
                    TaskManager.update_task_status(task_id, TaskState.Canceled)
                elif (
                    state == TaskState.AwaitingCompletion
                    and task.status.state == TaskState.AwaitingCompletion
                ):
                    logger.info(
                        "event=await_timeout task_id=%s from=%s to=%s",
                        task_id,
                        TaskState.AwaitingCompletion,
                        TaskState.Completed,
                    )
                    TaskManager.update_task_status(task_id, TaskState.Completed)
            except asyncio.CancelledError:
                pass

        cls._await_timers[task_id] = asyncio.create_task(
            _wait_then_transition(), name=f"await-timeout-{task_id}"
        )


async def _run_urban_pipeline(
    task_id: str,
    user_text: str,
    cancel_event: asyncio.Event,
    timeout_ms: int | None,
) -> None:
    """后台执行阶段：Working -> (AwaitingInput | AwaitingCompletion | Failed)。"""

    async def _work():
        if cancel_event.is_set():
            return
        # 进入 Working
        current = TaskManager.get_task(task_id)
        logger.info(
            "event=state_transition task_id=%s from=%s to=%s",
            task_id,
            getattr(current.status, "state", None),
            TaskState.Working,
        )
        TaskManager.update_task_status(task_id, TaskState.Working)

        # 结构化分析（合并历史 requirements）
        prev = getattr(current, "_urban_requirements", None)
        logger.info("event=llm_analyze_start task_id=%s", task_id)
        analysis = await analyze_requirements(
            user_text or "", previous_requirements=prev
        )
        # 如果补充信息超范围，则提示继续补充
        if analysis.get("decision", "accept") == "reject":
            guidance = analysis.get(
                "reason", "补充信息超出城区范围，请提供城六区内需求"
            )
            logger.info(
                "event=state_transition task_id=%s from=%s to=%s reason=analysis_reject",
                task_id,
                TaskState.Working,
                TaskState.AwaitingInput,
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            # 安排等待输入阶段超时
            t = TaskManager.get_task(task_id)
            timeout_ms = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            UrbanJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms
            )
            return

        requirements = analysis.get("requirements", {})
        # 持久化到任务对象（用于后续 Continue 合并）
        setattr(current, "_urban_requirements", requirements)

        # 判断是否缺少必要信息
        missing = requirements.get("missingFields") or []
        if isinstance(missing, list) and len(missing) > 0:
            guidance = "缺少必要信息: " + ",".join(map(str, missing))
            logger.info(
                "event=state_transition task_id=%s from=%s to=%s reason=missing_fields fields=%s",
                task_id,
                TaskState.Working,
                TaskState.AwaitingInput,
                ",".join(map(str, missing)),
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            # 安排等待输入阶段超时
            t = TaskManager.get_task(task_id)
            timeout_ms = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            UrbanJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms
            )
            return

        # 生成产出
        logger.info("event=llm_produce_start task_id=%s", task_id)
        plan = await produce_plan(requirements)
        if cancel_event.is_set():
            return
        product = Product(
            id=f"product-{uuid.uuid4()}",
            name="北京城区旅游规划",
            dataItems=[TextDataItem(text=plan)],
        )
        TaskManager.set_products(task_id, [product])
        latest = TaskManager.get_task(task_id)
        if latest and latest.status.state == TaskState.Failed:
            # 产品过大导致的失败不应被覆盖
            return
        logger.info(
            "event=state_transition task_id=%s from=%s to=%s",
            task_id,
            TaskState.Working,
            TaskState.AwaitingCompletion,
        )
        TaskManager.update_task_status(task_id, TaskState.AwaitingCompletion)
        # 安排等待确认阶段超时
        t2 = TaskManager.get_task(task_id)
        timeout_ms2 = getattr(
            t2,
            "_aip_awaiting_completion_timeout_ms",
            DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS,
        )
        UrbanJobManager.schedule_await_timeout(
            task_id, TaskState.AwaitingCompletion, timeout_ms2
        )

    try:
        if timeout_ms and timeout_ms > 0:
            await asyncio.wait_for(_work(), timeout=timeout_ms / 1000.0)
        else:
            await _work()
    except asyncio.TimeoutError:
        logger.error(
            "event=state_transition task_id=%s to=%s reason=timeout",
            task_id,
            TaskState.Failed,
        )
        TaskManager.update_task_status(
            task_id,
            TaskState.Failed,
            [TextDataItem(text="任务执行超时")],
        )


handlers = CommandHandlers(
    on_start=on_start,  # Start：分析并决定接受/拒绝；接受则后台执行
    on_continue=on_continue,  # Continue：补充输入 -> 再分析 -> 后台执行
    on_cancel=on_cancel,  # Cancel：真正取消后台作业后，按协议置 Canceled
)

# --- 将 AIP RPC 路由添加到 FastAPI 应用中 ---
add_aip_rpc_router(app, AIP_ENDPOINT, handlers)
logger.info(
    "event=app_start agent_id=%s endpoint=%s model=%s log_level=%s",
    AGENT_ID,
    AIP_ENDPOINT,
    LLM_MODEL,
    LOG_LEVEL,
)


# --- 可选：添加一个根路径用于健康检查 ---
@app.get("/")
def read_root():
    return {"message": f"欢迎使用 {AGENT_ID}. AIP 协议端点位于 {AIP_ENDPOINT}."}


if __name__ == "__main__":
    import uvicorn
    import ssl

    # 加载mTLS配置
    json_path = os.path.join(os.path.dirname(__file__), "beijing_urban.json")
    mtls_config = load_mtls_config_from_json(json_path)

    logger.info(
        "event=server_start host=0.0.0.0 port=8011 mtls=enabled aic=%s", mtls_config.aic
    )

    uvicorn.run(
        "beijing_urban:app",
        host="0.0.0.0",
        port=8011,
        reload=True,
        workers=1,
        ssl_keyfile=str(mtls_config.key_file),
        ssl_certfile=str(mtls_config.cert_file),
        ssl_ca_certs=str(mtls_config.ca_cert_file),
        ssl_cert_reqs=ssl.CERT_REQUIRED,  # 如果在测试环境中客户端没有证书，请注释这一行。
    )
