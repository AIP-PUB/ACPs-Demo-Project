import os
import sys
import json
import uuid
import asyncio
from typing import Callable, Awaitable
from json import JSONDecodeError
from fastapi import FastAPI
from dotenv import load_dotenv
import openai

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

# ============================
# 环境加载
# ============================
load_dotenv()

# --- Agent 配置 ---
AGENT_ID = os.getenv("BEIJING_CATERING_AGENT_ID", "beijing_catering_agent_001")
AIP_ENDPOINT = os.getenv("BEIJING_CATERING_AIP_ENDPOINT", "/acps-aip-v1/rpc")
LOG_LEVEL = os.getenv("BEIJING_CATERING_LOG_LEVEL", "INFO").upper()

logger = get_agent_logger(
    "agent.beijing_catering", "BEIJING_CATERING_LOG_LEVEL", LOG_LEVEL
)

# --- OpenAI 配置 ---
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("OPENAI_MODEL", "Doubao-pro-32k")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="北京美食推荐 Agent",
    description="一个符合ACPs协议的，专为北京本地美食与餐厅匹配提供规划建议的Agent.",
)


def _load_capabilities_snippet() -> str:
    json_path = os.path.join(os.path.dirname(__file__), "beijing_catering.json")
    fallback = (
        "职责：根据口味偏好、预算与动线提供北京本地美食与餐厅建议；"
        "范围：仅限北京餐饮推荐；拒绝非餐饮/跨城/行程规划等请求。"
    )
    return load_capabilities_snippet_from_json(json_path, fallback)


CAPABILITIES_SNIPPET = _load_capabilities_snippet()

# --- 缺省超时/时长与产出限制（毫秒/字节） ---
DEFAULT_RESPONSE_TIMEOUT_MS = int(
    os.getenv("BEIJING_CATERING_RESPONSE_TIMEOUT_MS", "5000")
)
DEFAULT_AWAITING_INPUT_TIMEOUT_MS = int(
    os.getenv("BEIJING_CATERING_AWAITING_INPUT_TIMEOUT_MS", "60000")
)
DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS = int(
    os.getenv("BEIJING_CATERING_AWAITING_COMPLETION_TIMEOUT_MS", "60000")
)
DEFAULT_MAX_PRODUCTS_BYTES = int(
    os.getenv("BEIJING_CATERING_MAX_PRODUCTS_BYTES", "1048576")
)
DEFAULT_WORK_TIMEOUT_MS = int(os.getenv("BEIJING_CATERING_WORK_TIMEOUT_MS", "10000"))

# --- 提示词模板 ---
DECIDE_PROMPT = (
    "你是【北京美食推荐 Agent】的请求门卫。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 只判断该请求是否属于北京餐饮推荐范围，是否应由本 Agent 处理。\n"
    "- 不要做餐厅推荐和方案生成。\n\n"
    "[输出：严格 JSON，仅此一段]\n"
    "{\n"
    '  "decision": "accept" | "reject",\n'
    '  "reason": "string（decision=reject 必填，说明不在北京餐饮范围或不合规）"\n'
    "}"
)

ANALYZE_PROMPT = (
    "你是【北京美食推荐 Agent】的需求分析助手。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "1) 分析用户输入（可能是初始需求或补充需求），生成结构化 requirements。\n"
    "2) 若内容超出北京餐饮范围或非餐饮主题，请给出提示并将 decision 标记为 reject。\n\n"
    "[输出要求：严格 JSON，仅此一段]\n"
    "{\n"
    '  "decision": "accept" | "reject",\n'
    '  "reason": "string（decision=reject 必填）",\n'
    '  "requirements": {\n'
    '    "preferences": ["不辣", "清真", "素食"],\n'
    '    "dietaryRestrictions": ["无葱蒜", "无花生"],\n'
    '    "budgetLevel": "low|medium|high|null",\n'
    '    "areas": ["东城区", "西城区"],\n'
    '    "meals": ["breakfast", "lunch", "dinner", "snack"],\n'
    '    "missingFields": ["若缺少必要信息，这里列出字段名"]\n'
    "  }\n"
    "}"
)

PRODUCE_PROMPT = (
    "你是【北京美食推荐 Agent】的产出生成助手。\n\n"
    "[Agent 职责与范围]"
    f"\n{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据 requirements 生成可执行的北京餐饮推荐方案（纯文本，不含 JSON）。\n"
    "- 结构包含餐次、人均预算、位置动线、招牌菜与文化背景提示。\n"
)


async def decide_accept(user_text: str) -> dict:
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
        obj["reason"] = "不满足北京餐饮范围或规范"
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
        obj = {
            "decision": "accept",
            "requirements": {"preferences": [], "missingFields": ["meals"]},
        }
    if "decision" not in obj:
        obj["decision"] = "accept"
    if obj.get("decision") == "accept" and "requirements" not in obj:
        obj["requirements"] = {"preferences": [], "missingFields": ["meals"]}
    if obj.get("decision") == "reject" and "reason" not in obj:
        obj["reason"] = "不满足北京餐饮范围或规范"
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

    estimated_first_llm_ms = 2000
    if response_timeout_ms is not None and response_timeout_ms < estimated_first_llm_ms:
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

    gate = await decide_accept(user_text)
    if gate.get("decision", "accept") == "reject":
        reason = gate.get("reason", "不满足北京餐饮范围或规范")
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

    accepted = TaskManager.create_task(message, initial_state=TaskState.Accepted)
    setattr(accepted, "_aip_awaiting_input_timeout_ms", awaiting_input_timeout_ms)
    setattr(
        accepted, "_aip_awaiting_completion_timeout_ms", awaiting_completion_timeout_ms
    )
    setattr(accepted, "_aip_max_products_bytes", max_products_bytes)

    logger.info(
        "event=job_schedule task_id=%s work_timeout_ms=%s",
        accepted.id,
        str(work_timeout_ms),
    )
    CateringJobManager.start_job(
        accepted.id,
        lambda cancel_event: _run_catering_pipeline(
            accepted.id, user_text, cancel_event, work_timeout_ms
        ),
    )
    return TaskManager.get_task(accepted.id)


async def on_cancel(message: Message, task: Task) -> Task:
    CateringJobManager.cancel_job(task.id)
    return await DefaultHandlers.cancel(message, task)


async def on_continue(message: Message, task: Task) -> Task:
    TaskManager.add_message_to_history(task.id, message)
    if task.status.state not in (TaskState.AwaitingInput, TaskState.AwaitingCompletion):
        logger.info(
            "event=continue_ignored task_id=%s state=%s", task.id, task.status.state
        )
        return task
    user_text = extract_text_from_message(message)
    if not user_text.strip():
        logger.info("event=continue_missing_text task_id=%s", task.id)
        return task

    work_timeout_ms = DEFAULT_WORK_TIMEOUT_MS
    logger.info(
        "event=job_schedule task_id=%s work_timeout_ms=%s via=continue",
        task.id,
        str(work_timeout_ms),
    )
    CateringJobManager.start_job(
        task.id,
        lambda cancel_event: _run_catering_pipeline(
            task.id, user_text, cancel_event, work_timeout_ms
        ),
    )
    return TaskManager.get_task(task.id)


class CateringJobManager:
    _jobs: dict[str, dict] = {}
    _await_timers: dict[str, asyncio.Task] = {}

    @classmethod
    def start_job(
        cls,
        task_id: str,
        coro_factory: Callable[[asyncio.Event], Awaitable[None]],
    ) -> bool:
        job = cls._jobs.get(task_id)
        if job and not job["task"].done():
            return False

        cancel_event = asyncio.Event()

        async def _runner():
            try:
                await coro_factory(cancel_event)
            except asyncio.CancelledError:
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
            except Exception as e:
                logger.exception("event=job_exception task_id=%s", task_id)
                TaskManager.update_task_status(
                    task_id,
                    TaskState.Failed,
                    data_items=[TextDataItem(text=f"后台执行异常: {str(e)}")],
                )
            finally:
                cls._jobs.pop(task_id, None)

        t = asyncio.create_task(_runner(), name=f"catering-job-{task_id}")
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
        t = cls._await_timers.pop(task_id, None)
        if t and not t.done():
            t.cancel()

    @classmethod
    def schedule_await_timeout(
        cls, task_id: str, state: TaskState, timeout_ms: int | None
    ) -> None:
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


async def _run_catering_pipeline(
    task_id: str,
    user_text: str,
    cancel_event: asyncio.Event,
    timeout_ms: int | None,
) -> None:
    async def _work():
        if cancel_event.is_set():
            return
        current = TaskManager.get_task(task_id)
        logger.info(
            "event=state_transition task_id=%s from=%s to=%s",
            task_id,
            getattr(current.status, "state", None),
            TaskState.Working,
        )
        TaskManager.update_task_status(task_id, TaskState.Working)

        prev = getattr(current, "_catering_requirements", None)
        logger.info("event=llm_analyze_start task_id=%s", task_id)
        analysis = await analyze_requirements(
            user_text or "", previous_requirements=prev
        )
        if analysis.get("decision", "accept") == "reject":
            guidance = analysis.get("reason", "请提供与北京餐饮相关的补充信息")
            logger.info(
                "event=state_transition task_id=%s from=%s to=%s reason=analysis_reject",
                task_id,
                TaskState.Working,
                TaskState.AwaitingInput,
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            t = TaskManager.get_task(task_id)
            timeout_ms2 = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            CateringJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms2
            )
            return

        requirements = analysis.get("requirements", {})
        setattr(current, "_catering_requirements", requirements)

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
            t = TaskManager.get_task(task_id)
            timeout_ms2 = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            CateringJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms2
            )
            return

        logger.info("event=llm_produce_start task_id=%s", task_id)
        plan = await produce_plan(requirements)
        if cancel_event.is_set():
            return
        product = Product(
            id=f"product-{uuid.uuid4()}",
            name="北京美食推荐方案",
            dataItems=[TextDataItem(text=plan)],
        )
        TaskManager.set_products(task_id, [product])
        latest = TaskManager.get_task(task_id)
        if latest and latest.status.state == TaskState.Failed:
            return
        logger.info(
            "event=state_transition task_id=%s from=%s to=%s",
            task_id,
            TaskState.Working,
            TaskState.AwaitingCompletion,
        )
        TaskManager.update_task_status(task_id, TaskState.AwaitingCompletion)
        t2 = TaskManager.get_task(task_id)
        timeout_ms3 = getattr(
            t2,
            "_aip_awaiting_completion_timeout_ms",
            DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS,
        )
        CateringJobManager.schedule_await_timeout(
            task_id, TaskState.AwaitingCompletion, timeout_ms3
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
    on_start=on_start,
    on_continue=on_continue,
    on_cancel=on_cancel,
)

# 注册 AIP RPC 路由（基于 CommandHandlers）
add_aip_rpc_router(app, AIP_ENDPOINT, handlers)
logger.info(
    "event=app_start agent_id=%s endpoint=%s model=%s log_level=%s",
    AGENT_ID,
    AIP_ENDPOINT,
    LLM_MODEL,
    LOG_LEVEL,
)


# 根路径健康检查
@app.get("/")
def read_root():
    return {
        "message": f"欢迎使用 {AGENT_ID}. AIP 协议端点位于 {AIP_ENDPOINT}.",
        "agent_id": AGENT_ID,
    }


if __name__ == "__main__":
    import uvicorn
    import ssl

    # 加载mTLS配置
    json_path = os.path.join(os.path.dirname(__file__), "beijing_catering.json")
    mtls_config = load_mtls_config_from_json(json_path)

    logger.info(
        "event=server_start host=0.0.0.0 port=8013 mtls=enabled aic=%s", mtls_config.aic
    )

    uvicorn.run(
        "beijing_catering:app",
        host="0.0.0.0",
        port=8013,
        reload=True,
        workers=1,
        ssl_keyfile=str(mtls_config.key_file),
        ssl_certfile=str(mtls_config.cert_file),
        ssl_ca_certs=str(mtls_config.ca_cert_file),
        ssl_cert_reqs=ssl.CERT_REQUIRED,  # 要求客户端必须提供证书（mTLS双向认证）
    )
