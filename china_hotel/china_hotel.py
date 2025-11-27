import os
import sys
import json
import asyncio
import uuid
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
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

AGENT_ID = os.getenv("CHINA_HOTEL_AGENT_ID", "china_hotel_agent_001")
AIP_ENDPOINT = os.getenv("CHINA_HOTEL_AIP_ENDPOINT", "/acps-aip-v1/rpc")

# 日志配置
LOG_LEVEL = os.getenv("CHINA_HOTEL_LOG_LEVEL", "INFO").upper()
logger = get_agent_logger("agent.china_hotel", "CHINA_HOTEL_LOG_LEVEL", LOG_LEVEL)

# 大模型相关配置
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("OPENAI_MODEL", "Doubao-pro-32k")
CLASSIFIER_MAX_TOKENS = int(os.getenv("CHINA_HOTEL_CLASSIFIER_MAX_TOKENS", "500"))
EXEC_MAX_TOKENS = int(os.getenv("CHINA_HOTEL_EXEC_MAX_TOKENS", "1500"))

# 缺省超时/时长与产出限制（毫秒/字节）
DEFAULT_RESPONSE_TIMEOUT_MS = int(os.getenv("CHINA_HOTEL_RESPONSE_TIMEOUT_MS", "5000"))
DEFAULT_AWAITING_INPUT_TIMEOUT_MS = int(
    os.getenv("CHINA_HOTEL_AWAITING_INPUT_TIMEOUT_MS", "60000")
)
DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS = int(
    os.getenv("CHINA_HOTEL_AWAITING_COMPLETION_TIMEOUT_MS", "60000")
)
DEFAULT_MAX_PRODUCTS_BYTES = int(os.getenv("CHINA_HOTEL_MAX_PRODUCTS_BYTES", "1048576"))
DEFAULT_WORK_TIMEOUT_MS = int(os.getenv("CHINA_HOTEL_WORK_TIMEOUT_MS", "10000"))

# 初始化主应用程序
app = FastAPI(
    title="全国酒店预订推荐 Agent",
    description="符合 ACPs 协议的全国酒店智能体：基于行程与预算输出结构化酒店推荐方案或拒绝理由。",
)


# ============================
# 多技能编排：目录与提示词模板
# ============================

# 统一技能目录（与 china_hotel.json 保持一致的 ID）
SKILL_CATALOG: Dict[str, Dict[str, Any]] = {
    "china_hotel.hotel-recommendation": {
        "name": "酒店推荐服务",
        "required_slots": ["city", "dates_or_nights"],
        "optional_slots": [
            "budget",
            "people",
            "preferences",
            "activity_areas",
            "transport",
            "membership",
        ],
    },
    "china_hotel.reservation-assistance": {
        "name": "预订协助服务",
        # 二选一：要么给出 city+dates+people；要么给出 hotel_candidates+dates+people
        "required_slots_anyof": [
            ["city", "dates_or_nights", "people"],
            ["hotel_candidates", "dates_or_nights", "people"],
        ],
        "optional_slots": ["platforms", "membership", "invoice", "cancel_policy"],
    },
    "china_hotel.accommodation-optimization": {
        "name": "住宿优化建议",
        "required_slots": ["city", "dates_or_nights", "activity_areas"],
        "optional_slots": ["budget", "transport", "preferences"],
    },
    "china_hotel.price-comparison": {
        "name": "价格比较服务",
        "required_slots": ["hotel_candidates", "dates_or_nights", "city"],
        "optional_slots": [
            "budget",
            "room_meal",
            "cancel_policy",
            "membership",
            "platforms",
        ],
    },
}


def _load_skill_descriptions_from_spec() -> None:
    """
    从 agent 规范文件（china_hotel.json）加载各技能的中文描述，
    并写入 SKILL_CATALOG，用于增强分类与执行阶段的提示词质量。
    """
    try:
        spec_path = os.path.join(os.path.dirname(__file__), "china_hotel.json")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        for sk in spec.get("skills", []):
            sid = sk.get("id")
            if sid in SKILL_CATALOG:
                SKILL_CATALOG[sid]["description"] = sk.get("description", "")
    except Exception as e:
        logger.warning("event=load_skill_desc_failed error=%s", e)


# 加载技能描述
_load_skill_descriptions_from_spec()


def _load_capabilities_snippet() -> str:
    json_path = os.path.join(os.path.dirname(__file__), "china_hotel.json")
    fallback = (
        "职责：提供中国境内酒店推荐、预订协助、住宿优化、价格比较；"
        "范围：仅限中国境内酒店相关；拒绝境外/非住宿类需求；演示模式不因实时价格/库存而拒绝。"
    )
    return load_capabilities_snippet_from_json(json_path, fallback)


CAPABILITIES_SNIPPET = _load_capabilities_snippet()

# ============================
# 任务上下文（内存）
# ============================
TASK_CTX: Dict[str, Dict[str, Any]] = {}


GLOBAL_CLASSIFIER_PROMPT = (
    ""  # 已不再直接使用全局分类器提示词（保留占位，以便向后兼容）
)


async def _call_llm(
    messages: List[Dict[str, str]], temperature: float, max_tokens: int | None
) -> str:
    raw = await call_openai_chat(
        messages,
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens or 512,
    )
    return (raw or "").strip()


def _merge_requirements(
    prev: Dict[str, Any] | None, now: Dict[str, Any] | None
) -> Dict[str, Any]:
    """简易合并器：以 now 覆盖 prev，仅对 requirements/global 层做浅合并。"""
    prev = prev or {}
    now = now or {}
    merged = {**prev}
    # 合并 global 槽位
    g_prev = (prev.get("global") or {}) if isinstance(prev.get("global"), dict) else {}
    g_now = (now.get("global") or {}) if isinstance(now.get("global"), dict) else {}
    merged["global"] = {**g_prev, **{k: v for k, v in g_now.items() if v}}
    # 合并 selectedSkills（去重保序）
    skills_prev = [s for s in prev.get("selectedSkills", []) if isinstance(s, str)]
    skills_now = [s for s in now.get("selectedSkills", []) if isinstance(s, str)]
    seen = set()
    merged["selectedSkills"] = (
        [s for s in skills_prev + skills_now if not (s in seen or seen.add(s))]
        or skills_prev
        or skills_now
    )
    return merged


async def decide_accept(user_text: str) -> dict:
    """Start 门卫判断：accept | reject。"""
    DECIDE_PROMPT = (
        "你是【全国酒店预订推荐 Agent】的请求门卫。\n\n"
        "[Agent 职责与范围]\n"
        f"{CAPABILITIES_SNIPPET}\n\n"
        "[你的任务]\n"
        "- 仅判断该请求是否属于中国境内酒店相关，是否应由本 Agent 处理。\n"
        "- 不要因为缺少日期/人数/预算等细节而拒绝；只要主题明确为中国境内酒店/住宿相关，即判定为 accept（后续由系统引导补充信息）。\n"
        "- 仅在明确越界时才判定为 reject：例如境外酒店、非住宿类任务（机票/签证/景点规划/餐饮/营销文案/相机价格对比等）。\n\n"
        "- 不要做需求结构化，也不要给出方案。\n\n"
        "[输出：严格 JSON，仅此一段]\n"
        "{\n"
        '  "decision": "accept" | "reject",\n'
        '  "reason": "string（decision=reject 必填，说明越界/不合规）"\n'
        "}"
    )
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
        obj["reason"] = "请求不在中国境内酒店范围或不合规"
    return obj


def _validate_skill_slots(skill_id: str, slots: Dict[str, Any]) -> List[str]:
    """
    基于技能元数据校验槽位完整性。
    - required_slots: 所有字段必须存在。
    - required_slots_anyof: 多个备选组中至少满足一组；若均不满足，返回缺失字段最少的那一组。
    返回：缺失字段名称列表（为空表示可执行）。
    """
    meta = SKILL_CATALOG.get(skill_id, {})
    missing: List[str] = []
    # 全量必填（all-of）：列出的字段都需存在。
    for k in meta.get("required_slots", []):
        v = slots.get(k)
        if not v:
            missing.append(k)
    # 备选必填（any-of）：多个字段组满足其一即可。
    any_of: List[List[str]] = meta.get("required_slots_anyof", []) or []
    if any_of:
        # 每个组代表一套可选的完整必填集合；至少满足其中一组。
        group_missing_list: List[List[str]] = []
        for group in any_of:
            group_missing = [k for k in group if not slots.get(k)]
            group_missing_list.append(group_missing)
        # 若全部分组都不完整，则给出“最少缺失字段”的那一组以尽量降低补充成本。
        if not any(len(gm) == 0 for gm in group_missing_list):
            # 选择缺失项最少的分组，降低用户补充负担。
            best_missing = min(group_missing_list, key=lambda gm: len(gm))
            missing.extend(best_missing)
    return missing


def _skill_prompt(
    skill_id: str, slots: Dict[str, Any], user_request: str
) -> List[Dict[str, str]]:
    """
    构造技能执行提示词（system 消息）。
    内容包含：技能名称与描述、演示模式约束、槽位回显、用户原始需求。
    返回：单条 system 消息组成的 messages 列表。
    """
    skill_meta = SKILL_CATALOG.get(skill_id, {})
    skill_name = skill_meta.get("name", skill_id)
    skill_desc = skill_meta.get("description", "")
    guidance = (
        f"你正在执行技能：{skill_name} (id={skill_id})。\n"
        + (f"技能描述：{skill_desc}\n" if skill_desc else "")
        + "你处于演示模式：不要因无法实时价格/库存拒绝；产出价格请给区间或约数并附标准免责声明。\n"
        + "请输出该技能的结果正文（纯文本，不要 JSON、不加多余说明）。\n"
        + "若槽位不足以完成本技能，请先仅列出需要补充的字段清单（中文、简洁），不要生成方案。\n\n"
        + "可用槽位（可能为空）：\n"
        + "\n".join(
            [
                f"- {k}: {slots.get(k) or ''}"
                for k in [
                    "city",
                    "dates_or_nights",
                    "people",
                    "budget",
                    "preferences",
                    "activity_areas",
                    "transport",
                    "membership",
                    "hotel_candidates",
                    "platforms",
                ]
            ]
        )
        + "\n\n用户原始需求：\n"
        + user_request
    )
    return [
        {"role": "system", "content": guidance},
    ]


async def execute_skill(
    skill_id: str, slots: Dict[str, Any], user_request: str
) -> Tuple[Optional[str], Optional[List[str]]]:
    """执行单个技能（异步）。若缺失必填，返回缺失字段；否则返回文本结果。"""
    missing = _validate_skill_slots(skill_id, slots)
    if missing:
        return None, missing
    messages = _skill_prompt(skill_id, slots, user_request)
    text = await _call_llm(messages, temperature=0.5, max_tokens=EXEC_MAX_TOKENS)
    return (text.strip() if text else ""), None


# ----------------- 结构化需求分析与产出 -----------------
ANALYZE_PROMPT = (
    "你是【全国酒店预订推荐 Agent】的需求分析助手。\n\n"
    "[Agent 职责与范围]\n"
    f"{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据用户输入（可能是初始或补充）生成结构化需求 requirements。\n"
    "- 识别应使用哪些技能（selectedSkills），并提取通用槽位 global：city, dates_or_nights, people, budget, preferences, activity_areas, transport, membership, hotel_candidates, platforms。\n"
    "- 不要自行进行价格/库存实时校验。\n\n"
    "[输出：严格 JSON，仅此一段]\n"
    "{\n"
    '  "decision": "accept" | "reject",\n'
    '  "reason": "string（decision=reject 必填）",\n'
    '  "requirements": {\n'
    '    "selectedSkills": ["china_hotel.hotel-recommendation"],\n'
    '    "global": {\n'
    '      "city": "string|null",\n'
    '      "dates_or_nights": "string|null",\n'
    '      "people": "string|null",\n'
    '      "budget": "string|null",\n'
    '      "preferences": "string|null",\n'
    '      "activity_areas": "string|null",\n'
    '      "transport": "string|null",\n'
    '      "membership": "string|null",\n'
    '      "hotel_candidates": "string|null",\n'
    '      "platforms": "string|null"\n'
    "    }\n"
    "  }\n"
    "}"
)


async def analyze_requirements(
    user_text: str, previous_requirements: Dict[str, Any] | None = None
) -> Dict[str, Any]:
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
        max_tokens=600,
    )
    try:
        obj = json.loads(raw)
    except JSONDecodeError:
        obj = {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
            },
        }
    # 兜底字段
    if obj.get("decision") not in ("accept", "reject"):
        obj["decision"] = "accept"
    if obj.get("decision") == "reject" and not obj.get("reason"):
        obj["reason"] = "请求不在中国境内酒店范围或不合规"
    req = obj.get("requirements")
    if not isinstance(req, dict):
        req = {"selectedSkills": ["china_hotel.hotel-recommendation"], "global": {}}
        obj["requirements"] = req
    # 合并 previous
    if previous_requirements:
        req = _merge_requirements(previous_requirements, req)
        obj["requirements"] = req
    # 默认技能
    skills = (
        req.get("selectedSkills") if isinstance(req.get("selectedSkills"), list) else []
    )
    skills = [s for s in skills if s in SKILL_CATALOG]
    if not skills:
        skills = ["china_hotel.hotel-recommendation"]
    req["selectedSkills"] = skills
    # 全局缺失
    g = req.get("global") if isinstance(req.get("global"), dict) else {}
    global_missing: List[str] = []
    for k in ["city", "dates_or_nights"]:
        if not (g.get(k) and str(g.get(k)).strip()):
            global_missing.append(k)
    req["globalMissing"] = global_missing
    # 技能缺失
    per_skill_missing: Dict[str, List[str]] = {}
    for sid in skills:
        miss = _validate_skill_slots(sid, g)
        if miss:
            per_skill_missing[sid] = miss
    req["perSkillMissing"] = per_skill_missing
    return obj


PRODUCE_PROMPT = (
    "你是【全国酒店预订推荐 Agent】的产出生成助手。\n\n"
    "[Agent 职责与范围]\n"
    f"{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据 requirements 执行所选技能并组织为清晰的文本输出（纯文本）。\n"
    "- 价格与库存为演示估算，请附上标准免责声明。\n"
)


async def produce_output(requirements: Dict[str, Any], user_text: str) -> str:
    g = (
        requirements.get("global")
        if isinstance(requirements.get("global"), dict)
        else {}
    )
    skills = (
        requirements.get("selectedSkills", [])
        if isinstance(requirements.get("selectedSkills"), list)
        else []
    )
    sections: List[str] = []
    for sid in skills:
        text, missing = await execute_skill(sid, g, user_text)
        if text:
            header = SKILL_CATALOG[sid]["name"]
            sections.append(f"【{header}】\n" + text)
    if not sections:
        return ""
    plan = "\n\n——\n".join(sections)
    plan += "\n\n【免责声明】价格/库存为演示估算，实际以官方/OTA 实时为准。"
    return plan.strip()


# ---------------- Agent 内部：后台任务管理器（基于 asyncio） ----------------
class HotelJobManager:
    _jobs: Dict[str, Dict[str, Any]] = {}
    _await_timers: Dict[str, asyncio.Task] = {}

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

        t = asyncio.create_task(_runner(), name=f"hotel-job-{task_id}")
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


async def _run_hotel_pipeline(
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

        prev = getattr(current, "_hotel_requirements", None)
        logger.info("event=llm_analyze_start task_id=%s", task_id)
        analysis = await analyze_requirements(
            user_text or "", previous_requirements=prev
        )
        if analysis.get("decision", "accept") == "reject":
            guidance = analysis.get("reason", "请求不在中国境内酒店范围或不合规")
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
            timeout_ms = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            HotelJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms
            )
            return

        requirements = analysis.get("requirements", {})
        setattr(current, "_hotel_requirements", requirements)

        g = (
            requirements.get("global")
            if isinstance(requirements.get("global"), dict)
            else {}
        )
        global_missing = requirements.get("globalMissing", [])
        per_skill_missing = requirements.get("perSkillMissing", {})
        # 若缺失信息，则 AwaitingInput
        if (isinstance(global_missing, list) and global_missing) or (
            isinstance(per_skill_missing, dict) and any(per_skill_missing.values())
        ):
            parts: List[str] = []
            if global_missing:
                tips_map = {
                    "city": "城市/目的地（可多城市顺序）",
                    "dates_or_nights": "住宿时长（晚数或入住/退房日期）",
                }
                human = "；".join([tips_map.get(m, str(m)) for m in global_missing])
                parts.append("全局必填缺失：" + human)
            if isinstance(per_skill_missing, dict):
                for sid, miss in per_skill_missing.items():
                    if miss:
                        name = SKILL_CATALOG.get(sid, {}).get("name", sid)
                        parts.append(f"[{name}] 需要：" + "、".join(map(str, miss)))
            guidance = (
                "信息不足：\n"
                + "\n".join(parts)
                + "\n请补充后再次提交；示例：城市=北京；日期=2025-10-01~2025-10-04；人数=2成人；预算=¥500-700/晚。"
            )
            logger.info(
                "event=state_transition task_id=%s from=%s to=%s reason=missing_fields",
                task_id,
                TaskState.Working,
                TaskState.AwaitingInput,
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            t = TaskManager.get_task(task_id)
            timeout_ms = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            HotelJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms
            )
            return

        # 生成产出
        logger.info("event=llm_produce_start task_id=%s", task_id)
        plan = await produce_output(requirements, user_text)
        if cancel_event.is_set():
            return
        if not plan:
            TaskManager.update_task_status(
                task_id,
                TaskState.Failed,
                [TextDataItem(text="未能生成可用结果，请补充更多上下文。")],
            )
            return
        product = Product(
            id=f"product-{uuid.uuid4()}",
            name="全国酒店多技能方案",
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
        timeout_ms2 = getattr(
            t2,
            "_aip_awaiting_completion_timeout_ms",
            DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS,
        )
        HotelJobManager.schedule_await_timeout(
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


# ----------------- AIP CommandHandlers 实现 -----------------
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
        reason = gate.get("reason", "请求不在中国境内酒店范围或不合规")
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
    HotelJobManager.start_job(
        accepted.id,
        lambda cancel_event: _run_hotel_pipeline(
            accepted.id, user_text, cancel_event, work_timeout_ms
        ),
    )
    return TaskManager.get_task(accepted.id)


async def on_cancel(message: Message, task: Task) -> Task:
    HotelJobManager.cancel_job(task.id)
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
    HotelJobManager.start_job(
        task.id,
        lambda cancel_event: _run_hotel_pipeline(
            task.id, user_text, cancel_event, work_timeout_ms
        ),
    )
    return TaskManager.get_task(task.id)


handlers = CommandHandlers(
    on_start=on_start,
    on_continue=on_continue,
    on_cancel=on_cancel,
)

# 注册 RPC 路由
add_aip_rpc_router(app, AIP_ENDPOINT, handlers)


@app.get("/")
def read_root():
    """
    健康检查与基础信息接口。
    返回：欢迎语与当前 Agent ID、RPC 端点路径。
    """
    return {
        "message": f"欢迎使用 {AGENT_ID}. AIP 协议端点位于 {AIP_ENDPOINT}.",
        "agent_id": AGENT_ID,
    }


if __name__ == "__main__":
    import uvicorn
    import ssl

    # 加载mTLS配置
    json_path = os.path.join(os.path.dirname(__file__), "china_hotel.json")
    mtls_config = load_mtls_config_from_json(json_path)

    logger.info(
        "event=server_start host=0.0.0.0 port=8015 mtls=enabled aic=%s", mtls_config.aic
    )

    uvicorn.run(
        "china_hotel:app",
        host="0.0.0.0",
        port=8015,
        reload=True,
        workers=1,
        ssl_keyfile=str(mtls_config.key_file),
        ssl_certfile=str(mtls_config.cert_file),
        ssl_ca_certs=str(mtls_config.ca_cert_file),
        ssl_cert_reqs=ssl.CERT_REQUIRED,  # 要求客户端必须提供证书（mTLS双向认证）
    )
