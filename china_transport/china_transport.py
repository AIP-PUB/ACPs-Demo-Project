import os
import sys
import json
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple
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

AGENT_ID = os.getenv("CHINA_TRANSPORT_AGENT_ID", "china_transport_agent_001")
AIP_ENDPOINT = os.getenv("CHINA_TRANSPORT_AIP_ENDPOINT", "/acps-aip-v1/rpc")

# 日志配置
LOG_LEVEL = os.getenv("CHINA_TRANSPORT_LOG_LEVEL", "INFO").upper()
logger = get_agent_logger(
    "agent.china_transport", "CHINA_TRANSPORT_LOG_LEVEL", LOG_LEVEL
)

# 大模型配置
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("OPENAI_MODEL", "Doubao-pro-32k")
CLASSIFIER_MAX_TOKENS = int(os.getenv("CHINA_TRANSPORT_CLASSIFIER_MAX_TOKENS", "500"))
EXEC_MAX_TOKENS = int(os.getenv("CHINA_TRANSPORT_EXEC_MAX_TOKENS", "1500"))

# 缺省超时/时长与产出限制（毫秒/字节）
DEFAULT_RESPONSE_TIMEOUT_MS = int(
    os.getenv("CHINA_TRANSPORT_RESPONSE_TIMEOUT_MS", "5000")
)
DEFAULT_AWAITING_INPUT_TIMEOUT_MS = int(
    os.getenv("CHINA_TRANSPORT_AWAITING_INPUT_TIMEOUT_MS", "60000")
)
DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS = int(
    os.getenv("CHINA_TRANSPORT_AWAITING_COMPLETION_TIMEOUT_MS", "60000")
)
DEFAULT_MAX_PRODUCTS_BYTES = int(
    os.getenv("CHINA_TRANSPORT_MAX_PRODUCTS_BYTES", "1048576")
)
DEFAULT_WORK_TIMEOUT_MS = int(os.getenv("CHINA_TRANSPORT_WORK_TIMEOUT_MS", "10000"))

# 初始化主应用程序
app = FastAPI(
    title="全国城际交通规划 Agent",
    description="符合 ACPs 协议的全国城际交通智能体：输出城市与城市之间的交通方案与对比，拒绝城市内与境外交通。",
)

"""
此模块实现“全国城际交通规划 Agent”的多技能编排逻辑，遵循 AIP 2.1.2 的状态转移：
- Start：门卫判定 accept/reject；accept 后异步执行（Working）；
- Working：
    - 结构化需求分析（选择技能 + 槽位提取 + 必填校验）；
    - 必填缺失 → AwaitingInput；
    - 否则执行技能产出 → 设置 Product → AwaitingCompletion；
- Continue：在 AwaitingInput/AwaitingCompletion 时触发，再次进入 Working 并重复上述流程；
- Cancel/Complete：按规范进入 Canceled/Completed。
"""

# ============================
# 多技能编排：目录与提示词模板（与 JSON 规范保持一致的 ID）
# ============================

# 统一通用槽位键约定
# - from_city / to_city：城市对
# - city_sequence：有序城市序列（例如：上海→南京→杭州）
# - dates_or_days：出行日期范围或天数
# - people：人数
# - budget：预算区间（¥）
# - preference：偏好（时间优先/费用优先/舒适度）
# - luggage_special：行李/特殊需求
# - transfer_constraints：换乘限制（直达/最少换乘/避夜行等）
# - daily_time_window：每日起止时间窗口
SKILL_CATALOG: Dict[str, Dict[str, Any]] = {
    "china_transport.intercity-transportation-planning": {
        "name": "城际交通规划",
        # 二选一：from/to + 日期；或 city_sequence + 日期
        "required_slots_anyof": [
            ["from_city", "to_city", "dates_or_days"],
            ["city_sequence", "dates_or_days"],
        ],
        "optional_slots": [
            "people",
            "budget",
            "preference",
            "luggage_special",
            "transfer_constraints",
        ],
    },
    "china_transport.route-optimization": {
        "name": "路线优化服务",
        "required_slots": ["city_sequence", "dates_or_days"],
        "optional_slots": [
            "budget",
            "preference",
            "daily_time_window",
            "transfer_constraints",
            "people",
            "luggage_special",
        ],
    },
    "china_transport.special-needs-transportation": {
        "name": "特殊需求交通",
        # 二选一： (from/to + 日期 + 人数 + 特殊) 或 (city_sequence + 日期 + 人数 + 特殊)
        "required_slots_anyof": [
            ["from_city", "to_city", "dates_or_days", "people", "luggage_special"],
            ["city_sequence", "dates_or_days", "people", "luggage_special"],
        ],
        "optional_slots": [
            "budget",
            "preference",
            "transfer_constraints",
        ],
    },
}


def _load_skill_descriptions_from_spec() -> None:
    """
    从 china_transport.json 加载技能描述，写入 SKILL_CATALOG。
    """
    try:
        spec_path = os.path.join(os.path.dirname(__file__), "china_transport.json")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        for sk in spec.get("skills", []):
            sid = sk.get("id")
            if sid in SKILL_CATALOG:
                SKILL_CATALOG[sid]["description"] = sk.get("description", "")
    except Exception as e:
        logger.warning("event=load_transport_skill_desc_failed error=%s", e)


# 模块导入时加载描述
_load_skill_descriptions_from_spec()


def _load_capabilities_snippet() -> str:
    json_path = os.path.join(os.path.dirname(__file__), "china_transport.json")
    fallback = (
        "职责：提供中国大陆城际交通规划、路线优化、特殊需求交通建议；"
        "范围：仅限中国境内城市与城市之间交通；拒绝境外与城市内出行；演示模式不因实时票价/库存而拒绝。"
    )
    return load_capabilities_snippet_from_json(json_path, fallback)


CAPABILITIES_SNIPPET = _load_capabilities_snippet()

# ============================
# 任务上下文（内存）
# ============================
# 保存每个 task 的累计槽位、选定技能与补充历史，避免 Continue 时被重新分类误判。
TASK_CTX: Dict[str, Dict[str, Any]] = {}


def _build_global_classifier_prompt() -> str:
    """
    构造分类器系统提示：范围与拒绝项、技能目录、严格 JSON 输出结构。
    """
    skills_lines = []
    for k, v in SKILL_CATALOG.items():
        name = v.get("name", k)
        desc = v.get("description", "")
        skills_lines.append(f"- {k}: {name} — {desc}")
    return (
        "你是一个城际交通请求分拣与槽位抽取助手，服务于中国大陆境内城市与城市之间的交通规划。\n"
        "- 明确拒绝：城市内出行（地铁/公交/出租/网约车/步行等）；境外或跨国交通；与交通无关的内容。\n"
        "- 不得因‘无法实时价格/库存/购票’而拒绝。\n\n"
        "任务：\n"
        "1) 判断是否在本 Agent 范围（in_scope）。\n"
        "2) 识别是否显式提到技能 ID（explicit_skill_ids），否则基于意图与技能映射（inferred_skills）。\n"
        "3) 抽取或归一化槽位 slots：from_city, to_city, city_sequence, dates_or_days, people, budget, preference, luggage_special, transfer_constraints, daily_time_window。\n"
        "4) 标注缺失的全局必填（missing_global）：(from_city+to_city 或 city_sequence) 与 dates_or_days。\n\n"
        "技能目录（id: name — description）：\n"
        + "\n".join(skills_lines)
        + "\n\n严格返回 JSON：{\n"
        '  "in_scope": true|false,\n'
        '  "reason": string,\n'
        '  "explicit_skill_ids": string[],\n'
        '  "inferred_skills": string[],\n'
        '  "missing_global": string[],\n'
        '  "slots": { "from_city": string|null, "to_city": string|null, "city_sequence": string|null, \n'
        '              "dates_or_days": string|null, "people": string|null, "budget": string|null, \n'
        '              "preference": string|null, "luggage_special": string|null, "transfer_constraints": string|null, \n'
        '              "daily_time_window": string|null }\n}'
    )


GLOBAL_CLASSIFIER_PROMPT = _build_global_classifier_prompt()


async def _call_llm(
    messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int]
) -> str:
    raw = await call_openai_chat(
        messages,
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens or 512,
    )
    return (raw or "").strip()


async def _llm_merge_slots(
    prior_slots: Dict[str, Any], supplement: str
) -> Dict[str, Any]:
    """使用一次 LLM 将补充文本与既有槽位合并，返回严格 JSON（仅已知键）。"""
    allowed = [
        "from_city",
        "to_city",
        "city_sequence",
        "dates_or_days",
        "people",
        "budget",
        "preference",
        "luggage_special",
        "transfer_constraints",
        "daily_time_window",
    ]
    sys = (
        "你是字段合并器。给定‘既有槽位 JSON’与‘补充文本’，只返回严格 JSON（不含多余文本）。\n"
        "规则：仅输出以下已知键；若无法确定某键则省略；值必须为字符串。\n"
        + ", ".join(allowed)
    )
    prior_json = json.dumps(
        {
            k: v
            for k, v in prior_slots.items()
            if k in allowed and isinstance(v, str) and v.strip()
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": f"[既有槽位]\n{prior_json}\n\n[补充]\n{supplement}",
        },
    ]
    raw = await _call_llm(messages, temperature=0.0, max_tokens=300)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        merged: Dict[str, Any] = {}
        for k in allowed:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                merged[k] = v.strip()
        return merged
    except Exception:
        logger.error("event=merge_slots_json_error preview=%s", truncate(raw, 160))
        return {}


async def classify_request(user_request: str) -> Dict[str, Any]:
    """
    调用分类器模型，判断范围、识别技能、抽取槽位与全局缺失项（missing_global）。
    失败时返回安全默认值（提示补充 from/to 或 city_sequence 与 dates_or_days）。
    """
    messages = [
        {"role": "system", "content": GLOBAL_CLASSIFIER_PROMPT},
        {"role": "user", "content": user_request},
    ]
    raw = await _call_llm(messages, temperature=0.0, max_tokens=CLASSIFIER_MAX_TOKENS)
    try:
        data = json.loads(raw)
        data.setdefault("in_scope", True)
        data.setdefault("reason", "")
        data.setdefault("explicit_skill_ids", [])
        data.setdefault("inferred_skills", [])
        data.setdefault("missing_global", [])
        data.setdefault("slots", {})
        return data
    except Exception:
        retry_messages = messages + [
            {
                "role": "system",
                "content": "请仅返回严格合法的 JSON，结构与键名必须与上文要求完全一致，不要添加任何解释或注释。",
            }
        ]
        raw2 = await _call_llm(
            retry_messages, temperature=0.0, max_tokens=CLASSIFIER_MAX_TOKENS
        )
        try:
            data = json.loads(raw2)
            data.setdefault("in_scope", True)
            data.setdefault("reason", "")
            data.setdefault("explicit_skill_ids", [])
            data.setdefault("inferred_skills", [])
            data.setdefault("missing_global", [])
            data.setdefault("slots", {})
            return data
        except Exception:
            return {
                "in_scope": True,
                "reason": "classifier-json-parse-failed",
                "explicit_skill_ids": [],
                "inferred_skills": [],
                "missing_global": [
                    "from_city/to_city 或 city_sequence",
                    "dates_or_days",
                ],
                "slots": {},
            }


def _validate_skill_slots(skill_id: str, slots: Dict[str, Any]) -> List[str]:
    """
    按技能元数据校验槽位：
    - required_slots：全部需要；
    - required_slots_anyof：给定分组中至少满足一组；若均不满足，返回缺失字段最少的那组。
    返回缺失字段列表（为空表示可执行）。
    """
    meta = SKILL_CATALOG.get(skill_id, {})
    missing: List[str] = []
    for k in meta.get("required_slots", []):
        if not slots.get(k):
            missing.append(k)
    any_of: List[List[str]] = meta.get("required_slots_anyof", []) or []
    if any_of:
        group_missing_list: List[List[str]] = []
        for group in any_of:
            group_missing = [k for k in group if not slots.get(k)]
            group_missing_list.append(group_missing)
        if not any(len(gm) == 0 for gm in group_missing_list):
            best_missing = min(group_missing_list, key=lambda gm: len(gm))
            missing.extend(best_missing)
    return missing


def _skill_prompt(
    skill_id: str, slots: Dict[str, Any], user_request: str
) -> List[Dict[str, str]]:
    """
    构造单技能执行的提示词（system）。包含：技能名/描述、演示模式、槽位回显、用户需求。
    """
    meta = SKILL_CATALOG.get(skill_id, {})
    name = meta.get("name", skill_id)
    desc = meta.get("description", "")
    guidance = (
        f"你正在执行技能：{name} (id={skill_id})。\n"
        + (f"技能描述：{desc}\n" if desc else "")
        + "你处于演示模式：不要因无法实时价格/库存/购票而拒绝；价格/耗时给出区间或约数，并在末尾附标准免责声明。\n"
        + "请输出该技能的结果正文（纯文本，不要 JSON、不加多余说明）。\n"
        + "若槽位不足以完成本技能，请先仅列出需要补充的字段清单（中文、简洁），不要生成方案。\n\n"
        + "可用槽位（可能为空）：\n"
        + "\n".join(
            [
                f"- {k}: {slots.get(k) or ''}"
                for k in [
                    "from_city",
                    "to_city",
                    "city_sequence",
                    "dates_or_days",
                    "people",
                    "budget",
                    "preference",
                    "luggage_special",
                    "transfer_constraints",
                    "daily_time_window",
                ]
            ]
        )
        + "\n\n用户原始需求：\n"
        + user_request
    )
    return [{"role": "system", "content": guidance}]


async def execute_skill(
    skill_id: str, slots: Dict[str, Any], user_request: str
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    执行单个技能。返回 (result_text, missing_fields)。有缺失则返回 (None, 缺失字段列表)。
    """
    missing = _validate_skill_slots(skill_id, slots)
    if missing:
        return None, missing
    messages = _skill_prompt(skill_id, slots, user_request)
    text = await _call_llm(messages, temperature=0.5, max_tokens=EXEC_MAX_TOKENS)
    return (text or "").strip(), None


# ----------------- 结构化需求分析与产出 -----------------
def _merge_requirements(
    prev: Dict[str, Any] | None, now: Dict[str, Any] | None
) -> Dict[str, Any]:
    prev = prev or {}
    now = now or {}
    merged = {**prev}
    g_prev = (prev.get("global") or {}) if isinstance(prev.get("global"), dict) else {}
    g_now = (now.get("global") or {}) if isinstance(now.get("global"), dict) else {}
    merged["global"] = {**g_prev, **{k: v for k, v in g_now.items() if v}}
    skills_prev = [s for s in prev.get("selectedSkills", []) if isinstance(s, str)]
    skills_now = [s for s in now.get("selectedSkills", []) if isinstance(s, str)]
    seen = set()
    merged["selectedSkills"] = (
        [s for s in skills_prev + skills_now if not (s in seen or seen.add(s))]
        or skills_prev
        or skills_now
    )
    return merged


ANALYZE_PROMPT = (
    "你是【全国城际交通规划 Agent】的需求分析助手。\n\n"
    "[Agent 职责与范围]\n"
    f"{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据用户输入（可能是初始或补充）生成结构化需求 requirements。\n"
    "- 识别应使用哪些技能（selectedSkills），并提取通用槽位 global：from_city, to_city, city_sequence, dates_or_days, people, budget, preference, luggage_special, transfer_constraints, daily_time_window。\n"
    "- 不要自行进行实时票价/库存校验。\n\n"
    "[输出：严格 JSON，仅此一段]\n"
    "{\n"
    '  "decision": "accept" | "reject",\n'
    '  "reason": "string（decision=reject 必填）",\n'
    '  "requirements": {\n'
    '    "selectedSkills": ["china_transport.intercity-transportation-planning"],\n'
    '    "global": {\n'
    '      "from_city": "string|null",\n'
    '      "to_city": "string|null",\n'
    '      "city_sequence": "string|null",\n'
    '      "dates_or_days": "string|null",\n'
    '      "people": "string|null",\n'
    '      "budget": "string|null",\n'
    '      "preference": "string|null",\n'
    '      "luggage_special": "string|null",\n'
    '      "transfer_constraints": "string|null",\n'
    '      "daily_time_window": "string|null"\n'
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
            {
                "previous": previous_requirements,
                "supplement": user_text,
            },
            ensure_ascii=False,
        )
    raw = await call_openai_chat(
        [
            {"role": "system", "content": ANALYZE_PROMPT},
            {"role": "user", "content": payload},
        ],
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=700,
    )
    try:
        obj = json.loads(raw or "{}")
    except Exception:
        # 兜底：accept + 空 requirements
        obj = {
            "decision": "accept",
            "requirements": {"selectedSkills": [], "global": {}},
        }
    if obj.get("decision") not in ("accept", "reject"):
        obj["decision"] = "accept"
    if obj.get("decision") == "reject" and not obj.get("reason"):
        obj["reason"] = "请求不在城际交通规划范围"
    req = obj.get("requirements")
    if not isinstance(req, dict):
        req = {"selectedSkills": [], "global": {}}
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
        skills = ["china_transport.intercity-transportation-planning"]
    req["selectedSkills"] = skills
    # 全局缺失
    g = req.get("global") if isinstance(req.get("global"), dict) else {}
    global_missing: List[str] = []
    # 需要 (from/to 或 city_sequence) + dates_or_days
    has_pair = bool((g.get("from_city") and g.get("to_city")) or g.get("city_sequence"))
    if not has_pair:
        global_missing.append("from_city/to_city 或 city_sequence")
    if not g.get("dates_or_days"):
        global_missing.append("dates_or_days")
    req["globalMissing"] = global_missing
    # 技能缺失
    per_skill_missing: Dict[str, List[str]] = {}
    for sid in skills:
        miss = _validate_skill_slots(sid, g)
        if miss:
            per_skill_missing[sid] = miss
    req["perSkillMissing"] = per_skill_missing
    return obj


PRODUCE_PROMPT_NOTE = (
    "你是【全国城际交通规划 Agent】的产出生成助手。\n\n"
    "[Agent 职责与范围]\n"
    f"{CAPABILITIES_SNIPPET}\n\n"
    "[你的任务]\n"
    "- 根据 requirements 执行所选技能并组织为清晰的文本输出（纯文本）。\n"
    "- 票价/时长为演示估算，请附上标准免责声明。\n"
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
        result_text, missing = await execute_skill(sid, g, user_text)
        if missing:
            # 即使个别技能缺失，统一在 analyze 阶段已处理，这里忽略
            continue
        if result_text:
            header = SKILL_CATALOG.get(sid, {}).get("name", sid)
            sections.append(f"【{header}】\n" + result_text)
    if not sections:
        return "未能生成可用结果，请补充更多上下文。"
    plan = "\n\n——\n".join(sections)
    plan += (
        "\n\n【免责声明】数据为演示估算，票价/班次/库存以 12306/航空公司/OTA 实时为准。"
    )
    return plan.strip()


class TransportJobManager:
    _jobs: Dict[str, Dict[str, Any]] = {}
    _await_timers: Dict[str, asyncio.Task] = {}

    @classmethod
    def _get_job(cls, task_id: str) -> Dict[str, Any] | None:
        return cls._jobs.get(task_id)

    @classmethod
    def cancel(cls, task_id: str) -> None:
        job = cls._jobs.get(task_id)
        if not job:
            return
        event: asyncio.Event = job.get("cancel_event")
        if event:
            event.set()
        task: asyncio.Task = job.get("task")
        if task and not task.done():
            task.cancel()
        cls._jobs.pop(task_id, None)
        # 取消等待定时器
        t = cls._await_timers.pop(task_id, None)
        if t and not t.done():
            t.cancel()

    @classmethod
    def start_job(
        cls,
        task_id: str,
        user_text: str,
        timeout_ms: int | None,
        previous_requirements: Dict[str, Any] | None = None,
    ) -> None:
        cancel_event = asyncio.Event()
        coro = _run_transport_pipeline(
            task_id, user_text, cancel_event, timeout_ms, previous_requirements
        )
        t = asyncio.create_task(coro)
        cls._jobs[task_id] = {"task": t, "cancel_event": cancel_event}

    @classmethod
    def schedule_await_timeout(
        cls, task_id: str, state: TaskState, timeout_ms: int | None
    ) -> None:
        # 取消同一任务此前的等待定时器
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
                    TaskManager.update_task_status(task_id, TaskState.Canceled)
                elif (
                    state == TaskState.AwaitingCompletion
                    and task.status.state == TaskState.AwaitingCompletion
                ):
                    TaskManager.update_task_status(task_id, TaskState.Completed)
            except asyncio.CancelledError:
                pass

        cls._await_timers[task_id] = asyncio.create_task(
            _wait_then_transition(), name=f"transport-await-timeout-{task_id}"
        )


async def decide_accept(user_text: str) -> dict:
    DECIDE_PROMPT = (
        "你是【全国城际交通规划 Agent】的请求门卫。\n\n"
        "[Agent 职责与范围]\n"
        f"{CAPABILITIES_SNIPPET}\n\n"
        "[你的任务]\n"
        "- 仅判断该请求是否属于中国境内城际交通相关，是否应由本 Agent 处理。\n"
        "- 不要因为缺少日期/人数/预算等细节而拒绝；只要主题明确为中国境内‘城际/跨城’交通相关，即判定为 accept（后续由系统引导补充信息）。\n"
        "- 明确判定为 accept 的典型表达：包含两座不同城市与‘A→B’/‘A到B’/‘A至B’等跨城措辞，或出现‘城际交通/高铁/动车/长途客车’等关键词。\n"
        "- 仅在明确越界时才判定为 reject：例如‘城市内’交通（地铁/公交/打车/步行等）、境外或非交通类任务。\n\n"
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
        obj = json.loads(raw or "{}")
    except Exception:
        obj = {"decision": "accept"}
    if obj.get("decision") not in ("accept", "reject"):
        obj["decision"] = "accept"
    if obj.get("decision") == "reject" and not obj.get("reason"):
        obj["reason"] = "请求不在城际交通规划范围"
    logger.info("event=decide_accept result=%s", obj)
    return obj


async def _run_transport_pipeline(
    task_id: str,
    user_text: str,
    cancel_event: asyncio.Event,
    timeout_ms: int | None,
    previous_requirements: Dict[str, Any] | None,
) -> None:
    async def _work():
        # 将任务置为 Working
        TaskManager.update_task_status(task_id, TaskState.Working)
        # 上下文合并
        prev_req = previous_requirements
        if not prev_req:
            ctx = TASK_CTX.get(task_id, {})
            prev_req = (
                ctx.get("requirements")
                if isinstance(ctx.get("requirements"), dict)
                else None
            )
        # 分析需求
        analyzed = await analyze_requirements(user_text, prev_req)
        req = analyzed.get("requirements") or {}
        # 保存上下文
        ctx = TASK_CTX.setdefault(task_id, {})
        ctx["requirements"] = req
        ctx.setdefault("history", []).append(user_text)
        logger.info(
            "event=analyze_requirements task_id=%s analyzed=%s", task_id, analyzed
        )
        # 如果被判定 reject（仅首次考虑），不直接拒绝，而是引导用户补充/澄清，进入 AwaitingInput。
        if analyzed.get("decision") == "reject" and not prev_req:
            reason = analyzed.get("reason", "请求不在城际交通规划范围")
            # 结合已计算的缺失信息，给出更清晰的补充指引
            g_missing = req.get("globalMissing") or []
            per_skill_missing = req.get("perSkillMissing") or {}
            parts: List[str] = []
            if g_missing:
                tips_map = {
                    "from_city/to_city 或 city_sequence": "出发与到达城市（或城市序列）",
                    "dates_or_days": "出行日期或天数",
                }
                human = "；".join([tips_map.get(m, m) for m in g_missing])
                parts.append("[全局必填]：" + human)
            for sid, miss in per_skill_missing.items():
                if not miss:
                    continue
                name = SKILL_CATALOG.get(sid, {}).get("name", sid)
                parts.append(f"[{name}] 需要：" + "、".join(miss))
            tips = ("\n" + "\n".join(parts)) if parts else ""
            guidance = (
                f"请求需要澄清：{reason}。请确认需求为中国境内城市与城市之间的出行，并补充必要信息。"
                + tips
                + "\n示例：从=上海；到=北京；日期=2025-10-02；人数=2；偏好=时间优先；预算=¥400-700/人。"
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            t = TaskManager.get_task(task_id)
            timeout_ms2 = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            TransportJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms2
            )
            return

        # 必填校验
        g_missing = req.get("globalMissing", [])
        per_skill_missing = req.get("perSkillMissing", {}) or {}
        if g_missing or any(per_skill_missing.values()):
            parts: List[str] = []
            if g_missing:
                tips_map = {
                    "from_city/to_city 或 city_sequence": "出发与到达城市（或城市序列）",
                    "dates_or_days": "出行日期或天数",
                }
                human = "；".join([tips_map.get(m, m) for m in g_missing])
                parts.append("[全局必填]：" + human)
            for sid, miss in per_skill_missing.items():
                if not miss:
                    continue
                name = SKILL_CATALOG.get(sid, {}).get("name", sid)
                parts.append(f"[{name}] 需要：" + "、".join(miss))
            guidance = (
                "信息不足：\n"
                + "\n".join(parts)
                + "\n示例：从=上海；到=北京；日期=2025-10-02；人数=2；偏好=时间优先；预算=¥400-700/人。"
            )
            TaskManager.update_task_status(
                task_id, TaskState.AwaitingInput, [TextDataItem(text=guidance)]
            )
            t = TaskManager.get_task(task_id)
            timeout_ms2 = getattr(
                t, "_aip_awaiting_input_timeout_ms", DEFAULT_AWAITING_INPUT_TIMEOUT_MS
            )
            TransportJobManager.schedule_await_timeout(
                task_id, TaskState.AwaitingInput, timeout_ms2
            )
            return

        # 生成产出
        plan = await produce_output(req, user_text)
        product = Product(
            id=f"product-{uuid.uuid4()}",
            name="全国城际交通多技能方案",
            dataItems=[TextDataItem(text=plan)],
        )
        TaskManager.set_products(task_id, [product])
        # 若产品写入因大小限制导致任务已失败，则不要覆盖为 AwaitingCompletion
        current = TaskManager.get_task(task_id)
        if current and current.status.state == TaskState.Failed:
            return
        TaskManager.update_task_status(task_id, TaskState.AwaitingCompletion)
        t2 = TaskManager.get_task(task_id)
        timeout_ms3 = getattr(
            t2,
            "_aip_awaiting_completion_timeout_ms",
            DEFAULT_AWAITING_COMPLETION_TIMEOUT_MS,
        )
        TransportJobManager.schedule_await_timeout(
            task_id, TaskState.AwaitingCompletion, timeout_ms3
        )

    try:
        if timeout_ms and timeout_ms > 0:
            await asyncio.wait_for(_work(), timeout=timeout_ms / 1000.0)
        else:
            await _work()
    except asyncio.TimeoutError:
        TaskManager.update_task_status(
            task_id, TaskState.Failed, [TextDataItem(text="执行超时，请稍后重试。")]
        )
    except asyncio.CancelledError:
        # 在某些运行时环境中，wait_for 超时会导致内部任务收到取消信号；这里也统一标记为 Failed
        TaskManager.update_task_status(
            task_id, TaskState.Failed, [TextDataItem(text="任务被取消或超时")]
        )
    except Exception as e:
        TaskManager.update_task_status(
            task_id,
            TaskState.Failed,
            [TextDataItem(text=f"处理请求时发生错误: {str(e)}")],
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
    work_timeout_ms = params.get("workTimeout") or DEFAULT_WORK_TIMEOUT_MS

    user_text = extract_text_from_message(message)

    # 若响应时限极短，预判无法按时决策，直接拒绝（与酒店代理保持一致的体验）
    estimated_first_llm_ms = 2000
    if (
        response_timeout_ms is not None
        and isinstance(response_timeout_ms, int)
        and response_timeout_ms < estimated_first_llm_ms
    ):
        rejected = TaskManager.create_task(
            message,
            initial_state=TaskState.Rejected,
            data_items=[TextDataItem(text="无法在指定 responseTimeout 内完成决策")],
        )
        return rejected

    # 门卫判断
    gate = await decide_accept(user_text)
    if gate.get("decision", "accept") == "reject":
        rejected = TaskManager.create_task(
            message,
            initial_state=TaskState.Rejected,
            data_items=[
                TextDataItem(text=gate.get("reason", "请求不在城际交通规划范围"))
            ],
        )
        return rejected

    # 接受任务
    accepted = TaskManager.create_task(message, initial_state=TaskState.Accepted)
    setattr(accepted, "_aip_awaiting_input_timeout_ms", awaiting_input_timeout_ms)
    setattr(
        accepted, "_aip_awaiting_completion_timeout_ms", awaiting_completion_timeout_ms
    )
    setattr(accepted, "_aip_max_products_bytes", max_products_bytes)

    # 调度后台作业
    TransportJobManager.start_job(accepted.id, user_text, work_timeout_ms, None)
    logger.info("event=task_accepted task_id=%s", accepted.id)
    return accepted


async def on_cancel(message: Message, task: Task) -> Task:
    TransportJobManager.cancel(task.id)
    return await DefaultHandlers.cancel(message, task)


async def on_continue(message: Message, task: Task) -> Task:
    # 仅支持在 AwaitingInput/AwaitingCompletion 下继续
    if task.status.state not in (TaskState.AwaitingInput, TaskState.AwaitingCompletion):
        return task
    user_text = extract_text_from_message(message)
    # 读取已存在的 requirements
    prev_req = None
    ctx = TASK_CTX.get(task.id)
    if ctx:
        prev_req = (
            ctx.get("requirements")
            if isinstance(ctx.get("requirements"), dict)
            else None
        )
    # 安排后台重新工作
    TransportJobManager.start_job(task.id, user_text, DEFAULT_WORK_TIMEOUT_MS, prev_req)
    # 按默认行为：添加消息历史，不改变状态（由后台切换至 Working）
    TaskManager.add_message_to_history(task.id, message)
    return task


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
    json_path = os.path.join(os.path.dirname(__file__), "china_transport.json")
    mtls_config = load_mtls_config_from_json(json_path)

    logger.info(
        "event=server_start host=0.0.0.0 port=8016 mtls=enabled aic=%s", mtls_config.aic
    )

    uvicorn.run(
        "china_transport:app",
        host="0.0.0.0",
        port=8016,
        reload=True,
        workers=1,
        ssl_keyfile=str(mtls_config.key_file),
        ssl_certfile=str(mtls_config.cert_file),
        ssl_ca_certs=str(mtls_config.ca_cert_file),
        ssl_cert_reqs=ssl.CERT_REQUIRED,  # 要求客户端必须提供证书（mTLS双向认证）
    )
