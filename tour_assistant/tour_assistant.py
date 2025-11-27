import os
import sys
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base import get_agent_logger, truncate

from acps_aip.aip_rpc_client import AipRpcClient
from acps_aip.aip_base_model import TaskState
from acps_aip.mtls_config import load_mtls_config_from_json

# ============================
# 环境与基础配置加载
# ============================
load_dotenv()  # 加载 .env 文件中的环境变量

# 领导（协调）Agent 标识
LEADER_ID = os.getenv("LEADER_AGENT_ID", "tour-assistant-leader-001")
LOG_LEVEL = os.getenv("LEADER_LOG_LEVEL", "DEBUG").upper()
# 示例配置文件目录（用于简易“服务发现”）仅支持新版：各自目录下（如 china_hotel/china_hotel.json）
# 仓库根目录（tour_assistant.py 的上一级目录）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# OpenAI / 大模型配置（兼容 base_url 简单形式；保持与其他文件一致）
openai.api_key = os.getenv("OPENAI_API_KEY")
_raw_base_url = os.getenv("OPENAI_BASE_URL")
if _raw_base_url and not _raw_base_url.endswith("/"):
    _raw_base_url += "/"
openai.base_url = _raw_base_url
LLM_MODEL = os.getenv("OPENAI_MODEL", "Doubao-pro-32k")
DISCOVERY_BASE_URL = os.getenv("DISCOVERY_BASE_URL")
_DISCOVERY_TIMEOUT_SECONDS = 5.0

# ============================
# FastAPI 应用实例
# ============================
app = FastAPI(
    title="Tour Assistant (Leader)",
    description="旅游助理（协调者）：分析用户需求 → 调度 Partner → 整合结果。",
)

# --- CORS (前端在同域静态托管时仅预防未来跨域；允许本地开发端口) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 若需限制可改为具体域名列表
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = get_agent_logger("agent.tour_assistant", "LEADER_LOG_LEVEL", LOG_LEVEL)
logger.info(
    "event=app_start leader_id=%s model=%s log_level=%s",
    LEADER_ID,
    LLM_MODEL,
    LOG_LEVEL,
)

# ============================
# mTLS 配置加载
# ============================
_mtls_json_path = os.path.join(os.path.dirname(__file__), "tour_assistant.json")
_mtls_config = load_mtls_config_from_json(_mtls_json_path)
_client_ssl_context = _mtls_config.create_client_ssl_context()
logger.info(
    "event=mtls_config_loaded aic=%s cert_dir=%s",
    _mtls_config.aic,
    _mtls_config.cert_dir,
)

# ============================
# 会话与上下文存储（内存版；真实系统应使用数据库/缓存）
# sessions[session_id] = { messages: [...], last_analysis: {...}, created_at: ... }
# ============================
sessions: Dict[str, Dict[str, Any]] = {}


class UserRequest(BaseModel):
    """用户请求数据模型：允许复用已有 session，或创建新的会话"""

    session_id: str | None = None
    query: str


_BEIJING_DISCOVERY_QUERIES: Dict[str, str] = {
    "beijing_urban": "北京城区旅游智能体",
    "beijing_rural": "北京郊区自然景点智能体",
    "beijing_catering": "北京美食推荐智能体",
}


def _load_static_agent(agent_type: str) -> dict | None:
    mapping = {
        "beijing_urban": ("beijing_urban", "beijing_urban.json"),
        "beijing_rural": ("beijing_rural", "beijing_rural.json"),
        "beijing_catering": ("beijing_catering", "beijing_catering.json"),
        "china_hotel": ("china_hotel", "china_hotel.json"),
        "china_transport": ("china_transport", "china_transport.json"),
    }
    entry = mapping.get(agent_type)
    if not entry:
        return None
    file_path = os.path.join(ROOT_DIR, entry[0], entry[1])
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(
                "event=service_discovery_static_success agent_type=%s file=%s",
                agent_type,
                file_path,
            )
            return data
    except FileNotFoundError:
        logger.error(
            "event=service_discovery_static_missing agent_type=%s file=%s",
            agent_type,
            file_path,
        )
    except json.JSONDecodeError:
        logger.error(
            "event=service_discovery_static_invalid_json agent_type=%s file=%s",
            agent_type,
            file_path,
        )
    return None


def _discover_beijing_agent(agent_type: str) -> dict | None:
    if not DISCOVERY_BASE_URL:
        logger.debug(
            "event=discovery_skipped reason=no_base_url agent_type=%s", agent_type
        )
        return None
    query = _BEIJING_DISCOVERY_QUERIES.get(agent_type)
    if not query:
        return None
    url = DISCOVERY_BASE_URL.rstrip("/") + "/api/discovery/"
    try:
        with httpx.Client(timeout=_DISCOVERY_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json={"query": query, "limit": 1})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(
            "event=discovery_request_failed agent_type=%s error=%s",
            agent_type,
            exc,
        )
        return None

    agents = payload.get("agents") if isinstance(payload, dict) else None
    if not agents:
        logger.warning(
            "event=discovery_empty_result agent_type=%s payload_preview=%s",
            agent_type,
            truncate(str(payload), 120),
        )
        return None

    first = agents[0]
    candidate = None
    if isinstance(first, dict):
        if "acs" in first and isinstance(first["acs"], dict):
            candidate = first["acs"]
        elif "acs" in first and isinstance(first["acs"], str):
            try:
                candidate = json.loads(first["acs"])
            except json.JSONDecodeError:
                logger.warning(
                    "event=discovery_invalid_acs_string agent_type=%s",
                    agent_type,
                )

    if candidate:
        logger.info(
            "event=discovery_success agent_type=%s aic=%s",
            agent_type,
            candidate.get("aic"),
        )
        return candidate

    logger.warning(
        "event=discovery_result_unusable agent_type=%s first_preview=%s",
        agent_type,
        truncate(str(first), 120),
    )
    return None


def find_agent_service(agent_type: str) -> dict | None:
    """优先通过 discovery-server 动态发现北京 Agent，失败时回退到本地静态配置。"""
    if agent_type in _BEIJING_DISCOVERY_QUERIES:
        discovered = _discover_beijing_agent(agent_type)
        if discovered:
            return discovered
    return _load_static_agent(agent_type)


def extract_jsonrpc_endpoint(agent_info: dict) -> str | None:
    """从能力描述中提取 JSONRPC 的 endpoint URL。

    期望结构示例（可能的变体）:
    {
      "endPoints": [
         {"transport":"JSONRPC","url":"http://host:8011/acps-aip-v1/rpc"},
         {"transport":"SSE","url":"..."}
      ]
    }

    返回首个 transport == JSONRPC 的 url。
    """
    if not agent_info:
        return None
    # 兼容不同字段命名: endPoints | endpoints | endpoint
    endpoints = (
        agent_info.get("endPoints")
        or agent_info.get("endpoints")
        or agent_info.get("endpoint")
        or []
    )
    # 如果是字典（例如 {"rpc": {..}, "sse": {..}}）转为列表
    if isinstance(endpoints, dict):
        endpoints = list(endpoints.values())
    if not isinstance(endpoints, list):
        return None
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        transport = str(ep.get("transport", "")).upper()
        if transport == "JSONRPC":
            url = ep.get("url") or ep.get("URI") or ep.get("endpoint")
            if url:
                logger.debug(
                    "event=extract_endpoint transport=%s url=%s", transport, url
                )
                return url
    return None


# ============================
# 多阶段提示词模板（借鉴早期 main.py 结构思想，并做简化）
# ============================
ANALYSIS_PROMPT_TEMPLATE = """你是资深“多维旅游需求分析与路由助手”。请对用户输入进行五大维度拆解：
1) intercity_transport 城际交通（跨城市移动、到达/离开方式、段间衔接）
2) hotel 住宿（酒店/民宿/晚数/预算/位置偏好）
3) urban 城市城区人文/文化/博物馆/地标/历史/市内体验
4) rural 郊区/自然/长城/山/湖/户外/生态/周边拓展
5) food 美食/餐厅/本地饮食/三餐与小吃

当前已部署可用代理（agentId）与其覆盖：
 - china_transport_agent_001 : intercity_transport (全国)
 - china_hotel_agent_001          : hotel (全国)
 - beijing_urban_agent_001        : urban (仅北京)
 - beijing_rural_agent_001        : rural (仅北京)
 - beijing_catering_agent_001     : food  (仅北京)

若目的地为非北京城市：只能实际调用 hotel 与 intercity_transport；其它需要的维度列入 unavailable_dimensions。

【输出要求】
严格输出 JSON（无多余文本），结构如下（不要出现注释）：
{
  "destination_city": "...",
  "request_type": "new_plan|modify_plan|add_detail|question",
  "user_needs": {
     "duration": "...",
     "budget": "...",
     "preferences": ["..."],
     "special_requirements": "..."
  },
  "dimensions": {
     "intercity_transport": {"needed": true,  "reason": "...", "sub_query": "..."},
     "hotel":              {"needed": true,  "reason": "...", "sub_query": "..."},
     "urban":              {"needed": false, "reason": "...", "sub_query": "..."},
     "rural":              {"needed": false, "reason": "...", "sub_query": "..."},
     "food":               {"needed": false, "reason": "...", "sub_query": "..."}
  },
  "required_agents": ["china_transport_agent_001", "china_hotel_agent_001"],
  "unavailable_dimensions": ["urban", "food"],
  "routing_reason": "总体路由理由",
  "notes": "补充说明，可为空"
}

规范：
1. 所有 needed=false 仍需给出 reason；若 needed=true 必须给出 sub_query（可精炼原始需求仅保留该维度相关要素）。
2. sub_query：避免包含其它维度的冗余内容，使用简洁指令式语句。
3. preferences 用字符串数组，不足则空数组。缺失字段给空字符串或空数组，不要 null。
4. 不允许添加未定义的顶层字段。

【历史上下文】
{context}

【当前请求】
{query}
"""


# 简单占位符替换工具：仅替换已知 {name}，不会解析其它大括号，避免 JSON 中的花括号触发 format KeyError
def _fill_template(tpl: str, **kwargs) -> str:
    for k, v in kwargs.items():
        tpl = tpl.replace("{" + k + "}", v)
    return tpl


INTEGRATION_PROMPT_TEMPLATE = """你是旅游方案整合助手，请基于分析结果、分解子查询与多个智能体产出，生成用户可直接采用的最终方案。

【会话上下文】
{context}

【用户请求】
{query}

【需求分析 JSON】
{analysis_json}

【任务分解（decomposition）】
{decomposition_block}

【各智能体结果汇总】
{partner_results_block}

【整合要求】
1. 若 request_type = new_plan：提供 行程总览 / 分时安排 / 预算提示 / 注意事项 / 组合价值说明（若多代理）。
2. 若为 modify_plan 或 add_detail：说明变化点，并给出更新后的完整方案。
3. 多代理时需合并去重：若城区与郊区天数需要拆分，明确分日或分段结构。
4. 重点突出亮点与差异化体验，避免简单拼接；删除互相冲突或重复景点。
5. 结构清晰，使用分级标题或条目。无信息的板块不输出标题。

直接输出最终方案文本（不要再输出 JSON）。"""


# ============================
# 会话 & 辅助函数
# ============================
def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],  # [{role, content, time}]
            "last_analysis": None,
            # 记录与各 Partner 的进行中任务与状态
            # partner_tasks[agent_id] = {"task_id": str, "state": TaskState, "last_product": str|None,
            #                            "awaiting_prompt": str|None, "updated_at": iso, "sub_query": str|None}
            "partner_tasks": {},
            # 当前等待用户补充的信息队列（用于前端提示）
            # [{"agent_id": str, "task_id": str, "question": str, "time": iso}]
            "pending_questions": [],
        }
    return sessions[session_id]


def append_message(session_id: str, role: str, content: str):
    sess = get_session(session_id)
    sess["messages"].append(
        {
            "role": role,
            "content": content,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


def _extract_text_from_data_items(items: List[Any] | None) -> str:
    if not items:
        return ""
    texts: List[str] = []
    for di in items:
        try:
            # TextDataItem has attribute 'text'
            t = getattr(di, "text", None)
            if t:
                texts.append(str(t))
        except Exception:
            continue
    return "\n".join(texts).strip()


def _save_partner_task_state(
    session_id: str,
    agent_id: str,
    task_id: str,
    state: Any,
    sub_query: str | None = None,
    last_product: str | None = None,
    awaiting_prompt: str | None = None,
):
    sess = get_session(session_id)
    sess["partner_tasks"][agent_id] = {
        "task_id": task_id,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_product": last_product,
        "awaiting_prompt": awaiting_prompt,
        "sub_query": sub_query
        or sess["partner_tasks"].get(agent_id, {}).get("sub_query"),
    }
    # 若处于 AwaitingInput，同时记录 pending question 供前端展示
    sess.setdefault("pending_questions", [])
    sess["pending_questions"] = [
        item for item in sess["pending_questions"] if item.get("agent_id") != agent_id
    ]
    if state == TaskState.AwaitingInput and awaiting_prompt:
        sess["pending_questions"].append(
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "question": awaiting_prompt,
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )


def _build_continue_payload(
    *,
    user_input: str,
    agent_id: str | None = None,
    sub_query: str | None = None,
    awaiting_prompt: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Package the Continue payload with helpful context for the partner agent."""
    # Keep the message human-readable so downstream LLM partners can interpret it easily.
    sections: list[str] = []
    if sub_query:
        sections.append(f"子任务指令：{sub_query.strip()}")
    if awaiting_prompt:
        sections.append(f"待补充问题：{awaiting_prompt.strip()}")
    sections.append(f"用户回复：{user_input.strip()}")
    return "\n\n".join(sections)


# 删除 LLM 字段抽取：Partner 现已支持在 Continue 时自行解析/合并补充


def build_context(session_id: str) -> str:
    sess = get_session(session_id)
    msgs = sess["messages"][-5:]  # 只取最近 5 条对话
    parts = ["=== 最近对话 ==="]
    for m in msgs:
        cn_role = "用户" if m["role"] == "user" else "助理"
        parts.append(f"{cn_role}: {m['content']}")
    return "\n".join(parts)


# ============================
# LLM 调用阶段函数
# ============================
def llm_analysis(query: str, session_id: str) -> Dict[str, Any]:
    """阶段1：五维度需求分析 → 返回结构化 JSON（失败时使用启发式回退）"""
    context = build_context(session_id)
    prompt = _fill_template(ANALYSIS_PROMPT_TEMPLATE, context=context, query=query)

    # ---------------- Heuristic helpers ----------------
    CITY_KWS = [
        "北京",
        "上海",
        "广州",
        "深圳",
        "杭州",
        "成都",
        "西安",
        "南京",
        "重庆",
        "厦门",
        "昆明",
        "青岛",
        "天津",
        "苏州",
        "桂林",
        "三亚",
    ]

    urban_kw = [
        "博物馆",
        "历史",
        "文化",
        "城区",
        "市区",
        "老街",
        "胡同",
        "地标",
        "艺术",
        "展览",
    ]
    rural_kw = [
        "郊区",
        "自然",
        "山",
        "湖",
        "徒步",
        "露营",
        "长城",
        "森林",
        "草原",
        "湿地",
    ]
    food_kw = ["美食", "餐厅", "吃", "早餐", "午餐", "晚餐", "小吃", "餐饮", "饭店"]
    hotel_kw = ["酒店", "住宿", "住", "客栈", "民宿", "晚", "房"]
    transport_kw = ["高铁", "火车", "航班", "飞机", "动车", "交通", "出行", "到", "→"]

    def heuristic_detect(text: str) -> Dict[str, Any]:
        lower = text.lower()
        # destination city: pick first city keyword appearing
        dest_city = next((c for c in CITY_KWS if c in text), "未指定")
        has_multi_cities = (
            sum(1 for c in CITY_KWS if c in text) >= 2 or "→" in text or "到" in text
        )
        dim = {}
        dim["intercity_transport"] = has_multi_cities or any(
            k in text for k in transport_kw
        )
        dim["hotel"] = any(k in text for k in hotel_kw)
        dim["urban"] = any(k in text for k in urban_kw)
        dim["rural"] = any(k in text for k in rural_kw)
        dim["food"] = any(k in text for k in food_kw)
        # Always at least one dimension; if none found assume urban baseline
        if not any(dim.values()):
            dim["urban"] = True
        return {"destination_city": dest_city, "dim_flags": dim}

    heur_basic = heuristic_detect(query)

    def build_sub_query(dimension: str, original: str, city: str) -> str:
        prefix_map = {
            "intercity_transport": "请基于以下需求规划城际交通段并给出高铁/航班/衔接建议:",
            "hotel": "请根据以下需求提炼住宿晚数/预算/区域偏好并给出酒店分档建议:",
            "urban": f"请聚焦{city if city!='未指定' else '目的地'}城区文化/人文/地标体验规划, 忽略自然郊野要素:",
            "rural": f"请聚焦{city if city!='未指定' else '目的地'}郊区/自然/户外/长城/生态体验规划, 忽略城区要素:",
            "food": f"请为{city if city!='未指定' else '目的地'}行程生成本地美食与餐厅安排 (三餐+特色小吃):",
        }
        return prefix_map.get(dimension, "请针对该维度生成子任务:") + " " + original

    # Try LLM first
    llm_data: Dict[str, Any] | None = None
    logger.info(
        "event=analysis_start session_id=%s query_chars=%d", session_id, len(query)
    )
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1100,
        )
        content = resp.choices[0].message.content.strip()
        logger.info(
            "event=analysis_llm_end session_id=%s raw_chars=%d preview=%s",
            session_id,
            len(content),
            truncate(content.replace("\n", " "), 160),
        )
        try:
            llm_data = json.loads(content)
            logger.debug(
                "event=analysis_json_parse_success session_id=%s keys=%s",
                session_id,
                list(llm_data.keys()),
            )
        except json.JSONDecodeError:
            logger.error(
                "event=analysis_json_parse_error session_id=%s preview=%s",
                session_id,
                truncate(content, 120),
            )
            llm_data = None
    except Exception as e:
        logger.exception("event=analysis_llm_exception session_id=%s", session_id)
        llm_data = None

    if llm_data is None:
        # ---- Heuristic fallback structure ----
        logger.info(
            "event=analysis_fallback_heuristic session_id=%s dest=%s",
            session_id,
            heur_basic["destination_city"],
        )
        flags = heur_basic["dim_flags"]
        city = heur_basic["destination_city"]
        dimensions_obj = {}
        for d, enabled in flags.items():
            dimensions_obj[d] = {
                "needed": bool(enabled),
                "reason": (
                    "启发式判定包含相关关键词" if enabled else "未检测到相关关键词"
                ),
                **({"sub_query": build_sub_query(d, query, city)} if enabled else {}),
            }
        llm_data = {
            "destination_city": city,
            "request_type": "new_plan",
            "user_needs": {
                "duration": "",
                "budget": "",
                "preferences": [],
                "special_requirements": query,
            },
            "dimensions": dimensions_obj,
            "routing_reason": "LLM解析失败，使用启发式结果。",
            "notes": "fallback",
        }

    # ---- Post-process / enrich ----
    city = llm_data.get("destination_city") or heur_basic["destination_city"]
    dims = llm_data.get("dimensions") or {}
    # Ensure all five present
    for d in ["intercity_transport", "hotel", "urban", "rural", "food"]:
        if d not in dims:
            dims[d] = {"needed": False, "reason": "缺失补齐", "sub_query": ""}
        else:
            # normalize fields
            dims[d]["needed"] = bool(dims[d].get("needed"))
            if dims[d]["needed"] and not dims[d].get("sub_query"):
                dims[d]["sub_query"] = build_sub_query(d, query, city)
            if not dims[d].get("reason"):
                dims[d]["reason"] = "未提供理由"
    llm_data["dimensions"] = dims

    # Determine available agents
    available_agents: List[str] = []
    unavailable_dimensions: List[str] = []
    dest_is_beijing = city and "北京" in city
    if dims["intercity_transport"]["needed"]:
        available_agents.append("china_transport_agent_001")
    if dims["hotel"]["needed"]:
        available_agents.append("china_hotel_agent_001")
    if dims["urban"]["needed"]:
        if dest_is_beijing:
            available_agents.append("beijing_urban_agent_001")
        else:
            unavailable_dimensions.append("urban")
    if dims["rural"]["needed"]:
        if dest_is_beijing:
            available_agents.append("beijing_rural_agent_001")
        else:
            unavailable_dimensions.append("rural")
    if dims["food"]["needed"]:
        if dest_is_beijing:
            available_agents.append("beijing_catering_agent_001")
        else:
            unavailable_dimensions.append("food")

    llm_data["required_agents"] = available_agents
    llm_data["unavailable_dimensions"] = unavailable_dimensions
    # For backwards compatibility keep task_priority same order
    llm_data["task_priority"] = available_agents[:]
    logger.info(
        "event=analysis_resolved session_id=%s city=%s agents=%s unavailable=%s",
        session_id,
        city,
        ",".join(available_agents) or "-",
        ",".join(unavailable_dimensions) or "-",
    )

    get_session(session_id)["last_analysis"] = llm_data
    return llm_data


async def call_partner(
    agent_id: str, user_query: str, session_id: str, *, sub_query: str | None = None
) -> Dict[str, Any]:
    """通用调用函数：根据 agent_id 查找 ACS → 提取 JSONRPC → 发起 AIP RPC 交互。

    返回结构:
      {
        "agent_id": agent_id,
        "success": bool,
        "state": state_value,
        "product_text": str|None,
        "raw_task": task_json|None,
        "error": 错误信息（可选）
      }
    """
    # 将内部 ID 映射为查找文件用 key
    lookup_key = None
    if agent_id == "beijing_urban_agent_001":
        lookup_key = "beijing_urban"
    elif agent_id == "beijing_rural_agent_001":
        lookup_key = "beijing_rural"
    elif agent_id == "china_transport_agent_001":
        lookup_key = "china_transport"
    elif agent_id == "china_hotel_agent_001":
        lookup_key = "china_hotel"
    elif agent_id == "beijing_catering_agent_001":
        lookup_key = "beijing_catering"
    if not lookup_key:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unknown-agent",
            "product_text": None,
            "error": "unsupported agent id",
        }
    logger.info(
        "event=partner_call_start session_id=%s agent_id=%s lookup_key=%s",
        session_id,
        agent_id,
        lookup_key,
    )
    agent_info = find_agent_service(lookup_key)
    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        logger.error(
            "event=partner_endpoint_missing session_id=%s agent_id=%s",
            session_id,
            agent_id,
        )
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "error": f"{agent_id} JSONRPC endpoint not available",
        }
    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
    try:
        # 新建任务：Start
        task = await client.start_task(session_id=session_id, user_input=user_query)
        logger.debug(
            "event=partner_task_started session_id=%s agent_id=%s task_id=%s state=%s",
            session_id,
            agent_id,
            task.id,
            task.status.state,
        )
        # 轮询与状态驱动
        max_loops = 60  # 最长 60s 轮询（每秒一次）
        loops = 0
        last_state = None
        did_auto_continue = False  # 避免在 AwaitingInput 时重复持续发送 Continue
        while True:
            state = task.status.state
            if state != last_state:
                logger.debug(
                    "event=partner_state_change session_id=%s agent_id=%s task_id=%s state=%s",
                    session_id,
                    agent_id,
                    task.id,
                    state,
                )
                last_state = state
            # 保存当前状态
            _save_partner_task_state(
                session_id,
                agent_id,
                task.id,
                state,
                sub_query=sub_query,
                last_product=(
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products and task.products[0].dataItems
                    else None
                ),
                awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
            )

            # 终态
            if state in (
                TaskState.Completed,
                TaskState.Canceled,
                TaskState.Failed,
                TaskState.Rejected,
            ):
                break

            # 需要用户/领导补充
            if state == TaskState.AwaitingInput:
                prompt_text = _extract_text_from_data_items(task.status.dataItems)
                logger.info(
                    "event=partner_awaiting_input session_id=%s agent_id=%s prompt=%s",
                    session_id,
                    agent_id,
                    truncate(prompt_text, 160),
                )
                # Leader 不再自动构造补充，直接返回由前端收集用户输入再继续
                break

            # 有产出，等待确认
            if state == TaskState.AwaitingCompletion:
                product_text = (
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products
                    else ""
                )
                satisfied, feedback = _evaluate_product_satisfaction(
                    sub_query or user_query, product_text
                )
                if satisfied:
                    task = await client.complete_task(
                        task_id=task.id, session_id=session_id
                    )
                    break
                else:
                    # Leader 不再改写反馈内容，直接使用用户输入进行下一次 Continue
                    task = await client.continue_task(
                        task_id=task.id,
                        session_id=session_id,
                        user_input=(feedback or "请继续完善上述方案。"),
                    )
                    loops += 1
                    if loops >= max_loops:
                        break
                    await asyncio.sleep(1)
                    task = await client.get_task(task.id, session_id)
                    continue

            # 仍在处理
            loops += 1
            if loops >= max_loops:
                break
            await asyncio.sleep(1)
            task = await client.get_task(task.id, session_id)

        # 汇总返回
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        if (not product_text) and task.status.state == TaskState.AwaitingInput:
            product_text = _extract_text_from_data_items(task.status.dataItems)

        result = {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
        logger.info(
            "event=partner_call_end session_id=%s agent_id=%s state=%s success=%s product_chars=%s",
            session_id,
            agent_id,
            task.status.state,
            result["success"],
            len(product_text) if product_text else 0,
        )
        return result
    except Exception as e:
        logger.exception(
            "event=partner_call_exception session_id=%s agent_id=%s",
            session_id,
            agent_id,
        )
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "error",
            "product_text": None,
            "error": str(e),
        }
    finally:
        await client.close()


async def continue_partner(
    agent_id: str,
    task_id: str,
    user_input: str,
    session_id: str,
    *,
    sub_query: str | None = None,
    awaiting_prompt: str | None = None,
) -> Dict[str, Any]:
    """继续已有任务：发送 Continue → 轮询 → 根据 AwaitingCompletion 进行评估或继续。

    返回结构同 call_partner。
    """
    # 将内部 ID 映射为查找文件用 key（沿用上方逻辑）
    lookup_key = None
    if agent_id == "beijing_urban_agent_001":
        lookup_key = "beijing_urban"
    elif agent_id == "beijing_rural_agent_001":
        lookup_key = "beijing_rural"
    elif agent_id == "china_transport_agent_001":
        lookup_key = "china_transport"
    elif agent_id == "china_hotel_agent_001":
        lookup_key = "china_hotel"
    elif agent_id == "beijing_catering_agent_001":
        lookup_key = "beijing_catering"
    if not lookup_key:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unknown-agent",
            "product_text": None,
            "error": "unsupported agent id",
        }
    agent_info = find_agent_service(lookup_key)
    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "error": f"{agent_id} JSONRPC endpoint not available",
        }
    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
    try:
        # Continue → 轮询 → 分支
        task = await client.continue_task(
            task_id=task_id,
            session_id=session_id,
            user_input=_build_continue_payload(
                user_input=user_input,
                sub_query=sub_query,
                awaiting_prompt=awaiting_prompt,
            ),
        )
        # 与 call_partner 相同的轮询/评估逻辑
        max_loops = 60
        loops = 0
        last_state = None
        while True:
            state = task.status.state
            if state != last_state:
                last_state = state
                logger.debug(
                    "event=partner_state_change session_id=%s agent_id=%s task_id=%s state=%s",
                    session_id,
                    agent_id,
                    task.id,
                    state,
                )
            _save_partner_task_state(
                session_id,
                agent_id,
                task.id,
                state,
                sub_query=sub_query,
                last_product=(
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products and task.products[0].dataItems
                    else None
                ),
                awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
            )
            if state in (
                TaskState.Completed,
                TaskState.Canceled,
                TaskState.Failed,
                TaskState.Rejected,
            ):
                break
            if state == TaskState.AwaitingInput:
                break  # 继续等待用户
            if state == TaskState.AwaitingCompletion:
                product_text = (
                    _extract_text_from_data_items(task.products[0].dataItems)
                    if task.products
                    else ""
                )
                satisfied, feedback = _evaluate_product_satisfaction(
                    sub_query or user_input, product_text
                )
                if satisfied:
                    task = await client.complete_task(
                        task_id=task.id, session_id=session_id
                    )
                    break
                else:
                    task = await client.continue_task(
                        task_id=task.id,
                        session_id=session_id,
                        user_input=_build_continue_payload(
                            user_input=(feedback or "请继续完善上述方案。"),
                            sub_query=sub_query,
                            awaiting_prompt=awaiting_prompt,
                        ),
                    )
                    loops += 1
                    if loops >= max_loops:
                        break
                    await asyncio.sleep(1)
                    task = await client.get_task(task.id, session_id)
                    continue
            loops += 1
            if loops >= max_loops:
                break
            await asyncio.sleep(1)
            task = await client.get_task(task.id, session_id)
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        if (not product_text) and task.status.state == TaskState.AwaitingInput:
            product_text = _extract_text_from_data_items(task.status.dataItems)
        return {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
    finally:
        await client.close()


async def complete_partner(
    agent_id: str,
    task_id: str,
    session_id: str,
    *,
    sub_query: str | None = None,
) -> Dict[str, Any]:
    """尝试直接完成处于 AwaitingCompletion 的任务（用户明确表示已满足/请完成）。"""
    # 将内部 ID 映射为查找文件用 key
    if agent_id == "beijing_urban_agent_001":
        lookup_key = "beijing_urban"
    elif agent_id == "beijing_rural_agent_001":
        lookup_key = "beijing_rural"
    elif agent_id == "china_transport_agent_001":
        lookup_key = "china_transport"
    elif agent_id == "china_hotel_agent_001":
        lookup_key = "china_hotel"
    elif agent_id == "beijing_catering_agent_001":
        lookup_key = "beijing_catering"
    else:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unknown-agent",
            "product_text": None,
            "error": "unsupported agent id",
        }
    agent_info = find_agent_service(lookup_key)
    partner_url = extract_jsonrpc_endpoint(agent_info) if agent_info else None
    if not partner_url:
        return {
            "agent_id": agent_id,
            "success": False,
            "state": "unavailable",
            "product_text": None,
            "error": f"{agent_id} JSONRPC endpoint not available",
        }
    client = AipRpcClient(
        partner_url=partner_url, leader_id=LEADER_ID, ssl_context=_client_ssl_context
    )
    try:
        task = await client.complete_task(task_id=task_id, session_id=session_id)
        # 简短轮询一次，获取最终产品文本
        await asyncio.sleep(0.5)
        task = await client.get_task(task.id, session_id)
        _save_partner_task_state(
            session_id,
            agent_id,
            task.id,
            task.status.state,
            sub_query=sub_query,
            last_product=(
                _extract_text_from_data_items(task.products[0].dataItems)
                if task.products and task.products[0].dataItems
                else None
            ),
            awaiting_prompt=_extract_text_from_data_items(task.status.dataItems),
        )
        product_text = None
        if task.products and task.products[0].dataItems:
            product_text = _extract_text_from_data_items(task.products[0].dataItems)
        return {
            "agent_id": agent_id,
            "success": task.status.state not in (TaskState.Failed, TaskState.Rejected),
            "state": task.status.state,
            "product_text": product_text,
            "raw_task": task.model_dump(),
            "needs_user_input": task.status.state == TaskState.AwaitingInput,
        }
    finally:
        await client.close()


def _evaluate_product_satisfaction(
    requirement_text: str, product_text: str
) -> Tuple[bool, str | None]:
    """使用 LLM 对产出物进行轻量评估。返回 (是否满意, 如不满足给出反馈指令)。"""
    try:
        prompt = (
            "请判断给定的‘产出物’是否覆盖并满足‘需求’。只返回两行：\n"
            "第一行：YES 或 NO；\n"
            "第二行：若 NO，请用一句中文给出需要补充或修正的具体指令；若 YES 留空。\n\n"
            f"需求：\n{requirement_text}\n\n产出物：\n{product_text}"
        )
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        text = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ok = lines and lines[0].upper().startswith("YES")
        feedback = None
        if not ok and len(lines) >= 2:
            feedback = lines[1]
        return ok, feedback
    except Exception:
        # 回退策略：无法判断则默认满意，避免阻塞
        return True, None


def llm_integrate(
    query: str,
    analysis: Dict[str, Any],
    partner_results: Dict[str, Dict[str, Any]],
    session_id: str,
) -> str:
    """阶段3：整合结果生成最终答复（支持多代理）。"""
    context = build_context(session_id)
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    # 构造 partner results 文本块
    blocks = []
    for aid, res in partner_results.items():
        if aid == "beijing_urban_agent_001":
            label = "城区景点代理"
        elif aid == "beijing_rural_agent_001":
            label = "郊区景点代理"
        elif aid == "china_transport_agent_001":
            label = "城际交通代理"
        elif aid == "china_hotel_agent_001":
            label = "酒店代理"
        elif aid == "beijing_catering_agent_001":
            label = "美食代理"
        else:
            label = aid
        text_part = res.get("product_text") or res.get("error") or "(无产出)"
        blocks.append(f"[{label} {aid}]\n{text_part}\n")
    partner_results_block = "\n".join(blocks) if blocks else "(无代理产出)"
    decomposition_block = json.dumps(
        analysis.get("decomposition", {}), ensure_ascii=False, indent=2
    )
    prompt = _fill_template(
        INTEGRATION_PROMPT_TEMPLATE,
        context=context,
        query=query,
        analysis_json=analysis_json,
        partner_results_block=partner_results_block,
        decomposition_block=decomposition_block,
    )
    logger.info(
        "event=integration_start session_id=%s query_chars=%d partner_count=%d",
        session_id,
        len(query),
        len(partner_results),
    )
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1600,
        )
        text = resp.choices[0].message.content.strip()
        logger.info(
            "event=integration_llm_end session_id=%s chars=%d preview=%s",
            session_id,
            len(text),
            truncate(text.replace("\n", " "), 160),
        )
        return text
    except Exception as e:
        logger.exception("event=integration_exception session_id=%s", session_id)
        return f"整合阶段发生错误: {str(e)}\n\n原始多代理结果:\n{partner_results_block}"


@app.post("/user_api")
async def handle_user_request(user_request: UserRequest):
    """统一入口：多阶段执行流程

    阶段流程：
    1) 需求分析（LLM）
    2) 调用 Partner（当前仅北京城区景点规划师）
    3) 结果整合（LLM）
    4) 返回最终方案
    """
    session_id = user_request.session_id or f"session-{uuid.uuid4()}"
    is_new_session = user_request.session_id is None
    user_query = user_request.query
    logger.info(
        "event=request_received session_id=%s new_session=%s query_chars=%d",
        session_id,
        is_new_session,
        len(user_query),
    )

    # 记录用户消息
    append_message(session_id, "user", user_query)

    sess = get_session(session_id)
    partner_results: Dict[str, Dict[str, Any]] = {}
    partner_subqueries: Dict[str, str] = {}

    t0 = datetime.now(timezone.utc)
    analysis = llm_analysis(user_query, session_id)
    t1 = datetime.now(timezone.utc)
    dimensions = analysis.get("dimensions") or {}
    required_agents = analysis.get("required_agents") or []
    if isinstance(required_agents, str):
        required_agents = [required_agents]

    # 判断：这次输入是否用于补充进行中的任务（等待用户/完成确认）
    pending = [
        (aid, info)
        for aid, info in sess.get("partner_tasks", {}).items()
        if info.get("state") in (TaskState.AwaitingInput, TaskState.AwaitingCompletion)
    ]
    is_supplement = len(pending) > 0 and not is_new_session

    agent_dim_map = {
        "china_transport_agent_001": "intercity_transport",
        "china_hotel_agent_001": "hotel",
        "beijing_urban_agent_001": "urban",
        "beijing_rural_agent_001": "rural",
        "beijing_catering_agent_001": "food",
    }

    def resolve_sub_query(agent_id: str) -> str:
        dim_key = agent_dim_map.get(agent_id)
        if dim_key and dim_key in dimensions:
            sub_obj = dimensions.get(dim_key) or {}
            candidate = sub_obj.get("sub_query")
            if candidate:
                return candidate
        return (
            sess.get("partner_tasks", {}).get(agent_id, {}).get("sub_query")
            or user_query
        )

    if is_supplement:
        logger.info(
            "event=input_classified supplement=true session_id=%s waiting_agents=%s",
            session_id,
            ",".join([aid for aid, _ in pending]),
        )
        sub_payloads = []
        for aid, info in pending:
            sub_q = resolve_sub_query(aid)
            partner_subqueries[aid] = sub_q
            if aid in sess.get("partner_tasks", {}):
                sess["partner_tasks"][aid]["sub_query"] = sub_q
            wants_complete = ("完成" in user_query) or ("整理结果" in user_query)
            if wants_complete and info.get("state") == TaskState.AwaitingCompletion:
                sub_payloads.append(
                    complete_partner(
                        aid,
                        info["task_id"],
                        session_id,
                        sub_query=sub_q,
                    )
                )
            else:
                sub_payloads.append(
                    continue_partner(
                        aid,
                        info["task_id"],
                        user_query,
                        session_id,
                        sub_query=sub_q,
                        awaiting_prompt=info.get("awaiting_prompt"),
                    )
                )
        results = await asyncio.gather(*sub_payloads)
        for r in results:
            partner_results[r["agent_id"]] = r
            logger.info(
                "event=supplement_continue_result session_id=%s agent_id=%s state=%s success=%s",
                session_id,
                r["agent_id"],
                r.get("state"),
                r.get("success"),
            )

    # 根据最新分析结果，决定是否需要启动新的 Partner 任务
    supported_ids = set(agent_dim_map.keys())
    active_states = {
        TaskState.Accepted,
        TaskState.Working,
        TaskState.AwaitingInput,
        TaskState.AwaitingCompletion,
    }
    call_list: list[str] = []
    for aid in required_agents:
        if aid not in supported_ids:
            continue
        current_info = sess.get("partner_tasks", {}).get(aid)
        if current_info and current_info.get("state") in active_states:
            # 仍在执行/等待中，沿用当前任务
            partner_subqueries.setdefault(aid, resolve_sub_query(aid))
            continue
        call_list.append(aid)
        partner_subqueries[aid] = resolve_sub_query(aid)

    if call_list:
        sub_payloads = [
            call_partner(
                aid,
                partner_subqueries[aid],
                session_id,
                sub_query=partner_subqueries[aid],
            )
            for aid in call_list
        ]
        logger.info(
            "event=partner_batch_start session_id=%s agents=%s",
            session_id,
            ",".join(call_list),
        )
        results = await asyncio.gather(*sub_payloads)
        logger.info(
            "event=partner_batch_end session_id=%s agents=%s",
            session_id,
            ",".join(call_list),
        )
        for r in results:
            partner_results[r["agent_id"]] = r

    # 阶段 3：结果整合
    t2 = datetime.now(timezone.utc)
    final_text = llm_integrate(user_query, analysis, partner_results, session_id)
    t3 = datetime.now(timezone.utc)
    logger.info(
        "event=request_complete session_id=%s analysis_ms=%d partner_ms=%d integration_ms=%d total_ms=%d",
        session_id,
        int((t1 - t0).total_seconds() * 1000),
        int((t2 - t1).total_seconds() * 1000),
        int((t3 - t2).total_seconds() * 1000),
        int((t3 - t0).total_seconds() * 1000),
    )

    # 记录助理输出
    append_message(session_id, "assistant", final_text)

    return {
        "session_id": session_id,
        "analysis": analysis,
        "partner_results": partner_results,
        "partner_subqueries": partner_subqueries,
        "final_response": final_text,
        # 额外返回：前端可据此展示和区分
        "pending_questions": get_session(session_id).get("pending_questions", []),
        "partner_tasks": get_session(session_id).get("partner_tasks", {}),
    }


@app.get("/")
def read_root():
    """健康检查 / 简易欢迎信息"""
    return {"message": f"欢迎使用旅游助理 {LEADER_ID}. 调用 /user_api 进行交互"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tour_assistant:app", host="0.0.0.0", port=8019, reload=True)
