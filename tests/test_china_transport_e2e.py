"""
china_transport 的端到端 HTTP 集成测试（需要真实服务进程）。

运行示例：
        export END_TO_END=1
        export CHINA_TRANSPORT_BASE_URL=http://localhost:8016
        export CHINA_TRANSPORT_AIP_ENDPOINT=/acps-aip-v1/rpc
        venv/bin/python -m pytest -q tests/test_china_transport_e2e.py
"""

import os
import time
import uuid
from datetime import datetime, timezone

import pytest
import requests


pytestmark = pytest.mark.skipif(
    os.getenv("END_TO_END") != "1",
    reason="Set END_TO_END=1 to run live HTTP integration tests.",
)


# Shadow the conftest autouse patch_openai with a no-op to ensure real OpenAI is used
@pytest.fixture(autouse=True)
def patch_openai():
    yield


BASE_URL = os.getenv("CHINA_TRANSPORT_BASE_URL", "http://localhost:8016").rstrip("/")
ENDPOINT = os.getenv("CHINA_TRANSPORT_AIP_ENDPOINT", "/acps-aip-v1/rpc")
RPC_URL = f"{BASE_URL}{ENDPOINT}"


def _build_rpc_payload(
    text: str,
    task_id: str,
    command: str = "start",
    command_params: dict | None = None,
):
    sent_at = datetime.now(timezone.utc).isoformat()
    message = {
        "type": "message",
        "id": str(uuid.uuid4()),
        "sentAt": sent_at,
        "senderRole": "leader",
        "senderId": "e2e-http",
        "command": command,
        "dataItems": [{"type": "text", "text": text}],
        "taskId": task_id,
        "sessionId": "session-e2e",
    }
    payload = {
        "jsonrpc": "2.0",
        "method": "rpc",
        "id": str(uuid.uuid4()),
        "params": {"message": message},
    }
    if command_params:
        payload["params"]["message"]["commandParams"] = command_params
    return payload


def _post(payload: dict, timeout=30):
    try:
        r = requests.post(RPC_URL, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Cannot connect to server at {RPC_URL}: {e}")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def _get(task_id: str):
    return _post(_build_rpc_payload("", task_id, command="get"))


def _state(resp_json):
    return ((resp_json or {}).get("result") or {}).get("status", {}).get("state")


def _wait_for_state(
    task_id: str, expect_set: set[str], timeout_s: float = 6.0, interval_s: float = 0.2
):
    deadline = time.time() + timeout_s
    last_state = None
    body = None
    while time.time() < deadline:
        body = _get(task_id)
        last_state = _state(body)
        if last_state in expect_set:
            return last_state, body
        time.sleep(interval_s)
    return last_state, body


def _products(resp_json):
    return ((resp_json or {}).get("result") or {}).get("products") or []


def _first_product_text(resp_json: dict) -> str:
    products = _products(resp_json)
    if not products:
        return ""
    items = (products[0] or {}).get("dataItems") or []
    for it in items:
        if (it or {}).get("type") == "text":
            return it.get("text") or ""
    return ""


def test_http_live_intercity_planning_end_to_end():
    """验证：常规城际交通规划请求，从 Start 到 AwaitingCompletion，且产出至少一个产品。"""
    task_id = f"task-{uuid.uuid4()}"
    text = "上海→北京，2025-10-02~2025-10-05，2人，时间优先，预算¥400-700/人。"
    body = _post(_build_rpc_payload(text, task_id))
    assert _state(body) in {
        "accepted",
        "working",
        "awaiting-input",
        "awaiting-completion",
    }
    st, latest = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    assert len(_products(latest)) >= 1


def test_http_live_explicit_route_optimization():
    """验证：显式指定路线优化技能，应达 AwaitingCompletion 且有产出。"""
    task_id = f"task-{uuid.uuid4()}"
    text = "请执行技能ID：china_transport.route-optimization。城市序列=上海→南京→杭州；日期=2025-10-01~2025-10-06；偏好=效率优先。"
    body = _post(_build_rpc_payload(text, task_id))
    assert _state(body) in {
        "accepted",
        "working",
        "awaiting-input",
        "awaiting-completion",
    }
    st, latest = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    assert len(_products(latest)) >= 1


def test_http_live_out_of_scope_reject():
    """验证：超出域（城市内交通）被直接拒绝。"""
    task_id = f"task-{uuid.uuid4()}"
    text = "请帮我规划上海地铁出行路线。"
    body = _post(_build_rpc_payload(text, task_id))
    assert _state(body) == "rejected"


def test_http_live_complete_flow():
    """验证：进入 AwaitingCompletion 后，Complete 使任务进入 Completed。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("广州→深圳，2025-10-01~2025-10-02。", task_id))
    st, _ = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    complete = _post(_build_rpc_payload("完成", task_id, command="complete"))
    assert _state(complete) == "completed"


def test_http_live_continue_from_awaiting_input_to_awaiting_completion():
    """验证：AwaitingInput 时补充必要槽位，推进到 AwaitingCompletion 并有产出。"""
    task_id = f"task-{uuid.uuid4()}"
    # 使用更明确的“城际交通”措辞以降低误拒绝概率；若仍被拒绝，则跳过（取决于实际模型配置）。
    _post(_build_rpc_payload("需要从上海到北京的城际交通建议", task_id))
    st, _ = _wait_for_state(task_id, {"awaiting-input", "rejected"})
    if st == "rejected":
        pytest.skip("门卫将初始请求判定为拒绝，环境模型配置可能较为严格；跳过该用例。")
    assert st == "awaiting-input"
    cont = _post(
        _build_rpc_payload(
            "出发地=上海；目的地=北京；日期=2025-10-02~2025-10-05；偏好时间优先。",
            task_id,
            command="continue",
        )
    )
    assert _state(cont) in {"working", "awaiting-input", "awaiting-completion"}
    st2, latest = _wait_for_state(task_id, {"awaiting-completion"})
    assert st2 == "awaiting-completion"
    assert len(_products(latest)) >= 1


def test_http_live_continue_on_awaiting_completion_with_feedback():
    """验证：AwaitingCompletion 时 Continue 提供偏好，不减少产出数量。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("广州→深圳，2025-10-01~2025-10-02。", task_id))
    st, before = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    before_products = len(_products(before))
    cont = _post(
        _build_rpc_payload("尽量安排高铁，时间优先。", task_id, command="continue")
    )
    assert _state(cont) in {"working", "awaiting-completion"}
    st2, after = _wait_for_state(task_id, {"awaiting-completion"})
    assert st2 == "awaiting-completion"
    assert len(_products(after)) >= before_products


def test_http_live_cancel_flow():
    """验证：非终态下 Cancel，将进入 Canceled；考虑并发，先等到进入工作相关态再取消。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("上海→南京，2025-10-10~2025-10-12。", task_id))
    st, _ = _wait_for_state(
        task_id, {"working", "awaiting-input", "awaiting-completion"}
    )
    assert st in {"working", "awaiting-input", "awaiting-completion"}
    cancel = _post(_build_rpc_payload("取消", task_id, command="cancel"))
    assert _state(cancel) in {
        "canceled",
        "working",
        "awaiting-input",
        "awaiting-completion",
    }
    st2, _ = _wait_for_state(task_id, {"canceled"}, timeout_s=3.0)
    assert st2 == "canceled"


def test_http_live_response_timeout_prejudge_rejected():
    """验证：设置很短的 responseTimeout，Start 阶段预判超时，任务直接 Rejected。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"responseTimeout": 100}
    body = _post(
        _build_rpc_payload(
            "上海→北京，2025-10-02~2025-10-05。",
            task_id,
            command_params=params,
        )
    )
    assert _state(body) == "rejected"


def test_http_live_awaiting_completion_timeout_to_completed():
    """验证：AwaitingCompletion 长时间无输入，超时自动转为 Completed。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"awaitingCompletionTimeout": 200}
    _post(
        _build_rpc_payload(
            "杭州→宁波，2025-10-10~2025-10-12，时间优先。",
            task_id,
            command_params=params,
        )
    )
    st, _ = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    st2, _ = _wait_for_state(task_id, {"completed"}, timeout_s=6.0)
    assert st2 == "completed"


def test_http_live_awaiting_input_timeout_to_canceled():
    """验证：AwaitingInput 长时间无补充信息，超时自动转为 Canceled。若被门卫拒绝，则跳过（受模型配置影响）。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"awaitingInputTimeout": 200}
    _post(
        _build_rpc_payload(
            "需要从上海到北京的城际交通建议", task_id, command_params=params
        )
    )
    st, _ = _wait_for_state(task_id, {"awaiting-input", "rejected"})
    if st == "rejected":
        pytest.skip("门卫将初始请求判定为拒绝，环境模型配置可能较为严格；跳过该用例。")
    assert st == "awaiting-input"
    st2, _ = _wait_for_state(task_id, {"canceled"}, timeout_s=6.0)
    assert st2 == "canceled"


def test_http_live_products_size_limit_failed():
    """验证：设置很小的 maxProductsBytes，产物落库失败，进入 Failed。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"maxProductsBytes": 10}
    _post(
        _build_rpc_payload(
            "上海→北京，2025-10-02~2025-10-05。",
            task_id,
            command_params=params,
        )
    )
    st, _ = _wait_for_state(task_id, {"failed"}, timeout_s=6.0)
    assert st == "failed"


def test_http_live_complete_ignored_when_not_awaiting_completion():
    """验证：非 AwaitingCompletion 态调用 Complete，应被忽略并保持原状态。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("需要从上海到北京的城际交通建议", task_id))
    st, _ = _wait_for_state(task_id, {"awaiting-input", "rejected"})
    if st == "rejected":
        pytest.skip("门卫将初始请求判定为拒绝；跳过该用例。")
    assert st == "awaiting-input"
    res = _post(_build_rpc_payload("完成", task_id, command="complete"))
    assert _state(res) == "awaiting-input"


def test_http_live_continue_ignored_when_terminal_rejected():
    """验证：Rejected 终态下 Continue 被忽略，保持 Rejected。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("请帮我规划上海地铁出行路线。", task_id))
    st, _ = _wait_for_state(task_id, {"rejected"})
    assert st == "rejected"
    res = _post(_build_rpc_payload("追加信息：改为城际", task_id, command="continue"))
    assert _state(res) == "rejected"


def test_http_live_cancel_on_completed_is_idempotent():
    """验证：Completed 态再次 Cancel 是幂等操作（无效动作），保持 Completed。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"awaitingCompletionTimeout": 200}
    _post(
        _build_rpc_payload(
            "上海→南京，2025-10-10~2025-10-12。",
            task_id,
            command_params=params,
        )
    )
    st_ac, _ = _wait_for_state(task_id, {"awaiting-completion"}, timeout_s=6.0)
    assert st_ac == "awaiting-completion"
    st, _ = _wait_for_state(task_id, {"completed"}, timeout_s=3.0)
    assert st == "completed"
    res = _post(_build_rpc_payload("取消", task_id, command="cancel"))
    assert _state(res) == "completed"


def test_http_live_skill_selection_via_output_headers_planning():
    """验证：当触发城际交通规划技能，产出正文应包含“【城际交通规划】”分段标题。"""
    task_id = f"task-{uuid.uuid4()}"
    body = _post(
        _build_rpc_payload(
            "广州→深圳，2025-10-01~2025-10-02。",
            task_id,
        )
    )
    assert _state(body) in {
        "accepted",
        "working",
        "awaiting-input",
        "awaiting-completion",
    }
    st, latest = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    text = _first_product_text(latest)
    assert "【城际交通规划】" in text


def test_http_live_skill_selection_via_output_headers_route_optimization():
    """验证：显式请求路线优化技能时，产出正文应包含“【路线优化服务】”分段标题。"""
    task_id = f"task-{uuid.uuid4()}"
    text = "请执行技能ID：china_transport.route-optimization。城市序列=上海→南京→杭州；日期=2025-10-01~2025-10-06。"
    body = _post(_build_rpc_payload(text, task_id))
    assert _state(body) in {
        "accepted",
        "working",
        "awaiting-input",
        "awaiting-completion",
    }
    st, latest = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    out = _first_product_text(latest)
    assert "【路线优化服务】" in out
