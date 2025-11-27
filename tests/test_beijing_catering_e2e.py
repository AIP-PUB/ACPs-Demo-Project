"""
beijing_catering 的端到端 HTTP 集成测试（需要真实服务进程）。

运行示例：
    export END_TO_END=1
    export BEIJING_CATERING_BASE_URL=http://localhost:8013
    export BEIJING_CATERING_AIP_ENDPOINT=/acps-aip-v1/rpc
    venv/bin/python -m pytest -q tests/test_beijing_catering_e2e.py
"""

import os
import uuid
import time
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


BASE_URL = os.getenv("BEIJING_CATERING_BASE_URL", "http://localhost:8013").rstrip("/")
ENDPOINT = os.getenv("BEIJING_CATERING_AIP_ENDPOINT", "/acps-aip-v1/rpc")
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


def _state(resp_json):
    return ((resp_json or {}).get("result") or {}).get("status", {}).get("state")


def _post(payload: dict, timeout=30):
    try:
        r = requests.post(RPC_URL, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Cannot connect to server at {RPC_URL}: {e}")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def _get(task_id: str):
    return _post(_build_rpc_payload("", task_id, command="get"))


def _wait_for_state(
    task_id: str, expect_set: set[str], timeout_s: float = 6.0, interval_s: float = 0.2
):
    """使用 GET 轮询直到状态进入期望集合或超时，返回 (state, body)。"""
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


def test_http_live_start_then_background():
    """验证：Start 后通过 GET 轮询，状态应进入工作相关态或被拒绝。"""
    task_id = f"task-{uuid.uuid4()}"
    body = _post(_build_rpc_payload("请推荐一份北京午餐路线，口味清淡。", task_id))
    assert _state(body) in {"accepted", "awaiting-input", "working", "rejected"}
    st, _ = _wait_for_state(
        task_id, {"awaiting-input", "working", "awaiting-completion", "rejected"}
    )
    assert st in {"awaiting-input", "working", "awaiting-completion", "rejected"}


def test_http_live_reject_out_of_scope():
    """验证：超出北京范围（跨城）的请求，返回 Rejected 或 AwaitingInput（提示澄清）。"""
    task_id = f"task-{uuid.uuid4()}"
    body = _post(_build_rpc_payload("请规划上海餐厅一日三餐（跨城）。", task_id))
    st, _ = _wait_for_state(task_id, {"rejected", "awaiting-input"})
    assert st in {"rejected", "awaiting-input"}


def test_http_live_continue_to_awaiting_completion():
    """验证：AwaitingInput 时补充必要槽位，推进到 AwaitingCompletion 并有产出。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("需要北京不辣的一日三餐建议", task_id))
    st, latest = _wait_for_state(
        task_id, {"awaiting-input", "working", "awaiting-completion"}
    )
    assert st in {"awaiting-input", "working", "awaiting-completion"}
    if st == "awaiting-completion":
        assert len(_products(latest)) >= 1
        return
    cont = _post(
        _build_rpc_payload(
            "区域=东城；预算=中等；餐次=午餐", task_id, command="continue"
        )
    )
    assert _state(cont) in {"awaiting-completion", "awaiting-input", "working"}
    st2, latest = _wait_for_state(task_id, {"awaiting-completion", "awaiting-input"})
    if st2 == "awaiting-input":
        cont2 = _post(
            _build_rpc_payload(
                "口味=清淡；餐次=午餐；人数=2；区域=东城或西城。",
                task_id,
                command="continue",
            )
        )
        assert _state(cont2) in {"awaiting-completion", "working", "awaiting-input"}
        st2, latest = _wait_for_state(task_id, {"awaiting-completion"}, timeout_s=6.0)
    assert st2 == "awaiting-completion"
    assert len(_products(latest)) >= 1


def test_http_live_complete_flow():
    """验证：进入 AwaitingCompletion 后，Complete 使任务进入 Completed。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("北京晚餐推荐，预算中等", task_id))
    st, _ = _wait_for_state(
        task_id, {"awaiting-completion", "awaiting-input", "rejected"}
    )
    if st == "rejected":
        pytest.skip("门卫拒绝了请求，跳过该用例。")
    if st == "awaiting-input":
        _post(
            _build_rpc_payload(
                "口味=清淡；预算=人均80；区域=东城。", task_id, command="continue"
            )
        )
        st, _ = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    comp = _post(_build_rpc_payload("完成", task_id, command="complete"))
    assert _state(comp) == "completed"


def test_http_live_cancel_flow():
    """验证：非终态下 Cancel，将进入 Canceled；考虑并发，先等到进入工作相关态再取消。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("北京午餐推荐。", task_id))
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


def test_http_live_awaiting_completion_timeout_to_completed():
    """验证：AwaitingCompletion 长时间无输入，超时自动转为 Completed。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"awaitingCompletionTimeout": 200}
    _post(
        _build_rpc_payload(
            "北京晚餐推荐，预算中等。",
            task_id,
            command_params=params,
        )
    )
    st, _ = _wait_for_state(
        task_id, {"awaiting-completion", "awaiting-input", "rejected"}
    )
    if st == "rejected":
        pytest.skip("门卫拒绝了请求，跳过该用例。")
    if st == "awaiting-input":
        _post(
            _build_rpc_payload(
                "口味=清淡；预算=人均80；区域=东城。", task_id, command="continue"
            )
        )
        st, _ = _wait_for_state(task_id, {"awaiting-completion"})
    assert st == "awaiting-completion"
    st2, _ = _wait_for_state(task_id, {"completed"}, timeout_s=6.0)
    assert st2 == "completed"


def test_http_live_awaiting_input_timeout_to_canceled():
    """验证：AwaitingInput 长时间无补充信息，超时自动转为 Canceled。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"awaitingInputTimeout": 200}
    _post(
        _build_rpc_payload("需要北京不辣的一日三餐建议", task_id, command_params=params)
    )
    st, _ = _wait_for_state(task_id, {"awaiting-input", "awaiting-completion"})
    if st == "awaiting-completion":
        pytest.skip("未进入 AwaitingInput（直接达成 AwaitingCompletion），跳过该用例。")
    assert st == "awaiting-input"
    st2, _ = _wait_for_state(task_id, {"canceled"}, timeout_s=6.0)
    assert st2 == "canceled"


def test_http_live_products_size_limit_failed():
    """验证：设置很小的 maxProductsBytes，产物落库失败，进入 Failed。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"maxProductsBytes": 10}
    _post(
        _build_rpc_payload(
            "北京晚餐推荐，预算中等。",
            task_id,
            command_params=params,
        )
    )
    st, _ = _wait_for_state(
        task_id, {"failed", "awaiting-input", "rejected"}, timeout_s=6.0
    )
    if st == "rejected":
        pytest.skip("门卫拒绝了请求，跳过 size limit 用例。")
    if st == "awaiting-input":
        _post(
            _build_rpc_payload(
                "口味=清淡；预算=人均80；区域=东城。", task_id, command="continue"
            )
        )
        st, _ = _wait_for_state(task_id, {"failed"}, timeout_s=6.0)
    assert st == "failed"


def test_http_live_complete_ignored_when_not_awaiting_completion():
    """验证：非 AwaitingCompletion 态调用 Complete，应被忽略并保持原状态。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("需要北京不辣的一日三餐建议", task_id))
    st, _ = _wait_for_state(task_id, {"awaiting-input", "awaiting-completion"})
    if st == "awaiting-completion":
        pytest.skip("已进入 AwaitingCompletion，无法验证忽略逻辑，跳过。")
    res = _post(_build_rpc_payload("完成", task_id, command="complete"))
    assert _state(res) == "awaiting-input"


def test_http_live_continue_ignored_when_terminal_rejected():
    """验证：Rejected 终态下 Continue 被忽略，保持 Rejected。"""
    task_id = f"task-{uuid.uuid4()}"
    _post(_build_rpc_payload("请规划上海餐厅一日三餐（跨城）。", task_id))
    st, _ = _wait_for_state(task_id, {"rejected", "awaiting-input"})
    if st == "awaiting-input":
        pytest.skip("分析未直接拒绝而是引导澄清，跳过。")
    assert st == "rejected"
    res = _post(_build_rpc_payload("追加信息：改为北京", task_id, command="continue"))
    assert _state(res) == "rejected"


def test_http_live_response_timeout_prejudge_rejected():
    """验证：设置很短的 responseTimeout，Start 阶段预判超时，任务直接 Rejected。"""
    task_id = f"task-{uuid.uuid4()}"
    params = {"responseTimeout": 100}
    body = _post(
        _build_rpc_payload(
            "北京晚餐推荐，预算中等。",
            task_id,
            command_params=params,
        )
    )
    assert _state(body) == "rejected"
