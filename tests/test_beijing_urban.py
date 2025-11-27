import time
import uuid
import asyncio
import pytest
import httpx

from acps_aip.aip_base_model import TaskState, TaskCommand


def _post(client, payload):
    resp = client.post("/acps-aip-v1/rpc", json=payload)
    assert resp.status_code == 200
    return resp.json()


def _mk(task_id, text, command=TaskCommand.Start):
    from tests.conftest import build_rpc_payload

    return build_rpc_payload(text, task_id, command=command)


def _state(body):
    return body.get("result", {}).get("status", {}).get("state")


def _products(body):
    return body.get("result", {}).get("products") or []


def _result_task(body):
    return body.get("result", {})


def _wait_for_state(client, task_id, expect_set, timeout_s=1.5, interval_s=0.05):
    """Poll GET until state is in expect_set or timeout."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        body = _post(client, _mk(task_id, "", command=TaskCommand.Get))
        last = _state(body)
        if last in expect_set:
            return last, body
        time.sleep(interval_s)
    return last, body


@pytest.mark.usefixtures("patch_openai")
def test_start_empty_text_awaiting_input(client_urban, new_task_id):
    """验证：空文本启动任务，可能 Accepted 或 Rejected；若 Accepted，则后台很快推进到 Working 或 AwaitingInput。"""
    payload = _mk(new_task_id, "", command=TaskCommand.Start)
    payload["params"]["message"]["dataItems"] = []
    body = _post(client_urban, payload)
    assert _state(body) in {TaskState.Accepted.value, TaskState.Rejected.value}
    time.sleep(0.2)
    get_body = _post(client_urban, _mk(new_task_id, "", command=TaskCommand.Get))
    if _state(body) == TaskState.Accepted.value:
        assert _state(get_body) in {
            TaskState.Working.value,
            TaskState.AwaitingInput.value,
        }


@pytest.mark.usefixtures("patch_openai")
def test_start_reject(client_urban, new_task_id):
    """验证：触发明确拒绝信号时，任务立即进入 Rejected。"""
    body = _post(client_urban, _mk(new_task_id, "PLEASE REJECT THIS"))
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
def test_start_normal_background_to_awaiting_input(client_urban, new_task_id):
    """验证：普通文本启动后，初态 Accepted，后台推进至 Working 或 AwaitingInput。"""
    start_body = _post(client_urban, _mk(new_task_id, "任意文本"))
    assert _state(start_body) == TaskState.Accepted.value
    s, _ = _wait_for_state(
        client_urban,
        new_task_id,
        {TaskState.Working.value, TaskState.AwaitingInput.value},
        timeout_s=2.0,
    )
    assert s in {TaskState.Working.value, TaskState.AwaitingInput.value}


@pytest.mark.usefixtures("patch_openai")
def test_continue_reject_from_awaiting_input(client_urban, new_task_id):
    """验证：AwaitingInput 时 Continue 若触发拒绝提示，仍保持 AwaitingInput 并给出引导。"""
    _post(client_urban, _mk(new_task_id, "任意文本"))
    time.sleep(0.2)
    cont_body = _post(
        client_urban,
        _mk(new_task_id, "REJECT in supplement", command=TaskCommand.Continue),
    )
    assert _state(cont_body) == TaskState.AwaitingInput.value


@pytest.mark.usefixtures("patch_openai")
def test_continue_success_to_awaiting_completion(
    client_urban, new_task_id, monkeypatch
):
    """验证：Continue 后需求充分，进入 AwaitingCompletion，并产生至少一个产品。"""
    import beijing_urban.beijing_urban as urban

    # Patch analysis to produce complete requirements (no missingFields)
    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "scope": "city-core-only",
                "theme": "文化深度",
                "days": 1,
                "preferences": [],
                "budgetLevel": "medium",
                "mustSee": ["故宫"],
                "avoid": [],
                "missingFields": [],
            },
        }

    async def fake_produce(requirements: dict) -> str:
        return "OK PLAN"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    # Start -> immediate Accepted, let bg move to Working then see no missing -> AwaitingCompletion
    _post(client_urban, _mk(new_task_id, "给我一个城区一日游"))
    # Continue with any supplement to kick re-run (or first run if waiting input)
    _post(
        client_urban, _mk(new_task_id, "补充: 文化主题", command=TaskCommand.Continue)
    )

    # Wait until AwaitingCompletion
    s, get_body = _wait_for_state(
        client_urban, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(get_body)) >= 1


@pytest.mark.usefixtures("patch_openai")
def test_cancel_semantics(client_urban, new_task_id):
    """验证：在非终态下调用 Cancel，返回 Canceled。"""
    _post(client_urban, _mk(new_task_id, "任意文本"))
    cancel_body = _post(
        client_urban, _mk(new_task_id, "现在取消", command=TaskCommand.Cancel)
    )
    assert _state(cancel_body) == TaskState.Canceled.value


@pytest.mark.usefixtures("patch_openai")
def test_complete_from_awaiting_completion(client_urban, new_task_id, monkeypatch):
    """验证：AwaitingCompletion 状态下调用 Complete，进入 Completed。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "scope": "city-core-only",
                "missingFields": [],
            },
        }

    async def fake_produce(_):
        return "OK PLAN 2"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    _post(client_urban, _mk(new_task_id, "请直接生成路线"))
    s, get_body = _wait_for_state(
        client_urban, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value

    complete_body = _post(
        client_urban, _mk(new_task_id, "完成", command=TaskCommand.Complete)
    )
    assert _state(complete_body) == TaskState.Completed.value


@pytest.mark.usefixtures("patch_openai")
def test_products_size_limit_failure(client_urban, new_task_id, monkeypatch):
    """验证：maxProductsBytes 过小导致产出写入失败，进入 Failed。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "X" * 50

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    # Start with very small maxProductsBytes
    payload = _mk(new_task_id, "执行并产出")
    payload["params"]["message"]["commandParams"] = {"maxProductsBytes": 10}
    _post(client_urban, payload)

    s, get_body = _wait_for_state(
        client_urban, new_task_id, {TaskState.Failed.value}, timeout_s=2.0
    )
    assert s == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
def test_timeout_behavior(client_urban, new_task_id, monkeypatch):
    """验证：工作阶段执行超时（缩短 DEFAULT_WORK_TIMEOUT_MS）导致 Failed。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": []},
        }

    async def fake_produce(_):
        # sleep long enough to exceed tiny timeout
        await asyncio.sleep(0.2)
        return "SLOW PLAN"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    # Tighten the internal per-cycle work timeout to force a timeout
    monkeypatch.setattr(urban, "DEFAULT_WORK_TIMEOUT_MS", 10)

    payload = _mk(new_task_id, "执行并超时")
    _post(client_urban, payload)

    time.sleep(0.3)
    get_body = _post(client_urban, _mk(new_task_id, "", command=TaskCommand.Get))
    assert _state(get_body) == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
def test_start_prejudge_response_timeout_reject(client_urban, new_task_id):
    """验证：Start 设置很短 responseTimeout 触发预判拒绝 Rejected。"""
    payload = _mk(new_task_id, "任意文本")
    payload["params"]["message"]["commandParams"] = {"responseTimeout": 100}
    body = _post(client_urban, payload)
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_awaiting_input_timeout_to_canceled(new_task_id, monkeypatch):
    """验证：AwaitingInput 超时（awaitingInputTimeout 很短）应自动转为 Canceled。"""
    import beijing_urban.beijing_urban as urban

    # Force analysis to require missing fields so it enters AwaitingInput
    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": ["theme"]},
        }

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)

    payload = _mk(new_task_id, "触发AwaitingInput")
    payload["params"]["message"]["commandParams"] = {"awaitingInputTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=urban.app), base_url="http://test"
    ) as client:
        resp = await client.post("/acps-aip-v1/rpc", json=payload)
        assert resp.status_code == 200
        await asyncio.sleep(0.05)
        get_payload = _mk(new_task_id, "", command=TaskCommand.Get)
        get_resp = await client.post("/acps-aip-v1/rpc", json=get_payload)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert _state(body) == TaskState.Canceled.value


@pytest.mark.usefixtures("patch_openai")
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_awaiting_completion_timeout_to_completed(new_task_id, monkeypatch):
    """验证：AwaitingCompletion 超时（awaitingCompletionTimeout 很短）应自动转为 Completed。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    payload = _mk(new_task_id, "触发AwaitingCompletion")
    payload["params"]["message"]["commandParams"] = {"awaitingCompletionTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=urban.app), base_url="http://test"
    ) as client:
        resp = await client.post("/acps-aip-v1/rpc", json=payload)
        assert resp.status_code == 200
        await asyncio.sleep(0.05)
        get_payload = _mk(new_task_id, "", command=TaskCommand.Get)
        get_resp = await client.post("/acps-aip-v1/rpc", json=get_payload)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert _state(body) == TaskState.Completed.value


@pytest.mark.usefixtures("patch_openai")
def test_awaiting_completion_continue_back_to_working(
    client_urban, new_task_id, monkeypatch
):
    """验证：AwaitingCompletion 时 Continue，可回到 Working（或仍为 AwaitingCompletion）。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    _post(client_urban, _mk(new_task_id, "先到AwaitingCompletion"))
    # Wait to reach AwaitingCompletion
    s, _ = _wait_for_state(
        client_urban, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value

    # Continue with more input triggers Working again
    _post(client_urban, _mk(new_task_id, "再优化一下", command=TaskCommand.Continue))
    s2, _ = _wait_for_state(
        client_urban,
        new_task_id,
        {TaskState.Working.value, TaskState.AwaitingCompletion.value},
    )
    assert s2 in {TaskState.Working.value, TaskState.AwaitingCompletion.value}


@pytest.mark.usefixtures("patch_openai")
def test_complete_ignored_when_not_awaiting_completion(
    client_urban, new_task_id, monkeypatch
):
    """验证：非 AwaitingCompletion 状态下调用 Complete，应被忽略并保持原状态（例如 AwaitingInput）。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": ["theme"]},
        }

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)

    _post(client_urban, _mk(new_task_id, "需要城区一日游"))
    s, _ = _wait_for_state(
        client_urban, new_task_id, {TaskState.AwaitingInput.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingInput.value

    res = _post(client_urban, _mk(new_task_id, "完成", command=TaskCommand.Complete))
    assert _state(res) == TaskState.AwaitingInput.value


@pytest.mark.usefixtures("patch_openai")
def test_cancel_on_completed_is_idempotent(client_urban, new_task_id, monkeypatch):
    """验证：Completed 状态再次 Cancel 为幂等操作，状态保持 Completed。"""
    import beijing_urban.beijing_urban as urban

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "city-core-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK"

    monkeypatch.setattr(urban, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(urban, "produce_plan", fake_produce)

    _post(client_urban, _mk(new_task_id, "请直接生成"))
    s, _ = _wait_for_state(
        client_urban, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value

    res = _post(client_urban, _mk(new_task_id, "完成", command=TaskCommand.Complete))
    assert _state(res) == TaskState.Completed.value

    res2 = _post(client_urban, _mk(new_task_id, "取消", command=TaskCommand.Cancel))
    assert _state(res2) == TaskState.Completed.value
