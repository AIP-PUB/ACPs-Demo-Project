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
    body = None
    while time.time() < deadline:
        body = _post(client, _mk(task_id, "", command=TaskCommand.Get))
        last = _state(body)
        if last in expect_set:
            return last, body
        time.sleep(interval_s)
    return last, body


@pytest.mark.usefixtures("patch_openai")
def test_start_empty_text_awaiting_input(client_rural, new_task_id):
    """空输入启动：通过门卫后进入 Accepted 或直接 Rejected；随后后台分析进入 Working 或 AwaitingInput。"""
    payload = _mk(new_task_id, "", command=TaskCommand.Start)
    payload["params"]["message"]["dataItems"] = []
    body = _post(client_rural, payload)
    assert _state(body) in {TaskState.Accepted.value, TaskState.Rejected.value}
    if _state(body) == TaskState.Accepted.value:
        s, _ = _wait_for_state(
            client_rural,
            new_task_id,
            {TaskState.Working.value, TaskState.AwaitingInput.value},
            timeout_s=2.0,
        )
        assert s in {TaskState.Working.value, TaskState.AwaitingInput.value}


@pytest.mark.usefixtures("patch_openai")
def test_start_reject(client_rural, new_task_id):
    """启动即被门卫拒绝：返回 Rejected。"""
    body = _post(client_rural, _mk(new_task_id, "PLEASE REJECT THIS"))
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
def test_start_normal_background_to_awaiting_input(client_rural, new_task_id):
    """正常启动：通过门卫后后台处理，状态进入 Working 或 AwaitingInput。"""
    start_body = _post(client_rural, _mk(new_task_id, "任意文本"))
    assert _state(start_body) == TaskState.Accepted.value
    s, _ = _wait_for_state(
        client_rural,
        new_task_id,
        {TaskState.Working.value, TaskState.AwaitingInput.value},
        timeout_s=2.0,
    )
    assert s in {TaskState.Working.value, TaskState.AwaitingInput.value}


@pytest.mark.usefixtures("patch_openai")
def test_continue_reject_from_awaiting_input(client_rural, new_task_id):
    """在 AwaitingInput 态补充信息被分析拒绝：保持 AwaitingInput。"""
    _post(client_rural, _mk(new_task_id, "任意文本"))
    time.sleep(0.2)
    cont_body = _post(
        client_rural,
        _mk(new_task_id, "REJECT in supplement", command=TaskCommand.Continue),
    )
    assert _state(cont_body) == TaskState.AwaitingInput.value


@pytest.mark.usefixtures("patch_openai")
def test_continue_success_to_awaiting_completion(
    client_rural, new_task_id, monkeypatch
):
    """AwaitingInput → Continue 成功：进入 AwaitingCompletion，且产物非空。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "scope": "rural-only",
                "theme": "自然摄影",
                "days": 1,
                "preferences": [],
                "budgetLevel": "medium",
                "mustSee": ["慕田峪长城"],
                "avoid": [],
                "missingFields": [],
            },
        }

    async def fake_produce(requirements: dict) -> str:
        return "OK RURAL PLAN"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    _post(client_rural, _mk(new_task_id, "给我一个郊区一日游"))
    _post(
        client_rural,
        _mk(new_task_id, "补充: 自然摄影主题", command=TaskCommand.Continue),
    )
    s, get_body = _wait_for_state(
        client_rural, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(get_body)) >= 1


@pytest.mark.usefixtures("patch_openai")
def test_cancel_semantics(client_rural, new_task_id):
    """在非终态调用 Cancel：进入 Canceled。"""
    _post(client_rural, _mk(new_task_id, "任意文本"))
    cancel_body = _post(
        client_rural, _mk(new_task_id, "现在取消", command=TaskCommand.Cancel)
    )
    assert _state(cancel_body) == TaskState.Canceled.value


@pytest.mark.usefixtures("patch_openai")
def test_complete_from_awaiting_completion(client_rural, new_task_id, monkeypatch):
    """AwaitingCompletion → Complete：进入 Completed。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK PLAN 2"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    _post(client_rural, _mk(new_task_id, "请直接生成路线"))
    time.sleep(0.2)
    get_body = _post(client_rural, _mk(new_task_id, "", command=TaskCommand.Get))
    assert _state(get_body) == TaskState.AwaitingCompletion.value
    complete_body = _post(
        client_rural, _mk(new_task_id, "完成", command=TaskCommand.Complete)
    )
    assert _state(complete_body) == TaskState.Completed.value


@pytest.mark.usefixtures("patch_openai")
def test_products_size_limit_failure(client_rural, new_task_id, monkeypatch):
    """产物大小超限：写入失败进入 Failed。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "X" * 50

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    payload = _mk(new_task_id, "执行并产出")
    payload["params"]["message"]["commandParams"] = {"maxProductsBytes": 10}
    _post(client_rural, payload)
    s, _ = _wait_for_state(
        client_rural, new_task_id, {TaskState.Failed.value}, timeout_s=2.0
    )
    assert s == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
def test_timeout_behavior(client_rural, new_task_id, monkeypatch):
    """工作阶段内部超时：进入 Failed。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        await asyncio.sleep(0.2)
        return "SLOW PLAN"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    monkeypatch.setattr(rural, "DEFAULT_WORK_TIMEOUT_MS", 10)
    _post(client_rural, _mk(new_task_id, "执行并超时"))
    time.sleep(0.3)
    get_body = _post(client_rural, _mk(new_task_id, "", command=TaskCommand.Get))
    assert _state(get_body) == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_awaiting_input_timeout_to_canceled(new_task_id, monkeypatch):
    """AwaitingInput 长时间无补充：超时自动进入 Canceled。"""
    import beijing_rural.beijing_rural as rural

    # Force analysis to require missing fields so it enters AwaitingInput
    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": ["theme"]},
        }

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)

    from tests.conftest import build_rpc_payload

    payload = build_rpc_payload(
        "触发AwaitingInput", new_task_id, command=TaskCommand.Start
    )
    payload["params"]["message"]["commandParams"] = {"awaitingInputTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rural.app), base_url="http://test"
    ) as client:
        resp = await client.post("/acps-aip-v1/rpc", json=payload)
        assert resp.status_code == 200
        await asyncio.sleep(0.05)
        get_payload = build_rpc_payload("", new_task_id, command=TaskCommand.Get)
        get_resp = await client.post("/acps-aip-v1/rpc", json=get_payload)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert _state(body) == TaskState.Canceled.value


@pytest.mark.usefixtures("patch_openai")
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_awaiting_completion_timeout_to_completed(new_task_id, monkeypatch):
    """AwaitingCompletion 长时间无输入：超时自动进入 Completed。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    from tests.conftest import build_rpc_payload

    payload = build_rpc_payload(
        "触发AwaitingCompletion", new_task_id, command=TaskCommand.Start
    )
    payload["params"]["message"]["commandParams"] = {"awaitingCompletionTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rural.app), base_url="http://test"
    ) as client:
        resp = await client.post("/acps-aip-v1/rpc", json=payload)
        assert resp.status_code == 200
        await asyncio.sleep(0.05)
        get_payload = build_rpc_payload("", new_task_id, command=TaskCommand.Get)
        get_resp = await client.post("/acps-aip-v1/rpc", json=get_payload)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert _state(body) == TaskState.Completed.value


@pytest.mark.usefixtures("patch_openai")
def test_awaiting_completion_continue_back_to_working(
    client_rural, new_task_id, monkeypatch
):
    """AwaitingCompletion 连续补充：可回到 Working。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    _post(client_rural, _mk(new_task_id, "先到AwaitingCompletion"))
    s, _ = _wait_for_state(
        client_rural, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value
    _post(client_rural, _mk(new_task_id, "再优化一下", command=TaskCommand.Continue))
    s2, _ = _wait_for_state(
        client_rural,
        new_task_id,
        {TaskState.Working.value, TaskState.AwaitingCompletion.value},
    )
    assert s2 in {TaskState.Working.value, TaskState.AwaitingCompletion.value}


@pytest.mark.usefixtures("patch_openai")
def test_start_prejudge_response_timeout_reject(client_rural, new_task_id):
    """Start 阶段预判响应超时：直接 Rejected。"""
    payload = _mk(new_task_id, "任意文本")
    payload["params"]["message"]["commandParams"] = {"responseTimeout": 100}
    body = _post(client_rural, payload)
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
def test_complete_ignored_when_not_awaiting_completion(client_rural, new_task_id):
    """非 AwaitingCompletion 调用 Complete：应被忽略并保持原状态。"""
    _post(client_rural, _mk(new_task_id, "需要北京郊区一日游建议"))
    s, _ = _wait_for_state(
        client_rural,
        new_task_id,
        {TaskState.AwaitingInput.value, TaskState.AwaitingCompletion.value},
        timeout_s=2.0,
    )
    if s == TaskState.AwaitingCompletion.value:
        pytest.skip("已在 AwaitingCompletion，无法验证忽略逻辑。")
    res = _post(client_rural, _mk(new_task_id, "完成", command=TaskCommand.Complete))
    assert _state(res) == TaskState.AwaitingInput.value


@pytest.mark.usefixtures("patch_openai")
def test_continue_ignored_when_terminal_rejected(client_rural, new_task_id):
    """终态 Rejected 下 Continue：保持 Rejected。"""
    body = _post(client_rural, _mk(new_task_id, "PLEASE REJECT THIS"))
    assert _state(body) == TaskState.Rejected.value
    res = _post(
        client_rural, _mk(new_task_id, "仍要继续", command=TaskCommand.Continue)
    )
    assert _state(res) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
def test_cancel_on_completed_is_idempotent(client_rural, new_task_id, monkeypatch):
    """对 Completed 再次 Cancel：应保持 Completed（幂等）。"""
    import beijing_rural.beijing_rural as rural

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {"scope": "rural-only", "missingFields": []},
        }

    async def fake_produce(_):
        return "OK PLAN"

    monkeypatch.setattr(rural, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(rural, "produce_plan", fake_produce)

    _post(client_rural, _mk(new_task_id, "请直接生成路线"))
    s, _ = _wait_for_state(
        client_rural, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value
    _post(client_rural, _mk(new_task_id, "完成", command=TaskCommand.Complete))
    s2, _ = _wait_for_state(client_rural, new_task_id, {TaskState.Completed.value})
    assert s2 == TaskState.Completed.value
    res = _post(client_rural, _mk(new_task_id, "取消", command=TaskCommand.Cancel))
    assert _state(res) == TaskState.Completed.value
