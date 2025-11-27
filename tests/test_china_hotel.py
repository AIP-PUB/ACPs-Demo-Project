import time
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


def _wait_for_state(client, task_id, expect_set, timeout_s=1.5, interval_s=0.05):
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
def test_start_empty_text_accepted_then_bg_progress(client_hotel, new_task_id):
    """验证：空文本启动任务，可能 Accepted 或 Rejected；若 Accepted，则后台很快推进到 Working 或 AwaitingInput。"""
    payload = _mk(new_task_id, "", command=TaskCommand.Start)
    payload["params"]["message"]["dataItems"] = []
    body = _post(client_hotel, payload)
    assert _state(body) in {TaskState.Accepted.value, TaskState.Rejected.value}

    time.sleep(0.2)
    get_body = _post(client_hotel, _mk(new_task_id, "", command=TaskCommand.Get))
    if _state(body) == TaskState.Accepted.value:
        assert _state(get_body) in {
            TaskState.Working.value,
            TaskState.AwaitingInput.value,
        }


@pytest.mark.usefixtures("patch_openai")
def test_start_reject(client_hotel, new_task_id):
    """验证：触发明确拒绝信号时，任务立即进入 Rejected。"""
    body = _post(client_hotel, _mk(new_task_id, "PLEASE REJECT THIS"))
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
def test_continue_success_to_awaiting_completion(
    client_hotel, new_task_id, monkeypatch
):
    """验证：Continue 后需求充分，进入 AwaitingCompletion，并产生至少一个产品。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {
                    "city": "北京",
                    "dates_or_nights": "2025-10-01~2025-10-04",
                },
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(req: dict, user_text: str):
        return "OK PLAN"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "给我一个酒店建议"))
    _post(client_hotel, _mk(new_task_id, "补充: 北京3晚", command=TaskCommand.Continue))

    s, get_body = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(get_body)) >= 1


@pytest.mark.usefixtures("patch_openai")
def test_cancel_semantics(client_hotel, new_task_id):
    """验证：在非终态下调用 Cancel，返回 Canceled。"""
    _post(client_hotel, _mk(new_task_id, "任意文本"))
    cancel_body = _post(
        client_hotel, _mk(new_task_id, "现在取消", command=TaskCommand.Cancel)
    )
    assert _state(cancel_body) == TaskState.Canceled.value


@pytest.mark.usefixtures("patch_openai")
def test_complete_from_awaiting_completion(client_hotel, new_task_id, monkeypatch):
    """验证：AwaitingCompletion 状态下调用 Complete，进入 Completed。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(_, __):
        return "OK PLAN 2"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "请直接生成方案"))
    s, _ = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value

    complete_body = _post(
        client_hotel, _mk(new_task_id, "完成", command=TaskCommand.Complete)
    )
    assert _state(complete_body) == TaskState.Completed.value


@pytest.mark.usefixtures("patch_openai")
def test_products_size_limit_failure(client_hotel, new_task_id, monkeypatch):
    """验证：maxProductsBytes 过小导致产出写入失败，进入 Failed。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(_, __):
        return "X" * 50

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    payload = _mk(new_task_id, "执行并产出")
    payload["params"]["message"]["commandParams"] = {"maxProductsBytes": 10}
    _post(client_hotel, payload)

    s, _ = _wait_for_state(
        client_hotel, new_task_id, {TaskState.Failed.value}, timeout_s=2.0
    )
    assert s == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
def test_timeout_behavior(client_hotel, new_task_id, monkeypatch):
    """验证：工作阶段执行超时（缩短 DEFAULT_WORK_TIMEOUT_MS）导致 Failed。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(_, __):
        await asyncio.sleep(0.2)
        return "SLOW PLAN"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    monkeypatch.setattr(hotel, "DEFAULT_WORK_TIMEOUT_MS", 10)
    _post(client_hotel, _mk(new_task_id, "执行并超时"))
    time.sleep(0.3)
    get_body = _post(client_hotel, _mk(new_task_id, "", command=TaskCommand.Get))
    assert _state(get_body) == TaskState.Failed.value


@pytest.mark.usefixtures("patch_openai")
def test_start_prejudge_response_timeout_reject(client_hotel, new_task_id):
    """验证：Start 设置很短 responseTimeout 触发预判拒绝 Rejected。"""
    payload = _mk(new_task_id, "任意文本")
    payload["params"]["message"]["commandParams"] = {"responseTimeout": 100}
    body = _post(client_hotel, payload)
    assert _state(body) == TaskState.Rejected.value


@pytest.mark.usefixtures("patch_openai")
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_awaiting_input_timeout_to_canceled(new_task_id, monkeypatch):
    """验证：AwaitingInput 超时（awaitingInputTimeout 很短）自动转为 Canceled。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(user_text: str, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": ["city"],
                "perSkillMissing": {},
            },
        }

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)

    payload = _mk(new_task_id, "触发AwaitingInput")
    payload["params"]["message"]["commandParams"] = {"awaitingInputTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=hotel.app), base_url="http://test"
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
    """验证：AwaitingCompletion 超时（awaitingCompletionTimeout 很短）自动转为 Completed。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(_, __):
        return "OK"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    payload = _mk(new_task_id, "触发AwaitingCompletion")
    payload["params"]["message"]["commandParams"] = {"awaitingCompletionTimeout": 10}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=hotel.app), base_url="http://test"
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
    client_hotel, new_task_id, monkeypatch
):
    """验证：AwaitingCompletion 时 Continue，可回到 Working（或仍为 AwaitingCompletion）。"""
    import china_hotel.china_hotel as hotel

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": ["china_hotel.hotel-recommendation"],
                "global": {},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(_, __):
        return "OK"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "先到AwaitingCompletion"))
    s, _ = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}
    )
    assert s == TaskState.AwaitingCompletion.value

    _post(client_hotel, _mk(new_task_id, "再优化一下", command=TaskCommand.Continue))
    s2, _ = _wait_for_state(
        client_hotel,
        new_task_id,
        {TaskState.Working.value, TaskState.AwaitingCompletion.value},
    )
    assert s2 in {TaskState.Working.value, TaskState.AwaitingCompletion.value}


@pytest.mark.usefixtures("patch_openai")
def test_skill_selection_recommendation_to_awaiting_completion(
    client_hotel, new_task_id, monkeypatch
):
    """验证：分析选择酒店推荐技能（china_hotel.hotel-recommendation），流程可达 AwaitingCompletion 并产出。"""
    import china_hotel.china_hotel as hotel

    expected = ["china_hotel.hotel-recommendation"]

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": expected,
                "global": {"city": "北京"},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(req, __):
        assert req.get("selectedSkills") == expected
        return "OK RECOMMENDATION"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "触发推荐技能"))
    s, body = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(body)) >= 1


@pytest.mark.usefixtures("patch_openai")
def test_skill_selection_price_comparison_to_awaiting_completion(
    client_hotel, new_task_id, monkeypatch
):
    """验证：分析选择比价技能（china_hotel.price-comparison），流程可达 AwaitingCompletion 并产出。"""
    import china_hotel.china_hotel as hotel

    expected = ["china_hotel.price-comparison"]

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": expected,
                "global": {"city": "上海"},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(req, __):
        assert req.get("selectedSkills") == expected
        return "OK PRICE"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "触发比价技能"))
    s, body = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(body)) >= 1


@pytest.mark.usefixtures("patch_openai")
def test_skill_selection_multi_skills_aggregate(client_hotel, new_task_id, monkeypatch):
    """验证：分析选择多个技能（推荐+比价），produce 收到 selectedSkills 并集并产出。"""
    import china_hotel.china_hotel as hotel

    expected = [
        "china_hotel.hotel-recommendation",
        "china_hotel.price-comparison",
    ]

    async def fake_analyze(_, previous_requirements=None):
        return {
            "decision": "accept",
            "requirements": {
                "selectedSkills": expected,
                "global": {"city": "南京"},
                "globalMissing": [],
                "perSkillMissing": {},
            },
        }

    async def fake_produce(req, __):
        assert set(req.get("selectedSkills") or []) == set(expected)
        return "OK MULTI"

    monkeypatch.setattr(hotel, "analyze_requirements", fake_analyze)
    monkeypatch.setattr(hotel, "produce_output", fake_produce)

    _post(client_hotel, _mk(new_task_id, "触发多技能"))
    s, body = _wait_for_state(
        client_hotel, new_task_id, {TaskState.AwaitingCompletion.value}, timeout_s=2.0
    )
    assert s == TaskState.AwaitingCompletion.value
    assert len(_products(body)) >= 1
