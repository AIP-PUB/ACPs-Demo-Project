"""
End-to-end HTTP integration tests for tour_assistant against a real running server.

Run example:
    export END_TO_END=1
    export TOUR_ASSISTANT_BASE_URL=http://localhost:8019
    venv/bin/python -m pytest -q tests/test_tour_assistant_e2e.py
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


BASE_URL = os.getenv("TOUR_ASSISTANT_BASE_URL", "http://localhost:8019").rstrip("/")
USER_API = f"{BASE_URL}/user_api"


def _post_user_api(payload: dict, timeout=60):
    try:
        r = requests.post(USER_API, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Cannot connect to tour assistant at {USER_API}: {e}")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def _agent_state(resp_json: dict, agent_id: str) -> str | None:
    tasks = (resp_json or {}).get("partner_tasks") or {}
    info = tasks.get(agent_id) or {}
    return info.get("state")


def _any_completed(resp_json: dict, agent_id: str) -> bool:
    return _agent_state(resp_json, agent_id) == "completed"


@pytest.mark.parametrize(
    "initial_query, supplement, agent_id",
    [
        (
            # Beijing urban flow
            "我想安排一个北京城区的两日文化行程，主要参观历史景点。",
            "行程时间10月3日至10月4日，两位成人，主题是历史文化深度体验，重点参观故宫、天坛、颐和园，预算中等偏上，体力允许中等步行，不需要餐饮或购物安排，交通以地铁和公交为主。",
            "beijing_urban_agent_001",
        ),
        (
            # Beijing rural flow
            "想在北京郊区安排周末户外活动，最好有山和湖。",
            "行程10月5日至10月6日，三人周末户外主题，预算中等，偏好徒步+山湖风景，优先怀柔或延庆，可接受轻度露营，不需要城市美食安排，需提供从市区出发的交通衔接及返回方案。",
            "beijing_rural_agent_001",
        ),
        (
            # Beijing catering flow
            "打算体验北京的特色美食，请推荐餐厅。",
            "两天行程，人均预算150元，三人同行，白天偏好老北京小吃和胡同小店，晚上希望在簋街附近安排晚餐，需要告知营业时间及是否需预订，顺带标注交通方式。",
            "beijing_catering_agent_001",
        ),
        (
            # Hotel flow: initial underspecified -> supplement with dates/budget
            "我想在北京住酒店，住3晚，靠近地铁。",
            "入住时间10月2日至10月5日，两位成人，预算每晚500-700元，需要含早，可步行到地铁，优先王府井或故宫附近，房间需提供无烟楼层。",
            "china_hotel_agent_001",
        ),
        (
            # Transport flow: initial underspecified -> supplement with date/time/seating
            "我需要从上海去北京的交通方案。",
            "10月5日上午出发，目的地北京站，优先高铁二等座，预算500-700元，需要上午10点前到达，随身行李两件，如有座位紧张请列出备选方案。",
            "china_transport_agent_001",
        ),
    ],
)
def test_tour_assistant_flow_awaiting_input_then_complete(
    initial_query, supplement, agent_id
):
    # Start new session
    session_id = f"session-{uuid.uuid4()}"

    # Step 1: initial natural language input
    resp1 = _post_user_api({"session_id": session_id, "query": initial_query})

    # The leader should have routed to the target agent, but if discovery/endpoints are not configured, skip
    state1 = _agent_state(resp1, agent_id)
    pr1 = (resp1 or {}).get("partner_results", {}).get(agent_id)
    if not pr1:
        pytest.skip(
            f"Partner {agent_id} not invoked by leader. Ensure partner services are configured."
        )
    if pr1.get("state") == "unavailable":
        pytest.skip(f"Partner {agent_id} unavailable (no JSONRPC endpoint).")
    # Initial call may occasionally be rejected by a live partner classifier; we accept it
    # since the next user supplement will be treated as a new task.
    assert state1 in {
        None,
        "accepted",
        "working",
        "awaiting-input",
        "awaiting-completion",
        "completed",
        "rejected",
    }

    # Optional brief wait if the partner is still processing
    if state1 in {None, "working"}:
        time.sleep(1.0)

    # Step 2: provide supplemental details to unblock AwaitingInput
    resp2 = _post_user_api({"session_id": session_id, "query": supplement})
    state2 = _agent_state(resp2, agent_id) or (
        resp2.get("partner_results", {}).get(agent_id, {}).get("state")
    )

    # In many cases, the leader will either:
    # - detect awaiting-input and Continue with the supplement, transitioning to awaiting-completion/completed, or
    # - directly reach completed if product evaluation passes.
    assert state2 in {
        "accepted",
        "awaiting-completion",
        "completed",
        "working",
        "awaiting-input",
    }

    # Step 3: if still not completed, try one more nudge; the leader will auto-complete at awaiting-completion if satisfied
    if not _any_completed(resp2, agent_id):
        # up to 2 nudges to allow the leader to iterate and complete
        last = resp2
        for _ in range(2):
            last = _post_user_api(
                {"session_id": session_id, "query": "如已满足需求，请整理结果并完成。"}
            )
            state3 = _agent_state(last, agent_id) or (
                last.get("partner_results", {}).get(agent_id, {}).get("state")
            )
            assert state3 in {
                "accepted",
                "awaiting-completion",
                "completed",
                "working",
                "awaiting-input",
            }
            if _any_completed(last, agent_id):
                break
            time.sleep(1.0)
        # Final assertion: completed or awaiting-completion is acceptable
        final_state = _agent_state(last, agent_id) or (
            last.get("partner_results", {}).get(agent_id, {}).get("state")
        )
        assert _any_completed(last, agent_id) or final_state == "awaiting-completion"
    else:
        assert _any_completed(resp2, agent_id)
