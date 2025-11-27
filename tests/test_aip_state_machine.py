import uuid
import pytest
from acps_aip.aip_base_model import TaskCommand
from conftest import build_rpc_payload


def _post_rpc(client, payload):
    res = client.post("/acps-aip-v1/rpc", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert "result" in body, body
    return body["result"]


def test_start_is_idempotent(client_urban, new_task_id):
    task_id = new_task_id
    # First start with text
    payload1 = build_rpc_payload("天安门一日游", task_id, TaskCommand.Start)
    result1 = _post_rpc(client_urban, payload1)
    assert result1["id"] == task_id
    first_state = result1["status"]["state"]

    # Start again must be idempotent and return the current task snapshot (state may have advanced asynchronously)
    payload2 = build_rpc_payload("", task_id, TaskCommand.Start)
    result2 = _post_rpc(client_urban, payload2)
    assert result2["id"] == task_id
    assert result2["status"]["state"] in (
        first_state,
        "working",
        "awaiting-input",
        "awaiting-completion",
    )


def test_continue_requires_text_and_state(client_urban, new_task_id):
    task_id = new_task_id
    # Start without text -> agent should place task into awaiting-input
    payload_start = build_rpc_payload("", task_id, TaskCommand.Start)
    result_start = _post_rpc(client_urban, payload_start)
    assert result_start["status"]["state"] in ("accepted", "awaiting-input")

    # Continue without text -> ignored by RPC server; state unchanged (remains not progressed)
    payload_cont_empty = build_rpc_payload("", task_id, TaskCommand.Continue)
    result_cont_empty = _post_rpc(client_urban, payload_cont_empty)
    assert result_cont_empty["status"]["state"] in ("accepted", "awaiting-input")

    # Continue with text -> agent processes, may reach awaiting-completion or still require input depending on analysis
    payload_cont_text = build_rpc_payload(
        "适合亲子的城区一日游", task_id, TaskCommand.Continue
    )
    result_cont_text = _post_rpc(client_urban, payload_cont_text)
    assert result_cont_text["status"]["state"] in (
        "awaiting-completion",
        "awaiting-input",
        "working",
    )


def test_complete_only_from_awaiting_completion(client_urban, new_task_id):
    task_id = new_task_id
    # Start without text to be in awaiting-input
    payload_start = build_rpc_payload("", task_id, TaskCommand.Start)
    result_start = _post_rpc(client_urban, payload_start)
    assert result_start["status"]["state"] in ("accepted", "awaiting-input")

    # Attempt to complete prematurely -> ignored
    payload_complete_ignored = build_rpc_payload("", task_id, TaskCommand.Complete)
    result_complete_ignored = _post_rpc(client_urban, payload_complete_ignored)
    assert result_complete_ignored["status"]["state"] in ("accepted", "awaiting-input")

    # Now provide text to progress the task
    payload_cont_text = build_rpc_payload("文化深度路线", task_id, TaskCommand.Continue)
    result_cont = _post_rpc(client_urban, payload_cont_text)
    assert result_cont["status"]["state"] in (
        "awaiting-completion",
        "awaiting-input",
        "working",
    )

    # Complete from awaiting-completion -> should become completed
    payload_complete = build_rpc_payload("", task_id, TaskCommand.Complete)
    result_complete = _post_rpc(client_urban, payload_complete)
    # If not yet in awaiting-completion, complete is a no-op; otherwise transitions to completed
    assert result_complete["status"]["state"] in (
        "completed",
        result_cont["status"]["state"],
    )


def test_cancel_is_idempotent_and_terminal_safe(client_urban, new_task_id):
    task_id = new_task_id
    # Start with text to establish a task
    payload_start = build_rpc_payload("休闲路线", task_id, TaskCommand.Start)
    _post_rpc(client_urban, payload_start)

    # Cancel once -> canceled
    payload_cancel = build_rpc_payload("", task_id, TaskCommand.Cancel)
    result_cancel1 = _post_rpc(client_urban, payload_cancel)
    assert result_cancel1["status"]["state"] == "canceled"

    # Cancel again -> remains canceled (idempotent)
    result_cancel2 = _post_rpc(client_urban, payload_cancel)
    assert result_cancel2["status"]["state"] == "canceled"
