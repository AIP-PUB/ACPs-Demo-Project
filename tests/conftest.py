import os
import sys
import uuid
import pathlib
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Ensure project root (parent of this tests directory) is on sys.path in case pytest
# changes the working directory during collection.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI app objects directly from each agent module
import beijing_urban.beijing_urban as beijing_urban
import beijing_rural.beijing_rural as beijing_rural
import beijing_catering.beijing_catering as beijing_catering
import china_hotel.china_hotel as china_hotel
import china_transport.china_transport as china_transport

from acps_aip.aip_base_model import Message, TextDataItem, TaskCommand

# Disable actual OpenAI network calls by monkeypatching the client method.
# We will craft a deterministic fake response JSON depending on the agent under test and input text.


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)


class DummyCompletion:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    """Auto patch openai.chat.completions.create so tests run offline.

    We produce different JSON to trigger accept / reject / failure branches.
    Input cues:
      - contains keyword REJECT -> decision=reject with reason.
      - contains keyword BADJSON -> return malformed JSON to test error handling.
      - otherwise decision=accept with simple plan.
    """
    import openai

    def fake_create(messages, model, **kwargs):  # noqa: D401
        user_content = messages[-1]["content"].lower()
        if "badjson" in user_content:
            return DummyCompletion("not a json string")
        if "reject" in user_content:
            return DummyCompletion('{"decision":"reject","reason":"测试拒绝分支"}')
        # accept
        return DummyCompletion(
            '{"decision":"accept","plan":"示例规划输出 plan for model %s"}' % model
        )

    monkeypatch.setattr(openai.chat.completions, "create", fake_create)


@pytest.fixture(scope="session")
def client_urban():
    return TestClient(beijing_urban.app)


@pytest.fixture(scope="session")
def client_rural():
    return TestClient(beijing_rural.app)


@pytest.fixture(scope="session")
def client_catering():
    return TestClient(beijing_catering.app)


@pytest.fixture(scope="session")
def client_hotel():
    return TestClient(china_hotel.app)


@pytest.fixture(scope="session")
def client_transport():
    return TestClient(china_transport.app)


def build_rpc_payload(
    app_message: str, task_id: str, command: TaskCommand = TaskCommand.Start
):
    # Minimal RPC request body aligned with RpcRequest schema
    sent_at = datetime.now(timezone.utc).isoformat()
    message = {
        "type": "message",
        "id": str(uuid.uuid4()),
        "sentAt": sent_at,
        "senderRole": "leader",
        "senderId": "test-suite",
        "command": command.value,
        "dataItems": [{"type": "text", "text": app_message}],
        "taskId": task_id,
        "sessionId": "session-test",
    }
    return {
        "jsonrpc": "2.0",
        "method": "rpc",
        "id": str(uuid.uuid4()),
        "params": {"message": message},
    }


@pytest.fixture
def new_task_id():
    return f"task-{uuid.uuid4()}"
