from __future__ import annotations

import pytest
from bilinc import BilincApiKeyRequired
from langchain_core.tools import ToolException

from langchain_bilinc import BilincRecall, BilincRemember
from tests.fakes import COMMIT_RESPONSE, RECALL_RESPONSE, FakeTransport, make_client

COMMIT = "/api/cloud/memory/commit"
RECALL = "/api/cloud/memory/recall"


def test_remember_commits_with_prefix_and_defaults() -> None:
    fake = FakeTransport({COMMIT: (201, COMMIT_RESPONSE)})
    tool = BilincRemember(client=make_client(fake), key_prefix="agent", session_id="s1")

    out = tool.invoke({"key": "user.timezone", "value": "Europe/Istanbul"})

    call = fake.calls[0]
    assert call["method"] == "POST" and call["path"] == COMMIT
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["key"] == "agent.user.timezone"
    assert call["json"]["value"] == "Europe/Istanbul"
    assert call["json"]["memoryType"] == "semantic"
    assert call["json"]["importance"] == 1.0
    assert call["json"]["source"] == "langchain"
    assert call["json"]["sessionId"] == "s1"
    assert out == {"saved": True, "key": "agent.user.timezone", "entry_version": "v1_new"}


def test_remember_without_key_uses_stable_content_key() -> None:
    fake = FakeTransport({COMMIT: (201, COMMIT_RESPONSE)})
    tool = BilincRemember(client=make_client(fake))
    first = tool.invoke({"value": {"fact": "a"}, "memory_type": "episodic", "importance": 0.5})
    second = tool.invoke({"value": {"fact": "a"}})
    assert first["key"] == second["key"]
    assert first["key"].startswith("memory.")
    assert fake.calls[0]["json"]["memoryType"] == "episodic"
    assert fake.calls[0]["json"]["importance"] == 0.5


def test_remember_error_is_tool_exception() -> None:
    error = {"error": "payment_required", "message": "Monthly credits used up"}
    fake = FakeTransport({COMMIT: (402, error)})
    tool = BilincRemember(client=make_client(fake))
    with pytest.raises(ToolException, match="not saved.*Monthly credits used up"):
        tool.invoke({"value": "x"})


def test_recall_returns_normalized_results() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    tool = BilincRecall(client=make_client(fake), limit=7, memory_types=["semantic"])

    out = tool.invoke({"query": "release"})

    body = fake.calls[0]["json"]
    assert body == {
        "query": "release",
        "profile": "balanced",
        "limit": 7,
        "memoryTypes": ["semantic"],
    }
    assert out[0]["key"] == "project.release"
    assert out[0]["memory_type"] == "episodic"
    assert out[1]["memory_type"] == "semantic"
    assert "entry_version" not in out[0]


async def test_recall_async_uses_executor() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    tool = BilincRecall(client=make_client(fake))
    out = await tool.ainvoke({"query": "release", "limit": 2})
    assert fake.calls[0]["json"]["limit"] == 2
    assert len(out) == 2


def test_recall_error_is_tool_exception() -> None:
    error = {"error": "capability_not_entitled", "message": "profile not in plan"}
    fake = FakeTransport({RECALL: (403, error)})
    tool = BilincRecall(client=make_client(fake), profile="deep")
    with pytest.raises(ToolException, match="Recall failed"):
        tool.invoke({"query": "x"})


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILINC_API_KEY", "env-key")
    tool = BilincRecall()
    assert tool.bilinc_api_key is not None
    assert tool.bilinc_api_key.get_secret_value() == "env-key"
    assert "env-key" not in repr(tool)
    assert tool._get_client().api_key == "env-key"


def test_missing_key_fails_only_on_first_call() -> None:
    tool = BilincRecall()  # constructing without a key is fine
    with pytest.raises(BilincApiKeyRequired):
        tool._get_client()
