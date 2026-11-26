"""Regression tests for the fifth review round: long session IDs (no network)."""

from __future__ import annotations

import langchain_bilinc._client as client_module
from langchain_bilinc import BilincRemember
from tests.fakes import COMMIT_RESPONSE, FakeTransport, make_client

COMMIT = "/api/cloud/memory/commit"


def remember(session_id: str | None) -> dict:
    fake = FakeTransport({COMMIT: (201, COMMIT_RESPONSE)})
    BilincRemember(client=make_client(fake), session_id=session_id).invoke({"value": "x"})
    return fake.calls[0]["json"]


def test_limit_matches_the_server() -> None:
    assert client_module.MAX_SESSION_ID_LENGTH == 256


def test_long_session_id_is_hashed() -> None:
    session = "s" * 300
    body = remember(session)
    assert body["sessionId"] == client_module._session_hash(session)
    assert len(body["sessionId"]) <= 256


def test_session_id_at_the_limit_is_sent_as_is() -> None:
    assert remember("t" * 256)["sessionId"] == "t" * 256


def test_short_and_blank_session_ids() -> None:
    assert remember(" s1 ")["sessionId"] == "s1"
    assert "sessionId" not in remember("   ")
    assert "sessionId" not in remember(None)


def test_same_hash_as_the_llamaindex_package_would_use() -> None:
    # Both packages use "h" + the first 24 hex characters of SHA-256.
    import hashlib

    session = "u" * 400
    expected = "h" + hashlib.sha256(session.encode("utf-8")).hexdigest()[:24]
    assert remember(session)["sessionId"] == expected
