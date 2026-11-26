"""Regression tests for the 2026-11-26 code review finding 7 (no network)."""

from __future__ import annotations

import logging

import pytest
from bilinc import BilincRateLimitError

from langchain_bilinc import BilincRetriever, BilincRetrieverError
from tests.fakes import RECALL_RESPONSE, FakeTransport, make_client

RECALL = "/api/cloud/memory/recall"


def test_k_is_capped_at_100() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    BilincRetriever(client=make_client(fake), k=500).invoke("release")
    BilincRetriever(client=make_client(fake)).invoke("release", k=1000)
    BilincRetriever(client=make_client(fake)).invoke("release", k=0)
    assert [c["json"]["limit"] for c in fake.calls] == [100, 100, 1]


def test_query_is_cut_at_4096_characters() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    BilincRetriever(client=make_client(fake)).invoke("q" * 5000)
    assert len(fake.calls[0]["json"]["query"]) == 4096


def test_blank_query_returns_nothing_without_a_call() -> None:
    fake = FakeTransport({})
    assert BilincRetriever(client=make_client(fake)).invoke("   ") == []
    assert fake.calls == []


def test_bilinc_error_becomes_a_clear_error(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeTransport({RECALL: (429, {"error": "rate_limited", "message": "slow down"})})
    retriever = BilincRetriever(client=make_client(fake))
    with caplog.at_level(logging.WARNING, logger="langchain_bilinc.retrievers"):
        with pytest.raises(BilincRetrieverError, match="Bilinc recall failed: .*slow down") as info:
            retriever.invoke("release")
    assert isinstance(info.value.__cause__, BilincRateLimitError)
    assert "Bilinc recall failed" in caplog.text


def test_missing_key_becomes_a_clear_error() -> None:
    with pytest.raises(BilincRetrieverError, match="No Bilinc Cloud API key"):
        BilincRetriever().invoke("release")
