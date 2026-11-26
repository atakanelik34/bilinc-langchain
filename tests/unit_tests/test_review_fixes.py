"""Regression tests for the code review findings (no network)."""

from __future__ import annotations

from langchain_bilinc import BilincRetriever
from tests.fakes import RECALL_RESPONSE, FakeTransport, make_client

RECALL = "/api/cloud/memory/recall"


def test_k_is_capped_at_100() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    BilincRetriever(client=make_client(fake), k=500).invoke("release")
    BilincRetriever(client=make_client(fake)).invoke("release", k=1000)
    assert [c["json"]["limit"] for c in fake.calls] == [100, 100]


def test_query_is_cut_at_4096_characters() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    BilincRetriever(client=make_client(fake)).invoke("q" * 5000)
    assert len(fake.calls[0]["json"]["query"]) == 4096


def test_blank_query_returns_nothing_without_a_call() -> None:
    fake = FakeTransport({})
    assert BilincRetriever(client=make_client(fake)).invoke("   ") == []
    assert fake.calls == []
