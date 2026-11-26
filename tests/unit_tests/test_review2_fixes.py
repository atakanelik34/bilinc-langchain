"""Regression tests for the second code review round (2026-11-26, no network)."""

from __future__ import annotations

import inspect
import logging

import bilinc.client
import pytest
from bilinc import BilincApiKeyRequired, BilincAuthError, BilincError, BilincRateLimitError

import langchain_bilinc.retrievers as retrievers_module
from langchain_bilinc import BilincRetriever
from tests.fakes import RECALL_RESPONSE, FakeTransport, make_client

RECALL = "/api/cloud/memory/recall"


# Finding 5: the SDK's own error types reach the caller -------------------------


@pytest.mark.parametrize(
    "status, payload, error_type",
    [
        (429, {"error": "rate_limited", "message": "slow down"}, BilincRateLimitError),
        (401, {"error": "invalid_api_key", "message": "bad key"}, BilincAuthError),
    ],
)
def test_bilinc_errors_propagate_unchanged(status, payload, error_type, caplog) -> None:
    fake = FakeTransport({RECALL: (status, payload)})
    retriever = BilincRetriever(client=make_client(fake))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(error_type):
            retriever.invoke("release")
    # existing `except BilincError` code keeps working
    with pytest.raises(BilincError):
        retriever.invoke("release")
    assert caplog.records == []  # raised, not also logged


def test_missing_key_raises_the_sdk_error() -> None:
    with pytest.raises(BilincApiKeyRequired):
        BilincRetriever().invoke("release")


# Finding 8: k <= 0 returns nothing and costs nothing ------------------------------


@pytest.mark.parametrize("k", [0, -1, -50])
def test_non_positive_k_makes_no_call(k: int) -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    assert BilincRetriever(client=make_client(fake), k=k).invoke("release") == []
    assert BilincRetriever(client=make_client(fake)).invoke("release", k=k) == []
    assert fake.calls == []


# Cleanup: the limits come from the SDK, not local copies --------------------------


def test_limits_are_the_sdk_constants() -> None:
    assert retrievers_module.MAX_RECALL_LIMIT is bilinc.client.MAX_RECALL_LIMIT
    assert retrievers_module.MAX_QUERY_LENGTH is bilinc.client.MAX_QUERY_LENGTH
    source = inspect.getsource(retrievers_module)
    assert "MAX_RECALL_LIMIT = " not in source and "MAX_QUERY_LENGTH = " not in source
    assert "logger" not in source
