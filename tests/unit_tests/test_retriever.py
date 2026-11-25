from __future__ import annotations

from langchain_bilinc import BilincRetriever
from tests.fakes import RECALL_RESPONSE, FakeTransport, make_client

RECALL = "/api/cloud/memory/recall"


def test_retriever_returns_documents() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    retriever = BilincRetriever(client=make_client(fake), k=3, profile="fast")

    docs = retriever.invoke("release")

    assert fake.calls[0]["json"] == {"query": "release", "profile": "fast", "limit": 3}
    assert docs[0].page_content == '{"decision": "ship 0.1.0 after review"}'
    assert docs[0].metadata["key"] == "project.release"
    assert docs[0].metadata["score"] == 0.42
    assert docs[1].page_content == "Europe/Istanbul"
    assert docs[1].metadata["entry_version"] == "v1_def"
    assert "value" not in docs[1].metadata


def test_retriever_k_override_at_call_time() -> None:
    fake = FakeTransport({RECALL: (200, {"results": []})})
    retriever = BilincRetriever(client=make_client(fake))
    assert retriever.invoke("x", k=9) == []
    assert fake.calls[0]["json"]["limit"] == 9


async def test_retriever_async() -> None:
    fake = FakeTransport({RECALL: (200, RECALL_RESPONSE)})
    retriever = BilincRetriever(client=make_client(fake))
    docs = await retriever.ainvoke("release")
    assert len(docs) == 2
