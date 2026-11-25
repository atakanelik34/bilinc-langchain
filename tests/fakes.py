"""A recording fake for the Bilinc SDK transport (no network)."""

from __future__ import annotations

import json
from typing import Any

from bilinc import CloudClient
from bilinc.client import error_for_response


class FakeTransport:
    def __init__(self, routes: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self.routes = routes
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, method: str, url: str, *, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> dict[str, Any]:
        path = url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path.split("?", 1)[0]
        self.calls.append(
            {
                "method": method,
                "path": path,
                "headers": headers,
                "json": json.loads(body) if body else None,
            }
        )
        status, payload = self.routes[path]
        if status >= 400:
            raise error_for_response(status, payload)
        return payload


def make_client(transport: FakeTransport) -> CloudClient:
    return CloudClient(api_key="test-key", transport=transport, read_retry_attempts=1)


RECALL_RESPONSE = {
    "query": "release",
    "results": [
        {
            "key": "project.release",
            "value": {"decision": "ship 0.1.0 after review"},
            "memory_type": "episodic",
            "importance": 0.9,
            "score": 0.42,
            "entry_version": "v1_abc",
            "source": "langchain",
            "updated_at": 1794394635.8,
        },
        {
            "key": "user.timezone",
            "value": "Europe/Istanbul",
            "memoryType": "semantic",
            "importance": 1.0,
            "score": 0.2,
            "entryVersion": "v1_def",
        },
    ],
}

COMMIT_RESPONSE = {"key": "x", "entryVersion": "v1_new"}
