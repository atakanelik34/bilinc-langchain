"""Client construction and result shaping shared by the Bilinc tools and retriever."""

from __future__ import annotations

import json
from typing import Any

from bilinc import CloudClient
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr


class BilincClientMixin(BaseModel):
    """Builds a :class:`bilinc.CloudClient` lazily.

    The key comes from ``api_key``, else ``BILINC_API_KEY``, else the key saved by
    ``bilinc login`` (the SDK's own lookup). Nothing is sent until the first call.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    bilinc_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("BILINC_API_KEY", default=None),
        alias="api_key",
    )
    """Bilinc Cloud API key."""

    base_url: str | None = None
    """Override the Bilinc Cloud URL. Defaults to the SDK's own choice."""

    client: CloudClient | None = Field(default=None, exclude=True)
    """Optional pre-built client; when set, ``api_key`` and ``base_url`` are ignored."""

    _client: CloudClient | None = PrivateAttr(default=None)

    def _get_client(self) -> CloudClient:
        if self.client is not None:
            return self.client
        if self._client is None:
            key = self.bilinc_api_key.get_secret_value() if self.bilinc_api_key else None
            self._client = CloudClient(api_key=key, base_url=self.base_url)
        return self._client


def _pick(item: dict[str, Any], *names: str) -> Any:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    return None


def recall_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a recall response into a list of plain result dicts."""

    items = response.get("results")
    if items is None:
        items = response.get("memories") or []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "key": item.get("key"),
                "value": item.get("value"),
                "memory_type": _pick(item, "memory_type", "memoryType"),
                "importance": item.get("importance"),
                "score": item.get("score"),
                "source": item.get("source"),
                "entry_version": _pick(item, "entry_version", "entryVersion"),
                "updated_at": _pick(item, "updated_at", "updatedAt"),
            }
        )
    return out


def value_to_text(value: Any) -> str:
    """Render a stored memory value as text for an LLM or a Document."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
