"""LangChain tools that write to and read from Bilinc agent memory."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from bilinc import BilincError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_bilinc._client import BilincClientMixin, recall_results, value_to_text

MemoryType = Literal["episodic", "procedural", "semantic", "working", "spatial"]
RecallProfile = Literal["fast", "balanced", "verified", "deep"]


class BilincRememberInput(BaseModel):
    """Input for :class:`BilincRemember`."""

    value: str | dict[str, Any] = Field(
        description="What to remember: a short fact or decision, as text or a JSON object."
    )
    key: str | None = Field(
        default=None,
        description=(
            "Optional stable, namespaced key such as 'user.preference.language'. "
            "Writing to an existing key replaces its value; the old value stays in history."
        ),
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="episodic (events), semantic (facts), procedural (how-to), working or spatial.",
    )
    importance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="How important this is, from 0.0 to 1.0."
    )


class BilincRemember(BilincClientMixin, BaseTool):
    """Commit a memory to Bilinc Cloud (``CloudClient.commit``).

    Setup:
        .. code-block:: bash

            pip install -U langchain-bilinc
            export BILINC_API_KEY="..."

    Instantiate:
        .. code-block:: python

            from langchain_bilinc import BilincRemember

            tool = BilincRemember(key_prefix="support-agent")

    Invoke:
        .. code-block:: python

            tool.invoke({"key": "user.timezone", "value": "Europe/Istanbul"})
    """

    name: str = "bilinc_remember"
    description: str = (
        "Save a durable fact, decision or preference to long-term memory so it can be "
        "recalled in later sessions. Only save what should still matter later."
    )
    args_schema: type[BaseModel] = BilincRememberInput

    key_prefix: str | None = None
    """Prefix added to every key, e.g. an agent or project name."""

    default_memory_type: MemoryType = "semantic"
    default_importance: float = 1.0
    source: str | None = "langchain"
    """The "saved by" label stored with each memory."""

    session_id: str | None = None
    metadata: dict[str, Any] | None = None
    """Extra metadata stored with every memory this tool writes."""

    def _full_key(self, key: str | None, value: Any) -> str:
        if not key:
            digest = hashlib.sha256(value_to_text(value).encode("utf-8")).hexdigest()[:16]
            key = f"memory.{digest}"
        return f"{self.key_prefix}.{key}" if self.key_prefix else key

    def _run(
        self,
        value: str | dict[str, Any],
        key: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        full_key = self._full_key(key, value)
        try:
            response = self._get_client().commit(
                full_key,
                value,
                memory_type=memory_type or self.default_memory_type,
                importance=self.default_importance if importance is None else importance,
                metadata=dict(self.metadata or {}),
                source=self.source,
                session_id=self.session_id,
            )
        except BilincError as exc:
            raise ToolException(f"Memory was not saved: {exc}") from exc
        return {
            "saved": True,
            "key": full_key,
            "entry_version": response.get("entryVersion") or response.get("entry_version"),
        }


class BilincRecallInput(BaseModel):
    """Input for :class:`BilincRecall`."""

    query: str = Field(description="What to look up in memory, in plain language.")
    limit: int | None = Field(default=None, ge=1, le=100, description="How many memories.")


class BilincRecall(BilincClientMixin, BaseTool):
    """Recall memories from Bilinc Cloud (``CloudClient.recall``).

    Instantiate:
        .. code-block:: python

            from langchain_bilinc import BilincRecall

            tool = BilincRecall(limit=5)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "what timezone is the user in?"})
    """

    name: str = "bilinc_recall"
    description: str = (
        "Search long-term memory for earlier facts, decisions and preferences. "
        "Call it before acting on something that may have been decided before."
    )
    args_schema: type[BaseModel] = BilincRecallInput

    limit: int = 5
    profile: RecallProfile = "balanced"
    """Recall profile. Higher profiles depend on the Bilinc plan."""

    memory_types: list[MemoryType] | None = None

    def _run(
        self,
        query: str,
        limit: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        try:
            response = self._get_client().recall(
                query,
                profile=self.profile,
                limit=limit or self.limit,
                memory_types=list(self.memory_types) if self.memory_types else None,
            )
        except BilincError as exc:
            raise ToolException(f"Recall failed: {exc}") from exc
        return [
            {k: v for k, v in item.items() if v is not None and k != "entry_version"}
            for item in recall_results(response)
        ]
