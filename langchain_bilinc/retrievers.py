"""A LangChain retriever over Bilinc agent memory."""

from __future__ import annotations

from typing import Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_bilinc._client import BilincClientMixin, recall_results, value_to_text


class BilincRetriever(BilincClientMixin, BaseRetriever):
    """Retrieve Bilinc memories as LangChain ``Document`` objects.

    Each memory becomes one document: ``page_content`` is the stored value (JSON
    values are serialized), and ``metadata`` holds the key, memory type,
    importance, score, source, entry version and update time.

    Example:
        .. code-block:: python

            from langchain_bilinc import BilincRetriever

            retriever = BilincRetriever(k=5)
            docs = retriever.invoke("what did we decide about the release?")
    """

    k: int = 5
    """Number of memories to return."""

    profile: Literal["fast", "balanced", "verified", "deep"] = "balanced"
    memory_types: list[str] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: object
    ) -> list[Document]:
        k = int(kwargs.get("k", self.k))  # type: ignore[call-overload]
        response = self._get_client().recall(
            query,
            profile=self.profile,
            limit=k,
            memory_types=list(self.memory_types) if self.memory_types else None,
        )
        docs: list[Document] = []
        for item in recall_results(response):
            metadata = {name: v for name, v in item.items() if name != "value" and v is not None}
            docs.append(Document(page_content=value_to_text(item["value"]), metadata=metadata))
        return docs
