"""LangChain integration for Bilinc: shared, versioned memory for AI agents."""

from importlib import metadata

from langchain_bilinc.retrievers import BilincRetriever
from langchain_bilinc.tools import (
    BilincRecall,
    BilincRecallInput,
    BilincRemember,
    BilincRememberInput,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = ""

__all__ = [
    "BilincRecall",
    "BilincRecallInput",
    "BilincRemember",
    "BilincRememberInput",
    "BilincRetriever",
    "__version__",
]
