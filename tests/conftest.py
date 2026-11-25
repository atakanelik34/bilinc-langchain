"""Test setup: no network, no real key, and no saved Bilinc login is read."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access is disabled in unit tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    for name in ("BILINC_API_KEY", "BILINC_BASE_URL", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    # Point the SDK at an empty config dir so a developer's `bilinc login` is never read.
    monkeypatch.setenv("BILINC_CONFIG_DIR", str(tmp_path / "bilinc-config"))
