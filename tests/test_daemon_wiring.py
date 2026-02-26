"""Tests for daemon auto-detection and _codex_client() backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.jobs_common import _should_use_daemon

MP = pytest.MonkeyPatch


class TestShouldUseDaemon:
    def test_env_direct_overrides_socket(self, monkeypatch: MP, tmp_path: Path) -> None:
        """AGM_BACKEND_MODE=direct forces direct even with socket."""
        sock = tmp_path / "test.sock"
        sock.touch()
        monkeypatch.setenv("AGM_BACKEND_MODE", "direct")
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)
        assert _should_use_daemon() is False

    def test_env_daemon_forces_daemon(self, monkeypatch: MP, tmp_path: Path) -> None:
        """AGM_BACKEND_MODE=daemon forces daemon even without socket."""
        sock = tmp_path / "nonexistent.sock"
        monkeypatch.setenv("AGM_BACKEND_MODE", "daemon")
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)
        assert _should_use_daemon() is True

    def test_auto_detect_socket_exists(self, monkeypatch: MP, tmp_path: Path) -> None:
        """Without env var, returns True when socket file exists."""
        sock = tmp_path / "test.sock"
        sock.touch()
        monkeypatch.delenv("AGM_BACKEND_MODE", raising=False)
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)
        assert _should_use_daemon() is True

    def test_auto_detect_no_socket(self, monkeypatch: MP, tmp_path: Path) -> None:
        """Without env var, returns False when socket missing."""
        sock = tmp_path / "nonexistent.sock"
        monkeypatch.delenv("AGM_BACKEND_MODE", raising=False)
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)
        assert _should_use_daemon() is False

    def test_env_case_insensitive(self, monkeypatch: MP, tmp_path: Path) -> None:
        """AGM_BACKEND_MODE is case-insensitive."""
        sock = tmp_path / "nonexistent.sock"
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)

        monkeypatch.setenv("AGM_BACKEND_MODE", "DAEMON")
        assert _should_use_daemon() is True

        monkeypatch.setenv("AGM_BACKEND_MODE", "Direct")
        assert _should_use_daemon() is False

    def test_unknown_value_falls_to_auto(self, monkeypatch: MP, tmp_path: Path) -> None:
        """Unknown AGM_BACKEND_MODE falls through to auto-detection."""
        sock = tmp_path / "test.sock"
        sock.touch()
        monkeypatch.setenv("AGM_BACKEND_MODE", "bogus")
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", sock)
        assert _should_use_daemon() is True
