"""Tests for `agm daemon start/stop/status` CLI commands."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from agm.cli import main

MP = pytest.MonkeyPatch


@pytest.fixture()
def runtime_dir(tmp_path: Path) -> Path:
    """Point all daemon paths to tmp_path so tests are isolated."""
    d = tmp_path / "agm"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _patch_paths(monkeypatch: MP, runtime_dir: Path) -> None:
    """Redirect daemon path constants to tmp runtime_dir."""
    monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", runtime_dir / "appserver.sock")
    monkeypatch.setattr("agm.daemon.DEFAULT_PID_PATH", runtime_dir / "appserver.pid")
    monkeypatch.setattr("agm.daemon.DEFAULT_LOG_PATH", runtime_dir / "daemon.log")


class TestDaemonStart:
    def test_already_running(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """If daemon is already running, reports it and exits 0."""
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text(str(os.getpid()))  # current process is alive

        monkeypatch.setattr("agm.cli.os.kill", lambda pid, sig: None)

        result = CliRunner().invoke(main, ["daemon", "start"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "already_running"
        assert data["pid"] == os.getpid()

    def test_starts_daemon(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """Spawns subprocess and waits for socket to appear."""
        sock = runtime_dir / "appserver.sock"
        pid_file = runtime_dir / "appserver.pid"

        # Fake Popen that simulates daemon creating socket + PID file
        def fake_popen(*_args, **_kwargs):
            pid_file.write_text(str(12345))
            sock.touch()
            return MagicMock(pid=12345)

        monkeypatch.setattr("agm.cli.subprocess.Popen", fake_popen)
        # os.kill(12345, 0) should succeed for our fake PID
        monkeypatch.setattr("agm.cli.os.kill", lambda pid, sig: None)
        monkeypatch.setattr("agm.cli.time.sleep", lambda _: None)

        result = CliRunner().invoke(main, ["daemon", "start"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["pid"] == 12345

    def test_start_failure(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """Reports failure when socket never appears."""

        def fake_popen(*_args, **_kwargs):
            return MagicMock(pid=99999)

        monkeypatch.setattr("agm.cli.subprocess.Popen", fake_popen)
        monkeypatch.setattr("agm.cli.time.sleep", lambda _: None)
        # os.kill raises — no PID file written, process not found
        monkeypatch.setattr(
            "agm.cli.os.kill",
            MagicMock(side_effect=ProcessLookupError),
        )

        result = CliRunner().invoke(main, ["daemon", "start"])
        assert result.exit_code != 0
        assert "failed to start" in result.output.lower()

    def test_cleans_stale_pid_file(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """Removes stale PID file from a dead process before starting."""
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text("99999")
        sock = runtime_dir / "appserver.sock"

        # First os.kill(99999, 0) raises — process dead (stale PID)
        # Then os.kill(12345, 0) succeeds — new daemon alive
        kills = {"99999": ProcessLookupError, "12345": None}

        def fake_kill(pid, sig):
            exc = kills.get(str(pid))
            if exc:
                raise exc

        def fake_popen(*_args, **_kwargs):
            pid_file.write_text(str(12345))
            sock.touch()
            return MagicMock(pid=12345)

        monkeypatch.setattr("agm.cli.os.kill", fake_kill)
        monkeypatch.setattr("agm.cli.subprocess.Popen", fake_popen)
        monkeypatch.setattr("agm.cli.time.sleep", lambda _: None)

        result = CliRunner().invoke(main, ["daemon", "start"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["pid"] == 12345


class TestDaemonStop:
    def test_not_running(self, runtime_dir: Path) -> None:
        """stop when no daemon is running is a no-op."""
        result = CliRunner().invoke(main, ["daemon", "stop"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "not_running"

    def test_stops_running_daemon(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """Sends SIGTERM and reports success."""
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text("12345")

        kill_calls: list[tuple[int, int]] = []
        # First call (sig 0) succeeds — process alive
        # SIGTERM call is recorded
        # Second sig-0 check raises — process exited
        call_count = 0

        def fake_kill(pid, sig):
            nonlocal call_count
            kill_calls.append((pid, sig))
            if sig == signal.SIGTERM:
                return
            call_count += 1
            if call_count >= 2:
                raise ProcessLookupError

        monkeypatch.setattr("agm.cli.os.kill", fake_kill)
        monkeypatch.setattr("agm.cli.time.sleep", lambda _: None)

        result = CliRunner().invoke(main, ["daemon", "stop"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["pid"] == 12345
        # Verify SIGTERM was sent
        assert any(sig == signal.SIGTERM for _, sig in kill_calls)


class TestDaemonStatus:
    def test_not_running(self, runtime_dir: Path) -> None:
        result = CliRunner().invoke(main, ["daemon", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["running"] is False
        assert data["pid"] is None

    def test_running(self, runtime_dir: Path, monkeypatch: MP) -> None:
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text(str(os.getpid()))
        sock = runtime_dir / "appserver.sock"
        sock.touch()

        monkeypatch.setattr("agm.cli.os.kill", lambda pid, sig: None)

        result = CliRunner().invoke(main, ["daemon", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["running"] is True
        assert data["pid"] == os.getpid()
        assert data["socket_exists"] is True


class TestDaemonThreads:
    def test_requires_running_daemon(self, runtime_dir: Path) -> None:
        """threads command should fail when daemon is not running."""
        result = CliRunner().invoke(main, ["daemon", "threads"])
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "not running" in data["error"].lower()

    def test_forwards_thread_list_search_params(self, runtime_dir: Path, monkeypatch: MP) -> None:
        """threads command should proxy filters to thread/list via daemon client."""
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("agm.cli.os.kill", lambda pid, sig: None)

        captured: dict[str, object] = {}

        class _FakeDaemonClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def request(self, method, params=None, timeout=120):
                captured["method"] = method
                captured["params"] = params
                captured["timeout"] = timeout
                return {"data": [{"id": "thread-1"}], "nextCursor": "cursor-2"}

        monkeypatch.setattr("agm.daemon_client.DaemonClient", _FakeDaemonClient)

        result = CliRunner().invoke(
            main,
            [
                "daemon",
                "threads",
                "--search",
                "upgrade",
                "--limit",
                "15",
                "--cursor",
                "cursor-1",
                "--archived",
                "--sort-key",
                "updated_at",
                "--cwd",
                "/tmp/project",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["nextCursor"] == "cursor-2"
        assert data["data"][0]["id"] == "thread-1"
        assert "status_type" in data["data"][0]
        assert "active_flags" in data["data"][0]
        assert captured["method"] == "thread/list"
        assert captured["params"] == {
            "searchTerm": "upgrade",
            "limit": 15,
            "cursor": "cursor-1",
            "archived": True,
            "sortKey": "updated_at",
            "cwd": "/tmp/project",
        }

    def test_surfaces_daemon_connection_errors_as_json_click_error(
        self, runtime_dir: Path, monkeypatch: MP
    ) -> None:
        """threads command should preserve JSON error shape on daemon connection failures."""
        pid_file = runtime_dir / "appserver.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("agm.cli.os.kill", lambda pid, sig: None)

        class _BrokenDaemonClient:
            async def __aenter__(self):
                raise FileNotFoundError("appserver.sock not found")

            async def __aexit__(self, *_exc):
                return None

        monkeypatch.setattr("agm.daemon_client.DaemonClient", _BrokenDaemonClient)

        result = CliRunner().invoke(main, ["daemon", "threads"])
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "failed to list daemon threads" in data["error"].lower()
