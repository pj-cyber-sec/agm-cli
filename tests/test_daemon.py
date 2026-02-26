"""Tests for the shared app-server daemon and DaemonClient.

Uses real asyncio Unix sockets with a mocked codex app-server subprocess.
Tests observable behaviors: request multiplexing, notification routing,
server request forwarding, idle timeout, crash recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agm.daemon import AppServerDaemon
from agm.daemon_client import DaemonClient

# ---------------------------------------------------------------------------
# Fake app-server subprocess
# ---------------------------------------------------------------------------


class FakeAppServerIO:
    """Simulates app-server subprocess stdio.

    Feed lines into stdout (from app-server perspective) and capture
    what was written to stdin (requests from daemon).
    """

    def __init__(self) -> None:
        self._stdout_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stdin_lines: list[dict[str, Any]] = []
        self._stdin_event = asyncio.Event()

    # -- stdout (daemon reads from here) --

    def feed_stdout(self, msg: dict) -> None:
        self._stdout_queue.put_nowait(json.dumps(msg).encode() + b"\n")

    def close_stdout(self) -> None:
        self._stdout_queue.put_nowait(b"")

    async def readline_stdout(self) -> bytes:
        return await self._stdout_queue.get()

    # -- stdin (daemon writes here) --

    def write_stdin(self, data: bytes) -> None:
        try:
            parsed = json.loads(data.strip())
            self._stdin_lines.append(parsed)
            self._stdin_event.set()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    async def wait_for_stdin(self, count: int = 1, timeout: float = 2) -> None:
        """Wait until at least `count` messages have been written to stdin."""
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self._stdin_lines) < count:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Expected {count} stdin messages, got {len(self._stdin_lines)}")
            self._stdin_event.clear()
            try:
                await asyncio.wait_for(self._stdin_event.wait(), timeout=remaining)
            except TimeoutError:
                break

    def pop_stdin(self) -> dict[str, Any]:
        return self._stdin_lines.pop(0)

    @property
    def stdin_lines(self) -> list[dict[str, Any]]:
        return self._stdin_lines


def make_fake_process(io: FakeAppServerIO) -> MagicMock:
    proc = MagicMock()
    proc.pid = 12345

    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock(side_effect=lambda data: io.write_stdin(data))
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=io.readline_stdout)

    fake_stderr = MagicMock()
    stderr_queue: asyncio.Queue[bytes] = asyncio.Queue()
    stderr_queue.put_nowait(b"")  # immediate EOF
    fake_stderr.readline = AsyncMock(side_effect=stderr_queue.get)
    proc.stderr = fake_stderr

    proc.wait = AsyncMock()
    return proc


# ---------------------------------------------------------------------------
# Auto-respond helper
# ---------------------------------------------------------------------------


class AutoResponder:
    """Watches stdin and auto-responds to requests with configurable responses."""

    def __init__(self, io: FakeAppServerIO) -> None:
        self._io = io
        self._task: asyncio.Task | None = None
        self._responses: dict[str, dict] = {}
        self._default_result: dict = {}

    def set_response(self, method: str, result: dict) -> None:
        self._responses[method] = result

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            await self._io.wait_for_stdin(len(self._io.stdin_lines) + 1, timeout=60)
            while self._io.stdin_lines:
                msg = self._io.pop_stdin()
                msg_id = msg.get("id")
                method = msg.get("method")
                if msg_id is not None and method is not None:
                    result = self._responses.get(method, self._default_result)
                    self._io.feed_stdout({"jsonrpc": "2.0", "id": msg_id, "result": result})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_socket(tmp_path: Path) -> Path:
    return tmp_path / "test.sock"


@pytest.fixture
def appserver_io() -> FakeAppServerIO:
    return FakeAppServerIO()


@pytest.fixture
def fake_proc(appserver_io: FakeAppServerIO) -> MagicMock:
    return make_fake_process(appserver_io)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaemonStartStop:
    @pytest.mark.asyncio
    async def test_daemon_creates_socket(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        # Auto-respond to initialize
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                assert tmp_socket.exists()
            finally:
                await daemon.stop()

        assert not tmp_socket.exists()

    @pytest.mark.asyncio
    async def test_daemon_writes_pid_file(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})
        pid_path = tmp_socket.with_suffix(".pid")

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                assert pid_path.exists()
                assert pid_path.read_text().strip().isdigit()
            finally:
                await daemon.stop()

        assert not pid_path.exists()


class TestRequestMultiplexing:
    @pytest.mark.asyncio
    async def test_single_worker_request_response(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """A worker's request flows through daemon to app-server and back."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        responder = AutoResponder(appserver_io)
        responder.set_response("model/list", {"data": ["gpt-4"]})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            responder.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    result = await client.request("model/list", timeout=5)
                    assert result == {"data": ["gpt-4"]}
            finally:
                await responder.stop()
                await daemon.stop()

    @pytest.mark.asyncio
    async def test_two_workers_get_correct_responses(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """Two workers sending requests simultaneously get their own responses."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        responder = AutoResponder(appserver_io)
        responder.set_response("thread/list", {"threads": []})
        responder.set_response("model/list", {"models": ["a"]})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            responder.start()
            try:
                async with (
                    DaemonClient(socket_path=tmp_socket) as c1,
                    DaemonClient(socket_path=tmp_socket) as c2,
                ):
                    r1, r2 = await asyncio.gather(
                        c1.request("thread/list", timeout=5),
                        c2.request("model/list", timeout=5),
                    )
                    assert r1 == {"threads": []}
                    assert r2 == {"models": ["a"]}
            finally:
                await responder.stop()
                await daemon.stop()

    @pytest.mark.asyncio
    async def test_rpc_error_propagated_to_worker(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """RPC errors from app-server are propagated to the requesting worker."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    # Manually respond with an error after the request arrives
                    req_task = asyncio.create_task(client.request("bad/method", timeout=5))
                    await asyncio.sleep(0.05)
                    # Find the forwarded request ID
                    await appserver_io.wait_for_stdin(2)  # init + this request
                    forwarded = appserver_io.stdin_lines[-1]
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": forwarded["id"],
                            "error": {"code": -32601, "message": "Method not found"},
                        }
                    )
                    from agm.client import RPCError

                    with pytest.raises(RPCError, match="Method not found"):
                        await req_task
            finally:
                await daemon.stop()


class TestThreadRegistrationAndNotificationRouting:
    @pytest.mark.asyncio
    async def test_thread_start_auto_registers_owner(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """thread/start response auto-registers the requesting worker as owner."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with (
                    DaemonClient(socket_path=tmp_socket) as c1,
                    DaemonClient(socket_path=tmp_socket) as c2,
                ):
                    # c1 starts a thread
                    req_task = asyncio.create_task(
                        c1.request("thread/start", {"model": "test"}, timeout=5)
                    )
                    await asyncio.sleep(0.05)
                    await appserver_io.wait_for_stdin(2)
                    forwarded = appserver_io.stdin_lines[-1]
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": forwarded["id"],
                            "result": {"thread": {"id": "thread-abc", "model": "test"}},
                        }
                    )
                    await req_task

                    # Now send a thread-scoped notification — only c1 should get it
                    c1_received: list[dict] = []
                    c2_received: list[dict] = []

                    def c1_handler(params: dict) -> None:
                        c1_received.append(params)

                    def c2_handler(params: dict) -> None:
                        c2_received.append(params)

                    c1.on_notification("turn/started", c1_handler)
                    c2.on_notification("turn/started", c2_handler)

                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "method": "turn/started",
                            "params": {
                                "threadId": "thread-abc",
                                "turn": {"id": "t1"},
                            },
                        }
                    )
                    await asyncio.sleep(0.1)

                    assert len(c1_received) == 1
                    assert c1_received[0]["threadId"] == "thread-abc"
                    assert len(c2_received) == 0

                    c1_thread_status: list[dict] = []
                    c2_thread_status: list[dict] = []
                    c1.on_notification(
                        "thread/status/changed",
                        lambda p: c1_thread_status.append(p),
                    )
                    c2.on_notification(
                        "thread/status/changed",
                        lambda p: c2_thread_status.append(p),
                    )
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "method": "thread/status/changed",
                            "params": {
                                "threadId": "thread-abc",
                                "oldStatus": "idle",
                                "newStatus": "running",
                            },
                        }
                    )
                    await asyncio.sleep(0.1)
                    assert len(c1_thread_status) == 1
                    assert c1_thread_status[0]["threadId"] == "thread-abc"
                    assert len(c2_thread_status) == 0
            finally:
                await daemon.stop()

    @pytest.mark.asyncio
    async def test_broadcast_notification(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """Notifications without threadId are broadcast to all workers."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with (
                    DaemonClient(socket_path=tmp_socket) as c1,
                    DaemonClient(socket_path=tmp_socket) as c2,
                ):
                    c1_got: list[dict] = []
                    c2_got: list[dict] = []
                    c1.on_notification("deprecationNotice", lambda p: c1_got.append(p))
                    c2.on_notification("deprecationNotice", lambda p: c2_got.append(p))
                    await asyncio.sleep(0.05)

                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "method": "deprecationNotice",
                            "params": {"message": "old API"},
                        }
                    )
                    await asyncio.sleep(0.1)

                    assert len(c1_got) == 1
                    assert len(c2_got) == 1
            finally:
                await daemon.stop()


class TestServerRequestForwarding:
    @pytest.mark.asyncio
    async def test_server_request_forwarded_to_thread_owner(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """Server requests are forwarded to the worker that owns the thread."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    # Set up approval handler
                    async def approve(method: str, params: dict) -> dict:
                        return {"decision": "accept"}

                    client.set_server_request_handler(approve)

                    # Register thread ownership via thread/start
                    req_task = asyncio.create_task(
                        client.request("thread/start", {"model": "test"}, timeout=5)
                    )
                    await asyncio.sleep(0.05)
                    await appserver_io.wait_for_stdin(2)
                    fwd = appserver_io.stdin_lines[-1]
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": fwd["id"],
                            "result": {"thread": {"id": "t-1"}},
                        }
                    )
                    await req_task

                    # App-server sends a server request for this thread
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": 900,
                            "method": "item/commandExecution/requestApproval",
                            "params": {"threadId": "t-1", "command": "ls"},
                        }
                    )
                    # Wait for the daemon to forward and worker to respond
                    await asyncio.sleep(0.2)

                    # Check that daemon forwarded the worker's response back
                    # to app-server
                    responses = [m for m in appserver_io.stdin_lines if m.get("id") == 900]
                    assert len(responses) == 1
                    assert responses[0]["result"] == {"decision": "accept"}
            finally:
                await daemon.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "params", "expected_response"),
        [
            (
                "execCommandApproval",
                {
                    "conversationId": "t-1",
                    "callId": "call-1",
                    "command": ["bash", "-lc", "echo hi"],
                    "cwd": "/tmp",
                    "parsedCmd": [],
                },
                {"decision": "approved"},
            ),
            (
                "applyPatchApproval",
                {
                    "conversationId": "t-1",
                    "callId": "patch-1",
                    "fileChanges": {},
                },
                {"decision": "approved"},
            ),
            (
                "item/tool/requestUserInput",
                {
                    "threadId": "t-1",
                    "turnId": "turn-1",
                    "itemId": "item-1",
                    "questions": [],
                },
                {"answers": {}},
            ),
            (
                "account/chatgptAuthTokens/refresh",
                {"reason": "expired"},
                {"accessToken": "tok-1", "chatgptAccountId": "acct-1"},
            ),
        ],
    )
    async def test_server_request_roundtrip_new_methods(
        self,
        tmp_socket: Path,
        appserver_io: FakeAppServerIO,
        fake_proc: MagicMock,
        method: str,
        params: dict[str, Any],
        expected_response: dict[str, Any],
    ) -> None:
        """New server request methods should round-trip through daemon + client."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    responses = {
                        "execCommandApproval": {"decision": "approved"},
                        "applyPatchApproval": {"decision": "approved"},
                        "item/tool/requestUserInput": {"answers": {}},
                        "account/chatgptAuthTokens/refresh": {
                            "accessToken": "tok-1",
                            "chatgptAccountId": "acct-1",
                        },
                    }

                    async def handler(request_method: str, _request_params: dict) -> dict:
                        return responses[request_method]

                    client.set_server_request_handler(handler)

                    # Register thread ownership via thread/start.
                    req_task = asyncio.create_task(
                        client.request("thread/start", {"model": "test"}, timeout=5)
                    )
                    await asyncio.sleep(0.05)
                    await appserver_io.wait_for_stdin(2)
                    fwd = appserver_io.stdin_lines[-1]
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": fwd["id"],
                            "result": {"thread": {"id": "t-1"}},
                        }
                    )
                    await req_task

                    request_id = 901
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": method,
                            "params": params,
                        }
                    )
                    await asyncio.sleep(0.2)

                    routed = [m for m in appserver_io.stdin_lines if m.get("id") == request_id]
                    assert len(routed) == 1
                    assert routed[0]["result"] == expected_response
            finally:
                await daemon.stop()


class TestWorkerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_unregisters_threads(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """When a worker disconnects, its threads are unregistered."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                # Connect, start thread, disconnect
                async with DaemonClient(socket_path=tmp_socket) as client:
                    req_task = asyncio.create_task(
                        client.request("thread/start", {"model": "x"}, timeout=5)
                    )
                    await asyncio.sleep(0.05)
                    await appserver_io.wait_for_stdin(2)
                    fwd = appserver_io.stdin_lines[-1]
                    appserver_io.feed_stdout(
                        {
                            "jsonrpc": "2.0",
                            "id": fwd["id"],
                            "result": {"thread": {"id": "thread-x"}},
                        }
                    )
                    await req_task

                # Worker disconnected — thread should now be unregistered
                await asyncio.sleep(0.1)
                assert "thread-x" not in daemon._threads
            finally:
                await daemon.stop()


class TestAppServerCrash:
    @pytest.mark.asyncio
    async def test_eof_fails_pending_worker_requests(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """When app-server exits, pending worker requests fail with RPCError."""
        from agm.client import RPCError

        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    req = asyncio.create_task(client.request("slow/method", timeout=5))
                    await asyncio.sleep(0.05)
                    # Simulate app-server crash
                    appserver_io.close_stdout()
                    with pytest.raises(RPCError, match="App-server process exited"):
                        await req
            finally:
                await daemon.stop()

    @pytest.mark.asyncio
    async def test_auto_restart_on_crash(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """Daemon auto-restarts app-server when it crashes with workers connected."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        # Second process for the restart
        io2 = FakeAppServerIO()
        io2.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})
        proc2 = make_fake_process(io2)

        call_count = 0

        async def mock_exec(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fake_proc
            return proc2

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=mock_exec),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            try:
                async with DaemonClient(socket_path=tmp_socket) as client:
                    await asyncio.sleep(0.05)
                    # Crash the first app-server
                    appserver_io.close_stdout()
                    # Wait for auto-restart
                    await asyncio.sleep(0.3)
                    assert call_count == 2

                    # New requests should work on the restarted app-server
                    responder = AutoResponder(io2)
                    responder.set_response("ping", {"pong": True})
                    responder.start()
                    try:
                        result = await client.request("ping", timeout=5)
                        assert result == {"pong": True}
                    finally:
                        await responder.stop()
            finally:
                await daemon.stop()


class TestDaemonClientEOF:
    @pytest.mark.asyncio
    async def test_daemon_disconnect_fails_pending(
        self, tmp_socket: Path, appserver_io: FakeAppServerIO, fake_proc: MagicMock
    ) -> None:
        """If daemon closes connection, pending worker requests fail."""
        appserver_io.feed_stdout({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}})

        with patch(
            "agm.daemon.asyncio.create_subprocess_exec",
            AsyncMock(return_value=fake_proc),
        ):
            daemon = AppServerDaemon(socket_path=tmp_socket, idle_timeout=0)
            await daemon.start()
            client = DaemonClient(socket_path=tmp_socket)
            await client.start()
            try:
                req = asyncio.create_task(client.request("slow/method", timeout=5))
                await asyncio.sleep(0.05)
                # Kill daemon while request is pending
                await daemon.stop()
                with pytest.raises(ConnectionError):
                    await req
            finally:
                await client.stop()
