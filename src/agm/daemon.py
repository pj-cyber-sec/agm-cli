"""Shared app-server daemon: one codex process, many workers.

The daemon owns a single ``codex app-server`` subprocess and exposes a
Unix domain socket for rq workers.  Workers send JSON-RPC requests over
the socket; the daemon multiplexes them onto the subprocess stdio pipe
and routes responses, notifications, and server requests back to the
correct worker.

Notification routing uses ``threadId``: the daemon auto-registers thread
ownership when a ``thread/start`` or ``thread/resume`` response flows
through.  Notifications without ``threadId`` are broadcast to all
connected workers.

Run directly::

    python -m agm.daemon          # foreground
    python -m agm.daemon --bg     # detach (for auto-start)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# -- Paths ----------------------------------------------------------------

_RUNTIME_DIR = Path(f"/run/user/{os.getuid()}/agm")
DEFAULT_SOCKET_PATH = _RUNTIME_DIR / "appserver.sock"
DEFAULT_PID_PATH = _RUNTIME_DIR / "appserver.pid"
DEFAULT_LOG_PATH = _RUNTIME_DIR / "daemon.log"
DEFAULT_IDLE_TIMEOUT = 300  # seconds


# -- Wire protocol (daemon <-> worker) -----------------------------------


def _encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode() + b"\n"


def _decode(line: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# -- Notifications: threadId extraction -----------------------------------

# Notifications that carry threadId (route to owning worker).
# Everything else is broadcast.
_THREAD_SCOPED_NOTIFICATIONS = frozenset(
    {
        "turn/started",
        "turn/completed",
        "turn/errored",
        "turn/plan/updated",
        "turn/diff/updated",
        "item/started",
        "item/completed",
        "item/agentMessage/delta",
        "item/reasoning/summaryTextDelta",
        "item/commandExecution/outputDelta",
        "item/fileChange/outputDelta",
        "thread/tokenUsage/updated",
        "thread/status/changed",
        "thread/name/updated",
        "thread/compacted",
        "model/rerouted",
        "error",
    }
)


def _extract_thread_id(method: str, params: dict) -> str | None:
    """Extract threadId from notification params, if present."""
    if method in _THREAD_SCOPED_NOTIFICATIONS:
        return params.get("threadId")
    return None


def _extract_server_request_thread_id(params: dict[str, Any]) -> str | None:
    """Extract thread ownership key from server request params."""
    for key in ("threadId", "conversationId"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# -- Methods whose responses carry a new threadId to auto-register --------

_THREAD_START_METHODS = frozenset({"thread/start", "thread/resume"})


# -- Worker connection ----------------------------------------------------


class _Worker:
    """State for one connected worker."""

    __slots__ = ("reader", "writer", "threads", "addr")

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.threads: set[str] = set()
        self.addr = writer.get_extra_info("peername") or "unknown"

    def send(self, msg: dict) -> None:
        """Queue a message to this worker (non-blocking)."""
        try:
            self.writer.write(_encode(msg))
        except Exception:
            log.debug("Failed to write to worker %s", self.addr)

    async def drain(self) -> None:
        await self.writer.drain()

    def close(self) -> None:
        self.writer.close()


# -- Daemon ---------------------------------------------------------------


class AppServerDaemon:
    """Multiplexing daemon for a shared codex app-server process."""

    def __init__(
        self,
        *,
        socket_path: Path = DEFAULT_SOCKET_PATH,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._idle_timeout = idle_timeout
        self._env = env

        # App-server subprocess
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

        # ID management
        self._next_id = 1
        # daemon_id -> (worker, worker_id, method)
        self._pending: dict[int, tuple[_Worker, int | str, str]] = {}

        # Thread ownership: threadId -> worker
        self._threads: dict[str, _Worker] = {}

        # Connected workers
        self._workers: set[_Worker] = set()

        # Unix socket server
        self._server: asyncio.AbstractServer | None = None

        # Daemon's own pending requests (for initialize, etc.)
        self._daemon_pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

        # Idle tracking
        self._idle_handle: asyncio.TimerHandle | None = None
        self._shutting_down = False

    # -- App-server lifecycle ---------------------------------------------

    async def _start_appserver(self) -> None:
        """Spawn the codex app-server subprocess."""
        log.info("Starting codex app-server subprocess")
        self._process = await asyncio.create_subprocess_exec(
            "codex",
            "app-server",
            env=self._env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._appserver_read_loop())
        self._stderr_task = asyncio.create_task(self._appserver_drain_stderr())
        await self._appserver_initialize()
        log.info("App-server ready (pid=%d)", self._process.pid or 0)

    async def _stop_appserver(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()
        if self._process and self._process.stdin:
            self._process.stdin.close()
        if self._process:
            await self._process.wait()
        self._process = None
        self._reader_task = None
        self._stderr_task = None
        log.info("App-server stopped")

    async def _appserver_initialize(self) -> None:
        """Send JSON-RPC initialize to the app-server."""
        resp = await self._appserver_request(
            "initialize",
            {
                "clientInfo": {"name": "agm-daemon", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
            timeout=30,
        )
        log.debug("Initialize response: %s", resp)

    async def _appserver_request(
        self,
        method: str,
        params: dict | None = None,
        timeout: float = 120,
    ) -> dict:
        """Send a request directly to app-server (daemon's own requests)."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("App-server not running")

        req_id = self._next_id
        self._next_id += 1

        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params

        future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        # Store with worker=None to indicate daemon's own request
        self._daemon_pending[req_id] = future

        self._process.stdin.write(_encode(msg))
        await self._process.stdin.drain()

        return await asyncio.wait_for(future, timeout=timeout)

    # -- Worker -> app-server forwarding ----------------------------------

    def _forward_to_appserver(
        self,
        worker: _Worker,
        worker_id: int | str,
        method: str,
        params: dict | None,
    ) -> None:
        """Remap worker request ID and forward to app-server."""
        if not self._process or not self._process.stdin:
            err = {"code": -1, "message": "App-server not running"}
            worker.send({"type": "response", "id": worker_id, "error": err})
            return

        daemon_id = self._next_id
        self._next_id += 1
        self._pending[daemon_id] = (worker, worker_id, method)

        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": daemon_id, "method": method}
        if params is not None:
            msg["params"] = params

        try:
            self._process.stdin.write(_encode(msg))
        except Exception:
            self._pending.pop(daemon_id, None)
            err = {"code": -1, "message": "Write failed"}
            worker.send({"type": "response", "id": worker_id, "error": err})

    # -- App-server -> worker routing -------------------------------------

    def _handle_appserver_response(self, msg_id: int | str, msg: dict) -> None:
        """Route response from app-server back to the requesting worker."""
        # Daemon always generates int IDs; narrow for dict lookups.
        if not isinstance(msg_id, int):
            return

        # Check if it's the daemon's own request
        daemon_future = self._daemon_pending.pop(msg_id, None)
        if daemon_future is not None:
            if "error" in msg:
                daemon_future.set_exception(RuntimeError(str(msg["error"])))
            else:
                daemon_future.set_result(msg.get("result", {}))
            return

        entry = self._pending.pop(msg_id, None)
        if entry is None:
            return
        worker, worker_id, method = entry

        # Auto-register threadId for thread/start and thread/resume
        if method in _THREAD_START_METHODS:
            result = msg.get("result", {})
            thread = result.get("thread", {})
            thread_id = thread.get("id")
            if thread_id:
                self._threads[thread_id] = worker
                worker.threads.add(thread_id)
                log.debug("Auto-registered thread %s -> worker %s", thread_id, worker.addr)

        # Remap ID back and send to worker
        if "error" in msg:
            worker.send({"type": "response", "id": worker_id, "error": msg["error"]})
        else:
            worker.send({"type": "response", "id": worker_id, "result": msg.get("result", {})})

    async def _handle_appserver_notification(self, method: str, params: dict) -> None:
        """Route notification to owning worker or broadcast."""
        thread_id = _extract_thread_id(method, params)

        if thread_id and thread_id in self._threads:
            worker = self._threads[thread_id]
            worker.send({"type": "notification", "method": method, "params": params})
        else:
            # Broadcast to all workers
            for worker in self._workers:
                worker.send({"type": "notification", "method": method, "params": params})

    async def _handle_appserver_server_request(
        self, msg_id: int | str, method: str, params: dict
    ) -> None:
        """Route server request to the owning worker by threadId."""
        thread_id = _extract_server_request_thread_id(params)
        worker = self._threads.get(thread_id) if thread_id else None
        if worker is None and thread_id is None and len(self._workers) == 1:
            worker = next(iter(self._workers))

        if worker is None:
            # No owner — reject
            log.warning("Server request %s with no thread owner (threadId=%s)", method, thread_id)
            err = {"code": -1, "message": "No thread owner"}
            await self._respond_to_appserver(msg_id, error=err)
            return

        # Forward to worker and wait for response
        worker.send({"type": "server_request", "id": msg_id, "method": method, "params": params})

    async def _respond_to_appserver(
        self,
        req_id: int | str,
        *,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        if not self._process or not self._process.stdin:
            return
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result or {}
        self._process.stdin.write(_encode(msg))
        await self._process.stdin.drain()

    # -- App-server read loop ---------------------------------------------

    async def _appserver_read_loop(self) -> None:
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                self._handle_appserver_eof()
                break
            msg = _decode(line)
            if msg is None:
                continue

            msg_id = msg.get("id")
            method = msg.get("method")

            if method is not None and msg_id is not None:
                await self._handle_appserver_server_request(msg_id, method, msg.get("params", {}))
            elif method is not None:
                await self._handle_appserver_notification(method, msg.get("params", {}))
            elif msg_id is not None:
                self._handle_appserver_response(msg_id, msg)

    async def _appserver_drain_stderr(self) -> None:
        assert self._process and self._process.stderr
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            log.debug("app-server stderr: %s", line.decode(errors="replace").rstrip())

    def _handle_appserver_eof(self) -> None:
        """App-server process exited — fail pending, notify workers, auto-restart."""
        log.error("App-server EOF — process exited")
        err = {"code": -1, "message": "App-server process exited"}
        for _daemon_id, (worker, worker_id, _method) in self._pending.items():
            worker.send({"type": "response", "id": worker_id, "error": err})
        self._pending.clear()

        for future in self._daemon_pending.values():
            if not future.done():
                future.set_exception(ConnectionError("App-server exited"))
        self._daemon_pending.clear()

        # Thread registrations are invalid after restart
        self._threads.clear()
        for worker in self._workers:
            worker.threads.clear()

        for worker in self._workers:
            worker.send({"type": "daemon_event", "event": "appserver_exited"})

        # Auto-restart if workers are still connected
        if self._workers and not self._shutting_down:
            asyncio.get_running_loop().create_task(self._auto_restart())

    async def _auto_restart(self) -> None:
        """Attempt to restart the app-server after an unexpected exit."""
        log.info("Auto-restarting app-server (workers still connected)")
        try:
            await self._start_appserver()
            for worker in self._workers:
                worker.send({"type": "daemon_event", "event": "appserver_restarted"})
        except Exception:
            log.exception("Failed to restart app-server — shutting down")
            self._shutting_down = True
            asyncio.get_running_loop().create_task(self.stop())

    # -- Worker connection handling ----------------------------------------

    async def _handle_worker(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        worker = _Worker(reader, writer)
        self._workers.add(worker)
        self._cancel_idle_timer()
        log.info("Worker connected: %s (total: %d)", worker.addr, len(self._workers))

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = _decode(line)
                if msg is None:
                    continue
                await self._dispatch_worker_message(worker, msg)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._disconnect_worker(worker)

    async def _dispatch_worker_message(self, worker: _Worker, msg: dict) -> None:
        msg_type = msg.get("type")

        if msg_type == "request":
            self._forward_to_appserver(
                worker,
                msg.get("id", 0),
                msg.get("method", ""),
                msg.get("params"),
            )
        elif msg_type == "server_response":
            # Worker responding to a server request we forwarded
            req_id = msg.get("id")
            if req_id is not None:
                if "error" in msg:
                    await self._respond_to_appserver(req_id, error=msg["error"])
                else:
                    await self._respond_to_appserver(req_id, result=msg.get("result", {}))
        else:
            log.warning("Unknown worker message type: %s", msg_type)

    def _disconnect_worker(self, worker: _Worker) -> None:
        self._workers.discard(worker)
        # Unregister all threads owned by this worker
        for thread_id in worker.threads:
            self._threads.pop(thread_id, None)
        # Fail pending requests from this worker
        to_remove = [daemon_id for daemon_id, (w, _, _) in self._pending.items() if w is worker]
        for daemon_id in to_remove:
            self._pending.pop(daemon_id, None)
        worker.close()
        log.info("Worker disconnected: %s (total: %d)", worker.addr, len(self._workers))
        self._maybe_start_idle_timer()

    # -- Idle timeout -----------------------------------------------------

    def _cancel_idle_timer(self) -> None:
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None

    def _maybe_start_idle_timer(self) -> None:
        if self._workers or self._threads or self._idle_timeout <= 0:
            return
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(self._idle_timeout, self._idle_expired)
        log.info("No workers — idle timeout in %ds", self._idle_timeout)

    def _idle_expired(self) -> None:
        if self._workers or self._threads:
            return  # Workers reconnected — cancel
        log.info("Idle timeout reached — shutting down")
        self._shutting_down = True
        asyncio.get_running_loop().create_task(self.stop())

    # -- Public API -------------------------------------------------------

    async def start(self) -> None:
        """Start the daemon: create socket, spawn app-server."""
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale socket
        if self._socket_path.exists():
            self._socket_path.unlink()

        await self._start_appserver()

        self._server = await asyncio.start_unix_server(
            self._handle_worker,
            path=str(self._socket_path),
        )
        # Make socket accessible to the user
        self._socket_path.chmod(0o600)

        # Write PID file
        pid_path = self._socket_path.with_suffix(".pid")
        pid_path.write_text(str(os.getpid()))

        log.info("Daemon listening on %s", self._socket_path)
        self._maybe_start_idle_timer()

    async def stop(self) -> None:
        """Stop the daemon: close socket, stop app-server."""
        self._cancel_idle_timer()
        # Close all worker connections so their read loops exit
        for worker in list(self._workers):
            worker.close()
        self._workers.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await self._stop_appserver()
        # Clean up socket and PID file
        if self._socket_path.exists():
            self._socket_path.unlink()
        pid_path = self._socket_path.with_suffix(".pid")
        if pid_path.exists():
            pid_path.unlink()
        log.info("Daemon stopped")

    async def serve_forever(self) -> None:
        """Run until shutdown signal or idle timeout."""
        assert self._server is not None
        stop_event = asyncio.Event()

        def on_signal() -> None:
            log.info("Signal received — shutting down")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, on_signal)

        await stop_event.wait()
        await self.stop()


# -- Entry point ----------------------------------------------------------


async def _main() -> None:
    from agm.paths import CODEX_HOME

    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CODEX_HOME": str(CODEX_HOME)}

    daemon = AppServerDaemon(env=env)
    await daemon.start()
    await daemon.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())


if __name__ == "__main__":
    main()
