"""JSON-RPC client for the Codex app-server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Type alias for the server request handler signature.
ServerRequestHandler = Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


@runtime_checkable
class CodexClient(Protocol):
    """Structural interface for Codex app-server clients.

    Both :class:`AppServerClient` (direct subprocess) and
    :class:`~agm.daemon_client.DaemonClient` (shared daemon) implement
    this interface.  Job code uses ``CodexClient`` for type annotations
    so it works with either backend.
    """

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = 120,
    ) -> dict[str, Any]: ...

    def on_notification(self, method: str, handler: Callable) -> None: ...

    def remove_notification_handler(self, method: str, handler: Callable) -> None: ...

    def set_server_request_handler(self, handler: ServerRequestHandler) -> None: ...

    async def __aenter__(self) -> CodexClient: ...

    async def __aexit__(self, *exc: object) -> None: ...


class AppServerClient:
    """Manages a codex app-server subprocess and communicates via JSON-RPC over stdio.

    Handles three types of messages from the server:
    - Responses: matched to pending requests by ID
    - Notifications: dispatched to registered handlers (no response needed)
    - Server requests: dispatched to the server request handler (response sent back)
    """

    def __init__(self, *, env: dict[str, str] | None = None) -> None:
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._notification_handlers: dict[str, list[Callable]] = {}
        self._server_request_handler: ServerRequestHandler | None = None

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            "codex",
            "app-server",
            env=self._env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,  # 10MB — codex responses can be large
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._initialize()

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            await self._process.wait()

    # -- Notification handling --

    def on_notification(self, method: str, handler: Callable) -> None:
        """Register a handler for a server notification method."""
        self._notification_handlers.setdefault(method, []).append(handler)

    def remove_notification_handler(self, method: str, handler: Callable) -> None:
        """Remove a previously registered notification handler."""
        handlers = self._notification_handlers.get(method, [])
        if handler in handlers:
            handlers.remove(handler)

    # -- Server request handling --

    def set_server_request_handler(self, handler: ServerRequestHandler) -> None:
        """Register handler for server requests (JSON-RPC requests FROM the server)."""
        self._server_request_handler = handler

    # -- Outgoing messages --

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = 120,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response.

        Args:
            timeout: Seconds to wait for a response. None = no limit.
                     Default 120s prevents hangs if app-server stalls.
        """
        if not self._process or not self._process.stdin:
            raise RuntimeError("Client not started")

        req_id = self._next_id
        self._next_id += 1

        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params

        # Register future BEFORE writing to stdin to avoid race where
        # response arrives before _pending is populated.
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        line = json.dumps(msg) + "\n"
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except Exception:
            self._pending.pop(req_id, None)
            raise

        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        except (TimeoutError, asyncio.CancelledError):
            # Clean up so a late response doesn't hit a cancelled future
            self._pending.pop(req_id, None)
            raise

    async def respond(self, req_id: int | str, result: dict[str, Any]) -> None:
        """Send a JSON-RPC response to a server request."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Client not started")
        msg = {"jsonrpc": "2.0", "id": req_id, "result": result}
        line = json.dumps(msg) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    async def respond_error(self, req_id: int | str, code: int, message: str) -> None:
        """Send a JSON-RPC error response to a server request."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Client not started")
        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }
        line = json.dumps(msg) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    # -- Internal --

    async def _initialize(self) -> None:
        from agm import __version__

        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "agm", "version": __version__},
                "capabilities": {"experimentalApi": True},
            },
        )

    async def _drain_stderr(self) -> None:
        """Read and log stderr to prevent OS pipe buffer from filling."""
        assert self._process and self._process.stderr
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            log.debug("app-server stderr: %s", line.decode(errors="replace").rstrip())

    def _handle_read_eof(self) -> None:
        """Fail all pending futures when app-server stdout closes."""
        err = ConnectionError("App-server connection closed")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(err)
        self._pending.clear()

    @staticmethod
    def _decode_json_line(line: bytes) -> dict[str, Any] | None:
        """Decode one JSON-RPC line, returning None for malformed JSON."""
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _handle_server_request(
        self,
        msg_id: int | str,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Handle JSON-RPC request sent from app-server."""
        if self._server_request_handler:
            try:
                result = await self._server_request_handler(method, params)
                await self.respond(msg_id, result)
            except Exception as e:
                log.exception("Server request handler error for %s", method)
                await self.respond_error(msg_id, -1, str(e))
            return
        log.warning("No handler for server request: %s", method)
        await self.respond_error(msg_id, -1, f"No handler for {method}")

    async def _dispatch_notification(self, method: str, params: dict[str, Any]) -> None:
        """Dispatch JSON-RPC notification handlers."""
        for handler in list(self._notification_handlers.get(method, [])):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(params)
                else:
                    handler(params)
            except Exception:
                log.exception("Notification handler error for %s", method)

    def _handle_response(self, msg_id: int | str, msg: dict[str, Any]) -> None:
        """Resolve pending client request future from response message."""
        if msg_id not in self._pending:
            return
        future = self._pending.pop(msg_id)
        if future.done():
            return  # Late response after timeout — discard
        if "error" in msg:
            future.set_exception(RPCError(msg["error"]))
            return
        future.set_result(msg.get("result", {}))

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                self._handle_read_eof()
                break
            msg = self._decode_json_line(line)
            if msg is None:
                continue

            msg_id = msg.get("id")
            method = msg.get("method")

            # Classify by JSON-RPC shape, not pending-ID membership.
            # This prevents mis-routing if server request IDs collide
            # with client request IDs.
            if method is not None and msg_id is not None:
                # Server request (has both method and id)
                await self._handle_server_request(msg_id, method, msg.get("params", {}))
            elif method is not None:
                # Notification (has method, no id)
                await self._dispatch_notification(method, msg.get("params", {}))
            elif msg_id is not None:
                # Response to our request (has result/error, no method)
                self._handle_response(msg_id, msg)

    async def __aenter__(self) -> AppServerClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


class RPCError(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        self.code = error.get("code", -1)
        self.data = error.get("data")
        super().__init__(error.get("message", "Unknown RPC error"))
