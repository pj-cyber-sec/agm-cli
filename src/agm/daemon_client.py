"""Worker-side client for the shared app-server daemon.

Implements :class:`~agm.client.CodexClient` so it can be used as a
drop-in replacement for :class:`~agm.client.AppServerClient` in job code.

Communication uses newline-delimited JSON over a Unix domain socket.
The daemon handles multiplexing onto a single ``codex app-server``
subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agm.client import RPCError, ServerRequestHandler
from agm.daemon import DEFAULT_SOCKET_PATH, _encode

log = logging.getLogger(__name__)


class DaemonClient:
    """Worker-side client that talks to the app-server daemon.

    Satisfies the :class:`~agm.client.CodexClient` protocol — job code
    can use it interchangeably with :class:`~agm.client.AppServerClient`.
    """

    def __init__(self, *, socket_path: Path = DEFAULT_SOCKET_PATH) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notification_handlers: dict[str, list[Callable]] = {}
        self._server_request_handler: ServerRequestHandler | None = None

    async def start(self) -> None:
        """Connect to the daemon's Unix socket."""
        self._reader, self._writer = await asyncio.open_unix_connection(str(self._socket_path))
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Disconnect from the daemon."""
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._reader = None
        self._writer = None
        self._reader_task = None

    # -- CodexClient interface ------------------------------------------------

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = 120,
    ) -> dict[str, Any]:
        """Send a request through the daemon to the app-server."""
        if not self._writer:
            raise RuntimeError("DaemonClient not connected")

        req_id = self._next_id
        self._next_id += 1

        msg: dict[str, Any] = {
            "type": "request",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future

        try:
            self._writer.write(_encode(msg))
            await self._writer.drain()
        except Exception:
            self._pending.pop(req_id, None)
            raise

        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        except (TimeoutError, asyncio.CancelledError):
            self._pending.pop(req_id, None)
            raise

    def on_notification(self, method: str, handler: Callable) -> None:
        """Register a handler for a daemon-forwarded notification."""
        self._notification_handlers.setdefault(method, []).append(handler)

    def remove_notification_handler(self, method: str, handler: Callable) -> None:
        """Remove a previously registered notification handler."""
        handlers = self._notification_handlers.get(method, [])
        if handler in handlers:
            handlers.remove(handler)

    def set_server_request_handler(self, handler: ServerRequestHandler) -> None:
        """Register handler for server requests forwarded by the daemon."""
        self._server_request_handler = handler

    # -- Internal read loop ---------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                self._handle_eof()
                break
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict):
                continue
            await self._dispatch(msg)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")

        if msg_type == "response":
            self._handle_response(msg)
        elif msg_type == "notification":
            await self._handle_notification(msg)
        elif msg_type == "server_request":
            await self._handle_server_request(msg)
        elif msg_type == "daemon_event":
            self._handle_daemon_event(msg)

    def _handle_response(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        if not isinstance(msg_id, int) or msg_id not in self._pending:
            return
        future = self._pending.pop(msg_id)
        if future.done():
            return
        if "error" in msg:
            future.set_exception(RPCError(msg["error"]))
        else:
            future.set_result(msg.get("result", {}))

    async def _handle_notification(self, msg: dict[str, Any]) -> None:
        method = msg.get("method", "")
        params = msg.get("params", {})
        for handler in list(self._notification_handlers.get(method, [])):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(params)
                else:
                    handler(params)
            except Exception:
                log.exception("Notification handler error for %s", method)

    async def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if self._server_request_handler:
            try:
                result = await self._server_request_handler(method, params)
                await self._send_server_response(req_id, result=result)
            except Exception as e:
                log.exception("Server request handler error for %s", method)
                await self._send_server_response(req_id, error={"code": -1, "message": str(e)})
            return

        log.warning("No handler for server request: %s", method)
        await self._send_server_response(
            req_id, error={"code": -1, "message": f"No handler for {method}"}
        )

    async def _send_server_response(
        self,
        req_id: int | str | None,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if not self._writer:
            return
        msg: dict[str, Any] = {"type": "server_response", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result or {}
        self._writer.write(_encode(msg))
        await self._writer.drain()

    def _handle_daemon_event(self, msg: dict[str, Any]) -> None:
        event = msg.get("event")
        if event == "appserver_exited":
            log.error("App-server exited — failing pending requests")
            err = ConnectionError("App-server process exited")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(err)
            self._pending.clear()
        elif event == "appserver_restarted":
            log.info("App-server restarted by daemon")

    def _handle_eof(self) -> None:
        """Daemon disconnected — fail all pending requests."""
        err = ConnectionError("Daemon connection closed")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(err)
        self._pending.clear()

    # -- Context manager ------------------------------------------------------

    async def __aenter__(self) -> DaemonClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
