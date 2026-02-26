"""Tests for the JSON-RPC client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agm.client import AppServerClient, RPCError


class FakeStdout:
    """Simulates subprocess stdout that delivers lines on demand."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, msg: dict) -> None:
        self._queue.put_nowait(json.dumps(msg).encode() + b"\n")

    def close(self) -> None:
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


def make_mock_process(stdout: FakeStdout) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()
    proc.wait = AsyncMock()
    proc.stdout = stdout
    # stderr needs an async readline that returns EOF immediately
    fake_stderr = FakeStdout()
    fake_stderr.close()  # enqueue EOF so _drain_stderr exits
    proc.stderr = fake_stderr
    return proc


@pytest.fixture
def fake_server():
    """Provide a fake stdout and a patched subprocess."""
    stdout = FakeStdout()
    proc = make_mock_process(stdout)

    # Auto-respond to initialize
    stdout.feed({"jsonrpc": "2.0", "id": 1, "result": {"userAgent": "codex/test"}})

    return stdout, proc


@pytest.mark.asyncio
async def test_request_response(fake_server):
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"data": []}})
            result = await client.request("thread/list", {})
            assert result == {"data": []}


@pytest.mark.asyncio
async def test_rpc_error(fake_server):
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            stdout.feed(
                {"jsonrpc": "2.0", "id": 2, "error": {"code": -1, "message": "bad request"}}
            )
            with pytest.raises(RPCError, match="bad request"):
                await client.request("bogus/method", {})


@pytest.mark.asyncio
async def test_notifications_dispatched(fake_server):
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            received = {}

            def handler(params):
                received.update(params)

            client.on_notification("turn/started", handler)

            # Notification triggers handler; request still works
            stdout.feed({"jsonrpc": "2.0", "method": "turn/started", "params": {"turnId": "t1"}})
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            result = await client.request("some/method", {})
            assert result == {"ok": True}
            # Give event loop a tick so notification handler runs
            await asyncio.sleep(0)
            assert received.get("turnId") == "t1"


@pytest.mark.asyncio
async def test_notification_handler_removal(fake_server):
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            calls = []

            def handler(params):
                calls.append(params)

            client.on_notification("turn/completed", handler)
            client.remove_notification_handler("turn/completed", handler)

            stdout.feed({"jsonrpc": "2.0", "method": "turn/completed", "params": {"x": 1}})
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {}})
            await client.request("ping", {})
            await asyncio.sleep(0)
            assert len(calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "params", "response"),
    [
        ("item/commandExecution/requestApproval", {"command": "ls"}, {"decision": "accept"}),
        (
            "execCommandApproval",
            {
                "conversationId": "thread-1",
                "callId": "call-1",
                "command": ["bash", "-lc", "echo hi"],
                "cwd": "/tmp",
                "parsedCmd": [],
            },
            {"decision": "approved"},
        ),
        (
            "applyPatchApproval",
            {"conversationId": "thread-1", "callId": "patch-1", "fileChanges": {}},
            {"decision": "approved"},
        ),
        (
            "item/tool/requestUserInput",
            {"threadId": "thread-1", "turnId": "turn-1", "itemId": "item-1", "questions": []},
            {"answers": {}},
        ),
        (
            "account/chatgptAuthTokens/refresh",
            {"reason": "expired"},
            {"accessToken": "tok-1", "chatgptAccountId": "acct-1"},
        ),
    ],
)
async def test_server_request_handled(fake_server, method, params, response):
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:

            async def handler(req_method, _params):
                assert req_method == method
                return response

            client.set_server_request_handler(handler)

            # Server sends a request (has method AND id, but not in our pending)
            stdout.feed(
                {
                    "jsonrpc": "2.0",
                    "id": 999,
                    "method": method,
                    "params": params,
                }
            )
            # Give the read loop time to process and respond
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            result = await client.request("ping", {})
            assert result == {"ok": True}

            # Check that a response was written back for the server request
            written_lines = [call.args[0] for call in proc.stdin.write.call_args_list]
            # Find the response to id 999
            for line in written_lines:
                msg = json.loads(line)
                if msg.get("id") == 999:
                    assert msg["result"] == response
                    break
            else:
                pytest.fail("No response sent for server request id 999")


@pytest.mark.asyncio
async def test_eof_fails_pending_futures(fake_server):
    """When stdout closes (EOF), all pending futures should fail with ConnectionError."""
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            # Send a request but don't feed a response — close stdout instead
            future = asyncio.ensure_future(client.request("some/method", {}))
            # Give the request time to register
            await asyncio.sleep(0)
            stdout.close()
            with pytest.raises(ConnectionError, match="connection closed"):
                await future


@pytest.mark.asyncio
async def test_server_request_id_collision(fake_server):
    """Server request with same ID as a client request should be classified
    as a server request (has method), not a response."""
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            received_requests = []

            async def handler(method, params):
                received_requests.append(method)
                return {"decision": "accept"}

            client.set_server_request_handler(handler)

            # Send a client request (will get id=2 after initialize)
            # Then inject a server request with the SAME id=2
            stdout.feed(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "item/fileChange/requestApproval",
                    "params": {"path": "/tmp/test"},
                }
            )
            # Now feed the actual response for our request (id=2)
            # Since the server request consumed id=2 via method-first classification,
            # we need to feed another response
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            await client.request("ping", {})
            # Give time for handler
            await asyncio.sleep(0)
            assert "item/fileChange/requestApproval" in received_requests


@pytest.mark.asyncio
async def test_future_registered_before_write(fake_server):
    """Immediate response during write should still resolve the request."""
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            original_write = proc.stdin.write

            def write_and_immediately_respond(data):
                msg = json.loads(data)
                if msg.get("method") == "test/method":
                    stdout.feed({"jsonrpc": "2.0", "id": msg["id"], "result": {"data": True}})
                return original_write(data)

            proc.stdin.write = MagicMock(side_effect=write_and_immediately_respond)

            result = await client.request("test/method", {})
            assert result == {"data": True}


@pytest.mark.asyncio
async def test_request_timeout(fake_server):
    """Request should raise TimeoutError if server doesn't respond in time."""
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            # Don't feed a response — request should time out
            with pytest.raises(TimeoutError):
                await client.request("test/slow", {}, timeout=0.1)


@pytest.mark.asyncio
async def test_request_timeout_cleans_up_pending(fake_server):
    """Late response for a timed-out request must not satisfy a later request."""
    stdout, proc = fake_server

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            original_write = proc.stdin.write
            request_ids: dict[str, int] = {}

            def record_request_ids(data):
                msg = json.loads(data)
                method = msg.get("method")
                if method is not None:
                    request_ids[method] = msg["id"]
                return original_write(data)

            proc.stdin.write = MagicMock(side_effect=record_request_ids)

            with pytest.raises(TimeoutError):
                await client.request("test/slow", {}, timeout=0.1)

            next_request = asyncio.create_task(client.request("test/next", {}, timeout=1))
            await asyncio.sleep(0)

            # Late response for the timed-out request should be ignored.
            stdout.feed(
                {"jsonrpc": "2.0", "id": request_ids["test/slow"], "result": {"late": True}}
            )
            await asyncio.sleep(0)
            assert not next_request.done()

            stdout.feed({"jsonrpc": "2.0", "id": request_ids["test/next"], "result": {"ok": True}})
            result = await next_request
            assert result == {"ok": True}


@pytest.mark.asyncio
async def test_env_passed_to_subprocess(fake_server):
    """When env is provided, it should be forwarded to create_subprocess_exec."""
    stdout, proc = fake_server
    custom_env = {"PATH": "/usr/bin", "CODEX_HOME": "/tmp/test-codex"}

    mock_exec = AsyncMock(return_value=proc)
    with patch("agm.client.asyncio.create_subprocess_exec", mock_exec):
        async with AppServerClient(env=custom_env) as client:
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            result = await client.request("test/method")
            assert result == {"ok": True}

    mock_exec.assert_called_once()
    assert mock_exec.call_args.kwargs["env"] is custom_env


@pytest.mark.asyncio
async def test_env_none_by_default(fake_server):
    """When no env is provided, None is passed (inherits parent environment)."""
    stdout, proc = fake_server

    mock_exec = AsyncMock(return_value=proc)
    with patch("agm.client.asyncio.create_subprocess_exec", mock_exec):
        async with AppServerClient() as client:
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            await client.request("test/method")

    assert mock_exec.call_args.kwargs["env"] is None


@pytest.mark.asyncio
async def test_stderr_drain(fake_server):
    """stderr should be drained without blocking the client."""
    stdout, proc = fake_server

    # Replace stderr with one that has data
    fake_stderr = FakeStdout()
    fake_stderr._queue.put_nowait(b"some debug output\n")
    fake_stderr.close()  # EOF after the line
    proc.stderr = fake_stderr

    with patch("agm.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        async with AppServerClient() as client:
            stdout.feed({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
            result = await client.request("test/method")
            assert result == {"ok": True}
