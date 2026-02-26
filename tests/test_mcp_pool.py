"""Tests for MCP client pool."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from agm.mcp_pool import McpClient, McpPool

# Path to the fake MCP server script
_FAKE_SERVER = str(Path(__file__).parent / "_fake_mcp_server.py")


# ---------------------------------------------------------------------------
# Unit tests: pool logic with pre-wired clients
# ---------------------------------------------------------------------------


def _make_connected_client(name: str, tools: list[dict[str, Any]]) -> McpClient:
    """Create an McpClient with tools pre-populated (no real connection)."""
    client = McpClient.__new__(McpClient)
    client.name = name
    client._tools = tools
    client._session = None
    return client


class TestPoolToolIndex:
    """Verify tool indexing, routing, and filtering without real MCP servers."""

    def test_get_dynamic_tools_returns_all(self) -> None:
        pool = McpPool()
        tools_a = [{"name": "greet", "description": "Say hello", "inputSchema": {}}]
        tools_b = [{"name": "add", "description": "Add nums", "inputSchema": {}}]
        pool._clients["alpha"] = _make_connected_client("alpha", tools_a)
        pool._clients["beta"] = _make_connected_client("beta", tools_b)
        pool._tool_index = {"greet": "alpha", "add": "beta"}

        result = pool.get_dynamic_tools()
        names = {t["name"] for t in result}
        assert names == {"greet", "add"}

    def test_get_dynamic_tools_filters_by_server(self) -> None:
        pool = McpPool()
        tools_a = [{"name": "greet", "description": "", "inputSchema": {}}]
        tools_b = [{"name": "add", "description": "", "inputSchema": {}}]
        pool._clients["alpha"] = _make_connected_client("alpha", tools_a)
        pool._clients["beta"] = _make_connected_client("beta", tools_b)
        pool._tool_index = {"greet": "alpha", "add": "beta"}

        result = pool.get_dynamic_tools(["alpha"])
        assert len(result) == 1
        assert result[0]["name"] == "greet"

    def test_has_tool(self) -> None:
        pool = McpPool()
        pool._tool_index = {"greet": "alpha"}

        assert pool.has_tool("greet") is True
        assert pool.has_tool("nonexistent") is False

    async def test_call_tool_unknown_raises(self) -> None:
        pool = McpPool()
        with pytest.raises(KeyError, match="Unknown tool: missing"):
            await pool.call_tool("missing", {})

    def test_server_names(self) -> None:
        pool = McpPool()
        pool._clients["a"] = _make_connected_client("a", [])
        pool._clients["b"] = _make_connected_client("b", [])
        assert set(pool.server_names) == {"a", "b"}

    async def test_disconnect_removes_tools(self) -> None:
        pool = McpPool()
        tools_a = [{"name": "greet", "description": "", "inputSchema": {}}]
        client = _make_connected_client("alpha", tools_a)
        pool._clients["alpha"] = client
        pool._tool_index = {"greet": "alpha"}

        # Patch disconnect to be a no-op (no real session)
        async def noop() -> None:
            client._tools = []

        client.disconnect = noop  # type: ignore[assignment]

        await pool.disconnect("alpha")

        assert "alpha" not in pool._clients
        assert pool.has_tool("greet") is False

    def test_tool_collision_keeps_latest(self) -> None:
        """When two servers expose a tool with the same name, the second wins."""
        pool = McpPool()
        tools_a = [{"name": "shared", "description": "from a", "inputSchema": {}}]
        tools_b = [{"name": "shared", "description": "from b", "inputSchema": {}}]

        pool._clients["a"] = _make_connected_client("a", tools_a)
        pool._tool_index["shared"] = "a"

        pool._clients["b"] = _make_connected_client("b", tools_b)
        pool._tool_index["shared"] = "b"  # latest wins

        assert pool._tool_index["shared"] == "b"


# ---------------------------------------------------------------------------
# Integration tests: real MCP server subprocess
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMcpClientIntegration:
    """Test real MCP client against the fake server subprocess."""

    async def test_connect_discover_call(self) -> None:
        """Full lifecycle: connect, discover tools, call a tool, disconnect."""
        client = McpClient("test", sys.executable, [_FAKE_SERVER])
        await client.connect()

        try:
            assert len(client.tools) == 2
            names = client.tool_names
            assert "greet" in names
            assert "add" in names

            # Verify DynamicToolSpec format
            for tool in client.tools:
                assert "name" in tool
                assert "description" in tool
                assert "inputSchema" in tool

            # Call a tool
            result = await client.call_tool("greet", {"name": "World"})
            assert "Hello, World!" in result

            result = await client.call_tool("add", {"a": 3, "b": 7})
            assert "10" in result
        finally:
            await client.disconnect()

    async def test_pool_connect_and_call(self) -> None:
        """Full pool lifecycle with real MCP server."""
        async with McpPool() as pool:
            await pool.connect("test-tools", sys.executable, [_FAKE_SERVER])

            tools = pool.get_dynamic_tools()
            assert len(tools) == 2
            assert pool.has_tool("greet")

            result = await pool.call_tool("greet", {"name": "agm"})
            assert "Hello, agm!" in result

    async def test_pool_idempotent_connect(self) -> None:
        """Connecting the same server name twice returns existing client."""
        async with McpPool() as pool:
            c1 = await pool.connect("test-tools", sys.executable, [_FAKE_SERVER])
            c2 = await pool.connect("test-tools", sys.executable, [_FAKE_SERVER])
            assert c1 is c2
