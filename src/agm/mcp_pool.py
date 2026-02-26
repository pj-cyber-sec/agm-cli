"""MCP client pool: spawn MCP servers on demand, discover tools, route calls.

The pool manages long-lived connections to MCP servers over stdio.
Each server is identified by a name and spawned from a command + args pair.
Tools are discovered at connect time and cached.  Tool calls are routed
to the owning server by name.

Usage::

    pool = McpPool()
    await pool.connect("sqlite", "npx", ["-y", "@anthropic/mcp-server-sqlite"])
    tools = pool.get_dynamic_tools(["sqlite"])  # DynamicToolSpec list
    result = await pool.call_tool("list_tables", {"database": "app.db"})
    await pool.close()
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

log = logging.getLogger(__name__)


class McpClient:
    """Single MCP server connection over stdio."""

    def __init__(self, name: str, command: str, args: list[str]) -> None:
        self.name = name
        self._server_params = StdioServerParameters(command=command, args=args)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Spawn the MCP server process and initialize the session."""
        read, write = await self._stack.enter_async_context(stdio_client(self._server_params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        await self._discover_tools()
        log.info("MCP server '%s' connected (%d tools)", self.name, len(self._tools))

    async def _discover_tools(self) -> None:
        assert self._session is not None
        result = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return discovered tools in DynamicToolSpec format."""
        return list(self._tools)

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(t["name"] for t in self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the text result."""
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    async def disconnect(self) -> None:
        await self._stack.aclose()
        self._session = None
        self._tools = []


class McpPool:
    """Pool of MCP server connections.

    Provides a unified interface for discovering tools across multiple
    MCP servers and routing tool calls to the correct server.
    """

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        # tool_name -> server_name for call routing
        self._tool_index: dict[str, str] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._clients)

    async def connect(self, name: str, command: str, args: list[str]) -> McpClient:
        """Connect to an MCP server.  Returns the client for inspection.

        If a server with this name is already connected, returns the
        existing client without reconnecting.
        """
        if name in self._clients:
            return self._clients[name]

        client = McpClient(name, command, args)
        await client.connect()

        self._clients[name] = client
        for tool_name in client.tool_names:
            if tool_name in self._tool_index:
                log.warning(
                    "Tool name collision: '%s' from '%s' shadows '%s'",
                    tool_name,
                    name,
                    self._tool_index[tool_name],
                )
            self._tool_index[tool_name] = name

        return client

    async def disconnect(self, name: str) -> None:
        """Disconnect a specific server."""
        client = self._clients.pop(name, None)
        if client is None:
            return
        # Remove tool index entries for this server
        self._tool_index = {k: v for k, v in self._tool_index.items() if v != name}
        await client.disconnect()

    def get_dynamic_tools(self, server_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return DynamicToolSpec list for the given servers (or all)."""
        tools: list[dict[str, Any]] = []
        for name, client in self._clients.items():
            if server_names is not None and name not in server_names:
                continue
            tools.extend(client.tools)
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the owning MCP server."""
        server_name = self._tool_index.get(name)
        if server_name is None:
            raise KeyError(f"Unknown tool: {name}")
        client = self._clients[server_name]
        return await client.call_tool(name, arguments)

    def has_tool(self, name: str) -> bool:
        return name in self._tool_index

    async def close(self) -> None:
        """Disconnect all servers."""
        errors: list[Exception] = []
        for client in list(self._clients.values()):
            try:
                await client.disconnect()
            except Exception as e:
                errors.append(e)
                log.warning("Error disconnecting MCP server '%s': %s", client.name, e)
        self._clients.clear()
        self._tool_index.clear()
        if errors:
            log.warning("%d errors during pool shutdown", len(errors))

    async def __aenter__(self) -> McpPool:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
