"""Minimal MCP server for testing.  Run with: python tests/_fake_mcp_server.py"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("test-tools")


@server.tool()
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


if __name__ == "__main__":
    server.run(transport="stdio")
