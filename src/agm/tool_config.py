"""Per-project MCP tool configuration.

Projects declare available MCP servers and which job types use them
in ``.agm/tools.toml``::

    [servers.context7]
    command = "npx"
    args = ["-y", "@upstash/context7-mcp"]

    [servers.sqlite]
    command = "npx"
    args = ["-y", "@anthropic/mcp-server-sqlite"]

    [tools]
    task_execution = ["context7", "sqlite"]
    review = ["context7"]

Only job types listed under ``[tools]`` get MCP tools injected.
Unlisted job types run without tools (current behavior).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_tool_config(project_dir: str | None) -> dict[str, Any] | None:
    """Load ``.agm/tools.toml`` from a project directory.

    Returns the parsed TOML dict, or None if the file doesn't exist
    or the project dir is unknown.
    """
    if not project_dir:
        return None

    path = Path(project_dir) / ".agm" / "tools.toml"
    if not path.exists():
        return None

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        log.warning("Failed to parse %s", path, exc_info=True)
        return None


def get_servers_for_job_type(
    config: dict[str, Any], job_type: str
) -> list[tuple[str, str, list[str]]]:
    """Return (name, command, args) tuples for MCP servers needed by a job type.

    Reads the ``[tools]`` section to find which server names are mapped
    to the given job type, then resolves each name from ``[servers]``.
    """
    tools_section = config.get("tools", {})
    server_names = tools_section.get(job_type, [])
    if not server_names:
        return []

    servers_section = config.get("servers", {})
    result: list[tuple[str, str, list[str]]] = []
    for name in server_names:
        server = servers_section.get(name)
        if server is None:
            log.warning("tools.toml: server '%s' referenced but not defined", name)
            continue
        command = server.get("command", "")
        args = server.get("args", [])
        if not command:
            log.warning("tools.toml: server '%s' has no command", name)
            continue
        result.append((name, command, args))

    return result
