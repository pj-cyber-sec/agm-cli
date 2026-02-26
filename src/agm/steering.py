"""Helpers for routing mid-turn steer requests to Codex threads."""

from __future__ import annotations

from typing import Any


def default_executor_recipient(task_id: str) -> str:
    """Return canonical recipient reference for a task executor."""
    return f"executor:{task_id[:8]}"


async def steer_active_turn(
    *,
    thread_id: str,
    active_turn_id: str,
    content: str,
    timeout: float = 30,
) -> dict[str, Any]:
    """Send turn/steer via daemon to an active thread turn."""
    from agm.daemon_client import DaemonClient

    async with DaemonClient() as client:
        return await client.request(
            "turn/steer",
            {
                "threadId": thread_id,
                "expectedTurnId": active_turn_id,
                "input": [{"type": "text", "text": content}],
            },
            timeout=timeout,
        )
