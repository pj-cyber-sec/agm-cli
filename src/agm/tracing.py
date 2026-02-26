"""Rich execution tracing for the agm pipeline.

Captures structured trace events during Codex turns and writes them to
SQLite for durable, queryable execution history. Every pipeline stage
(enrichment, exploration, planning, task creation, execution, review)
can produce trace events.
"""

from __future__ import annotations

import logging
import sqlite3

from agm.db import add_trace_event

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Truncation limits per field type
# ---------------------------------------------------------------------------

MAX_TRACE_STDOUT = 50_000
MAX_TRACE_FILE_CONTENT = 100_000
MAX_TRACE_TOOL_RESULT = 50_000
MAX_TRACE_REASONING = 10_000
MAX_TRACE_AGENT_MESSAGE = 50_000


def _truncate(text: str, limit: int) -> str:
    """Truncate text with a marker if over *limit* characters."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated from {len(text)} chars]"


def _next_trace_ordinal(conn: sqlite3.Connection, *, entity_type: str, entity_id: str) -> int:
    """Return the next ordinal for an entity's trace stream."""
    max_ordinal = _current_trace_ordinal(conn, entity_type=entity_type, entity_id=entity_id)
    return max_ordinal + 1


def _current_trace_ordinal(conn: sqlite3.Connection, *, entity_type: str, entity_id: str) -> int:
    """Return the current max ordinal for an entity's trace stream."""
    row = conn.execute(
        "SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal "
        "FROM trace_events WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    ).fetchone()
    if not row:
        return -1
    max_ordinal = row["max_ordinal"]
    if not isinstance(max_ordinal, int):
        return -1
    return max_ordinal


# ---------------------------------------------------------------------------
# TraceContext — one per Codex turn, writes trace events to SQLite
# ---------------------------------------------------------------------------


class TraceContext:
    """Captures structured trace events during a Codex turn.

    Best-effort: ``record()`` never raises. A trace write failure
    should never block the agent.
    """

    __slots__ = (
        "entity_type",
        "entity_id",
        "stage",
        "plan_id",
        "project",
        "_ordinal",
        "_turn_index",
        "_conn",
    )

    def __init__(
        self,
        *,
        entity_type: str,
        entity_id: str,
        stage: str,
        plan_id: str | None,
        project: str,
        conn: sqlite3.Connection,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.stage = stage
        self.plan_id = plan_id
        self.project = project
        self._ordinal = _next_trace_ordinal(conn, entity_type=entity_type, entity_id=entity_id)
        self._turn_index = 0
        self._conn = conn

    def set_turn_index(self, index: int) -> None:
        """Set the current turn index for subsequent events."""
        self._turn_index = index

    def record(self, event_type: str, status: str | None, data: dict) -> None:
        """Insert a trace event. Best-effort — catches and logs exceptions."""
        try:
            max_ordinal = _current_trace_ordinal(
                self._conn, entity_type=self.entity_type, entity_id=self.entity_id
            )
            if self._ordinal <= max_ordinal:
                self._ordinal = max_ordinal + 1
            add_trace_event(
                self._conn,
                entity_type=self.entity_type,
                entity_id=self.entity_id,
                stage=self.stage,
                turn_index=self._turn_index,
                ordinal=self._ordinal,
                event_type=event_type,
                status=status,
                data=data,
            )
            self._ordinal += 1
        except Exception:
            log.debug(
                "Failed to record trace event %s for %s:%s",
                event_type,
                self.entity_type,
                self.entity_id,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Per-event-type extractors
# ---------------------------------------------------------------------------


def _extract_command_execution(item: dict) -> dict:
    data: dict = {"command": item.get("command", "")}
    if item.get("exitCode") is not None:
        data["exit_code"] = item["exitCode"]
    if item.get("durationMs") is not None:
        data["duration_ms"] = item["durationMs"]
    stdout = item.get("stdout", "")
    if stdout:
        data["stdout"] = _truncate(stdout, MAX_TRACE_STDOUT)
    stderr = item.get("stderr", "")
    if stderr:
        data["stderr"] = _truncate(stderr, MAX_TRACE_STDOUT)
    return data


def _extract_file_read(item: dict) -> dict:
    data: dict = {"path": item.get("path", "")}
    content = item.get("content", "")
    if content:
        data["content"] = _truncate(content, MAX_TRACE_FILE_CONTENT)
    return data


def _extract_file_mutation(item: dict) -> dict:
    data: dict = {"path": item.get("path", "")}
    content = item.get("content", "")
    if content:
        data["content"] = _truncate(content, MAX_TRACE_FILE_CONTENT)
    diff = item.get("diff", "")
    if diff:
        data["diff"] = _truncate(diff, MAX_TRACE_FILE_CONTENT)
    return data


def _extract_file_change(item: dict) -> dict:
    files = []
    for c in item.get("changes", []):
        if not isinstance(c, dict):
            continue
        entry: dict = {"path": c.get("path", "")}
        if c.get("kind"):
            entry["kind"] = c["kind"]
        diff = c.get("diff", "")
        if diff:
            entry["diff"] = _truncate(diff, MAX_TRACE_FILE_CONTENT)
        files.append(entry)
    return {"files": files}


def _extract_mcp_tool_call(item: dict) -> dict:
    data: dict = {"server": item.get("server", ""), "tool": item.get("tool", "")}
    arguments = item.get("arguments")
    if arguments is not None:
        arg_str = str(arguments) if not isinstance(arguments, str) else arguments
        data["arguments"] = _truncate(arg_str, MAX_TRACE_TOOL_RESULT)
    result = item.get("result")
    if result is not None:
        result_str = str(result) if not isinstance(result, str) else result
        data["result"] = _truncate(result_str, MAX_TRACE_TOOL_RESULT)
    return data


def _extract_web_search(item: dict) -> dict:
    data: dict = {"query": item.get("query", "")}
    results = item.get("results")
    if results is not None:
        result_str = str(results) if not isinstance(results, str) else results
        data["results"] = _truncate(result_str, MAX_TRACE_TOOL_RESULT)
    return data


def _extract_reasoning(item: dict) -> dict:
    summaries = item.get("summary", [])
    if summaries:
        text = " ".join(str(s) for s in summaries)
        return {"text": _truncate(text, MAX_TRACE_REASONING)}
    return {}


def _extract_agent_message(item: dict) -> dict:
    text = item.get("text", "")
    if text:
        return {"text": _truncate(text, MAX_TRACE_AGENT_MESSAGE)}
    return {}


def _extract_plan(item: dict) -> dict:
    steps = item.get("steps", [])
    if not steps:
        text = item.get("text", "")
        if text:
            return {"text": _truncate(text, MAX_TRACE_AGENT_MESSAGE)}
        return {}
    return {
        "steps": [
            {"step": s.get("step", ""), "status": s.get("status", "")}
            for s in steps
            if isinstance(s, dict)
        ]
    }


_EXTRACTORS: dict[str, object] = {
    "commandExecution": _extract_command_execution,
    "fileReadTool": _extract_file_read,
    "fileEditTool": _extract_file_mutation,
    "fileWriteTool": _extract_file_mutation,
    "fileChange": _extract_file_change,
    "mcpToolCall": _extract_mcp_tool_call,
    "webSearch": _extract_web_search,
    "reasoning": _extract_reasoning,
    "agentMessage": _extract_agent_message,
    "plan": _extract_plan,
}


# ---------------------------------------------------------------------------
# Rich data extraction from Codex ThreadItems
# ---------------------------------------------------------------------------


def extract_rich_trace(item: dict) -> tuple[str, dict]:
    """Extract rich data from a Codex ThreadItem for SQLite storage.

    Returns ``(event_type, data_dict)`` with truncation applied.
    This is the rich counterpart to ``_extract_item_summary()`` in
    jobs_common.py — it captures full content instead of minimal summaries.
    """
    event_type = item.get("type", "unknown")
    extractor = _EXTRACTORS.get(event_type)
    if extractor is not None:
        return event_type, extractor(item)  # type: ignore[operator]
    return event_type, {}
