"""Tests for rich execution tracing (agm.tracing + agm.db trace helpers)."""

import sqlite3
import tempfile
from pathlib import Path

from agm.db import (
    SCHEMA_VERSION,
    add_trace_event,
    count_trace_events,
    get_connection,
    get_trace_summary,
    list_trace_events,
    purge_data,
    purge_preview_counts,
)
from agm.tracing import (
    MAX_TRACE_REASONING,
    MAX_TRACE_STDOUT,
    TraceContext,
    _truncate,
    extract_rich_trace,
)


def tmp_conn():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    return get_connection(db_path)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_schema_version_is_32():
    assert SCHEMA_VERSION == 32


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


def test_truncate_short_text_unchanged():
    assert _truncate("hello", 100) == "hello"


def test_truncate_at_exact_limit():
    text = "x" * 50
    assert _truncate(text, 50) == text


def test_truncate_over_limit_adds_marker():
    text = "x" * 200
    result = _truncate(text, 100)
    assert len(result) > 100  # marker appended
    assert result.startswith("x" * 100)
    assert "truncated from 200 chars" in result


def test_truncate_empty_string():
    assert _truncate("", 10) == ""


# ---------------------------------------------------------------------------
# extract_rich_trace — each event type
# ---------------------------------------------------------------------------


def test_extract_command_execution():
    item = {
        "type": "commandExecution",
        "command": "pytest -v",
        "exitCode": 0,
        "durationMs": 1234,
        "stdout": "PASSED",
        "stderr": "",
    }
    event_type, data = extract_rich_trace(item)
    assert event_type == "commandExecution"
    assert data["command"] == "pytest -v"
    assert data["exit_code"] == 0
    assert data["duration_ms"] == 1234
    assert data["stdout"] == "PASSED"
    assert "stderr" not in data  # empty stderr omitted


def test_extract_command_execution_truncates_stdout():
    item = {
        "type": "commandExecution",
        "command": "cat big.log",
        "stdout": "x" * (MAX_TRACE_STDOUT + 100),
    }
    _, data = extract_rich_trace(item)
    assert "truncated" in data["stdout"]
    assert data["stdout"].startswith("x" * MAX_TRACE_STDOUT)


def test_extract_file_read():
    item = {"type": "fileReadTool", "path": "src/foo.py", "content": "import os"}
    event_type, data = extract_rich_trace(item)
    assert event_type == "fileReadTool"
    assert data["path"] == "src/foo.py"
    assert data["content"] == "import os"


def test_extract_file_edit():
    item = {
        "type": "fileEditTool",
        "path": "src/bar.py",
        "content": "new content",
        "diff": "- old\n+ new",
    }
    _, data = extract_rich_trace(item)
    assert data["path"] == "src/bar.py"
    assert data["content"] == "new content"
    assert data["diff"] == "- old\n+ new"


def test_extract_file_write():
    item = {"type": "fileWriteTool", "path": "README.md", "content": "# Title"}
    _, data = extract_rich_trace(item)
    assert data["path"] == "README.md"
    assert data["content"] == "# Title"


def test_extract_file_change():
    item = {
        "type": "fileChange",
        "changes": [
            {"path": "a.py", "kind": "modified", "diff": "+ line"},
            {"path": "b.py", "kind": "created"},
        ],
    }
    _, data = extract_rich_trace(item)
    assert len(data["files"]) == 2
    assert data["files"][0]["path"] == "a.py"
    assert data["files"][0]["kind"] == "modified"
    assert data["files"][0]["diff"] == "+ line"
    assert data["files"][1]["path"] == "b.py"
    assert "diff" not in data["files"][1]  # no diff for this entry


def test_extract_mcp_tool_call():
    item = {
        "type": "mcpToolCall",
        "server": "sqlite",
        "tool": "query",
        "arguments": {"sql": "SELECT 1"},
        "result": "[[1]]",
    }
    _, data = extract_rich_trace(item)
    assert data["server"] == "sqlite"
    assert data["tool"] == "query"
    assert "SELECT 1" in data["arguments"]
    assert data["result"] == "[[1]]"


def test_extract_web_search():
    item = {"type": "webSearch", "query": "python sqlite", "results": "some results"}
    _, data = extract_rich_trace(item)
    assert data["query"] == "python sqlite"
    assert data["results"] == "some results"


def test_extract_reasoning():
    item = {"type": "reasoning", "summary": ["thinking about", "the problem"]}
    _, data = extract_rich_trace(item)
    assert data["text"] == "thinking about the problem"


def test_extract_reasoning_truncates():
    item = {"type": "reasoning", "summary": ["x" * (MAX_TRACE_REASONING + 100)]}
    _, data = extract_rich_trace(item)
    assert "truncated" in data["text"]


def test_extract_agent_message():
    item = {"type": "agentMessage", "text": "I'll help with that."}
    _, data = extract_rich_trace(item)
    assert data["text"] == "I'll help with that."


def test_extract_plan_with_steps():
    item = {
        "type": "plan",
        "steps": [
            {"step": "Read files", "status": "completed"},
            {"step": "Write code", "status": "in_progress"},
        ],
    }
    _, data = extract_rich_trace(item)
    assert len(data["steps"]) == 2
    assert data["steps"][0] == {"step": "Read files", "status": "completed"}


def test_extract_plan_with_text_fallback():
    item = {"type": "plan", "text": "Here is the plan..."}
    _, data = extract_rich_trace(item)
    assert data["text"] == "Here is the plan..."


def test_extract_unknown_type():
    item = {"type": "unknownThing", "foo": "bar"}
    event_type, data = extract_rich_trace(item)
    assert event_type == "unknownThing"
    assert data == {}  # unknown types produce empty data


def test_extract_missing_type():
    item = {"foo": "bar"}
    event_type, data = extract_rich_trace(item)
    assert event_type == "unknown"
    assert data == {}


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------


def test_trace_context_record_inserts_events():
    conn = tmp_conn()
    ctx = TraceContext(
        entity_type="task",
        entity_id="task-123",
        stage="execution",
        plan_id="plan-1",
        project="myproject",
        conn=conn,
    )
    ctx.record("commandExecution", "completed", {"command": "pytest"})
    ctx.record("fileReadTool", "completed", {"path": "src/a.py"})

    events = list_trace_events(conn, "task", "task-123")
    assert len(events) == 2
    assert events[0]["ordinal"] == 0
    assert events[0]["event_type"] == "commandExecution"
    assert events[1]["ordinal"] == 1
    assert events[1]["event_type"] == "fileReadTool"
    conn.close()


def test_trace_context_auto_increments_ordinal():
    conn = tmp_conn()
    ctx = TraceContext(
        entity_type="plan",
        entity_id="plan-1",
        stage="enrichment",
        plan_id="plan-1",
        project="proj",
        conn=conn,
    )
    for i in range(5):
        ctx.record("reasoning", None, {"text": f"step {i}"})

    events = list_trace_events(conn, "plan", "plan-1")
    ordinals = [e["ordinal"] for e in events]
    assert ordinals == [0, 1, 2, 3, 4]
    conn.close()


def test_trace_context_set_turn_index():
    conn = tmp_conn()
    ctx = TraceContext(
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        plan_id=None,
        project="p",
        conn=conn,
    )
    ctx.record("reasoning", None, {"text": "turn 0"})
    ctx.set_turn_index(1)
    ctx.record("reasoning", None, {"text": "turn 1"})

    events = list_trace_events(conn, "task", "t-1")
    assert events[0]["turn_index"] == 0
    assert events[1]["turn_index"] == 1
    conn.close()


def test_trace_context_best_effort_no_raise():
    """TraceContext.record() never raises, even on DB errors."""
    # Use a closed connection to force an error
    conn = tmp_conn()
    ctx = TraceContext(
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        plan_id=None,
        project="p",
        conn=conn,
    )
    conn.close()

    # Should not raise — best-effort
    ctx.record("commandExecution", "completed", {"command": "echo hi"})


# ---------------------------------------------------------------------------
# DB helpers: add_trace_event / list / count / summary
# ---------------------------------------------------------------------------


def test_add_trace_event_returns_id():
    conn = tmp_conn()
    eid = add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="commandExecution",
        status="completed",
        data={"command": "ls"},
    )
    assert isinstance(eid, str)
    assert len(eid) == 36  # UUID format
    conn.close()


def test_add_trace_event_stores_json_data():
    conn = tmp_conn()
    add_trace_event(
        conn,
        entity_type="plan",
        entity_id="p-1",
        stage="planning",
        turn_index=0,
        ordinal=0,
        event_type="fileReadTool",
        data={"path": "src/main.py", "content": "print('hi')"},
    )
    events = list_trace_events(conn, "plan", "p-1")
    assert events[0]["data"]["path"] == "src/main.py"
    assert events[0]["data"]["content"] == "print('hi')"
    conn.close()


def test_list_trace_events_filter_by_event_type():
    conn = tmp_conn()
    _add_events(
        conn,
        "task",
        "t-1",
        "execution",
        [
            ("commandExecution", "completed"),
            ("fileReadTool", "completed"),
            ("commandExecution", "completed"),
            ("reasoning", None),
        ],
    )
    result = list_trace_events(conn, "task", "t-1", event_type="commandExecution")
    assert len(result) == 2
    assert all(e["event_type"] == "commandExecution" for e in result)
    conn.close()


def test_list_trace_events_filter_by_stage():
    conn = tmp_conn()
    _add_events(
        conn,
        "plan",
        "p-1",
        "enrichment",
        [
            ("reasoning", None),
        ],
    )
    _add_events(
        conn,
        "plan",
        "p-1",
        "planning",
        [
            ("fileReadTool", "completed"),
        ],
    )
    enrichment = list_trace_events(conn, "plan", "p-1", stage="enrichment")
    assert len(enrichment) == 1
    assert enrichment[0]["stage"] == "enrichment"
    planning = list_trace_events(conn, "plan", "p-1", stage="planning")
    assert len(planning) == 1
    assert planning[0]["stage"] == "planning"
    conn.close()


def test_list_trace_events_filter_by_turn_index():
    conn = tmp_conn()
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="reasoning",
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=1,
        ordinal=1,
        event_type="commandExecution",
    )
    turn1 = list_trace_events(conn, "task", "t-1", turn_index=1)
    assert len(turn1) == 1
    assert turn1[0]["event_type"] == "commandExecution"
    conn.close()


def test_list_trace_events_limit():
    conn = tmp_conn()
    _add_events(
        conn,
        "task",
        "t-1",
        "execution",
        [
            ("reasoning", None),
            ("fileReadTool", "completed"),
            ("commandExecution", "completed"),
            ("fileWriteTool", "completed"),
        ],
    )
    result = list_trace_events(conn, "task", "t-1", limit=2)
    assert len(result) == 2
    assert result[0]["ordinal"] == 0
    assert result[1]["ordinal"] == 1
    conn.close()


def test_list_trace_events_ordered_by_ordinal():
    conn = tmp_conn()
    # Insert out of ordinal order
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=2,
        event_type="fileWriteTool",
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="fileReadTool",
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=1,
        event_type="commandExecution",
    )
    result = list_trace_events(conn, "task", "t-1")
    types = [e["event_type"] for e in result]
    assert types == ["fileReadTool", "commandExecution", "fileWriteTool"]
    conn.close()


def test_count_trace_events():
    conn = tmp_conn()
    _add_events(
        conn,
        "task",
        "t-1",
        "execution",
        [
            ("commandExecution", "completed"),
            ("commandExecution", "started"),
            ("fileReadTool", "completed"),
            ("reasoning", None),
            ("reasoning", None),
            ("reasoning", None),
        ],
    )
    counts = count_trace_events(conn, "task", "t-1")
    assert counts["commandExecution"] == 2
    assert counts["fileReadTool"] == 1
    assert counts["reasoning"] == 3
    conn.close()


def test_count_trace_events_empty():
    conn = tmp_conn()
    counts = count_trace_events(conn, "task", "nonexistent")
    assert counts == {}
    conn.close()


def test_get_trace_summary():
    conn = tmp_conn()
    # Build a realistic trace: read → edit → command → mcp tool
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="fileReadTool",
        status="completed",
        data={"path": "src/main.py"},
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=1,
        event_type="fileEditTool",
        status="completed",
        data={"path": "src/main.py", "diff": "+ new line"},
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=2,
        event_type="commandExecution",
        status="completed",
        data={"command": "pytest", "exit_code": 0},
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=3,
        event_type="mcpToolCall",
        status="completed",
        data={"server": "sqlite", "tool": "query"},
    )
    # Started events should not appear in summary lists
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=4,
        event_type="fileReadTool",
        status="started",
        data={"path": "src/other.py"},
    )

    summary = get_trace_summary(conn, "task", "t-1")
    assert summary["files_read"] == ["src/main.py"]
    assert summary["files_written"] == ["src/main.py"]
    assert summary["commands_run"] == [{"command": "pytest", "exit_code": 0}]
    assert summary["tools_called"] == [{"server": "sqlite", "tool": "query"}]
    assert summary["total_events"] == 5
    # started fileReadTool counted in event_counts but not in files_read
    assert summary["event_counts"]["fileReadTool"] == 2
    conn.close()


def test_get_trace_summary_deduplicates_files():
    conn = tmp_conn()
    for i in range(3):
        add_trace_event(
            conn,
            entity_type="task",
            entity_id="t-1",
            stage="execution",
            turn_index=0,
            ordinal=i,
            event_type="fileReadTool",
            status="completed",
            data={"path": "src/same.py"},
        )

    summary = get_trace_summary(conn, "task", "t-1")
    assert summary["files_read"] == ["src/same.py"]  # deduplicated
    conn.close()


def test_get_trace_summary_file_change_events():
    conn = tmp_conn()
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="t-1",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="fileChange",
        status="completed",
        data={"files": [{"path": "a.py"}, {"path": "b.py"}]},
    )
    summary = get_trace_summary(conn, "task", "t-1")
    assert set(summary["files_written"]) == {"a.py", "b.py"}
    conn.close()


def test_get_trace_summary_empty():
    conn = tmp_conn()
    summary = get_trace_summary(conn, "task", "nonexistent")
    assert summary["files_read"] == []
    assert summary["files_written"] == []
    assert summary["commands_run"] == []
    assert summary["tools_called"] == []
    assert summary["total_events"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# Purge includes trace_events
# ---------------------------------------------------------------------------


def test_purge_clears_trace_events(tmp_path):
    from agm.db import add_project, create_plan_request, create_task

    conn = tmp_conn()
    pid = add_project(conn, "proj", str(tmp_path))["id"]
    plan = create_plan_request(
        conn, project_id=pid, prompt="do stuff", caller="cli", backend="codex"
    )
    task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="task 1",
        description="desc",
    )

    # Add trace events for both plan and task
    add_trace_event(
        conn,
        entity_type="plan",
        entity_id=plan["id"],
        stage="enrichment",
        turn_index=0,
        ordinal=0,
        event_type="reasoning",
        data={"text": "plan trace"},
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id=task["id"],
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="commandExecution",
        data={"command": "ls"},
    )

    # Verify events exist
    assert len(list_trace_events(conn, "plan", plan["id"])) == 1
    assert len(list_trace_events(conn, "task", task["id"])) == 1

    result = purge_data(conn, pid)
    assert result["counts"]["trace_events"] >= 2

    # Verify events are gone
    assert len(list_trace_events(conn, "plan", plan["id"])) == 0
    assert len(list_trace_events(conn, "task", task["id"])) == 0
    conn.close()


def test_purge_preview_includes_trace_events():
    from agm.db import add_project, create_plan_request, create_task

    conn = tmp_conn()
    pid = add_project(conn, "proj", "/tmp/proj")["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="x", caller="cli", backend="codex")
    create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="t",
        description="d",
    )
    add_trace_event(
        conn,
        entity_type="plan",
        entity_id=plan["id"],
        stage="planning",
        turn_index=0,
        ordinal=0,
        event_type="fileReadTool",
    )

    counts = purge_preview_counts(conn, pid)
    assert "trace_events" in counts
    assert counts["trace_events"] >= 1
    conn.close()


# ---------------------------------------------------------------------------
# Isolation: events scoped to entity
# ---------------------------------------------------------------------------


def test_trace_events_isolated_between_entities():
    conn = tmp_conn()
    _add_events(
        conn,
        "task",
        "t-1",
        "execution",
        [
            ("commandExecution", "completed"),
            ("fileReadTool", "completed"),
        ],
    )
    _add_events(
        conn,
        "task",
        "t-2",
        "execution",
        [
            ("reasoning", None),
        ],
    )

    assert len(list_trace_events(conn, "task", "t-1")) == 2
    assert len(list_trace_events(conn, "task", "t-2")) == 1
    assert len(list_trace_events(conn, "plan", "t-1")) == 0  # wrong entity_type
    conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_events(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    stage: str,
    events: list[tuple[str, str | None]],
) -> None:
    """Convenience: add multiple events with auto-incrementing ordinal."""
    for i, (event_type, status) in enumerate(events):
        add_trace_event(
            conn,
            entity_type=entity_type,
            entity_id=entity_id,
            stage=stage,
            turn_index=0,
            ordinal=i,
            event_type=event_type,
            status=status,
        )
