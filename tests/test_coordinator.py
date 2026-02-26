"""Tests for coordinator steering logic."""

from __future__ import annotations

import json
import sqlite3

from agm.db import (
    create_plan_request,
    create_session,
    create_task,
    finalize_plan_request,
    get_connection,
    list_channel_messages,
    list_task_steers,
    set_plan_session_id,
    update_task_status,
)
from agm.jobs_coordinator import _int_env, run_plan_coordinator


def _project_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()
    assert row is not None
    return row["id"]


def _setup_plan_with_session(conn: sqlite3.Connection) -> tuple[dict, dict]:
    project_id = _project_id(conn)
    session = create_session(conn, project_id=project_id, trigger="do")
    plan = create_plan_request(
        conn,
        project_id=project_id,
        prompt="p",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, plan["id"], '{"title":"x","summary":"s","tasks":[]}')
    set_plan_session_id(conn, plan["id"], session["id"])
    return plan, session


def test_coordinator_int_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("AGM_COORDINATOR_MIN_RUNNING", "abc")
    assert _int_env("AGM_COORDINATOR_MIN_RUNNING", 3, min_value=1) == 3


def test_coordinator_emits_overlap_steers(db_conn_path, monkeypatch):
    conn, db_path = db_conn_path
    monkeypatch.setattr("agm.db.get_connection", lambda *_: get_connection(db_path))
    plan, session = _setup_plan_with_session(conn)
    t1 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="A",
        description="d",
        files=json.dumps(["src/api.py", "src/common.py"]),
    )
    t2 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="B",
        description="d",
        files=json.dumps(["src/api.py"]),
    )
    t3 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=2,
        title="C",
        description="d",
        files=json.dumps(["README.md"]),
    )
    for task in (t1, t2, t3):
        update_task_status(conn, task["id"], "running")
        conn.execute(
            "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
            (f"thread-{task['id'][:6]}", f"turn-{task['id'][:6]}", task["id"]),
        )
    conn.commit()

    async def _fake_live_steer(*, thread_id, active_turn_id, content, timeout=30):
        return {"turnId": active_turn_id}

    monkeypatch.setattr("agm.jobs_coordinator.steer_active_turn", _fake_live_steer)

    result = run_plan_coordinator(plan["id"])
    assert result.startswith("coordinator:sent=")

    steers = list_task_steers(conn, session_id=session["id"])
    assert len(steers) == 2
    assert {row["task_id"] for row in steers} == {t1["id"], t2["id"]}
    assert all(row["reason"] == "file_overlap" for row in steers)
    assert all(row["live_applied"] == 1 for row in steers)

    channel = list_channel_messages(conn, session["id"], kind="steer")
    assert len(channel) == 2


def test_coordinator_records_live_race_errors(db_conn_path, monkeypatch):
    conn, db_path = db_conn_path
    monkeypatch.setattr("agm.db.get_connection", lambda *_: get_connection(db_path))
    plan, session = _setup_plan_with_session(conn)
    tasks = [
        create_task(
            conn,
            plan_id=plan["id"],
            ordinal=i,
            title=f"T{i}",
            description="d",
            files=json.dumps([f"f{i}.py"]),
        )
        for i in range(3)
    ]
    for task in tasks:
        update_task_status(conn, task["id"], "running")
        conn.execute(
            "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
            (f"thread-{task['id'][:6]}", f"turn-{task['id'][:6]}", task["id"]),
        )
    # Force one task to look stuck.
    conn.execute(
        (
            "UPDATE tasks SET active_turn_started_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-1200 seconds') "
            "WHERE id = ?"
        ),
        (tasks[0]["id"],),
    )
    conn.commit()

    async def _raise_race(*, thread_id, active_turn_id, content, timeout=30):
        raise RuntimeError("expectedTurnId mismatch")

    monkeypatch.setattr("agm.jobs_coordinator.steer_active_turn", _raise_race)

    run_plan_coordinator(plan["id"])
    steers = list_task_steers(conn, session_id=session["id"])
    stuck_rows = [row for row in steers if row["reason"] == "stuck_turn"]
    assert len(stuck_rows) == 1
    assert "expectedTurnId mismatch" in (stuck_rows[0]["live_error"] or "")

    # Dedupe: second run should not emit the same steer within lookback window.
    run_plan_coordinator(plan["id"])
    steers_after = list_task_steers(conn, session_id=session["id"])
    assert len(steers_after) == len(steers)


def test_coordinator_retries_live_after_non_live_record(db_conn_path, monkeypatch):
    """A no-active-turn steer should not block later live steer attempts."""
    conn, db_path = db_conn_path
    monkeypatch.setattr("agm.db.get_connection", lambda *_: get_connection(db_path))
    plan, session = _setup_plan_with_session(conn)
    tasks = [
        create_task(
            conn,
            plan_id=plan["id"],
            ordinal=i,
            title=f"T{i}",
            description="d",
            files=json.dumps([f"f{i}.py"]),
        )
        for i in range(3)
    ]
    for task in tasks:
        update_task_status(conn, task["id"], "running")
    conn.execute(
        (
            "UPDATE tasks SET active_turn_started_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-1200 seconds') "
            "WHERE id = ?"
        ),
        (tasks[0]["id"],),
    )
    conn.commit()

    async def _fake_live_steer(*, thread_id, active_turn_id, content, timeout=30):
        return {"turnId": active_turn_id}

    live_calls: list[tuple[str, str]] = []

    async def _recording_live_steer(*, thread_id, active_turn_id, content, timeout=30):
        live_calls.append((thread_id, active_turn_id))
        return await _fake_live_steer(
            thread_id=thread_id,
            active_turn_id=active_turn_id,
            content=content,
            timeout=timeout,
        )

    monkeypatch.setattr("agm.jobs_coordinator.steer_active_turn", _recording_live_steer)

    # First run: no active turn/thread on the stuck task.
    run_plan_coordinator(plan["id"])
    steers = list_task_steers(conn, session_id=session["id"])
    assert len(steers) == 1
    assert steers[0]["live_requested"] == 0
    assert live_calls == []

    # Second run: active turn is available; coordinator should attempt live steer.
    conn.execute(
        "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
        ("thread-live", "turn-live", tasks[0]["id"]),
    )
    conn.commit()

    run_plan_coordinator(plan["id"])
    steers_after = list_task_steers(conn, session_id=session["id"])
    assert len(steers_after) == 2
    assert steers_after[0]["live_requested"] == 1
    assert steers_after[0]["applied_turn_id"] == "turn-live"
    assert live_calls == [("thread-live", "turn-live")]
