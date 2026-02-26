"""Tests for the plans database layer."""

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from agm.db import (
    add_plan_log,
    add_plan_question,
    add_project,
    answer_plan_question,
    create_plan_request,
    create_task,
    fail_stale_running_plan_for_doctor,
    finalize_plan_request,
    get_connection,
    get_plan_chain,
    get_plan_question,
    get_plan_request,
    list_plan_logs,
    list_plan_questions,
    list_plan_requests,
    list_plan_watch_events,
    list_recent_failed_plans,
    list_running_plan_workers,
    list_status_history,
    list_status_history_timing_rows,
    record_status_change,
    reset_plan_for_retry,
    set_plan_model,
    set_plan_request_thread_id,
    set_plan_request_worker,
    update_plan_request_status,
)


def get_project_id(conn):
    row = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()
    return row["id"]


def make_plan(conn, prompt="add auth"):
    pid = get_project_id(conn)
    return create_plan_request(conn, project_id=pid, prompt=prompt, caller="cli", backend="codex")


class _CASRaceConnection:
    """Wrap a connection and inject a status change between read and CAS update."""

    def __init__(self, conn, *, plan_id: str, update_sql_prefix: str, race_status: str) -> None:
        self._conn = conn
        self._plan_id = plan_id
        self._update_sql_prefix = update_sql_prefix
        self._race_status = race_status
        self._armed = False
        self._raced = False

    def execute(self, sql, params=()):
        if not self._armed and sql.startswith("SELECT status FROM plans WHERE id = ?"):
            self._armed = True
        elif self._armed and not self._raced and sql.startswith(self._update_sql_prefix):
            self._conn.execute(
                "UPDATE plans SET status = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (self._race_status, self._plan_id),
            )
            self._raced = True
        return self._conn.execute(sql, params)

    def commit(self):
        return self._conn.commit()


# -- plan CRUD --


def test_create_plan_request(db_conn):
    conn = db_conn
    p = make_plan(conn)
    assert p["status"] == "pending"
    assert p["caller"] == "cli"
    assert p["backend"] == "codex"
    assert p["prompt"] == "add auth"
    assert p["actor"] is not None
    assert p["pid"] is None
    assert p["thread_id"] is None
    assert p["parent_id"] is None
    # Token defaults via round-trip
    found = get_plan_request(conn, p["id"])
    assert found["input_tokens"] == 0
    assert found["output_tokens"] == 0


def test_get_connection_migrates_plan_token_columns_idempotently():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE plans (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            parent_id TEXT REFERENCES plans(id),
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            actor TEXT NOT NULL,
            caller TEXT NOT NULL,
            backend TEXT NOT NULL,
            input_tokens INTEGER,
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend, input_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("plan1", "p1", "legacy prompt", "alice", "cli", "codex", None),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(plans)").fetchall()}
    assert "input_tokens" in cols
    assert "output_tokens" in cols
    row = migrated.execute(
        "SELECT input_tokens, output_tokens FROM plans WHERE id = ?",
        ("plan1",),
    ).fetchone()
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    migrated.close()

    reopened = get_connection(db_path)
    row = reopened.execute(
        "SELECT input_tokens, output_tokens FROM plans WHERE id = ?",
        ("plan1",),
    ).fetchone()
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    reopened.close()


def test_migration_adds_plan_model_column_for_v5():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE plans (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            parent_id TEXT REFERENCES plans(id),
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            actor TEXT NOT NULL,
            caller TEXT NOT NULL,
            backend TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            pid INTEGER,
            thread_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("plan1", "p1", "legacy prompt", "alice", "cli", "codex"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row[1] for row in migrated.execute("PRAGMA table_info(plans)").fetchall()}
    assert "model" in cols
    row = migrated.execute("SELECT model FROM plans WHERE id = ?", ("plan1",)).fetchone()
    assert row["model"] is None

    reopened = get_connection(db_path)
    row = reopened.execute("SELECT model FROM plans WHERE id = ?", ("plan1",)).fetchone()
    assert row["model"] is None
    reopened.close()


@pytest.mark.parametrize(
    "caller, backend, match",
    [
        ("bogus", "codex", "Invalid caller"),
        ("cli", "bogus", "Invalid backend"),
    ],
)
def test_create_plan_request_rejects_invalid_enums(db_conn, caller, backend, match):
    conn = db_conn
    pid = get_project_id(conn)
    with pytest.raises(ValueError, match=match):
        create_plan_request(conn, project_id=pid, prompt="x", caller=caller, backend=backend)


def test_get_plan_request_not_found(db_conn):
    conn = db_conn
    assert get_plan_request(conn, "nope") is None


def test_set_plan_model(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    assert set_plan_model(conn, plan["id"], "gpt-5")
    found = get_plan_request(conn, plan["id"])
    assert found["model"] == "gpt-5"

    assert set_plan_model(conn, plan["id"], None)
    found = get_plan_request(conn, plan["id"])
    assert found["model"] is None

    assert not set_plan_model(conn, "missing", "gpt-5")


def test_list_plan_requests_by_project(db_conn):
    conn = db_conn
    pid = get_project_id(conn)
    add_project(conn, "other", "/tmp/other")
    other_id = conn.execute("SELECT id FROM projects WHERE name = 'other'").fetchone()["id"]
    create_plan_request(conn, project_id=pid, prompt="plan A", caller="cli", backend="codex")
    create_plan_request(conn, project_id=other_id, prompt="plan B", caller="cli", backend="codex")
    plans = list_plan_requests(conn, project_id=pid)
    assert len(plans) == 1
    assert plans[0]["prompt"] == "plan A"


def test_list_plan_requests_by_statuses(db_conn):
    conn = db_conn
    pid = get_project_id(conn)
    add_project(conn, "other", "/tmp/other")
    other_id = conn.execute("SELECT id FROM projects WHERE name = 'other'").fetchone()["id"]

    earliest = create_plan_request(
        conn, project_id=pid, prompt="pending plan", caller="cli", backend="codex"
    )
    middle = create_plan_request(
        conn, project_id=pid, prompt="running plan", caller="cli", backend="codex"
    )
    excluded = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    latest = create_plan_request(
        conn, project_id=pid, prompt="second pending", caller="cli", backend="codex"
    )
    assert update_plan_request_status(conn, middle["id"], "running")
    assert update_plan_request_status(conn, excluded["id"], "failed")

    other_plan = create_plan_request(
        conn, project_id=other_id, prompt="other running", caller="cli", backend="codex"
    )
    assert update_plan_request_status(conn, other_plan["id"], "running")

    conn.execute(
        "UPDATE plans SET created_at = ? WHERE id = ?", ("2026-01-01T00:00:00Z", earliest["id"])
    )
    conn.execute(
        "UPDATE plans SET created_at = ? WHERE id = ?", ("2026-01-01T00:00:00Z", middle["id"])
    )
    conn.execute(
        "UPDATE plans SET created_at = ? WHERE id = ?", ("2026-01-01T00:00:00Z", excluded["id"])
    )
    conn.execute(
        "UPDATE plans SET created_at = ? WHERE id = ?", ("2026-01-02T00:00:00Z", latest["id"])
    )
    conn.execute(
        "UPDATE plans SET created_at = ? WHERE id = ?", ("2026-01-01T12:00:00Z", other_plan["id"])
    )
    conn.commit()

    plans = list_plan_requests(conn, project_id=pid, statuses=("running", "pending"))
    assert len(plans) == 3
    assert [plan["id"] for plan in plans] == [earliest["id"], middle["id"], latest["id"]]
    assert [plan["status"] for plan in plans] == ["pending", "running", "pending"]


def test_list_plan_requests_status_precedence_over_statuses(db_conn):
    conn = db_conn
    pid = get_project_id(conn)
    create_plan_request(conn, project_id=pid, prompt="pending", caller="cli", backend="codex")
    running = create_plan_request(
        conn, project_id=pid, prompt="running", caller="cli", backend="codex"
    )
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed", caller="cli", backend="codex"
    )
    assert update_plan_request_status(conn, running["id"], "running")
    assert update_plan_request_status(conn, failed["id"], "failed")

    plans = list_plan_requests(
        conn, project_id=pid, status="failed", statuses=("pending", "running")
    )
    assert len(plans) == 1
    assert plans[0]["id"] == failed["id"]
    assert plans[0]["status"] == "failed"


def test_list_plan_requests_with_empty_statuses(db_conn):
    conn = db_conn
    pid = get_project_id(conn)
    create_plan_request(conn, project_id=pid, prompt="pending plan", caller="cli", backend="codex")
    create_plan_request(conn, project_id=pid, prompt="running plan", caller="cli", backend="codex")
    create_plan_request(conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex")

    all_plans = list_plan_requests(conn, project_id=pid, statuses=[])
    assert len(all_plans) == 3


@pytest.mark.parametrize("add_non_error_logs", [False, True])
def test_list_recent_failed_plans_with_no_error_logs(db_conn, add_non_error_logs):
    """Failed plans with no ERROR-level logs should have error=None."""
    conn = db_conn
    p = make_plan(conn, "fail plan")
    assert update_plan_request_status(conn, p["id"], "failed")
    if add_non_error_logs:
        add_plan_log(conn, plan_id=p["id"], level="INFO", message="non-error message")
        add_plan_log(conn, plan_id=p["id"], level="WARNING", message="also non-error")

    failures = list_recent_failed_plans(conn)
    assert len(failures) == 1
    assert failures[0]["id"] == p["id"]
    assert failures[0]["error"] is None
    assert failures[0]["project_name"] == "testproj"


def test_list_recent_failed_plans_error_message_prefers_latest_deterministically(db_conn):
    conn = db_conn
    p = make_plan(conn, "latest error wins")
    assert update_plan_request_status(conn, p["id"], "failed")

    old_message = "legacy failure: " + ("x" * 200)
    new_message = "latest failure: " + ("y" * 200)
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) "
        "VALUES (?, ?, 'ERROR', ?, ?)",
        (uuid.uuid4().hex[:12], p["id"], old_message, "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) "
        "VALUES (?, ?, 'ERROR', ?, ?)",
        (uuid.uuid4().hex[:12], p["id"], new_message, "2026-01-01T00:00:00Z"),
    )
    conn.commit()

    failures = list_recent_failed_plans(conn)
    assert len(failures) == 1
    assert failures[0]["id"] == p["id"]
    assert failures[0]["error"] == new_message
    assert len(failures[0]["error"]) == len(new_message)


def test_list_recent_failed_plans_filters_by_project(db_conn):
    conn = db_conn
    primary_id = get_project_id(conn)
    other = add_project(conn, "other", "/tmp/other")

    p1 = make_plan(conn, "primary failed plan")
    p2 = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other failed plan",
        caller="cli",
        backend="codex",
    )
    p3 = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other failed plan 2",
        caller="cli",
        backend="codex",
    )

    assert update_plan_request_status(conn, p1["id"], "failed")
    assert update_plan_request_status(conn, p2["id"], "failed")
    assert update_plan_request_status(conn, p3["id"], "failed")

    primary_only = list_recent_failed_plans(conn, project_id=primary_id)
    assert len(primary_only) == 1
    assert primary_only[0]["project_id"] == primary_id
    assert primary_only[0]["project_name"] == "testproj"

    other_only = list_recent_failed_plans(conn, project_id=other["id"])
    assert {row["id"] for row in other_only} == {p2["id"], p3["id"]}
    assert all(row["project_name"] == "other" for row in other_only)


def test_list_recent_failed_plans_respects_limit_and_recent_order(db_conn):
    conn = db_conn
    first = make_plan(conn, "first")
    second = make_plan(conn, "second")
    third = make_plan(conn, "third")
    assert update_plan_request_status(conn, first["id"], "failed")
    assert update_plan_request_status(conn, second["id"], "failed")
    assert update_plan_request_status(conn, third["id"], "failed")
    conn.execute(
        "UPDATE plans SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:01Z", first["id"]),
    )
    conn.execute(
        "UPDATE plans SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:02Z", second["id"]),
    )
    conn.execute(
        "UPDATE plans SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:03Z", third["id"]),
    )
    conn.commit()

    failures = list_recent_failed_plans(conn, limit=2)
    assert [row["id"] for row in failures] == [third["id"], second["id"]]


def test_finalize_plan_request(db_conn):
    conn = db_conn
    p = make_plan(conn, "add tests")
    plan_text = "## Plan\n1. Add unit tests\n2. Add integration tests"
    assert finalize_plan_request(conn, p["id"], plan_text)
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "finalized"
    assert "unit tests" in found["plan"]
    assert list_status_history(conn, entity_type="plan", entity_id=p["id"]) == []


def test_finalize_plan_request_not_found(db_conn):
    conn = db_conn
    assert finalize_plan_request(conn, "nope", "text") is False


def test_update_plan_request_status(db_conn):
    conn = db_conn
    p = make_plan(conn, "refactor")
    assert update_plan_request_status(conn, p["id"], "finalized")
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "finalized"
    assert list_status_history(conn, entity_type="plan", entity_id=p["id"]) == []


def test_update_plan_request_status_invalid(db_conn):
    conn = db_conn
    p = make_plan(conn, "x")
    with pytest.raises(ValueError, match="Invalid status"):
        update_plan_request_status(conn, p["id"], "bogus")


def test_update_plan_request_status_record_history_on_transition(db_conn):
    conn = db_conn
    p = make_plan(conn, "transition")
    assert update_plan_request_status(conn, p["id"], "running", record_history=True)
    history = list_status_history(conn, entity_type="plan", entity_id=p["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "pending"
    assert history[0]["new_status"] == "running"


def test_update_plan_request_status_cas_failure_returns_false_and_no_history(db_conn):
    conn = db_conn
    p = make_plan(conn, "race update")
    raced_conn = _CASRaceConnection(
        conn,
        plan_id=p["id"],
        update_sql_prefix="UPDATE plans SET status = ?",
        race_status="failed",
    )
    assert not update_plan_request_status(raced_conn, p["id"], "running", record_history=True)
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "failed"
    assert list_status_history(conn, entity_type="plan", entity_id=p["id"]) == []


def test_finalize_plan_request_record_history_and_already_finalized_edge_case(db_conn):
    conn = db_conn
    p = make_plan(conn, "finalize edge")
    assert finalize_plan_request(conn, p["id"], "first text", record_history=True)
    assert finalize_plan_request(conn, p["id"], "second text", record_history=True)
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "finalized"
    assert found["plan"] == "second text"
    history = list_status_history(conn, entity_type="plan", entity_id=p["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "pending"
    assert history[0]["new_status"] == "finalized"


def test_finalize_plan_request_cas_failure_returns_false_and_no_history(db_conn):
    conn = db_conn
    p = make_plan(conn, "race finalize")
    raced_conn = _CASRaceConnection(
        conn,
        plan_id=p["id"],
        update_sql_prefix="UPDATE plans SET plan = ?, status = 'finalized'",
        race_status="failed",
    )
    assert not finalize_plan_request(raced_conn, p["id"], "final text", record_history=True)
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "failed"
    assert found["plan"] is None
    assert list_status_history(conn, entity_type="plan", entity_id=p["id"]) == []


# -- plan status history --


def test_record_status_change_plan_validates_status_set(db_conn):
    conn = db_conn
    p = make_plan(conn, "x")
    with pytest.raises(ValueError, match="Invalid new_status 'review'"):
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=p["id"],
            old_status="pending",
            new_status="review",
        )


def test_plan_status_history_order_nullable_fields_and_timing_rows(db_conn):
    conn = db_conn
    p = make_plan(conn, "status timeline")
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status=None,
        new_status="pending",
        actor=None,
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="pending",
        new_status="running",
        actor="alice",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="running",
        new_status="awaiting_input",
        actor="alice",
        created_at="2026-01-01T00:00:05Z",
    )
    conn.commit()

    history = list_status_history(conn, entity_type="plan", entity_id=p["id"])
    assert [row["new_status"] for row in history] == ["pending", "running", "awaiting_input"]
    assert history[0]["old_status"] is None
    assert history[0]["actor"] is None
    assert history[1]["actor"] == "alice"
    assert history[0]["id"] < history[1]["id"]
    assert history[0]["created_at"] == history[1]["created_at"] == "2026-01-01T00:00:00Z"

    timing_rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=p["id"])
    assert [row["duration_seconds"] for row in timing_rows] == [0, 5, None]
    assert timing_rows[0]["next_created_at"] == "2026-01-01T00:00:00Z"
    assert timing_rows[1]["next_created_at"] == "2026-01-01T00:00:05Z"
    assert timing_rows[2]["next_created_at"] is None


def test_record_status_change_can_participate_in_caller_transaction(db_conn):
    conn = db_conn
    p = make_plan(conn, "atomic")

    conn.execute("BEGIN")
    conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (p["id"],))
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="pending",
        new_status="running",
    )
    conn.rollback()

    found = get_plan_request(conn, p["id"])
    assert found["status"] == "pending"
    assert list_status_history(conn, entity_type="plan", entity_id=p["id"]) == []


# -- plan continue (parent_id) --


def test_create_plan_request_with_parent(db_conn):
    conn = db_conn
    parent = make_plan(conn, "original plan")
    child = create_plan_request(
        conn,
        project_id=parent["project_id"],
        prompt="follow-up",
        caller="cli",
        backend="codex",
        parent_id=parent["id"],
    )
    assert child["parent_id"] == parent["id"]
    found = get_plan_request(conn, child["id"])
    assert found["parent_id"] == parent["id"]


def test_create_plan_request_parent_persists(db_conn):
    """parent_id should survive round-trip through DB."""
    conn = db_conn
    parent = make_plan(conn, "parent")
    child = create_plan_request(
        conn,
        project_id=parent["project_id"],
        prompt="child",
        caller="cli",
        backend="codex",
        parent_id=parent["id"],
    )
    found = get_plan_request(conn, child["id"])
    assert found["parent_id"] == parent["id"]
    assert found["prompt"] == "child"


# -- plan history (get_plan_chain) --


def test_get_plan_chain(db_conn):
    """get_plan_chain should return the full continuation chain."""
    conn = db_conn
    root = make_plan(conn, "root")
    child = create_plan_request(
        conn,
        project_id=root["project_id"],
        prompt="child",
        caller="cli",
        backend="codex",
        parent_id=root["id"],
    )
    grandchild = create_plan_request(
        conn,
        project_id=root["project_id"],
        prompt="grandchild",
        caller="cli",
        backend="codex",
        parent_id=child["id"],
    )
    # Query from the middle -- should get all 3
    chain = get_plan_chain(conn, child["id"])
    assert len(chain) == 3
    assert chain[0]["id"] == root["id"]
    assert chain[1]["id"] == child["id"]
    assert chain[2]["id"] == grandchild["id"]


def test_get_plan_chain_not_found(db_conn):
    conn = db_conn
    assert get_plan_chain(conn, "nonexistent") == []


# -- worker tracking --


def test_set_plan_request_worker(db_conn):
    conn = db_conn
    p = make_plan(conn)
    assert set_plan_request_worker(conn, p["id"], pid=12345, thread_id="thread_abc")
    found = get_plan_request(conn, p["id"])
    assert found["pid"] == 12345
    assert found["thread_id"] == "thread_abc"
    assert found["status"] == "pending"  # set_plan_worker does NOT change status


def test_set_plan_request_worker_no_thread(db_conn):
    conn = db_conn
    p = make_plan(conn)
    assert set_plan_request_worker(conn, p["id"], pid=99)
    found = get_plan_request(conn, p["id"])
    assert found["pid"] == 99
    assert found["thread_id"] is None
    assert found["status"] == "pending"  # status unchanged


def test_set_plan_request_thread_id(db_conn):
    conn = db_conn
    p = make_plan(conn)
    set_plan_request_worker(conn, p["id"], pid=1)
    assert set_plan_request_thread_id(conn, p["id"], "thread_xyz")
    found = get_plan_request(conn, p["id"])
    assert found["thread_id"] == "thread_xyz"


def test_list_running_plan_workers_filters_running_and_pid(db_conn):
    conn = db_conn
    main_project_id = get_project_id(conn)

    running_with_pid = make_plan(conn, "running with pid")
    update_plan_request_status(conn, running_with_pid["id"], "running")
    set_plan_request_worker(conn, running_with_pid["id"], pid=111, thread_id="thread-a")

    running_no_pid = make_plan(conn, "running no pid")
    update_plan_request_status(conn, running_no_pid["id"], "running")

    pending_with_pid = make_plan(conn, "pending with pid")
    set_plan_request_worker(conn, pending_with_pid["id"], pid=222)

    failed_with_pid = make_plan(conn, "failed with pid")
    set_plan_request_worker(conn, failed_with_pid["id"], pid=333)
    update_plan_request_status(conn, failed_with_pid["id"], "failed")

    other = add_project(conn, "otherproj", "/tmp/otherproj")
    other_plan = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other running",
        caller="cli",
        backend="codex",
    )
    update_plan_request_status(conn, other_plan["id"], "running")
    set_plan_request_worker(conn, other_plan["id"], pid=444)

    all_running = list_running_plan_workers(conn)
    all_ids = {plan["id"] for plan in all_running}
    assert running_with_pid["id"] in all_ids
    assert other_plan["id"] in all_ids
    assert running_no_pid["id"] not in all_ids
    assert pending_with_pid["id"] not in all_ids
    assert failed_with_pid["id"] not in all_ids

    main_only = list_running_plan_workers(conn, project_id=main_project_id)
    main_ids = {plan["id"] for plan in main_only}
    assert main_ids == {running_with_pid["id"]}
    assert main_only[0]["project_dir"] == "/tmp/testproj"


# -- plan questions --


def test_add_plan_question(db_conn):
    conn = db_conn
    p = make_plan(conn)
    q = add_plan_question(conn, plan_id=p["id"], question="Which auth method?", options="OAuth,JWT")
    assert q["question"] == "Which auth method?"
    assert q["options"] == "OAuth,JWT"
    assert q["header"] is None
    assert q["multi_select"] is False
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "pending"  # add_plan_question does NOT change status


def test_add_plan_question_with_structured_fields(db_conn):
    conn = db_conn
    p = make_plan(conn)
    import json

    opts = json.dumps(
        [
            {"label": "OAuth2", "description": "Industry standard"},
            {"label": "JWT", "description": "Stateless"},
        ]
    )
    q = add_plan_question(
        conn,
        plan_id=p["id"],
        question="Which auth method?",
        options=opts,
        header="Auth method",
        multi_select=True,
    )
    assert q["header"] == "Auth method"
    assert q["multi_select"] is True
    assert q["options"] == opts
    # Verify round-trip through DB
    found = get_plan_question(conn, q["id"])
    assert found["header"] == "Auth method"
    assert found["multi_select"] == 1  # SQLite stores as int


def test_answer_plan_question(db_conn):
    conn = db_conn
    p = make_plan(conn)
    q = add_plan_question(conn, plan_id=p["id"], question="Which auth?")
    assert answer_plan_question(conn, q["id"], "OAuth", answered_by="testuser")
    found_q = get_plan_question(conn, q["id"])
    assert found_q["answer"] == "OAuth"
    assert found_q["answered_by"] == "testuser"
    assert found_q["answered_at"] is not None
    found_p = get_plan_request(conn, p["id"])
    assert found_p["status"] == "pending"  # answer does NOT change plan status


def test_answer_already_answered(db_conn):
    conn = db_conn
    p = make_plan(conn)
    q = add_plan_question(conn, plan_id=p["id"], question="Pick one")
    answer_plan_question(conn, q["id"], "first")
    assert answer_plan_question(conn, q["id"], "second") is False


def test_list_plan_questions_unanswered(db_conn):
    conn = db_conn
    p = make_plan(conn)
    q1 = add_plan_question(conn, plan_id=p["id"], question="Q1")
    add_plan_question(conn, plan_id=p["id"], question="Q2")
    answer_plan_question(conn, q1["id"], "answer1")
    unanswered = list_plan_questions(conn, p["id"], unanswered_only=True)
    assert len(unanswered) == 1
    assert unanswered[0]["question"] == "Q2"


def test_get_plan_question_not_found(db_conn):
    conn = db_conn
    assert get_plan_question(conn, "nope") is None


# -- plan logs --


def test_add_and_list_plan_logs(db_conn):
    conn = db_conn
    p = make_plan(conn)
    entry = add_plan_log(conn, plan_id=p["id"], level="INFO", message="first")
    assert entry["plan_id"] == p["id"]
    assert entry["level"] == "INFO"
    assert "id" in entry
    add_plan_log(conn, plan_id=p["id"], level="WARNING", message="second")
    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 2
    assert logs[0]["message"] == "first"
    assert logs[1]["message"] == "second"


def test_list_plan_logs_by_level(db_conn):
    conn = db_conn
    p = make_plan(conn)
    add_plan_log(conn, plan_id=p["id"], level="INFO", message="info msg")
    add_plan_log(conn, plan_id=p["id"], level="ERROR", message="error msg")
    add_plan_log(conn, plan_id=p["id"], level="INFO", message="info msg 2")
    logs = list_plan_logs(conn, p["id"], level="ERROR")
    assert len(logs) == 1
    assert logs[0]["message"] == "error msg"


def test_list_plan_logs_empty(db_conn):
    conn = db_conn
    p = make_plan(conn)
    logs = list_plan_logs(conn, p["id"])
    assert logs == []


def test_plan_log_persists(db_conn):
    """Round-trip: insert via add_plan_log, read back via list_plan_logs."""
    conn = db_conn
    p = make_plan(conn)
    add_plan_log(conn, plan_id=p["id"], level="INFO", message="persisted")
    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 1
    assert logs[0]["message"] == "persisted"
    assert logs[0]["created_at"] is not None


def test_plan_db_handler(db_conn):
    """PlanDBHandler writes to DB via the logging system."""
    import logging

    from agm.jobs import PlanDBHandler

    conn = db_conn
    p = make_plan(conn)
    handler = PlanDBHandler(conn, p["id"])
    logger = logging.getLogger("test_plan_db_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("handler test message")
        logger.warning("handler warning")
    finally:
        logger.removeHandler(handler)
    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 2
    assert logs[0]["level"] == "INFO"
    assert "handler test message" in logs[0]["message"]
    assert logs[1]["level"] == "WARNING"


def test_list_plan_watch_events_combines_sources_and_filters_other_plans(db_conn):
    conn = db_conn
    p1 = make_plan(conn)
    p2 = make_plan(conn, "other plan")
    t1 = create_task(conn, plan_id=p1["id"], ordinal=0, title="task 1", description="a")
    t2 = create_task(conn, plan_id=p2["id"], ordinal=0, title="task 2", description="b")

    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], p1["id"], "INFO", "plan-old", "2026-01-01T00:00:01Z"),
    )
    conn.execute(
        "INSERT INTO task_logs (id, task_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], t1["id"], "INFO", "task-old", "2026-01-01T00:00:02Z"),
    )
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], p1["id"], "INFO", "plan-new", "2026-01-01T00:00:03Z"),
    )
    conn.execute(
        "INSERT INTO task_logs (id, task_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], t1["id"], "INFO", "task-new", "2026-01-01T00:00:04Z"),
    )
    # Unrelated plan rows should never appear.
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], p2["id"], "INFO", "other-plan", "2026-01-01T00:00:05Z"),
    )
    conn.execute(
        "INSERT INTO task_logs (id, task_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], t2["id"], "INFO", "other-task", "2026-01-01T00:00:06Z"),
    )
    conn.commit()

    events = list_plan_watch_events(conn, p1["id"], limit=3)
    assert len(events) == 3
    assert [e["message"] for e in events] == ["task-new", "plan-new", "task-old"]
    assert events[0]["timestamp"] > events[1]["timestamp"] > events[2]["timestamp"]
    assert events[0]["source"] == "task"
    assert events[0]["task_id"] == t1["id"]
    assert events[1]["source"] == "plan"
    assert events[1]["task_id"] is None
    assert {"other-plan", "other-task"}.isdisjoint({e["message"] for e in events})
    expected_keys = {
        "source",
        "task_id",
        "timestamp",
        "level",
        "message",
        "order_rowid",
        "order_source_rank",
    }
    assert all(set(event.keys()) == expected_keys for event in events)
    assert all(
        (event["source"] == "plan" and event["task_id"] is None and event["order_source_rank"] == 1)
        or (
            event["source"] == "task"
            and event["task_id"] is not None
            and event["order_source_rank"] == 0
        )
        for event in events
    )


def test_list_plan_watch_events_deterministic_tie_break_and_per_source_limits(db_conn):
    conn = db_conn
    p = make_plan(conn)
    t = create_task(conn, plan_id=p["id"], ordinal=0, title="task", description="a")

    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], p["id"], "INFO", "plan-1", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], p["id"], "INFO", "plan-2", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO task_logs (id, task_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], t["id"], "INFO", "task-1", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO task_logs (id, task_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex[:12], t["id"], "INFO", "task-2", "2026-01-01T00:00:00Z"),
    )
    conn.commit()

    events = list_plan_watch_events(conn, p["id"], limit=2)
    assert len(events) == 2
    assert [e["message"] for e in events] == ["plan-2", "task-2"]
    assert events[0]["timestamp"] == "2026-01-01T00:00:00Z"
    assert events[1]["timestamp"] == "2026-01-01T00:00:00Z"
    assert events[0]["order_rowid"] == 2
    assert events[1]["order_rowid"] == 2
    assert events[0]["order_source_rank"] > events[1]["order_source_rank"]
    assert {"plan-1", "task-1"}.isdisjoint({e["message"] for e in events})


def test_list_plan_watch_events_empty_inputs(db_conn):
    conn = db_conn
    p = make_plan(conn)
    assert list_plan_watch_events(conn, p["id"]) == []
    assert list_plan_watch_events(conn, "does-not-exist") == []
    assert list_plan_watch_events(conn, p["id"], limit=0) == []


# -- plan retry --


def test_reset_plan_for_retry(db_conn):
    """reset_plan_for_retry should reset a failed plan to pending."""
    conn = db_conn
    p = make_plan(conn)
    update_plan_request_status(conn, p["id"], "running")
    set_plan_request_worker(conn, p["id"], pid=123, thread_id="t1")
    update_plan_request_status(conn, p["id"], "failed")
    assert reset_plan_for_retry(conn, p["id"])
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "pending"
    assert found["pid"] is None
    assert found["thread_id"] is None


def test_reset_plan_for_retry_not_failed(db_conn):
    """reset_plan_for_retry should refuse to reset a non-failed plan."""
    conn = db_conn
    p = make_plan(conn)
    assert not reset_plan_for_retry(conn, p["id"])
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "pending"


def test_reset_plan_for_retry_not_found(db_conn):
    """reset_plan_for_retry should return False for nonexistent plan."""
    conn = db_conn
    assert not reset_plan_for_retry(conn, "nonexistent")


# -- doctor stale running remediation --


def test_fail_stale_running_plan_for_doctor(db_conn):
    conn = db_conn
    p = make_plan(conn)
    update_plan_request_status(conn, p["id"], "running")
    set_plan_request_worker(conn, p["id"], pid=123, thread_id="thread-old")

    assert fail_stale_running_plan_for_doctor(
        conn,
        p["id"],
        reason="stale worker pid no longer exists",
    )
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "failed"
    assert found["pid"] is None
    assert found["thread_id"] is None

    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 1
    assert logs[0]["level"] == "ERROR"
    assert "Doctor --fix remediation" in logs[0]["message"]
    assert "stale running plan" in logs[0]["message"]
    assert "stale worker pid no longer exists" in logs[0]["message"]

    history = list_status_history(conn, entity_type="plan", entity_id=p["id"])
    assert any(
        h["old_status"] == "running" and h["new_status"] == "failed" and h["actor"] == "doctor"
        for h in history
    )


def test_fail_stale_running_plan_for_doctor_ineligible_noop(db_conn):
    conn = db_conn
    p = make_plan(conn)
    set_plan_request_worker(conn, p["id"], pid=55, thread_id="thread-pending")

    assert not fail_stale_running_plan_for_doctor(
        conn,
        p["id"],
        reason="should not apply to pending",
    )
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "pending"
    assert found["pid"] == 55
    assert found["thread_id"] == "thread-pending"
    assert list_plan_logs(conn, p["id"]) == []


def test_fail_stale_running_plan_for_doctor_idempotent_repeat(db_conn):
    conn = db_conn
    p = make_plan(conn)
    update_plan_request_status(conn, p["id"], "running")
    set_plan_request_worker(conn, p["id"], pid=777, thread_id="thread-repeat")

    assert fail_stale_running_plan_for_doctor(conn, p["id"], reason="first pass")
    assert not fail_stale_running_plan_for_doctor(conn, p["id"], reason="second pass")
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "failed"
    assert found["pid"] is None
    assert found["thread_id"] is None

    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 1
    assert "first pass" in logs[0]["message"]


def test_fail_stale_plan_for_doctor_awaiting_input(db_conn):
    """Doctor fix handles awaiting_input plans (not just running)."""
    conn = db_conn
    p = make_plan(conn)
    update_plan_request_status(conn, p["id"], "running")
    update_plan_request_status(conn, p["id"], "awaiting_input")
    set_plan_request_worker(conn, p["id"], pid=999, thread_id="thread-wait")

    assert fail_stale_running_plan_for_doctor(
        conn, p["id"], old_status="awaiting_input", reason="stale pid"
    )
    found = get_plan_request(conn, p["id"])
    assert found["status"] == "failed"
    assert found["pid"] is None

    logs = list_plan_logs(conn, p["id"])
    assert len(logs) == 1
    assert "stale awaiting_input plan" in logs[0]["message"]

    history = list_status_history(conn, entity_type="plan", entity_id=p["id"])
    assert any(
        h["old_status"] == "awaiting_input"
        and h["new_status"] == "failed"
        and h["actor"] == "doctor"
        for h in history
    )
