"""Tests for the task database layer."""

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from agm.db import (
    DEFAULT_BACKEND,
    SCHEMA_VERSION,
    VALID_TASK_STATUSES,
    add_channel_message,
    add_plan_question,
    add_project,
    add_task_block,
    add_task_log,
    cancel_tasks_batch,
    claim_task,
    clear_stale_task_git_refs_for_doctor,
    clear_task_git_refs,
    count_tasks_by_plan,
    create_plan_request,
    create_quick_plan_and_task,
    create_session,
    create_task,
    create_tasks_batch,
    fail_stale_running_task_for_doctor,
    finalize_plan_request,
    get_connection,
    get_plan_request,
    get_task,
    get_task_block,
    get_task_rejection_count,
    get_unanswered_question_count,
    get_unresolved_block_count,
    get_unresolved_blocker_summary,
    list_channel_messages,
    list_recent_task_events,
    list_running_task_workers,
    list_status_history,
    list_status_history_timing_rows,
    list_task_blocks,
    list_task_cleanup_candidates,
    list_task_logs,
    list_task_worktree_project_refs,
    list_tasks,
    record_status_change,
    reset_task_for_reexecution,
    reset_task_for_retry,
    resolve_backend,
    resolve_blockers_for_terminal_task,
    resolve_stale_blockers,
    resolve_task_block,
    set_plan_session_id,
    set_task_active_turn_id,
    set_task_failure_reason,
    set_task_model,
    set_task_priority,
    set_task_reviewer_thread_id,
    set_task_thread_id,
    set_task_worker,
    update_plan_enrichment,
    update_plan_request_status,
    update_plan_task_creation_status,
    update_plan_tokens,
    update_task_status,
    update_task_tokens,
)


def get_project_id(conn):
    row = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()
    return row["id"]


def make_plan(conn, prompt="do stuff"):
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt=prompt, caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"test","summary":"s","tasks":[]}')
    return plan


class _TaskCASRaceConnection:
    """Wrap a connection and inject a task status change between read and CAS update."""

    def __init__(
        self,
        conn,
        *,
        task_id: str,
        select_sql_prefix: str,
        update_sql_prefix: str,
        race_sql: str,
        race_params: tuple = (),
    ) -> None:
        self._conn = conn
        self._task_id = task_id
        self._select_sql_prefix = select_sql_prefix
        self._update_sql_prefix = update_sql_prefix
        self._race_sql = race_sql
        self._race_params = race_params
        self._armed = False
        self._raced = False

    def execute(self, sql, params=()):
        if not self._armed and sql.startswith(self._select_sql_prefix):
            self._armed = True
        elif self._armed and not self._raced and sql.startswith(self._update_sql_prefix):
            self._conn.execute(
                self._race_sql,
                self._race_params or (self._task_id,),
            )
            self._raced = True
        return self._conn.execute(sql, params)

    def commit(self):
        return self._conn.commit()


# -- create_task --


def test_create_task(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="Do X", description="Details")
    assert t["id"]
    assert t["plan_id"] == plan["id"]
    assert t["ordinal"] == 0
    assert t["title"] == "Do X"
    assert t["description"] == "Details"
    assert t["status"] == "blocked"
    assert t["priority"] is None
    assert t["files"] is None
    # Token defaults via round-trip
    found = get_task(conn, t["id"])
    assert found["input_tokens"] == 0
    assert found["output_tokens"] == 0


def test_get_connection_migrates_task_token_columns_idempotently():
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
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            output_tokens INTEGER,
            pid INTEGER,
            thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            reviewer_thread_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("plan1", "p1", "legacy prompt", "alice", "cli", "codex"),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description, output_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy desc", None),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "input_tokens" in cols
    assert "output_tokens" in cols
    row = migrated.execute(
        "SELECT input_tokens, output_tokens FROM tasks WHERE id = ?",
        ("task1",),
    ).fetchone()
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    migrated.close()

    reopened = get_connection(db_path)
    row = reopened.execute(
        "SELECT input_tokens, output_tokens FROM tasks WHERE id = ?",
        ("task1",),
    ).fetchone()
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    reopened.close()


def test_create_task_with_optional_fields(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    files = json.dumps(["src/foo.py", "src/bar.py"])
    t = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="Do X",
        description="Details",
        files=files,
        priority="HIGH",
    )
    assert t["files"] == files
    assert t["priority"] == "high"
    found = get_task(conn, t["id"])
    assert found["priority"] == "high"
    assert found["files"] == files


def test_get_task_not_found(db_conn):
    conn = db_conn
    assert get_task(conn, "nonexistent") is None


# -- list_tasks --


def test_list_tasks_by_plan(db_conn):
    conn = db_conn
    plan1 = make_plan(conn, "plan 1")
    plan2 = make_plan(conn, "plan 2")
    create_task(conn, plan_id=plan1["id"], ordinal=0, title="A", description="a")
    create_task(conn, plan_id=plan2["id"], ordinal=0, title="B", description="b")
    tasks = list_tasks(conn, plan_id=plan1["id"])
    assert len(tasks) == 1
    assert tasks[0]["title"] == "A"


def test_list_tasks_by_project(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    project_id = get_project_id(conn)
    tasks = list_tasks(conn, project_id=project_id)
    assert len(tasks) == 1
    assert tasks[0]["title"] == "A"


def test_list_tasks_by_status(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    update_task_status(conn, t1["id"], "ready")
    tasks = list_tasks(conn, status="ready")
    assert len(tasks) == 1
    assert tasks[0]["title"] == "A"


def test_count_tasks_by_plan_counts_by_status_and_total(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    t3 = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="c")
    t4 = create_task(conn, plan_id=plan["id"], ordinal=3, title="D", description="d")

    update_task_status(conn, t2["id"], "ready")
    update_task_status(conn, t3["id"], "completed")
    update_task_status(conn, t4["id"], "failed")
    counts = count_tasks_by_plan(conn, plan["id"])

    expected = {status: 0 for status in VALID_TASK_STATUSES}
    expected.update(
        {
            "blocked": 1,
            "ready": 1,
            "completed": 1,
            "failed": 1,
            "total": 4,
        }
    )
    assert counts == expected


def test_count_tasks_by_plan_for_plan_with_no_tasks_returns_zeros(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    counts = count_tasks_by_plan(conn, plan["id"])

    expected = {status: 0 for status in VALID_TASK_STATUSES}
    expected.update({"total": 0})
    assert counts == expected


def test_list_tasks_by_priority_with_medium_semantics(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    high = create_task(
        conn, plan_id=plan["id"], ordinal=0, title="High", description="a", priority="high"
    )
    medium_null = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Medium Null", description="b", priority="medium"
    )
    legacy_medium = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Legacy Medium", description="c", priority="low"
    )
    low = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="Low", description="d", priority="low"
    )
    conn.execute("UPDATE tasks SET priority = 'medium' WHERE id = ?", (legacy_medium["id"],))
    conn.commit()

    high_rows = list_tasks(conn, priority="high")
    assert [row["id"] for row in high_rows] == [high["id"]]

    low_rows = list_tasks(conn, priority="low")
    assert [row["id"] for row in low_rows] == [low["id"]]

    medium_rows = list_tasks(conn, priority="medium")
    assert [row["id"] for row in medium_rows] == [medium_null["id"], legacy_medium["id"]]


def test_list_task_cleanup_candidates_filters_and_orders(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    project_id = get_project_id(conn)

    # Keep: terminal + both refs
    keep1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    keep2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    keep3 = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="c")
    update_task_status(conn, keep1["id"], "failed")
    update_task_status(conn, keep2["id"], "cancelled")
    update_task_status(conn, keep3["id"], "completed")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ?, updated_at = ? WHERE id = ?",
        ("agm/a", "/tmp/a", "2026-01-01T00:00:01Z", keep1["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ?, updated_at = ? WHERE id = ?",
        ("agm/b", "/tmp/b", "2026-01-01T00:00:02Z", keep2["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ?, updated_at = ? WHERE id = ?",
        ("agm/c", "/tmp/c", "2026-01-01T00:00:03Z", keep3["id"]),
    )

    # Skip: missing refs, wrong status, wrong project
    skip_missing_worktree = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="D", description="d"
    )
    update_task_status(conn, skip_missing_worktree["id"], "completed")
    conn.execute("UPDATE tasks SET branch = ? WHERE id = ?", ("agm/d", skip_missing_worktree["id"]))

    skip_missing_branch = create_task(
        conn, plan_id=plan["id"], ordinal=4, title="E", description="e"
    )
    update_task_status(conn, skip_missing_branch["id"], "failed")
    conn.execute(
        "UPDATE tasks SET worktree = ? WHERE id = ?",
        ("/tmp/e", skip_missing_branch["id"]),
    )

    skip_status = create_task(conn, plan_id=plan["id"], ordinal=5, title="F", description="f")
    update_task_status(conn, skip_status["id"], "running")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/f", "/tmp/f", skip_status["id"]),
    )

    other = add_project(conn, "otherproj", "/tmp/otherproj")
    other_plan = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, other_plan["id"], '{"title":"test","summary":"s","tasks":[]}')
    other_task = create_task(conn, plan_id=other_plan["id"], ordinal=0, title="X", description="x")
    update_task_status(conn, other_task["id"], "failed")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/x", "/tmp/x", other_task["id"]),
    )
    conn.commit()

    found = list_task_cleanup_candidates(conn, project_id)
    found_ids = [t["id"] for t in found]
    assert found_ids == [keep1["id"], keep2["id"], keep3["id"]]
    assert skip_missing_worktree["id"] not in found_ids
    assert skip_missing_branch["id"] not in found_ids
    assert skip_status["id"] not in found_ids
    assert other_task["id"] not in found_ids

    for task_id in found_ids:
        row = conn.execute(
            "SELECT tasks.status, tasks.branch, tasks.worktree, plans.project_id "
            "FROM tasks JOIN plans ON tasks.plan_id = plans.id WHERE tasks.id = ?",
            (task_id,),
        ).fetchone()
        assert row["project_id"] == project_id
        assert row["status"] in {"completed", "cancelled", "failed"}
        assert row["branch"] is not None
        assert row["worktree"] is not None

    row = conn.execute(
        "SELECT status, branch, worktree FROM tasks WHERE id = ?",
        (skip_missing_worktree["id"],),
    ).fetchone()
    assert row["status"] == "completed"
    assert row["branch"] == "agm/d"
    assert row["worktree"] is None

    row = conn.execute(
        "SELECT status, branch, worktree FROM tasks WHERE id = ?",
        (skip_missing_branch["id"],),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["branch"] is None
    assert row["worktree"] == "/tmp/e"

    row = conn.execute(
        "SELECT status, branch, worktree FROM tasks WHERE id = ?",
        (skip_status["id"],),
    ).fetchone()
    assert row["status"] == "running"
    assert row["branch"] == "agm/f"
    assert row["worktree"] == "/tmp/f"


# -- update_task_status --


@pytest.mark.parametrize("record_history", [False, True])
def test_update_task_status(db_conn, record_history):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert update_task_status(conn, t["id"], "ready", record_history=record_history)
    found = get_task(conn, t["id"])
    assert found["status"] == "ready"
    history = list_status_history(conn, entity_type="task", entity_id=t["id"])
    if record_history:
        assert len(history) == 1
        assert history[0]["old_status"] == "blocked"
        assert history[0]["new_status"] == "ready"
        assert history[0]["actor"] is None
    else:
        assert history == []


def test_update_task_status_noop_when_already_target_status(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert not update_task_status(conn, t["id"], "blocked", record_history=True)
    found = get_task(conn, t["id"])
    assert found["status"] == "blocked"
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


def test_update_task_status_cas_failure_returns_false_and_no_history(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    raced_conn = _TaskCASRaceConnection(
        conn,
        task_id=t["id"],
        select_sql_prefix="SELECT status, actor FROM tasks WHERE id = ?",
        update_sql_prefix="UPDATE tasks SET status = ?, updated_at = strftime(",
        race_sql=(
            "UPDATE tasks SET status = 'running', updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
        ),
    )
    assert not update_task_status(raced_conn, t["id"], "ready", record_history=True)
    found = get_task(conn, t["id"])
    assert found["status"] == "running"
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


def test_update_task_status_invalid(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    with pytest.raises(ValueError, match="Invalid task status"):
        update_task_status(conn, t["id"], "bogus")


def test_update_task_status_cancelled(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert update_task_status(conn, t["id"], "cancelled")
    found = get_task(conn, t["id"])
    assert found["status"] == "cancelled"


# -- task status history --


def test_record_status_change_task_validates_status_set(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    with pytest.raises(ValueError, match="Invalid new_status 'finalized'"):
        record_status_change(
            conn,
            entity_type="task",
            entity_id=t["id"],
            old_status="blocked",
            new_status="finalized",
        )
    with pytest.raises(ValueError, match="Invalid old_status 'finalized'"):
        record_status_change(
            conn,
            entity_type="task",
            entity_id=t["id"],
            old_status="finalized",
            new_status="running",
        )


def test_task_status_history_chronological_order_and_timing_rows(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status=None,
        new_status="blocked",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="blocked",
        new_status="ready",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="ready",
        new_status="running",
        created_at="2026-01-01T00:00:03Z",
    )
    conn.commit()

    history = list_status_history(conn, entity_type="task", entity_id=t["id"])
    assert [row["new_status"] for row in history] == ["blocked", "ready", "running"]
    assert history[0]["id"] < history[1]["id"]
    assert history[0]["created_at"] == history[1]["created_at"] == "2026-01-01T00:00:00Z"

    timing_rows = list_status_history_timing_rows(conn, entity_type="task", entity_id=t["id"])
    assert [row["duration_seconds"] for row in timing_rows] == [0, 3, None]
    assert timing_rows[0]["next_created_at"] == "2026-01-01T00:00:00Z"
    assert timing_rows[1]["next_created_at"] == "2026-01-01T00:00:03Z"
    assert timing_rows[2]["next_created_at"] is None


def test_get_task_rejection_count_counts_review_rejected_transitions(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="running",
        new_status="review",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="review",
        new_status="rejected",
        created_at="2026-01-01T00:00:01Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="ready",
        new_status="review",
        created_at="2026-01-01T00:00:02Z",
    )
    conn.commit()
    assert get_task_rejection_count(conn, t["id"]) == 1


def test_get_task_rejection_count_zero_when_no_review_rejections(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    record_status_change(
        conn,
        entity_type="task",
        entity_id=t["id"],
        old_status="running",
        new_status="ready",
        created_at="2026-01-01T00:00:00Z",
    )
    conn.commit()
    assert get_task_rejection_count(conn, t["id"]) == 0


def test_set_task_priority(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    assert set_task_priority(conn, t["id"], "high")
    assert get_task(conn, t["id"])["priority"] == "high"

    assert set_task_priority(conn, t["id"], "low")
    assert get_task(conn, t["id"])["priority"] == "low"

    assert set_task_priority(conn, t["id"], "medium")
    assert get_task(conn, t["id"])["priority"] is None

    assert set_task_priority(conn, t["id"], None)
    assert get_task(conn, t["id"])["priority"] is None

    assert not set_task_priority(conn, "missing", "high")


@pytest.mark.parametrize(
    "invalid_priority",
    ["", "   ", "urgent", 123, True, math.nan, math.inf, -math.inf],
)
def test_task_priority_invalid_values(db_conn, invalid_priority):
    conn = db_conn
    plan = make_plan(conn)
    project_id = get_project_id(conn)

    with pytest.raises(ValueError, match="Invalid task priority"):
        create_task(
            conn,
            plan_id=plan["id"],
            ordinal=0,
            title="A",
            description="a",
            priority=invalid_priority,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Invalid task priority"):
        set_task_priority(conn, "any-task", invalid_priority)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid task priority"):
        list_tasks(conn, priority=invalid_priority)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid task priority"):
        create_quick_plan_and_task(
            conn,
            project_id=project_id,
            prompt="Quick fix",
            title="Quick fix",
            description="Do it fast",
            caller="cli",
            backend="codex",
            priority=invalid_priority,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Invalid task priority"):
        create_tasks_batch(
            conn,
            plan["id"],
            [
                {
                    "ordinal": 0,
                    "title": "Invalid",
                    "description": "d",
                    "status": "ready",
                    "priority": invalid_priority,
                }
            ],
        )


def test_cancel_tasks_batch(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    update_task_status(conn, t2["id"], "ready")

    cancellations = [
        {"task_id": t1["id"], "reason": "superseded"},
        {"task_id": t2["id"], "reason": "duplicate"},
    ]
    result = cancel_tasks_batch(conn, cancellations)
    assert result["cancelled"] == 2

    assert get_task(conn, t1["id"])["status"] == "cancelled"
    assert get_task(conn, t2["id"])["status"] == "cancelled"
    assert list_status_history(conn, entity_type="task", entity_id=t1["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t2["id"]) == []

    # Verify logs were created
    logs1 = list_task_logs(conn, t1["id"])
    assert len(logs1) == 1
    assert "superseded" in logs1[0]["message"]
    logs2 = list_task_logs(conn, t2["id"])
    assert len(logs2) == 1
    assert "duplicate" in logs2[0]["message"]


def test_cancel_tasks_batch_record_history_only_for_changed_rows(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    pending_task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    ready_task = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    running_task = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="c")
    update_task_status(conn, ready_task["id"], "ready")
    update_task_status(conn, running_task["id"], "running")

    result = cancel_tasks_batch(
        conn,
        [
            {"task_id": pending_task["id"], "reason": "obsolete"},
            {"task_id": ready_task["id"], "reason": "duplicate"},
            {"task_id": running_task["id"], "reason": "should noop"},
        ],
        record_history=True,
    )
    assert result["cancelled"] == 2
    assert get_task(conn, pending_task["id"])["status"] == "cancelled"
    assert get_task(conn, ready_task["id"])["status"] == "cancelled"
    assert get_task(conn, running_task["id"])["status"] == "running"

    pending_history = list_status_history(conn, entity_type="task", entity_id=pending_task["id"])
    ready_history = list_status_history(conn, entity_type="task", entity_id=ready_task["id"])
    running_history = list_status_history(conn, entity_type="task", entity_id=running_task["id"])
    assert [(row["old_status"], row["new_status"]) for row in pending_history] == [
        ("blocked", "cancelled")
    ]
    assert [(row["old_status"], row["new_status"]) for row in ready_history] == [
        ("ready", "cancelled")
    ]
    assert running_history == []

    assert len(list_task_logs(conn, pending_task["id"])) == 1
    assert len(list_task_logs(conn, ready_task["id"])) == 1
    # Skipped tasks get an INFO log explaining why
    running_logs = list_task_logs(conn, running_task["id"])
    assert len(running_logs) == 1
    assert "Batch cancel skipped" in running_logs[0]["message"]


def test_cancel_tasks_batch_cas_failure_skips_log_and_history(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    raced_conn = _TaskCASRaceConnection(
        conn,
        task_id=t["id"],
        select_sql_prefix="SELECT status, actor FROM tasks WHERE id = ?",
        update_sql_prefix="UPDATE tasks SET status = 'cancelled', ",
        race_sql=(
            "UPDATE tasks SET status = 'running', updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
        ),
    )
    result = cancel_tasks_batch(
        raced_conn,
        [{"task_id": t["id"], "reason": "race"}],
        record_history=True,
    )
    assert result["cancelled"] == 0
    found = get_task(conn, t["id"])
    assert found["status"] == "running"
    assert list_task_logs(conn, t["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


def test_cancel_tasks_batch_skips_non_cancellable(db_conn):
    """Only blocked/ready tasks can be cancelled."""
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "running")

    result = cancel_tasks_batch(conn, [{"task_id": t["id"], "reason": "stale"}])
    assert result["cancelled"] == 0
    assert get_task(conn, t["id"])["status"] == "running"
    # Skipped tasks get an INFO log explaining why
    logs = list_task_logs(conn, t["id"])
    assert len(logs) == 1
    assert "Batch cancel skipped" in logs[0]["message"]
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


# -- task claim --


def test_claim_task(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")

    assert claim_task(conn, t["id"], actor="alice", caller="cli")
    found = get_task(conn, t["id"])
    assert found["status"] == "running"
    assert found["actor"] == "alice"
    assert found["caller"] == "cli"

    # Verify claim is logged
    logs = list_task_logs(conn, t["id"])
    assert len(logs) == 1
    assert "alice" in logs[0]["message"]
    assert "cli" in logs[0]["message"]
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


def test_claim_task_record_history_on_transition(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")

    assert claim_task(conn, t["id"], actor="alice", caller="cli", record_history=True)
    history = list_status_history(conn, entity_type="task", entity_id=t["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "ready"
    assert history[0]["new_status"] == "running"
    assert history[0]["actor"] == "alice"


def test_claim_task_not_ready(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    # blocked, not ready
    assert not claim_task(conn, t["id"], actor="alice", caller="cli")
    found = get_task(conn, t["id"])
    assert found["status"] == "blocked"
    assert list_task_logs(conn, t["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


def test_claim_task_default_actor(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")

    assert claim_task(conn, t["id"], caller="cli")
    found = get_task(conn, t["id"])
    assert found["actor"] is not None  # defaults to $USER


def test_claim_task_cas_failure_returns_false_and_keeps_logs_and_history_unchanged(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")
    raced_conn = _TaskCASRaceConnection(
        conn,
        task_id=t["id"],
        select_sql_prefix="SELECT status FROM tasks WHERE id = ?",
        update_sql_prefix="UPDATE tasks SET status = 'running', actor = ?, caller = ?, ",
        race_sql=(
            "UPDATE tasks SET status = 'running', actor = 'other', caller = 'cli', updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
        ),
    )

    assert not claim_task(raced_conn, t["id"], actor="alice", caller="cli", record_history=True)
    found = get_task(conn, t["id"])
    assert found["status"] == "running"
    assert found["actor"] == "other"
    assert found["caller"] == "cli"
    assert list_task_logs(conn, t["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t["id"]) == []


# -- task retry --


def test_reset_task_for_retry(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "failed")
    conn.execute(
        "UPDATE tasks SET pid = ?, thread_id = ?, actor = ?, caller = ? WHERE id = ?",
        (1234, "thread-old", "alice", "cli", t["id"]),
    )
    conn.commit()

    assert reset_task_for_retry(conn, t["id"])
    found = get_task(conn, t["id"])
    assert found["status"] == "blocked"
    assert found["pid"] is None
    assert found["thread_id"] is None
    assert found["actor"] is None
    assert found["caller"] is None


def test_reset_task_for_retry_not_failed(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert not reset_task_for_retry(conn, t["id"])
    found = get_task(conn, t["id"])
    assert found["status"] == "blocked"


def test_reset_task_for_retry_not_found(db_conn):
    conn = db_conn
    assert not reset_task_for_retry(conn, "nonexistent")


def test_fail_stale_running_task_for_doctor(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")
    claim_task(conn, t["id"], actor="alice", caller="cli")
    set_task_worker(conn, t["id"], pid=101, thread_id="thread-old")

    assert fail_stale_running_task_for_doctor(
        conn,
        t["id"],
        reason="worker process disappeared",
    )
    found = get_task(conn, t["id"])
    assert found["status"] == "failed"
    assert found["pid"] is None
    assert found["thread_id"] is None
    assert found["actor"] == "alice"
    assert found["caller"] == "cli"

    logs = list_task_logs(conn, t["id"])
    assert len(logs) == 2
    assert logs[1]["level"] == "ERROR"
    assert "Doctor --fix remediation" in logs[1]["message"]
    assert "stale running task" in logs[1]["message"]
    assert "worker process disappeared" in logs[1]["message"]

    history = list_status_history(conn, entity_type="task", entity_id=t["id"])
    assert any(
        h["old_status"] == "running" and h["new_status"] == "failed" and h["actor"] == "doctor"
        for h in history
    )

    # Idempotent: second call is a no-op since task is already failed
    assert not fail_stale_running_task_for_doctor(conn, t["id"], reason="second")
    assert len(list_task_logs(conn, t["id"])) == 2  # no new logs


def test_fail_stale_running_task_for_doctor_ineligible_noop(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "failed")
    conn.execute(
        "UPDATE tasks SET pid = ?, thread_id = ? WHERE id = ?",
        (444, "already-failed-thread", t["id"]),
    )
    conn.commit()

    assert not fail_stale_running_task_for_doctor(
        conn,
        t["id"],
        reason="should not apply",
    )
    found = get_task(conn, t["id"])
    assert found["status"] == "failed"
    assert found["pid"] == 444
    assert found["thread_id"] == "already-failed-thread"
    assert list_task_logs(conn, t["id"]) == []


def test_fail_stale_task_for_doctor_review_status(db_conn):
    """Doctor fix handles review tasks (not just running)."""
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, t["id"], "ready")
    claim_task(conn, t["id"], actor="alice", caller="cli")
    update_task_status(conn, t["id"], "review")
    set_task_worker(conn, t["id"], pid=202)

    assert fail_stale_running_task_for_doctor(
        conn, t["id"], old_status="review", reason="reviewer pid gone"
    )
    found = get_task(conn, t["id"])
    assert found["status"] == "failed"
    assert found["pid"] is None

    logs = list_task_logs(conn, t["id"])
    doctor_logs = [entry for entry in logs if "Doctor --fix" in entry["message"]]
    assert len(doctor_logs) == 1
    assert "stale review task" in doctor_logs[0]["message"]

    history = list_status_history(conn, entity_type="task", entity_id=t["id"])
    assert any(
        h["old_status"] == "review" and h["new_status"] == "failed" and h["actor"] == "doctor"
        for h in history
    )


def test_clear_task_git_refs(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    db_path = conn.execute("PRAGMA database_list").fetchone()["file"]
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ?, updated_at = ? WHERE id = ?",
        ("agm/test", "/tmp/wt", "2000-01-01T00:00:00Z", t["id"]),
    )
    conn.commit()

    assert clear_task_git_refs(conn, t["id"])
    cleared = conn.execute(
        "SELECT branch, worktree, updated_at FROM tasks WHERE id = ?",
        (t["id"],),
    ).fetchone()
    assert cleared["branch"] is None
    assert cleared["worktree"] is None
    assert cleared["updated_at"] != "2000-01-01T00:00:00Z"

    first_updated_at = cleared["updated_at"]
    assert not clear_task_git_refs(conn, t["id"])
    after_second_clear = conn.execute(
        "SELECT branch, worktree, updated_at FROM tasks WHERE id = ?",
        (t["id"],),
    ).fetchone()
    assert after_second_clear["branch"] is None
    assert after_second_clear["worktree"] is None
    assert after_second_clear["updated_at"] == first_updated_at

    # Not-found case returns False
    assert not clear_task_git_refs(conn, "nonexistent")

    conn.close()

    reopened = get_connection(Path(db_path))
    persisted = reopened.execute(
        "SELECT branch, worktree, updated_at FROM tasks WHERE id = ?",
        (t["id"],),
    ).fetchone()
    assert persisted["branch"] is None
    assert persisted["worktree"] is None
    assert persisted["updated_at"] == first_updated_at
    reopened.close()


def test_clear_stale_task_git_refs_for_doctor(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/stale", "/tmp/wt-stale", t["id"]),
    )
    conn.commit()

    result = clear_stale_task_git_refs_for_doctor(
        conn,
        t["id"],
        reason="orphan cleanup removed worktree",
    )
    assert result == {"changed": True, "result": "changed"}
    found = get_task(conn, t["id"])
    assert found["branch"] is None
    assert found["worktree"] is None

    logs = list_task_logs(conn, t["id"])
    assert len(logs) == 1
    assert logs[0]["level"] == "INFO"
    assert "Doctor --fix remediation" in logs[0]["message"]
    assert "orphan cleanup removed worktree" in logs[0]["message"]


def test_clear_stale_task_git_refs_for_doctor_not_changed_and_idempotent(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    first = clear_stale_task_git_refs_for_doctor(conn, t["id"], reason="already clear")
    second = clear_stale_task_git_refs_for_doctor(conn, t["id"], reason="repeat")
    assert first == {"changed": False, "result": "not-changed"}
    assert second == {"changed": False, "result": "not-changed"}
    assert list_task_logs(conn, t["id"]) == []


# -- set_task_worker --


def test_set_task_worker(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert set_task_worker(conn, t["id"], pid=1234, thread_id="thread-abc")
    found = get_task(conn, t["id"])
    assert found["pid"] == 1234
    assert found["thread_id"] == "thread-abc"


def test_set_task_worker_no_thread(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert set_task_worker(conn, t["id"], pid=5678)
    found = get_task(conn, t["id"])
    assert found["pid"] == 5678
    assert found["thread_id"] is None


def test_set_task_model(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    assert set_task_model(conn, t["id"], "gpt-5-work")
    found = get_task(conn, t["id"])
    assert found["model"] == "gpt-5-work"

    assert set_task_model(conn, t["id"], None)
    found = get_task(conn, t["id"])
    assert found["model"] is None

    assert not set_task_model(conn, "missing", "gpt-5")


def test_set_task_worker_preserves_existing_thread_id(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    # Simulate executor setting thread_id
    set_task_thread_id(conn, t["id"], "executor-thread-abc")
    # Simulate reviewer calling set_task_worker without thread_id
    assert set_task_worker(conn, t["id"], pid=9999)
    found = get_task(conn, t["id"])
    assert found["pid"] == 9999
    assert found["thread_id"] == "executor-thread-abc"  # preserved, not clobbered


def test_list_running_task_workers_filters_running_and_pid(db_conn):
    conn = db_conn
    main_project_id = get_project_id(conn)
    plan = make_plan(conn)

    running_with_pid = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, running_with_pid["id"], "running")
    set_task_worker(conn, running_with_pid["id"], pid=111)

    running_no_pid = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    update_task_status(conn, running_no_pid["id"], "running")

    ready_with_pid = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="c")
    update_task_status(conn, ready_with_pid["id"], "ready")
    set_task_worker(conn, ready_with_pid["id"], pid=222)

    review_with_pid = create_task(conn, plan_id=plan["id"], ordinal=3, title="D", description="d")
    update_task_status(conn, review_with_pid["id"], "review")
    set_task_worker(conn, review_with_pid["id"], pid=333)

    other = add_project(conn, "otherproj-running", "/tmp/otherproj-running")
    other_plan = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other running",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, other_plan["id"], '{"title":"other","summary":"s","tasks":[]}')
    other_running = create_task(
        conn, plan_id=other_plan["id"], ordinal=0, title="Other", description="x"
    )
    update_task_status(conn, other_running["id"], "running")
    set_task_worker(conn, other_running["id"], pid=444)

    all_running = list_running_task_workers(conn)
    all_ids = {task["id"] for task in all_running}
    assert running_with_pid["id"] in all_ids
    assert other_running["id"] in all_ids
    assert running_no_pid["id"] not in all_ids
    assert ready_with_pid["id"] not in all_ids
    assert review_with_pid["id"] not in all_ids

    main_only = list_running_task_workers(conn, project_id=main_project_id)
    main_ids = {task["id"] for task in main_only}
    assert main_ids == {running_with_pid["id"]}
    assert main_only[0]["project_dir"] == "/tmp/testproj"


def test_list_task_worktree_project_refs_correlates_project_metadata(db_conn):
    conn = db_conn
    main_project_id = get_project_id(conn)
    plan = make_plan(conn)

    with_both_refs = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    with_branch_only = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    no_refs = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="c")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/a", "/tmp/a", with_both_refs["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/b", None, with_branch_only["id"]),
    )

    other = add_project(conn, "otherproj-refs", "/tmp/otherproj-refs")
    other_plan = create_plan_request(
        conn,
        project_id=other["id"],
        prompt="other refs",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, other_plan["id"], '{"title":"other","summary":"s","tasks":[]}')
    other_task = create_task(conn, plan_id=other_plan["id"], ordinal=0, title="X", description="x")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/x", "/tmp/x", other_task["id"]),
    )
    conn.commit()

    all_rows = list_task_worktree_project_refs(conn)
    all_ids = {row["id"] for row in all_rows}
    assert with_both_refs["id"] in all_ids
    assert with_branch_only["id"] in all_ids
    assert other_task["id"] in all_ids
    assert no_refs["id"] not in all_ids

    branch_only_row = next(row for row in all_rows if row["id"] == with_branch_only["id"])
    assert branch_only_row["project_id"] == main_project_id
    assert branch_only_row["project_dir"] == "/tmp/testproj"
    assert branch_only_row["branch"] == "agm/b"
    assert branch_only_row["worktree"] is None

    filtered = list_task_worktree_project_refs(conn, project_id=main_project_id)
    filtered_ids = {row["id"] for row in filtered}
    assert filtered_ids == {with_both_refs["id"], with_branch_only["id"]}
    assert other_task["id"] not in filtered_ids


# -- set_task_thread_id --


def test_set_task_thread_id(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert set_task_thread_id(conn, t["id"], "thread-xyz")
    found = get_task(conn, t["id"])
    assert found["thread_id"] == "thread-xyz"


def test_set_task_active_turn_id(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    assert set_task_active_turn_id(conn, t["id"], "turn-123")
    found = get_task(conn, t["id"])
    assert found["active_turn_id"] == "turn-123"
    assert found["active_turn_started_at"] is not None

    assert set_task_active_turn_id(conn, t["id"], None)
    cleared = get_task(conn, t["id"])
    assert cleared["active_turn_id"] is None
    assert cleared["active_turn_started_at"] is None


# -- task blocks --


def test_add_task_block_internal(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    block = add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t1["id"])
    assert block["task_id"] == t2["id"]
    assert block["blocked_by_task_id"] == t1["id"]
    assert block["resolved"] == 0


def test_add_task_block_external(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    block = add_task_block(
        conn,
        task_id=t["id"],
        external_factor="API key",
        reason="Need production credentials",
    )
    assert block["external_factor"] == "API key"
    assert block["reason"] == "Need production credentials"


def test_resolve_task_block(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    block = add_task_block(conn, task_id=t["id"], external_factor="API key", reason="need it")
    assert resolve_task_block(conn, block["id"])
    found = get_task_block(conn, block["id"])
    assert found["resolved"] == 1
    assert found["resolved_at"] is not None


def test_resolve_task_block_already_resolved(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    block = add_task_block(conn, task_id=t["id"], external_factor="API key", reason="need it")
    assert resolve_task_block(conn, block["id"])
    assert not resolve_task_block(conn, block["id"])


def test_list_task_blocks(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t1["id"])
    add_task_block(conn, task_id=t2["id"], external_factor="Design", reason="pending")
    blocks = list_task_blocks(conn, t2["id"])
    assert len(blocks) == 2


def test_list_task_blocks_unresolved_only(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    b1 = add_task_block(conn, task_id=t["id"], external_factor="Key", reason="need")
    add_task_block(conn, task_id=t["id"], external_factor="Review", reason="waiting")
    resolve_task_block(conn, b1["id"])
    blocks = list_task_blocks(conn, t["id"], unresolved_only=True)
    assert len(blocks) == 1
    assert blocks[0]["external_factor"] == "Review"


def test_list_task_blocks_by_plan(db_conn):
    conn = db_conn
    plan1 = make_plan(conn, "plan 1")
    plan2 = make_plan(conn, "plan 2")
    t1 = create_task(conn, plan_id=plan1["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan2["id"], ordinal=0, title="B", description="b")
    add_task_block(conn, task_id=t1["id"], external_factor="Key", reason="need")
    add_task_block(conn, task_id=t2["id"], external_factor="Review", reason="waiting")
    blocks = list_task_blocks(conn, plan_id=plan1["id"])
    assert len(blocks) == 1
    assert blocks[0]["task_id"] == t1["id"]


def test_list_task_blocks_by_project(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    add_task_block(conn, task_id=t["id"], external_factor="Key", reason="need")
    pid = get_project_id(conn)
    blocks = list_task_blocks(conn, project_id=pid)
    assert len(blocks) == 1
    assert blocks[0]["task_id"] == t["id"]


def test_get_unresolved_block_count(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    add_task_block(conn, task_id=t["id"], external_factor="Key", reason="need")
    add_task_block(conn, task_id=t["id"], external_factor="Review", reason="waiting")
    assert get_unresolved_block_count(conn, t["id"]) == 2


def test_get_unresolved_block_count_after_resolve(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    b = add_task_block(conn, task_id=t["id"], external_factor="Key", reason="need")
    resolve_task_block(conn, b["id"])
    assert get_unresolved_block_count(conn, t["id"]) == 0


def test_blocker_summary_with_dead_blockers(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    blocker = create_task(conn, plan_id=plan["id"], ordinal=0, title="Blocker", description="b")
    update_task_status(conn, blocker["id"], "cancelled")
    t = create_task(conn, plan_id=plan["id"], ordinal=1, title="Blocked", description="d")
    add_task_block(conn, task_id=t["id"], blocked_by_task_id=blocker["id"], reason="dep")
    summary = get_unresolved_blocker_summary(conn, t["id"])
    assert summary["total"] == 1
    assert summary["dead"] == 1
    assert summary["external"] == 0


def test_blocker_summary_mixed(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    alive = create_task(conn, plan_id=plan["id"], ordinal=0, title="Alive", description="a")
    dead = create_task(conn, plan_id=plan["id"], ordinal=1, title="Dead", description="d")
    update_task_status(conn, dead["id"], "failed")
    t = create_task(conn, plan_id=plan["id"], ordinal=2, title="Blocked", description="b")
    add_task_block(conn, task_id=t["id"], blocked_by_task_id=alive["id"], reason="dep")
    add_task_block(conn, task_id=t["id"], blocked_by_task_id=dead["id"], reason="dep")
    add_task_block(conn, task_id=t["id"], external_factor="API key", reason="need key")
    summary = get_unresolved_blocker_summary(conn, t["id"])
    assert summary["total"] == 3
    assert summary["dead"] == 1
    assert summary["external"] == 1


def test_blocker_summary_no_blockers(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="Free", description="f")
    summary = get_unresolved_blocker_summary(conn, t["id"])
    assert summary == {"total": 0, "dead": 0, "external": 0}


# -- task logs --


def test_add_and_list_task_logs(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    entry = add_task_log(conn, task_id=t["id"], level="INFO", message="a")
    assert entry["task_id"] == t["id"]
    assert entry["level"] == "INFO"
    add_task_log(conn, task_id=t["id"], level="ERROR", message="b")
    logs = list_task_logs(conn, t["id"])
    assert len(logs) == 2


def test_list_task_logs_by_level(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    add_task_log(conn, task_id=t["id"], level="INFO", message="a")
    add_task_log(conn, task_id=t["id"], level="ERROR", message="b")
    logs = list_task_logs(conn, t["id"], level="ERROR")
    assert len(logs) == 1
    assert logs[0]["message"] == "b"


def test_list_recent_task_events_task_scope_newest_first_rowid_limit_and_excludes_other_tasks(
    db_conn,
):
    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    other_task = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    add_task_log(conn, task_id=task["id"], level="INFO", message="first")
    add_task_log(conn, task_id=task["id"], level="ERROR", message="second")
    conn.execute(
        "UPDATE task_logs SET created_at = ? WHERE task_id = ? AND message = ?",
        ("2026-01-01T00:00:03Z", task["id"], "first"),
    )
    conn.execute(
        "UPDATE task_logs SET created_at = ? WHERE task_id = ? AND message = ?",
        ("2026-01-01T00:00:01Z", task["id"], "second"),
    )
    add_task_log(conn, task_id=other_task["id"], level="INFO", message="other-task-newest")
    conn.commit()

    rows = list_recent_task_events(conn, task_id=task["id"], limit=2)
    assert len(rows) == 2
    assert [row["message"] for row in rows] == ["second", "first"]
    assert all(row["task_id"] == task["id"] for row in rows)
    assert rows[0]["source_rowid"] > rows[1]["source_rowid"]
    assert rows[0]["created_at"] < rows[1]["created_at"]
    expected_keys = {"task_id", "created_at", "message", "level", "source", "source_rowid"}
    assert set(rows[0].keys()) == expected_keys


def test_list_recent_task_events_plan_scope_filters_to_plan_and_optional_visible_statuses(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    other_plan = make_plan(conn, "other plan")
    active = create_task(conn, plan_id=plan["id"], ordinal=0, title="active", description="a")
    done = create_task(conn, plan_id=plan["id"], ordinal=1, title="done", description="b")
    other = create_task(conn, plan_id=other_plan["id"], ordinal=0, title="other", description="c")
    update_task_status(conn, active["id"], "running")
    update_task_status(conn, done["id"], "completed")
    update_task_status(conn, other["id"], "running")
    add_task_log(conn, task_id=active["id"], level="INFO", message="active msg")
    add_task_log(conn, task_id=other["id"], level="INFO", message="other msg")
    add_task_log(conn, task_id=done["id"], level="INFO", message="done msg")

    all_rows = list_recent_task_events(conn, plan_id=plan["id"], limit=10)
    assert [row["message"] for row in all_rows] == ["done msg", "active msg"]
    assert {row["task_id"] for row in all_rows} == {active["id"], done["id"]}

    active_only = list_recent_task_events(
        conn,
        plan_id=plan["id"],
        limit=10,
        visible_statuses=["running", "running"],
    )
    assert [row["message"] for row in active_only] == ["active msg"]

    done_only = list_recent_task_events(
        conn,
        plan_id=plan["id"],
        limit=10,
        visible_statuses=("completed",),
    )
    assert [row["message"] for row in done_only] == ["done msg"]


def test_list_recent_task_events_project_scope_joins_plans_and_excludes_other_projects(db_conn):
    conn = db_conn
    main_project_id = get_project_id(conn)
    other_project = add_project(conn, "otherproj", "/tmp/otherproj")

    plan_a = make_plan(conn, "plan A")
    plan_b = make_plan(conn, "plan B")
    other_plan = create_plan_request(
        conn,
        project_id=other_project["id"],
        prompt="other plan",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, other_plan["id"], '{"title":"other","summary":"s","tasks":[]}')

    task_a = create_task(conn, plan_id=plan_a["id"], ordinal=0, title="A", description="a")
    task_b = create_task(conn, plan_id=plan_b["id"], ordinal=0, title="B", description="b")
    task_other = create_task(conn, plan_id=other_plan["id"], ordinal=0, title="O", description="o")
    add_task_log(conn, task_id=task_a["id"], level="INFO", message="main-old")
    add_task_log(conn, task_id=task_other["id"], level="INFO", message="other-newest")
    add_task_log(conn, task_id=task_b["id"], level="INFO", message="main-new")

    rows = list_recent_task_events(conn, project_id=main_project_id, limit=2)
    assert len(rows) == 2
    assert [row["message"] for row in rows] == ["main-new", "main-old"]
    assert rows[0]["source_rowid"] > rows[1]["source_rowid"]
    assert {row["task_id"] for row in rows} == {task_a["id"], task_b["id"]}


def test_list_recent_task_events_empty_when_no_rows_match(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    project_id = get_project_id(conn)

    assert list_recent_task_events(conn, task_id=task["id"], limit=5) == []
    assert list_recent_task_events(conn, plan_id=plan["id"], limit=5) == []
    assert list_recent_task_events(conn, project_id=project_id, limit=5) == []

    update_task_status(conn, task["id"], "running")
    add_task_log(conn, task_id=task["id"], level="INFO", message="running msg")
    assert (
        list_recent_task_events(
            conn,
            plan_id=plan["id"],
            limit=5,
            visible_statuses=["completed"],
        )
        == []
    )


def test_list_recent_task_events_requires_exactly_one_scope(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    project_id = get_project_id(conn)

    with pytest.raises(ValueError, match="Exactly one of task_id, plan_id, or project_id"):
        list_recent_task_events(conn, limit=1)
    with pytest.raises(ValueError, match="Exactly one of task_id, plan_id, or project_id"):
        list_recent_task_events(conn, task_id=task["id"], plan_id=plan["id"], limit=1)
    with pytest.raises(ValueError, match="Exactly one of task_id, plan_id, or project_id"):
        list_recent_task_events(conn, plan_id=plan["id"], project_id=project_id, limit=1)


@pytest.mark.parametrize(
    "task_scope, expected_error",
    [
        ("", "task_id must be a non-empty string when provided"),
        ("   ", "task_id must be a non-empty string when provided"),
        (123, "task_id must be a non-empty string when provided"),
    ],
)
def test_list_recent_task_events_invalid_task_scope_values(db_conn, task_scope, expected_error):
    conn = db_conn
    with pytest.raises(ValueError, match=expected_error):
        list_recent_task_events(conn, task_id=task_scope, limit=1)


@pytest.mark.parametrize(
    "limit",
    [
        -1,
        0,
        None,
        "",
        [],
        1.5,
        math.nan,
        math.inf,
        -math.inf,
        True,
    ],
)
def test_list_recent_task_events_invalid_limit_values(db_conn, limit):
    conn = db_conn
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        list_recent_task_events(conn, task_id="abc123", limit=limit)


def test_list_recent_task_events_visible_statuses_validation(db_conn):
    conn = db_conn
    with pytest.raises(ValueError, match="visible_statuses must be a collection of status strings"):
        list_recent_task_events(conn, task_id="abc123", limit=1, visible_statuses="running")
    with pytest.raises(ValueError, match="visible_statuses must contain only non-empty strings"):
        list_recent_task_events(conn, task_id="abc123", limit=1, visible_statuses=["running", ""])

    assert list_recent_task_events(conn, task_id="abc123", limit=1, visible_statuses=[]) == []


def test_list_recent_task_events_accepts_large_limit(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    add_task_log(conn, task_id=task["id"], level="INFO", message="only message")

    rows = list_recent_task_events(conn, task_id=task["id"], limit=10**9)
    assert len(rows) == 1
    assert rows[0]["message"] == "only message"


# -- plan task creation status --


def test_update_plan_task_creation_status(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    assert update_plan_task_creation_status(conn, plan["id"], "pending")
    row = conn.execute(
        "SELECT task_creation_status FROM plans WHERE id = ?", (plan["id"],)
    ).fetchone()
    assert row["task_creation_status"] == "pending"


def test_update_plan_task_creation_status_rejects_invalid(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    with pytest.raises(ValueError, match="Invalid task_creation_status"):
        update_plan_task_creation_status(conn, plan["id"], "bogus")


# -- plan terminal guard --


def test_plan_terminal_guard_blocks_transition_from_cancelled(db_conn):
    """Terminal guard should prevent transitions out of cancelled status."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "cancelled")
    # Try to transition cancelled → running (simulates worker race)
    result = update_plan_request_status(conn, plan["id"], "running")
    assert result is False
    row = conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert row["status"] == "cancelled"


def test_plan_terminal_guard_blocks_transition_from_finalized(db_conn):
    """Terminal guard should prevent transitions out of finalized status."""
    conn = db_conn
    plan = make_plan(conn)  # make_plan finalizes
    result = update_plan_request_status(conn, plan["id"], "running")
    assert result is False
    row = conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert row["status"] == "finalized"


def test_plan_cancelled_sets_finished_at(db_conn):
    """Transitioning a plan to cancelled should set finished_at."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "cancelled")
    row = conn.execute("SELECT finished_at FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert row["finished_at"] is not None


# -- force_cancel_plan --


def test_force_cancel_plan_from_finalized(db_conn):
    """force_cancel_plan should cancel a finalized plan."""
    from agm.db import force_cancel_plan

    conn = db_conn
    plan = make_plan(conn)  # make_plan finalizes
    old = force_cancel_plan(conn, plan["id"])
    assert old == "finalized"
    row = conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert row["status"] == "cancelled"


def test_force_cancel_plan_from_failed(db_conn):
    """force_cancel_plan should cancel a failed plan."""
    from agm.db import force_cancel_plan

    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="t", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "running")
    update_plan_request_status(conn, plan["id"], "failed")
    old = force_cancel_plan(conn, plan["id"])
    assert old == "failed"
    row = conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert row["status"] == "cancelled"


def test_force_cancel_plan_already_cancelled(db_conn):
    """force_cancel_plan returns None for already-cancelled plans."""
    from agm.db import force_cancel_plan

    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="t", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "cancelled")
    assert force_cancel_plan(conn, plan["id"]) is None


def test_force_cancel_plan_not_found(db_conn):
    """force_cancel_plan returns None for nonexistent plan."""
    from agm.db import force_cancel_plan

    conn = db_conn
    assert force_cancel_plan(conn, "nonexistent") is None


def test_force_cancel_plan_records_status_history(db_conn):
    """force_cancel_plan should record status history."""
    from agm.db import force_cancel_plan

    conn = db_conn
    plan = make_plan(conn)
    force_cancel_plan(conn, plan["id"])
    rows = conn.execute(
        "SELECT old_status, new_status FROM status_history "
        "WHERE entity_id = ? AND new_status = 'cancelled'",
        (plan["id"],),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["old_status"] == "finalized"


# -- create_tasks_batch --


def test_create_tasks_batch(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "First",
            "description": "Do first thing",
            "files": json.dumps(["a.py"]),
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
        },
        {
            "ordinal": 1,
            "title": "Second",
            "description": "Do second thing",
            "files": None,
            "status": "blocked",
            "blocked_by": [0],
            "blocked_by_existing": [],
            "external_blockers": [],
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)
    assert 0 in mapping
    assert 1 in mapping
    assert mapping[0] != mapping[1]

    # Verify tasks exist
    t0 = get_task(conn, mapping[0])
    assert t0["title"] == "First"
    assert t0["status"] == "ready"

    t1 = get_task(conn, mapping[1])
    assert t1["title"] == "Second"
    assert t1["status"] == "blocked"

    # Verify blocker wired
    blocks = list_task_blocks(conn, mapping[1])
    assert len(blocks) == 1
    assert blocks[0]["blocked_by_task_id"] == mapping[0]


def test_create_tasks_batch_with_external_blockers(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "Needs key",
            "description": "Requires API key",
            "status": "blocked",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [
                {"factor": "API key", "reason": "Need production credentials"},
            ],
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)
    blocks = list_task_blocks(conn, mapping[0])
    assert len(blocks) == 1
    assert blocks[0]["external_factor"] == "API key"
    assert blocks[0]["reason"] == "Need production credentials"


def test_create_tasks_batch_with_existing_dep(db_conn):
    conn = db_conn
    plan1 = make_plan(conn, "first plan")
    # Create an existing task from plan1
    existing = create_task(conn, plan_id=plan1["id"], ordinal=0, title="Existing", description="x")

    plan2 = make_plan(conn, "second plan")
    tasks_data = [
        {
            "ordinal": 0,
            "title": "New task",
            "description": "Depends on existing",
            "status": "blocked",
            "blocked_by": [],
            "blocked_by_existing": [existing["id"]],
            "external_blockers": [],
        },
    ]
    mapping = create_tasks_batch(conn, plan2["id"], tasks_data)
    blocks = list_task_blocks(conn, mapping[0])
    assert len(blocks) == 1
    assert blocks[0]["blocked_by_task_id"] == existing["id"]


def test_create_tasks_batch_priority_persistence(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "High",
            "description": "d",
            "status": "ready",
            "priority": "high",
        },
        {
            "ordinal": 1,
            "title": "Medium",
            "description": "d",
            "status": "ready",
            "priority": "medium",
        },
        {
            "ordinal": 2,
            "title": "Low",
            "description": "d",
            "status": "ready",
            "priority": "low",
        },
        {
            "ordinal": 3,
            "title": "Implicit Medium",
            "description": "d",
            "status": "ready",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    assert get_task(conn, mapping[0])["priority"] == "high"
    assert get_task(conn, mapping[1])["priority"] is None
    assert get_task(conn, mapping[2])["priority"] == "low"
    assert get_task(conn, mapping[3])["priority"] is None


# -- resolve_blockers_for_terminal_task --


def test_resolve_blockers_for_terminal_task(db_conn):
    """Completing a task resolves internal blockers and promotes dependents to ready."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "blocked")
    # t1 blocked by t0
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    assert get_unresolved_block_count(conn, t1["id"]) == 1

    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert t1["id"] in promoted
    assert cascade_cancelled == []
    assert get_unresolved_block_count(conn, t1["id"]) == 0
    refreshed = get_task(conn, t1["id"])
    assert refreshed["status"] == "ready"
    assert list_status_history(conn, entity_type="task", entity_id=t1["id"]) == []


def test_resolve_blockers_multiple_dependents(db_conn):
    """Completing a task promotes multiple blocked dependents."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Task 2", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "blocked")
    update_task_status(conn, t2["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t0["id"])

    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert set(promoted) == {t1["id"], t2["id"]}
    assert cascade_cancelled == []


def test_resolve_blockers_partial_unblock(db_conn):
    """Task with multiple blockers stays blocked when only one is resolved."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Task 2", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "ready")
    update_task_status(conn, t2["id"], "blocked")
    # t2 blocked by both t0 and t1
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t1["id"])

    # Only complete t0 — t2 should stay blocked (still blocked by t1)
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert promoted == []
    assert cascade_cancelled == []
    assert get_unresolved_block_count(conn, t2["id"]) == 1
    refreshed = get_task(conn, t2["id"])
    assert refreshed["status"] == "blocked"


def test_resolve_blockers_no_dependents(db_conn):
    """No-op when completed task has no dependents."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        conn, t0["id"], record_history=True
    )
    assert promoted == []
    assert cascade_cancelled == []


def test_resolve_blockers_skips_cancelled(db_conn):
    """Cancelled downstream tasks are not promoted to ready even if all blockers resolve."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "cancelled")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert promoted == []
    assert cascade_cancelled == []
    refreshed = get_task(conn, t1["id"])
    assert refreshed["status"] == "cancelled"


def test_resolve_blockers_record_history_for_promoted_only(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Task 2", description="d")
    t3 = create_task(conn, plan_id=plan["id"], ordinal=3, title="Task 3", description="d")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "blocked")
    update_task_status(conn, t2["id"], "cancelled")
    update_task_status(conn, t3["id"], "ready")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t3["id"], blocked_by_task_id=t0["id"])

    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        conn, t0["id"], record_history=True
    )
    assert promoted == [t1["id"]]
    assert cascade_cancelled == []
    assert get_task(conn, t1["id"])["status"] == "ready"
    assert get_task(conn, t2["id"])["status"] == "cancelled"
    assert get_task(conn, t3["id"])["status"] == "ready"

    history_t1 = list_status_history(conn, entity_type="task", entity_id=t1["id"])
    assert len(history_t1) == 1
    assert history_t1[0]["old_status"] == "blocked"
    assert history_t1[0]["new_status"] == "ready"
    assert list_status_history(conn, entity_type="task", entity_id=t2["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t3["id"]) == []


def test_resolve_blockers_duplicate_blocks_idempotent(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    first, _ = resolve_blockers_for_terminal_task(conn, t0["id"], record_history=True)
    assert first == [t1["id"]]
    assert get_unresolved_block_count(conn, t1["id"]) == 0

    history = list_status_history(conn, entity_type="task", entity_id=t1["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "blocked"
    assert history[0]["new_status"] == "ready"

    second, _ = resolve_blockers_for_terminal_task(conn, t0["id"], record_history=True)
    assert second == []
    assert len(list_status_history(conn, entity_type="task", entity_id=t1["id"])) == 1


def test_resolve_blockers_promotion_requires_real_status_change(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    raced_conn = _TaskCASRaceConnection(
        conn,
        task_id=t1["id"],
        select_sql_prefix="SELECT status, actor FROM tasks WHERE id = ?",
        update_sql_prefix="UPDATE tasks SET status = 'ready', updated_at = strftime(",
        race_sql=(
            "UPDATE tasks SET status = 'running', updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
        ),
    )
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        raced_conn, t0["id"], record_history=True
    )
    assert promoted == []
    assert cascade_cancelled == []
    assert get_unresolved_block_count(conn, t1["id"]) == 0
    assert get_task(conn, t1["id"])["status"] == "running"
    assert list_status_history(conn, entity_type="task", entity_id=t1["id"]) == []


# -- resolve_stale_blockers --


def test_resolve_stale_blockers(db_conn):
    """Sweep resolves blockers pointing at completed tasks and promotes dependents."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t1["id"], "blocked")
    # t1 blocked by t0, but t0 is already completed
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")

    # Blocker still unresolved (simulating race — completion happened before blocker was created)
    assert get_unresolved_block_count(conn, t1["id"]) == 1

    promoted = resolve_stale_blockers(conn)
    assert t1["id"] in promoted
    assert get_unresolved_block_count(conn, t1["id"]) == 0
    assert get_task(conn, t1["id"])["status"] == "ready"
    assert list_status_history(conn, entity_type="task", entity_id=t1["id"]) == []


def test_resolve_stale_blockers_no_stale(db_conn):
    """No-op when there are no stale blockers."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    update_task_status(conn, t0["id"], "ready")
    promoted = resolve_stale_blockers(conn, record_history=True)
    assert promoted == []
    assert list_status_history(conn, entity_type="task", entity_id=t0["id"]) == []


def test_resolve_stale_blockers_partial(db_conn):
    """Task with one stale and one active blocker stays blocked."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Task 2", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "ready")
    update_task_status(conn, t2["id"], "blocked")
    # t2 blocked by t0 (completed) and t1 (still ready)
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t1["id"])

    promoted = resolve_stale_blockers(conn)
    assert promoted == []
    assert get_unresolved_block_count(conn, t2["id"]) == 1
    assert get_task(conn, t2["id"])["status"] == "blocked"


def test_resolve_stale_blockers_record_history_for_promoted_only(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Task 2", description="d")
    t3 = create_task(conn, plan_id=plan["id"], ordinal=3, title="Task 3", description="d")

    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t2["id"], "cancelled")
    update_task_status(conn, t3["id"], "ready")

    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t3["id"], blocked_by_task_id=t0["id"])

    promoted = resolve_stale_blockers(conn, record_history=True)
    assert promoted == [t1["id"]]
    assert get_task(conn, t1["id"])["status"] == "ready"
    assert get_task(conn, t2["id"])["status"] == "cancelled"
    assert get_task(conn, t3["id"])["status"] == "ready"

    history_t1 = list_status_history(conn, entity_type="task", entity_id=t1["id"])
    assert len(history_t1) == 1
    assert history_t1[0]["old_status"] == "blocked"
    assert history_t1[0]["new_status"] == "ready"
    assert list_status_history(conn, entity_type="task", entity_id=t2["id"]) == []
    assert list_status_history(conn, entity_type="task", entity_id=t3["id"]) == []


def test_resolve_stale_blockers_promotion_requires_real_status_change(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")

    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "completed")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    raced_conn = _TaskCASRaceConnection(
        conn,
        task_id=t1["id"],
        select_sql_prefix="SELECT status, actor FROM tasks WHERE id = ?",
        update_sql_prefix="UPDATE tasks SET status = 'ready', updated_at = strftime(",
        race_sql=(
            "UPDATE tasks SET status = 'running', updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
        ),
    )
    promoted = resolve_stale_blockers(raced_conn, record_history=True)
    assert promoted == []
    assert get_unresolved_block_count(conn, t1["id"]) == 0
    assert get_task(conn, t1["id"])["status"] == "running"
    assert list_status_history(conn, entity_type="task", entity_id=t1["id"]) == []


def test_resolve_stale_blockers_idempotent(db_conn):
    """Repeated stale sweeps are idempotent. Only completed blockers are swept."""
    conn = db_conn
    plan = make_plan(conn)
    completed_blocker_1 = create_task(
        conn, plan_id=plan["id"], ordinal=0, title="Completed blocker 1", description="d"
    )
    completed_blocker_2 = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Completed blocker 2", description="d"
    )
    t1 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Blocked task", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=3, title="Another blocked", description="d")

    update_task_status(conn, completed_blocker_1["id"], "running")
    update_task_status(conn, completed_blocker_1["id"], "completed")
    update_task_status(conn, completed_blocker_2["id"], "running")
    update_task_status(conn, completed_blocker_2["id"], "completed")
    update_task_status(conn, t1["id"], "blocked")
    update_task_status(conn, t2["id"], "blocked")

    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=completed_blocker_1["id"])
    add_task_block(conn, task_id=t2["id"], blocked_by_task_id=completed_blocker_2["id"])

    first = resolve_stale_blockers(conn, record_history=True)
    assert set(first) == {t1["id"], t2["id"]}
    assert get_task(conn, t1["id"])["status"] == "ready"
    assert get_task(conn, t2["id"])["status"] == "ready"

    second = resolve_stale_blockers(conn, record_history=True)
    assert second == []
    assert get_task(conn, t1["id"])["status"] == "ready"
    assert get_task(conn, t2["id"])["status"] == "ready"


# -- resolve blockers for cancelled/failed tasks --


def test_resolve_blockers_for_cancelled_task(db_conn):
    """Cancelling a blocker task cascade-cancels downstream dependents."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    assert get_unresolved_block_count(conn, t1["id"]) == 1

    # Cancel t0 — t1 should be cascade-cancelled (not promoted)
    update_task_status(conn, t0["id"], "cancelled")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert promoted == []
    assert t1["id"] in cascade_cancelled
    assert get_unresolved_block_count(conn, t1["id"]) == 0
    assert get_task(conn, t1["id"])["status"] == "cancelled"
    # Verify cascade log message
    logs = list_task_logs(conn, t1["id"])
    assert any("Cascade cancelled" in log["message"] for log in logs)


def test_resolve_blockers_for_failed_task(db_conn):
    """Failing a blocker task does NOT resolve blockers — downstream stays blocked."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 0", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    # Fail t0 — t1 should stay blocked (awaiting retry)
    update_task_status(conn, t0["id"], "failed")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, t0["id"])
    assert promoted == []
    assert cascade_cancelled == []
    assert get_unresolved_block_count(conn, t1["id"]) == 1
    assert get_task(conn, t1["id"])["status"] == "blocked"


def test_cancelled_task_cascades_downstream(db_conn):
    """Cancelling A cascade-cancels B (A→B chain)."""
    conn = db_conn
    plan = make_plan(conn)
    a = create_task(conn, plan_id=plan["id"], ordinal=0, title="Step A", description="d")
    b = create_task(conn, plan_id=plan["id"], ordinal=1, title="Step B", description="d")
    c = create_task(conn, plan_id=plan["id"], ordinal=2, title="Step C", description="d")
    update_task_status(conn, a["id"], "ready")
    update_task_status(conn, b["id"], "blocked")
    update_task_status(conn, c["id"], "blocked")
    add_task_block(conn, task_id=b["id"], blocked_by_task_id=a["id"])
    add_task_block(conn, task_id=c["id"], blocked_by_task_id=b["id"])

    update_task_status(conn, a["id"], "cancelled")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, a["id"])
    assert promoted == []
    assert set(cascade_cancelled) == {b["id"], c["id"]}
    assert get_task(conn, b["id"])["status"] == "cancelled"
    assert get_task(conn, c["id"])["status"] == "cancelled"


def test_cancelled_task_cascades_transitively(db_conn):
    """Cancel A → cascades to B, C, D (A→B, A→C, B→D)."""
    conn = db_conn
    plan = make_plan(conn)
    a = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="d")
    b = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="d")
    c = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="d")
    d = create_task(conn, plan_id=plan["id"], ordinal=3, title="D", description="d")
    update_task_status(conn, a["id"], "ready")
    update_task_status(conn, b["id"], "blocked")
    update_task_status(conn, c["id"], "blocked")
    update_task_status(conn, d["id"], "blocked")
    add_task_block(conn, task_id=b["id"], blocked_by_task_id=a["id"])
    add_task_block(conn, task_id=c["id"], blocked_by_task_id=a["id"])
    add_task_block(conn, task_id=d["id"], blocked_by_task_id=b["id"])

    update_task_status(conn, a["id"], "cancelled")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, a["id"])
    assert promoted == []
    assert set(cascade_cancelled) == {b["id"], c["id"], d["id"]}
    for tid in [b["id"], c["id"], d["id"]]:
        assert get_task(conn, tid)["status"] == "cancelled"
        logs = list_task_logs(conn, tid)
        assert any("Cascade cancelled" in log["message"] for log in logs)


def test_cancelled_task_cascade_skips_terminal(db_conn):
    """Cascade cancellation skips already-terminal downstream tasks."""
    conn = db_conn
    plan = make_plan(conn)
    a = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="d")
    b = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="d")
    c = create_task(conn, plan_id=plan["id"], ordinal=2, title="C", description="d")
    update_task_status(conn, a["id"], "ready")
    update_task_status(conn, b["id"], "blocked")
    # c already completed before cascade
    update_task_status(conn, c["id"], "running")
    update_task_status(conn, c["id"], "completed")
    add_task_block(conn, task_id=b["id"], blocked_by_task_id=a["id"])
    add_task_block(conn, task_id=c["id"], blocked_by_task_id=a["id"])

    update_task_status(conn, a["id"], "cancelled")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, a["id"])
    assert promoted == []
    assert cascade_cancelled == [b["id"]]
    assert get_task(conn, b["id"])["status"] == "cancelled"
    assert get_task(conn, c["id"])["status"] == "completed"  # unchanged


def test_cancelled_task_cascade_records_history(db_conn):
    """Cascade cancellation records status history when requested."""
    conn = db_conn
    plan = make_plan(conn)
    a = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="d")
    b = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="d")
    update_task_status(conn, a["id"], "ready")
    update_task_status(conn, b["id"], "blocked")
    add_task_block(conn, task_id=b["id"], blocked_by_task_id=a["id"])

    update_task_status(conn, a["id"], "cancelled")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        conn, a["id"], record_history=True
    )
    assert cascade_cancelled == [b["id"]]
    history = list_status_history(conn, entity_type="task", entity_id=b["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "blocked"
    assert history[0]["new_status"] == "cancelled"


def test_bucket_failure_blocks_downstream(db_conn):
    """Bucket tasks: #0 fails, #1 and #2 stay blocked (not promoted)."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "Step 1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
        {
            "ordinal": 1,
            "title": "Step 2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
        {
            "ordinal": 2,
            "title": "Step 3",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Fail first task
    update_task_status(conn, mapping[0], "running")
    update_task_status(conn, mapping[0], "failed")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, mapping[0])

    # Nothing promoted or cascade-cancelled: failed tasks leave downstream blocked
    assert promoted == []
    assert cascade_cancelled == []
    assert get_task(conn, mapping[1])["status"] == "blocked"
    assert get_task(conn, mapping[2])["status"] == "blocked"
    assert get_unresolved_block_count(conn, mapping[1]) == 1


def test_bucket_retry_completes_unblocks(db_conn):
    """Bucket tasks: #0 fails, retry succeeds, #1 promoted."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "Step 1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
        {
            "ordinal": 1,
            "title": "Step 2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Fail first task — blockers stay unresolved
    update_task_status(conn, mapping[0], "running")
    update_task_status(conn, mapping[0], "failed")
    resolve_blockers_for_terminal_task(conn, mapping[0])
    assert get_unresolved_block_count(conn, mapping[1]) == 1

    # Retry: reset to blocked, then complete
    reset_task_for_retry(conn, mapping[0])
    update_task_status(conn, mapping[0], "ready")
    update_task_status(conn, mapping[0], "running")
    update_task_status(conn, mapping[0], "completed")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, mapping[0])

    # Now downstream is promoted
    assert mapping[1] in promoted
    assert cascade_cancelled == []
    assert get_task(conn, mapping[1])["status"] == "ready"
    assert get_unresolved_block_count(conn, mapping[1]) == 0


def test_resolve_stale_blockers_only_sweeps_completed(db_conn):
    """Stale blocker sweep only resolves blockers pointing at completed tasks."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Cancelled", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Failed", description="d")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Completed", description="d")
    t3 = create_task(conn, plan_id=plan["id"], ordinal=3, title="Blocked A", description="d")
    t4 = create_task(conn, plan_id=plan["id"], ordinal=4, title="Blocked B", description="d")
    t5 = create_task(conn, plan_id=plan["id"], ordinal=5, title="Blocked C", description="d")

    update_task_status(conn, t0["id"], "cancelled")
    update_task_status(conn, t1["id"], "ready")
    update_task_status(conn, t1["id"], "running")
    update_task_status(conn, t1["id"], "failed")
    update_task_status(conn, t2["id"], "running")
    update_task_status(conn, t2["id"], "completed")

    # Blockers created after terminal state (simulating race)
    add_task_block(conn, task_id=t3["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=t4["id"], blocked_by_task_id=t1["id"])
    add_task_block(conn, task_id=t5["id"], blocked_by_task_id=t2["id"])

    promoted = resolve_stale_blockers(conn)
    # Only t5 promoted (blocked by completed t2). t3 and t4 stay blocked.
    assert promoted == [t5["id"]]
    assert get_unresolved_block_count(conn, t3["id"]) == 1
    assert get_unresolved_block_count(conn, t4["id"]) == 1
    assert get_task(conn, t5["id"])["status"] == "ready"
    assert get_unresolved_block_count(conn, t5["id"]) == 0


def test_cancel_tasks_batch_does_not_resolve_blockers(db_conn):
    """Batch cancellation does NOT resolve blockers or promote downstream."""
    conn = db_conn
    plan = make_plan(conn)
    t0 = create_task(conn, plan_id=plan["id"], ordinal=0, title="To cancel", description="d")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Blocked", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])

    result = cancel_tasks_batch(conn, [{"task_id": t0["id"], "reason": "superseded"}])
    assert result["cancelled"] == 1
    assert "promoted_task_ids" not in result
    assert get_task(conn, t0["id"])["status"] == "cancelled"
    # t1 stays blocked — batch cancel doesn't promote or cascade
    assert get_unresolved_block_count(conn, t1["id"]) == 1
    assert get_task(conn, t1["id"])["status"] == "blocked"


def test_cancel_tasks_batch_no_promotion(db_conn):
    """Batch cancellation no longer promotes or resolves blockers."""
    conn = db_conn
    plan = make_plan(conn)
    blocker_1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Blocker A", description="d")
    blocker_2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Blocker B", description="d")
    dependent_1 = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Dependent A", description="d"
    )
    dependent_2 = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="Dependent B", description="d"
    )

    update_task_status(conn, blocker_1["id"], "ready")
    update_task_status(conn, blocker_2["id"], "ready")
    update_task_status(conn, dependent_1["id"], "blocked")
    update_task_status(conn, dependent_2["id"], "blocked")

    add_task_block(conn, task_id=dependent_1["id"], blocked_by_task_id=blocker_1["id"])
    add_task_block(conn, task_id=dependent_2["id"], blocked_by_task_id=blocker_1["id"])
    add_task_block(conn, task_id=dependent_2["id"], blocked_by_task_id=blocker_2["id"])

    result = cancel_tasks_batch(
        conn,
        [
            {"task_id": blocker_1["id"], "reason": "superseded"},
            {"task_id": blocker_2["id"], "reason": "superseded"},
        ],
        record_history=True,
    )
    assert result["cancelled"] == 2
    assert "promoted_task_ids" not in result

    # Dependents stay blocked with unresolved blockers
    assert get_task(conn, dependent_1["id"])["status"] == "blocked"
    assert get_task(conn, dependent_2["id"])["status"] == "blocked"
    assert get_unresolved_block_count(conn, dependent_1["id"]) == 1
    assert get_unresolved_block_count(conn, dependent_2["id"]) == 2

    history = list_status_history(conn, entity_type="task", entity_id=blocker_1["id"])
    assert len(history) == 1
    assert history[0]["old_status"] == "ready"
    assert history[0]["new_status"] == "cancelled"


def test_cancel_tasks_batch_idempotent(db_conn):
    """Duplicate cancellation requests are idempotent."""
    conn = db_conn
    plan = make_plan(conn)
    blocker = create_task(conn, plan_id=plan["id"], ordinal=0, title="Blocker", description="d")
    update_task_status(conn, blocker["id"], "ready")

    first = cancel_tasks_batch(
        conn,
        [{"task_id": blocker["id"], "reason": "superseded"}],
        record_history=True,
    )
    assert first["cancelled"] == 1

    second = cancel_tasks_batch(
        conn,
        [
            {"task_id": blocker["id"], "reason": "noop"},
            {"task_id": blocker["id"], "reason": "again"},
        ],
        record_history=True,
    )
    assert second["cancelled"] == 0
    assert get_task(conn, blocker["id"])["status"] == "cancelled"
    blocker_history = list_status_history(conn, entity_type="task", entity_id=blocker["id"])
    assert len(blocker_history) == 1
    assert blocker_history[0]["old_status"] == "ready"
    assert blocker_history[0]["new_status"] == "cancelled"
    # 1 cancel log + 2 skip logs (noop + again on already-cancelled)
    assert len(list_task_logs(conn, blocker["id"])) == 3


# -- reviewer thread id --


def test_set_task_failure_reason(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    assert set_task_failure_reason(conn, t["id"], "dependency missing")
    found = get_task(conn, t["id"])
    assert found["failure_reason"] == "dependency missing"

    assert set_task_failure_reason(conn, t["id"], None)
    found = get_task(conn, t["id"])
    assert found["failure_reason"] is None


def test_set_task_failure_reason_not_found_returns_false(db_conn):
    conn = db_conn
    assert not set_task_failure_reason(conn, "missing", "anything")


def test_set_task_reviewer_thread_id(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    assert set_task_reviewer_thread_id(conn, t["id"], "review-thread-xyz")
    found = get_task(conn, t["id"])
    assert found["reviewer_thread_id"] == "review-thread-xyz"


def test_reviewer_thread_id_preserves_executor_thread(db_conn):
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    # Set executor thread first
    set_task_thread_id(conn, t["id"], "exec-thread-123")
    # Set reviewer thread — should NOT overwrite executor thread
    set_task_reviewer_thread_id(conn, t["id"], "review-thread-456")
    found = get_task(conn, t["id"])
    assert found["thread_id"] == "exec-thread-123"
    assert found["reviewer_thread_id"] == "review-thread-456"


# -- quick mode --


def test_create_quick_plan_and_task(db_conn):
    """create_quick_plan_and_task creates a finalized plan + ready task."""
    conn = db_conn
    pid = get_project_id(conn)
    plan, task = create_quick_plan_and_task(
        conn,
        project_id=pid,
        prompt="Fix the bug",
        title="Fix the bug",
        description="Fix the bug in module X",
        caller="cli",
        backend="codex",
    )
    assert plan["status"] == "finalized"
    assert plan["mode"] == "quick"
    assert plan["task_creation_status"] == "completed"
    assert plan["thread_id"] is None
    assert task["status"] == "ready"
    assert task["plan_id"] == plan["id"]
    assert task["ordinal"] == 1
    assert task["skip_review"] == 0
    assert task["skip_merge"] == 0

    # Verify in DB
    db_plan = conn.execute("SELECT * FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert db_plan["status"] == "finalized"
    assert db_plan["mode"] == "quick"
    assert db_plan["task_creation_status"] == "completed"
    db_task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert db_task["status"] == "ready"
    assert db_task["skip_review"] == 0
    assert db_task["skip_merge"] == 0


def test_create_quick_plan_and_task_with_flags(db_conn):
    """create_quick_plan_and_task respects skip_review and skip_merge."""
    conn = db_conn
    pid = get_project_id(conn)
    plan, task = create_quick_plan_and_task(
        conn,
        project_id=pid,
        prompt="Quick fix",
        title="Quick fix",
        description="Do it fast",
        caller="cli",
        backend="codex",
        files='["src/foo.py"]',
        skip_review=True,
        skip_merge=True,
        priority="low",
    )
    assert task["skip_review"] == 1
    assert task["skip_merge"] == 1
    assert task["priority"] == "low"
    assert task["files"] == '["src/foo.py"]'

    db_task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert db_task["skip_review"] == 1
    assert db_task["skip_merge"] == 1
    assert db_task["priority"] == "low"
    assert db_task["files"] == '["src/foo.py"]'


def test_create_quick_plan_and_task_invalid_caller(db_conn):
    """create_quick_plan_and_task rejects invalid callers."""
    conn = db_conn
    pid = get_project_id(conn)
    with pytest.raises(ValueError, match="Invalid caller"):
        create_quick_plan_and_task(
            conn,
            project_id=pid,
            prompt="x",
            title="x",
            description="x",
            caller="bogus",
            backend="codex",
        )


# -- token tracking --


def test_update_plan_tokens_additive(db_conn):
    """update_plan_tokens should add to existing counts."""
    conn = db_conn
    plan = make_plan(conn)
    p = get_plan_request(conn, plan["id"])
    assert p["input_tokens"] == 0
    assert p["output_tokens"] == 0

    update_plan_tokens(conn, plan["id"], 100, 50)
    p = get_plan_request(conn, plan["id"])
    assert p["input_tokens"] == 100
    assert p["output_tokens"] == 50

    # Add more — should be additive
    update_plan_tokens(conn, plan["id"], 200, 75)
    p = get_plan_request(conn, plan["id"])
    assert p["input_tokens"] == 300
    assert p["output_tokens"] == 125


def test_update_task_tokens_additive(db_conn):
    """update_task_tokens should add to existing counts."""
    conn = db_conn
    plan = make_plan(conn)
    t = create_task(conn, plan_id=plan["id"], ordinal=0, title="t", description="d")
    task = get_task(conn, t["id"])
    assert task["input_tokens"] == 0
    assert task["output_tokens"] == 0

    update_task_tokens(conn, t["id"], 500, 200)
    task = get_task(conn, t["id"])
    assert task["input_tokens"] == 500
    assert task["output_tokens"] == 200

    update_task_tokens(conn, t["id"], 300, 100)
    task = get_task(conn, t["id"])
    assert task["input_tokens"] == 800
    assert task["output_tokens"] == 300


# -- task buckets --


def test_create_tasks_batch_with_buckets(db_conn):
    """Auto-blocker injected on second task in same bucket."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "First DB task",
            "description": "Add column",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
        {
            "ordinal": 1,
            "title": "Second DB task",
            "description": "Use column",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Second task should have an auto-injected blocker on first
    blocks = list_task_blocks(conn, mapping[1])
    assert len(blocks) == 1
    assert blocks[0]["blocked_by_task_id"] == mapping[0]

    # Second task forced to blocked (non-first in bucket)
    t1 = get_task(conn, mapping[1])
    assert t1["status"] == "blocked"

    # First task remains ready
    t0 = get_task(conn, mapping[0])
    assert t0["status"] == "ready"

    # Both tasks have bucket set
    assert t0["bucket"] == "db-layer"
    assert t1["bucket"] == "db-layer"


def test_create_tasks_batch_bucket_no_duplicate_blockers(db_conn):
    """Agent + system both set blocker; verify single row (no duplicate)."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "First",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "core",
        },
        {
            "ordinal": 1,
            "title": "Second",
            "description": "d",
            "status": "blocked",
            "blocked_by": [0],  # Agent already set this
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "core",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Should have exactly one blocker, not two
    blocks = list_task_blocks(conn, mapping[1])
    assert len(blocks) == 1
    assert blocks[0]["blocked_by_task_id"] == mapping[0]


def test_create_tasks_batch_bucket_forces_blocked(db_conn):
    """Second task marked 'ready' by agent gets forced to 'blocked' by bucket logic."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "First",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "cli",
        },
        {
            "ordinal": 1,
            "title": "Second",
            "description": "d",
            "status": "ready",  # Agent says ready, but bucket forces blocked
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "cli",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    t0 = get_task(conn, mapping[0])
    t1 = get_task(conn, mapping[1])
    assert t0["status"] == "ready"
    assert t1["status"] == "blocked"


def test_create_tasks_batch_multiple_buckets(db_conn):
    """Intra-bucket blockers only — no cross-bucket auto-blockers."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "DB-1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
        {
            "ordinal": 1,
            "title": "CLI-1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "cli-cmds",
        },
        {
            "ordinal": 2,
            "title": "DB-2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
        {
            "ordinal": 3,
            "title": "CLI-2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "cli-cmds",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # DB-2 blocked by DB-1 (same bucket)
    db2_blocks = list_task_blocks(conn, mapping[2])
    assert len(db2_blocks) == 1
    assert db2_blocks[0]["blocked_by_task_id"] == mapping[0]

    # CLI-2 blocked by CLI-1 (same bucket)
    cli2_blocks = list_task_blocks(conn, mapping[3])
    assert len(cli2_blocks) == 1
    assert cli2_blocks[0]["blocked_by_task_id"] == mapping[1]

    # First tasks in each bucket: no blockers
    assert list_task_blocks(conn, mapping[0]) == []
    assert list_task_blocks(conn, mapping[1]) == []

    # First in each bucket stays ready; second in each forced to blocked
    assert get_task(conn, mapping[0])["status"] == "ready"
    assert get_task(conn, mapping[1])["status"] == "ready"
    assert get_task(conn, mapping[2])["status"] == "blocked"
    assert get_task(conn, mapping[3])["status"] == "blocked"


def test_singleton_bucket_stripped_on_creation(db_conn):
    """A bucket with only one task gets its label NULLed out at creation time."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "DB-1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
        {
            "ordinal": 1,
            "title": "DB-2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "db-layer",
        },
        {
            "ordinal": 2,
            "title": "Solo CLI task",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "cli-commands",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Solo task's bucket should be stripped to NULL
    solo = get_task(conn, mapping[2])
    assert solo["bucket"] is None

    # Multi-task bucket preserved
    t0 = get_task(conn, mapping[0])
    t1 = get_task(conn, mapping[1])
    assert t0["bucket"] == "db-layer"
    assert t1["bucket"] == "db-layer"

    # Auto-serialization only applied to the 2-task bucket
    assert list_task_blocks(conn, mapping[1]) != []  # DB-2 blocked by DB-1
    assert list_task_blocks(conn, mapping[2]) == []  # Solo task has no blockers

    # Solo task stays ready (not forced to blocked)
    assert solo["status"] == "ready"


def test_bucket_pipeline_cascade(db_conn):
    """Complete first task in bucket; second promoted, third stays blocked."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "Step 1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
        {
            "ordinal": 1,
            "title": "Step 2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
        {
            "ordinal": 2,
            "title": "Step 3",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "pipeline",
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # Verify initial state: only first is ready
    assert get_task(conn, mapping[0])["status"] == "ready"
    assert get_task(conn, mapping[1])["status"] == "blocked"
    assert get_task(conn, mapping[2])["status"] == "blocked"

    # Complete first task
    update_task_status(conn, mapping[0], "running")
    update_task_status(conn, mapping[0], "completed")
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, mapping[0])

    # Second should be promoted to ready
    assert mapping[1] in promoted
    assert cascade_cancelled == []
    assert get_task(conn, mapping[1])["status"] == "ready"

    # Third should still be blocked (blocked by second, which isn't completed yet)
    assert mapping[2] not in promoted
    assert get_task(conn, mapping[2])["status"] == "blocked"


def test_create_tasks_batch_null_bucket_no_auto_blockers(db_conn):
    """Tasks with null bucket don't get auto-blockers."""
    conn = db_conn
    plan = make_plan(conn)
    tasks_data = [
        {
            "ordinal": 0,
            "title": "Standalone A",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": None,
        },
        {
            "ordinal": 1,
            "title": "Standalone B",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": None,
        },
    ]
    mapping = create_tasks_batch(conn, plan["id"], tasks_data)

    # No auto-blockers for null-bucket tasks
    assert list_task_blocks(conn, mapping[0]) == []
    assert list_task_blocks(conn, mapping[1]) == []

    # Both remain ready
    assert get_task(conn, mapping[0])["status"] == "ready"
    assert get_task(conn, mapping[1])["status"] == "ready"


def test_migration_adds_bucket_column():
    """Migration adds bucket column to existing tasks table."""
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
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            pid INTEGER,
            thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            reviewer_thread_id TEXT,
            skip_review INTEGER NOT NULL DEFAULT 0,
            skip_merge INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.commit()

    # Open via get_connection which runs migrations
    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "bucket" in cols
    migrated.close()


def test_migration_adds_task_model_column_for_v5():
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
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            priority TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            pid INTEGER,
            thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            reviewer_thread_id TEXT,
            skip_review INTEGER NOT NULL DEFAULT 0,
            skip_merge INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("proj1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("plan1", "proj1", "legacy prompt", "alice", "cli", "codex"),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description) VALUES (?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy description"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "model" in cols
    row = migrated.execute("SELECT model FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["model"] is None
    migrated.close()

    reopened = get_connection(db_path)
    row = reopened.execute("SELECT model FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["model"] is None
    reopened.close()


def test_migration_adds_priority_column_idempotently():
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
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            pid INTEGER,
            thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            reviewer_thread_id TEXT,
            skip_review INTEGER NOT NULL DEFAULT 0,
            skip_merge INTEGER NOT NULL DEFAULT 0,
            bucket TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("proj1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("plan1", "proj1", "legacy prompt", "alice", "cli", "codex"),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description) VALUES (?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy description"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "priority" in cols
    row = migrated.execute("SELECT priority FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["priority"] is None
    migrated.execute("UPDATE tasks SET priority = 'medium' WHERE id = ?", ("task1",))
    migrated.commit()
    migrated.close()

    reopened = get_connection(db_path)
    cols = {row["name"] for row in reopened.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "priority" in cols
    row = reopened.execute("SELECT priority FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["priority"] == "medium"
    reopened.close()


def test_migration_adds_failure_reason_column():
    """Migration adds failure_reason column to existing tasks table."""
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
            pid INTEGER,
            thread_id TEXT,
            task_creation_status TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            priority TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            pid INTEGER,
            thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            reviewer_thread_id TEXT,
            skip_review INTEGER NOT NULL DEFAULT 0,
            skip_merge INTEGER NOT NULL DEFAULT 0,
            bucket TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy", "/tmp/legacy"),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("plan1", "p1", "legacy prompt", "alice", "cli", "codex"),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description) VALUES (?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy description"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "failure_reason" in cols
    row = migrated.execute("SELECT failure_reason FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["failure_reason"] is None
    migrated.close()

    reopened = get_connection(db_path)
    cols = {row["name"] for row in reopened.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "failure_reason" in cols
    row = reopened.execute("SELECT failure_reason FROM tasks WHERE id = ?", ("task1",)).fetchone()
    assert row["failure_reason"] is None
    reopened.close()


def test_schema_version_set_on_fresh_db():
    """Fresh DB gets SCHEMA_VERSION set via PRAGMA user_version."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = get_connection(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "failure_reason" in cols
    assert "active_turn_id" in cols
    assert "active_turn_started_at" in cols


def test_schema_version_set_after_legacy_migration():
    """Legacy DB (user_version=0) gets migrated and version set."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    # Create a legacy DB without user_version
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
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            actor TEXT NOT NULL,
            caller TEXT NOT NULL,
            backend TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            pid INTEGER,
            thread_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE task_blocks (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id),
            blocked_by_task_id TEXT REFERENCES tasks(id),
            external_factor TEXT,
            reason TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            resolved_at TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE plan_questions (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            question TEXT NOT NULL,
            options TEXT,
            answer TEXT,
            answered_by TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            answered_at TEXT
        );
        CREATE TABLE plan_logs (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE task_logs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id),
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.commit()

    # Open via get_connection — triggers migration
    migrated = get_connection(db_path)
    version = migrated.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    # Verify migration added missing columns
    cols = {row[1] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "bucket" in cols
    assert "priority" in cols
    assert "skip_review" in cols
    assert "failure_reason" in cols
    assert "active_turn_id" in cols
    assert "active_turn_started_at" in cols
    migrated.close()


def test_schema_version_skips_migrate_when_current(monkeypatch):
    """When user_version matches SCHEMA_VERSION, _migrate is not called."""
    from agm import db

    db_path = Path(tempfile.mktemp(suffix=".db"))
    # First connection: sets version
    conn1 = get_connection(db_path)
    conn1.close()

    # Patch _migrate to track calls
    migrate_calls = []
    original_migrate = db._migrate

    def tracking_migrate(conn, from_version):
        migrate_calls.append(from_version)
        return original_migrate(conn, from_version)

    monkeypatch.setattr(db, "_migrate", tracking_migrate)

    # Second connection: should skip _migrate
    conn2 = get_connection(db_path)
    conn2.close()
    assert len(migrate_calls) == 0


# --- Failed sibling context ---


def test_get_failed_sibling_context(db_conn):
    """_get_failed_sibling_context should list failed siblings from same plan."""
    from agm.jobs import _get_failed_sibling_context

    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="DB helpers", description="d1")
    create_task(conn, plan_id=plan["id"], ordinal=1, title="CLI stuff", description="d2")
    t3 = create_task(conn, plan_id=plan["id"], ordinal=2, title="Docs", description="d3")

    # t1 failed, t2 still active, t3 is the one asking
    update_task_status(conn, t1["id"], "failed")

    ctx = _get_failed_sibling_context(conn, t3)
    assert "DB helpers" in ctx
    assert "NOT implemented" in ctx
    assert "CLI stuff" not in ctx  # not failed


def test_get_failed_sibling_context_none_failed(db_conn):
    """_get_failed_sibling_context should return empty when no siblings failed."""
    from agm.jobs import _get_failed_sibling_context

    conn = db_conn
    plan = make_plan(conn)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task A", description="d1")

    ctx = _get_failed_sibling_context(conn, t1)
    assert ctx == ""


# --- Channel context ---


def test_get_channel_context_no_session(db_conn):
    """Returns empty when plan has no session."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    result = _get_channel_context(conn, task)
    assert result == ""


def test_get_channel_context_no_messages(db_conn):
    """Returns empty when session has no context messages from others."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    plan = get_plan_request(conn, plan["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    result = _get_channel_context(conn, task)
    assert result == ""


def test_get_channel_context_includes_sibling_messages(db_conn):
    """Includes context messages from other executors."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    plan = get_plan_request(conn, plan["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="executor:aabbccdd",
        content="Created UserService with authenticate()",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="executor",
        content="Task completed: DB helpers",
    )

    result = _get_channel_context(conn, task)
    assert "executor-aabb" in result
    assert "Created UserService" in result
    assert "Task completed: DB helpers" in result
    assert "Context from other agents" in result


def test_get_channel_context_filters_own_messages(db_conn):
    """Filters out messages from this task's own executor."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    plan = get_plan_request(conn, plan["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    task_short = task["id"][:8]
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender=f"executor:{task_short}",
        content="My own message",
    )

    result = _get_channel_context(conn, task)
    assert result == ""


def test_get_channel_context_filters_executing_autopost(db_conn):
    """Filters bare 'Executing:' auto-posts but keeps 'Task completed:' messages."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    plan = get_plan_request(conn, plan["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    # Bare auto-post (noise — should be filtered)
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="executor",
        content="Executing: Add payment integration",
    )
    # Completion message (useful — should be kept)
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="executor",
        content="Task completed: Add payment integration",
    )

    result = _get_channel_context(conn, task)
    assert "Executing:" not in result
    assert "Task completed: Add payment integration" in result


def test_get_channel_context_prioritizes_targeted_steer_messages(db_conn):
    """Targeted steer messages should be included ahead of generic context noise."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    plan = get_plan_request(conn, plan["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="planner:11111111",
        content="Minor note",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="steer",
        sender="coordinator:22222222",
        recipient=f"executor:{task['id'][:8]}",
        content="Use the adapter boundary; avoid direct imports.",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="dm",
        sender="coordinator:33333333",
        recipient="executor:someoneelse",
        content="Ignore this",
    )

    result = _get_channel_context(conn, task)
    assert "Use the adapter boundary" in result
    assert "Ignore this" not in result
    assert "[steer|coordinator-2222]" in result


def test_publish_execution_context_summary_writes_channel_message(db_conn):
    """Executor summary distillation should persist a context message."""
    from agm.jobs_execution import _publish_execution_context_summary

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task A", description="d")

    _publish_execution_context_summary(conn, task, "Implemented adapter + migration wiring.")
    messages = list_channel_messages(conn, session["id"], kind="context")
    assert len(messages) == 1
    assert messages[0]["sender"] == f"executor:{task['id'][:8]}"
    assert "Implemented adapter" in messages[0]["content"]


def test_get_channel_context_prefers_latest_messages_when_capped(db_conn):
    """Context loader should sample the latest messages, not the oldest."""
    from agm.jobs import _get_channel_context

    conn = db_conn
    pid = get_project_id(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = make_plan(conn)
    set_plan_session_id(conn, plan["id"], session["id"])
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    for i in range(520):
        add_channel_message(
            conn,
            session_id=session["id"],
            kind="context",
            sender=f"planner:{i:08d}",
            content=f"old-{i}",
        )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="steer",
        sender="coordinator:feedbeef",
        recipient=f"executor:{task['id'][:8]}",
        content="Latest steer should be visible",
    )

    result = _get_channel_context(conn, task)
    assert "Latest steer should be visible" in result


# -- quality gate config --


def test_load_quality_gate_default():
    """_load_quality_gate returns default config when None."""
    from agm.jobs import _default_quality_gate, _load_quality_gate

    config = _load_quality_gate(None)
    assert config == _default_quality_gate()
    assert "auto_fix" in config
    assert "checks" in config


def test_load_quality_gate_custom():
    """_load_quality_gate parses custom JSON config."""
    import json

    from agm.jobs import _load_quality_gate

    custom = json.dumps(
        {
            "auto_fix": [{"name": "black", "cmd": ["black", "."]}],
            "checks": [{"name": "mypy", "cmd": ["mypy", "src/"], "timeout": 120}],
        }
    )
    config = _load_quality_gate(custom)
    assert len(config["auto_fix"]) == 1
    assert config["auto_fix"][0]["name"] == "black"
    assert len(config["checks"]) == 1
    assert config["checks"][0]["name"] == "mypy"


def test_load_quality_gate_invalid_json():
    """_load_quality_gate returns default on invalid JSON."""
    from agm.jobs import _default_quality_gate, _load_quality_gate

    config = _load_quality_gate("not json")
    assert config == _default_quality_gate()


def test_load_quality_gate_missing_checks():
    """_load_quality_gate returns default when checks key missing."""
    import json

    from agm.jobs import _default_quality_gate, _load_quality_gate

    config = _load_quality_gate(json.dumps({"auto_fix": []}))
    assert config == _default_quality_gate()


def test_project_quality_gate_db(db_conn):
    """set/get project quality gate in DB."""
    from agm.db import get_project_quality_gate, set_project_quality_gate

    conn = db_conn
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]

    # Initially None
    assert get_project_quality_gate(conn, pid) is None

    # Set custom
    custom = '{"checks": [{"name": "mypy", "cmd": ["mypy"]}]}'
    set_project_quality_gate(conn, pid, custom)
    assert get_project_quality_gate(conn, pid) == custom

    # Reset to None
    set_project_quality_gate(conn, pid, None)
    assert get_project_quality_gate(conn, pid) is None


def test_migration_adds_quality_gate_column(db_conn):
    """Schema migration should add quality_gate column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "quality_gate" in cols


def test_migration_v3_adds_default_backend(db_conn):
    """Schema migration v3 should add default_backend column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "default_backend" in cols


def test_resolve_backend_prefers_explicit_and_falls_back():
    """resolve_backend honors explicit backend then hard default."""
    assert resolve_backend("codex") == "codex"
    assert resolve_backend(None) == DEFAULT_BACKEND


def test_resolve_backend_rejects_invalid():
    """resolve_backend rejects invalid explicit backend values."""
    with pytest.raises(ValueError, match="Invalid backend"):
        resolve_backend("invalid")


def test_migration_v6_adds_plan_approval(db_conn):
    """Schema migration v6 should add plan_approval column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "plan_approval" in cols


def test_set_and_get_project_plan_approval(db_conn):
    """set/get project plan_approval round-trip."""
    from agm.db import get_project_plan_approval, set_project_plan_approval

    conn = db_conn
    proj = add_project(conn, "approval-test", "/tmp/approval-test")

    # Initially 'auto' (NULL → auto)
    assert get_project_plan_approval(conn, proj["id"]) == "auto"

    # Set to manual
    set_project_plan_approval(conn, proj["id"], "manual")
    assert get_project_plan_approval(conn, proj["id"]) == "manual"

    # Reset to None (→ auto)
    set_project_plan_approval(conn, proj["id"], None)
    assert get_project_plan_approval(conn, proj["id"]) == "auto"


def test_set_project_plan_approval_validates(db_conn):
    """Invalid plan_approval value should raise ValueError."""
    from agm.db import set_project_plan_approval

    conn = db_conn
    proj = add_project(conn, "approval-val", "/tmp/approval-val")

    with pytest.raises(ValueError, match="Invalid plan_approval"):
        set_project_plan_approval(conn, proj["id"], "invalid")


def test_migration_v29_adds_app_server_approval_policy(db_conn):
    """Schema migration v29 should add app_server_approval_policy column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "app_server_approval_policy" in cols


def test_set_and_get_project_app_server_approval_policy(db_conn):
    """set/get project app-server approval policy round-trip with defaults."""
    from agm.db import (
        get_project_app_server_approval_policy,
        set_project_app_server_approval_policy,
    )

    conn = db_conn
    proj = add_project(conn, "app-server-approval-test", "/tmp/app-server-approval-test")

    default_policy = get_project_app_server_approval_policy(conn, proj["id"])
    assert default_policy["item/commandExecution/requestApproval"] == "accept"
    assert default_policy["execCommandApproval"] == "approved"

    set_project_app_server_approval_policy(
        conn,
        proj["id"],
        {
            "item/commandExecution/requestApproval": "decline",
            "execCommandApproval": "denied",
        },
    )
    updated = get_project_app_server_approval_policy(conn, proj["id"])
    assert updated["item/commandExecution/requestApproval"] == "decline"
    assert updated["execCommandApproval"] == "denied"
    assert updated["item/fileChange/requestApproval"] == "accept"

    set_project_app_server_approval_policy(conn, proj["id"], None)
    reset = get_project_app_server_approval_policy(conn, proj["id"])
    assert reset["item/commandExecution/requestApproval"] == "accept"
    assert reset["execCommandApproval"] == "approved"


def test_set_project_app_server_approval_policy_validates(db_conn):
    """Invalid app-server approval policy values should raise ValueError."""
    from agm.db import set_project_app_server_approval_policy

    conn = db_conn
    proj = add_project(conn, "app-server-approval-validate", "/tmp/app-server-approval-validate")

    with pytest.raises(ValueError, match="Unknown app-server approval methods"):
        set_project_app_server_approval_policy(
            conn,
            proj["id"],
            {"unknown/method": "accept"},
        )

    with pytest.raises(ValueError, match="Invalid decision"):
        set_project_app_server_approval_policy(
            conn,
            proj["id"],
            {"execCommandApproval": "accept"},
        )


def test_migration_v30_adds_app_server_ask_for_approval(db_conn):
    """Schema migration v30 should add app_server_ask_for_approval column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "app_server_ask_for_approval" in cols


def test_migration_v31_adds_active_turn_tracking_columns(db_conn):
    """Schema migration v31 should add active turn tracking columns to tasks."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "active_turn_id" in cols
    assert "active_turn_started_at" in cols


def test_migration_v32_adds_task_steers_table(db_conn):
    """Schema migration v32 should add task_steers table."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(task_steers)").fetchall()}
    assert "task_id" in cols
    assert "session_id" in cols
    assert "live_error" in cols


def test_set_and_get_project_app_server_ask_for_approval(db_conn):
    """set/get AskForApproval policy should round-trip with sane defaults."""
    from agm.db import (
        get_project_app_server_ask_for_approval,
        set_project_app_server_ask_for_approval,
    )

    conn = db_conn
    proj = add_project(conn, "app-server-ask-policy-test", "/tmp/app-server-ask-policy-test")

    assert get_project_app_server_ask_for_approval(conn, proj["id"]) == "never"

    set_project_app_server_ask_for_approval(conn, proj["id"], "on-request")
    assert get_project_app_server_ask_for_approval(conn, proj["id"]) == "on-request"

    set_project_app_server_ask_for_approval(
        conn,
        proj["id"],
        {"reject": {"mcp_elicitations": True, "rules": False, "sandbox_approval": True}},
    )
    assert get_project_app_server_ask_for_approval(conn, proj["id"]) == {
        "reject": {"mcp_elicitations": True, "rules": False, "sandbox_approval": True}
    }

    set_project_app_server_ask_for_approval(conn, proj["id"], None)
    assert get_project_app_server_ask_for_approval(conn, proj["id"]) == "never"


def test_set_project_app_server_ask_for_approval_validates(db_conn):
    """Invalid AskForApproval policies should raise ValueError."""
    from agm.db import set_project_app_server_ask_for_approval

    conn = db_conn
    proj = add_project(conn, "app-server-ask-validate", "/tmp/app-server-ask-validate")

    with pytest.raises(ValueError, match="Invalid app-server ask-for-approval policy"):
        set_project_app_server_ask_for_approval(conn, proj["id"], "always")

    with pytest.raises(ValueError, match="Reject ask-for-approval policy must include only"):
        set_project_app_server_ask_for_approval(
            conn,
            proj["id"],
            {"reject": {"rules": True}},
        )

    with pytest.raises(ValueError, match="must be boolean"):
        set_project_app_server_ask_for_approval(
            conn,
            proj["id"],
            {"reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": "yes"}},
        )


# -- Schema v23: post_merge_command --


def test_migration_v23_adds_post_merge_command(db_conn):
    """Schema migration v23 should add post_merge_command column to projects."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    assert "post_merge_command" in cols


def test_set_get_post_merge_command(db_conn):
    """set/get project post_merge_command round-trip."""
    from agm.db import get_project_post_merge_command, set_project_post_merge_command

    conn = db_conn
    proj = add_project(conn, "pmc-test", "/tmp/pmc-test")

    # Initially None
    assert get_project_post_merge_command(conn, proj["id"]) is None

    # Set a command
    set_project_post_merge_command(conn, proj["id"], "scripts/post-merge.sh")
    assert get_project_post_merge_command(conn, proj["id"]) == "scripts/post-merge.sh"

    # Reset to None
    set_project_post_merge_command(conn, proj["id"], None)
    assert get_project_post_merge_command(conn, proj["id"]) is None


def test_awaiting_approval_is_valid_task_creation_status():
    """awaiting_approval should be in VALID_TASK_CREATION_STATUSES."""
    from agm.db import VALID_TASK_CREATION_STATUSES

    assert "awaiting_approval" in VALID_TASK_CREATION_STATUSES


# -- Schema v7: enrichment columns --


def test_schema_v7_enrichment_columns_exist(db_conn):
    """Schema v7 should add enriched_prompt and enrichment_thread_id columns."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(plans)").fetchall()}
    assert "enriched_prompt" in cols
    assert "enrichment_thread_id" in cols


def test_update_plan_enrichment_sets_enriched_prompt(db_conn):
    """update_plan_enrichment should set enriched_prompt."""
    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_enrichment(conn, plan["id"], enriched_prompt="Enriched: do stuff")
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["enriched_prompt"] == "Enriched: do stuff"
    assert found["enrichment_thread_id"] is None


def test_update_plan_enrichment_sets_thread_id(db_conn):
    """update_plan_enrichment should set enrichment_thread_id."""
    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_enrichment(conn, plan["id"], enrichment_thread_id="thread-123")
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["enrichment_thread_id"] == "thread-123"
    assert found["enriched_prompt"] is None


def test_update_plan_enrichment_sets_both(db_conn):
    """update_plan_enrichment should set both fields at once."""
    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_enrichment(
        conn, plan["id"], enriched_prompt="enriched", enrichment_thread_id="tid"
    )
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["enriched_prompt"] == "enriched"
    assert found["enrichment_thread_id"] == "tid"


def test_update_plan_enrichment_returns_false_for_missing(db_conn):
    """update_plan_enrichment should return False for nonexistent plan."""
    conn = db_conn
    ok = update_plan_enrichment(conn, "nonexistent", enriched_prompt="x")
    assert ok is False


# -- exploration columns (schema v26) --


def test_schema_v26_exploration_columns_exist(db_conn):
    """Schema v26 should add exploration_context and exploration_thread_id columns."""
    conn = db_conn
    cols = {row[1] for row in conn.execute("PRAGMA table_info(plans)").fetchall()}
    assert "exploration_context" in cols
    assert "exploration_thread_id" in cols


def test_update_plan_exploration_sets_context(db_conn):
    """update_plan_exploration should set exploration_context."""
    from agm.db import update_plan_exploration

    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_exploration(conn, plan["id"], exploration_context='{"summary":"test"}')
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["exploration_context"] == '{"summary":"test"}'
    assert found["exploration_thread_id"] is None


def test_update_plan_exploration_sets_thread_id(db_conn):
    """update_plan_exploration should set exploration_thread_id."""
    from agm.db import update_plan_exploration

    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_exploration(conn, plan["id"], exploration_thread_id="thread-exp-1")
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["exploration_thread_id"] == "thread-exp-1"
    assert found["exploration_context"] is None


def test_update_plan_exploration_sets_both(db_conn):
    """update_plan_exploration should set both fields at once."""
    from agm.db import update_plan_exploration

    conn = db_conn
    plan = make_plan(conn)
    ok = update_plan_exploration(
        conn,
        plan["id"],
        exploration_context='{"summary":"x"}',
        exploration_thread_id="tid-exp",
    )
    assert ok is True
    found = get_plan_request(conn, plan["id"])
    assert found["exploration_context"] == '{"summary":"x"}'
    assert found["exploration_thread_id"] == "tid-exp"


def test_update_plan_exploration_returns_false_for_missing(db_conn):
    """update_plan_exploration should return False for nonexistent plan."""
    from agm.db import update_plan_exploration

    conn = db_conn
    ok = update_plan_exploration(conn, "nonexistent", exploration_context="x")
    assert ok is False


# -- get_unanswered_question_count --


def test_get_unanswered_question_count_zero(db_conn):
    """No questions should return 0."""
    conn = db_conn
    plan = make_plan(conn)
    assert get_unanswered_question_count(conn, plan["id"]) == 0


def test_get_unanswered_question_count_tracks_answers(db_conn):
    """Should count only unanswered questions."""
    conn = db_conn
    plan = make_plan(conn)
    add_plan_question(conn, plan_id=plan["id"], question="Question 1?")
    add_plan_question(conn, plan_id=plan["id"], question="Question 2?")
    assert get_unanswered_question_count(conn, plan["id"]) == 2

    # Answer one
    q_id = conn.execute(
        "SELECT id FROM plan_questions WHERE plan_id = ? LIMIT 1", (plan["id"],)
    ).fetchone()["id"]
    conn.execute("UPDATE plan_questions SET answer = 'yes' WHERE id = ?", (q_id,))
    conn.commit()
    assert get_unanswered_question_count(conn, plan["id"]) == 1


# -- reset_task_for_reexecution --


def test_reset_task_for_reexecution_approved_to_running(db_conn):
    """reset_task_for_reexecution should transition approved -> running."""
    conn = db_conn
    plan = make_plan(conn)
    finalize_plan_request(conn, plan["id"], "The plan")
    task = create_task(conn, plan_id=plan["id"], ordinal=1, title="Test", description="d")
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    set_task_thread_id(conn, task["id"], "old-thread")
    set_task_reviewer_thread_id(conn, task["id"], "old-reviewer")
    claim_task(conn, task["id"], actor="test", caller="cli")

    ok = reset_task_for_reexecution(
        conn, task["id"], "new-branch", "/tmp/new-wt", record_history=True
    )
    assert ok is True

    found = get_task(conn, task["id"])
    assert found["status"] == "running"
    assert found["branch"] == "new-branch"
    assert found["worktree"] == "/tmp/new-wt"
    assert found["thread_id"] is None
    assert found["reviewer_thread_id"] is None

    history = list_status_history(conn, entity_type="task", entity_id=task["id"])
    assert history[-1]["old_status"] == "approved"
    assert history[-1]["new_status"] == "running"


def test_reset_task_for_reexecution_rejects_non_approved(db_conn):
    """reset_task_for_reexecution should fail for non-approved tasks."""
    conn = db_conn
    plan = make_plan(conn)
    finalize_plan_request(conn, plan["id"], "The plan")
    task = create_task(conn, plan_id=plan["id"], ordinal=1, title="Test", description="d")
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")

    ok = reset_task_for_reexecution(conn, task["id"], "branch", "/tmp/wt")
    assert ok is False

    found = get_task(conn, task["id"])
    assert found["status"] == "running"  # Unchanged


# -- Schema v7 migration from v6 --


def test_schema_v7_migration_from_v6():
    """DB at schema v6 should gain enrichment columns on migration to v7."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    # Create a DB at schema version 6 (pre-enrichment)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            default_backend TEXT,
            model_config TEXT,
            plan_approval TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE plans (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            actor TEXT NOT NULL,
            caller TEXT NOT NULL,
            backend TEXT NOT NULL,
            pid INTEGER,
            thread_id TEXT,
            parent_id TEXT,
            task_creation_status TEXT,
            model TEXT,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'blocked',
            pid INTEGER,
            thread_id TEXT,
            reviewer_thread_id TEXT,
            actor TEXT,
            caller TEXT,
            branch TEXT,
            worktree TEXT,
            bucket TEXT,
            priority TEXT,
            skip_review INTEGER NOT NULL DEFAULT 0,
            skip_merge INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE task_blocks (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id),
            blocked_by_task_id TEXT REFERENCES tasks(id),
            external_factor TEXT,
            reason TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            resolved_at TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE plan_questions (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            question TEXT NOT NULL,
            options TEXT,
            answer TEXT,
            answered_by TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            answered_at TEXT
        );
        CREATE TABLE plan_logs (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE task_logs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id),
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            actor TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    # Open via get_connection — triggers migration
    migrated = get_connection(db_path)
    version = migrated.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION

    # Verify enrichment columns exist
    cols = {row[1] for row in migrated.execute("PRAGMA table_info(plans)").fetchall()}
    assert "enriched_prompt" in cols
    assert "enrichment_thread_id" in cols
    migrated.close()
