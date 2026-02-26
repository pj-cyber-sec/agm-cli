"""Tests for the database layer."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from agm.db import (
    SCHEMA_VERSION,
    _build_timeline_filters,
    add_channel_message,
    add_project,
    add_task_block,
    add_task_steer,
    connect,
    create_plan_request,
    create_session,
    create_task,
    finalize_plan_request,
    finish_session,
    get_channel_message,
    get_connection,
    get_project,
    get_project_base_branch,
    get_project_model_config,
    get_session,
    inspect_sqlite_integrity,
    list_channel_messages,
    list_plan_requests,
    list_plan_timeline_rows,
    list_projects,
    list_sessions,
    list_status_history,
    list_task_blocks,
    list_task_steers,
    purge_data,
    purge_preview_counts,
    record_status_change,
    remove_project,
    set_plan_session_id,
    set_project_base_branch,
    set_project_model_config,
    update_session_status,
    update_task_status,
)


def tmp_conn():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    return get_connection(db_path)


def test_add_and_list():
    conn = tmp_conn()
    add_project(conn, "myapp", "/tmp/myapp")
    projects = list_projects(conn)
    assert len(projects) == 1
    assert projects[0]["name"] == "myapp"
    assert projects[0]["dir"] == "/tmp/myapp"
    conn.close()


def test_get_by_name():
    conn = tmp_conn()
    p = add_project(conn, "foo", "/tmp/foo")
    found = get_project(conn, "foo")
    assert found is not None
    assert found["id"] == p["id"]
    conn.close()


def test_get_by_id():
    conn = tmp_conn()
    p = add_project(conn, "bar", "/tmp/bar")
    found = get_project(conn, p["id"])
    assert found is not None
    assert found["name"] == "bar"
    conn.close()


def test_get_not_found():
    conn = tmp_conn()
    assert get_project(conn, "nope") is None
    conn.close()


def test_remove():
    conn = tmp_conn()
    add_project(conn, "gone", "/tmp/gone")
    result = remove_project(conn, "gone")
    assert result is not None
    assert result["plan_ids"] == []
    assert result["task_ids"] == []
    assert list_projects(conn) == []
    conn.close()


def test_remove_not_found():
    conn = tmp_conn()
    assert remove_project(conn, "nope") is None
    conn.close()


def test_remove_with_cross_project_blockers():
    """Removing a project cleans up task_blocks where its tasks are the blocker.

    Regression test: previously only task_blocks where task_id belonged to the
    project were deleted. Cross-project blocked_by_task_id refs were left dangling,
    causing sqlite3.IntegrityError on the subsequent tasks DELETE.
    """
    conn = tmp_conn()
    # Project A has a task that blocks a task in Project B
    add_project(conn, "proj-a", "/tmp/proj-a")
    add_project(conn, "proj-b", "/tmp/proj-b")
    pid_a = get_project(conn, "proj-a")["id"]
    pid_b = get_project(conn, "proj-b")["id"]

    plan_a = create_plan_request(conn, project_id=pid_a, prompt="a", caller="cli", backend="codex")
    finalize_plan_request(conn, plan_a["id"], '{"title":"a","summary":"s","tasks":[]}')
    task_a = create_task(conn, plan_id=plan_a["id"], ordinal=0, title="A", description="d")

    plan_b = create_plan_request(conn, project_id=pid_b, prompt="b", caller="cli", backend="codex")
    finalize_plan_request(conn, plan_b["id"], '{"title":"b","summary":"s","tasks":[]}')
    task_b = create_task(conn, plan_id=plan_b["id"], ordinal=0, title="B", description="d")

    # B is blocked by A (cross-project blocker)
    add_task_block(conn, task_id=task_b["id"], blocked_by_task_id=task_a["id"])
    assert len(list_task_blocks(conn, task_b["id"])) == 1

    # Removing project A should not cause FK constraint error
    result = remove_project(conn, "proj-a")
    assert result is not None
    assert task_a["id"] in result["task_ids"]

    # The cross-project blocker should be cleaned up
    remaining = list_task_blocks(conn, task_b["id"])
    assert len(remaining) == 0

    # Project B still intact
    assert get_project(conn, "proj-b") is not None
    conn.close()


def test_duplicate_name():
    conn = tmp_conn()
    add_project(conn, "dup", "/tmp/a")
    try:
        add_project(conn, "dup", "/tmp/b")
        raise AssertionError("Should have raised")
    except Exception:
        pass
    conn.close()


# -- index tests --

EXPECTED_INDEXES = {
    "idx_plans_project_id": ("plans", ["project_id"]),
    "idx_plans_parent_id": ("plans", ["parent_id"]),
    "idx_plans_status": ("plans", ["status"]),
    "idx_tasks_plan_id": ("tasks", ["plan_id"]),
    "idx_tasks_status": ("tasks", ["status"]),
    "idx_task_blocks_task_id": ("task_blocks", ["task_id"]),
    "idx_task_blocks_blocker": ("task_blocks", ["blocked_by_task_id", "resolved"]),
    "idx_plan_questions_plan_id": ("plan_questions", ["plan_id"]),
    "idx_plan_logs_plan_id": ("plan_logs", ["plan_id"]),
    "idx_task_logs_task_id": ("task_logs", ["task_id"]),
    "idx_status_history_entity_timeline": (
        "status_history",
        ["entity_type", "entity_id", "created_at", "id"],
    ),
    "idx_status_history_stale_age": (
        "status_history",
        ["entity_type", "new_status", "created_at", "id"],
    ),
}


def _get_custom_indexes(conn):
    """Get all non-autoindex custom indexes from the database."""
    rows = conn.execute(
        "SELECT name, tbl_name FROM sqlite_master "
        "WHERE type = 'index' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _get_index_columns(conn, index_name):
    """Get ordered column names for an index."""
    rows = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
    return [row[2] for row in sorted(rows, key=lambda r: r[0])]


def test_indexes_exist():
    """All expected custom indexes are created on connection."""
    conn = tmp_conn()
    indexes = _get_custom_indexes(conn)
    for name, (table, _cols) in EXPECTED_INDEXES.items():
        assert name in indexes, f"Missing index {name}"
        assert indexes[name] == table, f"Index {name} on wrong table"
    conn.close()


def test_index_columns():
    """Indexes cover the expected columns in the right order."""
    conn = tmp_conn()
    for name, (_table, expected_cols) in EXPECTED_INDEXES.items():
        actual_cols = _get_index_columns(conn, name)
        assert actual_cols == expected_cols, (
            f"Index {name}: expected {expected_cols}, got {actual_cols}"
        )
    conn.close()


def test_indexes_idempotent():
    """Reopening the same DB doesn't error or duplicate indexes."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn1 = get_connection(db_path)
    indexes1 = _get_custom_indexes(conn1)
    conn1.close()

    conn2 = get_connection(db_path)
    indexes2 = _get_custom_indexes(conn2)
    conn2.close()

    assert indexes1 == indexes2


def test_get_connection_migrates_status_history_table_idempotently():
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
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row["name"] for row in migrated.execute("PRAGMA table_info(status_history)").fetchall()}
    assert cols == {
        "id",
        "entity_type",
        "entity_id",
        "old_status",
        "new_status",
        "actor",
        "created_at",
    }
    migrated.close()

    reopened = get_connection(db_path)
    cols = {row["name"] for row in reopened.execute("PRAGMA table_info(status_history)").fetchall()}
    assert cols == {
        "id",
        "entity_type",
        "entity_id",
        "old_status",
        "new_status",
        "actor",
        "created_at",
    }
    reopened.close()


def test_migration_adds_base_branch_column_for_v4():
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
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            actor TEXT NOT NULL,
            caller TEXT NOT NULL,
            backend TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy", "/tmp/legacy"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    cols = {row[1] for row in migrated.execute("PRAGMA table_info(projects)").fetchall()}
    assert "base_branch" in cols
    row = migrated.execute("SELECT base_branch FROM projects WHERE id = ?", ("p1",)).fetchone()
    assert row["base_branch"] is None
    assert get_project_base_branch(migrated, "p1") == "main"
    migrated.close()


def test_migration_v4_is_idempotent_for_existing_db():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            base_branch TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir, base_branch) VALUES (?, ?, ?, ?)",
        ("p1", "with-base", "/tmp/with-base", "develop"),
    )
    conn.commit()
    conn.close()

    first = get_connection(db_path)
    assert first.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols_first = {row[1] for row in first.execute("PRAGMA table_info(projects)").fetchall()}
    row_first = first.execute("SELECT base_branch FROM projects WHERE id = ?", ("p1",)).fetchone()
    assert "base_branch" in cols_first
    assert row_first["base_branch"] == "develop"
    first.close()

    second = get_connection(db_path)
    assert second.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols_second = {row[1] for row in second.execute("PRAGMA table_info(projects)").fetchall()}
    row_second = second.execute("SELECT base_branch FROM projects WHERE id = ?", ("p1",)).fetchone()
    assert cols_second == cols_first
    assert row_second["base_branch"] == "develop"
    second.close()


def test_set_project_base_branch_rejects_empty_or_whitespace():
    conn = tmp_conn()
    project = add_project(conn, "base-branch-empty", "/tmp/base-branch-empty")
    with pytest.raises(ValueError, match="base_branch must be a non-empty string"):
        set_project_base_branch(conn, project["id"], "")
    with pytest.raises(ValueError, match="base_branch must be a non-empty string"):
        set_project_base_branch(conn, project["id"], "   ")
    conn.close()


def test_get_project_base_branch_fallbacks_to_main():
    conn = tmp_conn()
    project = add_project(conn, "base-branch-fallback", "/tmp/base-branch-fallback")

    assert get_project_base_branch(conn, project["id"]) == "main"
    set_project_base_branch(conn, project["id"], "develop")
    assert get_project_base_branch(conn, project["id"]) == "develop"
    set_project_base_branch(conn, project["id"], None)
    assert get_project_base_branch(conn, project["id"]) == "main"
    assert get_project_base_branch(conn, "missing") == "main"
    conn.close()


def test_get_project_base_branch_falls_back_when_column_missing():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        ("p1", "legacy-no-base", "/tmp/legacy-no-base"),
    )
    conn.commit()
    assert get_project_base_branch(conn, "p1") == "main"
    conn.close()


def test_set_and_get_project_model_config():
    conn = tmp_conn()
    project = add_project(conn, "model-config", "/tmp/model-config")

    assert get_project_model_config(conn, project["id"]) is None
    set_project_model_config(conn, project["id"], '{"planner":"think","task":"work"}')
    assert get_project_model_config(conn, project["id"]) == '{"planner":"think","task":"work"}'
    set_project_model_config(conn, project["id"], None)
    assert get_project_model_config(conn, project["id"]) is None
    assert get_project_model_config(conn, "missing") is None
    conn.close()


def test_get_project_model_config_falls_back_when_column_missing():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            base_branch TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)", ("p1", "legacy", "/tmp/legacy")
    )
    conn.commit()

    assert get_project_model_config(conn, "p1") is None
    conn.close()


def test_migration_adds_model_fields_for_v5():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            base_branch TEXT,
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

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            priority TEXT,
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
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description) VALUES (?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy desc"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    project_cols = {row[1] for row in migrated.execute("PRAGMA table_info(projects)").fetchall()}
    plan_cols = {row[1] for row in migrated.execute("PRAGMA table_info(plans)").fetchall()}
    task_cols = {row[1] for row in migrated.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "model_config" in project_cols
    assert "model" in plan_cols
    assert "model" in task_cols

    assert (
        migrated.execute("SELECT model_config FROM projects WHERE id = ?", ("p1",)).fetchone()[
            "model_config"
        ]
        is None
    )
    assert (
        migrated.execute("SELECT model FROM plans WHERE id = ?", ("plan1",)).fetchone()["model"]
        is None
    )
    assert (
        migrated.execute("SELECT model FROM tasks WHERE id = ?", ("task1",)).fetchone()["model"]
        is None
    )
    migrated.close()

    reopened = get_connection(db_path)
    assert (
        reopened.execute("SELECT model_config FROM projects WHERE id = ?", ("p1",)).fetchone()[
            "model_config"
        ]
        is None
    )
    assert (
        reopened.execute("SELECT model FROM plans WHERE id = ?", ("plan1",)).fetchone()["model"]
        is None
    )
    assert (
        reopened.execute("SELECT model FROM tasks WHERE id = ?", ("task1",)).fetchone()["model"]
        is None
    )
    reopened.close()


def test_migration_v5_handles_partially_migrated_models():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            dir TEXT NOT NULL,
            base_branch TEXT,
            model_config TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE plans (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            parent_id TEXT REFERENCES plans(id),
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            plan TEXT,
            model TEXT,
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

        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            ordinal INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            files TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            priority TEXT,
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
        "INSERT INTO projects (id, name, dir, model_config) VALUES (?, ?, ?, ?)",
        ("p1", "partial", "/tmp/partial", '{"planner":"legacy"}'),
    )
    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, actor, caller, backend, model) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("plan1", "p1", "legacy prompt", "alice", "cli", "codex", "gpt-5"),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description) VALUES (?, ?, ?, ?, ?)",
        ("task1", "plan1", 0, "legacy task", "legacy desc"),
    )
    conn.commit()
    conn.close()

    migrated = get_connection(db_path)
    assert (
        migrated.execute("SELECT model_config FROM projects WHERE id = ?", ("p1",)).fetchone()[
            "model_config"
        ]
        == '{"planner":"legacy"}'
    )
    assert (
        migrated.execute("SELECT model FROM plans WHERE id = ?", ("plan1",)).fetchone()["model"]
        == "gpt-5"
    )
    assert (
        migrated.execute("SELECT model FROM tasks WHERE id = ?", ("task1",)).fetchone()["model"]
        is None
    )
    migrated.close()


def test_migration_v15_adds_question_fields_and_migrates_options():
    """v15 adds header + multi_select to plan_questions and migrates string options."""
    import json

    from agm.db import _migrate_to_v15

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Minimal plan_questions table without header/multi_select
    conn.executescript(
        """
        CREATE TABLE plan_questions (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            question TEXT NOT NULL,
            options TEXT,
            answer TEXT, answered_by TEXT,
            created_at TEXT, answered_at TEXT
        );
        """
    )
    # Insert old-format string-array options
    old_opts = json.dumps(["OAuth2", "JWT", "API Key"])
    conn.execute(
        "INSERT INTO plan_questions (id, plan_id, question, options) VALUES (?, ?, ?, ?)",
        ("q1", "plan1", "Which auth?", old_opts),
    )
    # Insert a question with no options (should be untouched)
    conn.execute(
        "INSERT INTO plan_questions (id, plan_id, question, options) VALUES (?, ?, ?, ?)",
        ("q2", "plan1", "Any preference?", None),
    )
    # Insert already-migrated object options (should be untouched)
    new_opts = json.dumps([{"label": "A", "description": "x"}])
    conn.execute(
        "INSERT INTO plan_questions (id, plan_id, question, options) VALUES (?, ?, ?, ?)",
        ("q3", "plan1", "Pick one", new_opts),
    )
    conn.commit()

    _migrate_to_v15(conn)

    # Verify new columns exist
    cols = {r[1] for r in conn.execute("PRAGMA table_info(plan_questions)").fetchall()}
    assert "header" in cols
    assert "multi_select" in cols
    # Verify string-array options were migrated to object format
    q1 = conn.execute("SELECT options FROM plan_questions WHERE id = 'q1'").fetchone()
    parsed = json.loads(q1["options"])
    assert len(parsed) == 3
    assert parsed[0] == {"label": "OAuth2", "description": ""}
    assert parsed[2] == {"label": "API Key", "description": ""}
    # Verify null options untouched
    q2 = conn.execute("SELECT options FROM plan_questions WHERE id = 'q2'").fetchone()
    assert q2["options"] is None
    # Verify already-migrated options untouched
    q3 = conn.execute("SELECT options FROM plan_questions WHERE id = 'q3'").fetchone()
    assert json.loads(q3["options"]) == [{"label": "A", "description": "x"}]
    conn.close()


def test_record_status_change_rejects_invalid_entity_type():
    conn = tmp_conn()
    with pytest.raises(ValueError, match="Invalid entity_type 'job'"):
        record_status_change(
            conn,
            entity_type="job",
            entity_id="abc123",
            old_status=None,
            new_status="running",
        )
    conn.close()


def test_record_status_change_noops_when_status_is_unchanged():
    conn = tmp_conn()
    add_project(conn, "status-noop", "/tmp/status-noop")
    pid = get_project(conn, "status-noop")["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="noop", caller="cli", backend="codex")

    result = record_status_change(
        conn,
        entity_type="plan",
        entity_id=plan["id"],
        old_status="running",
        new_status="running",
    )
    assert result == {}
    rows = list_status_history(conn, entity_type="plan", entity_id=plan["id"])
    assert rows == []
    conn.close()


def test_record_status_change_noop_still_validates_actor():
    conn = tmp_conn()
    add_project(conn, "status-noop-actor", "/tmp/status-noop-actor")
    pid = get_project(conn, "status-noop-actor")["id"]
    plan = create_plan_request(
        conn,
        project_id=pid,
        prompt="noop actor",
        caller="cli",
        backend="codex",
    )

    with pytest.raises(ValueError, match="actor must be a string or None"):
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=plan["id"],
            old_status="running",
            new_status="running",
            actor=123,  # type: ignore[arg-type]
        )
    conn.close()


def test_build_timeline_filters_builds_entity_and_timerange_conditions():
    filters, params = _build_timeline_filters(
        project_id=None,
        plan_id="  plan-001  ",
        since=None,
        until=None,
    )
    assert filters == ["entity_id = ?"]
    assert params == ["plan-001"]


def test_list_plan_timeline_rows_filters_plan_scope_and_timerange():
    conn = tmp_conn()
    add_project(conn, "timelines-a", "/tmp/timelines-a")
    add_project(conn, "timelines-b", "/tmp/timelines-b")
    project_a = get_project(conn, "timelines-a")
    project_b = get_project(conn, "timelines-b")

    plan_a = create_plan_request(
        conn,
        project_id=project_a["id"],
        prompt="plan a",
        caller="cli",
        backend="codex",
    )
    plan_b = create_plan_request(
        conn,
        project_id=project_b["id"],
        prompt="plan b",
        caller="cli",
        backend="codex",
    )

    record_status_change(
        conn,
        entity_type="plan",
        entity_id=plan_a["id"],
        old_status="pending",
        new_status="running",
        created_at="2026-01-01T10:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=plan_a["id"],
        old_status="running",
        new_status="finalized",
        created_at="2026-01-03T10:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=plan_b["id"],
        old_status="pending",
        new_status="running",
        created_at="2026-01-02T10:00:00Z",
    )

    full_rows = list_plan_timeline_rows(conn, f"  {plan_a['id']}  ")
    assert [row["created_at"] for row in full_rows] == [
        "2026-01-01T10:00:00Z",
        "2026-01-03T10:00:00Z",
    ]

    scoped_rows = list_plan_timeline_rows(
        conn,
        plan_a["id"],
        project_id=project_a["id"],
    )
    assert len(scoped_rows) == 2

    mismatched_scope_rows = list_plan_timeline_rows(
        conn,
        plan_a["id"],
        project_id=project_b["id"],
    )
    assert mismatched_scope_rows == []

    since_rows = list_plan_timeline_rows(
        conn,
        plan_a["id"],
        since="2026-01-03T00:00:00Z",
    )
    assert [row["created_at"] for row in since_rows] == ["2026-01-03T10:00:00Z"]

    until_rows = list_plan_timeline_rows(
        conn,
        plan_a["id"],
        until="2026-01-01T23:59:59Z",
    )
    assert [row["created_at"] for row in until_rows] == ["2026-01-01T10:00:00Z"]

    bounded_rows = list_plan_timeline_rows(
        conn,
        plan_a["id"],
        since="2026-01-01T10:00:00Z",
        until="2026-01-03T10:00:00Z",
    )
    assert [row["created_at"] for row in bounded_rows] == [
        "2026-01-01T10:00:00Z",
        "2026-01-03T10:00:00Z",
    ]

    conn.close()


def test_list_plan_timeline_rows_keeps_plan_rows_when_prompt_row_non_enriching():
    conn = tmp_conn()
    add_project(conn, "timeline", "/tmp/timeline")
    pid = get_project(conn, "timeline")["id"]
    p = create_plan_request(conn, project_id=pid, prompt="plan", caller="cli", backend="codex")

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
        entity_type="prompt",
        entity_id=p["id"],
        old_status="pending",
        new_status="finalized",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="running",
        new_status="awaiting_input",
        actor="alice",
        created_at="2026-01-01T00:00:10Z",
    )
    conn.commit()

    rows = list_plan_timeline_rows(conn, p["id"])
    assert [row["created_at"] for row in rows] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:10Z",
    ]
    assert rows[0]["entity_type"] in {"plan", "prompt"}
    assert rows[1]["entity_type"] in {"plan", "prompt"}
    assert rows[0]["entity_type"] != rows[1]["entity_type"]
    assert rows[2]["entity_type"] == "plan"
    assert rows[0]["duration_seconds"] == 0
    assert rows[1]["duration_seconds"] == 10
    assert rows[2]["duration_seconds"] is None
    assert rows[2]["next_created_at"] is None
    conn.close()


def test_list_plan_timeline_rows_drops_plan_running_when_prompt_is_enriching():
    conn = tmp_conn()
    add_project(conn, "timeline2", "/tmp/timeline2")
    pid = get_project(conn, "timeline2")["id"]
    p = create_plan_request(conn, project_id=pid, prompt="plan", caller="cli", backend="codex")

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
        entity_type="prompt",
        entity_id=p["id"],
        old_status="pending",
        new_status="enriching",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="running",
        new_status="awaiting_input",
        actor="alice",
        created_at="2026-01-01T00:00:10Z",
    )
    conn.commit()

    rows = list_plan_timeline_rows(conn, p["id"])
    assert [row["entity_type"] for row in rows] == ["prompt", "plan"]
    assert [row["old_status"] for row in rows] == ["pending", "running"]
    assert [row["new_status"] for row in rows] == ["enriching", "awaiting_input"]
    assert rows[0]["duration_seconds"] == 10
    assert rows[1]["duration_seconds"] is None
    assert rows[1]["next_created_at"] is None
    conn.close()


def test_list_plan_timeline_rows_keeps_plan_running_after_awaiting_input_with_prompt_enriching():
    conn = tmp_conn()
    add_project(conn, "timeline3", "/tmp/timeline3")
    pid = get_project(conn, "timeline3")["id"]
    p = create_plan_request(conn, project_id=pid, prompt="plan", caller="cli", backend="codex")

    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="awaiting_input",
        new_status="running",
        actor="alice",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="prompt",
        entity_id=p["id"],
        old_status="pending",
        new_status="enriching",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="running",
        new_status="awaiting_input",
        actor="alice",
        created_at="2026-01-01T00:00:10Z",
    )
    conn.commit()

    rows = list_plan_timeline_rows(conn, p["id"])
    assert [row["entity_type"] for row in rows] == ["plan", "prompt", "plan"]
    assert rows[0]["old_status"] == "awaiting_input"
    assert rows[0]["new_status"] == "running"
    assert rows[1]["old_status"] == "pending"
    assert rows[1]["new_status"] == "enriching"
    assert rows[2]["old_status"] == "running"
    assert rows[2]["new_status"] == "awaiting_input"
    assert rows[0]["duration_seconds"] == 0
    assert rows[1]["duration_seconds"] == 10
    assert rows[2]["duration_seconds"] is None
    assert rows[2]["next_created_at"] is None
    conn.close()


# -- task status transitions --


def test_update_task_status_blocks_completed_to_running():
    conn = tmp_conn()
    add_project(conn, "terminal-task", "/tmp/terminal-task")
    project_id = get_project(conn, "terminal-task")["id"]
    plan = create_plan_request(
        conn, project_id=project_id, prompt="plan", caller="cli", backend="codex"
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="task", description="desc")
    conn.execute("UPDATE tasks SET status = 'completed' WHERE id = ?", (task["id"],))
    conn.commit()

    before = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()

    result = update_task_status(conn, task["id"], "running")
    assert result is False
    after = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert after["status"] == "completed"
    assert after["status"] == before["status"]
    assert after["updated_at"] == before["updated_at"]
    conn.close()


def test_update_task_status_blocks_failed_to_ready():
    conn = tmp_conn()
    add_project(conn, "terminal-task-2", "/tmp/terminal-task-2")
    project_id = get_project(conn, "terminal-task-2")["id"]
    plan = create_plan_request(
        conn, project_id=project_id, prompt="plan", caller="cli", backend="codex"
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="task", description="desc")
    conn.execute("UPDATE tasks SET status = 'failed' WHERE id = ?", (task["id"],))
    conn.commit()

    before = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()

    result = update_task_status(conn, task["id"], "ready")
    assert result is False
    after = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert after["status"] == "failed"
    assert after["status"] == before["status"]
    assert after["updated_at"] == before["updated_at"]
    conn.close()


def test_update_task_status_blocks_cancelled_to_ready():
    conn = tmp_conn()
    add_project(conn, "terminal-task-3", "/tmp/terminal-task-3")
    project_id = get_project(conn, "terminal-task-3")["id"]
    plan = create_plan_request(
        conn, project_id=project_id, prompt="plan", caller="cli", backend="codex"
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="task", description="desc")
    conn.execute("UPDATE tasks SET status = 'cancelled' WHERE id = ?", (task["id"],))
    conn.commit()

    before = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()

    result = update_task_status(conn, task["id"], "ready")
    assert result is False
    after = conn.execute(
        "SELECT status, updated_at FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert after["status"] == "cancelled"
    assert after["status"] == before["status"]
    assert after["updated_at"] == before["updated_at"]
    conn.close()


def test_update_task_status_allows_non_terminal_transition():
    conn = tmp_conn()
    add_project(conn, "terminal-task-4", "/tmp/terminal-task-4")
    project_id = get_project(conn, "terminal-task-4")["id"]
    plan = create_plan_request(
        conn, project_id=project_id, prompt="plan", caller="cli", backend="codex"
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="task", description="desc")
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task["id"],))
    conn.commit()

    result = update_task_status(conn, task["id"], "running")
    assert result is True
    current = conn.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert current["status"] == "running"
    conn.close()


# -- busy_timeout --


def test_busy_timeout_set():
    """get_connection sets busy_timeout pragma."""
    conn = tmp_conn()
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 10000
    conn.close()


# -- connect() context manager --


def test_connect_context_manager():
    """connect() yields a usable connection and closes it on exit."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    with connect(db_path) as conn:
        add_project(conn, "ctx", "/tmp/ctx")
        projects = list_projects(conn)
        assert len(projects) == 1
    # Connection should be closed after exiting the context
    # Attempting to use it should raise
    try:
        conn.execute("SELECT 1")
        closed = False
    except Exception:
        closed = True
    assert closed


def test_connect_closes_on_exception():
    """connect() closes connection even when body raises."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    try:
        with connect(db_path) as conn:
            add_project(conn, "err", "/tmp/err")
            raise ValueError("boom")
    except ValueError:
        pass
    try:
        conn.execute("SELECT 1")
        closed = False
    except Exception:
        closed = True
    assert closed


# -- doctor read helpers --


def test_inspect_sqlite_integrity_ok():
    conn = tmp_conn()
    result = inspect_sqlite_integrity(conn)
    assert result["ok"] is True
    assert result["rows"] == ["ok"]
    assert result["failures"] == []
    conn.close()


def test_inspect_sqlite_integrity_non_ok_rows_normalized():
    class _FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _FakeConn:
        def execute(self, query):
            assert query == "PRAGMA integrity_check"
            return _FakeCursor(
                [
                    (" ok ",),
                    ("*** in database main ***",),
                    ("",),
                    (None,),
                ]
            )

    result = inspect_sqlite_integrity(_FakeConn())
    assert result["ok"] is False
    assert result["rows"] == ["ok", "*** in database main ***"]
    assert result["failures"] == ["*** in database main ***"]


def test_aegis_is_builtin_caller():
    """aegis is a built-in caller."""
    from agm.callers import get_all_callers

    assert "aegis" in get_all_callers()


def test_schema_v12_is_noop():
    """Schema v12 migration is a no-op (query-mode columns removed)."""
    conn = tmp_conn()
    task_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "mode" not in task_cols
    assert "result" not in task_cols
    assert "output_schema" not in task_cols
    conn.close()


# -- sessions --


def _make_project(conn, name="testproj"):
    """Helper to create a project and return its ID."""
    add_project(conn, name, f"/tmp/{name}")
    return get_project(conn, name)["id"]


def test_create_session_returns_open_session():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="plan_request", trigger_prompt="do X")
    assert session["status"] == "open"
    assert session["project_id"] == pid
    assert session["trigger"] == "plan_request"
    assert session["trigger_prompt"] == "do X"
    assert session["started_at"] is None
    assert session["finished_at"] is None
    assert len(session["id"]) == 12
    conn.close()


def test_get_session_found_and_missing():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    found = get_session(conn, session["id"])
    assert found is not None
    assert found["id"] == session["id"]
    assert get_session(conn, "nonexistent") is None
    conn.close()


def test_list_sessions_filters_by_project_and_status():
    conn = tmp_conn()
    pid_a = _make_project(conn, "proj-a")
    pid_b = _make_project(conn, "proj-b")
    s1 = create_session(conn, project_id=pid_a, trigger="do")
    s2 = create_session(conn, project_id=pid_a, trigger="plan_request")
    s3 = create_session(conn, project_id=pid_b, trigger="do")
    update_session_status(conn, s2["id"], "active")

    # all sessions
    assert len(list_sessions(conn)) == 3
    # filter by project
    assert {s["id"] for s in list_sessions(conn, project_id=pid_a)} == {s1["id"], s2["id"]}
    # filter by status
    assert [s["id"] for s in list_sessions(conn, status="active")] == [s2["id"]]
    # filter by statuses (multi)
    open_sessions = list_sessions(conn, statuses=["open"])
    assert {s["id"] for s in open_sessions} == {s1["id"], s3["id"]}
    conn.close()


def test_update_session_status_rejects_invalid():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    with pytest.raises(ValueError, match="Invalid session status 'bogus'"):
        update_session_status(conn, session["id"], "bogus")
    conn.close()


def test_update_session_status_to_active_sets_started_at():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    assert get_session(conn, session["id"])["started_at"] is None

    update_session_status(conn, session["id"], "active")
    row = get_session(conn, session["id"])
    assert row["status"] == "active"
    assert row["started_at"] is not None
    first_started = row["started_at"]

    # calling active again should NOT overwrite started_at
    update_session_status(conn, session["id"], "active")
    row = get_session(conn, session["id"])
    assert row["started_at"] == first_started
    conn.close()


def test_finish_session_sets_terminal_status_and_finished_at():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    update_session_status(conn, session["id"], "active")

    finish_session(conn, session["id"], "completed")
    row = get_session(conn, session["id"])
    assert row["status"] == "completed"
    assert row["finished_at"] is not None
    conn.close()


def test_finish_session_rejects_non_terminal_status():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    with pytest.raises(ValueError, match="Invalid terminal status 'active'"):
        finish_session(conn, session["id"], "active")
    conn.close()


def test_set_plan_session_id_links_plan_to_session():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="plan_request")
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")

    # plan starts with no session
    plans = list_plan_requests(conn, project_id=pid)
    assert plans[0]["session_id"] is None

    set_plan_session_id(conn, plan["id"], session["id"])
    plans = list_plan_requests(conn, project_id=pid)
    assert plans[0]["session_id"] == session["id"]

    # list_plan_requests can filter by session_id
    by_session = list_plan_requests(conn, session_id=session["id"])
    assert len(by_session) == 1
    assert by_session[0]["id"] == plan["id"]
    conn.close()


# -- channel messages --


def test_add_and_list_channel_messages():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")

    m1 = add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="enrichment",
        content="Prompt enriched",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="broadcast",
        sender="planner",
        content="Plan finalized with 3 tasks",
    )
    m3 = add_channel_message(
        conn,
        session_id=session["id"],
        kind="dm",
        sender="user",
        content="Focus on the API",
        recipient="executor",
    )

    assert len(m1["id"]) == 12
    assert m1["kind"] == "context"
    assert m1["recipient"] is None
    assert m3["recipient"] == "executor"

    # list all messages in order
    messages = list_channel_messages(conn, session["id"])
    assert len(messages) == 3
    assert [m["sender"] for m in messages] == ["enrichment", "planner", "user"]

    # filter by kind
    dm_messages = list_channel_messages(conn, session["id"], kind="dm")
    assert len(dm_messages) == 1
    assert dm_messages[0]["id"] == m3["id"]

    # filter by sender
    planner_messages = list_channel_messages(conn, session["id"], sender="planner")
    assert len(planner_messages) == 1
    assert planner_messages[0]["content"] == "Plan finalized with 3 tasks"

    # filter by recipient
    executor_messages = list_channel_messages(conn, session["id"], recipient="executor")
    assert len(executor_messages) == 1
    assert executor_messages[0]["id"] == m3["id"]

    # limit
    limited = list_channel_messages(conn, session["id"], limit=2)
    assert len(limited) == 2

    # offset
    paged = list_channel_messages(conn, session["id"], limit=1, offset=1)
    assert len(paged) == 1
    assert paged[0]["id"] == messages[1]["id"]
    conn.close()


def test_add_channel_message_rejects_invalid_kind():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    with pytest.raises(ValueError, match="Invalid message kind 'shout'"):
        add_channel_message(
            conn, session_id=session["id"], kind="shout", sender="user", content="hey"
        )
    conn.close()


def test_get_channel_message():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    m = add_channel_message(
        conn,
        session_id=session["id"],
        kind="steer",
        sender="user",
        content="skip tests",
        metadata='{"plan_id": "abc123"}',
    )
    found = get_channel_message(conn, m["id"])
    assert found is not None
    assert found["content"] == "skip tests"
    assert found["metadata"] == '{"plan_id": "abc123"}'
    assert get_channel_message(conn, "nonexistent") is None
    conn.close()


def test_add_channel_message_with_metadata():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    m = add_channel_message(
        conn,
        session_id=session["id"],
        kind="question",
        sender="enrichment",
        content="Which API?",
        metadata='{"plan_id": "p1"}',
    )
    assert m["metadata"] == '{"plan_id": "p1"}'
    stored = get_channel_message(conn, m["id"])
    assert stored["metadata"] == '{"plan_id": "p1"}'
    conn.close()


# -- task steers --


def test_add_and_list_task_steers():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    plan = create_plan_request(conn, project_id=pid, prompt="p", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"x","summary":"s","tasks":[]}')
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    msg = add_channel_message(
        conn,
        session_id=session["id"],
        kind="steer",
        sender="operator:api",
        recipient=f"executor:{task['id'][:8]}",
        content="Do this first",
    )
    steer = add_task_steer(
        conn,
        task_id=task["id"],
        session_id=session["id"],
        message_id=msg["id"],
        sender="operator:api",
        recipient=f"executor:{task['id'][:8]}",
        content="Do this first",
        reason="manual",
        live_requested=True,
        live_applied=False,
        live_error="expectedTurnId mismatch",
        thread_id="thread-1",
        expected_turn_id="turn-1",
    )
    assert steer["task_id"] == task["id"]
    rows = list_task_steers(conn, task_id=task["id"])
    assert len(rows) == 1
    assert rows[0]["message_id"] == msg["id"]
    assert rows[0]["live_requested"] == 1
    assert rows[0]["live_applied"] == 0
    assert rows[0]["live_error"] == "expectedTurnId mismatch"
    conn.close()


# -- purge cascade for sessions --


def test_purge_deletes_sessions_and_messages():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    add_channel_message(
        conn, session_id=session["id"], kind="context", sender="system", content="msg1"
    )
    add_channel_message(
        conn, session_id=session["id"], kind="broadcast", sender="planner", content="msg2"
    )

    # also create a plan to satisfy purge_data logic
    create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")

    result = purge_data(conn, pid)
    assert result["counts"]["sessions"] == 1
    assert result["counts"]["channel_messages"] == 2

    # verify they're actually gone
    assert get_session(conn, session["id"]) is None
    assert list_channel_messages(conn, session["id"]) == []
    conn.close()


def test_purge_preview_counts_includes_sessions_and_messages():
    conn = tmp_conn()
    pid = _make_project(conn)
    session = create_session(conn, project_id=pid, trigger="do")
    add_channel_message(
        conn, session_id=session["id"], kind="context", sender="system", content="msg"
    )
    create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")

    counts = purge_preview_counts(conn, pid)
    assert counts["sessions"] == 1
    assert counts["channel_messages"] == 1
    assert counts["plans"] == 1
    conn.close()


# -- schema version --


def test_schema_version_is_32():
    assert SCHEMA_VERSION == 32


def test_sessions_table_exists_with_expected_columns():
    conn = tmp_conn()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    expected = {
        "id",
        "project_id",
        "trigger",
        "trigger_prompt",
        "status",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    }
    assert expected == cols
    conn.close()


def test_channel_messages_table_exists_with_expected_columns():
    conn = tmp_conn()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(channel_messages)").fetchall()}
    expected = {
        "id",
        "session_id",
        "kind",
        "sender",
        "recipient",
        "content",
        "metadata",
        "created_at",
    }
    assert expected == cols
    conn.close()


def test_task_steers_table_exists_with_expected_columns():
    conn = tmp_conn()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(task_steers)").fetchall()}
    expected = {
        "id",
        "task_id",
        "session_id",
        "message_id",
        "sender",
        "recipient",
        "content",
        "reason",
        "metadata",
        "live_requested",
        "live_applied",
        "live_error",
        "thread_id",
        "expected_turn_id",
        "applied_turn_id",
        "created_at",
    }
    assert expected == cols
    conn.close()


def test_plans_table_has_session_id_column():
    conn = tmp_conn()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(plans)").fetchall()}
    assert "session_id" in cols
    conn.close()
