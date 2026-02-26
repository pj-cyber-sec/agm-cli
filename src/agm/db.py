"""SQLite database for agm state."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from agm.backends import IMPLEMENTED_BACKENDS
from agm.callers import get_all_callers
from agm.paths import DEFAULT_DB_PATH

VALID_BACKENDS = {"codex"}
DEFAULT_BACKEND = "codex"
VALID_PLAN_STATUSES = {"pending", "running", "awaiting_input", "finalized", "failed", "cancelled"}
PLAN_TERMINAL_STATUSES = {"finalized", "failed", "cancelled"}
VALID_TASK_STATUSES = {
    "blocked",
    "ready",
    "running",
    "review",
    "rejected",
    "approved",
    "completed",
    "failed",
    "cancelled",
}
TASK_TERMINAL_STATUSES = {"completed", "cancelled", "failed"}
VALID_TASK_PRIORITIES = {"high", "medium", "low"}
VALID_TASK_CREATION_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "awaiting_approval",
}
VALID_PROMPT_STATUSES = {
    "pending",
    "enriching",
    "awaiting_input",
    "finalized",
    "failed",
    "cancelled",
}

VALID_SESSION_STATUSES = {"open", "active", "completed", "failed"}
SESSION_TERMINAL_STATUSES = {"completed", "failed"}
VALID_MESSAGE_KINDS = {"steer", "question", "broadcast", "dm", "context"}


def _utcnow() -> str:
    """ISO 8601 UTC timestamp matching SQLite strftime format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Bump when adding migrations. 0 = legacy (pre-versioning).
SCHEMA_VERSION = 32

SCHEMA = """\
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    dir TEXT UNIQUE NOT NULL,
    model_config TEXT,
    base_branch TEXT,
    plan_approval TEXT,
    post_merge_command TEXT,
    quality_gate TEXT,
    setup_result TEXT,
    app_server_approval_policy TEXT,
    app_server_ask_for_approval TEXT,
    default_backend TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    trigger TEXT NOT NULL,
    trigger_prompt TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    parent_id TEXT REFERENCES plans(id),
    prompt TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt_status TEXT NOT NULL DEFAULT 'pending',
    task_creation_status TEXT,
    model TEXT,
    plan TEXT,
    actor TEXT NOT NULL,
    caller TEXT NOT NULL,
    backend TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    thread_id TEXT,
    enriched_prompt TEXT,
    enrichment_thread_id TEXT,
    exploration_context TEXT,
    exploration_thread_id TEXT,
    started_at TEXT,
    finished_at TEXT,
    mode TEXT NOT NULL DEFAULT 'standard',
    session_id TEXT REFERENCES sessions(id),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS plan_questions (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    question TEXT NOT NULL,
    options TEXT,
    header TEXT,
    multi_select INTEGER NOT NULL DEFAULT 0,
    answer TEXT,
    answered_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    answered_at TEXT
);

CREATE TABLE IF NOT EXISTS plan_logs (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'plan',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    files TEXT,
    status TEXT NOT NULL DEFAULT 'blocked',
    failure_reason TEXT,
    model TEXT,
    priority TEXT,
    actor TEXT,
    caller TEXT,
    branch TEXT,
    worktree TEXT,
    reviewer_thread_id TEXT,
    skip_review INTEGER NOT NULL DEFAULT 0,
    skip_merge INTEGER NOT NULL DEFAULT 0,
    bucket TEXT,
    merge_commit TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    thread_id TEXT,
    active_turn_id TEXT,
    active_turn_started_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS task_blocks (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    blocked_by_task_id TEXT REFERENCES tasks(id),
    external_factor TEXT,
    reason TEXT,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS task_logs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'task',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS channel_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL,
    sender TEXT NOT NULL,
    recipient TEXT,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS task_steers (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    message_id TEXT REFERENCES channel_messages(id),
    sender TEXT NOT NULL,
    recipient TEXT,
    content TEXT NOT NULL,
    reason TEXT,
    metadata TEXT,
    live_requested INTEGER NOT NULL DEFAULT 0,
    live_applied INTEGER NOT NULL DEFAULT 0,
    live_error TEXT,
    thread_id TEXT,
    expected_turn_id TEXT,
    applied_turn_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS trace_events (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('plan', 'task')),
    entity_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    turn_index INTEGER NOT NULL DEFAULT 0,
    ordinal INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT,
    data TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


# -- Row TypedDicts matching table schemas --


class ProjectRow(TypedDict):
    id: str
    name: str
    dir: str
    model_config: str | None
    base_branch: str | None
    plan_approval: str | None
    post_merge_command: str | None
    quality_gate: str | None
    setup_result: str | None
    app_server_approval_policy: str | None
    app_server_ask_for_approval: str | None
    default_backend: str | None
    created_at: str


class SessionRow(TypedDict):
    id: str
    project_id: str
    trigger: str
    trigger_prompt: str | None
    status: str
    started_at: str | None
    finished_at: str | None
    created_at: str
    updated_at: str


class PlanRow(TypedDict):
    id: str
    project_id: str
    parent_id: str | None
    prompt: str
    status: str
    prompt_status: str
    task_creation_status: str | None
    model: str | None
    plan: str | None
    actor: str
    caller: str
    backend: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    pid: int | None
    thread_id: str | None
    enriched_prompt: str | None
    enrichment_thread_id: str | None
    exploration_context: str | None
    exploration_thread_id: str | None
    started_at: str | None
    finished_at: str | None
    mode: str
    session_id: str | None
    created_at: str
    updated_at: str


class TaskRow(TypedDict):
    id: str
    plan_id: str
    ordinal: int
    title: str
    description: str
    files: str | None
    status: str
    failure_reason: str | None
    model: str | None
    priority: str | None
    actor: str | None
    caller: str | None
    branch: str | None
    worktree: str | None
    reviewer_thread_id: str | None
    skip_review: int
    skip_merge: int
    bucket: str | None
    merge_commit: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    pid: int | None
    thread_id: str | None
    active_turn_id: str | None
    active_turn_started_at: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str
    updated_at: str


class PlanQuestionRow(TypedDict):
    id: str
    plan_id: str
    question: str
    options: str | None
    header: str | None
    multi_select: int
    answer: str | None
    answered_by: str | None
    created_at: str
    answered_at: str | None


class TaskBlockRow(TypedDict):
    id: str
    task_id: str
    blocked_by_task_id: str | None
    external_factor: str | None
    reason: str | None
    resolved: int
    resolved_at: str | None
    created_at: str


class ChannelMessageRow(TypedDict):
    id: str
    session_id: str
    kind: str
    sender: str
    recipient: str | None
    content: str
    metadata: str | None
    created_at: str


class TaskSteerRow(TypedDict):
    id: str
    task_id: str
    session_id: str
    message_id: str | None
    sender: str
    recipient: str | None
    content: str
    reason: str | None
    metadata: str | None
    live_requested: int
    live_applied: int
    live_error: str | None
    thread_id: str | None
    expected_turn_id: str | None
    applied_turn_id: str | None
    created_at: str


class TraceEventRow(TypedDict):
    id: str
    entity_type: str
    entity_id: str
    stage: str
    turn_index: int
    ordinal: int
    event_type: str
    status: str | None
    data: str
    created_at: str


_INSERT_TASK_LOG = (
    "INSERT INTO task_logs (id, task_id, level, message, source) VALUES (?, ?, ?, ?, ?)"
)


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(SCHEMA)

    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if current_version < SCHEMA_VERSION:
        _migrate(conn, current_version)
        _create_indexes(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    return conn


@contextlib.contextmanager
def connect(db_path: Path = DEFAULT_DB_PATH):
    """Context manager wrapper for get_connection().

    Usage:
        with connect() as conn:
            do_stuff(conn)
    # conn.close() is guaranteed even on exceptions.
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_def: str, cols: set[str]
) -> None:
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    """Pre-v1: all columns added after the initial schema."""
    cols = _table_columns(conn, "plans")
    for col, defn in [
        ("pid", "INTEGER"),
        ("thread_id", "TEXT"),
        ("parent_id", "TEXT REFERENCES plans(id)"),
        ("task_creation_status", "TEXT"),
        ("input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("output_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ]:
        _add_column_if_missing(conn, "plans", col, defn, cols)
    conn.execute("UPDATE plans SET input_tokens = 0 WHERE input_tokens IS NULL")
    conn.execute("UPDATE plans SET output_tokens = 0 WHERE output_tokens IS NULL")
    conn.execute("UPDATE plans SET status = 'pending' WHERE status = 'drafting'")

    task_cols = _table_columns(conn, "tasks")
    for col, defn in [
        ("actor", "TEXT"),
        ("caller", "TEXT"),
        ("branch", "TEXT"),
        ("worktree", "TEXT"),
        ("reviewer_thread_id", "TEXT"),
        ("input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("output_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("skip_review", "INTEGER NOT NULL DEFAULT 0"),
        ("skip_merge", "INTEGER NOT NULL DEFAULT 0"),
        ("bucket", "TEXT"),
        ("priority", "TEXT"),
    ]:
        _add_column_if_missing(conn, "tasks", col, defn, task_cols)
    conn.execute("UPDATE tasks SET input_tokens = 0 WHERE input_tokens IS NULL")
    conn.execute("UPDATE tasks SET output_tokens = 0 WHERE output_tokens IS NULL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL CHECK (entity_type IN ('plan', 'task', 'prompt')),
            entity_id TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            actor TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_status_history_entity_timeline
            ON status_history(entity_type, entity_id, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_status_history_stale_age
            ON status_history(entity_type, new_status, created_at, id);
    """)


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "quality_gate", "TEXT", cols)


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "default_backend", "TEXT", cols)


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "base_branch", "TEXT", cols)


def _migrate_to_v5(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "model_config", "TEXT", cols)
    _add_column_if_missing(conn, "plans", "model", "TEXT", _table_columns(conn, "plans"))
    _add_column_if_missing(conn, "tasks", "model", "TEXT", _table_columns(conn, "tasks"))


def _migrate_to_v6(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "plan_approval", "TEXT", cols)


def _migrate_to_v7(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "enriched_prompt", "TEXT", cols)
    _add_column_if_missing(conn, "plans", "enrichment_thread_id", "TEXT", cols)


def _migrate_to_v8(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "tasks")
    _add_column_if_missing(conn, "tasks", "merge_commit", "TEXT", cols)


def _migrate_to_v9(conn: sqlite3.Connection) -> None:
    """Add started_at column to plans and tasks, backfill from status_history."""
    plan_cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "started_at", "TEXT", plan_cols)
    task_cols = _table_columns(conn, "tasks")
    _add_column_if_missing(conn, "tasks", "started_at", "TEXT", task_cols)

    # Backfill from status_history: first transition to 'running'
    conn.execute("""
        UPDATE plans SET started_at = (
            SELECT MIN(created_at) FROM status_history
            WHERE entity_type = 'plan' AND entity_id = plans.id
              AND new_status = 'running'
        ) WHERE started_at IS NULL
    """)
    conn.execute("""
        UPDATE tasks SET started_at = (
            SELECT MIN(created_at) FROM status_history
            WHERE entity_type = 'task' AND entity_id = tasks.id
              AND new_status = 'running'
        ) WHERE started_at IS NULL
    """)


def _migrate_to_v10(conn: sqlite3.Connection) -> None:
    """Add finished_at column to plans and tasks, backfill from status_history."""
    plan_cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "finished_at", "TEXT", plan_cols)
    task_cols = _table_columns(conn, "tasks")
    _add_column_if_missing(conn, "tasks", "finished_at", "TEXT", task_cols)

    # Backfill plans: timestamp of transition to terminal status
    conn.execute("""
        UPDATE plans SET finished_at = (
            SELECT MIN(created_at) FROM status_history
            WHERE entity_type = 'plan' AND entity_id = plans.id
              AND new_status IN ('finalized', 'failed')
        ) WHERE finished_at IS NULL
          AND status IN ('finalized', 'failed')
    """)
    # Backfill tasks: timestamp of transition to terminal status
    conn.execute("""
        UPDATE tasks SET finished_at = (
            SELECT MIN(created_at) FROM status_history
            WHERE entity_type = 'task' AND entity_id = tasks.id
              AND new_status IN ('completed', 'failed', 'cancelled')
        ) WHERE finished_at IS NULL
          AND status IN ('completed', 'failed', 'cancelled')
    """)


def _migrate_to_v11(conn: sqlite3.Connection) -> None:
    """Add unique index on projects(dir) to prevent duplicate directory registrations."""
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_dir ON projects(dir)")


def _migrate_to_v12(conn: sqlite3.Connection) -> None:
    """No-op (previously added query-mode columns, now removed)."""


def _migrate_to_v13(conn: sqlite3.Connection) -> None:
    """Make plans.project_id nullable for ephemeral (projectless) queries."""
    # Check if project_id is already nullable (fresh DB from SCHEMA)
    for row in conn.execute("PRAGMA table_info(plans)").fetchall():
        if row[1] == "project_id":
            if row[3] == 0:  # notnull=0 means already nullable
                return
            break
    # Existing DB: project_id is NOT NULL — must rebuild table.
    # Commit current txn so we can toggle PRAGMA foreign_keys.
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    # Clean up leftover temp table from a previously interrupted migration.
    conn.execute("DROP TABLE IF EXISTS plans_new")
    old_cols = {r[1] for r in conn.execute("PRAGMA table_info(plans)").fetchall()}
    conn.execute(
        "CREATE TABLE plans_new ("
        "id TEXT PRIMARY KEY, "
        "project_id TEXT REFERENCES projects(id), "
        "prompt TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', "
        "plan TEXT, "
        "actor TEXT NOT NULL, "
        "caller TEXT NOT NULL, "
        "backend TEXT NOT NULL, "
        "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), "
        "updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), "
        "pid INTEGER, "
        "thread_id TEXT, "
        "parent_id TEXT REFERENCES plans(id), "
        "task_creation_status TEXT, "
        "input_tokens INTEGER NOT NULL DEFAULT 0, "
        "output_tokens INTEGER NOT NULL DEFAULT 0, "
        "model TEXT, "
        "enriched_prompt TEXT, "
        "enrichment_thread_id TEXT, "
        "exploration_context TEXT, "
        "exploration_thread_id TEXT, "
        "started_at TEXT, "
        "finished_at TEXT"
        ")"
    )
    new_cols = {r[1] for r in conn.execute("PRAGMA table_info(plans_new)").fetchall()}
    shared = sorted(old_cols & new_cols)
    col_list = ", ".join(shared)
    conn.execute(f"INSERT INTO plans_new ({col_list}) SELECT {col_list} FROM plans")
    conn.execute("DROP TABLE plans")
    conn.execute("ALTER TABLE plans_new RENAME TO plans")
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


def _migrate_to_v14(conn: sqlite3.Connection) -> None:
    """Add cached_input_tokens and reasoning_tokens columns to plans and tasks."""
    for table in ("plans", "tasks"):
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in ("cached_input_tokens", "reasoning_tokens"):
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")


def _migrate_to_v15(conn: sqlite3.Connection) -> None:
    """Add header, multi_select to plan_questions; migrate options to object format."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(plan_questions)").fetchall()}
    if "header" not in existing:
        conn.execute("ALTER TABLE plan_questions ADD COLUMN header TEXT")
    if "multi_select" not in existing:
        conn.execute(
            "ALTER TABLE plan_questions ADD COLUMN multi_select INTEGER NOT NULL DEFAULT 0"
        )
    # Migrate existing string-array options to object-array format:
    # ["A", "B"] → [{"label": "A", "description": ""}, {"label": "B", "description": ""}]
    rows = conn.execute(
        "SELECT id, options FROM plan_questions WHERE options IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            parsed = json.loads(row["options"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            migrated = [{"label": opt, "description": ""} for opt in parsed]
            conn.execute(
                "UPDATE plan_questions SET options = ? WHERE id = ?",
                (json.dumps(migrated), row["id"]),
            )


def _migrate_to_v16(conn: sqlite3.Connection) -> None:
    """Pipeline modes refactor: prompt_status, blocked/rejected statuses, remove query mode."""
    # Add prompt_status column to plans
    plan_cols = _table_columns(conn, "plans")
    _add_column_if_missing(
        conn, "plans", "prompt_status", "TEXT NOT NULL DEFAULT 'pending'", plan_cols
    )

    # Rename task status 'pending' → 'blocked'
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE status = 'pending'")

    # Backfill prompt_status from existing plan state:
    # Plans that already have enriched_prompt → finalized
    conn.execute(
        "UPDATE plans SET prompt_status = 'finalized' "
        "WHERE enriched_prompt IS NOT NULL AND prompt_status = 'pending'"
    )
    # Plans in awaiting_input → awaiting_input
    conn.execute(
        "UPDATE plans SET prompt_status = 'awaiting_input' "
        "WHERE status = 'awaiting_input' AND prompt_status = 'pending'"
    )
    # Terminal plans → finalized prompt_status (enrichment is done or irrelevant)
    conn.execute(
        "UPDATE plans SET prompt_status = 'finalized' "
        "WHERE status IN ('finalized', 'failed', 'cancelled') AND prompt_status = 'pending'"
    )
    # Running plans with enrichment_thread_id but no enriched_prompt → enriching
    conn.execute(
        "UPDATE plans SET prompt_status = 'enriching' "
        "WHERE status = 'running' AND enrichment_thread_id IS NOT NULL "
        "AND enriched_prompt IS NULL AND prompt_status = 'pending'"
    )


def _migrate_to_v17(conn: sqlite3.Connection) -> None:
    """Remove Claude backend: reset default_backend='claude' to NULL (falls back to codex)."""
    conn.execute("UPDATE projects SET default_backend = NULL WHERE default_backend = 'claude'")


def _migrate_to_v18(conn: sqlite3.Connection) -> None:
    """No-op (previously added agent_instructions_mode column, now removed)."""


def _migrate_to_v19(conn: sqlite3.Connection) -> None:
    """Add source column to plan_logs."""
    cols = _table_columns(conn, "plan_logs")
    _add_column_if_missing(conn, "plan_logs", "source", "TEXT NOT NULL DEFAULT 'plan'", cols)


def _migrate_to_v20(conn: sqlite3.Connection) -> None:
    """Widen status_history CHECK constraint to allow 'prompt' entity_type."""
    # SQLite cannot ALTER CHECK constraints — must rebuild the table.
    conn.execute("PRAGMA foreign_keys = OFF")
    # Clean up leftover temp table from a previously interrupted migration.
    conn.execute("DROP TABLE IF EXISTS status_history_new")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS status_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL CHECK (entity_type IN ('plan', 'task', 'prompt')),
            entity_id TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            actor TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)
    conn.execute("""
        INSERT INTO status_history_new (id, entity_type, entity_id, old_status,
            new_status, actor, created_at)
        SELECT id, entity_type, entity_id, old_status, new_status, actor, created_at
        FROM status_history
    """)
    conn.execute("DROP TABLE status_history")
    conn.execute("ALTER TABLE status_history_new RENAME TO status_history")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_status_history_entity_timeline "
        "ON status_history(entity_type, entity_id, created_at, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_status_history_stale_age "
        "ON status_history(entity_type, new_status, created_at, id)"
    )
    conn.execute("PRAGMA foreign_keys = ON")


def _migrate_to_v21(conn: sqlite3.Connection) -> None:
    """Add source column to task_logs for actor distinction."""
    cols = _table_columns(conn, "task_logs")
    _add_column_if_missing(conn, "task_logs", "source", "TEXT NOT NULL DEFAULT 'task'", cols)


def _migrate_to_v22(conn: sqlite3.Connection) -> None:
    """Add mode column to plans for quick vs standard distinction."""
    cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "mode", "TEXT NOT NULL DEFAULT 'standard'", cols)


def _migrate_to_v23(conn: sqlite3.Connection) -> None:
    """Add post_merge_command column to projects."""
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "post_merge_command", "TEXT", cols)


def _migrate_to_v24(conn: sqlite3.Connection) -> None:
    """Add failure_reason column to tasks."""
    cols = _table_columns(conn, "tasks")
    _add_column_if_missing(conn, "tasks", "failure_reason", "TEXT", cols)


def _migrate_to_v25(conn: sqlite3.Connection) -> None:
    """Add sessions table, channel_messages table, and plans.session_id."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            trigger TEXT NOT NULL,
            trigger_prompt TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            started_at TEXT,
            finished_at TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS channel_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            kind TEXT NOT NULL,
            sender TEXT NOT NULL,
            recipient TEXT,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    plan_cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "session_id", "TEXT REFERENCES sessions(id)", plan_cols)


def _migrate_to_v26(conn: sqlite3.Connection) -> None:
    """Add exploration columns to plans."""
    cols = _table_columns(conn, "plans")
    _add_column_if_missing(conn, "plans", "exploration_context", "TEXT", cols)
    _add_column_if_missing(conn, "plans", "exploration_thread_id", "TEXT", cols)


def _migrate_to_v27(conn: sqlite3.Connection) -> None:
    """Add trace_events table for rich execution tracing."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trace_events (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL CHECK (entity_type IN ('plan', 'task')),
            entity_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            ordinal INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT,
            data TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)


def _migrate_to_v28(conn: sqlite3.Connection) -> None:
    """Add setup_result column to projects for traceability."""
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "setup_result", "TEXT", cols)


def _migrate_to_v29(conn: sqlite3.Connection) -> None:
    """Add app_server_approval_policy column to projects."""
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "app_server_approval_policy", "TEXT", cols)


def _migrate_to_v30(conn: sqlite3.Connection) -> None:
    """Add app_server_ask_for_approval column to projects."""
    cols = _table_columns(conn, "projects")
    _add_column_if_missing(conn, "projects", "app_server_ask_for_approval", "TEXT", cols)


def _migrate_to_v31(conn: sqlite3.Connection) -> None:
    """Add active-turn tracking columns to tasks."""
    cols = _table_columns(conn, "tasks")
    _add_column_if_missing(conn, "tasks", "active_turn_id", "TEXT", cols)
    _add_column_if_missing(conn, "tasks", "active_turn_started_at", "TEXT", cols)


def _migrate_to_v32(conn: sqlite3.Connection) -> None:
    """Add task_steers table for steer audit and replay."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_steers (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id),
            session_id TEXT NOT NULL REFERENCES sessions(id),
            message_id TEXT REFERENCES channel_messages(id),
            sender TEXT NOT NULL,
            recipient TEXT,
            content TEXT NOT NULL,
            reason TEXT,
            metadata TEXT,
            live_requested INTEGER NOT NULL DEFAULT 0,
            live_applied INTEGER NOT NULL DEFAULT 0,
            live_error TEXT,
            thread_id TEXT,
            expected_turn_id TEXT,
            applied_turn_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)


_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migrate_to_v1),
    (2, _migrate_to_v2),
    (3, _migrate_to_v3),
    (4, _migrate_to_v4),
    (5, _migrate_to_v5),
    (6, _migrate_to_v6),
    (7, _migrate_to_v7),
    (8, _migrate_to_v8),
    (9, _migrate_to_v9),
    (10, _migrate_to_v10),
    (11, _migrate_to_v11),
    (12, _migrate_to_v12),
    (13, _migrate_to_v13),
    (14, _migrate_to_v14),
    (15, _migrate_to_v15),
    (16, _migrate_to_v16),
    (17, _migrate_to_v17),
    (18, _migrate_to_v18),
    (19, _migrate_to_v19),
    (20, _migrate_to_v20),
    (21, _migrate_to_v21),
    (22, _migrate_to_v22),
    (23, _migrate_to_v23),
    (24, _migrate_to_v24),
    (25, _migrate_to_v25),
    (26, _migrate_to_v26),
    (27, _migrate_to_v27),
    (28, _migrate_to_v28),
    (29, _migrate_to_v29),
    (30, _migrate_to_v30),
    (31, _migrate_to_v31),
    (32, _migrate_to_v32),
]


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Run schema migrations from from_version to SCHEMA_VERSION.

    Each migration function contains idempotent column-existence checks so
    it's safe for both legacy DBs (upgrading) and fresh DBs (all columns
    already in SCHEMA, checks are no-ops). Commit is handled by the caller.
    """
    for version, migration_fn in _MIGRATIONS:
        if from_version < version:
            migration_fn(conn)


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Create non-PK indexes for common query patterns. Idempotent."""
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_plans_project_id ON plans(project_id);
        CREATE INDEX IF NOT EXISTS idx_plans_parent_id ON plans(parent_id);
        CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_plan_id ON tasks(plan_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_task_blocks_task_id ON task_blocks(task_id);
        CREATE INDEX IF NOT EXISTS idx_task_blocks_task_resolved
            ON task_blocks(task_id, resolved);
        CREATE INDEX IF NOT EXISTS idx_task_blocks_blocker
            ON task_blocks(blocked_by_task_id, resolved);
        CREATE INDEX IF NOT EXISTS idx_plan_questions_plan_id
            ON plan_questions(plan_id);
        CREATE INDEX IF NOT EXISTS idx_plan_logs_plan_id ON plan_logs(plan_id);
        CREATE INDEX IF NOT EXISTS idx_task_logs_task_id ON task_logs(task_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions(project_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        CREATE INDEX IF NOT EXISTS idx_channel_messages_session_id
            ON channel_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_channel_messages_session_kind
            ON channel_messages(session_id, kind);
        CREATE INDEX IF NOT EXISTS idx_task_steers_task_id
            ON task_steers(task_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_task_steers_session_id
            ON task_steers(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_plans_session_id ON plans(session_id);
        CREATE INDEX IF NOT EXISTS idx_trace_events_entity
            ON trace_events(entity_type, entity_id, ordinal);
        CREATE INDEX IF NOT EXISTS idx_trace_events_type
            ON trace_events(entity_type, entity_id, event_type);
    """)


def _validate_base_branch(base_branch: str | None) -> str | None:
    if base_branch is None:
        return None
    if not isinstance(base_branch, str):
        raise ValueError("base_branch must be a non-empty string or None.")
    normalized = base_branch.strip()
    if not normalized:
        raise ValueError("base_branch must be a non-empty string or None.")
    return normalized


def _status_set_for_entity_type(entity_type: str) -> set[str]:
    if entity_type == "plan":
        return VALID_PLAN_STATUSES
    if entity_type == "task":
        return VALID_TASK_STATUSES
    if entity_type == "prompt":
        return VALID_PROMPT_STATUSES
    raise ValueError(
        f"Invalid entity_type '{entity_type}'. Must be one of: {['plan', 'task', 'prompt']}"
    )


def _validate_status_for_entity(
    *, entity_type: str, status: str | None, field_name: str, allow_none: bool = False
) -> str | None:
    valid_statuses = _status_set_for_entity_type(entity_type)
    if status is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be a non-empty string.")
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = status.strip()
    if normalized not in valid_statuses:
        raise ValueError(
            f"Invalid {field_name} '{status}' for entity_type '{entity_type}'. "
            f"Must be one of: {sorted(valid_statuses)}"
        )
    return normalized


def _validate_status_history_scope(entity_type: str, entity_id: str) -> None:
    _status_set_for_entity_type(entity_type)
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id must be a non-empty string.")


def record_status_change(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    entity_id: str,
    new_status: str,
    old_status: str | None = None,
    actor: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Record a validated status transition for a plan or task.

    This helper intentionally does not commit. Callers can compose status updates
    and history inserts atomically inside a caller-managed transaction.
    """
    _validate_status_history_scope(entity_type, entity_id)
    normalized_new_status = _validate_status_for_entity(
        entity_type=entity_type, status=new_status, field_name="new_status"
    )
    normalized_old_status = _validate_status_for_entity(
        entity_type=entity_type, status=old_status, field_name="old_status", allow_none=True
    )
    if actor is not None and not isinstance(actor, str):
        raise ValueError("actor must be a string or None.")
    if created_at is not None and (not isinstance(created_at, str) or not created_at.strip()):
        raise ValueError("created_at must be a non-empty string when provided.")
    if normalized_old_status == normalized_new_status:
        return {}

    if created_at is None:
        cursor = conn.execute(
            "INSERT INTO status_history (entity_type, entity_id, old_status, new_status, actor) "
            "VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id.strip(), normalized_old_status, normalized_new_status, actor),
        )
    else:
        cursor = conn.execute(
            "INSERT INTO status_history "
            "(entity_type, entity_id, old_status, new_status, actor, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entity_type,
                entity_id.strip(),
                normalized_old_status,
                normalized_new_status,
                actor,
                created_at.strip(),
            ),
        )

    row = conn.execute(
        "SELECT id, entity_type, entity_id, old_status, new_status, actor, created_at "
        "FROM status_history WHERE id = ?",
        (cursor.lastrowid,),
    ).fetchone()
    return dict(row) if row else {}


def list_status_history(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    entity_id: str,
) -> list[dict]:
    """List status history oldest-first for a specific plan or task."""
    _validate_status_history_scope(entity_type, entity_id)
    rows = conn.execute(
        "SELECT id, entity_type, entity_id, old_status, new_status, actor, created_at "
        "FROM status_history "
        "WHERE entity_type = ? AND entity_id = ? "
        "ORDER BY created_at, id",
        (entity_type, entity_id.strip()),
    ).fetchall()
    return [dict(row) for row in rows]


def get_task_rejection_count(conn: sqlite3.Connection, task_id: str) -> int:
    """Count task status_history transitions from review to rejected."""
    _validate_status_history_scope("task", task_id)
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM status_history
        WHERE entity_type = 'task'
          AND entity_id = ?
          AND old_status = 'review'
          AND new_status = 'rejected'
        """,
        (task_id.strip(),),
    ).fetchone()
    return int(row["count"]) if row else 0


def list_status_history_timing_rows(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    entity_id: str,
) -> list[dict]:
    """Return oldest-first history rows with next timestamps/durations for timing math."""
    _validate_status_history_scope(entity_type, entity_id)
    rows = conn.execute(
        """
        WITH ordered AS (
            SELECT
                id,
                entity_type,
                entity_id,
                old_status,
                new_status,
                actor,
                created_at,
                LEAD(created_at) OVER (ORDER BY created_at, id) AS next_created_at
            FROM status_history
            WHERE entity_type = ? AND entity_id = ?
        )
        SELECT
            id,
            entity_type,
            entity_id,
            old_status,
            new_status,
            actor,
            created_at,
            next_created_at,
            CASE
                WHEN next_created_at IS NULL THEN NULL
                ELSE
                    CAST(strftime('%s', next_created_at) AS INTEGER)
                    - CAST(strftime('%s', created_at) AS INTEGER)
            END AS duration_seconds
        FROM ordered
        ORDER BY created_at, id
        """,
        (entity_type, entity_id.strip()),
    ).fetchall()
    return [dict(row) for row in rows]


def _build_timeline_filters(
    project_id: str | None,
    plan_id: str | None,
    since: str | None,
    until: str | None,
) -> tuple[list[str], list[object]]:
    """Build reusable SQL predicates and parameters for plan timeline queries."""
    conditions: list[str] = []
    params: list[object] = []

    if plan_id is not None:
        conditions.append("entity_id = ?")
        params.append(plan_id.strip())

    if project_id is not None:
        conditions.append("entity_id IN (SELECT id FROM plans WHERE project_id = ?)")
        params.append(project_id.strip())

    if since is not None:
        conditions.append("created_at >= ?")
        params.append(since.strip())

    if until is not None:
        conditions.append("created_at <= ?")
        params.append(until.strip())

    return conditions, params


def _collapse_timeline_timestamp_duplicates(rows: list[dict]) -> list[dict]:
    """Collapse plan/prompt transitions that share the same timestamp."""
    rows_by_timestamp: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_timestamp.setdefault(row["created_at"], []).append(row)

    collapsed: list[dict] = []
    for timestamp_rows in rows_by_timestamp.values():
        collapsed.extend(_collapse_same_timestamp_timeline_rows(timestamp_rows))

    return sorted(collapsed, key=lambda row: (row["created_at"], row["id"]))


def _collapse_same_timestamp_timeline_rows(timestamp_rows: list[dict]) -> list[dict]:
    """Collapse one timestamp bucket of plan/prompt timeline rows."""
    prompt_rows = [row for row in timestamp_rows if row["entity_type"] == "prompt"]
    has_enriching_prompt = any(row["new_status"] == "enriching" for row in prompt_rows)
    plan_rows = [
        row
        for row in timestamp_rows
        if row["entity_type"] == "plan"
        and not (
            has_enriching_prompt
            and row["old_status"] == "pending"
            and row["new_status"] == "running"
        )
    ]
    return [*prompt_rows, *plan_rows]


def _recompute_timeline_row_durations(rows: list[dict]) -> list[dict]:
    """Recompute timeline endpoints and durations from a deduplicated sequence."""
    for index, row in enumerate(rows):
        if index + 1 == len(rows):
            row["next_created_at"] = None
            row["duration_seconds"] = None
            continue

        next_row = rows[index + 1]
        row["next_created_at"] = next_row["created_at"]
        row["duration_seconds"] = int(
            (
                datetime.fromisoformat(next_row["created_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
        )

    return rows


def list_plan_timeline_rows(
    conn: sqlite3.Connection,
    plan_id: str,
    *,
    project_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Return combined plan + prompt timeline with durations.

    Merges plan-level and prompt-level status transitions into one
    chronological timeline. Prompt transitions (enriching, awaiting_input,
    finalized) show enrichment phases that are otherwise hidden within
    the plan's 'running' status.

    Redundant plan transitions that overlap with more specific prompt
    transitions at the same timestamp are filtered out.
    """
    timeline_filters, params = _build_timeline_filters(project_id, plan_id, since, until)
    where_conditions = ["entity_type IN ('plan', 'prompt')"]
    if timeline_filters:
        where_conditions = timeline_filters + where_conditions

    rows = conn.execute(
        """
        WITH combined AS (
            SELECT id, entity_type, old_status, new_status, actor, created_at
            FROM status_history
            WHERE {conditions}
        ),
        ordered AS (
            SELECT *,
                LEAD(created_at) OVER (ORDER BY created_at, id) AS next_created_at
            FROM combined
        )
        SELECT
            id,
            entity_type,
            old_status,
            new_status,
            actor,
            created_at,
            next_created_at,
            CASE
                WHEN next_created_at IS NULL THEN NULL
            ELSE
                CAST(strftime('%s', next_created_at) AS INTEGER)
                - CAST(strftime('%s', created_at) AS INTEGER)
            END AS duration_seconds
        FROM ordered
        ORDER BY created_at, id
        """.format(
            conditions=" AND ".join(where_conditions),
        ),
        params,
    ).fetchall()
    timeline_rows = [dict(row) for row in rows]
    deduped_rows = _collapse_timeline_timestamp_duplicates(timeline_rows)
    return _recompute_timeline_row_durations(deduped_rows)


def bulk_active_runtime_seconds(
    conn: sqlite3.Connection,
    entity_type: str,
    active_statuses: frozenset[str],
) -> dict[str, int]:
    """Compute active runtime for all entities of the given type in one query.

    Returns ``{entity_id: seconds}`` for entities that have any active-status
    history.  Uses the same LEAD() window function as
    ``list_status_history_timing_rows`` but aggregated per entity.  Open-ended
    rows (currently in an active status) use ``strftime('%s','now')`` as the
    end boundary.
    """
    if not active_statuses:
        return {}
    placeholders = ",".join("?" for _ in active_statuses)
    rows = conn.execute(
        f"""
        WITH ordered AS (
            SELECT
                entity_id,
                new_status,
                created_at,
                LEAD(created_at) OVER (
                    PARTITION BY entity_id ORDER BY created_at, id
                ) AS next_created_at
            FROM status_history
            WHERE entity_type = ?
        )
        SELECT
            entity_id,
            SUM(
                CAST(strftime('%s', COALESCE(
                    next_created_at,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )) AS INTEGER)
                - CAST(strftime('%s', created_at) AS INTEGER)
            ) AS active_seconds
        FROM ordered
        WHERE new_status IN ({placeholders})
        GROUP BY entity_id
        """,
        (entity_type, *active_statuses),
    ).fetchall()
    return {row["entity_id"]: max(row["active_seconds"], 0) for row in rows}


def get_project_by_dir(conn: sqlite3.Connection, directory: str) -> ProjectRow | None:
    """Look up a project by its directory path."""
    row = conn.execute("SELECT * FROM projects WHERE dir = ?", (directory,)).fetchone()
    return cast(ProjectRow, dict(row)) if row else None


def add_project(conn: sqlite3.Connection, name: str, directory: str) -> dict:
    # Check for duplicate directory before insert for a clear error message
    existing = get_project_by_dir(conn, directory)
    if existing:
        raise ValueError(
            f"Directory '{directory}' is already registered as project '{existing['name']}'"
        )
    project_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO projects (id, name, dir) VALUES (?, ?, ?)",
        (project_id, name, directory),
    )
    conn.commit()
    return {"id": project_id, "name": name, "dir": directory}


def list_projects(conn: sqlite3.Connection) -> list[ProjectRow]:
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
    return [cast(ProjectRow, dict(row)) for row in rows]


def get_project(conn: sqlite3.Connection, name_or_id: str) -> ProjectRow | None:
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ? OR name = ?",
        (name_or_id, name_or_id),
    ).fetchone()
    return cast(ProjectRow, dict(row)) if row else None


def update_project_dir(
    conn: sqlite3.Connection, name_or_id: str, new_dir: str
) -> ProjectRow | None:
    """Update a project's directory path. Returns updated project or None if not found."""
    proj = get_project(conn, name_or_id)
    if not proj:
        return None
    existing = get_project_by_dir(conn, new_dir)
    if existing and existing["id"] != proj["id"]:
        raise ValueError(
            f"Directory '{new_dir}' is already registered as project '{existing['name']}'"
        )
    conn.execute("UPDATE projects SET dir = ? WHERE id = ?", (new_dir, proj["id"]))
    conn.commit()
    proj["dir"] = new_dir
    return proj


def rename_project(conn: sqlite3.Connection, name_or_id: str, new_name: str) -> dict | None:
    """Rename a project. Returns updated project or None if not found."""
    proj = get_project(conn, name_or_id)
    if not proj:
        return None
    existing = get_project(conn, new_name)
    if existing and existing["id"] != proj["id"]:
        raise ValueError(f"Project name '{new_name}' is already taken")
    conn.execute("UPDATE projects SET name = ? WHERE id = ?", (new_name, proj["id"]))
    conn.commit()
    old_name = proj["name"]
    proj["name"] = new_name
    return {"project": proj, "old_name": old_name}


def remove_project(conn: sqlite3.Connection, name_or_id: str) -> dict | None:
    """Remove a project and cascade-delete all dependent data.

    Returns a summary dict with counts of deleted rows, or None if project not found.
    """
    row = conn.execute(
        "SELECT id FROM projects WHERE id = ? OR name = ?",
        (name_or_id, name_or_id),
    ).fetchone()
    if not row:
        return None
    project_id = row["id"]

    # Collect plan and task IDs for log cleanup (caller needs these)
    plan_rows = conn.execute("SELECT id FROM plans WHERE project_id = ?", (project_id,)).fetchall()
    plan_ids = [r["id"] for r in plan_rows]
    task_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE plan_id IN (SELECT id FROM plans WHERE project_id = ?)",
            (project_id,),
        ).fetchall()
    ]

    # Delete leaf tables first (deepest FK dependencies)
    conn.execute(
        "DELETE FROM status_history WHERE entity_id IN "
        "(SELECT id FROM tasks WHERE plan_id IN (SELECT id FROM plans WHERE project_id = ?))",
        (project_id,),
    )
    conn.execute(
        "DELETE FROM status_history WHERE entity_id IN (SELECT id FROM plans WHERE project_id = ?)",
        (project_id,),
    )
    conn.execute(
        "DELETE FROM task_logs WHERE task_id IN "
        "(SELECT id FROM tasks WHERE plan_id IN (SELECT id FROM plans WHERE project_id = ?))",
        (project_id,),
    )
    # Delete task_blocks where this project's tasks are the blocked task
    conn.execute(
        "DELETE FROM task_blocks WHERE task_id IN "
        "(SELECT id FROM tasks WHERE plan_id IN (SELECT id FROM plans WHERE project_id = ?))",
        (project_id,),
    )
    # Delete task_blocks where this project's tasks are the blocker (cross-project refs)
    conn.execute(
        "DELETE FROM task_blocks WHERE blocked_by_task_id IN "
        "(SELECT id FROM tasks WHERE plan_id IN (SELECT id FROM plans WHERE project_id = ?))",
        (project_id,),
    )
    plan_subq = "SELECT id FROM plans WHERE project_id = ?"
    conn.execute(
        f"DELETE FROM plan_logs WHERE plan_id IN ({plan_subq})",
        (project_id,),
    )
    conn.execute(
        f"DELETE FROM plan_questions WHERE plan_id IN ({plan_subq})",
        (project_id,),
    )
    conn.execute(
        f"DELETE FROM tasks WHERE plan_id IN ({plan_subq})",
        (project_id,),
    )
    conn.execute("DELETE FROM plans WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()

    return {"plan_ids": plan_ids, "task_ids": task_ids}


def set_project_quality_gate(
    conn: sqlite3.Connection, project_id: str, quality_gate: str | None
) -> None:
    """Set quality gate config (JSON string of check commands, or None to use defaults)."""
    conn.execute(
        "UPDATE projects SET quality_gate = ? WHERE id = ?",
        (quality_gate, project_id),
    )
    conn.commit()


def get_project_quality_gate(conn: sqlite3.Connection, project_id: str) -> str | None:
    """Get quality gate config JSON, or None for defaults."""
    row = conn.execute(
        "SELECT quality_gate FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    return row["quality_gate"] if row else None


def set_project_model_config(
    conn: sqlite3.Connection, project_id: str, model_config: str | None
) -> None:
    """Persist project-level model settings (JSON string), or clear with None."""
    if "model_config" not in _table_columns(conn, "projects"):
        return

    conn.execute(
        "UPDATE projects SET model_config = ? WHERE id = ?",
        (model_config, project_id),
    )
    conn.commit()


def get_project_model_config(conn: sqlite3.Connection, project_id: str) -> str | None:
    """Return project model configuration, or None for missing row/column/unset value."""
    if "model_config" not in _table_columns(conn, "projects"):
        return None

    row = conn.execute(
        "SELECT model_config FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    return row["model_config"] if row else None


def resolve_backend(
    explicit_backend: str | None,
    default: str = DEFAULT_BACKEND,
) -> str:
    """Resolve a backend using precedence: explicit > hard default.

    Only Codex is supported.
    """
    backend = explicit_backend or default

    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: {sorted(VALID_BACKENDS)}")

    if backend not in IMPLEMENTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not yet implemented. "
            f"Supported: {', '.join(sorted(IMPLEMENTED_BACKENDS))}"
        )

    return backend


VALID_PLAN_APPROVAL_MODES = {"auto", "manual"}
APP_SERVER_APPROVAL_POLICY_DEFAULTS: dict[str, str] = {
    "item/commandExecution/requestApproval": "accept",
    "item/fileChange/requestApproval": "accept",
    "skill/requestApproval": "approve",
    "execCommandApproval": "approved",
    "applyPatchApproval": "approved",
}
APP_SERVER_APPROVAL_POLICY_ALLOWED_DECISIONS: dict[str, set[str]] = {
    "item/commandExecution/requestApproval": {
        "accept",
        "acceptForSession",
        "decline",
        "cancel",
    },
    "item/fileChange/requestApproval": {
        "accept",
        "acceptForSession",
        "decline",
        "cancel",
    },
    "skill/requestApproval": {"approve", "decline"},
    "execCommandApproval": {"approved", "approved_for_session", "denied", "abort"},
    "applyPatchApproval": {"approved", "approved_for_session", "denied", "abort"},
}
APP_SERVER_ASK_FOR_APPROVAL_DEFAULT = "never"
APP_SERVER_ASK_FOR_APPROVAL_ALLOWED = {"untrusted", "on-failure", "on-request", "never"}
APP_SERVER_ASK_FOR_APPROVAL_REJECT_KEYS = (
    "mcp_elicitations",
    "rules",
    "sandbox_approval",
)


def set_project_plan_approval(conn: sqlite3.Connection, project_id: str, mode: str | None) -> None:
    """Set plan approval mode for a project, or None to reset (falls back to auto)."""
    if mode is not None and mode not in VALID_PLAN_APPROVAL_MODES:
        raise ValueError(
            f"Invalid plan_approval mode '{mode}'. "
            f"Must be one of: {sorted(VALID_PLAN_APPROVAL_MODES)}"
        )
    conn.execute(
        "UPDATE projects SET plan_approval = ? WHERE id = ?",
        (mode, project_id),
    )
    conn.commit()


def get_project_plan_approval(conn: sqlite3.Connection, project_id: str) -> str:
    """Get plan approval mode for a project. Returns 'auto' or 'manual'."""
    row = conn.execute(
        "SELECT plan_approval FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    return (row["plan_approval"] or "auto") if row else "auto"


def normalize_app_server_approval_policy(
    policy: Mapping[str, object] | None,
) -> dict[str, str]:
    """Validate and normalize app-server approval policy over default decisions."""
    normalized = dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS)
    if policy is None:
        return normalized
    if not isinstance(policy, Mapping):
        raise ValueError("Approval policy must be a JSON object.")

    unknown_methods = sorted(str(method) for method in policy if method not in normalized)
    if unknown_methods:
        raise ValueError(
            f"Unknown app-server approval methods: {unknown_methods}. "
            f"Expected subset of: {sorted(normalized)}"
        )

    for method, decision in policy.items():
        if not isinstance(decision, str):
            raise ValueError(f"Decision for '{method}' must be a string.")
        allowed = APP_SERVER_APPROVAL_POLICY_ALLOWED_DECISIONS[method]
        if decision not in allowed:
            raise ValueError(
                f"Invalid decision '{decision}' for '{method}'. Allowed: {sorted(allowed)}"
            )
        normalized[method] = decision

    return normalized


def parse_app_server_approval_policy(policy: object | None) -> dict[str, str]:
    """Parse/normalize a persisted app-server approval policy, defaulting on invalid input."""
    if not policy:
        return dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS)

    parsed: object
    if isinstance(policy, Mapping):
        parsed = policy
    elif isinstance(policy, str):
        try:
            parsed = json.loads(policy)
        except (TypeError, json.JSONDecodeError):
            return dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS)
    else:
        return dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS)

    try:
        return normalize_app_server_approval_policy(cast(Mapping[str, object], parsed))
    except ValueError:
        return dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS)


def normalize_app_server_ask_for_approval(
    policy: object | None,
) -> str | dict[str, dict[str, bool]]:
    """Validate/normalize app-server AskForApproval policy."""
    if policy is None:
        return APP_SERVER_ASK_FOR_APPROVAL_DEFAULT
    if isinstance(policy, str):
        if policy not in APP_SERVER_ASK_FOR_APPROVAL_ALLOWED:
            raise ValueError(
                f"Invalid app-server ask-for-approval policy '{policy}'. "
                f"Allowed: {sorted(APP_SERVER_ASK_FOR_APPROVAL_ALLOWED)}"
            )
        return policy
    if not isinstance(policy, Mapping):
        raise ValueError("Ask-for-approval policy must be a string or JSON object.")

    if set(policy.keys()) != {"reject"}:
        raise ValueError("Reject ask-for-approval policy must be {'reject': {...}}.")
    reject = policy.get("reject")
    if not isinstance(reject, Mapping):
        raise ValueError("Reject ask-for-approval policy requires a 'reject' object.")

    missing = [k for k in APP_SERVER_ASK_FOR_APPROVAL_REJECT_KEYS if k not in reject]
    unknown = sorted(str(k) for k in reject if k not in APP_SERVER_ASK_FOR_APPROVAL_REJECT_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unknown:
            details.append(f"unknown keys: {unknown}")
        raise ValueError(
            "Reject ask-for-approval policy must include only "
            f"{list(APP_SERVER_ASK_FOR_APPROVAL_REJECT_KEYS)} ({'; '.join(details)})."
        )

    normalized_reject: dict[str, bool] = {}
    for key in APP_SERVER_ASK_FOR_APPROVAL_REJECT_KEYS:
        value = reject.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"Reject ask-for-approval key '{key}' must be boolean.")
        normalized_reject[key] = value

    return {"reject": normalized_reject}


def parse_app_server_ask_for_approval(
    policy: object | None,
) -> str | dict[str, dict[str, bool]]:
    """Parse/normalize persisted app-server AskForApproval policy."""
    if not policy:
        return APP_SERVER_ASK_FOR_APPROVAL_DEFAULT

    parsed: object
    if isinstance(policy, Mapping):
        parsed = policy
    elif isinstance(policy, str):
        try:
            parsed = json.loads(policy)
        except (TypeError, json.JSONDecodeError):
            parsed = policy
    else:
        return APP_SERVER_ASK_FOR_APPROVAL_DEFAULT

    try:
        return normalize_app_server_ask_for_approval(parsed)
    except ValueError:
        return APP_SERVER_ASK_FOR_APPROVAL_DEFAULT


def set_project_app_server_approval_policy(
    conn: sqlite3.Connection,
    project_id: str,
    policy: Mapping[str, object] | None,
) -> None:
    """Set project app-server approval policy, or None to reset to defaults."""
    serialized: str | None = None
    if policy is not None:
        normalized = normalize_app_server_approval_policy(policy)
        serialized = json.dumps(normalized, sort_keys=True)
    conn.execute(
        "UPDATE projects SET app_server_approval_policy = ? WHERE id = ?",
        (serialized, project_id),
    )
    conn.commit()


def get_project_app_server_approval_policy(
    conn: sqlite3.Connection,
    project_id: str,
) -> dict[str, str]:
    """Return effective app-server approval policy for a project."""
    row = conn.execute(
        "SELECT app_server_approval_policy FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    raw = row["app_server_approval_policy"] if row else None
    return parse_app_server_approval_policy(raw)


def set_project_app_server_ask_for_approval(
    conn: sqlite3.Connection,
    project_id: str,
    policy: object | None,
) -> None:
    """Set project app-server AskForApproval policy, or None to reset default."""
    serialized: str | None = None
    if policy is not None:
        normalized = normalize_app_server_ask_for_approval(policy)
        serialized = json.dumps(normalized, sort_keys=True)
    conn.execute(
        "UPDATE projects SET app_server_ask_for_approval = ? WHERE id = ?",
        (serialized, project_id),
    )
    conn.commit()


def get_project_app_server_ask_for_approval(
    conn: sqlite3.Connection,
    project_id: str,
) -> str | dict[str, dict[str, bool]]:
    """Return effective app-server AskForApproval policy for a project."""
    row = conn.execute(
        "SELECT app_server_ask_for_approval FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    raw = row["app_server_ask_for_approval"] if row else None
    return parse_app_server_ask_for_approval(raw)


def set_project_post_merge_command(
    conn: sqlite3.Connection, project_id: str, command: str | None
) -> None:
    """Set post-merge command for a project, or None to clear."""
    conn.execute(
        "UPDATE projects SET post_merge_command = ? WHERE id = ?",
        (command, project_id),
    )
    conn.commit()


def get_project_post_merge_command(conn: sqlite3.Connection, project_id: str) -> str | None:
    """Get post-merge command for a project. Returns None if unset."""
    row = conn.execute(
        "SELECT post_merge_command FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    return row["post_merge_command"] if row else None


def set_project_setup_result(
    conn: sqlite3.Connection, project_id: str, setup_result_json: str | None
) -> None:
    """Store the full setup result JSON for traceability."""
    conn.execute(
        "UPDATE projects SET setup_result = ? WHERE id = ?",
        (setup_result_json, project_id),
    )
    conn.commit()


def get_project_setup_result(conn: sqlite3.Connection, project_id: str) -> str | None:
    """Get the stored setup result JSON, or None if not yet run."""
    row = conn.execute(
        "SELECT setup_result FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    return row["setup_result"] if row else None


def set_project_base_branch(
    conn: sqlite3.Connection, project_id: str, base_branch: str | None
) -> None:
    """Set base branch for a project, or None to reset to default behavior."""
    normalized_base_branch = _validate_base_branch(base_branch)
    conn.execute(
        "UPDATE projects SET base_branch = ? WHERE id = ?",
        (normalized_base_branch, project_id),
    )
    conn.commit()


def get_project_base_branch(conn: sqlite3.Connection, project_id: str) -> str:
    """Return a concrete project base branch, defaulting to 'main'."""
    if "base_branch" not in _table_columns(conn, "projects"):
        return "main"

    row = conn.execute("SELECT base_branch FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        return "main"

    base_branch = row["base_branch"]
    if base_branch is None:
        return "main"

    normalized = str(base_branch).strip()
    return normalized if normalized else "main"


# -- doctor read helpers --


def inspect_sqlite_integrity(conn: sqlite3.Connection) -> dict:
    """Run PRAGMA integrity_check and normalize output for diagnostics.

    Returns:
      {
        "ok": bool,                # True only when every non-empty row is "ok"
        "rows": list[str],         # normalized raw rows
        "failures": list[str],     # non-ok rows suitable for surfacing
      }
    """
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    normalized: list[str] = []
    for row in rows:
        text = str(row[0]).strip() if row and row[0] is not None else ""
        if text:
            normalized.append(text)

    if not normalized:
        return {
            "ok": False,
            "rows": [],
            "failures": ["integrity_check returned no rows"],
        }

    failures = [row for row in normalized if row.lower() != "ok"]
    return {"ok": not failures, "rows": normalized, "failures": failures}


def list_running_plan_workers(
    conn: sqlite3.Connection, project_id: str | None = None
) -> list[dict]:
    """List running plans that have worker pid metadata."""
    query = (
        "SELECT p.*, pr.name AS project_name, pr.dir AS project_dir "
        "FROM plans p "
        "LEFT JOIN projects pr ON p.project_id = pr.id "
        "WHERE p.status = 'running' AND p.pid IS NOT NULL"
    )
    params: list[str] = []
    if project_id:
        query += " AND p.project_id = ?"
        params.append(project_id)
    query += " ORDER BY p.updated_at, p.created_at, p.id"
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def list_running_task_workers(
    conn: sqlite3.Connection, project_id: str | None = None
) -> list[dict]:
    """List running tasks that have worker pid metadata."""
    query = (
        "SELECT t.*, p.project_id AS project_id, "
        "pr.name AS project_name, pr.dir AS project_dir "
        "FROM tasks t "
        "JOIN plans p ON t.plan_id = p.id "
        "LEFT JOIN projects pr ON p.project_id = pr.id "
        "WHERE t.status = 'running' AND t.pid IS NOT NULL"
    )
    params: list[str] = []
    if project_id:
        query += " AND p.project_id = ?"
        params.append(project_id)
    query += " ORDER BY t.updated_at, t.created_at, t.ordinal, t.id"
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def list_task_worktree_project_refs(
    conn: sqlite3.Connection, project_id: str | None = None
) -> list[dict]:
    """Correlate task git refs with project metadata for orphan analysis."""
    query = (
        "SELECT t.*, p.project_id AS project_id, p.status AS plan_status, "
        "pr.name AS project_name, pr.dir AS project_dir "
        "FROM tasks t "
        "JOIN plans p ON t.plan_id = p.id "
        "LEFT JOIN projects pr ON p.project_id = pr.id "
        "WHERE (t.branch IS NOT NULL OR t.worktree IS NOT NULL)"
    )
    params: list[str] = []
    if project_id:
        query += " AND p.project_id = ?"
        params.append(project_id)
    query += " ORDER BY t.updated_at, t.created_at, t.ordinal, t.id"
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


# -- sessions --


def create_session(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    trigger: str,
    trigger_prompt: str | None = None,
) -> SessionRow:
    """Create a new session with status 'open'."""
    session_id = uuid.uuid4().hex[:12]
    now = _utcnow()
    conn.execute(
        "INSERT INTO sessions "
        "(id, project_id, trigger, trigger_prompt, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'open', ?, ?)",
        (session_id, project_id, trigger, trigger_prompt, now, now),
    )
    conn.commit()
    return cast(
        SessionRow,
        {
            "id": session_id,
            "project_id": project_id,
            "trigger": trigger,
            "trigger_prompt": trigger_prompt,
            "status": "open",
            "started_at": None,
            "finished_at": None,
            "created_at": now,
            "updated_at": now,
        },
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> SessionRow | None:
    """Lookup a session by ID."""
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return cast(SessionRow, dict(row)) if row else None


def list_sessions(
    conn: sqlite3.Connection,
    project_id: str | None = None,
    status: str | None = None,
    statuses: Sequence[str] | None = None,
) -> list[SessionRow]:
    """List sessions with optional project/status filtering."""
    query = "SELECT * FROM sessions"
    conditions: list[str] = []
    params: list[str] = []
    if project_id:
        conditions.append("project_id = ?")
        params.append(project_id)
    if status:
        conditions.append("status = ?")
        params.append(status)
    elif statuses:
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [cast(SessionRow, dict(row)) for row in rows]


def update_session_status(conn: sqlite3.Connection, session_id: str, status: str) -> None:
    """Update session status. Validates against VALID_SESSION_STATUSES.

    When transitioning to 'active', also sets started_at (preserving any
    existing value).
    """
    if status not in VALID_SESSION_STATUSES:
        raise ValueError(
            f"Invalid session status '{status}'. Must be one of: {sorted(VALID_SESSION_STATUSES)}"
        )
    now = _utcnow()
    if status == "active":
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ?, "
            "started_at = COALESCE(started_at, ?) WHERE id = ?",
            (status, now, now, session_id),
        )
    else:
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, session_id),
        )
    conn.commit()


def finish_session(conn: sqlite3.Connection, session_id: str, status: str) -> None:
    """Finish a session by setting a terminal status and finished_at."""
    if status not in SESSION_TERMINAL_STATUSES:
        raise ValueError(
            f"Invalid terminal status '{status}'. "
            f"Must be one of: {sorted(SESSION_TERMINAL_STATUSES)}"
        )
    now = _utcnow()
    conn.execute(
        "UPDATE sessions SET status = ?, finished_at = ?, updated_at = ? WHERE id = ?",
        (status, now, now, session_id),
    )
    conn.commit()


def set_plan_session_id(conn: sqlite3.Connection, plan_id: str, session_id: str) -> None:
    """Link a plan to a session."""
    conn.execute(
        "UPDATE plans SET session_id = ? WHERE id = ?",
        (session_id, plan_id),
    )
    conn.commit()


# -- session reconciliation --


def _is_effectively_terminal_task_row(task: TaskRow) -> bool:
    """Return whether a task should count as terminal for session completion."""
    status = task.get("status")
    if status in TASK_TERMINAL_STATUSES:
        return True
    # do --no-merge intentionally stops at approved.
    return status == "approved" and bool(task.get("skip_merge"))


def infer_terminal_session_status(conn: sqlite3.Connection, session_id: str) -> str | None:
    """Infer terminal session status when all linked plans/tasks are effectively terminal."""
    plans = list_plan_requests(conn, session_id=session_id)
    if not plans:
        return None
    if not all(p["status"] in PLAN_TERMINAL_STATUSES for p in plans):
        return None

    any_failed = any(p["status"] == "failed" for p in plans)
    for plan in plans:
        task_creation_status = str(plan.get("task_creation_status") or "")
        if task_creation_status in {"pending", "running", "awaiting_approval"}:
            return None
        if task_creation_status == "failed":
            any_failed = True
        tasks = list_tasks(conn, plan_id=plan["id"])
        if tasks and not all(_is_effectively_terminal_task_row(task) for task in tasks):
            return None
        if not any_failed and any(task["status"] == "failed" for task in tasks):
            any_failed = True
    return "failed" if any_failed else "completed"


def reconcile_session_statuses(
    conn: sqlite3.Connection, session_id: str | None = None
) -> list[dict[str, str]]:
    """Backfill session terminal statuses for already-finished pipelines.

    Returns list of transitions: ``{"id": ..., "old_status": ..., "new_status": ...}``.
    """
    if session_id:
        sessions = [get_session(conn, session_id)]
    else:
        sessions = list_sessions(conn, statuses=["open", "active"])

    updates: list[dict[str, str]] = []
    for session in sessions:
        if not session:
            continue
        target_status = infer_terminal_session_status(conn, session["id"])
        if not target_status or session["status"] == target_status:
            continue
        old_status = session["status"]
        finish_session(conn, session["id"], target_status)
        updates.append(
            {
                "id": session["id"],
                "old_status": old_status,
                "new_status": target_status,
            }
        )
    return updates


# -- channel messages --


def add_channel_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    kind: str,
    sender: str,
    content: str,
    recipient: str | None = None,
    metadata: str | None = None,
) -> ChannelMessageRow:
    """Add a message to a session's communication channel."""
    if kind not in VALID_MESSAGE_KINDS:
        raise ValueError(
            f"Invalid message kind '{kind}'. Must be one of: {sorted(VALID_MESSAGE_KINDS)}"
        )
    message_id = uuid.uuid4().hex[:12]
    now = _utcnow()
    conn.execute(
        "INSERT INTO channel_messages "
        "(id, session_id, kind, sender, recipient, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (message_id, session_id, kind, sender, recipient, content, metadata, now),
    )
    conn.commit()
    return cast(
        ChannelMessageRow,
        {
            "id": message_id,
            "session_id": session_id,
            "kind": kind,
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "metadata": metadata,
            "created_at": now,
        },
    )


def list_channel_messages(
    conn: sqlite3.Connection,
    session_id: str,
    kind: str | None = None,
    sender: str | None = None,
    recipient: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ChannelMessageRow]:
    """List messages for a session in chronological order."""
    query = "SELECT * FROM channel_messages WHERE session_id = ?"
    params: list[str | int] = [session_id]
    if kind:
        query += " AND kind = ?"
        params.append(kind)
    if sender:
        if ":" in sender:
            query += " AND sender = ?"
            params.append(sender)
        else:
            query += " AND (sender = ? OR sender LIKE ?)"
            params.extend([sender, f"{sender}:%"])
    if recipient:
        if ":" in recipient:
            query += " AND recipient = ?"
            params.append(recipient)
        else:
            query += " AND (recipient = ? OR recipient LIKE ?)"
            params.extend([recipient, f"{recipient}:%"])
    query += " ORDER BY created_at ASC, rowid ASC LIMIT ? OFFSET ?"
    params.append(limit)
    params.append(offset)
    rows = conn.execute(query, params).fetchall()
    return [cast(ChannelMessageRow, dict(row)) for row in rows]


def get_channel_message(conn: sqlite3.Connection, message_id: str) -> ChannelMessageRow | None:
    """Lookup a channel message by ID."""
    row = conn.execute("SELECT * FROM channel_messages WHERE id = ?", (message_id,)).fetchone()
    return cast(ChannelMessageRow, dict(row)) if row else None


def add_task_steer(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    session_id: str,
    sender: str,
    content: str,
    message_id: str | None = None,
    recipient: str | None = None,
    reason: str | None = None,
    metadata: str | None = None,
    live_requested: bool = False,
    live_applied: bool = False,
    live_error: str | None = None,
    thread_id: str | None = None,
    expected_turn_id: str | None = None,
    applied_turn_id: str | None = None,
) -> TaskSteerRow:
    """Persist a steer action (audit + replay surface)."""
    steer_id = uuid.uuid4().hex[:12]
    now = _utcnow()
    conn.execute(
        "INSERT INTO task_steers "
        "(id, task_id, session_id, message_id, sender, recipient, content, reason, metadata, "
        "live_requested, live_applied, live_error, thread_id, expected_turn_id, applied_turn_id, "
        "created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            steer_id,
            task_id,
            session_id,
            message_id,
            sender,
            recipient,
            content,
            reason,
            metadata,
            1 if live_requested else 0,
            1 if live_applied else 0,
            live_error,
            thread_id,
            expected_turn_id,
            applied_turn_id,
            now,
        ),
    )
    conn.commit()
    return cast(
        TaskSteerRow,
        {
            "id": steer_id,
            "task_id": task_id,
            "session_id": session_id,
            "message_id": message_id,
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "reason": reason,
            "metadata": metadata,
            "live_requested": 1 if live_requested else 0,
            "live_applied": 1 if live_applied else 0,
            "live_error": live_error,
            "thread_id": thread_id,
            "expected_turn_id": expected_turn_id,
            "applied_turn_id": applied_turn_id,
            "created_at": now,
        },
    )


def list_task_steers(
    conn: sqlite3.Connection,
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[TaskSteerRow]:
    """List persisted steer entries, newest first."""
    query = "SELECT * FROM task_steers"
    clauses: list[str] = []
    params: list[str | int] = []
    if task_id:
        clauses.append("task_id = ?")
        params.append(task_id)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    return [cast(TaskSteerRow, dict(row)) for row in rows]


def has_recent_task_steer(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    reason: str,
    content: str,
    lookback_seconds: int,
    require_live_attempt: bool = False,
) -> bool:
    """Return True when a matching steer was recorded in the recent lookback window."""
    live_clause = " AND live_requested = 1" if require_live_attempt else ""
    row = conn.execute(
        "SELECT 1 FROM task_steers "
        "WHERE task_id = ? AND reason = ? AND content = ? "
        f"{live_clause} "
        "AND created_at >= strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?) "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (task_id, reason, content, f"-{max(0, lookback_seconds)} seconds"),
    ).fetchone()
    return row is not None


# -- trace events --


VALID_TRACE_STAGES = {
    "enrichment",
    "exploration",
    "planning",
    "task_creation",
    "execution",
    "review",
}


def add_trace_event(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    entity_id: str,
    stage: str,
    turn_index: int,
    ordinal: int,
    event_type: str,
    status: str | None = None,
    data: dict | None = None,
) -> str:
    """Insert a trace event. Returns the event id."""
    event_id = str(uuid.uuid4())
    data_json = json.dumps(data or {})
    conn.execute(
        "INSERT INTO trace_events "
        "(id, entity_type, entity_id, stage, turn_index, ordinal, "
        "event_type, status, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            entity_type,
            entity_id,
            stage,
            turn_index,
            ordinal,
            event_type,
            status,
            data_json,
        ),
    )
    conn.commit()
    return event_id


def list_trace_events(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    *,
    event_type: str | None = None,
    stage: str | None = None,
    turn_index: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return trace events ordered by ordinal."""
    query = "SELECT * FROM trace_events WHERE entity_type = ? AND entity_id = ?"
    params: list[object] = [entity_type, entity_id]
    if event_type is not None:
        query += " AND event_type = ?"
        params.append(event_type)
    if stage is not None:
        query += " AND stage = ?"
        params.append(stage)
    if turn_index is not None:
        query += " AND turn_index = ?"
        params.append(turn_index)
    query += " ORDER BY ordinal ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        try:
            d["data"] = json.loads(d["data"]) if d.get("data") else {}
        except (json.JSONDecodeError, TypeError):
            d["data"] = {}
        result.append(d)
    return result


def get_latest_trace_events_by_entity_ids(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    entity_ids: Sequence[str],
    event_type: str,
) -> dict[str, dict]:
    """Return latest trace event per entity_id for the given event_type."""
    ids = [str(entity_id).strip() for entity_id in entity_ids if str(entity_id).strip()]
    if not ids:
        return {}

    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        (
            "SELECT te.entity_id, te.status, te.data, te.created_at "
            "FROM trace_events te "
            "JOIN ("
            "  SELECT entity_id, MAX(ordinal) AS max_ordinal "
            "  FROM trace_events "
            f"  WHERE entity_type = ? AND event_type = ? AND entity_id IN ({placeholders}) "
            "  GROUP BY entity_id"
            ") latest "
            "ON te.entity_id = latest.entity_id AND te.ordinal = latest.max_ordinal "
            "WHERE te.entity_type = ? AND te.event_type = ?"
        ),
        (entity_type, event_type, *ids, entity_type, event_type),
    ).fetchall()

    latest: dict[str, dict] = {}
    for row in rows:
        data: dict
        try:
            data = json.loads(row["data"]) if row["data"] else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
        latest[str(row["entity_id"])] = {
            "status": row["status"],
            "data": data,
            "created_at": row["created_at"],
        }
    return latest


def count_trace_events(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
) -> dict[str, int]:
    """Return event counts grouped by event_type."""
    rows = conn.execute(
        "SELECT event_type, COUNT(*) as cnt FROM trace_events "
        "WHERE entity_type = ? AND entity_id = ? GROUP BY event_type",
        (entity_type, entity_id),
    ).fetchall()
    return {row["event_type"]: row["cnt"] for row in rows}


def _accumulate_trace_summary(
    ev: dict,
    data: dict,
    summary: dict,
) -> None:
    """Accumulate one completed trace event into *summary* accumulators."""
    et = ev["event_type"]
    if et == "fileReadTool":
        path = data.get("path")
        if path and path not in summary["files_read"]:
            summary["files_read"].append(path)
    elif et in ("fileEditTool", "fileWriteTool"):
        path = data.get("path")
        if path and path not in summary["files_written"]:
            summary["files_written"].append(path)
    elif et == "fileChange":
        for f in data.get("files", []):
            path = f.get("path") if isinstance(f, dict) else None
            if path and path not in summary["files_written"]:
                summary["files_written"].append(path)
    elif et == "commandExecution":
        summary["commands_run"].append(
            {"command": data.get("command", ""), "exit_code": data.get("exit_code")}
        )
    elif et == "mcpToolCall":
        summary["tools_called"].append(
            {"server": data.get("server", ""), "tool": data.get("tool", "")}
        )


def get_trace_summary(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
) -> dict:
    """Return aggregated trace summary for an entity."""
    events = list_trace_events(conn, entity_type, entity_id)
    summary: dict = {
        "files_read": [],
        "files_written": [],
        "commands_run": [],
        "tools_called": [],
        "event_counts": {},
        "total_events": len(events),
    }

    for ev in events:
        summary["event_counts"][ev["event_type"]] = (
            summary["event_counts"].get(ev["event_type"], 0) + 1
        )
        data = ev.get("data", {})
        if isinstance(data, dict) and ev.get("status") == "completed":
            _accumulate_trace_summary(ev, data, summary)

    return summary


# -- plans --


def create_plan_request(
    conn: sqlite3.Connection,
    *,
    project_id: str | None,
    prompt: str,
    caller: str,
    backend: str,
    actor: str | None = None,
    parent_id: str | None = None,
) -> dict:
    all_callers = get_all_callers()
    if caller not in all_callers:
        raise ValueError(f"Invalid caller '{caller}'. Must be one of: {sorted(all_callers)}")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: {sorted(VALID_BACKENDS)}")
    if backend not in IMPLEMENTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not yet implemented. "
            f"Supported: {', '.join(sorted(IMPLEMENTED_BACKENDS))}"
        )
    if actor is None:
        actor = os.environ.get("USER", "unknown")

    plan_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO plans "
        "(id, project_id, parent_id, prompt, status, "
        "actor, caller, backend, input_tokens, output_tokens) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, 0, 0)",
        (plan_id, project_id, parent_id, prompt, actor, caller, backend),
    )
    conn.commit()
    return {
        "id": plan_id,
        "project_id": project_id,
        "parent_id": parent_id,
        "prompt": prompt,
        "status": "pending",
        "actor": actor,
        "caller": caller,
        "backend": backend,
        "input_tokens": 0,
        "output_tokens": 0,
        "pid": None,
        "thread_id": None,
    }


def get_plan_request(conn: sqlite3.Connection, plan_id: str) -> PlanRow | None:
    row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    return cast(PlanRow, dict(row)) if row else None


def set_plan_model(conn: sqlite3.Connection, plan_id: str, model: str | None) -> bool:
    """Store runtime model selection for a plan."""
    plan_cols = {row[1] for row in conn.execute("PRAGMA table_info(plans)").fetchall()}
    if "model" not in plan_cols:
        return False

    cursor = conn.execute(
        "UPDATE plans SET model = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ?",
        (model, plan_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def list_plan_requests(
    conn: sqlite3.Connection,
    project_id: str | None = None,
    status: str | None = None,
    statuses: Sequence[str] | None = None,
    project_id_is_null: bool = False,
    session_id: str | None = None,
) -> list[PlanRow]:
    """List plans for a project, with optional status/session filtering.

    When ``status`` is provided, it takes precedence over ``statuses``.
    """
    query = "SELECT * FROM plans"
    conditions = []
    params: list[str] = []
    if project_id:
        conditions.append("project_id = ?")
        params.append(project_id)
    if project_id_is_null:
        conditions.append("project_id IS NULL")
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if status:
        conditions.append("status = ?")
        params.append(status)
    elif statuses:
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at, rowid"
    rows = conn.execute(query, params).fetchall()
    return [cast(PlanRow, dict(row)) for row in rows]


def list_recent_failed_plans(
    conn: sqlite3.Connection, project_id: str | None = None, limit: int | None = None
) -> list[dict]:
    """List failed plans with project metadata and latest ERROR message."""
    if limit is not None and limit <= 0:
        return []

    query = """
        WITH latest_error_logs AS (
            SELECT
                plan_id,
                message,
                ROW_NUMBER() OVER (
                    PARTITION BY plan_id
                    ORDER BY created_at DESC, rowid DESC
                ) AS rn
            FROM plan_logs
            WHERE level = 'ERROR'
        )
        SELECT
            p.*,
            pr.name AS project_name,
            pr.dir AS project_dir,
            rel.message AS error
        FROM plans p
        LEFT JOIN projects pr ON p.project_id = pr.id
        LEFT JOIN latest_error_logs rel ON rel.plan_id = p.id AND rel.rn = 1
        WHERE p.status = 'failed'
    """
    params: list[object] = []
    if project_id:
        query += " AND p.project_id = ?"
        params.append(project_id)

    query += " ORDER BY p.updated_at DESC, p.rowid DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def finalize_plan_request(
    conn: sqlite3.Connection,
    plan_id: str,
    plan_text: str,
    *,
    record_history: bool = False,
) -> bool:
    current = conn.execute("SELECT status FROM plans WHERE id = ?", (plan_id,)).fetchone()
    if not current:
        return False
    old_status = current["status"]

    cursor = conn.execute(
        "UPDATE plans SET plan = ?, status = 'finalized', "
        "finished_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = ?",
        (plan_text, plan_id, old_status),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False

    if record_history and old_status != "finalized":
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=plan_id,
            old_status=old_status,
            new_status="finalized",
        )
    conn.commit()
    return True


def update_plan_request_status(
    conn: sqlite3.Connection,
    plan_id: str,
    status: str,
    *,
    record_history: bool = False,
) -> bool:
    if status not in VALID_PLAN_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {VALID_PLAN_STATUSES}")
    current = conn.execute("SELECT status FROM plans WHERE id = ?", (plan_id,)).fetchone()
    if not current:
        return False
    old_status = current["status"]

    # Prevent transitions out of terminal statuses (worker race protection).
    # plan retry uses reset_plan_for_retry (raw SQL), not this function.
    if old_status in PLAN_TERMINAL_STATUSES:
        return False

    # Set started_at on first transition to running
    extra_clauses = ""
    if status == "running":
        extra_clauses += (
            ", started_at = COALESCE(started_at, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"
        )
    # Set finished_at on terminal statuses
    if status in ("finalized", "failed", "cancelled"):
        extra_clauses += ", finished_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"

    cursor = conn.execute(
        "UPDATE plans SET status = ?,"
        f" updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'){extra_clauses}"
        " WHERE id = ? AND status = ?",
        (status, plan_id, old_status),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False

    if record_history and old_status != status:
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=plan_id,
            old_status=old_status,
            new_status=status,
        )
    conn.commit()
    return True


def count_plan_requests_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) as cnt FROM plans GROUP BY status").fetchall()
    return {row["status"]: row["cnt"] for row in rows}


def count_tasks_by_plan(conn: sqlite3.Connection, plan_id: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM tasks WHERE plan_id = ? GROUP BY status",
        (plan_id,),
    ).fetchall()
    counts = {status: 0 for status in VALID_TASK_STATUSES}
    total = 0
    for row in rows:
        counts[row["status"]] = row["cnt"]
        total += row["cnt"]
    counts["total"] = total
    return counts


def reset_plan_for_retry(conn: sqlite3.Connection, plan_id: str) -> bool:
    """Reset a failed plan to pending, clearing worker state."""
    cursor = conn.execute(
        "UPDATE plans SET status = 'pending', pid = NULL, thread_id = NULL, "
        "started_at = NULL, finished_at = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = 'failed'",
        (plan_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def force_cancel_plan(conn: sqlite3.Connection, plan_id: str) -> str | None:
    """Cancel a plan regardless of current status, bypassing the terminal guard.

    Returns the old status on success, or None if plan not found or already cancelled.
    Uses raw SQL like reset_plan_for_retry to bypass update_plan_request_status guard.
    """
    current = conn.execute("SELECT status FROM plans WHERE id = ?", (plan_id,)).fetchone()
    if not current or current["status"] == "cancelled":
        return None
    old_status = current["status"]
    cursor = conn.execute(
        "UPDATE plans SET status = 'cancelled', "
        "finished_at = COALESCE(finished_at, strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = ?",
        (plan_id, old_status),
    )
    if cursor.rowcount > 0:
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=plan_id,
            old_status=old_status,
            new_status="cancelled",
        )
    conn.commit()
    return old_status if cursor.rowcount > 0 else None


def _doctor_reason(reason: str | None) -> str:
    """Normalize doctor remediation reason text for audit messages."""
    if reason and reason.strip():
        return reason.strip()
    return "unspecified reason"


def fail_stale_running_plan_for_doctor(
    conn: sqlite3.Connection, plan_id: str, *, old_status: str = "running", reason: str
) -> bool:
    """Mark a stale active plan as failed, clearing worker fields and logging audit trail.

    Handles running and awaiting_input statuses via old_status parameter.
    Records status_history for audit trail continuity.
    """
    cursor = conn.execute(
        "UPDATE plans SET status = 'failed', pid = NULL, thread_id = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = ?",
        (plan_id, old_status),
    )
    if cursor.rowcount > 0:
        log_id = uuid.uuid4().hex[:12]
        conn.execute(
            "INSERT INTO plan_logs (id, plan_id, level, message) VALUES (?, ?, ?, ?)",
            (
                log_id,
                plan_id,
                "ERROR",
                (
                    f"Doctor --fix remediation: marked stale {old_status} plan as failed; "
                    f"reason: {_doctor_reason(reason)}"
                ),
            ),
        )
        record_status_change(
            conn,
            entity_type="plan",
            entity_id=plan_id,
            old_status=old_status,
            new_status="failed",
            actor="doctor",
        )
    conn.commit()
    return cursor.rowcount > 0


def update_plan_tokens(
    conn: sqlite3.Connection,
    plan_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> bool:
    """Add token counts to a plan (additive, COALESCE-safe)."""
    cursor = conn.execute(
        "UPDATE plans SET "
        "input_tokens = COALESCE(input_tokens, 0) + ?, "
        "output_tokens = COALESCE(output_tokens, 0) + ?, "
        "cached_input_tokens = COALESCE(cached_input_tokens, 0) + ?, "
        "reasoning_tokens = COALESCE(reasoning_tokens, 0) + ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, plan_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_task_tokens(
    conn: sqlite3.Connection,
    task_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> bool:
    """Add token counts to a task (additive, COALESCE-safe)."""
    cursor = conn.execute(
        "UPDATE tasks SET "
        "input_tokens = COALESCE(input_tokens, 0) + ?, "
        "output_tokens = COALESCE(output_tokens, 0) + ?, "
        "cached_input_tokens = COALESCE(cached_input_tokens, 0) + ?, "
        "reasoning_tokens = COALESCE(reasoning_tokens, 0) + ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def set_task_failure_reason(
    conn: sqlite3.Connection, task_id: str, failure_reason: str | None
) -> bool:
    """Persist a task failure reason without changing status."""
    cursor = conn.execute(
        "UPDATE tasks SET failure_reason = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (failure_reason, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def set_plan_request_worker(
    conn: sqlite3.Connection, plan_id: str, pid: int, thread_id: str | None = None
) -> bool:
    """Record worker pid and optional thread_id on a plan. Does not change status."""
    cursor = conn.execute(
        "UPDATE plans SET pid = ?, thread_id = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (pid, thread_id, plan_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_plan_enrichment(
    conn: sqlite3.Connection,
    plan_id: str,
    *,
    enriched_prompt: str | None = None,
    enrichment_thread_id: str | None = None,
) -> bool:
    """Set enrichment fields on a plan. Only updates non-None arguments."""
    sets: list[str] = []
    params: list[object] = []
    if enriched_prompt is not None:
        sets.append("enriched_prompt = ?")
        params.append(enriched_prompt)
    if enrichment_thread_id is not None:
        sets.append("enrichment_thread_id = ?")
        params.append(enrichment_thread_id)
    if not sets:
        return False
    sets.append("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')")
    params.append(plan_id)
    cursor = conn.execute(
        f"UPDATE plans SET {', '.join(sets)} WHERE id = ?",
        params,
    )
    conn.commit()
    return cursor.rowcount > 0


def update_plan_exploration(
    conn: sqlite3.Connection,
    plan_id: str,
    *,
    exploration_context: str | None = None,
    exploration_thread_id: str | None = None,
) -> bool:
    """Set exploration fields on a plan. Only updates non-None arguments."""
    sets: list[str] = []
    params: list[object] = []
    if exploration_context is not None:
        sets.append("exploration_context = ?")
        params.append(exploration_context)
    if exploration_thread_id is not None:
        sets.append("exploration_thread_id = ?")
        params.append(exploration_thread_id)
    if not sets:
        return False
    sets.append("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')")
    params.append(plan_id)
    cursor = conn.execute(
        f"UPDATE plans SET {', '.join(sets)} WHERE id = ?",
        params,
    )
    conn.commit()
    return cursor.rowcount > 0


def update_prompt_status(
    conn: sqlite3.Connection,
    plan_id: str,
    status: str,
) -> bool:
    """Update the prompt_status column on a plan.

    Validates against VALID_PROMPT_STATUSES and records transition in
    status_history (entity_type='prompt') for timeline visibility.
    """
    if status not in VALID_PROMPT_STATUSES:
        raise ValueError(
            f"Invalid prompt_status '{status}'. Must be one of: {sorted(VALID_PROMPT_STATUSES)}"
        )
    old_row = conn.execute("SELECT prompt_status FROM plans WHERE id = ?", (plan_id,)).fetchone()
    old_status = old_row["prompt_status"] if old_row else None
    cursor = conn.execute(
        "UPDATE plans SET prompt_status = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (status, plan_id),
    )
    if cursor.rowcount > 0:
        record_status_change(
            conn,
            entity_type="prompt",
            entity_id=plan_id,
            old_status=old_status,
            new_status=status,
        )
    conn.commit()
    return cursor.rowcount > 0


def get_unanswered_question_count(conn: sqlite3.Connection, plan_id: str) -> int:
    """Count unanswered questions for a plan."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM plan_questions WHERE plan_id = ? AND answer IS NULL",
        (plan_id,),
    ).fetchone()
    return row["cnt"]


def set_plan_request_thread_id(conn: sqlite3.Connection, plan_id: str, thread_id: str) -> bool:
    cursor = conn.execute(
        "UPDATE plans SET thread_id = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (thread_id, plan_id),
    )
    conn.commit()
    return cursor.rowcount > 0


# -- plan questions --


def add_plan_question(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    question: str,
    options: str | None = None,
    header: str | None = None,
    multi_select: bool = False,
) -> dict:
    """Insert a question for a plan. Does not change plan status."""
    q_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO plan_questions (id, plan_id, question, options, header, multi_select)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (q_id, plan_id, question, options, header, int(multi_select)),
    )
    conn.commit()
    return {
        "id": q_id,
        "plan_id": plan_id,
        "question": question,
        "options": options,
        "header": header,
        "multi_select": multi_select,
    }


def answer_plan_question(
    conn: sqlite3.Connection,
    question_id: str,
    answer: str,
    answered_by: str | None = None,
) -> bool:
    """Record an answer to a plan question. Does not change plan status."""
    if answered_by is None:
        answered_by = os.environ.get("USER", "unknown")
    cursor = conn.execute(
        "UPDATE plan_questions SET answer = ?, answered_by = ?, "
        "answered_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ? AND answer IS NULL",
        (answer, answered_by, question_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def list_plan_questions(
    conn: sqlite3.Connection, plan_id: str, unanswered_only: bool = False
) -> list[PlanQuestionRow]:
    if unanswered_only:
        rows = conn.execute(
            "SELECT * FROM plan_questions WHERE plan_id = ? AND answer IS NULL ORDER BY created_at",
            (plan_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM plan_questions WHERE plan_id = ? ORDER BY created_at",
            (plan_id,),
        ).fetchall()
    return [cast(PlanQuestionRow, dict(row)) for row in rows]


def get_plan_question(conn: sqlite3.Connection, question_id: str) -> PlanQuestionRow | None:
    row = conn.execute("SELECT * FROM plan_questions WHERE id = ?", (question_id,)).fetchone()
    return cast(PlanQuestionRow, dict(row)) if row else None


# -- plan logs --


def add_plan_log(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    level: str,
    message: str,
    source: str = "plan",
) -> dict:
    """Insert a log entry for a plan."""
    log_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, source) VALUES (?, ?, ?, ?, ?)",
        (log_id, plan_id, level, message, source),
    )
    conn.commit()
    return {"id": log_id, "plan_id": plan_id, "level": level, "message": message}


def list_plan_logs(conn: sqlite3.Connection, plan_id: str, level: str | None = None) -> list[dict]:
    if level:
        rows = conn.execute(
            "SELECT * FROM plan_logs WHERE plan_id = ? AND level = ? ORDER BY created_at",
            (plan_id, level),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM plan_logs WHERE plan_id = ? ORDER BY created_at",
            (plan_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_plan_watch_events(
    conn: sqlite3.Connection,
    plan_id: str,
    limit: int = 100,
    *,
    since_plan_rowid: int = 0,
    since_task_rowid: int = 0,
    since_status_rowid: int = 0,
) -> list[dict]:
    """Return a normalized newest-first event stream for a plan.

    Combines plan_logs, task_logs (for tasks in this plan only), and task/plan
    status_history transitions, with explicit per-source descending LIMIT
    queries and deterministic merge ordering.

    When since_plan_rowid/since_task_rowid are provided, only events with
    rowid > the given value are returned (for incremental streaming).
    """
    if limit <= 0:
        return []

    plan_rows = conn.execute(
        "SELECT rowid AS order_rowid, created_at, level, message, source "
        "FROM plan_logs "
        "WHERE plan_id = ? AND rowid > ? "
        "ORDER BY rowid DESC "
        "LIMIT ?",
        (plan_id, since_plan_rowid, limit),
    ).fetchall()

    task_rows = conn.execute(
        "SELECT tl.rowid AS order_rowid, tl.task_id, tl.created_at, "
        "tl.level, tl.message, tl.source "
        "FROM task_logs tl "
        "JOIN tasks t ON t.id = tl.task_id "
        "WHERE t.plan_id = ? AND tl.rowid > ? "
        "ORDER BY tl.rowid DESC "
        "LIMIT ?",
        (plan_id, since_task_rowid, limit),
    ).fetchall()

    plan_status_rows = conn.execute(
        "SELECT sh.id AS order_rowid, sh.created_at, sh.old_status, sh.new_status "
        "FROM status_history sh "
        "WHERE sh.entity_type = 'plan' AND sh.entity_id = ? AND sh.id > ? "
        "ORDER BY sh.id DESC "
        "LIMIT ?",
        (plan_id, since_status_rowid, limit),
    ).fetchall()

    task_status_rows = conn.execute(
        "SELECT sh.id AS order_rowid, sh.entity_id AS task_id, sh.created_at, "
        "sh.old_status, sh.new_status "
        "FROM status_history sh "
        "JOIN tasks t ON t.id = sh.entity_id "
        "WHERE sh.entity_type = 'task' AND t.plan_id = ? AND sh.id > ? "
        "ORDER BY sh.id DESC "
        "LIMIT ?",
        (plan_id, since_status_rowid, limit),
    ).fetchall()

    events: list[dict] = []
    for row in plan_rows:
        events.append(
            {
                "source": row["source"],
                "task_id": None,
                "timestamp": row["created_at"],
                "level": row["level"],
                "message": row["message"],
                "order_rowid": row["order_rowid"],
                "order_source_rank": 1,
            }
        )

    for row in task_rows:
        events.append(
            {
                "source": row["source"],
                "task_id": row["task_id"],
                "timestamp": row["created_at"],
                "level": row["level"],
                "message": row["message"],
                "order_rowid": row["order_rowid"],
                "order_source_rank": 0,
            }
        )

    for row in plan_status_rows:
        events.append(
            {
                "source": "status",
                "task_id": None,
                "timestamp": row["created_at"],
                "level": "STATUS",
                "message": f"Status changed: {row['old_status']} -> {row['new_status']}",
                "order_rowid": row["order_rowid"],
                "order_source_rank": 2,
            }
        )

    for row in task_status_rows:
        events.append(
            {
                "source": "status",
                "task_id": row["task_id"],
                "timestamp": row["created_at"],
                "level": "STATUS",
                "message": f"Status changed: {row['old_status']} -> {row['new_status']}",
                "order_rowid": row["order_rowid"],
                "order_source_rank": 2,
            }
        )

    events.sort(
        key=lambda event: (
            event["timestamp"],
            event["order_rowid"],
            event["order_source_rank"],
        ),
        reverse=True,
    )
    return events[:limit]


def get_plan_event_watermarks(conn: sqlite3.Connection, plan_id: str) -> tuple[int, int]:
    """Return (max_plan_log_rowid, max_task_log_rowid) for a plan.

    Used to initialize streaming watch watermarks without relying on
    a limited event fetch that might miss one source entirely.
    """
    plan_row = conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) AS m FROM plan_logs WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()
    task_row = conn.execute(
        "SELECT COALESCE(MAX(tl.rowid), 0) AS m "
        "FROM task_logs tl JOIN tasks t ON t.id = tl.task_id "
        "WHERE t.plan_id = ?",
        (plan_id,),
    ).fetchone()
    return (plan_row["m"], task_row["m"])


# -- plan history --


def get_plan_chain(conn: sqlite3.Connection, plan_id: str) -> list[PlanRow]:
    """Return the continuation chain containing a plan.

    Walks up to the root, then down through the latest child at each level.
    If a plan has multiple continuations (branches), only the most recent
    branch is followed. Returns plans in chronological order.

    Uses two recursive CTEs to fetch the entire chain in a single query.
    """
    rows = conn.execute(
        """
        WITH RECURSIVE
        ancestors(id, depth) AS (
            SELECT id, 0 FROM plans WHERE id = ?
            UNION ALL
            SELECT p.id, a.depth + 1
            FROM plans p
            JOIN ancestors a ON p.id = (SELECT parent_id FROM plans WHERE id = a.id)
            WHERE p.id IS NOT NULL
        ),
        root AS (
            SELECT id FROM ancestors ORDER BY depth DESC LIMIT 1
        ),
        descendants(id) AS (
            SELECT id FROM root
            UNION ALL
            SELECT child.id
            FROM descendants d
            JOIN (
                SELECT id, parent_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY parent_id
                           ORDER BY created_at DESC, rowid DESC
                       ) AS rn
                FROM plans
                WHERE parent_id IS NOT NULL
            ) child ON child.parent_id = d.id AND child.rn = 1
        )
        SELECT p.* FROM descendants d JOIN plans p ON p.id = d.id
        """,
        (plan_id,),
    ).fetchall()
    return [cast(PlanRow, dict(r)) for r in rows]


# -- plan task creation status --


def update_plan_task_creation_status(conn: sqlite3.Connection, plan_id: str, status: str) -> bool:
    if status not in VALID_TASK_CREATION_STATUSES:
        raise ValueError(
            f"Invalid task_creation_status '{status}'. "
            f"Must be one of: {VALID_TASK_CREATION_STATUSES}"
        )
    cursor = conn.execute(
        "UPDATE plans SET task_creation_status = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (status, plan_id),
    )
    conn.commit()
    return cursor.rowcount > 0


# -- tasks --


def _normalize_task_priority(priority: str | None) -> str | None:
    if priority is None:
        return None
    if not isinstance(priority, str) or not priority.strip():
        raise ValueError(
            f"Invalid task priority '{priority}'. Must be one of: {VALID_TASK_PRIORITIES}"
        )
    normalized = priority.strip().lower()
    if normalized not in VALID_TASK_PRIORITIES:
        raise ValueError(
            f"Invalid task priority '{priority}'. Must be one of: {VALID_TASK_PRIORITIES}"
        )
    if normalized == "medium":
        return None
    return normalized


def create_task(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    ordinal: int,
    title: str,
    description: str,
    files: str | None = None,
    priority: str | None = None,
) -> dict:
    normalized_priority = _normalize_task_priority(priority)
    task_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO tasks "
        "(id, plan_id, ordinal, title, description, files, priority, input_tokens, output_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
        (task_id, plan_id, ordinal, title, description, files, normalized_priority),
    )
    conn.commit()
    return {
        "id": task_id,
        "plan_id": plan_id,
        "ordinal": ordinal,
        "title": title,
        "description": description,
        "files": files,
        "status": "blocked",
        "priority": normalized_priority,
        "input_tokens": 0,
        "output_tokens": 0,
        "pid": None,
        "thread_id": None,
    }


def create_quick_plan_and_task(
    conn: sqlite3.Connection,
    *,
    project_id: str | None = None,
    prompt: str,
    title: str,
    description: str,
    caller: str,
    backend: str,
    files: str | None = None,
    skip_review: bool = False,
    skip_merge: bool = False,
    priority: str | None = None,
    actor: str | None = None,
) -> tuple[dict, dict]:
    """Create a synthetic plan + single ready task for quick execution.

    The plan is created already finalized (no AI planner involved).
    The task enters the pipeline at 'ready' — executor/reviewer/merge
    work unchanged from there.

    Returns (plan_dict, task_dict).
    """
    all_callers = get_all_callers()
    if caller not in all_callers:
        raise ValueError(f"Invalid caller '{caller}'. Must be one of: {sorted(all_callers)}")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: {sorted(VALID_BACKENDS)}")
    if backend not in IMPLEMENTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not yet implemented. "
            f"Supported: {', '.join(sorted(IMPLEMENTED_BACKENDS))}"
        )
    if actor is None:
        actor = os.environ.get("USER", "unknown")
    normalized_priority = _normalize_task_priority(priority)

    plan_id = uuid.uuid4().hex[:12]
    task_id = uuid.uuid4().hex[:12]

    conn.execute(
        "INSERT INTO plans (id, project_id, prompt, status, prompt_status, plan, actor, "
        "caller, backend, task_creation_status, mode) "
        "VALUES (?, ?, ?, 'finalized', 'finalized', ?, ?, ?, ?, 'completed', 'quick')",
        (plan_id, project_id, prompt, prompt, actor, caller, backend),
    )
    conn.execute(
        "INSERT INTO tasks (id, plan_id, ordinal, title, description, files, status, "
        "priority, skip_review, skip_merge) "
        "VALUES (?, ?, 1, ?, ?, ?, 'ready', ?, ?, ?)",
        (
            task_id,
            plan_id,
            title,
            description,
            files,
            normalized_priority,
            int(skip_review),
            int(skip_merge),
        ),
    )
    conn.commit()

    plan_dict = {
        "id": plan_id,
        "project_id": project_id,
        "prompt": prompt,
        "status": "finalized",
        "prompt_status": "finalized",
        "plan": prompt,
        "actor": actor,
        "caller": caller,
        "backend": backend,
        "task_creation_status": "completed",
        "thread_id": None,
        "mode": "quick",
    }
    task_dict = {
        "id": task_id,
        "plan_id": plan_id,
        "ordinal": 1,
        "title": title,
        "description": description,
        "files": files,
        "status": "ready",
        "priority": normalized_priority,
        "skip_review": int(skip_review),
        "skip_merge": int(skip_merge),
    }
    return plan_dict, task_dict


def get_task(conn: sqlite3.Connection, task_id: str) -> TaskRow | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return cast(TaskRow, dict(row)) if row else None


def list_tasks(
    conn: sqlite3.Connection,
    plan_id: str | None = None,
    project_id: str | None = None,
    status: str | None = None,
    statuses: Sequence[str] | None = None,
    priority: str | None = None,
    project_id_is_null: bool = False,
) -> list[TaskRow]:
    query = "SELECT tasks.* FROM tasks"
    conditions: list[str] = []
    params: list[str] = []
    needs_plan_join = bool(project_id) or project_id_is_null
    if needs_plan_join:
        query += " JOIN plans ON tasks.plan_id = plans.id"
    if project_id:
        conditions.append("plans.project_id = ?")
        params.append(project_id)
    if project_id_is_null:
        conditions.append("plans.project_id IS NULL")
    if plan_id:
        conditions.append("tasks.plan_id = ?")
        params.append(plan_id)
    if status:
        conditions.append("tasks.status = ?")
        params.append(status)
    elif statuses:
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"tasks.status IN ({placeholders})")
        params.extend(statuses)
    if priority is not None:
        normalized_priority = _normalize_task_priority(priority)
        if normalized_priority is None:
            conditions.append("(tasks.priority IS NULL OR tasks.priority = ?)")
            params.append("medium")
        else:
            conditions.append("tasks.priority = ?")
            params.append(normalized_priority)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY tasks.created_at, tasks.ordinal"
    rows = conn.execute(query, params).fetchall()
    return [cast(TaskRow, dict(row)) for row in rows]


def _terminal_task_status_in_clause() -> str:
    return ", ".join(f"'{status}'" for status in sorted(TASK_TERMINAL_STATUSES))


def list_task_cleanup_candidates(conn: sqlite3.Connection, project_id: str) -> list[TaskRow]:
    """List terminal tasks in a project that still have branch/worktree refs."""
    terminal_statuses = _terminal_task_status_in_clause()
    rows = conn.execute(
        "SELECT tasks.* FROM tasks "
        "JOIN plans ON tasks.plan_id = plans.id "
        "WHERE plans.project_id = ? "
        f"AND tasks.status IN ({terminal_statuses}) "
        "AND tasks.branch IS NOT NULL "
        "AND tasks.worktree IS NOT NULL "
        "ORDER BY tasks.updated_at, tasks.created_at, tasks.ordinal, tasks.id",
        (project_id,),
    ).fetchall()
    return [cast(TaskRow, dict(row)) for row in rows]


def update_task_status(
    conn: sqlite3.Connection,
    task_id: str,
    status: str,
    *,
    record_history: bool = False,
) -> bool:
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"Invalid task status '{status}'. Must be one of: {VALID_TASK_STATUSES}")
    current = conn.execute("SELECT status, actor FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not current:
        return False
    old_status = current["status"]
    if old_status == status:
        return False

    # Prevent transitions out of terminal statuses (worker race protection).
    if old_status in TASK_TERMINAL_STATUSES:
        return False

    # Set started_at on first transition to running
    extra_clauses = ""
    if status == "running":
        extra_clauses += (
            ", started_at = COALESCE(started_at, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"
        )
    # Set finished_at on terminal statuses
    if status in TASK_TERMINAL_STATUSES:
        extra_clauses += ", finished_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    if status != "running":
        extra_clauses += ", active_turn_id = NULL, active_turn_started_at = NULL"

    cursor = conn.execute(
        "UPDATE tasks SET status = ?,"
        f" updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'){extra_clauses}"
        " WHERE id = ? AND status = ?",
        (status, task_id, old_status),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False

    if record_history:
        record_status_change(
            conn,
            entity_type="task",
            entity_id=task_id,
            old_status=old_status,
            new_status=status,
            actor=current["actor"],
        )
    conn.commit()
    return True


def set_task_priority(conn: sqlite3.Connection, task_id: str, priority: str | None) -> bool:
    normalized_priority = _normalize_task_priority(priority)
    cursor = conn.execute(
        "UPDATE tasks SET priority = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ?",
        (normalized_priority, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str | None = None,
    caller: str | None = None,
    branch: str | None = None,
    worktree: str | None = None,
    record_history: bool = False,
) -> bool:
    """Claim a ready task for execution, recording who took ownership.

    Sets status to 'running' and records actor/caller/branch/worktree.
    Only works on 'ready' tasks — returns False otherwise.
    """
    if actor is None:
        actor = os.environ.get("USER", "unknown")
    all_callers = get_all_callers()
    if caller and caller not in all_callers:
        raise ValueError(f"Invalid caller '{caller}'. Must be one of: {sorted(all_callers)}")
    current = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not current:
        return False
    old_status = current["status"]
    if old_status != "ready":
        return False

    cursor = conn.execute(
        "UPDATE tasks SET status = 'running', actor = ?, caller = ?, "
        "branch = ?, worktree = ?, "
        "started_at = COALESCE(started_at, strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = ?",
        (actor, caller, branch, worktree, task_id, old_status),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False

    if record_history:
        record_status_change(
            conn,
            entity_type="task",
            entity_id=task_id,
            old_status=old_status,
            new_status="running",
            actor=actor,
        )

    if cursor.rowcount > 0:
        log_id = uuid.uuid4().hex[:12]
        conn.execute(
            _INSERT_TASK_LOG,
            (log_id, task_id, "INFO", f"Claimed by {actor} via {caller}", "system"),
        )
    conn.commit()
    return True


def reset_task_for_retry(conn: sqlite3.Connection, task_id: str) -> bool:
    """Reset a failed task to blocked, clearing worker and ownership state."""
    cursor = conn.execute(
        "UPDATE tasks SET status = 'blocked', pid = NULL, thread_id = NULL, "
        "actor = NULL, caller = NULL, branch = NULL, worktree = NULL, "
        "started_at = NULL, finished_at = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = 'failed'",
        (task_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def reset_task_for_reexecution(
    conn: sqlite3.Connection,
    task_id: str,
    branch: str,
    worktree: str,
    *,
    record_history: bool = False,
) -> bool:
    """Reset an approved task for re-execution after a merge conflict.

    CAS: only transitions approved → running. Clears thread_id and
    reviewer_thread_id (force fresh execution), sets new branch/worktree.
    """
    current = conn.execute("SELECT status, actor FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not current:
        return False
    old_status = current["status"]
    if old_status != "approved":
        return False

    cursor = conn.execute(
        "UPDATE tasks SET status = 'running', "
        "thread_id = NULL, reviewer_thread_id = NULL, "
        "branch = ?, worktree = ?, "
        "started_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = 'approved'",
        (branch, worktree, task_id),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False

    if record_history:
        record_status_change(
            conn,
            entity_type="task",
            entity_id=task_id,
            old_status=old_status,
            new_status="running",
            actor=current["actor"],
        )
    conn.commit()
    return True


def fail_stale_running_task_for_doctor(
    conn: sqlite3.Connection, task_id: str, *, old_status: str = "running", reason: str
) -> bool:
    """Mark a stale active task as failed, clearing worker fields and logging audit trail.

    Handles running and review statuses via old_status parameter.
    Records status_history for audit trail continuity.
    """
    cursor = conn.execute(
        "UPDATE tasks SET status = 'failed', pid = NULL, thread_id = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND status = ?",
        (task_id, old_status),
    )
    if cursor.rowcount > 0:
        log_id = uuid.uuid4().hex[:12]
        conn.execute(
            _INSERT_TASK_LOG,
            (
                log_id,
                task_id,
                "ERROR",
                (
                    f"Doctor --fix remediation: marked stale {old_status} task as failed; "
                    f"reason: {_doctor_reason(reason)}"
                ),
                "system",
            ),
        )
        record_status_change(
            conn,
            entity_type="task",
            entity_id=task_id,
            old_status=old_status,
            new_status="failed",
            actor="doctor",
        )
    conn.commit()
    return cursor.rowcount > 0


def clear_task_git_refs(conn: sqlite3.Connection, task_id: str) -> bool:
    """Clear branch/worktree refs on a task. Returns True only when a row changed."""
    cursor = conn.execute(
        "UPDATE tasks SET branch = NULL, worktree = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND (branch IS NOT NULL OR worktree IS NOT NULL)",
        (task_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def clear_stale_task_git_refs_for_doctor(
    conn: sqlite3.Connection, task_id: str, *, reason: str
) -> dict[str, bool | str]:
    """Clear stale task git refs for doctor --fix with changed/not-changed feedback."""
    cursor = conn.execute(
        "UPDATE tasks SET branch = NULL, worktree = NULL, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND (branch IS NOT NULL OR worktree IS NOT NULL)",
        (task_id,),
    )
    changed = cursor.rowcount > 0
    if changed:
        log_id = uuid.uuid4().hex[:12]
        conn.execute(
            _INSERT_TASK_LOG,
            (
                log_id,
                task_id,
                "INFO",
                (
                    "Doctor --fix remediation: cleared stale task git refs; "
                    f"reason: {_doctor_reason(reason)}"
                ),
                "system",
            ),
        )
    conn.commit()
    return {
        "changed": changed,
        "result": "changed" if changed else "not-changed",
    }


_SENTINEL: str | None = object()  # type: ignore[assignment]


def set_task_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    pid: int,
    thread_id: str | None = _SENTINEL,
) -> bool:
    """Record worker pid and optional thread_id on a task. Does not change status.

    If thread_id is not provided, the existing thread_id is preserved.
    """
    if thread_id is _SENTINEL:
        cursor = conn.execute(
            "UPDATE tasks SET pid = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
            (pid, task_id),
        )
    else:
        cursor = conn.execute(
            "UPDATE tasks SET pid = ?, thread_id = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
            (pid, thread_id, task_id),
        )
    conn.commit()
    return cursor.rowcount > 0


def set_task_thread_id(conn: sqlite3.Connection, task_id: str, thread_id: str | None) -> bool:
    """Update just the thread_id on a task (called after thread/start)."""
    cursor = conn.execute(
        "UPDATE tasks SET thread_id = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (thread_id, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def set_task_active_turn_id(
    conn: sqlite3.Connection, task_id: str, active_turn_id: str | None
) -> bool:
    """Track active turn state for mid-turn steer support."""
    if active_turn_id:
        cursor = conn.execute(
            "UPDATE tasks SET active_turn_id = ?, "
            "active_turn_started_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = ?",
            (active_turn_id, task_id),
        )
    else:
        cursor = conn.execute(
            "UPDATE tasks SET active_turn_id = NULL, active_turn_started_at = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = ?",
            (task_id,),
        )
    conn.commit()
    return cursor.rowcount > 0


def set_task_merge_commit(conn: sqlite3.Connection, task_id: str, sha: str) -> bool:
    """Record the merge commit SHA on a task after successful merge."""
    cursor = conn.execute(
        "UPDATE tasks SET merge_commit = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (sha, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def set_task_model(conn: sqlite3.Connection, task_id: str, model: str | None) -> bool:
    """Store runtime model selection for a task."""
    task_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "model" not in task_cols:
        return False

    cursor = conn.execute(
        "UPDATE tasks SET model = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ?",
        (model, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def set_task_reviewer_thread_id(conn: sqlite3.Connection, task_id: str, thread_id: str) -> bool:
    """Record the reviewer's thread_id without touching the executor's thread_id."""
    cursor = conn.execute(
        "UPDATE tasks SET reviewer_thread_id = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (thread_id, task_id),
    )
    conn.commit()
    return cursor.rowcount > 0


# -- task blocks --


def add_task_block(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    blocked_by_task_id: str | None = None,
    external_factor: str | None = None,
    reason: str | None = None,
) -> dict:
    block_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO task_blocks (id, task_id, blocked_by_task_id, external_factor, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (block_id, task_id, blocked_by_task_id, external_factor, reason),
    )
    conn.commit()
    return {
        "id": block_id,
        "task_id": task_id,
        "blocked_by_task_id": blocked_by_task_id,
        "external_factor": external_factor,
        "reason": reason,
        "resolved": 0,
    }


def resolve_task_block(conn: sqlite3.Connection, block_id: str) -> bool:
    cursor = conn.execute(
        "UPDATE task_blocks SET resolved = 1, "
        "resolved_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ? AND resolved = 0",
        (block_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def _promote_newly_unblocked_pending_tasks(
    conn: sqlite3.Connection,
    task_ids: list[str],
    *,
    record_history: bool = False,
) -> list[str]:
    """Promote newly-unblocked blocked tasks to ready via CAS-safe updates."""
    promoted: list[str] = []
    seen: set[str] = set()
    now_expr = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"

    for tid in task_ids:
        if tid in seen:
            continue
        seen.add(tid)

        remaining = conn.execute(
            "SELECT COUNT(*) as cnt FROM task_blocks WHERE task_id = ? AND resolved = 0",
            (tid,),
        ).fetchone()["cnt"]
        if remaining != 0:
            continue

        current = conn.execute("SELECT status, actor FROM tasks WHERE id = ?", (tid,)).fetchone()
        if not current:
            continue
        old_status = current["status"]
        if old_status != "blocked":
            continue

        cursor = conn.execute(
            f"UPDATE tasks SET status = 'ready', updated_at = {now_expr} "
            "WHERE id = ? AND status = ?",
            (tid, old_status),
        )
        if cursor.rowcount == 0:
            continue

        promoted.append(tid)
        if record_history:
            record_status_change(
                conn,
                entity_type="task",
                entity_id=tid,
                old_status=old_status,
                new_status="ready",
                actor=current["actor"],
            )

    return promoted


def _resolve_blockers_for_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    record_history: bool = False,
) -> list[str]:
    """Internal: resolve blockers for a task without committing.

    Used by resolve_blockers_for_terminal_task (commits) and
    cancel_tasks_batch (caller commits).
    """
    blocks = conn.execute(
        "SELECT id, task_id FROM task_blocks WHERE blocked_by_task_id = ? AND resolved = 0",
        (task_id,),
    ).fetchall()

    if not blocks:
        return []

    now_expr = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    block_ids = [block["id"] for block in blocks]
    placeholders = ", ".join("?" for _ in block_ids)
    conn.execute(
        f"UPDATE task_blocks SET resolved = 1, resolved_at = {now_expr} "
        f"WHERE id IN ({placeholders}) AND resolved = 0",
        block_ids,
    )

    return _promote_newly_unblocked_pending_tasks(
        conn,
        [block["task_id"] for block in blocks],
        record_history=record_history,
    )


def _cascade_cancel_downstream(
    conn: sqlite3.Connection,
    cancelled_task_id: str,
    *,
    record_history: bool = False,
) -> list[str]:
    """Cancel all tasks transitively blocked by a cancelled task.

    Resolves blocker rows (so block bookkeeping is clean) and cancels
    each downstream non-terminal task. Returns list of cascade-cancelled task IDs.
    """
    now_expr = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    to_process = [cancelled_task_id]
    cancelled_ids: list[str] = []
    visited = {cancelled_task_id}

    while to_process:
        current_id = to_process.pop(0)
        blocks = conn.execute(
            "SELECT tb.id AS block_id, tb.task_id FROM task_blocks tb "
            "WHERE tb.blocked_by_task_id = ? AND tb.resolved = 0",
            (current_id,),
        ).fetchall()

        # Batch-resolve all blocker rows for this level
        block_ids = [block["block_id"] for block in blocks]
        if block_ids:
            placeholders = ", ".join("?" for _ in block_ids)
            conn.execute(
                f"UPDATE task_blocks SET resolved = 1, resolved_at = {now_expr} "
                f"WHERE id IN ({placeholders}) AND resolved = 0",
                block_ids,
            )

        for block in blocks:
            downstream_id = block["task_id"]

            if downstream_id in visited:
                continue
            visited.add(downstream_id)

            downstream = conn.execute(
                "SELECT id, status, actor FROM tasks WHERE id = ?",
                (downstream_id,),
            ).fetchone()
            if not downstream or downstream["status"] in TASK_TERMINAL_STATUSES:
                continue

            # CAS cancel
            old_status = downstream["status"]
            cursor = conn.execute(
                f"UPDATE tasks SET status = 'cancelled', finished_at = {now_expr}, "
                f"updated_at = {now_expr} WHERE id = ? AND status = ?",
                (downstream_id, old_status),
            )
            if cursor.rowcount > 0:
                cancelled_ids.append(downstream_id)
                log_id = uuid.uuid4().hex[:12]
                conn.execute(
                    _INSERT_TASK_LOG,
                    (
                        log_id,
                        downstream_id,
                        "INFO",
                        f"Cascade cancelled: dependency {cancelled_task_id[:8]} was cancelled",
                        "system",
                    ),
                )
                if record_history:
                    record_status_change(
                        conn,
                        entity_type="task",
                        entity_id=downstream_id,
                        old_status=old_status,
                        new_status="cancelled",
                        actor=downstream["actor"],
                    )
                to_process.append(downstream_id)

    return cancelled_ids


def resolve_blockers_for_terminal_task(
    conn: sqlite3.Connection,
    terminal_task_id: str,
    *,
    record_history: bool = False,
) -> tuple[list[str], list[str]]:
    """Handle blocker resolution based on terminal state.

    - completed: resolve blockers, promote downstream tasks to ready
    - cancelled: cascade-cancel all transitively-dependent tasks
    - failed: do nothing — downstream stays blocked awaiting retry

    Returns (promoted_task_ids, cascade_cancelled_task_ids).
    """
    task = conn.execute("SELECT id, status FROM tasks WHERE id = ?", (terminal_task_id,)).fetchone()
    if not task:
        conn.commit()
        return [], []

    status = task["status"]

    if status == "completed":
        promoted = _resolve_blockers_for_task(conn, terminal_task_id, record_history=record_history)
        conn.commit()
        return promoted, []

    if status == "cancelled":
        cascade_cancelled = _cascade_cancel_downstream(
            conn, terminal_task_id, record_history=record_history
        )
        conn.commit()
        return [], cascade_cancelled

    # failed — do nothing, downstream stays blocked awaiting retry
    conn.commit()
    return [], []


def resolve_stale_blockers(conn: sqlite3.Connection, *, record_history: bool = False) -> list[str]:
    """Resolve unresolved internal blockers where the blocking task completed.

    Only sweeps blockers pointing at completed tasks (not failed/cancelled).
    Failed tasks leave downstream blocked awaiting retry. Cancelled tasks
    are handled by cascade cancellation at cancel time. Idempotent.
    Returns list of task IDs promoted to ready.
    """
    stale = conn.execute(
        "SELECT tb.id, tb.task_id FROM task_blocks tb "
        "JOIN tasks t ON tb.blocked_by_task_id = t.id "
        "WHERE tb.resolved = 0 AND t.status = 'completed'",
    ).fetchall()

    if not stale:
        return []

    now_expr = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
    for row in stale:
        conn.execute(
            f"UPDATE task_blocks SET resolved = 1, resolved_at = {now_expr} "
            "WHERE id = ? AND resolved = 0",
            (row["id"],),
        )

    promoted = _promote_newly_unblocked_pending_tasks(
        conn,
        [row["task_id"] for row in stale],
        record_history=record_history,
    )

    conn.commit()
    return promoted


def list_task_blocks(
    conn: sqlite3.Connection,
    task_id: str | None = None,
    *,
    plan_id: str | None = None,
    project_id: str | None = None,
    unresolved_only: bool = False,
) -> list[TaskBlockRow]:
    parts = ["SELECT tb.* FROM task_blocks tb"]
    joins = []
    conditions = []
    params: list = []

    if task_id:
        conditions.append("tb.task_id = ?")
        params.append(task_id)
    if plan_id or project_id:
        joins.append("JOIN tasks t ON tb.task_id = t.id")
    if plan_id:
        conditions.append("t.plan_id = ?")
        params.append(plan_id)
    if project_id:
        joins.append("JOIN plans p ON t.plan_id = p.id")
        conditions.append("p.project_id = ?")
        params.append(project_id)
    if unresolved_only:
        conditions.append("tb.resolved = 0")

    query = " ".join(parts + joins)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY tb.created_at"

    rows = conn.execute(query, params).fetchall()
    return [cast(TaskBlockRow, dict(row)) for row in rows]


def get_task_block(conn: sqlite3.Connection, block_id: str) -> TaskBlockRow | None:
    row = conn.execute("SELECT * FROM task_blocks WHERE id = ?", (block_id,)).fetchone()
    return cast(TaskBlockRow, dict(row)) if row else None


def get_unresolved_block_count(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM task_blocks WHERE task_id = ? AND resolved = 0",
        (task_id,),
    ).fetchone()
    return row["cnt"]


def get_unresolved_blocker_summary(conn: sqlite3.Connection, task_id: str) -> dict[str, int]:
    """Return a breakdown of unresolved blockers for a task.

    Returns dict with keys: total, dead (cancelled/failed blockers), external.
    """
    rows = conn.execute(
        "SELECT tb.blocked_by_task_id, tb.external_factor, t.status "
        "FROM task_blocks tb "
        "LEFT JOIN tasks t ON tb.blocked_by_task_id = t.id "
        "WHERE tb.task_id = ? AND tb.resolved = 0",
        (task_id,),
    ).fetchall()
    total = len(rows)
    dead = sum(
        1 for r in rows if r["blocked_by_task_id"] and r["status"] in ("cancelled", "failed")
    )
    external = sum(1 for r in rows if r["external_factor"])
    return {"total": total, "dead": dead, "external": external}


def get_unresolved_blocker_summaries_batch(
    conn: sqlite3.Connection, task_ids: list[str]
) -> dict[str, dict[str, int]]:
    """Batch version of :func:`get_unresolved_blocker_summary`.

    Returns ``{task_id: {total, dead, external}}`` for each requested task.
    Tasks with no unresolved blockers are included with zeroed counts.
    """
    if not task_ids:
        return {}
    placeholders = ",".join("?" * len(task_ids))
    rows = conn.execute(
        "SELECT tb.task_id, tb.blocked_by_task_id, tb.external_factor, t.status "
        "FROM task_blocks tb "
        "LEFT JOIN tasks t ON tb.blocked_by_task_id = t.id "
        f"WHERE tb.task_id IN ({placeholders}) AND tb.resolved = 0",
        task_ids,
    ).fetchall()
    result: dict[str, dict[str, int]] = {
        tid: {"total": 0, "dead": 0, "external": 0} for tid in task_ids
    }
    for r in rows:
        tid = r["task_id"]
        result[tid]["total"] += 1
        if r["blocked_by_task_id"] and r["status"] in ("cancelled", "failed"):
            result[tid]["dead"] += 1
        if r["external_factor"]:
            result[tid]["external"] += 1
    return result


# -- task logs --


def add_task_log(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    level: str,
    message: str,
    source: str = "task",
) -> dict:
    log_id = uuid.uuid4().hex[:12]
    conn.execute(
        _INSERT_TASK_LOG,
        (log_id, task_id, level, message, source),
    )
    conn.commit()
    return {"id": log_id, "task_id": task_id, "level": level, "message": message, "source": source}


def list_task_logs(conn: sqlite3.Connection, task_id: str, level: str | None = None) -> list[dict]:
    if level:
        rows = conn.execute(
            "SELECT * FROM task_logs WHERE task_id = ? AND level = ? ORDER BY created_at",
            (task_id, level),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM task_logs WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _validate_recent_task_event_scope(
    task_id: str | None,
    plan_id: str | None,
    project_id: str | None,
) -> tuple[str, str]:
    """Validate and return the requested event scope as (name, value)."""
    scopes = {
        "task_id": task_id,
        "plan_id": plan_id,
        "project_id": project_id,
    }
    provided: list[str] = []
    for name, value in scopes.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string when provided.")
        provided.append(name)
    if len(provided) != 1:
        raise ValueError("Exactly one of task_id, plan_id, or project_id must be provided.")
    name = provided[0]
    return name, str(scopes[name])


def _validate_recent_task_event_limit(limit: int) -> None:
    """Validate recent-event query limit."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")


def _normalize_recent_task_event_statuses(
    visible_statuses: list[str] | tuple[str, ...] | set[str] | None,
) -> list[str] | None:
    """Normalize optional visible-status filter list."""
    if visible_statuses is None:
        return None
    if isinstance(visible_statuses, str):
        raise ValueError("visible_statuses must be a collection of status strings.")

    statuses: list[str] = []
    for status in visible_statuses:
        if not isinstance(status, str) or not status:
            raise ValueError("visible_statuses must contain only non-empty strings.")
        statuses.append(status)
    if not statuses:
        return []
    return sorted(set(statuses))


def _build_recent_task_events_query(
    *,
    scope_name: str,
    scope_value: str,
    limit: int,
    since_rowid: int,
    statuses: list[str] | None,
) -> tuple[str, list[object]]:
    """Build SQL and params for recent task event lookup."""
    query = (
        "SELECT tl.task_id AS task_id, tl.created_at AS created_at, "
        "tl.message AS message, tl.level AS level, tl.source AS source, "
        "tl.rowid AS source_rowid "
        "FROM task_logs tl JOIN tasks t ON tl.task_id = t.id"
    )
    conditions: list[str] = []
    params: list[object] = []

    if scope_name == "task_id":
        conditions.append("tl.task_id = ?")
    elif scope_name == "plan_id":
        conditions.append("t.plan_id = ?")
    else:
        query += " JOIN plans p ON t.plan_id = p.id"
        conditions.append("p.project_id = ?")
    params.append(scope_value)

    if since_rowid > 0:
        conditions.append("tl.rowid > ?")
        params.append(since_rowid)

    if statuses is not None:
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"t.status IN ({placeholders})")
        params.extend(statuses)

    query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY tl.rowid DESC LIMIT ?"
    params.append(limit)
    return query, params


def list_recent_task_events(
    conn: sqlite3.Connection,
    *,
    task_id: str | None = None,
    plan_id: str | None = None,
    project_id: str | None = None,
    limit: int,
    visible_statuses: list[str] | tuple[str, ...] | set[str] | None = None,
    since_rowid: int = 0,
) -> list[dict]:
    """List recent task log events for exactly one scope, newest first.

    When since_rowid is provided, only events with tl.rowid > the given
    value are returned (for incremental streaming).
    """
    scope_name, scope_value = _validate_recent_task_event_scope(task_id, plan_id, project_id)
    _validate_recent_task_event_limit(limit)
    statuses = _normalize_recent_task_event_statuses(visible_statuses)
    if statuses == []:
        return []

    query, params = _build_recent_task_events_query(
        scope_name=scope_name,
        scope_value=scope_value,
        limit=limit,
        since_rowid=since_rowid,
        statuses=statuses,
    )
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_task_event_watermark(
    conn: sqlite3.Connection,
    *,
    task_id: str | None = None,
    plan_id: str | None = None,
    project_id: str | None = None,
) -> int:
    """Return max task_log rowid for the given scope.

    Used to initialize streaming watch watermarks reliably.
    """
    query = (
        "SELECT COALESCE(MAX(tl.rowid), 0) AS m FROM task_logs tl JOIN tasks t ON tl.task_id = t.id"
    )
    params: list[object] = []
    if task_id is not None:
        query += " WHERE tl.task_id = ?"
        params.append(task_id)
    elif plan_id is not None:
        query += " WHERE t.plan_id = ?"
        params.append(plan_id)
    elif project_id is not None:
        query += " JOIN plans p ON t.plan_id = p.id WHERE p.project_id = ?"
        params.append(project_id)
    return conn.execute(query, params).fetchone()["m"]


# -- task batch insert --


def cancel_tasks_batch(
    conn: sqlite3.Connection,
    cancellations: list[dict],
    *,
    record_history: bool = False,
) -> dict[str, int]:
    """Cancel tasks by ID with a reason logged to task_logs.

    cancellations: list of dicts with keys: task_id, reason
    Only cancels tasks in blocked/ready status. Tasks in other statuses are
    skipped with an INFO task_log explaining why. Does NOT resolve blockers
    or promote downstream tasks — cancelled tasks should not unblock dependents.
    The task agent manages the full landscape; if it wants downstream cancelled,
    it includes them in the batch.
    Returns: {"cancelled": count of tasks moved to cancelled}
    """
    count = 0
    for c in cancellations:
        task_id = c["task_id"]
        current = conn.execute(
            "SELECT status, actor FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not current:
            continue
        old_status = current["status"]
        if old_status not in {"blocked", "ready"}:
            log_id = uuid.uuid4().hex[:12]
            conn.execute(
                _INSERT_TASK_LOG,
                (
                    log_id,
                    task_id,
                    "INFO",
                    f"Batch cancel skipped: task is '{old_status}'"
                    " (only blocked/ready can be batch-cancelled)",
                    "system",
                ),
            )
            continue

        cursor = conn.execute(
            "UPDATE tasks SET status = 'cancelled', "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = ? AND status = ?",
            (task_id, old_status),
        )
        if cursor.rowcount > 0:
            count += 1
            log_id = uuid.uuid4().hex[:12]
            conn.execute(
                _INSERT_TASK_LOG,
                (log_id, task_id, "INFO", f"Cancelled: {c.get('reason', 'no reason')}", "system"),
            )
            if record_history:
                record_status_change(
                    conn,
                    entity_type="task",
                    entity_id=task_id,
                    old_status=old_status,
                    new_status="cancelled",
                    actor=current["actor"],
                )
    conn.commit()
    return {"cancelled": count}


def _insert_task_dependency_block(
    conn: sqlite3.Connection, task_id: str, blocked_by_task_id: str
) -> None:
    block_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO task_blocks (id, task_id, blocked_by_task_id) VALUES (?, ?, ?)",
        (block_id, task_id, blocked_by_task_id),
    )


def _insert_task_external_block(
    conn: sqlite3.Connection, task_id: str, factor: str, reason: str | None
) -> None:
    block_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO task_blocks (id, task_id, external_factor, reason) VALUES (?, ?, ?, ?)",
        (block_id, task_id, factor, reason),
    )


def _insert_task_batch_row(
    conn: sqlite3.Connection,
    plan_id: str,
    task_data: dict,
    ordinal_to_id: dict[int, str],
) -> None:
    task_id = uuid.uuid4().hex[:12]
    ordinal_to_id[task_data["ordinal"]] = task_id
    conn.execute(
        "INSERT INTO tasks "
        "(id, plan_id, ordinal, title, description, files, status,"
        " input_tokens, output_tokens, bucket, priority) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)",
        (
            task_id,
            plan_id,
            task_data["ordinal"],
            task_data["title"],
            task_data["description"],
            task_data.get("files"),
            task_data.get("status", "blocked"),
            task_data.get("bucket"),
            _normalize_task_priority(task_data.get("priority")),
        ),
    )


def _insert_task_batch_blocks(
    conn: sqlite3.Connection,
    task_data: dict,
    task_id: str,
    ordinal_to_id: dict[int, str],
) -> None:
    for dep_ordinal in task_data.get("blocked_by", []):
        dep_id = ordinal_to_id.get(dep_ordinal)
        if dep_id is None:
            continue
        _insert_task_dependency_block(conn, task_id, dep_id)

    for existing_id in task_data.get("blocked_by_existing", []):
        _insert_task_dependency_block(conn, task_id, existing_id)

    for ext in task_data.get("external_blockers", []):
        _insert_task_external_block(conn, task_id, ext["factor"], ext.get("reason"))


def _group_tasks_by_bucket(tasks_data: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for task_data in tasks_data:
        bucket = task_data.get("bucket")
        if bucket:
            buckets.setdefault(bucket, []).append(task_data)
    return buckets


def _strip_singleton_buckets(
    conn: sqlite3.Connection,
    buckets: dict[str, list[dict]],
    ordinal_to_id: dict[int, str],
) -> dict[str, list[dict]]:
    """Remove bucket labels from tasks that are alone in their bucket."""
    multi: dict[str, list[dict]] = {}
    for label, tasks in buckets.items():
        if len(tasks) >= 2:
            multi[label] = tasks
        else:
            for task_data in tasks:
                task_id = ordinal_to_id[task_data["ordinal"]]
                conn.execute("UPDATE tasks SET bucket = NULL WHERE id = ?", (task_id,))
    return multi


def _apply_bucket_auto_serialization(
    conn: sqlite3.Connection,
    buckets: dict[str, list[dict]],
    ordinal_to_id: dict[int, str],
    explicit_blockers: dict[int, set[int]],
) -> None:
    for bucket_tasks in buckets.values():
        bucket_tasks.sort(key=lambda task: task["ordinal"])
        for prev, cur in zip(bucket_tasks, bucket_tasks[1:], strict=False):
            cur_id = ordinal_to_id[cur["ordinal"]]
            prev_ordinal = prev["ordinal"]
            if prev_ordinal not in explicit_blockers.get(cur["ordinal"], set()):
                prev_id = ordinal_to_id[prev_ordinal]
                _insert_task_dependency_block(conn, cur_id, prev_id)
            conn.execute(
                "UPDATE tasks SET status = 'blocked' WHERE id = ? AND status != 'blocked'",
                (cur_id,),
            )


def create_tasks_batch(
    conn: sqlite3.Connection, plan_id: str, tasks_data: list[dict]
) -> dict[int, str]:
    """Insert all tasks + blocks in a single transaction.

    tasks_data: list of dicts with keys:
        ordinal, title, description, files (JSON string or None),
        status, blocked_by (list of ordinals), blocked_by_existing (list of task IDs),
        external_blockers (list of {factor, reason}), bucket (str or None),
        priority (str | None; high/medium/low where medium stores as NULL)

    Bucket auto-serialization: tasks sharing a non-null bucket are serialized
    by ordinal within the bucket. Each task is auto-blocked by the previous
    task in its bucket (skipped if the blocker already exists). Non-first
    bucket tasks are forced to "blocked" (they have blockers).

    Returns {ordinal: task_id} mapping.
    """
    ordinal_to_id: dict[int, str] = {}
    for task_data in tasks_data:
        _insert_task_batch_row(conn, plan_id, task_data, ordinal_to_id)

    explicit_blockers = {
        task_data["ordinal"]: set(task_data.get("blocked_by", [])) for task_data in tasks_data
    }

    for task_data in tasks_data:
        _insert_task_batch_blocks(
            conn,
            task_data,
            ordinal_to_id[task_data["ordinal"]],
            ordinal_to_id,
        )

    buckets = _group_tasks_by_bucket(tasks_data)
    buckets = _strip_singleton_buckets(conn, buckets, ordinal_to_id)
    _apply_bucket_auto_serialization(
        conn,
        buckets,
        ordinal_to_id,
        explicit_blockers,
    )

    conn.commit()
    return ordinal_to_id


# -- purge --


def purge_data(conn: sqlite3.Connection, project_id: str | None) -> dict:
    """Delete operational rows and return purge summary + related IDs."""
    if project_id:
        plan_rows = conn.execute(
            "SELECT id FROM plans WHERE project_id = ?", (project_id,)
        ).fetchall()
    else:
        plan_rows = conn.execute("SELECT id FROM plans").fetchall()

    plan_ids = [row["id"] for row in plan_rows]

    task_ids: list[str] = []
    if plan_ids:
        placeholders = ",".join("?" * len(plan_ids))
        task_rows = conn.execute(
            f"SELECT id FROM tasks WHERE plan_id IN ({placeholders})",
            plan_ids,
        ).fetchall()
        task_ids = [row["id"] for row in task_rows]

    counts: dict[str, int] = {}

    if task_ids:
        placeholders = ",".join("?" * len(task_ids))
        for table, column in [
            ("status_history", "entity_id"),
            ("task_logs", "task_id"),
            ("task_blocks", "task_id"),
            ("task_steers", "task_id"),
        ]:
            cur = conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", task_ids)
            counts[table] = counts.get(table, 0) + cur.rowcount

        cur = conn.execute(
            f"DELETE FROM task_blocks WHERE blocked_by_task_id IN ({placeholders})",
            task_ids,
        )
        counts["task_blocks"] = counts.get("task_blocks", 0) + cur.rowcount

        cur = conn.execute(
            "DELETE FROM trace_events"
            f" WHERE entity_type = 'task' AND entity_id IN ({placeholders})",
            task_ids,
        )
        counts["trace_events"] = counts.get("trace_events", 0) + cur.rowcount
    else:
        counts["status_history"] = counts.get("status_history", 0)
        counts["task_logs"] = 0
        counts["task_blocks"] = 0
        counts["task_steers"] = 0
        counts["trace_events"] = counts.get("trace_events", 0)

    if plan_ids:
        placeholders = ",".join("?" * len(plan_ids))
        cur = conn.execute(
            f"DELETE FROM status_history WHERE entity_id IN ({placeholders})",
            plan_ids,
        )
        counts["status_history"] = counts.get("status_history", 0) + cur.rowcount
        for table, column in [("plan_logs", "plan_id"), ("plan_questions", "plan_id")]:
            cur = conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", plan_ids)
            counts[table] = cur.rowcount

        cur = conn.execute(
            "DELETE FROM trace_events"
            f" WHERE entity_type = 'plan' AND entity_id IN ({placeholders})",
            plan_ids,
        )
        counts["trace_events"] = counts.get("trace_events", 0) + cur.rowcount

    if task_ids:
        placeholders = ",".join("?" * len(task_ids))
        cur = conn.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", task_ids)
        counts["tasks"] = cur.rowcount
    else:
        counts["tasks"] = 0

    if plan_ids:
        placeholders = ",".join("?" * len(plan_ids))
        cur = conn.execute(f"DELETE FROM plans WHERE id IN ({placeholders})", plan_ids)
        counts["plans"] = cur.rowcount
    else:
        counts["plans"] = 0

    # Delete channel_messages and sessions scoped to the project
    if project_id:
        session_rows = conn.execute(
            "SELECT id FROM sessions WHERE project_id = ?", (project_id,)
        ).fetchall()
    else:
        session_rows = conn.execute("SELECT id FROM sessions").fetchall()
    session_ids = [row["id"] for row in session_rows]

    if session_ids:
        placeholders = ",".join("?" * len(session_ids))
        cur = conn.execute(
            f"DELETE FROM task_steers WHERE session_id IN ({placeholders})",
            session_ids,
        )
        counts["task_steers"] = counts.get("task_steers", 0) + cur.rowcount
        cur = conn.execute(
            f"DELETE FROM channel_messages WHERE session_id IN ({placeholders})",
            session_ids,
        )
        counts["channel_messages"] = cur.rowcount
        cur = conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
        counts["sessions"] = cur.rowcount
    else:
        counts["task_steers"] = counts.get("task_steers", 0)
        counts["channel_messages"] = 0
        counts["sessions"] = 0

    conn.commit()
    return {"counts": counts, "plan_ids": plan_ids, "task_ids": task_ids}


def purge_preview_counts(conn: sqlite3.Connection, project_id: str | None) -> dict[str, int]:
    """Count rows that would be deleted by a purge operation."""
    counts: dict[str, int] = {}
    if project_id:
        plan_subq = "SELECT id FROM plans WHERE project_id = ?"
        task_subq = f"SELECT id FROM tasks WHERE plan_id IN ({plan_subq})"
        session_subq = "SELECT id FROM sessions WHERE project_id = ?"
        params: tuple[str, ...] = (project_id,)
    else:
        plan_subq = "SELECT id FROM plans"
        task_subq = "SELECT id FROM tasks"
        session_subq = "SELECT id FROM sessions"
        params = ()

    queries: list[tuple[str, str, tuple[str, ...]]] = [
        ("plans", plan_subq, params),
        ("tasks", task_subq, params),
        ("task_logs", f"SELECT id FROM task_logs WHERE task_id IN ({task_subq})", params),
        (
            "task_blocks",
            (
                "SELECT id FROM task_blocks WHERE "
                f"task_id IN ({task_subq}) OR blocked_by_task_id IN ({task_subq})"
            ),
            params + params,
        ),
        ("plan_logs", f"SELECT id FROM plan_logs WHERE plan_id IN ({plan_subq})", params),
        ("plan_questions", f"SELECT id FROM plan_questions WHERE plan_id IN ({plan_subq})", params),
        (
            "status_history",
            (
                "SELECT id FROM status_history WHERE "
                f"entity_id IN ({plan_subq}) OR entity_id IN ({task_subq})"
            ),
            params + params,
        ),
        (
            "trace_events",
            (
                "SELECT id FROM trace_events WHERE "
                f"(entity_type = 'plan' AND entity_id IN ({plan_subq})) OR "
                f"(entity_type = 'task' AND entity_id IN ({task_subq}))"
            ),
            params + params,
        ),
        (
            "channel_messages",
            f"SELECT id FROM channel_messages WHERE session_id IN ({session_subq})",
            params,
        ),
        (
            "task_steers",
            (
                "SELECT id FROM task_steers WHERE "
                f"task_id IN ({task_subq}) OR session_id IN ({session_subq})"
            ),
            params + params,
        ),
        ("sessions", session_subq, params),
    ]
    for table, subq, qparams in queries:
        row = conn.execute(f"SELECT COUNT(*) as c FROM ({subq})", qparams).fetchone()
        counts[table] = row["c"] if row else 0

    return counts
