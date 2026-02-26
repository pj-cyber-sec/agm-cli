"""JSON stdin/stdout dispatch layer for agm-web (and any non-CLI consumer).

Protocol:
    stdin:  {"method": "task.show", "params": {"id": "abc123"}}
    stdout: {"ok": true, "data": {...}}
    stdout: {"ok": false, "error": "Not found", "code": "NOT_FOUND"}

Always exits 0. Always returns JSON on stdout. No stderr parsing needed.
Entry point: ``agm-api`` console script (pyproject.toml).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from agm.agents_config import (
    SUPPORTED_ROLES,
    get_effective_role_config,
    get_global_role_text,
    get_project_role_text,
)
from agm.backends import (
    _env_model_lookup_for_backend,
    _get_default_effort_for_backend_tier,
    _get_default_model_for_backend_tier,
    resolve_model_config,
)
from agm.callers import BUILTIN_CALLERS, get_all_callers
from agm.daemon_threads import normalize_daemon_thread_list
from agm.db import (
    VALID_MESSAGE_KINDS,
    ProjectRow,
    add_channel_message,
    add_task_steer,
    bulk_active_runtime_seconds,
    connect,
    get_plan_chain,
    get_plan_request,
    get_project,
    get_project_base_branch,
    get_project_model_config,
    get_session,
    get_task,
    get_task_rejection_count,
    get_trace_summary,
    get_unresolved_block_count,
    list_channel_messages,
    list_plan_logs,
    list_plan_questions,
    list_plan_requests,
    list_plan_watch_events,
    list_projects,
    list_recent_task_events,
    list_sessions,
    list_status_history_timing_rows,
    list_task_blocks,
    list_task_logs,
    list_task_steers,
    list_tasks,
    list_trace_events,
    parse_app_server_approval_policy,
    parse_app_server_ask_for_approval,
    reconcile_session_statuses,
)
from agm.queries import (
    PLAN_ACTIVE_STATUSES,
    PLAN_WATCH_RECENT_EVENTS_ROWS,
    TASK_ACTIVE_STATUSES,
    build_plan_stats_data,
    build_plan_watch_snapshot,
    build_task_watch_snapshot,
    effective_task_priority,
    enrich_plan_list_rows,
    enrich_task_list_rows,
    format_duration_seconds,
    format_plan_failure_error,
    format_plan_failure_prompt,
    gather_project_summaries,
    is_effectively_terminal_task,
    latest_task_thread_statuses,
    model_usage_counts,
    normalize_logs,
    normalize_plan_chain,
    normalize_plan_questions,
    normalize_task_blocks,
    normalize_timeline_rows,
    plan_failure_diagnostic,
    plan_watch_terminal_state,
    project_token_totals,
    resolve_project_names_for_tasks,
    status_counts,
    task_failure_diagnostic,
    task_list_filter_rows,
    task_watch_terminal_state,
    watch_short_id,
    watch_truncate,
)
from agm.status_reference import get_status_reference
from agm.steering import default_executor_recipient, steer_active_turn

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

NOT_FOUND = "NOT_FOUND"
INVALID_PARAMS = "INVALID_PARAMS"
INVALID_METHOD = "INVALID_METHOD"
INTERNAL = "INTERNAL"


class ApiError(Exception):
    """Raised by handlers to produce a structured error response."""

    def __init__(self, message: str, code: str = INTERNAL):
        super().__init__(message)
        self.code = code


def _require(params: dict, key: str) -> str:
    """Extract a required string param, raising ApiError if missing."""
    val = params.get(key)
    if not val:
        raise ApiError(f"Missing required param: {key}", INVALID_PARAMS)
    return str(val)


def _optional(params: dict, key: str) -> str | None:
    val = params.get(key)
    return str(val) if val is not None else None


def _optional_bool(params: dict, key: str, default: bool = False) -> bool:
    val = params.get(key)
    if val is None:
        return default
    return bool(val)


def _optional_non_negative_int(params: dict, key: str, default: int) -> int:
    val = params.get(key)
    if val is None:
        return default
    try:
        parsed = int(val)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"Param '{key}' must be an integer", INVALID_PARAMS) from exc
    if parsed < 0:
        raise ApiError(f"Param '{key}' must be >= 0", INVALID_PARAMS)
    return parsed


def _resolve_project_id(conn, params: dict) -> str | None:
    """Resolve project_id from either ``project_id`` (UUID) or ``project`` (name).

    Returns the project UUID, or None if neither param is provided.
    Raises ApiError if project name is provided but not found.
    """
    project_id = _optional(params, "project_id")
    if project_id:
        return project_id
    project_name = _optional(params, "project")
    if not project_name:
        return None
    proj = get_project(conn, project_name)
    if not proj:
        raise ApiError(f"Project '{project_name}' not found", NOT_FOUND)
    return proj["id"]


WATCH_RECENT_EVENT_FETCH_LIMIT = 200


# ---------------------------------------------------------------------------
# Handlers — each takes (conn, params) and returns JSON-serializable data
# ---------------------------------------------------------------------------

# -- tasks --


def _handle_task_show(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    return dict(t)


def _handle_task_list(conn, params):
    plan_id = _optional(params, "plan_id")
    project_id = _resolve_project_id(conn, params)
    status = _optional(params, "status")
    show_all = _optional_bool(params, "show_all")

    tasks = list_tasks(conn, plan_id=plan_id, project_id=project_id, status=status)
    filtered = task_list_filter_rows(tasks, show_all=show_all, status=status)

    project_names = resolve_project_names_for_tasks(conn, filtered)
    runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)

    return enrich_task_list_rows(filtered, project_names, runtimes)


def _handle_task_logs(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    level = _optional(params, "level")
    logs = list_task_logs(conn, task_id, level=level)
    normalized = normalize_logs(logs)
    return {
        "task_id": task_id,
        "level": level,
        "count": len(normalized),
        "logs": normalized,
    }


def _handle_task_blocks(conn, params):
    task_id = _optional(params, "id")
    plan_id = _optional(params, "plan_id")
    project_id = _optional(params, "project_id")
    unresolved_only = _optional_bool(params, "unresolved_only")

    blocks = list_task_blocks(conn, task_id=task_id, plan_id=plan_id, project_id=project_id)
    normalized = normalize_task_blocks(blocks, unresolved_only)

    project_name = None
    if project_id:
        proj = get_project(conn, project_id)
        if proj:
            project_name = proj["name"]

    return {
        "scope": {
            "task_id": task_id,
            "plan_id": plan_id,
            "project_id": project_id,
            "project_name": project_name,
            "unresolved_only": unresolved_only,
        },
        "count": len(normalized),
        "blocks": normalized,
    }


def _handle_task_timeline(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    rows = list_status_history_timing_rows(conn, entity_type="task", entity_id=task_id)
    return {
        "task_id": task_id,
        "timeline": normalize_timeline_rows(rows),
    }


def _handle_task_trace(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    event_type = _optional(params, "event_type")
    stage = _optional(params, "stage")
    limit = int(params["limit"]) if params.get("limit") else None
    events = list_trace_events(
        conn, "task", task_id, event_type=event_type, stage=stage, limit=limit
    )
    return {"task_id": task_id, "count": len(events), "events": events}


def _handle_task_trace_summary(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    data = get_trace_summary(conn, "task", task_id)
    data["task_id"] = task_id
    return data


def _handle_task_diff(conn, params):
    task_id = _require(params, "id")
    t = get_task(conn, task_id)
    if not t:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    branch = t.get("branch")
    merge_commit = t.get("merge_commit")
    if not branch and not merge_commit:
        raise ApiError(f"Task '{task_id}' has no branch or merge commit", INVALID_PARAMS)
    plan = get_plan_request(conn, t["plan_id"])
    if not plan:
        raise ApiError(f"Plan '{t['plan_id']}' not found", NOT_FOUND)
    proj = get_project(conn, plan["project_id"])
    if not proj:
        raise ApiError(f"Project '{plan['project_id']}' not found", NOT_FOUND)
    base_branch = get_project_base_branch(conn, proj["id"])
    project_dir = proj["dir"]

    diff_text = _task_diff_resolve_text(project_dir, base_branch, branch, merge_commit)
    return {"task_id": task_id, "diff": diff_text}


def _handle_task_failures(conn, params):
    project_id = _resolve_project_id(conn, params)
    plan_id = _optional(params, "plan_id")

    tasks = list_tasks(conn, plan_id=plan_id, project_id=project_id, status="failed")
    failures = []
    for t in reversed(tasks):
        plan = get_plan_request(conn, t["plan_id"])
        proj = get_project(conn, plan["project_id"]) if plan else None
        project = proj["name"] if proj else (plan["project_id"] if plan else "-")
        source, error = task_failure_diagnostic(conn, t["id"])
        failures.append(
            {
                "task_id": t["id"],
                "plan_id": t["plan_id"],
                "project_id": plan["project_id"] if plan else None,
                "project": project,
                "title": t.get("title", ""),
                "title_snippet": watch_truncate(t.get("title", ""), 50),
                "source": source,
                "error": error,
                "error_snippet": format_plan_failure_error(error),
                "priority": effective_task_priority(t.get("priority")),
                "created_at": t.get("created_at"),
                "updated_at": t.get("updated_at"),
                "failed": t.get("updated_at"),
            }
        )
    return failures


# -- plans --


def _handle_plan_show(conn, params):
    plan_id = _require(params, "id")
    show_tasks = _optional_bool(params, "show_tasks")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    result = dict(p)
    if show_tasks:
        tasks = list_tasks(conn, plan_id=plan_id)
        project_names = resolve_project_names_for_tasks(conn, tasks)
        runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)
        result["tasks"] = enrich_task_list_rows(tasks, project_names, runtimes)
    return result


def _handle_plan_list(conn, params):
    project_id = _resolve_project_id(conn, params)
    status = _optional(params, "status")
    show_all = _optional_bool(params, "show_all")

    # Match CLI: default to active statuses unless show_all or explicit status
    default_statuses = None if (show_all or status) else ["pending", "running", "awaiting_input"]

    plans = list_plan_requests(
        conn,
        project_id=project_id,
        status=status,
        statuses=default_statuses,
    )
    runtimes = bulk_active_runtime_seconds(conn, "plan", PLAN_ACTIVE_STATUSES)
    return enrich_plan_list_rows(conn, plans, runtimes)


def _handle_plan_logs(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    level = _optional(params, "level")
    logs = list_plan_logs(conn, plan_id, level=level)
    normalized = normalize_logs(logs)
    return {
        "plan_id": plan_id,
        "level": level,
        "count": len(normalized),
        "logs": normalized,
    }


def _handle_plan_timeline(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=plan_id)
    return {
        "plan_id": plan_id,
        "timeline": normalize_timeline_rows(rows),
    }


def _handle_plan_questions(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    unanswered_only = _optional_bool(params, "unanswered_only")
    questions = list_plan_questions(conn, plan_id, unanswered_only=unanswered_only)
    normalized = normalize_plan_questions(questions)
    return {
        "plan_id": plan_id,
        "unanswered_only": unanswered_only,
        "count": len(normalized),
        "questions": normalized,
    }


def _handle_plan_history(conn, params):
    plan_id = _require(params, "id")
    chain = get_plan_chain(conn, plan_id)
    if not chain:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    return {
        "plan_id": plan_id,
        "chain": normalize_plan_chain(chain, plan_id),
    }


def _handle_plan_stats(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    tasks = list_tasks(conn, plan_id=plan_id)
    timing_rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=plan_id)
    return build_plan_stats_data(conn, plan_id, p, tasks, timing_rows)


def _handle_plan_failures(conn, params):
    project_id = _resolve_project_id(conn, params)

    plans = list_plan_requests(conn, project_id=project_id, status="failed")
    failures = []
    for p in reversed(plans):
        proj = get_project(conn, p["project_id"])
        project = proj["name"] if proj else p["project_id"]
        source, error = plan_failure_diagnostic(conn, p["id"])
        failures.append(
            {
                "plan_id": p["id"],
                "project_id": p["project_id"],
                "project": project,
                "source": source,
                "prompt": p.get("prompt", ""),
                "prompt_snippet": format_plan_failure_prompt(p.get("prompt", "")),
                "error": error,
                "error_snippet": format_plan_failure_error(error),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at"),
                "failed": p.get("updated_at"),
            }
        )
    return failures


def _handle_plan_trace(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    event_type = _optional(params, "event_type")
    stage = _optional(params, "stage")
    limit = int(params["limit"]) if params.get("limit") else None
    events = list_trace_events(
        conn, "plan", plan_id, event_type=event_type, stage=stage, limit=limit
    )
    return {"plan_id": plan_id, "count": len(events), "events": events}


def _handle_plan_trace_summary(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    data = get_trace_summary(conn, "plan", plan_id)
    data["plan_id"] = plan_id
    return data


def _handle_plan_watch(conn, params):
    plan_id = _require(params, "id")
    p = get_plan_request(conn, plan_id)
    if not p:
        raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
    tasks = list_tasks(conn, plan_id=plan_id)
    recent_events = list_plan_watch_events(conn, plan_id, limit=PLAN_WATCH_RECENT_EVENTS_ROWS)
    timing_rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=plan_id)
    terminal_state = plan_watch_terminal_state(p, tasks)
    return build_plan_watch_snapshot(
        p,
        tasks,
        recent_events,
        terminal_state=terminal_state,
        timing_rows=timing_rows,
    )


def _handle_task_watch(conn, params):
    task_id = _optional(params, "task_id")
    plan_id = _optional(params, "plan_id")
    project_name = _optional(params, "project")
    show_all = _optional_bool(params, "show_all")

    # Validate exactly one scope
    provided = [
        name
        for name, val in [("task_id", task_id), ("plan_id", plan_id), ("project", project_name)]
        if val
    ]
    if not provided:
        raise ApiError("Provide exactly one of: task_id, plan_id, project", INVALID_PARAMS)
    if len(provided) > 1:
        raise ApiError("Provide exactly one of: task_id, plan_id, project", INVALID_PARAMS)

    # Resolve scope
    plan_backends: dict[str, str] = {}
    if task_id:
        scope, event_kwargs = _task_watch_scope_for_task(conn, task_id, plan_backends)
    elif project_name:
        scope, event_kwargs = _task_watch_scope_for_project(conn, project_name)
    else:
        scope, event_kwargs = _task_watch_scope_for_plan(conn, plan_id, plan_backends)

    # Load tasks
    tasks = _task_watch_load_tasks(conn, task_id, plan_id, project_name)
    _task_watch_fill_plan_backends(conn, tasks, plan_backends)

    blocker_counts: dict[str, int] = {
        str(t["id"]): get_unresolved_block_count(conn, str(t["id"])) for t in tasks
    }
    events = list_recent_task_events(
        conn,
        task_id=event_kwargs["task_id"],
        plan_id=event_kwargs["plan_id"],
        project_id=event_kwargs["project_id"],
        limit=WATCH_RECENT_EVENT_FETCH_LIMIT,
    )
    t_runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)

    if task_id or show_all:
        visible = tasks
    else:
        visible = [t for t in tasks if not is_effectively_terminal_task(t)]
    scope_rt_secs = sum(t_runtimes.get(str(t["id"]), 0) for t in tasks) or None
    watch_elapsed = format_duration_seconds(scope_rt_secs) if scope_rt_secs else "-"
    thread_status_by_task = latest_task_thread_statuses(conn, tasks)

    terminal_state = task_watch_terminal_state(tasks)
    return build_task_watch_snapshot(
        scope=scope,
        tasks=tasks,
        visible_tasks=visible,
        recent_events=events,
        blocker_counts=blocker_counts,
        plan_backends=plan_backends,
        watch_elapsed=watch_elapsed,
        terminal_state=terminal_state,
        thread_status_by_task=thread_status_by_task,
    )


# -- task watch helpers (inline, mirrors cli.py) --


_EventScope = dict[str, str | None]


def _task_watch_scope_for_task(
    conn,
    task_id: str,
    plan_backends: dict[str, str],
) -> tuple[dict[str, object], _EventScope]:
    task_row = get_task(conn, task_id)
    if not task_row:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    task_plan = get_plan_request(conn, task_row["plan_id"])
    task_backend = (task_plan.get("backend") if task_plan else None) or "codex"
    plan_backends[task_row["plan_id"]] = task_backend
    scope: dict[str, object] = {
        "type": "task",
        "task_id": task_row["id"],
        "task_id_short": watch_short_id(task_row["id"]),
        "plan_id": task_row.get("plan_id"),
        "session_id": task_plan.get("session_id") if task_plan else None,
        "backend": task_backend,
    }
    return scope, {"task_id": task_row["id"], "plan_id": None, "project_id": None}


def _task_watch_scope_for_project(
    conn,
    project_name: str,
) -> tuple[dict[str, object], _EventScope]:
    proj = get_project(conn, project_name)
    if not proj:
        raise ApiError(f"Project '{project_name}' not found", NOT_FOUND)
    scope: dict[str, object] = {
        "type": "project",
        "project_id": proj["id"],
        "project_id_short": watch_short_id(proj["id"]),
        "project_name": proj["name"],
    }
    return scope, {"task_id": None, "plan_id": None, "project_id": proj["id"]}


def _task_watch_scope_for_plan(
    conn,
    plan_id: str | None,
    plan_backends: dict[str, str],
) -> tuple[dict[str, object], _EventScope]:
    plan_id_value = plan_id or ""
    scope: dict[str, object] = {
        "type": "plan",
        "plan_id": plan_id,
        "plan_id_short": watch_short_id(plan_id_value),
    }
    if plan_id:
        resolved_plan = get_plan_request(conn, plan_id)
        if not resolved_plan:
            raise ApiError(f"Plan '{plan_id}' not found", NOT_FOUND)
        backend = resolved_plan.get("backend") or "codex"
        scope["plan_status"] = resolved_plan.get("status")
        scope["session_id"] = resolved_plan.get("session_id")
        scope["backend"] = backend
        plan_backends[plan_id] = backend
        scope["title"] = resolved_plan.get("prompt") or ""
        scope["title_truncated"] = watch_truncate(resolved_plan.get("prompt") or "", 60)
    return scope, {"task_id": None, "plan_id": plan_id, "project_id": None}


def _task_watch_load_tasks(conn, task_id, plan_id, project_name):
    if task_id:
        task_row = get_task(conn, task_id)
        if not task_row:
            return []
        enriched = dict(task_row)
        enriched["rejection_count"] = get_task_rejection_count(conn, task_row["id"])
        return [enriched]
    project_id = None
    if project_name:
        proj = get_project(conn, project_name)
        if proj:
            project_id = proj["id"]
    tasks = list_tasks(conn, plan_id=plan_id, project_id=project_id)
    result = []
    for t in tasks:
        enriched = dict(t)
        enriched["rejection_count"] = get_task_rejection_count(conn, t["id"])
        result.append(enriched)
    return result


def _task_watch_fill_plan_backends(conn, tasks, plan_backends):
    for task_row in tasks:
        pid = task_row.get("plan_id")
        if not pid or pid in plan_backends:
            continue
        plan_row = get_plan_request(conn, pid)
        plan_backends[pid] = (plan_row.get("backend") if plan_row else None) or "codex"


# -- sessions --


def _handle_session_show(conn, params):
    session_id = _require(params, "id")
    reconcile_session_statuses(conn, session_id=session_id)
    s = get_session(conn, session_id)
    if not s:
        raise ApiError(f"Session '{session_id}' not found", NOT_FOUND)
    plans = list_plan_requests(conn, session_id=session_id)
    return {"session": dict(s), "plans": [dict(p) for p in plans]}


def _handle_session_list(conn, params):
    reconcile_session_statuses(conn)
    project_id = _resolve_project_id(conn, params)
    status = _optional(params, "status")
    show_all = _optional_bool(params, "show_all")

    if status:
        sessions = list_sessions(conn, project_id=project_id, status=status)
    elif show_all:
        sessions = list_sessions(conn, project_id=project_id)
    else:
        sessions = list_sessions(conn, project_id=project_id, statuses=["open", "active"])
    return [dict(s) for s in sessions]


def _handle_session_messages(conn, params):
    session_id = _require(params, "id")
    reconcile_session_statuses(conn, session_id=session_id)
    s = get_session(conn, session_id)
    if not s:
        raise ApiError(f"Session '{session_id}' not found", NOT_FOUND)
    kind = _optional(params, "kind")
    sender = _optional(params, "sender")
    recipient = _optional(params, "recipient")
    limit = _optional_non_negative_int(params, "limit", 100)
    offset = _optional_non_negative_int(params, "offset", 0)
    messages = list_channel_messages(
        conn,
        session_id,
        kind=kind,
        sender=sender,
        recipient=recipient,
        limit=limit,
        offset=offset,
    )
    normalized_messages = [dict(msg) for msg in messages]
    return {
        "session_id": session_id,
        "kind": kind,
        "sender": sender,
        "recipient": recipient,
        "limit": limit,
        "offset": offset,
        "count": len(normalized_messages),
        "messages": normalized_messages,
    }


def _serialize_channel_metadata(metadata: object | None) -> str | None:
    if metadata is None:
        return None
    if isinstance(metadata, str):
        return metadata
    if isinstance(metadata, dict):
        return json.dumps(metadata, sort_keys=True)
    raise ApiError("Param 'metadata' must be an object or string", INVALID_PARAMS)


def _normalize_sender(sender: str | None) -> str:
    raw = (sender or "").strip()
    if not raw:
        return "operator:api"
    if ":" in raw:
        return raw
    return f"{raw}:api"


def _emit_session_message(
    conn,
    *,
    session_id: str,
    kind: str,
    sender: str,
    recipient: str | None,
    message_id: str,
    metadata: str | None,
) -> None:
    from agm.queue import publish_event

    session = get_session(conn, session_id)
    project_name = ""
    if session:
        project = get_project(conn, session.get("project_id", ""))
        project_name = project["name"] if project else ""
    parsed_metadata: Any = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            parsed_metadata = metadata
    publish_event(
        "session:message",
        session_id,
        kind,
        project=project_name,
        extra={
            "session_id": session_id,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "message_id": message_id,
            "metadata": parsed_metadata,
        },
    )


def _handle_session_post(conn, params):
    session_id = _require(params, "id")
    content = _require(params, "content").strip()
    if not content:
        raise ApiError("Param 'content' must be non-empty", INVALID_PARAMS)
    session = get_session(conn, session_id)
    if not session:
        raise ApiError(f"Session '{session_id}' not found", NOT_FOUND)

    kind = (_optional(params, "kind") or "context").strip().lower()
    if kind not in VALID_MESSAGE_KINDS:
        raise ApiError(
            f"Invalid message kind '{kind}'. Must be one of: {sorted(VALID_MESSAGE_KINDS)}",
            INVALID_PARAMS,
        )

    sender = _normalize_sender(_optional(params, "sender"))
    recipient = _optional(params, "recipient")
    metadata = _serialize_channel_metadata(params.get("metadata"))
    msg = add_channel_message(
        conn,
        session_id=session_id,
        kind=kind,
        sender=sender,
        content=content,
        recipient=recipient,
        metadata=metadata,
    )
    _emit_session_message(
        conn,
        session_id=session_id,
        kind=kind,
        sender=sender,
        recipient=recipient,
        message_id=msg["id"],
        metadata=metadata,
    )
    return dict(msg)


def _handle_task_steer(conn, params):
    task_id = _require(params, "id")
    content = _require(params, "content").strip()
    if not content:
        raise ApiError("Param 'content' must be non-empty", INVALID_PARAMS)
    task = get_task(conn, task_id)
    if not task:
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    plan = get_plan_request(conn, task["plan_id"])
    if not plan or not plan.get("session_id"):
        raise ApiError(f"Task '{task_id}' has no session channel", INVALID_PARAMS)

    session_id = cast(str, plan["session_id"])
    sender = _normalize_sender(_optional(params, "sender"))
    recipient = _optional(params, "recipient") or default_executor_recipient(task_id)
    live_requested = _optional_bool(params, "live", True)
    if "metadata" in params:
        raw_metadata = params.get("metadata")
    else:
        raw_metadata = {
            "phase": "execution",
            "status": "steer_requested",
            "task_id": task_id,
            "live": live_requested,
        }
    metadata = _serialize_channel_metadata(raw_metadata)
    msg = add_channel_message(
        conn,
        session_id=session_id,
        kind="steer",
        sender=sender,
        content=content,
        recipient=recipient,
        metadata=metadata,
    )
    _emit_session_message(
        conn,
        session_id=session_id,
        kind="steer",
        sender=sender,
        recipient=recipient,
        message_id=msg["id"],
        metadata=metadata,
    )

    response: dict[str, Any] = {
        "task_id": task_id,
        "session_id": session_id,
        "message_id": msg["id"],
        "live_requested": live_requested,
        "live_applied": False,
    }

    record_args: dict[str, Any] = {
        "task_id": task_id,
        "session_id": session_id,
        "message_id": msg["id"],
        "sender": sender,
        "recipient": recipient,
        "content": content,
        "reason": "manual",
        "metadata": metadata,
        "live_requested": live_requested,
        "live_applied": False,
    }

    if not live_requested:
        add_task_steer(conn, **record_args)
        return response
    if task.get("status") != "running":
        response["live_error"] = "task is not running"
        record_args["live_error"] = response["live_error"]
        add_task_steer(conn, **record_args)
        return response
    thread_id = task.get("thread_id")
    active_turn_id = task.get("active_turn_id")
    record_args["thread_id"] = thread_id
    record_args["expected_turn_id"] = active_turn_id
    if not thread_id or not active_turn_id:
        response["live_error"] = "task has no active turn"
        record_args["live_error"] = response["live_error"]
        add_task_steer(conn, **record_args)
        return response

    try:
        steer_response = asyncio.run(
            steer_active_turn(
                thread_id=thread_id,
                active_turn_id=active_turn_id,
                content=content,
            )
        )
        response["live_applied"] = True
        response["turn_id"] = steer_response.get("turnId")
        record_args["live_applied"] = True
        record_args["applied_turn_id"] = response.get("turn_id")
    except Exception as exc:
        response["live_error"] = str(exc)
        record_args["live_error"] = response["live_error"]
    add_task_steer(conn, **record_args)
    return response


def _handle_task_steers(conn, params):
    task_id = _require(params, "id")
    if not get_task(conn, task_id):
        raise ApiError(f"Task '{task_id}' not found", NOT_FOUND)
    limit = _optional_non_negative_int(params, "limit", 100)
    if limit <= 0:
        raise ApiError("Param 'limit' must be > 0", INVALID_PARAMS)
    offset = _optional_non_negative_int(params, "offset", 0)
    rows = list_task_steers(conn, task_id=task_id, limit=limit, offset=offset)
    return {
        "task_id": task_id,
        "limit": limit,
        "offset": offset,
        "count": len(rows),
        "items": [dict(row) for row in rows],
    }


# -- projects --


def _parse_json_column(value: object | None):
    """Parse a JSON-string column, returning None or the parsed object."""
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value  # Return raw string if not valid JSON


def _handle_project_show(conn, params):
    name_or_id = _require(params, "id")
    p = get_project(conn, name_or_id)
    if not p:
        raise ApiError(f"Project '{name_or_id}' not found", NOT_FOUND)
    model_config_json = get_project_model_config(conn, p["id"])
    model_payload = _parse_project_model_config(p, model_config_json)
    result = dict(p)
    result["model_config"] = model_payload
    result["quality_gate"] = _parse_json_column(p.get("quality_gate"))
    result["setup_result"] = _parse_json_column(p.get("setup_result"))
    result["app_server_approval_policy"] = parse_app_server_approval_policy(
        p.get("app_server_approval_policy")
    )
    result["app_server_ask_for_approval"] = parse_app_server_ask_for_approval(
        p.get("app_server_ask_for_approval")
    )
    return result


def _handle_project_list(conn, _params):
    projects = []
    for p in list_projects(conn):
        d = dict(p)
        d["quality_gate"] = _parse_json_column(p.get("quality_gate"))
        d["setup_result"] = _parse_json_column(p.get("setup_result"))
        d["app_server_approval_policy"] = parse_app_server_approval_policy(
            p.get("app_server_approval_policy")
        )
        d["app_server_ask_for_approval"] = parse_app_server_ask_for_approval(
            p.get("app_server_ask_for_approval")
        )
        projects.append(d)
    return projects


def _handle_project_stats(conn, params):
    name_or_id = _require(params, "id")
    p = get_project(conn, name_or_id)
    if not p:
        raise ApiError(f"Project '{name_or_id}' not found", NOT_FOUND)
    plans = list_plan_requests(conn, project_id=p["id"])
    tasks = list_tasks(conn, project_id=p["id"])
    return {
        "project": p["name"],
        "total_plans": len(plans),
        "plan_counts": status_counts(plans),
        "total_tasks": len(tasks),
        "task_counts": status_counts(tasks),
        "plan_model_counts": model_usage_counts(plans, "model"),
        "task_model_counts": model_usage_counts(tasks, "model"),
        "tokens": project_token_totals(plans, tasks),
    }


def _handle_project_setup_status(conn, params):
    from agm.queue import get_job, inspect_queue_jobs

    name_or_id = _require(params, "id")
    project = get_project(conn, name_or_id)
    if not project:
        raise ApiError(f"Project '{name_or_id}' not found", NOT_FOUND)

    setup_job_id = f"setup-{project['id']}"
    queue_rows = inspect_queue_jobs("agm:setup")
    queue_row = next((row for row in queue_rows if row.get("job_id") == setup_job_id), None)

    rq_status = None
    exc_info = None
    job = get_job(setup_job_id)
    if job is not None:
        rq_status = str(job.get_status(refresh=True))
        exc_info = getattr(job, "exc_info", None)

    return {
        "project_id": project["id"],
        "project_name": project["name"],
        "setup_job_id": setup_job_id,
        "rq_status": rq_status,
        "queue": queue_row,
        "setup_result": _parse_json_column(project.get("setup_result")),
        "error": (
            exc_info.splitlines()[-1].strip() if isinstance(exc_info, str) and exc_info else None
        ),
    }


# -- status / infra --


def _handle_status(conn, _params):
    from agm.queue import (
        get_active_external_jobs,
        get_codex_rate_limits_safe,
        get_queue_counts_safe,
    )

    project_summaries = gather_project_summaries(conn)
    queue_info = get_queue_counts_safe()
    rate_limits = get_codex_rate_limits_safe()
    external_jobs = get_active_external_jobs()

    codex_cfg = resolve_model_config("codex", None)
    models = {
        "codex": {"think": codex_cfg["think_model"], "work": codex_cfg["work_model"]},
    }

    return {
        "models": models,
        "projects": project_summaries,
        "queue": queue_info,
        "codex_rate_limits": rate_limits,
        "external_jobs": external_jobs,
    }


def _handle_queue_status(_conn, _params):
    from agm.queue import get_queue_counts

    try:
        return get_queue_counts()
    except Exception as exc:
        raise ApiError(f"Redis unavailable: {exc}", INTERNAL) from exc


def _handle_queue_failed(_conn, params):
    from agm.queue import get_failed_jobs

    queue_name = _optional(params, "queue")
    return get_failed_jobs(queue_name)


def _handle_queue_inspect(_conn, params):
    from agm.queue import inspect_queue_jobs

    queue_name = _optional(params, "queue")
    limit = (
        _optional_non_negative_int(params, "limit", 0) if params.get("limit") is not None else None
    )
    try:
        return inspect_queue_jobs(queue_name, limit=limit)
    except Exception as exc:
        raise ApiError(f"Redis unavailable: {exc}", INTERNAL) from exc


def _handle_doctor(_conn, params):
    from agm.doctor import run_doctor

    fix = _optional_bool(params, "fix")
    return run_doctor(fix=fix)


def _handle_conflicts(conn, params):
    from agm.git_ops import detect_worktree_conflicts

    project_name = _optional(params, "project")
    results = []
    if project_name:
        proj = get_project(conn, project_name)
        if not proj:
            raise ApiError(f"Project '{project_name}' not found", NOT_FOUND)
        projects = [proj]
    else:
        projects = list_projects(conn)
    for proj in projects:
        result = detect_worktree_conflicts(proj["dir"])
        result["project"] = proj["name"]
        results.append(result)
    return results


def _handle_help_status(_conn, _params):
    return get_status_reference()


def _handle_caller_list(_conn, _params):
    all_callers = get_all_callers()
    return {
        "builtin": sorted(BUILTIN_CALLERS),
        "custom": sorted(all_callers - BUILTIN_CALLERS),
        "all": sorted(all_callers),
    }


def _daemon_pid() -> int | None:
    """Read daemon PID file and verify process liveness."""
    from agm.daemon import DEFAULT_PID_PATH

    if not DEFAULT_PID_PATH.exists():
        return None
    try:
        pid = int(DEFAULT_PID_PATH.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _handle_daemon_status(_conn, _params):
    from agm.daemon import DEFAULT_LOG_PATH, DEFAULT_SOCKET_PATH

    pid = _daemon_pid()
    return {
        "running": pid is not None,
        "pid": pid,
        "socket": str(DEFAULT_SOCKET_PATH),
        "socket_exists": DEFAULT_SOCKET_PATH.exists(),
        "log": str(DEFAULT_LOG_PATH),
    }


def _handle_daemon_threads(_conn, params):
    if _daemon_pid() is None:
        raise ApiError("Daemon is not running. Start it with: agm daemon start", INTERNAL)

    payload: dict[str, Any] = {}
    search = (
        _optional(params, "search")
        or _optional(params, "search_term")
        or _optional(params, "searchTerm")
    )
    if search:
        payload["searchTerm"] = search

    if "limit" in params and params.get("limit") is not None:
        limit = _optional_non_negative_int(params, "limit", 0)
        if limit <= 0:
            raise ApiError("Param 'limit' must be > 0", INVALID_PARAMS)
        payload["limit"] = limit

    cursor = _optional(params, "cursor")
    if cursor:
        payload["cursor"] = cursor

    if "archived" in params and params.get("archived") is not None:
        payload["archived"] = bool(params.get("archived"))

    sort_key = _optional(params, "sort_key") or _optional(params, "sortKey")
    if sort_key:
        if sort_key not in {"created_at", "updated_at"}:
            raise ApiError(
                "Param 'sort_key' must be one of: created_at, updated_at",
                INVALID_PARAMS,
            )
        payload["sortKey"] = sort_key

    cwd = _optional(params, "cwd")
    if cwd:
        payload["cwd"] = cwd

    async def _request_threads() -> dict[str, Any]:
        from agm.daemon_client import DaemonClient

        async with DaemonClient() as client:
            return await client.request("thread/list", payload)

    try:
        return normalize_daemon_thread_list(asyncio.run(_request_threads()))
    except Exception as exc:
        raise ApiError(f"Failed to list daemon threads: {exc}", INTERNAL) from exc


def _handle_agents_show(conn, params):
    project_name = _optional(params, "project")
    project: ProjectRow | None = None
    if project_name:
        project = get_project(conn, project_name)
        if not project:
            raise ApiError(f"Project '{project_name}' not found", NOT_FOUND)
    project_dir = str(project["dir"]) if project and project.get("dir") else None
    roles: dict[str, dict[str, str]] = {}
    for role in SUPPORTED_ROLES:
        global_text = get_global_role_text(role)
        project_text = get_project_role_text(project_dir, role) if project_dir else ""
        effective = get_effective_role_config(project_dir, role)
        roles[role] = {
            "global": global_text,
            "project": project_text,
            "effective": effective,
        }
    return {"roles": roles}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_diff_resolve_text(
    project_dir: str,
    base_branch: str,
    branch: str | None,
    merge_commit: str | None,
) -> str:
    """Resolve diff text for a task branch."""
    if branch:
        diff_text = _git_diff(project_dir, f"{base_branch}...{branch}")
        if diff_text:
            return diff_text
    if merge_commit:
        return _git_diff(project_dir, f"{merge_commit}^1..{merge_commit}")
    return ""


def _git_diff(project_dir: str, revspec: str) -> str:
    result = subprocess.run(
        ["git", "diff", revspec],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def _parse_project_model_config(
    project: ProjectRow, model_config_json: str | None
) -> dict[str, Any]:
    """Parse model config for project show output.

    Returns the same shape as CLI ``project show --json`` model_config.
    """
    import os

    parsed: dict[str, str] = {}
    warnings: list[str] = []
    if model_config_json:
        try:
            parsed = json.loads(model_config_json)
        except (json.JSONDecodeError, TypeError):
            warnings.append(f"Malformed model_config JSON: {model_config_json!r}")

    backend = project.get("default_backend") or "codex"
    resolved = resolve_model_config(backend, parsed or None)

    # Env overrides (same logic as cli.py _resolve_model_env_overrides)
    env_lookup = _env_model_lookup_for_backend(backend)
    env_effort_think = os.environ.get("AGM_MODEL_THINK_EFFORT")
    env_effort_work = os.environ.get("AGM_MODEL_WORK_EFFORT")
    env_values: dict[str, str] = {}
    for k, v in env_lookup.items():
        if v is not None:
            env_values[k] = v
    if env_effort_think:
        env_values["think_effort"] = env_effort_think
    if env_effort_work:
        env_values["work_effort"] = env_effort_work

    # Defaults
    think_model = _get_default_model_for_backend_tier(backend, "think")
    work_model = _get_default_model_for_backend_tier(backend, "work")
    defaults = {
        "think_model": think_model,
        "work_model": work_model,
        "think_effort": _get_default_effort_for_backend_tier(backend, "think", think_model),
        "work_effort": _get_default_effort_for_backend_tier(backend, "work", work_model),
    }

    return {
        "project_backend": backend,
        "configured": parsed,
        "active": {
            "think_model": resolved["think_model"],
            "think_effort": resolved["think_effort"],
            "work_model": resolved["work_model"],
            "work_effort": resolved["work_effort"],
        },
        "sources": {
            "env": env_values,
            "default": defaults,
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable] = {
    # tasks
    "task.show": _handle_task_show,
    "task.list": _handle_task_list,
    "task.logs": _handle_task_logs,
    "task.blocks": _handle_task_blocks,
    "task.timeline": _handle_task_timeline,
    "task.trace": _handle_task_trace,
    "task.trace.summary": _handle_task_trace_summary,
    "task.steers": _handle_task_steers,
    "task.diff": _handle_task_diff,
    "task.failures": _handle_task_failures,
    # plans
    "plan.show": _handle_plan_show,
    "plan.list": _handle_plan_list,
    "plan.logs": _handle_plan_logs,
    "plan.timeline": _handle_plan_timeline,
    "plan.questions": _handle_plan_questions,
    "plan.history": _handle_plan_history,
    "plan.stats": _handle_plan_stats,
    "plan.failures": _handle_plan_failures,
    "plan.trace": _handle_plan_trace,
    "plan.trace.summary": _handle_plan_trace_summary,
    "plan.watch": _handle_plan_watch,
    "task.watch": _handle_task_watch,
    # sessions
    "session.show": _handle_session_show,
    "session.list": _handle_session_list,
    "session.messages": _handle_session_messages,
    "session.post": _handle_session_post,
    "task.steer": _handle_task_steer,
    # projects
    "project.show": _handle_project_show,
    "project.list": _handle_project_list,
    "project.stats": _handle_project_stats,
    "project.setup_status": _handle_project_setup_status,
    # status / infra
    "status": _handle_status,
    "queue.status": _handle_queue_status,
    "queue.failed": _handle_queue_failed,
    "queue.inspect": _handle_queue_inspect,
    "doctor": _handle_doctor,
    "conflicts": _handle_conflicts,
    "help_status": _handle_help_status,
    "caller.list": _handle_caller_list,
    "daemon.status": _handle_daemon_status,
    "daemon.threads": _handle_daemon_threads,
    "agents.show": _handle_agents_show,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def dispatch(request: dict, *, db_path: Path | None = None) -> dict:
    """Process a single API request and return the response dict.

    Args:
        request: ``{"method": "...", "params": {...}}``
        db_path: Override the default database path. When ``None``,
            uses ``DEFAULT_DB_PATH`` (``~/.config/agm/agm.db`` or
            ``AGM_DB_PATH`` env var).
    """
    method = request.get("method")
    if not method or not isinstance(method, str):
        return {"ok": False, "error": "Missing or invalid 'method'", "code": INVALID_METHOD}

    handler = METHODS.get(method)
    if not handler:
        return {"ok": False, "error": f"Unknown method: {method}", "code": INVALID_METHOD}

    params = request.get("params") or {}

    try:
        with connect(db_path) if db_path else connect() as conn:
            data = handler(conn, params)
        return {"ok": True, "data": data}
    except ApiError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "code": INTERNAL}


def main() -> None:
    """Read JSON request from stdin, dispatch, write JSON response to stdout."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            response = {"ok": False, "error": "Empty request", "code": INVALID_PARAMS}
        else:
            request = json.loads(raw)
            response = dispatch(request)
    except json.JSONDecodeError as exc:
        response = {"ok": False, "error": f"Invalid JSON: {exc}", "code": INVALID_PARAMS}
    except Exception as exc:
        response = {"ok": False, "error": str(exc), "code": INTERNAL}

    sys.stdout.write(json.dumps(response, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
