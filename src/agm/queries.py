"""Shared query/enrichment logic — used by both CLI and API entry points.

All functions are pure data transformations or DB reads.
No Click imports, no stdout/stderr output.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, TypedDict

from agm.db import (
    PlanQuestionRow,
    PlanRow,
    TaskBlockRow,
    TaskRow,
    get_latest_trace_events_by_entity_ids,
    get_plan_request,
    get_project,
    get_task,
    list_plan_logs,
    list_plan_requests,
    list_plan_watch_events,
    list_projects,
    list_task_logs,
    list_tasks,
)

WATCH_RECENT_EVENT_FETCH_LIMIT = 200


# ---------------------------------------------------------------------------
# Failure diagnostics
# ---------------------------------------------------------------------------


def plan_failure_diagnostic(conn: sqlite3.Connection, plan_id: str) -> tuple[str, str]:
    """Return (source, error_message) for a failed plan."""
    for event in list_plan_watch_events(conn, plan_id, limit=WATCH_RECENT_EVENT_FETCH_LIMIT):
        if str(event.get("level")).upper() == "ERROR":
            return format_plan_failure_source(event), str(event.get("message") or "")

    logs = list_plan_logs(conn, plan_id, level="ERROR")
    if logs:
        return "plan", str(logs[-1].get("message") or "")
    return "plan", ""


def task_failure_diagnostic(conn: sqlite3.Connection, task_id: str) -> tuple[str, str]:
    """Return (source, error_message) for a failed task."""

    def _failure_reason_message(raw: str | None) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        if not isinstance(parsed, dict):
            return text
        message = str(parsed.get("message") or "").strip()
        exc_type = str(parsed.get("exception_type") or "").strip()
        if message:
            return message
        if exc_type:
            return exc_type
        return text

    logs = list_task_logs(conn, task_id, level="ERROR")
    if logs:
        latest = str(logs[-1].get("message") or "")
        if latest.startswith("Task execution failed:"):
            suffix = latest.partition(":")[2].strip()
            if suffix:
                return "task", latest
            task = get_task(conn, task_id)
            if task:
                fallback = _failure_reason_message(task.get("failure_reason"))
                if fallback:
                    return "task", f"Task execution failed: {fallback}"
            return "task", "Task execution failed"
        if latest.strip():
            return "task", latest
    reviews = list_task_logs(conn, task_id, level="REVIEW")
    if reviews:
        return "review", str(reviews[-1].get("message") or "")
    task = get_task(conn, task_id)
    if task:
        fallback = _failure_reason_message(task.get("failure_reason"))
        if fallback:
            return "task", fallback
    return "task", ""


def task_merge_failure_signal(conn: sqlite3.Connection, task_id: str) -> tuple[bool, str]:
    """Return whether the latest task log indicates a merge failure and its message."""
    logs = list_task_logs(conn, task_id)
    if not logs:
        return False, ""
    latest = logs[-1]
    latest_level = str(latest.get("level", "")).upper()
    latest_message = str(latest.get("message") or "")
    if latest_level == "MERGE_CONFLICT":
        return True, latest_message
    if latest_level == "ERROR" and "merge" in latest_message.lower():
        return True, latest_message
    return False, latest_message


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_timeline_rows(rows: list[dict]) -> list[dict]:
    """Normalize status-history timing rows for JSON output."""
    normalized: list[dict] = []
    for row in rows:
        duration_seconds = status_duration_seconds(row)
        entry: dict = {
            "id": row.get("id"),
            "created_at": row.get("created_at"),
            "old_status": row.get("old_status"),
            "new_status": row.get("new_status"),
            "actor": row.get("actor"),
            "next_created_at": row.get("next_created_at"),
            "duration_seconds": duration_seconds,
            "duration_label": "elapsed" if row.get("next_created_at") is None else "for",
            "duration": format_duration_value(duration_seconds),
        }
        entity_type = row.get("entity_type")
        if entity_type:
            entry["entity_type"] = entity_type
        normalized.append(entry)
    return normalized


def normalize_logs(logs: list[dict]) -> list[dict]:
    """Normalize log entries for JSON output."""
    return [
        {
            "id": log["id"],
            "level": log.get("level"),
            "message": log.get("message"),
            "created_at": log.get("created_at"),
        }
        for log in logs
    ]


def normalize_task_blocks(
    blocks: list[TaskBlockRow], unresolved_only: bool
) -> list[dict[str, Any]]:
    """Normalize task block rows for JSON output."""
    normalized = []
    for block in blocks:
        blocked_by = block.get("blocked_by_task_id")
        target = block.get("task_id")
        normalized.append(
            {
                "id": block["id"],
                "task_id": target,
                "blocked_by_task_id": blocked_by,
                "external_factor": block.get("external_factor"),
                "reason": block.get("reason"),
                "resolved": bool(block.get("resolved")),
                "unresolved_only_filter": unresolved_only,
                "created_at": block.get("created_at"),
                "resolved_at": block.get("resolved_at"),
                "task_id_short": watch_short_id(target or ""),
                "blocked_by_short": watch_short_id(blocked_by or ""),
                "is_external": bool(block.get("external_factor")),
            }
        )
    return normalized


def normalize_plan_chain(chain: list[PlanRow], target_plan_id: str) -> list[dict[str, Any]]:
    """Normalize plan history chain entries for JSON output."""
    total = len(chain)
    normalized: list[dict] = []
    for index, plan_row in enumerate(chain, start=1):
        prompt = plan_row.get("prompt") or ""
        normalized.append(
            {
                "id": plan_row["id"],
                "status": plan_row.get("status"),
                "prompt": prompt,
                "prompt_preview": prompt[:50],
                "created_at": plan_row.get("created_at"),
                "updated_at": plan_row.get("updated_at"),
                "is_target": plan_row["id"] == target_plan_id,
                "position": index,
                "total": total,
            }
        )
    return normalized


def normalize_plan_questions(questions: list[PlanQuestionRow]) -> list[dict[str, Any]]:
    """Normalize plan question rows for JSON output."""
    import json as _json

    normalized = []
    for q in questions:
        question = q.get("question") or ""
        raw_options = q.get("options")
        if isinstance(raw_options, str):
            try:
                raw_options = _json.loads(raw_options)
            except (ValueError, TypeError):
                raw_options = None
        normalized.append(
            {
                "id": q["id"],
                "question": question,
                "question_preview": question[:60],
                "header": q.get("header"),
                "options": raw_options,
                "multi_select": bool(q.get("multi_select", False)),
                "answer": q.get("answer"),
                "answered_by": q.get("answered_by"),
                "answered_at": q.get("answered_at"),
                "status": "answered" if q.get("answer") else "pending",
                "created_at": q.get("created_at"),
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Plan list enrichment
# ---------------------------------------------------------------------------


def plan_list_error(conn: sqlite3.Connection, plan_row: Mapping[str, Any]) -> str | None:
    """Return error snippet for a failed plan, or None."""
    if plan_row["status"] != "failed":
        return None
    _, error_message = plan_failure_diagnostic(conn, plan_row["id"])
    return format_plan_failure_error(error_message) or None


def enrich_plan_list_rows(
    conn: sqlite3.Connection,
    plans: Sequence[Mapping[str, Any]],
    runtimes: dict[str, int],
) -> list[dict[str, Any]]:
    """Add error + active_runtime_seconds to plan list rows."""
    return [
        {
            **plan_row,
            "error": plan_list_error(conn, plan_row),
            "active_runtime_seconds": runtimes.get(plan_row["id"]),
        }
        for plan_row in plans
    ]


# ---------------------------------------------------------------------------
# Task list enrichment
# ---------------------------------------------------------------------------


def resolve_project_names_for_tasks(
    conn: sqlite3.Connection, tasks: list[TaskRow]
) -> dict[str, str]:
    """Build a plan_id -> project_name lookup for a list of tasks."""
    plan_ids = {t["plan_id"] for t in tasks}
    result: dict[str, str] = {}
    for pid in plan_ids:
        if pid in result:
            continue
        plan = get_plan_request(conn, pid)
        if not plan:
            continue
        proj = get_project(conn, plan["project_id"])
        result[pid] = proj["name"] if proj else ""
    return result


def task_list_filter_rows(
    tasks: list[TaskRow], *, show_all: bool, status: str | None
) -> list[TaskRow]:
    """Filter task list for default display (hide terminal states)."""
    if show_all or status:
        return tasks
    return [t for t in tasks if not is_effectively_terminal_task(t)]


def is_effectively_terminal_task(task: Mapping[str, Any]) -> bool:
    """Return whether a task should be treated as terminal in UX/status views."""
    status = task.get("status")
    if status in ("completed", "cancelled", "failed"):
        return True
    # do --no-merge intentionally stops at approved and should not appear active.
    return status == "approved" and bool(task.get("skip_merge"))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count rows by status."""
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def task_status_counts(tasks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count tasks by status, preserving display order."""
    order = ["ready", "running", "review", "rejected", "approved", "blocked"]
    counts: dict[str, int] = {}
    for t in tasks:
        s = t.get("status")
        if s:
            counts[s] = counts.get(s, 0) + 1
    return {s: counts[s] for s in order if s in counts}


def model_usage_counts(rows: Sequence[Mapping[str, Any]], model_key: str) -> dict[str, int]:
    """Count rows by model field."""
    counts: dict[str, int] = {}
    for row in rows:
        model = row.get(model_key)
        if not model:
            continue
        counts[model] = counts.get(model, 0) + 1
    return counts


def project_token_totals(plans: list[PlanRow], tasks: list[TaskRow]) -> dict[str, int]:
    """Aggregate project token totals across plans and tasks."""
    plan_in = sum(int_value(p.get("input_tokens")) for p in plans)
    plan_out = sum(int_value(p.get("output_tokens")) for p in plans)
    task_in = sum(int_value(t.get("input_tokens")) for t in tasks)
    task_out = sum(int_value(t.get("output_tokens")) for t in tasks)
    return {
        "plan_input": plan_in,
        "plan_output": plan_out,
        "task_input": task_in,
        "task_output": task_out,
        "total_input": plan_in + task_in,
        "total_output": plan_out + task_out,
        "total_cached": sum(int_value(r.get("cached_input_tokens")) for r in [*plans, *tasks]),
        "total_reasoning": sum(int_value(r.get("reasoning_tokens")) for r in [*plans, *tasks]),
    }


# ---------------------------------------------------------------------------
# Plan stats
# ---------------------------------------------------------------------------


def count_plan_rejections(conn: sqlite3.Connection, tasks: list[TaskRow]) -> int:
    """Count total reviewer rejections across tasks."""
    rejections = 0
    for t in tasks:
        task_logs = list_task_logs(conn, t["id"], level="REVIEW")
        rejections += sum(
            1 for entry in task_logs if "rejected" in (entry.get("message") or "").lower()
        )
    return rejections


def single_model_count(model_name: str | None) -> dict[str, int]:
    """Return one-item model count mapping when model is present."""
    if not model_name:
        return {}
    return {model_name: 1}


def plan_wall_time_seconds(
    p: PlanRow,
    tasks: list[TaskRow],
    timing_rows: list[dict] | None,
) -> int | None:
    """Compute active runtime for plan stats output."""
    if timing_rows is not None:
        active = compute_active_runtime_seconds(timing_rows, entity_type="plan")
        if active is not None:
            return active
    return _plan_wall_time_fallback_seconds(p, tasks)


def _plan_wall_time_fallback_seconds(p: PlanRow, tasks: list[TaskRow]) -> int | None:
    """Compute plan wall time from timestamps when timing rows are unavailable."""
    started = parse_iso_z(p.get("started_at")) or parse_iso_z(p.get("created_at"))
    end_candidates = [p.get("finished_at") or p.get("updated_at")]
    end_candidates += [t.get("finished_at") or t.get("updated_at") for t in tasks]
    latest_end = max(
        (dt for ts in end_candidates if ts and (dt := parse_iso_z(ts)) is not None),
        default=None,
    )
    return int((latest_end - started).total_seconds()) if started and latest_end else None


def build_plan_stats_data(
    conn: sqlite3.Connection,
    plan_id: str,
    p: PlanRow,
    tasks: list[TaskRow],
    timing_rows: list[dict] | None = None,
) -> dict:
    """Assemble plan stats data dict."""
    rejections = count_plan_rejections(conn, tasks)
    task_counts_data = status_counts(tasks)
    token_totals = project_token_totals([p], tasks)
    plan_model_counts = single_model_count(p.get("model"))
    wall_time = plan_wall_time_seconds(p, tasks, timing_rows)

    return {
        "plan_id": plan_id,
        "plan_status": p.get("status"),
        "task_creation_status": p.get("task_creation_status"),
        "total_tasks": len(tasks),
        "task_counts": task_counts_data,
        "rejections": rejections,
        "plan_model_counts": plan_model_counts,
        "task_model_counts": model_usage_counts(tasks, "model"),
        "tokens": token_totals,
        "active_runtime_seconds": wall_time,
    }


# ---------------------------------------------------------------------------
# Status dashboard
# ---------------------------------------------------------------------------


def active_plan_rows(plans: list[PlanRow]) -> list[PlanRow]:
    """Return plans considered active in status dashboard."""
    return [p for p in plans if p["status"] in ("pending", "running", "awaiting_input")]


def active_task_rows(tasks: list[TaskRow]) -> list[TaskRow]:
    """Return tasks considered active in status dashboard."""
    return [t for t in tasks if not is_effectively_terminal_task(t)]


def summarize_active_plans(active_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize active plans for status output."""
    return [
        {
            "id": plan_row["id"][:12],
            "status": plan_row["status"],
            "prompt": (plan_row.get("prompt") or "")[:60],
        }
        for plan_row in active_plans
    ]


def summarize_active_tasks(active_tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize active tasks for status output."""
    return [
        {
            "id": task_row["id"][:12],
            "plan_id": task_row["plan_id"][:12],
            "status": task_row["status"],
            "title": (task_row.get("title") or "")[:60],
            "ordinal": task_row.get("ordinal", 0),
            "bucket": task_row.get("bucket"),
        }
        for task_row in active_tasks
    ]


def recent_project_failures(conn: sqlite3.Connection, plans: list[PlanRow]) -> list[dict]:
    """Return sorted failed-plan diagnostics for status dashboard."""
    failures = []
    for plan_row in plans:
        if plan_row["status"] != "failed":
            continue
        failures.append(
            {
                "plan_id": plan_row["id"],
                "prompt": (plan_row.get("prompt") or "")[:80],
                "status": plan_row["status"],
                "error_snippet": format_plan_failure_error(
                    plan_failure_diagnostic(conn, plan_row["id"])[1]
                ),
                "failed": plan_row.get("updated_at"),
            }
        )
    return sorted(failures, key=lambda row: row["failed"] or "", reverse=True)


def orphan_activity_summary(conn: sqlite3.Connection) -> dict | None:
    """Return status summary for project-less integrations activity."""
    orphan_plans = list_plan_requests(
        conn,
        project_id_is_null=True,
        statuses=["pending", "running", "awaiting_input"],
    )
    orphan_tasks_raw = list_tasks(
        conn,
        project_id_is_null=True,
        statuses=["blocked", "ready", "running", "review", "approved"],
    )
    orphan_tasks = active_task_rows(orphan_tasks_raw)
    if not orphan_plans and not orphan_tasks:
        return None
    return {
        "project": "(integrations)",
        "active_plans": len(orphan_plans),
        "active_tasks": len(orphan_tasks),
        "task_breakdown": task_status_counts(orphan_tasks),
        "plans": summarize_active_plans(orphan_plans),
        "tasks": summarize_active_tasks(orphan_tasks),
        "recent_failures": [],
    }


def gather_project_summaries(conn: sqlite3.Connection) -> list[dict]:
    """Collect per-project plan/task activity summaries for status dashboard."""
    summaries: list[dict] = []
    for project in list_projects(conn):
        plans = list_plan_requests(conn, project_id=project["id"])
        tasks = list_tasks(conn, project_id=project["id"])
        ap = active_plan_rows(plans)
        at = active_task_rows(tasks)
        failures = recent_project_failures(conn, plans)
        summaries.append(
            {
                "project": project["name"],
                "active_plans": len(ap),
                "active_tasks": len(at),
                "task_breakdown": task_status_counts(at),
                "plans": summarize_active_plans(ap),
                "tasks": summarize_active_tasks(at),
                "recent_failures": failures,
            }
        )

    orphan = orphan_activity_summary(conn)
    if orphan:
        summaries.append(orphan)
    return summaries


# ---------------------------------------------------------------------------
# Watch snapshot computation
# ---------------------------------------------------------------------------


def watch_status_counts(
    tasks: Sequence[Mapping[str, Any]],
    ordered_statuses: list[str],
) -> dict[str, int]:
    counts = {status: 0 for status in ordered_statuses}
    for task in tasks:
        status = task.get("status")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts


class WatchTerminalState(TypedDict):
    reached: bool
    reason: str
    message: str | None


class PlanWatchTerminalState(WatchTerminalState):
    is_plan_terminal: bool
    is_all_tasks_terminal: bool


def _earliest_task_phase_timestamp(
    tasks: Sequence[Mapping[str, Any]],
    *,
    statuses: set[str] | None = None,
) -> str | None:
    candidates: list[tuple[datetime, str]] = []
    for task in tasks:
        status = task.get("status")
        if statuses is not None and status not in statuses:
            continue
        ts = task.get("started_at") or task.get("updated_at")
        parsed = parse_iso_z(ts)
        if parsed and isinstance(ts, str):
            candidates.append((parsed, ts))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def derive_plan_watch_phase(
    plan_row: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, str | None]:
    """Derive current plan phase, when it started, and blocking reason."""
    status = str(plan_row.get("status") or "")
    prompt_status = str(plan_row.get("prompt_status") or "")
    task_creation_status = str(plan_row.get("task_creation_status") or "")
    updated_at = plan_row.get("updated_at")
    created_at = plan_row.get("created_at")

    if status == "pending":
        return "queued", created_at, "Waiting for worker pickup."
    if prompt_status == "enriching":
        return "enrichment", updated_at, None
    if prompt_status == "awaiting_input":
        return "awaiting_input", updated_at, "Waiting for answers to plan questions."
    if status == "running":
        if plan_row.get("exploration_thread_id") and not plan_row.get("exploration_context"):
            return "exploration", updated_at, None
        return "planning", updated_at, None
    if status in {"failed", "cancelled"}:
        return status, plan_row.get("finished_at") or updated_at, None

    if task_creation_status == "awaiting_approval":
        return "awaiting_approval", updated_at, "Manual plan approval required."
    if task_creation_status in {"pending", "running"}:
        return "task_creation", updated_at, None

    active_tasks = [task for task in tasks if not is_effectively_terminal_task(task)]
    if active_tasks:
        if any(task.get("status") == "blocked" for task in active_tasks):
            return (
                "execution_blocked",
                _earliest_task_phase_timestamp(active_tasks, statuses={"blocked"}),
                "Tasks are blocked by dependencies or external blockers.",
            )
        if any(task.get("status") == "review" for task in active_tasks):
            return (
                "review",
                _earliest_task_phase_timestamp(active_tasks, statuses={"review"}),
                None,
            )
        if any(task.get("status") == "approved" for task in active_tasks):
            return (
                "merge_pending",
                _earliest_task_phase_timestamp(active_tasks, statuses={"approved"}),
                "Tasks are approved and waiting for merge.",
            )
        return "execution", _earliest_task_phase_timestamp(active_tasks), None

    return "completed", plan_row.get("finished_at") or updated_at, None


def derive_task_watch_phase(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None, str | None]:
    """Derive current task-watch phase, when it started, and blocking reason."""
    if not tasks:
        return "empty", None, "No tasks in scope."

    active = [task for task in tasks if not is_effectively_terminal_task(task)]
    if not active:
        latest_candidates = [
            timestamp
            for task in tasks
            for timestamp in [task.get("finished_at") or task.get("updated_at")]
            if isinstance(timestamp, str)
        ]
        latest = max(latest_candidates, default=None)
        return "completed", latest, None

    if any(task.get("status") == "blocked" for task in active):
        return (
            "blocked",
            _earliest_task_phase_timestamp(active, statuses={"blocked"}),
            "Tasks are blocked by dependencies or external blockers.",
        )
    if any(task.get("status") == "running" for task in active):
        return (
            "execution",
            _earliest_task_phase_timestamp(active, statuses={"running"}),
            None,
        )
    if any(task.get("status") == "review" for task in active):
        return (
            "review",
            _earliest_task_phase_timestamp(active, statuses={"review"}),
            None,
        )
    if any(task.get("status") == "approved" for task in active):
        return (
            "merge_pending",
            _earliest_task_phase_timestamp(active, statuses={"approved"}),
            "Tasks are approved and waiting for merge.",
        )
    if any(task.get("status") == "ready" for task in active):
        return (
            "ready",
            _earliest_task_phase_timestamp(active, statuses={"ready"}),
            "Tasks are ready but not yet picked up by a worker.",
        )
    return "active", _earliest_task_phase_timestamp(active), None


def plan_watch_terminal_state(plan_row: PlanRow, tasks: list[TaskRow]) -> PlanWatchTerminalState:
    status = plan_row.get("status")
    if status not in {"finalized", "failed", "cancelled"}:
        return {
            "reached": False,
            "reason": "in_progress",
            "message": None,
            "is_plan_terminal": False,
            "is_all_tasks_terminal": False,
        }

    tcs = plan_row.get("task_creation_status")
    if tcs and tcs not in ("completed", "failed"):
        return {
            "reached": False,
            "reason": "task_creation_in_progress",
            "message": None,
            "is_plan_terminal": True,
            "is_all_tasks_terminal": False,
        }

    if not tasks:
        return {
            "reached": True,
            "reason": "plan_terminal_no_tasks",
            "message": f"Plan {status}.",
            "is_plan_terminal": True,
            "is_all_tasks_terminal": True,
        }

    active_tasks = [t for t in tasks if not is_effectively_terminal_task(t)]
    if active_tasks:
        return {
            "reached": False,
            "reason": "active_tasks",
            "message": None,
            "is_plan_terminal": True,
            "is_all_tasks_terminal": False,
        }

    return {
        "reached": True,
        "reason": "all_tasks_terminal",
        "message": "All tasks reached terminal state.",
        "is_plan_terminal": True,
        "is_all_tasks_terminal": True,
    }


def plan_watch_elapsed(
    plan_row: PlanRow,
    tasks: list[TaskRow],
    timing_rows: list[dict] | None = None,
) -> str:
    """Active runtime: sum of time spent in 'running' status only.

    Uses status_history timing rows when available to exclude idle states
    (pending, awaiting_input, etc.). Falls back to wall-clock subtraction
    when timing_rows is not provided.
    """
    if timing_rows is not None:
        runtime = plan_watch_elapsed_from_timing_rows(plan_row, timing_rows)
        return runtime or ""
    start_ts = plan_row.get("started_at") or plan_row.get("created_at")
    return format_elapsed(start_ts, end_ts=plan_watch_elapsed_fallback_end(plan_row, tasks))


def plan_watch_elapsed_from_timing_rows(plan_row: PlanRow, timing_rows: list[dict]) -> str | None:
    seconds = compute_active_runtime_seconds(timing_rows, entity_type="plan")
    if seconds is None:
        return None
    return format_duration_seconds(seconds)


def plan_watch_elapsed_fallback_end(plan_row: PlanRow, tasks: list[TaskRow]) -> str | None:
    all_terminal = (
        all(is_effectively_terminal_task(task) for task in tasks)
        if tasks
        else plan_row.get("status") in ("finalized", "failed", "cancelled")
    )
    if not all_terminal:
        return None
    timestamps = [plan_row.get("finished_at") or plan_row.get("updated_at")]
    timestamps.extend(task.get("finished_at") or task.get("updated_at") for task in tasks)
    return max((ts for ts in timestamps if ts), default=None)


def aggregate_plan_tokens(plan_row: PlanRow, tasks: list[TaskRow]) -> dict[str, int]:
    """Aggregate token usage across plan + tasks."""
    plan_in = int_value(plan_row.get("input_tokens"))
    plan_out = int_value(plan_row.get("output_tokens"))
    plan_cached = int_value(plan_row.get("cached_input_tokens"))
    plan_reasoning = int_value(plan_row.get("reasoning_tokens"))
    task_in = task_out = task_cached = task_reasoning = 0
    for t in tasks:
        task_in += int_value(t.get("input_tokens"))
        task_out += int_value(t.get("output_tokens"))
        task_cached += int_value(t.get("cached_input_tokens"))
        task_reasoning += int_value(t.get("reasoning_tokens"))
    return {
        "plan_input": plan_in,
        "plan_output": plan_out,
        "task_input": task_in,
        "task_output": task_out,
        "total_input": plan_in + task_in,
        "total_output": plan_out + task_out,
        "total_cached": plan_cached + task_cached,
        "total_reasoning": plan_reasoning + task_reasoning,
    }


def build_plan_watch_snapshot(
    plan_row: PlanRow,
    tasks: list[TaskRow],
    recent_events: list[dict],
    *,
    terminal_state: PlanWatchTerminalState,
    timing_rows: list[dict] | None = None,
) -> dict[str, object]:
    status_counts_map = watch_status_counts(tasks, WATCH_TASK_STATUS_ORDER)
    active_tasks = [t for t in tasks if not is_effectively_terminal_task(t)]
    tokens = aggregate_plan_tokens(plan_row, tasks)
    phase, phase_since, blocking_reason = derive_plan_watch_phase(plan_row, tasks)
    return {
        "schema": "plan_watch_snapshot_v1",
        "scope": {
            "type": "plan",
            "plan_id": plan_row["id"],
            "plan_id_short": watch_short_id(plan_row["id"]),
            "session_id": plan_row.get("session_id"),
            "title": plan_row.get("prompt") or "",
            "title_truncated": watch_truncate(plan_row.get("prompt") or "", 60),
        },
        "plan": {
            "status": plan_row.get("status"),
            "backend": plan_row.get("backend") or "codex",
            "task_creation_status": plan_row.get("task_creation_status"),
            "created_at": plan_row.get("created_at"),
            "updated_at": plan_row.get("updated_at"),
        },
        "runtime": plan_watch_elapsed(plan_row, tasks, timing_rows),
        "counts": {
            "tasks_total": len(tasks),
            "tasks_active": len(active_tasks),
            "status_summary": status_counts_map,
        },
        "tokens": tokens,
        "phase": phase,
        "phase_since": phase_since,
        "blocking_reason": blocking_reason,
        "recent_events": normalize_plan_watch_events(recent_events),
        "terminal_state": terminal_state,
    }


def task_watch_terminal_state(tasks: Sequence[Mapping[str, Any]]) -> WatchTerminalState:
    if not tasks:
        return {
            "reached": False,
            "reason": "no_tasks",
            "message": None,
        }

    active_tasks = [t for t in tasks if not is_effectively_terminal_task(t)]
    if active_tasks:
        return {
            "reached": False,
            "reason": "active_tasks",
            "message": None,
        }

    return {
        "reached": True,
        "reason": "all_tasks_terminal",
        "message": "All tasks reached terminal state.",
    }


def build_task_watch_snapshot(
    *,
    scope: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    visible_tasks: Sequence[Mapping[str, Any]],
    recent_events: list[dict[str, Any]],
    blocker_counts: dict[str, int],
    plan_backends: dict[str, str],
    watch_elapsed: str,
    terminal_state: WatchTerminalState,
    thread_status_by_task: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    summary = {
        status: count
        for status, count in watch_status_counts(tasks, WATCH_SUMMARY_STATUSES).items()
    }
    visible_summary = {
        status: count
        for status, count in watch_status_counts(visible_tasks, WATCH_SUMMARY_STATUSES).items()
    }
    task_tok_in = sum(int_value(t.get("input_tokens")) for t in tasks)
    task_tok_out = sum(int_value(t.get("output_tokens")) for t in tasks)
    phase, phase_since, blocking_reason = derive_task_watch_phase(tasks)
    normalized_recent_events = normalize_task_watch_events(recent_events)
    resolved_thread_status = (
        {str(task_id): dict(status) for task_id, status in thread_status_by_task.items()}
        if thread_status_by_task
        else {}
    )
    thread_status_summary: dict[str, int] = {}
    for status in resolved_thread_status.values():
        status_name = str(status.get("new_status") or status.get("status") or "unknown")
        thread_status_summary[status_name] = thread_status_summary.get(status_name, 0) + 1

    return {
        "schema": "task_watch_snapshot_v1",
        "scope": scope,
        "watching": watch_elapsed,
        "counts": {
            "tasks_total": len(tasks),
            "tasks_visible": len(visible_tasks),
            "tasks_active": len([t for t in tasks if not is_effectively_terminal_task(t)]),
            "status_summary": summary,
            "status_summary_visible": visible_summary,
        },
        "tokens": {
            "task_input": task_tok_in,
            "task_output": task_tok_out,
            "total_input": task_tok_in,
            "total_output": task_tok_out,
        },
        "phase": phase,
        "phase_since": phase_since,
        "blocking_reason": blocking_reason,
        "tasks": [
            {
                "id": t["id"],
                "id_short": watch_short_id(t["id"]),
                "status": t.get("status"),
                "plan_id": t.get("plan_id"),
                "backend": plan_backends.get(t.get("plan_id", ""), "codex"),
                "title": t.get("title") or "",
                "title_truncated": watch_truncate(t.get("title") or "", 80),
                "priority": t.get("priority"),
                "bucket": t.get("bucket"),
                "blocked_count": blocker_counts.get(t["id"], 0),
                "rejection_count": int_value(t.get("rejection_count", 0)),
                "updated_at": t.get("updated_at"),
                "thread_status": resolved_thread_status.get(str(t["id"])),
            }
            for t in visible_tasks
        ],
        "thread_status_summary": thread_status_summary,
        "recent_events": normalized_recent_events,
        "terminal_state": terminal_state,
    }


# -- Relocated from display.py --

# Statuses where agents are actively working (not idle/waiting).
PLAN_ACTIVE_STATUSES = frozenset({"running"})
TASK_ACTIVE_STATUSES = frozenset({"running", "review"})

PLAN_WATCH_RECENT_EVENTS_ROWS = 50
PLAN_WATCH_EVENT_MAX_LEN = 80
WATCH_FALLBACK_POLL_INTERVAL = 30.0  # seconds between fallback DB checks (event-driven primary)
WATCH_TASK_STATUS_ORDER = [
    "running",
    "review",
    "rejected",
    "approved",
    "ready",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]
WATCH_SUMMARY_STATUSES = ["ready", "running", "review", "rejected", "approved", "blocked"]
WATCH_RECENT_EVENT_LINES = 50
WATCH_RECENT_EVENT_MSG_LEN = 70
PLAN_FAILURE_PROMPT_SNIPPET_LEN = 50
PLAN_FAILURE_ERROR_SNIPPET_LEN = 80


def parse_iso_z(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def format_elapsed(ts: str | None, end_ts: str | None = None) -> str:
    """Format elapsed time from ts to end_ts (or now if end_ts is None)."""
    dt = parse_iso_z(ts)
    if not dt:
        return ""
    if end_ts:
        end_dt = parse_iso_z(end_ts)
        if not end_dt:
            end_dt = datetime.now(UTC)
    else:
        end_dt = datetime.now(UTC)
    seconds = int((end_dt - dt).total_seconds())
    if seconds < 0:
        return ""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    m = minutes % 60
    return f"{hours}h {m}m" if m else f"{hours}h"


def elapsed_seconds(ts: str | None) -> int | None:
    dt = parse_iso_z(ts)
    if not dt:
        return None
    seconds = int((datetime.now(UTC) - dt).total_seconds())
    if seconds < 0:
        return 0
    return seconds


def format_duration_seconds(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    rem_mins = minutes % 60
    return f"{hours}h {rem_mins}m" if rem_mins else f"{hours}h"


def format_duration_value(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    return format_duration_seconds(max(int(seconds), 0))


def format_compact_datetime(ts: str | None) -> str:
    """Format a timestamp as compact local datetime: 'Feb 12 14:05' or 'Jan 03 2025 09:30'."""
    dt = parse_iso_z(ts)
    if not dt:
        return "-"
    local_dt = dt.astimezone()
    now = datetime.now(UTC).astimezone()
    if local_dt.year == now.year:
        return local_dt.strftime("%b %d %H:%M")
    return local_dt.strftime("%b %d %Y %H:%M")


def status_duration_seconds(row: dict) -> int | None:
    duration_seconds = row.get("duration_seconds")
    if duration_seconds is not None:
        try:
            return max(int(duration_seconds), 0)
        except (TypeError, ValueError):
            return None
    return elapsed_seconds(row.get("created_at"))


def compute_active_runtime_seconds(
    timing_rows: list[dict],
    entity_type: str = "plan",
) -> int | None:
    """Sum only the time spent in active statuses from status_history rows.

    For plans: 'running' is active (enrichment/planning/task creation).
    For tasks: 'running' and 'review' are active (execution/review).
    Idle states (pending, awaiting_input, finalized, etc.) are excluded.

    If the entity is currently in an active status and the last row has
    no duration_seconds (still in progress), adds elapsed time since that
    row's created_at to capture the live interval.
    """
    active_statuses = TASK_ACTIVE_STATUSES if entity_type == "task" else PLAN_ACTIVE_STATUSES
    total = 0
    has_any = False
    for row in timing_rows:
        status = row.get("new_status", "")
        if status not in active_statuses:
            continue
        seconds = status_duration_seconds(row)
        if seconds is not None:
            total += seconds
            has_any = True
    # If currently in an active status and last row is open-ended,
    # status_duration_seconds already falls back to elapsed_seconds.
    return total if has_any else None


def format_watch_hms(ts: str | None) -> str:
    dt = parse_iso_z(ts)
    if not dt:
        return "--:--:--"
    return dt.astimezone(UTC).strftime("%H:%M:%S")


def watch_format_hhmmss(ts: str | None) -> str:
    dt = parse_iso_z(ts)
    if not dt:
        return "--:--:--"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%H:%M:%S")


def effective_task_priority(priority: str | None) -> str:
    if priority is None:
        return "medium"
    normalized = str(priority).strip().lower()
    if normalized == "":
        return "medium"
    return normalized


def enrich_task_list_row(
    task: TaskRow, project_name: str, active_runtime_seconds: int | None
) -> dict:
    """Return a task row with task-list JSON enrichment fields."""
    enriched = dict(task)
    enriched["project_name"] = project_name
    enriched["active_runtime_seconds"] = active_runtime_seconds
    return enriched


def enrich_task_list_rows(
    tasks: list[TaskRow],
    project_names: dict[str, str],
    runtimes: dict[str, int],
) -> list[dict]:
    """Add task-list JSON enrichment fields to every task row."""
    return [
        enrich_task_list_row(
            task,
            project_name=project_names.get(task["plan_id"], ""),
            active_runtime_seconds=runtimes.get(task["id"]),
        )
        for task in tasks
    ]


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def watch_short_id(full_id: str) -> str:
    return full_id


def watch_truncate(text: str, max_len: int = 40) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def format_plan_failure_error(message: str | None) -> str:
    """Format a compact error snippet for failure listings."""
    normalized = str(message or "").replace("\n", " ").strip()
    return watch_truncate(normalized, PLAN_FAILURE_ERROR_SNIPPET_LEN)


def format_plan_failure_source(event: dict) -> str:
    """Return a compact source label for a failure event."""
    source = event.get("source") or "plan"
    if source == "task":
        return watch_short_id(str(event.get("task_id") or "task"))
    return str(source)


def format_plan_failure_prompt(text: str | None) -> str:
    """Format a compact prompt preview for failure listings."""
    return watch_truncate(text or "", PLAN_FAILURE_PROMPT_SNIPPET_LEN)


def format_plan_watch_event_line(event: dict) -> str:
    source = event.get("source") or "plan"
    if source == "task":
        source = watch_short_id(event.get("task_id") or "task")
    message = watch_truncate(str(event.get("message") or ""), PLAN_WATCH_EVENT_MAX_LEN)
    return f"  {format_watch_hms(event.get('timestamp'))}  [{source}]  {message}"


def normalize_plan_watch_events(events: list[dict]) -> list[dict]:
    """Normalize plan watch events for machine-readable snapshots."""
    normalized: list[dict] = []
    for event in events[:PLAN_WATCH_RECENT_EVENTS_ROWS]:
        message = str(event.get("message") or "")
        normalized.append(
            {
                "timestamp": event.get("timestamp"),
                "source": event.get("source") or "plan",
                "task_id": event.get("task_id"),
                "task_id_short": watch_short_id(str(event.get("task_id") or "")),
                "level": event.get("level"),
                "message": message,
                "message_truncated": watch_truncate(
                    message.replace("\n", " ").strip(),
                    PLAN_WATCH_EVENT_MAX_LEN,
                ),
                "line": format_plan_watch_event_line(event),
            }
        )
    return normalized


def normalize_task_watch_events(events: list[dict]) -> list[dict]:
    """Normalize task watch events for machine-readable snapshots."""
    normalized: list[dict] = []
    for event in events[:WATCH_RECENT_EVENT_LINES]:
        message = str(event.get("message") or "")
        normalized_message = message.replace("\n", " ").strip()
        task_id = event.get("task_id")
        timestamp = event.get("created_at")
        source = event.get("source") or "task"
        normalized.append(
            {
                "timestamp": timestamp,
                "source": source,
                "task_id": task_id,
                "task_id_short": watch_short_id(str(task_id or "")),
                "level": event.get("level"),
                "message": message,
                "message_truncated": watch_truncate(
                    normalized_message,
                    WATCH_RECENT_EVENT_MSG_LEN,
                ),
                "line": f"  {watch_format_hhmmss(timestamp)}  "
                f"[{source}]  "
                f"{watch_truncate(normalized_message, WATCH_RECENT_EVENT_MSG_LEN)}",
            }
        )
    return normalized


def latest_task_thread_statuses(
    conn: sqlite3.Connection,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return latest structured thread-status trace event for each task."""
    task_ids = [str(task.get("id") or "").strip() for task in tasks]
    latest_rows = get_latest_trace_events_by_entity_ids(
        conn,
        entity_type="task",
        entity_ids=task_ids,
        event_type="threadStatus",
    )
    result: dict[str, dict[str, Any]] = {}
    for task_id, row in latest_rows.items():
        raw_data = row.get("data")
        data: Mapping[str, Any] = raw_data if isinstance(raw_data, Mapping) else {}
        new_status = data.get("new_status")
        old_status = data.get("old_status")
        thread_id = data.get("thread_id")
        kind = row.get("status") or "changed"
        result[task_id] = {
            "kind": str(kind),
            "status": str(new_status or data.get("status") or "unknown"),
            "new_status": str(new_status) if new_status is not None else None,
            "old_status": str(old_status) if old_status is not None else None,
            "thread_id": str(thread_id) if thread_id is not None else None,
            "recorded_at": row.get("created_at"),
        }
    return result


def extract_plan_event_rowids(events: list[dict]) -> tuple[int, int]:
    """Extract max plan_rowid and max task_rowid from plan watch events."""
    max_plan = 0
    max_task = 0
    for e in events:
        rowid = e.get("order_rowid", 0)
        if e.get("source") == "task":
            max_task = max(max_task, rowid)
        else:
            max_plan = max(max_plan, rowid)
    return max_plan, max_task


def extract_task_event_max_rowid(events: list[dict]) -> int:
    """Extract max source_rowid from task watch events."""
    max_rowid = 0
    for e in events:
        rowid = e.get("source_rowid", 0)
        max_rowid = max(max_rowid, rowid)
    return max_rowid


def entity_elapsed(row: dict) -> str:
    """Compute elapsed time for a plan or task.

    Uses started_at -> finished_at for terminal entities (actual work time).
    Uses started_at -> now for active entities (time since work started).
    Falls back to created_at/updated_at when dedicated columns are absent.
    """
    terminal = {"finalized", "failed", "completed", "cancelled"}
    status = row.get("status") or ""
    started = row.get("started_at") or row.get("created_at")
    if not started:
        return ""
    if status in terminal:
        ended = row.get("finished_at") or row.get("updated_at")
        return format_elapsed(started, end_ts=ended)
    return format_elapsed(started)


def entity_elapsed_seconds(row: dict) -> int | None:
    """Compute elapsed seconds for a plan or task (for JSON output)."""
    terminal = {"finalized", "failed", "completed", "cancelled"}
    status = row.get("status") or ""
    started = row.get("started_at") or row.get("created_at")
    start_dt = parse_iso_z(started)
    if not start_dt:
        return None
    if status in terminal:
        ended = row.get("finished_at") or row.get("updated_at")
        end_dt = parse_iso_z(ended)
        if not end_dt:
            return None
    else:
        end_dt = datetime.now(UTC)
    seconds = int((end_dt - start_dt).total_seconds())
    return max(seconds, 0)
