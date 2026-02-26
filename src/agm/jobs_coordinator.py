"""Coordinator job for inter-agent overlap/stuck steering."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from itertools import combinations

from agm.db import (
    PlanRow,
    TaskRow,
    add_task_log,
    add_task_steer,
    connect,
    get_plan_request,
    has_recent_task_steer,
    list_tasks,
)
from agm.jobs_common import _emit, _post_channel_message, _resolve_project_name
from agm.steering import default_executor_recipient, steer_active_turn

log = logging.getLogger(__name__)


def _int_env(name: str, default: int, *, min_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        log.warning("Invalid %s=%r; falling back to %d", name, raw, default)
        return default


COORDINATOR_MIN_RUNNING = _int_env("AGM_COORDINATOR_MIN_RUNNING", 3, min_value=1)
COORDINATOR_STUCK_SECONDS = _int_env("AGM_COORDINATOR_STUCK_SECONDS", 600, min_value=1)
COORDINATOR_DEDUPE_SECONDS = _int_env("AGM_COORDINATOR_DEDUPE_SECONDS", 300, min_value=0)


def _parse_task_files(task: TaskRow) -> set[str]:
    files_raw = task.get("files")
    if not files_raw:
        return set()
    try:
        parsed = json.loads(files_raw)
    except (TypeError, json.JSONDecodeError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(path).strip() for path in parsed if isinstance(path, str) and path.strip()}


def _active_turn_age_seconds(task: TaskRow) -> int | None:
    active_since = task.get("active_turn_started_at")
    if not active_since:
        return None
    try:
        started = datetime.strptime(active_since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return int((datetime.now(UTC) - started).total_seconds())


def _manual_coordinator_sender(plan: PlanRow) -> str:
    return f"coordinator:{str(plan['id'])[:8]}"


def _apply_live_steer(task: TaskRow, content: str) -> tuple[bool, bool, str | None, str | None]:
    thread_id = task.get("thread_id")
    active_turn_id = task.get("active_turn_id")
    if not thread_id or not active_turn_id:
        return False, False, "task has no active turn", None
    try:
        response = asyncio.run(
            steer_active_turn(
                thread_id=thread_id,
                active_turn_id=active_turn_id,
                content=content,
            )
        )
        return True, True, None, response.get("turnId")
    except Exception as exc:
        return True, False, str(exc), None


def _record_coordinator_steer(
    conn: sqlite3.Connection,
    *,
    plan: PlanRow,
    task: TaskRow,
    content: str,
    reason: str,
    run_id: str,
) -> bool:
    session_id = plan.get("session_id")
    if not session_id:
        return False
    if has_recent_task_steer(
        conn,
        task_id=task["id"],
        reason=reason,
        content=content,
        lookback_seconds=COORDINATOR_DEDUPE_SECONDS,
        require_live_attempt=True,
    ):
        return False

    recipient = default_executor_recipient(task["id"])
    metadata = {
        "phase": "execution",
        "status": "coordinator_steer",
        "task_id": task["id"],
        "reason": reason,
        "run_id": run_id,
    }
    msg = _post_channel_message(
        conn,
        plan,
        kind="steer",
        sender=_manual_coordinator_sender(plan),
        recipient=recipient,
        content=content,
        metadata=metadata,
    )
    live_requested, live_applied, live_error, applied_turn_id = _apply_live_steer(task, content)
    add_task_steer(
        conn,
        task_id=task["id"],
        session_id=session_id,
        message_id=msg.get("id") if msg else None,
        sender=_manual_coordinator_sender(plan),
        recipient=recipient,
        content=content,
        reason=reason,
        metadata=json.dumps(metadata, sort_keys=True),
        live_requested=live_requested,
        live_applied=live_applied,
        live_error=live_error,
        thread_id=task.get("thread_id"),
        expected_turn_id=task.get("active_turn_id"),
        applied_turn_id=applied_turn_id,
    )
    add_task_log(
        conn,
        task_id=task["id"],
        level="INFO",
        message=f"Coordinator steer ({reason}): {content}",
        source="coordinator",
    )
    return True


def _overlap_messages(running_tasks: list[TaskRow]) -> list[tuple[TaskRow, str]]:
    messages: list[tuple[TaskRow, str]] = []
    file_sets: dict[str, set[str]] = {task["id"]: _parse_task_files(task) for task in running_tasks}
    by_id = {task["id"]: task for task in running_tasks}

    for left_id, right_id in combinations(file_sets.keys(), 2):
        overlap = file_sets[left_id] & file_sets[right_id]
        if not overlap:
            continue
        overlap_sample = ", ".join(sorted(overlap)[:3])
        left = by_id[left_id]
        right = by_id[right_id]
        messages.append(
            (
                left,
                (
                    f"Potential file overlap with task {right['id'][:8]} "
                    f"on: {overlap_sample}. Coordinate changes to avoid merge conflicts."
                ),
            )
        )
        messages.append(
            (
                right,
                (
                    f"Potential file overlap with task {left['id'][:8]} "
                    f"on: {overlap_sample}. Coordinate changes to avoid merge conflicts."
                ),
            )
        )
    return messages


def _stuck_messages(running_tasks: list[TaskRow]) -> list[tuple[TaskRow, str]]:
    messages: list[tuple[TaskRow, str]] = []
    for task in running_tasks:
        age_seconds = _active_turn_age_seconds(task)
        if age_seconds is None or age_seconds < COORDINATOR_STUCK_SECONDS:
            continue
        messages.append(
            (
                task,
                (
                    f"Active turn appears stuck (>{COORDINATOR_STUCK_SECONDS}s). "
                    "Summarize current blocker and either continue with a smaller "
                    "step or request input."
                ),
            )
        )
    return messages


def run_plan_coordinator(plan_id: str) -> str:
    """Run one coordinator pass for a plan and emit steering as needed."""
    with connect() as conn:
        plan = get_plan_request(conn, plan_id)
        if not plan:
            return "skipped:plan_missing"
        running_tasks = list_tasks(conn, plan_id=plan_id, status="running")
        if len(running_tasks) < COORDINATOR_MIN_RUNNING:
            return "skipped:not_enough_running_tasks"

        run_id = uuid.uuid4().hex[:10]
        steers_sent = 0
        for task, content in _overlap_messages(running_tasks):
            if _record_coordinator_steer(
                conn,
                plan=plan,
                task=task,
                content=content,
                reason="file_overlap",
                run_id=run_id,
            ):
                steers_sent += 1
        for task, content in _stuck_messages(running_tasks):
            if _record_coordinator_steer(
                conn,
                plan=plan,
                task=task,
                content=content,
                reason="stuck_turn",
                run_id=run_id,
            ):
                steers_sent += 1

        project_name = _resolve_project_name(conn, plan["project_id"])
        _emit(
            "plan:coordinator",
            plan_id,
            "completed",
            project=project_name,
            plan_id=plan_id,
            extra={
                "plan_id": plan_id,
                "run_id": run_id,
                "running_tasks": len(running_tasks),
                "steers_sent": steers_sent,
            },
        )
        return f"coordinator:sent={steers_sent}"
