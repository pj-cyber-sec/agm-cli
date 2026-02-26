"""Task creation + refresh functions.

_trigger_task_creation, run_task_creation, _insert_tasks_from_output,
bucket/file verification helpers, auto-trigger sorting, run_task_refresh.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from typing import Any

from agm.db import (
    PlanRow,
    ProjectRow,
    TaskRow,
    add_plan_log,
    cancel_tasks_batch,
    connect,
    create_tasks_batch,
    get_plan_request,
    get_project,
    get_task,
    list_tasks,
    resolve_stale_blockers,
    set_plan_model,
    update_plan_task_creation_status,
    update_plan_tokens,
)
from agm.jobs_common import (
    _TASK_PRIORITY_RANK,
    PlanDBHandler,
    _apply_project_app_server_ask_for_approval,
    _codex_client,
    _codex_turn,
    _effective_task_priority,
    _emit,
    _get_plan_backend,
    _load_project_app_server_approval_policy,
    _load_project_model_config,
    _make_server_request_handler,
    _merge_developer_instructions,
    _normalize_output_task_priority,
    _resolve_project_model_config,
    _resolve_project_name,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-trigger sorting
# ---------------------------------------------------------------------------


def _sort_task_ids_for_auto_trigger(conn: sqlite3.Connection, task_ids: list[str]) -> list[str]:
    """Deterministically sort task IDs for auto-trigger by effective priority."""
    if not task_ids:
        return []

    unique_ids = list(dict.fromkeys(task_ids))
    found_tasks = []
    missing_ids = []
    for tid in unique_ids:
        task = get_task(conn, tid)
        if task:
            found_tasks.append(task)
        else:
            missing_ids.append(tid)

    found_tasks.sort(
        key=lambda task: (
            _TASK_PRIORITY_RANK[_effective_task_priority(task.get("priority"))],
            task.get("ordinal") if isinstance(task.get("ordinal"), int) else 2**31 - 1,
            task.get("created_at") or "",
            task["id"],
        )
    )
    return [task["id"] for task in found_tasks] + missing_ids


def _auto_trigger_execution_for_ready_tasks(conn: sqlite3.Connection, task_ids: list[str]) -> None:
    from agm.jobs_merge import _trigger_task_execution  # deferred

    for tid in _sort_task_ids_for_auto_trigger(conn, task_ids):
        _trigger_task_execution(conn, tid)


# ---------------------------------------------------------------------------
# Trigger task creation
# ---------------------------------------------------------------------------


def _trigger_task_creation(conn: sqlite3.Connection, plan_id: str) -> None:
    """Auto-trigger task creation after plan finalization.

    If the project's plan_approval mode is 'manual', sets
    task_creation_status to 'awaiting_approval' instead of enqueuing.
    The user must then run ``plan approve`` to proceed.

    Sets task_creation_status to 'pending' and enqueues.
    Logs but does NOT re-raise on failure (plan is already finalized).
    """
    try:
        from agm.db import get_project_plan_approval

        plan = get_plan_request(conn, plan_id)
        project_name = _resolve_project_name(conn, plan["project_id"]) if plan else ""
        if plan:
            approval_mode = get_project_plan_approval(conn, plan["project_id"])
            if approval_mode == "manual":
                update_plan_task_creation_status(conn, plan_id, "awaiting_approval")
                _emit("plan:task_creation", plan_id, "awaiting_approval", project=project_name)
                add_plan_log(
                    conn,
                    plan_id=plan_id,
                    level="INFO",
                    message="Plan awaiting approval — run `agm plan approve` to proceed.",
                )
                log.info("Plan %s awaiting approval (plan_approval=manual)", plan_id)
                return

        from agm.queue import enqueue_task_creation

        update_plan_task_creation_status(conn, plan_id, "pending")
        _emit("plan:task_creation", plan_id, "pending", project=project_name)
        enqueue_task_creation(plan_id)
        log.info("Task creation enqueued for plan %s", plan_id)
    except Exception as e:
        log.error("Failed to enqueue task creation for plan %s: %s", plan_id, e)
        update_plan_task_creation_status(conn, plan_id, "failed")
        plan = get_plan_request(conn, plan_id)
        if plan:
            pname = _resolve_project_name(conn, plan["project_id"])
            _emit("plan:task_creation", plan_id, "failed", project=pname)


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------


def run_task_creation(plan_id: str) -> str:
    """Create tasks from a finalized plan via the task agent.

    Called by rq worker. Logs go to plan_logs since this is plan-level work.
    """
    with connect() as conn:
        plan = get_plan_request(conn, plan_id)
        if not plan:
            log.warning("Plan %s not found (deleted?), skipping", plan_id)
            return "skipped:entity_missing"

        if plan["status"] != "finalized":
            raise ValueError(f"Plan {plan_id} is '{plan['status']}', not 'finalized'")

        if not plan.get("plan"):
            raise ValueError(f"Plan {plan_id} has no plan text")

        db_handler = PlanDBHandler(conn, plan_id, source="task_creation")
        db_handler.setLevel(logging.DEBUG)
        log.addHandler(db_handler)
        prev_log_level = log.level
        if log.level > logging.DEBUG or log.level == logging.NOTSET:
            log.setLevel(logging.DEBUG)

        project_name = _resolve_project_name(conn, plan["project_id"])
        update_plan_task_creation_status(conn, plan_id, "running")
        _emit("plan:task_creation", plan_id, "running", project=project_name)
        backend = _get_plan_backend(conn, plan)
        set_plan_model(
            conn,
            plan_id,
            _resolve_project_model_config(conn, plan["project_id"], backend)["think_model"],
        )
        log.info("Task creation started for plan %s (backend=%s)", plan_id, backend)

        try:
            result = _run_task_creation_codex(conn, plan)
            update_plan_task_creation_status(conn, plan_id, "completed")
            _emit("plan:task_creation", plan_id, "completed", project=project_name)
            log.info("Task creation completed for plan %s", plan_id)
            return result
        except Exception:
            log.exception("Task creation failed for plan %s", plan_id)
            update_plan_task_creation_status(conn, plan_id, "failed")
            _emit("plan:task_creation", plan_id, "failed", project=project_name)
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def _run_task_creation_codex(conn: sqlite3.Connection, plan: PlanRow) -> str:
    """Run task creation via the Codex backend."""
    return asyncio.run(_run_task_creation_codex_async(conn, plan))


def _extract_task_files(task: dict | TaskRow) -> set[str]:
    files_value = task.get("files")
    if files_value is None:
        return set()

    parsed_files: list[Any] | tuple[Any, ...] | set[Any]
    if isinstance(files_value, str):
        stripped = files_value.strip()
        if not stripped:
            return set()
        try:
            loaded = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return {stripped}
        if not isinstance(loaded, list):
            return set()
        parsed_files = loaded
    elif isinstance(files_value, (list, tuple, set)):
        parsed_files = files_value
    else:
        return set()

    normalized: set[str] = set()
    for file_path in parsed_files:
        if not isinstance(file_path, str):
            continue
        cleaned = file_path.strip()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def _normalize_bucket_value(bucket: Any) -> str | None:
    if not isinstance(bucket, str):
        return None
    normalized = bucket.strip()
    return normalized or None


def _bucket_assignment_findings(left: dict, right: dict) -> list[str]:
    overlap = sorted(left["files"] & right["files"])
    left_bucket = left["bucket"]
    right_bucket = right["bucket"]
    same_bucket = bool(left_bucket and left_bucket == right_bucket)

    findings: list[str] = []
    if overlap and not same_bucket:
        findings.append(
            "WARNING: Bucket/file overlap mismatch between "
            f"task ordinal={left['ordinal']} ({left_bucket or 'none'}) and "
            f"task ordinal={right['ordinal']} ({right_bucket or 'none'}); "
            f"shared files: {', '.join(overlap)}"
        )
    if same_bucket and not overlap:
        findings.append(
            "INFO: Bucket grouping without shared files between "
            f"task ordinal={left['ordinal']} and task ordinal={right['ordinal']} "
            f"for bucket '{left_bucket}'"
        )
    return findings


def _verify_bucket_assignments(tasks: list[dict]) -> list[str]:
    findings: list[str] = []
    enriched = [
        {
            "ordinal": task.get("ordinal"),
            "title": task.get("title"),
            "bucket": _normalize_bucket_value(task.get("bucket")),
            "files": _extract_task_files(task),
        }
        for task in tasks
    ]

    for i, left in enumerate(enriched):
        for right in enriched[i + 1 :]:
            findings.extend(_bucket_assignment_findings(left, right))
    return findings


def _warn_cross_plan_file_overlaps(
    current_plan_id: str, new_tasks: list[dict], existing_active: list[TaskRow]
) -> None:
    existing_other_plans = [
        task for task in existing_active if task.get("plan_id") != current_plan_id
    ]
    if not existing_other_plans:
        return

    for new_task in new_tasks:
        new_files = _extract_task_files(new_task)
        if not new_files:
            continue

        for existing_task in existing_other_plans:
            existing_files = _extract_task_files(existing_task)
            overlap = sorted(new_files & existing_files)
            if not overlap:
                continue
            log.warning(
                "Cross-plan file overlap: new task ordinal=%s shares files [%s] "
                "with active task %s from plan %s",
                new_task.get("ordinal"),
                ", ".join(overlap),
                existing_task["id"],
                existing_task["plan_id"],
            )


async def _run_task_creation_codex_async(conn: sqlite3.Connection, plan: PlanRow) -> str:
    """Async implementation of codex task creation.

    Fresh thread (not resuming plan thread). Plan JSON is self-contained;
    task agent needs its own config/instructions.
    """
    from agm.backends import (
        build_task_creation_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    thread_config = get_runtime_thread_config(
        plan["backend"], "task_creation", _load_project_model_config(conn, plan["project_id"])
    )
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    _merge_developer_instructions(thread_config, project["dir"], "task_agent")

    # Load existing non-terminal tasks for the project
    existing_tasks = list_tasks(conn, project_id=plan["project_id"], status=None)
    active_statuses = {"blocked", "ready", "running", "review", "approved", "failed"}
    existing_active = [t for t in existing_tasks if t["status"] in active_statuses]

    # Build prompt with plan JSON + existing tasks
    existing_summary = ""
    if existing_active:
        task_lines = []
        for t in existing_active:
            bucket_part = f', bucket="{t["bucket"]}"' if t.get("bucket") else ""
            priority_part = f", priority={_effective_task_priority(t.get('priority'))}"
            task_lines.append(
                f'  - id={t["id"]}, title="{t["title"]}", status={t["status"]}'
                f"{priority_part}{bucket_part}"
            )
        existing_summary = (
            "\n\nExisting tasks for this project (check for overlap before creating new ones):\n"
            + "\n".join(task_lines)
        )

    assert plan["plan"] is not None
    prompt = build_task_creation_prompt(plan["plan"], existing_summary)
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="task_creation",
        plan_id=plan["id"],
        project=project["name"],
        conn=conn,
    )

    approval_policy = _load_project_app_server_approval_policy(conn, plan["project_id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))

        def on_thread_ready(thread_id: str) -> None:
            log.info("Started task creation thread %s for plan %s", thread_id, plan["id"])

        pmc = _load_project_model_config(conn, plan["project_id"])
        turn_config = get_runtime_turn_config(plan["backend"], "task_creation", pmc)

        _, output_text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
            on_thread_ready=on_thread_ready,
            trace_context=trace_ctx,
        )
        # Task creation tokens attributed to the plan
        update_plan_tokens(conn, plan["id"], **tokens)

        if not output_text:
            raise RuntimeError(f"No output produced by task agent for plan {plan['id']}")

        with contextlib.suppress(json.JSONDecodeError, TypeError, AttributeError):
            parsed = json.loads(output_text)
            candidate_tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
            if isinstance(candidate_tasks, list):
                _warn_cross_plan_file_overlaps(plan["id"], candidate_tasks, existing_active)

        task_count = _insert_tasks_from_output(conn, plan["id"], output_text)
        log.info("Created %d tasks for plan %s", task_count, plan["id"])
        return output_text


def _resolve_stale_and_trigger(conn: sqlite3.Connection, ready_task_ids: list[str]) -> None:
    """Resolve stale blockers and auto-trigger all ready tasks."""
    promoted = resolve_stale_blockers(conn, record_history=True)
    if promoted:
        log.info("Resolved stale blockers, promoted %d tasks to ready", len(promoted))
    ready_task_ids.extend(promoted)
    _auto_trigger_execution_for_ready_tasks(conn, ready_task_ids)


def _validate_task_references(conn: sqlite3.Connection, tasks: list[dict]) -> list[dict]:
    """Validate existing task refs and normalize each task into insertion format."""
    validated = []
    for t in tasks:
        valid_refs = []
        for ref_id in t.get("blocked_by_existing", []):
            if get_task(conn, ref_id):
                valid_refs.append(ref_id)
            else:
                log.warning(
                    "Task ordinal=%d references nonexistent task %s — skipping",
                    t.get("ordinal", -1),
                    ref_id,
                )
        validated.append(
            {
                "ordinal": t["ordinal"],
                "title": t["title"],
                "description": t["description"],
                "files": json.dumps(t.get("files", [])),
                "status": t.get("status", "blocked"),
                "blocked_by": t.get("blocked_by", []),
                "blocked_by_existing": valid_refs,
                "external_blockers": t.get("external_blockers", []),
                "bucket": t.get("bucket"),
                "priority": _normalize_output_task_priority(t.get("priority")),
            }
        )
    return validated


def _parse_task_creation_output(output_text: str) -> dict:
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Task agent output is not valid JSON: {e}") from e


def _process_task_cancellations(conn: sqlite3.Connection, data: dict) -> list[str]:
    cancellations = data.get("cancel_tasks", [])
    if cancellations:
        cancel_result = cancel_tasks_batch(conn, cancellations, record_history=True)
        log.info("Cancelled %d tasks", cancel_result["cancelled"])
    return cancellations


def _ensure_ready_task_present(tasks: list[dict]) -> None:
    if any(task.get("status") == "ready" for task in tasks):
        return
    log.warning("Task agent produced no 'ready' tasks — forcing first task to ready")
    tasks[0]["status"] = "ready"


def _log_bucket_assignment_findings(findings: list[str]) -> None:
    for finding in findings:
        if finding.startswith("INFO: "):
            log.info("%s", finding.removeprefix("INFO: "))
            continue
        if finding.startswith("WARNING: "):
            log.warning("%s", finding.removeprefix("WARNING: "))
            continue
        log.warning("%s", finding)


def _emit_task_created_events(
    conn: sqlite3.Connection, plan_id: str, ordinal_to_id: dict[int, str]
) -> None:
    if not ordinal_to_id:
        return
    plan = get_plan_request(conn, plan_id)
    project_name = _resolve_project_name(conn, plan["project_id"]) if plan else ""
    for task_id in ordinal_to_id.values():
        _emit("task:created", task_id, "created", project=project_name, plan_id=plan_id)
    log.info("Created %d task(s) for plan %s", len(ordinal_to_id), plan_id)


def _collect_ready_task_ids(
    validated_tasks: list[dict], ordinal_to_id: dict[int, str]
) -> list[str]:
    return [
        ordinal_to_id[task["ordinal"]]
        for task in validated_tasks
        if task.get("status") == "ready" and task["ordinal"] in ordinal_to_id
    ]


def _insert_tasks_from_output(
    conn: sqlite3.Connection, plan_id: str | None, output_text: str
) -> int:
    """Parse task agent output and insert tasks + process cancellations.

    plan_id may be None for refresh (reconcile-only) runs.
    Returns the number of tasks created.
    """
    data = _parse_task_creation_output(output_text)
    cancellations = _process_task_cancellations(conn, data)
    tasks = data.get("tasks", [])
    ready_task_ids: list[str] = []

    if not tasks:
        if cancellations:
            _resolve_stale_and_trigger(conn, ready_task_ids)
            return 0
        raise RuntimeError(f"Task agent produced no tasks and no cancellations for plan {plan_id}")

    if not plan_id:
        log.warning("Refresh produced new tasks but no plan_id — skipping creation")
        _resolve_stale_and_trigger(conn, ready_task_ids)
        return 0

    _ensure_ready_task_present(tasks)
    validated_tasks = _validate_task_references(conn, tasks)
    _log_bucket_assignment_findings(_verify_bucket_assignments(validated_tasks))

    ordinal_to_id = create_tasks_batch(conn, plan_id, validated_tasks)
    _emit_task_created_events(conn, plan_id, ordinal_to_id)
    ready_task_ids.extend(_collect_ready_task_ids(validated_tasks, ordinal_to_id))
    _resolve_stale_and_trigger(conn, ready_task_ids)
    return len(validated_tasks)


def on_task_creation_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when a task creation job fails."""
    plan_id = job.args[0] if job.args else None
    if plan_id:
        with connect() as conn:
            plan = get_plan_request(conn, plan_id)
            if not plan:
                log.warning("Task creation %s failed callback skipped: plan not found", plan_id)
                return
            update_plan_task_creation_status(conn, plan_id, "failed")
            pname = _resolve_project_name(conn, plan["project_id"])
            _emit("plan:task_creation", plan_id, "failed", project=pname)
            add_plan_log(
                conn,
                plan_id=plan_id,
                level="ERROR",
                message=f"Task creation failed: {exc_value}",
            )
        log.warning("Task creation for plan %s failed: %s", plan_id, exc_value)


# ---------------------------------------------------------------------------
# Task refresh
# ---------------------------------------------------------------------------


def run_task_refresh(project_id: str, prompt: str | None = None, backend: str = "codex") -> str:
    """Invoke the task agent to review and clean up tasks for a project.

    Called by rq worker. Unlike task creation, this is not tied to a plan.
    The agent sees all project tasks and can cancel stale/duplicate ones.
    """
    with connect() as conn:
        project = get_project(conn, project_id)
        if not project:
            log.warning("Project %s not found (deleted?), skipping", project_id)
            return "skipped:entity_missing"

        if backend != "codex":
            raise ValueError(f"Unknown backend: {backend}")

        log.info("Task refresh started for project %s (backend=%s)", project_id, backend)
        result = _run_task_refresh_codex(conn, project, prompt)
        return result


def _run_task_refresh_codex(
    conn: sqlite3.Connection, project: ProjectRow, prompt: str | None
) -> str:
    """Run task refresh via the Codex backend."""
    return asyncio.run(_run_task_refresh_codex_async(conn, project, prompt))


async def _run_task_refresh_codex_async(
    conn: sqlite3.Connection, project: ProjectRow, prompt: str | None
) -> str:
    """Async implementation of codex task refresh."""
    from agm.backends import (
        REFRESH_PROMPT_SUFFIX,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )

    thread_config = get_runtime_thread_config(
        "codex", "task_creation", _load_project_model_config(conn, project["id"])
    )
    _apply_project_app_server_ask_for_approval(conn, project["id"], thread_config)
    _merge_developer_instructions(thread_config, project["dir"], "task_agent")

    # Load all non-terminal tasks for the project
    all_tasks = list_tasks(conn, project_id=project["id"])
    active_tasks = [t for t in all_tasks if t["status"] not in ("completed", "failed")]

    if not active_tasks:
        log.info("No active tasks for project %s — nothing to refresh", project["id"])
        return "No active tasks to refresh"

    task_lines = []
    for t in active_tasks:
        effective_priority = _effective_task_priority(t.get("priority"))
        task_lines.append(
            f'  - id={t["id"]}, title="{t["title"]}", status={t["status"]}, '
            f"priority={effective_priority}, plan_id={t['plan_id']}"
        )
    task_summary = "\n".join(task_lines)

    user_prompt = prompt or "Review and clean up the task landscape."
    full_prompt = (
        f"{user_prompt}\n\nCurrent tasks for this project:\n{task_summary}{REFRESH_PROMPT_SUFFIX}"
    )
    approval_policy = _load_project_app_server_approval_policy(conn, project["id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))

        def on_thread_ready(thread_id: str) -> None:
            log.info(
                "Started refresh thread %s for project %s",
                thread_id,
                project["id"],
            )

        turn_config = get_runtime_turn_config(
            "codex", "task_creation", _load_project_model_config(conn, project["id"])
        )

        _, output_text, _tokens = await _codex_turn(
            client,
            prompt=full_prompt,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
            on_thread_ready=on_thread_ready,
        )
        if not output_text:
            raise RuntimeError(f"No output produced by task agent for project {project['id']}")

        task_count = _insert_tasks_from_output(conn, None, output_text)
        log.info(
            "Refresh complete for project %s: %d new tasks",
            project["id"],
            task_count,
        )
        return output_text
