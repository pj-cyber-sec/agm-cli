"""Task execution functions.

run_task_execution, on_task_execution_failure, codex backend,
predecessor context, failed sibling context, merge conflict prompt section.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any

from agm.db import (
    TaskRow,
    add_task_log,
    connect,
    get_plan_request,
    get_project_quality_gate,
    get_task,
    list_tasks,
    resolve_blockers_for_terminal_task,
    set_plan_model,
    set_task_active_turn_id,
    set_task_failure_reason,
    set_task_model,
    set_task_thread_id,
    set_task_worker,
    update_task_status,
    update_task_tokens,
)
from agm.jobs_common import (
    _COMMIT_NUDGE,
    CHANNEL_TOOL_INSTRUCTIONS,
    CHANNEL_TOOL_SPEC,
    MAX_COMMIT_NUDGES,
    TaskDBHandler,
    TurnEventContext,
    _apply_project_app_server_ask_for_approval,
    _codex_client,
    _codex_turn,
    _compact_codex_thread,
    _emit,
    _fallback_thread_config_for_resolved_model,
    _get_latest_review,
    _get_plan_backend,
    _get_project_dir_for_task,
    _get_project_id_for_task,
    _get_rejection_count,
    _has_uncommitted_changes,
    _load_project_app_server_approval_policy,
    _load_project_model_config,
    _make_server_request_handler,
    _maybe_complete_session,
    _merge_developer_instructions,
    _post_channel_message,
    _resolve_project_model_config,
    _resolve_project_name,
)
from agm.queue import publish_task_model_escalation
from agm.tool_config import get_servers_for_job_type, load_tool_config

log = logging.getLogger(__name__)


def _get_predecessor_context(conn: sqlite3.Connection, task_id: str) -> tuple[str, list[str]]:
    """Build context summary from completed predecessor tasks.

    Looks up resolved internal blockers for this task, finds the completed
    predecessor tasks, and returns a summary of what each one implemented
    (title + description excerpt). This helps the executor understand naming
    conventions and patterns established by earlier tasks in the chain.

    Returns ``(context_text, injected_predecessor_ids)``.
    """
    from agm.db import list_task_blocks

    blocks = list_task_blocks(conn, task_id)
    candidate_ids: list[str] = [
        b["blocked_by_task_id"]
        for b in blocks
        if b["blocked_by_task_id"] is not None and b.get("resolved")
    ]
    if not candidate_ids:
        return "", []

    summaries = []
    injected_ids: list[str] = []
    for pred_id in candidate_ids:
        pred = get_task(conn, pred_id)
        if not pred or pred["status"] != "completed":
            continue
        injected_ids.append(pred_id)
        desc_excerpt = (pred.get("description") or "")[:200]
        summaries.append(f"  - {pred['title']}: {desc_excerpt}")

    if not summaries:
        return "", []

    return (
        "\nPredecessor tasks (already merged to main — "
        "read their code to follow established patterns):\n" + "\n".join(summaries),
        injected_ids,
    )


def _get_failed_sibling_context(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Build context about sibling tasks from the same plan that failed.

    Warns the executor not to reference or depend on features from these tasks,
    since they were never implemented.
    """
    rows = conn.execute(
        "SELECT title FROM tasks WHERE plan_id = ? AND status = 'failed' AND id != ?",
        (task["plan_id"], task["id"]),
    ).fetchall()
    if not rows:
        return ""
    titles = [r["title"] for r in rows]
    lines = "\n".join(f"  - {t}" for t in titles)
    return (
        "\nWARNING — these sibling tasks from the same plan FAILED and were "
        "NOT implemented. Do not reference, import, or document features "
        "from these tasks — their code does not exist:\n" + lines
    )


def _get_channel_context(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Build context from channel messages posted by other agents in this plan.

    Pulls recent context/steer/dm/question channel entries, prioritizes
    directly-targeted steer/context, and emits a deterministic, bounded prompt
    section.
    """
    plan = get_plan_request(conn, task["plan_id"])
    if not plan:
        return ""
    session_id = plan.get("session_id")
    if not session_id:
        return ""

    rows = conn.execute(
        "SELECT * FROM channel_messages WHERE session_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 500",
        (session_id,),
    ).fetchall()
    messages = [dict(row) for row in reversed(rows)]
    if not messages:
        return ""

    task_short = task["id"][:8]
    own_sender = f"executor:{task_short}"
    own_role = "executor"
    target_kinds = {"context", "steer", "question", "dm"}
    kind_rank = {"steer": 0, "question": 1, "dm": 2, "context": 3, "broadcast": 4}

    def _recipient_matches(recipient: str | None) -> bool:
        if not recipient:
            return True
        if recipient == own_sender:
            return True
        if ":" in recipient:
            role, ident = recipient.split(":", 1)
            return role == own_role and (ident == task_short or ident == "")
        return recipient == own_role

    ranked: list[tuple[tuple[int, int, int], int, Any]] = []
    for idx, msg in enumerate(messages):
        kind = msg.get("kind", "context")
        if kind not in target_kinds:
            continue
        # Skip own executor messages
        if msg["sender"] == own_sender:
            continue
        # Skip status broadcasts from executors (noise for prompt context)
        if msg["sender"].startswith("executor") and msg["content"].startswith("Executing:"):
            continue
        # Skip system messages about this specific task
        if msg["sender"].startswith("system") and task_short in msg["content"]:
            continue
        recipient = msg.get("recipient")
        if kind in {"dm", "question", "steer"} and recipient and not _recipient_matches(recipient):
            continue
        recipient_rank = 2
        if recipient == own_sender:
            recipient_rank = 0
        elif recipient and recipient.startswith(f"{own_role}:"):
            recipient_rank = 1
        ranked.append(((kind_rank.get(kind, 4), recipient_rank, -idx), idx, msg))

    if not ranked:
        return ""

    ranked.sort(key=lambda row: row[0])
    selected = ranked[:12]
    selected.sort(key=lambda row: row[1])  # Preserve chronological readability.

    lines = []
    for _, _, msg in selected:
        sender = msg["sender"]
        if ":" in sender:
            role, sid = sender.split(":", 1)
            label = f"{role}-{sid[:4]}"
        else:
            label = sender
        kind = msg.get("kind", "context")
        prefix = f"{kind}|" if kind != "context" else ""
        lines.append(f"  - [{prefix}{label}] {msg['content']}")

    return "\nContext from other agents working on this plan:\n" + "\n".join(lines)


def _summarize_execution_context(summary: str) -> str:
    """Normalize task-turn output into a compact channel context line."""
    compact = " ".join((summary or "").split())
    if not compact:
        return ""
    if len(compact) > 420:
        compact = compact[:417].rstrip() + "..."
    return compact


def _publish_execution_context_summary(
    conn: sqlite3.Connection, task: TaskRow, summary: str
) -> None:
    """Publish distilled executor output so sibling tasks can consume it."""
    compact = _summarize_execution_context(summary)
    if not compact:
        return
    plan = get_plan_request(conn, task["plan_id"])
    if not plan or not plan.get("session_id"):
        return
    task_short = task["id"][:8]
    title = task.get("title") or task_short
    _post_channel_message(
        conn,
        plan,
        kind="context",
        sender=f"executor:{task_short}",
        content=f"[{title}] {compact}",
        metadata={
            "phase": "execution",
            "status": "context_published",
            "task_id": task["id"],
            "source": "turn_completed",
        },
    )


def _build_merge_conflict_prompt_section(diff: str) -> str:
    """Build a prompt section for merge conflict re-execution."""
    return (
        "\n\n## MERGE CONFLICT RE-EXECUTION\n\n"
        "Your previous implementation was correct but couldn't merge to main due to "
        "conflicts with other changes that landed since you started. You are now working "
        "on a fresh worktree based on the current main.\n\n"
        "Here is your previous diff for reference — re-implement the same changes, "
        "adapted to the current state of the codebase:\n\n"
        f"```diff\n{diff}\n```"
    )


def _get_quality_gate_fail_context(conn: sqlite3.Connection, task_id: str) -> str | None:
    """Fetch failure details from the most recent QUALITY_GATE_FAIL task_log."""
    from agm.db import list_task_logs

    logs = list_task_logs(conn, task_id, level="QUALITY_GATE_FAIL")
    if not logs:
        return None
    message = logs[-1]["message"]
    marker = "\n\n"
    idx = message.find(marker)
    if idx >= 0:
        return message[idx + len(marker) :]
    return message


def _build_quality_gate_fail_prompt_section(context: str) -> str:
    """Build a prompt section for quality gate failure re-execution."""
    return (
        "\n\n## QUALITY GATE FAILURE RE-EXECUTION\n\n"
        "Your previous implementation passed review but failed quality gate checks "
        "at merge time. You are now working on a fresh worktree based on the current "
        "base branch.\n\n"
        "Here are the failures and your previous diff for reference — fix the quality "
        "gate issues while re-implementing your changes:\n\n"
        f"{context}"
    )


def _parse_quality_gate_config(quality_gate_json: str | None) -> dict:
    """Parse project quality gate JSON or return default config."""
    from agm.backends import DEFAULT_QUALITY_GATE

    if not quality_gate_json:
        return DEFAULT_QUALITY_GATE
    try:
        config = json.loads(quality_gate_json)
    except (json.JSONDecodeError, TypeError):
        return DEFAULT_QUALITY_GATE
    return config if isinstance(config, dict) else DEFAULT_QUALITY_GATE


def _format_quality_gate_cmd(value: object) -> str:
    """Format command field from quality gate config."""
    return " ".join(value) if isinstance(value, list) else str(value)


def _append_quality_gate_auto_fix_lines(lines: list[str], auto_fix: list[dict]) -> None:
    """Append auto-fix command lines to quality gate prompt output."""
    lines.append("\nAuto-fix commands (run these first, in order):")
    for cmd_spec in auto_fix:
        cmd_name = cmd_spec.get("name", "unnamed")
        cmd_str = _format_quality_gate_cmd(cmd_spec.get("cmd", []))
        lines.append(f"  - {cmd_name}: {cmd_str}")


def _append_quality_gate_check_lines(lines: list[str], checks: list[dict]) -> None:
    """Append strict check lines to quality gate prompt output."""
    lines.append("\nStrict checks (must pass):")
    for check_spec in checks:
        check_name = check_spec.get("name", "unnamed")
        cmd_str = _format_quality_gate_cmd(check_spec.get("cmd", []))
        timeout = check_spec.get("timeout", 60)
        lines.append(f"  - {check_name}: {cmd_str} (timeout: {timeout}s)")


def _build_execution_failure_reason(task_id: str, exc_value: object | None, *, path: str) -> str:
    """Build a stable JSON payload for execution failure metadata."""
    message = ""
    if exc_value is not None:
        message = str(exc_value)
        if not message and isinstance(exc_value, BaseException):
            message = exc_value.__class__.__name__
    return json.dumps(
        {
            "context": {"path": path},
            "exception_type": exc_value.__class__.__name__
            if isinstance(exc_value, BaseException)
            else type(exc_value).__name__,
            "message": message,
            "source": "execution",
            "task_id": task_id,
        },
        sort_keys=True,
    )


def _build_quality_gate_prompt(conn: sqlite3.Connection, project_id: str) -> str:
    """Build a quality-gate prompt section from project config or defaults.

    Reads the project's quality gate config from DB and formats auto-fix
    and strict check commands into a readable prompt section.
    Falls back to DEFAULT_QUALITY_GATE (empty) if no custom config.
    When the gate is empty, returns a discovery prompt telling the executor
    to inspect the repo and run appropriate tooling.
    """
    quality_gate_json = get_project_quality_gate(conn, project_id)
    config = _parse_quality_gate_config(quality_gate_json)

    auto_fix = config.get("auto_fix", [])
    checks = config.get("checks", [])

    # Empty gate — return discovery prompt
    if not auto_fix and not checks:
        return (
            "\n\nQUALITY GATE — No quality gate configured for this project.\n"
            "Inspect the repo to discover its tooling (look for config files like "
            "package.json, pyproject.toml, Makefile, Cargo.toml, go.mod, etc.) and "
            "run the appropriate linter, formatter, and test commands before committing. "
            "If unsure, at minimum run the test suite."
        )

    lines = ["\n\nQUALITY GATE — Project command sequence:"]
    if auto_fix:
        _append_quality_gate_auto_fix_lines(lines, auto_fix)
    if checks:
        _append_quality_gate_check_lines(lines, checks)

    return "\n".join(lines)


def _resolve_execution_result(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Run task execution via the selected backend."""
    return _run_task_execution_codex(conn, task)


def _finalize_skip_review_execution_success(
    conn: sqlite3.Connection, task: TaskRow, task_id: str, project_name: str
) -> None:
    """Persist auto-approved task and trigger merge when configured."""
    update_task_status(conn, task_id, "approved", record_history=True)
    _emit(
        "task:status",
        task_id,
        "approved",
        project=project_name,
        plan_id=task["plan_id"],
    )
    log.info("Task %s auto-approved (skip_review)", task_id)
    if task.get("skip_merge"):
        _maybe_complete_session(conn, task["plan_id"])
        return
    # Deferred import to avoid circular dependency
    from agm.jobs_merge import _trigger_task_merge

    _trigger_task_merge(task_id)


def _finalize_review_execution_success(
    conn: sqlite3.Connection, task: TaskRow, task_id: str, project_name: str
) -> None:
    """Move task to review and trigger reviewer workflow."""
    update_task_status(conn, task_id, "review", record_history=True)
    _emit(
        "task:status",
        task_id,
        "review",
        project=project_name,
        plan_id=task["plan_id"],
    )
    log.info("Task %s submitted for review", task_id)
    # Deferred import to avoid circular dependency
    from agm.jobs_merge import _trigger_task_review

    _trigger_task_review(task_id)


def _finalize_execution_success(
    conn: sqlite3.Connection,
    task: TaskRow,
    task_id: str,
    project_name: str,
) -> None:
    """Persist successful execution and route to review or merge."""
    plan = get_plan_request(conn, task["plan_id"])
    if plan:
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"executor:{task_id[:8]}",
            content=f"Task completed: {task.get('title', task_id)}",
            metadata={
                "phase": "execution",
                "status": "completed",
                "task_id": task_id,
                "task_title": task.get("title"),
            },
        )
    if task.get("skip_review"):
        _finalize_skip_review_execution_success(conn, task, task_id, project_name)
        return
    _finalize_review_execution_success(conn, task, task_id, project_name)


def _finalize_execution_failure(
    conn: sqlite3.Connection,
    task: TaskRow,
    task_id: str,
    project_name: str,
    failure_reason: str | None = None,
) -> None:
    """Persist failure state and emit cancellation events."""
    plan = get_plan_request(conn, task["plan_id"])
    if plan:
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"executor:{task_id[:8]}",
            content=f"Execution failed: {task.get('title', task_id)}",
            metadata={
                "phase": "execution",
                "status": "failed",
                "task_id": task_id,
                "failure_reason": failure_reason,
            },
        )
    if failure_reason is not None:
        set_task_failure_reason(conn, task_id, failure_reason)
    update_task_status(conn, task_id, "failed", record_history=True)
    if failure_reason is not None:
        set_task_failure_reason(conn, task_id, failure_reason)
    _emit("task:status", task_id, "failed", project=project_name, plan_id=task["plan_id"])
    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        conn, task_id, record_history=True
    )
    if promoted:
        log.info("Failed task %s unblocked %d task(s)", task_id, len(promoted))
    for cid in cascade_cancelled:
        _emit(
            "task:status",
            cid,
            "cancelled",
            project=project_name,
            plan_id=task["plan_id"],
        )
    _maybe_complete_session(conn, task["plan_id"])


def run_task_execution(task_id: str) -> str:
    """Execute a task via the appropriate backend.

    Called by rq worker. Updates task status as it progresses:
    running -> review (on success) or running -> failed (on error).
    """
    with connect() as conn:
        task = get_task(conn, task_id)
        if not task:
            log.warning("Task %s not found (deleted?), skipping", task_id)
            return "skipped:entity_missing"

        if task["status"] != "running":
            raise ValueError(f"Task {task_id} is '{task['status']}', not 'running'")

        if not task.get("worktree"):
            raise ValueError(f"Task {task_id} has no worktree")

        db_handler = TaskDBHandler(conn, task_id, source="executor")
        db_handler.setLevel(logging.DEBUG)
        # Attach at the shared "agm" namespace so logs emitted from helper
        # modules (e.g. jobs_common fallback/timeout warnings) are captured too.
        task_log_logger = logging.getLogger("agm")
        task_log_logger.addHandler(db_handler)
        prev_log_level = task_log_logger.level
        if task_log_logger.level > logging.DEBUG or task_log_logger.level == logging.NOTSET:
            task_log_logger.setLevel(logging.DEBUG)

        pid = os.getpid()
        set_task_worker(conn, task_id, pid=pid)
        backend = _get_plan_backend(conn, task)
        project_id = _get_project_id_for_task(conn, task)
        project_name = _resolve_project_name(conn, project_id) if project_id else ""
        model_config = _resolve_project_model_config(conn, project_id, backend)
        set_task_model(conn, task_id, model_config["work_model"])
        # Backfill plan model for quick-mode plans (no planner sets it).
        plan_model = conn.execute(
            "SELECT model FROM plans WHERE id = ?", (task["plan_id"],)
        ).fetchone()
        if plan_model and not plan_model[0]:
            set_plan_model(conn, task["plan_id"], model_config["work_model"])
        plan = get_plan_request(conn, task["plan_id"])
        log.info(
            "Worker pid=%d started task %s (backend=%s)",
            pid,
            task_id,
            backend,
        )
        if plan:
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"executor:{task_id[:8]}",
                content=f"Executing: {task.get('title', task_id)}",
                metadata={
                    "phase": "execution",
                    "status": "running",
                    "task_id": task_id,
                    "task_title": task.get("title"),
                    "worktree": task.get("worktree"),
                },
            )
            _maybe_enqueue_plan_coordinator(conn, plan["id"])

        try:
            result = _resolve_execution_result(conn, task)
            _finalize_execution_success(conn, task, task_id, project_name)
            return result
        except Exception as exc:
            log.error("Task %s failed", task_id)
            failure_reason = _build_execution_failure_reason(
                task_id,
                exc,
                path="direct",
            )
            _finalize_execution_failure(
                conn,
                task,
                task_id,
                project_name,
                failure_reason=failure_reason,
            )
            raise
        finally:
            task_log_logger.removeHandler(db_handler)
            task_log_logger.setLevel(prev_log_level)


def _maybe_enqueue_plan_coordinator(conn: sqlite3.Connection, plan_id: str) -> None:
    """Best-effort coordinator enqueue when enough tasks are running."""
    from agm.jobs_coordinator import COORDINATOR_MIN_RUNNING

    min_running = COORDINATOR_MIN_RUNNING
    running = list_tasks(conn, plan_id=plan_id, status="running")
    if len(running) < max(1, min_running):
        return
    try:
        from agm.queue import enqueue_plan_coordinator

        enqueue_plan_coordinator(plan_id)
    except Exception:
        log.debug("Coordinator enqueue skipped for plan %s", plan_id, exc_info=True)


def _run_task_execution_codex(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Run task execution via the Codex backend."""
    return asyncio.run(_run_task_execution_codex_async(conn, task))


def _append_prompt_if_present(parts: list[str], section: str) -> None:
    """Append a section to prompt parts if non-empty."""
    if section:
        parts.append(section)


def _append_files_prompt_section(parts: list[str], files_json: str | None) -> None:
    """Append 'Files to work on' section when task file hints are available."""
    if not files_json:
        return
    try:
        files = json.loads(files_json)
    except (json.JSONDecodeError, TypeError):
        return
    if files:
        parts.append("\nFiles to work on:\n" + "\n".join(f"  - {f}" for f in files))


def _build_executor_rejection_prompt(
    conn: sqlite3.Connection,
    task: TaskRow,
    quality_gate_section: str,
    executor_prompt_suffix: str,
) -> str:
    """Build execution prompt for rejection retry on an existing thread."""
    review_text = _get_latest_review(conn, task["id"])
    channel_context = _get_channel_context(conn, task)
    if review_text:
        parts = [
            f"REVIEWER FEEDBACK — your changes were rejected.\n\n"
            f"Findings:\n{review_text}\n\n"
            f"Instructions:\n"
            f"1. Make targeted fixes ONLY to the specific code the reviewer "
            f"flagged. Do not rewrite or restructure working code.\n"
            f"2. Run the project's test suite after fixing to verify "
            f"nothing regressed.\n"
            f"3. Commit the fix with a clear message describing what you changed.",
        ]
        if channel_context:
            parts.append(channel_context)
        parts.append(quality_gate_section)
        parts.append(executor_prompt_suffix)
        return "\n".join(p for p in parts if p)
    parts = [
        "Your previous changes were rejected. "
        "Review your implementation for issues, fix them, run tests, "
        "and commit.",
    ]
    if channel_context:
        parts.append(channel_context)
    parts.append(quality_gate_section)
    parts.append(executor_prompt_suffix)
    return "\n".join(p for p in parts if p)


def _build_executor_fresh_prompt(
    conn: sqlite3.Connection,
    task: TaskRow,
    quality_gate_section: str,
    executor_prompt_suffix: str,
) -> str:
    """Build execution prompt for first-run task execution."""
    parts = [f"Task: {task['title']}\n\n{task['description']}"]
    _append_files_prompt_section(parts, task.get("files"))
    predecessor_text, predecessor_ids = _get_predecessor_context(conn, task["id"])
    _append_prompt_if_present(parts, predecessor_text)
    if predecessor_ids:
        add_task_log(
            conn,
            task_id=task["id"],
            level="PREDECESSOR_CONTEXT",
            message=json.dumps(
                {
                    "predecessor_task_ids": predecessor_ids,
                    "count": len(predecessor_ids),
                }
            ),
            source="executor",
        )
    _append_prompt_if_present(parts, _get_failed_sibling_context(conn, task))
    _append_prompt_if_present(parts, _get_channel_context(conn, task))

    # Merge conflict re-execution context (deferred import)
    from agm.jobs_merge import _get_merge_conflict_context

    merge_conflict_diff = _get_merge_conflict_context(conn, task["id"])
    if merge_conflict_diff:
        parts.append(_build_merge_conflict_prompt_section(merge_conflict_diff))

    qg_fail_context = _get_quality_gate_fail_context(conn, task["id"])
    if qg_fail_context:
        parts.append(_build_quality_gate_fail_prompt_section(qg_fail_context))

    _append_prompt_if_present(parts, quality_gate_section)
    parts.append(executor_prompt_suffix)
    return "\n".join(parts)


def _build_executor_prompt(
    conn: sqlite3.Connection,
    task: TaskRow,
    quality_gate_section: str,
    executor_prompt_suffix: str,
) -> str:
    """Build execution prompt for first-run or rejection retry."""
    if task.get("thread_id"):
        return _build_executor_rejection_prompt(
            conn, task, quality_gate_section, executor_prompt_suffix
        )
    return _build_executor_fresh_prompt(conn, task, quality_gate_section, executor_prompt_suffix)


def _is_thread_not_found(exc: BaseException) -> bool:
    """Detect thread-not-found errors from Codex API."""
    msg = str(exc).lower()
    return "not found" in msg or "not_found" in msg


def _maybe_escalate_codex_config(
    conn: sqlite3.Connection,
    task: TaskRow,
    existing_thread_id: str | None,
    turn_config: dict,
    thread_config: dict,
    runtime_model_config: dict[str, str],
) -> str | None:
    """Escalate Codex effort/model on rejection retries.

    Returns the thread ID to use downstream — ``None`` means force a new
    thread (e.g. after model escalation clears the stale thread).
    """
    if not existing_thread_id:
        return None
    rejection_count = _get_rejection_count(conn, task["id"])
    if rejection_count >= 1:
        turn_config["effort"] = "xhigh"
        log.info(
            "Escalating effort to xhigh for task %s (rejection %d)",
            task["id"],
            rejection_count,
        )
    if rejection_count >= 2:
        work_model = runtime_model_config["work_model"]
        think_model = runtime_model_config["think_model"]
        if work_model != think_model:
            project_id = _get_project_id_for_task(conn, task)
            project_name = _resolve_project_name(conn, project_id) if project_id else ""
            publish_task_model_escalation(
                task["id"],
                work_model,
                think_model,
                rejection_count,
                project=project_name,
                plan_id=task["plan_id"],
            )
            thread_config["model"] = think_model
            set_task_model(conn, task["id"], think_model)
            set_task_thread_id(conn, task["id"], None)
            log.info(
                "Escalating model %s → %s for task %s (rejection %d), clearing thread",
                work_model,
                think_model,
                task["id"],
                rejection_count,
            )
            return None
    return existing_thread_id


def _build_codex_post_initial_turn(
    task: TaskRow,
) -> Callable[[Callable[[str], Awaitable[dict]], str], Awaitable[None]]:
    """Build post-turn callback that nudges Codex to commit changes."""

    async def _post_initial_turn(
        run_turn: Callable[[str], Awaitable[dict]], _thread_id: str
    ) -> None:
        assert task["worktree"] is not None
        for nudge in range(MAX_COMMIT_NUDGES):
            if not _has_uncommitted_changes(task["worktree"]):
                break
            log.warning(
                "Task %s has uncommitted changes after turn, nudging (%d/%d)",
                task["id"],
                nudge + 1,
                MAX_COMMIT_NUDGES,
            )
            await run_turn(_COMMIT_NUDGE)

        if _has_uncommitted_changes(task["worktree"]):
            log.warning(
                "Task %s still has uncommitted changes after %d nudges",
                task["id"],
                MAX_COMMIT_NUDGES,
            )

    return _post_initial_turn


async def _execute_codex_turn(
    conn: sqlite3.Connection,
    task: TaskRow,
    client,
    prompt: str,
    turn_config: dict,
    thread_config: dict,
    fallback_config: dict | None,
    existing_thread_id: str | None,
    post_initial_turn: Callable[[Callable[[str], Awaitable[dict]], str], Awaitable[None]],
    event_context: TurnEventContext | None = None,
    trace_context: Any | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Run Codex execution turn for new or resumed thread.

    If the existing thread has expired server-side (thread not found),
    falls through to the fresh-thread path instead of crashing.
    """
    # -- Rollback/compact on rejection 2+ (may detect dead thread early) --
    if existing_thread_id:
        rejection_count = _get_rejection_count(conn, task["id"])
        if rejection_count >= 2:
            try:
                await client.request(
                    "thread/rollback",
                    {"threadId": existing_thread_id, "numTurns": 1},
                    timeout=30,
                )
                log.info(
                    "Rolled back 1 turn on thread %s before retry (rejection %d)",
                    existing_thread_id,
                    rejection_count,
                )
            except Exception as exc:
                if _is_thread_not_found(exc):
                    log.warning(
                        "Thread %s not found during rollback for task %s, will start fresh",
                        existing_thread_id,
                        task["id"],
                    )
                    existing_thread_id = None
                else:
                    log.warning(
                        "Failed to rollback thread %s for task %s: %s",
                        existing_thread_id,
                        task["id"],
                        exc,
                    )

        if existing_thread_id and rejection_count >= 2:
            try:
                await _compact_codex_thread(client, existing_thread_id)
                log.info(
                    "Compacted thread %s before retry (rejection %d)",
                    existing_thread_id,
                    rejection_count,
                )
            except Exception as exc:
                if _is_thread_not_found(exc):
                    log.warning(
                        "Thread %s not found during compact for task %s, will start fresh",
                        existing_thread_id,
                        task["id"],
                    )
                    existing_thread_id = None
                else:
                    log.warning(
                        "Failed to compact thread %s for task %s: %s",
                        existing_thread_id,
                        task["id"],
                        exc,
                    )

    # -- Resume existing thread --
    if existing_thread_id:

        def on_thread_ready(thread_id: str) -> None:
            log.info(
                "Resumed execution thread %s for task %s",
                thread_id,
                task["id"],
            )

        try:
            _, summary, tokens = await _codex_turn(
                client,
                prompt=prompt,
                turn_config=turn_config,
                resume_thread_id=existing_thread_id,
                resume_thread_params={
                    "model": thread_config["model"],
                    "approvalPolicy": thread_config["approvalPolicy"],
                },
                on_thread_ready=on_thread_ready,
                post_initial_turn=post_initial_turn,
                event_context=event_context,
                trace_context=trace_context,
            )
            return summary, tokens
        except Exception as exc:
            if _is_thread_not_found(exc):
                log.warning(
                    "Thread %s not found during resume for task %s, falling back to fresh thread",
                    existing_thread_id,
                    task["id"],
                )
                add_task_log(
                    conn,
                    task_id=task["id"],
                    level="WARNING",
                    message=f"Thread {existing_thread_id} expired, starting fresh thread",
                    source="executor",
                )
            else:
                raise

    # -- Fresh thread (also reached via thread-not-found fallback) --
    def on_thread_ready(thread_id: str) -> None:
        set_task_thread_id(conn, task["id"], thread_id)
        log.info(
            "Started execution thread %s for task %s",
            thread_id,
            task["id"],
        )

    _, summary, tokens = await _codex_turn(
        client,
        prompt=prompt,
        turn_config=turn_config,
        start_thread_params={"cwd": task["worktree"], **thread_config},
        on_thread_ready=on_thread_ready,
        post_initial_turn=post_initial_turn,
        fallback_thread_params=(
            {"cwd": task["worktree"], **fallback_config} if fallback_config else None
        ),
        event_context=event_context,
        trace_context=trace_context,
    )
    return summary, tokens


async def _run_task_execution_codex_async(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Async implementation of codex task execution.

    If the task has a thread_id (re-run after rejection), resumes the
    executor's existing thread with the rejection context. Otherwise
    starts a fresh thread. After the main turn, checks for uncommitted
    changes and nudges the agent to commit (up to MAX_COMMIT_NUDGES
    times on the same thread).
    """
    from agm.backends import (
        EXECUTOR_PROMPT_SUFFIX,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )

    project_id = _get_project_id_for_task(conn, task)
    backend = _get_plan_backend(conn, task)
    project_model_config = _load_project_model_config(conn, project_id)
    runtime_model_config = _resolve_project_model_config(conn, project_id, backend)

    thread_config = get_runtime_thread_config(
        backend,
        "task_execution",
        project_model_config,
    )
    _apply_project_app_server_ask_for_approval(conn, project_id, thread_config)
    fallback_config = _fallback_thread_config_for_resolved_model(
        thread_config,
        runtime_model_config["work_model"],
    )
    project_dir = _get_project_dir_for_task(conn, task)
    _merge_developer_instructions(thread_config, project_dir, "executor")

    quality_gate_section = _build_quality_gate_prompt(conn, project_id) if project_id else ""
    existing_thread_id = task.get("thread_id")
    prompt = _build_executor_prompt(
        conn,
        task,
        quality_gate_section,
        EXECUTOR_PROMPT_SUFFIX,
    )

    project_name = _resolve_project_name(conn, project_id) if project_id else ""
    from agm.tracing import TraceContext

    trace_ctx = TraceContext(
        entity_type="task",
        entity_id=task["id"],
        stage="execution",
        plan_id=task.get("plan_id"),
        project=project_name,
        conn=conn,
    )

    approval_policy = _load_project_app_server_approval_policy(conn, project_id)
    set_task_active_turn_id(conn, task["id"], None)

    async with _codex_client() as client:
        # -- MCP tool injection (from .agm/tools.toml) --
        mcp_pool = None
        tool_config = load_tool_config(project_dir)
        if tool_config:
            servers = get_servers_for_job_type(tool_config, "task_execution")
            if servers:
                from agm.mcp_pool import McpPool

                mcp_pool = McpPool()
                for name, command, args in servers:
                    try:
                        await mcp_pool.connect(name, command, args)
                    except Exception:
                        log.warning("Failed to connect MCP server '%s'", name, exc_info=True)
                dynamic_tools = mcp_pool.get_dynamic_tools()
                if dynamic_tools:
                    thread_config.setdefault("dynamicTools", []).extend(dynamic_tools)
                    log.info(
                        "Injected %d dynamicTools from %d servers",
                        len(dynamic_tools),
                        len(servers),
                    )

        # -- Channel tool injection (post_channel_message) --
        channel_poster: Callable[[dict[str, Any]], None] | None = None
        plan = get_plan_request(conn, task["plan_id"])
        if plan and plan.get("session_id"):
            task_short = task["id"][:8]
            task_title = task.get("title", task_short)

            def _post_to_channel(args: dict[str, Any]) -> None:
                content = str(args.get("content", "")).strip()
                if not content:
                    return
                kind = str(args.get("kind", "context")).strip() or "context"
                recipient = args.get("recipient")
                metadata = args.get("metadata")
                _post_channel_message(
                    conn,
                    plan,
                    kind=kind,
                    sender=f"executor:{task_short}",
                    content=f"[{task_title}] {content}",
                    recipient=str(recipient) if recipient else None,
                    metadata=metadata if isinstance(metadata, dict) else None,
                )

            channel_poster = _post_to_channel
            thread_config.setdefault("dynamicTools", []).append(CHANNEL_TOOL_SPEC)
            dev_inst = thread_config.get("developerInstructions", "")
            thread_config["developerInstructions"] = dev_inst + CHANNEL_TOOL_INSTRUCTIONS

        # -- Set server request handler (MCP + channel + approval tracing) --
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(
                _make_server_request_handler(
                    mcp_pool,
                    channel_poster=channel_poster,
                    trace_context=trace_ctx,
                    approval_policy=approval_policy,
                )
            )

        try:
            turn_config = get_runtime_turn_config(backend, "task_execution", project_model_config)
            existing_thread_id = _maybe_escalate_codex_config(
                conn, task, existing_thread_id, turn_config, thread_config, runtime_model_config
            )
            if existing_thread_id is None and task.get("thread_id"):
                # Model escalated — no fallback from escalated model
                fallback_config = None
            post_initial_turn = _build_codex_post_initial_turn(task)

            def _on_turn_started(turn_id: str | None) -> None:
                normalized_turn_id = turn_id if isinstance(turn_id, str) else None
                set_task_active_turn_id(conn, task["id"], normalized_turn_id)

            def _on_turn_completed(_turn_id: str | None) -> None:
                set_task_active_turn_id(conn, task["id"], None)

            event_ctx = TurnEventContext(
                task_id=task["id"],
                plan_id=task.get("plan_id"),
                project=project_name,
                on_turn_started=_on_turn_started,
                on_turn_completed=_on_turn_completed,
                owner_role="executor",
                model=runtime_model_config.get("work_model"),
                model_provider=backend,
            )

            summary, tokens = await _execute_codex_turn(
                conn=conn,
                task=task,
                client=client,
                prompt=prompt,
                turn_config=turn_config,
                thread_config=thread_config,
                fallback_config=fallback_config,
                existing_thread_id=existing_thread_id,
                post_initial_turn=post_initial_turn,
                event_context=event_ctx,
                trace_context=trace_ctx,
            )
            update_task_tokens(conn, task["id"], **tokens)
            _publish_execution_context_summary(conn, task, summary or "")
            return summary or "Execution completed"
        finally:
            set_task_active_turn_id(conn, task["id"], None)
            if mcp_pool is not None:
                await mcp_pool.close()


def on_task_execution_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when a task execution job fails. Marks task as failed in DB."""
    task_id = job.args[0] if job.args else None
    if task_id:
        with connect() as conn:
            task = get_task(conn, task_id)
            if not task:
                log.warning("Task %s execution failed callback skipped: task not found", task_id)
                return
            set_task_failure_reason(
                conn,
                task_id,
                _build_execution_failure_reason(
                    task_id,
                    exc_value,
                    path="callback",
                ),
            )
            update_task_status(conn, task_id, "failed", record_history=True)
            set_task_active_turn_id(conn, task_id, None)
            project_id = _get_project_id_for_task(conn, task)
            _emit(
                "task:status",
                task_id,
                "failed",
                project=_resolve_project_name(conn, project_id) if project_id else "",
                plan_id=task["plan_id"],
            )
            failure_label = str(exc_value) if exc_value is not None else ""
            if not failure_label and isinstance(exc_value, BaseException):
                failure_label = exc_value.__class__.__name__
            add_task_log(
                conn,
                task_id=task_id,
                level="ERROR",
                message=f"Task execution failed: {failure_label}",
                source="executor",
            )
            plan = get_plan_request(conn, task["plan_id"])
            if plan:
                _post_channel_message(
                    conn,
                    plan,
                    kind="broadcast",
                    sender=f"executor:{task_id[:8]}",
                    content=f"Execution failed: {task.get('title', task_id)}",
                    metadata={
                        "phase": "execution",
                        "status": "failed",
                        "task_id": task_id,
                        "error": failure_label,
                    },
                )
            # no-op for failed tasks — downstream stays blocked awaiting retry
            resolve_blockers_for_terminal_task(conn, task_id, record_history=True)
            _maybe_complete_session(conn, task["plan_id"])
        log.warning("Task %s execution failed: %s", task_id, exc_value)
