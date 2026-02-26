"""Task review functions.

run_task_review, on_task_review_failure, _handle_review_verdict,
_gather_review_git_context, _check_pre_review_gates, _build_review_prompt,
_prepare_review, codex backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable

from agm.db import (
    TaskRow,
    add_task_log,
    connect,
    get_plan_request,
    get_project_quality_gate,
    get_task,
    resolve_blockers_for_terminal_task,
    set_task_failure_reason,
    set_task_reviewer_thread_id,
    set_task_worker,
    update_task_status,
    update_task_tokens,
)
from agm.jobs_common import (
    CHANNEL_TOOL_INSTRUCTIONS,
    CHANNEL_TOOL_SPEC,
    MAX_DIFF_CHARS,
    MAX_REJECTIONS,
    TaskDBHandler,
    TurnEventContext,
    _apply_project_app_server_ask_for_approval,
    _codex_client,
    _codex_turn,
    _emit,
    _fallback_thread_config_for_resolved_model,
    _get_plan_backend,
    _get_project_dir_for_task,
    _get_project_id_for_task,
    _get_rejection_count,
    _load_project_app_server_approval_policy,
    _load_project_model_config,
    _make_server_request_handler,
    _maybe_complete_session,
    _merge_developer_instructions,
    _parse_task_files,
    _post_channel_message,
    _resolve_effective_base_branch,
    _resolve_project_model_config,
    _resolve_project_name,
)
from agm.jobs_quality_gate import (
    _run_quality_checks,
    _serialize_quality_gate_result,
)

log = logging.getLogger(__name__)


def _validate_review_task_state(task: TaskRow, task_id: str) -> None:
    """Validate review task state before reviewer execution."""
    if task["status"] != "review":
        raise ValueError(f"Task {task_id} is '{task['status']}', not 'review'")
    if not task.get("worktree"):
        raise ValueError(f"Task {task_id} has no worktree")


def _attach_task_db_handler(
    conn: sqlite3.Connection, task_id: str, *, source: str = "reviewer"
) -> tuple[TaskDBHandler, int]:
    """Attach task log handler and return (handler, previous log level)."""
    db_handler = TaskDBHandler(conn, task_id, source=source)
    db_handler.setLevel(logging.DEBUG)
    log.addHandler(db_handler)
    prev_log_level = log.level
    if log.level > logging.DEBUG or log.level == logging.NOTSET:
        log.setLevel(logging.DEBUG)
    return db_handler, prev_log_level


def _log_reviewer_start(
    conn: sqlite3.Connection,
    task: TaskRow,
    backend: str,
    task_id: str,
    pid: int,
) -> None:
    """Log reviewer model context and start event."""
    project_id = _get_project_id_for_task(conn, task)
    reviewer_model_config = _resolve_project_model_config(conn, project_id, backend)
    reviewer_model = reviewer_model_config["think_model"]
    executor_model = task.get("model")
    if executor_model and reviewer_model != executor_model:
        log.info(
            "Task %s reviewer model (%s) differs from executor model (%s)",
            task_id,
            reviewer_model,
            executor_model,
        )
    log.info("Reviewer pid=%d started for task %s (backend=%s)", pid, task_id, backend)


def _run_reviewer_backend(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Execute review against the configured backend."""
    return _run_task_review_codex(conn, task)


def _parse_reviewer_output(result: str) -> dict:
    """Parse reviewer JSON output."""
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"Reviewer output is not valid JSON: {e}") from e


def _build_review_failure_reason(
    task_id: str, summary: object | str | None, rejection_count: int
) -> str:
    """Build structured failure reason for max-rejection review terminal states."""
    summary_text = summary if isinstance(summary, str) else str(summary or "")
    return json.dumps(
        {
            "code": "max_rejections",
            "task_id": task_id,
            "rejection_count": rejection_count,
            "summary_snippet": summary_text[:500],
        },
        sort_keys=True,
    )


def run_task_review(task_id: str) -> str:
    """Review a task via the reviewer agent.

    Called by rq worker. Validates task is in 'review' status, launches
    a codex agent to evaluate the executor's changes, then transitions:
    - approve → 'approved'
    - reject → 'running' (with REVIEW log for executor context)

    On exception, task stays in 'review' — executor's work is preserved.
    """
    with connect() as conn:
        task = get_task(conn, task_id)
        if not task:
            log.warning("Task %s not found (deleted?), skipping", task_id)
            return "skipped:entity_missing"

        _validate_review_task_state(task, task_id)
        db_handler, prev_log_level = _attach_task_db_handler(conn, task_id)

        pid = __import__("os").getpid()
        set_task_worker(conn, task_id, pid=pid)
        backend = _get_plan_backend(conn, task)
        _log_reviewer_start(conn, task, backend, task_id, pid)
        plan = get_plan_request(conn, task["plan_id"])
        if plan:
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"reviewer:{task_id[:8]}",
                content=f"Review started: {task.get('title', task_id)}",
                metadata={
                    "phase": "review",
                    "status": "running",
                    "task_id": task_id,
                    "backend": backend,
                },
            )

        try:
            result = _run_reviewer_backend(conn, task)
            review = _parse_reviewer_output(result)
            _handle_review_verdict(conn, task, review)
            return result
        except Exception:
            log.error("Review of task %s failed", task_id)
            # Do NOT change status — task stays in 'review', executor work preserved
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def _handle_review_verdict(conn: sqlite3.Connection, task: TaskRow, review: dict) -> None:
    """Process reviewer verdict: approve, reject (with retry/fail), or error."""

    def _handle_approval(summary: str, project_name: str) -> None:
        update_task_status(conn, task_id, "approved", record_history=True)
        _emit("task:status", task_id, "approved", project=project_name, plan_id=task["plan_id"])
        log.info("Task %s approved: %s", task_id, summary)
        if not task.get("skip_merge"):
            # Deferred import to avoid circular dependency
            from agm.jobs_merge import _trigger_task_merge

            _trigger_task_merge(task_id)
            return
        _maybe_complete_session(conn, task["plan_id"])

    def _build_review_message(summary: str, findings: list[dict]) -> str:
        findings_text = "\n".join(
            f"  [{f['severity']}] {f['file']}: {f['description']}" for f in findings
        )
        return f"{summary}\n\nFindings:\n{findings_text}" if findings_text else summary

    def _transition_rejected_task(summary: str, project_name: str) -> None:
        rejection_count = _get_rejection_count(conn, task_id)
        if rejection_count >= MAX_REJECTIONS:
            set_task_failure_reason(
                conn,
                task_id,
                _build_review_failure_reason(task_id, summary, rejection_count),
            )
            update_task_status(conn, task_id, "failed", record_history=True)
            _emit("task:status", task_id, "failed", project=project_name, plan_id=task["plan_id"])
            log.warning("Task %s failed after %d rejections", task_id, rejection_count)
            _, cascade_cancelled = resolve_blockers_for_terminal_task(
                conn, task_id, record_history=True
            )
            for cid in cascade_cancelled:
                _emit(
                    "task:status",
                    cid,
                    "cancelled",
                    project=project_name,
                    plan_id=task["plan_id"],
                )
            _maybe_complete_session(conn, task["plan_id"])
            return

        # Record rejected state (briefly visible in watch/display)
        update_task_status(conn, task_id, "rejected", record_history=True)
        _emit("task:status", task_id, "rejected", project=project_name, plan_id=task["plan_id"])
        log.info("Task %s rejected (%d/%d): %s", task_id, rejection_count, MAX_REJECTIONS, summary)

        # Auto-transition to running and re-trigger executor
        update_task_status(conn, task_id, "running", record_history=True)
        _emit("task:status", task_id, "running", project=project_name, plan_id=task["plan_id"])
        log.info(
            "Task %s re-running after rejection (%d/%d)",
            task_id,
            rejection_count,
            MAX_REJECTIONS,
        )
        # Deferred import to avoid circular dependency
        from agm.jobs_merge import _trigger_task_execution

        _trigger_task_execution(conn, task_id)

    task_id = task["id"]
    verdict = review.get("verdict")
    summary = review.get("summary", "")
    findings = review.get("findings", [])
    project_id = _get_project_id_for_task(conn, task)
    project_name = _resolve_project_name(conn, project_id) if project_id else ""
    plan = get_plan_request(conn, task["plan_id"])

    if verdict == "approve":
        _handle_approval(summary, project_name)
        if plan:
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"reviewer:{task_id[:8]}",
                content=f"Review: approved — {summary[:80]}" if summary else "Review: approved",
                metadata={
                    "phase": "review",
                    "status": "approved",
                    "task_id": task_id,
                    "summary": summary,
                    "findings_count": len(findings),
                },
            )
        return

    if verdict != "reject":
        raise RuntimeError(f"Unknown verdict: {verdict}")

    # Log findings at REVIEW level for executor resume context
    add_task_log(
        conn,
        task_id=task_id,
        level="REVIEW",
        message=_build_review_message(summary, findings),
        source="reviewer",
    )
    _transition_rejected_task(summary, project_name)
    if plan:
        _post_channel_message(
            conn,
            plan,
            kind="steer",
            sender=f"reviewer:{task_id[:8]}",
            content=f"Review: rejected — {summary[:80]}" if summary else "Review: rejected",
            recipient=f"executor:{task_id[:8]}",
            metadata={
                "phase": "review",
                "status": "rejected",
                "task_id": task_id,
                "summary": summary,
                "findings_count": len(findings),
            },
        )
        _post_channel_message(
            conn,
            plan,
            kind="dm",
            sender=f"reviewer:{task_id[:8]}",
            recipient=f"executor:{task_id[:8]}",
            content=(
                "Address reviewer findings, rerun quality gates/tests, and resubmit for review."
            ),
            metadata={
                "phase": "review",
                "status": "action_required",
                "task_id": task_id,
            },
        )


def _run_task_review_codex(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Run task review via the Codex backend."""
    return asyncio.run(_run_task_review_codex_async(conn, task))


def _gather_review_git_context(
    worktree: str, base_branch: str, task_files: list[str]
) -> tuple[str, str, str]:
    """Gather diff, commit log, and out-of-scope note for review.

    Returns (diff_text, commit_log, out_of_scope_note).
    """
    import subprocess

    def _run_git_stdout(args: list[str]) -> str:
        return (
            subprocess.run(
                args,
                cwd=worktree,
                capture_output=True,
                text=True,
            ).stdout
            or ""
        )

    def _build_out_of_scope_note() -> str:
        if not task_files:
            return ""
        full_stat = _run_git_stdout(["git", "diff", f"{base_branch}...HEAD", "--stat"])
        scoped_stat = _run_git_stdout(
            ["git", "diff", f"{base_branch}...HEAD", "--stat", "--"] + task_files
        )
        full_count = sum(1 for line in full_stat.strip().splitlines() if "|" in line)
        scoped_count = sum(1 for line in scoped_stat.strip().splitlines() if "|" in line)
        extra = full_count - scoped_count
        if extra <= 0:
            return ""
        return (
            f"\n(Note: {extra} additional file(s) changed outside the "
            "task's file list — not shown in diff.)\n"
        )

    # Scope diff to task files when available
    diff_cmd = ["git", "diff", f"{base_branch}...HEAD"]
    if task_files:
        diff_cmd += ["--"] + task_files
    diff_text = _run_git_stdout(diff_cmd)
    out_of_scope_note = _build_out_of_scope_note()

    # Truncate large diffs
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = (
            diff_text[:MAX_DIFF_CHARS] + f"\n\n...(diff truncated at {MAX_DIFF_CHARS} chars)\n"
        )

    commit_log = _run_git_stdout(["git", "log", "--oneline", f"{base_branch}..HEAD"])

    return diff_text, commit_log, out_of_scope_note


def _check_pre_review_gates(
    conn: sqlite3.Connection,
    task: TaskRow,
    diff_text: str,
    commit_log: str,
    base_branch: str,
) -> str | None:
    """Run empty-submission and quality gates. Returns rejection JSON or None."""
    if not diff_text.strip() and not commit_log.strip():
        return json.dumps(
            {
                "verdict": "reject",
                "summary": "No commits or changes found — executor produced no code.",
                "findings": [
                    {
                        "severity": "critical",
                        "file": "(none)",
                        "description": (
                            f"The branch has no commits ahead of {base_branch} "
                            "and no diff. The executor failed to write or commit any code."
                        ),
                    }
                ],
            }
        )

    plan = get_plan_request(conn, task["plan_id"])
    project_quality_gate = None
    if plan:
        project_quality_gate = get_project_quality_gate(conn, plan["project_id"])
    assert task["worktree"] is not None
    qg_result = _run_quality_checks(
        task["worktree"],
        quality_gate_json=project_quality_gate,
    )

    # Always log structured results when checks ran
    if qg_result.checks:
        add_task_log(
            conn,
            task_id=task["id"],
            level="QUALITY_GATE",
            message=json.dumps(_serialize_quality_gate_result(qg_result)),
            source="reviewer",
        )

    if not qg_result.passed:
        failures = qg_result.failures
        return json.dumps(
            {
                "verdict": "reject",
                "summary": (
                    "Quality gate failed: "
                    + ", ".join(c.name for c in failures)
                    + ". Fix these before review."
                ),
                "findings": [
                    {
                        "severity": "critical",
                        "file": c.name,
                        "description": f"{c.name} failed:\n{c.output[:500]}",
                    }
                    for c in failures
                ],
            }
        )

    return None


def _build_review_prompt(
    conn: sqlite3.Connection,
    task: TaskRow,
    task_files: list[str],
    diff_text: str,
    commit_log: str,
    out_of_scope_note: str,
) -> str:
    """Build the reviewer prompt from task context and git data."""
    from agm.backends import REVIEWER_PROMPT_SUFFIX

    parts = [f"Task: {task['title']}\n\n{task['description']}"]
    if task_files:
        parts.append("\nFiles listed in task:\n" + "\n".join(f"  - {f}" for f in task_files))

    parts.append(f"\nCommits:\n```\n{commit_log}```")
    parts.append(f"\nDiff:\n```diff\n{diff_text}```")
    if out_of_scope_note:
        parts.append(out_of_scope_note)

    project_id = _get_project_id_for_task(conn, task)
    if project_id:
        from agm.jobs_execution import _build_quality_gate_prompt

        quality_gate_section = _build_quality_gate_prompt(conn, project_id)
        if quality_gate_section:
            parts.append(quality_gate_section)

    parts.append(REVIEWER_PROMPT_SUFFIX)
    return "\n".join(parts)


def _prepare_review(conn: sqlite3.Connection, task: TaskRow) -> tuple[str, None] | tuple[None, str]:
    """Shared pre-review: gather context, run gates, build prompt.

    Returns (prompt, None) on success or (None, rejection_json) on gate failure.
    """
    task_files = _parse_task_files(task)
    assert task["worktree"] is not None
    base_branch = _resolve_effective_base_branch(conn, task)
    diff_text, commit_log, out_of_scope_note = _gather_review_git_context(
        task["worktree"],
        base_branch,
        task_files,
    )

    rejection = _check_pre_review_gates(
        conn,
        task,
        diff_text,
        commit_log,
        base_branch,
    )
    if rejection:
        return None, rejection

    prompt = _build_review_prompt(
        conn,
        task,
        task_files,
        diff_text,
        commit_log,
        out_of_scope_note,
    )
    return prompt, None


async def _run_task_review_codex_async(conn: sqlite3.Connection, task: TaskRow) -> str:
    """Run codex task review. Returns raw JSON output string."""
    from agm.backends import get_runtime_thread_config, get_runtime_turn_config

    project_id = _get_project_id_for_task(conn, task)

    prompt, rejection = _prepare_review(conn, task)
    if rejection:
        return rejection
    assert prompt is not None

    worktree = task["worktree"]
    backend = _get_plan_backend(conn, task)
    project_model_config = _load_project_model_config(conn, project_id)
    runtime_model_config = _resolve_project_model_config(conn, project_id, backend)
    thread_config = get_runtime_thread_config(
        backend,
        "task_review",
        project_model_config,
    )
    _apply_project_app_server_ask_for_approval(conn, project_id, thread_config)
    turn_config = get_runtime_turn_config(backend, "task_review", project_model_config)
    _merge_developer_instructions(
        thread_config,
        _get_project_dir_for_task(conn, task),
        "reviewer",
    )
    fallback_config = _fallback_thread_config_for_resolved_model(
        thread_config,
        runtime_model_config["think_model"],
    )

    project_name = _resolve_project_name(conn, project_id) if project_id else ""
    from agm.tracing import TraceContext

    trace_ctx = TraceContext(
        entity_type="task",
        entity_id=task["id"],
        stage="review",
        plan_id=task.get("plan_id"),
        project=project_name,
        conn=conn,
    )

    approval_policy = _load_project_app_server_approval_policy(conn, project_id)

    async with _codex_client() as client:
        channel_poster: Callable[[dict[str, object]], None] | None = None
        plan = get_plan_request(conn, task["plan_id"])
        if plan and plan.get("session_id"):
            sender = f"reviewer:{task['id'][:8]}"

            def _post_to_channel(args: dict[str, object]) -> None:
                content = str(args.get("content", "")).strip()
                if not content:
                    return
                kind = str(args.get("kind", "context")) or "context"
                recipient = args.get("recipient")
                metadata = args.get("metadata")
                _post_channel_message(
                    conn,
                    plan,
                    kind=kind,
                    sender=sender,
                    recipient=str(recipient) if recipient else None,
                    content=content,
                    metadata=metadata if isinstance(metadata, dict) else None,
                )

            channel_poster = _post_to_channel
            thread_config.setdefault("dynamicTools", []).append(CHANNEL_TOOL_SPEC)
            dev_inst = thread_config.get("developerInstructions", "")
            thread_config["developerInstructions"] = dev_inst + CHANNEL_TOOL_INSTRUCTIONS

        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(
                _make_server_request_handler(
                    channel_poster=channel_poster,
                    trace_context=trace_ctx,
                    approval_policy=approval_policy,
                )
            )

        def on_thread_ready(thread_id: str) -> None:
            set_task_reviewer_thread_id(conn, task["id"], thread_id)
            log.info("Started review thread %s for task %s", thread_id, task["id"])

        event_ctx = TurnEventContext(
            task_id=task["id"],
            plan_id=task.get("plan_id"),
            project=project_name,
            owner_role="reviewer",
            model=runtime_model_config.get("think_model"),
            model_provider=backend,
        )

        _, output_text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            start_thread_params={"cwd": worktree, **thread_config},
            on_thread_ready=on_thread_ready,
            fallback_thread_params=(
                {"cwd": worktree, **fallback_config} if fallback_config else None
            ),
            event_context=event_ctx,
            trace_context=trace_ctx,
        )
        update_task_tokens(conn, task["id"], **tokens)
        if not output_text:
            raise RuntimeError("No output produced by reviewer agent")
        return output_text


def on_task_review_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when a task review job fails.

    Logs error but does NOT change task status — task stays in 'review'
    so executor's work is preserved. User can re-run the reviewer.
    Emits event so web UI is notified.
    """
    task_id = job.args[0] if job.args else None
    if task_id:
        with connect() as conn:
            task = get_task(conn, task_id)
            if not task:
                log.warning("Task %s review failed callback skipped: task not found", task_id)
                return
            add_task_log(
                conn,
                task_id=task_id,
                level="ERROR",
                message=f"Task review failed: {exc_value}",
                source="reviewer",
            )
            project_id = _get_project_id_for_task(conn, task)
            project_name = _resolve_project_name(conn, project_id) if project_id else ""
            _emit(
                "task:status",
                task_id,
                "review",
                project=project_name,
                plan_id=task["plan_id"],
            )
            plan = get_plan_request(conn, task["plan_id"])
            if plan:
                _post_channel_message(
                    conn,
                    plan,
                    kind="broadcast",
                    sender=f"reviewer:{task_id[:8]}",
                    content=f"Review failed: {task.get('title', task_id)}",
                    metadata={
                        "phase": "review",
                        "status": "failed",
                        "task_id": task_id,
                        "error": str(exc_value),
                    },
                )
        log.warning("Task %s review failed: %s", task_id, exc_value)
