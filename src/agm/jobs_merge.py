"""Merge + auto-trigger functions.

run_task_merge, on_task_merge_failure, capture/conflict helpers,
_rollback_claim, and all auto-trigger wiring between pipeline stages.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sqlite3
import subprocess

from agm.db import (
    ProjectRow,
    TaskRow,
    add_task_log,
    claim_task,
    connect,
    get_plan_request,
    get_project,
    get_project_post_merge_command,
    get_project_quality_gate,
    get_task,
    list_task_logs,
    reset_task_for_reexecution,
    resolve_blockers_for_terminal_task,
    set_task_failure_reason,
    set_task_merge_commit,
    update_task_status,
)
from agm.jobs_common import (
    MAX_DIFF_CHARS,
    TaskDBHandler,
    _emit,
    _get_project_id_for_task,
    _maybe_complete_session,
    _post_channel_message,
    _resolve_effective_base_branch,
    _resolve_project_name,
)
from agm.jobs_quality_gate import (
    _run_quality_checks,
    _serialize_quality_gate_result,
)

log = logging.getLogger(__name__)


WORKTREE_SYNC_WARNING_MESSAGE = (
    "Merge succeeded but working tree sync failed. Run git checkout HEAD -- . manually."
)


def _capture_branch_diff(project_dir: str, branch: str, base_branch: str, task_id: str) -> str:
    """Capture the diff between a branch and base for re-execution context.

    Uses git diff base...branch (three-dot = changes introduced by branch).
    Truncates at MAX_DIFF_CHARS to prevent prompt bloat.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...{branch}"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("Failed to capture branch diff for task %s: %s", task_id, e)
        return "(diff capture failed)"

    if not diff:
        return "(no diff)"
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + "\n\n... (truncated)"
    return diff


def _reexecute_task(
    conn: sqlite3.Connection,
    task: TaskRow,
    proj: ProjectRow,
    base_branch: str,
    *,
    log_level: str,
    log_message: str,
) -> str:
    """Tear down worktree, create fresh one, and re-trigger executor.

    Shared by merge conflict and quality gate failure recovery paths.
    Returns a status message describing what happened.
    """
    from agm.git_ops import create_worktree, remove_worktree

    task_id = task["id"]
    assert task["branch"] is not None
    assert task["worktree"] is not None

    # Capture the diff before teardown
    diff = _capture_branch_diff(proj["dir"], task["branch"], base_branch, task_id)

    # Log the failure + saved diff
    add_task_log(
        conn,
        task_id=task_id,
        level=log_level,
        message=f"{log_message}\n\n{diff}",
        source="merger",
    )

    # Tear down old worktree + branch, create fresh one
    remove_worktree(proj["dir"], task["worktree"], task["branch"])
    new_branch, new_worktree = create_worktree(proj["dir"], task_id, task["title"], base_branch)
    reset_task_for_reexecution(conn, task_id, new_branch, new_worktree, record_history=True)
    _emit(
        "task:status",
        task_id,
        "running",
        project=proj["name"],
        plan_id=task["plan_id"],
    )
    log.info("Task %s reset for re-execution on %s", task_id, new_branch)

    _trigger_task_execution(conn, task_id)
    return f"{log_level} — re-execution triggered for {task_id}"


def _get_merge_conflict_context(conn: sqlite3.Connection, task_id: str) -> str | None:
    """Fetch the diff from the most recent MERGE_CONFLICT task_log."""
    logs = list_task_logs(conn, task_id, level="MERGE_CONFLICT")
    if not logs:
        return None
    message = logs[-1]["message"]
    # Extract diff portion after the header line
    marker = "\n\n"
    idx = message.find(marker)
    if idx >= 0:
        return message[idx + len(marker) :]
    return message


def _rollback_claim(
    conn: sqlite3.Connection,
    task_id: str,
    project_dir: str | None,
    branch: str,
    worktree_path: str,
) -> None:
    """Roll back a task claim: reset to ready, remove worktree.

    Called when enqueue fails after a successful claim to prevent
    the task from being stuck in 'running' with no job.
    """
    try:
        conn.execute(
            "UPDATE tasks SET status = 'ready', pid = NULL, thread_id = NULL, "
            "actor = NULL, caller = NULL, branch = NULL, worktree = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = ? AND status = 'running'",
            (task_id,),
        )
        conn.commit()
        if project_dir:
            from agm.git_ops import remove_worktree

            remove_worktree(project_dir, worktree_path, branch)
        log.info("Rolled back claim for task %s → ready", task_id)
    except Exception as e:
        log.error("Failed to roll back claim for task %s: %s", task_id, e)


def _prepare_ready_task_execution_claim(
    conn: sqlite3.Connection,
    task: TaskRow,
) -> tuple[str, str, str] | None:
    """Create worktree + claim a ready task for auto execution."""
    task_id = task["id"]
    plan = get_plan_request(conn, task["plan_id"])
    if not plan:
        log.error("_trigger_task_execution: plan for task %s not found", task_id)
        return None
    proj = get_project(conn, plan["project_id"])
    if not proj:
        log.error("_trigger_task_execution: project for task %s not found", task_id)
        return None

    project_dir = proj["dir"]
    base_branch = _resolve_effective_base_branch(conn, proj)
    from agm.git_ops import create_worktree

    branch, worktree_path = create_worktree(
        project_dir,
        task_id,
        task["title"],
        base_branch=base_branch,
    )
    if not claim_task(
        conn,
        task_id,
        actor="agm-auto",
        caller="agm-auto",
        branch=branch,
        worktree=worktree_path,
        record_history=True,
    ):
        log.error("_trigger_task_execution: failed to claim task %s", task_id)
        return None

    _emit("task:status", task_id, "running", project=proj["name"], plan_id=task["plan_id"])
    log.info("Auto-claimed task %s (branch=%s)", task_id, branch)
    return project_dir, branch, worktree_path


def _trigger_task_execution(conn: sqlite3.Connection, task_id: str) -> None:
    """Auto-trigger executor for a task that is ready or running (after rejection).

    Ready tasks: look up project dir, create worktree, claim, then enqueue.
    Running tasks (after rejection): already have worktree, just enqueue.
    Logs but does NOT re-raise on failure.

    If enqueue fails after a successful claim, rolls back: resets task to ready
    and removes the worktree so it's not stuck running with no job.
    """
    try:
        from agm.queue import enqueue_task_execution

        task = get_task(conn, task_id)
        if not task:
            log.error("_trigger_task_execution: task %s not found", task_id)
            return

        newly_claimed = False
        project_dir = None
        branch = ""
        worktree_path = ""

        if task["status"] == "ready":
            claim_context = _prepare_ready_task_execution_claim(conn, task)
            if claim_context is None:
                return
            project_dir, branch, worktree_path = claim_context
            newly_claimed = True
        elif task["status"] == "running":
            if not task.get("worktree"):
                log.error("_trigger_task_execution: running task %s has no worktree", task_id)
                return
        else:
            log.warning(
                "_trigger_task_execution: task %s is '%s', skipping", task_id, task["status"]
            )
            return

        try:
            enqueue_task_execution(task_id)
            log.info("Auto-enqueued execution for task %s", task_id)
        except Exception:
            if newly_claimed:
                log.error("Enqueue failed after claiming task %s — rolling back to ready", task_id)
                _rollback_claim(conn, task_id, project_dir, branch, worktree_path)
            raise
    except Exception as e:
        log.error("_trigger_task_execution failed for task %s: %s", task_id, e)


def _trigger_task_review(task_id: str) -> None:
    """Auto-trigger reviewer for a task in review status.

    Logs but does NOT re-raise on failure.
    """
    try:
        from agm.queue import enqueue_task_review

        enqueue_task_review(task_id)
        log.info("Auto-enqueued review for task %s", task_id)
    except Exception as e:
        log.error("_trigger_task_review failed for task %s: %s", task_id, e)


def _trigger_task_merge(task_id: str) -> None:
    """Auto-trigger merge for an approved task.

    Logs but does NOT re-raise on failure.
    """
    try:
        from agm.queue import enqueue_task_merge

        enqueue_task_merge(task_id)
        log.info("Auto-enqueued merge for task %s", task_id)
    except Exception as e:
        log.error("_trigger_task_merge failed for task %s: %s", task_id, e)


# -- task merge --


def _check_pre_merge_quality_gate(
    conn: sqlite3.Connection,
    task: TaskRow,
    proj: ProjectRow,
    base_branch: str,
) -> str | None:
    """Run quality gate before merge. Returns message if re-execution triggered, else None.

    Raises RuntimeError on second quality gate failure (task stays approved).
    """
    task_id = task["id"]
    plan = get_plan_request(conn, task["plan_id"])
    project_quality_gate = get_project_quality_gate(conn, plan["project_id"]) if plan else None
    if not project_quality_gate:
        return None

    assert task["worktree"] is not None
    qg_result = _run_quality_checks(task["worktree"], quality_gate_json=project_quality_gate)

    # Always log structured results when checks ran
    if qg_result.checks:
        add_task_log(
            conn,
            task_id=task_id,
            level="QUALITY_GATE",
            message=json.dumps(_serialize_quality_gate_result(qg_result)),
            source="merger",
        )

    if qg_result.passed:
        return None

    existing_qg_logs = list_task_logs(conn, task_id, level="QUALITY_GATE_FAIL")
    if existing_qg_logs:
        log.warning("Task %s: second quality gate failure at merge, leaving approved", task_id)
        raise RuntimeError(
            "Quality gate failed at merge (second attempt): "
            + ", ".join(c.name for c in qg_result.failures)
        )

    failure_details = "\n".join(f"- {c.name}: {c.output[:500]}" for c in qg_result.failures)
    log.info("Task %s: quality gate failed at merge, initiating re-execution", task_id)
    return _reexecute_task(
        conn,
        task,
        proj,
        base_branch,
        log_level="QUALITY_GATE_FAIL",
        log_message=(
            "Quality gate failed at merge. Re-execution triggered.\n\n"
            f"Failures:\n{failure_details}\n\nPrevious diff:"
        ),
    )


def _run_post_merge_command(project_dir: str, command: str, merge_sha: str, task_id: str) -> None:
    """Run the project's configured post-merge command.

    The command is executed in the project directory as argv (no shell). The
    merge SHA is exposed via AGM_MERGE_SHA for consumers that need it.
    Failures are logged but do not block the merge pipeline.
    """
    args = shlex.split(command)
    env = os.environ.copy()
    env["AGM_MERGE_SHA"] = merge_sha
    env["AGM_TASK_ID"] = task_id
    try:
        result = subprocess.run(
            args,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode == 0:
            log.info("Post-merge command succeeded for task %s (SHA %s)", task_id, merge_sha[:8])
        else:
            log.error("Post-merge command failed for task %s: %s", task_id, result.stderr.strip())
    except Exception as e:
        log.error("Post-merge command skipped for task %s: %s", task_id, e)


def _complete_merge(
    conn: sqlite3.Connection,
    task: TaskRow,
    proj: ProjectRow,
    base_branch: str,
    merge_sha: str | None,
    sync_failure: bool = False,
) -> str:
    """Post-merge bookkeeping: status update, blocker resolution, cleanup, auto-trigger."""
    from agm.git_ops import remove_worktree

    task_id = task["id"]
    log.info("Merged %s -> %s for task %s", task["branch"], base_branch, task_id)

    if merge_sha and isinstance(merge_sha, str):
        set_task_merge_commit(conn, task_id, merge_sha)
    if merge_sha and isinstance(merge_sha, str) and sync_failure:
        add_task_log(
            conn,
            task_id=task_id,
            level="WARNING",
            message=WORKTREE_SYNC_WARNING_MESSAGE,
            source="merger",
        )

    update_task_status(conn, task_id, "completed", record_history=True)
    _emit("task:status", task_id, "completed", project=proj["name"], plan_id=task["plan_id"])
    plan = get_plan_request(conn, task["plan_id"])
    if plan:
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"merger:{task_id[:8]}",
            content=f"Merged to {base_branch}",
            metadata={
                "phase": "merge",
                "status": "completed",
                "task_id": task_id,
                "base_branch": base_branch,
                "merge_commit": merge_sha,
            },
        )
    _maybe_complete_session(conn, task["plan_id"])
    log.info("Task %s -> completed", task_id)

    promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
        conn, task_id, record_history=True
    )
    for pid in promoted:
        log.info("Unblocked task %s -> ready", pid)
    for cid in cascade_cancelled:
        _emit("task:status", cid, "cancelled", project=proj["name"], plan_id=task["plan_id"])

    assert task["worktree"] is not None
    assert task["branch"] is not None
    remove_worktree(proj["dir"], task["worktree"], task["branch"])
    log.info("Cleaned up worktree for task %s: %s", task_id, task["branch"])

    if merge_sha:
        plan = get_plan_request(conn, task["plan_id"])
        post_merge_cmd = get_project_post_merge_command(conn, plan["project_id"]) if plan else None
        if post_merge_cmd:
            _run_post_merge_command(proj["dir"], post_merge_cmd, merge_sha, task_id)

    from agm.jobs_task_creation import _auto_trigger_execution_for_ready_tasks

    _auto_trigger_execution_for_ready_tasks(conn, promoted)
    return f"Merged {task['branch']} -> {base_branch}"


def _validate_merge_task_state(task: TaskRow, task_id: str) -> None:
    """Validate a task can be merged."""
    if task["status"] != "approved":
        raise ValueError(f"Task {task_id} is '{task['status']}', not 'approved'")
    if not task.get("worktree") or not task.get("branch"):
        raise ValueError(f"Task {task_id} has no worktree/branch")


def _load_merge_project_context(
    conn: sqlite3.Connection,
    task: TaskRow,
    task_id: str,
) -> tuple[ProjectRow, str]:
    """Resolve project + effective base branch for merge."""
    plan = get_plan_request(conn, task["plan_id"])
    if not plan:
        raise ValueError(f"Plan for task {task_id} not found")
    proj = get_project(conn, plan["project_id"])
    if not proj:
        raise ValueError(f"Project for task {task_id} not found")
    base_branch = _resolve_effective_base_branch(conn, proj)
    return proj, base_branch


def _warn_out_of_scope_merge_files(task: TaskRow, proj: ProjectRow, base_branch: str) -> None:
    """Warn when a branch touches files outside task scope."""
    if not task.get("files"):
        return

    from agm.git_ops import check_branch_file_scope

    assert task["files"] is not None
    assert task["branch"] is not None
    allowed = json.loads(task["files"])
    out_of_scope = check_branch_file_scope(
        proj["dir"],
        task["branch"],
        allowed,
        base_branch=base_branch,
    )
    if out_of_scope:
        log.warning(
            "Branch touches %d out-of-scope file(s): %s",
            len(out_of_scope),
            ", ".join(out_of_scope),
        )


def _build_merge_conflict_failure_reason(conflict_context: str | None) -> str:
    """Build the deterministic merge-conflict exhaustion reason."""
    context = conflict_context or "(no conflict context available)"
    return f"[MERGE_CONFLICT] Attempt 3/3: merge conflict after re-executions.\n\n{context}"


def _merge_or_reexecute_on_conflict(
    conn: sqlite3.Connection,
    task: TaskRow,
    proj: ProjectRow,
    base_branch: str,
) -> tuple[str | None, str | None, bool]:
    """Merge task branch; on first/second conflict trigger re-execution."""
    from agm.git_ops import merge_to_main

    task_id = task["id"]
    assert task["branch"] is not None
    sync_failures: list[str] = []
    try:
        merge_sha = merge_to_main(
            proj["dir"],
            task["branch"],
            task_id,
            task["title"],
            base_branch=base_branch,
            worktree_path=task["worktree"],
            on_sync_failure=sync_failures.append,
        )
        return merge_sha, None, len(sync_failures) > 0
    except RuntimeError as merge_err:
        if "conflict" not in str(merge_err).lower():
            raise

        existing_conflict_logs = list_task_logs(conn, task_id, level="MERGE_CONFLICT")
        attempt_number = len(existing_conflict_logs) + 1
        if attempt_number >= 3:
            failure_reason = _build_merge_conflict_failure_reason(
                _get_merge_conflict_context(conn, task_id)
            )
            add_task_log(
                conn,
                task_id=task_id,
                level="ERROR",
                message=failure_reason,
                source="merger",
            )
            set_task_failure_reason(conn, task_id, failure_reason)
            update_task_status(conn, task_id, "failed", record_history=True)
            project_id = _get_project_id_for_task(conn, task)
            project_name = _resolve_project_name(conn, project_id) if project_id else ""
            _emit(
                "task:status",
                task_id,
                "failed",
                project=project_name,
                plan_id=task["plan_id"],
            )
            promoted, cascade_cancelled = resolve_blockers_for_terminal_task(
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
            if promoted:
                log.info("Failed task %s unblocked %d task(s)", task_id, len(promoted))
            plan = get_plan_request(conn, task["plan_id"])
            if plan:
                _post_channel_message(
                    conn,
                    plan,
                    kind="broadcast",
                    sender=f"merger:{task_id[:8]}",
                    content=f"Merge failed: {task.get('title', task_id)}",
                    metadata={
                        "phase": "merge",
                        "status": "failed",
                        "task_id": task_id,
                        "failure_reason": failure_reason,
                    },
                )
            _maybe_complete_session(conn, task["plan_id"])
            log.warning(
                "Task %s: merge conflict attempts exhausted (3/3), transitioning to failed",
                task_id,
            )
            return None, failure_reason, False

        attempt_number += 1
        log.info(
            "Task %s: merge conflict, attempt %d/3, re-executing",
            task_id,
            attempt_number,
        )
        return (
            None,
            _reexecute_task(
                conn,
                task,
                proj,
                base_branch,
                log_level="MERGE_CONFLICT",
                log_message=(
                    f"[MERGE_CONFLICT] Attempt {attempt_number}/3: "
                    "re-executing against updated main\n\nPrevious diff:"
                ),
            ),
            False,
        )


def run_task_merge(task_id: str) -> str:
    """Merge an approved task's branch into the configured base branch.

    Called by rq worker on the serialized agm:merge queue.
    On success: completed, blockers resolved, worktree cleaned up,
    newly-ready tasks auto-triggered for execution.
    On conflict: attempts 1 and 2 capture diff, tear down worktree, create fresh one,
    and re-trigger executor. Attempt 3 transitions task to failed.
    On non-conflict failures: task stays approved.
    """
    with connect() as conn:
        task = get_task(conn, task_id)
        if not task:
            log.warning("Task %s not found (deleted?), skipping", task_id)
            return "skipped:entity_missing"

        _validate_merge_task_state(task, task_id)

        db_handler = TaskDBHandler(conn, task_id, source="merger")
        db_handler.setLevel(logging.DEBUG)
        log.addHandler(db_handler)
        prev_log_level = log.level
        if log.level > logging.DEBUG or log.level == logging.NOTSET:
            log.setLevel(logging.DEBUG)

        try:
            proj, base_branch = _load_merge_project_context(conn, task, task_id)
            _warn_out_of_scope_merge_files(task, proj, base_branch)

            # Pre-merge quality gate (only if project has a configured gate)
            qg_result = _check_pre_merge_quality_gate(conn, task, proj, base_branch)
            if qg_result is not None:
                return qg_result

            merge_sha, reexecute_result, sync_failure = _merge_or_reexecute_on_conflict(
                conn, task, proj, base_branch
            )
            if reexecute_result is not None:
                return reexecute_result

            return _complete_merge(
                conn, task, proj, base_branch, merge_sha, sync_failure=sync_failure
            )
        except Exception:
            log.error("Merge failed for task %s", task_id)
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def on_task_merge_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when a task merge job fails.

    Logs error but does NOT change task status — task stays approved.
    Emits event so web UI is notified.
    """
    task_id = job.args[0] if job.args else None
    if task_id:
        with connect() as conn:
            task = get_task(conn, task_id)
            if not task:
                log.warning("Task %s merge failed callback skipped: task not found", task_id)
                return
            add_task_log(
                conn,
                task_id=task_id,
                level="ERROR",
                message=f"Task merge failed: {exc_value}",
                source="merger",
            )
            project_id = _get_project_id_for_task(conn, task)
            project_name = _resolve_project_name(conn, project_id) if project_id else ""
            _emit(
                "task:status",
                task_id,
                "approved",
                project=project_name,
                plan_id=task["plan_id"],
            )
        log.warning("Task %s merge failed: %s", task_id, exc_value)
