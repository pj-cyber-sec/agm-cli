"""Enrichment job functions.

run_enrichment, on_enrichment_failure, all enrichment variants
(fresh/resume/continuation), and enrichment processing helpers.

Extracted from jobs_plan.py to separate enrichment from planning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from typing import Any

from agm.db import (
    PlanRow,
    TaskRow,
    add_plan_log,
    add_plan_question,
    connect,
    get_plan_request,
    get_project,
    list_plan_questions,
    list_tasks,
    set_plan_request_worker,
    update_plan_enrichment,
    update_plan_request_status,
    update_plan_tokens,
    update_prompt_status,
)
from agm.jobs_common import (
    CHANNEL_TOOL_INSTRUCTIONS,
    CHANNEL_TOOL_SPEC,
    PlanDBHandler,
    _apply_project_app_server_ask_for_approval,
    _codex_client,
    _codex_turn,
    _emit,
    _load_project_app_server_approval_policy,
    _load_project_model_config,
    _make_server_request_handler,
    _maybe_complete_session,
    _merge_developer_instructions,
    _post_channel_message,
    _resolve_project_name,
)

log = logging.getLogger(__name__)


def run_enrichment(plan_id: str) -> str:
    """Execute the enrichment phase for a plan.

    Called by rq worker on the agm:enrichment queue. Updates prompt_status
    as it progresses: pending → enriching → (awaiting_input →) finalized.

    On finalization, auto-enqueues planning via enqueue_plan_request().
    """
    with connect() as conn:
        plan = get_plan_request(conn, plan_id)
        if not plan:
            log.warning("Plan %s not found (deleted?), skipping", plan_id)
            return "skipped:entity_missing"

        if plan["status"] == "cancelled":
            log.info("Plan %s already cancelled, skipping", plan_id)
            return "skipped:cancelled"

        db_handler = PlanDBHandler(conn, plan_id, source="enrichment")
        db_handler.setLevel(logging.DEBUG)
        log.addHandler(db_handler)
        prev_log_level = log.level
        if log.level > logging.DEBUG or log.level == logging.NOTSET:
            log.setLevel(logging.DEBUG)

        project_name = _resolve_project_name(conn, plan["project_id"])

        pid = os.getpid()
        set_plan_request_worker(conn, plan_id, pid=pid)
        # Set plan to running if still pending
        if plan["status"] == "pending":
            update_plan_request_status(conn, plan_id, "running", record_history=True)
            _emit("plan:status", plan_id, "running", project=project_name)
        update_prompt_status(conn, plan_id, "enriching")
        _emit("prompt:status", plan_id, "enriching", project=project_name)
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"enrichment:{plan_id[:8]}",
            content="Prompt enrichment started",
            metadata={
                "phase": "enrichment",
                "status": "enriching",
                "plan_id": plan_id,
                "backend": plan["backend"],
            },
        )
        log.info("Enriching prompt for plan %s (backend=%s)", plan_id, plan["backend"])

        try:
            planner_prompt = _resolve_enrichment(conn, plan)

            if planner_prompt is None:
                # Questions were asked, plan is now awaiting_input
                update_prompt_status(conn, plan_id, "awaiting_input")
                _emit("prompt:status", plan_id, "awaiting_input", project=project_name)
                _post_channel_message(
                    conn,
                    plan,
                    kind="question",
                    sender=f"enrichment:{plan_id[:8]}",
                    content="Awaiting user answers",
                    metadata={
                        "phase": "enrichment",
                        "status": "awaiting_input",
                        "plan_id": plan_id,
                    },
                )
                log.info("Enrichment paused — awaiting user answers for plan %s", plan_id)
                return "awaiting_input"

            # Enrichment complete — finalize prompt_status and trigger planning
            update_prompt_status(conn, plan_id, "finalized")
            _emit("prompt:status", plan_id, "finalized", project=project_name)
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"enrichment:{plan_id[:8]}",
                content="Prompt enriched",
                metadata={
                    "phase": "enrichment",
                    "status": "finalized",
                    "plan_id": plan_id,
                },
            )

            log.info("Plan %s enrichment finalized, enqueueing exploration", plan_id)
            from agm.queue import enqueue_explorer

            enqueue_explorer(plan_id)
            return "enrichment_finalized"

        except Exception:
            log.error("Enrichment for plan %s failed", plan_id)
            update_prompt_status(conn, plan_id, "failed")
            _emit("prompt:status", plan_id, "failed", project=project_name)
            update_plan_request_status(conn, plan_id, "failed", record_history=True)
            _emit("plan:status", plan_id, "failed", project=project_name)
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def _resolve_enrichment(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    """Run enrichment phase and return the prompt for the planner.

    Returns:
    - str: enriched prompt ready for the planner
    - None: questions were asked, plan transitioned to awaiting_input
    """
    # Re-read plan to get enrichment fields (may have been set in a prior run)
    plan = get_plan_request(conn, plan["id"]) or plan

    if plan.get("enriched_prompt"):
        # Enrichment already complete
        log.info("Plan %s using enriched prompt", plan["id"])
        return plan["enriched_prompt"]

    if plan.get("enrichment_thread_id"):
        # Resume enrichment (questions were answered)
        log.info("Plan %s resuming enrichment", plan["id"])
        return _run_enrichment_resume_codex(conn, plan)

    if plan.get("parent_id"):
        # Continuation — enrich with parent context
        log.info("Plan %s starting continuation enrichment", plan["id"])
        return _run_enrichment_continuation_codex(conn, plan)

    # Fresh enrichment
    log.info("Plan %s starting enrichment", plan["id"])
    return _run_enrichment_codex(conn, plan)


def _enrichment_has_substance(text: str) -> bool:
    """Check if enrichment output is a real prompt vs meta-narration."""
    stripped = text.strip()
    if len(stripped) < 50:
        return False
    lower = stripped.lower()
    meta_indicators = [
        "i will scan",
        "i'm delegating",
        "i am delegating",
        "discovery is complete",
        "sub-agent",
        "sub agent",
        "i'm scanning",
        "i am scanning",
        "spawned an explorer",
        "spawned a reviewer",
        "delegating parallel",
    ]
    meta_hits = sum(1 for ind in meta_indicators if ind in lower)
    return meta_hits < 2


def _process_enrichment_output(
    conn: sqlite3.Connection,
    plan: PlanRow,
    output_text: str,
    thread_id: str,
    tokens: dict[str, int],
) -> str | None:
    """Process enrichment agent output: store result, handle questions.

    Returns enriched prompt string, or None if questions were asked.
    """
    update_plan_tokens(conn, plan["id"], **tokens)

    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("Enrichment output not valid JSON, using raw prompt: %s", e)
        update_plan_enrichment(conn, plan["id"], enriched_prompt=plan["prompt"])
        return plan["prompt"]

    enriched_prompt = data.get("enriched_prompt", "")
    questions = data.get("questions", [])

    if not enriched_prompt or not _enrichment_has_substance(enriched_prompt):
        if enriched_prompt:
            log.warning(
                "Enrichment output is meta-narration for plan %s, using raw prompt",
                plan["id"],
            )
        else:
            log.warning(
                "Enrichment produced empty prompt for plan %s, using raw prompt",
                plan["id"],
            )
        enriched_prompt = plan["prompt"]

    if questions:
        # Store thread_id for resume, insert questions
        update_plan_enrichment(conn, plan["id"], enrichment_thread_id=thread_id)
        for q in questions:
            options_json = json.dumps(q["options"]) if q.get("options") else None
            add_plan_question(
                conn,
                plan_id=plan["id"],
                question=q["question"],
                options=options_json,
                header=q.get("header"),
                multi_select=bool(q.get("multi_select", False)),
            )
        update_plan_request_status(conn, plan["id"], "awaiting_input", record_history=True)
        _emit(
            "plan:status",
            plan["id"],
            "awaiting_input",
            project=_resolve_project_name(conn, plan["project_id"]),
        )
        log.info(
            "Enrichment agent asked %d question(s) for plan %s",
            len(questions),
            plan["id"],
        )
        return None

    # No questions — store enriched prompt and thread_id, proceed
    update_plan_enrichment(
        conn, plan["id"], enriched_prompt=enriched_prompt, enrichment_thread_id=thread_id
    )
    log.info("Plan %s enriched (%d chars)", plan["id"], len(enriched_prompt))
    return enriched_prompt


# -- Fresh enrichment --


def _run_enrichment_codex(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    """Run fresh enrichment via Codex backend."""
    return asyncio.run(_run_enrichment_codex_async(conn, plan))


async def _run_enrichment_codex_async(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    from agm.backends import (
        build_enrichment_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    pmc = _load_project_model_config(conn, plan["project_id"])
    thread_config = get_runtime_thread_config(plan["backend"], "prompt_enrichment", pmc)
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    turn_config = get_runtime_turn_config(plan["backend"], "prompt_enrichment", pmc)
    prompt = build_enrichment_prompt(plan["prompt"])
    _merge_developer_instructions(thread_config, project["dir"], "enrichment")
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="enrichment",
        plan_id=plan["id"],
        project=project["name"],
        conn=conn,
    )
    approval_policy = _load_project_app_server_approval_policy(conn, plan["project_id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))
        if plan.get("session_id"):
            sender = f"enrichment:{plan['id'][:8]}"

            def _post_to_channel(args: dict[str, Any]) -> None:
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

            if callable(set_handler):
                set_handler(
                    _make_server_request_handler(
                        channel_poster=_post_to_channel,
                        trace_context=trace_ctx,
                        approval_policy=approval_policy,
                    )
                )
            thread_config.setdefault("dynamicTools", []).append(CHANNEL_TOOL_SPEC)
            dev_inst = thread_config.get("developerInstructions", "")
            thread_config["developerInstructions"] = dev_inst + CHANNEL_TOOL_INSTRUCTIONS

        def on_thread_ready(thread_id: str) -> None:
            log.info("Started enrichment thread %s for plan %s", thread_id, plan["id"])

        thread_id, output_text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
            on_thread_ready=on_thread_ready,
            trace_context=trace_ctx,
        )

    return _process_enrichment_output(conn, plan, output_text, thread_id, tokens)


# -- Resume enrichment --


def _run_enrichment_resume_codex(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    """Resume enrichment via Codex backend after questions were answered."""
    return asyncio.run(_run_enrichment_resume_codex_async(conn, plan))


async def _run_enrichment_resume_codex_async(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    from agm.backends import (
        build_enrichment_resume_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    answered = list_plan_questions(conn, plan["id"], unanswered_only=False)
    answered_with_answers = [q for q in answered if q.get("answer")]
    prompt = build_enrichment_resume_prompt(answered_with_answers)

    pmc = _load_project_model_config(conn, plan["project_id"])
    thread_config = get_runtime_thread_config(plan["backend"], "prompt_enrichment", pmc)
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    turn_config = get_runtime_turn_config(plan["backend"], "prompt_enrichment", pmc)
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="enrichment",
        plan_id=plan["id"],
        project=project["name"],
        conn=conn,
    )
    approval_policy = _load_project_app_server_approval_policy(conn, plan["project_id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))
        if plan.get("session_id"):
            sender = f"enrichment:{plan['id'][:8]}"

            def _post_to_channel(args: dict[str, Any]) -> None:
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

            if callable(set_handler):
                set_handler(
                    _make_server_request_handler(
                        channel_poster=_post_to_channel,
                        trace_context=trace_ctx,
                        approval_policy=approval_policy,
                    )
                )
            thread_config.setdefault("dynamicTools", []).append(CHANNEL_TOOL_SPEC)
            dev_inst = thread_config.get("developerInstructions", "")
            thread_config["developerInstructions"] = dev_inst + CHANNEL_TOOL_INSTRUCTIONS

        def on_thread_ready(thread_id: str) -> None:
            log.info("Resumed enrichment thread %s for plan %s", thread_id, plan["id"])

        thread_id, output_text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            resume_thread_id=plan["enrichment_thread_id"],
            resume_thread_params={
                "model": thread_config["model"],
                "approvalPolicy": thread_config["approvalPolicy"],
            },
            on_thread_ready=on_thread_ready,
            trace_context=trace_ctx,
        )

    return _process_enrichment_output(conn, plan, output_text, thread_id, tokens)


# -- Continuation enrichment --


def _summarize_task_status_counts(tasks: list[TaskRow]) -> str:
    status_counts: dict[str, int] = {}
    for task in tasks:
        status = task.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return ", ".join(f"{count} {status}" for status, count in sorted(status_counts.items()))


def _append_files_summary(line: str, files: object) -> str:
    parsed_files: Any = files
    if parsed_files and isinstance(parsed_files, str):
        try:
            parsed_files = json.loads(parsed_files)
        except (json.JSONDecodeError, TypeError):
            parsed_files = []
    if not parsed_files:
        return line
    line += f", files: {', '.join(parsed_files[:5])}"
    if len(parsed_files) > 5:
        line += f" (+{len(parsed_files) - 5} more)"
    return line


def _format_parent_task_outcome_line(task: TaskRow) -> str:
    priority = task.get("priority") or "medium"
    title = task.get("title", "untitled")
    status = task.get("status", "unknown")
    line = f'  - "{title}" — status: {status}, priority: {priority}'
    return _append_files_summary(line, task.get("files"))


def _build_parent_task_outcomes_summary(conn: sqlite3.Connection, parent_plan_id: str) -> str:
    """Build a summary of task outcomes from a parent plan."""
    tasks = list_tasks(conn, plan_id=parent_plan_id)
    if not tasks:
        return "No tasks were created for the previous plan."

    lines = [f"Summary: {_summarize_task_status_counts(tasks)}"]
    lines.extend(_format_parent_task_outcome_line(task) for task in tasks)
    return "\n".join(lines)


def _run_enrichment_continuation_codex(conn: sqlite3.Connection, plan: PlanRow) -> str | None:
    """Run continuation enrichment via Codex backend."""
    return asyncio.run(_run_enrichment_continuation_codex_async(conn, plan))


async def _run_enrichment_continuation_codex_async(
    conn: sqlite3.Connection, plan: PlanRow
) -> str | None:
    from agm.backends import (
        build_enrichment_continuation_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    assert plan["parent_id"] is not None
    parent = get_plan_request(conn, plan["parent_id"])
    task_outcomes = _build_parent_task_outcomes_summary(conn, plan["parent_id"]) if parent else None

    prompt = build_enrichment_continuation_prompt(
        raw_prompt=plan["prompt"],
        parent_enriched_prompt=parent.get("enriched_prompt") if parent else None,
        parent_plan_text=parent.get("plan") if parent else None,
        task_outcomes_summary=task_outcomes,
    )

    parent_enrichment_thread = parent.get("enrichment_thread_id") if parent else None

    pmc = _load_project_model_config(conn, plan["project_id"])
    thread_config = get_runtime_thread_config(plan["backend"], "prompt_enrichment", pmc)
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    turn_config = get_runtime_turn_config(plan["backend"], "prompt_enrichment", pmc)
    _merge_developer_instructions(thread_config, project["dir"], "enrichment")
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="enrichment",
        plan_id=plan["id"],
        project=project["name"],
        conn=conn,
    )
    approval_policy = _load_project_app_server_approval_policy(conn, plan["project_id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))
        if plan.get("session_id"):
            sender = f"enrichment:{plan['id'][:8]}"

            def _post_to_channel(args: dict[str, Any]) -> None:
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

            if callable(set_handler):
                set_handler(
                    _make_server_request_handler(
                        channel_poster=_post_to_channel,
                        trace_context=trace_ctx,
                        approval_policy=approval_policy,
                    )
                )
            thread_config.setdefault("dynamicTools", []).append(CHANNEL_TOOL_SPEC)
            dev_inst = thread_config.get("developerInstructions", "")
            thread_config["developerInstructions"] = dev_inst + CHANNEL_TOOL_INSTRUCTIONS

        def on_thread_ready(thread_id: str) -> None:
            log.info(
                "Continuation enrichment thread %s for plan %s",
                thread_id,
                plan["id"],
            )

        turn_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "turn_config": turn_config,
            "on_thread_ready": on_thread_ready,
            "trace_context": trace_ctx,
        }
        if parent_enrichment_thread:
            turn_kwargs["resume_thread_id"] = parent_enrichment_thread
            turn_kwargs["resume_thread_params"] = {
                "model": thread_config["model"],
                "approvalPolicy": thread_config["approvalPolicy"],
            }
        else:
            turn_kwargs["start_thread_params"] = {
                "cwd": project["dir"],
                **thread_config,
            }

        thread_id, output_text, tokens = await _codex_turn(client, **turn_kwargs)

    return _process_enrichment_output(conn, plan, output_text, thread_id, tokens)


def on_enrichment_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when an enrichment job fails. Marks plan as failed in DB."""
    plan_id = job.args[0] if job.args else None
    if plan_id:
        with connect() as conn:
            plan = get_plan_request(conn, plan_id)
            if not plan:
                log.warning(
                    "Plan %s enrichment failure callback skipped: not found",
                    plan_id,
                )
                return
            update_prompt_status(conn, plan_id, "failed")
            update_plan_request_status(conn, plan_id, "failed", record_history=True)
            pname = _resolve_project_name(conn, plan["project_id"])
            _emit("plan:status", plan_id, "failed", project=pname)
            add_plan_log(
                conn,
                plan_id=plan_id,
                level="ERROR",
                message=f"Enrichment failed: {exc_value}",
            )
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"enrichment:{plan_id[:8]}",
                content=f"Enrichment failed: {exc_value}",
                metadata={
                    "phase": "enrichment",
                    "status": "failed",
                    "plan_id": plan_id,
                    "error": str(exc_value),
                },
            )
            _maybe_complete_session(conn, plan_id)
        log.warning("Plan %s enrichment failed: %s", plan_id, exc_value)
