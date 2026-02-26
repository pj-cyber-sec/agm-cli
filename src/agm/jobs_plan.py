"""Plan request functions (planning only — enrichment is in jobs_enrichment.py).

run_plan_request, on_plan_request_failure, codex planner backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3

from agm.db import (
    add_plan_log,
    connect,
    finalize_plan_request,
    get_plan_request,
    get_project,
    set_plan_model,
    set_plan_request_thread_id,
    set_plan_request_worker,
    update_plan_request_status,
    update_plan_tokens,
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
    _resolve_project_model_config,
    _resolve_project_name,
)

log = logging.getLogger(__name__)


def run_plan_request(plan_id: str) -> str:
    """Execute planning via the appropriate backend.

    Called by rq worker on the agm:plans queue after enrichment completes.
    Expects prompt_status='finalized' and enriched_prompt to be set.

    Status flow: pending → running → finalized | failed
    """
    with connect() as conn:
        plan = get_plan_request(conn, plan_id)
        if not plan:
            log.warning("Plan %s not found (deleted?), skipping", plan_id)
            return "skipped:entity_missing"

        if plan["status"] == "cancelled":
            log.info("Plan %s already cancelled, skipping", plan_id)
            return "skipped:cancelled"

        db_handler = PlanDBHandler(conn, plan_id)
        db_handler.setLevel(logging.DEBUG)
        log.addHandler(db_handler)
        prev_log_level = log.level
        if log.level > logging.DEBUG or log.level == logging.NOTSET:
            log.setLevel(logging.DEBUG)

        project_name = _resolve_project_name(conn, plan["project_id"])

        pid = os.getpid()
        set_plan_request_worker(conn, plan_id, pid=pid)
        # Only set running if not already running (enrichment may have set it)
        if plan["status"] != "running":
            update_plan_request_status(conn, plan_id, "running", record_history=True)
            _emit("plan:status", plan_id, "running", project=project_name)
        set_plan_model(
            conn,
            plan_id,
            _resolve_project_model_config(conn, plan["project_id"], plan["backend"])["think_model"],
        )
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"planner:{plan_id[:8]}",
            content="Planning started",
            metadata={
                "phase": "planning",
                "status": "running",
                "plan_id": plan_id,
                "backend": plan["backend"],
            },
        )
        log.info(
            "Planner pid=%d started plan %s (backend=%s)",
            pid,
            plan_id,
            plan["backend"],
        )

        try:
            backend = plan["backend"]

            # Use enriched prompt if available, otherwise raw prompt
            planner_prompt = plan.get("enriched_prompt") or plan["prompt"]
            plan = dict(plan)  # mutable copy
            plan["prompt"] = planner_prompt

            if backend == "codex":
                result = _run_plan_request_codex(conn, plan)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            return result
        except Exception:
            log.exception("Plan %s failed", plan_id)
            update_plan_request_status(conn, plan_id, "failed", record_history=True)
            _emit("plan:status", plan_id, "failed", project=project_name)
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def _run_plan_request_codex(conn: sqlite3.Connection, plan: dict) -> str:
    """Run a plan via the Codex backend."""
    return asyncio.run(_run_plan_request_codex_async(conn, plan))


async def _run_plan_request_codex_async(conn: sqlite3.Connection, plan: dict) -> str:
    """Async implementation of codex plan execution."""
    from agm.backends import (
        build_plan_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    # Check if this is a continuation of a previous plan
    parent_thread_id = None
    if plan.get("parent_id"):
        parent = get_plan_request(conn, plan["parent_id"])
        if parent and parent.get("thread_id"):
            parent_thread_id = parent["thread_id"]
            log.info(
                "Plan %s continues parent %s (thread %s)",
                plan["id"],
                plan["parent_id"],
                parent_thread_id,
            )
        else:
            log.warning(
                "Plan %s has parent_id %s but parent has no thread_id, starting fresh",
                plan["id"],
                plan.get("parent_id"),
            )

    pmc = _load_project_model_config(conn, plan["project_id"])
    thread_config = get_runtime_thread_config(plan["backend"], "plan_request", pmc)
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    turn_config = get_runtime_turn_config(plan["backend"], "plan_request", pmc)
    prompt = build_plan_prompt(plan["prompt"], exploration_context=plan.get("exploration_context"))
    _merge_developer_instructions(thread_config, project["dir"], "planner")
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="planning",
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
            plan_id_short = str(plan["id"])[:8]

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
                    sender=f"planner:{plan_id_short}",
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

        if parent_thread_id:

            def on_thread_ready(thread_id: str) -> None:
                set_plan_request_thread_id(conn, plan["id"], thread_id)
                log.info("Resumed thread %s for plan %s", thread_id, plan["id"])

            _, plan_text, tokens = await _codex_turn(
                client,
                prompt=prompt,
                turn_config=turn_config,
                resume_thread_id=parent_thread_id,
                resume_thread_params={
                    "model": thread_config["model"],
                    "approvalPolicy": thread_config["approvalPolicy"],
                },
                on_thread_ready=on_thread_ready,
                trace_context=trace_ctx,
            )
        else:

            def on_thread_ready(thread_id: str) -> None:
                set_plan_request_thread_id(conn, plan["id"], thread_id)
                log.info("Started thread %s for plan %s", thread_id, plan["id"])

            _, plan_text, tokens = await _codex_turn(
                client,
                prompt=prompt,
                turn_config=turn_config,
                start_thread_params={"cwd": project["dir"], **thread_config},
                on_thread_ready=on_thread_ready,
                trace_context=trace_ctx,
            )

        update_plan_tokens(conn, plan["id"], **tokens)

        if plan_text:
            finalize_plan_request(conn, plan["id"], plan_text, record_history=True)
            _emit("plan:status", plan["id"], "finalized", project=project["name"])
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"planner:{plan['id'][:8]}",
                content="Plan finalized",
                metadata={
                    "phase": "planning",
                    "status": "finalized",
                    "plan_id": plan["id"],
                    "plan_chars": len(plan_text),
                },
            )
            log.info("Plan %s finalized (%d chars)", plan["id"], len(plan_text))
            from agm.jobs_task_creation import _trigger_task_creation

            _trigger_task_creation(conn, plan["id"])
            return plan_text
        raise RuntimeError("No plan text produced by backend")


def on_plan_request_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when a plan job fails. Marks plan as failed in DB."""
    plan_id = job.args[0] if job.args else None
    if plan_id:
        with connect() as conn:
            plan = get_plan_request(conn, plan_id)
            if not plan:
                log.warning("Plan %s failed callback skipped: plan not found", plan_id)
                return
            update_plan_request_status(conn, plan_id, "failed", record_history=True)
            pname = _resolve_project_name(conn, plan["project_id"])
            _emit("plan:status", plan_id, "failed", project=pname)
            add_plan_log(
                conn,
                plan_id=plan_id,
                level="ERROR",
                message=f"Plan failed: {exc_value}",
            )
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"planner:{plan_id[:8]}",
                content=f"Plan failed: {exc_value}",
                metadata={
                    "phase": "planning",
                    "status": "failed",
                    "plan_id": plan_id,
                    "error": str(exc_value),
                },
            )
            _maybe_complete_session(conn, plan_id)
        log.warning("Plan %s failed: %s", plan_id, exc_value)
