"""Codebase exploration job functions.

run_explorer, on_explorer_failure, and exploration processing helpers.

The explorer runs between enrichment and planning. It does one thorough
codebase exploration guided by the enriched prompt, stores structured
findings on the plan row, posts them to the session channel, and hands
off to the planner. Explorer failure is non-fatal — the planner still
runs without exploration context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3

from agm.db import (
    PlanRow,
    add_plan_log,
    connect,
    get_plan_request,
    get_project,
    set_plan_request_worker,
    update_plan_exploration,
    update_plan_tokens,
)
from agm.jobs_common import (
    CHANNEL_TOOL_INSTRUCTIONS,
    CHANNEL_TOOL_SPEC,
    PlanDBHandler,
    _apply_project_app_server_ask_for_approval,
    _codex_client,
    _codex_turn,
    _load_project_app_server_approval_policy,
    _load_project_model_config,
    _make_server_request_handler,
    _merge_developer_instructions,
    _post_channel_message,
    _resolve_project_name,
)

log = logging.getLogger(__name__)
EXPLORER_TURN_TIMEOUT_SECONDS = 900


def _format_exception_message(exc: BaseException) -> str:
    """Return stable exception text with type for observability surfaces."""
    detail = str(exc).strip()
    exc_type = type(exc).__name__
    if detail:
        return f"{exc_type}: {detail}"
    return exc_type


def run_explorer(plan_id: str) -> str:
    """Execute codebase exploration for a plan.

    Called by rq worker on the agm:explorer queue after enrichment
    completes. Explores the codebase and stores findings for the planner.

    Explorer failure is non-fatal: on any error, the planner is still
    enqueued with the enriched prompt (same as if explorer didn't exist).
    """
    with connect() as conn:
        plan = get_plan_request(conn, plan_id)
        if not plan:
            log.warning("Plan %s not found (deleted?), skipping", plan_id)
            return "skipped:entity_missing"

        if plan["status"] == "cancelled":
            log.info("Plan %s already cancelled, skipping", plan_id)
            return "skipped:cancelled"

        db_handler = PlanDBHandler(conn, plan_id, source="explorer")
        db_handler.setLevel(logging.DEBUG)
        log.addHandler(db_handler)
        prev_log_level = log.level
        if log.level > logging.DEBUG or log.level == logging.NOTSET:
            log.setLevel(logging.DEBUG)

        _resolve_project_name(conn, plan["project_id"])  # validate project exists
        pid = os.getpid()
        set_plan_request_worker(conn, plan_id, pid=pid)
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"explorer:{plan_id[:8]}",
            content="Codebase exploration started",
            metadata={
                "phase": "exploration",
                "status": "running",
                "plan_id": plan_id,
                "backend": plan["backend"],
            },
        )
        log.info(
            "Explorer pid=%d started for plan %s (backend=%s)",
            pid,
            plan_id,
            plan["backend"],
        )

        try:
            enriched_prompt = plan.get("enriched_prompt") or plan["prompt"]
            plan_dict = dict(plan)
            plan_dict["prompt"] = enriched_prompt

            output_text = _run_explorer_codex(conn, plan_dict)
            exploration_context = _process_exploration_output(conn, plan, output_text)

            if exploration_context:
                _post_channel_message(
                    conn,
                    plan,
                    kind="context",
                    sender=f"explorer:{plan_id[:8]}",
                    content=_format_exploration_for_channel(json.loads(exploration_context)),
                    metadata={
                        "phase": "exploration",
                        "status": "completed",
                        "plan_id": plan_id,
                    },
                )
                log.info(
                    "Explorer for plan %s completed (%d chars)",
                    plan_id,
                    len(exploration_context),
                )
            else:
                log.warning(
                    "Explorer for plan %s produced no usable output",
                    plan_id,
                )

            # Always enqueue planner after exploration
            from agm.queue import enqueue_plan_request

            enqueue_plan_request(plan_id)
            return "exploration_completed"

        except Exception as exc:
            error_text = _format_exception_message(exc)
            log.error(
                "Explorer for plan %s failed, falling back to planner: %s",
                plan_id,
                error_text,
                exc_info=True,
            )
            # Non-fatal: enqueue planner anyway
            from agm.queue import enqueue_plan_request

            enqueue_plan_request(plan_id)
            raise
        finally:
            log.removeHandler(db_handler)
            log.setLevel(prev_log_level)


def _run_explorer_codex(conn: sqlite3.Connection, plan: dict) -> str:
    """Run exploration via the Codex backend."""
    return asyncio.run(_run_explorer_codex_async(conn, plan))


async def _run_explorer_codex_async(conn: sqlite3.Connection, plan: dict) -> str:
    """Async implementation of codex exploration."""
    from agm.backends import (
        EXPLORATION_OUTPUT_SCHEMA,
        build_exploration_prompt,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )
    from agm.tracing import TraceContext

    project = get_project(conn, plan["project_id"])
    if not project:
        raise ValueError(f"Project {plan['project_id']} not found")

    pmc = _load_project_model_config(conn, plan["project_id"])
    thread_config = get_runtime_thread_config(plan["backend"], "codebase_exploration", pmc)
    _apply_project_app_server_ask_for_approval(conn, plan["project_id"], thread_config)
    turn_config = get_runtime_turn_config(plan["backend"], "codebase_exploration", pmc)
    turn_config["outputSchema"] = EXPLORATION_OUTPUT_SCHEMA
    prompt = build_exploration_prompt(plan["prompt"])
    _merge_developer_instructions(thread_config, project["dir"], "explorer")
    trace_ctx = TraceContext(
        entity_type="plan",
        entity_id=plan["id"],
        stage="exploration",
        plan_id=plan["id"],
        project=project["name"],
        conn=conn,
    )

    def on_thread_ready(thread_id: str) -> None:
        update_plan_exploration(conn, plan["id"], exploration_thread_id=thread_id)
        log.info("Started exploration thread %s for plan %s", thread_id, plan["id"])

    approval_policy = _load_project_app_server_approval_policy(conn, plan["project_id"])

    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))

        if plan.get("session_id"):
            sender = f"explorer:{plan['id'][:8]}"

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

        _, output_text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
            on_thread_ready=on_thread_ready,
            trace_context=trace_ctx,
            turn_timeout=EXPLORER_TURN_TIMEOUT_SECONDS,
        )

    update_plan_tokens(conn, plan["id"], **tokens)

    if not output_text:
        raise RuntimeError("Explorer produced no output")

    return output_text


def _process_exploration_output(
    conn: sqlite3.Connection,
    plan: PlanRow,
    output_text: str,
) -> str | None:
    """Process exploration output: validate JSON, store on plan row.

    Returns the raw JSON string, or None if parsing failed.
    """
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("Explorer output not valid JSON: %s", e)
        return None

    # Validate required fields
    required = {"summary", "architecture", "relevant_files"}
    if not required.issubset(data.keys()):
        log.warning(
            "Explorer output missing required fields: %s",
            required - data.keys(),
        )
        return None

    # Store exploration context on plan row
    update_plan_exploration(conn, plan["id"], exploration_context=output_text)
    return output_text


def _format_exploration_for_channel(data: dict) -> str:
    """Format structured exploration data as readable text for the channel."""
    lines: list[str] = []

    summary = data.get("summary", "")
    if summary:
        lines.append(f"Codebase exploration: {summary}")

    architecture = data.get("architecture", "")
    if architecture:
        lines.append(f"\nArchitecture: {architecture}")

    relevant_files = data.get("relevant_files", [])
    if relevant_files:
        lines.append("\nRelevant files:")
        for f in relevant_files:
            symbols = ", ".join(f.get("key_symbols", []))
            suffix = f" (key: {symbols})" if symbols else ""
            lines.append(f"  - {f['path']}: {f['description']}{suffix}")

    patterns = data.get("patterns_to_follow", [])
    if patterns:
        lines.append("\nPatterns to follow:")
        for p in patterns:
            lines.append(f"  - {p}")

    helpers = data.get("reusable_helpers", [])
    if helpers:
        lines.append("\nReusable helpers:")
        for h in helpers:
            lines.append(f"  - {h['path']}:{h['symbol']} — {h['description']}")

    test_locations = data.get("test_locations", [])
    if test_locations:
        lines.append("\nTest locations:")
        for t in test_locations:
            lines.append(f"  - {t}")

    return "\n".join(lines)


def on_explorer_failure(job, _connection, _exc_type, exc_value, _traceback):
    """Callback when an explorer job fails.

    Non-fatal: logs the failure but does NOT mark the plan as failed.
    The planner is enqueued by the main run_explorer function even on
    exception (before the rq failure callback fires).
    """
    plan_id = job.args[0] if job.args else None
    if plan_id:
        error_text = _format_exception_message(exc_value)
        with connect() as conn:
            plan = get_plan_request(conn, plan_id)
            if not plan:
                return
            add_plan_log(
                conn,
                plan_id=plan_id,
                level="WARNING",
                message=f"Exploration failed (non-fatal): {error_text}",
            )
            _post_channel_message(
                conn,
                plan,
                kind="broadcast",
                sender=f"explorer:{plan_id[:8]}",
                content=f"Exploration failed (proceeding without): {error_text}",
                metadata={
                    "phase": "exploration",
                    "status": "failed",
                    "plan_id": plan_id,
                    "error": error_text,
                    "error_type": type(exc_value).__name__,
                },
            )
        log.warning("Explorer for plan %s failed: %s", plan_id, error_text)
