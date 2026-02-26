"""Project setup agent — auto-configure quality gate + post-merge via LLM inspection."""

from __future__ import annotations

import asyncio
import json
import logging

from agm.db import (
    ProjectRow,
    connect,
    get_project,
    parse_app_server_approval_policy,
    parse_app_server_ask_for_approval,
    set_project_post_merge_command,
    set_project_quality_gate,
    set_project_setup_result,
)
from agm.jobs_common import (
    _codex_client,
    _codex_turn,
    _load_project_model_config,
    _make_server_request_handler,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output parsing / validation
# ---------------------------------------------------------------------------


def _parse_setup_output(text: str) -> dict:
    """Parse and validate LLM output for project setup.

    Returns dict with ``quality_gate``, ``post_merge_command``, ``stack``,
    ``warnings``.  Raises ``ValueError`` on invalid output.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output is not a JSON object")

    # quality_gate
    qg = parsed.get("quality_gate")
    if not isinstance(qg, dict):
        raise ValueError("Output missing 'quality_gate' object")
    if not isinstance(qg.get("auto_fix"), list):
        raise ValueError("quality_gate missing 'auto_fix' array")
    if not isinstance(qg.get("checks"), list):
        raise ValueError("quality_gate missing 'checks' array")

    # stack
    stack = parsed.get("stack")
    if not isinstance(stack, dict):
        raise ValueError("Output missing 'stack' object")
    if not isinstance(stack.get("languages"), list):
        raise ValueError("stack missing 'languages' array")
    if not isinstance(stack.get("tools"), list):
        raise ValueError("stack missing 'tools' array")

    # warnings
    if not isinstance(parsed.get("warnings"), list):
        raise ValueError("Output missing 'warnings' array")

    # post_merge_command: string | null — accept missing as None
    if "post_merge_command" not in parsed:
        parsed["post_merge_command"] = None

    # reasoning: array — accept missing for backward compat
    reasoning = parsed.get("reasoning", [])
    if not isinstance(reasoning, list):
        reasoning = []

    return {
        "quality_gate": {"auto_fix": qg["auto_fix"], "checks": qg["checks"]},
        "post_merge_command": parsed["post_merge_command"],
        "stack": {
            "languages": stack["languages"],
            "package_manager": stack.get("package_manager"),
            "tools": stack["tools"],
        },
        "warnings": parsed["warnings"],
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Apply results to project
# ---------------------------------------------------------------------------


def _apply_setup_result(
    project_id: str,
    result: dict,
    *,
    dry_run: bool = False,
    project_dir: str | None = None,
) -> dict:
    """Apply setup result to the project row.

    Returns the result dict augmented with ``quality_gate_applied`` and
    ``post_merge_applied`` booleans.
    """
    applied = {"quality_gate_applied": False, "post_merge_applied": False}

    if not dry_run:
        qg = result.get("quality_gate")
        has_qg = qg and (qg.get("auto_fix") or qg.get("checks"))
        pmc = result.get("post_merge_command")

        with connect() as conn:
            if has_qg:
                set_project_quality_gate(conn, project_id, json.dumps(qg))
                applied["quality_gate_applied"] = True

            if pmc:
                set_project_post_merge_command(conn, project_id, pmc)
                applied["post_merge_applied"] = True

            # Store full setup result for traceability
            set_project_setup_result(conn, project_id, json.dumps(result))

        # Create .agm/agents.toml scaffold if missing
        if project_dir:
            _ensure_agents_toml(project_dir)

    return {**result, **applied}


def _ensure_agents_toml(project_dir: str) -> bool:
    """Create .agm/agents.toml scaffold if it doesn't exist. Returns True if created."""
    from pathlib import Path

    from agm.agents_config import build_agents_toml_scaffold

    agm_dir = Path(project_dir) / ".agm"
    agents_toml = agm_dir / "agents.toml"
    if agents_toml.exists():
        return False

    try:
        agm_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        agents_toml.write_text(build_agents_toml_scaffold())
        return True
    except OSError as exc:
        log.debug("Failed to create agents.toml for %s: %s", project_dir, exc)
        return False


# ---------------------------------------------------------------------------
# Codex backend
# ---------------------------------------------------------------------------


async def _run_project_setup_codex(project: ProjectRow, backend: str, model_config: dict) -> dict:
    from agm.backends import (
        PROJECT_SETUP_PROMPT,
        PROJECT_SETUP_SCHEMA,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )

    # Borrow enrichment's model/effort/sandbox (think tier, read-only).
    thread_config = get_runtime_thread_config(backend, "prompt_enrichment", model_config)
    thread_config["approvalPolicy"] = parse_app_server_ask_for_approval(
        project.get("app_server_ask_for_approval")
    )
    thread_config["developerInstructions"] = (
        "You are a project setup inspector. Read config files and build "
        "pipeline configuration as JSON matching the output schema. "
        "Keep it minimal and accurate: format, lint, typecheck, test, "
        "post-merge. Do not explore broadly, write code, or spawn sub-agents."
    )
    turn_config = get_runtime_turn_config(backend, "prompt_enrichment", model_config)
    turn_config["outputSchema"] = PROJECT_SETUP_SCHEMA

    approval_policy = parse_app_server_approval_policy(project.get("app_server_approval_policy"))
    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))
        _, output_text, _ = await _codex_turn(
            client,
            prompt=PROJECT_SETUP_PROMPT,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
        )

    return _parse_setup_output(output_text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_project_setup(
    project_name_or_id: str,
    *,
    backend: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the project setup agent: inspect project, auto-apply configuration.

    Returns a dict with ``quality_gate``, ``post_merge_command``, ``stack``,
    ``warnings``, and ``quality_gate_applied``/``post_merge_applied`` booleans.
    """
    with connect() as conn:
        project = get_project(conn, project_name_or_id)
        if not project:
            raise ValueError(f"Project '{project_name_or_id}' not found")

        if not backend:
            backend = project.get("default_backend") or "codex"

        model_config = _load_project_model_config(conn, project["id"])

    if backend != "codex":
        raise ValueError(f"Unknown backend: {backend}")

    result = asyncio.run(_run_project_setup_codex(project, backend, model_config))
    return _apply_setup_result(project["id"], result, dry_run=dry_run, project_dir=project["dir"])


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _emit_setup(
    status: str,
    project_id: str,
    project_name: str,
    *,
    job_id: str | None = None,
    extra: dict | None = None,
) -> None:
    """Emit ``project:setup`` event to Redis Stream (best-effort)."""
    from agm.queue import publish_event

    payload_extra = dict(extra or {})
    if job_id:
        payload_extra["job_id"] = job_id

    publish_event(
        "project:setup",
        project_id,
        status,
        project=project_name,
        extra=payload_extra or None,
    )


# ---------------------------------------------------------------------------
# Worker entry point (rq)
# ---------------------------------------------------------------------------


def run_project_setup_worker(
    project_id: str, project_name: str, backend: str | None = None
) -> dict:
    """rq worker entry point for async project setup.

    Called on the ``agm:setup`` queue.  Runs the LLM inspector, applies
    the result, and emits ``project:setup`` events throughout.
    """
    from rq import get_current_job

    current_job = get_current_job()
    job_id = current_job.id if current_job else None

    _emit_setup("running", project_id, project_name, job_id=job_id)
    log.info("Setup job started for project %s (%s)", project_name, project_id)

    try:
        result = run_project_setup(project_id, backend=backend)
        _emit_setup(
            "completed",
            project_id,
            project_name,
            job_id=job_id,
            extra={
                "quality_gate_applied": result.get("quality_gate_applied", False),
                "post_merge_applied": result.get("post_merge_applied", False),
                "warnings": result.get("warnings", []),
            },
        )
        log.info("Setup job completed for project %s", project_name)
        return result
    except Exception as exc:
        log.exception("Setup job failed for project %s", project_name)
        _emit_setup("failed", project_id, project_name, job_id=job_id, extra={"error": str(exc)})
        raise


def on_setup_failure(job, _connection, _exc_type, exc_value, _traceback):  # type: ignore[no-untyped-def]
    """rq failure callback — emit ``project:setup`` failed event."""
    project_id = job.args[0] if job.args else None
    project_name = job.args[1] if job.args and len(job.args) > 1 else ""
    if project_id:
        job_id = getattr(job, "id", None)
        _emit_setup(
            "failed",
            project_id,
            project_name,
            job_id=str(job_id) if job_id is not None else None,
            extra={"error": str(exc_value)},
        )
