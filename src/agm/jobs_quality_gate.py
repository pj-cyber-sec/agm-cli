"""Quality gate functions.

Default gate, loading, running checks, generating configs via LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from agm.db import (
    ProjectRow,
    connect,
    get_project,
    parse_app_server_approval_policy,
    parse_app_server_ask_for_approval,
)
from agm.jobs_common import (
    _codex_client,
    _codex_turn,
    _load_project_model_config,
    _make_server_request_handler,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------


@dataclass
class QualityCheckResult:
    """Result of a single quality gate check."""

    name: str
    passed: bool
    output: str
    duration_ms: int


@dataclass
class QualityGateResult:
    """Full result of running the quality gate."""

    auto_fix_ran: bool
    auto_fix_committed: bool
    checks: list[QualityCheckResult]

    @property
    def failures(self) -> list[QualityCheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _serialize_quality_gate_result(result: QualityGateResult) -> dict:
    """Serialize QualityGateResult to JSON-safe dict for storage."""
    return {
        "auto_fix_ran": result.auto_fix_ran,
        "auto_fix_committed": result.auto_fix_committed,
        "passed": result.passed,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "output": c.output[:500],
                "duration_ms": c.duration_ms,
            }
            for c in result.checks
        ],
    }


def _default_quality_gate() -> dict:
    """Return the default quality gate config.

    Empty — agm makes no assumptions about project tooling.
    """
    return {"auto_fix": [], "checks": []}


def _load_quality_gate(quality_gate_json: str | None) -> dict:
    """Load quality gate config from JSON string, or return defaults."""
    if not quality_gate_json:
        return _default_quality_gate()

    try:
        config = json.loads(quality_gate_json)
        if not isinstance(config, dict):
            return _default_quality_gate()
        # Validate structure: must have "checks" list
        if "checks" not in config or not isinstance(config["checks"], list):
            return _default_quality_gate()
        return config
    except (json.JSONDecodeError, TypeError):
        return _default_quality_gate()


def _run_quality_checks(worktree: str, quality_gate_json: str | None = None) -> QualityGateResult:
    """Run quality gate checks in a worktree.

    If quality_gate_json is provided (from project config), uses those commands.
    Otherwise uses the default empty gate (no checks — agm makes no assumptions
    about project tooling).

    Auto-fixes first (if configured), commits if changed,
    then runs strict checks.

    Returns a ``QualityGateResult`` with per-check pass/fail, output,
    timing, and auto-fix metadata.
    """
    import subprocess

    config = _load_quality_gate(quality_gate_json)

    # Phase 1: Auto-fix commands (optional)
    auto_fix_cmds = config.get("auto_fix", [])
    auto_fix_ran = bool(auto_fix_cmds)
    for fix_cmd in auto_fix_cmds:
        cmd = fix_cmd.get("cmd", [])
        if cmd:
            subprocess.run(
                cmd,
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=fix_cmd.get("timeout", 60),
            )

    # Commit auto-fixes if anything changed
    auto_fix_committed = False
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        subprocess.run(
            ["git", "add", "."],
            cwd=worktree,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Auto-format: quality gate auto-fix"],
            cwd=worktree,
            capture_output=True,
        )
        auto_fix_committed = True

    # Phase 2: Strict checks
    checks: list[QualityCheckResult] = []
    for check in config.get("checks", []):
        cmd = check.get("cmd", [])
        name = check.get("name", " ".join(cmd))
        if not cmd:
            continue
        start = time.monotonic()
        result = subprocess.run(
            cmd,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=check.get("timeout", 120),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        checks.append(
            QualityCheckResult(
                name=name,
                passed=result.returncode == 0,
                output=(result.stdout + result.stderr).strip(),
                duration_ms=elapsed_ms,
            )
        )

    return QualityGateResult(
        auto_fix_ran=auto_fix_ran,
        auto_fix_committed=auto_fix_committed,
        checks=checks,
    )


# -- Quality gate generate --------------------------------------------------


def _parse_quality_gate_output(text: str) -> dict:
    """Parse and validate LLM output for quality gate generation.

    Returns dict with "auto_fix" and "checks" arrays.
    Raises ValueError on invalid output.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output is not a JSON object")
    if "checks" not in parsed or not isinstance(parsed["checks"], list):
        raise ValueError("Output missing 'checks' array")
    if "auto_fix" not in parsed or not isinstance(parsed["auto_fix"], list):
        raise ValueError("Output missing 'auto_fix' array")
    return {"auto_fix": parsed["auto_fix"], "checks": parsed["checks"]}


def generate_quality_gate(project_name_or_id: str, backend: str | None = None) -> dict:
    """Generate a quality gate config by having an LLM inspect the project.

    Returns parsed config dict (not stored — caller decides whether to apply).
    """
    with connect() as conn:
        project = get_project(conn, project_name_or_id)
        if not project:
            raise ValueError(f"Project '{project_name_or_id}' not found")

        # Resolve backend: explicit > project default > codex
        if not backend:
            backend = project.get("default_backend") or "codex"

        model_config = _load_project_model_config(conn, project["id"])

    if backend == "codex":
        return asyncio.run(_generate_quality_gate_codex(project, backend, model_config))
    else:
        raise ValueError(f"Unknown backend: {backend}")


async def _generate_quality_gate_codex(
    project: ProjectRow, backend: str, model_config: dict
) -> dict:
    from agm.backends import (
        QUALITY_GATE_GENERATE_PROMPT,
        QUALITY_GATE_GENERATE_SCHEMA,
        get_runtime_thread_config,
        get_runtime_turn_config,
    )

    # Borrow enrichment's model/effort/sandbox (think tier, read-only) but
    # override developerInstructions — QG generation is not prompt engineering.
    thread_config = get_runtime_thread_config(backend, "prompt_enrichment", model_config)
    thread_config["approvalPolicy"] = parse_app_server_ask_for_approval(
        project.get("app_server_ask_for_approval")
    )
    thread_config["developerInstructions"] = (
        "You are a project inspector. Read config files (pyproject.toml, "
        "package.json, Makefile, etc.) and produce a quality gate config "
        "as JSON matching the output schema. "
        "Keep it minimal: format, lint, typecheck, test. "
        "Do not explore broadly, write code, or spawn sub-agents."
    )
    turn_config = get_runtime_turn_config(backend, "prompt_enrichment", model_config)
    turn_config["outputSchema"] = QUALITY_GATE_GENERATE_SCHEMA

    approval_policy = parse_app_server_approval_policy(project.get("app_server_approval_policy"))
    async with _codex_client() as client:
        set_handler = getattr(client, "set_server_request_handler", None)
        if callable(set_handler):
            set_handler(_make_server_request_handler(approval_policy=approval_policy))
        _, output_text, _ = await _codex_turn(
            client,
            prompt=QUALITY_GATE_GENERATE_PROMPT,
            turn_config=turn_config,
            start_thread_params={"cwd": project["dir"], **thread_config},
        )

    return _parse_quality_gate_output(output_text)
