"""Shared infrastructure for job modules.

Constants, helpers, DB handler classes, codex infrastructure,
task helpers, and project memory utilities used across all job modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, cast

from agm.agents_config import get_effective_role_config
from agm.db import (
    PLAN_TERMINAL_STATUSES,
    TASK_TERMINAL_STATUSES,
    PlanRow,
    ProjectRow,
    TaskRow,
    add_channel_message,
    add_plan_log,
    add_task_log,
    finish_session,
    get_plan_request,
    get_project,
    get_project_app_server_approval_policy,
    get_project_app_server_ask_for_approval,
    get_project_base_branch,
    list_plan_requests,
    list_task_logs,
    list_tasks,
    normalize_app_server_approval_policy,
    normalize_app_server_ask_for_approval,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REJECTIONS = 3
MAX_DIFF_CHARS = 30_000  # Truncate large diffs in reviewer prompt
MAX_COMMIT_NUDGES = 2
_COMMIT_NUDGE = (
    "You have uncommitted changes in the worktree. "
    "Please `git add` all relevant files and `git commit` with a clear message. "
    "Do not leave any work uncommitted — a reviewer agent will inspect your commits next."
)
_TASK_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
_PROJECT_INSTRUCTIONS_SECTION_DELIMITER = "\n\n--- Project-specific instructions ---\n"

# -- Channel tool (dynamicTools spec for agent → channel posting) ----------

CHANNEL_TOOL_SPEC = {
    "name": "post_channel_message",
    "description": (
        "Post a message to the shared communication channel for this plan. "
        "Other agents working on tasks in the same plan can see these messages. "
        "Use this to share important decisions, interface definitions, "
        "new APIs or types you created, architectural choices, or warnings "
        "about constraints you discovered."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message to share with other agents.",
            },
            "kind": {
                "type": "string",
                "enum": ["broadcast", "context", "dm", "steer", "question"],
                "description": (
                    "Message kind. Use broadcast for status, context for findings, "
                    "steer for guidance, question for questions, dm for targeted messages."
                ),
            },
            "recipient": {
                "type": "string",
                "description": "Optional target agent in role:id format for dm/steer/question.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional structured metadata (files, tools, token counts, etc.).",
                "additionalProperties": True,
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    },
}

CHANNEL_TOOL_INSTRUCTIONS = (
    "\n\nYou have a post_channel_message tool. Use it to share important "
    "context with other agents working on the same plan — interface "
    "definitions, new types or APIs you created, architectural decisions, "
    "or constraints you discovered. Prefer kinds intentionally: "
    "broadcast=status updates, context=findings, steer=guidance, "
    "question=questions, dm=targeted notes. Include structured metadata "
    "when useful (files, commands, tools, token counts). Post once per "
    "significant decision, not after every file edit."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(
    event_type: str,
    entity_id: str,
    status: str,
    *,
    project: str,
    plan_id: str | None = None,
    extra: dict | None = None,
) -> None:
    """Emit pipeline event. Best-effort wrapper around queue.publish_event."""
    from agm.queue import publish_event

    publish_event(event_type, entity_id, status, project=project, plan_id=plan_id, extra=extra)


def _post_channel_message(
    conn: sqlite3.Connection,
    plan: PlanRow | dict,
    *,
    kind: str,
    sender: str,
    content: str,
    recipient: str | None = None,
    metadata: str | dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Best-effort post to the plan's session channel. No-op if no session."""
    session_id = plan.get("session_id")
    if not session_id:
        return None
    fallback_id = str(plan.get("id", ""))[:8] or str(session_id)[:8] or "unknown"
    sender_ref = _normalize_agent_ref(sender, fallback_id)
    recipient_ref = _normalize_agent_ref(recipient, fallback_id) if recipient else None
    normalized_kind = _normalize_channel_kind(kind)
    metadata_json = _serialize_channel_metadata(metadata)
    try:
        msg = add_channel_message(
            conn,
            session_id=session_id,
            kind=normalized_kind,
            sender=sender_ref,
            content=content,
            recipient=recipient_ref,
            metadata=metadata_json,
        )
        metadata_payload = _deserialize_channel_metadata(metadata_json)
        project = _resolve_project_name(conn, plan.get("project_id", ""))
        _emit(
            "session:message",
            session_id,
            normalized_kind,
            project=project,
            extra={
                "session_id": session_id,
                "sender": sender_ref,
                "recipient": recipient_ref,
                "kind": normalized_kind,
                "message_id": msg["id"],
                "metadata": metadata_payload,
            },
        )
        return cast(dict[str, Any], msg)
    except Exception:
        log.debug("Failed to post channel message for session %s", session_id, exc_info=True)
    return None


def _normalize_agent_ref(agent_ref: str | None, fallback_id: str) -> str:
    """Normalize agent references to role:id format."""
    raw = (agent_ref or "").strip()
    if not raw:
        return f"unknown:{fallback_id}"
    if ":" in raw:
        role, ident = raw.split(":", 1)
        role_norm = role.strip() or "unknown"
        ident_norm = ident.strip() or fallback_id
        return f"{role_norm}:{ident_norm}"
    return f"{raw}:{fallback_id}"


def _normalize_channel_kind(kind: str | None) -> str:
    """Normalize/repair message kinds before persisting channel entries."""
    raw = (kind or "").strip().lower()
    if raw in {"broadcast", "context", "dm", "steer", "question"}:
        return raw
    if raw in {"status", "update"}:
        return "broadcast"
    return "context"


def _serialize_channel_metadata(metadata: str | dict[str, Any] | None) -> str | None:
    """Serialize channel metadata to DB text column."""
    if metadata is None:
        return None
    if isinstance(metadata, str):
        return metadata
    try:
        return json.dumps(metadata, sort_keys=True)
    except Exception:
        return None


def _deserialize_channel_metadata(metadata_json: str | None) -> Any:
    """Best-effort parse for SSE payload enrichment."""
    if metadata_json is None:
        return None
    try:
        return json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return metadata_json


def _maybe_complete_session(conn: sqlite3.Connection, plan_id: str) -> None:
    """Check if all plans and tasks in a session are terminal; if so, finish it.

    Called after a plan or task reaches a terminal state.
    """

    def _is_effectively_terminal_task(task: Mapping[str, object]) -> bool:
        status = task.get("status")
        if status in TASK_TERMINAL_STATUSES:
            return True
        # do --no-merge intentionally stops at approved.
        return status == "approved" and bool(task.get("skip_merge"))

    plan = get_plan_request(conn, plan_id)
    if not plan:
        return
    session_id = plan.get("session_id")
    if not session_id:
        return

    session_plans = list_plan_requests(conn, session_id=session_id)
    all_plans_terminal = all(p["status"] in PLAN_TERMINAL_STATUSES for p in session_plans)
    if not all_plans_terminal:
        return

    # Also verify all tasks under these plans are terminal
    for sp in session_plans:
        tasks = list_tasks(conn, plan_id=sp["id"])
        if tasks and not all(_is_effectively_terminal_task(t) for t in tasks):
            return

    any_failed = any(p["status"] == "failed" for p in session_plans)
    # Check task failures too
    if not any_failed:
        for sp in session_plans:
            tasks = list_tasks(conn, plan_id=sp["id"])
            if any(t["status"] == "failed" for t in tasks):
                any_failed = True
                break
    final_status = "failed" if any_failed else "completed"
    try:
        finish_session(conn, session_id, final_status)
        project = _resolve_project_name(conn, plan.get("project_id", ""))
        _emit("session:status", session_id, final_status, project=project)
        _post_channel_message(
            conn,
            plan,
            kind="broadcast",
            sender=f"system:{str(session_id)[:8]}",
            content=f"Session {final_status}",
            metadata={
                "phase": "session",
                "status": final_status,
                "session_id": session_id,
            },
        )
    except Exception:
        log.debug("Failed to complete session %s", session_id, exc_info=True)


def _resolve_project_name(conn: sqlite3.Connection, project_id: str) -> str:
    """Resolve project name from ID. Returns empty string on failure."""
    proj = get_project(conn, project_id)
    return proj["name"] if proj else ""


def _merge_developer_instructions(thread_config: dict, project_dir: str | None, role: str) -> None:
    """Merge agents.toml instructions into thread_config developerInstructions (in-place).

    Appends to existing developerInstructions if present, or sets it if None.
    """
    instructions = get_effective_role_config(project_dir, role)
    if not instructions:
        return
    existing = thread_config.get("developerInstructions")
    if existing:
        thread_config["developerInstructions"] = (
            f"{existing}{_PROJECT_INSTRUCTIONS_SECTION_DELIMITER}{instructions}"
        )
    else:
        thread_config["developerInstructions"] = instructions


def _make_server_request_handler(
    mcp_pool: Any = None,
    channel_poster: Callable[[dict[str, Any]], None] | None = None,
    trace_context: Any | None = None,
    approval_policy: Mapping[str, object] | None = None,
) -> Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]:
    """Build a server-request handler with policy-driven approvals and tool routing.

    When *mcp_pool* is provided, ``item/tool/call`` requests are routed to
    the pool.  When *channel_poster* is provided, ``post_channel_message``
    calls are routed to it.  Approval decisions are resolved from
    ``approval_policy`` (or defaults when omitted). When *trace_context* is
    provided, handled approval requests are recorded.
    """
    effective_approval_policy = normalize_app_server_approval_policy(approval_policy)

    def _nested_lookup(payload: Any, keys: tuple[str, ...]) -> str | None:
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
            for value in payload.values():
                found = _nested_lookup(value, keys)
                if found:
                    return found
            return None
        if isinstance(payload, list):
            for item in payload:
                found = _nested_lookup(item, keys)
                if found:
                    return found
        return None

    def _load_chatgpt_auth_tokens() -> dict[str, str | None]:
        from agm.paths import CODEX_HOME

        auth_path = CODEX_HOME / "auth.json"
        try:
            raw = json.loads(auth_path.read_text())
        except (OSError, json.JSONDecodeError):
            raw = {}

        access_token = _nested_lookup(raw, ("accessToken", "access_token"))
        account_id = _nested_lookup(raw, ("chatgptAccountId", "chatgpt_account_id", "accountId"))
        plan_type = _nested_lookup(raw, ("chatgptPlanType", "chatgpt_plan_type"))
        return {
            "accessToken": access_token or "",
            "chatgptAccountId": account_id or "",
            "chatgptPlanType": plan_type,
        }

    def _approval_context(params: dict[str, Any]) -> dict[str, Any]:
        skill_metadata = params.get("skillMetadata")
        skill_info = skill_metadata if isinstance(skill_metadata, Mapping) else {}
        return {
            "thread_id": params.get("threadId")
            or params.get("thread_id")
            or params.get("conversationId"),
            "turn_id": params.get("turnId") or params.get("turn_id"),
            "item_id": params.get("itemId") or params.get("item_id") or params.get("callId"),
            "approval_id": params.get("approvalId") or params.get("approval_id"),
            "reason": params.get("reason"),
            "skill_name": params.get("skillName")
            or params.get("skill_name")
            or skill_info.get("name"),
            "skill_path": params.get("skillPath")
            or params.get("skill_path")
            or skill_info.get("path"),
            "permission_profile": params.get("permissionProfile")
            or params.get("permission_profile"),
            "additional_permissions": params.get("additionalPermissions")
            or params.get("additional_permissions"),
            "network_approval_context": params.get("networkApprovalContext")
            or params.get("network_approval_context"),
        }

    def _record_approval_request(
        method: str,
        decision: str,
        params: dict[str, Any],
    ) -> None:
        context = _approval_context(params)
        log.info(
            "Approval handled: method=%s decision=%s item_id=%s approval_id=%s "
            "skill_name=%s additional_permissions=%s",
            method,
            decision,
            context.get("item_id"),
            context.get("approval_id"),
            context.get("skill_name"),
            context.get("additional_permissions"),
        )
        if trace_context is not None and hasattr(trace_context, "record"):
            trace_context.record(
                "approvalRequest",
                decision,
                {
                    "method": method,
                    "decision": decision,
                    **context,
                },
            )

    async def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method in effective_approval_policy:
            decision = effective_approval_policy[method]
            _record_approval_request(method, decision, params)
            return {"decision": decision}

        if method == "item/tool/requestUserInput":
            return {"answers": {}}

        if method == "account/chatgptAuthTokens/refresh":
            tokens = _load_chatgpt_auth_tokens()
            if not tokens["accessToken"] or not tokens["chatgptAccountId"]:
                log.warning("ChatGPT token refresh requested but no auth tokens are available.")
            return tokens

        if method == "item/tool/call":
            tool_name = params.get("tool", "")
            arguments = params.get("arguments", {})

            if tool_name == "post_channel_message" and channel_poster is not None:
                if not isinstance(arguments, dict):
                    arguments = {}
                try:
                    channel_poster(arguments)
                    return {
                        "success": True,
                        "contentItems": [{"type": "inputText", "text": "Message posted."}],
                    }
                except Exception as e:
                    log.warning("Channel post failed: %s", e)
                    return {
                        "success": False,
                        "contentItems": [{"type": "inputText", "text": str(e)}],
                    }

            if mcp_pool is not None:
                try:
                    result = await mcp_pool.call_tool(tool_name, arguments)
                    return {
                        "success": True,
                        "contentItems": [{"type": "inputText", "text": result}],
                    }
                except Exception as e:
                    log.warning("MCP tool call failed: %s(%s): %s", tool_name, arguments, e)
                    return {
                        "success": False,
                        "contentItems": [{"type": "inputText", "text": str(e)}],
                    }

        raise ValueError(f"Unsupported server request method: {method}")

    return handler


def _get_plan_backend(conn: sqlite3.Connection, plan_or_task: PlanRow | TaskRow) -> str:
    """Resolve backend from a plan dict, or from a task dict via plan chain.

    Returns the plan's backend field, defaulting to 'codex'.
    """
    backend = plan_or_task.get("backend")
    if backend:
        return backend

    # Task dict — look up via plan_id
    plan_id = plan_or_task.get("plan_id")
    if plan_id:
        plan = get_plan_request(conn, plan_id)
        if plan:
            return plan.get("backend", "codex")
    return "codex"


def _resolve_effective_base_branch(
    conn: sqlite3.Connection, record: ProjectRow | TaskRow | None
) -> str:
    """Resolve an effective base branch from project context.

    Accepts either a project row (bearing project_id/dir) or a task row
    that can be resolved via plan→project.
    Falls back to ``main`` when resolution fails.
    """
    if not isinstance(record, dict):
        return "main"

    project_id = record.get("project_id")
    if project_id:
        return get_project_base_branch(conn, project_id)

    plan_id = record.get("plan_id")
    if plan_id:
        plan = get_plan_request(conn, plan_id)
        if plan:
            return get_project_base_branch(conn, plan["project_id"])

    project_row_id = record.get("id")
    if project_row_id and ("dir" in record or "name" in record):
        return get_project_base_branch(conn, project_row_id)

    return "main"


def _load_project_model_config(conn: sqlite3.Connection, project_id: str | None) -> dict:
    """Load and parse a project's `model_config` JSON (if present)."""
    if not project_id:
        return {}

    from agm.db import get_project_model_config

    raw = get_project_model_config(conn, project_id)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw

    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_project_app_server_approval_policy(
    conn: sqlite3.Connection,
    project_id: str | None,
) -> dict[str, str]:
    """Load effective per-project app-server approval policy."""
    if not project_id:
        return normalize_app_server_approval_policy(None)
    return get_project_app_server_approval_policy(conn, project_id)


def _load_project_app_server_ask_for_approval(
    conn: sqlite3.Connection,
    project_id: str | None,
) -> str | dict[str, dict[str, bool]]:
    """Load effective project AskForApproval policy for thread start/resume."""
    if not project_id:
        return normalize_app_server_ask_for_approval(None)
    return get_project_app_server_ask_for_approval(conn, project_id)


def _apply_project_app_server_ask_for_approval(
    conn: sqlite3.Connection,
    project_id: str | None,
    thread_config: dict[str, Any],
) -> None:
    """Apply project AskForApproval policy to thread/start params."""
    ask_policy = _load_project_app_server_ask_for_approval(conn, project_id)
    thread_config["approvalPolicy"] = copy.deepcopy(ask_policy)


def _resolve_project_model_config(
    conn: sqlite3.Connection, project_id: str | None, backend: str
) -> dict[str, str]:
    """Resolve effective model config for a project/backend."""
    from agm.backends import resolve_model_config

    return resolve_model_config(backend, _load_project_model_config(conn, project_id))


def _fallback_thread_config_for_resolved_model(
    thread_config: dict, primary_model: str
) -> dict | None:
    """Return a retry thread config tied to the resolved primary model.

    Uses WORK_MODEL_FALLBACK for the retry model, but only when the
    primary model matches the resolved runtime model.
    """
    from agm.backends import WORK_MODEL_FALLBACK

    if thread_config.get("model") != primary_model:
        return None
    if primary_model == WORK_MODEL_FALLBACK:
        return None

    fb = copy.deepcopy(thread_config)
    fb["model"] = WORK_MODEL_FALLBACK
    return fb


# ---------------------------------------------------------------------------
# Logging handler classes
# ---------------------------------------------------------------------------


class PlanDBHandler(logging.Handler):
    """Logging handler that persists log records to the plan_logs table."""

    def __init__(self, conn: sqlite3.Connection, plan_id: str, *, source: str = "plan"):
        super().__init__()
        self.conn = conn
        self.plan_id = plan_id
        self.source = source

    def emit(self, record: logging.LogRecord) -> None:
        try:
            add_plan_log(
                self.conn,
                plan_id=self.plan_id,
                level=record.levelname,
                message=self.format(record),
                source=self.source,
            )
        except Exception:
            self.handleError(record)


class TaskDBHandler(logging.Handler):
    """Logging handler that persists log records to the task_logs table."""

    def __init__(self, conn: sqlite3.Connection, task_id: str, *, source: str = "task"):
        super().__init__()
        self.conn = conn
        self.task_id = task_id
        self.source = source

    def emit(self, record: logging.LogRecord) -> None:
        try:
            add_task_log(
                self.conn,
                task_id=self.task_id,
                level=record.levelname,
                message=self.format(record),
                source=self.source,
            )
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Codex infrastructure
# ---------------------------------------------------------------------------


def _ensure_codex_home() -> Path:
    """Bootstrap agm's isolated Codex home directory.

    Creates ``~/.config/agm/.codex/`` with:
    - A symlink to ``~/.codex/auth.json`` (shared auth credentials)
    - An empty ``skills/`` directory (agm controls skill injection)

    Idempotent — safe to call on every job.
    """
    from agm.paths import CODEX_HOME

    CODEX_HOME.mkdir(parents=True, exist_ok=True)

    real_auth = Path.home() / ".codex" / "auth.json"
    local_auth = CODEX_HOME / "auth.json"
    if real_auth.exists() and not local_auth.exists():
        local_auth.symlink_to(real_auth)

    (CODEX_HOME / "skills").mkdir(exist_ok=True)

    return CODEX_HOME


def _should_use_daemon() -> bool:
    """Determine whether to use the shared daemon or direct app-server.

    Checks ``AGM_BACKEND_MODE`` env var first (``daemon`` or ``direct``),
    then falls back to auto-detection: use daemon if the socket exists.
    """
    from agm.daemon import DEFAULT_SOCKET_PATH

    mode = os.environ.get("AGM_BACKEND_MODE", "").lower()
    if mode == "direct":
        return False
    if mode == "daemon":
        return True
    return DEFAULT_SOCKET_PATH.exists()


async def _try_daemon_client():
    """Attempt to connect to the shared daemon. Returns client or None."""
    from agm.daemon_client import DaemonClient

    client = DaemonClient()
    try:
        await client.start()
        return client
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        log.debug("Daemon not available, falling back to direct mode")
        return None


@contextlib.asynccontextmanager
async def _codex_client(
    *,
    approval_policy: Mapping[str, object] | None = None,
):
    """Create a Codex client with the standard server-request handler.

    Tries the shared daemon first (if socket exists or ``AGM_BACKEND_MODE=daemon``).
    Falls back to spawning a direct ``codex app-server`` subprocess.

    Populates the live model cache on first connection per worker process.

    Usage:
        async with _codex_client() as client:
            ...
    """
    from agm.backends import get_live_models, set_live_models
    from agm.client import AppServerClient

    handle_server_request = _make_server_request_handler(approval_policy=approval_policy)

    client = None
    daemon_client = None

    if _should_use_daemon():
        daemon_client = await _try_daemon_client()

    if daemon_client is not None:
        client = daemon_client
        client.set_server_request_handler(handle_server_request)
        try:
            if get_live_models() is None:
                try:
                    result = await client.request("model/list", {"includeHidden": True}, timeout=30)
                    set_live_models(result.get("data", []))
                except Exception:
                    log.info("Failed to fetch model list, using static catalog")
            yield client
        finally:
            await daemon_client.stop()
    else:
        codex_home = _ensure_codex_home()
        env = {**os.environ, "CODEX_HOME": str(codex_home)}

        async with AppServerClient(env=env) as client:
            client.set_server_request_handler(handle_server_request)

            if get_live_models() is None:
                try:
                    result = await client.request("model/list", {"includeHidden": True}, timeout=30)
                    set_live_models(result.get("data", []))
                except Exception:
                    log.info("Failed to fetch model list, using static catalog")

            yield client


def _classify_codex_error(error_info: Any) -> tuple[str, dict[str, Any]]:
    """Classify a CodexErrorInfo value into (error_type, details).

    CodexErrorInfo is a discriminated union:
    - Plain strings: "contextWindowExceeded", "serverOverloaded", etc.
    - Object variants: {"httpConnectionFailed": {"httpStatusCode": 502}},
      {"responseStreamDisconnected": {"httpStatusCode": N}}, etc.

    Returns ("unknown", {}) if the value is None or unrecognizable.
    """
    if error_info is None:
        return ("unknown", {})
    if isinstance(error_info, str):
        return (error_info, {})
    if isinstance(error_info, dict):
        # Object variant — single key is the discriminator
        for key, value in error_info.items():
            return (key, value if isinstance(value, dict) else {})
    return ("unknown", {})


def _format_turn_error(error: dict[str, Any]) -> str:
    """Format a TurnError dict into a human-readable message with structured info.

    Extracts codexErrorInfo when present to provide richer diagnostics than
    just the raw message string.  Appends ``additionalDetails`` when available.
    """
    message = error.get("message", "unknown error")
    extra = error.get("additionalDetails")
    extra_suffix = f" | {extra}" if extra else ""
    info = error.get("codexErrorInfo")
    if info is None:
        return f"Turn failed: {message}{extra_suffix}"

    error_type, details = _classify_codex_error(info)

    if error_type == "modelCap":
        model = details.get("model", "unknown")
        reset = details.get("reset_after_seconds")
        suffix = f" (resets in {reset}s)" if reset is not None else ""
        return f"Turn failed [modelCap]: {message} — model={model}{suffix}{extra_suffix}"

    if error_type in (
        "httpConnectionFailed",
        "responseStreamConnectionFailed",
        "responseStreamDisconnected",
        "responseTooManyFailedAttempts",
    ):
        status = details.get("httpStatusCode")
        suffix = f" (HTTP {status})" if status is not None else ""
        return f"Turn failed [{error_type}]: {message}{suffix}{extra_suffix}"

    # String variants with specific diagnostics
    _HINTS = {
        "contextWindowExceeded": "consider thread/compact",
        "usageLimitExceeded": "API usage limit reached",
        "unauthorized": "check codex login status",
        "badRequest": "check request params",
        "sandboxError": "sandbox policy violation",
        "threadRollbackFailed": "rollback could not complete",
    }
    hint = _HINTS.get(error_type)
    if hint:
        return f"Turn failed [{error_type}]: {message} — {hint}{extra_suffix}"

    # serverOverloaded, internalServerError, other, or unrecognized
    return f"Turn failed [{error_type}]: {message}{extra_suffix}"


def _log_codex_model_rerouted(params: dict) -> None:
    from_model = params.get("fromModel", "?")
    to_model = params.get("toModel", "?")
    reason = params.get("reason", "unknown")
    log.warning(
        "Model rerouted: %s → %s (reason: %s, thread=%s, turn=%s)",
        from_model,
        to_model,
        reason,
        params.get("threadId", "?"),
        params.get("turnId", "?"),
    )


def _log_codex_mid_turn_error(params: dict) -> None:
    will_retry = params.get("willRetry", False)
    error = params.get("error", {})
    info = error.get("codexErrorInfo")
    error_type, details = _classify_codex_error(info)
    retry_label = "will retry" if will_retry else "NOT retrying"
    detail_parts = [f"type={error_type}"]
    if error_type == "modelCap":
        reset = details.get("reset_after_seconds")
        if reset is not None:
            detail_parts.append(f"reset_in={reset}s")
    elif details.get("httpStatusCode") is not None:
        detail_parts.append(f"http={details['httpStatusCode']}")
    log.warning(
        "Mid-turn error (%s): %s [%s] thread=%s turn=%s",
        retry_label,
        error.get("message", "?"),
        ", ".join(detail_parts),
        params.get("threadId", "?"),
        params.get("turnId", "?"),
    )


def _format_exception_label(exc: BaseException) -> str:
    """Return a concise '<Type>: message' label, handling empty exception strings."""
    msg = str(exc).strip()
    if not msg:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {msg}"


def _extract_reasoning_summaries(turn_completed_params: dict) -> list[str]:
    """Extract reasoning summary texts from a turn/completed notification."""
    turn = turn_completed_params.get("turn", {})
    items = turn.get("items", []) if isinstance(turn, dict) else []
    summaries: list[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for part in item.get("summary", []):
            if isinstance(part, dict) and part.get("type") == "summary_text":
                text = part.get("text", "").strip()
                if text:
                    summaries.append(text)
    return summaries


def _update_codex_turn_token_usage(
    turn_tokens: dict[str, tuple[int, int, int, int]], params: dict
) -> None:
    turn_id = params.get("turnId", "unknown")
    usage = params.get("tokenUsage", {})
    last = usage.get("last", {})
    turn_tokens[turn_id] = (
        last.get("inputTokens", 0),
        last.get("outputTokens", 0),
        last.get("cachedInputTokens", 0),
        last.get("reasoningOutputTokens", 0),
    )


_AGENT_MESSAGE_TRUNCATE = 500


def _extract_item_summary(item: dict) -> dict:
    """Extract a minimal summary from a Codex ThreadItem for event publishing.

    Returns a dict with ``item_type`` plus type-specific details (command text,
    file paths, tool name, etc.).  Kept intentionally small — just enough for
    agm-web to render a progress indicator.
    """
    item_type = item.get("type", "unknown")
    summary: dict = {"item_type": item_type}

    if item_type == "commandExecution":
        summary["command"] = item.get("command", "")
        summary["status"] = item.get("status", "")
        if item.get("exitCode") is not None:
            summary["exit_code"] = item["exitCode"]
        if item.get("durationMs") is not None:
            summary["duration_ms"] = item["durationMs"]
        actions = item.get("commandActions")
        if actions:
            summary["command_actions"] = [
                {"type": a.get("type", "unknown")} for a in actions if isinstance(a, dict)
            ]
    elif item_type == "fileChange":
        changes = item.get("changes", [])
        summary["files"] = [c.get("path", "") for c in changes if c.get("path")]
        summary["status"] = item.get("status", "")
    elif item_type == "mcpToolCall":
        summary["server"] = item.get("server", "")
        summary["tool"] = item.get("tool", "")
        summary["status"] = item.get("status", "")
    elif item_type == "webSearch":
        summary["query"] = item.get("query", "")
    elif item_type == "reasoning":
        summaries = item.get("summary", [])
        if summaries:
            summary["reasoning"] = " ".join(str(s) for s in summaries)
    elif item_type == "agentMessage":
        text = item.get("text", "")
        if text:
            summary["text"] = text[:_AGENT_MESSAGE_TRUNCATE]

    return summary


def _extract_plan_steps(params: dict) -> list[dict]:
    """Extract plan steps from a turn/plan/updated notification."""
    return [
        {"step": s.get("step", ""), "status": s.get("status", "")}
        for s in params.get("plan", [])
        if isinstance(s, dict)
    ]


class TurnEventContext:
    """Context for publishing streaming events during a Codex turn.

    When passed to ``_codex_turn()``, enables Codex notifications to be
    republished as Redis Stream events for agm-web consumption:

    - ``turn/started`` / ``turn/completed`` → ``task:turn``
    - ``item/started`` → ``task:item_started``
    - ``item/completed`` → ``task:item_completed`` (with reasoning, agent
      message text, command actions, file paths)
    - ``turn/plan/updated`` → ``task:plan_updated``
    - ``turn/diff/updated`` → ``task:turn_diff``
    - ``thread/tokenUsage/updated`` → ``task:token_usage``
    - ``thread/status/changed`` → ``task:thread_status``
    - ``model/rerouted`` → ``task:model_rerouted``
    - ``error`` → ``task:backend_error``
    - ``deprecationNotice`` / ``configWarning`` → ``task:backend_warning``
    """

    __slots__ = (
        "task_id",
        "plan_id",
        "project",
        "on_turn_started",
        "on_turn_completed",
        "run_id",
        "owner_role",
        "model",
        "model_provider",
        "has_active_steer",
    )

    def __init__(
        self,
        *,
        task_id: str,
        plan_id: str | None = None,
        project: str = "",
        on_turn_started: Callable[[str | None], None] | None = None,
        on_turn_completed: Callable[[str | None], None] | None = None,
        run_id: str | None = None,
        owner_role: str = "executor",
        model: str | None = None,
        model_provider: str | None = None,
        has_active_steer: bool = False,
    ) -> None:
        self.task_id = task_id
        self.plan_id = plan_id
        self.project = project
        self.on_turn_started = on_turn_started
        self.on_turn_completed = on_turn_completed
        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.owner_role = owner_role
        self.model = model
        self.model_provider = model_provider
        self.has_active_steer = has_active_steer


def _build_event_thread_context(
    event_context: TurnEventContext,
    *,
    thread_id: str | None = None,
    thread_status: object | None = None,
    active_turn_id: str | None = None,
    last_turn_event_at: str | None = None,
    turn_sequence: int = 0,
) -> dict[str, Any]:
    """Build a stable thread_context payload for task stream events."""
    return {
        "thread_id": thread_id,
        "thread_status": thread_status,
        "active_turn_id": active_turn_id,
        "last_turn_event_at": last_turn_event_at,
        "owner_role": event_context.owner_role,
        "model": event_context.model,
        "provider": event_context.model_provider,
        "has_active_steer": event_context.has_active_steer,
        "run_id": event_context.run_id,
        "turn_sequence": turn_sequence,
    }


def _snake_case_rate_limit_window(w: dict | None) -> dict | None:
    """Convert a camelCase RateLimitWindow to snake_case."""
    if not w:
        return None
    return {
        "used_percent": w.get("usedPercent", 0),
        "resets_at": w.get("resetsAt"),
        "window_duration_mins": w.get("windowDurationMins"),
    }


def _snake_case_rate_limit_credits(c: dict | None) -> dict | None:
    """Convert a camelCase CreditsSnapshot to snake_case."""
    if not c:
        return None
    return {
        "balance": c.get("balance"),
        "has_credits": c.get("hasCredits", False),
        "unlimited": c.get("unlimited", False),
    }


def _store_full_rate_limits(resp: dict, captured: str) -> None:
    """Store all buckets from a GetAccountRateLimitsResponse."""
    from agm.queue import store_codex_rate_limits

    by_id = resp.get("rateLimitsByLimitId") or {}
    for limit_id, rl in by_id.items():
        snapshot = {
            "limit_id": rl.get("limitId") or limit_id,
            "limit_name": rl.get("limitName"),
            "plan_type": rl.get("planType"),
            "primary": _snake_case_rate_limit_window(rl.get("primary")),
            "secondary": _snake_case_rate_limit_window(rl.get("secondary")),
            "credits": _snake_case_rate_limit_credits(rl.get("credits")),
            "captured_at": captured,
        }
        store_codex_rate_limits(snapshot)


def _register_codex_notification_handlers(
    client: Any,
    turn_done: asyncio.Event,
    turn_state: dict[str, Any],
    turn_tokens: dict[str, tuple[int, int, int, int]],
    event_context: TurnEventContext | None = None,
    trace_context: Any | None = None,
) -> dict[str, Callable[[dict], None]]:
    thread_state: dict[str, Any] = {
        "thread_id": None,
        "thread_status": None,
        "last_turn_event_at": None,
        "turn_sequence": 0,
    }

    def _thread_context_payload() -> dict[str, Any] | None:
        if event_context is None:
            return None
        return {
            "thread_id": thread_state.get("thread_id"),
            "thread_status": thread_state.get("thread_status"),
            "active_turn_id": turn_state.get("active_turn_id"),
            "last_turn_event_at": thread_state.get("last_turn_event_at"),
            "owner_role": event_context.owner_role,
            "model": event_context.model,
            "provider": event_context.model_provider,
            "has_active_steer": event_context.has_active_steer,
            "run_id": event_context.run_id,
            "turn_sequence": thread_state.get("turn_sequence"),
        }

    def _with_thread_context(extra: dict[str, Any] | None) -> dict[str, Any] | None:
        payload = dict(extra) if extra else {}
        thread_context = _thread_context_payload()
        if thread_context is not None:
            payload["thread_context"] = thread_context
        return payload or None

    def _extract_notification_token_usage(params: dict) -> dict[str, int]:
        usage = params.get("tokenUsage", {})
        last = usage.get("last", {}) if isinstance(usage, dict) else {}
        return {
            "input": int(last.get("inputTokens", params.get("inputTokens", 0)) or 0),
            "output": int(last.get("outputTokens", params.get("outputTokens", 0)) or 0),
            "cached_input": int(
                last.get("cachedInputTokens", params.get("cachedInputTokens", 0)) or 0
            ),
            "reasoning": int(
                last.get("reasoningOutputTokens", params.get("reasoningOutputTokens", 0)) or 0
            ),
        }

    def _record_trace(event_type: str, status: str | None, data: dict) -> None:
        if trace_context is None:
            return
        trace_context.record(event_type, status, data)

    def _publish_task_event(
        event_type: str,
        status: str,
        *,
        extra: dict | None = None,
    ) -> None:
        if event_context is None:
            return
        from agm.queue import publish_event

        publish_event(
            event_type,
            event_context.task_id,
            status,
            project=event_context.project,
            plan_id=event_context.plan_id,
            extra=_with_thread_context(extra),
        )

    def on_turn_started(params: dict) -> None:
        turn = params.get("turn", {})
        turn_id = turn.get("id")
        thread_state["thread_id"] = params.get("threadId") or thread_state.get("thread_id")
        turn_state["active_turn_id"] = turn_id
        thread_state["last_turn_event_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        thread_state["turn_sequence"] = int(thread_state.get("turn_sequence", 0)) + 1
        if event_context is not None and event_context.on_turn_started:
            try:
                event_context.on_turn_started(turn_id)
            except Exception:
                log.debug("Turn start callback failed", exc_info=True)
        _record_trace(
            "turn",
            "started",
            {
                "turn_id": turn_id,
                "turn_status": turn.get("status"),
            },
        )
        _publish_task_event(
            "task:turn",
            "started",
            extra={
                "turn_id": turn_id,
                "turn_status": turn.get("status"),
            },
        )

    def on_turn_completed(params: dict) -> None:
        turn_state["result"] = params
        turn_state["active_turn_id"] = None
        turn = params.get("turn", {})
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        thread_state["thread_id"] = params.get("threadId") or thread_state.get("thread_id")
        thread_state["last_turn_event_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if event_context is not None and event_context.on_turn_completed:
            try:
                event_context.on_turn_completed(turn_id)
            except Exception:
                log.debug("Turn completion callback failed", exc_info=True)
        error = turn.get("error") if isinstance(turn, dict) else None
        trace_data = {
            "turn_id": turn_id,
            "turn_status": turn.get("status") if isinstance(turn, dict) else None,
        }
        if isinstance(error, dict):
            trace_data["error"] = error
        _record_trace("turn", "completed", trace_data)
        _publish_task_event("task:turn", "completed", extra=trace_data)
        for summary_text in _extract_reasoning_summaries(params):
            log.info("Reasoning: %s", summary_text)
        turn_done.set()

    def on_token_usage(params: dict) -> None:
        _update_codex_turn_token_usage(turn_tokens, params)
        token_usage = _extract_notification_token_usage(params)
        _record_trace(
            "tokenUsage",
            "updated",
            {
                "turn_id": params.get("turnId"),
                **token_usage,
            },
        )

    def on_model_rerouted(params: dict) -> None:
        _log_codex_model_rerouted(params)
        _record_trace(
            "modelRerouted",
            "updated",
            {"notification": params},
        )
        _publish_task_event(
            "task:model_rerouted",
            "updated",
            extra={"notification": params},
        )

    def on_thread_status_changed(params: dict) -> None:
        thread: dict = params.get("thread", {}) if isinstance(params.get("thread"), dict) else {}
        thread_id = params.get("threadId") or params.get("thread_id") or thread.get("id")
        old_status = (
            params.get("oldStatus")
            or params.get("previousStatus")
            or params.get("previous_status")
            or thread.get("oldStatus")
            or thread.get("previousStatus")
        )
        new_status = params.get("newStatus") or params.get("status") or thread.get("status")
        status_label = str(new_status or "changed")
        thread_state["thread_id"] = thread_id or thread_state.get("thread_id")
        thread_state["thread_status"] = new_status
        trace_data = {
            "thread_id": thread_id,
            "old_status": old_status,
            "new_status": new_status,
            "notification": params,
        }
        _record_trace("threadStatus", "changed", trace_data)
        _publish_task_event(
            "task:thread_status",
            status_label,
            extra=trace_data,
        )
        if old_status and new_status:
            log.info(
                "Codex thread status changed: %s -> %s (thread_id=%s)",
                old_status,
                new_status,
                thread_id or "unknown",
            )
        elif new_status:
            log.info(
                "Codex thread status updated: %s (thread_id=%s)",
                new_status,
                thread_id or "unknown",
            )
        else:
            log.info(
                "Codex thread status notification received (thread_id=%s)",
                thread_id or "unknown",
            )

    def on_mid_turn_error(params: dict) -> None:
        _log_codex_mid_turn_error(params)
        _record_trace("backendError", "error", {"notification": params})
        _publish_task_event(
            "task:backend_error",
            "error",
            extra={"notification": params},
        )

    def on_deprecation_notice(params: dict) -> None:
        log.warning("Codex deprecation notice: %s", params.get("message", params))
        _record_trace(
            "deprecationNotice",
            "warning",
            {"message": params.get("message", params)},
        )
        _publish_task_event(
            "task:backend_warning",
            "deprecation",
            extra={"message": params.get("message", params)},
        )

    def on_config_warning(params: dict) -> None:
        log.warning("Codex config warning: %s", params.get("message", params))
        _record_trace(
            "configWarning",
            "warning",
            {"message": params.get("message", params)},
        )
        _publish_task_event(
            "task:backend_warning",
            "config",
            extra={"message": params.get("message", params)},
        )

    rate_limits_fetched = False

    async def on_rate_limits_updated(params: dict) -> None:
        nonlocal rate_limits_fetched
        from agm.queue import store_codex_rate_limits

        rl = params.get("rateLimits", {})
        _record_trace(
            "rateLimits",
            "updated",
            {
                "limit_id": rl.get("limitId"),
                "limit_name": rl.get("limitName"),
                "plan_type": rl.get("planType"),
            },
        )
        captured = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        snapshot = {
            "limit_id": rl.get("limitId"),
            "limit_name": rl.get("limitName"),
            "plan_type": rl.get("planType"),
            "primary": _snake_case_rate_limit_window(rl.get("primary")),
            "secondary": _snake_case_rate_limit_window(rl.get("secondary")),
            "credits": _snake_case_rate_limit_credits(rl.get("credits")),
            "captured_at": captured,
        }
        store_codex_rate_limits(snapshot)

        # Fetch full multi-bucket data once per session.
        if not rate_limits_fetched:
            rate_limits_fetched = True
            try:
                resp = await client.request("account/rateLimits/read", timeout=10)
                _store_full_rate_limits(resp, captured)
            except Exception:
                rate_limits_fetched = False  # retry next notification

    def on_item_completed_capture(params: dict) -> None:
        """Capture last agentMessage/plan text from item/completed notifications.

        Ephemeral threads reject ``thread/read`` with ``includeTurns`` and
        the ``turn/completed`` notification carries empty items, so this is
        the only reliable way to capture the response text.
        """
        item = params.get("item", {})
        item_type = item.get("type")
        if item_type in ("agentMessage", "plan"):
            turn_state["last_text_item"] = item.get("text", "")

    handlers: dict[str, Callable[..., Any]] = {
        "turn/started": on_turn_started,
        "turn/completed": on_turn_completed,
        "item/completed": on_item_completed_capture,
        "thread/tokenUsage/updated": on_token_usage,
        "thread/status/changed": on_thread_status_changed,
        "model/rerouted": on_model_rerouted,
        "error": on_mid_turn_error,
        "deprecationNotice": on_deprecation_notice,
        "configWarning": on_config_warning,
        "account/rateLimits/updated": on_rate_limits_updated,
    }

    # Streaming notifications — only when event_context is provided.
    if event_context is not None:

        def on_item_started(params: dict) -> None:
            item = params.get("item", {})
            _publish_task_event(
                "task:item_started",
                item.get("type", "unknown"),
                extra=_extract_item_summary(item),
            )

        def on_item_completed(params: dict) -> None:
            on_item_completed_capture(params)
            item = params.get("item", {})
            _publish_task_event(
                "task:item_completed",
                item.get("type", "unknown"),
                extra=_extract_item_summary(item),
            )

        def on_plan_updated(params: dict) -> None:
            steps = _extract_plan_steps(params)
            _publish_task_event(
                "task:plan_updated",
                "plan",
                extra={"steps": steps},
            )

        def on_turn_diff_updated(params: dict) -> None:
            diff_text = params.get("diff", "")
            _publish_task_event(
                "task:turn_diff",
                "updated",
                extra={
                    "diff_bytes": len(diff_text.encode()),
                    "diff_files": diff_text.count("diff --git"),
                },
            )

        def on_token_usage_stream(params: dict) -> None:
            on_token_usage(params)
            token_usage = _extract_notification_token_usage(params)
            _publish_task_event(
                "task:token_usage",
                "updated",
                extra={
                    **token_usage,
                },
            )

        handlers["item/started"] = on_item_started
        handlers["item/completed"] = on_item_completed
        handlers["turn/plan/updated"] = on_plan_updated
        handlers["turn/diff/updated"] = on_turn_diff_updated
        handlers["thread/tokenUsage/updated"] = on_token_usage_stream
    # Rich trace capture for item events — writes to SQLite for durable tracing.
    if trace_context is not None:
        from agm.tracing import extract_rich_trace

        tctx = trace_context

        def on_item_started_trace(params: dict) -> None:
            item = params.get("item", {})
            event_type, data = extract_rich_trace(item)
            tctx.record(event_type, "started", data)

        def on_item_completed_trace(params: dict) -> None:
            item = params.get("item", {})
            event_type, data = extract_rich_trace(item)
            tctx.record(event_type, "completed", data)

        # Chain with existing handlers if present.
        existing_started = handlers.get("item/started")
        if existing_started:
            orig_started = existing_started

            def combined_started(params: dict) -> None:
                orig_started(params)
                on_item_started_trace(params)

            handlers["item/started"] = combined_started
        else:
            handlers["item/started"] = on_item_started_trace

        existing_completed = handlers.get("item/completed")
        if existing_completed:
            orig_completed = existing_completed

            def combined_completed(params: dict) -> None:
                orig_completed(params)
                on_item_completed_trace(params)

            handlers["item/completed"] = combined_completed
        else:
            # Still need on_item_completed_capture for text extraction.
            def trace_with_capture(params: dict) -> None:
                on_item_completed_capture(params)
                on_item_completed_trace(params)

            handlers["item/completed"] = trace_with_capture

    for name, handler in handlers.items():
        client.on_notification(name, handler)
    return handlers


def _remove_codex_notification_handlers(
    client: Any, handlers: dict[str, Callable[[dict], None]]
) -> None:
    for name, handler in handlers.items():
        client.remove_notification_handler(name, handler)


async def _compact_codex_thread(
    client: Any,
    thread_id: str,
    timeout: float = 120,
) -> None:
    """Compact a Codex thread and wait for the compacted notification.

    Sends ``thread/compact/start`` and blocks until the ``thread/compacted``
    notification arrives for the matching thread.
    """
    compact_done = asyncio.Event()

    def on_compacted(params: dict) -> None:
        if params.get("threadId") == thread_id:
            compact_done.set()

    client.on_notification("thread/compacted", on_compacted)
    try:
        await client.request("thread/compact/start", {"threadId": thread_id}, timeout=30)
        await asyncio.wait_for(compact_done.wait(), timeout=timeout)
    finally:
        client.remove_notification_handler("thread/compacted", on_compacted)


async def _start_or_resume_codex_thread(
    client: Any,
    *,
    start_thread_params: dict | None,
    resume_thread_id: str | None,
    resume_thread_params: dict | None,
    on_thread_ready: Callable[[str], None] | None,
) -> tuple[str, dict[str, Any] | None]:
    def _extract_resume_thread_metadata(resume_response: dict[str, Any]) -> dict[str, Any]:
        thread = resume_response.get("thread", {})
        if not isinstance(thread, dict):
            return {}

        thread_status = thread.get("status")
        thread_status_type = (
            thread_status.get("type")
            if isinstance(thread_status, dict) and isinstance(thread_status.get("type"), str)
            else None
        )
        thread_active_flags = (
            thread_status.get("activeFlags")
            if isinstance(thread_status, dict)
            and isinstance(thread_status.get("activeFlags"), list)
            else []
        )

        turns = thread.get("turns", [])
        latest_turn: dict[str, Any] = {}
        if isinstance(turns, list) and turns and isinstance(turns[-1], dict):
            latest_turn = turns[-1]
        latest_items = latest_turn.get("items", [])
        latest_turn_item_types = (
            [
                item.get("type")
                for item in latest_items
                if isinstance(item, dict) and isinstance(item.get("type"), str)
            ]
            if isinstance(latest_items, list)
            else []
        )

        return {
            "thread_id": thread.get("id"),
            "thread_status": thread_status,
            "thread_status_type": thread_status_type,
            "thread_active_flags": thread_active_flags,
            "latest_turn_id": latest_turn.get("id"),
            "latest_turn_status": latest_turn.get("status"),
            "latest_turn_item_count": len(latest_items) if isinstance(latest_items, list) else 0,
            "latest_turn_item_types": latest_turn_item_types,
        }

    resume_metadata: dict[str, Any] | None = None
    if resume_thread_id is not None:
        resume_params: dict[str, Any] = {"threadId": resume_thread_id}
        if resume_thread_params:
            resume_params.update(resume_thread_params)
        resume_response = await client.request("thread/resume", resume_params, timeout=300)
        thread_id = resume_response["thread"]["id"]
        resume_metadata = _extract_resume_thread_metadata(resume_response)
    elif start_thread_params is not None:
        thread_response = await client.request("thread/start", start_thread_params, timeout=300)
        thread_id = thread_response["thread"]["id"]
    else:
        raise ValueError("Either start_thread_params or resume_thread_id must be provided")

    if on_thread_ready:
        on_thread_ready(thread_id)
    return thread_id, resume_metadata


HEARTBEAT_INTERVAL = 30  # seconds between executor heartbeat events
TURN_TIMEOUT = 600  # 10 minutes max per turn


async def _wait_with_heartbeat(
    turn_done: asyncio.Event,
    event_context: TurnEventContext | None,
    *,
    thread_id: str | None = None,
    turn_state: dict[str, Any] | None = None,
    timeout: int = TURN_TIMEOUT,
) -> None:
    """Wait for turn completion, emitting periodic heartbeat events."""
    import time

    from agm.queue import publish_event

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        wait_time = min(HEARTBEAT_INTERVAL, remaining)
        try:
            await asyncio.wait_for(turn_done.wait(), timeout=wait_time)
            return  # Turn completed
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise
            # Emit heartbeat
            if event_context:
                elapsed = int(timeout - (deadline - time.monotonic()))
                extra: dict[str, Any] = {"elapsed_seconds": elapsed}
                extra["thread_context"] = _build_event_thread_context(
                    event_context,
                    thread_id=thread_id,
                    active_turn_id=(
                        cast(str | None, turn_state.get("active_turn_id"))
                        if turn_state is not None
                        else None
                    ),
                )
                publish_event(
                    "task:heartbeat",
                    event_context.task_id,
                    "working",
                    project=event_context.project,
                    plan_id=event_context.plan_id,
                    extra=extra,
                )


async def _run_codex_turn_once(
    client: Any,
    *,
    thread_id: str,
    prompt: str,
    turn_config: dict,
    turn_done: asyncio.Event,
    turn_state: dict[str, Any],
    event_context: TurnEventContext | None = None,
    trace_context: Any | None = None,
    turn_timeout: int = TURN_TIMEOUT,
    turn_index: int = 0,
) -> dict:
    if trace_context is not None and hasattr(trace_context, "set_turn_index"):
        trace_context.set_turn_index(turn_index)
    turn_done.clear()
    turn_state["result"] = {}
    turn_state.pop("last_text_item", None)
    turn_params: dict[str, Any] = {
        "threadId": thread_id,
        "input": [{"type": "text", "text": prompt}],
        **turn_config,
    }
    await client.request("turn/start", turn_params)
    try:
        await _wait_with_heartbeat(
            turn_done,
            event_context,
            thread_id=thread_id,
            turn_state=turn_state,
            timeout=turn_timeout,
        )
    except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt, SystemExit) as exc:
        active_turn_id = turn_state.get("active_turn_id")
        if active_turn_id:
            try:
                await client.request(
                    "turn/interrupt",
                    {"threadId": thread_id, "turnId": active_turn_id},
                    timeout=10,
                )
                label = "cancelled" if not isinstance(exc, TimeoutError) else "timed-out"
                log.warning("Interrupted %s turn %s on thread %s", label, active_turn_id, thread_id)
            except Exception:
                log.warning("Failed to interrupt turn %s on thread %s", active_turn_id, thread_id)
        raise

    turn = turn_state.get("result", {}).get("turn", {})
    if turn.get("status") == "failed":
        error = turn.get("error", {})
        raise RuntimeError(_format_turn_error(error))
    return turn


async def _run_initial_codex_turn(
    client: Any,
    *,
    prompt: str,
    turn_config: dict,
    turn_done: asyncio.Event,
    turn_state: dict[str, Any],
    thread_id: str,
    fallback_thread_params: dict | None,
    resume_thread_id: str | None,
    on_thread_ready: Callable[[str], None] | None,
    event_context: TurnEventContext | None = None,
    trace_context: Any | None = None,
    turn_timeout: int = TURN_TIMEOUT,
    turn_index: int = 0,
) -> str:
    try:
        await _run_codex_turn_once(
            client,
            thread_id=thread_id,
            prompt=prompt,
            turn_config=turn_config,
            turn_done=turn_done,
            turn_state=turn_state,
            event_context=event_context,
            trace_context=trace_context,
            turn_timeout=turn_timeout,
            turn_index=turn_index,
        )
        return thread_id
    except (RuntimeError, TimeoutError) as first_err:
        if fallback_thread_params is None or resume_thread_id is not None:
            raise
        fallback_model = fallback_thread_params.get("model", "unknown")
        error_label = _format_exception_label(first_err)
        trace_fallback_retry_data = {
            "failed_thread_id": thread_id,
            "fallback_model": fallback_model,
            "reason": error_label,
        }
        log.warning(
            "Primary turn failed on thread %s (%s); retrying with fallback model %s",
            thread_id,
            error_label,
            fallback_model,
        )
        if trace_context is not None:
            trace_context.record("executionFallback", "retrying", trace_fallback_retry_data)
        if event_context is not None:
            from agm.queue import publish_event

            publish_event(
                "task:execution_fallback",
                event_context.task_id,
                "retrying",
                project=event_context.project,
                plan_id=event_context.plan_id,
                extra={
                    **trace_fallback_retry_data,
                    "thread_context": _build_event_thread_context(
                        event_context,
                        thread_id=thread_id,
                        active_turn_id=turn_state.get("active_turn_id"),
                    ),
                },
            )
        thread_response = await client.request("thread/start", fallback_thread_params, timeout=300)
        fallback_thread_id = thread_response["thread"]["id"]
        if on_thread_ready:
            on_thread_ready(fallback_thread_id)
        trace_fallback_started_data = {
            "failed_thread_id": thread_id,
            "fallback_thread_id": fallback_thread_id,
            "fallback_model": fallback_model,
            "reason": error_label,
        }
        if trace_context is not None:
            trace_context.record("executionFallback", "started", trace_fallback_started_data)
        if event_context is not None:
            from agm.queue import publish_event

            publish_event(
                "task:execution_fallback",
                event_context.task_id,
                "started",
                project=event_context.project,
                plan_id=event_context.plan_id,
                extra={
                    **trace_fallback_started_data,
                    "thread_context": _build_event_thread_context(
                        event_context,
                        thread_id=fallback_thread_id,
                        active_turn_id=turn_state.get("active_turn_id"),
                    ),
                },
            )
        await _run_codex_turn_once(
            client,
            thread_id=fallback_thread_id,
            prompt=prompt,
            turn_config=turn_config,
            turn_done=turn_done,
            turn_state=turn_state,
            event_context=event_context,
            trace_context=trace_context,
            turn_timeout=turn_timeout,
            turn_index=turn_index,
        )
        return fallback_thread_id


def _summarize_codex_token_usage(
    turn_tokens: dict[str, tuple[int, int, int, int]],
) -> dict[str, int]:
    total_input = sum(t[0] for t in turn_tokens.values())
    total_output = sum(t[1] for t in turn_tokens.values())
    total_cached = sum(t[2] for t in turn_tokens.values())
    total_reasoning = sum(t[3] for t in turn_tokens.values())
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cached_input_tokens": total_cached,
        "reasoning_tokens": total_reasoning,
    }


async def _codex_turn(
    client: Any,
    *,
    prompt: str,
    turn_config: dict,
    start_thread_params: dict | None = None,
    resume_thread_id: str | None = None,
    resume_thread_params: dict | None = None,
    on_thread_ready: Callable[[str], None] | None = None,
    post_initial_turn: (
        Callable[[Callable[[str], Awaitable[dict]], str], Awaitable[None]] | None
    ) = None,
    fallback_thread_params: dict | None = None,
    event_context: TurnEventContext | None = None,
    trace_context: Any | None = None,
    turn_timeout: int = TURN_TIMEOUT,
) -> tuple[str, str, dict[str, int]]:
    """Run one Codex turn orchestration and return extracted output text + token usage."""
    turn_done = asyncio.Event()
    turn_state: dict[str, Any] = {"result": {}, "active_turn_id": None}
    turn_tokens: dict[str, tuple[int, int, int, int]] = {}
    handlers = _register_codex_notification_handlers(
        client, turn_done, turn_state, turn_tokens, event_context, trace_context
    )

    try:
        thread_id, resume_metadata = await _start_or_resume_codex_thread(
            client,
            start_thread_params=start_thread_params,
            resume_thread_id=resume_thread_id,
            resume_thread_params=resume_thread_params,
            on_thread_ready=on_thread_ready,
        )
        if resume_metadata:
            if trace_context is not None and hasattr(trace_context, "record"):
                trace_context.record("threadResume", "loaded", resume_metadata)
            status_label = resume_metadata.get("thread_status_type")
            if event_context is not None and isinstance(status_label, str):
                from agm.queue import publish_event

                publish_event(
                    "task:thread_status",
                    event_context.task_id,
                    status_label,
                    project=event_context.project,
                    plan_id=event_context.plan_id,
                    extra={
                        "thread_id": resume_metadata.get("thread_id"),
                        "old_status": None,
                        "new_status": resume_metadata.get("thread_status"),
                        "source": "thread/resume",
                        "thread_active_flags": resume_metadata.get("thread_active_flags"),
                        "latest_turn_id": resume_metadata.get("latest_turn_id"),
                        "latest_turn_status": resume_metadata.get("latest_turn_status"),
                        "latest_turn_item_count": resume_metadata.get("latest_turn_item_count"),
                        "latest_turn_item_types": resume_metadata.get("latest_turn_item_types"),
                        "thread_context": _build_event_thread_context(
                            event_context,
                            thread_id=cast(str | None, resume_metadata.get("thread_id")),
                            thread_status=resume_metadata.get("thread_status"),
                            active_turn_id=None,
                            turn_sequence=0,
                        ),
                    },
                )
        thread_ref = {"id": thread_id}
        thread_ref["id"] = await _run_initial_codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            turn_done=turn_done,
            turn_state=turn_state,
            thread_id=thread_ref["id"],
            fallback_thread_params=fallback_thread_params,
            resume_thread_id=resume_thread_id,
            on_thread_ready=on_thread_ready,
            event_context=event_context,
            trace_context=trace_context,
            turn_timeout=turn_timeout,
            turn_index=0,
        )
        next_turn_index = 1

        async def run_turn(text: str) -> dict:
            nonlocal next_turn_index
            turn_index = next_turn_index
            next_turn_index += 1
            return await _run_codex_turn_once(
                client,
                thread_id=thread_ref["id"],
                prompt=text,
                turn_config=turn_config,
                turn_done=turn_done,
                turn_state=turn_state,
                event_context=event_context,
                trace_context=trace_context,
                turn_timeout=turn_timeout,
                turn_index=turn_index,
            )

        if post_initial_turn:
            await post_initial_turn(run_turn, thread_ref["id"])

        read_result: dict[str, Any] = {}
        try:
            read_result = await client.request(
                "thread/read",
                {"threadId": thread_ref["id"], "includeTurns": True},
                timeout=300,
            )
        except Exception as exc:
            # Codex ephemeral threads reject includeTurns. Fall back to a
            # minimal thread/read and extract text from the completed turn.
            # Keep this narrow so unrelated RPC failures still bubble.
            from agm.client import RPCError

            if not isinstance(exc, RPCError) or "includeTurns" not in str(exc):
                raise
            read_result = await client.request(
                "thread/read",
                {"threadId": thread_ref["id"]},
                timeout=300,
            )
        summary = (
            _extract_plan_text(read_result)
            or _extract_turn_text(turn_state.get("result", {}))
            or turn_state.get("last_text_item", "")
        )
        token_usage = _summarize_codex_token_usage(turn_tokens)
        return thread_ref["id"], summary, token_usage
    finally:
        _remove_codex_notification_handlers(client, handlers)


def _extract_plan_text(thread_read_result: dict) -> str:
    """Extract plan text from the last turn in thread/read response.

    Only reads the final turn — the one just completed. This prevents
    continuation plans from accidentally finalizing with stale parent output
    if the new turn produced no plan/agentMessage items.
    """
    thread = thread_read_result.get("thread", {})
    turns = thread.get("turns", [])
    if not turns:
        return ""

    last_turn = turns[-1]
    last_plan = ""
    last_agent = ""

    for item in last_turn.get("items", []):
        item_type = item.get("type")
        if item_type == "plan":
            last_plan = item.get("text", "")
        elif item_type == "agentMessage":
            last_agent = item.get("text", "")

    # Prefer last plan item; fall back to last agent message
    return last_plan or last_agent


def _extract_turn_text(turn_completed_result: dict) -> str:
    """Extract plan text from a turn/completed payload.

    Fallback used when thread/read cannot provide turns (for example,
    ephemeral Codex threads that reject includeTurns).
    """
    turn = turn_completed_result.get("turn", {}) if isinstance(turn_completed_result, dict) else {}
    items = turn.get("items", []) if isinstance(turn, dict) else []
    if not isinstance(items, list):
        return ""

    last_plan = ""
    last_agent = ""
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "plan":
            last_plan = item.get("text", "")
        elif item_type == "agentMessage":
            last_agent = item.get("text", "")
    return last_plan or last_agent


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def _get_project_id_for_task(conn: sqlite3.Connection, task: TaskRow) -> str | None:
    """Resolve project ID from task → plan chain."""
    plan = get_plan_request(conn, task.get("plan_id", ""))
    if not plan:
        return None
    return plan.get("project_id")


def _get_project_dir_for_task(conn: sqlite3.Connection, task: TaskRow) -> str | None:
    """Resolve project directory from task → plan → project chain."""
    plan = get_plan_request(conn, task.get("plan_id", ""))
    if not plan:
        return None
    proj = get_project(conn, plan.get("project_id", ""))
    return proj["dir"] if proj else None


def _effective_task_priority(priority: Any) -> str:
    """Return high/medium/low for display/sorting, treating NULL as medium."""
    if isinstance(priority, str):
        normalized = priority.strip().lower()
        if normalized in _TASK_PRIORITY_RANK:
            return normalized
    return "medium"


def _normalize_output_task_priority(priority: Any) -> str | None:
    """Normalize task-agent priority (medium -> NULL)."""
    if priority is None:
        return None
    if not isinstance(priority, str) or not priority.strip():
        raise RuntimeError("Task agent produced invalid priority; expected high/medium/low")
    normalized = priority.strip().lower()
    if normalized not in _TASK_PRIORITY_RANK:
        raise RuntimeError(
            f"Task agent produced invalid priority '{priority}'; expected high/medium/low"
        )
    if normalized == "medium":
        return None
    return normalized


def _get_rejection_count(conn: sqlite3.Connection, task_id: str) -> int:
    """Count REVIEW-level task_logs for the current execution cycle.

    Only counts rejections after the most recent 'Claimed' log, so that
    task retry resets the counter (claim happens at the start of each cycle).
    Uses rowid for ordering since timestamps have second-level precision.
    """
    rows = conn.execute(
        "SELECT level, message FROM task_logs WHERE task_id = ? ORDER BY rowid",
        (task_id,),
    ).fetchall()
    # Find the position of the most recent claim (start of cycle)
    last_claim_idx = -1
    for i, row in enumerate(rows):
        if row[0] == "INFO" and "Claimed by" in row[1]:
            last_claim_idx = i
    # Count REVIEW logs after the last claim
    return sum(1 for row in rows[last_claim_idx + 1 :] if row[0] == "REVIEW")


def _get_latest_review(conn: sqlite3.Connection, task_id: str) -> str | None:
    """Fetch the latest REVIEW-level task_log message for a task."""
    logs = list_task_logs(conn, task_id, level="REVIEW")
    if logs:
        return logs[-1]["message"]
    return None


def _parse_task_files(task: TaskRow) -> list[str]:
    """Parse task's files JSON field into a list of paths."""
    task_files: list[str] = []
    if task.get("files"):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            task_files = json.loads(task["files"] or "") or []
    return task_files


def _has_uncommitted_changes(worktree: str) -> bool:
    """Check if a worktree has uncommitted or untracked changes."""
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())
