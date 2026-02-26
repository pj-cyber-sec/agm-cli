"""Protocol schema validation tests.

Validates that every outgoing message agm constructs matches the
protocol schema files in schemas/ and schemas/v2/. Catches contract
drift (unknown fields, wrong types, missing required fields) at test
time — before it reaches production.

Design note: The v2 schema files don't set ``additionalProperties: false``,
so standard ``jsonschema.validate()`` won't reject unknown top-level keys.
``_validate_strict()`` adds a top-level unknown-field check on top of
standard validation, catching both type errors AND stray fields (the exact
bug class that caused ``collaborationMode`` on ``thread/start``).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError, validate

from agm.backends import (
    THREAD_CONFIGS,
    TURN_CONFIGS,
    get_runtime_thread_config,
)

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"
V2_SCHEMAS_DIR = SCHEMAS_DIR / "v2"

# All 7 job types that have backend configs.
JOB_TYPES = sorted(THREAD_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_v2_schema(filename: str) -> dict:
    """Load a JSON schema from schemas/v2/."""
    path = V2_SCHEMAS_DIR / filename
    return json.loads(path.read_text())


def _load_schema(filename: str) -> dict:
    """Load a JSON schema from schemas/."""
    path = SCHEMAS_DIR / filename
    return json.loads(path.read_text())


def _validate(instance: Any, schema: dict) -> None:
    """Standard jsonschema validation."""
    validate(instance=instance, schema=schema)


def _validate_strict(instance: dict, schema: dict) -> None:
    """Validate + reject unknown top-level keys not in schema['properties'].

    Standard jsonschema won't reject unknown fields when the schema lacks
    ``additionalProperties: false``. This function adds that check.
    """
    _validate(instance, schema)
    allowed_keys = set(schema.get("properties", {}).keys())
    unknown = set(instance.keys()) - allowed_keys
    if unknown:
        raise ValidationError(
            f"Unknown top-level keys: {sorted(unknown)}. Allowed: {sorted(allowed_keys)}"
        )


def _iter_string_literals(node: ast.AST) -> list[str]:
    """Extract direct string literals from a constant or tuple/list/set literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
        return values
    return []


def _collect_server_request_methods() -> set[str]:
    """Collect server request methods handled by _make_server_request_handler()."""
    filepath = Path(__file__).resolve().parent.parent / "src" / "agm" / "jobs_common.py"
    tree = ast.parse(filepath.read_text())
    found: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "method":
            continue
        for comparator in node.comparators:
            for method_name in _iter_string_literals(comparator):
                if method_name.startswith(("item/", "skill/", "account/")) or method_name in (
                    "applyPatchApproval",
                    "execCommandApproval",
                ):
                    found.add(method_name)
    return found


def _collect_notification_methods() -> set[str]:
    """Collect notification methods registered via jobs_common notification wiring."""
    filepath = Path(__file__).resolve().parent.parent / "src" / "agm" / "jobs_common.py"
    tree = ast.parse(filepath.read_text())
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "handlers"
                    and isinstance(node.value, ast.Dict)
                ):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            found.add(key.value)
                if isinstance(target, ast.Subscript):
                    if not isinstance(target.value, ast.Name) or target.value.id != "handlers":
                        continue
                    key = target.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        found.add(key.value)

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "on_notification"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)

    return found


def _collect_server_notification_methods_from_schema() -> set[str]:
    """Collect notification methods advertised in ServerNotification.json."""
    schema = _load_schema("ServerNotification.json")
    found: set[str] = set()
    for variant in schema.get("oneOf", []):
        method_prop = variant.get("properties", {}).get("method", {})
        for method in method_prop.get("enum", []):
            if isinstance(method, str):
                found.add(method)
    return found


# ---------------------------------------------------------------------------
# (b) Static config → wire message validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_thread_config_validates_against_schema(job_type: str) -> None:
    """THREAD_CONFIGS entries (plus cwd) must match ThreadStartParams schema.
    Uses strict validation to catch unknown fields."""
    schema = _load_v2_schema("ThreadStartParams.json")
    config = dict(THREAD_CONFIGS[job_type])

    # cwd is always added at call site.
    config["cwd"] = "/tmp/test"

    _validate_strict(config, schema)


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_turn_config_validates_against_schema(job_type: str) -> None:
    """TURN_CONFIGS entries + runtime fields must match TurnStartParams schema.
    Uses strict validation to catch unknown fields."""
    schema = _load_v2_schema("TurnStartParams.json")
    config = dict(TURN_CONFIGS[job_type])

    # Runtime fields always added by _codex_turn()
    config["threadId"] = "test-thread-id"
    config["input"] = [{"type": "text", "text": "test prompt"}]

    _validate_strict(config, schema)


# ---------------------------------------------------------------------------
# (c) Runtime-resolved config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_runtime_thread_config_validates(job_type: str) -> None:
    """get_runtime_thread_config() output must match ThreadStartParams schema."""
    schema = _load_v2_schema("ThreadStartParams.json")
    config = get_runtime_thread_config("codex", job_type)

    config["cwd"] = "/tmp/test"

    _validate_strict(config, schema)


# ---------------------------------------------------------------------------
# (d) Fixed message validation
# ---------------------------------------------------------------------------


def test_initialize_params_validate() -> None:
    """Initialize params must match InitializeParams in ClientRequest.json."""
    schema = _load_schema("ClientRequest.json")

    # Inline $refs so ClientInfo resolves
    init_schema_resolved = {
        "type": "object",
        "required": ["clientInfo"],
        "properties": {
            "capabilities": {
                "anyOf": [
                    schema["definitions"]["InitializeCapabilities"],
                    {"type": "null"},
                ]
            },
            "clientInfo": schema["definitions"]["ClientInfo"],
        },
    }

    params = {
        "clientInfo": {"name": "agm", "version": "0.1.0"},
        "capabilities": {},
    }
    _validate(params, init_schema_resolved)


def test_initialize_capabilities_strict() -> None:
    """Capabilities must only contain known fields — catches stale fields
    like the removed optOutNotificationMethods."""
    schema = _load_schema("ClientRequest.json")
    caps_schema = schema["definitions"]["InitializeCapabilities"]

    capabilities: dict[str, Any] = {}
    _validate_strict(capabilities, caps_schema)

    # Verify unknown fields are caught
    with pytest.raises(ValidationError, match="Unknown top-level keys"):
        _validate_strict({"totallyBogusField": True}, caps_schema)


def test_approval_response_validates() -> None:
    """Approval response must match all approval response schemas."""
    cmd_schema = _load_schema("CommandExecutionRequestApprovalResponse.json")
    file_schema = _load_schema("FileChangeRequestApprovalResponse.json")
    skill_schema = _load_schema("SkillRequestApprovalResponse.json")
    exec_schema = _load_schema("ExecCommandApprovalResponse.json")
    patch_schema = _load_schema("ApplyPatchApprovalResponse.json")

    std_response = {"decision": "accept"}
    skill_response = {"decision": "approve"}
    review_response = {"decision": "approved"}
    _validate(std_response, cmd_schema)
    _validate(std_response, file_schema)
    _validate(skill_response, skill_schema)
    _validate(review_response, exec_schema)
    _validate(review_response, patch_schema)


def test_tool_request_user_input_response_validates() -> None:
    """ToolRequestUserInput response must match schema."""
    schema = _load_schema("ToolRequestUserInputResponse.json")
    response = {"answers": {}}
    _validate(response, schema)


def test_chatgpt_auth_tokens_refresh_response_validates() -> None:
    """account/chatgptAuthTokens/refresh response must match schema."""
    schema = _load_schema("ChatgptAuthTokensRefreshResponse.json")
    response = {
        "accessToken": "token-123",
        "chatgptAccountId": "acct-456",
        "chatgptPlanType": "pro",
    }
    _validate(response, schema)


def test_dynamic_tool_call_response_validates() -> None:
    """item/tool/call response must match DynamicToolCallResponse schema."""
    schema = _load_schema("DynamicToolCallResponse.json")
    response = {
        "success": True,
        "contentItems": [{"type": "inputText", "text": "ok"}],
    }
    _validate(response, schema)


def test_thread_read_params_validate() -> None:
    """thread/read params must match ThreadReadParams schema."""
    schema = _load_v2_schema("ThreadReadParams.json")
    params = {"threadId": "test-id", "includeTurns": True}
    _validate_strict(params, schema)


def test_thread_resume_params_validate() -> None:
    """thread/resume params must match ThreadResumeParams schema."""
    schema = _load_v2_schema("ThreadResumeParams.json")
    params = {"threadId": "test-id"}
    _validate_strict(params, schema)


def test_thread_list_params_validate() -> None:
    """thread/list params must match ThreadListParams schema."""
    schema = _load_v2_schema("ThreadListParams.json")
    params = {
        "searchTerm": "upgrade",
        "limit": 25,
        "cursor": "cursor-1",
        "archived": False,
        "cwd": "/tmp/project",
        "sortKey": "updated_at",
    }
    _validate_strict(params, schema)


# ---------------------------------------------------------------------------
# (e) Field placement guards
# ---------------------------------------------------------------------------


def test_collaboration_mode_not_in_thread_start_schema() -> None:
    """collaborationMode must NOT be a ThreadStartParams property.
    It was removed in 0.102.0 — components flattened to direct properties."""
    schema = _load_v2_schema("ThreadStartParams.json")
    assert "collaborationMode" not in schema.get("properties", {})


def test_collaboration_mode_not_in_turn_start_schema() -> None:
    """collaborationMode was removed from TurnStartParams in 0.102.0.
    Its components are now direct properties: effort, model, sandboxPolicy."""
    schema = _load_v2_schema("TurnStartParams.json")
    assert "collaborationMode" not in schema.get("properties", {})


def test_effort_in_turn_start_schema() -> None:
    """effort (from former collaborationMode.settings.reasoning_effort) is
    a direct TurnStartParams property in 0.102.0."""
    schema = _load_v2_schema("TurnStartParams.json")
    assert "effort" in schema.get("properties", {})


def test_summary_in_turn_start_schema() -> None:
    """summary (reasoning summaries) is a TurnStartParams property."""
    schema = _load_v2_schema("TurnStartParams.json")
    assert "summary" in schema.get("properties", {})


def test_model_list_params_validate() -> None:
    """model/list params must match ModelListParams schema."""
    schema = _load_v2_schema("ModelListParams.json")
    params: dict[str, Any] = {"includeHidden": True}
    _validate_strict(params, schema)


def test_thread_rollback_params_validate() -> None:
    """thread/rollback params must match ThreadRollbackParams schema."""
    schema = _load_v2_schema("ThreadRollbackParams.json")
    params: dict[str, Any] = {"threadId": "test-thread-id", "numTurns": 1}
    _validate_strict(params, schema)


def test_thread_compact_start_params_validate() -> None:
    """thread/compact/start params must match ThreadCompactStartParams schema."""
    schema = _load_v2_schema("ThreadCompactStartParams.json")
    params: dict[str, Any] = {"threadId": "test-thread-id"}
    _validate_strict(params, schema)


def test_model_list_response_validates() -> None:
    """model/list response shape must match ModelListResponse schema."""
    schema = _load_v2_schema("ModelListResponse.json")
    response = {
        "data": [
            {
                "id": "gpt-5.3-codex",
                "model": "gpt-5.3-codex",
                "displayName": "GPT-5.3 Codex",
                "description": "Full Codex model",
                "hidden": False,
                "isDefault": True,
                "defaultReasoningEffort": "high",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "high", "description": "Best quality"},
                ],
            }
        ],
        "nextCursor": None,
    }
    _validate(response, schema)


def test_get_account_rate_limits_response_validates() -> None:
    """account/rateLimits/read response must match GetAccountRateLimitsResponse."""
    schema = _load_v2_schema("GetAccountRateLimitsResponse.json")
    response = {
        "rateLimits": {
            "planType": "pro",
            "primary": {"usedPercent": 0, "resetsAt": 1740000000, "windowDurationMins": 300},
            "secondary": {"usedPercent": 22, "resetsAt": 1740500000, "windowDurationMins": 10080},
            "credits": {"hasCredits": True, "unlimited": False, "balance": None},
        },
        "rateLimitsByLimitId": {
            "codex": {
                "limitId": "codex",
                "limitName": "5 hour usage limit",
                "planType": "pro",
                "primary": {"usedPercent": 0, "resetsAt": 1740000000, "windowDurationMins": 300},
                "secondary": {
                    "usedPercent": 22,
                    "resetsAt": 1740500000,
                    "windowDurationMins": 10080,
                },
                "credits": {"hasCredits": True, "unlimited": False, "balance": None},
            },
        },
    }
    _validate(response, schema)


# ---------------------------------------------------------------------------
# (g) Coverage guards — fail if new methods/handlers added without tests
# ---------------------------------------------------------------------------

# Methods agm sends to the Codex app-server.
_COVERED_CLIENT_METHODS = {
    "account/rateLimits/read",
    "initialize",
    "model/list",
    "thread/compact/start",
    "thread/rollback",
    "thread/start",
    "thread/resume",
    "thread/read",
    "turn/start",
    "turn/interrupt",
}

# Server request methods agm handles responses for.
_COVERED_SERVER_REQUEST_METHODS = {
    "account/chatgptAuthTokens/refresh",
    "applyPatchApproval",
    "execCommandApproval",
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/tool/call",
    "item/tool/requestUserInput",
    "skill/requestApproval",
}


def test_all_client_methods_have_schema_coverage() -> None:
    """Every method agm sends must be covered by schema validation tests.

    If this fails, a new client.request() call was added without updating
    the schema tests. Add a test for the new method and include it in
    _COVERED_CLIENT_METHODS.
    """
    import ast

    # Scan client.py and jobs_common.py for client.request() calls
    found_methods: set[str] = set()

    files_to_scan = [
        Path(__file__).resolve().parent.parent / "src" / "agm" / "client.py",
        Path(__file__).resolve().parent.parent / "src" / "agm" / "jobs_common.py",
    ]

    for filepath in files_to_scan:
        tree = ast.parse(filepath.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match: *.request("method", ...) or await *.request("method", ...)
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "request"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found_methods.add(node.args[0].value)

    uncovered = found_methods - _COVERED_CLIENT_METHODS
    assert not uncovered, (
        f"New client methods found without schema test coverage: {sorted(uncovered)}. "
        f"Add schema validation tests and update _COVERED_CLIENT_METHODS."
    )


def test_all_server_request_methods_have_schema_coverage() -> None:
    """Every server request method agm handles must have schema coverage.

    If this fails, a new server request handler was added without updating
    the schema tests. Add a test for the response shape and include the
    method in _COVERED_SERVER_REQUEST_METHODS.
    """
    found_methods = _collect_server_request_methods()

    uncovered = found_methods - _COVERED_SERVER_REQUEST_METHODS
    assert not uncovered, (
        f"New server request methods found without schema test coverage: {sorted(uncovered)}. "
        f"Add response shape validation tests and update _COVERED_SERVER_REQUEST_METHODS."
    )


# ---------------------------------------------------------------------------
# (h) Notification schema validation
# ---------------------------------------------------------------------------

# Notification methods agm subscribes to in _codex_turn.
_COVERED_NOTIFICATION_METHODS = {
    "account/rateLimits/updated",
    "turn/started",
    "turn/completed",
    "thread/compacted",
    "thread/tokenUsage/updated",
    "thread/status/changed",
    "model/rerouted",
    "error",
    "deprecationNotice",
    "configWarning",
    # Streaming notifications (conditionally registered via event_context)
    "item/started",
    "item/completed",
    "turn/diff/updated",
    "turn/plan/updated",
}


def test_model_rerouted_notification_schema_exists() -> None:
    """ModelReroutedNotification.json must exist in schemas/v2/."""
    schema = _load_v2_schema("ModelReroutedNotification.json")
    assert schema["title"] == "ModelReroutedNotification"
    assert "fromModel" in schema["properties"]
    assert "toModel" in schema["properties"]
    assert "reason" in schema["properties"]


def test_model_rerouted_notification_validates() -> None:
    """A sample model/rerouted notification must match the schema."""
    schema = _load_v2_schema("ModelReroutedNotification.json")
    params = {
        "fromModel": "gpt-5.3-codex-spark",
        "toModel": "gpt-5.3",
        "reason": "highRiskCyberActivity",
        "threadId": "thread-123",
        "turnId": "turn-1",
    }
    _validate_strict(params, schema)


def test_error_notification_validates() -> None:
    """A sample error notification with object variant must match the schema."""
    schema = _load_v2_schema("ErrorNotification.json")
    params = {
        "error": {
            "message": "Connection refused",
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 502}},
        },
        "threadId": "thread-123",
        "turnId": "turn-1",
        "willRetry": True,
    }
    _validate_strict(params, schema)


def test_error_notification_plain_string_error_info() -> None:
    """ErrorNotification with plain string codexErrorInfo validates."""
    schema = _load_v2_schema("ErrorNotification.json")
    params = {
        "error": {
            "message": "Server busy",
            "codexErrorInfo": "serverOverloaded",
        },
        "threadId": "thread-123",
        "turnId": "turn-1",
        "willRetry": True,
    }
    _validate_strict(params, schema)


def test_thread_compacted_notification_validates() -> None:
    """thread/compacted notification must match ContextCompactedNotification schema."""
    schema = _load_schema("ServerNotification.json")
    compact_schema = schema["definitions"]["ContextCompactedNotification"]
    params: dict[str, Any] = {"threadId": "test-thread-id", "turnId": "test-turn-id"}
    _validate_strict(params, compact_schema)


def test_item_started_notification_validates() -> None:
    """item/started notification must match ItemStartedNotification schema."""
    schema = _load_v2_schema("ItemStartedNotification.json")
    assert "item" in schema["properties"]
    assert "threadId" in schema["properties"]
    assert "turnId" in schema["properties"]


def test_item_completed_notification_validates() -> None:
    """item/completed notification must match ItemCompletedNotification schema."""
    schema = _load_v2_schema("ItemCompletedNotification.json")
    assert "item" in schema["properties"]
    assert "threadId" in schema["properties"]
    assert "turnId" in schema["properties"]


def test_turn_plan_updated_notification_validates() -> None:
    """turn/plan/updated notification must match TurnPlanUpdatedNotification."""
    schema = _load_v2_schema("TurnPlanUpdatedNotification.json")
    params = {
        "threadId": "thread-123",
        "turnId": "turn-1",
        "plan": [
            {"step": "Read the source file", "status": "completed"},
            {"step": "Apply the fix", "status": "inProgress"},
            {"step": "Run tests", "status": "pending"},
        ],
    }
    _validate_strict(params, schema)


def test_turn_diff_updated_notification_validates() -> None:
    """turn/diff/updated notification must match TurnDiffUpdatedNotification."""
    schema = _load_v2_schema("TurnDiffUpdatedNotification.json")
    params = {
        "threadId": "thread-123",
        "turnId": "turn-1",
        "diff": "diff --git a/foo b/foo",
    }
    _validate_strict(params, schema)


def test_thread_status_changed_notification_validates() -> None:
    """thread/status/changed notification must match ThreadStatusChangedNotification."""
    schema = _load_v2_schema("ThreadStatusChangedNotification.json")
    params = {
        "threadId": "thread-123",
        "status": {"type": "idle"},
    }
    _validate_strict(params, schema)


def test_account_rate_limits_updated_notification_validates() -> None:
    """account/rateLimits/updated notification must match its v2 schema."""
    schema = _load_v2_schema("AccountRateLimitsUpdatedNotification.json")
    params = {
        "rateLimits": {
            "limitId": "codex",
            "limitName": "5 hour usage limit",
            "planType": "pro",
            "primary": {"usedPercent": 0, "resetsAt": 1740000000, "windowDurationMins": 300},
            "secondary": {"usedPercent": 22, "resetsAt": 1740500000, "windowDurationMins": 10080},
            "credits": {"hasCredits": True, "unlimited": False, "balance": None},
        }
    }
    _validate_strict(params, schema)


def test_thread_token_usage_updated_notification_validates() -> None:
    """thread/tokenUsage/updated notification must match its v2 schema."""
    schema = _load_v2_schema("ThreadTokenUsageUpdatedNotification.json")
    usage = {
        "inputTokens": 100,
        "outputTokens": 200,
        "cachedInputTokens": 5,
        "reasoningOutputTokens": 20,
        "totalTokens": 325,
    }
    params = {
        "threadId": "thread-123",
        "turnId": "turn-1",
        "tokenUsage": {"last": usage, "total": usage, "modelContextWindow": 200000},
    }
    _validate_strict(params, schema)


def test_all_notification_methods_have_coverage() -> None:
    """Every notification method agm subscribes to must be covered.

    If this fails, a new on_notification() call was added without updating
    the tests. Add a schema test and include it in _COVERED_NOTIFICATION_METHODS.
    """
    found_methods = _collect_notification_methods()

    uncovered = found_methods - _COVERED_NOTIFICATION_METHODS
    assert not uncovered, (
        f"New notification subscriptions found without coverage: {sorted(uncovered)}. "
        f"Add schema validation tests and update _COVERED_NOTIFICATION_METHODS."
    )


def test_subscribed_notifications_exist_in_server_notification_schema() -> None:
    """Subscribed notification methods must exist in ServerNotification schema."""
    subscribed = _collect_notification_methods()
    available = _collect_server_notification_methods_from_schema()
    missing = subscribed - available
    assert not missing, (
        "Notification subscriptions missing from ServerNotification.json: "
        f"{sorted(missing)}. Regenerate schemas and/or update subscriptions."
    )
