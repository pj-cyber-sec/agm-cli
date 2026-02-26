"""Tests for the tool-aware server request handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agm.jobs_common import _make_server_request_handler


class FakePool:
    """Fake MCP pool for testing tool call routing."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name in self._results:
            return self._results[name]
        raise KeyError(f"Unknown tool: {name}")


class TestMakeServerRequestHandler:
    async def test_approval_without_pool(self) -> None:
        handler = _make_server_request_handler()
        result = await handler("item/commandExecution/requestApproval", {})
        assert result == {"decision": "accept"}

    async def test_file_change_approval(self) -> None:
        handler = _make_server_request_handler()
        result = await handler("item/fileChange/requestApproval", {})
        assert result == {"decision": "accept"}

    async def test_skill_approval(self) -> None:
        handler = _make_server_request_handler()
        result = await handler("skill/requestApproval", {"skillName": "demo"})
        assert result == {"decision": "approve"}

    async def test_unknown_method_raises(self) -> None:
        handler = _make_server_request_handler()
        with pytest.raises(ValueError, match="Unsupported"):
            await handler("some/unknown/method", {})

    async def test_tool_call_without_pool_raises(self) -> None:
        handler = _make_server_request_handler()
        with pytest.raises(ValueError, match="Unsupported"):
            await handler("item/tool/call", {"tool": "greet", "arguments": {}})

    async def test_exec_command_approval(self) -> None:
        handler = _make_server_request_handler()
        result = await handler(
            "execCommandApproval",
            {
                "callId": "call-1",
                "conversationId": "thread-1",
                "command": ["bash", "-lc", "echo hi"],
                "cwd": "/tmp",
                "parsedCmd": [],
                "approvalId": "approval-1",
                "additionalPermissions": {"write": ["/tmp"]},
            },
        )
        assert result == {"decision": "approved"}

    async def test_exec_command_approval_records_trace_and_log(self, caplog) -> None:
        records: list[tuple[str, str | None, dict[str, Any]]] = []

        class _TraceRecorder:
            def record(self, event_type: str, status: str | None, data: dict[str, Any]) -> None:
                records.append((event_type, status, data))

        handler = _make_server_request_handler(trace_context=_TraceRecorder())
        with caplog.at_level("INFO"):
            await handler(
                "execCommandApproval",
                {
                    "conversationId": "thread-1",
                    "callId": "call-1",
                    "command": ["bash", "-lc", "echo hi"],
                    "cwd": "/tmp",
                    "parsedCmd": [],
                    "approvalId": "approval-1",
                    "additionalPermissions": {"write": ["/tmp"]},
                },
            )

        assert any(
            "Approval handled: method=execCommandApproval" in rec.message for rec in caplog.records
        )
        assert any("approval_id=approval-1" in rec.message for rec in caplog.records)
        assert records[0][0] == "approvalRequest"
        assert records[0][1] == "approved"
        assert records[0][2]["method"] == "execCommandApproval"
        assert records[0][2]["approval_id"] == "approval-1"
        assert records[0][2]["additional_permissions"] == {"write": ["/tmp"]}

    async def test_skill_approval_records_trace_and_log(self, caplog) -> None:
        records: list[tuple[str, str | None, dict[str, Any]]] = []

        class _TraceRecorder:
            def record(self, event_type: str, status: str | None, data: dict[str, Any]) -> None:
                records.append((event_type, status, data))

        handler = _make_server_request_handler(trace_context=_TraceRecorder())
        with caplog.at_level("INFO"):
            result = await handler(
                "skill/requestApproval",
                {
                    "itemId": "item-skill-1",
                    "skillName": "demo-skill",
                    "skillPath": "skills/demo-skill/SKILL.md",
                    "permissionProfile": {"network": False},
                },
            )

        assert result == {"decision": "approve"}
        assert any("method=skill/requestApproval" in rec.message for rec in caplog.records)
        assert any("skill_name=demo-skill" in rec.message for rec in caplog.records)
        assert records[0][0] == "approvalRequest"
        assert records[0][1] == "approve"
        assert records[0][2]["method"] == "skill/requestApproval"
        assert records[0][2]["item_id"] == "item-skill-1"
        assert records[0][2]["skill_name"] == "demo-skill"
        assert records[0][2]["skill_path"] == "skills/demo-skill/SKILL.md"
        assert records[0][2]["permission_profile"] == {"network": False}

    async def test_skill_approval_uses_skill_metadata_fallback(self) -> None:
        records: list[tuple[str, str | None, dict[str, Any]]] = []

        class _TraceRecorder:
            def record(self, event_type: str, status: str | None, data: dict[str, Any]) -> None:
                records.append((event_type, status, data))

        handler = _make_server_request_handler(trace_context=_TraceRecorder())
        result = await handler(
            "skill/requestApproval",
            {
                "itemId": "item-skill-2",
                "skillName": "demo-skill",
                "skillMetadata": {"name": "demo-skill-meta", "path": "skills/meta/SKILL.md"},
            },
        )

        assert result == {"decision": "approve"}
        assert records[0][2]["skill_name"] == "demo-skill"
        assert records[0][2]["skill_path"] == "skills/meta/SKILL.md"

    async def test_apply_patch_approval(self) -> None:
        handler = _make_server_request_handler()
        result = await handler("applyPatchApproval", {"callId": "patch-1"})
        assert result == {"decision": "approved"}

    async def test_custom_approval_policy_overrides_decisions(self) -> None:
        handler = _make_server_request_handler(
            approval_policy={
                "item/commandExecution/requestApproval": "decline",
                "item/fileChange/requestApproval": "cancel",
                "skill/requestApproval": "decline",
                "execCommandApproval": "denied",
                "applyPatchApproval": "abort",
            }
        )
        assert await handler("item/commandExecution/requestApproval", {}) == {"decision": "decline"}
        assert await handler("item/fileChange/requestApproval", {}) == {"decision": "cancel"}
        assert await handler("skill/requestApproval", {"skillName": "demo"}) == {
            "decision": "decline"
        }
        assert await handler("execCommandApproval", {"callId": "exec-1"}) == {"decision": "denied"}
        assert await handler("applyPatchApproval", {"callId": "patch-1"}) == {"decision": "abort"}

    def test_invalid_approval_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown app-server approval methods"):
            _make_server_request_handler(approval_policy={"unknown/method": "accept"})

    async def test_tool_request_user_input_returns_empty_answers(self) -> None:
        handler = _make_server_request_handler()
        result = await handler(
            "item/tool/requestUserInput",
            {"toolCallId": "call-1", "questions": []},
        )
        assert result == {"answers": {}}

    async def test_chatgpt_auth_tokens_refresh_uses_auth_file(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        auth_home = tmp_path / ".codex"
        auth_home.mkdir()
        (auth_home / "auth.json").write_text(
            json.dumps(
                {
                    "accessToken": "token-123",
                    "chatgptAccountId": "acct-456",
                    "chatgptPlanType": "pro",
                }
            )
        )
        monkeypatch.setattr("agm.paths.CODEX_HOME", auth_home)

        handler = _make_server_request_handler()
        result = await handler("account/chatgptAuthTokens/refresh", {"reason": "expired"})
        assert result["accessToken"] == "token-123"
        assert result["chatgptAccountId"] == "acct-456"
        assert result["chatgptPlanType"] == "pro"

    async def test_chatgpt_auth_tokens_refresh_falls_back_to_empty(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        auth_home = tmp_path / ".codex"
        auth_home.mkdir()
        monkeypatch.setattr("agm.paths.CODEX_HOME", auth_home)

        handler = _make_server_request_handler()
        result = await handler("account/chatgptAuthTokens/refresh", {"reason": "expired"})
        assert result["accessToken"] == ""
        assert result["chatgptAccountId"] == ""

    async def test_tool_call_routed_to_pool(self) -> None:
        pool = FakePool({"greet": "Hello, World!"})
        handler = _make_server_request_handler(mcp_pool=pool)

        result = await handler(
            "item/tool/call",
            {"tool": "greet", "arguments": {"name": "World"}},
        )

        assert result["success"] is True
        assert result["contentItems"][0]["text"] == "Hello, World!"
        assert pool.calls == [("greet", {"name": "World"})]

    async def test_tool_call_failure_returns_error(self) -> None:
        pool = FakePool()  # no results registered
        handler = _make_server_request_handler(mcp_pool=pool)

        result = await handler(
            "item/tool/call",
            {"tool": "missing_tool", "arguments": {}},
        )

        assert result["success"] is False
        assert "missing_tool" in result["contentItems"][0]["text"]

    async def test_approval_still_works_with_pool(self) -> None:
        """Approval requests are handled even when a pool is registered."""
        pool = FakePool()
        handler = _make_server_request_handler(mcp_pool=pool)

        result = await handler("item/commandExecution/requestApproval", {})
        assert result == {"decision": "accept"}
        assert pool.calls == []  # pool not touched
