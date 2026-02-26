"""Tests for Codex event forwarding to Redis."""

from __future__ import annotations

from agm.jobs_common import _extract_item_summary


class TestExtractItemSummary:
    """Tests for _extract_item_summary() field extraction."""

    def test_command_execution_basic(self) -> None:
        item = {"type": "commandExecution", "command": "pytest", "status": "completed"}
        result = _extract_item_summary(item)
        assert result["item_type"] == "commandExecution"
        assert result["command"] == "pytest"
        assert result["status"] == "completed"

    def test_command_execution_with_exit_code_and_duration(self) -> None:
        item = {
            "type": "commandExecution",
            "command": "ruff check",
            "status": "completed",
            "exitCode": 1,
            "durationMs": 342,
        }
        result = _extract_item_summary(item)
        assert result["exit_code"] == 1
        assert result["duration_ms"] == 342

    def test_command_execution_with_actions(self) -> None:
        item = {
            "type": "commandExecution",
            "command": "rg pattern src/",
            "status": "completed",
            "commandActions": [
                {"type": "search", "query": "pattern", "path": "src/"},
            ],
        }
        result = _extract_item_summary(item)
        assert result["command_actions"] == [{"type": "search"}]

    def test_command_execution_no_actions(self) -> None:
        item = {"type": "commandExecution", "command": "ls", "status": "completed"}
        result = _extract_item_summary(item)
        assert "command_actions" not in result

    def test_file_change(self) -> None:
        item = {
            "type": "fileChange",
            "changes": [
                {"path": "src/foo.py", "kind": {"type": "update"}},
                {"path": "src/bar.py", "kind": {"type": "update"}},
            ],
            "status": "completed",
        }
        result = _extract_item_summary(item)
        assert result["files"] == ["src/foo.py", "src/bar.py"]

    def test_reasoning(self) -> None:
        item = {
            "type": "reasoning",
            "summary": ["Planning test implementation", "Reviewing existing code"],
        }
        result = _extract_item_summary(item)
        assert result["reasoning"] == "Planning test implementation Reviewing existing code"

    def test_reasoning_empty_summary(self) -> None:
        item = {"type": "reasoning", "summary": []}
        result = _extract_item_summary(item)
        assert "reasoning" not in result

    def test_agent_message(self) -> None:
        item = {"type": "agentMessage", "text": "I'll create the calculator package."}
        result = _extract_item_summary(item)
        assert result["text"] == "I'll create the calculator package."

    def test_agent_message_truncated(self) -> None:
        item = {"type": "agentMessage", "text": "x" * 1000}
        result = _extract_item_summary(item)
        assert len(result["text"]) == 500

    def test_agent_message_empty(self) -> None:
        item = {"type": "agentMessage", "text": ""}
        result = _extract_item_summary(item)
        assert "text" not in result

    def test_unknown_type(self) -> None:
        item = {"type": "somethingNew", "data": 42}
        result = _extract_item_summary(item)
        assert result == {"item_type": "somethingNew"}

    def test_mcp_tool_call(self) -> None:
        item = {"type": "mcpToolCall", "server": "sqlite", "tool": "query", "status": "completed"}
        result = _extract_item_summary(item)
        assert result["server"] == "sqlite"
        assert result["tool"] == "query"

    def test_web_search(self) -> None:
        item = {"type": "webSearch", "query": "python asyncio"}
        result = _extract_item_summary(item)
        assert result["query"] == "python asyncio"
