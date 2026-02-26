"""Tests for .agm/tools.toml loading and server resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.tool_config import get_servers_for_job_type, load_tool_config


class TestLoadToolConfig:
    def test_returns_none_for_no_project_dir(self) -> None:
        assert load_tool_config(None) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert load_tool_config(str(tmp_path)) is None

    def test_parses_valid_toml(self, tmp_path: Path) -> None:
        agm_dir = tmp_path / ".agm"
        agm_dir.mkdir()
        (agm_dir / "tools.toml").write_text(
            '[servers.context7]\ncommand = "npx"\nargs = ["-y", "@upstash/context7-mcp"]\n'
            "\n"
            "[tools]\n"
            'task_execution = ["context7"]\n'
        )
        config = load_tool_config(str(tmp_path))
        assert config is not None
        assert "servers" in config
        assert "context7" in config["servers"]
        assert config["tools"]["task_execution"] == ["context7"]

    def test_returns_none_for_malformed_toml(self, tmp_path: Path) -> None:
        agm_dir = tmp_path / ".agm"
        agm_dir.mkdir()
        (agm_dir / "tools.toml").write_text("this is not [valid toml\n")
        assert load_tool_config(str(tmp_path)) is None


class TestGetServersForJobType:
    @pytest.fixture()
    def config(self) -> dict:
        return {
            "servers": {
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@upstash/context7-mcp"],
                },
                "sqlite": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-server-sqlite"],
                },
            },
            "tools": {
                "task_execution": ["context7", "sqlite"],
                "review": ["context7"],
            },
        }

    def test_returns_servers_for_mapped_job_type(self, config: dict) -> None:
        result = get_servers_for_job_type(config, "task_execution")
        assert len(result) == 2
        names = [name for name, _, _ in result]
        assert names == ["context7", "sqlite"]

        # Verify command and args
        name, cmd, args = result[0]
        assert name == "context7"
        assert cmd == "npx"
        assert args == ["-y", "@upstash/context7-mcp"]

    def test_returns_subset_for_review(self, config: dict) -> None:
        result = get_servers_for_job_type(config, "review")
        assert len(result) == 1
        assert result[0][0] == "context7"

    def test_returns_empty_for_unmapped_job_type(self, config: dict) -> None:
        assert get_servers_for_job_type(config, "enrichment") == []

    def test_skips_undefined_server(self, config: dict) -> None:
        config["tools"]["task_execution"] = ["context7", "nonexistent"]
        result = get_servers_for_job_type(config, "task_execution")
        assert len(result) == 1
        assert result[0][0] == "context7"

    def test_skips_server_without_command(self, config: dict) -> None:
        config["servers"]["broken"] = {"args": ["foo"]}
        config["tools"]["task_execution"] = ["broken"]
        result = get_servers_for_job_type(config, "task_execution")
        assert result == []

    def test_empty_tools_section(self) -> None:
        config = {"servers": {"a": {"command": "echo", "args": []}}}
        assert get_servers_for_job_type(config, "task_execution") == []
