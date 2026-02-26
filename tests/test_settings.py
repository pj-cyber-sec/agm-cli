"""Tests for the agm settings command."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agm.cli import main


@pytest.fixture()
def _mock_targets(tmp_path):
    """Patch _SETTINGS_TARGETS to use temp paths."""
    targets = {
        "codex-instructions": {
            "path": tmp_path / ".codex" / "AGENTS.md",
            "description": "Codex CLI global instructions",
        },
        "codex-settings": {
            "path": tmp_path / ".codex" / "config.toml",
            "description": "Codex CLI settings",
        },
    }
    with patch("agm.cli._SETTINGS_TARGETS", targets):
        yield targets


def test_settings_list(_mock_targets):
    """--list returns JSON list of targets."""
    runner = CliRunner()
    result = runner.invoke(main, ["settings", "--list"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 2
    names = {t["name"] for t in data}
    assert "codex-instructions" in names
    assert "codex-settings" in names
    for t in data:
        assert "path" in t
        assert "exists" in t


def test_settings_edit_launches_editor(_mock_targets):
    """--edit opens the target file in $EDITOR."""
    runner = CliRunner()
    with patch("agm.cli._launch_editor") as mock_editor:
        result = runner.invoke(main, ["settings", "--edit", "codex-instructions"])
    assert result.exit_code == 0
    called_path = mock_editor.call_args[0][0]
    assert str(called_path).endswith("AGENTS.md")


def test_settings_edit_unknown_target():
    """--edit with an unknown target name is rejected by click."""
    runner = CliRunner()
    result = runner.invoke(main, ["settings", "--edit", "nonexistent"])
    assert result.exit_code != 0


def test_settings_read_missing(_mock_targets):
    """--read returns empty content for a missing file."""
    runner = CliRunner()
    result = runner.invoke(main, ["settings", "--read", "codex-instructions"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "codex-instructions"
    assert data["content"] == ""


def test_settings_read_existing(_mock_targets):
    """--read returns file content."""
    targets = _mock_targets
    path = targets["codex-instructions"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# My Instructions\n")

    runner = CliRunner()
    result = runner.invoke(main, ["settings", "--read", "codex-instructions"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["content"] == "# My Instructions\n"


def test_settings_write(_mock_targets):
    """--write saves stdin to the target file."""
    targets = _mock_targets
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["settings", "--write", "codex-instructions"],
        input="# Updated\nNew content\n",
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "codex-instructions"
    assert data["bytes_written"] > 0
    path = targets["codex-instructions"]["path"]
    assert path.read_text() == "# Updated\nNew content\n"
