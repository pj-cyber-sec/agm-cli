import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agm.agents_config import (
    _load_agent_instructions,
    build_agents_toml_scaffold,
    get_effective_role_config,
    get_global_role_text,
    get_project_role_text,
    reset_global_role_instructions,
    reset_project_role_instructions,
    set_global_role_instructions,
    set_project_role_instructions,
)


def _write_toml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_load_agent_instructions_merges_global_then_project(tmp_path):
    """Global instructions are prepended before project instructions."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    _write_toml(
        home / ".config" / "agm" / "agents.toml",
        '[planner]\ninstructions = "global instructions"\n',
    )
    _write_toml(
        project / ".agm" / "agents.toml",
        '[planner]\ninstructions = "project instructions"\n',
    )

    with patch("pathlib.Path.home", return_value=home):
        instructions = _load_agent_instructions(str(project), "planner")

    assert instructions == "global instructions\n\nproject instructions"


def test_load_agent_instructions_ignores_unknown_roles_and_wrong_types(tmp_path):
    """Only supported roles are used; unsupported/invalid role definitions are ignored."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    _write_toml(
        home / ".config" / "agm" / "agents.toml",
        (
            '[planner]\ninstructions = "global planner"\n'
            "[reviewer]\ninstructions = [1, 2, 3]\n"
            '[extra]\ninstructions = "ignored"\n'
        ),
    )
    _write_toml(
        project / ".agm" / "agents.toml",
        '[executor]\ninstructions = "project executor"\n'
        '[planner]\ninstructions = "project planner"\n',
    )

    with patch("pathlib.Path.home", return_value=home):
        assert (
            _load_agent_instructions(str(project), "planner") == "global planner\n\nproject planner"
        )
        assert _load_agent_instructions(str(project), "executor") == "project executor"
        assert _load_agent_instructions(str(project), "reviewer") == ""
        assert _load_agent_instructions(str(project), "not-a-role") == ""


def test_load_agent_instructions_returns_empty_without_configs(tmp_path):
    """Missing both global and project configs yields no instructions."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    with patch("pathlib.Path.home", return_value=home):
        assert _load_agent_instructions(str(project), "planner") == ""


def test_load_agent_instructions_handles_empty_sections(tmp_path):
    """Empty sections are treated as absent instructions."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    _write_toml(
        home / ".config" / "agm" / "agents.toml",
        '[planner]\ninstructions = ""\n',
    )
    _write_toml(
        project / ".agm" / "agents.toml",
        '[planner]\ninstructions = "   "\n',
    )

    with patch("pathlib.Path.home", return_value=home):
        assert _load_agent_instructions(str(project), "planner") == ""


def test_load_agent_instructions_fallback_on_malformed_toml(tmp_path):
    """Malformed TOML in one file does not prevent valid config from the other file."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    _write_toml(home / ".config" / "agm" / "agents.toml", '[planner\ninstructions = "broken"\n')
    _write_toml(
        project / ".agm" / "agents.toml",
        '[planner]\ninstructions = "project instructions"\n',
    )

    with patch("pathlib.Path.home", return_value=home):
        assert _load_agent_instructions(str(project), "planner") == "project instructions"


def test_agents_config_cli_helpers():
    """CLI helpers expose effective config and scaffold generation."""
    assert get_effective_role_config(None, "planner") == _load_agent_instructions(None, "planner")
    scaffold = build_agents_toml_scaffold("planner")
    assert "[planner]" in scaffold
    assert "Add planner instructions here." in scaffold


# -- get/set/reset tests --


def test_set_get_roundtrip(tmp_path):
    """Setting role instructions and reading them back returns the same text."""
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Be concise.")
    assert get_project_role_text(str(project), "planner") == "Be concise."


def test_set_preserves_other_roles(tmp_path):
    """Setting one role does not erase another."""
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Plan well.")
    set_project_role_instructions(str(project), "executor", "Execute fast.")
    assert get_project_role_text(str(project), "planner") == "Plan well."
    assert get_project_role_text(str(project), "executor") == "Execute fast."


def test_set_overwrites_existing(tmp_path):
    """Setting the same role again overwrites the previous text."""
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "version 1")
    set_project_role_instructions(str(project), "planner", "version 2")
    assert get_project_role_text(str(project), "planner") == "version 2"


def test_reset_removes_role(tmp_path):
    """Resetting a role removes it from the file."""
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Plan well.")
    set_project_role_instructions(str(project), "executor", "Execute fast.")
    reset_project_role_instructions(str(project), "planner")
    assert get_project_role_text(str(project), "planner") == ""
    assert get_project_role_text(str(project), "executor") == "Execute fast."


def test_reset_deletes_file_when_empty(tmp_path):
    """Resetting the last role deletes the file entirely."""
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Plan well.")
    reset_project_role_instructions(str(project), "planner")
    assert not (project / ".agm" / "agents.toml").exists()


def test_reset_nonexistent_role_is_safe(tmp_path):
    """Resetting a role that was never set does not error."""
    project = tmp_path / "project"
    project.mkdir()
    reset_project_role_instructions(str(project), "planner")  # no file at all


def test_set_creates_agm_dir(tmp_path):
    """Setting instructions creates .agm/ if it doesn't exist."""
    project = tmp_path / "project"
    project.mkdir()
    assert not (project / ".agm").exists()
    set_project_role_instructions(str(project), "reviewer", "Review carefully.")
    assert (project / ".agm" / "agents.toml").exists()
    assert get_project_role_text(str(project), "reviewer") == "Review carefully."


def test_set_invalid_role_raises(tmp_path):
    """Setting an unknown role raises ValueError."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="Unknown role"):
        set_project_role_instructions(str(project), "invalid_role", "text")


def test_reset_invalid_role_raises(tmp_path):
    """Resetting an unknown role raises ValueError."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="Unknown role"):
        reset_project_role_instructions(str(project), "invalid_role")


def test_get_project_role_text_no_global_merge(tmp_path):
    """get_project_role_text reads only project-level, ignoring global."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    _write_toml(
        home / ".config" / "agm" / "agents.toml",
        '[planner]\ninstructions = "global only"\n',
    )
    set_project_role_instructions(str(project), "planner", "project only")

    with patch("pathlib.Path.home", return_value=home):
        # get_project_role_text should NOT include global
        assert get_project_role_text(str(project), "planner") == "project only"
        # get_effective_role_config SHOULD include global
        assert "global only" in get_effective_role_config(str(project), "planner")


# -- CLI flag tests --


def test_agents_json_output(tmp_path):
    """--json flag produces valid JSON with the expected shape."""
    from click.testing import CliRunner

    from agm.cli import main

    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "executor", "Run fast.")

    runner = CliRunner()
    mock_project = {"dir": str(project), "name": "test"}
    with (
        patch("agm.cli._resolve_agents_project", return_value=mock_project),
        patch("pathlib.Path.home", return_value=home),
    ):
        result = runner.invoke(main, ["agents", "show"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "roles" in data
    assert data["roles"]["executor"]["project"] == "Run fast."
    assert data["roles"]["executor"]["effective"] == "Run fast."
    assert data["roles"]["executor"]["global"] == ""
    # Roles with no instructions should have empty strings
    assert data["roles"]["planner"]["project"] == ""
    assert data["roles"]["planner"]["global"] == ""


def test_agents_set_from_stdin(tmp_path):
    """agents set reads instructions from stdin and writes them."""
    from click.testing import CliRunner

    from agm.cli import main

    project = tmp_path / "project"
    project.mkdir()

    runner = CliRunner()
    with patch(
        "agm.cli._resolve_project_by_name",
        return_value={"dir": str(project), "name": "test"},
    ):
        result = runner.invoke(
            main, ["agents", "set", "planner", "-p", "test"], input="Be brief.\n"
        )

    assert result.exit_code == 0, result.output
    assert get_project_role_text(str(project), "planner") == "Be brief."


def test_agents_reset_via_cli(tmp_path):
    """agents reset clears role instructions."""
    from click.testing import CliRunner

    from agm.cli import main

    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Something.")

    runner = CliRunner()
    with patch(
        "agm.cli._resolve_project_by_name",
        return_value={"dir": str(project), "name": "test"},
    ):
        result = runner.invoke(main, ["agents", "reset", "planner", "-p", "test"])

    assert result.exit_code == 0, result.output
    assert get_project_role_text(str(project), "planner") == ""


def test_agents_set_empty_stdin_resets(tmp_path):
    """agents set with empty stdin implicitly resets the role."""
    from click.testing import CliRunner

    from agm.cli import main

    project = tmp_path / "project"
    project.mkdir()
    set_project_role_instructions(str(project), "planner", "Something.")

    runner = CliRunner()
    with patch(
        "agm.cli._resolve_project_by_name",
        return_value={"dir": str(project), "name": "test"},
    ):
        result = runner.invoke(main, ["agents", "set", "planner", "-p", "test"], input="")

    assert result.exit_code == 0, result.output
    assert get_project_role_text(str(project), "planner") == ""


def test_agents_set_requires_project():
    """agents set without -p and no cwd project should error."""
    from click.testing import CliRunner

    from agm.cli import main

    runner = CliRunner()
    with patch("agm.cli._resolve_agents_project", return_value=None):
        result = runner.invoke(main, ["agents", "set", "planner"], input="text")

    assert result.exit_code != 0
    assert "No project found" in result.output


def test_agents_set_invalid_role():
    """agents set with unknown role should error."""
    from click.testing import CliRunner

    from agm.cli import main

    runner = CliRunner()
    with patch(
        "agm.cli._resolve_project_by_name",
        return_value={"dir": "/tmp/fake", "name": "test"},
    ):
        result = runner.invoke(main, ["agents", "set", "bogus", "-p", "test"], input="text")

    assert result.exit_code != 0
    assert "Unknown role" in result.output


# -- Global CRUD tests --


def test_global_set_get_roundtrip(tmp_path):
    """Setting global role instructions and reading them back works."""
    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Global planner rules.")
        assert get_global_role_text("planner") == "Global planner rules."


def test_global_set_preserves_other_roles(tmp_path):
    """Setting one global role does not erase another."""
    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Plan globally.")
        set_global_role_instructions("executor", "Execute globally.")
        assert get_global_role_text("planner") == "Plan globally."
        assert get_global_role_text("executor") == "Execute globally."


def test_global_reset_removes_role(tmp_path):
    """Resetting a global role removes it."""
    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Plan globally.")
        set_global_role_instructions("executor", "Execute globally.")
        reset_global_role_instructions("planner")
        assert get_global_role_text("planner") == ""
        assert get_global_role_text("executor") == "Execute globally."


def test_global_reset_deletes_file_when_empty(tmp_path):
    """Resetting the last global role deletes the file."""
    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Plan globally.")
        reset_global_role_instructions("planner")
        assert not (home / ".config" / "agm" / "agents.toml").exists()


def test_global_set_invalid_role_raises(tmp_path):
    """Setting an unknown global role raises ValueError."""
    home = tmp_path / "home"
    home.mkdir()
    with (
        patch("pathlib.Path.home", return_value=home),
        pytest.raises(ValueError, match="Unknown role"),
    ):
        set_global_role_instructions("invalid_role", "text")


def test_global_reset_invalid_role_raises(tmp_path):
    """Resetting an unknown global role raises ValueError."""
    home = tmp_path / "home"
    home.mkdir()
    with (
        patch("pathlib.Path.home", return_value=home),
        pytest.raises(ValueError, match="Unknown role"),
    ):
        reset_global_role_instructions("invalid_role")


# -- Global CLI flag tests --


def test_agents_global_set_from_stdin(tmp_path):
    """agents set --global reads instructions from stdin and writes to global agents.toml."""
    from click.testing import CliRunner

    from agm.cli import main

    home = tmp_path / "home"
    home.mkdir()

    runner = CliRunner()
    with patch("pathlib.Path.home", return_value=home):
        result = runner.invoke(
            main, ["agents", "set", "--global", "planner"], input="Global rules.\n"
        )

    assert result.exit_code == 0, result.output
    with patch("pathlib.Path.home", return_value=home):
        assert get_global_role_text("planner") == "Global rules."


def test_agents_global_reset_via_cli(tmp_path):
    """agents reset --global clears global role instructions."""
    from click.testing import CliRunner

    from agm.cli import main

    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Something.")

    runner = CliRunner()
    with patch("pathlib.Path.home", return_value=home):
        result = runner.invoke(main, ["agents", "reset", "--global", "planner"])

    assert result.exit_code == 0, result.output
    with patch("pathlib.Path.home", return_value=home):
        assert get_global_role_text("planner") == ""


def test_agents_json_includes_global_field(tmp_path):
    """agents show --json output includes the global field per role."""
    from click.testing import CliRunner

    from agm.cli import main

    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    with patch("pathlib.Path.home", return_value=home):
        set_global_role_instructions("planner", "Global planner.")
    set_project_role_instructions(str(project), "planner", "Project planner.")

    runner = CliRunner()
    mock_project = {"dir": str(project), "name": "test"}
    with (
        patch("agm.cli._resolve_agents_project", return_value=mock_project),
        patch("pathlib.Path.home", return_value=home),
    ):
        result = runner.invoke(main, ["agents", "show"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["roles"]["planner"]["global"] == "Global planner."
    assert data["roles"]["planner"]["project"] == "Project planner."
    assert "Global planner." in data["roles"]["planner"]["effective"]
    assert "Project planner." in data["roles"]["planner"]["effective"]
