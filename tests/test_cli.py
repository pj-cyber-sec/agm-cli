"""Tests for the CLI commands."""

from __future__ import annotations

import json
import os
import subprocess as sp
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agm.cli import main
from agm.db import (
    add_channel_message,
    add_plan_log,
    add_plan_question,
    add_project,
    add_task_block,
    add_task_log,
    answer_plan_question,
    claim_task,
    create_plan_request,
    create_session,
    create_task,
    finalize_plan_request,
    get_connection,
    get_project,
    record_status_change,
    set_plan_request_thread_id,
    set_project_base_branch,
    set_task_failure_reason,
    update_plan_request_status,
    update_task_status,
)
from agm.jobs_quality_gate import QualityCheckResult, QualityGateResult
from agm.queries import (
    format_plan_failure_error,
    watch_short_id,
    watch_truncate,
)
from agm.status_reference import STATUS_REFERENCE_SCHEMA, get_status_reference


def _qg_pass() -> QualityGateResult:
    return QualityGateResult(auto_fix_ran=False, auto_fix_committed=False, checks=[])


def _qg_fail(name: str, output: str) -> QualityGateResult:
    return QualityGateResult(
        auto_fix_ran=False,
        auto_fix_committed=False,
        checks=[QualityCheckResult(name=name, passed=False, output=output, duration_ms=100)],
    )


def _has_ansi(text: str) -> bool:
    return "\x1b[" in text


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# JSON error handling (group-level)
# ---------------------------------------------------------------------------


def test_json_flag_formats_usage_error_as_json():
    """Unknown options produce JSON error when --json is present."""
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--no-such-flag"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    err = payload["error"].lower()
    assert "no-such-flag" in err or "no such option" in err


def test_json_flag_formats_missing_arg_as_json(db_conn_path):
    """Missing required arguments produce JSON error when --json is present."""
    conn, db_path = db_conn_path
    runner = CliRunner()
    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = runner.invoke(main, ["plan", "show"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "plan_id" in payload["error"].lower() or "missing" in payload["error"].lower()


def test_error_always_json():
    """All errors are now JSON (no human-only mode)."""
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--no-such-flag"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert (
        "no-such-flag" in payload["error"].lower() or "no such option" in payload["error"].lower()
    )


def test_init_registers_project_and_gitignore(tmp_path, monkeypatch):
    """agm init registers the project and adds .agm/ to .gitignore."""
    # Create a fake git repo with a commit
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("node_modules/\n")
    db_path = tmp_path / "agm.db"

    # Mock subprocess: git rev-parse succeeds (has commits), backends not found
    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=0)
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    monkeypatch.chdir(tmp_path)
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", side_effect=mock_run),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["name"] == tmp_path.name
        assert payload["dir"] == str(tmp_path)
        assert "warnings" in payload
        assert "codex cli not found" in payload["warnings"][0].lower()
        # Check .gitignore was updated
        gitignore_content = (tmp_path / ".gitignore").read_text()
        assert ".agm/" in gitignore_content


def test_init_accepts_positional_path(tmp_path):
    """agm init accepts a positional repository path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".gitignore").write_text("node_modules/\n")
    db_path = tmp_path / "agm.db"

    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=0)
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", side_effect=mock_run),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(repo)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["name"] == "repo"
        assert payload["dir"] == str(repo)


def test_init_rejects_path_and_dir_together(tmp_path):
    """agm init rejects using positional PATH and --dir simultaneously."""
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    db_path = tmp_path / "agm.db"

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(repo), "--dir", str(other)])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "either path or --dir" in payload["error"].lower()


def test_init_rejects_name_collision_in_different_directory(tmp_path):
    """agm init errors when --name matches an existing project in another directory."""
    db_path = tmp_path / "agm.db"
    existing = tmp_path / "existing"
    target = tmp_path / "target"
    existing.mkdir()
    target.mkdir()
    (existing / ".git").mkdir()
    (target / ".git").mkdir()

    conn = get_connection(db_path)
    add_project(conn, "same-name", str(existing))
    conn.close()

    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=0)
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", side_effect=mock_run),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target), "--name", "same-name"])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "already registered" in payload["error"].lower()


def test_init_name_collision_does_not_initialize_git(tmp_path):
    """Name collisions are validated before any git bootstrap side effects."""
    db_path = tmp_path / "agm.db"
    existing = tmp_path / "existing"
    target = tmp_path / "target"
    existing.mkdir()
    target.mkdir()
    (existing / ".git").mkdir()

    conn = get_connection(db_path)
    add_project(conn, "same-name", str(existing))
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target), "--name", "same-name"])
        assert result.exit_code != 0
        assert not (target / ".git").exists()


def test_init_git_commit_failure_is_json_error(tmp_path):
    """agm init wraps git commit errors as ClickException JSON output."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    db_path = tmp_path / "agm.db"

    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=1)
        if cmd[0] == "git" and cmd[1] == "add":
            return MagicMock(returncode=0)
        if cmd[0] == "git" and cmd[1] == "commit":
            raise sp.CalledProcessError(returncode=1, cmd=cmd, stderr="Author identity unknown")
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", side_effect=mock_run),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(repo)])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "failed to create initial commit" in payload["error"].lower()


def test_agents_show_defaults_to_most_specific_project_context(tmp_path, monkeypatch):
    """agm agents show uses the deepest matching registered project path."""
    root = tmp_path / "workspace" / "repo"
    child = root / "nested"
    (child / "subdir").mkdir(parents=True, exist_ok=True)
    parent_dir = tmp_path / "workspace"
    home = tmp_path / "home"
    db_path = tmp_path / "agm.db"

    conn = get_connection(db_path)
    add_project(conn, "parent", str(parent_dir))
    add_project(conn, "child", str(root))
    conn.close()

    (parent_dir / ".agm").mkdir(exist_ok=True)
    (parent_dir / ".agm" / "agents.toml").write_text('[planner]\ninstructions = "parent planner"\n')
    (root / ".agm").mkdir(exist_ok=True)
    (root / ".agm" / "agents.toml").write_text('[planner]\ninstructions = "child planner"\n')

    monkeypatch.chdir(child / "subdir")
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["agents", "show"])
        assert result.exit_code == 0
        assert "child planner" in result.output
        assert "parent planner" not in result.output


def test_agents_show_falls_back_to_global_when_no_project_context(tmp_path, monkeypatch):
    """agm agents show falls back to global config when no project matches cwd."""
    home = tmp_path / "home"
    db_path = tmp_path / "agm.db"
    conn = get_connection(db_path)
    add_project(conn, "other", str(tmp_path / "other"))
    conn.close()

    (home / ".config" / "agm").mkdir(parents=True, exist_ok=True)
    (home / ".config" / "agm" / "agents.toml").write_text(
        '[planner]\ninstructions = "global only"\n'
    )

    monkeypatch.chdir(tmp_path)
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["agents", "show"])
        assert result.exit_code == 0
        assert "global only" in result.output


def test_agents_show_global_flag_forces_global_context(tmp_path, monkeypatch):
    """agm agents show --global shows global config even when in a project."""
    project_root = tmp_path / "project"
    (project_root / ".agm").mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    db_path = tmp_path / "agm.db"

    conn = get_connection(db_path)
    add_project(conn, "project", str(project_root))
    conn.close()

    (project_root / ".agm" / "agents.toml").write_text(
        '[planner]\ninstructions = "project planner"\n'
    )
    (home / ".config" / "agm").mkdir(parents=True, exist_ok=True)
    (home / ".config" / "agm" / "agents.toml").write_text(
        '[planner]\ninstructions = "global planner"\n'
    )

    monkeypatch.chdir(project_root)
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["agents", "show", "--global"])
        assert result.exit_code == 0
        assert "global planner" in result.output
        assert "project planner" not in result.output


def test_agents_init_creates_project_and_global_scaffolds(tmp_path, monkeypatch):
    """agm agents init creates scaffold files for project and global contexts."""
    project_root = tmp_path / "project"
    home = tmp_path / "home"
    (project_root).mkdir()
    db_path = tmp_path / "agm.db"
    conn = get_connection(db_path)
    add_project(conn, "project", str(project_root))
    conn.close()

    monkeypatch.chdir(project_root)
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
    ):
        runner = CliRunner()

        project_result = runner.invoke(main, ["agents", "init"])
        assert project_result.exit_code == 0
        scaffold_path = project_root / ".agm" / "agents.toml"
        assert scaffold_path.exists()
        contents = scaffold_path.read_text()
        assert "[planner]" in contents
        assert "Add planner instructions here." in contents

        global_result = runner.invoke(main, ["agents", "init", "--global"])
        assert global_result.exit_code == 0
        global_path = home / ".config" / "agm" / "agents.toml"
        assert global_path.exists()
        global_contents = global_path.read_text()
        assert "[reviewer]" in global_contents
        assert "Add reviewer instructions here." in global_contents


def test_agents_edit_requires_editor_and_target_file(tmp_path, monkeypatch):
    """agm agents edit reports clear errors when prerequisites are missing."""
    home = tmp_path / "home"
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    conn = get_connection(db_path)
    add_project(conn, "project", str(project_dir))
    conn.close()

    target = project_dir / ".agm" / "agents.toml"
    (target.parent).mkdir(parents=True, exist_ok=True)
    target.write_text('[planner]\ninstructions = "project"\n')

    monkeypatch.chdir(project_dir)
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
        patch.dict(os.environ, {"EDITOR": ""}, clear=False),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["agents", "edit"])
        assert result.exit_code != 0
        assert "No EDITOR configured" in result.output

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.home", return_value=home),
        patch.dict(os.environ, {"EDITOR": "vim"}, clear=False),
        patch("subprocess.run") as run_editor,
    ):
        target.unlink()
        runner = CliRunner()
        result = runner.invoke(main, ["agents", "edit"])
        assert result.exit_code != 0
        assert "Missing target file" in result.output
        run_editor.assert_not_called()


def test_status_command(db_conn_path):
    """status shows overview of active plans, tasks, and queue health."""
    conn, db_path = db_conn_path
    proj_id = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()[0]
    plan = create_plan_request(
        conn,
        project_id=proj_id,
        prompt="test prompt",
        actor="alice",
        caller="cli",
        backend="codex",
    )
    create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="status task",
        description="active task",
    )
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with (
        patch("agm.cli.connect", _mock_connect),
        patch(
            "agm.queue.get_queue_counts_safe",
            return_value={
                "ok": True,
                "queues": {"default": {"queued": 2, "running": 1, "failed": 0}},
            },
        ),
        patch("agm.queue.get_codex_rate_limits_safe", return_value=None),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "projects" in payload
        assert any(p["project"] == "testproj" for p in payload["projects"])
        assert payload["queue"]["queues"]["default"]["queued"] == 2


def test_status_json_includes_recent_failures(db_conn_path):
    """status --json should include recent failure diagnostics per project."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    failed = create_plan_request(
        conn,
        project_id=pid,
        prompt="failure summary",
        actor="alice",
        caller="cli",
        backend="codex",
    )
    update_plan_request_status(conn, failed["id"], "failed")
    add_plan_log(
        conn,
        plan_id=failed["id"],
        level="ERROR",
        message="json failure details",
    )
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with (
        patch("agm.cli.connect", _mock_connect),
        patch(
            "agm.queue.get_queue_counts_safe",
            return_value={"ok": True, "queues": {}},
        ),
        patch("agm.queue.get_codex_rate_limits_safe", return_value=None),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert len(payload["projects"]) == 1
        project = payload["projects"][0]
        assert project["project"] == "testproj"
        assert len(project["recent_failures"]) == 1
        failure = project["recent_failures"][0]
        assert failure["plan_id"] == failed["id"]
        assert failure["status"] == "failed"
        assert failure["error_snippet"] == format_plan_failure_error("json failure details")
        assert failure["failed"] is not None


def test_help_status_json_is_parseable_and_stable():
    """help-status --json should emit a stable lifecycle payload."""
    runner = CliRunner()
    first = runner.invoke(main, ["help-status"], color=True)
    second = runner.invoke(main, ["help-status"], color=True)
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert not _has_ansi(first.output)
    assert not _has_ansi(second.output)

    payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert payload == second_payload
    assert payload == get_status_reference()
    assert payload["schema"] == STATUS_REFERENCE_SCHEMA
    assert payload["lifecycles"][0]["type"] == "plan"
    assert payload["lifecycles"][1]["type"] == "task"
    assert payload["lifecycles"][2]["type"] == "task_creation"

    assert any(
        status["status"] == "awaiting_input" for status in payload["lifecycles"][0]["statuses"]
    )
    assert any(status["status"] == "review" for status in payload["lifecycles"][1]["statuses"])
    assert any(status["status"] == "finalized" for status in payload["lifecycles"][0]["statuses"])


def test_plan_stats_model_usage_breakdown(db_conn_path):
    """plan stats should include model usage summaries for non-empty model data."""
    conn, db_path = db_conn_path
    proj_id = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()[0]
    plan = create_plan_request(
        conn,
        project_id=proj_id,
        prompt="test prompt",
        actor="alice",
        caller="cli",
        backend="codex",
    )
    plan_id = plan["id"]
    conn.execute("UPDATE plans SET model = 'plan-model' WHERE id = ?", (plan_id,))
    task_one = create_task(
        conn,
        plan_id=plan_id,
        ordinal=0,
        title="Task one",
        description="d",
    )
    task_two = create_task(
        conn,
        plan_id=plan_id,
        ordinal=1,
        title="Task two",
        description="d",
    )
    conn.execute("UPDATE tasks SET model = ? WHERE id = ?", ("task-model-a", task_one["id"]))
    conn.execute("UPDATE tasks SET model = ? WHERE id = ?", ("task-model-b", task_two["id"]))
    conn.commit()
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with patch("agm.cli.connect", _mock_connect):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "stats", plan_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["plan_model_counts"] == {"plan-model": 1}
        assert data["task_model_counts"] == {"task-model-a": 1, "task-model-b": 1}


def test_project_stats_command(db_conn_path):
    """project stats shows pipeline analytics for a project."""
    conn, db_path = db_conn_path
    proj_id = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()[0]
    create_plan_request(
        conn,
        project_id=proj_id,
        prompt="test prompt",
        actor="alice",
        caller="cli",
        backend="codex",
    )
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with patch("agm.cli.connect", _mock_connect):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "stats", "testproj"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["project"] == "testproj"
        assert data["plan_model_counts"] == {}
        assert data["task_model_counts"] == {}


def test_plan_request_rejects_invalid_backend(db_conn_path):
    """An invalid backend should be rejected by Click's choice validation."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "request", "test prompt", "-p", "testproj", "--backend", "invalid"],
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


def test_plan_request_marks_failed_on_enqueue_error(db_conn_path):
    """If enqueue fails, the plan should be marked as failed in the DB."""
    conn, db_path = db_conn_path
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.queue.get_queue",
            side_effect=ConnectionError("Redis unavailable"),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "request", "test prompt", "-p", "testproj"],
        )
        assert result.exit_code != 0
        assert "Failed to enqueue" in result.output

    # Verify the plan was marked failed in DB
    verify_conn = get_connection(db_path)
    plans = verify_conn.execute("SELECT * FROM plans").fetchall()
    assert len(plans) == 1
    assert plans[0]["status"] == "failed"
    verify_conn.close()


def test_project_list(db_conn_path):
    """project list should display registered projects."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "list"])
        assert result.exit_code == 0
        assert "testproj" in result.output


def test_project_remove_project(db_conn_path):
    """project remove --yes should delete project data and report counts."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn,
        project_id=pid,
        prompt="cleanup plan",
        actor="cli",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T0", description="task")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli._project_remove_queue_jobs", return_value=1) as mock_remove_queue_jobs,
        patch("agm.cli._project_remove_log_files", return_value=1) as mock_remove_log_files,
        patch("agm.cli._project_remove_agm_dir", return_value=False),
        patch("agm.cli._project_remove_setup_log", return_value=False),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "remove", "--yes", "testproj"])
        assert result.exit_code == 0
        mock_remove_queue_jobs.assert_called_once_with([plan["id"]], [task["id"]])
        mock_remove_log_files.assert_called_once_with([plan["id"]], [task["id"]])

    verify_conn = get_connection(db_path)
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM projects WHERE name = 'testproj'").fetchone()[0]
        == 0
    )
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM plans WHERE project_id = ?", (pid,)).fetchone()[0]
        == 0
    )
    verify_conn.close()


def test_project_remove_declines_without_confirmation(db_conn_path):
    """project remove should ask for confirmation before destructive work."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn,
        project_id=pid,
        prompt="cleanup plan",
        actor="cli",
        caller="cli",
        backend="codex",
    )
    create_task(conn, plan_id=plan["id"], ordinal=0, title="T0", description="task")
    conn.commit()
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli.remove_project") as mock_remove_project,
        patch("agm.cli._project_remove_queue_jobs") as mock_remove_queue_jobs,
        patch("agm.cli._project_remove_log_files") as mock_remove_log_files,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "remove", "testproj"], input="n\n")
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_remove_project.assert_not_called()
        mock_remove_queue_jobs.assert_not_called()
        mock_remove_log_files.assert_not_called()

    verify_conn = get_connection(db_path)
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM projects WHERE name = 'testproj'").fetchone()[0]
        == 1
    )
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM plans WHERE project_id = ?", (pid,)).fetchone()[0]
        == 1
    )
    verify_conn.close()


def test_project_remove_approves_with_prompt(db_conn_path):
    """project remove should delete data when prompt is approved."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn,
        project_id=pid,
        prompt="cleanup plan",
        actor="cli",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T0", description="task")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli._project_remove_queue_jobs", return_value=1) as mock_remove_queue_jobs,
        patch("agm.cli._project_remove_log_files", return_value=1) as mock_remove_log_files,
        patch("agm.cli._project_remove_agm_dir", return_value=False),
        patch("agm.cli._project_remove_setup_log", return_value=False),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "remove", "testproj"], input="y\n")
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_remove_queue_jobs.assert_called_once_with([plan["id"]], [task["id"]])
        mock_remove_log_files.assert_called_once_with([plan["id"]], [task["id"]])

    verify_conn = get_connection(db_path)
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM projects WHERE name = 'testproj'").fetchone()[0]
        == 0
    )
    assert (
        verify_conn.execute("SELECT COUNT(*) FROM plans WHERE project_id = ?", (pid,)).fetchone()[0]
        == 0
    )
    verify_conn.close()


def test_project_remove_cleans_agm_dir(tmp_path, db_conn_path):
    """project remove cleans up the .agm/ directory."""
    conn, db_path = db_conn_path
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    agm_dir = project_dir / ".agm"
    agm_dir.mkdir()
    (agm_dir / "agents.toml").write_text("[enrichment]\n")
    (agm_dir / "memory.md").write_text("# Memory\n")

    add_project(conn, "agm-dir-test", str(project_dir))

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli._project_remove_queue_jobs", return_value=0),
        patch("agm.cli._project_remove_log_files", return_value=0),
        patch("agm.cli._project_remove_setup_log", return_value=False),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "remove", "--yes", "agm-dir-test"])
        assert result.exit_code == 0

    assert not agm_dir.exists()


@pytest.mark.parametrize(
    "command",
    [
        ["plan", "list"],
        ["task", "list"],
    ],
)
def test_empty_list_returns_empty_json(db_conn_path, command):
    """list commands with no data should return an empty JSON array."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, command)
        assert result.exit_code == 0
        assert json.loads(result.output) == []


@pytest.mark.parametrize(
    "command, option",
    [
        (["plan", "list", "--status", "bogus"], "--status"),
        (["task", "list", "--status", "bogus"], "--status"),
        (["task", "list", "--priority", "bogus"], "--priority"),
    ],
)
def test_invalid_enum_params(db_conn_path, command, option):
    """Commands with invalid enum options should error at parse time."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, command)
        assert result.exit_code != 0
        assert f"Invalid value for '{option}'" in result.output


def test_plan_list_json_contract(db_conn_path):
    """plan list --json should include stable per-plan fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="json list", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list"], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        row = parsed[0]
        assert row["id"] == p["id"]
        assert row["prompt"] == "json list"
        assert row["status"] == "pending"
        assert row["backend"] == "codex"
        assert row["error"] is None
        assert set(row.keys()) >= {
            "id",
            "project_id",
            "prompt",
            "status",
            "actor",
            "caller",
            "backend",
            "input_tokens",
            "output_tokens",
            "created_at",
            "updated_at",
            "error",
        }


def test_plan_list_renders_colored_status_labels(db_conn_path):
    """plan list should render colorized status labels in human output."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    pending = create_plan_request(
        conn, project_id=pid, prompt="pending plan", caller="cli", backend="codex"
    )
    finalized = create_plan_request(
        conn, project_id=pid, prompt="finalized plan", caller="cli", backend="codex"
    )
    finalize_plan_request(conn, finalized["id"], '{"title":"done","tasks":[]}')
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list", "--all"], color=True)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert f"{pending['id']}" in clean and "pending" in clean
        assert f"{finalized['id']}" in clean and "finalized" in clean


def test_plan_list_failed_row_includes_truncated_error_context(db_conn_path):
    """plan list should append a truncated error snippet for failed rows."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    add_plan_log(
        conn,
        plan_id=failed["id"],
        level="ERROR",
        message="x" * 200,
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list", "--all"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert failed["id"] in clean
        assert "x" * 50 in clean
        assert "x" * 200 not in clean


def test_plan_list_json_includes_failed_error_field(db_conn_path):
    """plan list --json should include truncated failure snippets and null for missing logs."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    failed_with_log = create_plan_request(
        conn, project_id=pid, prompt="failed with log", caller="cli", backend="codex"
    )
    failed_no_log = create_plan_request(
        conn, project_id=pid, prompt="failed without log", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed_with_log["id"], "failed")
    update_plan_request_status(conn, failed_no_log["id"], "failed")
    add_plan_log(
        conn,
        plan_id=failed_with_log["id"],
        level="ERROR",
        message="x" * 200,
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list", "--all"], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        rows = json.loads(result.output)
        assert len(rows) == 2
        rows_by_id = {row["id"]: row for row in rows}
        assert rows_by_id[failed_with_log["id"]]["error"] == format_plan_failure_error("x" * 200)
        assert len(rows_by_id[failed_with_log["id"]]["error"]) <= 80
        assert rows_by_id[failed_no_log["id"]]["error"] is None


def test_plan_list_active_default_hides_terminal_states(db_conn_path):
    """plan list should show active plans only by default."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    active_pending = create_plan_request(
        conn, project_id=pid, prompt="active plan", caller="cli", backend="codex"
    )
    active_running = create_plan_request(
        conn, project_id=pid, prompt="running plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, active_running["id"], "running")
    active_awaiting = create_plan_request(
        conn, project_id=pid, prompt="awaiting_input plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, active_awaiting["id"], "awaiting_input")
    finalized = create_plan_request(
        conn, project_id=pid, prompt="finalized plan", caller="cli", backend="codex"
    )
    finalize_plan_request(conn, finalized["id"], '{"title":"done","summary":"s","tasks":[]}')
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    cancelled = create_plan_request(
        conn, project_id=pid, prompt="cancelled plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, cancelled["id"], "cancelled")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list"], color=True)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert active_pending["id"] in clean
        assert active_running["id"] in clean
        assert active_awaiting["id"] in clean
        assert finalized["id"] not in clean
        assert failed["id"] not in clean
        assert cancelled["id"] not in clean


def test_plan_list_status_filter_finalized(db_conn_path):
    """plan list -s finalized should return finalized plans only."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    active = create_plan_request(
        conn, project_id=pid, prompt="active plan", caller="cli", backend="codex"
    )
    finalized = create_plan_request(
        conn, project_id=pid, prompt="finalized plan", caller="cli", backend="codex"
    )
    finalize_plan_request(conn, finalized["id"], '{"title":"done","summary":"s","tasks":[]}')
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list", "-s", "finalized"], color=True)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert active["id"] not in clean
        assert finalized["id"] in clean
        assert failed["id"] not in clean


def test_plan_list_all_includes_terminal_history(db_conn_path):
    """plan list --all should include finalized, cancelled, and failed plans."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    active = create_plan_request(
        conn, project_id=pid, prompt="active plan", caller="cli", backend="codex"
    )
    finalized = create_plan_request(
        conn, project_id=pid, prompt="finalized plan", caller="cli", backend="codex"
    )
    finalize_plan_request(conn, finalized["id"], '{"title":"done","summary":"s","tasks":[]}')
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    cancelled = create_plan_request(
        conn, project_id=pid, prompt="cancelled plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, cancelled["id"], "cancelled")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result_all = runner.invoke(main, ["plan", "list", "--all"], color=True)
        assert result_all.exit_code == 0
        clean_all = _strip_ansi(result_all.output)
        assert active["id"] in clean_all
        assert finalized["id"] in clean_all
        assert failed["id"] in clean_all
        assert cancelled["id"] in clean_all

        result_short = runner.invoke(main, ["plan", "list", "-a"], color=True)
        assert result_short.exit_code == 0
        clean_short = _strip_ansi(result_short.output)
        assert active["id"] in clean_short
        assert finalized["id"] in clean_short
        assert failed["id"] in clean_short
        assert cancelled["id"] in clean_short


def test_plan_list_all_with_explicit_status_prefers_status(db_conn_path):
    """plan list --status should remain authoritative over --all."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    active = create_plan_request(
        conn, project_id=pid, prompt="active plan", caller="cli", backend="codex"
    )
    finalized = create_plan_request(
        conn, project_id=pid, prompt="finalized plan", caller="cli", backend="codex"
    )
    finalize_plan_request(conn, finalized["id"], '{"title":"done","summary":"s","tasks":[]}')
    failed = create_plan_request(
        conn, project_id=pid, prompt="failed plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    cancelled = create_plan_request(
        conn, project_id=pid, prompt="cancelled plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, cancelled["id"], "cancelled")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "list", "--all", "-s", "finalized"], color=True)
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert finalized["id"] in clean
        assert active["id"] not in clean
        assert failed["id"] not in clean
        assert cancelled["id"] not in clean


def test_plan_list_help_documents_active_default_and_all_option():
    """plan list --help should document active-default and --all filtering."""
    runner = CliRunner()
    result = runner.invoke(main, ["plan", "list", "--help"])
    assert result.exit_code == 0
    help_text = " ".join(result.output.replace("\n", " ").split())
    assert "Shows active plans by default." in help_text
    assert "Use --all (-a) to include finalized/cancelled/failed history." in help_text


def test_plan_failures_default_listing(db_conn_path):
    """plan failures should list recent failed plans with compact diagnostics."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    failed = create_plan_request(
        conn,
        project_id=pid,
        prompt="plan failure list prompt " * 3,
        caller="cli",
        backend="codex",
    )
    update_plan_request_status(conn, failed["id"], "failed")
    from agm.db import add_plan_log

    add_plan_log(
        conn,
        plan_id=failed["id"],
        level="ERROR",
        message="x" * 250,
    )
    create_plan_request(
        conn,
        project_id=pid,
        prompt="should not show",
        caller="cli",
        backend="codex",
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "failures"])
        assert result.exit_code == 0
        items = json.loads(result.output)
        assert len(items) >= 1
        fail_entry = next(e for e in items if e["plan_id"] == failed["id"])
        assert fail_entry["project"] == "testproj"


def test_plan_failures_project_filter(db_conn_path):
    """plan failures with --project should only list matching project failures."""
    conn, db_path = db_conn_path
    add_project(conn, "other", "/tmp/other")
    conn.commit()
    proj_a = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    proj_b = conn.execute("SELECT id FROM projects WHERE name = 'other'").fetchone()["id"]
    plan_a = create_plan_request(
        conn, project_id=proj_a, prompt="project A failure", caller="cli", backend="codex"
    )
    plan_b = create_plan_request(
        conn, project_id=proj_b, prompt="project B failure", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, plan_a["id"], "failed")
    update_plan_request_status(conn, plan_b["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "failures", "-p", "other"])
        assert result.exit_code == 0
        assert plan_b["id"] in result.output
        assert plan_a["id"] not in result.output


def test_plan_failures_empty(db_conn_path):
    """plan failures with no failed plans should return empty JSON array."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "failures"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []


def test_plan_failures_json_parseable(db_conn_path):
    """plan failures --json should emit parseable JSON with required fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    failed = create_plan_request(
        conn, project_id=pid, prompt="json failure", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, failed["id"], "failed")
    from agm.db import add_plan_log

    add_plan_log(
        conn,
        plan_id=failed["id"],
        level="ERROR",
        message="failure details",
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "failures"], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        entry = parsed[0]
        assert set(entry.keys()) >= {
            "plan_id",
            "project_id",
            "project",
            "source",
            "prompt",
            "prompt_snippet",
            "error",
            "error_snippet",
            "created_at",
            "updated_at",
            "failed",
        }
        assert entry["plan_id"] == failed["id"]
        assert entry["project"] == "testproj"
        assert entry["source"] == "plan"


def test_plan_show(db_conn_path):
    """plan show should return JSON plan details."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="test plan", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "show", p["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["prompt"] == "test plan"
        assert payload["id"] == p["id"]
        assert payload["status"] == "pending"


def test_plan_show_displays_model_name(db_conn_path):
    """plan show should display model when present in both human and json outputs."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="test plan", caller="cli", backend="codex")
    conn.execute("UPDATE plans SET model = 'plan-view-model' WHERE id = ?", (p["id"],))
    conn.commit()
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with patch("agm.cli.connect", _mock_connect):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "show", p["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["model"] == "plan-view-model"


def test_plan_show_json_contract(db_conn_path):
    """plan show should return the raw plan row contract."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="json view", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "show", p["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert parsed["id"] == p["id"]
        assert parsed["prompt"] == "json view"
        assert parsed["status"] == "pending"
        assert parsed["backend"] == "codex"
        assert set(parsed.keys()) >= {
            "id",
            "project_id",
            "prompt",
            "status",
            "actor",
            "caller",
            "backend",
            "input_tokens",
            "output_tokens",
            "created_at",
            "updated_at",
        }
        assert "Identity:" not in result.output
        assert "tasks" not in parsed


def test_plan_show_tasks_json_includes_task_shape(db_conn_path):
    """plan show --tasks --json should mirror task list JSON rows."""
    conn, db_path = db_conn_path
    plan, _ = _make_task(conn, db_path)
    failed_task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="Failed task",
        description="d",
    )
    conn.commit()
    completed = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=2,
        title="Completed task",
        description="d",
    )
    update_task_status(conn, completed["id"], "completed")
    update_task_status(conn, failed_task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        task_list_result = runner.invoke(main, ["task", "list", "--plan", plan["id"]])
        assert task_list_result.exit_code == 0
        task_rows = json.loads(task_list_result.output)
        assert isinstance(task_rows, list)
        assert completed["id"] not in {row["id"] for row in task_rows}
        assert failed_task["id"] not in {row["id"] for row in task_rows}

        show_result = runner.invoke(main, ["plan", "show", plan["id"], "--tasks"])
        assert show_result.exit_code == 0
        payload = json.loads(show_result.output)
        assert payload["id"] == plan["id"]
        assert "tasks" in payload
        assert payload["tasks"] == task_rows

        no_tasks_result = runner.invoke(main, ["plan", "show", plan["id"]])
        assert no_tasks_result.exit_code == 0
        no_tasks_payload = json.loads(no_tasks_result.output)
        assert "tasks" not in no_tasks_payload


def test_plan_show_not_found(db_conn_path):
    """plan show for nonexistent plan should error."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

        json_result = runner.invoke(main, ["plan", "show", "nonexistent"])
        assert json_result.exit_code != 0
        assert "not found" in json_result.output


def test_plan_logs_empty(db_conn_path):
    """plan logs for a plan with no logs should return empty JSON."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "logs", p["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 0
        assert payload["logs"] == []


def test_plan_history_json_is_parseable(db_conn_path):
    """plan history --json should emit parseable payload and required chain fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    root = create_plan_request(
        conn, project_id=pid, prompt="root prompt", caller="cli", backend="codex"
    )
    child = create_plan_request(
        conn,
        project_id=pid,
        prompt="child prompt",
        caller="cli",
        backend="codex",
        parent_id=root["id"],
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "history", child["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["plan_id"] == child["id"]
        chain = payload["chain"]
        assert isinstance(chain, list)
        assert len(chain) == 2
        assert chain[0]["id"] == root["id"]
        assert chain[1]["id"] == child["id"]
        assert chain[1]["is_target"] is True
        assert chain[0]["position"] == 1
        assert chain[1]["position"] == 2
        assert chain[1]["total"] == 2
        assert set(chain[1].keys()) >= {
            "id",
            "status",
            "prompt",
            "prompt_preview",
            "created_at",
            "updated_at",
            "is_target",
            "position",
            "total",
        }


def test_plan_timeline_json_is_parseable(db_conn_path):
    """plan timeline --json should emit parseable timeline entries with durations."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="timeline plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, p["id"], "running", record_history=True)
    update_plan_request_status(conn, p["id"], "finalized", record_history=True)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "timeline", p["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["plan_id"] == p["id"]
        timeline = payload["timeline"]
        assert isinstance(timeline, list)
        assert timeline
        entry = timeline[0]
        assert set(entry.keys()) >= {
            "id",
            "created_at",
            "old_status",
            "new_status",
            "actor",
            "next_created_at",
            "duration_seconds",
            "duration_label",
            "duration",
        }


def test_plan_questions_json_is_parseable(db_conn_path):
    """plan questions --json should emit parseable payload and required fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="ask question", caller="cli", backend="codex"
    )
    add_plan_question(conn, plan_id=p["id"], question="what is first?")
    q2 = add_plan_question(conn, plan_id=p["id"], question="what is second?")
    answer_plan_question(conn, q2["id"], "because")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "questions", p["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["plan_id"] == p["id"]
        assert payload["unanswered_only"] is False
        assert payload["count"] == 2
        assert len(payload["questions"]) == 2
        status_map = {q["question"]: q["status"] for q in payload["questions"]}
        assert status_map["what is first?"] == "pending"
        assert status_map["what is second?"] == "answered"
        assert set(payload["questions"][0].keys()) >= {
            "id",
            "question",
            "question_preview",
            "header",
            "options",
            "multi_select",
            "answer",
            "answered_by",
            "answered_at",
            "status",
            "created_at",
        }


def test_plan_answer_auto_resumes_when_all_answered(db_conn_path):
    """plan answer should auto-resume enrichment when last question is answered."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="build auth", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, p["id"], "running")
    update_plan_request_status(conn, p["id"], "awaiting_input")
    q1 = add_plan_question(conn, plan_id=p["id"], question="OAuth or JWT?")
    q2 = add_plan_question(conn, plan_id=p["id"], question="Which DB?")
    conn.close()

    mock_enqueue = MagicMock()

    # Answer first question — should not resume yet
    with (
        patch("agm.db.get_connection", return_value=get_connection(db_path)),
        patch("agm.queue.enqueue_enrichment", mock_enqueue),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "answer", q1["id"], "OAuth"])
        assert result.exit_code == 0
        mock_enqueue.assert_not_called()

    # Answer second question — should auto-resume
    with (
        patch("agm.db.get_connection", return_value=get_connection(db_path)),
        patch("agm.queue.enqueue_enrichment", mock_enqueue),
    ):
        result = runner.invoke(main, ["plan", "answer", q2["id"], "PostgreSQL"])
        assert result.exit_code == 0
        mock_enqueue.assert_called_once_with(p["id"])

    # Verify prompt_status transitioned to enriching
    verify_conn = get_connection(db_path)
    found = verify_conn.execute(
        "SELECT prompt_status FROM plans WHERE id = ?", (p["id"],)
    ).fetchone()
    assert found["prompt_status"] == "enriching"
    verify_conn.close()


def test_plan_answer_no_resume_when_not_awaiting_input(db_conn_path):
    """plan answer should not auto-resume if plan is not awaiting_input."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="do stuff", caller="cli", backend="codex")
    update_plan_request_status(conn, p["id"], "running")
    q = add_plan_question(conn, plan_id=p["id"], question="question?")
    # Plan stays 'running' — not 'awaiting_input'
    conn.close()

    mock_enqueue = MagicMock()

    with (
        patch("agm.db.get_connection", return_value=get_connection(db_path)),
        patch("agm.queue.enqueue_plan_request", mock_enqueue),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "answer", q["id"], "answer"])
        assert result.exit_code == 0
        # Should NOT resume — plan is not awaiting_input
        mock_enqueue.assert_not_called()
        assert "Resuming" not in result.output


def test_plan_logs_json_is_parseable(db_conn_path):
    """plan logs --json should emit parseable payload and required fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="logged plan", caller="cli", backend="codex"
    )
    add_plan_log(conn, plan_id=p["id"], level="INFO", message="plan started")
    add_plan_log(conn, plan_id=p["id"], level="ERROR", message="plan failed")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "logs", p["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["plan_id"] == p["id"]
        assert payload["count"] == 2
        assert payload["level"] is None
        assert len(payload["logs"]) == 2
        assert set(payload["logs"][0].keys()) >= {
            "id",
            "level",
            "message",
            "created_at",
        }


def test_plan_retry(db_conn_path):
    """plan retry should reset a failed plan and re-enqueue it."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="retry me", caller="cli", backend="codex")
    update_plan_request_status(conn, p["id"], "failed")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = f"plan-{p['id']}"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retry", p["id"]])
        assert result.exit_code == 0

    # Verify plan was reset to pending then re-enqueued
    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM plans WHERE id = ?", (p["id"],)).fetchone()
    assert found["status"] == "pending"
    assert found["pid"] is None
    assert found["thread_id"] is None
    verify_conn.close()


def test_plan_retry_not_failed(db_conn_path):
    """plan retry should reject non-failed plans."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="still running", caller="cli", backend="codex"
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retry", p["id"]])
        assert result.exit_code != 0
        assert "not 'failed'" in result.output


def test_plan_retry_finalized_suggests_continue(db_conn_path):
    """plan retry on a finalized plan should suggest plan continue."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="done", caller="cli", backend="codex")
    finalize_plan_request(conn, p["id"], "the plan text")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retry", p["id"]])
        assert result.exit_code != 0
        assert "plan continue" in result.output


# -- plan continue --


def test_plan_continue(db_conn_path):
    """plan continue should create a child plan and enqueue it."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    parent = create_plan_request(
        conn, project_id=pid, prompt="original", caller="cli", backend="codex"
    )
    set_plan_request_thread_id(conn, parent["id"], "thread-abc")
    finalize_plan_request(conn, parent["id"], "plan text")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "plan-child123"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "continue", parent["id"], "review the fixes"],
        )
        assert result.exit_code == 0

    # Verify child plan was created with parent_id
    verify_conn = get_connection(db_path)
    plans = verify_conn.execute(
        "SELECT * FROM plans WHERE parent_id = ?", (parent["id"],)
    ).fetchall()
    assert len(plans) == 1
    assert plans[0]["prompt"] == "review the fixes"
    assert plans[0]["parent_id"] == parent["id"]
    assert plans[0]["backend"] == "codex"
    verify_conn.close()


def test_plan_continue_not_finalized(db_conn_path):
    """plan continue should reject non-finalized plans."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="still pending", caller="cli", backend="codex"
    )
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "continue", p["id"], "follow up"])
        assert result.exit_code != 0
        assert "not 'finalized'" in result.output


def test_plan_continue_no_thread(db_conn_path):
    """plan continue should reject plans without a thread_id."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="no thread", caller="cli", backend="codex")
    finalize_plan_request(conn, p["id"], "text")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "continue", p["id"], "follow up"])
        assert result.exit_code != 0
        assert "no thread_id" in result.output


# -- plan history --


def test_plan_history_chain(db_conn_path):
    """plan history should show the full continuation chain."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    root = create_plan_request(conn, project_id=pid, prompt="root", caller="cli", backend="codex")
    set_plan_request_thread_id(conn, root["id"], "t1")
    finalize_plan_request(conn, root["id"], "text")
    child = create_plan_request(
        conn,
        project_id=pid,
        prompt="child",
        caller="cli",
        backend="codex",
        parent_id=root["id"],
    )
    finalize_plan_request(conn, child["id"], "text2")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "history", child["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["plan_id"] == child["id"]
        chain_ids = [entry["id"] for entry in payload["chain"]]
        assert root["id"] in chain_ids
        assert child["id"] in chain_ids


def test_plan_timeline_orders_rows_and_uses_elapsed_for_latest(db_conn_path):
    """plan timeline should be chronological with latest row showing elapsed time."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(conn, project_id=pid, prompt="timeline", caller="cli", backend="codex")
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status=None,
        new_status="pending",
        actor=None,
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="pending",
        new_status="running",
        actor="alice",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="plan",
        entity_id=p["id"],
        old_status="running",
        new_status="awaiting_input",
        actor=None,
        created_at="2026-01-01T00:00:10Z",
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "timeline", p["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["plan_id"] == p["id"]
        tl = payload["timeline"]
        assert len(tl) == 3
        # First transition: initial -> pending
        assert tl[0]["new_status"] == "pending"
        # Second: pending -> running, by alice
        assert tl[1]["old_status"] == "pending"
        assert tl[1]["new_status"] == "running"
        assert tl[1]["actor"] == "alice"
        # Third: running -> awaiting_input
        assert tl[2]["old_status"] == "running"
        assert tl[2]["new_status"] == "awaiting_input"


def _make_task(conn, db_path):
    """Helper to create a project + plan + task for CLI tests."""
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Test task", description="Do it")
    return plan, task


def test_task_list(db_conn_path):
    """task list should show tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list"])
        assert result.exit_code == 0
        assert task["id"] in result.output
        assert "Test task" in result.output


def test_task_list_json_preserves_full_title(db_conn_path):
    """task list JSON preserves full titles without truncation."""
    conn, db_path = db_conn_path
    plan, _ = _make_task(conn, db_path)
    long = create_task(conn, plan_id=plan["id"], ordinal=2, title="B" * 55, description="d")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list"])
        assert result.exit_code == 0
        items = json.loads(result.output)
        long_item = next(t for t in items if t["id"] == long["id"])
        assert long_item["title"] == "B" * 55


def test_task_list_json_is_parseable(db_conn_path):
    """task list --json should emit parseable JSON output and keep default filtering."""
    conn, db_path = db_conn_path
    plan, active = _make_task(conn, db_path)
    completed = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="Completed task",
        description="d",
    )
    update_task_status(conn, completed["id"], "completed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list"], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        ids = {row["id"] for row in parsed}
        assert active["id"] in ids
        assert completed["id"] not in ids

        result_all = runner.invoke(main, ["task", "list", "--all"], color=True)
        assert result_all.exit_code == 0
        assert not _has_ansi(result_all.output)
        parsed_all = json.loads(result_all.output)
        assert isinstance(parsed_all, list)
        ids_all = {row["id"] for row in parsed_all}
        assert active["id"] in ids_all
        assert completed["id"] in ids_all


def test_task_list_json_includes_project_name(db_conn_path):
    """task list --json should include project_name on each task."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) >= 1
        row = next(r for r in parsed if r["id"] == task["id"])
        assert row["project_name"] == "testproj"


def test_task_list_hides_terminal_by_default(db_conn_path):
    """task list should hide completed, cancelled, and failed tasks by default."""
    conn, db_path = db_conn_path
    plan, _ = _make_task(conn, db_path)
    tasks = {}
    for status in ("completed", "cancelled", "failed"):
        t = create_task(
            conn, plan_id=plan["id"], ordinal=0, title=f"Task {status}", description="d"
        )
        update_task_status(conn, t["id"], status)
        tasks[status] = t
    active = create_task(conn, plan_id=plan["id"], ordinal=1, title="Active task", description="d")
    update_task_status(conn, active["id"], "ready")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list"])
        assert result.exit_code == 0
        for status, t in tasks.items():
            assert t["id"] not in result.output, f"{status} task should be hidden"
        assert active["id"] in result.output

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list", "--all"])
        assert result.exit_code == 0
        for t in tasks.values():
            assert t["id"] in result.output
        assert active["id"] in result.output


def test_task_list_priority_filtering_semantics(db_conn_path):
    """task list --priority should compose correctly with status/plan/project/default hiding."""
    conn, db_path = db_conn_path
    test_project_id = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()[
        "id"
    ]
    add_project(conn, "otherproj", "/tmp/otherproj")
    other_project_id = conn.execute("SELECT id FROM projects WHERE name = 'otherproj'").fetchone()[
        "id"
    ]

    plan_a = create_plan_request(
        conn,
        project_id=test_project_id,
        prompt="plan a",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, plan_a["id"], '{"title":"a","summary":"s","tasks":[]}')
    plan_b = create_plan_request(
        conn,
        project_id=test_project_id,
        prompt="plan b",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, plan_b["id"], '{"title":"b","summary":"s","tasks":[]}')
    other_plan = create_plan_request(
        conn,
        project_id=other_project_id,
        prompt="other",
        caller="cli",
        backend="codex",
    )
    finalize_plan_request(conn, other_plan["id"], '{"title":"o","summary":"s","tasks":[]}')

    high_ready_a = create_task(
        conn,
        plan_id=plan_a["id"],
        ordinal=0,
        title="High ready A",
        description="d",
        priority="high",
    )
    update_task_status(conn, high_ready_a["id"], "ready")
    high_completed_a = create_task(
        conn,
        plan_id=plan_a["id"],
        ordinal=1,
        title="High completed A",
        description="d",
        priority="high",
    )
    update_task_status(conn, high_completed_a["id"], "completed")
    medium_ready_a = create_task(
        conn,
        plan_id=plan_a["id"],
        ordinal=2,
        title="Medium ready A",
        description="d",
        priority="medium",
    )
    update_task_status(conn, medium_ready_a["id"], "ready")
    low_ready_a = create_task(
        conn,
        plan_id=plan_a["id"],
        ordinal=3,
        title="Low ready A",
        description="d",
        priority="low",
    )
    update_task_status(conn, low_ready_a["id"], "ready")
    high_ready_b = create_task(
        conn,
        plan_id=plan_b["id"],
        ordinal=0,
        title="High ready B",
        description="d",
        priority="high",
    )
    update_task_status(conn, high_ready_b["id"], "ready")
    high_ready_other = create_task(
        conn,
        plan_id=other_plan["id"],
        ordinal=0,
        title="High ready other",
        description="d",
        priority="high",
    )
    update_task_status(conn, high_ready_other["id"], "ready")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()

        result = runner.invoke(main, ["task", "list", "--priority", "high"])
        assert result.exit_code == 0
        assert high_ready_a["id"] in result.output
        assert high_ready_b["id"] in result.output
        assert high_ready_other["id"] in result.output
        assert high_completed_a["id"] not in result.output
        assert medium_ready_a["id"] not in result.output
        assert low_ready_a["id"] not in result.output

        result = runner.invoke(main, ["task", "list", "--priority", "high", "--all"])
        assert result.exit_code == 0
        assert high_completed_a["id"] in result.output

        result = runner.invoke(
            main, ["task", "list", "--priority", "high", "--status", "completed"]
        )
        assert result.exit_code == 0
        assert high_completed_a["id"] in result.output
        assert high_ready_a["id"] not in result.output
        assert high_ready_b["id"] not in result.output

        result = runner.invoke(
            main, ["task", "list", "--project", "testproj", "--priority", "high"]
        )
        assert result.exit_code == 0
        assert high_ready_a["id"] in result.output
        assert high_ready_b["id"] in result.output
        assert high_ready_other["id"] not in result.output

        result = runner.invoke(
            main,
            ["task", "list", "--plan", plan_a["id"], "--priority", "high", "--all"],
        )
        assert result.exit_code == 0
        assert high_ready_a["id"] in result.output
        assert high_completed_a["id"] in result.output
        assert high_ready_b["id"] not in result.output
        assert high_ready_other["id"] not in result.output


def test_task_list_renders_priority(db_conn_path):
    """task list JSON includes raw priority field."""
    conn, db_path = db_conn_path
    plan, _ = _make_task(conn, db_path)
    high = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="High", description="d", priority="high"
    )
    medium_null = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=2,
        title="Medium null",
        description="d",
        priority="medium",
    )
    low = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="Low", description="d", priority="low"
    )
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "list", "--all"])
        assert result.exit_code == 0
        items = json.loads(result.output)
        by_id = {t["id"]: t for t in items}
        assert by_id[high["id"]]["priority"] == "high"
        # create_task("medium") stores None in DB
        assert by_id[medium_null["id"]]["priority"] in (None, "medium")
        assert by_id[low["id"]]["priority"] == "low"


def test_task_show_approved_task_returns_json(db_conn_path):
    """task show for approved task returns JSON with status."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "approved")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = CliRunner().invoke(main, ["task", "show", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "approved"
        assert payload["id"] == task["id"]


def test_task_show_displays_model_name(db_conn_path):
    """task show should display model when present in both human and json outputs."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.execute("UPDATE tasks SET model = 'task-view-model' WHERE id = ?", (task["id"],))
    conn.commit()
    conn.close()

    @contextmanager
    def _mock_connect():
        c = get_connection(Path(db_path))
        try:
            yield c
        finally:
            c.close()

    with patch("agm.cli.connect", _mock_connect):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "show", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["model"] == "task-view-model"


def test_task_show_displays_failure_reason(db_conn_path):
    """task show should include failure_reason in both human and json outputs."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.execute(
        "UPDATE tasks SET failure_reason = ? WHERE id = ?",
        ("dependency missing", task["id"]),
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "show", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["failure_reason"] == "dependency missing"


def test_task_show_json_is_parseable(db_conn_path):
    """task show --json should emit a parseable raw task object."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "show", task["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert parsed["id"] == task["id"]
        assert parsed["title"] == "Test task"
        assert parsed["failure_reason"] is None
        assert "Identity:" not in result.output
        assert "Timing summary:" not in result.output


def test_task_timeline_orders_rows(db_conn_path):
    """task timeline should return JSON with chronological transitions."""
    conn, db_path = db_conn_path
    _plan, task = _make_task(conn, db_path)
    record_status_change(
        conn,
        entity_type="task",
        entity_id=task["id"],
        old_status=None,
        new_status="blocked",
        actor=None,
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=task["id"],
        old_status="blocked",
        new_status="ready",
        actor=None,
        created_at="2026-01-01T00:00:03Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=task["id"],
        old_status="ready",
        new_status="running",
        actor="worker",
        created_at="2026-01-01T00:00:08Z",
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "timeline", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["task_id"] == task["id"]
        tl = payload["timeline"]
        assert len(tl) == 3
        assert tl[0]["new_status"] == "blocked"
        assert tl[1]["new_status"] == "ready"
        assert tl[2]["new_status"] == "running"
        assert tl[2]["actor"] == "worker"


def test_task_show_timing_summary_uses_history_totals(db_conn_path):
    """task show timing summary should include history-derived per-status elapsed timing."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    record_status_change(
        conn,
        entity_type="task",
        entity_id=task["id"],
        old_status=None,
        new_status="blocked",
        created_at="2026-01-01T00:00:00Z",
    )
    record_status_change(
        conn,
        entity_type="task",
        entity_id=task["id"],
        old_status="blocked",
        new_status="ready",
        created_at="2026-01-01T00:00:07Z",
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "show", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["id"] == task["id"]
        assert "status" in payload


def test_task_show_shows_effective_medium_priority_for_null_and_legacy(db_conn_path):
    """task show should show medium when DB priority is NULL or legacy 'medium'."""
    conn, db_path = db_conn_path
    plan, _ = _make_task(conn, db_path)
    medium_null = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="Medium null",
        description="d",
        priority="medium",
    )
    legacy_medium = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Legacy medium", description="d", priority="low"
    )
    conn.execute("UPDATE tasks SET priority = 'medium' WHERE id = ?", (legacy_medium["id"],))
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "show", medium_null["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        # create_task stores "medium" as the DB default; raw value may be None or "medium"
        assert payload["priority"] in (None, "medium")

        result = runner.invoke(main, ["task", "show", legacy_medium["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["priority"] == "medium"


@pytest.mark.parametrize(
    "command",
    [
        ["task", "show", "nonexistent"],
        ["task", "set-priority", "nonexistent", "high"],
        ["plan", "history", "nonexistent"],
    ],
)
def test_not_found_errors(db_conn_path, command):
    """Commands targeting nonexistent entities should error with 'not found'."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, command)
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


def test_task_set_priority_success(db_conn_path):
    """task set-priority should update DB priority and map medium to NULL."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()

        result = runner.invoke(main, ["task", "set-priority", task["id"], "high"])
        assert result.exit_code == 0

        verify = get_connection(db_path)
        row = verify.execute("SELECT priority FROM tasks WHERE id = ?", (task["id"],)).fetchone()
        assert row["priority"] == "high"
        verify.close()

        result = runner.invoke(main, ["task", "set-priority", task["id"], "medium"])
        assert result.exit_code == 0

        verify = get_connection(db_path)
        row = verify.execute("SELECT priority FROM tasks WHERE id = ?", (task["id"],)).fetchone()
        assert row["priority"] is None
        verify.close()


def test_task_set_priority_invalid_priority(db_conn_path):
    """task set-priority should validate priority choices."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "set-priority", "task123", "bogus"])
        assert result.exit_code != 0
        assert "Invalid value for '{high|low|medium}'" in result.output


def test_task_blocks(db_conn_path):
    """task blocks should show blockers for a task."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    add_task_block(conn, task_id=task["id"], external_factor="API key", reason="need it")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "blocks", task["id"]])
        assert result.exit_code == 0
        assert "API key" in result.output


def test_task_blocks_by_plan(db_conn_path):
    """task blocks --plan should show blockers across all tasks in a plan."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    add_task_block(conn, task_id=t0["id"], external_factor="API key", reason="need it")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "blocks", "--plan", plan["id"]])
        assert result.exit_code == 0
        assert "API key" in result.output
        assert t0["id"][:12] in result.output


def test_task_blocks_by_project(db_conn_path):
    """task blocks --project should show blockers across all project tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    add_task_block(conn, task_id=task["id"], external_factor="Design review", reason="pending")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "blocks", "--project", "testproj"])
        assert result.exit_code == 0
        assert "Design review" in result.output


def test_task_timeline_json_is_parseable(db_conn_path):
    """task timeline --json should emit parseable timeline entries with durations."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn, project_id=pid, prompt="timeline task plan", caller="cli", backend="codex"
    )
    task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="timed task",
        description="timeline test",
    )
    update_task_status(conn, task["id"], "ready", record_history=True)
    update_task_status(conn, task["id"], "running", record_history=True)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "timeline", task["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["task_id"] == task["id"]
        timeline = payload["timeline"]
        assert isinstance(timeline, list)
        assert timeline
        entry = timeline[0]
        assert set(entry.keys()) >= {
            "id",
            "created_at",
            "old_status",
            "new_status",
            "actor",
            "next_created_at",
            "duration_seconds",
            "duration_label",
            "duration",
        }


def test_task_blocks_json_is_parseable(db_conn_path):
    """task blocks --json should emit parseable blockers payload."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn, project_id=pid, prompt="blocker plan", caller="cli", backend="codex"
    )
    blocker = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="blocker",
        description="blocking task",
    )
    blocked = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="blocked",
        description="blocked task",
    )
    add_task_block(conn, task_id=blocked["id"], blocked_by_task_id=blocker["id"], reason="depends")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "blocks", blocked["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["scope"]["task_id"] == blocked["id"]
        assert payload["count"] == 1
        assert len(payload["blocks"]) == 1
        block = payload["blocks"][0]
        assert block["task_id"] == blocked["id"]
        assert block["blocked_by_task_id"] == blocker["id"]
        assert set(block.keys()) >= {
            "id",
            "task_id",
            "blocked_by_task_id",
            "external_factor",
            "reason",
            "resolved",
            "created_at",
            "resolved_at",
            "task_id_short",
            "blocked_by_short",
            "is_external",
            "unresolved_only_filter",
        }


def test_task_logs_json_is_parseable(db_conn_path):
    """task logs --json should emit parseable payload and required fields."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(
        conn, project_id=pid, prompt="logged task plan", caller="cli", backend="codex"
    )
    task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="logged task",
        description="logged test",
    )
    add_task_log(conn, task_id=task["id"], level="INFO", message="task created")
    add_task_log(conn, task_id=task["id"], level="ERROR", message="task failed")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "logs", task["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["task_id"] == task["id"]
        assert payload["count"] == 2
        assert payload["level"] is None
        assert len(payload["logs"]) == 2
        assert set(payload["logs"][0].keys()) >= {"id", "level", "message", "created_at"}


def test_task_blocks_requires_filter(db_conn_path):
    """task blocks with no args should error."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "blocks"])
        assert result.exit_code != 0
        assert "Provide a TASK_ID" in result.output


def test_task_unblock(db_conn_path):
    """task unblock should resolve an external blocker."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    block = add_task_block(conn, task_id=task["id"], external_factor="API key", reason="need it")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "unblock", block["id"]])
        assert result.exit_code == 0


def test_task_unblock_rejects_internal(db_conn_path):
    """task unblock should reject internal (task-to-task) blockers."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], "{}")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="B", description="b")
    block = add_task_block(conn, task_id=t2["id"], blocked_by_task_id=t1["id"])
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "unblock", block["id"]])
        assert result.exit_code != 0
        assert "Cannot manually resolve" in result.output


def test_task_claim(db_conn_path):
    """task claim should mark a ready task as running and create a worktree."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "release")
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree", return_value=("agm/test-task-abc123", "/tmp/worktree")
        ) as mock_create,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "claim", task["id"]])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(
            "/tmp/testproj",
            task["id"],
            "Test task",
            "release",
        )

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "running"
    assert found["actor"] is not None
    assert found["branch"] == "agm/test-task-abc123"
    assert found["worktree"] == "/tmp/worktree"
    verify_conn.close()


def test_task_claim_not_ready(db_conn_path):
    """task claim should reject non-ready tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "claim", task["id"]])
        assert result.exit_code != 0
        assert "not 'ready'" in result.output


def test_task_review(db_conn_path):
    """task review should transition running → review."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "review", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "review"
    verify_conn.close()


def test_task_reject(db_conn_path):
    """task reject should transition review → running."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "reject", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "running"
    verify_conn.close()


def test_task_approve(db_conn_path):
    """task approve should transition review → approved."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "approve", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "approved"
    verify_conn.close()


def test_task_complete(db_conn_path):
    """task complete should transition approved → completed."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "complete", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "completed"
    verify_conn.close()


def test_task_complete_unblocks_dependents(db_conn_path):
    """task complete should resolve internal blockers and promote dependents."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "review")
    update_task_status(conn, t0["id"], "approved")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "complete", t0["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t1["id"],)).fetchone()
    assert found["status"] == "ready"
    verify_conn.close()


def test_task_complete_not_approved(db_conn_path):
    """task complete should reject non-approved tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "complete", task["id"]])
        assert result.exit_code != 0
        assert "not 'approved'" in result.output


def test_task_fail_running(db_conn_path):
    """task fail should transition allowed statuses to failed with cleanup and logging."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="agm/fail", worktree="/tmp/fail-task")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.git_ops.remove_worktree") as mock_remove,
        patch(
            "agm.cli.resolve_blockers_for_terminal_task", return_value=([], ["downstream"])
        ) as mock_resolve,
        patch("agm.cli._emit_task_event") as mock_emit_event,
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["task", "fail", task["id"], "--reason", "  test reason  ", "--yes"],
        )
        assert result.exit_code == 0
        mock_remove.assert_called_once_with(
            "/tmp/testproj",
            "/tmp/fail-task",
            "agm/fail",
        )
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args[1] == task["id"]
        assert mock_resolve.call_args.kwargs == {"record_history": True}
        assert any(
            call.args[2] == "failed" and call.args[1] == task["id"]
            for call in mock_emit_event.call_args_list
        )
        assert any(
            call.args[2] == "cancelled" and call.args[1] == "downstream"
            for call in mock_emit_event.call_args_list
        )

    verify_conn = get_connection(db_path)
    found = verify_conn.execute(
        "SELECT status, failure_reason FROM tasks WHERE id = ?",
        (task["id"],),
    ).fetchone()
    assert found["status"] == "failed"
    assert found["failure_reason"] == "test reason"
    refs = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert refs["branch"] is None
    assert refs["worktree"] is None
    log_row = verify_conn.execute(
        "SELECT level, source, message FROM task_logs "
        "WHERE task_id = ? ORDER BY rowid DESC LIMIT 1",
        (task["id"],),
    ).fetchone()
    assert log_row["source"] == "cli"
    assert "test reason" in log_row["message"]
    verify_conn.close()


def test_task_fail_confirmation_required(db_conn_path):
    """task fail should abort on confirmation decline with no mutation."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli.resolve_blockers_for_terminal_task") as mock_resolve,
        patch("agm.cli.clear_task_git_refs") as mock_clear_refs,
        patch("agm.cli._emit_task_event") as mock_emit_event,
        patch("agm.cli._task_cancel_cleanup_worktree") as mock_cleanup,
    ):
        runner = CliRunner()
        result = runner.invoke(
            main, ["task", "fail", task["id"], "--reason", "won't run"], input="n\n"
        )
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_resolve.assert_not_called()
        mock_clear_refs.assert_not_called()
        mock_emit_event.assert_not_called()
        mock_cleanup.assert_not_called()

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "running"
    logs = verify_conn.execute(
        "SELECT COUNT(*) AS c FROM task_logs WHERE task_id = ? AND source = 'cli'",
        (task["id"],),
    ).fetchone()
    assert logs["c"] == 0
    verify_conn.close()


def test_task_fail_not_found(db_conn_path):
    """task fail should reject unknown task IDs."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "fail", "missing", "--reason", "oops", "--yes"])
        assert result.exit_code != 0
        assert "Task 'missing' not found." in result.output


def test_task_fail_invalid_source_status(db_conn_path):
    """task fail should reject source statuses outside running, review, or approved."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "completed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "fail", task["id"], "--reason", "bad", "--yes"])
        assert result.exit_code != 0
        assert "Only 'running', 'review', or 'approved' tasks can fail." in result.output


def test_task_fail_whitespace_reason(db_conn_path):
    """task fail should reject whitespace-only reasons."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "fail", task["id"], "--reason", "   \t   ", "--yes"])
        assert result.exit_code != 0
        assert "Reason cannot be blank." in result.output


def test_plan_cancel(db_conn_path):
    """plan cancel should cancel a pending plan."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main, ["plan", "cancel", "--yes", "--reason", "not needed", plan["id"]]
        )
        assert result.exit_code == 0, result.output

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert found["status"] == "cancelled"
    assert found["finished_at"] is not None
    log = verify_conn.execute(
        "SELECT message FROM plan_logs WHERE plan_id = ?", (plan["id"],)
    ).fetchone()
    assert "not needed" in log["message"]
    verify_conn.close()


def test_plan_cancel_aborts_when_declined(db_conn_path):
    """plan cancel should prompt and abort when declined."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.db.update_plan_request_status") as mock_update_status,
        patch("agm.db.add_plan_log") as mock_add_plan_log,
        patch("agm.db.force_cancel_plan") as mock_force_cancel_plan,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", plan["id"]], input="n\n")
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_update_status.assert_not_called()
        mock_add_plan_log.assert_not_called()
        mock_force_cancel_plan.assert_not_called()

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert found["status"] == "pending"
    verify_conn.close()


def test_plan_cancel_approves_with_prompt(db_conn_path):
    """plan cancel should proceed when prompt is approved."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", plan["id"]], input="y\n")
        assert result.exit_code == 0, result.output
        assert "Proceed?" in result.output

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT status FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert found["status"] == "cancelled"
    log = verify_conn.execute(
        "SELECT message FROM plan_logs WHERE plan_id = ?", (plan["id"],)
    ).fetchone()
    assert log["message"] == "Cancelled via CLI"
    verify_conn.close()


def test_plan_cancel_with_tasks(db_conn_path):
    """plan cancel should cascade-cancel all non-terminal tasks."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "running")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 1", description="d1")
    t2 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 2", description="d2")
    update_task_status(conn, t2["id"], "completed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0, result.output

    verify_conn = get_connection(db_path)
    found_plan = verify_conn.execute("SELECT * FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert found_plan["status"] == "cancelled"
    found_task = verify_conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert found_task["status"] == "cancelled"
    # Completed task should NOT be cancelled
    found_t2 = verify_conn.execute("SELECT status FROM tasks WHERE id = ?", (t2["id"],)).fetchone()
    assert found_t2["status"] == "completed"
    verify_conn.close()


def test_plan_cancel_running_task_stops_active_job_before_cleanup_and_status_update(db_conn_path):
    """plan cancel should stop running task jobs before cleanup and status update."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 1", description="d1")
    update_task_status(conn, task["id"], "ready")
    assert claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    conn.close()

    calls: list[str] = []
    mock_job = MagicMock()
    mock_job.cancel.side_effect = lambda: calls.append("cancel_job")
    mock_job.stop.side_effect = lambda: calls.append("stop_job")
    mock_queue = MagicMock()
    mock_queue.connection = object()
    remove_mock = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("remove_worktree"))
    update_status_mock = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("update_status")
    )

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue", return_value=mock_queue),
        patch("rq.job.Job.fetch", return_value=mock_job) as mock_fetch,
        patch("agm.git_ops.remove_worktree", side_effect=remove_mock),
        patch("agm.cli.update_task_status", side_effect=update_status_mock),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0, result.output

    assert calls == ["cancel_job", "stop_job", "remove_worktree", "update_status"]
    mock_fetch.assert_called_once_with(f"exec-{task['id']}", connection=mock_queue.connection)
    remove_mock.assert_called_once()
    update_status_mock.assert_called_once()
    assert update_status_mock.call_args.args[1:] == (task["id"], "cancelled")


def test_plan_cancel_running_task_job_fetch_failures_keep_cascade_flow(db_conn_path):
    """plan cancel should continue cascade if fetching a task job fails."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 1", description="d1")
    update_task_status(conn, task["id"], "ready")
    assert claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    conn.close()

    calls: list[str] = []
    mock_queue = MagicMock()
    mock_queue.connection = object()
    remove_mock = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("remove_worktree"))
    update_status_mock = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("update_status")
    )

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue", return_value=mock_queue),
        patch("rq.job.Job.fetch", side_effect=RuntimeError("missing")),
        patch("agm.git_ops.remove_worktree", side_effect=remove_mock),
        patch("agm.cli.update_task_status", side_effect=update_status_mock),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0, result.output

    assert calls == ["remove_worktree", "update_status"]


def test_plan_cancel_non_active_task_without_job_does_not_attempt_job_fetch(db_conn_path):
    """plan cancel should keep normal flow for non-active statuses with no job id."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task 1", description="d1")
    update_task_status(conn, task["id"], "ready")
    assert claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    update_task_status(conn, task["id"], "failed")
    conn.close()

    calls: list[str] = []
    mock_fetch = MagicMock()
    mock_get_queue = MagicMock(return_value=MagicMock(connection=object()))
    remove_mock = MagicMock(side_effect=lambda *_args, **_kwargs: calls.append("remove_worktree"))
    update_status_mock = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("update_status")
    )

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue", mock_get_queue),
        patch("rq.job.Job.fetch", mock_fetch),
        patch("agm.git_ops.remove_worktree", side_effect=remove_mock),
        patch("agm.cli.update_task_status", side_effect=update_status_mock),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0, result.output

    assert calls == ["remove_worktree", "update_status"]
    mock_fetch.assert_not_called()
    mock_get_queue.assert_not_called()


def test_plan_cancel_awaiting_input(db_conn_path):
    """plan cancel should cancel an awaiting_input plan."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "running")
    update_plan_request_status(conn, plan["id"], "awaiting_input")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0, result.output

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    assert found["status"] == "cancelled"
    verify_conn.close()


def test_plan_cancel_already_cancelled(db_conn_path):
    """plan cancel should reject an already-cancelled plan."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "cancelled")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code != 0
        assert "already cancelled" in result.output


def test_plan_cancel_finalized(db_conn_path):
    """plan cancel should force-cancel a finalized plan."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0


def test_plan_cancel_not_found(db_conn_path):
    """plan cancel should error on unknown plan ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_plan_cancel_failed(db_conn_path):
    """plan cancel should force-cancel a failed plan."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    update_plan_request_status(conn, plan["id"], "running")
    update_plan_request_status(conn, plan["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "cancel", "--yes", plan["id"]])
        assert result.exit_code == 0


def test_task_cancel_cascade_cancels_dependents(db_conn_path):
    """task cancel should cascade-cancel downstream dependent tasks."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", t0["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    blocker = verify_conn.execute(
        "SELECT resolved FROM task_blocks WHERE task_id = ? AND blocked_by_task_id = ?",
        (t1["id"], t0["id"]),
    ).fetchone()
    assert blocker is not None and blocker["resolved"] == 1
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t1["id"],)).fetchone()
    assert found["status"] == "cancelled"
    verify_conn.close()


def test_task_cancel_cascades_even_with_remaining_blockers(db_conn_path):
    """task cancel should cascade-cancel dependents even when another blocker remains.

    The dependency chain from the cancelled task is dead — the downstream task
    can never complete its full dependency set, so it gets cascade-cancelled.
    """
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    remaining = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Remaining blocker", description="d"
    )
    blocked = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Blocked task", description="d"
    )
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, remaining["id"], "blocked")
    update_task_status(conn, blocked["id"], "blocked")
    add_task_block(conn, task_id=blocked["id"], blocked_by_task_id=t0["id"])
    add_task_block(conn, task_id=blocked["id"], blocked_by_task_id=remaining["id"])
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", t0["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    # Blocker row from t0 is resolved (cleaned up by cascade)
    blocked_row = verify_conn.execute(
        "SELECT resolved FROM task_blocks WHERE task_id = ? AND blocked_by_task_id = ?",
        (blocked["id"], t0["id"]),
    ).fetchone()
    assert blocked_row is not None and blocked_row["resolved"] == 1
    # blocked task is cascade-cancelled
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (blocked["id"],)).fetchone()
    assert found["status"] == "cancelled"
    verify_conn.close()


def test_task_cancel_failed_task(db_conn_path):
    """task cancel should leave failed tasks terminal and keep downstream blocked."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Downstream", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "failed")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", t0["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t0["id"],)).fetchone()
    assert found["status"] == "failed"
    downstream = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t1["id"],)).fetchone()
    assert downstream["status"] == "blocked"
    verify_conn.close()


def test_task_cancel_completed_rejected(db_conn_path):
    """task cancel should reject already-completed tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    update_task_status(conn, task["id"], "completed")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", task["id"]])
        assert result.exit_code != 0
        assert "already 'completed'" in result.output


def test_task_cancel(db_conn_path):
    """task cancel should cancel a ready task."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", task["id"], "--reason", "stale"])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "cancelled"
    # Check log was written
    log = verify_conn.execute(
        "SELECT message FROM task_logs WHERE task_id = ?", (task["id"],)
    ).fetchone()
    assert "stale" in log["message"]
    verify_conn.close()


def test_task_cancel_running_with_worktree(db_conn_path):
    """task cancel should clean up worktree for a running task."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.git_ops.remove_worktree") as mock_remove,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", task["id"]])
        assert result.exit_code == 0
        mock_remove.assert_called_once()
        call_args = mock_remove.call_args
        assert call_args[0][1] == "/tmp/wt"
        assert call_args[0][2] == "agm/test"


def test_task_cancel_running_stops_rq_job(db_conn_path):
    """task cancel on a running task should cancel and stop its RQ job."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    conn.close()

    fake_job = MagicMock()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("rq.job.Job.fetch", return_value=fake_job),
    ):
        mock_get_queue.return_value.connection = MagicMock()
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", task["id"]])
        assert result.exit_code == 0
        fake_job.cancel.assert_called_once()
        fake_job.stop.assert_called_once()


def test_task_cancel_ready_skips_job_cancellation(db_conn_path):
    """task cancel on a non-running task should not attempt RQ job cancellation."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue") as mock_get_queue,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--yes", task["id"]])
        assert result.exit_code == 0
        mock_get_queue.assert_not_called()


def test_task_cancel_already_completed(db_conn_path):
    """task cancel should reject completed tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "completed")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", task["id"]])
        assert result.exit_code != 0
        assert "already 'completed'" in result.output


def test_task_cancel_already_cancelled(db_conn_path):
    """task cancel should reject already-cancelled tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "cancelled")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", task["id"]])
        assert result.exit_code != 0
        assert "already 'cancelled'" in result.output


def test_task_cancel_not_found(db_conn_path):
    """task cancel should error on unknown task ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_cancel_bulk_by_status(db_conn_path):
    """task cancel --status failed should skip transitions from terminal task states."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="T1", description="d")
    update_task_status(conn, t0["id"], "failed")
    update_task_status(conn, t1["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--status", "failed", "--yes"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["cancelled"] == 2
        assert payload["status"] == "failed"

    verify = get_connection(db_path)
    for tid in (t0["id"], t1["id"]):
        row = verify.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "failed"
    verify.close()


def test_task_cancel_bulk_confirms(db_conn_path):
    """task cancel --status should prompt for confirmation and return JSON."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--status", "failed"], input="y\n")
        assert result.exit_code == 0
        # Output contains confirmation prompt + JSON; extract last line
        json_line = [ln for ln in result.output.strip().splitlines() if ln.startswith("{")][-1]
        payload = json.loads(json_line)
        assert payload["cancelled"] == 1


def test_task_cancel_bulk_abort(db_conn_path):
    """task cancel --status should abort when user declines."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--status", "failed"], input="n\n")
        assert result.exit_code == 0

    verify = get_connection(db_path)
    row = verify.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert row["status"] == "failed"
    verify.close()


def test_task_cancel_bulk_json(db_conn_path):
    """task cancel --status --yes returns structured JSON output."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--status", "failed", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "failed"
        assert data["cancelled"] == 1


def test_task_cancel_bulk_no_tasks(db_conn_path):
    """task cancel --status with no matching tasks returns cancelled=0."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cancel", "--status", "failed", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "failed"
        assert data["cancelled"] == 0


def test_task_cancel_mutually_exclusive():
    """task cancel with both TASK_ID and --status should error."""
    runner = CliRunner()
    result = runner.invoke(main, ["task", "cancel", "some-id", "--status", "failed"])
    assert result.exit_code != 0
    assert "Cannot combine" in result.output


def test_task_cancel_requires_id_or_status():
    """task cancel with neither TASK_ID nor --status should error."""
    runner = CliRunner()
    result = runner.invoke(main, ["task", "cancel"])
    assert result.exit_code != 0
    assert "Specify TASK_ID or --status" in result.output


def test_task_retry(db_conn_path):
    """task retry should reset a failed task to pending."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "failed")
    conn.execute(
        "UPDATE tasks SET pid = ?, thread_id = ? WHERE id = ?",
        (9999, "thread-old", task["id"]),
    )
    conn.commit()
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "blocked"
    assert found["pid"] is None
    assert found["thread_id"] is None
    verify_conn.close()


def test_task_retry_cleans_worktree(db_conn_path):
    """task retry should remove the old worktree and branch."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="agm/test", worktree="/tmp/wt")
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.git_ops.remove_worktree") as mock_remove,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", task["id"]])
        assert result.exit_code == 0
        mock_remove.assert_called_once()
        call_args = mock_remove.call_args
        assert call_args[0][1] == "/tmp/wt"
        assert call_args[0][2] == "agm/test"


def test_task_retry_not_failed(db_conn_path):
    """task retry should reject non-failed tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", task["id"]])
        assert result.exit_code != 0
        assert "not 'failed'" in result.output


def test_task_retry_merge_failed_approved_no_signal(db_conn_path):
    """task retry should reject approved tasks without merge-failure signal."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "approved")
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="merge failed while applying hunk",
        source="executor",
    )
    add_task_log(
        conn,
        task_id=task["id"],
        level="INFO",
        message="retry requested after cleanup",
        source="executor",
    )
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", task["id"]])
        assert result.exit_code != 0
        assert (
            f"Task '{task['id']}' is 'approved' but has no merge-failure signal." in result.output
        )


def test_task_retry_merge_failed_approved_to_ready(db_conn_path):
    """task retry should move approved merge-failed tasks to ready."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "approved")
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="Merge failed and needs review",
        source="executor",
    )
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "ready"
    verify_conn.close()


def test_task_retry_not_found(db_conn_path):
    """task retry should error on unknown task ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_cleanup_no_candidates(db_conn_path):
    """task cleanup should succeed with a no-op message when nothing needs cleanup."""
    conn, db_path = db_conn_path
    plan, terminal_no_refs = _make_task(conn, db_path)
    running_with_refs = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Running", description="d"
    )
    terminal_branch_only = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Branch only", description="d"
    )
    terminal_worktree_only = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="Worktree only", description="d"
    )

    update_task_status(conn, terminal_no_refs["id"], "completed")
    update_task_status(conn, running_with_refs["id"], "running")
    update_task_status(conn, terminal_branch_only["id"], "cancelled")
    update_task_status(conn, terminal_worktree_only["id"], "failed")

    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/running", "/tmp/wt-running", running_with_refs["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/branch-only", None, terminal_branch_only["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        (None, "/tmp/wt-only", terminal_worktree_only["id"]),
    )
    conn.commit()
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli._remove_worktree") as mock_cleanup,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cleanup", "-p", "testproj"])
        assert result.exit_code == 0
        mock_cleanup.assert_not_called()


def test_task_cleanup_project_not_found(db_conn_path):
    """task cleanup should error if the project does not exist."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cleanup", "-p", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_cleanup_cleans_eligible_tasks_and_clears_refs(db_conn_path):
    """task cleanup should only clean terminal tasks with both git refs."""
    conn, db_path = db_conn_path
    plan, cleaned_completed = _make_task(conn, db_path)
    cleaned_cancelled = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Cancelled", description="d"
    )
    skip_running = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Running", description="d"
    )
    skip_completed_no_refs = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="No refs", description="d"
    )
    skip_failed_branch_only = create_task(
        conn, plan_id=plan["id"], ordinal=4, title="Branch only", description="d"
    )
    skip_failed_worktree_only = create_task(
        conn, plan_id=plan["id"], ordinal=5, title="Worktree only", description="d"
    )

    update_task_status(conn, cleaned_completed["id"], "completed")
    update_task_status(conn, cleaned_cancelled["id"], "cancelled")
    update_task_status(conn, skip_running["id"], "running")
    update_task_status(conn, skip_completed_no_refs["id"], "completed")
    update_task_status(conn, skip_failed_branch_only["id"], "failed")
    update_task_status(conn, skip_failed_worktree_only["id"], "failed")

    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/completed", "/tmp/wt-completed", cleaned_completed["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/cancelled", "/tmp/wt-cancelled", cleaned_cancelled["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/running", "/tmp/wt-running", skip_running["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/branch-only", None, skip_failed_branch_only["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        (None, "/tmp/wt-only", skip_failed_worktree_only["id"]),
    )
    conn.commit()
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.cli._remove_worktree") as mock_cleanup,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cleanup", "--yes", "--project", "testproj"])
        assert result.exit_code == 0
        assert mock_cleanup.call_count == 2
        cleaned_calls = {(c.args[1], c.args[2]) for c in mock_cleanup.call_args_list}
        assert cleaned_calls == {
            ("/tmp/wt-completed", "agm/completed"),
            ("/tmp/wt-cancelled", "agm/cancelled"),
        }
        assert all(c.args[0] == "/tmp/testproj" for c in mock_cleanup.call_args_list)

    verify_conn = get_connection(db_path)
    cleaned_completed_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (cleaned_completed["id"],)
    ).fetchone()
    cleaned_cancelled_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (cleaned_cancelled["id"],)
    ).fetchone()
    running_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (skip_running["id"],)
    ).fetchone()
    branch_only_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (skip_failed_branch_only["id"],)
    ).fetchone()
    worktree_only_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (skip_failed_worktree_only["id"],)
    ).fetchone()
    assert cleaned_completed_row["branch"] is None and cleaned_completed_row["worktree"] is None
    assert cleaned_cancelled_row["branch"] is None and cleaned_cancelled_row["worktree"] is None
    assert running_row["branch"] == "agm/running"
    assert running_row["worktree"] == "/tmp/wt-running"
    assert branch_only_row["branch"] == "agm/branch-only"
    assert branch_only_row["worktree"] is None
    assert worktree_only_row["branch"] is None
    assert worktree_only_row["worktree"] == "/tmp/wt-only"
    verify_conn.close()


def test_task_cleanup_processes_all_candidates_and_reports_summary(db_conn_path):
    """task cleanup should continue after per-task failures and report counts."""
    conn, db_path = db_conn_path
    plan, fail_first = _make_task(conn, db_path)
    keep_ok = create_task(conn, plan_id=plan["id"], ordinal=1, title="OK", description="d")
    skip_running = create_task(
        conn, plan_id=plan["id"], ordinal=2, title="Running", description="d"
    )
    skip_branch_only = create_task(
        conn, plan_id=plan["id"], ordinal=3, title="No worktree", description="d"
    )

    update_task_status(conn, fail_first["id"], "failed")
    update_task_status(conn, keep_ok["id"], "completed")
    update_task_status(conn, skip_running["id"], "running")
    update_task_status(conn, skip_branch_only["id"], "cancelled")

    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/fail", "/tmp/wt-fail", fail_first["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/ok", "/tmp/wt-ok", keep_ok["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/skip", "/tmp/wt-skip", skip_running["id"]),
    )
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/no-worktree", None, skip_branch_only["id"]),
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00Z", fail_first["id"]),
    )
    conn.execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:01Z", keep_ok["id"]),
    )
    conn.commit()
    conn.close()

    def _mock_cleanup(_project_dir, _worktree_path, branch):
        if branch == "agm/fail":
            raise RuntimeError("boom")

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.cli._remove_worktree", side_effect=_mock_cleanup) as mock_cleanup,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "cleanup", "--yes", "--project", "testproj"])
        assert result.exit_code == 0
        assert mock_cleanup.call_count == 2
        assert mock_cleanup.call_args_list[0].args[2] == "agm/fail"
        assert mock_cleanup.call_args_list[1].args[2] == "agm/ok"

    verify_conn = get_connection(db_path)
    fail_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (fail_first["id"],)
    ).fetchone()
    ok_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (keep_ok["id"],)
    ).fetchone()
    skip_running_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (skip_running["id"],)
    ).fetchone()
    skip_branch_only_row = verify_conn.execute(
        "SELECT branch, worktree FROM tasks WHERE id = ?", (skip_branch_only["id"],)
    ).fetchone()
    assert fail_row["branch"] is None and fail_row["worktree"] is None
    assert ok_row["branch"] is None and ok_row["worktree"] is None
    assert skip_running_row["branch"] == "agm/skip"
    assert skip_running_row["worktree"] == "/tmp/wt-skip"
    assert skip_branch_only_row["branch"] == "agm/no-worktree"
    assert skip_branch_only_row["worktree"] is None
    verify_conn.close()


def test_purge_not_found_json_payload_and_exit_code(db_conn_path):
    """purge --json for unknown project should return error payload and exit 1."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["purge", "-p", "nonexistent"])
        assert result.exit_code == 1
        output_lines = [line for line in result.output.splitlines() if line.strip()]
        payload = json.loads(output_lines[0])
        assert payload["ok"] is False
        assert "not found" in payload["error"]
        assert "nonexistent" in payload["error"]


def test_purge_confirmation_includes_blocked_by_task_rows_and_aborts(db_conn_path):
    """purge should show blocker rows in preview even when only blocked_by_task_id matches."""
    conn, db_path = db_conn_path
    plan, blocker = _make_task(conn, db_path)
    blocked = create_task(conn, plan_id=plan["id"], ordinal=1, title="Blocked", description="d")
    add_task_block(conn, task_id=blocked["id"], blocked_by_task_id=blocker["id"], reason="blocked")
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["purge", "-p", "testproj"], input="n\n")
        assert result.exit_code == 0

    verify = get_connection(db_path)
    assert (
        verify.execute("SELECT COUNT(*) FROM plans WHERE id = ?", (plan["id"],)).fetchone()[0] == 1
    )
    assert verify.execute("SELECT COUNT(*) FROM task_blocks").fetchone()[0] == 1
    verify.close()


def test_purge_json_summary_and_blocking_counts(db_conn_path):
    """purge --json should emit required summary fields and blocked row counts."""
    conn, db_path = db_conn_path
    plan, blocker = _make_task(conn, db_path)
    blocked = create_task(conn, plan_id=plan["id"], ordinal=1, title="Blocked", description="d")
    add_task_block(conn, task_id=blocked["id"], blocked_by_task_id=blocker["id"], reason="blocked")
    conn.commit()
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.remove_jobs_for_entities", return_value=7),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["purge", "-p", "testproj", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert set(data.keys()) == {"purged", "total", "redis_jobs_cleaned"}
        assert data["redis_jobs_cleaned"] == 7
        assert data["purged"]["task_blocks"] == 1
        assert data["total"] == sum(data["purged"].values())


def test_purge_redis_cleanup_best_effort(db_conn_path):
    """purge --json should keep the command successful if Redis cleanup fails."""
    conn, db_path = db_conn_path
    plan, task = _make_task(conn, db_path)
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.queue.remove_jobs_for_entities",
            side_effect=RuntimeError("redis unavailable"),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["purge", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["redis_jobs_cleaned"] == 0
        assert data["purged"]["plans"] == 1


def test_task_refresh(db_conn_path):
    """task refresh should enqueue a refresh job."""
    conn, db_path = db_conn_path
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "refresh-testproj"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["task", "refresh", "-p", "testproj"])
        assert result.exit_code == 0
        assert mock_q.enqueue.call_args.args[3] == "codex"


def test_task_refresh_uses_default_when_flag_missing(db_conn_path):
    """task refresh uses project default backend."""
    conn, db_path = db_conn_path
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "refresh-testproj"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["task", "refresh", "-p", "testproj"])
        assert result.exit_code == 0
        assert mock_q.enqueue.call_args.args[3] == "codex"


def test_task_refresh_project_not_found(db_conn_path):
    """task refresh should error if project doesn't exist."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "refresh", "-p", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_refresh_prefers_explicit_backend_over_project_default(db_conn_path):
    """task refresh should prefer explicit --backend."""
    conn, db_path = db_conn_path
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "refresh-testproj-explicit"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["task", "refresh", "-p", "testproj", "--backend", "codex"],
        )
        assert result.exit_code == 0
        assert mock_q.enqueue.call_args.args[3] == "codex"


def test_task_refresh_invalid_backend_is_rejected(db_conn_path):
    """task refresh rejects invalid --backend values."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "refresh", "-p", "testproj", "--backend", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output


# -- task run --


def test_task_run_auto_claims_ready(db_conn_path):
    """task run on a ready task should auto-claim and enqueue."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "release")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = f"exec-{task['id']}"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/test-task-abc", "/tmp/worktree"),
        ) as mock_create,
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", task["id"]])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(
            "/tmp/testproj",
            task["id"],
            "Test task",
            "release",
        )

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "running"
    assert found["worktree"] == "/tmp/worktree"
    verify_conn.close()


def test_task_run_already_running(db_conn_path):
    """task run on a running task with worktree should just enqueue."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.execute(
        "UPDATE tasks SET worktree = ? WHERE id = ?",
        ("/tmp/worktree", task["id"]),
    )
    conn.commit()
    conn.close()

    mock_job = MagicMock()
    mock_job.id = f"exec-{task['id']}"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", task["id"]])
        assert result.exit_code == 0


def test_task_run_running_no_worktree(db_conn_path):
    """task run on a running task without worktree should error."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", task["id"]])
        assert result.exit_code != 0
        assert "no worktree" in result.output


def test_task_run_wrong_status(db_conn_path):
    """task run should reject tasks in other statuses."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    # Task is pending (not ready or running)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", task["id"]])
        assert result.exit_code != 0
        assert "Only 'ready' or 'running'" in result.output


def test_task_run_not_found(db_conn_path):
    """task run should error on unknown task ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_run_enqueue_failure_marks_failed(db_conn_path):
    """task run should mark task failed if enqueue fails."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/test-task-abc", "/tmp/worktree"),
        ),
        patch(
            "agm.queue.get_queue",
            side_effect=ConnectionError("Redis unavailable"),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "run", task["id"]])
        assert result.exit_code != 0
        assert "Failed to enqueue" in result.output

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "failed"
    assert "Enqueue failed" in (found["failure_reason"] or "")
    # Verify enqueue failure is logged
    logs = verify_conn.execute(
        "SELECT * FROM task_logs WHERE task_id = ? AND level = 'ERROR'",
        (task["id"],),
    ).fetchall()
    assert len(logs) >= 1
    assert "Enqueue failed" in logs[0]["message"]
    verify_conn.close()


# -- task review dual-mode --


def test_task_review_running_transitions(db_conn_path):
    """task review on a running task should transition to review."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "review", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "review"
    verify_conn.close()


def test_task_review_launches_reviewer(db_conn_path):
    """task review on a task in review should enqueue reviewer job."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = f"review-{task['id']}"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(main, ["task", "review", task["id"]])
        assert result.exit_code == 0


def test_task_review_wrong_status(db_conn_path):
    """task review should error for tasks in blocked/approved/etc."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    # Task is blocked (not running or review)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "review", task["id"]])
        assert result.exit_code != 0
        assert "blocked" in result.output


def test_task_review_not_found(db_conn_path):
    """task review should error on unknown task ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "review", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


# -- task merge --


def test_task_merge_success(db_conn_path):
    """task merge should merge to base branch and resolve blockers."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "release")
    t1 = create_task(conn, plan_id=plan["id"], ordinal=1, title="Task 1", description="d")
    update_task_status(conn, t0["id"], "ready")
    update_task_status(conn, t0["id"], "running")
    update_task_status(conn, t0["id"], "review")
    update_task_status(conn, t0["id"], "approved")
    update_task_status(conn, t1["id"], "blocked")
    add_task_block(conn, task_id=t1["id"], blocked_by_task_id=t0["id"])
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/test-task-abc123", "/tmp/worktree", t0["id"]),
    )
    conn.commit()
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.cli._merge_to_main") as mock_merge,
        patch("agm.cli._remove_worktree") as mock_cleanup,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "merge", t0["id"]])
        assert result.exit_code == 0

        mock_merge.assert_called_once_with(
            "/tmp/testproj",
            "agm/test-task-abc123",
            t0["id"],
            "Test task",
            base_branch="release",
            worktree_path="/tmp/worktree",
        )
    mock_cleanup.assert_called_once_with(
        "/tmp/testproj",
        "/tmp/worktree",
        "agm/test-task-abc123",
    )

    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t0["id"],)).fetchone()
    assert found["status"] == "completed"
    dep = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (t1["id"],)).fetchone()
    assert dep["status"] == "ready"
    verify_conn.close()


def test_task_merge_wrong_status(db_conn_path):
    """task merge should reject non-approved tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "merge", task["id"]])
        assert result.exit_code != 0
        assert "not 'approved'" in result.output


def test_task_merge_not_found(db_conn_path):
    """task merge should error on unknown task ID."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "merge", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_merge_no_worktree(db_conn_path):
    """task merge should reject approved tasks without worktree/branch."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "merge", task["id"]])
        assert result.exit_code != 0
        assert "no worktree" in result.output


def test_task_merge_conflict(db_conn_path):
    """task merge should error on conflict, task stays approved."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    conn.execute(
        "UPDATE tasks SET branch = ?, worktree = ? WHERE id = ?",
        ("agm/conflict-branch", "/tmp/worktree", task["id"]),
    )
    conn.commit()
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._merge_to_main",
            side_effect=click.ClickException("Merge conflict:\nCONFLICT (content)"),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "merge", task["id"]])
        assert result.exit_code != 0
        assert "Merge conflict" in result.output

    # Task should still be approved
    verify_conn = get_connection(db_path)
    found = verify_conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "approved"
    verify_conn.close()


# -- watch helpers --


def testwatch_short_id():
    assert watch_short_id("abc12345deadbeef") == "abc12345deadbeef"


def test_watch_short_id_short_input():
    assert watch_short_id("abc") == "abc"


def test_watch_truncate_short():
    assert watch_truncate("hello", 10) == "hello"


def test_watch_truncate_long():
    result = watch_truncate("a" * 50, 10)
    assert len(result) == 10
    assert result.endswith("\u2026")


def test_watch_truncate_exact():
    assert watch_truncate("a" * 10, 10) == "a" * 10


# -- task watch --


def test_task_watch_requires_filter():
    """task watch without TASK_ID/--plan/--project should error."""
    runner = CliRunner()
    result = runner.invoke(main, ["task", "watch"])
    assert result.exit_code != 0
    assert "Provide TASK_ID, --plan, or --project" in result.output


def test_task_watch_rejects_ambiguous_scope():
    """task watch should reject multiple scope selectors."""
    runner = CliRunner()
    result = runner.invoke(main, ["task", "watch", "task123", "--plan", "plan123"])
    assert result.exit_code != 0
    assert "Provide exactly one scope" in result.output


def test_task_watch_rejects_all_ambiguous_scope_paths():
    """task watch should reject every multi-scope selector combination."""
    runner = CliRunner()
    commands = [
        ["task", "watch", "task123", "--project", "testproj"],
        ["task", "watch", "--plan", "plan123", "--project", "testproj"],
    ]
    for command in commands:
        result = runner.invoke(main, command)
        assert result.exit_code != 0
        assert "Provide exactly one scope: TASK_ID, --plan, or --project." in result.output


def test_task_watch_shows_tasks(db_conn_path):
    """task watch should return JSON snapshot for terminal tasks."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="Second task",
        description="d",
        priority="high",
    )
    update_task_status(conn, t0["id"], "completed")
    update_task_status(conn, t1["id"], "failed")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "watch", "--plan", plan["id"], "--all"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "tasks" in payload
        assert "phase" in payload
        assert "phase_since" in payload
        assert "blocking_reason" in payload
        assert payload["terminal_state"] is not None


def test_task_watch_hides_completed(db_conn_path):
    """task watch without --all should exit cleanly on all-terminal tasks."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    update_task_status(conn, t0["id"], "completed")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "watch", "--plan", plan["id"]])
        assert result.exit_code == 0


def test_task_watch_single_task_shows_details_and_events(db_conn_path):
    """task watch TASK_ID should show task details and events."""
    conn, db_path = db_conn_path
    plan, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "completed")
    add_task_log(
        conn,
        task_id=task["id"],
        level="INFO",
        message="first line\nsecond line",
    )
    conn.commit()
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "watch", task["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["terminal_state"] is not None
        assert any(t["status"] == "completed" for t in payload["tasks"])


def test_task_watch_json_includes_rejection_count(db_conn_path):
    """task watch --json should include stable per-task rejection_count."""
    conn, db_path = db_conn_path
    plan, t0 = _make_task(conn, db_path)
    t1 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="Second task",
        description="d",
    )
    update_task_status(conn, t0["id"], "ready", record_history=True)
    update_task_status(conn, t0["id"], "review", record_history=True)
    update_task_status(conn, t0["id"], "rejected", record_history=True)
    update_task_status(conn, t1["id"], "ready", record_history=True)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "watch", "--plan", plan["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["schema"] == "task_watch_snapshot_v1"
        assert "session_id" in payload["scope"]
        tasks = {row["id"]: row["rejection_count"] for row in payload["tasks"]}
        assert tasks[t0["id"]] == 1
        assert tasks[t1["id"]] == 0
        assert isinstance(tasks[t0["id"]], int)


# -- plan watch --


def test_plan_watch_not_found(db_conn_path):
    """plan watch for nonexistent plan should error."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "watch", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_plan_watch_json_is_parseable_and_schema(db_conn_path):
    """plan watch --json should emit deterministic snapshot output."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn,
        project_id=pid,
        prompt="json plan watch",
        caller="cli",
        backend="codex",
    )
    task = create_task(
        conn,
        plan_id=p["id"],
        ordinal=0,
        title="watch task",
        description="done",
    )
    update_task_status(conn, task["id"], "completed")
    update_plan_request_status(conn, p["id"], "finalized")
    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        ("plog001", p["id"], "INFO", "plan finished", "2026-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "watch", p["id"]], color=True)
        assert result.exit_code == 0
        assert not _has_ansi(result.output)
        payload = json.loads(result.output)
        assert payload["schema"] == "plan_watch_snapshot_v1"
        assert payload["plan"]["status"] == "finalized"
        scope = payload["scope"]
        assert scope["type"] == "plan"
        assert scope["plan_id"] == p["id"]
        assert "session_id" in scope
        assert scope["title"] == "json plan watch"
        assert payload["counts"]["tasks_total"] == 1
        assert payload["counts"]["tasks_active"] == 0
        assert payload["counts"]["status_summary"]["completed"] == 1
        assert "phase" in payload
        assert "phase_since" in payload
        assert "blocking_reason" in payload
        assert isinstance(payload["recent_events"], list)
        assert len(payload["recent_events"]) >= 1
        assert set(payload["recent_events"][0].keys()) >= {
            "timestamp",
            "source",
            "task_id",
            "message",
            "message_truncated",
            "line",
        }
        assert payload["terminal_state"]["reached"] is True


def test_plan_watch_shows_plan_and_exits(db_conn_path):
    """plan watch should show plan info and exit on terminal state."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    p = create_plan_request(
        conn, project_id=pid, prompt="watch test plan", caller="cli", backend="codex"
    )
    update_plan_request_status(conn, p["id"], "failed")
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "watch", p["id"]])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["scope"]["title"] == "watch test plan"
        assert payload["plan"]["backend"] == "codex"
        assert payload["plan"]["status"] == "failed"
        assert payload["terminal_state"] is not None


def test_plan_watch_renders_combined_recent_events(db_conn_path):
    """plan watch should show events from both plan and task logs."""
    conn, db_path = db_conn_path
    plan, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "completed")

    conn.execute(
        "INSERT INTO plan_logs (id, plan_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
        ("plog001", plan["id"], "INFO", "plan-msg", "2026-01-01T12:34:56Z"),
    )
    conn.commit()
    conn.close()

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(db_path),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "watch", plan["id"]])
        assert result.exit_code == 0
        assert "plan-msg" in result.output


def test_do_creates_plan_and_task_and_enqueues(db_conn_path):
    """agm do should create a synthetic plan + task, claim, and enqueue."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "release")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "exec-quick123"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/fix-the-bug-abc", "/tmp/wt-quick"),
        ) as mock_create,
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["do", "Fix the bug in auth", "-p", "testproj"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "plan_id" in data
        assert "task_id" in data
        assert "session_id" in data
        assert data["execution_job_id"] == "exec-quick123"
        assert data["title"] == "Fix the bug in auth"
        assert data["status"] == "running"
        assert data["expected_terminal_status"] == "completed"
        assert data["flags"] == []

    # Verify DB state
    verify_conn = get_connection(db_path)
    plans = verify_conn.execute("SELECT * FROM plans").fetchall()
    assert len(plans) == 1
    assert plans[0]["status"] == "finalized"
    assert plans[0]["task_creation_status"] == "completed"
    assert plans[0]["thread_id"] is None

    tasks = verify_conn.execute("SELECT * FROM tasks").fetchall()
    task_id = tasks[0]["id"]
    assert len(tasks) == 1
    assert tasks[0]["status"] == "running"  # claimed
    assert tasks[0]["branch"] == "agm/fix-the-bug-abc"
    assert tasks[0]["worktree"] == "/tmp/wt-quick"
    assert tasks[0]["title"] == "Fix the bug in auth"[:60]
    assert tasks[0]["skip_review"] == 0
    assert tasks[0]["skip_merge"] == 0
    mock_create.assert_called_once_with(
        "/tmp/testproj",
        task_id,
        "Fix the bug in auth",
        "release",
    )
    verify_conn.close()


def test_do_json_output(db_conn_path):
    """agm do --json should emit parseable structured output."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "release")
    conn.close()

    special_title = 'Fix "auth" bug with \\\\ paths and emoji 🚀'
    mock_job = MagicMock()
    mock_job.id = "exec-quick-json"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/fix-json-abc", "/tmp/wt-json"),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["do", "Fix it quickly", "-p", "testproj", "--title", special_title],
        )
        assert result.exit_code == 0
        output = result.output.strip()
        payload = json.loads(output)
        assert set(payload.keys()) == {
            "plan_id",
            "task_id",
            "session_id",
            "execution_job_id",
            "title",
            "status",
            "expected_terminal_status",
            "flags",
        }
        assert payload["execution_job_id"] == "exec-quick-json"
        assert payload["title"] == special_title
        assert payload["status"] == "running"
        assert payload["expected_terminal_status"] == "completed"
        assert payload["flags"] == []
        for marker in ("plan:", "task:", "job:", "flags:", "Monitor:"):
            assert marker not in output

    verify_conn = get_connection(db_path)
    task = verify_conn.execute("SELECT * FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()
    plan = verify_conn.execute(
        "SELECT * FROM plans WHERE id = ?",
        (task["plan_id"],),
    ).fetchone()
    assert payload["task_id"] == task["id"]
    assert payload["plan_id"] == plan["id"]
    assert payload["title"] == task["title"]
    verify_conn.close()


def test_do_with_flags(db_conn_path):
    """agm do --no-review --no-merge sets flags on the task."""
    conn, db_path = db_conn_path
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "exec-quick-flags"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/quick-abc", "/tmp/wt"),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["do", "Quick fix", "-p", "testproj", "--no-review", "--no-merge"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "task_id" in payload
        assert "plan_id" in payload
        assert payload["execution_job_id"] == "exec-quick-flags"
        assert payload["expected_terminal_status"] == "approved"
        assert payload["flags"] == ["skip-review", "skip-merge"]

    verify_conn = get_connection(db_path)
    task = verify_conn.execute("SELECT * FROM tasks").fetchone()
    assert task["skip_review"] == 1
    assert task["skip_merge"] == 1
    verify_conn.close()


def test_session_messages_supports_sender_recipient_and_offset(db_conn_path):
    """session messages should expose sender/recipient filters and offset pagination."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    session = create_session(conn, project_id=pid, trigger="do")
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="planner:a1b2c3d4",
        recipient="executor:e1f2a3b4",
        content="first",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="planner:ff00aa11",
        recipient="executor:99887766",
        content="second",
    )
    add_channel_message(
        conn,
        session_id=session["id"],
        kind="context",
        sender="planner:11223344",
        recipient="reviewer:bbccdd11",
        content="other",
    )
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "session",
                "messages",
                session["id"],
                "--sender",
                "planner",
                "--recipient",
                "executor",
                "--limit",
                "1",
                "--offset",
                "1",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["sender"] == "planner"
        assert payload["recipient"] == "executor"
        assert payload["limit"] == 1
        assert payload["offset"] == 1
        assert payload["count"] == 1
        assert payload["messages"][0]["content"] == "second"


def test_session_post_creates_message(db_conn_path):
    """session post should append a channel entry."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    session = create_session(conn, project_id=pid, trigger="do")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "session",
                "post",
                session["id"],
                "Route through adapter",
                "--kind",
                "steer",
                "--sender",
                "operator",
                "--recipient",
                "executor:deadbeef",
                "--metadata",
                '{"source":"cli-test"}',
            ],
        )
        assert result.exit_code == 0
        verify = runner.invoke(main, ["session", "messages", session["id"]])
        assert verify.exit_code == 0
        payload = json.loads(verify.output)
        assert payload["count"] == 1
        assert payload["messages"][0]["kind"] == "steer"
        assert payload["messages"][0]["sender"] == "operator:cli"


def test_task_steer_posts_message_and_attempts_live_turn(db_conn_path):
    """task steer should write a steer message and call live steer when task is active."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    session = create_session(conn, project_id=pid, trigger="do")
    plan = create_plan_request(conn, project_id=pid, prompt="p", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"x","summary":"s","tasks":[]}')
    conn.execute("UPDATE plans SET session_id = ? WHERE id = ?", (session["id"], plan["id"]))
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    update_task_status(conn, task["id"], "running")
    conn.execute(
        "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
        ("thread-live-1", "turn-live-1", task["id"]),
    )
    conn.commit()
    conn.close()

    captured: dict[str, str] = {}

    async def _fake_steer_active_turn(*, thread_id, active_turn_id, content, timeout=30):
        captured["thread_id"] = thread_id
        captured["active_turn_id"] = active_turn_id
        captured["content"] = content
        return {"turnId": active_turn_id}

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.cli.steer_active_turn", _fake_steer_active_turn),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "steer", task["id"], "Apply migration first"])
        assert result.exit_code == 0
        verify = runner.invoke(main, ["session", "messages", session["id"], "--kind", "steer"])
        assert verify.exit_code == 0
        payload = json.loads(verify.output)
        assert payload["count"] == 1
        assert payload["messages"][0]["recipient"] == f"executor:{task['id'][:8]}"

    assert captured == {
        "thread_id": "thread-live-1",
        "active_turn_id": "turn-live-1",
        "content": "Apply migration first",
    }


def test_task_steer_log_lists_history(db_conn_path):
    """task steer-log should return persisted steer audit rows."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    session = create_session(conn, project_id=pid, trigger="do")
    plan = create_plan_request(conn, project_id=pid, prompt="p", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"x","summary":"s","tasks":[]}')
    conn.execute("UPDATE plans SET session_id = ? WHERE id = ?", (session["id"], plan["id"]))
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        first = runner.invoke(main, ["task", "steer", task["id"], "First steer", "--no-live"])
        second = runner.invoke(main, ["task", "steer", task["id"], "Second steer", "--no-live"])
        assert first.exit_code == 0
        assert second.exit_code == 0

        result = runner.invoke(main, ["task", "steer-log", task["id"], "--limit", "10"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 2
        assert payload["items"][0]["content"] == "Second steer"
        assert payload["items"][1]["content"] == "First steer"


def test_do_with_title_and_files(db_conn_path):
    """agm do --title and --files are passed through."""
    conn, db_path = db_conn_path
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "exec-with-opts"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/custom-abc", "/tmp/wt"),
        ),
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "do",
                "Add OAuth support",
                "-p",
                "testproj",
                "--title",
                "Add OAuth",
                "-f",
                "src/auth.py",
                "-f",
                "src/routes.py",
            ],
        )
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    task = verify_conn.execute("SELECT * FROM tasks").fetchone()
    assert task["title"] == "Add OAuth"
    import json

    files = json.loads(task["files"])
    assert files == ["src/auth.py", "src/routes.py"]
    verify_conn.close()


def test_do_project_not_found(db_conn_path):
    """agm do should error if the project doesn't exist."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["do", "Fix it", "-p", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_do_enqueue_failure_marks_task_failed(db_conn_path):
    """agm do should mark task failed if enqueue fails."""
    conn, db_path = db_conn_path
    conn.close()

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/fail-abc", "/tmp/wt"),
        ),
        patch(
            "agm.queue.get_queue",
            side_effect=ConnectionError("Redis unavailable"),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["do", "Fails", "-p", "testproj"])
        assert result.exit_code != 0
        assert "Failed to enqueue" in result.output

    verify_conn = get_connection(db_path)
    task = verify_conn.execute("SELECT * FROM tasks").fetchone()
    assert task["status"] == "failed"
    assert "Enqueue failed" in (task["failure_reason"] or "")
    verify_conn.close()


# -- queue flush --


def test_queue_failed_json_empty():
    """queue failed --json should return empty array when no failures."""
    with patch("agm.queue.get_failed_jobs", return_value=[]):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "failed"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == []


def test_queue_flush_no_failed():
    """queue flush with no failed jobs should say so."""
    with (
        patch(
            "agm.queue.get_queue_counts",
            return_value={"agm:plans": {"failed": 0}},
        ),
        patch(
            "agm.queue.flush_failed_jobs", return_value={"agm:plans": 0}
        ) as mock_flush_failed_jobs,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "flush", "--yes"])
        assert result.exit_code == 0
        mock_flush_failed_jobs.assert_not_called()


def test_queue_flush_clears_jobs():
    """queue flush should report flushed counts."""
    with (
        patch(
            "agm.queue.get_queue_counts",
            return_value={"agm:plans": {"failed": 5}, "agm:exec": {"failed": 3}},
        ),
        patch(
            "agm.queue.flush_failed_jobs",
            return_value={"agm:plans": 5, "agm:exec": 3, "agm:tasks": 0, "agm:merge": 0},
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "flush", "--yes"])
        assert result.exit_code == 0


def test_queue_flush_prompts_and_approves():
    """queue flush should prompt and flush jobs when approved."""
    with (
        patch(
            "agm.queue.get_queue_counts",
            return_value={"agm:plans": {"failed": 5}, "agm:exec": {"failed": 3}},
        ),
        patch(
            "agm.queue.flush_failed_jobs",
            return_value={"agm:plans": 5, "agm:exec": 3, "agm:tasks": 0, "agm:merge": 0},
        ) as mock_flush_failed_jobs,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "flush"], input="y\n")
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_flush_failed_jobs.assert_called_once_with(None)


def test_queue_flush_prompts_and_aborts_on_decline():
    """queue flush should prompt for confirmation and abort when declined."""
    with (
        patch(
            "agm.queue.get_queue_counts",
            return_value={"agm:plans": {"failed": 1}},
        ),
        patch(
            "agm.queue.flush_failed_jobs", return_value={"agm:plans": 1}
        ) as mock_flush_failed_jobs,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "flush"], input="n\n")
        assert result.exit_code == 0
        assert "Proceed?" in result.output
        mock_flush_failed_jobs.assert_not_called()


def test_queue_inspect_json():
    """queue inspect should return structured inspection rows."""
    rows = [{"job_id": "exec-abc", "queue": "agm:exec", "status": "started"}]
    with patch("agm.queue.inspect_queue_jobs", return_value=rows):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "inspect", "-q", "agm:exec", "-n", "10"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload[0]["job_id"] == "exec-abc"


# -- queue clean --


def test_queue_clean_requires_flag():
    """queue clean without --logs, --finished, or --all should error."""
    runner = CliRunner()
    result = runner.invoke(main, ["queue", "clean"])
    assert result.exit_code != 0
    assert "Specify --logs, --finished, or --all" in result.output


def test_queue_clean_finished():
    """queue clean --finished should clear finished job metadata."""
    with patch(
        "agm.queue.clean_finished_jobs",
        return_value={"agm:plans": 3, "agm:exec": 0, "agm:tasks": 1, "agm:merge": 0},
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "clean", "--yes", "--finished"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["finished_jobs"]["agm:plans"] == 3
        assert payload["finished_jobs"]["agm:tasks"] == 1


def test_queue_clean_logs():
    """queue clean --logs should delete log files."""
    with patch(
        "agm.queue.clean_log_files",
        return_value={"deleted": 5, "kept": 2},
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "clean", "--yes", "--logs"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["logs"]["deleted"] == 5
        assert payload["logs"]["kept"] == 2


def test_queue_clean_all_json():
    """queue clean --all returns combined JSON output."""
    with (
        patch(
            "agm.queue.clean_finished_jobs",
            return_value={"agm:plans": 2, "agm:exec": 0, "agm:tasks": 0, "agm:merge": 0},
        ),
        patch(
            "agm.queue.clean_log_files",
            return_value={"deleted": 3, "kept": 0},
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "clean", "--all", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["finished_jobs"]["agm:plans"] == 2
        assert data["logs"]["deleted"] == 3


def test_queue_clean_no_finished_jobs():
    """queue clean --finished with nothing to clean returns JSON with zero counts."""
    with patch(
        "agm.queue.clean_finished_jobs",
        return_value={"agm:plans": 0, "agm:exec": 0, "agm:tasks": 0, "agm:merge": 0},
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["queue", "clean", "--yes", "--finished"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert all(v == 0 for v in payload["finished_jobs"].values())


# -- plan retask --


def test_plan_retask_success(db_conn_path):
    """plan retask should reset task_creation_status and enqueue."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "job-123"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_task_creation", return_value=mock_job) as mock_enqueue,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retask", plan["id"]])
        assert result.exit_code == 0
        mock_enqueue.assert_called_once_with(plan["id"])

    # Verify status was reset
    verify_conn = get_connection(db_path)
    p = verify_conn.execute("SELECT * FROM plans WHERE id = ?", (plan["id"],)).fetchone()
    # Note: enqueue_task_creation is mocked, so task_creation_status stays "pending"
    # (the real enqueue triggers run_task_creation which sets "running")
    assert p["task_creation_status"] == "pending"
    verify_conn.close()


def test_plan_retask_not_finalized(db_conn_path):
    """plan retask should reject non-finalized plans."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects").fetchone()["id"]
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retask", plan["id"]])
        assert result.exit_code != 0
        assert "not 'finalized'" in result.output


def test_plan_retask_not_found(db_conn_path):
    """plan retask should error on nonexistent plan."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "retask", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


# --- task retry --run ---


def test_task_retry_run(db_conn_path):
    """task retry --run should reset, claim, and enqueue in one step."""
    conn, db_path = db_conn_path
    plan, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "failed")
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    set_project_base_branch(conn, pid, "feature/xyz")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "exec-test"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/test-branch", "/tmp/wt"),
        ) as mock_create,
        patch("agm.queue.enqueue_task_execution", return_value=mock_job),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", "--run", task["id"]])
        assert result.exit_code == 0, result.output
        mock_create.assert_called_once_with(
            "/tmp/testproj",
            task["id"],
            "Test task",
            "feature/xyz",
        )

    verify = get_connection(db_path)
    found = verify.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert found["status"] == "running"
    assert found["branch"] == "agm/test-branch"
    verify.close()


def test_task_retry_run_merge_failed_clears_thread_id(db_conn_path):
    """task retry --run should clear merge-failed execution context before re-run."""
    conn, db_path = db_conn_path
    plan, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "approved")
    conn.execute(
        "UPDATE tasks SET thread_id = ?, reviewer_thread_id = ?, pid = ?, actor = ?, "
        "caller = ? WHERE id = ?",
        ("stale-thread", "stale-reviewer-thread", 999, "cli", "cli", task["id"]),
    )
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="Merge failed and needs rerun",
        source="executor",
    )
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "exec-merge-rerun"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.cli._create_worktree",
            return_value=("agm/test-branch", "/tmp/wt"),
        ),
        patch("agm.queue.enqueue_task_execution", return_value=mock_job),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", "--run", task["id"]])
        assert result.exit_code == 0

    verify_conn = get_connection(db_path)
    found = verify_conn.execute(
        "SELECT thread_id, reviewer_thread_id FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()
    assert found["thread_id"] == ""
    assert found["reviewer_thread_id"] == ""
    verify_conn.close()


def test_task_retry_run_not_failed(db_conn_path):
    """task retry --run should reject non-failed tasks."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "retry", "--run", task["id"]])
        assert result.exit_code != 0
        assert "not 'failed'" in result.output


# --- task check ---


def test_task_check_no_worktree(db_conn_path):
    """task check should error when task has no worktree."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "check", task["id"]])
        assert result.exit_code != 0
        assert "no worktree" in result.output


def test_task_check_passes(db_conn_path):
    """task check should report success when quality gate passes."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="b", worktree="/tmp/wt-check")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.jobs._run_quality_checks", return_value=_qg_pass()),
        patch("os.path.isdir", return_value=True),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "check", task["id"]])
        assert result.exit_code == 0


def test_task_check_fails(db_conn_path):
    """task check should report failures from quality gate."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="b", worktree="/tmp/wt-check")
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.jobs._run_quality_checks",
            return_value=_qg_fail("ruff check", "E501 line too long"),
        ),
        patch("os.path.isdir", return_value=True),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "check", task["id"]])
        assert result.exit_code == 1
        assert "Quality checks failed" in result.output


def test_task_check_output_is_truncated(db_conn_path):
    """task check should cap quality output at the configured limit."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="b", worktree="/tmp/wt-check")
    conn.close()

    long_output = "x" * 2500
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.jobs._run_quality_checks",
            return_value=_qg_fail("ruff check", long_output),
        ),
        patch("os.path.isdir", return_value=True),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "check", task["id"]])
        assert result.exit_code == 1
        assert "Quality checks failed" in result.output


# --- task diff ---


def test_task_diff_no_branch_or_merge_commit(db_conn_path):
    """task diff should error when task has no branch and no merge commit."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", task["id"]])
        assert result.exit_code != 0
        assert "no branch or merge commit" in result.output


def test_task_diff_not_found(db_conn_path):
    """task diff should error when task doesn't exist."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", "nonexistent-id"])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_task_diff_json(db_conn_path):
    """task diff --json returns structured output with diff text."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="test-branch", worktree="/tmp/wt")
    conn.close()

    mock_result = MagicMock()
    mock_result.stdout = "diff --git a/foo.py b/foo.py\n+hello\n"
    mock_result.returncode = 0

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", return_value=mock_result),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", task["id"]])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["task_id"] == task["id"]
        assert "diff --git" in data["diff"]


def test_task_diff_plain(db_conn_path):
    """task diff without --json outputs raw diff text."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="test-branch", worktree="/tmp/wt")
    conn.close()

    mock_result = MagicMock()
    mock_result.stdout = "diff --git a/foo.py b/foo.py\n+hello\n"
    mock_result.returncode = 0

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", return_value=mock_result),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", task["id"]])
        assert result.exit_code == 0
        assert "diff --git" in result.output


def test_task_diff_empty(db_conn_path):
    """task diff with no changes returns empty diff string."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="test-branch", worktree="/tmp/wt")
    conn.close()

    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.returncode = 0

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", return_value=mock_result),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", task["id"]])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["task_id"] == task["id"]
        assert data["diff"] == ""


def test_task_diff_merge_commit_fallback(db_conn_path):
    """task diff falls back to merge_commit when branch is gone."""
    from agm.db import set_task_merge_commit

    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    # Set merge_commit but no branch (simulates post-cleanup state)
    set_task_merge_commit(conn, task["id"], "abc123def456")
    conn.close()

    mock_result = MagicMock()
    mock_result.stdout = "diff --git a/foo.py b/foo.py\n+merged\n"
    mock_result.returncode = 0

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "diff", task["id"]])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "diff --git" in data["diff"]
        # Should have called git diff with merge_commit^1..merge_commit
        call_args = mock_run.call_args_list[-1]
        assert "abc123def456^1..abc123def456" in call_args[0][0]


# --- task logs --worker ---


def test_task_logs_worker(db_conn_path):
    """task logs --worker should read worker log file."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("Worker output line 1\nWorker output line 2\n")
        log_path = Path(f.name)

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value="Worker output line 1\n"),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "logs", "--worker", task["id"]])
        assert result.exit_code == 0
        assert "Worker output" in result.output

    log_path.unlink(missing_ok=True)


def test_task_logs_worker_not_found(db_conn_path):
    """task logs --worker returns null worker_log when file missing."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "logs", "--worker", task["id"]])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["task_id"] == task["id"]
        assert data["worker_log"] is None


# -- task failures --


def test_task_failures_none(db_conn_path):
    """task failures with no failed tasks returns empty list."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "failures"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


def test_task_failures_shows_failed(db_conn_path):
    """task failures shows failed tasks with error snippets."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    add_task_log(conn, task_id=task["id"], level="ERROR", message="ruff check failed: E501")
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "failures"])
        assert result.exit_code == 0
        assert task["id"] in result.output
        assert "ruff check failed" in result.output


def test_task_failures_json(db_conn_path):
    """task failures --json returns structured output."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    add_task_log(conn, task_id=task["id"], level="ERROR", message="pytest failed")
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "failures"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["task_id"] == task["id"]
        assert "pytest failed" in data[0]["error"]


def test_task_failures_uses_failure_reason_when_error_log_has_no_detail(db_conn_path):
    """task failures should fall back to stored failure_reason metadata."""
    conn, db_path = db_conn_path
    _, task = _make_task(conn, db_path)
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    add_task_log(conn, task_id=task["id"], level="ERROR", message="Task execution failed: ")
    set_task_failure_reason(
        conn,
        task["id"],
        json.dumps(
            {
                "source": "execution",
                "exception_type": "TimeoutError",
                "message": "",
                "task_id": task["id"],
                "context": {"path": "callback"},
            }
        ),
    )
    update_task_status(conn, task["id"], "failed")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["task", "failures"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["task_id"] == task["id"]
        assert "Task execution failed: TimeoutError" in data[0]["error"]


# -- plan troubleshoot --


def test_plan_troubleshoot_json(db_conn_path):
    """plan troubleshoot --json returns structured report."""
    conn, db_path = db_conn_path
    conn.close()

    queue_mock = {"ok": True, "queues": {}}
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_queue_counts_safe", return_value=queue_mock),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "troubleshoot"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "failed_plans" in data
        assert "failed_tasks" in data
        assert "stale_running_plans" in data
        assert "stale_running_tasks" in data
        assert "queue" in data


# -- project quality-gate --


def test_project_show_parses_json_columns(db_conn_path):
    """project show returns parsed JSON for project JSON-backed config columns."""
    conn, db_path = db_conn_path
    conn.execute(
        "UPDATE projects SET quality_gate = ?, setup_result = ?, app_server_ask_for_approval = ? "
        "WHERE name = ?",
        (
            json.dumps({"auto_fix": [], "checks": [{"name": "lint", "cmd": ["ruff", "check"]}]}),
            json.dumps({"warnings": ["note"]}),
            json.dumps("on-request"),
            "testproj",
        ),
    )
    conn.commit()
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "show", "testproj"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert isinstance(payload["quality_gate"], dict)
        assert payload["quality_gate"]["checks"][0]["name"] == "lint"
        assert isinstance(payload["setup_result"], dict)
        assert payload["setup_result"]["warnings"] == ["note"]
        assert payload["app_server_approval_policy"]["execCommandApproval"] == "approved"
        assert payload["app_server_ask_for_approval"] == "on-request"


def test_project_quality_gate_show_default(db_conn_path):
    """project quality-gate shows no quality gate when none set."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj"])
        assert result.exit_code == 0
        assert result.output.strip() == "null"


def test_project_quality_gate_set_and_view(db_conn_path):
    """project quality-gate --set stores custom config."""
    conn, db_path = db_conn_path
    conn.close()

    custom = json.dumps({"checks": [{"name": "mypy", "cmd": ["mypy", "src/"]}]})
    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--set", custom])
        assert result.exit_code == 0

        # Verify it's stored
        result = runner.invoke(main, ["project", "quality-gate", "testproj"])
        assert result.exit_code == 0
        assert "mypy" in result.output


def test_project_quality_gate_reset(db_conn_path):
    """project quality-gate --reset clears custom config."""
    conn, db_path = db_conn_path
    conn.close()

    custom = json.dumps({"checks": [{"name": "mypy", "cmd": ["mypy"]}]})
    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        runner.invoke(main, ["project", "quality-gate", "testproj", "--set", custom])
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--reset"])
        assert result.exit_code == 0

        result = runner.invoke(main, ["project", "quality-gate", "testproj"])
        assert result.output.strip() == "null"


def test_project_quality_gate_show_default_flag(db_conn_path):
    """project quality-gate --show-default shows default config without project lookup."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--show-default"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "auto_fix" in data
        assert "checks" in data


def test_project_quality_gate_invalid_json(db_conn_path):
    """project quality-gate --set with invalid JSON fails."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--set", "not json"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output


def test_project_quality_gate_list_presets():
    """--list-presets shows available presets without project lookup."""
    runner = CliRunner()
    result = runner.invoke(main, ["project", "quality-gate", "--list-presets"])
    assert result.exit_code == 0
    assert "python" in result.output
    assert "typescript" in result.output


def test_project_quality_gate_list_presets_json():
    """--list-presets --json returns structured preset data."""
    runner = CliRunner()
    result = runner.invoke(main, ["project", "quality-gate", "--list-presets"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "python" in data
    assert "typescript" in data
    assert "config" in data["python"]


def test_project_quality_gate_apply_preset(db_conn_path):
    """--preset applies a preset config to a project."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--preset", "python"])
        assert result.exit_code == 0

        # Verify it's stored
        result = runner.invoke(main, ["project", "quality-gate", "testproj"])
        assert result.exit_code == 0
        assert "ruff" in result.output


def test_project_quality_gate_unknown_preset(db_conn_path):
    """--preset with unknown name fails with helpful error."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--preset", "rust"])
        assert result.exit_code != 0
        assert "Unknown preset" in result.output
        assert "python" in result.output  # Shows available presets


def test_project_quality_gate_generate_json(db_conn_path):
    """--generate --json returns just the config."""
    mock_config = {
        "auto_fix": [],
        "checks": [{"name": "pytest", "cmd": ["pytest"], "timeout": 300}],
    }
    conn, db_path = db_conn_path
    conn.close()

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.jobs.generate_quality_gate", return_value=mock_config),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "quality-gate", "testproj", "--generate"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "checks" in data
        assert data["checks"][0]["name"] == "pytest"


def test_project_quality_gate_no_project_required_for_list_presets():
    """--list-presets works without a project name."""
    runner = CliRunner()
    result = runner.invoke(main, ["project", "quality-gate", "--list-presets"])
    assert result.exit_code == 0


def test_project_quality_gate_missing_name_errors():
    """Operations that need a project fail without name_or_id."""
    runner = CliRunner()
    result = runner.invoke(main, ["project", "quality-gate", "--reset"])
    assert result.exit_code != 0
    assert "NAME_OR_ID" in result.output


# -- project model-config --


def test_project_model_config_set_unknown_model_emits_warning(db_conn_path):
    """project model-config --set warns for unknown model IDs without failing."""
    conn, db_path = db_conn_path
    conn.close()

    unknown = json.dumps({"think_model": "not-a-real-model"})

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "model-config", "testproj", "--set", unknown])
        assert result.exit_code == 0
        assert "unknown to catalog metadata" in result.output.lower()

    conn, db_path = db_conn_path
    conn.close()

    custom = json.dumps({"think_model": "gpt-5.3-codex", "work_model": "gpt-5.3-codex-spark"})
    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        runner.invoke(main, ["project", "model-config", "testproj", "--set", custom])
        result = runner.invoke(main, ["project", "show", "testproj"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "model_config" in payload
        assert "configured" in payload["model_config"]
        assert payload["model_config"]["configured"]["think_model"] == "gpt-5.3-codex"


def test_project_base_branch_show_set_reset(db_conn_path):
    """project base-branch command should show effective value, set, and reset."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "base-branch", "testproj"])
        assert result.exit_code == 0
        assert '"base_branch": "main"' in result.output

        result = runner.invoke(main, ["project", "base-branch", "testproj", "release"])
        assert result.exit_code == 0

        result = runner.invoke(main, ["project", "base-branch", "testproj"])
        assert result.exit_code == 0
        assert '"base_branch": "release"' in result.output

        result = runner.invoke(main, ["project", "base-branch", "testproj", "--reset"])
        assert result.exit_code == 0

        result = runner.invoke(main, ["project", "base-branch", "testproj"])
        assert result.exit_code == 0
        assert '"base_branch": "main"' in result.output

        verify_conn = get_connection(db_path)
        row = verify_conn.execute(
            "SELECT base_branch FROM projects WHERE id = ?",
            (pid,),
        ).fetchone()
        verify_conn.close()
        assert row["base_branch"] is None


def test_plan_request_defaults_to_project_backend(db_conn_path):
    """plan request without --backend should use project's default_backend."""
    conn, db_path = db_conn_path
    conn.close()

    mock_enqueue = MagicMock()
    mock_enqueue.return_value = MagicMock(id="job-123")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_plan_request", mock_enqueue),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "request", "test prompt", "-p", "testproj"])
        assert result.exit_code == 0

        # The plan should have been created with backend='codex'
        with get_connection(db_path) as check_conn:
            check_conn.row_factory = __import__("sqlite3").Row
            plans = check_conn.execute("SELECT backend FROM plans").fetchall()
            assert any(p["backend"] == "codex" for p in plans)


def test_plan_request_prefers_explicit_backend_over_project_default(db_conn_path):
    """plan request should prefer explicit --backend over project default."""
    conn, db_path = db_conn_path
    conn.close()

    mock_enqueue = MagicMock()
    mock_enqueue.return_value = MagicMock(id="job-456")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_plan_request", mock_enqueue),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "request", "test prompt", "-p", "testproj", "--backend", "codex"],
        )
        assert result.exit_code == 0

        with get_connection(db_path) as check_conn:
            check_conn.row_factory = __import__("sqlite3").Row
            plans = check_conn.execute("SELECT backend FROM plans").fetchall()
            assert any(p["backend"] == "codex" for p in plans)


def test_do_defaults_to_project_backend(db_conn_path):
    """do command without --backend should use project's default_backend."""
    conn, db_path = db_conn_path
    conn.close()

    mock_enqueue = MagicMock()
    mock_enqueue.return_value = MagicMock(id="job-456")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_task_execution", mock_enqueue),
        patch("agm.cli._create_worktree", return_value=("branch-test", "/tmp/wt")),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["do", "test task", "-p", "testproj"])
        assert result.exit_code == 0

        # The plan should have backend='codex'
        with get_connection(db_path) as check_conn:
            check_conn.row_factory = __import__("sqlite3").Row
            plans = check_conn.execute("SELECT backend FROM plans").fetchall()
            assert any(p["backend"] == "codex" for p in plans)


def test_do_prefers_explicit_backend_over_project_default(db_conn_path):
    """do should use explicit --backend over project default."""
    conn, db_path = db_conn_path
    conn.close()

    mock_enqueue = MagicMock()
    mock_enqueue.return_value = MagicMock(id="job-do-explicit")

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(db_path),
        ),
        patch("agm.queue.enqueue_task_execution", mock_enqueue),
        patch("agm.cli._create_worktree", return_value=("branch-test", "/tmp/wt")),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["do", "test task", "-p", "testproj", "--backend", "codex"],
        )
        assert result.exit_code == 0

        with get_connection(db_path) as check_conn:
            check_conn.row_factory = __import__("sqlite3").Row
            plans = check_conn.execute("SELECT backend FROM plans").fetchall()
            assert any(p["backend"] == "codex" for p in plans)


def test_do_invalid_backend_is_rejected(db_conn_path):
    """do rejects invalid --backend values."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", return_value=get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["do", "test task", "-p", "testproj", "--backend", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output


# -- Plan approval gate tests --


def test_project_plan_approval_show_default(db_conn_path):
    """project plan-approval shows auto when no mode is set."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "plan-approval", "testproj"])
        assert result.exit_code == 0
        assert '"plan_approval": "auto"' in result.output


def test_project_plan_approval_set_manual(db_conn_path):
    """project plan-approval manual sets and confirms."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "plan-approval", "testproj", "manual"])
        assert result.exit_code == 0

        # Verify it stuck
        result = runner.invoke(main, ["project", "plan-approval", "testproj"])
        assert '"plan_approval": "manual"' in result.output


def test_project_plan_approval_reset(db_conn_path):
    """project plan-approval --reset goes back to auto."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        runner.invoke(main, ["project", "plan-approval", "testproj", "manual"])
        result = runner.invoke(main, ["project", "plan-approval", "testproj", "--reset"])
        assert result.exit_code == 0


def test_project_app_server_approval_show_default(db_conn_path):
    """project app-server-approval shows effective default policy."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "app-server-approval", "testproj"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        policy = payload["app_server_approval_policy"]
        assert policy["item/commandExecution/requestApproval"] == "accept"
        assert policy["execCommandApproval"] == "approved"


def test_project_app_server_approval_set_and_reset(db_conn_path):
    """project app-server-approval can set via JSON and reset to defaults."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        set_payload = json.dumps(
            {
                "item/commandExecution/requestApproval": "decline",
                "execCommandApproval": "denied",
            }
        )
        result = runner.invoke(
            main,
            ["project", "app-server-approval", "testproj", "--set", set_payload],
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(main, ["project", "app-server-approval", "testproj"])
        payload = json.loads(result.output)
        policy = payload["app_server_approval_policy"]
        assert policy["item/commandExecution/requestApproval"] == "decline"
        assert policy["execCommandApproval"] == "denied"

        result = runner.invoke(main, ["project", "app-server-approval", "testproj", "--reset"])
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["project", "app-server-approval", "testproj"])
        payload = json.loads(result.output)
        policy = payload["app_server_approval_policy"]
        assert policy["item/commandExecution/requestApproval"] == "accept"
        assert policy["execCommandApproval"] == "approved"


def test_project_app_server_approval_preset_and_invalid_json(db_conn_path):
    """project app-server-approval supports presets and rejects invalid payloads."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["project", "app-server-approval", "testproj", "--preset", "deny-all"],
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["project", "app-server-approval", "testproj"])
        policy = json.loads(result.output)["app_server_approval_policy"]
        assert policy["item/commandExecution/requestApproval"] == "decline"
        assert policy["execCommandApproval"] == "denied"

        bad = runner.invoke(
            main,
            ["project", "app-server-approval", "testproj", "--set", '["not-an-object"]'],
        )
        assert bad.exit_code != 0
        assert "Policy config must be a JSON object" in bad.output


def test_project_app_server_ask_for_approval_show_default(db_conn_path):
    """project app-server-ask-for-approval shows default AskForApproval policy."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "app-server-ask-for-approval", "testproj"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["app_server_ask_for_approval"] == "never"


def test_project_app_server_ask_for_approval_set_and_reset(db_conn_path):
    """project app-server-ask-for-approval supports presets, JSON set, and reset."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["project", "app-server-ask-for-approval", "testproj", "--preset", "reject-all"],
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(main, ["project", "app-server-ask-for-approval", "testproj"])
        payload = json.loads(result.output)
        assert payload["app_server_ask_for_approval"] == {
            "reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": True}
        }

        result = runner.invoke(
            main,
            [
                "project",
                "app-server-ask-for-approval",
                "testproj",
                "--set",
                '"on-request"',
            ],
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["project", "app-server-ask-for-approval", "testproj"])
        payload = json.loads(result.output)
        assert payload["app_server_ask_for_approval"] == "on-request"

        result = runner.invoke(
            main, ["project", "app-server-ask-for-approval", "testproj", "--reset"]
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["project", "app-server-ask-for-approval", "testproj"])
        payload = json.loads(result.output)
        assert payload["app_server_ask_for_approval"] == "never"


def test_project_app_server_ask_for_approval_invalid_payload(db_conn_path):
    """Invalid AskForApproval payload should return a validation error."""
    conn, db_path = db_conn_path
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        bad = runner.invoke(
            main,
            [
                "project",
                "app-server-ask-for-approval",
                "testproj",
                "--set",
                '{"reject":{"rules":true}}',
            ],
        )
        assert bad.exit_code != 0
        assert "must include only" in bad.output


def test_plan_approve_triggers_task_creation(db_conn_path):
    """plan approve on awaiting_approval plan enqueues task creation."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan_row = create_plan_request(
        conn, project_id=pid, prompt="test plan", actor="test", caller="cli", backend="codex"
    )
    plan_id = plan_row["id"]
    update_plan_request_status(conn, plan_id, "running")
    finalize_plan_request(conn, plan_id, '{"title": "test", "tasks": []}')
    from agm.db import update_plan_task_creation_status

    update_plan_task_creation_status(conn, plan_id, "awaiting_approval")
    conn.close()

    mock_job = MagicMock()
    mock_job.id = "test-job-id"
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_task_creation", return_value=mock_job) as mock_enqueue,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "approve", plan_id])
        assert result.exit_code == 0, result.output
        mock_enqueue.assert_called_once_with(plan_id)


def test_plan_approve_rejects_non_finalized(db_conn_path):
    """plan approve on non-finalized plan fails."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan_row = create_plan_request(
        conn, project_id=pid, prompt="test plan", actor="test", caller="cli", backend="codex"
    )
    plan_id = plan_row["id"]
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "approve", plan_id])
        assert result.exit_code != 0
        assert "not 'finalized'" in result.output


def test_plan_approve_rejects_already_running(db_conn_path):
    """plan approve on plan with task_creation_status=running fails."""
    conn, db_path = db_conn_path
    pid = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()["id"]
    plan_row = create_plan_request(
        conn, project_id=pid, prompt="test plan", actor="test", caller="cli", backend="codex"
    )
    plan_id = plan_row["id"]
    update_plan_request_status(conn, plan_id, "running")
    finalize_plan_request(conn, plan_id, '{"title": "test", "tasks": []}')
    from agm.db import update_plan_task_creation_status

    update_plan_task_creation_status(conn, plan_id, "running")
    conn.close()

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "approve", plan_id])
        assert result.exit_code != 0


# -- Token display helpers --


def test_aggregate_tokens():
    """aggregate_plan_tokens sums plan + task tokens correctly."""
    from agm.queries import aggregate_plan_tokens

    plan = {"input_tokens": 100, "output_tokens": 50}
    tasks = [
        {"input_tokens": 200, "output_tokens": 100},
        {"input_tokens": 300, "output_tokens": 150},
    ]
    result = aggregate_plan_tokens(plan, tasks)
    assert result["total_input"] == 600
    assert result["total_output"] == 300

    # Handles None/missing fields
    result_none = aggregate_plan_tokens({}, [{"input_tokens": None, "output_tokens": None}])
    assert result_none["total_input"] == 0


# -- _check_backend_auth unit tests --


def test_check_backend_auth_codex_installed_authed():
    """_check_backend_auth returns (True, True, ...) for authed codex."""
    from agm.cli import _check_backend_auth

    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd == ["codex", "--version"]:
            return MagicMock(returncode=0, stdout="codex-cli 0.101.0\n", stderr="")
        if cmd == ["codex", "login", "status"]:
            return MagicMock(returncode=0, stdout="Logged in\n", stderr="")
        return orig_run(cmd, **kwargs)

    with patch("subprocess.run", side_effect=mock_run):
        installed, authed, detail = _check_backend_auth("codex")
    assert installed is True
    assert authed is True
    assert "authenticated" in detail
    assert "0.101.0" in detail


def test_check_backend_auth_codex_installed_not_authed():
    """_check_backend_auth returns (True, False, ...) for unauthed codex."""
    from agm.cli import _check_backend_auth

    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd == ["codex", "--version"]:
            return MagicMock(returncode=0, stdout="codex-cli 0.101.0\n", stderr="")
        if cmd == ["codex", "login", "status"]:
            return MagicMock(returncode=1, stdout="", stderr="Not logged in")
        return orig_run(cmd, **kwargs)

    with patch("subprocess.run", side_effect=mock_run):
        installed, authed, detail = _check_backend_auth("codex")
    assert installed is True
    assert authed is False
    assert "not logged in" in detail
    assert "codex login" in detail


def test_check_backend_auth_not_installed():
    """_check_backend_auth returns (False, False, 'not found') when not installed."""
    from agm.cli import _check_backend_auth

    def mock_run(cmd, **kwargs):
        raise FileNotFoundError

    with patch("subprocess.run", side_effect=mock_run):
        installed, authed, detail = _check_backend_auth("codex")
    assert installed is False
    assert authed is False
    assert detail == "not found"


# -- project setup CLI tests --


def test_project_setup_enqueues_job(db_conn_path):
    """project setup (non-dry-run) enqueues a setup job and returns silently."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-proj", "/tmp/setup-proj")
    proj = get_project(conn, "setup-proj")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup") as mock_enqueue,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-proj"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""  # silent on success
        mock_enqueue.assert_called_once_with(proj["id"], "setup-proj", backend=None)


def test_project_setup_wait_blocks_until_completed(db_conn_path):
    """project setup --wait blocks until project:setup completed event."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-wait-proj", "/tmp/setup-wait-proj")
    proj = get_project(conn, "setup-wait-proj")
    enqueued_job = MagicMock()
    enqueued_job.id = "setup-job-1"
    finished_job = MagicMock()
    finished_job.get_status.return_value = "finished"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup", return_value=enqueued_job) as mock_enqueue,
        patch(
            "agm.queue.subscribe_events",
            return_value=iter(
                [
                    {
                        "type": "project:setup",
                        "id": proj["id"],
                        "status": "running",
                        "job_id": "setup-job-1",
                    },
                    {
                        "type": "project:setup",
                        "id": proj["id"],
                        "status": "completed",
                        "job_id": "setup-job-1",
                    },
                ]
            ),
        ) as mock_subscribe,
        patch("agm.queue.get_job", return_value=finished_job) as mock_get_job,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-wait-proj", "--wait"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""
        mock_enqueue.assert_called_once_with(proj["id"], "setup-wait-proj", backend=None)
        mock_subscribe.assert_called_once_with(project="setup-wait-proj", timeout=5.0)
        mock_get_job.assert_called_once_with("setup-job-1")


def test_project_setup_wait_raises_on_failed_event(db_conn_path):
    """project setup --wait surfaces failed setup events as CLI errors."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-fail-proj", "/tmp/setup-fail-proj")
    proj = get_project(conn, "setup-fail-proj")
    enqueued_job = MagicMock()
    enqueued_job.id = "setup-job-2"
    failed_job = MagicMock()
    failed_job.get_status.return_value = "failed"
    failed_job.exc_info = "RuntimeError: boom"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup", return_value=enqueued_job) as mock_enqueue,
        patch(
            "agm.queue.subscribe_events",
            return_value=iter(
                [
                    {
                        "type": "project:setup",
                        "id": proj["id"],
                        "status": "failed",
                        "job_id": "setup-job-2",
                        "error": "boom",
                    }
                ]
            ),
        ),
        patch("agm.queue.get_job", return_value=failed_job),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-fail-proj", "--wait"])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "boom" in payload["error"].lower()
        mock_enqueue.assert_called_once_with(proj["id"], "setup-fail-proj", backend=None)


def test_project_setup_wait_uses_job_state_when_event_missing(db_conn_path):
    """project setup --wait succeeds if job is finished even without terminal event."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-race-proj", "/tmp/setup-race-proj")
    proj = get_project(conn, "setup-race-proj")
    job = MagicMock()
    job.id = "setup-job-1"
    finished_job = MagicMock()
    finished_job.get_status.return_value = "finished"

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup", return_value=job) as mock_enqueue,
        patch("agm.queue.subscribe_events", return_value=iter([None])),
        patch("agm.queue.get_job", return_value=finished_job) as mock_get_job,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-race-proj", "--wait"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""
        mock_enqueue.assert_called_once_with(proj["id"], "setup-race-proj", backend=None)
        mock_get_job.assert_called_once_with("setup-job-1")


def test_project_setup_wait_ignores_mismatched_job_event(db_conn_path):
    """project setup --wait must not complete from another setup job's event."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-other-job-proj", "/tmp/setup-other-job-proj")
    proj = get_project(conn, "setup-other-job-proj")
    enqueued_job = MagicMock()
    enqueued_job.id = "setup-job-ours"
    running_job = MagicMock()
    running_job.get_status.return_value = "started"

    class _SingleMismatchedEvent:
        def __init__(self) -> None:
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self._sent:
                self._sent = True
                return {
                    "type": "project:setup",
                    "id": proj["id"],
                    "status": "completed",
                    "job_id": "setup-job-other",
                }
            return None

        def close(self):
            return None

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup", return_value=enqueued_job),
        patch("agm.queue.subscribe_events", return_value=_SingleMismatchedEvent()),
        patch("agm.queue.get_job", return_value=running_job),
        patch("agm.cli.PROJECT_SETUP_WAIT_TIMEOUT_SECONDS", 1.0),
        patch("agm.cli.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-other-job-proj", "--wait"])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "setup wait timed out" in payload["error"].lower()


def test_project_setup_wait_rejects_dry_run(db_conn_path):
    """project setup rejects combining --wait with --dry-run."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-drywait-proj", "/tmp/setup-drywait-proj")

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["project", "setup", "setup-drywait-proj", "--dry-run", "--wait"],
        )
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "--wait cannot be used with --dry-run" in payload["error"].lower()


def test_project_setup_wait_times_out_without_terminal_event(db_conn_path):
    """project setup --wait errors when no terminal setup event arrives in time."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-timeout-proj", "/tmp/setup-timeout-proj")

    class _TimeoutSubscriber:
        def __iter__(self):
            return self

        def __next__(self):
            return None

        def close(self):
            return None

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch(
            "agm.queue.enqueue_project_setup",
            return_value=MagicMock(id="setup-job-timeout"),
        ),
        patch("agm.queue.subscribe_events", return_value=_TimeoutSubscriber()),
        patch(
            "agm.queue.get_job",
            return_value=MagicMock(get_status=MagicMock(return_value="started")),
        ),
        patch("agm.cli.PROJECT_SETUP_WAIT_TIMEOUT_SECONDS", 1.0),
        patch("agm.cli.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "setup-timeout-proj", "--wait"])
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "setup wait timed out" in payload["error"].lower()


def test_project_setup_enqueues_with_backend_override(db_conn_path):
    """project setup forwards backend override for async setup jobs."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-backend-proj", "/tmp/setup-backend-proj")
    proj = get_project(conn, "setup-backend-proj")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup") as mock_enqueue,
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["project", "setup", "setup-backend-proj", "--backend", "codex"],
        )
        assert result.exit_code == 0, result.output
        mock_enqueue.assert_called_once_with(proj["id"], "setup-backend-proj", backend="codex")


def test_project_setup_unknown_backend_errors(db_conn_path):
    """project setup rejects unknown backend before enqueue."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-unknown-backend-proj", "/tmp/setup-unknown-backend-proj")

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.enqueue_project_setup") as mock_enqueue,
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["project", "setup", "setup-unknown-backend-proj", "--backend", "bogus"],
        )
        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert "unknown backend" in payload["error"].lower()
        mock_enqueue.assert_not_called()


def test_project_setup_dry_run_passes_flag(db_conn_path):
    """project setup --dry-run passes dry_run=True to run_project_setup."""
    conn, db_path = db_conn_path
    add_project(conn, "dry-proj", "/tmp/dry-proj")

    fake_result = {
        "quality_gate": {"auto_fix": [], "checks": []},
        "post_merge_command": None,
        "stack": {"languages": [], "package_manager": None, "tools": []},
        "warnings": [],
        "quality_gate_applied": False,
        "post_merge_applied": False,
    }
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.jobs_setup.run_project_setup", return_value=fake_result) as mock_setup,
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup", "dry-proj", "--dry-run"])
        assert result.exit_code == 0, result.output
        mock_setup.assert_called_once_with("dry-proj", backend=None, dry_run=True)


def test_project_setup_status_reports_job_and_setup_result(db_conn_path):
    """project setup-status returns rq status, queue row, and parsed setup result."""
    conn, db_path = db_conn_path
    add_project(conn, "setup-status-proj", "/tmp/setup-status-proj")
    proj = get_project(conn, "setup-status-proj")
    assert proj is not None
    conn.execute(
        "UPDATE projects SET setup_result = ? WHERE id = ?",
        (json.dumps({"warnings": [], "post_merge_command": "make sync"}), proj["id"]),
    )
    conn.commit()

    setup_job_id = f"setup-{proj['id']}"
    mock_job = MagicMock()
    mock_job.get_status.return_value = "finished"
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)),
        patch("agm.queue.get_job", return_value=mock_job),
        patch(
            "agm.queue.inspect_queue_jobs",
            return_value=[{"job_id": setup_job_id, "queue": "agm:setup", "status": "finished"}],
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["project", "setup-status", "setup-status-proj"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["project_name"] == "setup-status-proj"
        assert payload["setup_job_id"] == setup_job_id
        assert payload["rq_status"] == "finished"
        assert payload["queue"]["queue"] == "agm:setup"
        assert payload["setup_result"]["post_merge_command"] == "make sync"


# -- init quality gate integration tests --


def _make_init_mock_run(tmp_path, *, codex_auth=False):
    """Create a mock subprocess.run for init tests with auth control."""
    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=0)
        # codex --version
        if cmd == ["codex", "--version"]:
            raise FileNotFoundError
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    return mock_run


def _make_auth_mock_run(tmp_path, *, codex_installed=True, codex_authed=True):
    """Create a mock subprocess.run for init tests with auth control."""
    orig_run = sp.run

    def mock_run(cmd, **kwargs):
        if cmd[0] == "git" and cmd[1] == "rev-parse":
            return MagicMock(returncode=0)
        # codex --version
        if cmd == ["codex", "--version"]:
            if not codex_installed:
                raise FileNotFoundError
            return MagicMock(returncode=0, stdout="codex-cli 0.101.0\n", stderr="")
        # codex login status
        if len(cmd) >= 3 and cmd[:3] == ["codex", "login", "status"]:
            return MagicMock(returncode=0 if codex_authed else 1, stdout="", stderr="")
        if cmd[0] == "codex":
            raise FileNotFoundError
        return orig_run(cmd, **kwargs)

    return mock_run
