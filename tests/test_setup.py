"""Tests for the project setup agent (jobs_setup.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from agm.db import (
    add_project,
    get_connection,
    get_project,
    get_project_post_merge_command,
    get_project_quality_gate,
)
from agm.jobs_setup import (
    _apply_setup_result,
    _ensure_agents_toml,
    _parse_setup_output,
    on_setup_failure,
    run_project_setup_worker,
)

# ---------------------------------------------------------------------------
# _parse_setup_output
# ---------------------------------------------------------------------------


def test_parse_setup_output_valid():
    """Parses valid full setup output."""
    raw = json.dumps(
        {
            "quality_gate": {
                "auto_fix": [{"name": "fmt", "cmd": ["ruff", "format", "."]}],
                "checks": [{"name": "lint", "cmd": ["ruff", "check", "."], "timeout": 60}],
            },
            "post_merge_command": "make install-bin",
            "stack": {
                "languages": ["python"],
                "package_manager": "uv",
                "tools": ["ruff", "pytest"],
            },
            "warnings": [],
        }
    )
    result = _parse_setup_output(raw)
    assert result["quality_gate"]["auto_fix"][0]["name"] == "fmt"
    assert result["post_merge_command"] == "make install-bin"
    assert result["stack"]["languages"] == ["python"]
    assert result["warnings"] == []


def test_parse_setup_output_null_post_merge():
    """Accepts null post_merge_command."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "post_merge_command": None,
            "stack": {"languages": [], "package_manager": None, "tools": []},
            "warnings": ["No test runner detected"],
        }
    )
    result = _parse_setup_output(raw)
    assert result["post_merge_command"] is None
    assert result["warnings"] == ["No test runner detected"]


def test_parse_setup_output_missing_quality_gate():
    """Raises ValueError when quality_gate is missing."""
    raw = json.dumps(
        {
            "post_merge_command": None,
            "stack": {"languages": [], "package_manager": None, "tools": []},
            "warnings": [],
        }
    )
    with pytest.raises(ValueError, match="quality_gate"):
        _parse_setup_output(raw)


def test_parse_setup_output_missing_stack():
    """Raises ValueError when stack is missing."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "post_merge_command": None,
            "warnings": [],
        }
    )
    with pytest.raises(ValueError, match="stack"):
        _parse_setup_output(raw)


def test_parse_setup_output_missing_warnings():
    """Raises ValueError when warnings is missing."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "post_merge_command": None,
            "stack": {"languages": [], "package_manager": None, "tools": []},
        }
    )
    with pytest.raises(ValueError, match="warnings"):
        _parse_setup_output(raw)


def test_parse_setup_output_invalid_json():
    """Raises on invalid JSON."""
    with pytest.raises(json.JSONDecodeError):
        _parse_setup_output("not json")


def test_parse_setup_output_not_object():
    """Raises on non-object JSON."""
    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_setup_output('"just a string"')


def test_parse_setup_output_missing_post_merge_defaults_to_none():
    """Missing post_merge_command defaults to None."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "stack": {"languages": [], "package_manager": None, "tools": []},
            "warnings": [],
        }
    )
    result = _parse_setup_output(raw)
    assert result["post_merge_command"] is None


def test_parse_setup_output_with_reasoning():
    """Parses reasoning array from setup output."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "post_merge_command": None,
            "stack": {"languages": ["python"], "package_manager": "uv", "tools": []},
            "warnings": [],
            "reasoning": [
                {"item": "ruff", "action": "configured", "detail": "Found in pyproject.toml"},
                {"item": "jest", "action": "not_found", "detail": "No jest config detected"},
            ],
        }
    )
    result = _parse_setup_output(raw)
    assert len(result["reasoning"]) == 2
    assert result["reasoning"][0]["item"] == "ruff"
    assert result["reasoning"][1]["action"] == "not_found"


def test_parse_setup_output_missing_reasoning_defaults_to_empty():
    """Missing reasoning defaults to empty array (backward compat)."""
    raw = json.dumps(
        {
            "quality_gate": {"auto_fix": [], "checks": []},
            "post_merge_command": None,
            "stack": {"languages": [], "package_manager": None, "tools": []},
            "warnings": [],
        }
    )
    result = _parse_setup_output(raw)
    assert result["reasoning"] == []


# ---------------------------------------------------------------------------
# _apply_setup_result
# ---------------------------------------------------------------------------


def _sample_result(*, with_qg: bool = True, with_pmc: bool = True) -> dict:
    """Build a sample setup result for apply tests."""
    return {
        "quality_gate": {
            "auto_fix": [{"name": "fmt", "cmd": ["ruff", "format", "."]}],
            "checks": [{"name": "lint", "cmd": ["ruff", "check", "."], "timeout": 60}],
        }
        if with_qg
        else {"auto_fix": [], "checks": []},
        "post_merge_command": "make install-bin" if with_pmc else None,
        "stack": {"languages": ["python"], "package_manager": "uv", "tools": ["ruff"]},
        "warnings": [],
        "reasoning": [
            {"item": "ruff", "action": "configured", "detail": "Found in pyproject.toml"},
        ],
    }


def test_apply_setup_result_applies_both(db_conn_path):
    """Applies quality gate and post-merge to the project."""
    conn, db_path = db_conn_path
    add_project(conn, "apply-test", "/tmp/apply-test")
    proj = get_project(conn, "apply-test")
    assert proj is not None

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = _apply_setup_result(proj["id"], _sample_result())

    assert result["quality_gate_applied"] is True
    assert result["post_merge_applied"] is True
    assert get_project_quality_gate(conn, proj["id"]) is not None
    assert get_project_post_merge_command(conn, proj["id"]) == "make install-bin"


def test_apply_setup_result_skips_empty_qg(db_conn_path):
    """Skips quality gate when auto_fix and checks are both empty."""
    conn, db_path = db_conn_path
    add_project(conn, "empty-qg", "/tmp/empty-qg")
    proj = get_project(conn, "empty-qg")
    assert proj is not None

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = _apply_setup_result(proj["id"], _sample_result(with_qg=False, with_pmc=False))

    assert result["quality_gate_applied"] is False
    assert result["post_merge_applied"] is False


def test_apply_setup_result_skips_null_pmc(db_conn_path):
    """Skips post-merge when command is null."""
    conn, db_path = db_conn_path
    add_project(conn, "null-pmc", "/tmp/null-pmc")
    proj = get_project(conn, "null-pmc")
    assert proj is not None

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = _apply_setup_result(proj["id"], _sample_result(with_pmc=False))

    assert result["quality_gate_applied"] is True
    assert result["post_merge_applied"] is False


def test_apply_setup_result_dry_run(db_conn_path):
    """dry_run=True returns result without writing to DB."""
    conn, db_path = db_conn_path
    add_project(conn, "dry-run", "/tmp/dry-run")
    proj = get_project(conn, "dry-run")
    assert proj is not None

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = _apply_setup_result(proj["id"], _sample_result(), dry_run=True)

    assert result["quality_gate_applied"] is False
    assert result["post_merge_applied"] is False
    assert get_project_quality_gate(conn, proj["id"]) is None
    assert get_project_post_merge_command(conn, proj["id"]) is None


def test_apply_setup_result_overwrites_existing(db_conn_path):
    """Re-running apply overwrites existing config."""
    conn, db_path = db_conn_path
    add_project(conn, "overwrite", "/tmp/overwrite")
    proj = get_project(conn, "overwrite")
    assert proj is not None
    from agm.db import set_project_quality_gate

    set_project_quality_gate(conn, proj["id"], '{"auto_fix": [], "checks": []}')

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        result = _apply_setup_result(proj["id"], _sample_result())

    assert result["quality_gate_applied"] is True
    qg = get_project_quality_gate(conn, proj["id"])
    assert qg is not None
    parsed = json.loads(qg)
    assert len(parsed["auto_fix"]) == 1  # New config applied, not empty


def test_apply_setup_result_stores_full_result(db_conn_path):
    """Apply stores the full setup result JSON in the setup_result column."""
    conn, db_path = db_conn_path
    add_project(conn, "full-result", "/tmp/full-result")
    proj = get_project(conn, "full-result")
    assert proj is not None

    from agm.db import get_project_setup_result

    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(db_path)):
        _apply_setup_result(proj["id"], _sample_result())

    stored = get_project_setup_result(conn, proj["id"])
    assert stored is not None
    parsed = json.loads(stored)
    assert parsed["stack"]["languages"] == ["python"]
    assert parsed["reasoning"][0]["item"] == "ruff"


def test_ensure_agents_toml_creates_scaffold(tmp_path):
    """Creates .agm/agents.toml scaffold if missing."""
    assert _ensure_agents_toml(str(tmp_path)) is True
    agents_toml = tmp_path / ".agm" / "agents.toml"
    assert agents_toml.exists()
    content = agents_toml.read_text()
    assert "[executor]" in content
    assert "[reviewer]" in content


def test_ensure_agents_toml_skips_existing(tmp_path):
    """Does not overwrite existing agents.toml."""
    agm_dir = tmp_path / ".agm"
    agm_dir.mkdir()
    agents_toml = agm_dir / "agents.toml"
    agents_toml.write_text("# custom")
    assert _ensure_agents_toml(str(tmp_path)) is False
    assert agents_toml.read_text() == "# custom"


# ---------------------------------------------------------------------------
# run_project_setup_worker
# ---------------------------------------------------------------------------


def test_worker_emits_running_and_completed_events():
    """Worker emits running then completed events on success."""
    fake_result = {**_sample_result(), "quality_gate_applied": True, "post_merge_applied": True}
    with (
        patch("agm.jobs_setup.run_project_setup", return_value=fake_result) as mock_setup,
        patch("agm.queue.publish_event") as mock_pub,
    ):
        result = run_project_setup_worker("proj-123", "my-proj")

    assert result["quality_gate_applied"] is True
    mock_setup.assert_called_once_with("proj-123", backend=None)
    assert mock_pub.call_count == 2
    # First call: running
    assert mock_pub.call_args_list[0] == call(
        "project:setup",
        "proj-123",
        "running",
        project="my-proj",
        extra=None,
    )
    # Second call: completed with extras
    completed_call = mock_pub.call_args_list[1]
    assert completed_call[0] == ("project:setup", "proj-123", "completed")
    assert completed_call[1]["extra"]["quality_gate_applied"] is True


def test_worker_passes_backend_override():
    """Worker forwards explicit backend override to run_project_setup."""
    fake_result = {**_sample_result(), "quality_gate_applied": True, "post_merge_applied": True}
    with (
        patch("agm.jobs_setup.run_project_setup", return_value=fake_result) as mock_setup,
        patch("agm.queue.publish_event"),
    ):
        run_project_setup_worker("proj-123", "my-proj", "codex")

    mock_setup.assert_called_once_with("proj-123", backend="codex")


def test_worker_emits_failed_event_on_error():
    """Worker emits running then failed events and re-raises on error."""
    with (
        patch("agm.jobs_setup.run_project_setup", side_effect=ValueError("bad")),
        patch("agm.queue.publish_event") as mock_pub,
        pytest.raises(ValueError, match="bad"),
    ):
        run_project_setup_worker("proj-err", "err-proj")

    assert mock_pub.call_count == 2
    failed_call = mock_pub.call_args_list[1]
    assert failed_call[0][2] == "failed"
    assert "bad" in failed_call[1]["extra"]["error"]


def test_on_setup_failure_emits_event():
    """rq failure callback emits a failed event."""
    job = MagicMock()
    job.args = ("proj-fail", "fail-proj")
    job.id = "setup-job-err"
    with patch("agm.queue.publish_event") as mock_pub:
        on_setup_failure(job, None, ValueError, ValueError("boom"), None)

    mock_pub.assert_called_once()
    assert mock_pub.call_args[0][2] == "failed"
    assert "boom" in mock_pub.call_args[1]["extra"]["error"]
    assert mock_pub.call_args[1]["extra"]["job_id"] == "setup-job-err"
