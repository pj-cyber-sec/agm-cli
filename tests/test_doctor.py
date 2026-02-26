"""Tests for doctor health-check orchestration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from rq.exceptions import NoSuchJobError

from agm.cli import main
from agm.db import add_project, connect, create_plan_request, create_task, get_connection
from agm.doctor import (
    _check_backends,
    _check_redis_stream_health,
    _check_stale_registries,
    _check_worktree_cleanliness,
    _fix_stale_logs,
    _fix_stale_pids,
    run_doctor,
)


def _project_id(conn) -> str:
    row = conn.execute("SELECT id FROM projects WHERE name = 'proj'").fetchone()
    return row["id"]


def _mock_subprocess_completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _mock_doctor_redis(
    *,
    exists: bool = True,
    stream_length: int = 1,
    xrevrange_return: list[tuple[object, object]] | None = None,
) -> MagicMock:
    redis_mock = MagicMock()
    if xrevrange_return is None:
        if exists:
            xrevrange_return = [(f"{int(time.time() * 1000) - 100_000}-0", {"data": "{}"})]
        else:
            xrevrange_return = []
            stream_length = 0
    redis_mock.exists.return_value = 1 if exists else 0
    redis_mock.xlen.return_value = stream_length
    redis_mock.xrevrange.return_value = xrevrange_return
    return redis_mock


def _mock_started_registry(job_ids_by_queue: dict[str, list[str]]) -> object:
    def _factory(queue_name: str, connection=None):
        registry = MagicMock()
        registry.get_job_ids.return_value = job_ids_by_queue.get(queue_name, [])
        return registry

    return _factory


def test_run_doctor_all_pass(tmp_path):
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    worktree_dir = project_dir / ".agm" / "worktrees" / "task-wt"
    worktree_dir.mkdir(parents=True)

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task", description="desc")
    conn.execute("UPDATE tasks SET worktree = ? WHERE id = ?", (str(worktree_dir), task["id"]))
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {
        "agm:plans": {"queued": 0, "running": 0, "failed": 0},
        "agm:tasks": {"queued": 0, "running": 0, "failed": 0},
        "agm:exec": {"queued": 0, "running": 0, "failed": 0},
        "agm:merge": {"queued": 0, "running": 0, "failed": 0},
    }

    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    # Use empty temp log dir so real orphaned logs don't cause warnings
    empty_log_dir = tmp_path / "logs"
    empty_log_dir.mkdir()

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.LOG_DIR", empty_log_dir),
        patch(
            "subprocess.run",
            side_effect=_mock_subprocess_for_doctor,
        ),
    ):
        report = run_doctor(db_path)

    assert report["status"] == "pass"
    checks = {check["name"]: check for check in report["checks"]}
    assert set(checks.keys()) == {
        "redis",
        "redis_stream",
        "stale_registries",
        "sqlite",
        "backends",
        "stale_pids",
        "stale_entities",
        "worktrees",
        "worktree_cleanliness",
        "conflicts",
        "disk_usage",
        "log_files",
    }
    # backends check may warn if codex/claude not on PATH in test env
    non_backend_checks = {k: v for k, v in checks.items() if k != "backends"}
    assert all(check["status"] == "pass" for check in non_backend_checks.values())
    for check in report["checks"]:
        assert check["summary"]
        assert isinstance(check["findings"], list)


def test_check_redis_stream_healthy(tmp_path):
    db_path = tmp_path / "agm.db"
    with connect(db_path) as conn:
        project_dir = tmp_path / "project"
        add_project(conn, "proj", str(project_dir))

    stream_entry = (f"{int(time.time() * 1000) - 100}-0", {"data": "{}"})
    redis_mock = _mock_doctor_redis(xrevrange_return=[stream_entry])

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
    ):
        report = _check_redis_stream_health(db_path)

    assert report["name"] == "redis_stream"
    assert report["status"] == "pass"
    assert report["findings"]
    finding = report["findings"][0]
    assert finding["status"] == "pass"
    assert finding["details"]["latest_entry_id"] == stream_entry[0]


def test_check_redis_stream_missing(tmp_path):
    db_path = tmp_path / "agm.db"
    with connect(db_path) as conn:
        project_dir = tmp_path / "project"
        add_project(conn, "proj", str(project_dir))
        plan = create_plan_request(
            conn,
            project_id=_project_id(conn),
            prompt="test",
            caller="cli",
            backend="codex",
        )
        create_task(conn, plan_id=plan["id"], ordinal=0, title="Task", description="d")

    redis_mock = _mock_doctor_redis(exists=False, stream_length=0)

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
    ):
        report = _check_redis_stream_health(db_path)

    assert report["status"] == "warning"
    finding = report["findings"][0]
    assert finding["status"] == "warning"
    assert finding["details"]["active_plan_count"] == 1
    assert finding["details"]["active_task_count"] == 1


def test_check_redis_stream_stale_with_active_entities(tmp_path):
    db_path = tmp_path / "agm.db"
    with connect(db_path) as conn:
        project_dir = tmp_path / "project"
        add_project(conn, "proj", str(project_dir))
        plan = create_plan_request(
            conn,
            project_id=_project_id(conn),
            prompt="test",
            caller="cli",
            backend="codex",
        )
        create_task(conn, plan_id=plan["id"], ordinal=0, title="Task", description="d")

    stream_entry = (f"{int(time.time() * 1000) - 600_000}-0", {"data": "{}"})
    redis_mock = _mock_doctor_redis(
        stream_length=2,
        xrevrange_return=[stream_entry],
    )

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
    ):
        report = _check_redis_stream_health(db_path)

    assert report["status"] == "warning"
    finding = report["findings"][0]
    assert finding["status"] == "warning"
    assert finding["details"]["age_seconds"] >= 300
    assert finding["details"]["latest_entry_id"] == stream_entry[0]
    assert finding["details"]["stream_length"] == 2
    assert finding["details"]["active_plan_count"] == 1
    assert finding["details"]["active_task_count"] == 1


def test_check_redis_stream_healthy_when_db_counts_fail(tmp_path):
    db_path = tmp_path / "agm.db"
    stream_entry = (f"{int(time.time() * 1000) - 100}-0", {"data": "{}"})
    redis_mock = _mock_doctor_redis(xrevrange_return=[stream_entry])

    with (
        patch(
            "agm.doctor._count_active_plan_and_task_records",
            side_effect=RuntimeError("db unavailable"),
        ),
        patch("agm.doctor.get_redis", return_value=redis_mock),
    ):
        report = _check_redis_stream_health(db_path)

    assert report["status"] == "pass"
    finding = report["findings"][0]
    assert finding["details"]["active_plan_count"] == 0
    assert finding["details"]["active_task_count"] == 0


def test_check_stale_registries_pass(tmp_path):
    redis_mock = _mock_doctor_redis()
    queue_name = "agm:plans"
    live_job = MagicMock()
    live_job.meta = {"worker_pid": 424242}
    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch(
            "agm.doctor.StartedJobRegistry",
            side_effect=_mock_started_registry({queue_name: ["job-alive"]}),
        ),
        patch("agm.doctor.Job.fetch", return_value=live_job),
        patch("agm.doctor._pid_is_alive", return_value=True),
    ):
        report = _check_stale_registries()

    assert report["name"] == "stale_registries"
    assert report["status"] == "pass"
    assert report["findings"][0]["status"] == "pass"
    assert report["findings"][0]["details"]["checked"] == 1


def test_check_stale_registries_warns_on_ghost_registry_entry(tmp_path):
    redis_mock = _mock_doctor_redis()
    queue_name = "agm:plans"
    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch(
            "agm.doctor.StartedJobRegistry",
            side_effect=_mock_started_registry({queue_name: ["ghost-job-id"]}),
        ),
        patch("agm.doctor.Job.fetch", side_effect=NoSuchJobError("missing")),
    ):
        report = _check_stale_registries()

    assert report["name"] == "stale_registries"
    assert report["status"] == "warning"
    finding = report["findings"][0]
    assert finding["status"] == "warning"
    assert finding["details"]["kind"] == "ghost_registry_entry"
    assert finding["details"]["reason"] == "job missing in queue registry"
    assert finding["details"]["queue"] == queue_name
    assert finding["details"]["job_id"] == "ghost-job-id"


def test_run_doctor_degraded_with_fail_and_warnings(tmp_path):
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    worktree_root = project_dir / ".agm" / "worktrees"
    worktree_root.mkdir(parents=True)

    alive_worktree = worktree_root / "alive"
    alive_worktree.mkdir()
    orphan_worktree = worktree_root / "orphan"
    orphan_worktree.mkdir()
    missing_worktree = worktree_root / "missing"

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )

    stale_task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Stale", description="d")
    perm_task = create_task(conn, plan_id=plan["id"], ordinal=1, title="Perm", description="d")

    conn.execute(
        "UPDATE tasks SET status = 'running', pid = ?, worktree = ? WHERE id = ?",
        (424242, str(missing_worktree), stale_task["id"]),
    )
    conn.execute(
        "UPDATE tasks SET status = 'running', pid = ?, worktree = ? WHERE id = ?",
        (525252, str(alive_worktree), perm_task["id"]),
    )
    conn.commit()
    conn.close()

    def fake_kill(pid: int, sig: int):
        assert sig == 0
        if pid == 424242:
            raise ProcessLookupError
        if pid == 525252:
            raise PermissionError
        return None

    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", side_effect=ConnectionError("redis down")),
        patch("agm.doctor.os.kill", side_effect=fake_kill),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    assert report["status"] == "fail"
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["redis"]["status"] == "fail"
    assert checks["sqlite"]["status"] == "pass"

    stale_check = checks["stale_pids"]
    assert stale_check["status"] == "warning"
    stale_pids = [
        finding.get("details", {}).get("pid")
        for finding in stale_check["findings"]
        if finding["status"] == "warning"
    ]
    assert 424242 in stale_pids
    assert 525252 not in stale_pids

    worktree_check = checks["worktrees"]
    assert worktree_check["status"] == "warning"
    kinds = [
        finding.get("details", {}).get("kind")
        for finding in worktree_check["findings"]
        if finding["status"] == "warning"
    ]
    assert "db_missing_worktree" in kinds
    assert "filesystem_orphan" in kinds


def test_fix_stale_pids(tmp_path):
    """--fix should mark stale-PID tasks as failed."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with connect(db_path) as conn:
        add_project(conn, "proj", str(project_dir))
        plan = create_plan_request(
            conn,
            project_id=_project_id(conn),
            prompt="test",
            caller="cli",
            backend="codex",
        )
        task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Stale", description="d")
        conn.execute(
            "UPDATE tasks SET status = 'running', pid = 999999 WHERE id = ?",
            (task["id"],),
        )
        conn.commit()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    # Use empty temp log dir so real orphaned logs don't cause warnings
    empty_log_dir = tmp_path / "logs"
    empty_log_dir.mkdir()

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.os.kill", side_effect=ProcessLookupError),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.LOG_DIR", empty_log_dir),
    ):
        report = run_doctor(db_path, fix=True)

    assert report["status"] == "pass"
    assert "stale_pids" in report.get("fix_actions", {})
    action = report["fix_actions"]["stale_pids"]
    assert action["attempted"] == 1
    assert action["fixed"] == 1

    # Verify task is now failed
    with connect(db_path) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    assert row["status"] == "failed"


def test_fix_stale_pids_reruns_stale_entities(tmp_path):
    """--fix should rerun stale-entity checks after stale-PID fixes."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with connect(db_path) as conn:
        add_project(conn, "proj", str(project_dir))
        plan = create_plan_request(
            conn,
            project_id=_project_id(conn),
            prompt="test",
            caller="cli",
            backend="codex",
        )
        task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Stale", description="d")
        conn.execute(
            "UPDATE tasks SET status = 'running', pid = 999999 WHERE id = ?",
            (task["id"],),
        )
        conn.commit()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}
    empty_log_dir = tmp_path / "logs"
    empty_log_dir.mkdir()

    stale_entities_checks = [
        {
            "name": "stale_entities",
            "status": "warning",
            "summary": "1 stale entity before remediation",
            "findings": [
                {
                    "status": "warning",
                    "message": "entity stale before fix",
                    "details": {"entity_type": "task", "id": task["id"]},
                }
            ],
        },
        {
            "name": "stale_entities",
            "status": "pass",
            "summary": "No stale entities after stale-PID remediation",
            "findings": [
                {
                    "status": "pass",
                    "message": "no stale entities",
                    "details": {"checked": 1},
                }
            ],
        },
    ]

    with (
        patch(
            "agm.doctor._check_stale_entities", side_effect=stale_entities_checks
        ) as check_entities,
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.os.kill", side_effect=ProcessLookupError),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.LOG_DIR", empty_log_dir),
    ):
        report = run_doctor(db_path, fix=True)

    checks = {check["name"]: check for check in report["checks"]}

    assert report["fix_actions"]["stale_pids"]["fixed"] == 1
    stale_entities = checks["stale_entities"]
    assert stale_entities["status"] == "pass"
    assert stale_entities["summary"] == stale_entities_checks[1]["summary"]

    # Stale entity check should be run twice: initial run + post-fix rerun.
    assert check_entities.call_count == 2


def test_stale_entities_warning_uses_status_history_timestamp(tmp_path):
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task", description="desc")

    # Keep entity timestamps fresh so staleness must come from status_history age.
    conn.execute(
        "UPDATE plans SET status = 'running', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ?",
        (plan["id"],),
    )
    conn.execute(
        "UPDATE tasks SET status = 'running', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = ?",
        (task["id"],),
    )
    conn.execute(
        "INSERT INTO status_history"
        " (entity_type, entity_id, old_status, new_status, actor, created_at) "
        "VALUES ('plan', ?, 'pending', 'running', 'test', "
        "strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-2 hours'))",
        (plan["id"],),
    )
    conn.execute(
        "INSERT INTO status_history"
        " (entity_type, entity_id, old_status, new_status, actor, created_at) "
        "VALUES ('task', ?, 'pending', 'running', 'test', "
        "strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-91 minutes'))",
        (task["id"],),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    stale_check = {check["name"]: check for check in report["checks"]}["stale_entities"]
    assert stale_check["status"] == "warning"
    warnings = [f for f in stale_check["findings"] if f["status"] == "warning"]
    assert len(warnings) == 2
    warning_by_id = {f["details"]["id"]: f for f in warnings}

    plan_warning = warning_by_id[plan["id"]]["details"]
    assert plan_warning["entity_type"] == "plan"
    assert plan_warning["current_status"] == "running"
    assert plan_warning["timestamp_source"] == "status_history"
    assert plan_warning["stuck_seconds"] > 3600

    task_warning = warning_by_id[task["id"]]["details"]
    assert task_warning["entity_type"] == "task"
    assert task_warning["current_status"] == "running"
    assert task_warning["timestamp_source"] == "status_history"
    assert task_warning["stuck_seconds"] > 3600


def test_stale_entities_warning_falls_back_without_status_history(tmp_path):
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Task", description="desc")

    conn.execute(
        "UPDATE plans SET status = 'running', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', "
        "'-2 hours') WHERE id = ?",
        (plan["id"],),
    )
    conn.execute(
        "UPDATE tasks SET status = 'review', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', "
        "'-75 minutes') WHERE id = ?",
        (task["id"],),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    stale_check = {check["name"]: check for check in report["checks"]}["stale_entities"]
    assert stale_check["status"] == "warning"
    warnings = [f for f in stale_check["findings"] if f["status"] == "warning"]
    assert len(warnings) == 2
    for warning in warnings:
        details = warning["details"]
        assert details["timestamp_source"] == "entity_timestamp"
        assert details["stuck_seconds"] > 3600
        assert details["current_status"] in {"running", "review"}


def _mock_subprocess_for_doctor(cmd, **kwargs):
    """Subprocess mock for doctor tests: allow codex, git status, and git worktree prune."""
    if cmd == ["codex", "--version"]:
        return _mock_subprocess_completed(stdout="codex 1.0.0")
    if cmd == ["git", "status", "--porcelain"]:
        return _mock_subprocess_completed()
    if cmd == ["git", "worktree", "prune"]:
        return _mock_subprocess_completed()
    raise FileNotFoundError()


def test_fix_orphaned_worktrees(tmp_path):
    """--fix should remove orphaned worktree dirs and clear dangling refs."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    wt_root = project_dir / ".agm" / "worktrees"
    orphan = wt_root / "orphan-wt"
    orphan.mkdir(parents=True)
    missing_wt = wt_root / "gone"

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Dangling", description="d")
    conn.execute(
        "UPDATE tasks SET worktree = ?, branch = 'old-branch' WHERE id = ?",
        (str(missing_wt), task["id"]),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    # Use empty temp log dir so real orphaned logs don't cause warnings
    empty_log_dir = tmp_path / "logs"
    empty_log_dir.mkdir()

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.LOG_DIR", empty_log_dir),
        patch(
            "subprocess.run",
            side_effect=_mock_subprocess_for_doctor,
        ),
    ):
        report = run_doctor(db_path, fix=True)

    assert report["status"] == "pass"
    action = report["fix_actions"]["orphaned_worktrees"]
    assert action["attempted"] == 2
    assert action["fixed"] == 2

    assert not orphan.exists()
    conn = get_connection(db_path)
    row = conn.execute("SELECT worktree, branch FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    conn.close()
    assert row["worktree"] is None
    assert row["branch"] is None


def test_doctor_cli_pass(tmp_path):
    """agm doctor should return JSON report with pass status."""
    import json

    db_path = tmp_path / "agm.db"
    conn = get_connection(db_path)
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    # Use empty temp log dir so real orphaned logs don't cause warnings
    empty_log_dir = tmp_path / "logs"
    empty_log_dir.mkdir()

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor._resolve_db_path", return_value=db_path),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.LOG_DIR", empty_log_dir),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        report = json.loads(result.output)
        assert report["status"] == "pass"
        assert isinstance(report["checks"], list)
        assert all("name" in c and "status" in c for c in report["checks"])


def test_doctor_cli_fix_flag(tmp_path):
    """agm doctor --fix should show fix results."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Stale", description="d")
    conn.execute(
        "UPDATE tasks SET status = 'running', pid = 888888 WHERE id = ?",
        (task["id"],),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor._resolve_db_path", return_value=db_path),
        patch("agm.doctor.os.kill", side_effect=ProcessLookupError),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["doctor", "--fix"])
        assert result.exit_code == 0
        import json

        report = json.loads(result.output)
        assert "fixes" in report or "checks" in report


def test_doctor_conflicts_check_no_clash(tmp_path):
    """Doctor should pass gracefully when clash is not installed."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    conn.close()

    clash_result = {
        "available": False,
        "worktrees": [],
        "conflicts": [],
        "error": "clash binary not found",
    }
    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    checks = {c["name"]: c for c in report["checks"]}
    assert checks["conflicts"]["status"] == "pass"
    assert "not installed" in checks["conflicts"]["summary"]


def test_doctor_conflicts_check_with_conflicts(tmp_path):
    """Doctor should warn when clash detects worktree conflicts."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    conn.close()

    clash_result = {
        "available": True,
        "worktrees": [
            {"id": "main", "path": "/p", "branch": "main", "status": "clean"},
            {"id": "wt1", "path": "/p/wt1", "branch": "feat/a", "status": "dirty"},
        ],
        "conflicts": [
            {
                "wt1_id": "main",
                "wt2_id": "wt1",
                "conflicting_files": ["src/db.py", "src/jobs.py"],
                "error": None,
            }
        ],
        "error": None,
    }
    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    checks = {c["name"]: c for c in report["checks"]}
    assert checks["conflicts"]["status"] == "warning"
    assert "1 worktree conflict pair" in checks["conflicts"]["summary"]
    finding = checks["conflicts"]["findings"][0]
    assert finding["details"]["files"] == ["src/db.py", "src/jobs.py"]


def test_doctor_conflicts_check_clean(tmp_path):
    """Doctor should pass when clash finds no conflicts."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    conn.close()

    clash_result = {
        "available": True,
        "worktrees": [
            {"id": "main", "path": "/p", "branch": "main", "status": "clean"},
            {"id": "wt1", "path": "/p/wt1", "branch": "feat/a", "status": "clean"},
        ],
        "conflicts": [
            {
                "wt1_id": "main",
                "wt2_id": "wt1",
                "conflicting_files": [],
                "error": None,
            }
        ],
        "error": None,
    }
    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    checks = {c["name"]: c for c in report["checks"]}
    assert checks["conflicts"]["status"] == "pass"
    assert "No worktree conflicts" in checks["conflicts"]["summary"]


def test_fix_terminal_task_worktrees(tmp_path):
    """--fix should clean up worktrees belonging to terminal (failed/completed/cancelled) tasks."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    wt_root = project_dir / ".agm" / "worktrees"

    # Create two worktree dirs for terminal tasks
    failed_wt = wt_root / "failed-task-wt"
    failed_wt.mkdir(parents=True)
    completed_wt = wt_root / "completed-task-wt"
    completed_wt.mkdir()

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    failed_task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Failed", description="d")
    completed_task = create_task(
        conn, plan_id=plan["id"], ordinal=1, title="Completed", description="d"
    )
    conn.execute(
        "UPDATE tasks SET status = 'failed', worktree = ?, branch = 'feat/failed' WHERE id = ?",
        (str(failed_wt), failed_task["id"]),
    )
    conn.execute(
        "UPDATE tasks SET status = 'completed', worktree = ?, branch = 'feat/done' WHERE id = ?",
        (str(completed_wt), completed_task["id"]),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
        patch("agm.doctor.remove_worktree"),
    ):
        # First check detection
        report = run_doctor(db_path)
        checks = {c["name"]: c for c in report["checks"]}
        wt_check = checks["worktrees"]
        assert wt_check["status"] == "warning"
        kinds = [
            f.get("details", {}).get("kind")
            for f in wt_check["findings"]
            if f["status"] == "warning"
        ]
        assert kinds.count("terminal_task_worktree") == 2

        # Now fix
        report = run_doctor(db_path, fix=True)

    action = report["fix_actions"]["orphaned_worktrees"]
    assert action["attempted"] == 2
    assert action["fixed"] == 2
    assert action["failed"] == 0

    # Verify DB refs cleared
    conn = get_connection(db_path)
    for tid in [failed_task["id"], completed_task["id"]]:
        row = conn.execute("SELECT worktree, branch FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["worktree"] is None
        assert row["branch"] is None
    conn.close()


def test_terminal_task_worktree_detection_only_terminal(tmp_path):
    """Non-terminal tasks with worktrees should NOT be flagged as terminal_task_worktree."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    wt_root = project_dir / ".agm" / "worktrees"
    running_wt = wt_root / "running-wt"
    running_wt.mkdir(parents=True)

    conn = get_connection(db_path)
    add_project(conn, "proj", str(project_dir))
    plan = create_plan_request(
        conn,
        project_id=_project_id(conn),
        prompt="test",
        caller="cli",
        backend="codex",
    )
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Running", description="d")
    conn.execute(
        "UPDATE tasks SET status = 'running', worktree = ?, branch = 'feat/wip' WHERE id = ?",
        (str(running_wt), task["id"]),
    )
    conn.commit()
    conn.close()

    redis_mock = _mock_doctor_redis()
    queue_counts = {"agm:plans": {"queued": 0, "running": 0, "failed": 0}}
    clash_result = {"available": True, "worktrees": [], "conflicts": [], "error": None}

    with (
        patch("agm.doctor.get_redis", return_value=redis_mock),
        patch("agm.doctor.get_queue_counts", return_value=queue_counts),
        patch("agm.doctor.detect_worktree_conflicts", return_value=clash_result),
    ):
        report = run_doctor(db_path)

    checks = {c["name"]: c for c in report["checks"]}
    wt_check = checks["worktrees"]
    # Running task with existing worktree — should be pass (consistent state)
    assert wt_check["status"] == "pass"


def test_check_worktree_cleanliness_clean(tmp_path):
    """worktree_cleanliness passes for clean worktrees."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    worktree_dir = project_dir / ".agm" / "worktrees" / "clean"
    worktree_dir.mkdir(parents=True)

    with connect(db_path) as conn:
        add_project(conn, "proj", str(project_dir))

    with patch(
        "subprocess.run",
        side_effect=lambda cmd, **kwargs: (
            _mock_subprocess_completed()
            if cmd == ["git", "status", "--porcelain"]
            else (_ for _ in ()).throw(FileNotFoundError())
        ),
    ):
        check = _check_worktree_cleanliness(db_path)

    assert check["name"] == "worktree_cleanliness"
    assert check["status"] == "pass"
    assert "all are clean" in check["summary"]


def test_check_worktree_cleanliness_dirty(tmp_path):
    """worktree_cleanliness emits warning for dirty worktrees."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    worktree_dir = project_dir / ".agm" / "worktrees" / "dirty"
    worktree_dir.mkdir(parents=True)

    with connect(db_path) as conn:
        add_project(conn, "proj", str(project_dir))

    with patch(
        "subprocess.run",
        side_effect=lambda cmd, **kwargs: (
            _mock_subprocess_completed(stdout=" M modified.txt\n?? new.txt")
            if cmd == ["git", "status", "--porcelain"]
            else (_ for _ in ()).throw(FileNotFoundError())
        ),
    ):
        check = _check_worktree_cleanliness(db_path)

    assert check["name"] == "worktree_cleanliness"
    assert check["status"] == "warning"
    assert "dirty worktree" in check["summary"]
    assert len(check["findings"]) == 1
    finding = check["findings"][0]
    assert finding["status"] == "warning"
    assert finding["details"]["kind"] == "dirty_worktree"


def test_check_worktree_cleanliness_no_worktrees(tmp_path):
    """worktree_cleanliness passes when no worktree directories exist."""
    db_path = tmp_path / "agm.db"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with connect(db_path) as conn:
        add_project(conn, "proj", str(project_dir))

    check = _check_worktree_cleanliness(db_path)

    assert check["name"] == "worktree_cleanliness"
    assert check["status"] == "pass"
    assert "No configured worktree" in check["summary"]
    assert check["findings"] == []


def test_backends_check_none_available():
    """Backends check fails when codex is not on PATH."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        report = _check_backends()
    assert report["name"] == "backends"
    assert report["status"] == "fail"
    assert "No backends" in report["summary"]
    assert len(report["findings"]) == 1


def test_backends_check_one_available():
    """Backends check passes when at least one backend is available."""

    def mock_run(cmd, **kwargs):
        if cmd[0] == "codex":
            result = MagicMock()
            result.returncode = 0
            result.stdout = "codex 1.0.0"
            result.stderr = ""
            return result
        raise FileNotFoundError

    with patch("subprocess.run", side_effect=mock_run):
        report = _check_backends()
    assert report["status"] == "pass"
    assert "1 backend(s)" in report["summary"]


# ---------------------------------------------------------------------------
# _fix_stale_logs tests
# ---------------------------------------------------------------------------


def test_fix_stale_logs_deletes_stale_files(tmp_path):
    """Stale log files are deleted."""
    log_file = tmp_path / "plan-abc123.log"
    log_file.write_text("old log data")

    check = {
        "name": "log_files",
        "status": "warning",
        "summary": "1 stale log file(s)",
        "findings": [
            {
                "status": "warning",
                "message": "stale log",
                "details": {"kind": "stale_log", "path": str(log_file)},
            }
        ],
    }

    action = _fix_stale_logs(check)
    assert action["attempted"] == 1
    assert action["fixed"] == 1
    assert action["failed"] == 0
    assert not log_file.exists()


def test_fix_stale_logs_skips_non_stale_findings():
    """Non-stale findings are skipped."""
    check = {
        "name": "log_files",
        "status": "pass",
        "summary": "ok",
        "findings": [
            {
                "status": "info",
                "message": "normal log",
                "details": {"kind": "active_log"},
            }
        ],
    }

    action = _fix_stale_logs(check)
    assert action["attempted"] == 0


def test_fix_stale_logs_handles_missing_path():
    """Finding with empty path is skipped."""
    check = {
        "name": "log_files",
        "status": "warning",
        "summary": "1 stale",
        "findings": [
            {
                "status": "warning",
                "message": "stale log",
                "details": {"kind": "stale_log", "path": ""},
            }
        ],
    }

    action = _fix_stale_logs(check)
    assert action["attempted"] == 0


def test_fix_stale_logs_counts_unlink_failure(tmp_path):
    """Failed unlink increments failed counter."""
    check = {
        "name": "log_files",
        "status": "warning",
        "summary": "1 stale",
        "findings": [
            {
                "status": "warning",
                "message": "stale log",
                "details": {"kind": "stale_log", "path": str(tmp_path / "no-exist.log")},
            }
        ],
    }

    # missing_ok=True means this won't fail for missing files.
    # Force a real error by making path a directory
    dir_path = tmp_path / "dir.log"
    dir_path.mkdir()
    (dir_path / "child").write_text("x")
    check["findings"][0]["details"]["path"] = str(dir_path)

    action = _fix_stale_logs(check)
    assert action["attempted"] == 1
    assert action["failed"] == 1
    assert len(action["failures"]) == 1


# ---------------------------------------------------------------------------
# _fix_stale_pids unknown entity tests
# ---------------------------------------------------------------------------


def test_fix_stale_pids_unknown_entity(tmp_path):
    """Unknown entity type in finding results in failure."""
    db_path = tmp_path / "agm.db"
    conn = get_connection(db_path)
    conn.close()

    check = {
        "name": "stale_pids",
        "status": "warning",
        "summary": "1 stale",
        "findings": [
            {
                "status": "warning",
                "message": "stale",
                "details": {
                    "entity": "widget",
                    "entity_id": "xyz",
                    "old_status": "running",
                    "pid": 99999,
                },
            }
        ],
    }

    action = _fix_stale_pids(db_path, check)
    assert action["failed"] == 1
    assert "unknown entity" in action["failures"][0]["reason"]
