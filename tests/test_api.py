"""Tests for agm.api — JSON dispatch layer."""

import io
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from agm.api import dispatch, main
from agm.db import (
    add_channel_message,
    add_plan_log,
    add_plan_question,
    add_task_block,
    add_task_log,
    add_trace_event,
    create_plan_request,
    create_session,
    create_task,
    get_project,
    set_plan_session_id,
    update_plan_request_status,
    update_plan_task_creation_status,
    update_session_status,
    update_task_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_id(conn: sqlite3.Connection) -> str:
    p = get_project(conn, "testproj")
    assert p is not None
    return p["id"]


def _create_plan(conn: sqlite3.Connection, **kwargs) -> dict:
    defaults = {
        "project_id": _project_id(conn),
        "prompt": "Test prompt",
        "actor": "test",
        "caller": "cli",
        "backend": "codex",
    }
    defaults.update(kwargs)
    return create_plan_request(conn, **defaults)


def _create_task(conn: sqlite3.Connection, plan_id: str, **kwargs) -> dict:
    defaults = {
        "plan_id": plan_id,
        "ordinal": 1,
        "title": "Test task",
        "description": "Do the thing",
    }
    defaults.update(kwargs)
    return create_task(conn, **defaults)


def _fake_connect(conn: sqlite3.Connection):
    """Return a context manager that yields the existing test connection."""
    import contextlib

    @contextlib.contextmanager
    def _connect(**_kwargs):
        yield conn

    return _connect


def _patch(monkeypatch, db_conn):
    monkeypatch.setattr("agm.api.connect", _fake_connect(db_conn))


# ---------------------------------------------------------------------------
# Dispatch protocol
# ---------------------------------------------------------------------------


class TestDispatchProtocol:
    def test_rejects_bad_methods(self):
        for bad in [{}, {"method": 42}, {"method": "nonexistent.thing"}]:
            result = dispatch(bad)
            assert result["ok"] is False
            assert result["code"] == "INVALID_METHOD"

    def test_missing_params_defaults_to_empty(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "project.list"})
        assert result["ok"] is True

    def test_missing_required_param_returns_invalid(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "task.show", "params": {}})
        assert result["ok"] is False
        assert result["code"] == "INVALID_PARAMS"


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------


class TestTaskHandlers:
    def test_show_and_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"], title="Build widget")

        result = dispatch({"method": "task.show", "params": {"id": task["id"]}})
        assert result["ok"] is True
        assert result["data"]["title"] == "Build widget"
        assert result["data"]["plan_id"] == plan["id"]

        # Not found
        result = dispatch({"method": "task.show", "params": {"id": "bogus"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"

    def test_list_with_enrichment_and_filtering(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        _create_task(db_conn, plan["id"], ordinal=2, title="Task B")

        # Both tasks returned with enrichment
        result = dispatch({"method": "task.list", "params": {"plan_id": plan["id"]}})
        assert result["ok"] is True
        assert len(result["data"]) == 2
        assert "project_name" in result["data"][0]

        # Status filter
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        result = dispatch(
            {"method": "task.list", "params": {"plan_id": plan["id"], "status": "running"}}
        )
        assert len(result["data"]) == 1
        assert result["data"][0]["status"] == "running"

    def test_list_hides_terminal_by_default(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        update_task_status(db_conn, task["id"], "completed")

        result = dispatch({"method": "task.list", "params": {"plan_id": plan["id"]}})
        assert len(result["data"]) == 0

        result = dispatch(
            {"method": "task.list", "params": {"plan_id": plan["id"], "show_all": True}}
        )
        assert len(result["data"]) == 1

    def test_logs_with_level_filter(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        add_task_log(db_conn, task_id=task["id"], level="INFO", message="Started")
        add_task_log(db_conn, task_id=task["id"], level="ERROR", message="Broke")

        # All logs
        result = dispatch({"method": "task.logs", "params": {"id": task["id"]}})
        assert result["data"]["count"] == 2

        # Filtered
        result = dispatch({"method": "task.logs", "params": {"id": task["id"], "level": "ERROR"}})
        assert result["data"]["count"] == 1
        assert result["data"]["logs"][0]["message"] == "Broke"

    def test_timeline(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready", record_history=True)
        update_task_status(db_conn, task["id"], "running", record_history=True)

        result = dispatch({"method": "task.timeline", "params": {"id": task["id"]}})
        assert result["ok"] is True
        assert result["data"]["task_id"] == task["id"]
        assert len(result["data"]["timeline"]) >= 1
        assert "new_status" in result["data"]["timeline"][0]

    def test_blocks(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task_a = _create_task(db_conn, plan["id"], title="A")
        task_b = _create_task(db_conn, plan["id"], ordinal=2, title="B")
        add_task_block(db_conn, task_id=task_b["id"], blocked_by_task_id=task_a["id"], reason="Dep")

        result = dispatch({"method": "task.blocks", "params": {"id": task_b["id"]}})
        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["blocks"][0]["blocked_by_task_id"] == task_a["id"]

    def test_failures(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        add_task_log(db_conn, task_id=task["id"], level="ERROR", message="Segfault")
        update_task_status(db_conn, task["id"], "failed")

        result = dispatch(
            {"method": "task.failures", "params": {"project_id": _project_id(db_conn)}}
        )
        assert result["ok"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["source"] == "task"
        assert "Segfault" in result["data"][0]["error"]


# ---------------------------------------------------------------------------
# Plan handlers
# ---------------------------------------------------------------------------


class TestPlanHandlers:
    def test_show_and_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn, prompt="Build a dashboard")

        result = dispatch({"method": "plan.show", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        assert result["data"]["prompt"] == "Build a dashboard"

        result = dispatch({"method": "plan.show", "params": {"id": "nope"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"

    def test_show_with_tasks(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        _create_task(db_conn, plan["id"])

        result = dispatch({"method": "plan.show", "params": {"id": plan["id"], "show_tasks": True}})
        assert result["ok"] is True
        assert len(result["data"]["tasks"]) == 1

    def test_list_with_enrichment(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        _create_plan(db_conn, prompt="Plan A")
        _create_plan(db_conn, prompt="Plan B")

        result = dispatch({"method": "plan.list", "params": {"project_id": _project_id(db_conn)}})
        assert result["ok"] is True
        assert len(result["data"]) == 2
        assert "error" in result["data"][0]
        assert "active_runtime_seconds" in result["data"][0]

    def test_logs(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        add_plan_log(db_conn, plan_id=plan["id"], level="INFO", message="Planning started")

        result = dispatch({"method": "plan.logs", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["logs"][0]["message"] == "Planning started"

    def test_questions(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        add_plan_question(db_conn, plan_id=plan["id"], question="What approach?", header="Approach")

        result = dispatch({"method": "plan.questions", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["questions"][0]["status"] == "pending"

    def test_history(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)

        result = dispatch({"method": "plan.history", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        chain = result["data"]["chain"]
        assert chain[0]["id"] == plan["id"]
        assert chain[0]["is_target"] is True

    def test_stats(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        _create_task(db_conn, plan["id"], title="T1")
        _create_task(db_conn, plan["id"], ordinal=2, title="T2")

        result = dispatch({"method": "plan.stats", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        assert result["data"]["total_tasks"] == 2
        assert "tokens" in result["data"]

    def test_failures(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        update_plan_request_status(db_conn, plan["id"], "running")
        add_plan_log(db_conn, plan_id=plan["id"], level="ERROR", message="Model refused")
        update_plan_request_status(db_conn, plan["id"], "failed")

        result = dispatch(
            {"method": "plan.failures", "params": {"project_id": _project_id(db_conn)}}
        )
        assert result["ok"] is True
        assert len(result["data"]) == 1
        assert "refused" in result["data"][0]["error"]


# ---------------------------------------------------------------------------
# Project handlers
# ---------------------------------------------------------------------------


class TestProjectHandlers:
    def test_show_and_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        db_conn.execute(
            "UPDATE projects SET app_server_ask_for_approval = ? WHERE name = ?",
            (json.dumps("on-request"), "testproj"),
        )
        db_conn.commit()

        result = dispatch({"method": "project.show", "params": {"id": "testproj"}})
        assert result["ok"] is True
        assert result["data"]["name"] == "testproj"
        assert "model_config" in result["data"]
        assert result["data"]["app_server_approval_policy"]["execCommandApproval"] == "approved"
        assert result["data"]["app_server_ask_for_approval"] == "on-request"

        result = dispatch({"method": "project.show", "params": {"id": "ghost"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"

    def test_list(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        db_conn.execute(
            "UPDATE projects SET app_server_ask_for_approval = ? WHERE name = ?",
            (
                json.dumps(
                    {
                        "reject": {
                            "mcp_elicitations": True,
                            "rules": True,
                            "sandbox_approval": False,
                        }
                    }
                ),
                "testproj",
            ),
        )
        db_conn.commit()
        result = dispatch({"method": "project.list"})
        assert result["ok"] is True
        names = [p["name"] for p in result["data"]]
        assert "testproj" in names
        testproj = next(p for p in result["data"] if p["name"] == "testproj")
        assert testproj["app_server_approval_policy"]["execCommandApproval"] == "approved"
        assert testproj["app_server_ask_for_approval"] == {
            "reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}
        }

    def test_stats(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        _create_plan(db_conn)

        result = dispatch({"method": "project.stats", "params": {"id": "testproj"}})
        assert result["ok"] is True
        assert result["data"]["total_plans"] >= 1
        assert "tokens" in result["data"]

    def test_setup_status(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        with (
            patch("agm.queue.inspect_queue_jobs", return_value=[]),
            patch("agm.queue.get_job", return_value=None),
        ):
            result = dispatch({"method": "project.setup_status", "params": {"id": "testproj"}})
        assert result["ok"] is True
        assert result["data"]["project_name"] == "testproj"
        assert result["data"]["setup_job_id"].startswith("setup-")


# ---------------------------------------------------------------------------
# Watch handlers
# ---------------------------------------------------------------------------


class TestPlanWatch:
    def test_plan_watch_snapshot(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn, prompt="Build dashboard")
        _create_task(db_conn, plan["id"], title="Setup")
        t2 = _create_task(db_conn, plan["id"], ordinal=2, title="Build")
        update_task_status(db_conn, t2["id"], "ready")
        update_task_status(db_conn, t2["id"], "running")

        result = dispatch({"method": "plan.watch", "params": {"id": plan["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["schema"] == "plan_watch_snapshot_v1"
        assert data["scope"]["type"] == "plan"
        assert data["scope"]["plan_id"] == plan["id"]
        assert "session_id" in data["scope"]
        assert data["counts"]["tasks_total"] == 2
        assert "tokens" in data
        assert "phase" in data
        assert "phase_since" in data
        assert "blocking_reason" in data
        assert "terminal_state" in data

    def test_plan_watch_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "plan.watch", "params": {"id": "bogus"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"


class TestTaskWatch:
    def test_by_task_id(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"], title="Widget")

        result = dispatch({"method": "task.watch", "params": {"task_id": task["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["schema"] == "task_watch_snapshot_v1"
        assert data["scope"]["type"] == "task"
        assert data["scope"]["task_id"] == task["id"]
        assert "session_id" in data["scope"]
        assert data["counts"]["tasks_total"] == 1
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["title"] == "Widget"
        assert "phase" in data
        assert "phase_since" in data
        assert "blocking_reason" in data

    def test_by_plan_id(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn, prompt="Big plan")
        _create_task(db_conn, plan["id"], title="A")
        _create_task(db_conn, plan["id"], ordinal=2, title="B")

        result = dispatch({"method": "task.watch", "params": {"plan_id": plan["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["scope"]["type"] == "plan"
        assert data["counts"]["tasks_total"] == 2

    def test_by_project(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        _create_task(db_conn, plan["id"], title="T1")

        result = dispatch({"method": "task.watch", "params": {"project": "testproj"}})
        assert result["ok"] is True
        data = result["data"]
        assert data["scope"]["type"] == "project"
        assert data["scope"]["project_name"] == "testproj"

    def test_includes_thread_status_details(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"], title="T1")
        add_trace_event(
            db_conn,
            entity_type="task",
            entity_id=task["id"],
            stage="execution",
            turn_index=0,
            ordinal=1,
            event_type="threadStatus",
            status="changed",
            data={
                "thread_id": "thread-abc",
                "old_status": "idle",
                "new_status": "running",
            },
        )

        result = dispatch({"method": "task.watch", "params": {"task_id": task["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["thread_status_summary"]["running"] == 1
        assert data["tasks"][0]["thread_status"]["thread_id"] == "thread-abc"
        assert data["tasks"][0]["thread_status"]["old_status"] == "idle"
        assert data["tasks"][0]["thread_status"]["new_status"] == "running"

    def test_no_scope_rejected(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "task.watch", "params": {}})
        assert result["ok"] is False
        assert result["code"] == "INVALID_PARAMS"

    def test_multiple_scopes_rejected(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        result = dispatch(
            {
                "method": "task.watch",
                "params": {"task_id": task["id"], "plan_id": plan["id"]},
            }
        )
        assert result["ok"] is False
        assert result["code"] == "INVALID_PARAMS"

    def test_show_all_includes_terminal(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        update_task_status(db_conn, task["id"], "completed")

        # Without show_all: no visible tasks
        result = dispatch({"method": "task.watch", "params": {"plan_id": plan["id"]}})
        assert result["ok"] is True
        assert result["data"]["counts"]["tasks_visible"] == 0

        # With show_all: task visible
        result = dispatch(
            {
                "method": "task.watch",
                "params": {"plan_id": plan["id"], "show_all": True},
            }
        )
        assert result["ok"] is True
        assert result["data"]["counts"]["tasks_visible"] == 1


# ---------------------------------------------------------------------------
# Plan list filtering and project name resolution
# ---------------------------------------------------------------------------


class TestPlanListFiltering:
    def test_default_hides_terminal(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        update_plan_request_status(db_conn, plan["id"], "running")
        update_plan_request_status(db_conn, plan["id"], "finalized")

        # Default: only active statuses
        result = dispatch({"method": "plan.list", "params": {}})
        assert result["ok"] is True
        plan_ids = [p["id"] for p in result["data"]]
        assert plan["id"] not in plan_ids

    def test_show_all_includes_terminal(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        update_plan_request_status(db_conn, plan["id"], "running")
        update_plan_request_status(db_conn, plan["id"], "finalized")

        result = dispatch({"method": "plan.list", "params": {"show_all": True}})
        assert result["ok"] is True
        plan_ids = [p["id"] for p in result["data"]]
        assert plan["id"] in plan_ids


class TestProjectNameResolution:
    def test_plan_list_by_project_name(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        _create_plan(db_conn)

        result = dispatch(
            {
                "method": "plan.list",
                "params": {"project": "testproj", "show_all": True},
            }
        )
        assert result["ok"] is True
        assert len(result["data"]) >= 1

    def test_plan_failures_by_project_name(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        update_plan_request_status(db_conn, plan["id"], "running")
        add_plan_log(db_conn, plan_id=plan["id"], level="ERROR", message="Boom")
        update_plan_request_status(db_conn, plan["id"], "failed")

        result = dispatch(
            {
                "method": "plan.failures",
                "params": {"project": "testproj"},
            }
        )
        assert result["ok"] is True
        assert len(result["data"]) == 1

    def test_task_failures_by_project_name(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        add_task_log(db_conn, task_id=task["id"], level="ERROR", message="Crash")
        update_task_status(db_conn, task["id"], "failed")

        result = dispatch(
            {
                "method": "task.failures",
                "params": {"project": "testproj"},
            }
        )
        assert result["ok"] is True
        assert len(result["data"]) == 1

    def test_task_list_by_project_name(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        plan = _create_plan(db_conn)
        _create_task(db_conn, plan["id"])

        result = dispatch(
            {
                "method": "task.list",
                "params": {"project": "testproj"},
            }
        )
        assert result["ok"] is True
        assert len(result["data"]) >= 1

    def test_unknown_project_name_returns_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch(
            {
                "method": "plan.list",
                "params": {"project": "nonexistent"},
            }
        )
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Doctor fix param
# ---------------------------------------------------------------------------


class TestDoctorFix:
    def test_doctor_default_no_fix(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "doctor", "params": {}})
        assert result["ok"] is True
        assert "checks" in result["data"]

    def test_doctor_with_fix(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "doctor", "params": {"fix": True}})
        assert result["ok"] is True
        assert "checks" in result["data"]


# ---------------------------------------------------------------------------
# Status / infra handlers
# ---------------------------------------------------------------------------


class TestInfraHandlers:
    def test_help_status(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "help_status"})
        assert result["ok"] is True
        assert "lifecycles" in result["data"]

    def test_caller_list(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "caller.list"})
        assert result["ok"] is True
        assert "cli" in result["data"]["all"]

    def test_daemon_status_not_running(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        runtime_dir = Path(tempfile.mkdtemp())
        monkeypatch.setattr("agm.daemon.DEFAULT_PID_PATH", runtime_dir / "appserver.pid")
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", runtime_dir / "appserver.sock")
        monkeypatch.setattr("agm.daemon.DEFAULT_LOG_PATH", runtime_dir / "daemon.log")

        result = dispatch({"method": "daemon.status"})
        assert result["ok"] is True
        data = result["data"]
        assert data["running"] is False
        assert data["pid"] is None
        assert data["socket_exists"] is False

    def test_daemon_status_running(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        runtime_dir = Path(tempfile.mkdtemp())
        pid_path = runtime_dir / "appserver.pid"
        socket_path = runtime_dir / "appserver.sock"
        log_path = runtime_dir / "daemon.log"
        pid_path.write_text(str(os.getpid()))
        socket_path.touch()
        monkeypatch.setattr("agm.daemon.DEFAULT_PID_PATH", pid_path)
        monkeypatch.setattr("agm.daemon.DEFAULT_SOCKET_PATH", socket_path)
        monkeypatch.setattr("agm.daemon.DEFAULT_LOG_PATH", log_path)
        monkeypatch.setattr("agm.api.os.kill", lambda _pid, _sig: None)

        result = dispatch({"method": "daemon.status"})
        assert result["ok"] is True
        data = result["data"]
        assert data["running"] is True
        assert data["pid"] == os.getpid()
        assert data["socket_exists"] is True

    def test_daemon_threads_forwards_filters(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        monkeypatch.setattr("agm.api._daemon_pid", lambda: 12345)
        captured: dict[str, object] = {}

        class _FakeDaemonClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def request(self, method, params=None, timeout=120):
                captured["method"] = method
                captured["params"] = params
                captured["timeout"] = timeout
                return {"data": [{"id": "thread-1"}], "nextCursor": "cursor-2"}

        monkeypatch.setattr("agm.daemon_client.DaemonClient", _FakeDaemonClient)
        result = dispatch(
            {
                "method": "daemon.threads",
                "params": {
                    "search": "upgrade",
                    "limit": 10,
                    "cursor": "cursor-1",
                    "archived": True,
                    "sort_key": "updated_at",
                    "cwd": "/tmp/project",
                },
            }
        )

        assert result["ok"] is True
        data = result["data"]
        assert data["nextCursor"] == "cursor-2"
        assert data["data"][0]["id"] == "thread-1"
        assert "status_type" in data["data"][0]
        assert "active_flags" in data["data"][0]
        assert captured["method"] == "thread/list"
        assert captured["params"] == {
            "searchTerm": "upgrade",
            "limit": 10,
            "cursor": "cursor-1",
            "archived": True,
            "sortKey": "updated_at",
            "cwd": "/tmp/project",
        }

    def test_daemon_threads_not_running_returns_error(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        monkeypatch.setattr("agm.api._daemon_pid", lambda: None)

        result = dispatch({"method": "daemon.threads", "params": {}})
        assert result["ok"] is False
        assert result["code"] == "INTERNAL"
        assert "daemon is not running" in result["error"].lower()


# ---------------------------------------------------------------------------
# Session handlers
# ---------------------------------------------------------------------------


class TestSessionHandlers:
    def test_show_returns_session_with_linked_plans(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(
            db_conn, project_id=pid, trigger="plan_request", trigger_prompt="do X"
        )
        update_session_status(db_conn, session["id"], "active")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])

        result = dispatch({"method": "session.show", "params": {"id": session["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["session"]["id"] == session["id"]
        assert data["session"]["status"] == "active"
        assert len(data["plans"]) == 1
        assert data["plans"][0]["id"] == plan["id"]

    def test_show_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "session.show", "params": {"id": "nonexistent"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"

    def test_list_defaults_to_open_active(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        s1 = create_session(db_conn, project_id=pid, trigger="do")
        s2 = create_session(db_conn, project_id=pid, trigger="plan_request")
        update_session_status(db_conn, s2["id"], "active")

        result = dispatch({"method": "session.list"})
        assert result["ok"] is True
        ids = {s["id"] for s in result["data"]}
        assert s1["id"] in ids  # open
        assert s2["id"] in ids  # active

    def test_list_with_show_all(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        s = create_session(db_conn, project_id=pid, trigger="do")
        from agm.db import finish_session

        finish_session(db_conn, s["id"], "completed")

        # default excludes completed
        default_result = dispatch({"method": "session.list"})
        default_ids = {r["id"] for r in default_result["data"]}
        assert s["id"] not in default_ids

        # show_all includes completed
        all_result = dispatch({"method": "session.list", "params": {"show_all": True}})
        all_ids = {r["id"] for r in all_result["data"]}
        assert s["id"] in all_ids

    def test_messages_returns_channel_messages(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="context",
            sender="enrichment",
            content="Prompt enriched",
        )
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="broadcast",
            sender="planner",
            content="Plan finalized",
        )

        result = dispatch({"method": "session.messages", "params": {"id": session["id"]}})
        assert result["ok"] is True
        data = result["data"]
        assert data["session_id"] == session["id"]
        assert data["count"] == 2
        assert len(data["messages"]) == 2
        assert data["messages"][0]["sender"] == "enrichment"
        assert data["messages"][1]["sender"] == "planner"

    def test_messages_filter_by_kind(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        add_channel_message(
            db_conn, session_id=session["id"], kind="context", sender="system", content="ctx"
        )
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="dm",
            sender="user",
            content="focus",
            recipient="executor",
        )

        result = dispatch(
            {
                "method": "session.messages",
                "params": {"id": session["id"], "kind": "dm"},
            }
        )
        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["messages"][0]["kind"] == "dm"

    def test_messages_filter_by_sender_and_recipient(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="dm",
            sender="planner:a1b2c3d4",
            recipient="executor:e1f2a3b4",
            content="Do the migration",
        )
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="dm",
            sender="planner:ff00aa11",
            recipient="reviewer:b2c3d4e5",
            content="Double-check edge cases",
        )

        result = dispatch(
            {
                "method": "session.messages",
                "params": {
                    "id": session["id"],
                    "sender": "planner",
                    "recipient": "executor",
                },
            }
        )
        assert result["ok"] is True
        data = result["data"]
        assert data["sender"] == "planner"
        assert data["recipient"] == "executor"
        assert data["count"] == 1
        assert data["messages"][0]["content"] == "Do the migration"

    def test_messages_metadata_is_json_string(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="context",
            sender="planner:abcd1234",
            content="details",
            metadata='{"files":["a.py"],"tokens":42}',
        )

        result = dispatch({"method": "session.messages", "params": {"id": session["id"]}})
        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert isinstance(result["data"]["messages"][0]["metadata"], str)
        assert result["data"]["messages"][0]["metadata"] == '{"files":["a.py"],"tokens":42}'

    def test_messages_supports_offset_pagination(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="context",
            sender="one",
            content="first",
        )
        add_channel_message(
            db_conn,
            session_id=session["id"],
            kind="context",
            sender="two",
            content="second",
        )

        result = dispatch(
            {
                "method": "session.messages",
                "params": {"id": session["id"], "limit": 1, "offset": 1},
            }
        )
        assert result["ok"] is True
        data = result["data"]
        assert data["limit"] == 1
        assert data["offset"] == 1
        assert data["count"] == 1
        assert data["messages"][0]["content"] == "second"

    def test_messages_rejects_negative_pagination_values(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")

        result = dispatch(
            {
                "method": "session.messages",
                "params": {"id": session["id"], "offset": -1},
            }
        )
        assert result["ok"] is False
        assert result["code"] == "INVALID_PARAMS"

    def test_messages_not_found(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        result = dispatch({"method": "session.messages", "params": {"id": "nonexistent"}})
        assert result["ok"] is False
        assert result["code"] == "NOT_FOUND"

    def test_session_post_creates_message(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        result = dispatch(
            {
                "method": "session.post",
                "params": {
                    "id": session["id"],
                    "content": "Need to keep API shape stable",
                    "kind": "steer",
                    "sender": "operator",
                    "recipient": "executor:abcd1234",
                    "metadata": {"source": "api-test"},
                },
            }
        )
        assert result["ok"] is True
        payload = result["data"]
        assert payload["session_id"] == session["id"]
        assert payload["kind"] == "steer"
        assert payload["sender"] == "operator:api"

    def test_task_steer_posts_channel_message_when_not_running(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])

        result = dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "Prefer adapter over direct import"},
            }
        )
        assert result["ok"] is True
        assert result["data"]["live_applied"] is False
        assert "not running" in result["data"]["live_error"]

        messages = dispatch({"method": "session.messages", "params": {"id": session["id"]}})
        assert messages["ok"] is True
        assert messages["data"]["count"] == 1
        assert messages["data"]["messages"][0]["kind"] == "steer"
        assert messages["data"]["messages"][0]["recipient"] == f"executor:{task['id'][:8]}"

    def test_task_steer_applies_live_when_active_turn(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "running")
        db_conn.execute(
            "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
            ("thread-live-1", "turn-live-1", task["id"]),
        )
        db_conn.commit()

        captured: dict[str, str] = {}

        async def _fake_steer_active_turn(*, thread_id, active_turn_id, content, timeout=30):
            captured["thread_id"] = thread_id
            captured["active_turn_id"] = active_turn_id
            captured["content"] = content
            return {"turnId": active_turn_id}

        monkeypatch.setattr("agm.api.steer_active_turn", _fake_steer_active_turn)

        result = dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "Focus on migration edge cases"},
            }
        )
        assert result["ok"] is True
        assert result["data"]["live_applied"] is True
        assert result["data"]["turn_id"] == "turn-live-1"
        assert captured == {
            "thread_id": "thread-live-1",
            "active_turn_id": "turn-live-1",
            "content": "Focus on migration edge cases",
        }

    def test_task_steer_default_metadata_tracks_live_flag(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])

        result = dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "Hold for now", "live": False},
            }
        )
        assert result["ok"] is True
        assert result["data"]["live_requested"] is False

        messages = dispatch({"method": "session.messages", "params": {"id": session["id"]}})
        assert messages["ok"] is True
        metadata = json.loads(messages["data"]["messages"][0]["metadata"])
        assert metadata["live"] is False

    def test_task_steer_expected_turn_mismatch_is_reported(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "running")
        db_conn.execute(
            "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
            ("thread-live", "turn-old", task["id"]),
        )
        db_conn.commit()

        async def _raise_mismatch(*, thread_id, active_turn_id, content, timeout=30):
            raise RuntimeError("expectedTurnId mismatch")

        monkeypatch.setattr("agm.api.steer_active_turn", _raise_mismatch)

        result = dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "try steer"},
            }
        )
        assert result["ok"] is True
        assert result["data"]["live_applied"] is False
        assert "expectedTurnId mismatch" in result["data"]["live_error"]

    def test_task_steer_turn_ended_mid_request_is_reported(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "running")
        db_conn.execute(
            "UPDATE tasks SET thread_id = ?, active_turn_id = ? WHERE id = ?",
            ("thread-live", "turn-live", task["id"]),
        )
        db_conn.commit()

        async def _raise_ended(*, thread_id, active_turn_id, content, timeout=30):
            raise RuntimeError("active turn ended before steer")

        monkeypatch.setattr("agm.api.steer_active_turn", _raise_ended)

        result = dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "try steer"},
            }
        )
        assert result["ok"] is True
        assert result["data"]["live_applied"] is False
        assert "active turn ended" in result["data"]["live_error"]

    def test_task_steers_lists_persisted_history(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        session = create_session(db_conn, project_id=_project_id(db_conn), trigger="do")
        plan = _create_plan(db_conn)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        task = _create_task(db_conn, plan["id"])

        dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "first", "live": False},
            }
        )
        dispatch(
            {
                "method": "task.steer",
                "params": {"id": task["id"], "content": "second", "live": False},
            }
        )

        result = dispatch({"method": "task.steers", "params": {"id": task["id"], "limit": 10}})
        assert result["ok"] is True
        assert result["data"]["count"] == 2
        assert result["data"]["items"][0]["content"] == "second"
        assert result["data"]["items"][1]["content"] == "first"

    def test_session_list_reconciles_active_when_all_work_is_terminal(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        update_session_status(db_conn, session["id"], "active")
        plan = _create_plan(db_conn, project_id=pid)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        update_plan_request_status(db_conn, plan["id"], "running")
        update_plan_request_status(db_conn, plan["id"], "finalized")
        task = _create_task(db_conn, plan["id"])
        update_task_status(db_conn, task["id"], "ready")
        update_task_status(db_conn, task["id"], "running")
        update_task_status(db_conn, task["id"], "approved")
        db_conn.execute("UPDATE tasks SET skip_merge = 1 WHERE id = ?", (task["id"],))
        db_conn.commit()

        result = dispatch({"method": "session.list", "params": {"show_all": True}})
        assert result["ok"] is True
        row = next(r for r in result["data"] if r["id"] == session["id"])
        assert row["status"] == "completed"

    def test_session_list_keeps_active_when_task_creation_running(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        pid = _project_id(db_conn)
        session = create_session(db_conn, project_id=pid, trigger="do")
        update_session_status(db_conn, session["id"], "active")
        plan = _create_plan(db_conn, project_id=pid)
        set_plan_session_id(db_conn, plan["id"], session["id"])
        update_plan_request_status(db_conn, plan["id"], "running")
        update_plan_request_status(db_conn, plan["id"], "finalized")
        update_plan_task_creation_status(db_conn, plan["id"], "running")

        result = dispatch({"method": "session.list", "params": {"show_all": True}})
        assert result["ok"] is True
        row = next(r for r in result["data"] if r["id"] == session["id"])
        assert row["status"] == "active"

    def test_queue_inspect_method(self, db_conn, monkeypatch):
        _patch(monkeypatch, db_conn)
        with patch(
            "agm.queue.inspect_queue_jobs",
            return_value=[{"job_id": "exec-abc", "queue": "agm:exec", "status": "started"}],
        ):
            result = dispatch({"method": "queue.inspect", "params": {"queue": "agm:exec"}})
        assert result["ok"] is True
        assert result["data"][0]["job_id"] == "exec-abc"


# ---------------------------------------------------------------------------
# Entry point (main)
# ---------------------------------------------------------------------------


class TestMain:
    def test_valid_request(self, db_conn, monkeypatch, capsys):
        _patch(monkeypatch, db_conn)
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"method": "project.list"})))
        main()
        response = json.loads(capsys.readouterr().out)
        assert response["ok"] is True

    def test_bad_input(self, monkeypatch, capsys):
        for bad_input in ["not json", ""]:
            monkeypatch.setattr("sys.stdin", io.StringIO(bad_input))
            main()
            response = json.loads(capsys.readouterr().out)
            assert response["ok"] is False
            assert response["code"] == "INVALID_PARAMS"
