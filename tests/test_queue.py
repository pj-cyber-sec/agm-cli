"""Tests for the rq-based queue layer."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from agm.backends import (
    EXECUTOR_PROMPT_SUFFIX,
    PLAN_PROMPT_SUFFIX,
    REFRESH_PROMPT_SUFFIX,
    REVIEWER_PROMPT_SUFFIX,
    TASK_PROMPT_SUFFIX,
)
from agm.db import (
    add_task_block,
    add_task_log,
    add_trace_event,
    claim_task,
    count_plan_requests_by_status,
    create_plan_request,
    create_session,
    create_task,
    finalize_plan_request,
    get_connection,
    get_plan_request,
    get_session,
    get_task,
    list_plan_logs,
    list_status_history,
    list_task_logs,
    list_tasks,
    list_trace_events,
    set_plan_session_id,
    set_project_app_server_ask_for_approval,
    set_project_base_branch,
    set_project_model_config,
    set_task_model,
    set_task_thread_id,
    update_task_status,
)
from agm.jobs_quality_gate import QualityCheckResult, QualityGateResult


def _qg_pass() -> QualityGateResult:
    """Return a QualityGateResult that passes (no checks configured)."""
    return QualityGateResult(auto_fix_ran=False, auto_fix_committed=False, checks=[])


def _qg_fail(name: str, output: str) -> QualityGateResult:
    """Return a QualityGateResult with one failed check."""
    return QualityGateResult(
        auto_fix_ran=False,
        auto_fix_committed=False,
        checks=[QualityCheckResult(name=name, passed=False, output=output, duration_ms=100)],
    )


def get_project_id(conn):
    row = conn.execute("SELECT id FROM projects WHERE name = 'testproj'").fetchone()
    return row["id"]


def make_plan(conn, prompt="do stuff"):
    pid = get_project_id(conn)
    return create_plan_request(conn, project_id=pid, prompt=prompt, caller="cli", backend="codex")


def task_transition_rows(conn, task_id: str):
    return list_status_history(conn, entity_type="task", entity_id=task_id)


PROJECT_INSTRUCTIONS_SECTION_DELIMITER = "\n\n--- Project-specific instructions ---\n"
WORKTREE_SYNC_WARNING_MESSAGE = (
    "Merge succeeded but working tree sync failed. Run git checkout HEAD -- . manually."
)


class _FakeCodexTurnClient:
    """Minimal app-server client stub for _codex_turn tests."""

    def __init__(
        self,
        *,
        thread_id="thread-123",
        output_text="agent output",
        failed_message=None,
        failed_error_info=None,
        token_usage=None,
        fail_include_turns=False,
        resume_response=None,
    ):
        self.thread_id = thread_id
        self.output_text = output_text
        self.failed_message = failed_message
        self.failed_error_info = failed_error_info
        self.per_turn_usage = token_usage or {"inputTokens": 100, "outputTokens": 200}
        self.fail_include_turns = fail_include_turns
        self.resume_response = resume_response
        self._cumulative_input = 0
        self._cumulative_output = 0
        self._turn_count = 0
        self.calls = []
        self.removed_handlers = []
        self._notification_handlers = {}

    def on_notification(self, method, handler):
        self._notification_handlers.setdefault(method, []).append(handler)

    def remove_notification_handler(self, method, handler):
        self.removed_handlers.append((method, handler))
        handlers = self._notification_handlers.get(method, [])
        if handler in handlers:
            handlers.remove(handler)

    async def request(self, method, params=None, timeout=120):
        self.calls.append((method, params, timeout))

        if method == "thread/start":
            return {"thread": {"id": self.thread_id}}
        if method == "thread/resume":
            if self.resume_response is not None:
                return self.resume_response
            return {"thread": {"id": self.thread_id}}
        if method == "turn/start":
            self._turn_count += 1
            turn_id = f"turn-{self._turn_count}"
            for handler in list(self._notification_handlers.get("turn/started", [])):
                handler({"turn": {"id": turn_id, "status": "running"}})
            turn = {
                "status": "completed",
                "items": [{"type": "agentMessage", "text": self.output_text}],
            }
            if self.failed_message is not None:
                error: dict = {"message": self.failed_message}
                if self.failed_error_info is not None:
                    error["codexErrorInfo"] = self.failed_error_info
                turn = {"status": "failed", "error": error}
            # Fire token usage notification (like real Codex app-server).
            # ``total`` is cumulative across the thread, ``last`` is per-turn.
            self._cumulative_input += self.per_turn_usage.get("inputTokens", 0)
            self._cumulative_output += self.per_turn_usage.get("outputTokens", 0)
            cumulative = {
                "inputTokens": self._cumulative_input,
                "outputTokens": self._cumulative_output,
            }
            for handler in list(self._notification_handlers.get("thread/tokenUsage/updated", [])):
                handler(
                    {
                        "turnId": turn_id,
                        "tokenUsage": {"last": self.per_turn_usage, "total": cumulative},
                    }
                )
            for handler in list(self._notification_handlers.get("turn/completed", [])):
                handler({"turn": turn})
            return {}
        if method == "thread/read":
            if self.fail_include_turns and params and params.get("includeTurns") is True:
                from agm.client import RPCError

                raise RPCError({"message": "ephemeral threads do not support includeTurns"})
            return {
                "thread": {
                    "turns": [
                        {
                            "items": [
                                {"type": "agentMessage", "text": self.output_text},
                            ]
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected request: {method}")


# -- DB helpers still used by queue --


def test_count_plan_requests_by_status(db_conn):
    conn = db_conn
    make_plan(conn, "a")
    make_plan(conn, "b")
    make_plan(conn, "c")
    counts = count_plan_requests_by_status(conn)
    assert counts["pending"] == 3


# -- enqueue_plan_request --


def test_enqueue_plan_request_calls_rq(db_conn):
    """enqueue_plan_request should call rq's Queue.enqueue with the right args."""
    conn = db_conn
    p = make_plan(conn)
    mock_job = MagicMock()
    mock_job.id = f"plan-{p['id']}"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker"),
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import enqueue_plan_request

        job = enqueue_plan_request(p["id"])

    assert job.id == f"plan-{p['id']}"


# -- queue helpers --


def test_get_queue_counts():
    """get_queue_counts should map all queues with correct field-to-registry mapping."""
    with patch("agm.queue.get_redis") as mock_redis:
        redis_conn = MagicMock()
        redis_conn.exists.return_value = True
        mock_redis.return_value = redis_conn

        with (
            patch("agm.queue.Queue") as MockQueue,
            patch("agm.queue.FailedJobRegistry") as MockFailed,
            patch("agm.queue.StartedJobRegistry") as MockStarted,
            patch("agm.queue.Job.fetch") as mock_fetch,
        ):
            # Distinct values per registry prove the mapping is correct — any swap fails
            mock_q = MagicMock()
            mock_q.__len__ = MagicMock(return_value=5)
            mock_q.connection = redis_conn
            MockQueue.return_value = mock_q

            mock_failed = MagicMock()
            mock_failed.__len__ = MagicMock(return_value=3)
            MockFailed.return_value = mock_failed

            mock_started = MagicMock()
            mock_started.get_job_ids.return_value = ["job-1"]
            MockStarted.return_value = mock_started
            mock_job = MagicMock()
            mock_job.get_status.return_value = "started"
            mock_job.worker_name = None
            mock_fetch.return_value = mock_job

            from agm.queue import AGM_QUEUE_NAMES, get_queue_counts

            counts = get_queue_counts()

    # All 7 queue names present
    assert set(counts.keys()) == set(AGM_QUEUE_NAMES)
    # Field-to-registry mapping correct for every queue
    for stats in counts.values():
        assert stats["queued"] == 5  # Queue.__len__
        assert stats["running"] == 1  # live started jobs
        assert stats["failed"] == 3  # FailedJobRegistry.__len__


def test_check_redis_connection_safe_healthy():
    with patch("agm.queue.get_redis") as mock_redis:
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn

        from agm.queue import check_redis_connection_safe

        probe = check_redis_connection_safe()

    assert probe == {"ok": True, "error_type": None, "error": None}


def test_check_redis_connection_safe_unavailable():
    with patch("agm.queue.get_redis") as mock_redis:
        mock_conn = MagicMock()
        mock_conn.ping.side_effect = ConnectionError("redis unavailable")
        mock_redis.return_value = mock_conn

        from agm.queue import check_redis_connection_safe

        probe = check_redis_connection_safe()

    assert probe["ok"] is False
    assert probe["error_type"] == "ConnectionError"
    assert "redis unavailable" in str(probe["error"])


def test_get_queue_counts_safe_healthy():
    with patch("agm.queue.get_redis") as mock_redis:
        mock_conn = MagicMock()
        mock_conn.exists.return_value = True
        mock_redis.return_value = mock_conn

        with (
            patch("agm.queue.Queue") as MockQueue,
            patch("agm.queue.FailedJobRegistry") as MockFailed,
            patch("agm.queue.StartedJobRegistry") as MockStarted,
            patch("agm.queue.Job.fetch") as mock_fetch,
        ):
            mock_q = MagicMock()
            mock_q.__len__ = MagicMock(return_value=4)
            mock_q.connection = mock_conn
            MockQueue.return_value = mock_q

            mock_failed = MagicMock()
            mock_failed.__len__ = MagicMock(return_value=2)
            MockFailed.return_value = mock_failed

            mock_started = MagicMock()
            mock_started.get_job_ids.return_value = ["job-1"]
            MockStarted.return_value = mock_started
            mock_job = MagicMock()
            mock_job.get_status.return_value = "started"
            mock_job.worker_name = None
            mock_fetch.return_value = mock_job

            from agm.queue import AGM_QUEUE_NAMES, get_queue_counts_safe

            counts = get_queue_counts_safe()

    assert counts["ok"] is True
    assert counts["error_type"] is None
    assert counts["error"] is None
    assert list(counts["queues"].keys()) == list(AGM_QUEUE_NAMES)
    for queue_name in AGM_QUEUE_NAMES:
        assert list(counts["queues"][queue_name].keys()) == ["queued", "running", "failed"]
        assert counts["queues"][queue_name] == {"queued": 4, "running": 1, "failed": 2}


def test_get_queue_counts_safe_redis_unavailable():
    with patch("agm.queue.get_redis", side_effect=ConnectionError("redis unavailable")):
        from agm.queue import AGM_QUEUE_NAMES, get_queue_counts_safe

        counts = get_queue_counts_safe()

    assert counts["ok"] is False
    assert counts["error_type"] == "ConnectionError"
    assert "redis unavailable" in str(counts["error"])
    assert list(counts["queues"].keys()) == list(AGM_QUEUE_NAMES)
    for queue_name in AGM_QUEUE_NAMES:
        assert list(counts["queues"][queue_name].keys()) == ["queued", "running", "failed"]
        assert counts["queues"][queue_name] == {"queued": None, "running": None, "failed": None}


def test_get_queue_counts_prunes_stale_started_entries():
    """Stale started jobs should be removed and excluded from running counts."""
    with patch("agm.queue.get_redis") as mock_redis:
        redis_conn = MagicMock()
        redis_conn.exists.return_value = True
        mock_redis.return_value = redis_conn

        with (
            patch("agm.queue.Queue") as MockQueue,
            patch("agm.queue.FailedJobRegistry") as MockFailed,
            patch("agm.queue.StartedJobRegistry") as MockStarted,
            patch("agm.queue.Job.fetch") as mock_fetch,
        ):
            mock_q = MagicMock()
            mock_q.__len__ = MagicMock(return_value=0)
            mock_q.connection = redis_conn
            MockQueue.return_value = mock_q

            mock_failed = MagicMock()
            mock_failed.__len__ = MagicMock(return_value=0)
            MockFailed.return_value = mock_failed

            mock_started = MagicMock()
            mock_started.get_job_ids.return_value = ["live", "missing", "finished", "dead-worker"]
            MockStarted.return_value = mock_started

            live_job = MagicMock()
            live_job.get_status.return_value = "started"
            live_job.worker_name = "worker-live"

            finished_job = MagicMock()
            finished_job.get_status.return_value = "finished"
            finished_job.worker_name = None

            dead_worker_job = MagicMock()
            dead_worker_job.get_status.return_value = "started"
            dead_worker_job.worker_name = "worker-dead"

            def _fetch_side_effect(job_id, connection=None):
                if job_id == "live":
                    return live_job
                if job_id == "missing":
                    raise RuntimeError("missing job")
                if job_id == "finished":
                    return finished_job
                if job_id == "dead-worker":
                    return dead_worker_job
                raise AssertionError(f"unexpected job id: {job_id}")

            mock_fetch.side_effect = _fetch_side_effect
            redis_conn.exists.side_effect = lambda key: key == "rq:worker:worker-live"

            from agm.queue import AGM_QUEUE_NAMES, get_queue_counts

            counts = get_queue_counts()

    for queue_name in AGM_QUEUE_NAMES:
        assert counts[queue_name]["running"] == 1
    # Three stale entries per queue should be removed.
    assert mock_started.remove.call_count == len(AGM_QUEUE_NAMES) * 3


def test_inspect_queue_jobs_includes_linkage_and_worker_health():
    with patch("agm.queue.get_redis") as mock_redis:
        redis_conn = MagicMock()
        redis_conn.exists.side_effect = lambda key: key == "rq:worker:w1"
        mock_redis.return_value = redis_conn

        with (
            patch("agm.queue.Queue") as MockQueue,
            patch("agm.queue.StartedJobRegistry") as MockStarted,
            patch("agm.queue.Job.fetch") as mock_fetch,
            patch(
                "agm.queue._resolve_job_entity_context",
                return_value={"task_id": "abc", "plan_id": "plan1", "session_id": "sess1"},
            ),
        ):
            mock_q = MagicMock()
            mock_q.job_ids = ["exec-abc"]
            mock_q.connection = redis_conn
            MockQueue.return_value = mock_q

            started = MagicMock()
            started.get_job_ids.return_value = ["exec-abc"]
            MockStarted.return_value = started

            job = MagicMock()
            job.description = "Execute task abc"
            job.get_status.return_value = "started"
            job.worker_name = "w1"
            job.enqueued_at = None
            job.started_at = None
            job.ended_at = None
            mock_fetch.return_value = job

            from agm.queue import inspect_queue_jobs

            rows = inspect_queue_jobs("agm:exec")

    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == "exec-abc"
    assert row["queue"] == "agm:exec"
    assert row["status"] == "started"
    assert row["stale_started"] is False
    assert row["worker_alive"] is True
    assert row["entity_type"] == "task"
    assert row["task_id"] == "abc"
    assert row["session_id"] == "sess1"


def test_inspect_queue_jobs_treats_enum_started_as_live():
    with patch("agm.queue.get_redis") as mock_redis:
        redis_conn = MagicMock()
        redis_conn.exists.return_value = True
        mock_redis.return_value = redis_conn

        with (
            patch("agm.queue.Queue") as MockQueue,
            patch("agm.queue.StartedJobRegistry") as MockStarted,
            patch("agm.queue.Job.fetch") as mock_fetch,
        ):
            mock_q = MagicMock()
            mock_q.job_ids = ["exec-abc"]
            mock_q.connection = redis_conn
            MockQueue.return_value = mock_q

            started = MagicMock()
            started.get_job_ids.return_value = ["exec-abc"]
            MockStarted.return_value = started

            job = MagicMock()
            job.description = "Execute task abc"
            job.get_status.return_value = "JobStatus.STARTED"
            job.worker_name = "w1"
            job.enqueued_at = None
            job.started_at = None
            job.ended_at = None
            mock_fetch.return_value = job

            from agm.queue import inspect_queue_jobs

            rows = inspect_queue_jobs("agm:exec")

    assert len(rows) == 1
    assert rows[0]["stale_started"] is False


# -- jobs --


def test_run_plan_request_sets_worker_pid(db_conn):
    """run_plan_request should set worker metadata and move plan to running."""
    conn = db_conn
    p = make_plan(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_plan_text = "## Plan\n1. Do stuff\n2. More stuff"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_plan._run_plan_request_codex",
            return_value=mock_plan_text,
        ),
    ):
        from agm.jobs import run_plan_request

        result = run_plan_request(p["id"])

    assert result == mock_plan_text
    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["pid"] is not None
    assert found["status"] == "running"  # backend was mocked
    history = list_status_history(verify_conn, entity_type="plan", entity_id=p["id"])
    assert [(row["old_status"], row["new_status"]) for row in history] == [("pending", "running")]
    assert [row["actor"] for row in history] == [None]
    verify_conn.close()


def test_run_plan_request_success_records_pending_running_finalized_timeline(db_conn):
    """run_plan_request success should record pending->running->finalized timeline."""
    conn = db_conn
    p = make_plan(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    class _NoopClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def set_server_request_handler(self, _handler):
            return None

    async def _fake_codex_turn(_client, **kwargs):
        on_thread_ready = kwargs.get("on_thread_ready")
        if on_thread_ready:
            on_thread_ready("thread-plan-success")
        return (
            "thread-plan-success",
            '{"title":"t","summary":"s","tasks":[]}',
            {"input_tokens": 11, "output_tokens": 7},
        )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.client.AppServerClient", return_value=_NoopClient()),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_task_creation._trigger_task_creation"),
    ):
        from agm.jobs import run_plan_request

        result = run_plan_request(p["id"])

    assert result == '{"title":"t","summary":"s","tasks":[]}'
    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["status"] == "finalized"
    assert found["thread_id"] == "thread-plan-success"
    history = list_status_history(verify_conn, entity_type="plan", entity_id=p["id"])
    assert [(row["old_status"], row["new_status"]) for row in history] == [
        ("pending", "running"),
        ("running", "finalized"),
    ]
    assert [row["actor"] for row in history] == [None, None]
    verify_conn.close()


def test_run_plan_request_marks_failed_on_error(db_conn):
    """run_plan_request should mark the plan as failed when backend raises."""
    conn = db_conn
    p = make_plan(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_plan._run_plan_request_codex",
            side_effect=RuntimeError("backend error"),
        ),
    ):
        from agm.jobs import run_plan_request

        with pytest.raises(RuntimeError, match="backend error"):
            run_plan_request(p["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["pid"] is not None
    assert found["status"] == "failed"
    history = list_status_history(verify_conn, entity_type="plan", entity_id=p["id"])
    assert [(row["old_status"], row["new_status"]) for row in history] == [
        ("pending", "running"),
        ("running", "failed"),
    ]
    assert [row["actor"] for row in history] == [None, None]
    verify_conn.close()


def test_run_plan_request_failure_logs_traceback(db_conn):
    """run_plan_request should log full traceback to plan_logs on failure."""
    conn = db_conn
    p = make_plan(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_plan._run_plan_request_codex",
            side_effect=RuntimeError("detailed backend error"),
        ),
    ):
        from agm.jobs import run_plan_request

        with pytest.raises(RuntimeError, match="detailed backend error"):
            run_plan_request(p["id"])

    verify_conn = get_connection(Path(db_path))
    logs = list_plan_logs(verify_conn, p["id"])
    error_logs = [entry for entry in logs if entry["level"] == "ERROR"]
    assert len(error_logs) >= 1
    # Should contain traceback text and the error message
    error_text = error_logs[0]["message"]
    assert "detailed backend error" in error_text
    assert "Traceback" in error_text
    verify_conn.close()


def test_run_plan_request_not_found(db_conn):
    """run_plan_request should return skipped for nonexistent plan."""
    conn = db_conn
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))):
        from agm.jobs import run_plan_request

        result = run_plan_request("nonexistent")
        assert result == "skipped:entity_missing"


def test_run_plan_request_persists_resolved_think_model(db_conn):
    """run_plan_request should persist the resolved think model."""
    import json

    conn = db_conn
    p = make_plan(conn)
    pid = get_project_id(conn)
    set_project_model_config(
        conn,
        pid,
        json.dumps({"think_model": "gpt-5.3-codex", "think_effort": "low"}),
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_plan._run_plan_request_codex",
            return_value="resolved model plan",
        ),
    ):
        from agm.jobs import run_plan_request

        run_plan_request(p["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["model"] == "gpt-5.3-codex"
    verify_conn.close()


@pytest.mark.asyncio
async def test_run_plan_request_codex_uses_resolved_model_config_in_runtime_thread(db_conn):
    """Plan request should start a thread with resolved Codex think model and effort."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"think_model": "gpt-5.3-codex", "think_effort": "low"}),
    )
    set_project_app_server_ask_for_approval(conn, get_project_id(conn), "on-request")
    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-plan",
            "plan output",
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("agm.jobs_plan._codex_client", _fake_codex_client),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
    ):
        from agm.jobs import _run_plan_request_codex_async

        result = await _run_plan_request_codex_async(conn, p)

    assert result == "plan output"
    start_thread_params = captured["start_thread_params"]
    assert start_thread_params["model"] == "gpt-5.3-codex"
    assert start_thread_params["approvalPolicy"] == "on-request"
    assert captured["turn_config"]["effort"] == "low"


def test_run_plan_request_resumes_parent_thread(db_conn):
    """run_plan_request with parent_id should go through continuation enrichment then resume."""
    conn = db_conn
    parent = make_plan(conn, "original plan")
    # Simulate parent being finalized with a thread
    conn.execute(
        "UPDATE plans SET status = 'finalized', thread_id = 'thread-parent-123' WHERE id = ?",
        (parent["id"],),
    )
    conn.commit()

    # Create child with parent_id
    pid = get_project_id(conn)
    from agm.db import create_plan_request as _create

    child = _create(
        conn,
        project_id=pid,
        prompt="follow-up",
        caller="cli",
        backend="codex",
        parent_id=parent["id"],
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_plan_text = '{"title":"Follow-up","summary":"Done","tasks":[]}'

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_plan._run_plan_request_codex",
            return_value=mock_plan_text,
        ),
    ):
        from agm.jobs import run_plan_request

        result = run_plan_request(child["id"])

    assert result == mock_plan_text
    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, child["id"])
    assert found["parent_id"] == parent["id"]
    assert found["status"] == "running"  # backend was mocked
    verify_conn.close()


def test_run_plan_request_codex_warns_when_parent_has_no_thread_id(db_conn, caplog):
    """Continuation plans should warn and start fresh when parent thread metadata is missing."""
    conn = db_conn
    parent = make_plan(conn, "original plan")
    pid = get_project_id(conn)
    from agm.db import create_plan_request as _create

    child = _create(
        conn,
        project_id=pid,
        prompt="follow-up",
        caller="cli",
        backend="codex",
        parent_id=parent["id"],
    )

    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-new",
            "child plan output",
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    with (
        patch("agm.jobs_plan._codex_client", _fake_codex_client),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_task_creation._trigger_task_creation"),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_plan_request_codex_async

        caplog.set_level(logging.WARNING)
        result = asyncio.run(_run_plan_request_codex_async(conn, child))

    assert result == "child plan output"
    assert (
        f"Plan {child['id']} has parent_id {parent['id']}"
        " but parent has no thread_id, starting fresh" in caplog.text
    )
    assert captured["prompt"] == "follow-up" + PLAN_PROMPT_SUFFIX
    assert captured.get("resume_thread_id") is None
    assert captured.get("start_thread_params", {}).get("cwd") == "/tmp/testproj"


def test_run_plan_request_codex_continuation_uses_child_prompt_and_parent_thread(db_conn):
    """Continuation should resume parent thread while using child prompt text."""
    conn = db_conn
    parent = make_plan(conn, "original plan")
    conn.execute(
        "UPDATE plans SET status = 'finalized', thread_id = 'thread-parent-789' WHERE id = ?",
        (parent["id"],),
    )
    conn.commit()

    pid = get_project_id(conn)
    from agm.db import create_plan_request as _create

    child = _create(
        conn,
        project_id=pid,
        prompt="child follow-up prompt",
        caller="cli",
        backend="codex",
        parent_id=parent["id"],
    )
    set_project_app_server_ask_for_approval(
        conn,
        pid,
        {"reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}},
    )

    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-child",
            "child plan output",
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    with (
        patch("agm.jobs_plan._codex_client", _fake_codex_client),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_task_creation._trigger_task_creation"),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_plan_request_codex_async

        result = asyncio.run(_run_plan_request_codex_async(conn, child))

    assert result == "child plan output"
    assert captured["resume_thread_id"] == "thread-parent-789"
    assert captured["resume_thread_params"] == {
        "model": "gpt-5.3-codex",
        "approvalPolicy": {
            "reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}
        },
    }
    assert captured.get("start_thread_params") is None
    assert captured["prompt"] == "child follow-up prompt" + PLAN_PROMPT_SUFFIX
    assert parent["prompt"] not in captured["prompt"]


def test_run_plan_request_codex_prompt_includes_planner_instructions(db_conn):
    """Planner instructions injected via developerInstructions for Codex."""
    conn = db_conn
    plan = make_plan(conn, "write tests for scheduler")
    expected_prompt = "write tests for scheduler" + PLAN_PROMPT_SUFFIX
    instructions = "Planner rules: prioritize security."

    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-1",
            "plan output",
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    with (
        patch("agm.jobs_plan._codex_client", _fake_codex_client),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_task_creation._trigger_task_creation"),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_plan_request_codex_async

        result = asyncio.run(_run_plan_request_codex_async(conn, plan))

    assert result == "plan output"
    assert captured["prompt"] == expected_prompt
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


# -- _extract_plan_text --


def test_extract_plan_text_prefers_plan_items():
    """Plan-type items should be preferred over agentMessage items."""
    from agm.jobs import _extract_plan_text

    result = {
        "thread": {
            "turns": [
                {
                    "items": [
                        {"type": "agentMessage", "text": "Here's my plan"},
                        {"type": "plan", "text": "## Step 1\nDo X"},
                    ]
                }
            ]
        }
    }
    text = _extract_plan_text(result)
    assert "Step 1" in text
    assert "Here's my plan" not in text


def test_extract_plan_text_falls_back_to_agent_message():
    """Should use agentMessage if no plan items exist."""
    from agm.jobs import _extract_plan_text

    result = {
        "thread": {
            "turns": [
                {
                    "items": [
                        {"type": "agentMessage", "text": "I suggest doing X and Y"},
                        {"type": "commandExecution", "command": "ls"},
                    ]
                }
            ]
        }
    }
    text = _extract_plan_text(result)
    assert text == "I suggest doing X and Y"


def test_extract_plan_text_empty():
    """Should return empty string if no relevant items."""
    from agm.jobs import _extract_plan_text

    result = {"thread": {"turns": [{"items": [{"type": "reasoning", "content": []}]}]}}
    assert _extract_plan_text(result) == ""


def test_extract_plan_text_takes_last_turn():
    """Should only read the last turn, ignoring earlier turns."""
    from agm.jobs import _extract_plan_text

    result = {
        "thread": {
            "turns": [
                {"items": [{"type": "agentMessage", "text": "Progress snapshot"}]},
                {"items": [{"type": "agentMessage", "text": "Final output"}]},
            ]
        }
    }
    text = _extract_plan_text(result)
    assert text == "Final output"
    assert "Progress" not in text


def test_extract_plan_text_no_stale_parent():
    """Continuation: if new turn has no plan text, should NOT return parent's text."""
    from agm.jobs import _extract_plan_text

    result = {
        "thread": {
            "turns": [
                # Parent's turn — has plan text
                {"items": [{"type": "plan", "text": "Parent plan output"}]},
                # Child's turn — only reasoning, no plan/agentMessage
                {"items": [{"type": "reasoning", "content": []}]},
            ]
        }
    }
    text = _extract_plan_text(result)
    assert text == ""
    assert "Parent" not in text


# -- _codex_turn --


@pytest.mark.asyncio
async def test_codex_turn_start_mode_returns_text_and_runs_on_thread_ready():
    """_codex_turn should start a thread, run a turn, and return extracted text."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(output_text="hello from agent")
    seen_thread_id = []

    def on_thread_ready(thread_id):
        seen_thread_id.append(thread_id)

    thread_id, output, tokens = await _codex_turn(
        client,
        prompt="do work",
        turn_config={"toolChoice": "none"},
        start_thread_params={"cwd": "/tmp/test"},
        on_thread_ready=on_thread_ready,
    )

    assert thread_id == "thread-123"
    assert output == "hello from agent"
    assert tokens == {
        "input_tokens": 100,
        "output_tokens": 200,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert seen_thread_id == ["thread-123"]
    methods = [method for method, _, _ in client.calls]
    assert methods == ["thread/start", "turn/start", "thread/read"]
    assert client.calls[0][2] == 300  # thread/start timeout
    assert client.calls[2][2] == 300  # thread/read timeout
    turn_params = client.calls[1][1]
    assert turn_params["input"] == [{"type": "text", "text": "do work"}]
    assert turn_params["toolChoice"] == "none"


@pytest.mark.asyncio
async def test_codex_turn_ephemeral_threads_fallback_when_include_turns_unsupported():
    """_codex_turn should recover when ephemeral threads reject includeTurns."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(output_text="ephemeral output", fail_include_turns=True)

    thread_id, output, tokens = await _codex_turn(
        client,
        prompt="do work",
        turn_config={"toolChoice": "none"},
        start_thread_params={"cwd": "/tmp/test", "ephemeral": True},
    )

    assert thread_id == "thread-123"
    assert output == "ephemeral output"
    assert tokens == {
        "input_tokens": 100,
        "output_tokens": 200,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    methods = [method for method, _, _ in client.calls]
    assert methods == ["thread/start", "turn/start", "thread/read", "thread/read"]
    assert client.calls[2][1] == {"threadId": "thread-123", "includeTurns": True}
    assert client.calls[3][1] == {"threadId": "thread-123"}


@pytest.mark.asyncio
async def test_codex_turn_passes_thread_params_directly():
    """Thread params are passed directly to thread/start (no collaborationMode extraction)."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(output_text="planned output")

    await _codex_turn(
        client,
        prompt="plan it",
        turn_config={"outputSchema": {"type": "object"}, "effort": "high"},
        start_thread_params={
            "cwd": "/tmp",
            "model": "m",
            "sandbox": "read-only",
            "developerInstructions": None,
        },
    )

    # thread/start params passed as-is
    thread_start_params = client.calls[0][1]
    assert thread_start_params["model"] == "m"
    assert thread_start_params["sandbox"] == "read-only"

    # turn/start has effort from turn_config
    turn_start_params = client.calls[1][1]
    assert turn_start_params["effort"] == "high"
    assert turn_start_params["outputSchema"] == {"type": "object"}


@pytest.mark.asyncio
async def test_codex_turn_resume_mode_runs_post_initial_turn_hook():
    """_codex_turn should resume an existing thread and allow extra turns via hook."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(thread_id="thread-resume", output_text="final summary")
    seen_thread_id = []

    async def post_initial_turn(run_turn, thread_id):
        seen_thread_id.append(thread_id)
        await run_turn("nudge prompt")

    thread_id, output, tokens = await _codex_turn(
        client,
        prompt="main prompt",
        turn_config={"sandbox": "workspace-write"},
        resume_thread_id="existing-thread-1",
        post_initial_turn=post_initial_turn,
    )

    assert thread_id == "thread-resume"
    assert output == "final summary"
    # Two turns (initial + nudge), each fires 100/200 token notification
    assert tokens == {
        "input_tokens": 200,
        "output_tokens": 400,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert seen_thread_id == ["thread-resume"]
    methods = [method for method, _, _ in client.calls]
    assert methods == ["thread/resume", "turn/start", "turn/start", "thread/read"]
    assert client.calls[0][1] == {"threadId": "existing-thread-1"}
    assert client.calls[2][1]["input"] == [{"type": "text", "text": "nudge prompt"}]


@pytest.mark.asyncio
async def test_codex_turn_resume_mode_records_resume_metadata_trace_and_event():
    """_codex_turn should trace metadata from thread/resume and emit thread status seed event."""
    from agm.jobs import TurnEventContext, _codex_turn

    client = _FakeCodexTurnClient(
        thread_id="thread-resume",
        output_text="final summary",
        resume_response={
            "thread": {
                "id": "thread-resume",
                "status": {"type": "active", "activeFlags": ["waitingOnApproval"]},
                "turns": [
                    {
                        "id": "turn-prev",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "previous output"}],
                    }
                ],
            }
        },
    )
    trace_context = MagicMock()

    with patch("agm.queue.publish_event") as mock_publish:
        thread_id, output, _tokens = await _codex_turn(
            client,
            prompt="main prompt",
            turn_config={"sandbox": "workspace-write"},
            resume_thread_id="existing-thread-1",
            event_context=TurnEventContext(task_id="task-1", plan_id="plan-1", project="testproj"),
            trace_context=trace_context,
        )

    assert thread_id == "thread-resume"
    assert output == "final summary"
    trace_context.record.assert_any_call(
        "threadResume",
        "loaded",
        {
            "thread_id": "thread-resume",
            "thread_status": {"type": "active", "activeFlags": ["waitingOnApproval"]},
            "thread_status_type": "active",
            "thread_active_flags": ["waitingOnApproval"],
            "latest_turn_id": "turn-prev",
            "latest_turn_status": "completed",
            "latest_turn_item_count": 1,
            "latest_turn_item_types": ["agentMessage"],
        },
    )
    assert any(
        call.args[:3] == ("task:thread_status", "task-1", "active")
        and call.kwargs.get("extra", {}).get("source") == "thread/resume"
        and call.kwargs.get("extra", {}).get("latest_turn_id") == "turn-prev"
        and call.kwargs.get("extra", {}).get("thread_context", {}).get("thread_id")
        == "thread-resume"
        for call in mock_publish.call_args_list
    )


@pytest.mark.asyncio
async def test_codex_turn_failed_turn_raises_and_cleans_notification_handler():
    """_codex_turn should raise on failed turn and always remove handler."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(failed_message="boom")

    with pytest.raises(RuntimeError, match="Turn failed: boom"):
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
            start_thread_params={"cwd": "/tmp/test"},
        )

    removed_methods = [m for m, _ in client.removed_handlers]
    assert "turn/started" in removed_methods
    assert "turn/completed" in removed_methods
    assert "thread/tokenUsage/updated" in removed_methods
    assert "model/rerouted" in removed_methods
    assert "error" in removed_methods
    assert "deprecationNotice" in removed_methods
    assert "configWarning" in removed_methods
    assert client._notification_handlers.get("turn/completed") == []
    assert client._notification_handlers.get("thread/tokenUsage/updated") == []


@pytest.mark.asyncio
async def test_codex_turn_requires_launch_mode_and_still_cleans_notification_handler():
    """_codex_turn should reject missing thread launch config."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient()

    with pytest.raises(ValueError, match="Either start_thread_params or resume_thread_id"):
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
        )

    removed_methods = [m for m, _ in client.removed_handlers]
    assert "turn/completed" in removed_methods
    assert "thread/tokenUsage/updated" in removed_methods
    assert "model/rerouted" in removed_methods
    assert "error" in removed_methods


@pytest.mark.asyncio
async def test_run_initial_codex_turn_emits_fallback_events():
    """Fallback thread rollover should publish explicit retry/start events."""
    from agm.jobs_common import TurnEventContext, _run_initial_codex_turn

    class _FallbackStartClient:
        def __init__(self):
            self._starts = 0

        async def request(self, method, params=None, timeout=120):
            if method != "thread/start":
                raise AssertionError(f"Unexpected request: {method}")
            self._starts += 1
            return {"thread": {"id": f"fallback-thread-{self._starts}"}}

    client = _FallbackStartClient()
    turn_done = asyncio.Event()
    turn_state: dict[str, object] = {}
    seen_threads: list[str] = []
    trace_records: list[tuple[str, str | None, dict]] = []

    class _TraceRecorder:
        def record(self, event_type: str, status: str | None, data: dict) -> None:
            trace_records.append((event_type, status, data))

    with (
        patch("agm.jobs_common._run_codex_turn_once", side_effect=[TimeoutError(), {}]),
        patch("agm.queue.publish_event") as mock_publish_event,
    ):
        result_thread = await _run_initial_codex_turn(
            client,
            prompt="do work",
            turn_config={"effort": "high"},
            turn_done=turn_done,
            turn_state=turn_state,
            thread_id="thread-primary",
            fallback_thread_params={"model": "fallback-model"},
            resume_thread_id=None,
            on_thread_ready=seen_threads.append,
            event_context=TurnEventContext(task_id="task-1", plan_id="plan-1", project="proj-1"),
            trace_context=_TraceRecorder(),
        )

    assert result_thread == "fallback-thread-1"
    assert seen_threads == ["fallback-thread-1"]
    assert mock_publish_event.call_count == 2

    retrying_call = mock_publish_event.call_args_list[0]
    assert retrying_call.args[:3] == ("task:execution_fallback", "task-1", "retrying")
    assert retrying_call.kwargs["project"] == "proj-1"
    assert retrying_call.kwargs["plan_id"] == "plan-1"
    assert retrying_call.kwargs["extra"]["failed_thread_id"] == "thread-primary"
    assert retrying_call.kwargs["extra"]["fallback_model"] == "fallback-model"
    assert retrying_call.kwargs["extra"]["reason"] == "TimeoutError"
    assert retrying_call.kwargs["extra"]["thread_context"]["thread_id"] == "thread-primary"

    started_call = mock_publish_event.call_args_list[1]
    assert started_call.args[:3] == ("task:execution_fallback", "task-1", "started")
    assert started_call.kwargs["extra"]["failed_thread_id"] == "thread-primary"
    assert started_call.kwargs["extra"]["fallback_thread_id"] == "fallback-thread-1"
    assert started_call.kwargs["extra"]["reason"] == "TimeoutError"
    assert started_call.kwargs["extra"]["thread_context"]["thread_id"] == "fallback-thread-1"

    assert trace_records == [
        (
            "executionFallback",
            "retrying",
            {
                "failed_thread_id": "thread-primary",
                "fallback_model": "fallback-model",
                "reason": "TimeoutError",
            },
        ),
        (
            "executionFallback",
            "started",
            {
                "failed_thread_id": "thread-primary",
                "fallback_thread_id": "fallback-thread-1",
                "fallback_model": "fallback-model",
                "reason": "TimeoutError",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_codex_turn_structured_model_cap_error():
    """_codex_turn should include modelCap details in RuntimeError message."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(
        failed_message="Rate limit exceeded",
        failed_error_info={"modelCap": {"model": "gpt-5.3", "reset_after_seconds": 120}},
    )

    with pytest.raises(RuntimeError, match=r"\[modelCap\].*model=gpt-5.3.*resets in 120s"):
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
            start_thread_params={"cwd": "/tmp/test"},
        )


@pytest.mark.asyncio
async def test_codex_turn_records_notification_trace_and_turn_indexes(db_conn):
    """Trace should include turn lifecycle notifications with incrementing turn_index."""
    from agm.jobs import _codex_turn
    from agm.jobs_common import TurnEventContext
    from agm.tracing import TraceContext

    conn = db_conn
    client = _FakeCodexTurnClient(output_text="agent output")
    trace_ctx = TraceContext(
        entity_type="task",
        entity_id="trace-task-1",
        stage="execution",
        plan_id="plan-1",
        project="testproj",
        conn=conn,
    )

    async def _post_initial_turn(run_turn, _thread_id):
        await run_turn("commit nudge")

    with patch("agm.queue.publish_event") as mock_publish_event:
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
            start_thread_params={"cwd": "/tmp/test"},
            post_initial_turn=_post_initial_turn,
            event_context=TurnEventContext(task_id="trace-task-1", plan_id="plan-1", project="p1"),
            trace_context=trace_ctx,
        )

    events = list_trace_events(conn, "task", "trace-task-1")
    turn_started = [e for e in events if e["event_type"] == "turn" and e["status"] == "started"]
    turn_completed = [e for e in events if e["event_type"] == "turn" and e["status"] == "completed"]
    token_updates = [e for e in events if e["event_type"] == "tokenUsage"]

    assert [e["turn_index"] for e in turn_started] == [0, 1]
    assert [e["turn_index"] for e in turn_completed] == [0, 1]
    assert [e["turn_index"] for e in token_updates] == [0, 1]
    assert all(e["data"]["input"] == 100 for e in token_updates)
    assert all(e["data"]["output"] == 200 for e in token_updates)

    published_types = [call.args[0] for call in mock_publish_event.call_args_list]
    assert published_types.count("task:turn") >= 2


def test_register_codex_notification_handlers_emits_backend_diagnostics():
    """Backend diagnostic notifications should publish events and persist trace records."""
    from agm.jobs_common import TurnEventContext, _register_codex_notification_handlers

    client = _FakeCodexTurnClient(output_text="ok")
    turn_done = asyncio.Event()
    turn_state: dict[str, object] = {}
    turn_tokens: dict[str, tuple[int, int, int, int]] = {}
    trace_records: list[tuple[str, str | None, dict]] = []

    class _TraceRecorder:
        def record(self, event_type: str, status: str | None, data: dict) -> None:
            trace_records.append((event_type, status, data))

    with patch("agm.queue.publish_event") as mock_publish_event:
        handlers = _register_codex_notification_handlers(
            client,
            turn_done,
            turn_state,
            turn_tokens,
            event_context=TurnEventContext(task_id="task-99", plan_id="plan-99", project="proj-99"),
            trace_context=_TraceRecorder(),
        )
        handlers["model/rerouted"]({"fromModel": "gpt-5-mini", "toModel": "gpt-5"})
        handlers["error"]({"message": "mid-turn failure", "turnId": "turn-1"})
        handlers["deprecationNotice"]({"message": "deprecated API"})
        handlers["configWarning"]({"message": "config mismatch"})
        handlers["thread/status/changed"](
            {"threadId": "thread-99", "oldStatus": "idle", "newStatus": "running"}
        )

    published = [(call.args[0], call.args[2]) for call in mock_publish_event.call_args_list]
    assert ("task:model_rerouted", "updated") in published
    assert ("task:backend_error", "error") in published
    assert ("task:backend_warning", "deprecation") in published
    assert ("task:backend_warning", "config") in published
    assert ("task:thread_status", "running") in published

    thread_events = [
        call for call in mock_publish_event.call_args_list if call.args[0] == "task:thread_status"
    ]
    assert len(thread_events) == 1
    assert thread_events[0].kwargs["extra"]["thread_id"] == "thread-99"
    assert thread_events[0].kwargs["extra"]["old_status"] == "idle"
    assert thread_events[0].kwargs["extra"]["new_status"] == "running"

    assert (
        "modelRerouted",
        "updated",
        {"notification": {"fromModel": "gpt-5-mini", "toModel": "gpt-5"}},
    ) in trace_records
    assert (
        "backendError",
        "error",
        {"notification": {"message": "mid-turn failure", "turnId": "turn-1"}},
    ) in trace_records
    assert ("deprecationNotice", "warning", {"message": "deprecated API"}) in trace_records
    assert ("configWarning", "warning", {"message": "config mismatch"}) in trace_records
    assert (
        "threadStatus",
        "changed",
        {
            "thread_id": "thread-99",
            "old_status": "idle",
            "new_status": "running",
            "notification": {
                "threadId": "thread-99",
                "oldStatus": "idle",
                "newStatus": "running",
            },
        },
    ) in trace_records


@pytest.mark.asyncio
async def test_wait_with_heartbeat_includes_thread_context():
    from agm.jobs_common import TurnEventContext, _wait_with_heartbeat

    turn_done = asyncio.Event()
    turn_state: dict[str, object] = {"active_turn_id": "turn-42"}
    event_context = TurnEventContext(
        task_id="task-42",
        plan_id="plan-42",
        project="proj-42",
        owner_role="executor",
        model="gpt-5",
        model_provider="openai",
    )

    with (
        patch("agm.jobs_common.HEARTBEAT_INTERVAL", 0.01),
        patch("agm.queue.publish_event") as mock_publish_event,
        pytest.raises(TimeoutError),
    ):
        await _wait_with_heartbeat(
            turn_done,
            event_context,
            thread_id="thread-42",
            turn_state=turn_state,
            timeout=0.03,
        )

    assert mock_publish_event.call_count >= 1
    heartbeat = mock_publish_event.call_args_list[0]
    assert heartbeat.args[:3] == ("task:heartbeat", "task-42", "working")
    thread_context = heartbeat.kwargs["extra"]["thread_context"]
    assert thread_context["thread_id"] == "thread-42"
    assert thread_context["active_turn_id"] == "turn-42"
    assert thread_context["owner_role"] == "executor"


def test_trace_context_continues_ordinals_from_existing_events(db_conn):
    """TraceContext should continue ordinals from the current max for the entity."""
    conn = db_conn
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="trace-task-order",
        stage="execution",
        turn_index=0,
        ordinal=4,
        event_type="turn",
        status="started",
        data={"n": 1},
    )
    add_trace_event(
        conn,
        entity_type="task",
        entity_id="trace-task-order",
        stage="execution",
        turn_index=0,
        ordinal=0,
        event_type="turn",
        status="completed",
        data={"n": 2},
    )
    from agm.tracing import TraceContext

    trace_ctx_a = TraceContext(
        entity_type="task",
        entity_id="trace-task-order",
        stage="execution",
        plan_id="plan-1",
        project="testproj",
        conn=conn,
    )
    trace_ctx_b = TraceContext(
        entity_type="task",
        entity_id="trace-task-order",
        stage="execution",
        plan_id="plan-1",
        project="testproj",
        conn=conn,
    )
    trace_ctx_a.record("turn", "completed", {"n": 3})
    trace_ctx_b.record("turn", "completed", {"n": 4})

    events = list_trace_events(conn, "task", "trace-task-order")
    assert [e["ordinal"] for e in events] == [0, 4, 5, 6]


@pytest.mark.asyncio
async def test_codex_turn_structured_http_connection_failed_error():
    """_codex_turn should include HTTP status in error message for connection failures."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(
        failed_message="Connection refused",
        failed_error_info={"httpConnectionFailed": {"httpStatusCode": 502}},
    )

    with pytest.raises(RuntimeError, match=r"\[httpConnectionFailed\].*HTTP 502"):
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
            start_thread_params={"cwd": "/tmp/test"},
        )


@pytest.mark.asyncio
async def test_codex_turn_plain_string_codex_error_info():
    """_codex_turn should include plain string error types in message."""
    from agm.jobs import _codex_turn

    client = _FakeCodexTurnClient(
        failed_message="Server is overloaded",
        failed_error_info="serverOverloaded",
    )

    with pytest.raises(RuntimeError, match=r"\[serverOverloaded\]"):
        await _codex_turn(
            client,
            prompt="do work",
            turn_config={},
            start_thread_params={"cwd": "/tmp/test"},
        )


# -- _classify_codex_error / _format_turn_error unit tests --


@pytest.mark.parametrize(
    "error_info, expected",
    [
        ("contextWindowExceeded", ("contextWindowExceeded", {})),
        ("other", ("other", {})),
        (
            {"modelCap": {"model": "gpt-5.3", "reset_after_seconds": 60}},
            ("modelCap", {"model": "gpt-5.3", "reset_after_seconds": 60}),
        ),
        (
            {"httpConnectionFailed": {"httpStatusCode": 503}},
            ("httpConnectionFailed", {"httpStatusCode": 503}),
        ),
        (None, ("unknown", {})),
    ],
)
def test_classify_codex_error(error_info, expected):
    from agm.jobs_common import _classify_codex_error

    assert _classify_codex_error(error_info) == expected


def test_format_turn_error_no_codex_info():
    from agm.jobs_common import _format_turn_error

    assert _format_turn_error({"message": "boom"}) == "Turn failed: boom"


@pytest.mark.parametrize(
    "reset_seconds, expect_resets",
    [(45, True), (None, False)],
    ids=["with_reset", "no_reset"],
)
def test_format_turn_error_model_cap(reset_seconds, expect_resets):
    from agm.jobs_common import _format_turn_error

    cap = {"model": "gpt-5.3"}
    if reset_seconds is not None:
        cap["reset_after_seconds"] = reset_seconds
    error = {"message": "Rate limit", "codexErrorInfo": {"modelCap": cap}}
    result = _format_turn_error(error)
    assert "[modelCap]" in result
    assert "model=gpt-5.3" in result
    if expect_resets:
        assert f"resets in {reset_seconds}s" in result
    else:
        assert "resets" not in result


def test_format_turn_error_plain_string_error_type():
    from agm.jobs_common import _format_turn_error

    error = {"message": "context too large", "codexErrorInfo": "contextWindowExceeded"}
    result = _format_turn_error(error)
    assert "[contextWindowExceeded]" in result
    assert "context too large" in result


# -- get_failed_jobs --


def test_get_failed_jobs():
    """get_failed_jobs with a specific queue returns its failed jobs."""
    mock_job = MagicMock()
    mock_job.id = "plan-abc123"
    mock_job.description = "Plan abc123"
    mock_job.exc_info = "RuntimeError: something broke"
    mock_job.enqueued_at = None
    mock_job.ended_at = None

    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.FailedJobRegistry") as MockFailed,
        patch("agm.queue.Job") as MockJob,
    ):
        mock_conn = MagicMock()
        mock_redis.return_value = mock_conn
        MockQueue.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_job_ids.return_value = ["plan-abc123"]
        MockFailed.return_value = mock_registry
        MockJob.fetch.return_value = mock_job

        from agm.queue import get_failed_jobs

        jobs = get_failed_jobs("agm:plans")

    assert len(jobs) == 1
    assert jobs[0]["id"] == "plan-abc123"
    assert "something broke" in jobs[0]["exc_info"]


def test_get_failed_jobs_query_mode_description():
    """get_failed_jobs normalizes exec descriptions for query tasks."""
    mock_job = MagicMock()
    mock_job.id = "exec-abc123"
    mock_job.description = "Execute task abc123"
    mock_job.exc_info = "RuntimeError: something broke"
    mock_job.enqueued_at = None
    mock_job.ended_at = None

    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.FailedJobRegistry") as MockFailed,
        patch("agm.queue.Job") as MockJob,
        patch("agm.queue._resolve_failed_job_description", return_value="Query task abc123"),
    ):
        mock_redis.return_value = MagicMock()
        MockQueue.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_job_ids.return_value = ["exec-abc123"]
        MockFailed.return_value = mock_registry
        MockJob.fetch.return_value = mock_job

        from agm.queue import get_failed_jobs

        jobs = get_failed_jobs("agm:exec")

    assert len(jobs) == 1
    assert jobs[0]["description"] == "Query task abc123"


def test_get_failed_jobs_all_queues():
    """get_failed_jobs without args iterates all queues."""
    mock_job = MagicMock()
    mock_job.id = "plan-abc123"
    mock_job.description = "Plan abc123"
    mock_job.exc_info = "RuntimeError: something broke"
    mock_job.enqueued_at = None
    mock_job.ended_at = None

    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.FailedJobRegistry") as MockFailed,
        patch("agm.queue.Job") as MockJob,
    ):
        mock_redis.return_value = MagicMock()
        MockQueue.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_job_ids.return_value = ["plan-abc123"]
        MockFailed.return_value = mock_registry
        MockJob.fetch.return_value = mock_job

        from agm.queue import get_failed_jobs

        jobs = get_failed_jobs()

    from agm.queue import AGM_QUEUE_NAMES

    # One failed mock job returned per configured queue.
    assert len(jobs) == len(AGM_QUEUE_NAMES)


def test_get_failed_jobs_empty():
    """get_failed_jobs should return empty list when no failures."""
    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.FailedJobRegistry") as MockFailed,
    ):
        mock_redis.return_value = MagicMock()
        MockQueue.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_job_ids.return_value = []
        MockFailed.return_value = mock_registry

        from agm.queue import get_failed_jobs

        jobs = get_failed_jobs()

    assert jobs == []


# -- on_plan_request_failure --


def test_on_plan_request_failure_logs_to_db(db_conn):
    """Failure callback should persist an ERROR log entry."""
    from agm.db import list_plan_logs

    conn = db_conn
    p = make_plan(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [p["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_plan_request_failure

        on_plan_request_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["status"] == "failed"
    history = list_status_history(verify_conn, entity_type="plan", entity_id=p["id"])
    assert [(row["old_status"], row["new_status"]) for row in history] == [("pending", "failed")]
    assert [row["actor"] for row in history] == [None]
    logs = list_plan_logs(verify_conn, p["id"])
    assert len(logs) == 1
    assert logs[0]["level"] == "ERROR"
    assert "boom" in logs[0]["message"]
    verify_conn.close()


def test_on_plan_request_failure_from_awaiting_input_records_transition(db_conn):
    """Failure callback should record awaiting_input->failed when status changes."""
    conn = db_conn
    p = make_plan(conn)
    conn.execute("UPDATE plans SET status = 'awaiting_input' WHERE id = ?", (p["id"],))
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [p["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_plan_request_failure

        on_plan_request_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["status"] == "failed"
    history = list_status_history(verify_conn, entity_type="plan", entity_id=p["id"])
    assert [(row["old_status"], row["new_status"]) for row in history] == [
        ("awaiting_input", "failed")
    ]
    assert [row["actor"] for row in history] == [None]
    verify_conn.close()


def test_on_plan_request_failure_noop_does_not_append_status_history(db_conn):
    """Failure callback should not append status_history when status is already failed."""
    conn = db_conn
    p = make_plan(conn)
    conn.execute("UPDATE plans SET status = 'failed' WHERE id = ?", (p["id"],))
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [p["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_plan_request_failure

        on_plan_request_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_plan_request(verify_conn, p["id"])
    assert found["status"] == "failed"
    assert list_status_history(verify_conn, entity_type="plan", entity_id=p["id"]) == []
    logs = list_plan_logs(verify_conn, p["id"])
    assert len(logs) == 1
    verify_conn.close()


def test_on_plan_request_failure_nonexistent_plan_skips_status_history(db_conn):
    """Failure callback should not append history when the plan row does not exist."""
    conn = db_conn
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = ["missing-plan-id"]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_plan_request_failure

        on_plan_request_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    total = verify_conn.execute("SELECT COUNT(*) AS cnt FROM status_history").fetchone()["cnt"]
    assert total == 0
    verify_conn.close()


# -- enqueue_task_creation --


def test_enqueue_task_creation_calls_rq(db_conn):
    """enqueue_task_creation should push to agm:tasks queue."""
    conn = db_conn
    p = make_plan(conn)
    mock_job = MagicMock()
    mock_job.id = f"tasks-{p['id']}"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_TASKS, enqueue_task_creation

        job = enqueue_task_creation(p["id"])

    assert job.id == f"tasks-{p['id']}"
    mock_get_queue.assert_called_once_with(QUEUE_TASKS)
    mock_spawn.assert_called_once_with(QUEUE_TASKS, single=True, job_id=f"tasks-{p['id']}")


# -- single-worker spawn --


def test_spawn_worker_single_skips_when_active():
    """_spawn_worker(single=True) should skip if a worker is already running."""
    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.StartedJobRegistry") as MockStarted,
        patch("agm.queue.subprocess") as mock_subprocess,
    ):
        mock_redis.return_value = MagicMock()
        MockQueue.return_value = MagicMock()
        mock_started = MagicMock()
        mock_started.__len__ = MagicMock(return_value=1)
        MockStarted.return_value = mock_started

        from agm.queue import QUEUE_TASKS, _spawn_worker

        _spawn_worker(QUEUE_TASKS, single=True)

    mock_subprocess.Popen.assert_not_called()


def test_spawn_worker_single_spawns_when_idle():
    """_spawn_worker(single=True) should spawn if no worker is running and lock is free."""
    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.StartedJobRegistry") as MockStarted,
        patch("agm.queue.subprocess") as mock_subprocess,
    ):
        mock_conn = MagicMock()
        mock_conn.set.return_value = True  # lock acquired
        mock_redis.return_value = mock_conn
        MockQueue.return_value = MagicMock()
        mock_started = MagicMock()
        mock_started.__len__ = MagicMock(return_value=0)
        MockStarted.return_value = mock_started

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.DEVNULL = subprocess.DEVNULL

        from agm.queue import QUEUE_TASKS, _spawn_worker

        _spawn_worker(QUEUE_TASKS, single=True)

    mock_subprocess.Popen.assert_called_once()
    mock_conn.set.assert_called_once_with("agm:worker-lock:agm:tasks", "1", nx=True, ex=300)


def test_spawn_worker_single_skips_when_lock_held():
    """_spawn_worker(single=True) should skip if the Redis lock is already held."""
    with (
        patch("agm.queue.get_redis") as mock_redis,
        patch("agm.queue.Queue") as MockQueue,
        patch("agm.queue.StartedJobRegistry") as MockStarted,
        patch("agm.queue.subprocess") as mock_subprocess,
    ):
        mock_conn = MagicMock()
        mock_conn.set.return_value = False  # lock NOT acquired
        mock_redis.return_value = mock_conn
        MockQueue.return_value = MagicMock()
        mock_started = MagicMock()
        mock_started.__len__ = MagicMock(return_value=0)
        MockStarted.return_value = mock_started

        from agm.queue import QUEUE_TASKS, _spawn_worker

        _spawn_worker(QUEUE_TASKS, single=True)

    mock_subprocess.Popen.assert_not_called()


# -- run_task_creation --


def test_run_task_creation_not_finalized(db_conn):
    """run_task_creation should reject non-finalized plans."""
    conn = db_conn
    p = make_plan(conn)
    # Don't finalize — leave as pending
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import run_task_creation

        with pytest.raises(ValueError, match="not 'finalized'"):
            run_task_creation(p["id"])


def test_run_task_creation_happy_path(db_conn):
    """run_task_creation should create tasks from mocked codex output."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], json.dumps({"title": "T", "summary": "S", "tasks": []}))
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "Task A",
                    "description": "Do A",
                    "files": ["a.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                },
            ],
            "cancel_tasks": [],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_task_creation._run_task_creation_codex",
            return_value=mock_output,
        ),
    ):
        from agm.jobs import run_task_creation

        result = run_task_creation(p["id"])

    assert result == mock_output
    verify_conn = get_connection(Path(db_path))
    plan = get_plan_request(verify_conn, p["id"])
    assert plan["task_creation_status"] == "completed"
    verify_conn.close()


def test_run_task_creation_persists_resolved_think_model(db_conn):
    """run_task_creation should persist the resolved think model."""
    import json

    conn = db_conn
    p = make_plan(conn)
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"think_model": "gpt-5.3-codex", "think_effort": "low"}),
    )
    finalize_plan_request(conn, p["id"], json.dumps({"title": "T", "summary": "S", "tasks": []}))
    mock_output = json.dumps({"tasks": [], "cancel_tasks": []})
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_task_creation._run_task_creation_codex",
            return_value=mock_output,
        ),
    ):
        from agm.jobs import run_task_creation

        run_task_creation(p["id"])

    verify_conn = get_connection(Path(db_path))
    plan = get_plan_request(verify_conn, p["id"])
    assert plan["model"] == "gpt-5.3-codex"
    verify_conn.close()


def test_run_task_creation_failure_logs_traceback(db_conn):
    """run_task_creation should log full traceback to plan_logs on failure."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], json.dumps({"title": "T", "summary": "S", "tasks": []}))
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_task_creation._run_task_creation_codex",
            side_effect=RuntimeError("task agent crashed"),
        ),
    ):
        from agm.jobs import run_task_creation

        with pytest.raises(RuntimeError, match="task agent crashed"):
            run_task_creation(p["id"])

    verify_conn = get_connection(Path(db_path))
    plan = get_plan_request(verify_conn, p["id"])
    assert plan["task_creation_status"] == "failed"
    logs = list_plan_logs(verify_conn, p["id"])
    error_logs = [entry for entry in logs if entry["level"] == "ERROR"]
    assert len(error_logs) >= 1
    error_text = error_logs[0]["message"]
    assert "task agent crashed" in error_text
    assert "Traceback" in error_text
    verify_conn.close()


@pytest.mark.asyncio
async def test_task_creation_prompt_includes_effective_priority_in_existing_summary(db_conn):
    """Task creation prompt should include effective priority (NULL -> medium)."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], json.dumps({"title": "T", "summary": "S", "tasks": []}))

    create_task(
        conn,
        plan_id=p["id"],
        ordinal=1,
        title="High task",
        description="d",
        priority="high",
    )
    create_task(
        conn,
        plan_id=p["id"],
        ordinal=2,
        title="Implicit medium task",
        description="d",
    )
    plan = get_plan_request(conn, p["id"])

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return (
            "thread-1",
            '{"tasks":[{"ordinal":0,"title":"A","description":"d"}]}',
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_creation_codex_async

        await _run_task_creation_codex_async(conn, plan)

    assert 'title="High task", status=blocked, priority=high' in captured["prompt"]
    assert 'title="Implicit medium task", status=blocked, priority=medium' in captured["prompt"]


@pytest.mark.asyncio
async def test_task_creation_codex_uses_resolved_model_config_in_runtime_thread(db_conn):
    """Task creation should use resolved Codex work model and effort."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"work_model": "gpt-5.3-codex-spark", "work_effort": "low"}),
    )
    plan_payload = json.dumps({"title": "T", "summary": "S", "tasks": []})
    finalize_plan_request(conn, p["id"], plan_payload)
    plan = get_plan_request(conn, p["id"])

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-task",
            '{"tasks":[],"cancel_tasks":[]}',
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
    ):
        from agm.jobs import _run_task_creation_codex_async

        result = await _run_task_creation_codex_async(conn, plan)

    assert result == '{"tasks":[],"cancel_tasks":[]}'
    start_thread_params = captured["start_thread_params"]
    assert start_thread_params["model"] == "gpt-5.3-codex-spark"
    assert captured["turn_config"]["effort"] == "low"


@pytest.mark.asyncio
async def test_task_creation_codex_prompt_is_exact_payload(db_conn):
    """Codex task creation should send exact plan payload + TASK_PROMPT_SUFFIX."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    plan_payload = json.dumps({"title": "T", "summary": "S", "tasks": []})
    finalize_plan_request(conn, p["id"], plan_payload)
    plan = get_plan_request(conn, p["id"])
    expected = f"Plan JSON:\n```json\n{plan_payload}\n```{TASK_PROMPT_SUFFIX}"

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-1",
            '{"tasks":[],"cancel_tasks":[]}',
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_creation_codex_async

        await _run_task_creation_codex_async(conn, plan)

    assert captured["prompt"] == expected


@pytest.mark.asyncio
async def test_task_creation_codex_prompt_includes_task_agent_instructions(db_conn):
    """Task-agent instructions injected via developerInstructions for Codex."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    plan_payload = json.dumps({"title": "T", "summary": "S", "tasks": []})
    finalize_plan_request(conn, p["id"], plan_payload)
    plan = get_plan_request(conn, p["id"])
    instructions = "Task agent rules: keep tasks short."
    expected_prompt = f"Plan JSON:\n```json\n{plan_payload}\n```{TASK_PROMPT_SUFFIX}"

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-1",
            '{"tasks":[],"cancel_tasks":[]}',
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_task_creation_codex_async

        await _run_task_creation_codex_async(conn, plan)

    assert captured["prompt"] == expected_prompt
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


@pytest.mark.asyncio
async def test_task_refresh_prompt_includes_effective_priority_in_summary(db_conn):
    """Refresh prompt should include effective priority (NULL -> medium)."""
    import contextlib

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], '{"title":"t","summary":"s","tasks":[]}')
    create_task(
        conn,
        plan_id=p["id"],
        ordinal=0,
        title="Needs refresh medium",
        description="d",
    )
    create_task(
        conn,
        plan_id=p["id"],
        ordinal=1,
        title="Needs refresh low",
        description="d",
        priority="low",
    )

    project_id = get_project_id(conn)
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    project = dict(row)

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "thread-1", '{"tasks":[],"cancel_tasks":[]}', {"input_tokens": 0, "output_tokens": 0}

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_refresh_codex_async

        await _run_task_refresh_codex_async(conn, project, "refresh now")

    assert 'title="Needs refresh medium", status=blocked, priority=medium' in captured["prompt"]
    assert 'title="Needs refresh low", status=blocked, priority=low' in captured["prompt"]


@pytest.mark.asyncio
async def test_task_refresh_codex_prompt_includes_task_agent_instructions(db_conn):
    """Task refresh instructions injected via developerInstructions for Codex."""
    import contextlib

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], '{"title":"t","summary":"s","tasks":[]}')
    task_low = create_task(
        conn,
        plan_id=p["id"],
        ordinal=0,
        title="Needs refresh medium",
        description="d",
    )
    task_high = create_task(
        conn,
        plan_id=p["id"],
        ordinal=1,
        title="Needs refresh low",
        description="d",
        priority="low",
    )

    project_id = get_project_id(conn)
    project = dict(conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())
    instructions = "Refresh-only task-agent guidance."
    expected_prompt = (
        "refresh now\n\n"
        "Current tasks for this project:\n"
        f'  - id={task_low["id"]}, title="Needs refresh medium", status=blocked, '
        f"priority=medium, plan_id={p['id']}\n"
        f'  - id={task_high["id"]}, title="Needs refresh low", status=blocked, '
        f"priority=low, plan_id={p['id']}"
        f"{REFRESH_PROMPT_SUFFIX}"
    )

    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return "thread-1", '{"tasks":[],"cancel_tasks":[]}', {"input_tokens": 0, "output_tokens": 0}

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_task_refresh_codex_async

        await _run_task_refresh_codex_async(conn, project, "refresh now")

    assert captured["prompt"] == expected_prompt
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


@pytest.mark.asyncio
async def test_task_refresh_codex_uses_resolved_model_config_in_runtime_thread(db_conn):
    """Task refresh should use resolved Codex work model and effort."""
    import contextlib
    import json

    conn = db_conn
    p = make_plan(conn)
    project_id = get_project_id(conn)
    set_project_model_config(
        conn,
        project_id,
        json.dumps({"work_model": "gpt-5.3-codex-spark", "work_effort": "low"}),
    )
    finalize_plan_request(conn, p["id"], '{"title":"t","summary":"s","tasks":[]}')
    create_task(
        conn,
        plan_id=p["id"],
        ordinal=0,
        title="Needs refresh medium",
        description="d",
    )
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-refresh",
            '{"tasks":[],"cancel_tasks":[]}',
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
    ):
        from agm.jobs import _run_task_refresh_codex_async

        await _run_task_refresh_codex_async(conn, dict(project), "refresh now")

    assert captured["start_thread_params"]["model"] == "gpt-5.3-codex-spark"
    # Effort is on turn_config (0.102.0 flattened format), not thread params
    assert captured["turn_config"]["effort"] == "low"


@pytest.mark.asyncio
async def test_task_creation_logs_cross_plan_file_overlap_warning(db_conn, caplog):
    """Task creation should warn when proposed files overlap active tasks from other plans."""
    import contextlib
    import json

    conn = db_conn
    p_existing = make_plan(conn, "existing plan")
    p_new = make_plan(conn, "new plan")
    finalize_plan_request(conn, p_existing["id"], json.dumps({"title": "A", "tasks": []}))
    finalize_plan_request(conn, p_new["id"], json.dumps({"title": "B", "tasks": []}))

    create_task(
        conn,
        plan_id=p_existing["id"],
        ordinal=0,
        title="Existing active",
        description="d",
        files=json.dumps(["src/shared.py"]),
    )

    plan = get_plan_request(conn, p_new["id"])
    assert plan is not None

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **_kwargs):
        return (
            "thread-1",
            json.dumps(
                {
                    "tasks": [
                        {
                            "ordinal": 0,
                            "title": "New task",
                            "description": "d",
                            "files": ["src/shared.py"],
                            "status": "ready",
                        }
                    ],
                    "cancel_tasks": [],
                }
            ),
            {"input_tokens": 0, "output_tokens": 0},
        )

    caplog.set_level(logging.WARNING, logger="agm.jobs_task_creation")
    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=1),
    ):
        from agm.jobs import _run_task_creation_codex_async

        await _run_task_creation_codex_async(conn, plan)

    assert any("Cross-plan file overlap" in record.message for record in caplog.records)


# -- _insert_tasks_from_output --


def test_insert_tasks_from_output(db_conn):
    """Valid JSON should produce correct DB rows."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "First",
                    "description": "Do first",
                    "files": ["x.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                },
                {
                    "ordinal": 1,
                    "title": "Second",
                    "description": "Do second",
                    "files": [],
                    "blocked_by": [0],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                },
            ],
            "cancel_tasks": [],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    count = _insert_tasks_from_output(conn, p["id"], output)
    assert count == 2

    tasks = list_tasks(conn, plan_id=p["id"])
    assert len(tasks) == 2
    assert tasks[0]["status"] == "ready"
    assert tasks[1]["status"] == "pending"


def test_insert_tasks_from_output_invalid_ordinal(db_conn):
    """Invalid ordinal reference should warn but not crash."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "Only task",
                    "description": "Has bad dep",
                    "files": [],
                    "blocked_by": [99],
                    "blocked_by_existing": ["nonexistent-id"],
                    "external_blockers": [],
                    "status": "ready",
                },
            ],
            "cancel_tasks": [],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    # Should not raise — invalid refs are skipped
    count = _insert_tasks_from_output(conn, p["id"], output)
    assert count == 1


def test_insert_tasks_no_ready_forces_first(db_conn):
    """If no tasks are ready, the first should be forced to ready."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "All pending",
                    "description": "No one is ready",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [
                        {"factor": "design", "reason": "waiting"},
                    ],
                    "status": "pending",
                },
            ],
            "cancel_tasks": [],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    count = _insert_tasks_from_output(conn, p["id"], output)
    assert count == 1

    tasks = list_tasks(conn, plan_id=p["id"])
    assert tasks[0]["status"] == "ready"


def test_insert_tasks_with_cancellations(db_conn):
    """Output with cancel_tasks should cancel existing tasks."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    # Create an existing task to cancel
    old_task = create_task(conn, plan_id=p["id"], ordinal=0, title="Old task", description="stale")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "Replacement",
                    "description": "New version",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                },
            ],
            "cancel_tasks": [
                {"task_id": old_task["id"], "reason": "superseded by new plan"},
            ],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    count = _insert_tasks_from_output(conn, p["id"], output)
    assert count == 1

    cancelled = get_task(conn, old_task["id"])
    assert cancelled["status"] == "cancelled"


def test_insert_tasks_from_output_passes_normalized_priority_to_batch(db_conn):
    """Task output priorities should be normalized before create_tasks_batch."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "A",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "priority": " HIGH ",
                },
                {
                    "ordinal": 1,
                    "title": "B",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "priority": "Medium",
                },
                {
                    "ordinal": 2,
                    "title": "C",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "priority": "low",
                },
            ],
            "cancel_tasks": [],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    captured = {}

    def _fake_create_tasks_batch(_conn, _plan_id, tasks_data):
        captured["tasks_data"] = tasks_data
        return {0: "new-0", 1: "new-1", 2: "new-2"}

    with (
        patch("agm.jobs_task_creation.create_tasks_batch", side_effect=_fake_create_tasks_batch),
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 3
    priorities = [t["priority"] for t in captured["tasks_data"]]
    assert priorities == ["high", None, "low"]


def test_insert_tasks_from_output_warns_on_bucket_overlap_mismatch(db_conn, caplog):
    """Shared files across different/null buckets should log WARNING findings."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "API task",
                    "description": "d",
                    "files": ["src/shared.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "bucket": "api",
                },
                {
                    "ordinal": 1,
                    "title": "No bucket overlap",
                    "description": "d",
                    "files": ["src/shared.py", "src/other.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "bucket": None,
                },
            ],
            "cancel_tasks": [],
        }
    )

    caplog.set_level(logging.INFO, logger="agm.jobs_task_creation")
    with (
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 2
    assert any(
        record.levelname == "WARNING" and "Bucket/file overlap mismatch" in record.message
        for record in caplog.records
    )


def test_insert_tasks_from_output_warns_on_null_bucket_overlap(db_conn, caplog):
    """Two tasks sharing files with both buckets None should still warn."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "Task A",
                    "description": "d",
                    "files": ["src/shared.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "bucket": None,
                },
                {
                    "ordinal": 1,
                    "title": "Task B",
                    "description": "d",
                    "files": ["src/shared.py", "src/other.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "bucket": None,
                },
            ],
            "cancel_tasks": [],
        }
    )

    caplog.set_level(logging.INFO, logger="agm.jobs_task_creation")
    with (
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 2
    assert any(
        record.levelname == "WARNING" and "Bucket/file overlap mismatch" in record.message
        for record in caplog.records
    )


def test_insert_tasks_from_output_infos_on_same_bucket_without_overlap(db_conn, caplog):
    """Same bucket with zero file overlap should log INFO findings."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "UI layout",
                    "description": "d",
                    "files": ["ui/layout.tsx"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "bucket": "ui",
                },
                {
                    "ordinal": 1,
                    "title": "UI state",
                    "description": "d",
                    "files": ["ui/state.ts"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "bucket": "ui",
                },
            ],
            "cancel_tasks": [],
        }
    )

    caplog.set_level(logging.INFO, logger="agm.jobs_task_creation")
    with (
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 2
    assert any(
        record.levelname == "INFO" and "Bucket grouping without shared files" in record.message
        for record in caplog.records
    )


def test_insert_tasks_from_output_clean_bucket_assignments_emit_no_findings(db_conn, caplog):
    """Clean bucket/file assignments should not emit bucket verification messages."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "Auth API",
                    "description": "d",
                    "files": ["src/auth.py", "src/common.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "bucket": "api",
                },
                {
                    "ordinal": 1,
                    "title": "Auth worker",
                    "description": "d",
                    "files": ["src/common.py", "src/worker.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "bucket": "api",
                },
                {
                    "ordinal": 2,
                    "title": "UI page",
                    "description": "d",
                    "files": ["ui/page.tsx"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "bucket": "ui",
                },
            ],
            "cancel_tasks": [],
        }
    )

    caplog.set_level(logging.INFO, logger="agm.jobs_task_creation")
    with (
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 3
    assert not any("Bucket/file overlap mismatch" in record.message for record in caplog.records)
    assert not any(
        "Bucket grouping without shared files" in record.message for record in caplog.records
    )


def test_insert_tasks_cancellations_only(db_conn):
    """Output with only cancellations should trigger promoted dependents."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    old_task = create_task(conn, plan_id=p["id"], ordinal=0, title="Old task", description="stale")
    dependent = create_task(
        conn, plan_id=p["id"], ordinal=1, title="Blocked", description="waiting on old"
    )
    add_task_block(conn, task_id=dependent["id"], blocked_by_task_id=old_task["id"])

    output = json.dumps(
        {
            "tasks": [],
            "cancel_tasks": [
                {"task_id": old_task["id"], "reason": "no longer needed"},
            ],
        }
    )

    with patch("agm.jobs_merge._trigger_task_execution") as mock_trigger:
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 0

    cancelled = get_task(conn, old_task["id"])
    assert cancelled["status"] == "cancelled"
    # Batch cancel no longer promotes — dependent stays blocked
    dep = get_task(conn, dependent["id"])
    assert dep["status"] == "blocked"
    mock_trigger.assert_not_called()


def test_insert_tasks_with_cancellations_and_stale_promotions_sorted(db_conn):
    """Stale sweeps + created tasks should keep priority ordering."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    old_task = create_task(
        conn,
        plan_id=p["id"],
        ordinal=0,
        title="Old task",
        description="superseded",
    )

    stale_promoted_high = create_task(
        conn,
        plan_id=p["id"],
        ordinal=1,
        title="Stale high",
        description="ready after stale",
        priority="high",
    )
    stale_promoted_medium = create_task(
        conn,
        plan_id=p["id"],
        ordinal=2,
        title="Stale medium",
        description="ready after stale",
        priority="medium",
    )

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 10,
                    "title": "Low priority",
                    "description": "output-only",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "priority": "low",
                }
            ],
            "cancel_tasks": [
                {"task_id": old_task["id"], "reason": "superseded by output"},
            ],
        }
    )

    with (
        patch(
            "agm.jobs_task_creation.resolve_stale_blockers",
            return_value=[stale_promoted_high["id"], stale_promoted_medium["id"]],
        ),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 1
    tasks_by_ordinal = {task["ordinal"]: task for task in list_tasks(conn, plan_id=p["id"])}
    created = tasks_by_ordinal[10]
    # Priority ordering: high → medium → low
    assert [call.args[1] for call in mock_trigger.call_args_list] == [
        stale_promoted_high["id"],
        stale_promoted_medium["id"],
        created["id"],
    ]


def test_insert_tasks_empty_tasks_and_no_cancellations_raises(db_conn):
    """Agent producing no tasks and no cancellations should raise RuntimeError."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps({"tasks": [], "cancel_tasks": []})

    from agm.jobs import _insert_tasks_from_output

    with pytest.raises(RuntimeError, match="no tasks and no cancellations"):
        _insert_tasks_from_output(conn, p["id"], output)


def test_insert_tasks_invalid_json_raises(db_conn):
    """Non-JSON agent output should raise RuntimeError."""
    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    from agm.jobs import _insert_tasks_from_output

    with pytest.raises(RuntimeError, match="not valid JSON"):
        _insert_tasks_from_output(conn, p["id"], "This is not JSON {broken")


def test_insert_tasks_refresh_no_plan_id_warns(db_conn, caplog):
    """Refresh with tasks but no plan_id should warn and return 0."""
    import json
    import logging

    conn = db_conn

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "T",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                }
            ],
            "cancel_tasks": [],
        }
    )

    caplog.set_level(logging.WARNING)

    with patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[]):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, None, output)

    assert count == 0
    assert "Refresh produced new tasks but no plan_id" in caplog.text


# -- enqueue_task_refresh --


def test_enqueue_task_refresh_calls_rq():
    """enqueue_task_refresh should push a refresh job to agm:tasks queue."""
    mock_job = MagicMock()
    mock_job.id = "refresh-proj123"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_TASKS, enqueue_task_refresh

        job = enqueue_task_refresh("proj123", "clean up stale tasks")

    assert job.id == "refresh-proj123"
    mock_get_queue.assert_called_once_with(QUEUE_TASKS)
    mock_spawn.assert_called_once_with(QUEUE_TASKS, single=True, job_id="refresh-proj123")


# -- on_task_creation_failure --


def test_on_task_creation_failure_logs_to_db(db_conn):
    """Failure callback should mark task_creation_status as failed and log."""
    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [p["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_task_creation_failure

        on_task_creation_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    plan = get_plan_request(verify_conn, p["id"])
    assert plan["task_creation_status"] == "failed"
    logs = list_plan_logs(verify_conn, p["id"])
    assert any("boom" in entry["message"] for entry in logs)
    verify_conn.close()


# -- task execution --


def _make_running_task(conn, worktree="/tmp/worktree"):
    """Helper: create a running task with worktree for execution tests."""
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Test task", description="Do it")
    update_task_status(conn, task["id"], "ready")
    claim_task(
        conn,
        task["id"],
        caller="cli",
        branch="agm/test-task",
        worktree=worktree,
    )
    return task


def test_trigger_task_execution_uses_project_base_branch(db_conn):
    """Auto-triggering execution should create worktree from project's configured base branch."""
    conn = db_conn
    pid = get_project_id(conn)
    set_project_base_branch(conn, pid, "release/main")
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="Test task",
        description="Do it",
    )
    update_task_status(conn, task["id"], "ready")

    with (
        patch("agm.git_ops.create_worktree") as mock_create_worktree,
        patch("agm.jobs_merge.claim_task", return_value=True),
        patch("agm.queue.enqueue_task_execution"),
    ):
        mock_create_worktree.return_value = ("agm/test-task", "/tmp/worktree")
        from agm.jobs import _trigger_task_execution

        _trigger_task_execution(conn, task["id"])

    mock_create_worktree.assert_called_once_with(
        "/tmp/testproj", task["id"], "Test task", base_branch="release/main"
    )


def test_run_task_execution_sets_worker_pid(db_conn):
    """run_task_execution should set PID and transition to review on success."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Implemented the feature",
        ),
    ):
        from agm.jobs import run_task_execution

        result = run_task_execution(task["id"])

    assert result == "Implemented the feature"
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["pid"] is not None
    assert found["status"] == "review"
    verify_conn.close()


def test_run_task_execution_persists_resolved_work_model(db_conn):
    """run_task_execution should persist the resolved work model."""
    import json

    conn = db_conn
    task = _make_running_task(conn)
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"work_model": "gpt-5.3-codex-spark", "work_effort": "low"}),
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Implemented the feature",
        ),
        patch("agm.jobs_merge._trigger_task_review"),
    ):
        from agm.jobs import run_task_execution

        run_task_execution(task["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["model"] == "gpt-5.3-codex-spark"
    verify_conn.close()


@pytest.mark.asyncio
async def test_run_task_execution_codex_uses_resolved_work_model_and_fallback_config(db_conn):
    """Codex execution should use resolved work model and fallback by resolved context."""
    import contextlib
    import json

    conn = db_conn
    task = _make_running_task(conn)
    task = get_task(conn, task["id"])
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"work_model": "custom-work-model", "work_effort": "low"}),
    )
    set_project_app_server_ask_for_approval(
        conn,
        get_project_id(conn),
        {"reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}},
    )
    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-work",
            "implemented",
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch(
            "agm.jobs_execution._codex_client",
            _fake_codex_client,
        ),
        patch("agm.jobs_execution._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_execution._has_uncommitted_changes", return_value=False),
        patch("agm.backends.WORK_MODEL_FALLBACK", "fallback-work-model"),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    assert captured["start_thread_params"]["model"] == "custom-work-model"
    assert captured["start_thread_params"]["approvalPolicy"] == {
        "reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}
    }
    assert captured["fallback_thread_params"]["model"] == "fallback-work-model"
    assert captured["fallback_thread_params"]["approvalPolicy"] == {
        "reject": {"mcp_elicitations": True, "rules": True, "sandbox_approval": False}
    }


@pytest.mark.asyncio
async def test_run_task_execution_codex_prompt_includes_executor_instructions_fresh_task(db_conn):
    """Codex execution prompt should include executor instructions for fresh runs."""
    conn = db_conn
    task = get_task(conn, _make_running_task(conn)["id"])
    instructions = "Executor policy: keep diffs small."
    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return "thread-111", "implemented", {"input_tokens": 0, "output_tokens": 0}

    with (
        patch("agm.jobs_execution._codex_client", _fake_codex_client),
        patch("agm.jobs_execution._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_execution._has_uncommitted_changes", return_value=False),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    assert captured["prompt"].startswith("Task: Test task\n\nDo it")
    assert captured["prompt"].endswith(EXECUTOR_PROMPT_SUFFIX)
    assert instructions not in captured["prompt"]
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


@pytest.mark.asyncio
async def test_run_task_execution_codex_rejection_prompt_includes_executor_instructions(db_conn):
    """Codex execution retry prompt should include executor instructions for rejected tasks."""
    conn = db_conn
    task = get_task(conn, _make_running_task(conn)["id"])
    set_task_thread_id(conn, task["id"], "resume-thread")
    add_task_log(
        conn,
        task_id=task["id"],
        level="REVIEW",
        message="Tests are missing",
    )
    task = get_task(conn, task["id"])
    instructions = "Executor retry guidance."
    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return "resume-thread", "implemented", {"input_tokens": 0, "output_tokens": 0}

    with (
        patch("agm.jobs_execution._codex_client", _fake_codex_client),
        patch("agm.jobs_execution._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_execution._has_uncommitted_changes", return_value=False),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    assert "REVIEWER FEEDBACK — your changes were rejected" in captured["prompt"]
    assert "Tests are missing" in captured["prompt"]
    assert captured["prompt"].endswith(EXECUTOR_PROMPT_SUFFIX)
    assert instructions not in captured["prompt"]


def test_run_task_execution_marks_failed_on_error(db_conn):
    """run_task_execution should mark task as failed when backend raises."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            side_effect=RuntimeError("execution error"),
        ),
    ):
        from agm.jobs import run_task_execution

        with pytest.raises(RuntimeError, match="execution error"):
            run_task_execution(task["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "failed"
    reason = json.loads(found["failure_reason"])
    assert reason["source"] == "execution"
    assert reason["task_id"] == task["id"]
    assert reason["exception_type"] == "RuntimeError"
    assert reason["message"] == "execution error"
    assert reason["context"]["path"] == "direct"
    verify_conn.close()


def test_run_task_execution_not_running(db_conn):
    """run_task_execution should reject tasks not in running status."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], "{}")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    # Task is pending, not running
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import run_task_execution

        with pytest.raises(ValueError, match="not 'running'"):
            run_task_execution(task["id"])


def test_run_task_execution_no_worktree(db_conn):
    """run_task_execution should reject tasks without a worktree."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], "{}")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    # Running but no worktree
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import run_task_execution

        with pytest.raises(ValueError, match="no worktree"):
            run_task_execution(task["id"])


def test_run_task_execution_failure_logs(db_conn):
    """run_task_execution should log ERROR with traceback to task_logs."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            side_effect=RuntimeError("agent crashed"),
        ),
    ):
        from agm.jobs import run_task_execution

        with pytest.raises(RuntimeError, match="agent crashed"):
            run_task_execution(task["id"])

    verify_conn = get_connection(Path(db_path))
    logs = list_task_logs(verify_conn, task["id"])
    error_logs = [entry for entry in logs if entry["level"] == "ERROR"]
    assert len(error_logs) >= 1
    error_text = error_logs[0]["message"]
    assert "failed" in error_text.lower()
    verify_conn.close()


def test_run_task_execution_captures_cross_module_warning_logs(db_conn):
    """Execution task logs should include warnings emitted by helper modules."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    def fake_execution(_conn, _task):
        logging.getLogger("agm.jobs_common").warning("cross-module warning marker")
        return "Implemented"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._resolve_execution_result",
            side_effect=fake_execution,
        ),
        patch("agm.jobs_merge._trigger_task_review"),
    ):
        from agm.jobs import run_task_execution

        result = run_task_execution(task["id"])

    assert result == "Implemented"
    verify_conn = get_connection(Path(db_path))
    warning_logs = list_task_logs(verify_conn, task["id"], level="WARNING")
    assert any("cross-module warning marker" in entry["message"] for entry in warning_logs)
    verify_conn.close()


# -- enqueue_task_execution --


def test_enqueue_task_execution_calls_rq():
    """enqueue_task_execution should push to agm:exec queue."""
    mock_job = MagicMock()
    mock_job.id = "exec-task123"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_EXEC, enqueue_task_execution

        job = enqueue_task_execution("task123")

    assert job.id == "exec-task123"
    mock_get_queue.assert_called_once_with(QUEUE_EXEC)
    mock_spawn.assert_called_once_with(QUEUE_EXEC, job_id="exec-task123")


def test_enqueue_task_execution_default_uses_exec():
    """enqueue_task_execution with default mode should use agm:exec queue."""
    mock_job = MagicMock()
    mock_job.id = "exec-task789"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_EXEC, enqueue_task_execution

        job = enqueue_task_execution("task789")

    assert job.id == "exec-task789"
    mock_get_queue.assert_called_once_with(QUEUE_EXEC)
    mock_spawn.assert_called_once_with(QUEUE_EXEC, job_id="exec-task789")


# -- on_task_execution_failure --


def test_on_task_execution_failure_logs_to_db(db_conn):
    """Failure callback should mark task as failed and log error."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [task["id"]]

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
    ):
        from agm.jobs import on_task_execution_failure

        on_task_execution_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "failed"
    reason = json.loads(found["failure_reason"])
    assert reason["source"] == "execution"
    assert reason["task_id"] == task["id"]
    assert reason["exception_type"] == "RuntimeError"
    assert reason["message"] == "boom"
    assert reason["context"]["path"] == "callback"
    logs = list_task_logs(verify_conn, task["id"])
    assert any("boom" in entry["message"] for entry in logs)
    verify_conn.close()


def test_on_task_execution_failure_uses_exception_type_when_message_empty(db_conn):
    """Failure callback should record exception class name when message is empty."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [task["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_task_execution_failure

        on_task_execution_failure(mock_job, None, TimeoutError, TimeoutError(), None)

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    reason = json.loads(found["failure_reason"])
    assert reason["exception_type"] == "TimeoutError"
    assert reason["message"] == "TimeoutError"
    logs = list_task_logs(verify_conn, task["id"])
    assert any("TimeoutError" in entry["message"] for entry in logs)
    verify_conn.close()


# -- TaskDBHandler --


def test_task_db_handler(db_conn):
    """TaskDBHandler should persist log records to task_logs."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], "{}")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")

    from agm.jobs import TaskDBHandler

    handler = TaskDBHandler(conn, task["id"])
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("test_task_db_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("test message from handler")

    logs = list_task_logs(conn, task["id"])
    assert len(logs) == 1
    assert logs[0]["level"] == "INFO"
    assert "test message" in logs[0]["message"]

    logger.removeHandler(handler)


# -- _has_uncommitted_changes --


def test_has_uncommitted_changes_clean(tmp_path):
    """Clean worktree should return False."""
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    from agm.jobs import _has_uncommitted_changes

    assert _has_uncommitted_changes(str(tmp_path)) is False


def test_has_uncommitted_changes_dirty(tmp_path):
    """Worktree with uncommitted files should return True."""
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    (tmp_path / "new_file.py").write_text("hello")

    from agm.jobs import _has_uncommitted_changes

    assert _has_uncommitted_changes(str(tmp_path)) is True


def test_has_uncommitted_changes_staged(tmp_path):
    """Worktree with staged but uncommitted changes should return True."""
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    (tmp_path / "staged.py").write_text("hello")
    subprocess.run(
        ["git", "add", "staged.py"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    from agm.jobs import _has_uncommitted_changes

    assert _has_uncommitted_changes(str(tmp_path)) is True


# -- commit nudge --


def test_run_task_execution_nudges_on_uncommitted(db_conn):
    """run_task_execution should nudge and still reach review when changes get committed."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    # First call: dirty, second call (after nudge): clean
    call_count = {"n": 0}

    def mock_uncommitted(worktree):
        call_count["n"] += 1
        return call_count["n"] <= 1  # dirty on first check, clean after nudge

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Implemented and committed",
        ),
    ):
        from agm.jobs import run_task_execution

        result = run_task_execution(task["id"])

    assert result == "Implemented and committed"
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "review"
    verify_conn.close()


# -- task review --


def _make_review_task(conn, worktree="/tmp/worktree"):
    """Helper: create a task in review status with worktree for review tests."""
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Test task", description="Do it")
    update_task_status(conn, task["id"], "ready")
    claim_task(
        conn,
        task["id"],
        caller="cli",
        branch="agm/test-task",
        worktree=worktree,
    )
    update_task_status(conn, task["id"], "review")
    return task


def test_run_task_review_approves(db_conn):
    """run_task_review should transition to approved on approve verdict."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "verdict": "approve",
            "summary": "Looks good",
            "findings": [],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_review._run_task_review_codex",
            return_value=mock_output,
        ),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import run_task_review

        result = run_task_review(task["id"])

    assert result == mock_output
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"
    verify_conn.close()


def test_run_task_review_rejects(db_conn):
    """run_task_review should transition to running on reject verdict + log findings."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "verdict": "reject",
            "summary": "Missing tests",
            "findings": [
                {"severity": "critical", "file": "src/foo.py", "description": "No test coverage"},
            ],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_review._run_task_review_codex",
            return_value=mock_output,
        ),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import run_task_review

        result = run_task_review(task["id"])

    assert result == mock_output
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "running"
    # Verify REVIEW-level log was created
    logs = list_task_logs(verify_conn, task["id"], level="REVIEW")
    assert len(logs) >= 1
    assert "Missing tests" in logs[-1]["message"]
    assert "No test coverage" in logs[-1]["message"]
    verify_conn.close()


def test_run_task_review_logs_reviewer_model_mismatch(db_conn, caplog):
    """Task review should log when reviewer and executor models differ."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    task_id = task["id"]
    set_task_model(conn, task_id, "executor-model")
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"think_model": "reviewer-model", "think_effort": "low"}),
    )
    mock_output = json.dumps(
        {
            "verdict": "approve",
            "summary": "Looks good",
            "findings": [],
        }
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        caplog.at_level(logging.INFO, logger="agm.jobs_review"),
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_review._run_task_review_codex",
            return_value=mock_output,
        ),
        patch("agm.jobs_merge._trigger_task_merge"),
    ):
        from agm.jobs import run_task_review

        run_task_review(task_id)

    assert (
        f"Task {task_id} reviewer model (reviewer-model) differs from executor "
        f"model (executor-model)" in caplog.text
    )


@pytest.mark.asyncio
async def test_run_task_review_codex_uses_resolved_model_config_in_runtime_thread(db_conn):
    """Task review should use resolved Codex think model and effort (reviewer = think tier)."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    set_project_model_config(
        conn,
        get_project_id(conn),
        json.dumps({"think_model": "gpt-5.3-codex", "think_effort": "low"}),
    )
    task = get_task(conn, task["id"])
    captured = {}

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-review",
            '{"verdict":"approve","summary":"ok","findings":[]}',
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("subprocess.run") as mock_run,
        patch("agm.jobs_review._run_quality_checks", return_value=_qg_pass()),
        patch("agm.jobs_review._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
    ):
        mock_run.side_effect = [
            MagicMock(stdout="diff", stderr="", returncode=0),
            MagicMock(stdout="commit log", stderr="", returncode=0),
        ]
        from agm.jobs import _run_task_review_codex_async

        await _run_task_review_codex_async(conn, task)

    assert captured["start_thread_params"]["model"] == "gpt-5.3-codex"
    assert captured["turn_config"]["effort"] == "low"


@pytest.mark.asyncio
async def test_run_task_review_codex_async_uses_project_base_branch_in_git_commands_and_message(
    db_conn,
):
    """Reviewer git diffs and empty-work log should use project's effective base branch."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    task = get_task(conn, task["id"])
    set_project_base_branch(conn, get_project_id(conn), "release/base")

    with (
        patch("subprocess.run") as mock_run,
        patch("agm.jobs_review._run_quality_checks", return_value=_qg_pass()),
    ):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        from agm.jobs import _run_task_review_codex_async

        result = await _run_task_review_codex_async(conn, task)

    payload = json.loads(result)
    assert "release/base" in payload["findings"][0]["description"]
    commands = [call.args[0] for call in mock_run.call_args_list]
    assert any(cmd[:2] == ["git", "diff"] and "release/base...HEAD" in cmd for cmd in commands)
    assert any(cmd[:2] == ["git", "log"] and "release/base..HEAD" in cmd for cmd in commands)


def test_run_task_review_not_review_status(db_conn):
    """run_task_review should reject tasks not in review status."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import run_task_review

        with pytest.raises(ValueError, match="not 'review'"):
            run_task_review(task["id"])


def test_run_task_review_no_worktree(db_conn):
    """run_task_review should reject tasks without a worktree."""
    conn = db_conn
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], "{}")
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="A", description="a")
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    update_task_status(conn, task["id"], "review")
    # In review but no worktree
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import run_task_review

        with pytest.raises(ValueError, match="no worktree"):
            run_task_review(task["id"])


def test_run_task_review_failure_keeps_review_status(db_conn):
    """Backend crash during review should leave task in review status."""
    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_review._run_task_review_codex",
            side_effect=RuntimeError("codex crashed"),
        ),
    ):
        from agm.jobs import run_task_review

        with pytest.raises(RuntimeError, match="codex crashed"):
            run_task_review(task["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "review"  # NOT failed
    verify_conn.close()


def test_on_task_review_failure_logs_not_status(db_conn):
    """rq failure callback should log error but not change task status."""
    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [task["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_task_review_failure

        on_task_review_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "review"  # Status unchanged
    logs = list_task_logs(verify_conn, task["id"])
    assert any("boom" in entry["message"] for entry in logs)
    verify_conn.close()


# -- enqueue_task_review --


def test_enqueue_task_review_calls_rq():
    """enqueue_task_review should push to agm:review queue."""
    mock_job = MagicMock()
    mock_job.id = "review-task123"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_REVIEW, enqueue_task_review

        job = enqueue_task_review("task123")

    assert job.id == "review-task123"
    mock_get_queue.assert_called_once_with(QUEUE_REVIEW)
    mock_spawn.assert_called_once_with(QUEUE_REVIEW, job_id="review-task123")


# -- executor resume after rejection --


def test_executor_resumes_thread_after_rejection(db_conn):
    """Task with thread_id (after rejection) should use thread/resume."""
    conn = db_conn
    task = _make_running_task(conn)
    # Simulate: executor ran, got rejected, task back to running with thread_id
    set_task_thread_id(conn, task["id"], "exec-thread-123")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Fixed the issues",
        ),
    ):
        from agm.jobs import run_task_execution

        result = run_task_execution(task["id"])

    assert result == "Fixed the issues"
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "review"
    verify_conn.close()


def test_executor_uses_rejection_context(db_conn):
    """Executor prompt should include review findings when resuming after rejection."""
    conn = db_conn
    task = _make_running_task(conn)
    set_task_thread_id(conn, task["id"], "exec-thread-123")
    # Add a REVIEW-level log (from reviewer rejection)
    add_task_log(
        conn,
        task_id=task["id"],
        level="REVIEW",
        message="Missing tests\n\nFindings:\n  [critical] src/foo.py: No test coverage",
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Fixed",
        ),
    ):
        from agm.jobs import run_task_execution

        # This exercises the resume path; we verify it doesn't crash
        # and the task transitions correctly
        result = run_task_execution(task["id"])

    assert result == "Fixed"


def test_executor_fresh_thread_no_prior(db_conn):
    """Task without thread_id should use thread/start (fresh execution)."""
    conn = db_conn
    task = _make_running_task(conn)
    # No thread_id set — first execution
    assert get_task(conn, task["id"])["thread_id"] is None
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Implemented",
        ),
    ):
        from agm.jobs import run_task_execution

        result = run_task_execution(task["id"])

    assert result == "Implemented"
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "review"
    verify_conn.close()


# -- _get_latest_review --


def test_get_latest_review(db_conn):
    """_get_latest_review should return the latest REVIEW-level log."""
    conn = db_conn
    task = _make_running_task(conn)
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="First review")
    add_task_log(conn, task_id=task["id"], level="INFO", message="Some info")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="Second review")

    from agm.jobs import _get_latest_review

    result = _get_latest_review(conn, task["id"])
    assert result == "Second review"


def test_get_latest_review_none(db_conn):
    """_get_latest_review should return None if no REVIEW logs exist."""
    conn = db_conn
    task = _make_running_task(conn)
    add_task_log(conn, task_id=task["id"], level="INFO", message="Some info")

    from agm.jobs import _get_latest_review

    result = _get_latest_review(conn, task["id"])
    assert result is None


# -- enqueue_task_merge --


def test_enqueue_task_merge_calls_rq():
    """enqueue_task_merge should push to agm:merge queue with single-worker."""
    mock_job = MagicMock()
    mock_job.id = "merge-task123"

    with (
        patch("agm.queue.get_queue") as mock_get_queue,
        patch("agm.queue._spawn_worker") as mock_spawn,
    ):
        mock_q = MagicMock()
        mock_q.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_q

        from agm.queue import QUEUE_MERGE, enqueue_task_merge

        job = enqueue_task_merge("task123")

    assert job.id == "merge-task123"
    mock_get_queue.assert_called_once_with(QUEUE_MERGE)
    mock_spawn.assert_called_once_with(QUEUE_MERGE, single=True, job_id="merge-task123")


# -- run_task_merge --


def _make_approved_task(conn, worktree="/tmp/worktree", branch="agm/test-task"):
    """Helper: create an approved task with worktree for merge tests."""
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="test", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=0,
        title="Test task",
        description="Do it",
    )
    update_task_status(conn, task["id"], "ready")
    claim_task(
        conn,
        task["id"],
        caller="cli",
        branch=branch,
        worktree=worktree,
    )
    update_task_status(conn, task["id"], "review")
    update_task_status(conn, task["id"], "approved")
    return task, plan


def test_run_task_merge_success(db_conn):
    """run_task_merge should merge, complete task, resolve blockers, trigger exec."""
    from agm.db import add_task_block

    conn = db_conn
    task, plan = _make_approved_task(conn)
    # Add a blocked dependent task
    dep = create_task(conn, plan_id=plan["id"], ordinal=1, title="Dep task", description="d")
    add_task_block(conn, task_id=dep["id"], blocked_by_task_id=task["id"])
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main") as mock_merge,
        patch("agm.git_ops.remove_worktree") as mock_cleanup,
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Merged" in result
    mock_merge.assert_called_once()
    mock_cleanup.assert_called_once()

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    dep_found = get_task(verify_conn, dep["id"])
    assert dep_found["status"] == "ready"
    # Trigger should be called for the promoted task
    mock_trigger.assert_called()
    verify_conn.close()


def test_run_task_merge_success_logs_sync_warning_when_checkout_fails(db_conn):
    """Merge completion should log the exact sync-failure hint when checkout sync fails."""
    task, _ = _make_approved_task(db_conn)

    db_path = db_conn.execute("PRAGMA database_list").fetchone()[2]

    def fake_merge(*_args, **kwargs):
        on_sync_failure = kwargs.get("on_sync_failure")
        if on_sync_failure is not None:
            on_sync_failure("manual checkout blocked")
        return "abc123"

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main", side_effect=fake_merge),
        patch("agm.git_ops.remove_worktree"),
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    assert "Merged" in result
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    logs = list_task_logs(verify_conn, task["id"], level="WARNING")
    assert any(entry["message"] == WORKTREE_SYNC_WARNING_MESSAGE for entry in logs)
    verify_conn.close()


def test_run_task_merge_success_skips_sync_warning_when_checkout_succeeds(db_conn):
    """Normal merge success should not emit the sync-failure hint."""
    task, _ = _make_approved_task(db_conn)
    db_path = db_conn.execute("PRAGMA database_list").fetchone()[2]

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main", return_value="abc123"),
        patch("agm.git_ops.remove_worktree"),
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    assert "Merged" in result
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    logs = list_task_logs(verify_conn, task["id"], level="WARNING")
    assert not any(entry["message"] == WORKTREE_SYNC_WARNING_MESSAGE for entry in logs)
    verify_conn.close()


def test_run_task_merge_uses_project_base_branch(db_conn):
    """run_task_merge should pass custom base branch to git helpers and return message."""
    import json

    from agm.db import add_task_block

    conn = db_conn
    task, plan = _make_approved_task(conn)
    set_project_base_branch(conn, get_project_id(conn), "release/base")
    conn.execute(
        "UPDATE tasks SET files = ? WHERE id = ?",
        (json.dumps(["src/foo.py"]), task["id"]),
    )
    conn.commit()

    dep = create_task(conn, plan_id=plan["id"], ordinal=1, title="Dep task", description="d")
    add_task_block(conn, task_id=dep["id"], blocked_by_task_id=task["id"])
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_merge._trigger_task_execution"),
        patch("agm.git_ops.check_branch_file_scope") as mock_scope,
        patch("agm.git_ops.merge_to_main") as mock_merge,
        patch("agm.git_ops.remove_worktree") as mock_cleanup,
    ):
        mock_scope.return_value = []
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "release/base" in result
    assert mock_scope.call_args.kwargs["base_branch"] == "release/base"
    assert mock_merge.call_args.kwargs["base_branch"] == "release/base"
    mock_cleanup.assert_called_once()


def test_run_task_merge_promoted_triggers_priority_order_with_stable_ties(db_conn):
    """Promoted tasks should auto-trigger high->medium->low with stable tie-breaking."""
    conn = db_conn
    task, plan = _make_approved_task(conn)
    promoted_medium_2 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=2,
        title="medium 2",
        description="d",
    )
    promoted_high = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=3,
        title="high",
        description="d",
        priority="high",
    )
    promoted_low = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=4,
        title="low",
        description="d",
        priority="low",
    )
    promoted_medium_1 = create_task(
        conn,
        plan_id=plan["id"],
        ordinal=1,
        title="medium 1",
        description="d",
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    promoted_scrambled = [
        promoted_low["id"],
        promoted_medium_2["id"],
        promoted_high["id"],
        promoted_medium_1["id"],
    ]

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main"),
        patch("agm.git_ops.remove_worktree"),
        patch(
            "agm.jobs_merge.resolve_blockers_for_terminal_task",
            return_value=(promoted_scrambled, []),
        ),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_merge

        run_task_merge(task["id"])

    assert [call.args[1] for call in mock_trigger.call_args_list] == [
        promoted_high["id"],
        promoted_medium_1["id"],
        promoted_medium_2["id"],
        promoted_low["id"],
    ]


def test_run_task_merge_conflict_triggers_reexecution(db_conn):
    """First merge conflict should capture diff, reset task, and re-trigger executor."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    set_project_base_branch(conn, get_project_id(conn), "release/base")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.git_ops.merge_to_main",
            side_effect=RuntimeError("Merge conflict in file.py"),
        ),
        patch(
            "agm.jobs_merge._capture_branch_diff",
            return_value="diff --git a/file.py\n+new line",
        ),
        patch("agm.git_ops.remove_worktree"),
        patch(
            "agm.git_ops.create_worktree",
            return_value=("new-branch", "/tmp/new-wt"),
        ) as mock_create_worktree,
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "re-execution triggered" in result

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "running"  # Reset for re-execution
    assert found["branch"] == "new-branch"
    assert found["worktree"] == "/tmp/new-wt"
    assert found["thread_id"] is None  # Cleared for fresh execution

    # Should have logged the conflict
    logs = list_task_logs(verify_conn, task["id"], level="MERGE_CONFLICT")
    assert len(logs) == 1
    assert (
        logs[0]["message"] == "[MERGE_CONFLICT] Attempt 2/3: re-executing against updated main\n\n"
        "Previous diff:\n\ndiff --git a/file.py\n+new line"
    )
    mock_create_worktree.assert_called_once_with(
        "/tmp/testproj", task["id"], task["title"], "release/base"
    )
    verify_conn.close()

    mock_trigger.assert_called_once()


def test_run_task_merge_second_conflict_retriggers_reexecution(db_conn):
    """Second merge conflict should re-run task again."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    set_project_base_branch(conn, get_project_id(conn), "release/base")
    # Pre-insert a MERGE_CONFLICT log to simulate prior re-execution
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="Prior conflict diff",
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.git_ops.merge_to_main",
            side_effect=RuntimeError("Merge conflict again"),
        ),
        patch(
            "agm.jobs_merge._capture_branch_diff",
            return_value="second diff",
        ),
        patch("agm.git_ops.remove_worktree"),
        patch(
            "agm.git_ops.create_worktree",
            return_value=("new-branch", "/tmp/new-wt"),
        ) as mock_create_worktree,
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "re-execution triggered" in result

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "running"
    assert found["branch"] == "new-branch"
    assert found["worktree"] == "/tmp/new-wt"
    assert found["thread_id"] is None
    logs = list_task_logs(verify_conn, task["id"], level="MERGE_CONFLICT")
    assert len(logs) == 2
    assert (
        logs[-1]["message"] == "[MERGE_CONFLICT] Attempt 3/3: re-executing against updated main\n\n"
        "Previous diff:\n\nsecond diff"
    )
    assert mock_create_worktree.call_args.args[3] == "release/base"
    mock_create_worktree.assert_called_once_with(
        "/tmp/testproj", task["id"], task["title"], "release/base"
    )
    mock_trigger.assert_called_once()
    verify_conn.close()


def test_run_task_merge_two_conflicts_then_success(db_conn):
    """Two merge conflicts can still succeed on the third attempt."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    set_project_base_branch(conn, get_project_id(conn), "release/base")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.git_ops.merge_to_main",
            side_effect=[
                RuntimeError("Merge conflict in file.py"),
                RuntimeError("Merge conflict again"),
                "abc123",
            ],
        ),
        patch(
            "agm.jobs_merge._capture_branch_diff",
            return_value="diff --git a/file.py\n+new line",
        ),
        patch("agm.git_ops.remove_worktree"),
        patch(
            "agm.git_ops.create_worktree",
            side_effect=[("retry-branch", "/tmp/retry-wt"), ("retry-branch-2", "/tmp/retry-wt-2")],
        ) as mock_create_worktree,
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import run_task_merge

        first_attempt = run_task_merge(task["id"])
        update_task_status(conn, task["id"], "approved")
        second_attempt = run_task_merge(task["id"])
        update_task_status(conn, task["id"], "approved")
        third_attempt = run_task_merge(task["id"])

    assert "re-execution triggered" in first_attempt
    assert "re-execution triggered" in second_attempt
    assert "Merged" in third_attempt

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    logs = list_task_logs(verify_conn, task["id"], level="MERGE_CONFLICT")
    assert len(logs) == 2
    assert logs[0]["message"].startswith("[MERGE_CONFLICT] Attempt 2/3")
    assert logs[1]["message"].startswith("[MERGE_CONFLICT] Attempt 3/3")
    mock_create_worktree.assert_any_call("/tmp/testproj", task["id"], task["title"], "release/base")
    assert all(call.args[3] == "release/base" for call in mock_create_worktree.call_args_list)
    assert len(mock_create_worktree.call_args_list) == 2
    verify_conn.close()


def test_run_task_merge_third_conflict_fails_terminal(db_conn):
    """Third merge conflict should fail task and emit terminal status event."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="first conflict diff",
    )
    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="second conflict diff without marker",
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.git_ops.merge_to_main",
            side_effect=RuntimeError("Merge conflict again"),
        ),
        patch(
            "agm.jobs_merge.resolve_blockers_for_terminal_task",
            return_value=([], []),
        ) as mock_resolve_blockers,
        patch("agm.jobs_merge._emit") as mock_emit,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Attempt 3/3" in result
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "failed"

    logs = list_task_logs(verify_conn, task["id"])
    failure_log = [
        entry
        for entry in logs
        if entry["level"] == "ERROR" and "merge conflict after re-executions" in entry["message"]
    ]
    assert failure_log
    assert "Attempt 3/3" in failure_log[0]["message"]
    assert "second conflict diff without marker" in failure_log[0]["message"]
    mock_resolve_blockers.assert_called_once_with(ANY, task["id"], record_history=True)
    mock_emit.assert_any_call(
        "task:status",
        task["id"],
        "failed",
        project=ANY,
        plan_id=task["plan_id"],
    )
    assert not any(
        call.args[:3] == ("task:status", task["id"], "approved")
        for call in mock_emit.call_args_list
    )
    verify_conn.close()


def test_run_task_merge_non_conflict_error_stays_approved(db_conn):
    """Non-conflict merge errors should leave task approved (no re-execution)."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.git_ops.merge_to_main",
            side_effect=RuntimeError("Permission denied"),
        ),
    ):
        from agm.jobs import run_task_merge

        with pytest.raises(RuntimeError, match="Permission denied"):
            run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"  # Stays approved
    verify_conn.close()


# -- merge-time quality gate --


def test_run_task_merge_quality_gate_triggers_reexecution(db_conn):
    """First quality gate failure at merge should reset task and re-trigger executor."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    # Set a quality gate on the project
    pid = conn.execute("SELECT project_id FROM plans WHERE id = ?", (task["plan_id"],)).fetchone()[
        0
    ]
    conn.execute(
        "UPDATE projects SET quality_gate = ? WHERE id = ?",
        ('{"checks": [{"name": "lint", "cmd": ["ruff", "check"], "timeout": 30}]}', pid),
    )
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_merge._run_quality_checks",
            return_value=_qg_fail("lint", "error: unused import"),
        ),
        patch(
            "agm.jobs_merge._capture_branch_diff",
            return_value="diff --git a/foo.py\n+new line",
        ),
        patch("agm.git_ops.remove_worktree"),
        patch(
            "agm.git_ops.create_worktree",
            return_value=("new-branch", "/tmp/new-wt"),
        ),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "re-execution triggered" in result

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "running"  # Reset for re-execution
    assert found["branch"] == "new-branch"
    assert found["worktree"] == "/tmp/new-wt"

    logs = list_task_logs(verify_conn, task["id"], level="QUALITY_GATE_FAIL")
    assert len(logs) == 1
    assert "Re-execution triggered" in logs[0]["message"]
    verify_conn.close()

    mock_trigger.assert_called_once()


def test_run_task_merge_quality_gate_second_failure_stays_approved(db_conn):
    """Second quality gate failure at merge should leave task approved."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    pid = conn.execute("SELECT project_id FROM plans WHERE id = ?", (task["plan_id"],)).fetchone()[
        0
    ]
    conn.execute(
        "UPDATE projects SET quality_gate = ? WHERE id = ?",
        ('{"checks": [{"name": "lint", "cmd": ["ruff", "check"], "timeout": 30}]}', pid),
    )
    conn.commit()
    # Pre-insert a QUALITY_GATE_FAIL log to simulate prior re-execution
    add_task_log(
        conn,
        task_id=task["id"],
        level="QUALITY_GATE_FAIL",
        message="Prior QG failure",
    )
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_merge._run_quality_checks",
            return_value=_qg_fail("lint", "error: unused import"),
        ),
    ):
        from agm.jobs import run_task_merge

        with pytest.raises(RuntimeError, match="second attempt"):
            run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"  # Stays approved on second failure
    verify_conn.close()


def test_run_task_merge_quality_gate_passes(db_conn):
    """Quality gate passing should allow merge to proceed normally."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    pid = conn.execute("SELECT project_id FROM plans WHERE id = ?", (task["plan_id"],)).fetchone()[
        0
    ]
    conn.execute(
        "UPDATE projects SET quality_gate = ? WHERE id = ?",
        ('{"checks": [{"name": "lint", "cmd": ["ruff", "check"], "timeout": 30}]}', pid),
    )
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_merge._run_quality_checks", return_value=_qg_pass()),
        patch("agm.git_ops.merge_to_main", return_value="abc123"),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.jobs_task_creation._auto_trigger_execution_for_ready_tasks"),
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Merged" in result

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    verify_conn.close()


def test_run_task_merge_quality_gate_skipped_when_unconfigured(db_conn):
    """No quality gate configured should skip checks and merge normally."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_merge._run_quality_checks") as mock_qc,
        patch("agm.git_ops.merge_to_main", return_value="abc123"),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.jobs_task_creation._auto_trigger_execution_for_ready_tasks"),
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Merged" in result
    mock_qc.assert_not_called()

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "completed"
    verify_conn.close()


def test_post_merge_command_runs_after_merge(db_conn):
    """Configured post_merge_command should be invoked with merge SHA after merge."""
    from agm.db import set_project_post_merge_command

    conn = db_conn
    task, _ = _make_approved_task(conn)
    pid = get_project_id(conn)
    set_project_post_merge_command(conn, pid, "scripts/post-merge.sh")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main", return_value="abc123"),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.jobs_task_creation._auto_trigger_execution_for_ready_tasks"),
        patch("agm.jobs_merge._run_post_merge_command") as mock_pmc,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Merged" in result
    mock_pmc.assert_called_once()
    call_args = mock_pmc.call_args[0]
    assert (
        call_args[0] == conn.execute("SELECT dir FROM projects WHERE id = ?", (pid,)).fetchone()[0]
    )
    assert call_args[1] == "scripts/post-merge.sh"
    assert call_args[2] == "abc123"
    assert call_args[3] == task["id"]


def test_post_merge_command_skipped_when_unconfigured(db_conn):
    """No post_merge_command configured should skip the command."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.git_ops.merge_to_main", return_value="abc123"),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.jobs_task_creation._auto_trigger_execution_for_ready_tasks"),
        patch("agm.jobs_merge._run_post_merge_command") as mock_pmc,
    ):
        from agm.jobs import run_task_merge

        result = run_task_merge(task["id"])

    assert "Merged" in result
    mock_pmc.assert_not_called()


def test_run_post_merge_command_executes_without_shell():
    """Post-merge command runs as argv list (no shell injection risk)."""
    from agm.jobs_merge import _run_post_merge_command

    with patch("agm.jobs_merge.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _run_post_merge_command("/tmp/repo", "echo hello", "abc def", "task-123")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["echo", "hello"]
    assert call_args[1].get("shell") is not True
    assert call_args[1]["cwd"] == "/tmp/repo"
    assert call_args[1]["env"]["AGM_MERGE_SHA"] == "abc def"
    assert call_args[1]["env"]["AGM_TASK_ID"] == "task-123"


# -- auto-trigger: execution after review --


def test_run_task_execution_triggers_review(db_conn):
    """run_task_execution should auto-trigger review on success."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch(
            "agm.jobs_execution._run_task_execution_codex",
            return_value="Done",
        ),
        patch("agm.jobs_merge._trigger_task_review") as mock_trigger,
    ):
        from agm.jobs import run_task_execution

        run_task_execution(task["id"])

    mock_trigger.assert_called_once_with(task["id"])


# -- auto-trigger: merge after approve --


def test_run_task_review_approve_triggers_merge(db_conn):
    """run_task_review approve should auto-trigger merge."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps({"verdict": "approve", "summary": "LGTM", "findings": []})

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_review._run_task_review_codex", return_value=mock_output),
        patch("agm.jobs_merge._trigger_task_merge") as mock_trigger,
    ):
        from agm.jobs import run_task_review

        run_task_review(task["id"])

    mock_trigger.assert_called_once_with(task["id"])


# -- auto-trigger: re-execution after rejection --


def test_run_task_review_reject_triggers_reexecution(db_conn):
    """run_task_review reject should auto-trigger re-execution."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "verdict": "reject",
            "summary": "Missing tests",
            "findings": [
                {
                    "severity": "critical",
                    "file": "foo.py",
                    "description": "No tests",
                },
            ],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_review._run_task_review_codex", return_value=mock_output),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_review

        run_task_review(task["id"])

    # Should trigger re-execution
    mock_trigger.assert_called_once()
    call_args = mock_trigger.call_args
    assert call_args[0][1] == task["id"]

    # Task should be running again
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "running"
    verify_conn.close()


# -- rejection limit --


def test_run_task_review_reject_exceeds_limit(db_conn):
    """3 rejections should fail the task instead of re-running."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    # Pre-seed 2 REVIEW logs (previous rejections)
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="First rejection")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="Second rejection")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "verdict": "reject",
            "summary": "Still broken",
            "findings": [
                {
                    "severity": "critical",
                    "file": "foo.py",
                    "description": "Still no tests",
                },
            ],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_review._run_task_review_codex", return_value=mock_output),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_review

        run_task_review(task["id"])

    # Should NOT trigger re-execution
    mock_trigger.assert_not_called()

    # Task should be failed
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "failed"
    reason = json.loads(found["failure_reason"])
    assert reason["code"] == "max_rejections"
    assert reason["task_id"] == task["id"]
    assert reason["rejection_count"] == 3
    assert reason["summary_snippet"] == "Still broken"
    verify_conn.close()


def test_run_task_review_reject_exceeds_limit_with_non_string_summary(db_conn):
    """Non-string review summaries should still fail task and persist failure_reason."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="First rejection")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="Second rejection")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps(
        {
            "verdict": "reject",
            "summary": None,
            "findings": [
                {
                    "severity": "critical",
                    "file": "foo.py",
                    "description": "Still no tests",
                },
            ],
        }
    )

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_review._run_task_review_codex", return_value=mock_output),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import run_task_review

        run_task_review(task["id"])

    mock_trigger.assert_not_called()
    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "failed"
    reason = json.loads(found["failure_reason"])
    assert reason["summary_snippet"] == ""
    verify_conn.close()


# -- auto-trigger: task creation triggers execution --


def test_task_creation_triggers_execution(db_conn):
    """_insert_tasks_from_output should auto-trigger execution for ready tasks."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "First",
                    "description": "Do first",
                    "files": ["x.py"],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                },
                {
                    "ordinal": 1,
                    "title": "Second",
                    "description": "Do second",
                    "files": [],
                    "blocked_by": [0],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                },
            ],
            "cancel_tasks": [],
        }
    )

    with patch("agm.jobs_merge._trigger_task_execution") as mock_trigger:
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 2
    # Only the ready task should be triggered
    assert mock_trigger.call_count == 1


def test_task_creation_trigger_order_priority_then_stable_ties(db_conn):
    """Ready task auto-trigger order should be high, medium, low with stable ties."""
    import json

    conn = db_conn
    p = make_plan(conn)
    finalize_plan_request(conn, p["id"], "{}")
    promoted_high = create_task(
        conn,
        plan_id=p["id"],
        ordinal=99,
        title="promoted high",
        description="d",
        priority="high",
    )

    output = json.dumps(
        {
            "tasks": [
                {
                    "ordinal": 0,
                    "title": "medium 1",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                },
                {
                    "ordinal": 1,
                    "title": "pending high",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "pending",
                    "priority": "high",
                },
                {
                    "ordinal": 2,
                    "title": "low",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "priority": "low",
                },
                {
                    "ordinal": 3,
                    "title": "medium 2",
                    "description": "d",
                    "files": [],
                    "blocked_by": [],
                    "blocked_by_existing": [],
                    "external_blockers": [],
                    "status": "ready",
                    "priority": "medium",
                },
            ],
            "cancel_tasks": [],
        }
    )

    with (
        patch("agm.jobs_task_creation.resolve_stale_blockers", return_value=[promoted_high["id"]]),
        patch("agm.jobs_merge._trigger_task_execution") as mock_trigger,
    ):
        from agm.jobs import _insert_tasks_from_output

        count = _insert_tasks_from_output(conn, p["id"], output)

    assert count == 4
    tasks_by_ordinal = {t["ordinal"]: t for t in list_tasks(conn, plan_id=p["id"])}
    expected_order = [
        promoted_high["id"],  # promoted high
        tasks_by_ordinal[0]["id"],  # medium tie -> lower ordinal first
        tasks_by_ordinal[3]["id"],  # medium tie
        tasks_by_ordinal[2]["id"],  # low
    ]
    assert [call.args[1] for call in mock_trigger.call_args_list] == expected_order


# -- _get_rejection_count --


def test_get_rejection_count(db_conn):
    """_get_rejection_count should count REVIEW-level logs in current cycle."""
    conn = db_conn
    task = _make_running_task(conn)
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R1")
    add_task_log(conn, task_id=task["id"], level="INFO", message="Info")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R2")

    from agm.jobs import _get_rejection_count

    assert _get_rejection_count(conn, task["id"]) == 2


def test_get_rejection_count_resets_on_reclaim(db_conn):
    """_get_rejection_count should only count REVIEW logs after the latest claim."""
    conn = db_conn
    task = _make_running_task(conn)
    # First cycle: 3 rejections
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R1")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R2")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R3")

    from agm.jobs import _get_rejection_count

    assert _get_rejection_count(conn, task["id"]) == 3

    # Simulate retry: new claim log marks start of new cycle
    add_task_log(conn, task_id=task["id"], level="INFO", message="Claimed by user via cli")

    # Old rejections should no longer count
    assert _get_rejection_count(conn, task["id"]) == 0

    # New rejection in the new cycle
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R4")
    assert _get_rejection_count(conn, task["id"]) == 1


# -- _maybe_escalate_codex_config --


def test_escalate_effort_only_on_rejection_1(db_conn):
    """Rejection 1 should escalate effort to xhigh but keep the same thread."""
    conn = db_conn
    task = _make_running_task(conn)
    set_task_thread_id(conn, task["id"], "thread-abc")
    task = get_task(conn, task["id"])
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R1")

    from agm.jobs_execution import _maybe_escalate_codex_config

    turn_config = {"effort": "high"}
    thread_config = {"model": "spark-model"}
    runtime_model_config = {"work_model": "spark-model", "think_model": "full-model"}

    with patch("agm.jobs_execution.publish_task_model_escalation") as mock_publish:
        result = _maybe_escalate_codex_config(
            conn, task, "thread-abc", turn_config, thread_config, runtime_model_config
        )
    assert result == "thread-abc"
    assert turn_config["effort"] == "xhigh"
    # Thread and model unchanged
    assert thread_config["model"] == "spark-model"
    mock_publish.assert_not_called()
    refreshed = get_task(conn, task["id"])
    assert refreshed["thread_id"] == "thread-abc"


def test_escalate_model_on_rejection_2(db_conn):
    """Rejection 2+ with different work/think models should escalate model and clear thread."""
    conn = db_conn
    task = _make_running_task(conn)
    set_task_thread_id(conn, task["id"], "thread-abc")
    task = get_task(conn, task["id"])
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R1")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R2")

    from agm.jobs_execution import _maybe_escalate_codex_config

    turn_config = {"effort": "high"}
    thread_config = {"model": "spark-model"}
    runtime_model_config = {"work_model": "spark-model", "think_model": "full-model"}

    with patch("agm.jobs_execution.publish_task_model_escalation") as mock_publish:
        result = _maybe_escalate_codex_config(
            conn, task, "thread-abc", turn_config, thread_config, runtime_model_config
        )
    assert result is None
    assert turn_config["effort"] == "xhigh"
    assert thread_config["model"] == "full-model"
    mock_publish.assert_called_once_with(
        task["id"],
        "spark-model",
        "full-model",
        2,
        project="testproj",
        plan_id=task["plan_id"],
    )
    refreshed = get_task(conn, task["id"])
    assert refreshed["thread_id"] is None


def test_no_model_escalation_when_models_match(db_conn):
    """Rejection 2+ with same work/think model should only escalate effort."""
    conn = db_conn
    task = _make_running_task(conn)
    set_task_thread_id(conn, task["id"], "thread-abc")
    task = get_task(conn, task["id"])
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R1")
    add_task_log(conn, task_id=task["id"], level="REVIEW", message="R2")

    from agm.jobs_execution import _maybe_escalate_codex_config

    turn_config = {"effort": "high"}
    thread_config = {"model": "same-model"}
    runtime_model_config = {"work_model": "same-model", "think_model": "same-model"}

    with patch("agm.jobs_execution.publish_task_model_escalation") as mock_publish:
        result = _maybe_escalate_codex_config(
            conn, task, "thread-abc", turn_config, thread_config, runtime_model_config
        )
    assert result == "thread-abc"
    assert turn_config["effort"] == "xhigh"
    assert thread_config["model"] == "same-model"
    mock_publish.assert_not_called()


def test_no_escalation_without_existing_thread(db_conn):
    """No thread ID should return None unchanged (no escalation)."""
    conn = db_conn
    task = _make_running_task(conn)

    from agm.jobs_execution import _maybe_escalate_codex_config

    turn_config = {"effort": "high"}
    thread_config = {"model": "spark-model"}
    runtime_model_config = {"work_model": "spark-model", "think_model": "full-model"}

    with patch("agm.jobs_execution.publish_task_model_escalation") as mock_publish:
        result = _maybe_escalate_codex_config(
            conn, task, None, turn_config, thread_config, runtime_model_config
        )
    assert result is None
    assert turn_config["effort"] == "high"
    mock_publish.assert_not_called()


def test_is_thread_not_found_detection():
    """_is_thread_not_found should detect thread-not-found error patterns."""
    from agm.jobs_execution import _is_thread_not_found

    assert _is_thread_not_found(Exception("Thread not found")) is True
    assert _is_thread_not_found(Exception("thread NOT_FOUND on server")) is True
    assert _is_thread_not_found(Exception("RPCError: not_found")) is True
    assert _is_thread_not_found(Exception("timeout waiting for response")) is False
    assert _is_thread_not_found(Exception("connection refused")) is False


# -- on_task_merge_failure --


def test_on_task_merge_failure_logs_not_status(db_conn):
    """Merge failure callback should log error but not change task status."""
    conn = db_conn
    task, _ = _make_approved_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_job = MagicMock()
    mock_job.args = [task["id"]]

    with patch(
        "agm.db.get_connection",
        side_effect=lambda *_: get_connection(Path(db_path)),
    ):
        from agm.jobs import on_task_merge_failure

        on_task_merge_failure(mock_job, None, RuntimeError, RuntimeError("conflict"), None)

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"  # Status unchanged
    logs = list_task_logs(verify_conn, task["id"])
    assert any("conflict" in entry["message"] for entry in logs)
    verify_conn.close()


# -- skip_review / skip_merge --


def _make_running_task_with_flags(conn, skip_review=False, skip_merge=False):
    """Helper: create a running task with skip flags for quick mode tests."""
    pid = get_project_id(conn)
    plan = create_plan_request(conn, project_id=pid, prompt="quick", caller="cli", backend="codex")
    finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="Quick task", description="Do it")
    update_task_status(conn, task["id"], "ready")
    claim_task(conn, task["id"], caller="cli", branch="agm/quick", worktree="/tmp/wt")
    # Set skip flags directly
    conn.execute(
        "UPDATE tasks SET skip_review = ?, skip_merge = ? WHERE id = ?",
        (int(skip_review), int(skip_merge), task["id"]),
    )
    conn.commit()
    return task


def test_run_task_execution_skip_review_auto_approves(db_conn):
    """skip_review=True should go to approved and trigger merge."""
    conn = db_conn
    task = _make_running_task_with_flags(conn, skip_review=True)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_execution._run_task_execution_codex", return_value="Done"),
        patch("agm.jobs_merge._trigger_task_merge") as mock_merge,
        patch("agm.jobs_merge._trigger_task_review") as mock_review,
    ):
        from agm.jobs import run_task_execution

        run_task_execution(task["id"])

    mock_merge.assert_called_once()
    mock_review.assert_not_called()

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"
    verify_conn.close()


def test_run_task_execution_skip_review_skip_merge(db_conn):
    """skip_review=True + skip_merge=True should approve but not merge."""
    conn = db_conn
    task = _make_running_task_with_flags(conn, skip_review=True, skip_merge=True)
    plan = get_plan_request(conn, task["plan_id"])
    assert plan is not None
    session = create_session(conn, project_id=plan["project_id"], trigger="do")
    set_plan_session_id(conn, plan["id"], session["id"])
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_execution._run_task_execution_codex", return_value="Done"),
        patch("agm.jobs_merge._trigger_task_merge") as mock_merge,
        patch("agm.jobs_merge._trigger_task_review") as mock_review,
    ):
        from agm.jobs import run_task_execution

        run_task_execution(task["id"])

    mock_merge.assert_not_called()
    mock_review.assert_not_called()

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"
    finished_session = get_session(verify_conn, session["id"])
    assert finished_session is not None
    assert finished_session["status"] == "completed"
    verify_conn.close()


def test_run_task_review_skip_merge_no_auto_merge(db_conn):
    """skip_merge=True on reviewer approve should not trigger merge."""
    import json

    conn = db_conn
    task = _make_review_task(conn)
    plan = get_plan_request(conn, task["plan_id"])
    assert plan is not None
    session = create_session(conn, project_id=plan["project_id"], trigger="do")
    set_plan_session_id(conn, plan["id"], session["id"])
    # Set skip_merge flag
    conn.execute("UPDATE tasks SET skip_merge = 1 WHERE id = ?", (task["id"],))
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    mock_output = json.dumps({"verdict": "approve", "summary": "LGTM", "findings": []})

    with (
        patch(
            "agm.db.get_connection",
            side_effect=lambda *_: get_connection(Path(db_path)),
        ),
        patch("agm.jobs_review._run_task_review_codex", return_value=mock_output),
        patch("agm.jobs_merge._trigger_task_merge") as mock_merge,
    ):
        from agm.jobs import run_task_review

        run_task_review(task["id"])

    mock_merge.assert_not_called()

    verify_conn = get_connection(Path(db_path))
    found = get_task(verify_conn, task["id"])
    assert found["status"] == "approved"
    finished_session = get_session(verify_conn, session["id"])
    assert finished_session is not None
    assert finished_session["status"] == "completed"
    verify_conn.close()


# -- task status history (job flows) --


def test_task_status_history_success_chain_execution_review_merge(db_conn):
    """Core success path should record running->review->approved->completed."""
    import json

    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    approve_output = json.dumps({"verdict": "approve", "summary": "LGTM", "findings": []})

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))),
        patch("agm.jobs_execution._run_task_execution_codex", return_value="done"),
        patch("agm.jobs_merge._trigger_task_review"),
        patch("agm.jobs_review._run_task_review_codex", return_value=approve_output),
        patch("agm.jobs_merge._trigger_task_merge"),
        patch("agm.git_ops.merge_to_main"),
        patch("agm.git_ops.remove_worktree"),
    ):
        from agm.jobs import run_task_execution, run_task_merge, run_task_review

        run_task_execution(task["id"])
        run_task_review(task["id"])
        run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    rows = task_transition_rows(verify_conn, task["id"])
    assert [(row["old_status"], row["new_status"]) for row in rows] == [
        ("running", "review"),
        ("review", "approved"),
        ("approved", "completed"),
    ]
    task_row = get_task(verify_conn, task["id"])
    expected = [task_row["actor"]] * 3
    assert [row["actor"] for row in rows] == expected
    verify_conn.close()


def test_task_status_history_rejection_retry_chain(db_conn):
    """Reject path should record running->review->running without duplicates."""
    import json

    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    reject_output = json.dumps(
        {
            "verdict": "reject",
            "summary": "Needs fixes",
            "findings": [
                {"severity": "high", "file": "src/a.py", "description": "broken behavior"},
            ],
        }
    )

    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))),
        patch("agm.jobs_execution._run_task_execution_codex", return_value="done"),
        patch("agm.jobs_merge._trigger_task_review"),
        patch("agm.jobs_review._run_task_review_codex", return_value=reject_output),
        patch("agm.jobs_merge._trigger_task_execution"),
    ):
        from agm.jobs import run_task_execution, run_task_review

        run_task_execution(task["id"])
        run_task_review(task["id"])

    verify_conn = get_connection(Path(db_path))
    rows = task_transition_rows(verify_conn, task["id"])
    assert [(row["old_status"], row["new_status"]) for row in rows] == [
        ("running", "review"),
        ("review", "rejected"),
        ("rejected", "running"),
    ]
    task_row = get_task(verify_conn, task["id"])
    expected_actors = [task_row["actor"]] * 3
    assert [row["actor"] for row in rows] == expected_actors
    verify_conn.close()


def test_task_status_history_failure_callback_is_nonduplicative(db_conn):
    """Failure callback should not append another failed transition if already failed."""
    conn = db_conn
    task = _make_running_task(conn)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))),
        patch("agm.jobs_execution._run_task_execution_codex", side_effect=RuntimeError("boom")),
    ):
        from agm.jobs import on_task_execution_failure, run_task_execution

        with pytest.raises(RuntimeError, match="boom"):
            run_task_execution(task["id"])

        mock_job = MagicMock()
        mock_job.args = [task["id"]]
        on_task_execution_failure(mock_job, None, RuntimeError, RuntimeError("boom"), None)

    verify_conn = get_connection(Path(db_path))
    rows = task_transition_rows(verify_conn, task["id"])
    assert [(row["old_status"], row["new_status"]) for row in rows] == [("running", "failed")]
    assert len(rows) == 1
    verify_conn.close()


def test_task_status_history_cancellation_records_only_real_changes(db_conn):
    """Cancellation flow should record pending/ready->cancelled and skip no-op writes."""
    import json

    conn = db_conn
    plan = make_plan(conn)
    pending_task = create_task(conn, plan_id=plan["id"], ordinal=0, title="p", description="d")
    ready_task = create_task(conn, plan_id=plan["id"], ordinal=1, title="r", description="d")
    noop_task = create_task(conn, plan_id=plan["id"], ordinal=2, title="n", description="d")
    update_task_status(conn, ready_task["id"], "ready")
    update_task_status(conn, noop_task["id"], "cancelled")

    output = json.dumps(
        {
            "tasks": [],
            "cancel_tasks": [
                {"task_id": pending_task["id"], "reason": "obsolete"},
                {"task_id": ready_task["id"], "reason": "obsolete"},
                {"task_id": noop_task["id"], "reason": "already cancelled"},
            ],
        }
    )

    from agm.jobs import _insert_tasks_from_output

    count = _insert_tasks_from_output(conn, plan["id"], output)
    assert count == 0

    assert get_task(conn, pending_task["id"])["status"] == "cancelled"
    assert get_task(conn, ready_task["id"])["status"] == "cancelled"
    assert get_task(conn, noop_task["id"])["status"] == "cancelled"

    pending_rows = task_transition_rows(conn, pending_task["id"])
    ready_rows = task_transition_rows(conn, ready_task["id"])
    noop_rows = task_transition_rows(conn, noop_task["id"])
    assert [(row["old_status"], row["new_status"]) for row in pending_rows] == [
        ("blocked", "cancelled")
    ]
    assert [(row["old_status"], row["new_status"]) for row in ready_rows] == [
        ("ready", "cancelled")
    ]
    assert noop_rows == []
    assert [row["actor"] for row in pending_rows] == [None]
    assert [row["actor"] for row in ready_rows] == [None]


def test_task_status_history_promotion_and_auto_claim_from_merge(db_conn):
    """Merge promotions should record pending->ready and auto-claim ready->running."""
    from agm.db import add_task_block

    conn = db_conn
    task, plan = _make_approved_task(conn)
    promoted = create_task(conn, plan_id=plan["id"], ordinal=1, title="blocked", description="d")
    add_task_block(conn, task_id=promoted["id"], blocked_by_task_id=task["id"])
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))),
        patch("agm.git_ops.merge_to_main"),
        patch("agm.git_ops.remove_worktree"),
        patch("agm.git_ops.create_worktree", return_value=("agm/promoted", "/tmp/promoted-wt")),
        patch("agm.queue.enqueue_task_execution"),
    ):
        from agm.jobs import run_task_merge

        run_task_merge(task["id"])

    verify_conn = get_connection(Path(db_path))
    promoted_task = get_task(verify_conn, promoted["id"])
    assert promoted_task["status"] == "running"

    promoted_rows = task_transition_rows(verify_conn, promoted["id"])
    assert [(row["old_status"], row["new_status"]) for row in promoted_rows] == [
        ("blocked", "ready"),
        ("ready", "running"),
    ]
    assert [row["actor"] for row in promoted_rows] == [None, "agm-auto"]
    verify_conn.close()


def test_task_status_history_sets_null_actor_when_unknown(db_conn):
    """When no owner/system actor is available, history actor should be NULL."""
    conn = db_conn
    task = _make_running_task(conn)
    conn.execute("UPDATE tasks SET actor = NULL WHERE id = ?", (task["id"],))
    conn.commit()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    with (
        patch("agm.db.get_connection", side_effect=lambda *_: get_connection(Path(db_path))),
        patch("agm.jobs_execution._run_task_execution_codex", side_effect=RuntimeError("boom")),
    ):
        from agm.jobs import run_task_execution

        with pytest.raises(RuntimeError, match="boom"):
            run_task_execution(task["id"])

    verify_conn = get_connection(Path(db_path))
    rows = task_transition_rows(verify_conn, task["id"])
    assert [(row["old_status"], row["new_status"], row["actor"]) for row in rows] == [
        ("running", "failed", None)
    ]
    verify_conn.close()


# -- quality-gate prompt injection --


@pytest.mark.asyncio
async def test_executor_prompt_includes_default_quality_gate(db_conn):
    """Executor prompt should include discovery prompt when no quality gate configured."""
    conn = db_conn
    task = _make_running_task(conn)
    # Refresh task to get worktree populated by claim_task
    task = get_task(conn, task["id"])

    captured = {}

    async def fake_codex_turn(client, **kwargs):
        captured.update(kwargs)
        return "thread-123", "Done", {"input_tokens": 100, "output_tokens": 50}

    with (
        patch("agm.jobs_execution._codex_turn", side_effect=fake_codex_turn),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    prompt = captured.get("prompt", "")
    # Should include discovery prompt when no quality gate configured
    assert "No quality gate configured" in prompt
    assert "Inspect the repo to discover its tooling" in prompt
    assert PROJECT_INSTRUCTIONS_SECTION_DELIMITER not in prompt


@pytest.mark.asyncio
async def test_executor_prompt_includes_custom_quality_gate(db_conn):
    """Executor prompt should include custom quality-gate commands when configured."""
    conn = db_conn
    pid = get_project_id(conn)

    # Set custom quality gate
    from agm.db import set_project_quality_gate

    custom_qg = {
        "auto_fix": [{"name": "black", "cmd": ["black", "."]}],
        "checks": [
            {"name": "mypy", "cmd": ["mypy", "src/"], "timeout": 90},
            {"name": "custom test", "cmd": ["npm", "test"], "timeout": 120},
        ],
    }
    import json

    set_project_quality_gate(conn, pid, json.dumps(custom_qg))

    task = _make_running_task(conn)
    # Refresh task to get worktree populated by claim_task
    task = get_task(conn, task["id"])

    captured = {}

    async def fake_codex_turn(client, **kwargs):
        captured.update(kwargs)
        return "thread-456", "Done", {"input_tokens": 100, "output_tokens": 50}

    with (
        patch("agm.jobs_execution._codex_turn", side_effect=fake_codex_turn),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    prompt = captured.get("prompt", "")
    # Should include custom quality-gate section
    assert "QUALITY GATE — Project command sequence:" in prompt
    assert "Auto-fix commands (run these first, in order):" in prompt
    assert "black" in prompt
    assert "black ." in prompt
    assert "Strict checks (must pass):" in prompt
    assert "mypy" in prompt
    assert "mypy src/" in prompt
    assert "custom test" in prompt
    assert "npm test" in prompt
    # Should NOT include default ruff/pytest
    assert "ruff format" not in prompt
    assert "pytest" not in prompt
    assert PROJECT_INSTRUCTIONS_SECTION_DELIMITER not in prompt


@pytest.mark.asyncio
async def test_reviewer_prompt_includes_quality_gate(db_conn):
    """Reviewer prompt should include quality-gate commands for context."""
    conn = db_conn
    task = _make_running_task(conn)
    # Refresh task to get worktree populated by claim_task
    task = get_task(conn, task["id"])

    captured = {}

    async def fake_codex_turn(client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-review-123",
            '{"verdict":"approve","summary":"LGTM","findings":[]}',
            {"input_tokens": 200, "output_tokens": 100},
        )

    with (
        patch("agm.jobs_review._codex_turn", side_effect=fake_codex_turn),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
        patch("subprocess.run") as mock_run,
    ):
        # Mock git commands for reviewer
        mock_run.return_value = MagicMock(stdout="commit-log", stderr="", returncode=0)

        from agm.jobs import _run_task_review_codex_async

        await _run_task_review_codex_async(conn, task)

    prompt = captured.get("prompt", "")
    # Should include discovery prompt when no quality gate configured
    assert "No quality gate configured" in prompt
    assert "Inspect the repo to discover its tooling" in prompt
    assert PROJECT_INSTRUCTIONS_SECTION_DELIMITER not in prompt


@pytest.mark.asyncio
async def test_executor_rejection_retry_includes_quality_gate(db_conn):
    """Executor prompt on rejection retry should include discovery prompt."""
    conn = db_conn
    task = _make_running_task(conn)
    set_task_thread_id(conn, task["id"], "existing-thread-999")
    add_task_log(
        conn,
        task_id=task["id"],
        level="REVIEW",
        message="Tests are missing",
    )
    # Refresh task to get both worktree and thread_id
    task = get_task(conn, task["id"])

    captured = {}

    async def fake_codex_turn(client, **kwargs):
        captured.update(kwargs)
        return "existing-thread-999", "Fixed", {"input_tokens": 150, "output_tokens": 75}

    with (
        patch("agm.jobs_execution._codex_turn", side_effect=fake_codex_turn),
        patch("agm.jobs_common.get_effective_role_config", return_value=""),
    ):
        from agm.jobs import _run_task_execution_codex_async

        await _run_task_execution_codex_async(conn, task)

    prompt = captured.get("prompt", "")
    # Rejection retry prompt should include discovery prompt
    assert "REVIEWER FEEDBACK — your changes were rejected" in prompt
    assert "Tests are missing" in prompt
    assert "No quality gate configured" in prompt
    assert "Inspect the repo to discover its tooling" in prompt
    assert PROJECT_INSTRUCTIONS_SECTION_DELIMITER not in prompt


@pytest.mark.asyncio
async def test_reviewer_prompt_includes_reviewer_instructions(db_conn):
    """Reviewer prompt should include reviewer-specific instructions from agents.toml."""
    conn = db_conn
    task = _make_running_task(conn)
    task = get_task(conn, task["id"])
    instructions = "Reviewer guidance: focus on regressions."
    captured = {}

    async def fake_codex_turn(client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-review-999",
            '{"verdict":"approve","summary":"LGTM","findings":[]}',
            {"input_tokens": 200, "output_tokens": 100},
        )

    with (
        patch("agm.jobs_review._codex_turn", side_effect=fake_codex_turn),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="commit-log", stderr="", returncode=0)

        from agm.jobs import _run_task_review_codex_async

        await _run_task_review_codex_async(conn, task)

    prompt = captured.get("prompt", "")
    assert prompt.endswith(REVIEWER_PROMPT_SUFFIX)
    assert instructions not in prompt
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


def test_trigger_task_creation_respects_manual_approval(db_conn):
    """_trigger_task_creation should set awaiting_approval when project is manual."""
    from agm.db import (
        get_plan_request,
        set_project_plan_approval,
        update_plan_request_status,
    )

    conn = db_conn
    plan = make_plan(conn)
    pid = get_project_id(conn)
    set_project_plan_approval(conn, pid, "manual")
    update_plan_request_status(conn, plan["id"], "running")
    finalize_plan_request(conn, plan["id"], '{"title":"t","tasks":[]}')

    from agm.jobs import _trigger_task_creation

    _trigger_task_creation(conn, plan["id"])

    updated = get_plan_request(conn, plan["id"])
    assert updated["task_creation_status"] == "awaiting_approval"


def test_trigger_task_creation_auto_enqueues(db_conn):
    """_trigger_task_creation should enqueue when project approval is auto."""
    from agm.db import get_plan_request, update_plan_request_status

    conn = db_conn
    plan = make_plan(conn)
    update_plan_request_status(conn, plan["id"], "running")
    finalize_plan_request(conn, plan["id"], '{"title":"t","tasks":[]}')

    with patch("agm.queue.enqueue_task_creation") as mock_enqueue:
        from agm.jobs import _trigger_task_creation

        _trigger_task_creation(conn, plan["id"])

    updated = get_plan_request(conn, plan["id"])
    assert updated["task_creation_status"] == "pending"
    mock_enqueue.assert_called_once_with(plan["id"])


# ---------------------------------------------------------------------------
# _process_enrichment_output tests
# ---------------------------------------------------------------------------


def test_process_enrichment_output_valid_no_questions(db_conn):
    """Enrichment output with enriched_prompt and no questions returns prompt."""
    from agm.jobs import _process_enrichment_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (plan["id"],))
    conn.commit()

    enriched = "Modify src/agm/cli.py to add the new command handler"
    output = json.dumps({"enriched_prompt": enriched, "questions": []})
    tokens = {"input_tokens": 10, "output_tokens": 20}
    result = _process_enrichment_output(conn, plan, output, "thread-1", tokens)

    assert result == enriched
    updated = get_plan_request(conn, plan["id"])
    assert updated["enriched_prompt"] == enriched


def test_process_enrichment_output_with_questions(db_conn):
    """Enrichment output with questions transitions to awaiting_input."""
    from agm.db import list_plan_questions
    from agm.jobs import _process_enrichment_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (plan["id"],))
    conn.commit()

    output = json.dumps(
        {
            "enriched_prompt": "Add auth module in src/auth.py with login() function",
            "questions": [
                {"question": "Which DB?", "options": ["postgres", "sqlite"]},
                {"question": "Auth method?", "options": None},
            ],
        }
    )
    tokens = {"input_tokens": 10, "output_tokens": 20}
    result = _process_enrichment_output(conn, plan, output, "thread-enrich", tokens)

    assert result is None
    updated = get_plan_request(conn, plan["id"])
    assert updated["status"] == "awaiting_input"
    assert updated["enrichment_thread_id"] == "thread-enrich"

    questions = list_plan_questions(conn, plan["id"])
    assert len(questions) == 2
    assert questions[0]["question"] == "Which DB?"


def test_process_enrichment_output_invalid_json(db_conn):
    """Invalid JSON falls back to raw prompt."""
    from agm.jobs import _process_enrichment_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    tokens = {"input_tokens": 5, "output_tokens": 10}
    result = _process_enrichment_output(conn, plan, "not json at all", "t1", tokens)

    assert result == "raw prompt"
    updated = get_plan_request(conn, plan["id"])
    assert updated["enriched_prompt"] == "raw prompt"


def test_process_enrichment_output_empty_prompt_fallback(db_conn):
    """Empty enriched_prompt falls back to raw prompt."""
    from agm.jobs import _process_enrichment_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    output = '{"enriched_prompt": "", "questions": []}'
    tokens = {"input_tokens": 5, "output_tokens": 10}
    result = _process_enrichment_output(conn, plan, output, "t1", tokens)

    assert result == "raw prompt"


# ---------------------------------------------------------------------------
# _process_exploration_output + _format_exploration_for_channel tests
# ---------------------------------------------------------------------------


def test_process_exploration_output_valid_json(db_conn):
    """Valid exploration JSON is stored on the plan row."""
    from agm.jobs import _process_exploration_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (plan["id"],))
    conn.commit()

    exploration = json.dumps(
        {
            "summary": "Flask app with SQLAlchemy ORM",
            "architecture": "MVC with blueprints",
            "relevant_files": [
                {"path": "src/models.py", "description": "ORM models", "key_symbols": ["User"]},
            ],
            "patterns_to_follow": ["Use factory pattern"],
            "reusable_helpers": [
                {"path": "src/utils.py", "symbol": "get_db", "description": "DB session helper"},
            ],
            "test_locations": ["tests/test_models.py"],
        }
    )
    result = _process_exploration_output(conn, plan, exploration)

    assert result == exploration
    updated = get_plan_request(conn, plan["id"])
    assert updated["exploration_context"] == exploration


def test_process_exploration_output_invalid_json(db_conn):
    """Invalid JSON returns None without crashing."""
    from agm.jobs import _process_exploration_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    result = _process_exploration_output(conn, plan, "not json at all")

    assert result is None
    updated = get_plan_request(conn, plan["id"])
    assert updated["exploration_context"] is None


def test_process_exploration_output_missing_required_fields(db_conn):
    """JSON missing required fields returns None."""
    from agm.jobs import _process_exploration_output

    conn = db_conn
    plan = make_plan(conn, prompt="raw prompt")
    result = _process_exploration_output(conn, plan, '{"summary": "just summary"}')

    assert result is None
    updated = get_plan_request(conn, plan["id"])
    assert updated["exploration_context"] is None


def test_format_exploration_for_channel():
    """Structured exploration data formats into readable channel text."""
    from agm.jobs import _format_exploration_for_channel

    data = {
        "summary": "Flask REST API",
        "architecture": "Layered MVC",
        "relevant_files": [
            {
                "path": "src/routes.py",
                "description": "API endpoints",
                "key_symbols": ["login", "register"],
            },
        ],
        "patterns_to_follow": ["Use decorators for auth"],
        "reusable_helpers": [
            {"path": "src/auth.py", "symbol": "require_token", "description": "Auth decorator"},
        ],
        "test_locations": ["tests/test_routes.py"],
    }
    result = _format_exploration_for_channel(data)

    assert "Flask REST API" in result
    assert "Layered MVC" in result
    assert "src/routes.py" in result
    assert "login, register" in result
    assert "Use decorators for auth" in result
    assert "src/auth.py:require_token" in result
    assert "tests/test_routes.py" in result


def test_format_exploration_for_channel_empty_sections():
    """Formatter handles empty arrays gracefully."""
    from agm.jobs import _format_exploration_for_channel

    data = {
        "summary": "Simple project",
        "architecture": "",
        "relevant_files": [],
        "patterns_to_follow": [],
        "reusable_helpers": [],
        "test_locations": [],
    }
    result = _format_exploration_for_channel(data)
    assert "Simple project" in result
    assert "Relevant files" not in result
    assert "Patterns" not in result


# ---------------------------------------------------------------------------
# _parse_task_files tests
# ---------------------------------------------------------------------------


def test_parse_task_files_valid_json():
    from agm.jobs import _parse_task_files

    assert _parse_task_files({"files": '["a.py", "b.py"]'}) == ["a.py", "b.py"]


def test_parse_task_files_none():
    from agm.jobs import _parse_task_files

    assert _parse_task_files({"files": None}) == []
    assert _parse_task_files({}) == []


def test_parse_task_files_invalid_json():
    from agm.jobs import _parse_task_files

    assert _parse_task_files({"files": "not json"}) == []


def test_parse_task_files_empty_array():
    from agm.jobs import _parse_task_files

    assert _parse_task_files({"files": "[]"}) == []


def test_parse_task_files_non_list_json():
    """Non-list JSON (e.g. a string) is returned via `or []` fallback only if falsy."""
    from agm.jobs import _parse_task_files

    # A JSON string is truthy — _parse_task_files returns it as-is
    assert _parse_task_files({"files": '"just a string"'}) == "just a string"
    # null is falsy — falls through to []
    assert _parse_task_files({"files": "null"}) == []


# ---------------------------------------------------------------------------
# _get_predecessor_context tests
# ---------------------------------------------------------------------------


def test_get_predecessor_context_no_blockers(db_conn):
    """Returns empty string and no IDs when task has no resolved blockers."""
    from agm.jobs import _get_predecessor_context

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    result, ids = _get_predecessor_context(conn, task["id"])
    assert result == ""
    assert ids == []


def test_get_predecessor_context_with_completed_predecessor(db_conn):
    """Returns formatted context for completed predecessors."""
    from agm.jobs import _get_predecessor_context

    conn = db_conn
    plan = make_plan(conn)
    pred = create_task(
        conn, plan_id=plan["id"], ordinal=0, title="Setup DB", description="Create schema"
    )
    child = create_task(conn, plan_id=plan["id"], ordinal=1, title="Add API", description="d")

    # Make predecessor completed
    update_task_status(conn, pred["id"], "ready")
    update_task_status(conn, pred["id"], "running")
    update_task_status(conn, pred["id"], "review")
    update_task_status(conn, pred["id"], "approved")
    update_task_status(conn, pred["id"], "completed")

    # Add resolved blocker
    add_task_block(conn, task_id=child["id"], blocked_by_task_id=pred["id"])
    conn.execute(
        "UPDATE task_blocks SET resolved = 1 WHERE task_id = ? AND blocked_by_task_id = ?",
        (child["id"], pred["id"]),
    )
    conn.commit()

    result, ids = _get_predecessor_context(conn, child["id"])
    assert "Setup DB" in result
    assert "Create schema" in result
    assert "Predecessor tasks" in result
    assert pred["id"] in ids


def test_get_predecessor_context_skips_non_completed(db_conn):
    """Skips predecessors that are not completed."""
    from agm.jobs import _get_predecessor_context

    conn = db_conn
    plan = make_plan(conn)
    pred = create_task(conn, plan_id=plan["id"], ordinal=0, title="Pending", description="d")
    child = create_task(conn, plan_id=plan["id"], ordinal=1, title="Child", description="d")

    # Blocker is resolved but predecessor is still pending (e.g. cancelled)
    add_task_block(conn, task_id=child["id"], blocked_by_task_id=pred["id"])
    conn.execute(
        "UPDATE task_blocks SET resolved = 1 WHERE task_id = ? AND blocked_by_task_id = ?",
        (child["id"], pred["id"]),
    )
    conn.commit()

    result, ids = _get_predecessor_context(conn, child["id"])
    assert result == ""
    assert ids == []


# ---------------------------------------------------------------------------
# _get_merge_conflict_context tests
# ---------------------------------------------------------------------------


def test_get_merge_conflict_context_no_logs(db_conn):
    """Returns None when no MERGE_CONFLICT logs exist."""
    from agm.jobs import _get_merge_conflict_context

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    result = _get_merge_conflict_context(conn, task["id"])
    assert result is None


def test_get_merge_conflict_context_extracts_diff(db_conn):
    """Extracts diff portion after double-newline marker."""
    from agm.jobs import _get_merge_conflict_context

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message=(
            "Merge conflict. Saved diff.\n\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        ),
    )

    result = _get_merge_conflict_context(conn, task["id"])
    assert result.startswith("--- a/file.py")
    assert "+new" in result


def test_get_merge_conflict_context_no_marker(db_conn):
    """Returns full message when no double-newline marker found."""
    from agm.jobs import _get_merge_conflict_context

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    add_task_log(
        conn,
        task_id=task["id"],
        level="MERGE_CONFLICT",
        message="conflict log without marker",
    )

    result = _get_merge_conflict_context(conn, task["id"])
    assert result == "conflict log without marker"


# ---------------------------------------------------------------------------
# _rollback_claim tests
# ---------------------------------------------------------------------------


def test_rollback_claim_resets_to_ready(db_conn):
    """Rollback claim resets running task to ready and clears ownership."""
    from agm.jobs import _rollback_claim

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")

    # Simulate a claimed task
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")
    conn.execute(
        "UPDATE tasks SET pid = 12345, actor = 'alice', caller = 'cli', "
        "branch = 'feature-br', worktree = '/tmp/wt' WHERE id = ?",
        (task["id"],),
    )
    conn.commit()

    with patch("agm.git_ops.remove_worktree") as mock_rm:
        _rollback_claim(conn, task["id"], "/project", "/tmp/wt", "feature-br")

    rolled = get_task(conn, task["id"])
    assert rolled["status"] == "ready"
    assert rolled["pid"] is None
    assert rolled["actor"] is None
    assert rolled["branch"] is None
    assert rolled["worktree"] is None
    mock_rm.assert_called_once_with("/project", "feature-br", "/tmp/wt")


def test_rollback_claim_no_project_dir(db_conn):
    """Rollback claim works without project_dir (skips worktree removal)."""
    from agm.jobs import _rollback_claim

    conn = db_conn
    plan = make_plan(conn)
    task = create_task(conn, plan_id=plan["id"], ordinal=0, title="T", description="d")
    update_task_status(conn, task["id"], "ready")
    update_task_status(conn, task["id"], "running")

    _rollback_claim(conn, task["id"], None, None, None)

    rolled = get_task(conn, task["id"])
    assert rolled["status"] == "ready"


# -- _parse_quality_gate_output --


def test_parse_quality_gate_output_valid():
    """_parse_quality_gate_output accepts valid JSON with auto_fix + checks."""
    from agm.jobs import _parse_quality_gate_output

    raw = json.dumps(
        {
            "auto_fix": [{"name": "fmt", "cmd": ["fmt", "."]}],
            "checks": [{"name": "lint", "cmd": ["lint", "."], "timeout": 60}],
        }
    )
    result = _parse_quality_gate_output(raw)
    assert result["auto_fix"][0]["name"] == "fmt"
    assert result["checks"][0]["name"] == "lint"


def test_parse_quality_gate_output_missing_checks():
    """_parse_quality_gate_output rejects JSON without checks array."""
    from agm.jobs import _parse_quality_gate_output

    with pytest.raises(ValueError, match="checks"):
        _parse_quality_gate_output(json.dumps({"auto_fix": []}))


def test_parse_quality_gate_output_missing_auto_fix():
    """_parse_quality_gate_output rejects JSON without auto_fix array."""
    from agm.jobs import _parse_quality_gate_output

    with pytest.raises(ValueError, match="auto_fix"):
        _parse_quality_gate_output(json.dumps({"checks": []}))


def test_parse_quality_gate_output_not_json():
    """_parse_quality_gate_output rejects non-JSON text."""
    from agm.jobs import _parse_quality_gate_output

    with pytest.raises(json.JSONDecodeError):
        _parse_quality_gate_output("not json at all")


def test_parse_quality_gate_output_not_object():
    """_parse_quality_gate_output rejects non-object JSON."""
    from agm.jobs import _parse_quality_gate_output

    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_quality_gate_output(json.dumps([1, 2, 3]))


def test_run_quality_checks_stages_all_auto_fix_file_changes(tmp_path):
    """Auto-fix mode should commit all staged and untracked file changes."""
    from agm.jobs_quality_gate import _run_quality_checks

    quality_gate_json = json.dumps(
        {
            "auto_fix": [{"name": "fmt", "cmd": ["fmt", "."]}],
            "checks": [],
        }
    )
    status_output = "\n".join(
        [
            " M tracked.py",
            "?? untracked.py",
        ]
    )

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=status_output, stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        qg_result = _run_quality_checks(str(tmp_path), quality_gate_json=quality_gate_json)

    assert qg_result.passed
    assert qg_result.auto_fix_ran
    assert qg_result.auto_fix_committed
    calls = [call.args[0] for call in mock_run.call_args_list]
    assert calls[0] == ["fmt", "."]
    assert calls[1] == ["git", "status", "--porcelain"]
    assert calls[2] == ["git", "add", "."]
    assert calls[3] == ["git", "commit", "-m", "Auto-format: quality gate auto-fix"]


# -- _merge_developer_instructions unit tests --


def test_merge_developer_instructions_sets_when_none():
    """No existing devInstructions should set them."""
    from agm.jobs_common import _merge_developer_instructions

    tc = {"model": "m"}
    with patch("agm.jobs_common.get_effective_role_config", return_value="agent rules"):
        _merge_developer_instructions(tc, "/tmp/proj", "executor")
    assert tc["developerInstructions"] == "agent rules"


def test_merge_developer_instructions_appends_to_existing():
    """Existing devInstructions should be appended with delimiter."""
    from agm.jobs_common import _merge_developer_instructions

    tc = {"model": "m", "developerInstructions": "system prompt"}
    with patch("agm.jobs_common.get_effective_role_config", return_value="agent rules"):
        _merge_developer_instructions(tc, "/tmp/proj", "reviewer")
    assert "system prompt" in tc["developerInstructions"]
    assert "agent rules" in tc["developerInstructions"]
    assert PROJECT_INSTRUCTIONS_SECTION_DELIMITER in tc["developerInstructions"]


def test_merge_developer_instructions_no_instructions_noop():
    """No instructions configured should be a no-op."""
    from agm.jobs_common import _merge_developer_instructions

    tc = {"model": "m"}
    with patch("agm.jobs_common.get_effective_role_config", return_value=""):
        _merge_developer_instructions(tc, "/tmp/proj", "planner")
    assert "developerInstructions" not in tc


# -- Integration: planner instructions injection --


def test_planner_instructions_in_developer_instructions(db_conn):
    """Planner instructions injected via developerInstructions (not user prompt)."""
    conn = db_conn
    plan = make_plan(conn, "test planner dev instructions")
    instructions = "Planner rules via developerInstructions."
    captured = {}

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return ("thread-1", "plan output", {"input_tokens": 0, "output_tokens": 0})

    with (
        patch("agm.jobs_plan._codex_client", _fake_codex_client),
        patch("agm.jobs_plan._codex_turn", AsyncMock(side_effect=_fake_codex_turn)),
        patch("agm.jobs_task_creation._trigger_task_creation"),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_plan_request_codex_async

        asyncio.run(_run_plan_request_codex_async(conn, plan))

    assert instructions not in captured["prompt"]
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr


# -- Integration: task creation instructions injection --


@pytest.mark.asyncio
async def test_task_creation_instructions_in_developer_instructions(db_conn):
    """Task creation instructions injected via developerInstructions (not user prompt)."""
    import contextlib

    conn = db_conn
    p = make_plan(conn)
    plan_payload = json.dumps({"title": "T", "summary": "S", "tasks": []})
    finalize_plan_request(conn, p["id"], plan_payload)
    plan = get_plan_request(conn, p["id"])
    instructions = "Task agent developer rules."
    captured = {}

    @contextlib.asynccontextmanager
    async def _fake_codex_client():
        yield object()

    async def _fake_codex_turn(_client, **kwargs):
        captured.update(kwargs)
        return (
            "thread-1",
            '{"tasks":[],"cancel_tasks":[]}',
            {"input_tokens": 0, "output_tokens": 0},
        )

    with (
        patch("agm.jobs_task_creation._codex_client", _fake_codex_client),
        patch("agm.jobs_task_creation._codex_turn", side_effect=_fake_codex_turn),
        patch("agm.jobs_task_creation._insert_tasks_from_output", return_value=0),
        patch("agm.jobs_common.get_effective_role_config", return_value=instructions),
    ):
        from agm.jobs import _run_task_creation_codex_async

        await _run_task_creation_codex_async(conn, plan)

    assert instructions not in captured["prompt"]
    dev_instr = captured.get("start_thread_params", {}).get("developerInstructions")
    assert dev_instr is not None and instructions in dev_instr
