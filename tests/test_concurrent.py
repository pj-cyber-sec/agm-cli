"""Deterministic concurrent SQLite tests."""

from __future__ import annotations

import threading
from pathlib import Path

from agm.db import (
    add_project,
    add_task_block,
    claim_task,
    create_plan_request,
    create_task,
    create_tasks_batch,
    finalize_plan_request,
    get_connection,
    get_task,
    get_unresolved_block_count,
    list_status_history,
    list_task_blocks,
    list_task_logs,
    list_tasks,
    resolve_blockers_for_terminal_task,
    update_task_status,
)


def _create_plan_db(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "concurrent.sqlite3"
    conn = get_connection(db_path)
    try:
        project = add_project(conn, "testproj", str(tmp_path / "testproj"))
        plan = create_plan_request(
            conn,
            project_id=project["id"],
            prompt="concurrency plan",
            caller="cli",
            backend="codex",
        )
        finalize_plan_request(conn, plan["id"], '{"title":"t","summary":"s","tasks":[]}')
        return db_path, plan["id"]
    finally:
        conn.close()


def _join_threads(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive(), f"Thread {thread.name} did not finish"


def test_concurrent_claim_task_prevents_double_claim(tmp_path: Path):
    db_path, plan_id = _create_plan_db(tmp_path)

    conn = get_connection(db_path)
    try:
        task = create_task(conn, plan_id=plan_id, ordinal=0, title="Claim me", description="d")
        assert update_task_status(conn, task["id"], "ready")
    finally:
        conn.close()

    barrier = threading.Barrier(3)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def _worker(actor: str) -> None:
        worker_conn = get_connection(db_path)
        try:
            barrier.wait(timeout=5)
            claimed = claim_task(worker_conn, task["id"], actor=actor, caller="cli")
            results.append((actor, claimed))
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            worker_conn.close()

    t1 = threading.Thread(target=_worker, args=("worker-a",), name="claim-a")
    t2 = threading.Thread(target=_worker, args=("worker-b",), name="claim-b")
    t1.start()
    t2.start()

    barrier.wait(timeout=5)
    _join_threads([t1, t2])

    assert errors == []
    assert len(results) == 2
    assert sum(1 for _, ok in results if ok) == 1
    assert sum(1 for _, ok in results if not ok) == 1

    winner = next(actor for actor, ok in results if ok)

    verify_conn = get_connection(db_path)
    try:
        found = get_task(verify_conn, task["id"])
        logs = list_task_logs(verify_conn, task["id"])
        assert found is not None
        assert found["status"] == "running"
        assert found["actor"] == winner
        assert found["caller"] == "cli"
        assert len(logs) == 1
        assert winner in logs[0]["message"]
    finally:
        verify_conn.close()


def test_concurrent_update_task_status_prevents_double_transition(tmp_path: Path):
    db_path, plan_id = _create_plan_db(tmp_path)

    conn = get_connection(db_path)
    try:
        task = create_task(conn, plan_id=plan_id, ordinal=0, title="Transition", description="d")
    finally:
        conn.close()

    barrier = threading.Barrier(3)
    results: list[bool] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        worker_conn = get_connection(db_path)
        try:
            barrier.wait(timeout=5)
            transitioned = update_task_status(
                worker_conn,
                task["id"],
                "ready",
                record_history=True,
            )
            results.append(transitioned)
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            worker_conn.close()

    t1 = threading.Thread(target=_worker, name="status-a")
    t2 = threading.Thread(target=_worker, name="status-b")
    t1.start()
    t2.start()

    barrier.wait(timeout=5)
    _join_threads([t1, t2])

    assert errors == []
    assert len(results) == 2
    assert sum(1 for ok in results if ok) == 1
    assert sum(1 for ok in results if not ok) == 1

    verify_conn = get_connection(db_path)
    try:
        found = get_task(verify_conn, task["id"])
        history = list_status_history(verify_conn, entity_type="task", entity_id=task["id"])
        assert found is not None
        assert found["status"] == "ready"
        assert len(history) == 1
        assert history[0]["old_status"] == "blocked"
        assert history[0]["new_status"] == "ready"
    finally:
        verify_conn.close()


def test_busy_timeout_lock_contention_eventually_succeeds(tmp_path: Path):
    db_path, plan_id = _create_plan_db(tmp_path)

    conn = get_connection(db_path)
    try:
        task = create_task(conn, plan_id=plan_id, ordinal=0, title="Lock me", description="d")
    finally:
        conn.close()

    lock_acquired = threading.Event()
    allow_commit = threading.Event()
    transition_conn_ready = threading.Event()
    transition_started = threading.Event()
    transition_finished = threading.Event()

    results: list[bool] = []
    errors: list[BaseException] = []

    def _locker() -> None:
        locker_conn = get_connection(db_path)
        try:
            locker_conn.execute("BEGIN IMMEDIATE")
            locker_conn.execute(
                "UPDATE tasks SET description = description WHERE id = ?",
                (task["id"],),
            )
            lock_acquired.set()
            assert allow_commit.wait(timeout=10)
            locker_conn.commit()
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            locker_conn.close()

    def _transitioner() -> None:
        transition_conn = get_connection(db_path)
        try:
            transition_conn_ready.set()
            assert lock_acquired.wait(timeout=10)
            transition_started.set()
            transitioned = update_task_status(
                transition_conn,
                task["id"],
                "ready",
                record_history=True,
            )
            results.append(transitioned)
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            transition_finished.set()
            transition_conn.close()

    transitioner = threading.Thread(target=_transitioner, name="transitioner")
    transitioner.start()
    assert transition_conn_ready.wait(timeout=5)

    locker = threading.Thread(target=_locker, name="locker")
    locker.start()

    assert lock_acquired.wait(timeout=5)
    assert transition_started.wait(timeout=5)
    assert not transition_finished.is_set()

    allow_commit.set()
    _join_threads([locker, transitioner])

    assert errors == []
    assert results == [True]

    verify_conn = get_connection(db_path)
    try:
        found = get_task(verify_conn, task["id"])
        history = list_status_history(verify_conn, entity_type="task", entity_id=task["id"])
        assert found is not None
        assert found["status"] == "ready"
        assert len(history) == 1
        assert history[0]["old_status"] == "blocked"
        assert history[0]["new_status"] == "ready"
    finally:
        verify_conn.close()


def test_concurrent_blocker_resolution_promotes_downstream_once(tmp_path: Path):
    db_path, plan_id = _create_plan_db(tmp_path)

    conn = get_connection(db_path)
    try:
        blocker = create_task(conn, plan_id=plan_id, ordinal=0, title="Blocker", description="d")
        downstream = create_task(
            conn, plan_id=plan_id, ordinal=1, title="Downstream", description="d"
        )
        assert update_task_status(conn, blocker["id"], "ready")
        assert update_task_status(conn, blocker["id"], "running")
        assert update_task_status(conn, blocker["id"], "completed")
        add_task_block(conn, task_id=downstream["id"], blocked_by_task_id=blocker["id"])
        assert get_unresolved_block_count(conn, downstream["id"]) == 1
    finally:
        conn.close()

    barrier = threading.Barrier(3)
    promoted_results: list[list[str]] = []
    errors: list[BaseException] = []

    def _resolver() -> None:
        resolver_conn = get_connection(db_path)
        try:
            barrier.wait(timeout=5)
            promoted, _cascade = resolve_blockers_for_terminal_task(
                resolver_conn,
                blocker["id"],
                record_history=True,
            )
            promoted_results.append(promoted)
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            resolver_conn.close()

    t1 = threading.Thread(target=_resolver, name="resolver-a")
    t2 = threading.Thread(target=_resolver, name="resolver-b")
    t1.start()
    t2.start()

    barrier.wait(timeout=5)
    _join_threads([t1, t2])

    assert errors == []
    assert len(promoted_results) == 2
    promoted_ids = [task_id for batch in promoted_results for task_id in batch]
    assert promoted_ids.count(downstream["id"]) == 1
    assert len(promoted_ids) == 1

    verify_conn = get_connection(db_path)
    try:
        found = get_task(verify_conn, downstream["id"])
        history = list_status_history(verify_conn, entity_type="task", entity_id=downstream["id"])
        blocks = list_task_blocks(verify_conn, downstream["id"])
        assert found is not None
        assert found["status"] == "ready"
        assert get_unresolved_block_count(verify_conn, downstream["id"]) == 0
        assert len(blocks) == 1
        assert blocks[0]["resolved"] == 1
        assert len(history) == 1
        assert history[0]["old_status"] == "blocked"
        assert history[0]["new_status"] == "ready"
    finally:
        verify_conn.close()


def test_concurrent_create_tasks_batch_same_plan(tmp_path: Path):
    db_path, plan_id = _create_plan_db(tmp_path)

    batch_a = [
        {
            "ordinal": 0,
            "title": "A-1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "batch-a",
        },
        {
            "ordinal": 1,
            "title": "A-2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "batch-a",
        },
    ]
    batch_b = [
        {
            "ordinal": 10,
            "title": "B-1",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "batch-b",
        },
        {
            "ordinal": 11,
            "title": "B-2",
            "description": "d",
            "status": "ready",
            "blocked_by": [],
            "blocked_by_existing": [],
            "external_blockers": [],
            "bucket": "batch-b",
        },
    ]

    barrier = threading.Barrier(3)
    mappings: dict[str, dict[int, str]] = {}
    successes: list[str] = []
    errors: list[BaseException] = []

    def _create(label: str, tasks_data: list[dict]) -> None:
        creator_conn = get_connection(db_path)
        try:
            barrier.wait(timeout=5)
            mappings[label] = create_tasks_batch(creator_conn, plan_id, tasks_data)
            successes.append(label)
        except BaseException as exc:  # pragma: no cover - assertion helper path
            errors.append(exc)
        finally:
            creator_conn.close()

    t1 = threading.Thread(target=_create, args=("a", batch_a), name="batch-a")
    t2 = threading.Thread(target=_create, args=("b", batch_b), name="batch-b")
    t1.start()
    t2.start()

    barrier.wait(timeout=5)
    _join_threads([t1, t2])

    assert errors == []
    assert len(successes) == 2
    assert set(successes) == {"a", "b"}

    verify_conn = get_connection(db_path)
    try:
        tasks = list_tasks(verify_conn, plan_id=plan_id)
        assert len(tasks) == 4
        assert {task["ordinal"] for task in tasks} == {0, 1, 10, 11}

        mapping_a = mappings["a"]
        mapping_b = mappings["b"]

        assert set(mapping_a.keys()) == {0, 1}
        assert set(mapping_b.keys()) == {10, 11}

        a1 = get_task(verify_conn, mapping_a[0])
        a2 = get_task(verify_conn, mapping_a[1])
        b1 = get_task(verify_conn, mapping_b[10])
        b2 = get_task(verify_conn, mapping_b[11])

        assert a1 is not None and a1["status"] == "ready"
        assert a2 is not None and a2["status"] == "blocked"
        assert b1 is not None and b1["status"] == "ready"
        assert b2 is not None and b2["status"] == "blocked"

        a2_blocks = list_task_blocks(verify_conn, mapping_a[1])
        b2_blocks = list_task_blocks(verify_conn, mapping_b[11])
        assert len(a2_blocks) == 1
        assert len(b2_blocks) == 1
        assert a2_blocks[0]["blocked_by_task_id"] == mapping_a[0]
        assert b2_blocks[0]["blocked_by_task_id"] == mapping_b[10]
    finally:
        verify_conn.close()
