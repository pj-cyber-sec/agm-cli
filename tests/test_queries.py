"""Tests for query helpers used by CLI/API status views."""

from agm.queries import (
    active_task_rows,
    is_effectively_terminal_task,
    plan_watch_terminal_state,
    task_list_filter_rows,
    task_watch_terminal_state,
)


def test_is_effectively_terminal_task_for_skip_merge_approved():
    assert is_effectively_terminal_task({"status": "approved", "skip_merge": 1}) is True
    assert is_effectively_terminal_task({"status": "approved", "skip_merge": 0}) is False
    assert is_effectively_terminal_task({"status": "completed", "skip_merge": 0}) is True


def test_task_watch_terminal_state_treats_skip_merge_approved_as_terminal():
    terminal = task_watch_terminal_state([{"status": "approved", "skip_merge": 1}])
    assert terminal["reached"] is True
    assert terminal["reason"] == "all_tasks_terminal"


def test_plan_watch_terminal_state_treats_skip_merge_approved_as_terminal():
    plan = {"status": "finalized", "task_creation_status": "completed"}
    tasks = [{"status": "approved", "skip_merge": 1}]
    terminal = plan_watch_terminal_state(plan, tasks)
    assert terminal["reached"] is True
    assert terminal["is_all_tasks_terminal"] is True


def test_task_filters_hide_skip_merge_approved_by_default():
    tasks = [
        {"id": "t1", "status": "approved", "skip_merge": 1},
        {"id": "t2", "status": "approved", "skip_merge": 0},
        {"id": "t3", "status": "running", "skip_merge": 0},
    ]
    filtered = task_list_filter_rows(tasks, show_all=False, status=None)
    assert [task["id"] for task in filtered] == ["t2", "t3"]

    active = active_task_rows(tasks)
    assert [task["id"] for task in active] == ["t2", "t3"]
