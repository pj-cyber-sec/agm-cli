"""Tests for pipeline event publishing and subscription."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import RedisError

from agm.queue import EVENTS_STREAM, EVENTS_STREAM_MAXLEN, EventSubscriber, publish_event


def test_publish_event_best_effort():
    """publish_event swallows Redis transport errors so pipeline is never affected."""
    with patch("agm.queue.get_redis", side_effect=RedisError("Redis down")):
        # Should not raise
        publish_event("plan:status", "plan-123", "running", project="myproject")


def test_publish_event_non_redis_exception_propagates():
    """Non-Redis exceptions raised while publishing should propagate."""
    with (
        patch("agm.queue.get_redis", side_effect=ValueError("boom")),
        pytest.raises(ValueError, match="boom"),
    ):
        publish_event("plan:status", "plan-123", "running", project="myproject")


def test_publish_event_payload_shape():
    """Verify the JSON payload contains all required fields."""
    mock_redis = MagicMock()
    with patch("agm.queue.get_redis", return_value=mock_redis):
        publish_event(
            "task:status",
            "task-abc",
            "review",
            project="myproject",
            plan_id="plan-xyz",
        )

    mock_redis.xadd.assert_called_once()
    call_args = mock_redis.xadd.call_args
    assert call_args[0][0] == EVENTS_STREAM

    payload = json.loads(call_args[0][1]["data"])
    assert payload["type"] == "task:status"
    assert payload["id"] == "task-abc"
    assert payload["plan_id"] == "plan-xyz"
    assert payload["project"] == "myproject"
    assert payload["status"] == "review"
    assert "ts" in payload

    assert call_args[1]["maxlen"] == EVENTS_STREAM_MAXLEN
    assert call_args[1]["approximate"] is True


def test_publish_event_plan_id_defaults_to_entity_id():
    """When plan_id is not provided, it defaults to the entity_id."""
    mock_redis = MagicMock()
    with patch("agm.queue.get_redis", return_value=mock_redis):
        publish_event("plan:status", "plan-123", "finalized", project="proj")

    payload = json.loads(mock_redis.xadd.call_args[0][1]["data"])
    assert payload["plan_id"] == "plan-123"


# ---------------------------------------------------------------------------
# EventSubscriber tests
# ---------------------------------------------------------------------------


def _make_stream_entry(event_data: dict, entry_id: str = "1-0") -> list:
    """Build a mock XREAD response with a single event."""
    return [[EVENTS_STREAM, [(entry_id, {"data": json.dumps(event_data)})]]]


def test_event_subscriber_returns_matching_event():
    """EventSubscriber returns events that match filters."""
    event = {"type": "task:status", "id": "task-1", "plan_id": "plan-1", "project": "proj"}
    mock_redis = MagicMock()
    mock_redis.xread.return_value = _make_stream_entry(event)

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(plan_id="plan-1", timeout=1.0)
        result = next(sub)

    assert result is not None
    assert result["id"] == "task-1"
    assert result["_stream_id"] == "1-0"


def test_event_subscriber_filters_by_plan_id():
    """EventSubscriber skips events that don't match plan_id filter."""
    wrong_event = _make_stream_entry(
        {"type": "task:status", "id": "task-1", "plan_id": "other-plan", "project": "proj"},
        entry_id="1-0",
    )
    mock_redis = MagicMock()
    # First call returns non-matching event, second call times out
    mock_redis.xread.side_effect = [wrong_event, []]

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(plan_id="plan-1", timeout=0.01)
        result = next(sub)

    assert result is None  # Timed out after filtering


def test_event_subscriber_filters_by_task_id():
    """EventSubscriber skips events that don't match task_id filter."""
    wrong_event = _make_stream_entry(
        {"type": "task:status", "id": "task-other", "plan_id": "plan-1", "project": "proj"},
    )
    mock_redis = MagicMock()
    mock_redis.xread.side_effect = [wrong_event, []]

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(task_id="task-1", timeout=0.01)
        result = next(sub)

    assert result is None


def test_event_subscriber_filters_by_project():
    """EventSubscriber skips events that don't match project filter."""
    wrong_event = _make_stream_entry(
        {"type": "task:status", "id": "task-1", "plan_id": "plan-1", "project": "other-proj"},
    )
    mock_redis = MagicMock()
    mock_redis.xread.side_effect = [wrong_event, []]

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(project="proj", timeout=0.01)
        result = next(sub)

    assert result is None


def test_event_subscriber_returns_none_on_timeout():
    """EventSubscriber returns None when XREAD times out."""
    mock_redis = MagicMock()
    mock_redis.xread.return_value = []

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(timeout=0.01)
        result = next(sub)

    assert result is None


def test_event_subscriber_graceful_redis_unavailable():
    """EventSubscriber sleeps and returns None when Redis is unavailable."""
    with patch("agm.queue.get_redis", side_effect=RedisError("down")):
        sub = EventSubscriber(timeout=0.01)
        assert not sub._redis_available
        result = next(sub)

    assert result is None


def test_event_subscriber_skips_malformed_events():
    """EventSubscriber skips entries with invalid JSON payload."""
    bad_entry = [[EVENTS_STREAM, [("1-0", {"data": "not-json{"})]]]
    mock_redis = MagicMock()
    mock_redis.xread.side_effect = [bad_entry, []]

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(timeout=0.01)
        result = next(sub)

    assert result is None


def test_event_subscriber_advances_cursor():
    """EventSubscriber advances its cursor after reading entries."""
    event = {"type": "task:status", "id": "t1", "plan_id": "p1", "project": "proj"}
    mock_redis = MagicMock()
    mock_redis.xread.return_value = _make_stream_entry(event, entry_id="42-1")

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(timeout=0.01)
        assert sub._cursor == "$"
        next(sub)
        assert sub._cursor == "42-1"


def test_event_subscriber_handles_bytes_entry_id():
    """EventSubscriber decodes bytes entry IDs from Redis."""
    event = {"type": "task:status", "id": "t1", "project": "proj"}
    entry = [[EVENTS_STREAM, [(b"99-0", {b"data": json.dumps(event).encode()})]]]
    mock_redis = MagicMock()
    mock_redis.xread.return_value = entry

    with patch("agm.queue.get_redis", return_value=mock_redis):
        sub = EventSubscriber(timeout=0.01)
        result = next(sub)

    assert result is not None
    assert result["_stream_id"] == "99-0"
