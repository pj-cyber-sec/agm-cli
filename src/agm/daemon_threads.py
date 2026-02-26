"""Normalization helpers for daemon thread/list responses."""

from __future__ import annotations

from typing import Any


def _status_fields(status_payload: Any) -> tuple[str | None, list[str]]:
    if not isinstance(status_payload, dict):
        return None, []
    status_type = status_payload.get("type")
    active_flags = status_payload.get("activeFlags")
    normalized_flags = (
        [str(flag) for flag in active_flags] if isinstance(active_flags, list) else []
    )
    return str(status_type) if isinstance(status_type, str) else None, normalized_flags


def normalize_daemon_thread_list(result: dict[str, Any]) -> dict[str, Any]:
    """Add stable, web-friendly fields to each thread row while preserving raw keys."""
    rows = result.get("data")
    if not isinstance(rows, list):
        return result

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        status_type, active_flags = _status_fields(row.get("status"))
        normalized["status_type"] = status_type
        normalized["active_flags"] = active_flags
        normalized["title"] = row.get("title") or row.get("name")
        normalized["created_at"] = row.get("createdAt") or row.get("created_at")
        normalized["updated_at"] = row.get("updatedAt") or row.get("updated_at")
        normalized["archived_at"] = row.get("archivedAt") or row.get("archived_at")
        normalized_rows.append(normalized)

    payload = dict(result)
    payload["data"] = normalized_rows
    return payload
