"""Health checks and optional remediation for agm infrastructure."""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import StartedJobRegistry

from agm.db import (
    DEFAULT_DB_PATH,
    TASK_TERMINAL_STATUSES,
    clear_stale_task_git_refs_for_doctor,
    connect,
    fail_stale_running_plan_for_doctor,
    fail_stale_running_task_for_doctor,
    list_projects,
)
from agm.git_ops import detect_worktree_conflicts, get_real_conflicts, remove_worktree
from agm.queue import (
    AGM_QUEUE_NAMES,
    EVENTS_STREAM,
    LOG_DIR,
    get_queue_counts,
    get_redis,
)

Status = Literal["pass", "warning", "fail"]
_STATUS_RANK: dict[Status, int] = {"pass": 0, "warning": 1, "fail": 2}


class _CheckFindingRequired(TypedDict):
    status: Status
    message: str


class CheckFinding(_CheckFindingRequired, total=False):
    details: dict[str, object]


class CheckReport(TypedDict):
    name: str
    status: Status
    summary: str
    findings: list[CheckFinding]


class FixAction(TypedDict):
    attempted: int
    fixed: int
    failed: int
    failures: list[dict[str, str]]


class _DoctorReportRequired(TypedDict):
    status: Status
    summary: str
    checks: list[CheckReport]


class DoctorReport(_DoctorReportRequired, total=False):
    fix_actions: dict[str, FixAction]


def run_doctor(db_path: Path | None = None, *, fix: bool = False) -> DoctorReport:
    """Run all health checks and optionally apply remediation."""
    resolved_db_path = _resolve_db_path(db_path)
    checks = [
        _check_redis(),
        _check_redis_stream_health(resolved_db_path),
        _check_stale_registries(),
        _check_sqlite_integrity(resolved_db_path),
        _check_backends(),
        _check_stale_pids(resolved_db_path),
        _check_stale_entities(resolved_db_path),
        _check_orphaned_worktrees(resolved_db_path),
        _check_worktree_conflicts(resolved_db_path),
        _check_worktree_cleanliness(resolved_db_path),
        _check_disk_usage(resolved_db_path),
        _check_log_files(resolved_db_path),
    ]

    if fix:
        fix_actions = _apply_fixes(resolved_db_path, checks)
        checks = _rerun_fixable_checks(resolved_db_path, checks)
        if fix_actions.get("stale_pids", {}).get("fixed", 0) > 0:
            for index, check in enumerate(checks):
                if check["name"] == "stale_entities":
                    checks[index] = _check_stale_entities(resolved_db_path)
                    break

    status = _worst_status([check["status"] for check in checks])
    summary = _report_summary(checks)
    report: DoctorReport = {
        "status": status,
        "summary": summary,
        "checks": checks,
    }
    if fix:
        report["fix_actions"] = fix_actions
    return report


def _resolve_db_path(db_path: Path | None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser()
    return DEFAULT_DB_PATH


def _worst_status(statuses: list[Status]) -> Status:
    if not statuses:
        return "pass"
    return max(statuses, key=lambda s: _STATUS_RANK[s])


def _report_summary(checks: list[CheckReport]) -> str:
    counts: dict[Status, int] = {"pass": 0, "warning": 0, "fail": 0}
    for check in checks:
        counts[check["status"]] += 1
    return f"{counts['pass']} checks passed, {counts['warning']} warnings, {counts['fail']} failed."


def _check_redis() -> CheckReport:
    findings: list[CheckFinding] = []
    try:
        redis = get_redis()
        redis.ping()
        counts = get_queue_counts()
    except Exception as exc:
        return {
            "name": "redis",
            "status": "fail",
            "summary": "Redis is unavailable.",
            "findings": [
                {
                    "status": "fail",
                    "message": f"Failed to connect to Redis or read queue counts: {exc}",
                }
            ],
        }

    total_jobs = 0
    for queue_name, stats in sorted(counts.items()):
        queued = int(stats.get("queued", 0))
        running = int(stats.get("running", 0))
        failed = int(stats.get("failed", 0))
        total_jobs += queued + running + failed
        findings.append(
            {
                "status": "pass",
                "message": f"{queue_name}: queued={queued}, running={running}, failed={failed}",
                "details": {
                    "queue": queue_name,
                    "queued": queued,
                    "running": running,
                    "failed": failed,
                },
            }
        )

    return {
        "name": "redis",
        "status": "pass",
        "summary": f"Redis reachable; {len(counts)} queue(s), {total_jobs} total job(s).",
        "findings": findings,
    }


_REDIS_STREAM_STALE_SECONDS = 300
_MAX_STALE_REGISTRY_FINDINGS = 10


def _count_active_plan_and_task_records(db_path: Path) -> tuple[int, int]:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                (SELECT COUNT(*)
                   FROM plans
                   WHERE status IN ('pending', 'running', 'awaiting_input')) AS active_plan_count,
                (SELECT COUNT(*)
                   FROM tasks
                   WHERE status IN (
                       'blocked',
                       'ready',
                       'running',
                       'review',
                       'approved'
                   )) AS active_task_count
            """
        ).fetchone()
    return int(row["active_plan_count"]), int(row["active_task_count"])


def _check_redis_stream_health(db_path: Path) -> CheckReport:
    stream = os.environ.get("AGM_EVENTS_STREAM", EVENTS_STREAM)
    active_plan_count = 0
    active_task_count = 0

    with contextlib.suppress(Exception):
        active_plan_count, active_task_count = _count_active_plan_and_task_records(db_path)

    try:
        redis = get_redis()
        stream_exists = bool(redis.exists(stream))

        if not stream_exists:
            return {
                "name": "redis_stream",
                "status": "warning",
                "summary": f"Redis events stream '{stream}' is missing.",
                "findings": [
                    {
                        "status": "warning",
                        "message": (
                            f"Redis stream '{stream}' is missing while "
                            f"active plans={active_plan_count}, active tasks={active_task_count}."
                        ),
                        "details": {
                            "stream": stream,
                            "active_plan_count": active_plan_count,
                            "active_task_count": active_task_count,
                        },
                    }
                ],
            }

        stream_length = int(cast(int, redis.xlen(stream)))
        latest_entries = cast(list[tuple[object, object]], redis.xrevrange(stream, count=1))

        if not latest_entries:
            status = "warning" if (active_plan_count + active_task_count) > 0 else "pass"
            summary = (
                "Active entities are being tracked but no stream entries were found."
                if status == "warning"
                else "No active entities and no stream entries were found."
            )
            return {
                "name": "redis_stream",
                "status": status,
                "summary": summary,
                "findings": [
                    {
                        "status": status,
                        "message": (
                            f"Redis stream '{stream}' is empty while "
                            f"active plans={active_plan_count}, active tasks={active_task_count}, "
                            f"stream length={stream_length}."
                        ),
                        "details": {
                            "stream": stream,
                            "stream_length": stream_length,
                            "active_plan_count": active_plan_count,
                            "active_task_count": active_task_count,
                        },
                    }
                ],
            }

        latest_entry_id = latest_entries[0][0]
        if isinstance(latest_entry_id, bytes):
            latest_entry_id = latest_entry_id.decode()
        latest_entry_id_str = str(latest_entry_id)
        latest_ms = int(latest_entry_id_str.split("-", 1)[0])
        age_seconds = int((time.time() * 1000 - latest_ms) // 1000)

        if (
            age_seconds > _REDIS_STREAM_STALE_SECONDS
            and (active_plan_count + active_task_count) > 0
        ):
            return {
                "name": "redis_stream",
                "status": "warning",
                "summary": "Redis events stream may be stale while active entities exist.",
                "findings": [
                    {
                        "status": "warning",
                        "message": (
                            f"Latest Redis stream entry is {age_seconds}s old for '{stream}'."
                        ),
                        "details": {
                            "stream": stream,
                            "stream_length": stream_length,
                            "latest_entry_id": latest_entry_id_str,
                            "age_seconds": age_seconds,
                            "active_plan_count": active_plan_count,
                            "active_task_count": active_task_count,
                        },
                    }
                ],
            }

        return {
            "name": "redis_stream",
            "status": "pass",
            "summary": (f"Redis stream '{stream}' is healthy; latest entry is {age_seconds}s old."),
            "findings": [
                {
                    "status": "pass",
                    "message": f"Latest stream entry for '{stream}' is recent.",
                    "details": {
                        "stream": stream,
                        "stream_length": stream_length,
                        "latest_entry_id": latest_entry_id_str,
                        "age_seconds": age_seconds,
                        "active_plan_count": active_plan_count,
                        "active_task_count": active_task_count,
                    },
                }
            ],
        }
    except Exception as exc:
        return {
            "name": "redis_stream",
            "status": "fail",
            "summary": "Redis stream is unavailable.",
            "findings": [
                {
                    "status": "fail",
                    "message": f"Failed to read Redis stream state: {exc}",
                }
            ],
        }


def _coerce_pid_value(raw_pid: object) -> int | None:
    if isinstance(raw_pid, bool) or raw_pid is None:
        return None
    if isinstance(raw_pid, (bytes, bytearray)):
        try:
            raw_pid = raw_pid.decode()
        except Exception:
            return None
    if isinstance(raw_pid, int):
        return raw_pid if raw_pid > 0 else None
    if isinstance(raw_pid, float):
        if not raw_pid.is_integer():
            return None
        return int(raw_pid) if raw_pid > 0 else None
    if not isinstance(raw_pid, str):
        return None
    try:
        pid = int(raw_pid)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _extract_job_pid(job: Any) -> int | None:
    metadata = getattr(job, "meta", None)
    if isinstance(metadata, dict):
        for key in ("worker_pid", "pid", "process_id", "worker_process_id"):
            pid = _coerce_pid_value(metadata.get(key))
            if pid is not None:
                return pid

        worker_meta = metadata.get("worker")
        if isinstance(worker_meta, dict):
            for key in ("pid", "process_id", "worker_pid", "worker_process_id"):
                pid = _coerce_pid_value(worker_meta.get(key))
                if pid is not None:
                    return pid

    for key in ("worker_pid", "pid", "process_id", "worker_process_id"):
        pid = _coerce_pid_value(getattr(job, key, None))
        if pid is not None:
            return pid

    return None


def _check_stale_registries() -> CheckReport:
    try:
        redis = get_redis()
    except Exception as exc:
        return {
            "name": "stale_registries",
            "status": "fail",
            "summary": "Started registries could not be queried.",
            "findings": [{"status": "fail", "message": f"Failed to connect to Redis: {exc}"}],
        }

    stale_findings: list[CheckFinding] = []
    checked = 0

    for queue_name in AGM_QUEUE_NAMES:
        try:
            registry = StartedJobRegistry(queue_name, connection=redis)
            job_ids = sorted(list(registry.get_job_ids()), key=str)
        except Exception as exc:
            return {
                "name": "stale_registries",
                "status": "fail",
                "summary": "Started registries could not be queried.",
                "findings": [
                    {
                        "status": "fail",
                        "message": (
                            f"Failed to inspect started registry for queue {queue_name}: {exc}"
                        ),
                    }
                ],
            }

        for raw_job_id in job_ids:
            checked += 1
            job_id = raw_job_id.decode() if isinstance(raw_job_id, bytes) else str(raw_job_id)
            try:
                job = Job.fetch(job_id, connection=redis)
            except NoSuchJobError:
                stale_findings.append(
                    {
                        "status": "warning",
                        "message": (
                            f"Orphaned registry entry (job no longer exists): "
                            f"queue={queue_name} job_id={job_id}"
                        ),
                        "details": {
                            "kind": "ghost_registry_entry",
                            "reason": "job missing in queue registry",
                            "queue": queue_name,
                            "job_id": job_id,
                        },
                    }
                )
                continue
            except Exception as exc:
                return {
                    "name": "stale_registries",
                    "status": "fail",
                    "summary": "Started registries could not be queried.",
                    "findings": [
                        {
                            "status": "fail",
                            "message": (
                                f"Failed to inspect started job {job_id} "
                                f"in queue {queue_name}: {exc}"
                            ),
                        }
                    ],
                }

            pid = _extract_job_pid(job)
            if pid is None:
                continue
            if _pid_is_alive(pid):
                continue

            stale_findings.append(
                {
                    "status": "warning",
                    "message": (
                        f"Worker process not running for started job: "
                        f"queue={queue_name} job_id={job_id} pid={pid}"
                    ),
                    "details": {
                        "kind": "worker_dead",
                        "reason": "worker process is not running",
                        "queue": queue_name,
                        "job_id": job_id,
                        "pid": pid,
                    },
                }
            )

    if stale_findings:
        total_stale = len(stale_findings)
        findings = stale_findings[:_MAX_STALE_REGISTRY_FINDINGS]
        summary = (
            f"Found {total_stale} stale queue-registry item(s). "
            f"Showing {len(findings)} of {total_stale}."
            if total_stale > _MAX_STALE_REGISTRY_FINDINGS
            else f"Found {total_stale} stale queue-registry item(s)."
        )
        return {
            "name": "stale_registries",
            "status": "warning",
            "summary": summary,
            "findings": findings,
        }

    return {
        "name": "stale_registries",
        "status": "pass",
        "summary": f"No stale registry entries found across {checked} started job(s).",
        "findings": [
            {
                "status": "pass",
                "message": f"No stale registry entries found across {checked} started job(s).",
                "details": {"checked": checked},
            }
        ],
    }


def _check_backends() -> CheckReport:
    """Check whether backend CLIs (codex) are available on PATH."""
    import subprocess

    findings: list[CheckFinding] = []
    available = 0

    # Check codex
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            findings.append(
                {
                    "status": "pass",
                    "message": f"codex: {version}",
                    "details": {"backend": "codex", "version": version},
                }
            )
            available += 1
        else:
            findings.append(
                {
                    "status": "warning",
                    "message": "codex found but returned an error",
                    "details": {"backend": "codex", "error": result.stderr.strip()},
                }
            )
    except FileNotFoundError:
        findings.append(
            {
                "status": "warning",
                "message": "codex not found on PATH",
                "details": {"backend": "codex"},
            }
        )
    except Exception as exc:
        findings.append(
            {
                "status": "warning",
                "message": f"codex check failed: {exc}",
                "details": {"backend": "codex", "error": str(exc)},
            }
        )

    if available == 0:
        return {
            "name": "backends",
            "status": "fail",
            "summary": "No backends available. Install codex CLI.",
            "findings": findings,
        }

    return {
        "name": "backends",
        "status": "pass",
        "summary": f"{available} backend(s) available.",
        "findings": findings,
    }


def _check_sqlite_integrity(db_path: Path) -> CheckReport:
    try:
        with connect(db_path) as conn:
            try:
                rows = conn.execute("PRAGMA integrity_check").fetchall()
                messages = [str(row[0]) for row in rows if row]
            except Exception as exc:
                return {
                    "name": "sqlite",
                    "status": "fail",
                    "summary": "SQLite integrity check could not run.",
                    "findings": [
                        {
                            "status": "fail",
                            "message": f"PRAGMA integrity_check failed for {db_path}: {exc}",
                        }
                    ],
                }
    except Exception as exc:
        return {
            "name": "sqlite",
            "status": "fail",
            "summary": "SQLite is unavailable.",
            "findings": [
                {
                    "status": "fail",
                    "message": f"Failed to open database {db_path}: {exc}",
                }
            ],
        }

    if messages == ["ok"]:
        return {
            "name": "sqlite",
            "status": "pass",
            "summary": "SQLite integrity check passed.",
            "findings": [
                {
                    "status": "pass",
                    "message": f"Database integrity is OK: {db_path}",
                }
            ],
        }

    return {
        "name": "sqlite",
        "status": "fail",
        "summary": f"SQLite integrity check failed with {len(messages)} issue(s).",
        "findings": [
            {"status": "fail", "message": message, "details": {"database": str(db_path)}}
            for message in messages
        ],
    }


def _check_stale_pids(db_path: Path) -> CheckReport:
    try:
        with connect(db_path) as conn:
            try:
                plan_rows = conn.execute(
                    "SELECT id, status, pid FROM plans "
                    "WHERE pid IS NOT NULL AND status IN ('running', 'awaiting_input')"
                ).fetchall()
                task_rows = conn.execute(
                    "SELECT id, status, pid FROM tasks "
                    "WHERE pid IS NOT NULL AND status IN ('running', 'review')"
                ).fetchall()
            except Exception as exc:
                return {
                    "name": "stale_pids",
                    "status": "fail",
                    "summary": "PID check could not run.",
                    "findings": [
                        {"status": "fail", "message": f"Failed to query active rows: {exc}"}
                    ],
                }
    except Exception as exc:
        return {
            "name": "stale_pids",
            "status": "fail",
            "summary": "PID check could not run.",
            "findings": [{"status": "fail", "message": f"Failed to open database: {exc}"}],
        }

    stale: list[CheckFinding] = []
    alive_count = 0

    for row in plan_rows:
        pid = int(row["pid"])
        if _pid_is_alive(pid):
            alive_count += 1
            continue
        stale.append(
            {
                "status": "warning",
                "message": f"Stale plan PID detected: plan={row['id']} pid={pid}",
                "details": {"entity": "plan", "id": row["id"], "status": row["status"], "pid": pid},
            }
        )

    for row in task_rows:
        pid = int(row["pid"])
        if _pid_is_alive(pid):
            alive_count += 1
            continue
        stale.append(
            {
                "status": "warning",
                "message": f"Stale task PID detected: task={row['id']} pid={pid}",
                "details": {"entity": "task", "id": row["id"], "status": row["status"], "pid": pid},
            }
        )

    checked = len(plan_rows) + len(task_rows)
    if stale:
        return {
            "name": "stale_pids",
            "status": "warning",
            "summary": f"Found {len(stale)} stale PID(s) across {checked} active record(s).",
            "findings": stale,
        }

    return {
        "name": "stale_pids",
        "status": "pass",
        "summary": f"No stale PIDs found across {checked} active record(s).",
        "findings": [
            {
                "status": "pass",
                "message": f"Active PIDs checked: {checked}, alive: {alive_count}",
                "details": {"checked": checked, "alive": alive_count},
            }
        ],
    }


_STALE_ENTITY_THRESHOLD_SECONDS = 3600


def _check_stale_entities(db_path: Path) -> CheckReport:
    """Warn when non-terminal plans/tasks have not changed status for over an hour."""
    try:
        with connect(db_path) as conn:
            try:
                rows = conn.execute(
                    "WITH latest_history AS ("
                    "    SELECT entity_type, entity_id, MAX(created_at) AS last_transition_at "
                    "    FROM status_history "
                    "    GROUP BY entity_type, entity_id"
                    "), "
                    "active_entities AS ("
                    "    SELECT "
                    "      'plan' AS entity_type, "
                    "      p.id AS id, "
                    "      p.status AS status, "
                    "      p.updated_at AS updated_at, "
                    "      p.created_at AS created_at, "
                    "      lh.last_transition_at AS last_transition_at "
                    "    FROM plans p "
                    "    LEFT JOIN latest_history lh "
                    "      ON lh.entity_type = 'plan' AND lh.entity_id = p.id "
                    "    WHERE p.status IN ('pending', 'running', 'awaiting_input') "
                    "    UNION ALL "
                    "    SELECT "
                    "      'task' AS entity_type, "
                    "      t.id AS id, "
                    "      t.status AS status, "
                    "      t.updated_at AS updated_at, "
                    "      t.created_at AS created_at, "
                    "      lh.last_transition_at AS last_transition_at "
                    "    FROM tasks t "
                    "    LEFT JOIN latest_history lh "
                    "      ON lh.entity_type = 'task' AND lh.entity_id = t.id "
                    "    WHERE t.status IN ('blocked', 'ready', 'running', 'review', 'approved')"
                    "      AND NOT (t.status = 'blocked' AND EXISTS ("
                    "        SELECT 1 FROM task_blocks tb "
                    "        WHERE tb.task_id = t.id AND tb.resolved = 0"
                    "      ))"
                    ") "
                    "SELECT "
                    "  entity_type, "
                    "  id, "
                    "  status, "
                    "  COALESCE(last_transition_at, updated_at, created_at) AS status_since, "
                    "  CASE "
                    "    WHEN last_transition_at IS NOT NULL THEN 'status_history' "
                    "    ELSE 'entity_timestamp' "
                    "  END AS timestamp_source, "
                    "  CAST(strftime('%s', 'now') AS INTEGER) - "
                    "  CAST(strftime('%s', COALESCE(last_transition_at, updated_at, created_at)) "
                    "    AS INTEGER) AS stuck_seconds "
                    "FROM active_entities "
                    "ORDER BY stuck_seconds DESC, entity_type, id"
                ).fetchall()
            except Exception as exc:
                return {
                    "name": "stale_entities",
                    "status": "fail",
                    "summary": "Stale-entity age check could not run.",
                    "findings": [
                        {"status": "fail", "message": f"Failed to query entity age: {exc}"}
                    ],
                }
    except Exception as exc:
        return {
            "name": "stale_entities",
            "status": "fail",
            "summary": "Stale-entity age check could not run.",
            "findings": [{"status": "fail", "message": f"Failed to open database: {exc}"}],
        }

    checked = len(rows)
    stale: list[CheckFinding] = []

    for row in rows:
        raw_stuck_seconds = row["stuck_seconds"]
        if raw_stuck_seconds is None:
            continue
        stuck_seconds = max(int(raw_stuck_seconds), 0)
        if stuck_seconds <= _STALE_ENTITY_THRESHOLD_SECONDS:
            continue
        stuck_duration = _format_stuck_duration(stuck_seconds)
        stale.append(
            {
                "status": "warning",
                "message": (
                    f"Stale {row['entity_type']} status: "
                    f"{row['entity_type']}={row['id']} status={row['status']} "
                    f"stuck_for={stuck_duration}"
                ),
                "details": {
                    "entity_type": row["entity_type"],
                    "id": row["id"],
                    "current_status": row["status"],
                    "stuck_duration": stuck_duration,
                    "stuck_seconds": stuck_seconds,
                    "status_since": row["status_since"],
                    "timestamp_source": row["timestamp_source"],
                },
            }
        )

    if stale:
        return {
            "name": "stale_entities",
            "status": "warning",
            "summary": (
                f"Found {len(stale)} entity(s) stuck for over 1h (out of {checked} active)."
            ),
            "findings": stale,
        }

    return {
        "name": "stale_entities",
        "status": "pass",
        "summary": f"No non-terminal entities stuck for over 1h across {checked} active record(s).",
        "findings": [
            {
                "status": "pass",
                "message": f"Checked {checked} active non-terminal plan/task record(s).",
                "details": {
                    "checked": checked,
                    "stale_threshold_seconds": _STALE_ENTITY_THRESHOLD_SECONDS,
                },
            }
        ],
    }


def _format_stuck_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, rem = divmod(seconds, 60)
        return f"{minutes}m {rem}s"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes}m"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _load_worktree_metadata(db_path: Path) -> tuple[list[Any], list[Any]]:
    """Load DB worktree references and project directories."""
    with connect(db_path) as conn:
        refs = conn.execute(
            "SELECT t.id AS task_id, t.status AS task_status, "
            "t.worktree, t.branch, "
            "p.id AS project_id, p.name AS project_name, p.dir AS project_dir "
            "FROM tasks t "
            "JOIN plans pl ON t.plan_id = pl.id "
            "JOIN projects p ON pl.project_id = p.id "
            "WHERE t.worktree IS NOT NULL"
        ).fetchall()
        projects = conn.execute("SELECT id, name, dir FROM projects").fetchall()
    return refs, projects


def _scan_db_worktree_refs(
    refs: list[Any], terminal_statuses: set[str]
) -> tuple[list[CheckFinding], set[str]]:
    """Validate DB worktree references and detect terminal-task leftovers."""
    findings: list[CheckFinding] = []
    referenced_worktrees: set[str] = set()
    for row in refs:
        project_dir = Path(str(row["project_dir"])).expanduser()
        worktree_path = Path(str(row["worktree"])).expanduser()
        if not worktree_path.is_absolute():
            worktree_path = project_dir / worktree_path
        referenced_worktrees.add(_normalize_path(worktree_path))
        if not worktree_path.exists():
            findings.append(
                {
                    "status": "warning",
                    "message": f"DB references missing worktree: task={row['task_id']}",
                    "details": {
                        "kind": "db_missing_worktree",
                        "project_id": row["project_id"],
                        "project_name": row["project_name"],
                        "task_id": row["task_id"],
                        "task_status": row["task_status"],
                        "worktree": str(worktree_path),
                    },
                }
            )
            continue
        if row["task_status"] in terminal_statuses:
            findings.append(
                {
                    "status": "warning",
                    "message": (
                        f"Terminal task still has worktree: "
                        f"task={row['task_id']} status={row['task_status']}"
                    ),
                    "details": {
                        "kind": "terminal_task_worktree",
                        "project_id": row["project_id"],
                        "project_name": row["project_name"],
                        "project_dir": str(project_dir),
                        "task_id": row["task_id"],
                        "task_status": row["task_status"],
                        "worktree": str(worktree_path),
                        "branch": row["branch"],
                    },
                }
            )
    return findings, referenced_worktrees


def _scan_filesystem_worktrees(
    projects: list[Any], findings: list[CheckFinding]
) -> dict[str, dict[str, object]]:
    """Collect on-disk worktrees and record invalid project root issues."""
    filesystem_worktrees: dict[str, dict[str, object]] = {}
    for row in projects:
        project_dir = Path(str(row["dir"])).expanduser()
        if not project_dir.exists():
            findings.append(
                {
                    "status": "warning",
                    "message": f"Project directory is missing: {project_dir}",
                    "details": {
                        "kind": "missing_project_dir",
                        "project_id": row["id"],
                        "project_name": row["name"],
                        "project_dir": str(project_dir),
                    },
                }
            )
            continue

        worktree_root = project_dir / ".agm" / "worktrees"
        if not worktree_root.exists():
            continue
        if not worktree_root.is_dir():
            findings.append(
                {
                    "status": "warning",
                    "message": f"Expected directory but found non-directory: {worktree_root}",
                    "details": {
                        "kind": "invalid_worktree_root",
                        "project_id": row["id"],
                        "project_name": row["name"],
                        "path": str(worktree_root),
                    },
                }
            )
            continue

        for child in worktree_root.iterdir():
            if not child.is_dir():
                continue
            filesystem_worktrees[_normalize_path(child)] = {
                "project_id": row["id"],
                "project_name": row["name"],
                "project_dir": str(project_dir),
                "worktree": str(child),
            }
    return filesystem_worktrees


def _append_orphan_worktree_findings(
    filesystem_worktrees: dict[str, dict[str, object]],
    referenced_worktrees: set[str],
    findings: list[CheckFinding],
) -> None:
    """Append warnings for on-disk worktrees not referenced in DB."""
    for normalized, meta in filesystem_worktrees.items():
        if normalized in referenced_worktrees:
            continue
        findings.append(
            {
                "status": "warning",
                "message": f"Orphaned on-disk worktree: {meta['worktree']}",
                "details": {
                    "kind": "filesystem_orphan",
                    "project_id": meta["project_id"],
                    "project_name": meta["project_name"],
                    "project_dir": meta["project_dir"],
                    "worktree": meta["worktree"],
                },
            }
        )


def _check_orphaned_worktrees(db_path: Path) -> CheckReport:
    try:
        refs, projects = _load_worktree_metadata(db_path)
    except Exception as exc:
        message = f"Failed to load worktree metadata: {exc}"
        return {
            "name": "worktrees",
            "status": "fail",
            "summary": "Worktree correlation check could not run.",
            "findings": [{"status": "fail", "message": message}],
        }

    findings, referenced_worktrees = _scan_db_worktree_refs(refs, TASK_TERMINAL_STATUSES)
    filesystem_worktrees = _scan_filesystem_worktrees(projects, findings)
    _append_orphan_worktree_findings(filesystem_worktrees, referenced_worktrees, findings)

    if findings:
        return {
            "name": "worktrees",
            "status": "warning",
            "summary": f"Found {len(findings)} worktree correlation warning(s).",
            "findings": findings,
        }
    return {
        "name": "worktrees",
        "status": "pass",
        "summary": "Worktree references and on-disk state are consistent.",
        "findings": [
            {
                "status": "pass",
                "message": (
                    f"Checked {len(refs)} DB worktree reference(s) and "
                    f"{len(filesystem_worktrees)} on-disk worktree(s)."
                ),
                "details": {
                    "db_refs": len(refs),
                    "filesystem_worktrees": len(filesystem_worktrees),
                },
            }
        ],
    }


def _normalize_path(path: Path) -> str:
    return str(path.resolve(strict=False))


def _check_worktree_conflicts(db_path: Path) -> CheckReport:
    """Detect merge conflicts between active worktrees using clash."""
    try:
        with connect(db_path) as conn:
            projects = conn.execute("SELECT id, name, dir FROM projects").fetchall()
    except Exception as exc:
        return {
            "name": "conflicts",
            "status": "fail",
            "summary": "Conflict check could not run.",
            "findings": [{"status": "fail", "message": f"Failed to query projects: {exc}"}],
        }

    if not projects:
        return {
            "name": "conflicts",
            "status": "pass",
            "summary": "No projects to check for worktree conflicts.",
            "findings": [],
        }

    findings: list[CheckFinding] = []
    total_conflicts = 0
    clash_available = True

    for proj in projects:
        project_dir = str(proj["dir"])
        result = detect_worktree_conflicts(project_dir)

        if not result["available"]:
            clash_available = False
            findings.append(
                {
                    "status": "pass",
                    "message": "clash not installed - skipping conflict detection",
                    "details": {"error": result["error"]},
                }
            )
            break

        if result["error"]:
            findings.append(
                {
                    "status": "warning",
                    "message": f"clash error for {proj['name']}: {result['error']}",
                    "details": {"project": proj["name"], "error": result["error"]},
                }
            )
            continue

        real = get_real_conflicts(result)
        if real:
            total_conflicts += len(real)
            for c in real:
                files = c["conflicting_files"]
                findings.append(
                    {
                        "status": "warning",
                        "message": (
                            f"{proj['name']}: {c['wt1_id']} vs {c['wt2_id']} "
                            f"conflict on {len(files)} file(s)"
                        ),
                        "details": {
                            "project": proj["name"],
                            "wt1": c["wt1_id"],
                            "wt2": c["wt2_id"],
                            "files": files,
                        },
                    }
                )
        else:
            wt_count = len(result["worktrees"])
            findings.append(
                {
                    "status": "pass",
                    "message": f"{proj['name']}: {wt_count} worktree(s), no conflicts",
                    "details": {"project": proj["name"], "worktrees": wt_count},
                }
            )

    if not clash_available:
        return {
            "name": "conflicts",
            "status": "pass",
            "summary": "clash not installed — conflict detection skipped.",
            "findings": findings,
        }

    if total_conflicts > 0:
        return {
            "name": "conflicts",
            "status": "warning",
            "summary": f"Found {total_conflicts} worktree conflict pair(s).",
            "findings": findings,
        }

    return {
        "name": "conflicts",
        "status": "pass",
        "summary": "No worktree conflicts detected.",
        "findings": findings,
    }


def _check_disk_usage(db_path: Path) -> CheckReport:
    try:
        with connect(db_path) as conn:
            try:
                projects = conn.execute("SELECT id, name, dir FROM projects").fetchall()
            except Exception as exc:
                return {
                    "name": "disk_usage",
                    "status": "fail",
                    "summary": "Disk-usage check could not run.",
                    "findings": [{"status": "fail", "message": f"Failed to query projects: {exc}"}],
                }
    except Exception as exc:
        return {
            "name": "disk_usage",
            "status": "fail",
            "summary": "Disk-usage check could not run.",
            "findings": [{"status": "fail", "message": f"Failed to open database: {exc}"}],
        }

    findings: list[CheckFinding] = []
    had_warning = False

    db_file = db_path.expanduser()
    db_file_size = db_file.stat().st_size if db_file.exists() else 0
    db_usage_target = db_file if db_file.exists() else db_file.parent
    try:
        db_usage = shutil.disk_usage(db_usage_target)
        findings.append(
            {
                "status": "pass",
                "message": f"Database disk usage captured for {db_usage_target}",
                "details": {
                    "database": str(db_file),
                    "database_bytes": db_file_size,
                    "disk_total_bytes": db_usage.total,
                    "disk_used_bytes": db_usage.used,
                    "disk_free_bytes": db_usage.free,
                },
            }
        )
    except OSError as exc:
        had_warning = True
        findings.append(
            {
                "status": "warning",
                "message": f"Failed to read disk usage for database path: {exc}",
                "details": {"database": str(db_file)},
            }
        )

    for row in projects:
        project_dir = Path(str(row["dir"])).expanduser()
        if not project_dir.exists():
            had_warning = True
            findings.append(
                {
                    "status": "warning",
                    "message": f"Project directory missing for disk report: {project_dir}",
                    "details": {
                        "project_id": row["id"],
                        "project_name": row["name"],
                        "project_dir": str(project_dir),
                    },
                }
            )
            continue

        try:
            usage = shutil.disk_usage(project_dir)
        except OSError as exc:
            had_warning = True
            findings.append(
                {
                    "status": "warning",
                    "message": f"Failed to read disk usage for {project_dir}: {exc}",
                    "details": {
                        "project_id": row["id"],
                        "project_name": row["name"],
                        "project_dir": str(project_dir),
                    },
                }
            )
            continue

        worktree_root = project_dir / ".agm" / "worktrees"
        worktree_bytes = _directory_size(worktree_root)
        findings.append(
            {
                "status": "pass",
                "message": f"Disk usage captured for project {row['name']}",
                "details": {
                    "project_id": row["id"],
                    "project_name": row["name"],
                    "project_dir": str(project_dir),
                    "disk_total_bytes": usage.total,
                    "disk_used_bytes": usage.used,
                    "disk_free_bytes": usage.free,
                    "worktree_bytes": worktree_bytes,
                },
            }
        )

    if had_warning:
        return {
            "name": "disk_usage",
            "status": "warning",
            "summary": "Disk usage captured with warnings.",
            "findings": findings,
        }

    return {
        "name": "disk_usage",
        "status": "pass",
        "summary": f"Disk usage captured for {1 + len(projects)} location(s).",
        "findings": findings,
    }


def _directory_size(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _check_worktree_cleanliness(db_path: Path) -> CheckReport:
    """Check whether configured worktrees are clean according to git status."""
    findings: list[CheckFinding] = []
    inspected_worktrees = 0

    try:
        with connect(db_path) as conn:
            try:
                projects = list_projects(conn)
            except Exception as exc:
                return {
                    "name": "worktree_cleanliness",
                    "status": "fail",
                    "summary": "Worktree cleanliness check could not run.",
                    "findings": [
                        {
                            "status": "fail",
                            "message": f"Failed to query projects: {exc}",
                        }
                    ],
                }
    except Exception as exc:
        return {
            "name": "worktree_cleanliness",
            "status": "fail",
            "summary": "Worktree cleanliness check could not run.",
            "findings": [{"status": "fail", "message": f"Failed to open database: {exc}"}],
        }

    import subprocess

    for project in projects:
        project_id = project["id"]
        project_name = project["name"]
        project_dir = Path(str(project["dir"])).expanduser()
        worktree_root = project_dir / ".agm" / "worktrees"

        if not worktree_root.exists():
            continue
        if not worktree_root.is_dir():
            findings.append(
                {
                    "status": "warning",
                    "message": f"Worktree root is not a directory: {worktree_root}",
                    "details": {
                        "kind": "invalid_worktree_root",
                        "project_id": project_id,
                        "project_name": project_name,
                        "worktree_root": str(worktree_root),
                    },
                }
            )
            continue

        try:
            children = sorted(worktree_root.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            findings.append(
                {
                    "status": "warning",
                    "message": f"Unable to list worktrees for project: {project_name}",
                    "details": {
                        "kind": "worktree_list_error",
                        "project_id": project_id,
                        "project_name": project_name,
                        "error": str(exc),
                    },
                }
            )
            continue

        for worktree_dir in children:
            if not worktree_dir.is_dir():
                continue
            inspected_worktrees += 1
            try:
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(worktree_dir),
                    capture_output=True,
                    text=True,
                )
            except Exception as exc:
                findings.append(
                    {
                        "status": "warning",
                        "message": f"Unable to run git status for worktree: {worktree_dir}",
                        "details": {
                            "kind": "worktree_status_unavailable",
                            "project_id": project_id,
                            "project_name": project_name,
                            "worktree": str(worktree_dir),
                            "error": str(exc),
                        },
                    }
                )
                continue

            if status.returncode != 0:
                findings.append(
                    {
                        "status": "warning",
                        "message": f"Could not check worktree cleanliness for {worktree_dir}",
                        "details": {
                            "kind": "worktree_status_unavailable",
                            "project_id": project_id,
                            "project_name": project_name,
                            "worktree": str(worktree_dir),
                            "return_code": status.returncode,
                            "stderr": status.stderr.strip(),
                        },
                    }
                )
                continue

            porcelain = status.stdout.strip()
            if not porcelain:
                continue
            dirty_paths = porcelain.splitlines()
            findings.append(
                {
                    "status": "warning",
                    "message": f"Dirty worktree detected: {worktree_dir}",
                    "details": {
                        "kind": "dirty_worktree",
                        "project_id": project_id,
                        "project_name": project_name,
                        "worktree": str(worktree_dir),
                        "status_line_count": len(dirty_paths),
                        "porcelain": porcelain,
                    },
                }
            )

    if inspected_worktrees == 0 and not findings:
        return {
            "name": "worktree_cleanliness",
            "status": "pass",
            "summary": "No configured worktree directories found to inspect.",
            "findings": [],
        }

    warning_count = len(findings)
    if warning_count:
        dirty_count = sum(
            1 for finding in findings if finding.get("details", {}).get("kind") == "dirty_worktree"
        )
        unavailable_count = warning_count - dirty_count
        if dirty_count:
            summary = (
                f"Found {dirty_count} dirty worktree(s) "
                f"and {unavailable_count} unavailable worktree(s)."
            )
        else:
            summary = f"Could not verify {unavailable_count} worktree(s)."
        return {
            "name": "worktree_cleanliness",
            "status": "warning",
            "summary": summary,
            "findings": findings,
        }

    return {
        "name": "worktree_cleanliness",
        "status": "pass",
        "summary": f"Inspected {inspected_worktrees} worktree(s); all are clean.",
        "findings": [
            {
                "status": "pass",
                "message": f"Inspected {inspected_worktrees} worktree(s), all clean.",
                "details": {"inspected_worktrees": inspected_worktrees},
            }
        ],
    }


# -- fix logic --


_STALE_LOG_DAYS = 7


def _collect_log_entity_ids(db_path: Path) -> tuple[set[str], set[str]]:
    """Collect known and active job-id prefixes from plans/tasks."""
    active_ids: set[str] = set()
    known_ids: set[str] = set()
    with connect(db_path) as conn:
        for row in conn.execute("SELECT id, status FROM plans").fetchall():
            for prefix in ("plan-", "tasks-"):
                known_ids.add(f"{prefix}{row['id']}")
            if row["status"] in ("pending", "running", "awaiting_input"):
                active_ids.add(f"plan-{row['id']}")
                active_ids.add(f"tasks-{row['id']}")
        for row in conn.execute("SELECT id, status FROM tasks").fetchall():
            for prefix in ("exec-", "review-", "merge-"):
                known_ids.add(f"{prefix}{row['id']}")
            if row["status"] in ("blocked", "ready", "running", "review", "approved"):
                active_ids.add(f"exec-{row['id']}")
                active_ids.add(f"review-{row['id']}")
                active_ids.add(f"merge-{row['id']}")
    return active_ids, known_ids


def _classify_worker_logs(
    log_files: list[Path],
    known_ids: set[str],
    active_ids: set[str],
    cutoff: float,
) -> tuple[int, list[Path], list[Path]]:
    """Classify worker logs into stale/orphaned groups and sum size."""
    total_bytes = 0
    stale_files: list[Path] = []
    orphaned_files: list[Path] = []
    for log_file in log_files:
        try:
            stat = log_file.stat()
        except OSError:
            continue
        total_bytes += stat.st_size
        job_id = log_file.stem
        if known_ids and job_id not in known_ids:
            orphaned_files.append(log_file)
            continue
        if stat.st_mtime < cutoff and job_id not in active_ids:
            stale_files.append(log_file)
    return total_bytes, stale_files, orphaned_files


def _check_log_files(db_path: Path) -> CheckReport:
    """Check worker log files for count, size, and staleness."""
    import time

    if not LOG_DIR.exists():
        return {
            "name": "log_files",
            "status": "pass",
            "summary": "No worker log directory found.",
            "findings": [
                {
                    "status": "pass",
                    "message": f"Log directory does not exist: {LOG_DIR}",
                }
            ],
        }

    active_ids: set[str] = set()
    known_ids: set[str] = set()
    with contextlib.suppress(Exception):
        active_ids, known_ids = _collect_log_entity_ids(db_path)

    log_files = list(LOG_DIR.glob("*.log"))
    cutoff = time.time() - (_STALE_LOG_DAYS * 86400)
    total_bytes, stale_files, orphaned_files = _classify_worker_logs(
        log_files, known_ids, active_ids, cutoff
    )

    findings: list[CheckFinding] = [
        {
            "status": "pass",
            "message": (f"{len(log_files)} log file(s), {total_bytes:,} bytes total in {LOG_DIR}"),
            "details": {
                "log_dir": str(LOG_DIR),
                "file_count": len(log_files),
                "total_bytes": total_bytes,
            },
        }
    ]

    if stale_files or orphaned_files:
        for sf in stale_files:
            findings.append(
                {
                    "status": "warning",
                    "message": f"Stale log file (>{_STALE_LOG_DAYS}d): {sf.name}",
                    "details": {
                        "kind": "stale_log",
                        "path": str(sf),
                        "name": sf.name,
                    },
                }
            )
        for of in orphaned_files:
            findings.append(
                {
                    "status": "warning",
                    "message": f"Orphaned log file (entity deleted): {of.name}",
                    "details": {
                        "kind": "stale_log",
                        "path": str(of),
                        "name": of.name,
                    },
                }
            )
        parts: list[str] = []
        if stale_files:
            parts.append(f"{len(stale_files)} stale (>{_STALE_LOG_DAYS}d old)")
        if orphaned_files:
            parts.append(f"{len(orphaned_files)} orphaned (entity deleted)")
        return {
            "name": "log_files",
            "status": "warning",
            "summary": f"{len(log_files)} log file(s), {', '.join(parts)}.",
            "findings": findings,
        }

    return {
        "name": "log_files",
        "status": "pass",
        "summary": f"{len(log_files)} log file(s), {total_bytes:,} bytes, none stale.",
        "findings": findings,
    }


def _fix_stale_logs(check: CheckReport) -> FixAction:
    """Remove stale log files."""
    action = _new_fix_action()
    stale = [
        f
        for f in check["findings"]
        if f.get("status") == "warning" and f.get("details", {}).get("kind") == "stale_log"
    ]
    for finding in stale:
        path_str = str(finding.get("details", {}).get("path", ""))
        if not path_str:
            continue
        action["attempted"] += 1
        try:
            Path(path_str).unlink(missing_ok=True)
            action["fixed"] += 1
        except Exception as exc:
            action["failed"] += 1
            action["failures"].append({"target": path_str, "reason": str(exc)})
    return action


def _new_fix_action() -> FixAction:
    return {"attempted": 0, "fixed": 0, "failed": 0, "failures": []}


def _apply_fixes(db_path: Path, checks: list[CheckReport]) -> dict[str, FixAction]:
    """Apply best-effort remediation for fixable checks."""
    actions: dict[str, FixAction] = {}
    checks_by_name = {c["name"]: c for c in checks}

    stale = checks_by_name.get("stale_pids")
    if stale and stale["status"] != "pass":
        actions["stale_pids"] = _fix_stale_pids(db_path, stale)

    orphans = checks_by_name.get("worktrees")
    if orphans and orphans["status"] != "pass":
        actions["orphaned_worktrees"] = _fix_orphaned_worktrees(db_path, orphans)

    logs = checks_by_name.get("log_files")
    if logs and logs["status"] != "pass":
        actions["stale_logs"] = _fix_stale_logs(logs)

    return actions


def _rerun_fixable_checks(db_path: Path, checks: list[CheckReport]) -> list[CheckReport]:
    """Re-run checks that can be fixed by --fix."""
    check_index_by_name = {check["name"]: index for index, check in enumerate(checks)}

    stale_index = check_index_by_name.get("stale_pids")
    if stale_index is not None:
        checks[stale_index] = _check_stale_pids(db_path)

    worktrees_index = check_index_by_name.get("worktrees")
    if worktrees_index is not None:
        checks[worktrees_index] = _check_orphaned_worktrees(db_path)

    log_index = check_index_by_name.get("log_files")
    if log_index is not None:
        checks[log_index] = _check_log_files(db_path)

    return checks


def _fix_stale_pids(db_path: Path, check: CheckReport) -> FixAction:
    """Mark stale-PID plans/tasks as failed using db.py doctor helpers.

    Uses fail_stale_running_plan_for_doctor / fail_stale_running_task_for_doctor
    which clear pid/thread_id, log an audit trail, and record status_history.
    """
    action = _new_fix_action()
    stale_findings = [f for f in check["findings"] if f.get("status") == "warning"]
    if not stale_findings:
        return action

    try:
        with connect(db_path) as conn:
            for finding in stale_findings:
                details: dict[str, Any] = finding.get("details", {})
                entity = details.get("entity", "")
                entity_id = str(details.get("id", ""))
                old_status = str(details.get("status", "running"))
                reason = f"PID {details.get('pid', '?')} no longer alive"
                action["attempted"] += 1
                try:
                    if entity == "plan":
                        fail_stale_running_plan_for_doctor(
                            conn, entity_id, old_status=old_status, reason=reason
                        )
                    elif entity == "task":
                        fail_stale_running_task_for_doctor(
                            conn, entity_id, old_status=old_status, reason=reason
                        )
                    else:
                        action["failed"] += 1
                        action["failures"].append(
                            {"target": entity_id, "reason": f"unknown entity: {entity}"}
                        )
                        continue
                    action["fixed"] += 1
                except Exception as exc:
                    action["failed"] += 1
                    action["failures"].append({"target": entity_id, "reason": str(exc)})
    except Exception as exc:
        action["attempted"] = len(stale_findings)
        action["failed"] = len(stale_findings)
        action["failures"].append({"target": "db", "reason": str(exc)})
    return action


def _is_fixable_orphaned_worktree_finding(finding: CheckFinding) -> bool:
    """Return whether finding kind is fixable by orphaned-worktree fixer."""
    if finding.get("status") != "warning":
        return False
    kind = finding.get("details", {}).get("kind")
    return kind in ("filesystem_orphan", "db_missing_worktree", "terminal_task_worktree")


def _apply_orphaned_worktree_fix(conn, details: dict[str, Any]) -> str:
    """Apply one orphaned-worktree fix and return target identifier."""
    kind = details.get("kind", "")
    if kind == "filesystem_orphan":
        worktree = str(details.get("worktree", ""))
        if worktree and Path(worktree).exists():
            shutil.rmtree(worktree)
        # Prune stale git worktree records after directory removal
        import subprocess

        project_dir = str(details.get("project_dir", ""))
        if project_dir and Path(project_dir).exists():
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=project_dir,
                capture_output=True,
                check=False,
            )
        return worktree

    task_id = str(details.get("task_id", ""))
    if kind == "db_missing_worktree":
        clear_stale_task_git_refs_for_doctor(
            conn,
            task_id,
            reason="worktree directory missing from filesystem",
        )
        return task_id

    if kind == "terminal_task_worktree":
        worktree = str(details.get("worktree", ""))
        branch = str(details.get("branch", ""))
        project_dir = str(details.get("project_dir", ""))
        task_status = details.get("task_status", "?")
        if worktree and project_dir:
            remove_worktree(project_dir, worktree, branch)
            if Path(worktree).exists():
                shutil.rmtree(worktree)
        clear_stale_task_git_refs_for_doctor(
            conn,
            task_id,
            reason=f"terminal task ({task_status}) worktree cleanup",
        )
        return task_id

    raise ValueError(f"Unsupported worktree fix kind: {kind}")


def _fix_orphaned_worktrees(db_path: Path, check: CheckReport) -> FixAction:
    """Remove orphaned on-disk worktrees, clear dangling DB refs, and clean up terminal tasks."""
    action = _new_fix_action()
    fixable = [
        finding for finding in check["findings"] if _is_fixable_orphaned_worktree_finding(finding)
    ]
    if not fixable:
        return action

    try:
        with connect(db_path) as conn:
            for finding in fixable:
                details: dict[str, Any] = finding.get("details", {})
                action["attempted"] += 1
                try:
                    _apply_orphaned_worktree_fix(conn, details)
                    action["fixed"] += 1
                except Exception as exc:
                    action["failed"] += 1
                    target = details.get("worktree") or details.get("task_id", "?")
                    action["failures"].append({"target": str(target), "reason": str(exc)})
    except Exception as exc:
        action["attempted"] = len(fixable)
        action["failed"] = len(fixable)
        action["failures"].append({"target": "db", "reason": str(exc)})
    return action
