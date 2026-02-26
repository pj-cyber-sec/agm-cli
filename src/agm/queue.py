"""rq-based job queue for agm.

Domain commands (plan create, task create, etc.) enqueue jobs here.
The queue group in the CLI is for monitoring and debugging only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime

from redis import ConnectionPool, Redis
from redis.exceptions import RedisError
from rq import Callback, Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

from agm.paths import AGM_CONFIG_DIR

log = logging.getLogger(__name__)

REDIS_URL = os.environ.get("AGM_REDIS_URL", "redis://localhost:6379/0")

# Named queues
QUEUE_ENRICHMENT = "agm:enrichment"
QUEUE_EXPLORER = "agm:explorer"
QUEUE_PLANS = "agm:plans"
QUEUE_TASKS = "agm:tasks"
QUEUE_EXEC = "agm:exec"
QUEUE_COORD = "agm:coordinator"
QUEUE_REVIEW = "agm:review"
QUEUE_MERGE = "agm:merge"
QUEUE_EXTERNAL = "agm:external"
QUEUE_SETUP = "agm:setup"
AGM_QUEUE_NAMES = (
    QUEUE_ENRICHMENT,
    QUEUE_EXPLORER,
    QUEUE_PLANS,
    QUEUE_TASKS,
    QUEUE_EXEC,
    QUEUE_COORD,
    QUEUE_REVIEW,
    QUEUE_MERGE,
    QUEUE_EXTERNAL,
    QUEUE_SETUP,
)

# Redis Stream for durable event delivery (XADD/XREAD)
FAILURE_TTL = 7 * 24 * 3600  # 7 days — auto-expire failed jobs from Redis

EVENTS_STREAM = "agm:events:stream"
# Max entries retained in the stream (~1000 keeps ~30min of busy pipeline)
EVENTS_STREAM_MAXLEN = int(os.environ.get("AGM_EVENTS_STREAM_MAXLEN", "1000"))

LOG_DIR = AGM_CONFIG_DIR / "logs"


_pool = ConnectionPool.from_url(REDIS_URL)


def get_redis() -> Redis:
    return Redis(connection_pool=_pool)


def get_queue(name: str = QUEUE_PLANS) -> Queue:
    # No rq-level timeout (-1 disables). Each job type owns its timeout
    # at the application level (e.g. asyncio.wait_for in jobs.py).
    # Burst workers exit when the job finishes — no resource leak risk.
    return Queue(name, connection=get_redis(), default_timeout=-1)


EVENT_VERSION = 1  # Bump when payload shape changes


def publish_event(
    event_type: str,
    entity_id: str,
    status: str,
    *,
    project: str,
    plan_id: str | None = None,
    source: str = "worker",
    extra: dict | None = None,
) -> None:
    """Publish a pipeline event to Redis Stream. Best-effort, never raises.

    *source* identifies the producer: ``"worker"`` (rq job) or ``"cli"``
    (interactive CLI command).

    *extra* is an optional dict of additional fields merged into the payload.
    Used by streaming notifications to include item/plan details.
    """
    import uuid

    event: dict = {
        "event_id": str(uuid.uuid4()),
        "type": event_type,
        "id": entity_id,
        "plan_id": plan_id or entity_id,
        "project": project,
        "status": status,
        "source": source,
        "v": EVENT_VERSION,
        "ts": datetime.now(UTC).isoformat(),
    }
    if extra:
        event.update(extra)
    payload = json.dumps(event)
    try:
        r = get_redis()
        r.xadd(EVENTS_STREAM, {"data": payload}, maxlen=EVENTS_STREAM_MAXLEN, approximate=True)
    except RedisError:
        log.warning("Event publish failed (Redis unavailable): %s %s", event_type, entity_id)


RATE_LIMITS_KEY = "agm:codex:rate-limits"
RATE_LIMITS_TTL = 21600  # 6 hours


def _is_legacy_snapshot(data: dict) -> bool:
    """True if data is an old flat snapshot rather than a keyed-bucket dict."""
    return "captured_at" in data


def store_codex_rate_limits(snapshot: dict) -> None:
    """Best-effort upsert of a rate-limit bucket keyed by limit_id."""
    try:
        r = get_redis()
        raw: bytes | None = r.get(RATE_LIMITS_KEY)  # type: ignore[assignment]
        existing: dict = json.loads(raw) if raw else {}
        if _is_legacy_snapshot(existing):
            existing = {}  # discard old format
        key = snapshot.get("limit_id") or "default"
        buckets = existing
        buckets[key] = snapshot
        r.set(RATE_LIMITS_KEY, json.dumps(buckets), ex=RATE_LIMITS_TTL)
    except (RedisError, json.JSONDecodeError):
        log.debug("Rate-limit storage failed (Redis unavailable)")


def get_codex_rate_limits_safe() -> list[dict] | None:
    """Read all rate-limit buckets as a list. Returns None on miss or error."""
    try:
        r = get_redis()
        raw: bytes | None = r.get(RATE_LIMITS_KEY)  # type: ignore[assignment]
        if not raw:
            return None
        data: dict = json.loads(raw)
        if _is_legacy_snapshot(data):
            # Migrate old flat snapshot: treat as single default bucket.
            data.setdefault("limit_id", None)
            data.setdefault("limit_name", None)
            return [data]
        return list(data.values()) if data else None
    except (RedisError, json.JSONDecodeError):
        return None


def publish_task_model_escalation(
    task_id: str,
    old_model: str,
    new_model: str,
    rejection_count: int,
    project: str,
    plan_id: str | None = None,
) -> None:
    """Publish model escalation event for task executor retries."""
    publish_event(
        "task:model_escalated",
        task_id,
        "escalated",
        project=project,
        plan_id=plan_id,
        extra={
            "task_id": task_id,
            "old_model": old_model,
            "new_model": new_model,
            "rejection_count": rejection_count,
        },
    )


class EventSubscriber:
    """Iterator over Redis Stream events with optional filtering.

    Uses ``XREAD BLOCK`` for durable, cursor-based event delivery.
    Blocks up to ``timeout`` seconds waiting for the next event. Returns
    the parsed event dict, or ``None`` on timeout (callers use this for
    fallback DB checks).  Gracefully degrades when Redis is unavailable —
    ``__next__`` sleeps for ``timeout`` and returns ``None``.
    """

    def __init__(
        self,
        *,
        plan_id: str | None = None,
        task_id: str | None = None,
        project: str | None = None,
        timeout: float = 30.0,
        cursor: str = "$",
    ):
        self.plan_id = plan_id
        self.task_id = task_id
        self.project = project
        self.timeout = timeout
        self._cursor = cursor  # "$" = only new entries, "0" = from beginning
        self._redis_available = True
        try:
            self._redis = get_redis()
            # Verify stream exists (or will be created on first XADD)
            self._redis.ping()
        except Exception:
            self._redis_available = False
            self._redis = None

    def __iter__(self):
        return self

    def _xread_batch(self) -> list:
        """Read one batch of stream entries."""
        assert self._redis is not None
        timeout_ms = int(self.timeout * 1000)
        return self._redis.xread(  # type: ignore[assignment]
            {EVENTS_STREAM: self._cursor}, block=timeout_ms, count=10
        )

    @staticmethod
    def _decode_stream_event(entry_id, fields) -> dict | None:
        """Decode one stream entry payload to an event dict."""
        data = fields.get("data") or fields.get(b"data")
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode()
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(event, dict):
            return None
        event["_stream_id"] = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
        return event

    def _matches_filters(self, event: dict) -> bool:
        """Return whether event matches subscriber scope filters."""
        if self.plan_id and event.get("plan_id") != self.plan_id:
            return False
        if self.task_id and event.get("id") != self.task_id:
            return False
        return not (self.project and event.get("project") != self.project)

    def __next__(self) -> dict | None:
        """Return the next matching event, or None on timeout."""
        import time

        if not self._redis_available:
            time.sleep(self.timeout)
            return None
        while True:
            # XREAD BLOCK returns list of [stream_name, [(entry_id, fields), ...]]
            result = self._xread_batch()
            if not result:
                return None  # timeout — caller can do fallback DB check
            for _stream_name, entries in result:
                for entry_id, fields in entries:
                    self._cursor = entry_id  # advance cursor
                    event = self._decode_stream_event(entry_id, fields)
                    if event is None:
                        continue
                    if not self._matches_filters(event):
                        continue
                    return event
            # All entries in this batch were filtered out — loop to XREAD again

    def close(self):
        """No-op for stream-based subscriber (no subscription to clean up)."""
        pass


def subscribe_events(
    *,
    plan_id: str | None = None,
    task_id: str | None = None,
    project: str | None = None,
    timeout: float = 30.0,
) -> EventSubscriber:
    """Convenience factory for EventSubscriber."""
    return EventSubscriber(plan_id=plan_id, task_id=task_id, project=project, timeout=timeout)


def enqueue_enrichment(plan_id: str) -> Job:
    """Enqueue a plan for enrichment and ensure a worker is running.

    Called by `agm plan request`. Pushes the job to the enrichment queue.
    After enrichment completes, the job auto-enqueues planning.
    """
    from agm.jobs import run_enrichment

    q = get_queue(QUEUE_ENRICHMENT)
    job_id = f"enrich-{plan_id}"
    job = q.enqueue(
        run_enrichment,
        plan_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_enrichment_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Enrich {plan_id}",
    )
    _spawn_worker(QUEUE_ENRICHMENT, job_id=job_id)
    return job


def enqueue_explorer(plan_id: str) -> Job:
    """Enqueue codebase exploration for a plan (post-enrichment).

    Called after enrichment finalizes. Explorer runs before planning
    to pre-compute codebase context.
    """
    from agm.jobs import run_explorer

    q = get_queue(QUEUE_EXPLORER)
    job_id = f"explore-{plan_id}"
    job = q.enqueue(
        run_explorer,
        plan_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_explorer_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Explore for {plan_id}",
    )
    _spawn_worker(QUEUE_EXPLORER, job_id=job_id)
    return job


def enqueue_plan_request(plan_id: str) -> Job:
    """Enqueue a plan for planning (post-enrichment + exploration).

    Called after explorer completes (or on explorer failure as fallback).
    Pushes the job to the plans queue.
    """
    from agm.jobs import run_plan_request

    q = get_queue(QUEUE_PLANS)
    job_id = f"plan-{plan_id}"
    job = q.enqueue(
        run_plan_request,
        plan_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_plan_request_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Plan {plan_id}",
    )
    _spawn_worker(QUEUE_PLANS, job_id=job_id)
    return job


def enqueue_task_creation(plan_id: str) -> Job:
    """Enqueue task creation for a finalized plan.

    Called automatically after plan finalization. Pushes the job to Redis
    and spawns a background worker.
    """
    from agm.jobs import run_task_creation

    q = get_queue(QUEUE_TASKS)
    job_id = f"tasks-{plan_id}"
    job = q.enqueue(
        run_task_creation,
        plan_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_task_creation_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Tasks for plan {plan_id}",
    )
    _spawn_worker(QUEUE_TASKS, single=True, job_id=job_id)
    return job


def enqueue_task_refresh(project_id: str, prompt: str | None = None, backend: str = "codex") -> Job:
    """Enqueue a task refresh (reconciliation) for a project.

    Invokes the task agent to review and clean up stale/duplicate tasks.
    Uses single-worker mode to avoid concurrent task agent runs.
    """
    from agm.jobs import run_task_refresh

    q = get_queue(QUEUE_TASKS)
    job_id = f"refresh-{project_id}"
    job = q.enqueue(
        run_task_refresh,
        project_id,
        prompt,
        backend,
        job_id=job_id,
        failure_ttl=FAILURE_TTL,
        description=f"Task refresh for project {project_id}",
    )
    _spawn_worker(QUEUE_TASKS, single=True, job_id=job_id)
    return job


def enqueue_task_review(task_id: str) -> Job:
    """Enqueue a task for review by the reviewer agent.

    Uses the review queue. No single-worker constraint — multiple reviews
    can run in parallel (each in its own worktree).
    """
    from agm.jobs import run_task_review

    q = get_queue(QUEUE_REVIEW)
    job_id = f"review-{task_id}"
    job = q.enqueue(
        run_task_review,
        task_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_task_review_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Review task {task_id}",
    )
    _spawn_worker(QUEUE_REVIEW, job_id=job_id)
    return job


def enqueue_task_execution(task_id: str) -> Job:
    """Enqueue a task for execution by the executor agent.

    No single-worker constraint — multiple executors run in parallel
    (each in its own worktree).
    """
    from agm.jobs import run_task_execution

    job_id = f"exec-{task_id}"
    q = get_queue(QUEUE_EXEC)
    job = q.enqueue(
        run_task_execution,
        task_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_task_execution_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Execute task {task_id}",
    )
    _spawn_worker(QUEUE_EXEC, job_id=job_id)
    return job


def enqueue_plan_coordinator(plan_id: str) -> Job:
    """Enqueue a coordinator pass for a plan (deduped while active)."""
    from agm.jobs import run_plan_coordinator

    q = get_queue(QUEUE_COORD)
    job_id = f"coord-{plan_id}"
    existing = q.fetch_job(job_id)
    if existing is not None:
        status = existing.get_status(refresh=True)
        if status in {"queued", "started", "deferred", "scheduled"}:
            return existing
        with suppress(Exception):
            existing.delete()

    job = q.enqueue(
        run_plan_coordinator,
        plan_id,
        job_id=job_id,
        failure_ttl=FAILURE_TTL,
        result_ttl=60,
        description=f"Coordinate plan {plan_id}",
    )
    _spawn_worker(QUEUE_COORD, single=True, job_id=job_id)
    return job


def enqueue_task_merge(task_id: str) -> Job:
    """Enqueue a task merge job.

    Uses a serialized queue (agm:merge) with single-worker mode
    to prevent concurrent merges to main from conflicting.
    """
    from agm.jobs import run_task_merge

    q = get_queue(QUEUE_MERGE)
    job_id = f"merge-{task_id}"
    job = q.enqueue(
        run_task_merge,
        task_id,
        job_id=job_id,
        on_failure=Callback("agm.jobs.on_task_merge_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Merge task {task_id}",
    )
    _spawn_worker(QUEUE_MERGE, single=True, job_id=job_id)
    return job


def enqueue_external(
    prompt: str,
    *,
    output_schema: str | None = None,
    cwd: str | None = None,
    caller: str = "cli",
    backend: str = "codex",
    developer_instructions: str | None = None,
    effort: str | None = None,
    timeout: int | None = None,
) -> Job:
    """Enqueue a stateless external LLM call.

    No plan/task/project rows in agm's DB. Worker runs LLM, stores
    result in Redis with TTL, emits ``external:status`` event.
    Returns the rq Job (job.id is the external request ID).

    ``developer_instructions`` overrides the thread's developerInstructions
    (role identity, behavioral constraints).  ``effort`` overrides the turn
    effort level (e.g. ``"high"`` for quality-sensitive work).
    """
    import uuid

    from agm.jobs import run_external

    job_id = f"ext-{uuid.uuid4().hex[:12]}"
    q = get_queue(QUEUE_EXTERNAL)
    job = q.enqueue(
        run_external,
        prompt,
        output_schema,
        cwd,
        caller,
        backend,
        developer_instructions,
        effort,
        timeout,
        job_id=job_id,
        failure_ttl=FAILURE_TTL,
        description=f"External {job_id}",
    )
    _spawn_worker(QUEUE_EXTERNAL, job_id=job_id)
    return job


def enqueue_project_setup(project_id: str, project_name: str, *, backend: str | None = None) -> Job:
    """Enqueue project setup (LLM inspection) and spawn a worker.

    Called by ``agm project setup``.  Pushes job to the setup queue;
    worker runs the LLM inspector and applies quality-gate + post-merge
    config to the project row.
    """
    from agm.jobs_setup import run_project_setup_worker

    job_id = f"setup-{project_id}"
    q = get_queue(QUEUE_SETUP)
    job = q.enqueue(
        run_project_setup_worker,
        project_id,
        project_name,
        backend,
        job_id=job_id,
        on_failure=Callback("agm.jobs_setup.on_setup_failure"),
        failure_ttl=FAILURE_TTL,
        description=f"Setup {project_name}",
    )
    _spawn_worker(QUEUE_SETUP, job_id=job_id)
    return job


def _spawn_worker(
    queue_name: str = QUEUE_PLANS, *, single: bool = False, job_id: str | None = None
) -> None:
    """Spawn a background rq worker process for a specific queue.

    The worker processes jobs until the queue is empty (burst mode),
    then exits. Each enqueue call spawns its own worker so jobs
    are never left waiting.

    When single=True, only one worker is allowed at a time for the queue.
    If a worker is already processing a job, the new job will be picked up
    when the current one finishes — no second worker is spawned. This
    serializes execution, preventing concurrent task agents from clashing.

    When job_id is provided, worker stdout/stderr is captured to a log file
    at ~/.config/agm/logs/{job_id}.log. This captures rq startup output
    that would otherwise be lost to DEVNULL.
    """
    if single:
        redis = get_redis()
        q = Queue(queue_name, connection=redis)
        started = StartedJobRegistry(queue=q)
        lock_key = f"agm:worker-lock:{queue_name}"
        if len(started) > 0:
            log.info("Worker already active for %s, skipping spawn", queue_name)
            return
        # Registry empty — any leftover lock is stale (worker already exited).
        redis.delete(lock_key)
        # Atomic lock prevents TOCTOU race between registry check and Popen.
        if not redis.set(lock_key, "1", nx=True, ex=300):
            log.info("Spawn lock held for %s, skipping spawn", queue_name)
            return
    cmd = [
        sys.executable,
        "-m",
        "rq.cli",
        "worker",
        "--burst",
        "--url",
        REDIS_URL,
        queue_name,
    ]

    log_fh = None
    try:
        if job_id:
            LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            log_fh = open(LOG_DIR / f"{job_id}.log", "w")  # noqa: SIM115
            stdout_target = log_fh
            stderr_target = subprocess.STDOUT
        else:
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )
    except Exception:
        if log_fh is not None:
            log_fh.close()
        raise

    if log_fh is not None:
        log_fh.close()  # parent closes its copy; child keeps writing

    log.info("Spawned worker pid=%d for %s", proc.pid, queue_name)


def get_job(job_id: str) -> Job | None:
    """Fetch a job by ID."""
    try:
        return Job.fetch(job_id, connection=get_redis())
    except Exception:
        return None


def _queue_counts_template(value: int | None = None) -> dict[str, dict[str, int | None]]:
    return {
        name: {
            "queued": value,
            "running": value,
            "failed": value,
        }
        for name in AGM_QUEUE_NAMES
    }


def _count_live_started_jobs(q: Queue, started: StartedJobRegistry) -> int:
    """Return live started-job count, pruning stale started-registry entries."""
    running = 0
    for job_id in started.get_job_ids():
        try:
            job = Job.fetch(job_id, connection=q.connection)  # type: ignore[arg-type]
        except Exception:
            with suppress(Exception):
                started.remove(job_id, delete_job=False)
            continue

        try:
            status = job.get_status()
        except Exception:
            status = None
        if status != "started":
            with suppress(Exception):
                started.remove(job_id, delete_job=False)
            continue

        worker_name = getattr(job, "worker_name", None)
        if worker_name:
            worker_key = f"rq:worker:{worker_name}"
            try:
                if not bool(q.connection.exists(worker_key)):  # type: ignore[union-attr]
                    with suppress(Exception):
                        started.remove(job_id, delete_job=False)
                    continue
            except Exception:
                # If worker heartbeat lookup fails, keep the started entry.
                pass
        running += 1
    return running


def _exception_context(exc: Exception) -> dict[str, str]:
    message = str(exc) or repr(exc)
    return {
        "error_type": type(exc).__name__,
        "error": message,
    }


def check_redis_connection_safe() -> dict[str, bool | str | None]:
    """Probe Redis connectivity and never raise."""
    try:
        get_redis().ping()
        return {"ok": True, "error_type": None, "error": None}
    except Exception as exc:
        return {"ok": False, **_exception_context(exc)}


def get_queue_counts_safe() -> dict[str, bool | str | None | dict[str, dict[str, int | None]]]:
    """Return deterministic queue counts for doctor checks without raising."""
    result: dict[str, bool | str | None | dict[str, dict[str, int | None]]] = {
        "ok": False,
        "queues": _queue_counts_template(None),
        "error_type": None,
        "error": None,
    }
    try:
        redis = get_redis()
        redis.ping()
        counts: dict[str, dict[str, int | None]] = {}
        for name in AGM_QUEUE_NAMES:
            q = Queue(name, connection=redis)
            failed = FailedJobRegistry(queue=q)
            started = StartedJobRegistry(queue=q)
            counts[name] = {
                "queued": len(q),
                "running": _count_live_started_jobs(q, started),
                "failed": len(failed),
            }
        result["ok"] = True
        result["queues"] = counts
        return result
    except Exception as exc:
        result.update(_exception_context(exc))
        return result


def get_active_external_jobs() -> list[dict]:
    """Return metadata for running + queued external jobs. Best-effort."""
    try:
        redis = get_redis()
        q = Queue(QUEUE_EXTERNAL, connection=redis)
        started = StartedJobRegistry(queue=q)
        jobs: list[dict] = []
        for jid in list(q.job_ids) + started.get_job_ids():
            try:
                job = Job.fetch(jid, connection=redis)
                prompt_raw = job.args[0] if job.args else ""
                caller = job.args[3] if job.args and len(job.args) > 3 else "unknown"
                jobs.append(
                    {
                        "id": jid,
                        "status": "running" if jid in started.get_job_ids() else "queued",
                        "caller": caller,
                        "prompt": str(prompt_raw)[:120],
                        "started_at": (job.started_at.isoformat() if job.started_at else None),
                        "enqueued_at": (job.enqueued_at.isoformat() if job.enqueued_at else None),
                    }
                )
            except Exception:
                continue
        return jobs
    except (RedisError, Exception):
        return []


def get_queue_counts() -> dict[str, dict[str, int]]:
    """Return job counts per queue for status display."""
    redis = get_redis()
    result = {}
    for name in AGM_QUEUE_NAMES:
        q = Queue(name, connection=redis)
        failed = FailedJobRegistry(queue=q)
        started = StartedJobRegistry(queue=q)
        result[name] = {
            "queued": len(q),
            "running": _count_live_started_jobs(q, started),
            "failed": len(failed),
        }
    return result


def flush_failed_jobs(queue_name: str | None = None) -> dict[str, int]:
    """Clear failed job registries. Returns {queue_name: count_flushed}.

    If queue_name is None, flushes all agm queues.
    """
    redis = get_redis()
    targets = [queue_name] if queue_name else list(AGM_QUEUE_NAMES)
    result: dict[str, int] = {}
    for name in targets:
        q = Queue(name, connection=redis)
        registry = FailedJobRegistry(queue=q)
        job_ids = registry.get_job_ids()
        for job_id in job_ids:
            try:
                registry.remove(job_id, delete_job=True)
            except Exception:
                # Job data already gone from Redis — remove the registry entry directly
                registry.remove(job_id, delete_job=False)
        result[name] = len(job_ids)
    return result


def clean_finished_jobs(queue_name: str | None = None) -> dict[str, int]:
    """Clear finished job registries and delete job data from Redis.

    Returns {queue_name: count_cleaned}.
    """
    import contextlib

    redis = get_redis()
    targets = [queue_name] if queue_name else list(AGM_QUEUE_NAMES)
    result: dict[str, int] = {}
    for name in targets:
        q = Queue(name, connection=redis)
        registry = FinishedJobRegistry(queue=q)
        job_ids = registry.get_job_ids()
        for jid in job_ids:
            with contextlib.suppress(Exception):
                registry.remove(jid, delete_job=True)
        result[name] = len(job_ids)
    return result


def clean_log_files(older_than_days: int = 0) -> dict[str, int]:
    """Delete worker log files.

    Args:
        older_than_days: Only delete logs older than this many days.
            0 means delete all.

    Returns {"deleted": count, "kept": count}.
    """
    import time

    if not LOG_DIR.is_dir():
        return {"deleted": 0, "kept": 0}

    cutoff = time.time() - (older_than_days * 86400) if older_than_days > 0 else None
    deleted = 0
    kept = 0
    for log_file in LOG_DIR.iterdir():
        if not log_file.is_file() or log_file.suffix != ".log":
            continue
        if cutoff and log_file.stat().st_mtime >= cutoff:
            kept += 1
            continue
        log_file.unlink(missing_ok=True)
        deleted += 1
    return {"deleted": deleted, "kept": kept}


def remove_jobs_for_entities(plan_ids: list[str], task_ids: list[str]) -> int:
    """Remove rq jobs (any state) that reference the given plan/task IDs.

    Used during project removal to clean up orphaned Redis jobs.
    Returns the number of jobs removed.
    """
    # Build the set of job IDs to remove
    target_ids: set[str] = set()
    for pid in plan_ids:
        target_ids.add(f"enrich-{pid}")
        target_ids.add(f"plan-{pid}")
        target_ids.add(f"tasks-{pid}")
    for tid in task_ids:
        target_ids.add(f"exec-{tid}")
        target_ids.add(f"review-{tid}")
        target_ids.add(f"merge-{tid}")

    if not target_ids:
        return 0

    redis = get_redis()
    removed = 0
    for name in AGM_QUEUE_NAMES:
        q = Queue(name, connection=redis)
        for registry_cls in (FailedJobRegistry, StartedJobRegistry, FinishedJobRegistry):
            registry = registry_cls(queue=q)
            for job_id in registry.get_job_ids():
                if job_id in target_ids:
                    try:
                        registry.remove(job_id, delete_job=True)
                    except Exception:
                        registry.remove(job_id, delete_job=False)
                    removed += 1
        # Also check queued jobs
        for job in q.jobs:
            if job.id in target_ids:
                job.cancel()
                job.delete()
                removed += 1
    return removed


def _resolve_job_caller(job_id: str) -> str | None:
    """Look up the caller for a job by parsing its ID convention.

    Job IDs follow the pattern ``{prefix}-{entity_id}`` where prefix is
    one of exec/review/merge (task) or enrich/explore/plan/tasks (plan).
    """
    from agm.db import connect, get_plan_request, get_task

    parts = job_id.split("-", 1)
    if len(parts) != 2:
        return None
    prefix, entity_id = parts
    try:
        with connect() as conn:
            if prefix in ("exec", "review", "merge"):
                task = get_task(conn, entity_id)
                return task.get("caller") if task else None
            elif prefix in ("enrich", "explore", "plan", "tasks"):
                plan = get_plan_request(conn, entity_id)
                return plan.get("caller") if plan else None
    except Exception:
        return None
    return None


def _resolve_failed_job_description(job_id: str, fallback_description: str | None) -> str | None:
    """Normalize failed-job description from job ID prefix."""
    parts = job_id.split("-", 1)
    if len(parts) != 2:
        return fallback_description
    prefix, entity_id = parts
    if prefix == "enrich":
        return f"Enrich plan {entity_id}"
    if prefix == "ext":
        return f"External {entity_id}"
    return fallback_description


def get_failed_jobs(queue_name: str | None = None) -> list[dict]:
    """List failed jobs with their exception info.

    If *queue_name* is None, iterates all agm queues.
    """
    redis = get_redis()
    targets = [queue_name] if queue_name else list(AGM_QUEUE_NAMES)
    jobs = []
    for name in targets:
        q = Queue(name, connection=redis)
        registry = FailedJobRegistry(queue=q)
        for job_id in registry.get_job_ids():
            job = Job.fetch(job_id, connection=redis)
            jobs.append(
                {
                    "id": job.id,
                    "description": _resolve_failed_job_description(job.id, job.description),
                    "caller": _resolve_job_caller(job.id),
                    "exc_info": job.exc_info,
                    "enqueued_at": (str(job.enqueued_at) if job.enqueued_at else None),
                    "ended_at": (str(job.ended_at) if job.ended_at else None),
                }
            )
    return jobs


def _job_entity_from_id(job_id: str) -> tuple[str | None, str | None]:
    """Map job-id prefix to entity type/id when applicable."""
    if "-" not in job_id:
        return None, None
    prefix, entity_id = job_id.split("-", 1)
    if prefix in {"enrich", "explore", "plan", "tasks"}:
        return "plan", entity_id
    if prefix in {"exec", "review", "merge"}:
        return "task", entity_id
    if prefix == "setup":
        return "project", entity_id
    if prefix == "ext":
        return "external", entity_id
    return None, None


def _worker_alive(redis: Redis, worker_name: str | None) -> bool | None:
    """Best-effort worker heartbeat check from Redis worker key."""
    if not worker_name:
        return None
    try:
        return bool(redis.exists(f"rq:worker:{worker_name}"))
    except Exception:
        return None


def _age_seconds_from(started_at: datetime | None) -> float | None:
    """Compute age in seconds from a timestamp to now (UTC)."""
    if started_at is None:
        return None
    try:
        dt = started_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max((datetime.now(UTC) - dt).total_seconds(), 0.0)
    except Exception:
        return None


def _resolve_job_entity_context(entity_type: str | None, entity_id: str | None) -> dict:
    """Resolve project/plan/task/session linkage for queue inspect rows."""
    if not entity_type or not entity_id:
        return {}

    from agm.db import connect, get_plan_request, get_project, get_task

    try:
        with connect() as conn:
            if entity_type == "plan":
                plan = get_plan_request(conn, entity_id)
                if not plan:
                    return {"plan_id": entity_id}
                return {
                    "plan_id": plan["id"],
                    "project_id": plan.get("project_id"),
                    "session_id": plan.get("session_id"),
                    "plan_status": plan.get("status"),
                    "prompt_status": plan.get("prompt_status"),
                    "task_creation_status": plan.get("task_creation_status"),
                    "caller": plan.get("caller"),
                }
            if entity_type == "task":
                task = get_task(conn, entity_id)
                if not task:
                    return {"task_id": entity_id}
                plan = get_plan_request(conn, task["plan_id"])
                return {
                    "task_id": task["id"],
                    "task_status": task.get("status"),
                    "plan_id": task.get("plan_id"),
                    "project_id": plan.get("project_id") if plan else None,
                    "session_id": plan.get("session_id") if plan else None,
                    "caller": task.get("caller"),
                }
            if entity_type == "project":
                project = get_project(conn, entity_id)
                if not project:
                    return {"project_id": entity_id}
                return {"project_id": project["id"], "project_name": project.get("name")}
    except Exception:
        return {}
    return {}


def inspect_queue_jobs(queue_name: str | None = None, *, limit: int | None = None) -> list[dict]:
    """Return live queue-inspection rows with entity linkage + worker heartbeat."""
    redis = get_redis()
    targets = [queue_name] if queue_name else list(AGM_QUEUE_NAMES)
    rows: list[dict] = []

    def _normalized_job_status(raw_status: object) -> str:
        """Normalize rq status values for stale-started checks.

        rq may return either plain strings (``started``) or enum-style
        repr strings (``JobStatus.STARTED``). Normalize both to lowercase
        terminal tokens.
        """
        status_text = str(raw_status).strip()
        if status_text.startswith("JobStatus."):
            status_text = status_text.split(".", 1)[1]
        return status_text.lower()

    for name in targets:
        q = Queue(name, connection=redis)
        started = StartedJobRegistry(queue=q)
        queued_ids = list(q.job_ids)
        started_ids = started.get_job_ids()
        seen: set[str] = set()

        def _collect(
            job_id: str,
            in_started_registry: bool,
            *,
            _seen: set[str] = seen,
            _started: StartedJobRegistry = started,
            _queue_name: str = name,
        ) -> None:
            if job_id in _seen:
                return
            _seen.add(job_id)
            try:
                job = Job.fetch(job_id, connection=redis)
            except Exception:
                if in_started_registry:
                    with suppress(Exception):
                        _started.remove(job_id, delete_job=False)
                rows.append(
                    {
                        "job_id": job_id,
                        "queue": _queue_name,
                        "status": "missing",
                        "in_started_registry": in_started_registry,
                        "stale_started": in_started_registry,
                    }
                )
                return

            rq_status = str(job.get_status(refresh=True))
            normalized_status = _normalized_job_status(rq_status)
            worker_name = getattr(job, "worker_name", None)
            worker_alive = _worker_alive(redis, worker_name)
            stale_started = in_started_registry and normalized_status != "started"
            if in_started_registry and (stale_started or worker_alive is False):
                with suppress(Exception):
                    _started.remove(job_id, delete_job=False)

            entity_type, entity_id = _job_entity_from_id(job_id)
            linkage = _resolve_job_entity_context(entity_type, entity_id)
            row = {
                "job_id": job_id,
                "queue": _queue_name,
                "status": rq_status,
                "description": job.description,
                "in_started_registry": in_started_registry,
                "stale_started": bool(
                    stale_started or (in_started_registry and worker_alive is False)
                ),
                "worker_name": worker_name,
                "worker_alive": worker_alive,
                "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "age_seconds": _age_seconds_from(job.started_at or job.enqueued_at),
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
            row.update(linkage)
            rows.append(row)

        for jid in started_ids:
            _collect(jid, True)
        for jid in queued_ids:
            _collect(jid, False)

    rows.sort(key=lambda row: (row.get("queue", ""), str(row.get("job_id", ""))))
    if limit is not None and limit >= 0:
        return rows[:limit]
    return rows
