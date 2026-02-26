"""External mode job — stateless LLM calls with no agm DB side effects.

run_external executes an LLM query, stores the result in Redis (TTL),
and emits ``external:status`` events. No plan/task/project rows are
created.  Intended for integration clients (SDK, aegis) that manage
their own persistence.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging

from agm.queue import publish_event

log = logging.getLogger(__name__)


def run_external(
    prompt: str,
    output_schema: str | None = None,
    cwd: str | None = None,
    caller: str = "cli",
    backend: str = "codex",
    developer_instructions: str | None = None,
    effort: str | None = None,
    timeout: int | None = None,
) -> str:
    """Execute a stateless LLM call (no agm DB rows).

    Called by rq worker on the ``agm:external`` queue.

    Result lifecycle:
        1. Emit ``external:status`` = ``running``
        2. Run LLM
        3. Store result in Redis key ``agm:external:{job_id}:result`` (1h TTL)
        4. Emit ``external:status`` = ``completed`` (or ``failed``)
        5. Return result string (rq stores this too)
    """
    import rq

    job = rq.get_current_job()
    job_id = job.id if job else "unknown"

    _emit_external("running", job_id, caller, extra={"prompt": prompt[:120]})
    log.info("External job %s started (backend=%s, caller=%s)", job_id, backend, caller)

    try:
        result, tokens = asyncio.run(
            _run_external_codex(
                prompt,
                output_schema,
                cwd,
                backend,
                developer_instructions,
                effort,
                timeout,
            )
        )

        # Store in Redis with 1h TTL
        _store_result(job_id, result)
        _emit_external("completed", job_id, caller, extra={"tokens": tokens})
        log.info("External job %s completed (%d chars)", job_id, len(result))
        return result

    except Exception as exc:
        log.exception("External job %s failed", job_id)
        _emit_external("failed", job_id, caller, extra={"error": str(exc)})
        raise


async def _run_external_codex(
    prompt: str,
    output_schema: str | None,
    cwd: str | None,
    backend: str,
    developer_instructions: str | None = None,
    effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    """Run external query via Codex backend. Returns (result_text, token_usage)."""
    from agm.backends import get_runtime_thread_config, get_runtime_turn_config
    from agm.jobs_common import TURN_TIMEOUT, _codex_client, _codex_turn

    thread_config = dict(get_runtime_thread_config(backend, "query", None))
    turn_config = dict(get_runtime_turn_config(backend, "query", None))

    # Caller-provided overrides
    if developer_instructions:
        thread_config["developerInstructions"] = developer_instructions
    if effort:
        turn_config["effort"] = effort

    if output_schema:
        prompt += (
            "\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{output_schema}\n```"
        )
        turn_config["outputSchema"] = _json.loads(output_schema)

    start_params = dict(thread_config)
    if cwd:
        start_params["cwd"] = cwd

    async with _codex_client() as client:
        _, text, tokens = await _codex_turn(
            client,
            prompt=prompt,
            turn_config=turn_config,
            start_thread_params=start_params,
            turn_timeout=timeout or TURN_TIMEOUT,
        )
    return text or "", tokens


def _store_result(job_id: str, result: str) -> None:
    """Store result in Redis with 1-hour TTL."""
    try:
        from agm.queue import get_redis

        r = get_redis()
        r.set(f"agm:external:{job_id}:result", result, ex=3600)
    except Exception:
        log.warning("Failed to store external result in Redis for %s", job_id)


def _emit_external(status: str, job_id: str, caller: str, *, extra: dict | None = None) -> None:
    """Emit an external:status event."""
    publish_event(
        event_type="external:status",
        entity_id=job_id,
        status=status,
        project="",
        source=caller,
        extra=extra,
    )
