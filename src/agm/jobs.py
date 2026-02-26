"""Job functions executed by rq workers.

This module is a re-export facade. All implementation lives in the
jobs_* submodules. Imports here ensure that existing callers
(queue.py string references, cli.py imports) continue to work.
"""

from __future__ import annotations

# Re-export agents_config function used via agm.jobs in tests
from agm.agents_config import get_effective_role_config  # noqa: F401

# Re-export db functions that were imported and used directly via agm.jobs
from agm.db import (  # noqa: F401
    claim_task,
    create_tasks_batch,
    resolve_blockers_for_terminal_task,
    resolve_stale_blockers,
)

# -- jobs_common: shared infrastructure --
from agm.jobs_common import (  # noqa: F401
    _COMMIT_NUDGE,
    _PROJECT_INSTRUCTIONS_SECTION_DELIMITER,
    _TASK_PRIORITY_RANK,
    MAX_COMMIT_NUDGES,
    MAX_DIFF_CHARS,
    MAX_REJECTIONS,
    PlanDBHandler,
    TaskDBHandler,
    TurnEventContext,
    _codex_client,
    _codex_turn,
    _compact_codex_thread,
    _effective_task_priority,
    _emit,
    _extract_item_summary,
    _extract_plan_steps,
    _extract_plan_text,
    _extract_reasoning_summaries,
    _fallback_thread_config_for_resolved_model,
    _get_latest_review,
    _get_plan_backend,
    _get_project_dir_for_task,
    _get_project_id_for_task,
    _get_rejection_count,
    _has_uncommitted_changes,
    _load_project_model_config,
    _normalize_output_task_priority,
    _parse_task_files,
    _resolve_effective_base_branch,
    _resolve_project_model_config,
    _resolve_project_name,
)

# -- jobs_coordinator: inter-agent coordination --
from agm.jobs_coordinator import run_plan_coordinator  # noqa: F401

# -- jobs_enrichment: enrichment pipeline --
from agm.jobs_enrichment import (  # noqa: F401
    _build_parent_task_outcomes_summary,
    _enrichment_has_substance,
    _process_enrichment_output,
    _resolve_enrichment,
    _run_enrichment_codex,
    _run_enrichment_codex_async,
    _run_enrichment_continuation_codex,
    _run_enrichment_continuation_codex_async,
    _run_enrichment_resume_codex,
    _run_enrichment_resume_codex_async,
    on_enrichment_failure,
    run_enrichment,
)

# -- jobs_execution: task execution --
from agm.jobs_execution import (  # noqa: F401
    _build_merge_conflict_prompt_section,
    _build_quality_gate_fail_prompt_section,
    _build_quality_gate_prompt,
    _get_channel_context,
    _get_failed_sibling_context,
    _get_predecessor_context,
    _get_quality_gate_fail_context,
    _run_task_execution_codex,
    _run_task_execution_codex_async,
    on_task_execution_failure,
    run_task_execution,
)

# -- jobs_explorer: codebase exploration --
from agm.jobs_explorer import (  # noqa: F401
    _format_exploration_for_channel,
    _process_exploration_output,
    _run_explorer_codex,
    _run_explorer_codex_async,
    on_explorer_failure,
    run_explorer,
)

# -- jobs_external: stateless external mode --
from agm.jobs_external import (  # noqa: F401
    _run_external_codex,
    _store_result,
    run_external,
)

# -- jobs_merge: merge + auto-triggers --
from agm.jobs_merge import (  # noqa: F401
    _capture_branch_diff,
    _get_merge_conflict_context,
    _rollback_claim,
    _trigger_task_execution,
    _trigger_task_merge,
    _trigger_task_review,
    on_task_merge_failure,
    run_task_merge,
)

# -- jobs_plan: plan request (planning only) --
from agm.jobs_plan import (  # noqa: F401
    _run_plan_request_codex,
    _run_plan_request_codex_async,
    on_plan_request_failure,
    run_plan_request,
)

# -- jobs_quality_gate: quality gate functions --
from agm.jobs_quality_gate import (  # noqa: F401
    QualityCheckResult,
    QualityGateResult,
    _default_quality_gate,
    _generate_quality_gate_codex,
    _load_quality_gate,
    _parse_quality_gate_output,
    _run_quality_checks,
    _serialize_quality_gate_result,
    generate_quality_gate,
)

# -- jobs_review: task review --
from agm.jobs_review import (  # noqa: F401
    _build_review_prompt,
    _check_pre_review_gates,
    _gather_review_git_context,
    _handle_review_verdict,
    _prepare_review,
    _run_task_review_codex,
    _run_task_review_codex_async,
    on_task_review_failure,
    run_task_review,
)

# -- jobs_setup: project setup agent --
from agm.jobs_setup import (  # noqa: F401
    _apply_setup_result,
    _ensure_agents_toml,
    _parse_setup_output,
    _run_project_setup_codex,
    on_setup_failure,
    run_project_setup,
    run_project_setup_worker,
)

# -- jobs_task_creation: task creation + refresh --
from agm.jobs_task_creation import (  # noqa: F401
    _auto_trigger_execution_for_ready_tasks,
    _extract_task_files,
    _insert_tasks_from_output,
    _normalize_bucket_value,
    _run_task_creation_codex,
    _run_task_creation_codex_async,
    _run_task_refresh_codex,
    _run_task_refresh_codex_async,
    _sort_task_ids_for_auto_trigger,
    _trigger_task_creation,
    _verify_bucket_assignments,
    _warn_cross_plan_file_overlaps,
    on_task_creation_failure,
    run_task_creation,
    run_task_refresh,
)

# -- tracing: rich execution tracing --
from agm.tracing import (  # noqa: F401
    TraceContext,
    extract_rich_trace,
)
