"""Backend configuration for job types.

Each job type (plan_request, task) has a fixed set of thread/start
and turn/start parameters. Workers always run with these exact
settings — no guessing, no per-project overrides.

Two-tier model strategy:
  - Think tier (planner, task agent, reviewer, enrichment): deep reasoning, high intelligence
  - Work tier (executor): fast execution, clear instructions

Codex model override via env vars:
  AGM_MODEL_THINK          — think-tier model (default: gpt-5.3-codex)
  AGM_MODEL_WORK           — model for executor (default: gpt-5.3-codex-spark)
  AGM_MODEL_WORK_FALLBACK  — fallback when work model unavailable (default: AGM_MODEL_THINK)
  AGM_MODEL_THINK_EFFORT   — effort level for think tier (default: per-model catalog)
  AGM_MODEL_WORK_EFFORT    — effort level for work tier (default: per-model catalog)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.db import PlanQuestionRow

# Backends that have a working job implementation.
# Used by CLI to reject requests before DB/queue submission.
IMPLEMENTED_BACKENDS = {"codex"}

SUPPORTED_BACKENDS = {"codex"}
VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_THINK_JOB_TYPES = {
    "prompt_enrichment",
    "codebase_exploration",
    "plan_request",
    "task_review",
    "query",
}


# -- Model catalog --------------------------------------------------------------
#
# The keys are model identifiers and the values describe runtime defaults and
# metadata for model selection (recommendeds and effort defaults). Unknown model
# IDs are allowed and carried through without validation errors so users can try
# pre-release providers.

MODEL_CATALOG: dict[str, dict[str, object]] = {
    # -- Codex backend --
    "gpt-5.3-codex": {
        "backend": "codex",
        "tier": "think",
        "speed": "standard",
        "description": "Full Codex model. Best quality for planning and complex reasoning.",
        "effort_defaults": {"think": "high", "work": "high"},
        "recommendation": {
            "default": True,
            "reason": "Recommended for think tier. Best plan quality.",
        },
        "notes": None,
    },
    "gpt-5.3-codex-spark": {
        "backend": "codex",
        "tier": "work",
        "speed": "fast",
        "description": "Fast Codex variant (~1000 tok/s on Cerebras). Good for execution.",
        "effort_defaults": {"think": "high", "work": "high"},
        "recommendation": {
            "default": True,
            "reason": "Recommended for work tier. Fast execution and review.",
        },
        "notes": "May struggle with lint fix iteration across rejection cycles.",
    },
}


# -- Live model cache ----------------------------------------------------------
# Populated once per worker process from model/list API. Falls back to
# MODEL_CATALOG when unavailable.

_live_models: list[dict] | None = None


def set_live_models(models: list[dict]) -> None:
    """Cache the model/list API response (called once per worker lifecycle)."""
    global _live_models
    _live_models = models


def get_live_models() -> list[dict] | None:
    """Return cached model/list response, or None if not yet fetched."""
    return _live_models


def get_known_model_ids() -> set[str]:
    """Return all known model IDs from live cache + static catalog."""
    ids = set(MODEL_CATALOG.keys())
    if _live_models is not None:
        ids.update(m.get("id", "") for m in _live_models if m.get("id"))
    return ids


def is_model_available(model_id: str) -> bool | None:
    """Check if a model is available on the server.

    Returns True/False when live data is available, None when unknown
    (live cache not populated).
    """
    if _live_models is None:
        return None
    return model_id in {m.get("id", "") for m in _live_models}


def _get_default_model_for_backend_tier(backend: str, tier: str) -> str:
    for model_id, spec in MODEL_CATALOG.items():
        if spec.get("backend") != backend:
            continue
        if spec.get("tier") not in (tier, "both"):
            continue
        recommendation = spec.get("recommendation")
        if isinstance(recommendation, dict) and recommendation.get("default") is True:
            return model_id
    # Fall back to the first matching model in deterministic catalog order if
    # recommendation metadata is missing or stale.
    for model_id, spec in MODEL_CATALOG.items():
        if spec.get("backend") != backend:
            continue
        if spec.get("tier") in (tier, "both"):
            return model_id
    raise ValueError(f"No catalog model for backend='{backend}' tier='{tier}'")


def _get_default_effort_for_backend_tier(backend: str, tier: str, model: str | None = None) -> str:
    if model:
        spec = MODEL_CATALOG.get(model)
        if isinstance(spec, dict) and spec.get("backend") == backend:
            ed = spec.get("effort_defaults")
            if isinstance(ed, dict) and ed.get(tier) in VALID_EFFORTS:
                return str(ed[tier])
    default_model = _get_default_model_for_backend_tier(backend, tier)
    spec = MODEL_CATALOG[default_model]
    ed = spec.get("effort_defaults")
    if isinstance(ed, dict):
        effort = ed.get(tier)
        if effort in VALID_EFFORTS:
            return str(effort)
    raise ValueError(
        f"Catalog effort missing for backend='{backend}' tier='{tier}' model='{default_model}'"
    )


def _normalize_effort(effort: object) -> str | None:
    if not isinstance(effort, str):
        return None
    normalized = effort.strip().lower()
    if normalized in VALID_EFFORTS:
        return normalized
    return None


def _model_env_prefix_for_backend(backend: str) -> str:
    return "AGM_MODEL"


def _env_model_lookup_for_backend(backend: str) -> dict[str, str | None]:
    prefix = _model_env_prefix_for_backend(backend)
    return {
        "think_model": os.environ.get(f"{prefix}_THINK"),
        "work_model": os.environ.get(f"{prefix}_WORK"),
    }


def _env_effort_lookup_for_backend(backend: str) -> dict[str, str | None]:
    prefix = _model_env_prefix_for_backend(backend)
    return {
        "think_effort": os.environ.get(f"{prefix}_THINK_EFFORT"),
        "work_effort": os.environ.get(f"{prefix}_WORK_EFFORT"),
    }


def _resolve_model_for_tier(
    backend: str,
    tier: str,
    project_model_config: dict,
    env_model_lookup: dict[str, str | None],
) -> str:
    model_key = f"{tier}_model"
    resolved_model = (
        project_model_config.get(model_key)
        or env_model_lookup[model_key]
        or _get_default_model_for_backend_tier(backend, tier)
    )
    return str(resolved_model)


def _resolve_effort_for_tier(
    backend: str,
    tier: str,
    model: str,
    project_model_config: dict,
    env_effort_lookup: dict[str, str | None],
) -> str:
    effort_key = f"{tier}_effort"
    project_effort = _normalize_effort(project_model_config.get(effort_key))
    if project_effort is not None:
        return project_effort
    default_effort = _get_default_effort_for_backend_tier(backend, tier, model)
    return _normalize_effort(env_effort_lookup[effort_key]) or default_effort


def resolve_model_config(
    backend: str,
    project_model_config: dict | None,
) -> dict[str, str]:
    """Resolve think/work model and effort values for a backend.

    Precedence is:
    1) `project_model_config`
    2) environment variables
    3) catalog defaults
    """
    backend = backend.strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Expected one of {sorted(SUPPORTED_BACKENDS)}"
        )

    config = project_model_config if isinstance(project_model_config, dict) else {}
    env_model_lookup = _env_model_lookup_for_backend(backend)
    env_effort_lookup = _env_effort_lookup_for_backend(backend)

    resolved: dict[str, str] = {}
    for tier in ("think", "work"):
        model = _resolve_model_for_tier(backend, tier, config, env_model_lookup)
        resolved[f"{tier}_model"] = model
        resolved[f"{tier}_effort"] = _resolve_effort_for_tier(
            backend,
            tier,
            model,
            config,
            env_effort_lookup,
        )
    return resolved


_VALID_JOB_TYPES = {
    "plan_request",
    "task_creation",
    "task_execution",
    "task_review",
    "prompt_enrichment",
    "codebase_exploration",
    "query",
}


def get_runtime_thread_config(
    backend: str,
    job_type: str,
    project_model_config: dict | None = None,
) -> dict:
    """Return a thread config with resolved model for a job type.

    Model is resolved from: project config > env vars > catalog defaults.
    Effort resolution is handled by get_runtime_turn_config() since effort
    moved to turn/start in the 0.102.0 protocol.
    """
    if job_type not in _VALID_JOB_TYPES:
        raise ValueError(
            f"Invalid job_type '{job_type}'. Must be one of: {sorted(_VALID_JOB_TYPES)}"
        )

    model_config = resolve_model_config(backend, project_model_config)
    import copy

    model = (
        model_config["think_model"] if job_type in _THINK_JOB_TYPES else model_config["work_model"]
    )

    base = copy.deepcopy(THREAD_CONFIGS[job_type])
    base["model"] = model
    return base


def get_runtime_turn_config(
    backend: str,
    job_type: str,
    project_model_config: dict | None = None,
) -> dict:
    """Return a turn config with resolved effort for a job type.

    Effort is resolved from: project config > env vars > catalog defaults,
    matching the same precedence as get_runtime_thread_config().
    """
    if job_type not in _VALID_JOB_TYPES:
        raise ValueError(
            f"Invalid job_type '{job_type}'. Must be one of: {sorted(_VALID_JOB_TYPES)}"
        )

    model_config = resolve_model_config(backend, project_model_config)
    effort = (
        model_config["think_effort"]
        if job_type in _THINK_JOB_TYPES
        else model_config["work_effort"]
    )

    base = dict(TURN_CONFIGS.get(job_type, {}))
    base["effort"] = effort
    return base


# -- Model defaults (overridable via env vars) --------------------------------

# Codex models
THINK_MODEL = os.environ.get(
    "AGM_MODEL_THINK", _get_default_model_for_backend_tier("codex", "think")
)
WORK_MODEL = os.environ.get("AGM_MODEL_WORK", _get_default_model_for_backend_tier("codex", "work"))
# Fallback when WORK_MODEL fails (e.g. spark infra issues). Defaults to THINK_MODEL.
WORK_MODEL_FALLBACK = os.environ.get("AGM_MODEL_WORK_FALLBACK", THINK_MODEL)


# -- Plan output schema ---------------------------------------------------
# Structured output for plan requests. Constrains the final assistant
# message so clients and task agents can consume it directly as JSON.

PLAN_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Short descriptive title for the plan",
        },
        "summary": {
            "type": "string",
            "description": "Brief overview of what the plan accomplishes",
        },
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short task title in imperative form",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "What needs to be done: implementation details, "
                            "acceptance criteria, edge cases"
                        ),
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to create or modify",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "0-based indices of tasks that must complete first",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "Task priority: high for critical path,"
                            " medium for normal, low for deferrable."
                        ),
                    },
                    "bucket": {
                        "type": ["string", "null"],
                        "description": (
                            "Bucket label for file-overlap serialization. "
                            "Tasks sharing a bucket run serially; different buckets"
                            "run in parallel. Null if no file overlap with other tasks."
                        ),
                    },
                    "external_blockers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "factor": {
                                    "type": "string",
                                    "description": (
                                        "What external thing is needed "
                                        "(e.g. API key, design review)"
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Why this blocks the task",
                                },
                            },
                            "required": ["factor", "reason"],
                            "additionalProperties": False,
                        },
                        "description": (
                            "External blockers not tied to other tasks — "
                            "things outside the codebase that a human must resolve"
                        ),
                    },
                },
                "required": [
                    "title",
                    "description",
                    "files",
                    "depends_on",
                    "priority",
                    "bucket",
                    "external_blockers",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "summary", "tasks"],
    "additionalProperties": False,
}

# -- Enrichment output schema -----------------------------------------------
# Structured output for the prompt enrichment agent. It refines the user's
# raw prompt into a detailed, repo-aware specification and optionally asks
# clarifying questions.

ENRICHMENT_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "enriched_prompt": {
            "type": "string",
            "description": (
                "A well-structured prompt crafted from the user's request. "
                "Contains acceptance criteria, constraints, non-goals, "
                "and edge cases. Written as instructions for a planning agent."
            ),
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A clarifying question for the user",
                    },
                    "header": {
                        "type": ["string", "null"],
                        "description": (
                            "Short category label (max 12 chars) displayed as a tag "
                            "above the question. Examples: 'Auth method', 'Library', "
                            "'Approach'. Null if no category applies."
                        ),
                    },
                    "options": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": (
                                        "Concise display text for this option (1-5 words)"
                                    ),
                                },
                                "description": {
                                    "type": "string",
                                    "description": (
                                        "Explanation of what this option means or "
                                        "what trade-offs it involves"
                                    ),
                                },
                            },
                            "required": ["label", "description"],
                            "additionalProperties": False,
                        },
                        "description": (
                            "Structured answer options with labels and descriptions, "
                            "or null for free-form text input"
                        ),
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": (
                            "True if the user can select multiple options. "
                            "Only meaningful when options is not null."
                        ),
                    },
                },
                "required": ["question", "header", "options", "multi_select"],
                "additionalProperties": False,
            },
            "description": (
                "Clarifying questions for the user. Empty array if the prompt is "
                "already clear enough to proceed."
            ),
        },
    },
    "required": ["enriched_prompt", "questions"],
    "additionalProperties": False,
}

# -- Exploration output schema ---------------------------------------------
# Structured output for the codebase exploration agent. It produces a
# thorough report of the codebase area relevant to the task, which feeds
# into the planner and downstream agents via the session channel.

EXPLORATION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "High-level overview of the codebase area relevant to the task",
        },
        "architecture": {
            "type": "string",
            "description": (
                "Module boundaries, key abstractions, data flow patterns. "
                "Written as context for a planning agent."
            ),
        },
        "relevant_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "description": {"type": "string"},
                    "key_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Functions, classes, or constants relevant to the task"),
                    },
                },
                "required": ["path", "description", "key_symbols"],
                "additionalProperties": False,
            },
        },
        "patterns_to_follow": {
            "type": "array",
            "items": {"type": "string"},
            "description": ("Coding patterns, naming conventions, error handling styles to match"),
        },
        "reusable_helpers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbol": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["path", "symbol", "description"],
                "additionalProperties": False,
            },
            "description": (
                "Existing functions/classes that should be reused instead of duplicated"
            ),
        },
        "test_locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Test file paths relevant to the task",
        },
    },
    "required": [
        "summary",
        "architecture",
        "relevant_files",
        "patterns_to_follow",
        "reusable_helpers",
        "test_locations",
    ],
    "additionalProperties": False,
}

# -- Prompt suffixes -------------------------------------------------------
# Appended to user prompts automatically by jobs.py.
# Behavioral guardrails (no file writes, plan-only mode) come from the
# built-in plan.md template — developer_instructions is set to None so
# the template is not replaced. These suffixes contain domain-specific
# rules only.

ENRICHMENT_PROMPT_SUFFIX = """\

---
<instructions>
You are transforming the user's raw request into a well-crafted prompt for \
a planning agent. Apply prompt engineering best practices.

Step 1 — Analyze: What is the user actually asking for? What is the core \
intent behind the words?

Step 2 — Identify gaps: What critical information is missing? What would \
a planner have to guess?

Step 3 — Decide: If the request is genuinely ambiguous with multiple valid \
interpretations, ask clarifying questions. Otherwise, infer reasonable \
defaults and state them explicitly in the prompt.

Step 4 — Craft: Write a clear, structured prompt with acceptance criteria, \
constraints, non-goals, and edge cases.
</instructions>

<good_prompt_traits>
- Opens with a specific, unambiguous description of what to build
- Lists explicit acceptance criteria (what "done" looks like)
- States constraints and non-goals (what NOT to do)
- Covers edge cases and error handling expectations
- Defines expected user experience or interface behavior
- Uses precise language with no room for interpretation
</good_prompt_traits>

<bad_prompt_traits>
- Restates the user's vague request without adding clarity
- Contains meta-narration ("I will scan...", "I'm delegating...")
- Focuses on HOW to implement instead of WHAT to build
- Omits acceptance criteria or success conditions
- Is shorter or vaguer than the original request
</bad_prompt_traits>

<examples>
<example>
<user_request>Add a health check</user_request>
<enriched_prompt>Add a CLI command `health` that reports the operational \
status of all system dependencies.

The command must:
1. Check Redis connectivity — report connection status, memory usage, uptime
2. Check database — file exists, is readable, schema version is current
3. Check worker processes — count running, identify stale/zombie workers

Output: a table with columns [Component, Status, Details]. Status is one \
of OK / WARN / ERROR. Exit code 0 if all OK, 1 if any WARN, 2 if any ERROR.

Constraints:
- 5-second timeout per check; report ERROR with "timeout" on expiry
- No new dependencies — use existing connection helpers
- Must work when Redis is down (report ERROR, continue checking others)

Non-goals:
- No continuous monitoring or watch mode
- No alerting or notification integration</enriched_prompt>
</example>

<example>
<user_request>Make the tests faster</user_request>
<enriched_prompt>Reduce the wall-clock time of the default test suite \
by at least 30%.

Acceptance criteria:
- Baseline: measure current suite duration (record in commit message)
- Target: suite completes in 70% or less of baseline time
- Zero test regressions — all existing tests must still pass
- No tests removed or reclassified to hit the target

Approach guidance:
- Profile first: identify the slowest 20 tests by duration
- Common wins: replace real I/O with mocks, share expensive fixtures \
across tests via session/module scope, parallelize independent test files
- Preserve test isolation — no shared mutable state between tests

Constraints:
- Do not change the public API or behavior of the code under test
- Do not add new test dependencies without justification</enriched_prompt>
</example>
</examples>

<questions_guidelines>
Ask `questions` ONLY when:
- The request is genuinely ambiguous (multiple valid interpretations)
- Critical details are missing that would fundamentally change the approach
- There are meaningful trade-offs the user should decide

Do NOT ask about:
- Implementation details the coding agent can figure out
- Obvious defaults (error handling, test coverage, naming)
- Things with a single reasonable interpretation

Return an empty `questions` array when you can produce a solid prompt \
without further input. Prefer stating inferred defaults explicitly over \
asking unnecessary questions.

Each question must include:
- `question`: full question text
- `header`: short category tag (max 12 chars, e.g. "Scope", "UX") or null
- `options`: array of `{label, description}` objects, or null for free-form
- `multi_select`: true if multiple selections are valid, false otherwise
</questions_guidelines>"""

EXPLORATION_PROMPT_SUFFIX = """\

---
<instructions>
Explore the codebase to understand the architecture, conventions, and \
existing code relevant to the task above. Your findings will be passed \
to a planning agent that will design the implementation.

Focus on:
1. Architecture: module boundaries, data flow, key abstractions
2. Relevant files: which files will likely be modified or referenced
3. Patterns: naming conventions, error handling, import ordering
4. Reusable helpers: existing functions/utilities to use (don't duplicate)
5. Test locations: where tests for the affected code live
6. Constraints: anything that limits implementation choices

Be thorough but focused — explore what matters for THIS task, not the \
entire codebase.
</instructions>
"""

PLAN_PROMPT_SUFFIX = """\

---
Produce a structured implementation plan as JSON matching the output schema.

Rules:
- Read the codebase first — understand architecture, naming conventions, \
and existing patterns before planning.
- The `tasks` array MUST contain at least one task. Never return an empty \
tasks array — if you need more time to explore the codebase, do that first, \
but your final output must include concrete tasks.
- Prefer fewer, larger tasks over many tiny ones. Each task spawns a \
full agent session — overhead is real. A task that touches 3 related \
files in one module is better than 3 single-file tasks, but each task \
must still be completable by a coding agent in one session.
- Specify which files each task creates or modifies.
- Define dependencies between tasks using 0-based indices.
- Use `external_blockers` for things outside the codebase that a human \
must resolve (API keys, design reviews, infrastructure). Use an empty \
array when a task has no external blockers.
- Task descriptions are the SOLE context executors receive. Be specific: \
name the exact functions, modules, and patterns they should use or follow. \
A vague description produces wrong code.
- NEVER include diffs, patches, or code in task descriptions. Describe \
WHAT to do and WHERE, not the exact code to write. A separate executor \
agent with full write access will implement each task — diffs in \
descriptions become stale and mislead the executor.
- When multiple tasks touch the same module, specify exact column names, \
function signatures, or variable names in each task description so \
independent executors make consistent choices.
- Do NOT create "run tests", "verify", or "validate" tasks. Testing and \
validation are part of each implementation task, not separate work items. \
Every task's executor is expected to run the test suite before committing.
- If the request is large enough that parallel tasks would conflict heavily \
(many tasks touching the same files, complex interdependencies), flag this \
in the plan summary and consider whether it should be split into sequential \
plans instead of one plan with many buckets.
- Always assign `priority` on each task: use `high` for urgent or \
critical-path work, `medium` for normal implementation sequencing, and \
`low` for deferrable or nice-to-have work.
- **Task buckets**: If two or more tasks modify any of the same files, \
assign them the same `bucket` label (a short descriptive string like \
"db-layer" or "cli-commands"). Tasks in the same bucket run serially so \
each sees the previous task's merged code. Different buckets run in \
parallel. Use `null` for tasks that don't share files with any other task. \
Never assign a bucket to a task unless at least one other task shares that \
bucket. A single-task bucket will be stripped by the system.

Context7 MCP tools are available for up-to-date library documentation:
- resolve-library-id: find the Context7 library ID
- query-docs: fetch current docs, API references, and code examples
Use these when you need accurate API details, versions, or patterns."""

# -- Task output schema ----------------------------------------------------
# Structured output for task creation. Constrains the final assistant
# message so the task agent produces machine-parseable task definitions.

TASK_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ordinal": {
                        "type": "integer",
                        "description": "Unique ordinal for this task (0-based)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short task title in imperative form",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Agent-ready description: what to do, acceptance criteria, "
                            "edge cases, patterns to follow"
                        ),
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to create or modify",
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Ordinals of other NEW tasks in this output that must complete first"
                        ),
                    },
                    "blocked_by_existing": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "IDs of existing tasks from prior plans that must complete first"
                        ),
                    },
                    "external_blockers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "factor": {
                                    "type": "string",
                                    "description": (
                                        "What external thing is needed "
                                        "(e.g. API key, design review)"
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Why this blocks the task",
                                },
                            },
                            "required": ["factor", "reason"],
                            "additionalProperties": False,
                        },
                        "description": "External blockers not tied to other tasks",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "ready"],
                        "description": "ready if no blockers at all, blocked if any blockers",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "Task priority: high for critical path,"
                            " medium for normal, low for deferrable."
                        ),
                    },
                    "bucket": {
                        "type": ["string", "null"],
                        "description": (
                            "Bucket label for file-overlap serialization. "
                            "Tasks sharing a bucket run serially; different buckets"
                            "run in parallel. Null if no file overlap with other tasks."
                        ),
                    },
                },
                "required": [
                    "ordinal",
                    "title",
                    "description",
                    "files",
                    "blocked_by",
                    "blocked_by_existing",
                    "external_blockers",
                    "status",
                    "priority",
                    "bucket",
                ],
                "additionalProperties": False,
            },
        },
        "cancel_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of an existing task to cancel",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this task should be cancelled",
                    },
                },
                "required": ["task_id", "reason"],
                "additionalProperties": False,
            },
            "description": ("Existing tasks to cancel (superseded, duplicate, or stale)"),
        },
    },
    "required": ["tasks", "cancel_tasks"],
    "additionalProperties": False,
}

DEFAULT_QUALITY_GATE: dict = {"auto_fix": [], "checks": []}

# -- Quality gate presets ---------------------------------------------------
# Quick-start configs for common stacks. Each preset is a dict with
# "description" (human-readable) and "config" (same shape as quality gate).

QUALITY_GATE_PRESETS: dict[str, dict] = {
    "python": {
        "description": "Python: ruff format/check + pytest",
        "config": {
            "auto_fix": [
                {"name": "ruff format", "cmd": ["ruff", "format", "."]},
                {"name": "ruff check --fix", "cmd": ["ruff", "check", "--fix", "."]},
            ],
            "checks": [
                {"name": "ruff check", "cmd": ["ruff", "check", "."], "timeout": 60},
                {"name": "pytest", "cmd": ["pytest", "-q"], "timeout": 300},
            ],
        },
    },
    "typescript": {
        "description": "TypeScript: biome format/lint + vitest",
        "config": {
            "auto_fix": [
                {"name": "biome format", "cmd": ["npx", "biome", "format", "--write", "."]},
                {"name": "biome lint --fix", "cmd": ["npx", "biome", "lint", "--fix", "."]},
            ],
            "checks": [
                {"name": "biome check", "cmd": ["npx", "biome", "check", "."], "timeout": 60},
                {"name": "vitest", "cmd": ["npx", "vitest", "run"], "timeout": 300},
            ],
        },
    },
}

# -- Shared quality gate schema fragments ------------------------------------
# Reused by both QUALITY_GATE_GENERATE_SCHEMA and PROJECT_SETUP_SCHEMA.

_QUALITY_GATE_AUTO_FIX_ITEMS: dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short label for the auto-fix command",
        },
        "cmd": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command + args to run",
        },
    },
    "required": ["name", "cmd"],
    "additionalProperties": False,
}

_QUALITY_GATE_CHECK_ITEMS: dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short label for the check command",
        },
        "cmd": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command + args to run",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 120)",
        },
    },
    "required": ["name", "cmd", "timeout"],
    "additionalProperties": False,
}

# -- Quality gate generate schema/prompt ------------------------------------
# Structured output for LLM-based quality gate generation.

QUALITY_GATE_GENERATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "auto_fix": {
            "type": "array",
            "items": _QUALITY_GATE_AUTO_FIX_ITEMS,
            "description": (
                "Commands that auto-fix issues (formatters, lint --fix). Run before checks."
            ),
        },
        "checks": {
            "type": "array",
            "items": _QUALITY_GATE_CHECK_ITEMS,
            "description": "Strict checks that must pass (lint, tests). Failures block review.",
        },
    },
    "required": ["auto_fix", "checks"],
    "additionalProperties": False,
}

QUALITY_GATE_GENERATE_PROMPT = """\
Inspect this project and generate a quality gate configuration as JSON \
matching the output schema.

Context: this gate runs in an isolated git worktree after the executor \
writes code and before the reviewer evaluates the diff. Only include \
checks that validate code correctness in a worktree — no doc freshness, \
no deploy, no build artifacts.

Rules:
- Look for config files: pyproject.toml, package.json, Cargo.toml, \
go.mod, Makefile, biome.json, .eslintrc*, ruff.toml, etc.
- Only include tools that are actually configured or installed in \
this project. Do NOT guess — if there's no pyproject.toml, don't add \
ruff or pytest.
- Use the project's package manager (npm/pnpm/yarn/bun for JS, \
uv/pip for Python, cargo for Rust, etc.).
- `auto_fix` order matters: formatter first (e.g. `ruff format`), \
then lint autofix (e.g. `ruff check --fix`). Format before fix \
avoids the fixer producing unformatted output.
- `checks` commands: strict checks that must pass (e.g. `ruff check .`, \
`pytest -q`, `npx vitest run`). Failures reject the executor's work.
- Do NOT add checks that are redundant with auto_fix (e.g. no \
`ruff format --check` if `ruff format` already runs in auto_fix).
- Set reasonable timeouts: 60s for lint/typecheck, 300s for tests.
- Return empty arrays if no tooling is detected — never fabricate \
commands."""

# -- Project setup schema/prompt --------------------------------------------
# Structured output for LLM-based project setup (quality gate + post-merge).

PROJECT_SETUP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "quality_gate": {
            "type": "object",
            "properties": {
                "auto_fix": {
                    "type": "array",
                    "items": _QUALITY_GATE_AUTO_FIX_ITEMS,
                    "description": (
                        "Commands that auto-fix issues (formatters, lint --fix). Run before checks."
                    ),
                },
                "checks": {
                    "type": "array",
                    "items": _QUALITY_GATE_CHECK_ITEMS,
                    "description": (
                        "Strict checks that must pass (lint, tests). Failures block review."
                    ),
                },
            },
            "required": ["auto_fix", "checks"],
            "additionalProperties": False,
        },
        "post_merge_command": {
            "type": ["string", "null"],
            "description": (
                "Shell command to run after task branches merge to the base branch. "
                "Receives merge context via environment variable AGM_MERGE_SHA. "
                "Null if not applicable."
            ),
        },
        "stack": {
            "type": "object",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Detected programming languages",
                },
                "package_manager": {
                    "type": ["string", "null"],
                    "description": "Primary package manager (uv, npm, pnpm, cargo, etc.)",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Detected dev tools (ruff, pytest, biome, vitest, etc.)",
                },
            },
            "required": ["languages", "package_manager", "tools"],
            "additionalProperties": False,
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Missing or risky tooling warnings "
                "(no test runner, no linter, no type checker, etc.)"
            ),
        },
        "reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Tool or config item considered (e.g. ruff, pytest, radon)",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["configured", "skipped", "not_found"],
                        "description": "What was done with this item",
                    },
                    "detail": {
                        "type": "string",
                        "description": "Why this action was taken",
                    },
                },
                "required": ["item", "action", "detail"],
                "additionalProperties": False,
            },
            "description": (
                "Per-tool reasoning: what was configured, what was skipped, "
                "and what was not found. One entry per tool considered."
            ),
        },
    },
    "required": ["quality_gate", "post_merge_command", "stack", "warnings", "reasoning"],
    "additionalProperties": False,
}

PROJECT_SETUP_PROMPT = """\
Inspect this project and generate pipeline configuration as JSON matching \
the output schema.

Context: this configuration drives an automated agent pipeline. Agents \
write code in isolated git worktrees, then quality gate checks run \
automatically before review and merge.

What to inspect:
1. Config files: pyproject.toml, package.json, Cargo.toml, go.mod, \
Makefile, biome.json, .eslintrc*, ruff.toml, tsconfig.json, etc.
2. Build scripts: Makefile targets, npm scripts, cargo commands
3. CI config: .github/workflows, .gitlab-ci.yml (for command hints)

Tool categories to check (inspect each, report in reasoning):
- Formatters: ruff format, black, prettier, biome format, gofmt, rustfmt
- Linters: ruff check, eslint, biome lint, clippy, golangci-lint
- Type checkers: pyright, mypy, tsc (tsconfig.json), flow
- Test runners: pytest, vitest, jest, mocha, cargo test, go test
- Complexity: radon (Python cc -n D gate), eslint complexity rules
- Build/rebuild: make targets, npm scripts, cargo build

Quality gate rules:
- Only include tools that are actually configured or installed in \
this project. Do NOT guess — if there's no pyproject.toml, don't add \
ruff or pytest.
- Use the project's package manager (npm/pnpm/yarn/bun for JS, \
uv/pip for Python, cargo for Rust, etc.).
- `auto_fix` order matters: formatter first (e.g. `ruff format`), \
then lint autofix (e.g. `ruff check --fix`). Format before fix \
avoids the fixer producing unformatted output.
- `checks` commands: strict checks that must pass. Include ALL \
applicable categories: lint, typecheck, complexity, and tests. \
Examples: `ruff check .`, `pyright src/`, `radon cc src/ -a -n D`, \
`pytest -q`, `npx vitest run`, `tsc --noEmit`.
- Do NOT add checks that are redundant with auto_fix (e.g. no \
`ruff format --check` if `ruff format` already runs in auto_fix).
- Set reasonable timeouts: 60s for lint/typecheck, 120s for \
complexity, 300s for tests.
- Return empty arrays if no tooling is detected — never fabricate \
commands.

Post-merge command:
- Command that should run after a task branch merges to the base branch.
- Common examples: `make install-bin` (rebuild global binary), \
`npm run build` (rebuild), `make sync` (update deps).
- Receives the merge commit SHA via `AGM_MERGE_SHA`.
- Set null if the project has no post-merge needs.

Stack identification:
- languages: detected programming languages
- package_manager: primary package manager (uv, npm, pnpm, cargo, etc.)
- tools: ALL detected dev tools, including formatters, linters, type \
checkers, test runners, and complexity tools

Warnings:
- Note missing tooling: "No test runner detected", "No linter configured"
- Note risks: "No type checker — type errors won't be caught", \
"No complexity gate — functions can grow unbounded"
- Empty array if everything looks good.

Reasoning (one entry per tool considered):
- For each tool you inspect, record what you did and why.
- action "configured": tool added to quality gate or post-merge.
- action "skipped": tool exists but was intentionally not added (explain why).
- action "not_found": looked for but not present in this project.
- You MUST check every tool category listed above and include a \
reasoning entry for each. This makes the inspection auditable.
- Examples:
  {"item": "ruff", "action": "configured", "detail": "Found in \
pyproject.toml [tool.ruff]; added format + check to quality gate"}
  {"item": "radon", "action": "configured", "detail": "Found in \
dev deps; added cc -a -n D check (flags D/E/F complexity)"}
  {"item": "pyright", "action": "configured", "detail": "Found \
pyrightconfig.json; added typecheck to quality gate"}
  {"item": "jest", "action": "not_found", "detail": "No jest config \
or test scripts detected"}"""

TASK_PROMPT_SUFFIX = """\

---
Refine this plan into agent-ready tasks as JSON matching the output schema.
Preserve the planner's intent. Refine descriptions, add missing details, \
and fix dependency ordering — do not rewrite tasks from scratch.
You may split plan tasks into smaller pieces (use new ordinals). You may \
read code to understand the codebase and write better task descriptions.

Rules:
- Each task description must be self-contained: what to do, acceptance \
criteria, edge cases, and patterns to follow.
- Never include diffs, patches, or code in task descriptions. Describe \
WHAT to do and WHERE, not the exact code to write.
- Use `blocked_by` for dependencies on other NEW tasks (ordinals from \
this output).
- Use `blocked_by_existing` for dependencies on existing tasks (IDs \
provided in the input).
- Use `external_blockers` for things outside the codebase (API keys, \
design reviews, infrastructure).
- Set `status: "ready"` for tasks with zero blockers of any kind.
- Set `status: "blocked"` for tasks with any blockers.
- **At least one task must be "ready"** — no circular dependency graphs \
where everything blocks everything.
- Always assign `priority` on each task: use `high` for urgent or \
critical-path work, `medium` for normal implementation sequencing, and \
`low` for deferrable or nice-to-have work.
- Specify which files each task creates or modifies.
- Do NOT create "run tests", "verify", or "run regressions" tasks. \
Testing and validation are part of each implementation task, not \
separate work items. Cancel any existing tasks like these too.
- **Check existing tasks before creating new ones.** If an existing \
task already covers the same work (same files, same goal), do NOT \
create a duplicate — skip it. Use `blocked_by_existing` to depend on \
the existing task instead. This applies to tasks in any active state \
(blocked, ready, running, review, approved, failed).
- Only create new tasks for work not covered by existing tasks.
- Use `cancel_tasks` to cancel existing blocked/ready tasks that are \
superseded, stale, or are test-only/verify-only tasks.
- **Task buckets**: If two or more tasks modify any of the same files, \
assign them the same `bucket` label (a short descriptive string like \
"db-layer" or "cli-commands"). The system automatically serializes tasks \
within a bucket — do NOT add `blocked_by` entries between same-bucket \
tasks (the system handles this). Use `null` for tasks that don't share \
files with any other task. Cross-bucket dependencies still use `blocked_by`. \
Never assign a bucket to a task unless at least one other task shares that \
bucket. A single-task bucket will be stripped by the system."""


def append_suffix_once(prompt: str, suffix: str) -> str:
    """Append a suffix once, even if callers pass already-suffixed text."""
    if not suffix:
        return prompt
    if prompt.endswith(suffix):
        return prompt
    return f"{prompt}{suffix}"


def build_enrichment_prompt(raw_prompt: str) -> str:
    """Build an enrichment prompt with the required suffix."""
    return append_suffix_once(raw_prompt, ENRICHMENT_PROMPT_SUFFIX)


def build_enrichment_resume_prompt(answered_questions: list[PlanQuestionRow]) -> str:
    """Build a prompt for resuming enrichment after questions are answered."""
    parts = ["The user has answered your clarifying questions:\n"]
    for q in answered_questions:
        header = q.get("header")
        prefix = f"[{header}] " if header else ""
        parts.append(f"Q: {prefix}{q['question']}")
        parts.append(f"A: {q['answer']}\n")
    parts.append(
        "Please update your enriched_prompt incorporating these answers. "
        "You may ask more questions if critical details are still missing, "
        "or return an empty questions array to proceed."
    )
    return "\n".join(parts)


MAX_PARENT_PLAN_CHARS = 8000


def build_enrichment_continuation_prompt(
    raw_prompt: str,
    parent_enriched_prompt: str | None,
    parent_plan_text: str | None,
    task_outcomes_summary: str | None,
) -> str:
    """Build an enrichment prompt for a plan continuation with parent context.

    Includes the parent's enriched prompt, plan output, and task outcomes
    so the enrichment agent can produce a context-aware specification.
    """
    parts: list[str] = [
        "This is a CONTINUATION of a previous plan. Context from the parent plan follows."
    ]

    if parent_enriched_prompt:
        parts.append("\n## Previous specification (enriched prompt)")
        parts.append(parent_enriched_prompt)

    if parent_plan_text:
        truncated = parent_plan_text[:MAX_PARENT_PLAN_CHARS]
        if len(parent_plan_text) > MAX_PARENT_PLAN_CHARS:
            truncated += "\n... (truncated)"
        parts.append("\n## Previous plan output")
        parts.append(truncated)

    if task_outcomes_summary:
        parts.append("\n## Task outcomes from previous plan")
        parts.append(task_outcomes_summary)

    parts.append("\n## Continuation request")
    parts.append(raw_prompt)

    return append_suffix_once("\n".join(parts), ENRICHMENT_PROMPT_SUFFIX)


def build_exploration_prompt(enriched_prompt: str) -> str:
    """Build an exploration prompt with the required suffix."""
    return append_suffix_once(enriched_prompt, EXPLORATION_PROMPT_SUFFIX)


def build_plan_prompt(plan_prompt: str, exploration_context: str | None = None) -> str:
    """Build a planning prompt with optional exploration context and suffix."""
    if exploration_context:
        prompt = (
            f"<codebase_exploration>\n{exploration_context}\n"
            f"</codebase_exploration>\n\n{plan_prompt}"
        )
    else:
        prompt = plan_prompt
    return append_suffix_once(prompt, PLAN_PROMPT_SUFFIX)


def build_task_creation_prompt(plan_text: str, existing_summary: str = "") -> str:
    """Build a task-creation prompt with optional overlap context."""
    base_prompt = f"Plan JSON:\n```json\n{plan_text}\n```{existing_summary}"
    return append_suffix_once(base_prompt, TASK_PROMPT_SUFFIX)


REVIEW_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approve", "reject"],
            "description": "Whether to approve or reject the changes",
        },
        "summary": {
            "type": "string",
            "description": "Brief review summary",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "major", "minor", "nit"],
                    },
                    "file": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["severity", "file", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "summary", "findings"],
    "additionalProperties": False,
}

EXECUTOR_PROMPT_SUFFIX = """\

---
You are an executor agent. Your job is to implement the task described above.

Rules:
- Read the files you plan to modify AND their tests. Understand the existing \
patterns, naming conventions, and test style before making changes.
- When adding a new function/method, look at adjacent functions in the same \
module for the pattern to follow (parameter order, return type, error handling, \
logging). Match the existing code style exactly — indentation, naming \
conventions, import ordering, docstring style.
- Before writing raw SQL, DB queries, or low-level operations, check if the \
relevant module already has a helper function for what you need. Use existing \
helpers rather than duplicating logic inline.
- Implement exactly what the task description specifies — no more, no less.
- Focus on the files listed in the task. Only modify other files if necessary.
- If the task mentions specific acceptance criteria, verify each one.
- Do not refactor unrelated code. Do not add features beyond the task scope.

Quality checks — ALL MANDATORY before committing:
1. Run the concrete quality-gate command list for this task’s runtime context
   (including any auto-fix commands), in order.
2. Re-run the entire quality-gate command list after each fix.
3. Write tests for your changes.
4. Run the same quality-gate command list end-to-end before attempting review.
Quality-gate expectations are injected by jobs at runtime; do not assume a
single language/tooling stack.

Fixing issues after rejection:
- Read the test output and error messages carefully before changing code. \
The failure message usually points directly at the fix — do not guess.
- Make targeted fixes ONLY to the specific code the reviewer flagged.
- Do not rewrite, restructure, or refactor code that the reviewer did \
not mention. Working code that was not flagged should not be touched.
- If a reviewer suggestion seems wrong or would break working code, push \
back — explain why and propose an alternative fix instead of blindly applying it.
- Re-run the project quality gate end-to-end after fixing to confirm nothing \
regressed.

CRITICAL — committing:
- You MUST `git add` and `git commit` all your changes before finishing.
- NEVER use `git add .` or `git add -A`. Always stage specific files by \
name (`git add path/to/file.py`). Broad staging can sweep in unrelated \
files and corrupt the branch history.
- NEVER commit .env files, credentials, secrets, API keys, tokens, or \
other sensitive material. If you encounter such files, leave them \
untracked.
- Never leave uncommitted work in the worktree. A reviewer agent will \
inspect your commits next.
- Use clear, descriptive commit messages that explain what and why.
- Prefer logical commits (one per meaningful change) over a single \
monolithic commit."""

REFRESH_PROMPT_SUFFIX = """\

---
Review the existing tasks for this project. Produce JSON matching the \
output schema.

Your job is to clean up the task landscape:
- Cancel stale, superseded, or duplicate tasks via `cancel_tasks`.
- Cancel "run tests", "verify", or "run regressions" tasks — testing \
is part of each implementation task, not a separate work item.
- Only blocked/ready tasks can be cancelled.
- Do NOT cancel tasks that are running, completed, or failed.
- The `tasks` array should be empty unless you need to create new work \
items (rare — only if you spot a gap).
- Do NOT create "run tests" or "verify" tasks.

If the user provided specific instructions, follow them."""

REVIEWER_PROMPT_SUFFIX = """\

---
You are a code reviewer. Examine the changes (diff + commits) against \
the task description and acceptance criteria.

Rules:
- Evaluate ONLY what this specific task asks for. Tasks are part of a \
dependency chain — downstream tasks will add integration, usage, and \
further features. Do not reject because the code is not yet used or \
integrated; that is a later task's job.
- Read the diff carefully. Check correctness, edge cases, and test coverage.
- Read surrounding code for context when the diff touches existing files.
- Verify that tests actually assert meaningful behavior — not just that \
code runs without crashing. Tests like `assert result is not None` or \
`assert True` provide no real coverage. Flag these as findings.
- Approve unless there are substantive issues within this task's scope.
- Only reject for: bugs, missing functionality described in THIS task, \
no tests for the code added, security issues, or significant deviation \
from the task description.
- ALWAYS reject if the diff contains committed .env files, credentials, \
API keys, secrets, or other sensitive material.
- Flag (as a finding) if new code uses raw SQL or inline logic when the \
module already provides a helper function for the same operation — this \
is a sign of duplicated logic that will drift.
- Minor style issues, formatting, and lint are findings but NOT grounds \
for rejection — the quality gate handles these mechanically.

Use runtime-injected quality-gate command outcomes for any post-change verification.

Actionable findings — REQUIRED for rejections:
- Every finding MUST include a concrete fix suggestion: which function or \
code block to change, what the correct behavior should be, and how to \
fix it. Do not just describe the problem — describe the solution.
- Verify your findings by reading the actual implementation code, not just \
theorizing from the diff. If you cannot confirm the issue exists in the \
code, do not report it.
- Example: Instead of "NaN bypasses validation", write "In validate_input() \
at line N, add a check for NaN/infinity values using the language's \
appropriate numeric validation (e.g. isfinite equivalent). Currently \
NaN passes the > 0 check."

- Produce structured JSON matching the output schema with your verdict, \
summary, and findings."""

# -- thread/start params by job type --------------------------------------

THREAD_CONFIGS: dict[str, dict] = {
    "prompt_enrichment": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": THINK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "live",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are a prompt engineer and project manager. "
            "Your sole job is to take a user's raw, often vague request "
            "and transform it into a clear, well-structured prompt that "
            "a downstream planning agent will act on.\n\n"
            "You do NOT explore the codebase. You do NOT write code. "
            "You do NOT investigate files or spawn sub-agents. "
            "You craft prompts.\n\n"
            "Think like a PM sitting with a client: understand what they "
            "really want, identify gaps, ask clarifying questions when "
            "genuinely needed, and produce a specification that "
            "eliminates ambiguity for the planner."
        ),
    },
    "plan_request": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": THINK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "live",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are an implementation planner. Your job is to analyze a "
            "codebase and produce a structured implementation plan that "
            "downstream executor agents will carry out.\n\n"
            "You MUST read the codebase directly — use file read and search "
            "tools to understand architecture, naming conventions, and "
            "existing patterns. Do NOT delegate to sub-agents or spawn "
            "parallel exploration tasks.\n\n"
            "Your output MUST be a concrete, actionable plan matching the "
            "output schema with real tasks, real file paths, and specific "
            "implementation details. Never output intermediate progress "
            "updates, exploration narration, or delegation status — only "
            "the final structured plan."
        ),
    },
    "task_creation": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": WORK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "disabled",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are a task decomposition agent. You receive a structured "
            "implementation plan and break it into concrete, agent-ready "
            "task definitions that match the output schema exactly.\n\n"
            "Each task description you write is the SOLE context an "
            "executor agent receives — that agent sees nothing else about "
            "the plan, the project, or other tasks. If a task description "
            "is vague, the executor will guess wrong. Be specific about "
            "which functions to create or modify, which modules to touch, "
            "which patterns to follow, and what the acceptance criteria are.\n\n"
            "You may read code to verify file paths and understand existing "
            "patterns, but do NOT explore broadly. Do NOT delegate to "
            "sub-agents or spawn parallel tasks. Do NOT rewrite the plan — "
            "preserve the planner's intent and decomposition. Refine, "
            "add missing details, and fix dependency ordering.\n\n"
            "Your output MUST be valid JSON matching the output schema. "
            "Never output progress updates, narration, or status — only "
            "the final structured task definitions."
        ),
    },
    "task_execution": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": WORK_MODEL,
        # Executor must be able to write git metadata under the parent repo
        # (e.g. .git/worktrees/*) during add/commit from an isolated worktree.
        # workspace-write can block those parent-path writes depending on
        # sandbox/runtime behavior, causing tasks to spin in running.
        "sandbox": "danger-full-access",
        "config": {
            "web_search": "disabled",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are an autonomous code executor agent. You work alone "
            "in an isolated git worktree and implement exactly what your "
            "task description specifies — no more, no less.\n\n"
            "You write code, run quality gates, write tests, and commit "
            "your changes. You do NOT delegate to sub-agents, spawn "
            "parallel tasks, or ask for human input. You do NOT explore "
            "the codebase beyond what is needed for your specific task.\n\n"
            "Before writing any code, read the files you will modify AND "
            "their tests. Match the existing code style exactly — naming, "
            "patterns, error handling, import ordering. Use existing "
            "helpers instead of duplicating logic.\n\n"
            "You MUST commit all changes before finishing. Never leave "
            "uncommitted work. A reviewer agent inspects your commits "
            "next — quality and correctness matter.\n\n"
            "Use the update_plan tool to track your progress through "
            "each step of the task. Prefer file editing tools "
            "(apply_patch) over shell commands for writing files."
        ),
    },
    "task_review": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": THINK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "disabled",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are a code reviewer agent. You evaluate changes made by "
            "an executor agent against a specific task description and "
            "its acceptance criteria.\n\n"
            "You review ONLY what this specific task asks for. Tasks are "
            "part of a dependency chain — downstream tasks handle "
            "integration, usage, and further features. Do not reject "
            "because code is unused or unintegrated; that is a later "
            "task's responsibility.\n\n"
            "Read the diff AND the surrounding code for full context. "
            "Verify tests assert meaningful behavior, not just that code "
            "runs. Every rejection MUST include actionable fix "
            "suggestions — which code to change and how.\n\n"
            "Your output MUST be valid JSON matching the output schema "
            "with a verdict, summary, and findings. Never output "
            "narration, progress updates, or delegation — only the "
            "final structured review."
        ),
    },
    "codebase_exploration": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": THINK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "disabled",
            "features": {"multi_agent": False},
        },
        "developerInstructions": (
            "You are a codebase exploration agent. Your job is to deeply "
            "understand the project's architecture, conventions, and relevant "
            "code so that a downstream planning agent can make informed decisions "
            "without redundant exploration.\n\n"
            "You MUST read files and search the codebase directly. Do NOT "
            "delegate to sub-agents or spawn parallel tasks.\n\n"
            "Focus your exploration on what is relevant to the task described "
            "in the prompt. Identify: architecture and module boundaries, "
            "naming conventions and patterns, existing helpers and utilities "
            "that should be reused, test file locations and testing patterns, "
            "and API surfaces or interfaces that the task will interact with.\n\n"
            "Your output MUST be a structured exploration report matching the "
            "output schema. Never output meta-narration, delegation status, "
            "or intermediate progress — only the final structured report."
        ),
    },
    "query": {
        "approvalPolicy": "never",
        "personality": "pragmatic",
        "model": THINK_MODEL,
        "sandbox": "read-only",
        "config": {
            "web_search": "disabled",
            "features": {"multi_agent": False},
        },
        "developerInstructions": None,
        "ephemeral": True,
    },
}

# -- turn/start params by job type ----------------------------------------

TURN_CONFIGS: dict[str, dict] = {
    "prompt_enrichment": {
        "outputSchema": ENRICHMENT_OUTPUT_SCHEMA,
        "effort": "high",
    },
    "plan_request": {
        "outputSchema": PLAN_OUTPUT_SCHEMA,
        "effort": "high",
    },
    "task_creation": {
        "outputSchema": TASK_OUTPUT_SCHEMA,
        "effort": "high",
    },
    "task_execution": {
        "effort": "high",
        "summary": "concise",
    },
    "task_review": {
        "outputSchema": REVIEW_OUTPUT_SCHEMA,
        "effort": "high",
        "summary": "concise",
    },
    "codebase_exploration": {
        "outputSchema": EXPLORATION_OUTPUT_SCHEMA,
        "effort": "high",
    },
    "query": {
        "effort": "minimal",
    },  # outputSchema set dynamically from task.output_schema
}
