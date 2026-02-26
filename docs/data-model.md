# Data Model

## Schema

```
project          a registered codebase (id, name, dir, default_backend, base_branch, plan_approval, quality_gate, post_merge_command, model_config)
  └─ plan        a plan request (prompt, metadata) → backend produces plan text
       │           statuses: pending → running → awaiting_input → finalized / failed / cancelled
       │           task_creation_status: awaiting_approval → pending → running → completed / failed
       │           tracks: pid, thread_id, parent_id, enriched_prompt, enrichment_thread_id,
       │                   exploration_context, exploration_thread_id, model
       ├─ plan_question   a question from the worker needing an answer
       ├─ plan_log        a worker log entry (level, message, timestamp)
       └─ task   a work item refined from plan tasks by the task agent
            │      statuses: pending / ready → running → review ⇄ running → approved → completed / failed / cancelled
            │      tracks: ordinal, title, description, files, bucket, priority, pid, thread_id, active_turn_id, reviewer_thread_id, actor, caller
            ├─ task_blocks   blocker relationships
            │                  internal: blocked_by_task_id (task-to-task dep)
            │                  external: external_factor + reason (human-resolvable)
            │                  resolved flag + resolved_at timestamp
            └─ task_logs     task-level log entries

status_history   pipeline audit trail (entity_type, entity_id, old_status, new_status, actor)

session          conversation-level container (project_id, status, trigger, trigger_prompt)
  └─ channel_message   inter-agent messages (kind, sender, content, metadata)
                         kinds: steer, question, broadcast, dm, context
                         metadata: optional/nullable JSON payload; structured data is serialized to
                         text for storage in SQLite

trace_events     execution trace data (entity_type, entity_id, stage, turn_index, ordinal, event_type, data)
                   polymorphic FK: entity_type=plan|task, entity_id=plan_id|task_id
                   captures per-event detail from all Codex sessions (files read, commands run, edits, reasoning)
```

A **plan request** is what `plan request` inserts — a prompt with metadata (project, caller, backend, actor). The `plan` column is NULL until the backend produces the actual plan text and the request is finalized.

A **plan continuation** (`plan continue`) creates a child plan with `parent_id` pointing to the parent. The continuation goes through enrichment with parent context (enriched prompt, plan output, task outcomes), resuming the parent's `enrichment_thread_id` when available. After enrichment, the planner resumes the parent's thread (`thread/resume`) so the backend agent has full conversation history, then starts a new turn with the enriched follow-up prompt.

## Prompt enrichment

Before the planner runs, two preparation stages fire:

1. **Enrichment agent** — takes the raw user prompt, reads the repo, and produces a structured enriched prompt with specific file references, acceptance criteria, and edge cases. Owns the question-asking capability.
2. **Explorer agent** — scans the codebase and produces structured findings (architecture, relevant files, reusable helpers, test locations). Output stored as `exploration_context` on the plan row and passed to the planner as context. Non-fatal: planner runs even if exploration fails.

Pipeline: `enrich → (ask questions?) → explore → plan → tasks → execute → review → merge`

Enrichment lifecycle tracked by `prompt_status` column:
- `pending` → fresh enrichment needed
- `enriching` → enrichment in progress
- `awaiting_input` → questions emitted, waiting for answers
- `finalized` → enrichment done, planner can run
- `failed` / `cancelled` → terminal

Skipped for: `agm do` (quick mode — prompt goes straight to executor). Plan continuations go through continuation enrichment with parent context (enriched prompt, plan output, task outcomes); resumes parent's `enrichment_thread_id` when available.

Auto-resume: `plan answer` checks remaining unanswered questions. When all answered, auto-transitions `awaiting_input → running` and re-enqueues `run_plan_request()`. The enrichment thread is resumed with the answered questions.

## Task priority model

- Effective values are `high`, `medium`, `low`.
- DB storage is nullable: `NULL` means `medium`; only `high` and `low` are stored explicitly.
- Legacy rows with literal `"medium"` are still treated as `medium` for filtering and display.

## Project base branch

Configurable base branch per project via `project base-branch NAME [BRANCH] [--reset]`. Default: `main` (NULL in DB). `_resolve_effective_base_branch()` in jobs_common.py resolves from task → plan → project chain. All git operations use the resolved base branch.

## Plan approval gate

Project setting: `plan_approval` column (TEXT, nullable, NULL = auto). Modes: `auto` (default, immediate task creation after plan finalization) or `manual` (pauses at `awaiting_approval` until `plan approve`). Set via `project plan-approval NAME [auto|manual]`.

## Quality gate

Runs before the reviewer agent and again at merge to catch mechanical issues cheaply:
- **Phase 1 (auto-fix)**: configured commands (formatting, lint --fix) → commit if changed
- **Phase 2 (strict)**: configured checks (lint, tests) → failures block review/merge

Results are structured (`QualityGateResult`): per-check name, passed, output, duration, auto-fix metadata. Logged at `QUALITY_GATE` level in task_logs.

Configuration:
- `project setup` — **recommended**: LLM-based setup agent inspects repo and auto-applies quality gate + post-merge config. Trigger explicitly after `agm init` (use `--wait` for completion feedback).
- `project quality-gate --preset python|typescript` — quick-start presets
- `project quality-gate --set '{"auto_fix": [...], "checks": [...]}'` — custom config
- `project quality-gate --generate` — LLM inspects repo, outputs config without applying

When unconfigured, the executor agent is prompted to discover and run appropriate tooling. Plan requests warn when no quality gate is set.

## Sessions & channels

Each plan request creates a `session` tied to the project. Agents post messages to the session's channel during execution:
- Explorer posts structured codebase findings (kind: `context`)
- System posts warnings (e.g., "No quality gate configured", kind: `context`)
- Agents can steer each other or ask questions via channel messages

`session list` and `session messages` CLI commands for inspection.

## Status history

`status_history` table tracks entity status transitions for diagnostics and timing. `record_status_change()` called from every pipeline transition in jobs.py. `plan timeline` / `task timeline` CLI commands show chronological transitions with durations. Timing summaries in `plan show` / `task show`.

## Project memory

`.agm/memory.md` per project. Auto-captures learnings from reviewer rejections (critical/major findings) and execution failures. Injected into executor and reviewer prompts so fresh agents avoid repeating past mistakes. Capped at 3000 chars (tail-truncated).

Memory updates use LLM-based distillation (`_update_project_memory()` in jobs_common.py): reads current memory + new event, calls the codex backend with a `memory_update` config to produce a deduplicated, distilled memory file. On any LLM failure, falls back to raw timestamped append (`_append_project_memory()`).
