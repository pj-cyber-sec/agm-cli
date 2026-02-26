# Architecture

## Overview

agm is a CLI that orchestrates AI agents through a fully automated pipeline. It uses the Codex backend (app-server JSON-RPC subprocess). The backend is selected per plan request via `--backend` flag, project `default_backend` column, or falls back to `codex`.

See also: [Data model](data-model.md) | [Pipeline](pipeline.md) | [Backends](backends.md) | [JSON contracts](json-contracts.md) | [Protocol](protocol.md)

## Components & responsibilities

Each module has a single, clear job. No overlap.

### CLI (`src/agm/cli.py`) — user/agent interface

Thin layer. Parses input, calls db + queue, formats output. **No business logic. No status transitions. No direct Redis access.**

Command groups:
- `agm init` - project registration/bootstrap (git init if needed, register project, ensure `.agm/` ignore)
- `agm project setup` - LLM-based setup agent (auto-configures quality gate + post-merge command)
- `agm status` - global dashboard (active projects, task breakdown, queue health)
- `agm project` - manage registered projects (add, list, show, remove, move, rename, setup, quality-gate, plan-approval, post-merge-command, base-branch, model-config, stats)
- `agm plan` - manage plans (request, failures, continue, list, view, history, questions, answer, logs, retry, retask, approve, troubleshoot, stats, timeline, trace, watch)
- `agm task` - manage tasks (list, view, blocks, unblock, set-priority, claim, run, steer, review, reject, approve, merge, complete, cancel, retry, check, cleanup, refresh, logs, failures, timeline, trace, watch)
- `agm queue` - monitor the job queue (status, failed, flush)
- `agm do` - quick single-task mode (skip planner/task agent)
- `agm session` - inspect and post inter-agent channel messages (`list`, `show`, `messages`, `post`)
- `agm doctor` - health checks + auto-remediation
- `agm conflict` - cross-worktree conflict detection (via clash)
- `agm agents` - manage per-role agent instruction templates (agents.toml)
- `agm help-status` - canonical lifecycle reference

All read commands output JSON unconditionally. All mutation commands are silent (exit 0 on success, ClickException on failure).

Domain commands (`plan request`, future `task create`) write to DB **and** enqueue — the caller never thinks about the queue. Queue commands are monitoring only.

Priority-related CLI surface:
- `task list --priority {high|medium|low}` filters by effective priority. `medium` matches DB `NULL` plus legacy `"medium"` rows.
- `task set-priority TASK_ID {high|medium|low}` updates priority (`medium` is persisted as DB `NULL`).
- Priority is rendered in `task list` (`pri:*`), `task show` (`priority:*`), and `task watch` (single-task + grouped views) using the same effective model (`NULL`/legacy `"medium"` shown as `medium`).

`help-status` uses `src/agm/status_reference.py` as the canonical source of truth for lifecycle meaning and transitions.

### Database (`src/agm/db.py`) — pure data storage

CRUD operations and enum validation. **No workflow logic. No status transitions. No queuing.**

Functions store and retrieve rows. They validate field values (caller, backend, status) but never decide *when* a status should change — that's the job layer's call.

Schema versioned via `PRAGMA user_version` (`SCHEMA_VERSION` constant). Migrations only run when `user_version < SCHEMA_VERSION`. `busy_timeout=10s` prevents SQLITE_BUSY under concurrent workers.

CAS-guarded status transitions with optional `record_history` parameter — pipeline callers (jobs.py) pass `True`; CLI callers use the default `False`.

Tables:
- `projects` - registered codebases (id, name, dir, default_backend, base_branch, plan_approval, quality_gate, post_merge_command, model_config)
- `plans` - plan requests (prompt, status, plan text, pid, thread_id, parent_id, task_creation_status, enriched_prompt, enrichment_thread_id, exploration_context, exploration_thread_id, model)
- `plan_questions` - questions from workers needing human/agent answers
- `plan_logs` - worker log entries persisted per plan (level, message)
- `tasks` - work items derived from plans (ordinal, title, description, files, status, bucket, priority, pid, thread_id, active_turn_id, reviewer_thread_id, actor, caller, branch, worktree, skip_review, skip_merge, model)
- `task_blocks` - blocker relationships (internal task deps, external factors)
- `task_logs` - task-level log entries (level, message)
- `sessions` - conversation-level containers (id, project_id, status, trigger, trigger_prompt)
- `channel_messages` - inter-agent communication within sessions (id, session_id, kind, sender, content)
- `channel_messages` are queryable via `agm session messages` with `kind`, `sender`, and `recipient` filters.
- `status_history` - entity status transitions (entity_type, entity_id, old_status, new_status, actor)
- `trace_events` - execution trace data (entity_type, entity_id, stage, turn_index, ordinal, event_type, data)

### Queue (`src/agm/queue.py`) — job dispatch, worker lifecycle & monitoring

rq/Redis integration. Enqueue jobs, spawn workers, connection management, queue inspection, event publishing (`publish_event()` dual-publishes to Redis Stream + pub/sub), event subscription (`EventSubscriber` via XREAD BLOCK). **No DB access. No business logic. No status decisions.**

### Jobs (`src/agm/jobs*.py`) — business logic & status transitions

`jobs.py` is a re-export facade. The actual logic lives in focused submodules:

- **jobs_common.py** — shared infrastructure (constants, DB handlers, codex client helpers, project memory read/append/distill, agent instruction merging, prompt builders)
- **jobs_enrichment.py** — prompt enrichment (`run_enrichment`, fresh/resume/continuation variants, enrichment processing)
- **jobs_explorer.py** — codebase exploration (`run_explorer`, structured findings for the planner, non-fatal on failure)
- **jobs_plan.py** — plan request orchestration (`run_plan_request`, planning turn, question handling)
- **jobs_task_creation.py** — task creation + refresh (`run_task_creation`, `run_task_refresh`, bucket verification)
- **jobs_execution.py** — task execution (`run_task_execution`, predecessor/sibling context, merge conflict prompt)
- **jobs_review.py** — task review (`run_task_review`, verdict handling, pre-review gates)
- **jobs_quality_gate.py** — quality gate checks (run checks, structured results, LLM-generated configs)
- **jobs_merge.py** — merge + auto-triggers (`run_task_merge`, trigger chains, rollback)
- **jobs_setup.py** — project setup agent (`run_project_setup`, inspects repo, auto-applies quality gate + post-merge config)
- **jobs_external.py** — stateless LLM calls (`run_external`, Redis result storage, no DB side effects; used by SDK/integration clients)
- **tracing.py** — execution tracing (`TraceContext`, `extract_rich_trace`, captures per-event data from Codex sessions)

Entry points that dispatch on backend: `run_plan_request`, `run_enrichment`, `run_explorer`, `run_task_creation`, `run_task_execution`, `run_task_review`, `run_task_refresh`. `run_task_merge` and `run_external` are backend-agnostic. Every status change in the system originates here.

### Git operations (`src/agm/git_ops.py`) — shared git helpers

Git worktree and merge operations shared by CLI and job workers. Raises `RuntimeError` on failure (not `ClickException`) so functions work from both contexts. CLI wraps to `ClickException`.

All git operations accept a `base_branch` parameter (default `"main"`) to support configurable base branches per project.

Functions:
- `slugify(text, max_len)` — branch-safe slug from title
- `create_worktree(project_dir, task_id, title, base_branch)` — creates branch + worktree under `.agm/worktrees/`
- `remove_worktree(project_dir, worktree_path, branch)` — best-effort cleanup
- `rebase_onto_main(worktree_path, base_branch)` — rebases task branch onto base branch if behind. Aborts and raises RuntimeError on conflict.
- `merge_to_main(project_dir, branch, task_id, title, worktree_path, base_branch)` — always uses detached temp worktree + `update-ref` to advance base branch. If `worktree_path` provided, rebases first.
- `check_branch_file_scope(project_dir, branch, allowed_files, base_branch)` — checks if branch touches files outside the allowed list. Returns out-of-scope files.
- `compute_directory_disk_usage(path)` — returns disk usage in bytes (used by doctor)
- `inspect_worktrees(project_dir)` — lists worktrees with metadata (used by doctor)
- `detect_worktree_conflicts(project_dir)` — runs `clash status --json` and returns parsed conflict data. Gracefully handles clash not being installed.
- `get_real_conflicts(clash_result)` — filters clash output to only real conflicts.

### Agent config (`src/agm/agents_config.py`) — role-specific instructions

Loads and merges role-specific agent instructions from TOML files. Two scopes: global (`~/.config/agm/agents.toml`) and project-level (`.agm/agents.toml`). Supports 6 roles in pipeline order: `enrichment`, `explorer`, `planner`, `task_agent`, `executor`, `reviewer`. `_load_agent_instructions()` merges both scopes (project overrides global). Consumed by `_append_merged_agent_instructions()` in jobs.py, which injects custom instructions into all agent prompts. `agm agents` CLI command for scaffolding, viewing, and editing.

### Doctor (`src/agm/doctor.py`) — health checks & remediation

`run_doctor(db_path, fix=)` returns structured `DoctorReport`. Checks: Redis connectivity, SQLite integrity, backend CLI availability (codex on PATH), stale PIDs, orphaned worktrees, worktree conflicts (via clash), disk usage, worker log files, stale running entities. `--fix` auto-remediates stale workers (→ failed), orphaned worktrees (remove dirs + clear DB refs), and stale log files (>7d, no active entity). Fix re-runs affected checks to confirm resolution. Non-zero exit on remaining failures.

### Status reference (`src/agm/status_reference.py`) — lifecycle definitions

Canonical source of truth for plan, task, and task-creation status lifecycles. Returns structured data consumed by `help-status`. Defines valid statuses, their meanings, and typical transitions.

## Message flow

```
agm CLI
  └─ AppServerClient
       ├─ stdin  → JSON-RPC requests  → codex app-server
       └─ stdout ← JSON-RPC responses ← codex app-server
                 ← JSON-RPC notifications (events, progress, approvals)
```

## Local workspace (`.agm/`)

Each registered project gets a `.agm/` directory (gitignored) for agent infrastructure:

```
~/Projects/my-app/
  ├── .agm/                            # agm local workspace (gitignored)
  │   ├── worktrees/                   # git worktrees for agents
  │   │   ├── fix-login-bug/           # agent A on branch fix-login-bug
  │   │   └── add-tests/              # agent B on branch add-tests
  │   └── (future: task-level files)
  ├── .git/
  ├── .gitignore
  └── src/
```

## Git worktrees

Each agent runs in its own git worktree so multiple agents can work on the same repo in parallel without conflicts.

agm manages the full lifecycle:
- Task claimed -> create branch + worktree under `.agm/worktrees/`
- Agent runs in the worktree with its own cwd
- Task completed -> worktree removed + branch deleted (cleanup is best-effort)

Uses native `git worktree` (no external tooling needed).

## Identity & traceability

Every plan request records:
- **actor** - the human responsible (OS `$USER`)
- **caller** - what software invoked it (`cli`, `claude-code`, `codex-cli`, `agm-auto`)
- **backend** - which agent system produces it (`codex`)
- **pid** - OS process ID of the worker handling this plan
- **thread_id** - backend session ID (e.g. codex thread)

Callers are validated against an approved list.

### Execution tracing

`trace_events` table (schema v27) captures structured per-event data from every Codex session across all pipeline stages. Each event records entity context (`entity_type` + `entity_id`), pipeline stage (e.g. `enrichment`, `execution`, `review`), event type (`commandExecution`, `fileReadTool`, `fileEditTool`, `reasoning`, etc.), and a JSON data blob with event-specific fields.

`TraceContext` (tracing.py) subscribes to Codex item/started and item/completed notifications and writes trace events to SQLite. Best-effort: never blocks the agent on write failures.

CLI access via `plan trace PLAN_ID` and `task trace TASK_ID`. Summary mode (`--summary`) returns aggregated counts (files read, commands run, edits made, etc.) without raw event data.
