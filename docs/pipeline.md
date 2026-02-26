# Pipeline

## Prompt assembly

Each pipeline agent receives a prompt built from multiple layers. The layers vary by role:

| Layer | Enrichment | Explorer | Planner | Task Agent | Executor | Reviewer |
|---|---|---|---|---|---|---|
| User prompt / plan JSON | Y | Y (enriched) | Y (enriched) | Y (plan JSON) | Y (task desc) | Y (task + diff) |
| Prompt suffix (`*_PROMPT_SUFFIX`) | Y | Y | Y | Y | Y | Y |
| Role prompts (`agents.toml`) | Y (enrichment) | Y (explorer) | Y (planner) | Y (task_agent) | Y (executor) | Y (reviewer) |
| Developer instructions | None | String | None | None | String | String |
| Exploration context | - | - | Y | - | - | - |
| Pipeline memory (`.agm/memory.md`) | - | - | - | - | Y | Y |
| Quality gate prompt | - | - | - | - | Y | Y |
| Predecessor context | - | - | - | - | Y | - |
| Failed sibling context | - | - | - | - | Y | - |
| Merge conflict context | - | - | - | - | Y | - |

Assembly order (executor example, most complex):
1. Task title + description + file list
2. Predecessor context (completed blockers)
3. Failed sibling context (warns about failed plan tasks)
4. Merge conflict context (previous diff on re-execution)
5. Pipeline memory (`.agm/memory.md`)
6. Quality gate prompt (configured checks or discovery prompt)
7. `EXECUTOR_PROMPT_SUFFIX` (hardcoded rules)
8. Role prompts from `agents.toml` (appended last via `_append_merged_agent_instructions()`)

Assembly order (planner example):
1. Enriched prompt (from enrichment agent)
2. Exploration context (structured codebase findings from explorer agent, if available)
3. `PLANNER_PROMPT_SUFFIX` (hardcoded rules)
4. Role prompts from `agents.toml`

Developer instructions are set at the backend config level (thread/start for Codex). They are separate from the prompt — they frame the agent's role. All pipeline jobs have explicit role-specific instructions. Only `query` mode uses `None` to preserve the backend's built-in template.

## Queue infrastructure

Job queue backed by **rq** (Redis Queue).

- **Redis** as the message broker (configurable via `AGM_REDIS_URL`, default `redis://localhost:6379/0`)
- **Named queues**: `agm:plans` (plan requests), `agm:tasks` (task creation/refresh), `agm:exec` (task execution/review — parallel, separate from creation), `agm:merge` (task merge — serialized, single-worker to prevent concurrent merges), `agm:query` (query-mode execution — parallel, separate from code execution)
- **No rq-level retry or timeout** — failures are immediate and visible. The plan is marked failed in DB and logged. Retries are explicit via `agm plan retry`. Each job type owns its timeout at the application level (e.g. `asyncio.wait_for` in jobs.py). Burst workers exit when the job finishes, so no resource leak risk.
- **Job dependencies (internal)**: rq uses `depends_on` for internal chaining (plan → tasks, task blocked by another task). Agents never set job dependencies directly — the pipeline wires them automatically.

```
agm plan request "add auth" -p myproject  # domain cmd: writes DB + enqueues + spawns worker
agm plan continue PLAN_ID "follow up"     # domain cmd: child plan resuming parent's thread
agm plan retry PLAN_ID                    # domain cmd: reset failed plan + re-enqueue
agm plan retask PLAN_ID                   # domain cmd: re-trigger task creation for finalized plan
agm queue status                          # ops: show job counts per queue
agm queue failed                          # ops: list failed jobs with errors
agm queue flush                           # ops: clear failed jobs from registries
```

Workers are **dynamic**: `enqueue_plan_request()` spawns a background worker for the job. The worker lives until the plan reaches a terminal state (finalized/failed), then exits. No idle workers. Worker stdout/stderr is captured to `~/.config/agm/logs/{job_id}.log` (e.g. `plan-<id>.log`, `exec-<id>.log`) so startup crashes are diagnosable.

## Job lifecycle

All status transitions happen in `jobs.py`. The flow for a plan request:

```
cli.py                    queue.py                  jobs.py (rq worker)        db.py
  │                         │                           │                        │
  ├─ create_plan_request()─►├─ write plan request ──────┼───────────────────────►│ INSERT (pending)
  ├─ enqueue_plan_request()►├─ push to Redis            │                        │
  │                         ├─ _spawn_worker() ─────────┤ (background process)   │
  │                         │                           │                        │
  │                         │  rq dispatches ──────────►├─ run_plan_request()    │
  │                         │                           ├─ set_plan_request_worker()►│ UPDATE pid
  │                         │                           ├─ update_plan_request_status()►│ UPDATE running
  │                         │                           │                        │
  │                         │                           ├─ [enrichment phase]    │
  │                         │                           ├─ run_enrichment()      │
  │                         │                           ├─ add_question() ──────►│ INSERT question
  │                         │                           ├─ update_plan_request_status()►│ UPDATE awaiting_input
  │                         │                           │  ... wait for answer   │
  ├─ plan answer ──────────►├── re-enqueue ────────────►├─ resume enrichment ───►│ UPDATE answer
  │                         │                           ├─ update_plan_enrichment()►│ UPDATE enriched_prompt
  │                         │                           │                        │
  │                         │                           ├─ [exploration phase]   │
  │                         │                           ├─ run_explorer()        │
  │                         │                           ├─ (structured findings)►│ UPDATE exploration_context
  │                         │                           │  (non-fatal on failure)│
  │                         │                           │                        │
  │                         │                           ├─ [planner phase]       │
  │                         │                           ├─ (enriched prompt +    │
  │                         │                           │   exploration context) │
  │                         │                           ├─ finalize_plan_request()►│ UPDATE finalized
  │                         │                           │                        │
```

## Task creation

When a plan is finalized, `_trigger_task_creation()` in jobs.py automatically enqueues a task creation job. The task agent refines plan tasks into agent-ready work items:

```
jobs.py                          backends.py    client.py                    codex app-server
  │                                │              │                              │
  ├─ plan finalized ──────────────┤              │                              │
  ├─ _trigger_task_creation() ────┤              │                              │
  │  set task_creation_status     │              │                              │
  │  enqueue to agm:tasks ────────┤              │                              │
  │                                │              │                              │
  ├─ run_task_creation() ─────────┤              │                              │
  ├─ load THREAD_CONFIGS["task_creation"]►│      │                              │
  ├─ build prompt (plan JSON + existing tasks)    │                              │
  │                                │              ├─ thread/start(config) ──────►│
  │                                │              │◄─ {thread: {id}} ───────────│
  │                                │              ├─ turn/start(prompt) ────────►│
  │                                │              │   ... agent works ...        │
  │                                │              │◄─ turn/completed ───────────│
  │                                │              ├─ thread/read ──────────────►│
  │  _insert_tasks_from_output() ◄┼──────────────┤                              │
  │  create_tasks_batch() ────────┼──────────────►│  (single transaction)       │
```

Key behaviors:
- **Fresh thread** — task agent gets its own thread, not resuming the plan thread. Plan JSON is self-contained input.
- **Task agent sees the full task landscape** — input includes plan JSON + all existing pending/ready tasks for the project with their IDs. Enables cross-plan dependency wiring.
- **Structured output** via `TASK_OUTPUT_SCHEMA` — ordinals, blocked_by (new task ordinals), blocked_by_existing (existing task IDs), external_blockers, status, optional `priority`, `bucket`.
- **Deadlock prevention** — at least one task must be `ready`. If the agent produces none, the first task is forced to `ready`.
- **Batch insert** — `create_tasks_batch()` inserts all tasks + blocks in a single transaction for atomicity. Auto-injects intra-bucket blockers (consecutive tasks in same bucket, sorted by ordinal). Skips if agent already set the blocker. Forces non-first bucket tasks to `pending`.
- **Task buckets** — `bucket` column (nullable TEXT) on tasks. Tasks sharing a non-null bucket run serially (auto-injected `blocked_by` between consecutive ordinals). Different buckets run in parallel. NULL = no auto-serialization (quick mode, standalone tasks). Replaces the unreliable "file-overlap dependencies" prompt instruction with system-enforced serialization.
- **Priority ingestion** — task-agent priority is validated as `high|medium|low`; omitted/`medium` is normalized to DB `NULL`; invalid values fail task creation.
- **Priority-aware prompt context** — existing task summaries in creation/refresh prompts include effective priority (`NULL` rendered as `medium`) so the agent can account for urgency alongside blockers and buckets.
- **Blocker types** — internal (task blocks task via `blocked_by_task_id`, including auto-injected bucket blockers), external (human-resolvable via `external_factor` + `reason`).
- **`task unblock`** — only resolves external blockers. Internal blocker resolution depends on the blocking task's terminal state via `resolve_blockers_for_terminal_task()`: **completed** → resolve blockers, promote downstream to ready; **failed** → do nothing (downstream stays blocked awaiting `task retry`); **cancelled** → cascade-cancel all transitively-dependent tasks.
- **Task cancellation** — task agent output includes `cancel_tasks` to cancel stale/superseded/duplicate pending/ready tasks (batch cancel does NOT resolve blockers or cascade). `task refresh` triggers a cleanup pass. `task cancel TASK_ID` manually cancels any non-terminal task (with optional `--reason`), cleans up worktree/branch if present, cascade-cancels transitively-dependent tasks, and logs the cancellation.
- **Task ownership** — `claim_task()` creates a git worktree, records `actor` (who), `caller` (how), `branch`, and `worktree` path on the task, plus a log entry for audit trail. `task retry` removes the old worktree+branch (so next executor starts clean) and clears all ownership fields.
- **`task cleanup --project`** — targets terminal tasks (`completed`/`cancelled`/`failed`) that still have stored git refs (`branch` + `worktree`). For each candidate, it does best-effort git worktree/branch cleanup, then clears `branch` and `worktree` in DB. Failures are reported per task, and cleanup continues for the remaining tasks before printing a final cleanup report (`cleaned`, `failed`, `total`).

## Task execution

Launches an executor agent to implement a task in its worktree:

```
cli.py                           queue.py                  jobs.py (rq worker)        db.py
  │                                │                           │                        │
  ├─ task run TASK_ID             │                           │                        │
  │  (auto-claim if ready) ──────┼──────────────────────────┼───────────────────────►│ claim_task()
  ├─ enqueue_task_execution() ──►├─ push to agm:exec        │                        │
  │                                ├─ _spawn_worker() ────────┤ (background process)   │
  │                                │                           │                        │
  │                                │  rq dispatches ─────────►├─ run_task_execution() │
  │                                │                           ├─ set_task_worker() ──►│ UPDATE pid
  │                                │                           ├─ [codex backend]      │
  │                                │                           │  thread/start (cwd=worktree)
  │                                │                           │  turn/start (task prompt)
  │                                │                           │  ... agent implements task ...
  │                                │                           │  turn/completed
  │                                │                           │  thread/read
  │                                │                           ├─ update_task_status() ►│ UPDATE review
  │                                │                           │                        │
```

Key behaviors:
- **Fresh thread in worktree** — executor gets its own thread with `cwd` set to the task's worktree, not the project root.
- **`sandbox: "danger-full-access"`** — no sandbox. The executor runs in an isolated git worktree which provides natural containment. `workspace-write` creates a git overlay in `/tmp` that traps commits outside the real repo, breaking the merge pipeline.
- **`developerInstructions`** — actual string describing the executor role (not `None` — no plan.md template to preserve outside plan mode).
- **No `outputSchema`** — executor produces code and commits, not structured JSON.
- **Separate queue** — `agm:exec` keeps long-running execution jobs from blocking `agm:tasks` (task creation/refresh). No single-worker constraint — multiple executors run in parallel (each in its own worktree).
- **On success** — task transitions to `review`. Reviewer agent picks it up via `task review TASK_ID`.
- **Predecessor context** — when the task has completed predecessors (resolved internal blockers), `_get_predecessor_context()` includes their titles and description excerpts in the executor prompt so the agent can follow established patterns.
- **Project memory** — `.agm/memory.md` learnings from past rejections and failures are injected into executor and reviewer prompts. Fresh agents avoid repeating past mistakes without needing thread history.
- **On failure** — task transitions to `failed`, blockers resolved (downstream tasks promoted). Traceback logged to `task_logs`. Failure reason captured to project memory.
- **Auto-claim** — `task run` on a ready task auto-creates a worktree and claims it before enqueuing. Already-running tasks (from `task claim`) can be run directly.
- **Claim-enqueue crash recovery** — if enqueue fails after auto-claim, `_rollback_claim()` resets the task to `ready` and removes the worktree, preventing stuck-in-running orphans.
- **Thread resume after rejection** — if the task already has a `thread_id` (previously executed, then rejected by reviewer), the executor resumes its existing thread via `thread/resume` instead of starting fresh. The prompt includes the reviewer's findings from the latest REVIEW-level task_log entry.
- **Execution tracing** — `TraceContext` subscribes to Codex notifications and records per-event trace data (files read, commands run, edits made) to `trace_events`, including `commandExecution` and `tokenUsage` events. Best-effort: never blocks the executor on write failures.

## Task review

The `task review` command is dual-mode:
- **`running` → `review`**: manual status transition (executor signals done)
- **`review` → launch reviewer**: enqueues the reviewer agent job

The reviewer agent evaluates the executor's changes and produces a structured verdict:

```
cli.py                           queue.py                  jobs.py (rq worker)        db.py
  │                                │                           │                        │
  ├─ task review TASK_ID          │                           │                        │
  │  (task in review) ────────────┤                           │                        │
  ├─ enqueue_task_review() ──────►├─ push to agm:exec        │                        │
  │                                ├─ _spawn_worker() ────────┤ (background process)   │
  │                                │                           │                        │
  │                                │  rq dispatches ─────────►├─ run_task_review()     │
  │                                │                           ├─ set_task_worker() ──►│ UPDATE pid
  │                                │                           ├─ git diff main...HEAD  │
  │                                │                           ├─ git log main..HEAD    │
  │                                │                           ├─ [codex backend]       │
  │                                │                           │  thread/start (read-only, cwd=worktree)
  │                                │                           │  turn/start (task + diff + commits + REVIEWER_PROMPT_SUFFIX)
  │                                │                           │  ... agent reviews ...
  │                                │                           │  turn/completed
  │                                │                           │  thread/read
  │                                │                           ├─ parse verdict JSON    │
  │                                │                           │                        │
  │  if approve:                   │                           ├─ update_task_status() ►│ UPDATE approved
  │  if reject:                    │                           ├─ add_task_log(REVIEW)─►│ INSERT review log
  │                                │                           ├─ update_task_status() ►│ UPDATE running
```

Key behaviors:
- **Fresh thread** — reviewer gets its own thread (no executor context bleed). The executor's `thread_id` is preserved; the reviewer's thread is stored in `reviewer_thread_id`. `set_task_worker()` uses a sentinel default so calling it without `thread_id` only updates `pid`, not clobbering the executor's thread.
- **`sandbox: "read-only"`** — reviewer can read code and run read-only commands but cannot write files.
- **`developerInstructions`** — actual string describing the reviewer role.
- **Structured output** via `REVIEW_OUTPUT_SCHEMA` — verdict (approve/reject), summary, findings with severity/file/description.
- **Empty submission gate** — before launching the reviewer agent, checks `git diff` and `git log` against main. If no commits and no diff, auto-rejects with a synthetic verdict (executor produced no code). Prevents the reviewer from reading existing code on main and incorrectly approving.
- **On approve** — task transitions to `approved`.
- **On reject** — findings are logged at REVIEW level to `task_logs`, task transitions back to `running`. Critical/major findings are captured to project memory (`.agm/memory.md`). The executor can be re-run (`task run TASK_ID`) and will resume its thread with the rejection context.
- **On failure** — task stays in `review` (executor's work is preserved). User re-runs the reviewer.
- **Same queue** as executor (`agm:exec`) — no single-worker constraint, parallel reviews OK.
- **Quality gate pre-check** — before launching the reviewer, runs configured quality gate checks (auto-fix phase commits fixes, strict phase blocks on failures). Results logged at `QUALITY_GATE` level in task_logs.
- **Review tracing** — `TraceContext` captures reviewer events to `trace_events`, same as execution.

## Task merge

Merges an approved task's branch into main — a deterministic git operation, not a codex agent. Available both as a CLI command and as an auto-triggered queue job (`run_task_merge` on `agm:merge`).

```
cli.py / jobs.py (auto-trigger)
  │
  ├─ task merge TASK_ID  (CLI)  or  _trigger_task_merge()  (auto)
  │  validate: status == approved, has worktree + branch
  │  look up project dir (task → plan → project)
  │
  ├─ merge_to_main(project_dir, branch, task_id, title, worktree_path)   [git_ops.py]
  │    ├─ rebase_onto_main() if worktree_path provided (handles non-conflicting drift)
  │    ├─ create detached temp worktree at main's SHA (avoids branch conflicts)
  │    ├─ git merge --no-ff in temp worktree
  │    ├─ git update-ref to advance main branch
  │    └─ cleanup temp worktree (git remove + shutil fallback)
  │    on rebase conflict: git rebase --abort, raise error
  │    on merge conflict: git merge --abort, raise error
  │
  ├─ update_task_status(completed)
  ├─ resolve_blockers_for_terminal_task()  →  completed: promotes, cancelled: cascade-cancels, failed: no-op
  ├─ remove_worktree()  →  removes task worktree + branch
  └─ _trigger_task_execution() for each promoted task  (auto-trigger only)
```

Key behaviors:
- **Rebase-before-merge** — when called with a `worktree_path`, rebases the task branch onto main before merging. Handles non-conflicting drift automatically (e.g., earlier tasks merged, moving main forward). On rebase conflict: aborts rebase, raises error.
- **Always uses detached temp worktree** — creates a temporary worktree at main's SHA with `--detach` (avoids "branch already checked out" conflicts), merges there, then advances main via `update-ref`. Never disturbs the user's working tree regardless of what branch they're on.
- **No-op detection** — before merging, checks `git rev-list --count main..branch`. If zero (branch has no commits ahead of main), raises RuntimeError instead of silently succeeding.
- **Merge conflict re-execution** — on first merge conflict: captures the branch's diff (`git diff base...branch`), logs it as MERGE_CONFLICT level task_log, tears down old worktree+branch, creates fresh worktree from current main, resets task `approved → running` (clears thread_id/reviewer_thread_id via `reset_task_for_reexecution()`), re-triggers executor with previous diff injected as reference context. Executor re-implements the same changes adapted to the current codebase. Capped at 1 re-execution attempt — checked via existing MERGE_CONFLICT task_logs. Second conflict leaves task `approved` for manual resolution. Non-conflict errors propagate as before.
- **On success** — full completion: status → completed, blockers resolved, worktree cleaned up, promoted tasks auto-triggered.
- **Quality gate pre-merge** — runs configured quality gate checks again before merging. Failures block the merge.
- **Post-merge command** — if configured on the project, runs after successful merge (merge SHA is available as `AGM_MERGE_SHA`). Common uses: rebuild binaries, sync dependencies.
- **Serialized queue** — `agm:merge` uses single-worker mode to prevent concurrent merges from conflicting.
- **`task complete` remains** — for manual "I already merged outside agm" cases. `task merge` = merge + complete.

## Auto-trigger pipeline

After plan finalization, the full task lifecycle runs hands-free. The pre-planning stages and post-plan cascade:

```
plan request enqueued
  └─ run_plan_request()
       ├─ run_enrichment()  →  enriched prompt (may ask questions → awaiting_input)
       ├─ run_explorer()    →  exploration context (non-fatal on failure)
       └─ planner turn      →  plan finalized
            └─ _trigger_task_creation()
                 └─ task agent produces tasks (ready / pending)
                      │
                      ├─ ready tasks (priority-sorted) ──► _trigger_task_execution()
                      │                    ├─ create worktree (git_ops)
                      │                    ├─ claim_task(caller="agm-auto")
                      │                    └─ enqueue to agm:exec
                      │                         └─ executor runs
                      │                              └─ _trigger_task_review()
                      │                                   └─ enqueue to agm:exec
                      │                                        └─ reviewer runs
                      │                                             ├─ approve ──► _trigger_task_merge()
                      │                                             │                └─ enqueue to agm:merge (serialized)
                      │                                             │                     └─ merge to main
                      │                                             │                          ├─ completed
                      │                                             │                          ├─ resolve blockers → promote tasks
                      │                                             │                          └─ _trigger_task_execution() for promoted
                      │                                             │
                      │                                             └─ reject ──► check rejection count
                      │                                                  ├─ < MAX_REJECTIONS (3): back to running
                      │                                                  │    └─ _trigger_task_execution() (resume thread)
                      │                                                  └─ >= MAX_REJECTIONS: failed (no retry)
                      │
                      └─ stale blocker resolution ──► _trigger_task_execution() for promoted (priority-sorted)
```

Key design:
- **Catch + log, never re-raise** — all `_trigger_*` functions follow the same pattern as `_trigger_task_creation()`. A trigger failure never takes down the parent job.
- **`agm-auto` caller** — auto-triggered claims use this caller to distinguish from manual `cli` claims.
- **Serialized merges** — `agm:merge` queue uses single-worker mode. Only one merge runs at a time, preventing concurrent merge conflicts.
- **Priority-first scheduling for newly-ready tasks** — `_auto_trigger_execution_for_ready_tasks()` sorts IDs by effective priority (`high` → `medium`/DB `NULL` → `low`) before enqueueing. Stable ties are broken by `ordinal`, then `created_at`, then task ID. Used for newly-created ready tasks and tasks promoted by blocker resolution (including post-merge promotion).
- **Rejection limit** — `MAX_REJECTIONS = 3`. After 3 reviewer rejections per execution cycle (counted from REVIEW-level task_logs after the most recent "Claimed" log entry), the task transitions to `failed` instead of cycling back to the executor. Counter resets on `task retry` because retry triggers a new claim.
- **Reviewer scope** — reviewer evaluates only what the specific task asks for. Tasks are part of a dependency chain — downstream tasks add integration and usage. Missing integration by later tasks is not grounds for rejection.
- **Both CLI and auto** — every stage can also be triggered manually via CLI (`task run`, `task review`, `task merge`). Auto-triggers don't prevent manual intervention.
- **Watch snapshots for integrations** — `plan watch` and `task watch` emit JSON snapshot schema names (`plan_watch_snapshot_v1`, `task_watch_snapshot_v1`) for downstream tooling consumers.
