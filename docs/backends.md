# Backends

## Configuration (`src/agm/backends.py`)

Fixed parameters per job type for each backend. Each job type (enrichment, exploration, plan_request, task_creation, task_execution, task_review, project_setup) has exact settings. **No guessing, no per-project overrides.**

Codex config dicts:
- `THREAD_CONFIGS` — thread/start params (model, sandbox, personality, developerInstructions, ephemeral)
- `TURN_CONFIGS` — turn/start params (effort, outputSchema for structured output)
- `get_runtime_turn_config()` — resolves runtime effort from project model config

Two-tier model strategy:
- **Think tier** (planner, task agent, reviewer, enrichment, explorer, project setup, query): deep reasoning, high intelligence
- **Work tier** (executor): fast execution, clear instructions
- **Codex**: `THINK_MODEL` / `WORK_MODEL` (env: `AGM_MODEL_THINK` / `AGM_MODEL_WORK`)

`MODEL_CATALOG` documents all known models with backend/tier/speed/recommendations/notes. `resolve_model_config(backend, project_model_config)` resolves model+effort per tier with precedence: project `model_config` JSON > env vars > catalog defaults. `get_runtime_thread_config()` builds per-job configs. `model` column on plans and tasks records which model actually ran.

Important boundary: Codex CLI harness config (`~/.codex/config.toml` and role-specific
agent files) does not control agm pipeline model selection. agm only uses
project `model_config`, AGM env vars, and catalog defaults.

`IMPLEMENTED_BACKENDS = {"codex"}` — validated on plan request.

## Codex client (`src/agm/client.py`)

`AppServerClient` manages a `codex app-server` subprocess and communicates via JSON-RPC 2.0 over stdio. **No DB. No queue. No status management.**

- Spawns the process and sends an `initialize` handshake on start
- Correlates request/response pairs by ID
- Dispatches **notifications** (server → client, no response) to registered handlers
- Handles **server requests** (server → client, response expected) via a pluggable handler
- Sends responses back to the server for approval requests, user input requests, etc.
- Request-level timeouts (default 120s, 300s for slow methods like thread/resume, thread/read)
- Drains subprocess stderr in background to prevent OS pipe buffer deadlock
- Cleans up pending futures on timeout, cancellation, or send failure
- Async context manager for clean lifecycle management

## Codex backend flow

The codex plan worker (`_run_plan_request_codex` in jobs.py) executes this flow:

```
jobs.py                          backends.py    client.py                    codex app-server
  │                                │              │                              │
  ├─ load THREAD_CONFIGS ─────────►│              │                              │
  ├─ asyncio.run(async flow) ─────┼─────────────►│                              │
  │                                │              ├─ thread/start(config) ──────►│
  │                                │              │◄─ {thread: {id}} ───────────│
  │  set_plan_request_thread_id()◄─┼──────────────┤                              │
  │                                │              ├─ turn/start(prompt) ────────►│
  │                                │              │◄─ {turn: {id, status}} ─────│
  │                                │              │   ... agent works ...        │
  │                                │              │◄─ turn/completed notif ─────│
  │                                │              ├─ thread/read(includeTurns) ─►│
  │                                │              │◄─ {thread: {turns: [items]}} │
  │  _extract_plan_text() ◄────────┼──────────────┤                              │
  │  finalize_plan_request() ◄─────┼──────────────┤                              │
```

Key behaviors:
- Thread config from `THREAD_CONFIGS` — model, sandbox, approval policy, personality (`pragmatic`), developerInstructions, ephemeral (fire-and-forget threads like memory_update/query)
- Turn config from `TURN_CONFIGS` — `outputSchema` constrains the assistant to produce structured JSON (plan tasks include dependencies, optional `priority`, and `bucket`)
- `approvalPolicy: "never"` — auto-approves all tool calls
- Server requests (requestApproval) are auto-approved by the server request handler
- Plan text is extracted from the **last turn only** — `plan` items first, falling back to `agentMessage` items. Scoping to the last turn prevents continuation plans from finalizing with stale parent output.
- **Token tracking** — `_codex_turn()` accumulates `input_tokens`, `output_tokens`, `cached_input_tokens`, and `reasoning_tokens` from `thread/tokenUsage/updated` notifications (`last` field per turn) and returns them alongside text. Job functions persist tokens via `update_plan_tokens()` / `update_task_tokens()` (additive, COALESCE-safe). Cached/reasoning tokens surfaced in plan/task/project views when non-zero.
- **Turn interrupt on timeout** — `_codex_turn()` listens for `turn/started` to capture the active turn ID, then sends `turn/interrupt` on timeout before re-raising. Prevents zombie Codex sessions.
- Application-level timeout per job type (not rq-level) — cleanup always runs
- Plan/task prompt suffixes include explicit priority rules (`high|medium|low`, omitted = medium) so planning and task refinement produce consistent priority metadata.
- Developer instructions tell the planner (`plan_request` job only) to use Context7 MCP tools for library documentation. Other jobs do not reference Context7.
- Context7 MCP server is configured at the Codex harness level: `codex mcp add context7 -- npx -y @upstash/context7-mcp`
- Per-agent config overrides (`config` dict in THREAD_CONFIGS): pipeline agents lock down `features.multi_agent = false` and set `web_search` per tier (think=live, work=disabled). `query` has no config override (inherits global defaults).

## Thread resumption

When a plan has a `parent_id`, the worker resumes the parent's thread instead of starting a new one:

```
jobs.py                          client.py                    codex app-server
  │                                │                              │
  ├─ look up parent thread_id      │                              │
  ├─ asyncio.run(async flow) ─────►│                              │
  │                                ├─ thread/resume(threadId) ───►│  (reload thread from disk)
  │                                │◄─ {thread: {id, turns}} ────│
  │                                ├─ turn/start(prompt) ────────►│  (agent has full history)
  │                                │   ... agent works ...        │
  │                                │◄─ turn/completed notif ─────│
  │                                ├─ thread/read(includeTurns) ─►│
  │  _extract_plan_text() ◄────────┤                              │
  │  finalize_plan_request() ◄─────┤                              │
```

The agent sees all prior turns — every file it read, every finding it made — and can build on that context with the new prompt.

All `thread/resume` calls pass the current model from `get_runtime_thread_config()` so resumed threads respect the latest project model config (not whatever model the original thread used).

## Backend selection

Backend is resolved per plan request with a fallback chain:

1. Explicit `--backend` CLI flag (highest priority)
2. Project `default_backend` column (set at project creation, always `codex`)
3. Fall back to `"codex"` (default)

All tasks inherit the backend from their plan. `_get_plan_backend()` resolves this from either a plan dict (has `backend` field) or a task dict (traverses plan_id → plan → backend).
