# JSON Contracts

Read-command JSON output shapes for machine consumers (agm-web, scripts, agents).

## status

`agm status` returns:
- `models`
- `projects`
- `queue`

Each project includes:
- `project`
- `active_plans`
- `active_tasks`
- `task_breakdown`
- `plans` (active plans only)
- `recent_failures` (`plan_id`, `status`, `error_snippet`, `failed`)

## plan

- `plan list` returns the plan list plus an `error` field on each row.
- `plan failures` returns failure objects with `plan_id`, `project_id`, `project`, `source`, `prompt`, `prompt_snippet`, `error`, `error_snippet`, `created_at`, `updated_at`, `failed`.
- `plan history` returns `{"plan_id": "...", "chain": [...]}` where each chain item includes `id`, `status`, `prompt`, `prompt_preview`, `created_at`, `updated_at`, `is_target`, `position`, `total`.
- `plan timeline` returns `{"plan_id": "...", "timeline": [...]}` with normalized transitions.
- `plan questions` returns `plan_id`, `unanswered_only`, `count`, `questions` with status and timestamps.
- `plan logs` returns `plan_id`, `level`, `count`, `logs` (`id`, `level`, `message`, `created_at`).
- `plan watch` returns a `plan_watch_snapshot_v1` payload.

`plan watch` snapshot fields:
- `schema`: `plan_watch_snapshot_v1`
- `scope` (`type`, `plan_id`, `plan_id_short`, `title`, `title_truncated`)
- `plan` (`status`, `backend`, `task_creation_status`, `created_at`, `updated_at`)
- `runtime`
- `counts` (`tasks_total`, `tasks_active`, `status_summary`)
- `recent_events`
- `terminal_state` (`reached`, `reason`, `message`, `is_plan_terminal`, `is_all_tasks_terminal`)

## task

- `task list` returns raw task rows.
- `task show` returns raw task rows.
- `task timeline` returns `{"task_id": "...", "timeline": [...]}` with normalized transitions.
- `task blocks` returns `scope`, `count`, and normalized `blocks`.
- `task logs` returns `task_id`, `level`, `count`, `logs`.
- `task steer-log` returns `task_id`, `limit`, `offset`, `count`, `items` (`task_steers` audit rows).
- `task watch` returns a `task_watch_snapshot_v1` payload.

`task watch` snapshot fields:
- `schema`: `task_watch_snapshot_v1`
- `scope` (`type` and task/project/plan ids or short ids, plus optional `project_name` / `plan_title_truncated`)
- `watching`
- `counts` (`tasks_total`, `tasks_visible`, `active_tasks`, `status_summary`, `status_summary_visible`)
- `tasks` (`id`, `id_short`, `status`, `plan_id`, `backend`, `title`, `title_truncated`, `priority`, `bucket`, `blocked_count`, `updated_at`)
- `recent_events`
- `terminal_state` (`reached`, `reason`, `message`)

## do

- `do` returns a plain object with keys:
  - `plan_id`
  - `task_id`
  - `title`
  - `status` (always `"running"`)

## project

- `project setup NAME` is a mutation command and returns no payload on success (enqueues async setup).
- `project setup NAME --wait` blocks until setup completes/fails, still returns no payload on success.
- `project setup NAME --dry-run` returns the setup result payload: `quality_gate`, `post_merge_command`, `stack`, `warnings`, `quality_gate_applied`, `post_merge_applied` (both `*_applied` fields are `false` in dry run).
- `project stats NAME` returns project-level statistics.

## trace

- `plan trace PLAN_ID` returns trace events for a plan. `--summary` returns aggregated counts.
- `task trace TASK_ID` returns trace events for a task. `--summary` returns aggregated counts.

## session

- `session list` returns sessions for a project.
- `session post` is a mutation command and returns no payload on success.
- `session messages SESSION_ID` returns channel messages within a session.

## agm-api write/read endpoints

- `session.post` writes a channel message and returns the created message row.
- `task.steer` writes a steer channel message and returns:
  - `task_id`, `session_id`, `message_id`
  - `live_requested`, `live_applied`
  - optional `turn_id` on live success, optional `live_error` on live failure.
- `task.steers` returns the persisted steer audit log:
  - `task_id`, `limit`, `offset`, `count`, `items`.

## help-status

`help-status` returns:
- `schema: status_reference_v1`
- `lifecycles` containing `type`, `label`, `description`, and per-status `status`/`meaning`/`typical_transitions`.
