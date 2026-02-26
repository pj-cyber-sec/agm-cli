# Codex App-Server Protocol

JSON-RPC 2.0 over stdio. Messages are newline-delimited JSON.

Codex CLI version: 0.105.0

## Connection lifecycle

1. Spawn `codex app-server` as subprocess
2. Send `initialize` request with client info + capabilities
3. Send requests, receive responses and notifications
4. Close stdin to shut down

## Request format

```json
{"jsonrpc": "2.0", "id": 1, "method": "thread/list", "params": {}}
```

## Response format

```json
{"jsonrpc": "2.0", "id": 1, "result": {"data": []}}
```

## Error format

```json
{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Invalid request: ..."}}
```

## Notification format (server → client, no response expected)

```json
{"jsonrpc": "2.0", "method": "turn/completed", "params": {"threadId": "...", "turn": {...}}}
```

## Initialize

```json
{
  "clientInfo": {"name": "agm", "version": "0.1.0"},
  "capabilities": {}
}
```

## Key methods (v2)

### Session messages

`session.messages` supports pagination using the `limit` and `offset` parameters.

### Thread management

| Method | Params | Description |
|--------|--------|-------------|
| `thread/start` | model, cwd, approvalPolicy, sandbox, ephemeral, developerInstructions, personality | Start a new thread |
| `thread/list` | cursor, archived, sortKey, searchTerm | List all threads (search by title) |
| `thread/read` | threadId, includeTurns | Read thread content |
| `thread/resume` | threadId, model, developerInstructions | Resume an existing thread (with optional overrides) |
| `thread/fork` | threadId, + config overrides | Fork a thread with new settings |
| `thread/archive` | threadId | Archive a thread |
| `thread/unarchive` | threadId | Unarchive a thread |
| `thread/name/set` | threadId, name | Set thread name |
| `thread/rollback` | threadId, numTurns | Drop N turns (does NOT revert files) |
| `thread/compact/start` | threadId | Trigger context compaction (agm calls proactively on rejection cycle 2+) |

### Turn management

| Method | Params | Description |
|--------|--------|-------------|
| `turn/start` | threadId, input, outputSchema, effort, sandboxPolicy, model, personality, summary | Send a message / start a turn |
| `turn/steer` | threadId, expectedTurnId, input | Inject input mid-turn |
| `turn/interrupt` | threadId, turnId | Cancel an active turn |

### Review (built-in)

| Method | Params | Description |
|--------|--------|-------------|
| `review/start` | threadId, target, delivery | Built-in code review |

Review targets: `uncommittedChanges`, `baseBranch` (branch name), `commit` (SHA), `custom` (instructions).
Delivery: `inline` (same thread) or `detached` (new thread, returns `reviewThreadId`).

### Account

| Method | Description |
|--------|-------------|
| `account/read` | Get account info |
| `account/rateLimits/read` | Check rate limits |
| `account/login/start` | Start login flow |
| `account/logout` | Logout |

### Config & models

| Method | Description |
|--------|-------------|
| `config/read` | Read config |
| `model/list` | List available models (agm caches per worker lifecycle) |
| `mcpServerStatus/list` | List MCP server statuses |
| `experimentalFeature/list` | List feature flags with stages |
| `command/exec` | Execute command in sandbox |

## Key notifications (server → client)

| Method | Description |
|--------|-------------|
| `turn/started` | Turn has begun (includes turn ID for steer/interrupt) |
| `turn/completed` | Turn finished (includes turn status and items) |
| `turn/diff/updated` | Aggregated diff across file changes |
| `turn/plan/updated` | Plan steps with status |
| `item/started` | An item started |
| `item/completed` | An item completed |
| `item/agentMessage/delta` | Streaming agent text |
| `item/commandExecution/outputDelta` | Streaming command output |
| `item/fileChange/outputDelta` | Streaming file change |
| `item/reasoning/textDelta` | Reasoning text delta |
| `item/reasoning/summaryTextDelta` | Reasoning summary delta |
| `thread/name/updated` | Thread was renamed |
| `thread/tokenUsage/updated` | Token usage (last + total with cached/reasoning breakdown) |
| `error` | Error with `willRetry` flag, structured `CodexErrorInfo`, and optional `additionalDetails` |
| `model/rerouted` | Model rerouted (fromModel, toModel, reason) |
| `deprecationNotice` | Server warning about deprecated API usage |
| `configWarning` | Server warning about configuration issues |

agm handles `turn/started` (captures turn ID for interrupt), `turn/completed` (extracts reasoning summaries for task_logs), `thread/tokenUsage/updated`, `model/rerouted`, `error`, `deprecationNotice`, and `configWarning` notifications unconditionally. Delta/streaming text notifications (`agentMessage/delta`, `outputDelta`, etc.) are ignored.

### Streaming progress notifications

When a `TurnEventContext` is provided (executor and reviewer turns), agm republishes task-stream events to Redis for agm-web consumption, including:
- `task:turn` (`turn/started`, `turn/completed`)
- `task:item_started`, `task:item_completed`
- `task:plan_updated`
- `task:turn_diff`
- `task:token_usage`
- `task:thread_status`
- `task:model_rerouted`
- `task:backend_error`, `task:backend_warning`
- `task:execution_fallback`
- `task:heartbeat`

All of these events include `extra.thread_context` with:
- `thread_id`
- `thread_status`
- `active_turn_id`
- `last_turn_event_at`
- `owner_role` (executor/reviewer)
- `model`, `provider`
- `has_active_steer`
- `run_id`
- `turn_sequence`

Item events include a minimal summary (item type, command text, file paths, tool name). Plan events include step text and status.

Reasoning summaries (`summary: "concise"`) are enabled on executor and reviewer turns. Summary texts from reasoning items in turn/completed are logged to task_logs for debugging rejections and reviewer decisions.

## AGM API coordination methods

agm also exposes app-server-aware coordination methods for downstream clients:
- `session.post`: append a channel message to a session and emit `session:message`.
- `task.steer`: append a steer message plus optional live `turn/steer` injection against the active turn.
- `task.steers`: list persisted steer audit rows for replay/debug tooling.

## Server requests (server → client, response expected)

| Method | Response shape | Description |
|--------|---------------|-------------|
| `item/commandExecution/requestApproval` | `{"decision": "accept"\|"acceptForSession"\|"decline"\|"cancel"}` | Approve a command |
| `item/fileChange/requestApproval` | `{"decision": "accept"\|"acceptForSession"\|"decline"\|"cancel"}` | Approve a file change |
| `skill/requestApproval` | `{"decision": "approve"\|"decline"}` | Approve skill execution |
| `item/tool/call` | Tool-specific | Execute a dynamic tool (MCP) |
| `item/tool/requestUserInput` | User-input-specific | Request user input |

agm auto-approves `requestApproval` requests with `{"decision": "accept"}`. Unsupported methods receive a JSON-RPC error response.

## Schema validation tests

`tests/test_protocol_schema.py` validates every outgoing message shape against the schema files in `schemas/` and `schemas/v2/`. This catches contract drift (unknown fields, wrong types, missing required fields) at test time. Any protocol change must pass these tests — coverage guards will fail if new methods or handlers are added without corresponding schema tests.

## Turn input format

```json
{
  "threadId": "abc123",
  "input": [{"type": "text", "text": "fix the bug in main.py"}]
}
```

Input types: `text`, `image` (url), `localImage` (path), `skill`, `mention`.

## Sandbox policies (turn/start)

```json
{"type": "dangerFullAccess"}
{"type": "readOnly"}
{"type": "workspaceWrite", "writableRoots": [], "networkAccess": false}
{"type": "externalSandbox", "networkAccess": "restricted"}
```

## Token usage breakdown

```json
{
  "last": {"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0, "reasoningOutputTokens": 0, "totalTokens": 0},
  "total": {"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0, "reasoningOutputTokens": 0, "totalTokens": 0},
  "modelContextWindow": 200000
}
```

## Error info (structured)

Variants: `contextWindowExceeded`, `usageLimitExceeded`, `internalServerError`,
`unauthorized`, `badRequest`, `threadRollbackFailed`, `sandboxError`, `other`.

Error notifications include `willRetry: boolean`.
