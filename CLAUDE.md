# CLAUDE.md

## Project

`agm-cli` is a Python CLI that orchestrates automated software tasks in git repositories.

Core stack:
- Python 3.12+, managed with `uv`
- CLI: `click`
- Storage: SQLite
- Queue: Redis + `rq`
- Quality tooling: `ruff`, `pyright`, `pytest`

## Fast Commands

```bash
uv sync
uv run agm --help
make check
```

Equivalent quality gate:

```bash
uv run ruff check src/ tests/
uv run pyright src/agm/
uv run python -m pytest -m "not slow" -q
```

## Documentation

- `docs/architecture.md` - architecture and module boundaries
- `docs/data-model.md` - data model and lifecycle states
- `docs/pipeline.md` - orchestration flow
- `docs/backends.md` - backend integrations
- `docs/json-contracts.md` - JSON output contracts
- `docs/protocol.md` - app-server protocol details

## Code Map

- `src/agm/cli.py` - CLI command surface
- `src/agm/db.py` - schema and persistence helpers
- `src/agm/jobs*.py` - orchestration/job logic
- `src/agm/queue.py` - queueing and worker interactions
- `src/agm/backends.py` - backend config and model routing
- `src/agm/client.py` - JSON-RPC client
- `src/agm/git_ops.py` - git/worktree operations
- `tests/` - behavior and contract coverage

## Conventions

- Read commands return JSON output; mutation commands are silent on success.
- Keep protocol changes aligned with schema updates under `schemas/` and protocol tests.
- Prefer small, focused edits with matching tests.
- Preserve module boundaries: CLI parses and dispatches, jobs drive lifecycle transitions, DB stores state.

## Change Workflow

1. Inspect related docs and tests before editing.
2. Implement minimal changes.
3. Run `make check`.
4. Update docs when behavior or interfaces change.

