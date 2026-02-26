# AGENTS.md

Agent guidance for this repository. This file is intentionally aligned with `CLAUDE.md`.

## Goal

Make minimal, correct changes with passing checks and clear JSON-compatible behavior.

## Setup

```bash
uv sync
```

## Required Validation

```bash
make check
```

## Working Rules

- Review relevant code and docs before editing.
- Keep changes scoped to the request.
- Add/update tests when behavior changes.
- Update `docs/` when architecture, commands, or contracts change.
- Avoid broad refactors unless explicitly requested.

## High-Value References

- `CLAUDE.md`
- `docs/architecture.md`
- `docs/data-model.md`
- `docs/pipeline.md`
- `docs/json-contracts.md`
- `docs/protocol.md`

