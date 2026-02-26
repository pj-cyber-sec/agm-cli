# Contributing

## Local Setup

```bash
git clone <your-agm-cli-repo-url>
cd agm-cli
uv sync
```

Prerequisites before running feature work:

1. Redis available at `AGM_REDIS_URL` (default `redis://localhost:6379/0`)
2. Codex CLI installed and logged in (`codex login`)
3. Python 3.12+ and `uv`
4. Node.js 18+ (`npx` available)

## Quality Gate Before Commit

Run all checks:

```bash
make check
```

Equivalent commands:

```bash
uv run ruff check src/ tests/
uv run pyright src/agm/
uv run python -m pytest -m "not slow" -q
```

## Change Rules

- Keep changes focused to one intent.
- Update tests when behavior changes.
- Keep command output JSON-compatible where applicable.
