# agm-cli

`agm-cli` is a command-line orchestrator for automated software tasks in git repositories.
This copy is sanitized for public demo use.

## What It Does

- Registers local repositories as projects
- Creates and tracks plans/tasks in SQLite
- Runs queued work via Redis workers
- Applies quality gates before task progression
- Provides JSON-first CLI output for automation

## Fresh Linux Setup

### 1) System dependencies

Install:

1. Python 3.12+
2. `uv`
3. `git`
4. Redis
5. Node.js 18+ (`npx` is used by the MCP sqlite helper)
6. Codex CLI with an authenticated account/subscription

### 2) Clone and install

```bash
git clone <your-agm-cli-repo-url>
cd agm-cli
uv sync
```

### 3) Start Redis

Example (local foreground):

```bash
redis-server
```

Or on systemd-based Linux:

```bash
sudo systemctl enable --now redis-server
```

### 4) Authenticate Codex CLI

```bash
codex login
```

### 5) Verify installation

```bash
uv run agm --help
uv run agm doctor
```

## Runtime Configuration

Common environment variables:

- `AGM_DB_PATH` (default: `~/.config/agm/agm.db`)
- `AGM_REDIS_URL` (default: `redis://localhost:6379/0`)
- `AGM_MODEL_THINK`
- `AGM_MODEL_WORK`
- `AGM_MODEL_WORK_FALLBACK`

## Developer Validation

```bash
make check
```

Equivalent commands:

```bash
uv run ruff check src/ tests/
uv run pyright src/agm/
uv run python -m pytest -m "not slow" -q
```
