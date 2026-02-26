"""Canonical filesystem paths for agm configuration and state."""

from __future__ import annotations

import os
from pathlib import Path

AGM_CONFIG_DIR = Path.home() / ".config" / "agm"

CODEX_HOME = AGM_CONFIG_DIR / ".codex"

_env_db = os.environ.get("AGM_DB_PATH")
DEFAULT_DB_PATH = Path(_env_db).expanduser() if _env_db else AGM_CONFIG_DIR / "agm.db"
