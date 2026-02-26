"""Custom caller management for agm.

Built-in callers are hardcoded. Custom callers are stored in
``~/.config/agm/callers`` (one name per line).
"""

from __future__ import annotations

import re

from agm.paths import AGM_CONFIG_DIR

BUILTIN_CALLERS = frozenset({"cli", "claude-code", "codex-cli", "agm-auto", "aegis"})

_CALLERS_FILE = AGM_CONFIG_DIR / "callers"
_CALLER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,49}$")


def _read_custom_callers() -> set[str]:
    """Read custom caller names from the callers file (one per line)."""
    try:
        text = _CALLERS_FILE.read_text()
    except FileNotFoundError:
        return set()
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def get_all_callers() -> set[str]:
    """Return the union of built-in and custom callers."""
    return set(BUILTIN_CALLERS) | _read_custom_callers()


def add_caller(name: str) -> None:
    """Register a custom caller name.

    Raises ``ValueError`` for invalid names, built-in names, or duplicates.
    """
    if not _CALLER_NAME_RE.match(name):
        raise ValueError(
            f"Invalid caller name '{name}'. Must match [a-z0-9][a-z0-9._-]{{{{0,49}}}}."
        )
    if name in BUILTIN_CALLERS:
        raise ValueError(f"'{name}' is a built-in caller and cannot be added.")
    existing = _read_custom_callers()
    if name in existing:
        raise ValueError(
            f"Caller '{name}' is already registered. Use 'agm caller list' to see all callers."
        )

    _CALLERS_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _CALLERS_FILE.open("a") as f:
        f.write(f"{name}\n")


def remove_caller(name: str) -> None:
    """Unregister a custom caller name.

    Raises ``ValueError`` for built-in names or names not found.
    """
    if name in BUILTIN_CALLERS:
        raise ValueError(f"'{name}' is a built-in caller and cannot be removed.")
    try:
        lines = _CALLERS_FILE.read_text().splitlines()
    except FileNotFoundError:
        raise ValueError(
            f"Caller '{name}' is not registered. Use 'agm caller list' to see registered callers."
        ) from None
    existing = {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}
    if name not in existing:
        raise ValueError(
            f"Caller '{name}' is not registered. Use 'agm caller list' to see registered callers."
        )

    remaining = [line for line in lines if line.strip() != name]
    _CALLERS_FILE.write_text("\n".join(remaining) + "\n" if remaining else "")
