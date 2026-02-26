"""Helpers for loading role-specific agent instructions from TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from string import Template
from typing import Any

SUPPORTED_ROLES = ("enrichment", "explorer", "planner", "task_agent", "executor", "reviewer")

_ROLE_DESCRIPTIONS: dict[str, str] = {
    "planner": "Role instructions for the planner.",
    "explorer": "Role instructions for the explorer.",
    "executor": "Role instructions for the executor.",
    "reviewer": "Role instructions for the reviewer.",
    "task_agent": "Role instructions for the task agent.",
    "enrichment": "Role instructions for the enrichment agent.",
}

_SECTION_TEMPLATE = Template(
    '''# ${description}
[${role}]
instructions = """
${instructions}
"""
'''
)


def _normalize_role(role: str | None) -> str | None:
    """Normalize and validate an agent role name."""
    if role is None:
        return None
    normalized = role.strip().lower()
    if normalized not in SUPPORTED_ROLES:
        return None
    return normalized


def _global_agents_toml() -> Path:
    """Return the global agents.toml path."""
    return Path.home() / ".config" / "agm" / "agents.toml"


def _project_agents_toml(project_dir: str | None) -> Path | None:
    """Return the project-level agents.toml path."""
    if not project_dir:
        return None
    return Path(project_dir) / ".agm" / "agents.toml"


def _read_toml_file(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning an empty dict on any read/parse failure."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _extract_role_text(document: dict[str, Any], role: str) -> str:
    """Return the normalized role instruction text from one TOML document."""
    raw = document.get(role)
    instructions: str
    if isinstance(raw, str):
        instructions = raw
    elif isinstance(raw, dict):
        instructions = raw.get("instructions", "")
        if not isinstance(instructions, str):
            return ""
    else:
        return ""

    value = instructions.strip()
    return value


def _load_agent_instructions(project_dir: str | None, role: str) -> str:
    """Load merged instructions for *role* from global then project agents.toml.

    Supported roles: planner, executor, reviewer, task_agent.
    Unknown roles return an empty string.
    """
    normalized_role = _normalize_role(role)
    if normalized_role is None:
        return ""

    chunks: list[str] = []
    for path in (_global_agents_toml(), _project_agents_toml(project_dir)):
        if path is None:
            continue
        raw = _read_toml_file(path)
        role_text = _extract_role_text(raw, normalized_role)
        if role_text:
            chunks.append(role_text)

    return "\n\n".join(chunks)


def get_effective_role_config(project_dir: str | None, role: str) -> str:
    """CLI-facing helper to read effective role instructions."""
    return _load_agent_instructions(project_dir, role)


def _render_roles(roles_text: dict[str, str]) -> str:
    """Render non-empty roles into TOML text via _SECTION_TEMPLATE."""
    sections = []
    for role in SUPPORTED_ROLES:
        text = roles_text.get(role, "").strip()
        if text:
            sections.append(
                _SECTION_TEMPLATE.substitute(
                    role=role,
                    description=_ROLE_DESCRIPTIONS[role],
                    instructions=text,
                ).strip()
            )
    return "\n\n".join(sections) + "\n" if sections else ""


def _validate_role(role: str) -> str:
    """Normalize and validate a role, raising ValueError on unknown roles."""
    normalized = _normalize_role(role)
    if normalized is None:
        msg = f"Unknown role: {role}. Supported: {', '.join(SUPPORTED_ROLES)}"
        raise ValueError(msg)
    return normalized


def _get_role_at_path(path: Path, role: str) -> str:
    """Read one role's instructions from a TOML file (no merge)."""
    raw = _read_toml_file(path)
    return _extract_role_text(raw, role)


def _set_role_at_path(path: Path, role: str, text: str) -> None:
    """Write instructions for *role* into a TOML file. Creates dirs if needed."""
    existing = _read_toml_file(path)
    roles_text: dict[str, str] = {}
    for r in SUPPORTED_ROLES:
        roles_text[r] = _extract_role_text(existing, r)
    roles_text[role] = text.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_roles(roles_text))


def _reset_role_at_path(path: Path, role: str) -> None:
    """Remove a role from a TOML file. Deletes the file if no roles remain."""
    existing = _read_toml_file(path)
    roles_text: dict[str, str] = {}
    for r in SUPPORTED_ROLES:
        if r != role:
            roles_text[r] = _extract_role_text(existing, r)
    content = _render_roles(roles_text)
    if content:
        path.write_text(content)
    elif path.exists():
        path.unlink()


# -- Project-level CRUD --


def get_project_role_text(project_dir: str, role: str) -> str:
    """Read ONLY the project-level agents.toml for one role (no global merge)."""
    normalized = _normalize_role(role)
    if normalized is None:
        return ""
    path = _project_agents_toml(project_dir)
    if path is None:
        return ""
    return _get_role_at_path(path, normalized)


def set_project_role_instructions(project_dir: str, role: str, text: str) -> None:
    """Write instructions for *role* into the project-level agents.toml."""
    normalized = _validate_role(role)
    path = _project_agents_toml(project_dir)
    if path is None:
        msg = "project_dir is required"
        raise ValueError(msg)
    _set_role_at_path(path, normalized, text)


def reset_project_role_instructions(project_dir: str, role: str) -> None:
    """Remove a role from the project-level agents.toml."""
    normalized = _validate_role(role)
    path = _project_agents_toml(project_dir)
    if path is None:
        msg = "project_dir is required"
        raise ValueError(msg)
    _reset_role_at_path(path, normalized)


# -- Global-level CRUD --


def get_global_role_text(role: str) -> str:
    """Read ONLY the global agents.toml for one role (no project merge)."""
    normalized = _normalize_role(role)
    if normalized is None:
        return ""
    return _get_role_at_path(_global_agents_toml(), normalized)


def set_global_role_instructions(role: str, text: str) -> None:
    """Write instructions for *role* into the global agents.toml."""
    normalized = _validate_role(role)
    _set_role_at_path(_global_agents_toml(), normalized, text)


def reset_global_role_instructions(role: str) -> None:
    """Remove a role from the global agents.toml."""
    normalized = _validate_role(role)
    _reset_role_at_path(_global_agents_toml(), normalized)


def build_agents_toml_scaffold(role: str | None = None) -> str:
    """Return an `agents.toml` scaffold containing role instruction sections.

    No TOML serialization dependency is used; this is plain string-template
    rendering for predictable output.
    """
    normalized_role = _normalize_role(role)
    if role is not None and normalized_role is None:
        return ""
    roles = (normalized_role,) if normalized_role else SUPPORTED_ROLES
    sections = [
        _SECTION_TEMPLATE.substitute(
            role=selected_role,
            description=_ROLE_DESCRIPTIONS[selected_role],
            instructions=f"Add {selected_role} instructions here.",
        )
        for selected_role in roles
    ]
    return "\n".join(section.strip() for section in sections)
