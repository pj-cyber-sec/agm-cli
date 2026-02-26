#!/usr/bin/env python3
"""Generate command reference markdown from Click introspection.

Walks the agm CLI command tree and outputs deterministic markdown
suitable for committing as a skill reference file.

Usage:
    uv run python scripts/generate_command_ref.py > skills/agm/references/commands.md
"""

from __future__ import annotations

import click

from agm.cli import main


def _format_choice(param: click.Parameter) -> str:
    """Format a Choice type as inline choices string."""
    if isinstance(param.type, click.Choice):
        return "Choices: " + ", ".join(f"`{c}`" for c in sorted(param.type.choices))
    return ""


def _format_param_description(param: click.Parameter) -> str:
    """Build description cell for a parameter row."""
    parts: list[str] = []

    if param.required:
        parts.append("(required)")

    help_text = getattr(param, "help", None) or ""
    if help_text:
        parts.append(help_text)

    choice_text = _format_choice(param)
    if choice_text and choice_text.lower() not in help_text.lower():
        parts.append(choice_text)

    if getattr(param, "is_flag", False) and not parts:
        parts.append("Flag")

    if param.multiple:
        parts.append("(repeatable)")

    return " ".join(parts)


def _format_option_name(param: click.Parameter) -> str:
    """Format option names, e.g. `-p`, `--project`."""
    opts = getattr(param, "opts", [])
    return ", ".join(f"`{o}`" for o in opts)


def _is_structured_paragraph(para: str) -> bool:
    """Check if a paragraph has line-by-line structure that must be preserved."""
    non_empty = [ln.strip() for ln in para.strip().splitlines() if ln.strip()]
    if len(non_empty) <= 1:
        return False
    # Lines starting with command-like or list-like prefixes
    prefixes = ("agm ", "$ ", "> ", "{", "- ", "* ", "#")
    structured = sum(1 for ln in non_empty if any(ln.startswith(p) for p in prefixes))
    return structured >= 2


def _format_paragraph(para: str) -> list[str]:
    """Format a detail paragraph, preserving structured content."""
    raw = para.strip()
    if not raw:
        return []

    # Click's \b marker means "preserve this paragraph's formatting"
    preserve = "\b" in raw
    clean = raw.replace("\b", "").strip()
    if not clean:
        return []

    if preserve or _is_structured_paragraph(clean):
        # Preserve individual lines (examples, configs, definitions)
        return [ln.strip() for ln in clean.splitlines() if ln.strip()]

    # Regular prose — collapse to single line
    return [" ".join(clean.split())]


def _format_help_text(cmd: click.Command) -> list[str]:
    """Extract full help text: first line as summary, rest as detail block."""
    full_help = (cmd.help or "").strip()
    if not full_help:
        return []

    paragraphs = full_help.split("\n\n")
    lines: list[str] = []

    # First paragraph = summary
    summary = " ".join(paragraphs[0].replace("\b", "").split())
    lines.append(summary)
    lines.append("")

    # Remaining paragraphs = detail (if any)
    if len(paragraphs) > 1:
        for para in paragraphs[1:]:
            formatted = _format_paragraph(para)
            if formatted:
                lines.extend(formatted)
                lines.append("")

    return lines


def _format_command(cmd: click.Command, prefix: str) -> list[str]:
    """Format a single command as markdown lines."""
    lines: list[str] = []

    # Build heading with arguments inline
    args = [p for p in cmd.params if isinstance(p, click.Argument)]
    opts = [
        p
        for p in cmd.params
        if isinstance(p, click.Option)
        and p.name not in ("help",)
        and not p.hidden
    ]

    heading_parts = [prefix, cmd.name or ""]
    for arg in args:
        name = (arg.name or "").upper()
        if not arg.required:
            name = f"[{name}]"
        heading_parts.append(name)

    lines.append(f"### {' '.join(heading_parts)}")
    lines.append("")

    # Full help text
    lines.extend(_format_help_text(cmd))

    # Options table
    if opts:
        lines.append("| Option | Description |")
        lines.append("|--------|-------------|")
        for opt in sorted(opts, key=lambda o: (o.opts or [""])[0]):
            name = _format_option_name(opt)
            desc = _format_param_description(opt)
            lines.append(f"| {name} | {desc} |")
        lines.append("")

    return lines


def _generate_group(
    group: click.Group, prefix: str, heading_level: int = 2
) -> list[str]:
    """Generate markdown for all commands in a group."""
    lines: list[str] = []
    hashes = "#" * heading_level
    lines.append(f"{hashes} {group.name or 'agm'}")
    lines.append("")

    cmds = sorted(group.list_commands(None))  # type: ignore[arg-type]
    for cmd_name in cmds:
        cmd = group.get_command(None, cmd_name)  # type: ignore[arg-type]
        if cmd is None:
            continue
        if isinstance(cmd, click.Group):
            lines.extend(_generate_group(cmd, f"{prefix} {cmd_name}", heading_level + 1))
        else:
            lines.extend(_format_command(cmd, prefix))

    return lines


def generate() -> str:
    """Generate the full command reference."""
    lines: list[str] = [
        "# agm Command Reference",
        "",
        "Auto-generated from Click command tree. Do not edit manually.",
        "Regenerate: `uv run python scripts/generate_command_ref.py`",
        "",
        "## Pipeline flow",
        "",
        "`plan request` → enrichment → planning → task creation"
        " → `task run` → executor → quality gate"
        " → `task review` → reviewer → `task approve` → `task merge`",
        "",
        "Key commands at each stage:",
        "- Start: `agm plan request -p PROJECT \"prompt\"`",
        "- Monitor: `agm plan watch PLAN_ID` / `agm task watch`",
        "- Check: `agm plan show PLAN_ID` / `agm task show TASK_ID`",
        "- Approve plan: `agm plan approve PLAN_ID` (only if manual approval)",
        "- Merge: `agm task merge TASK_ID` (usually automatic)",
        "- Quick single-task: `agm do -p PROJECT \"prompt\"` (skips planning)",
        "",
    ]

    # Collect top-level commands and groups
    top_cmds: list[click.Command] = []
    groups: list[click.Group] = []

    for name in sorted(main.list_commands(None)):  # type: ignore[arg-type]
        cmd = main.get_command(None, name)  # type: ignore[arg-type]
        if cmd is None:
            continue
        if isinstance(cmd, click.Group):
            groups.append(cmd)
        else:
            top_cmds.append(cmd)

    # Top-level commands first
    if top_cmds:
        lines.append("## Top-level commands")
        lines.append("")
        for cmd in top_cmds:
            lines.extend(_format_command(cmd, "agm"))

    # Groups in alphabetical order (they're already sorted)
    for group in groups:
        lines.extend(_generate_group(group, f"agm {group.name}"))

    # Ensure single trailing newline
    text = "\n".join(lines)
    return text.rstrip() + "\n"


if __name__ == "__main__":
    print(generate(), end="")
