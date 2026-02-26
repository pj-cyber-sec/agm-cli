from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import click

from agm import __version__
from agm.agents_config import (
    SUPPORTED_ROLES,
    _global_agents_toml,
    build_agents_toml_scaffold,
    get_effective_role_config,
    get_global_role_text,
    get_project_role_text,
    reset_global_role_instructions,
    reset_project_role_instructions,
    set_global_role_instructions,
    set_project_role_instructions,
)
from agm.backends import (
    MODEL_CATALOG,
    VALID_EFFORTS,
    _get_default_effort_for_backend_tier,
    _get_default_model_for_backend_tier,
    get_live_models,
    is_model_available,
    resolve_model_config,
)
from agm.callers import (
    BUILTIN_CALLERS,
    add_caller,
    get_all_callers,
    remove_caller,
)
from agm.daemon_threads import normalize_daemon_thread_list
from agm.db import (
    DEFAULT_BACKEND,
    VALID_BACKENDS,
    VALID_MESSAGE_KINDS,
    VALID_PLAN_STATUSES,
    VALID_TASK_PRIORITIES,
    VALID_TASK_STATUSES,
    PlanRow,
    ProjectRow,
    TaskRow,
    add_project,
    add_task_steer,
    answer_plan_question,
    bulk_active_runtime_seconds,
    claim_task,
    clear_task_git_refs,
    connect,
    create_plan_request,
    create_session,
    get_plan_chain,
    get_plan_request,
    get_project,
    get_project_base_branch,
    get_project_model_config,
    get_session,
    get_task,
    get_task_block,
    get_task_rejection_count,
    get_unresolved_block_count,
    list_channel_messages,
    list_plan_logs,
    list_plan_questions,
    list_plan_requests,
    list_plan_timeline_rows,
    list_plan_watch_events,
    list_projects,
    list_recent_task_events,
    list_sessions,
    list_status_history_timing_rows,
    list_task_blocks,
    list_task_cleanup_candidates,
    list_task_logs,
    list_task_steers,
    list_tasks,
    parse_app_server_approval_policy,
    parse_app_server_ask_for_approval,
    purge_data,
    purge_preview_counts,
    reconcile_session_statuses,
    remove_project,
    reset_plan_for_retry,
    reset_task_for_retry,
    resolve_backend,
    resolve_blockers_for_terminal_task,
    resolve_task_block,
    set_plan_session_id,
    set_project_base_branch,
    set_project_model_config,
    set_task_failure_reason,
    set_task_priority,
    set_task_reviewer_thread_id,
    set_task_thread_id,
    update_session_status,
    update_task_status,
)
from agm.queries import (
    PLAN_ACTIVE_STATUSES,
    PLAN_WATCH_RECENT_EVENTS_ROWS,
    TASK_ACTIVE_STATUSES,
    WATCH_RECENT_EVENT_FETCH_LIMIT,
    build_plan_stats_data,
    build_plan_watch_snapshot,
    build_task_watch_snapshot,
    effective_task_priority,
    enrich_plan_list_rows,
    enrich_task_list_rows,
    format_duration_seconds,
    format_elapsed,
    format_plan_failure_error,
    format_plan_failure_prompt,
    gather_project_summaries,
    is_effectively_terminal_task,
    latest_task_thread_statuses,
    model_usage_counts,
    normalize_logs,
    normalize_plan_chain,
    normalize_plan_questions,
    normalize_task_blocks,
    normalize_timeline_rows,
    plan_failure_diagnostic,
    plan_watch_terminal_state,
    project_token_totals,
    resolve_project_names_for_tasks,
    status_counts,
    task_failure_diagnostic,
    task_list_filter_rows,
    task_merge_failure_signal,
    task_watch_terminal_state,
    watch_short_id,
    watch_truncate,
)
from agm.status_reference import get_status_reference
from agm.steering import default_executor_recipient, steer_active_turn

log = logging.getLogger(__name__)


class CallerType(click.ParamType):
    """Click parameter type that validates against all registered callers."""

    name = "caller"

    def get_metavar(self, param: click.Parameter, **kwargs: object) -> str:
        return "CALLER"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> str:
        all_callers = get_all_callers()
        if value not in all_callers:
            self.fail(
                f"'{value}' is not a registered caller. Valid: {', '.join(sorted(all_callers))}",
                param,
                ctx,
            )
        return value

    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list:
        from click.shell_completion import CompletionItem

        return [CompletionItem(c) for c in sorted(get_all_callers()) if c.startswith(incomplete)]


CALLER_TYPE = CallerType()

MAX_PROMPT_LENGTH = 100_000  # 100k chars — reasonable upper bound for LLM prompts
PROJECT_SETUP_WAIT_TIMEOUT_SECONDS = 1800.0


def _validate_prompt(ctx: click.Context, param: click.Parameter, value: str) -> str:
    if len(value) > MAX_PROMPT_LENGTH:
        raise click.BadParameter(
            f"Prompt is {len(value):,} chars (max {MAX_PROMPT_LENGTH:,}).",
            ctx=ctx,
            param=param,
        )
    return value


class _JsonAwareGroup(click.Group):
    """Group that always outputs JSON errors with command suggestions.

    Click normally writes plain-text usage errors to stderr.  Since all
    commands now output JSON unconditionally, this subclass intercepts Click
    exceptions and emits a JSON error object on stdout.  Unknown commands
    get fuzzy-matched suggestions via ``difflib.get_close_matches``.
    """

    def resolve_command(self, ctx, args):  # type: ignore[override]
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args:
                import difflib

                cmd_name = args[0]
                matches = difflib.get_close_matches(
                    cmd_name, self.list_commands(ctx), n=2, cutoff=0.5
                )
                hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
                raise click.UsageError(f"No such command '{cmd_name}'.{hint}") from None
            raise

    def main(self, args=None, standalone_mode=True, **kwargs):  # type: ignore[override]
        try:
            rv = super().main(args=args, standalone_mode=False, **kwargs)
            if standalone_mode:
                raise SystemExit(rv or 0)
            return rv
        except click.ClickException as e:
            click.echo(json.dumps({"ok": False, "error": e.format_message()}))
            code = getattr(e, "exit_code", 1)
            if standalone_mode:
                raise SystemExit(code) from None
            return code
        except click.Abort:
            if standalone_mode:
                click.echo("Aborted!", err=True)
                raise SystemExit(1) from None
            raise


@click.group(cls=_JsonAwareGroup)
@click.version_option(version=__version__)
def main():
    """Orchestrate AI agents through an automated plan-execute-review-merge pipeline.

    \b
    Quick start:
      agm init                                    Register current directory as a project
      agm doctor                                  Check prerequisites (Redis, backends)
      agm do "Fix the login bug" -p PROJECT       One-shot: execute, review, merge
      agm plan request "Add OAuth" -p PROJECT     Full pipeline: plan, then execute tasks
      agm plan watch PLAN_ID                      Monitor progress (live TUI)

    \b
    Key concepts:
      project   A git repo registered with agm
      plan      A decomposition of work into tasks (created by an AI planner)
      task      A single unit of work executed in an isolated git worktree
      doctor    Health check for Redis, backends, stale jobs
    """


# -- init --


@main.command()
@click.argument(
    "path",
    required=False,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
@click.option("--name", "-n", default=None, help="Project name (default: directory name).")
@click.option(
    "--dir",
    "-d",
    "directory",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Project directory (default: current directory).",
)
def init(
    path: str | None,
    name: str | None,
    directory: str | None,
):
    """Set up agm in a git repo.

    Works on existing repos and empty directories. If the directory
    is not a git repo, agm will initialize one.
    """
    if path and directory:
        raise click.ClickException("Use either PATH or --dir, not both.")

    resolved_target = directory if directory is not None else path
    target = Path(resolved_target) if resolved_target is not None else Path.cwd()
    project_name = name or target.name

    with connect() as conn:
        from agm.db import get_project_by_dir

        existing_by_name = get_project(conn, project_name)
        existing_by_dir = get_project_by_dir(conn, str(target))
        if existing_by_name and not existing_by_dir and existing_by_name.get("dir") != str(target):
            raise click.ClickException(
                f"Project name '{project_name}' is already registered for "
                f"'{existing_by_name['dir']}'. Use --name to choose a different name."
            )

    _init_git_repo(target, quiet=True)

    with connect() as conn:
        from agm.db import get_project_by_dir

        # Check both name and directory to prevent duplicates
        existing_by_name = get_project(conn, project_name)
        existing_by_dir = get_project_by_dir(conn, str(target))

        if existing_by_dir:
            project_name = existing_by_dir["name"]
        elif not existing_by_name:
            add_project(conn, project_name, str(target))

    _ensure_gitignore_entry(target, quiet=True)
    codex_warnings = _init_codex_check(quiet=True)

    with connect() as conn:
        proj = get_project(conn, project_name)
    payload = dict(proj) if proj else {}
    if codex_warnings:
        payload["warnings"] = codex_warnings
    click.echo(json.dumps(payload))


def _init_git_repo(cwd: Path, *, quiet: bool = False) -> None:
    """Ensure cwd is a git repo with at least one commit."""
    git_dir = cwd / ".git"
    if not git_dir.exists():
        try:
            subprocess.run(
                ["git", "init"], cwd=str(cwd), check=True, capture_output=True, text=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise click.ClickException(f"Failed to initialize git repo: {exc}") from exc

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True
        )
        has_commits = result.returncode == 0
    except FileNotFoundError:
        raise click.ClickException("git not found on PATH") from None

    if not has_commits:
        gitignore = cwd / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".agm/\n")
        try:
            subprocess.run(
                ["git", "add", ".gitignore"],
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise click.ClickException("git not found on PATH") from None
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            suffix = f": {detail}" if detail else ""
            raise click.ClickException(f"Failed to create initial commit{suffix}") from exc


def _init_codex_check(*, quiet: bool = False) -> list[str]:
    """Check that Codex CLI is installed and authenticated during init."""
    installed, authed, _detail = _check_backend_auth("codex")
    if installed and authed:
        return []

    if not installed:
        return [
            "Codex CLI not found. Install Codex CLI and run `codex login` before running agents."
        ]
    return ["Codex CLI is not authenticated. Run `codex login` before running agents."]


def _backend_version_text(cmd: str) -> str | None:
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, timeout=10, text=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        return cmd
    return output.splitlines()[0]


def _codex_auth_detail(version_text: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True,
            timeout=10,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, f"{version_text} (auth check failed)"
    if result.returncode == 0:
        return True, f"{version_text} (authenticated)"
    return False, f"{version_text} (not logged in — run `codex login`)"


def _check_backend_auth(cmd: str) -> tuple[bool, bool, str]:
    """Check if a backend CLI is installed and authenticated.

    Returns (installed, authenticated, detail_message).
    """
    version_text = _backend_version_text(cmd)
    if version_text is None:
        return False, False, "not found"
    if cmd == "codex":
        authenticated, detail = _codex_auth_detail(version_text)
        return True, authenticated, detail
    return True, False, f"{version_text} (unknown backend)"


def _ensure_gitignore_entry(cwd: Path, *, quiet: bool = False):
    """Add .agm/ to .gitignore if not already present."""
    gitignore = cwd / ".gitignore"
    agm_entry = ".agm/"
    if gitignore.exists():
        content = gitignore.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == agm_entry or stripped == ".agm":
                return
        # Append to existing file
        with open(gitignore, "a") as f:
            if not content.endswith("\n"):
                f.write("\n")
            f.write(f"{agm_entry}\n")
    else:
        gitignore.write_text(f"{agm_entry}\n")


def _project_agents_toml_path(project_dir: str) -> Path:
    return Path(project_dir) / ".agm" / "agents.toml"


def _agents_project_match(
    project: ProjectRow,
    resolved_cwd: Path,
) -> tuple[int, str, str, ProjectRow] | None:
    raw_project_dir = project.get("dir")
    if not isinstance(raw_project_dir, str) or not raw_project_dir:
        return None
    try:
        project_dir = Path(raw_project_dir).resolve()
    except OSError:
        return None
    try:
        if resolved_cwd != project_dir and not resolved_cwd.is_relative_to(project_dir):
            return None
    except ValueError:
        return None
    return (
        len(project_dir.parts),
        str(project_dir).lower(),
        str(project.get("name") or ""),
        project,
    )


def _resolve_agents_project(cwd: Path | None = None) -> ProjectRow | None:
    resolved_cwd = (cwd or Path.cwd()).resolve()
    with connect() as conn:
        projects = list_projects(conn)

    matches: list[tuple[int, str, str, ProjectRow]] = []
    for project in projects:
        match = _agents_project_match(project, resolved_cwd)
        if match is not None:
            matches.append(match)

    if not matches:
        return None

    matches.sort(key=lambda row: (-row[0], row[1], row[2]))
    return matches[0][3]


def _resolve_agents_context(global_scope: bool) -> tuple[ProjectRow | None, Path]:
    project = None
    if not global_scope:
        project = _resolve_agents_project()
    target = (
        _global_agents_toml()
        if global_scope or project is None
        else _project_agents_toml_path(str(project["dir"]))
    )
    return project, target


def _launch_editor(file_path: Path) -> None:
    editor = os.environ.get("EDITOR", "").strip()
    if not editor:
        raise click.ClickException(
            "No EDITOR configured. Set EDITOR (for example 'vim' or 'code --wait')."
        )

    command = shlex.split(editor)
    if not command:
        raise click.ClickException(
            "EDITOR is not a valid command. Set EDITOR (for example 'vim' or 'code --wait')."
        )

    try:
        subprocess.run([*command, str(file_path)], check=True)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Failed to launch editor '{command[0]}'. Is EDITOR set correctly?"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Editor '{editor}' exited with status {exc.returncode}."
        ) from exc


def _emit_agents_json(project: ProjectRow | None) -> None:
    project_dir = str(project["dir"]) if project and project.get("dir") else None
    roles: dict[str, dict[str, str]] = {}
    for role in SUPPORTED_ROLES:
        global_text = get_global_role_text(role)
        project_text = get_project_role_text(project_dir, role) if project_dir else ""
        effective = get_effective_role_config(project_dir, role)
        roles[role] = {"global": global_text, "project": project_text, "effective": effective}
    click.echo(json.dumps({"roles": roles}, indent=2))


def _not_found(entity: str, identifier: str) -> click.ClickException:
    """Build a ClickException with an actionable suggestion for missing entities."""
    hints = {
        "project": "Run 'agm project list' to see registered projects.",
        "plan": "Run 'agm plan list -p PROJECT' to see plans.",
        "task": "Run 'agm task list -p PROJECT' to see tasks.",
        "question": "Run 'agm plan questions PLAN_ID' to see questions.",
        "block": "Run 'agm task blocks TASK_ID' to see blockers.",
    }
    msg = f"{entity.title()} '{identifier}' not found."
    hint = hints.get(entity)
    if hint:
        msg += f"\n{hint}"
    return click.ClickException(msg)


def _resolve_project_by_name(name: str) -> ProjectRow:
    with connect() as conn:
        proj = get_project(conn, name)
    if not proj:
        raise _not_found("project", name)
    return proj


def _resolve_project_id(
    project_name: str | None,
) -> str | None:
    if not project_name:
        return None
    return _resolve_project_by_name(project_name)["id"]


def _validate_role(role: str) -> str:
    """Normalize and validate a role name. Raises ClickException on invalid."""
    normalized = role.strip().lower()
    if normalized not in SUPPORTED_ROLES:
        raise click.ClickException(f"Unknown role: {role}. Supported: {', '.join(SUPPORTED_ROLES)}")
    return normalized


def _agents_set_role(role: str, global_scope: bool, project: ProjectRow | None, text: str) -> None:
    """Handle --set: write or reset role instructions from stdin text."""
    role_normalized = _validate_role(role)
    if global_scope:
        if not text.strip():
            reset_global_role_instructions(role_normalized)
        else:
            set_global_role_instructions(role_normalized, text)
    else:
        if not project:
            raise click.ClickException("No project found. Use -p PROJECT to specify one.")
        if not text.strip():
            reset_project_role_instructions(str(project["dir"]), role_normalized)
        else:
            set_project_role_instructions(str(project["dir"]), role_normalized, text)


def _agents_reset_role(role: str, global_scope: bool, project: ProjectRow | None) -> None:
    """Handle --reset: clear role instructions."""
    role_normalized = _validate_role(role)
    if global_scope:
        reset_global_role_instructions(role_normalized)
    else:
        if not project:
            raise click.ClickException("No project found. Use -p PROJECT to specify one.")
        reset_project_role_instructions(str(project["dir"]), role_normalized)


def _resolve_agents_project_for_command(
    global_scope: bool, project_name: str | None
) -> ProjectRow | None:
    """Resolve project context for agents subcommands."""
    if project_name:
        return _resolve_project_by_name(project_name)
    if not global_scope:
        return _resolve_agents_project()
    return None


def _resolve_agents_target_path(
    global_scope: bool,
    project_name: str | None,
    project: ProjectRow | None,
) -> Path:
    """Resolve target agents.toml path for init/edit flows."""
    if project_name and project:
        if global_scope:
            return _global_agents_toml()
        return _project_agents_toml_path(str(project["dir"]))
    _, target = _resolve_agents_context(global_scope)
    return target


_agents_shared_options = [
    click.option("--global", "global_scope", is_flag=True, help="Use the global agents.toml path."),
    click.option("-p", "--project", "project_name", default=None, help="Target project by name."),
]


def _apply_agents_options(fn):  # type: ignore[no-untyped-def]
    for decorator in reversed(_agents_shared_options):
        fn = decorator(fn)
    return fn


@main.group(invoke_without_command=True)
@click.pass_context
def agents(ctx: click.Context) -> None:
    """Manage per-role agent instruction templates."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@agents.command()
@_apply_agents_options
def show(global_scope: bool, project_name: str | None) -> None:
    """Show effective agent instructions."""
    project = _resolve_agents_project_for_command(global_scope, project_name)
    _emit_agents_json(project)


@agents.command("init")
@_apply_agents_options
def agents_init(global_scope: bool, project_name: str | None) -> None:
    """Create a scaffolded agents.toml."""
    project = _resolve_agents_project_for_command(global_scope, project_name)
    target = _resolve_agents_target_path(global_scope, project_name, project)
    if target.exists():
        raise click.ClickException(
            f"agents.toml already exists at '{target}'. Use 'agents edit' instead."
        )
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(build_agents_toml_scaffold())


@agents.command()
@_apply_agents_options
def edit(global_scope: bool, project_name: str | None) -> None:
    """Open the target agents.toml in $EDITOR."""
    project = _resolve_agents_project_for_command(global_scope, project_name)
    target = _resolve_agents_target_path(global_scope, project_name, project)
    if not target.exists():
        raise click.ClickException(f"Missing target file: {target}")
    _launch_editor(target)


@agents.command("set")
@_apply_agents_options
@click.argument("role")
def agents_set(global_scope: bool, project_name: str | None, role: str) -> None:
    """Set role instructions from stdin (e.g. agm agents set executor)."""
    import sys

    project = _resolve_agents_project_for_command(global_scope, project_name)
    _agents_set_role(role, global_scope, project, sys.stdin.read())


@agents.command()
@_apply_agents_options
@click.argument("role")
def reset(global_scope: bool, project_name: str | None, role: str) -> None:
    """Clear role instructions (e.g. agm agents reset reviewer)."""
    project = _resolve_agents_project_for_command(global_scope, project_name)
    _agents_reset_role(role, global_scope, project)


# -- status --


@main.command()
def status():
    """Show overview of active plans, tasks, and queue health."""
    import json as json_mod

    from agm.backends import resolve_model_config
    from agm.queue import (
        get_active_external_jobs,
        get_codex_rate_limits_safe,
        get_queue_counts_safe,
    )

    with connect() as conn:
        project_summaries = gather_project_summaries(conn)

    queue_info = get_queue_counts_safe()
    rate_limits = get_codex_rate_limits_safe()
    external_jobs = get_active_external_jobs()

    # Resolve defaults (no project config = pure defaults + env)
    codex_cfg = resolve_model_config("codex", None)
    models = {
        "codex": {"think": codex_cfg["think_model"], "work": codex_cfg["work_model"]},
    }

    click.echo(
        json_mod.dumps(
            {
                "models": models,
                "projects": project_summaries,
                "queue": queue_info,
                "codex_rate_limits": rate_limits,
                "external_jobs": external_jobs,
            },
            indent=2,
            default=str,
        )
    )


@main.command("help-status")
def help_status():
    """Show canonical status lifecycle definitions for plans, tasks, and task creation."""
    import json as json_mod

    payload = get_status_reference()
    click.echo(json_mod.dumps(payload, indent=2, sort_keys=False))


def _resolve_project_base_branch(conn, project_id: str) -> str:
    """Resolve the effective base branch for a project."""
    return get_project_base_branch(conn, project_id)


_MODEL_CONFIG_KEYS = ("think_model", "work_model", "think_effort", "work_effort")
_EFFECTIVE_MODEL_TIERS = {
    "think_model": "think",
    "work_model": "work",
    "think_effort": "think",
    "work_effort": "work",
}


def _normalize_model_effort(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    effort = value.strip().lower()
    return effort if effort in VALID_EFFORTS else None


def _normalize_model_config_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_env_model_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _decode_model_config(
    raw_config: str | None, *, strict: bool
) -> tuple[dict[str, object], str | None]:
    if raw_config is None:
        return {}, None
    if not isinstance(raw_config, str):
        return {}, "Stored model_config is invalid: expected a JSON string."
    if strict:
        config = json.loads(raw_config)
    else:
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            return {}, f"Stored model_config is invalid JSON: {exc}"
    if not isinstance(config, dict):
        if strict:
            raise ValueError("Model config must be a JSON object.")
        return {}, "Stored model_config must be a JSON object."
    return dict(config), None


def _validate_model_config_keys(config: dict[str, object], *, strict: bool) -> str | None:
    unknown_keys = sorted(key for key in config if key not in _MODEL_CONFIG_KEYS)
    if not unknown_keys:
        return None
    if strict:
        raise ValueError(
            "Model config keys must be one of: think_model, work_model, think_effort, work_effort."
        )
    return f"Stored model_config has unsupported keys: {', '.join(unknown_keys)}"


def _parse_model_config_entry(
    key: str, value: object, *, strict: bool
) -> tuple[str | None, str | None]:
    if key in {"think_model", "work_model"}:
        normalized_model = _normalize_model_config_value(value)
        if normalized_model is not None:
            return normalized_model, None
        if strict:
            raise ValueError(f"'{key}' must be a non-empty string.")
        return None, f"Stored model_config.{key} must be a non-empty string."
    normalized_effort = _normalize_model_effort(value)
    if normalized_effort is not None:
        return normalized_effort, None
    if strict:
        raise ValueError(f"'{key}' must be one of: low, medium, high.")
    return None, f"Stored model_config.{key} must be one of: low, medium, high."


def _parse_model_config_payload(
    raw_config: str | None, *, strict: bool = False
) -> tuple[dict[str, str], list[str]]:
    config, decode_error = _decode_model_config(raw_config, strict=strict)
    if decode_error:
        return {}, [decode_error]
    if not config:
        return {}, []
    key_error = _validate_model_config_keys(config, strict=strict)
    if key_error:
        return {}, [key_error]
    parsed: dict[str, str] = {}
    for key, value in config.items():
        normalized, value_error = _parse_model_config_entry(key, value, strict=strict)
        if value_error:
            return {}, [value_error]
        if normalized is None:
            continue
        parsed[key] = normalized
    return parsed, []


def _resolve_model_env_overrides(backend: str) -> dict[str, str | None]:
    backend_key = backend.strip().lower()
    if backend_key == "codex":
        return {
            "think_model": _normalize_env_model_value(os.environ.get("AGM_MODEL_THINK")),
            "work_model": _normalize_env_model_value(os.environ.get("AGM_MODEL_WORK")),
            "think_effort": _normalize_env_model_value(os.environ.get("AGM_MODEL_THINK_EFFORT")),
            "work_effort": _normalize_env_model_value(os.environ.get("AGM_MODEL_WORK_EFFORT")),
        }
    return {}


def _project_default_model_config(backend: str) -> dict[str, str]:
    think_model = _get_default_model_for_backend_tier(backend, "think")
    work_model = _get_default_model_for_backend_tier(backend, "work")
    return {
        "think_model": think_model,
        "work_model": work_model,
        "think_effort": _get_default_effort_for_backend_tier(backend, "think", think_model),
        "work_effort": _get_default_effort_for_backend_tier(backend, "work", work_model),
    }


def _model_recommendation_warning(model_id: str, backend: str, tier: str) -> str | None:
    spec = MODEL_CATALOG.get(model_id)
    if not isinstance(spec, dict):
        available = is_model_available(model_id)
        if available is False:
            return f"Model '{model_id}' is not available on the server."
        if available is True:
            return (
                f"Model '{model_id}' is available but has no catalog metadata"
                " (no tier/effort defaults)."
            )
        return f"Model '{model_id}' is unknown to catalog metadata."

    model_backend = str(spec.get("backend", "")).lower()
    if model_backend and model_backend != backend:
        return (
            f"Model '{model_id}' is cataloged for backend '{model_backend}' but project backend is "
            f"'{backend}'."
        )

    model_tier = str(spec.get("tier", ""))
    if model_tier not in {tier, "both"}:
        return (
            f"Model '{model_id}' is not intended for the {tier} tier "
            f"(metadata tier is '{model_tier}')."
        )

    recommendation = spec.get("recommendation")
    if not isinstance(recommendation, dict) or not recommendation.get("default"):
        note = ""
        if isinstance(recommendation, dict):
            reason = recommendation.get("reason")
            if isinstance(reason, str) and reason.strip():
                note = f" ({reason.strip()})"
        return f"Model '{model_id}' is not a recommended {tier} model{note}."

    return None


def _model_config_warnings(backend: str, resolved: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for key in ("think_model", "work_model"):
        warning = _model_recommendation_warning(
            resolved.get(key, ""), backend, _EFFECTIVE_MODEL_TIERS[key]
        )
        if warning:
            warnings.append(warning)
    return warnings


def _set_model_config_warnings(backend: str, model_config: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for key in ("think_model", "work_model"):
        model = model_config.get(key)
        if not model:
            continue
        spec = MODEL_CATALOG.get(model)
        if not isinstance(spec, dict):
            available = is_model_available(model)
            if available is False:
                warnings.append(f"Model '{model}' is not available on the server.")
            elif available is True:
                warnings.append(
                    f"Model '{model}' is available but has no catalog metadata "
                    "(no tier/effort defaults)."
                )
            else:
                warnings.append(f"Model '{model}' is unknown to catalog metadata.")
            continue
        model_backend = str(spec.get("backend", ""))
        if model_backend and model_backend != backend:
            warnings.append(
                f"{key} model '{model}' is for backend '{model_backend}' while "
                f"project backend is '{backend}'."
            )
    return warnings


def _show_model_presets_json() -> None:
    """Output the model catalog as JSON."""
    live = get_live_models()
    static_ids = set(MODEL_CATALOG.keys())
    live_by_id: dict[str, dict] = {}
    if live is not None:
        live_by_id = {m["id"]: m for m in live if m.get("id")}

    catalog: list[dict] = []
    for model_id, spec in MODEL_CATALOG.items():
        entry: dict = {"id": model_id, **spec}
        live_model = live_by_id.get(model_id)
        if live_model:
            entry["live"] = True
        catalog.append(entry)

    live_only = [{"id": mid, **mdata} for mid, mdata in live_by_id.items() if mid not in static_ids]

    click.echo(
        json.dumps(
            {"catalog": catalog, "live_only": live_only, "live_available": live is not None},
            indent=2,
            default=str,
        )
    )


def _model_config_display_lines(
    project: ProjectRow, project_model_config: dict[str, str]
) -> list[str]:
    backend = project.get("default_backend") or DEFAULT_BACKEND
    env_values = _resolve_model_env_overrides(backend)
    defaults = _project_default_model_config(backend)
    env_source = {k: v for k, v in env_values.items() if v is not None}
    resolved = resolve_model_config(backend, project_model_config)
    warnings = _model_config_warnings(backend, resolved)

    lines = [
        "  model_config:",
        f"    configured: {json.dumps(project_model_config, sort_keys=True) or '(empty)'}",
        f"    backend: {backend}",
        "    active:",
        f"      think: {resolved['think_model']} (effort={resolved['think_effort']})",
        f"      work: {resolved['work_model']} (effort={resolved['work_effort']})",
        "    sources:",
        f"      env: {json.dumps(env_source, sort_keys=True)}",
        f"      default: {json.dumps(defaults, sort_keys=True)}",
    ]
    for warning in warnings:
        lines.append(f"    warning: {warning}")
    return lines


def _coerce_model_json_text(json_text: str) -> dict[str, str]:
    parsed, _ = _parse_model_config_payload(json_text, strict=True)
    return parsed


def _model_config_payload(project: ProjectRow, project_model_config: dict[str, str]) -> dict:
    backend = project.get("default_backend") or DEFAULT_BACKEND
    env_values = _resolve_model_env_overrides(backend)
    defaults = _project_default_model_config(backend)
    resolved = resolve_model_config(backend, project_model_config)
    return {
        "project_backend": backend,
        "configured": project_model_config,
        "active": {
            "think_model": resolved["think_model"],
            "think_effort": resolved["think_effort"],
            "work_model": resolved["work_model"],
            "work_effort": resolved["work_effort"],
        },
        "sources": {
            "env": {k: v for k, v in env_values.items() if v is not None},
            "default": defaults,
        },
        "warnings": _model_config_warnings(backend, resolved),
    }


def _parse_json_column(value: object | None) -> Any:
    """Parse a JSON-string column for CLI output."""
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _doctor_has_auto_fixable_warnings(report: Any) -> bool:
    """Return whether report contains stale PID/worktree warnings."""
    return any(
        check["name"] in {"stale_pids", "worktrees"} and check["status"] != "pass"
        for check in report["checks"]
    )


@main.command()
@click.option("--fix", is_flag=True, help="Auto-fix stale PIDs and orphaned worktrees.")
def doctor(fix: bool):
    """Run health checks on agm infrastructure."""
    from agm.doctor import run_doctor

    report = run_doctor(fix=fix)
    click.echo(json.dumps(report))
    if report["status"] == "fail":
        raise click.ClickException("Doctor checks failed.")


# -- conflicts --


def _resolve_conflict_projects(project_name: str | None):
    with connect() as conn:
        if project_name:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)
            return [proj]

        projects = list_projects(conn)
        if not projects:
            raise click.ClickException(
                "No projects registered. Run 'agm project add NAME DIR' to register one."
            )
        return projects


def _emit_conflicts_for_project(proj, result) -> None:
    import json as json_mod

    click.echo(json_mod.dumps(result, indent=2))


@main.command()
@click.option("--project", "-p", "project_name", default=None, help="Project name or ID.")
def conflict(project_name: str | None):
    """Detect merge conflicts between active worktrees (requires clash)."""
    from agm.git_ops import detect_worktree_conflicts

    projects = _resolve_conflict_projects(project_name)
    for proj in projects:
        result = detect_worktree_conflicts(proj["dir"])
        _emit_conflicts_for_project(proj, result)


# -- project --


@main.group()
def project():
    """Register, configure, and inspect projects."""


@project.command("add")
@click.argument("name")
@click.option("--dir", "-d", "directory", required=True, type=click.Path(exists=True))
def project_add(name: str, directory: str):
    """Register a project."""
    directory = str(Path(directory).resolve())
    with connect() as conn:
        try:
            add_project(conn, name, directory)
        except Exception as e:
            raise click.ClickException(str(e)) from e


@project.command("list")
def project_list():
    """List all projects."""
    import json

    with connect() as conn:
        projects = []
        for p in list_projects(conn):
            item = dict(p)
            item["app_server_approval_policy"] = parse_app_server_approval_policy(
                p.get("app_server_approval_policy")
            )
            item["app_server_ask_for_approval"] = parse_app_server_ask_for_approval(
                p.get("app_server_ask_for_approval")
            )
            projects.append(item)
    click.echo(json.dumps(projects, indent=2, default=str))


@project.command("show")
@click.argument("name_or_id")
def project_show(name_or_id: str):
    """Show project details."""
    with connect() as conn:
        p = get_project(conn, name_or_id)
        if p:
            project_model_config, parse_warnings = _parse_model_config_payload(
                get_project_model_config(conn, p["id"])
            )
            model_payload = _model_config_payload(p, project_model_config)
            model_payload["warnings"].extend(parse_warnings)
    if not p:
        raise _not_found("project", name_or_id)
    output = dict(p)
    output["model_config"] = model_payload
    output["quality_gate"] = _parse_json_column(p.get("quality_gate"))
    output["setup_result"] = _parse_json_column(p.get("setup_result"))
    output["app_server_approval_policy"] = parse_app_server_approval_policy(
        p.get("app_server_approval_policy")
    )
    output["app_server_ask_for_approval"] = parse_app_server_ask_for_approval(
        p.get("app_server_ask_for_approval")
    )
    click.echo(json.dumps(output, indent=2, default=str))


def _project_or_error(conn, name_or_id: str) -> ProjectRow:
    proj = get_project(conn, name_or_id)
    if not proj:
        raise _not_found("project", name_or_id)
    return proj


def _show_project_model_config(conn, proj: ProjectRow) -> None:
    raw_project_config = get_project_model_config(conn, proj["id"])
    project_model_config, parse_warnings = _parse_model_config_payload(raw_project_config)
    model_payload = _model_config_payload(proj, project_model_config)
    model_payload["warnings"].extend(parse_warnings)
    click.echo(json.dumps(model_payload, indent=2, default=str))


def _update_project_model_config(conn, proj: ProjectRow, config_json: str) -> None:
    try:
        parsed_config = _coerce_model_json_text(config_json)
    except (ValueError, json.JSONDecodeError) as e:
        raise click.ClickException(f"Invalid model config: {e}") from e
    set_project_model_config(conn, proj["id"], json.dumps(parsed_config))
    warnings = _set_model_config_warnings(
        proj.get("default_backend") or DEFAULT_BACKEND,
        parsed_config,
    )
    click.echo(json.dumps({"updated": True, "warnings": warnings}))


@project.command("model-config")
@click.argument("name_or_id")
@click.option(
    "--set",
    "config_json",
    default=None,
    help="Set model config JSON (think_model/work_model/think_effort/work_effort).",
)
@click.option("--reset", is_flag=True, help="Reset project model config to defaults.")
@click.option("--presets", is_flag=True, help="Show all model presets from catalog.")
def project_model_config(name_or_id: str, config_json: str | None, reset: bool, presets: bool):
    """Show, set, or reset project model config."""
    if presets:
        _show_model_presets_json()
        return
    if config_json is not None and reset:
        raise click.ClickException("Cannot pass both --set and --reset.")

    with connect() as conn:
        proj = _project_or_error(conn, name_or_id)
        if reset:
            set_project_model_config(conn, proj["id"], None)
            return
        if config_json is not None:
            _update_project_model_config(conn, proj, config_json)
            return
        _show_project_model_config(conn, proj)


@project.command("base-branch")
@click.argument("name_or_id")
@click.argument("base_branch", required=False, default=None)
@click.option("--reset", is_flag=True, help="Reset to project-level fallback (main).")
def project_base_branch(name_or_id: str, base_branch: str | None, reset: bool):
    """Show, set, or reset the project base branch for CLI git flows."""
    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)

        if reset:
            set_project_base_branch(conn, proj["id"], None)
            return

        if base_branch is not None:
            set_project_base_branch(conn, proj["id"], base_branch)
            return

        click.echo(json.dumps({"base_branch": _resolve_project_base_branch(conn, proj["id"])}))


def _project_remove_queue_jobs(plan_ids: list[str], task_ids: list[str]) -> int:
    """Best-effort cleanup of orphaned queue jobs for removed entities."""
    try:
        from agm.queue import remove_jobs_for_entities

        return remove_jobs_for_entities(plan_ids, task_ids)
    except Exception as exc:
        log.debug("Failed to remove queue jobs during project remove: %s", exc, exc_info=True)
        return 0


def _project_remove_logs_for_ids(
    log_dir: Path, entity_ids: list[str], prefixes: tuple[str, ...]
) -> int:
    """Remove matching worker log files for the provided entity IDs."""
    removed_logs = 0
    for entity_id in entity_ids:
        for prefix in prefixes:
            log_file = log_dir / f"{prefix}{entity_id}.log"
            if log_file.exists():
                log_file.unlink(missing_ok=True)
                removed_logs += 1
    return removed_logs


def _project_remove_log_files(plan_ids: list[str], task_ids: list[str]) -> int:
    """Remove worker log files for removed plans/tasks."""
    from agm.queue import LOG_DIR as log_dir

    if not log_dir.exists():
        return 0
    removed_logs = _project_remove_logs_for_ids(log_dir, plan_ids, ("plan-", "tasks-"))
    removed_logs += _project_remove_logs_for_ids(log_dir, task_ids, ("exec-", "review-", "merge-"))
    return removed_logs


def _project_remove_agm_dir(project_dir: str) -> bool:
    """Remove the .agm/ directory from a project. Best-effort."""
    import shutil

    agm_dir = Path(project_dir) / ".agm"
    if not agm_dir.is_dir():
        return False
    try:
        shutil.rmtree(agm_dir)
        return True
    except OSError as exc:
        log.debug("Failed to remove .agm/ for %s: %s", project_dir, exc)
        return False


def _project_remove_setup_log(project_id: str) -> bool:
    """Remove the setup worker log file for a project. Best-effort."""
    from agm.queue import LOG_DIR as log_dir

    log_file = log_dir / f"setup-{project_id}.log"
    if log_file.exists():
        log_file.unlink(missing_ok=True)
        return True
    return False


@project.command("remove")
@click.argument("name_or_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def project_remove(name_or_id: str, yes: bool):
    """Remove a project and all its plans, tasks, and logs."""
    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)
    project_dir = proj["dir"]
    project_id = proj["id"]
    if not yes and not click.confirm("  Proceed?"):
        return
    with connect() as conn:
        result = remove_project(conn, name_or_id)
    if result is None:
        raise _not_found("project", name_or_id)

    _project_remove_queue_jobs(result["plan_ids"], result["task_ids"])
    _project_remove_log_files(result["plan_ids"], result["task_ids"])
    _project_remove_agm_dir(project_dir)
    _project_remove_setup_log(project_id)


@project.command("move")
@click.argument("name_or_id")
@click.option("--dir", "-d", "directory", required=True, type=click.Path(exists=True))
def project_move(name_or_id: str, directory: str):
    """Change a project's directory path (preserves all data)."""
    from agm.db import update_project_dir

    directory = str(Path(directory).resolve())
    with connect() as conn:
        try:
            proj = update_project_dir(conn, name_or_id, directory)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        if not proj:
            raise _not_found("project", name_or_id)


@project.command("rename")
@click.argument("name_or_id")
@click.argument("new_name")
def project_rename(name_or_id: str, new_name: str):
    """Rename a project (preserves all data)."""
    from agm.db import rename_project

    with connect() as conn:
        try:
            result = rename_project(conn, name_or_id, new_name)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        if not result:
            raise _not_found("project", name_or_id)


def _emit_quality_gate_presets(presets: dict[str, dict[str, Any]]) -> None:
    """Output available quality gate presets as JSON."""
    click.echo(json.dumps(presets, indent=2))


def _emit_default_quality_gate() -> None:
    """Render default (empty) quality gate config."""
    from agm.jobs import _default_quality_gate

    click.echo(json.dumps(_default_quality_gate(), indent=2))


def _emit_generated_quality_gate(name_or_id: str) -> None:
    """Generate and print LLM-derived quality gate config as JSON."""
    from agm.jobs import generate_quality_gate

    try:
        config = generate_quality_gate(name_or_id)
    except Exception as e:
        raise click.ClickException(f"Generate failed: {e}") from e
    click.echo(json.dumps(config, indent=2))


def _parse_quality_gate(config_json: str) -> None:
    """Validate quality gate JSON payload shape."""
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON: {e}") from e
    if not isinstance(parsed, dict) or "checks" not in parsed:
        raise click.ClickException(
            "Quality gate config must be a JSON object with a 'checks' array."
        )


def _emit_quality_gate(quality_gate: str | None) -> None:
    if quality_gate:
        click.echo(json.dumps(json.loads(quality_gate), indent=2))
        return
    click.echo(json.dumps(None))


def _resolve_quality_gate_project(conn, name_or_id: str):
    proj = get_project(conn, name_or_id)
    if not proj:
        raise _not_found("project", name_or_id)
    return proj


def _apply_quality_gate_change(
    conn,
    proj,
    *,
    reset: bool,
    preset: str | None,
    config_json: str | None,
    quality_gate_presets,
) -> bool:
    from agm.db import set_project_quality_gate

    if reset:
        set_project_quality_gate(conn, proj["id"], None)
        return True

    if preset:
        if preset not in quality_gate_presets:
            available = ", ".join(sorted(quality_gate_presets.keys()))
            raise click.ClickException(f"Unknown preset '{preset}'. Available: {available}")
        preset_config = quality_gate_presets[preset]["config"]
        set_project_quality_gate(conn, proj["id"], json.dumps(preset_config))
        return True

    if config_json is not None:
        _parse_quality_gate(config_json)
        set_project_quality_gate(conn, proj["id"], config_json)
        return True

    return False


@project.command("setup")
@click.argument("name_or_id")
@click.option("--dry-run", is_flag=True, help="Show config without applying.")
@click.option("--wait", "wait_for_completion", is_flag=True, help="Block until setup completes.")
@click.option("--backend", default=None, help="Backend override.")
def project_setup(name_or_id: str, dry_run: bool, wait_for_completion: bool, backend: str | None):
    """Auto-configure project pipeline settings via LLM inspection.

    Inspects the project's config files to detect tooling and generates:
    quality gate (format/lint/test), post-merge command, and stack info.

    Runs asynchronously via queue unless --dry-run is passed.
    """
    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)

    if backend is not None:
        normalized_backend = backend.strip().lower()
        if normalized_backend not in VALID_BACKENDS:
            options = ", ".join(sorted(VALID_BACKENDS))
            raise click.ClickException(f"Unknown backend '{backend}'. Valid: {options}")
        backend = normalized_backend

    if dry_run and wait_for_completion:
        raise click.ClickException("--wait cannot be used with --dry-run.")

    if dry_run:
        from agm.jobs_setup import run_project_setup

        try:
            result = run_project_setup(name_or_id, backend=backend, dry_run=True)
            click.echo(json.dumps(result, indent=2))
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    else:
        if wait_for_completion:
            from agm.queue import enqueue_project_setup, subscribe_events

            subscriber = subscribe_events(project=proj["name"], timeout=5.0)
            try:
                setup_job = enqueue_project_setup(proj["id"], proj["name"], backend=backend)
            except Exception as e:
                _close_event_subscriber(subscriber)
                raise click.ClickException(f"Failed to enqueue setup: {e}") from e
            _wait_for_project_setup_completion(proj["id"], subscriber, setup_job.id)
            return

        from agm.queue import enqueue_project_setup

        try:
            enqueue_project_setup(proj["id"], proj["name"], backend=backend)
        except Exception as e:
            raise click.ClickException(f"Failed to enqueue setup: {e}") from e


@project.command("setup-status")
@click.argument("name_or_id")
def project_setup_status(name_or_id: str):
    """Show setup job state for a project."""
    import json

    from agm.queue import get_job, inspect_queue_jobs

    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)
        setup_result = _parse_json_column(proj.get("setup_result"))

    setup_job_id = f"setup-{proj['id']}"
    queue_rows = inspect_queue_jobs("agm:setup")
    queue_row = next((row for row in queue_rows if row.get("job_id") == setup_job_id), None)

    rq_status = None
    exc_info = None
    job = get_job(setup_job_id)
    if job is not None:
        rq_status = str(job.get_status(refresh=True))
        exc_info = getattr(job, "exc_info", None)

    click.echo(
        json.dumps(
            {
                "project_id": proj["id"],
                "project_name": proj["name"],
                "setup_job_id": setup_job_id,
                "rq_status": rq_status,
                "queue": queue_row,
                "setup_result": setup_result,
                "error": (
                    exc_info.splitlines()[-1].strip()
                    if isinstance(exc_info, str) and exc_info
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )


def _wait_for_project_setup_completion(project_id: str, subscriber: Any, job_id: str) -> None:
    """Block until project setup emits a terminal completion event."""
    normalized_job_id = str(job_id).strip()
    deadline = time.monotonic() + PROJECT_SETUP_WAIT_TIMEOUT_SECONDS
    try:
        while True:
            if time.monotonic() >= deadline:
                job_status, job_error = _project_setup_job_terminal_state(normalized_job_id)
                if job_status == "finished":
                    return
                if job_status == "failed":
                    _raise_project_setup_failed(job_error)
                raise click.ClickException(
                    f"Setup wait timed out after {int(PROJECT_SETUP_WAIT_TIMEOUT_SECONDS)} seconds."
                )
            event = next(subscriber)
            if event is None:
                job_status, job_error = _project_setup_job_terminal_state(normalized_job_id)
                if job_status == "finished":
                    return
                if job_status == "failed":
                    _raise_project_setup_failed(job_error)
                continue
            if event.get("type") != "project:setup":
                continue
            if event.get("id") != project_id:
                continue

            event_job_id = event.get("job_id")
            if (
                isinstance(event_job_id, str)
                and event_job_id.strip()
                and event_job_id != normalized_job_id
            ):
                continue

            status = str(event.get("status", "")).strip().lower()
            if status == "completed":
                job_status, job_error = _project_setup_job_terminal_state(normalized_job_id)
                if job_status == "finished":
                    return
                if job_status == "failed":
                    _raise_project_setup_failed(job_error)
                continue
            if status == "failed":
                job_status, job_error = _project_setup_job_terminal_state(normalized_job_id)
                if job_status == "failed":
                    event_error = str(event.get("error") or "").strip() or None
                    _raise_project_setup_failed(job_error or event_error)
                if job_status == "finished":
                    return
                continue
    finally:
        _close_event_subscriber(subscriber)


def _project_setup_job_terminal_state(job_id: str) -> tuple[str | None, str | None]:
    from agm.queue import get_job

    job = get_job(job_id)
    if job is None:
        return None, None

    status = str(job.get_status(refresh=True)).strip().lower()
    if status == "finished":
        return "finished", None
    if status != "failed":
        return None, None

    detail = str(getattr(job, "exc_info", "") or "").strip()
    if not detail:
        return "failed", None
    return "failed", detail.splitlines()[-1].strip() or None


def _raise_project_setup_failed(detail: str | None) -> None:
    if detail:
        raise click.ClickException(f"Setup failed: {detail}")
    raise click.ClickException("Setup failed.")


def _close_event_subscriber(subscriber: Any) -> None:
    close_fn = getattr(subscriber, "close", None)
    if callable(close_fn):
        close_fn()


@project.command("quality-gate")
@click.argument("name_or_id", required=False, default=None)
@click.option("--set", "config_json", default=None, help="Set quality gate config (JSON string).")
@click.option("--reset", is_flag=True, help="Reset to no quality gate.")
@click.option("--show-default", is_flag=True, help="Show the default quality gate config.")
@click.option("--preset", default=None, help="Apply a preset (python, typescript).")
@click.option("--list-presets", is_flag=True, help="List available presets.")
@click.option("--generate", is_flag=True, help="Generate config via LLM inspection.")
def project_quality_gate(
    name_or_id: str | None,
    config_json: str | None,
    reset: bool,
    show_default: bool,
    preset: str | None,
    list_presets: bool,
    generate: bool,
):
    """View or set the quality gate config for a project.

    The quality gate runs after executor finishes, before reviewer.
    Default: none. Configure with --set for your project's tooling.

    Config format (JSON):
    \b
      {
        "auto_fix": [{"name": "...", "cmd": ["..."]}],
        "checks": [{"name": "...", "cmd": ["..."], "timeout": 120}]
      }
    """
    from agm.backends import QUALITY_GATE_PRESETS

    # --list-presets: no project needed
    if list_presets:
        _emit_quality_gate_presets(QUALITY_GATE_PRESETS)
        return

    if show_default:
        _emit_default_quality_gate()
        return

    # All other operations require a project name
    if not name_or_id:
        raise click.ClickException("Missing argument 'NAME_OR_ID'.")

    # --generate: LLM inspection (does NOT auto-apply)
    if generate:
        _emit_generated_quality_gate(name_or_id)
        return

    with connect() as conn:
        proj = _resolve_quality_gate_project(conn, name_or_id)
        if _apply_quality_gate_change(
            conn,
            proj,
            reset=reset,
            preset=preset,
            config_json=config_json,
            quality_gate_presets=QUALITY_GATE_PRESETS,
        ):
            return

        _emit_quality_gate(proj.get("quality_gate"))


@project.command("plan-approval")
@click.argument("name_or_id")
@click.argument("mode", required=False, type=click.Choice(["auto", "manual"]))
@click.option("--reset", is_flag=True, help="Reset to auto (no approval required).")
def project_plan_approval(name_or_id: str, mode: str | None, reset: bool):
    """View or set plan approval mode for a project.

    \b
    When set to 'manual', finalized plans wait for `plan approve`
    before task creation starts. Default is 'auto' (immediate).

    \b
    Examples:
      agm project plan-approval myproject           # show current
      agm project plan-approval myproject manual    # require approval
      agm project plan-approval myproject auto      # disable gate
      agm project plan-approval myproject --reset   # same as auto
    """
    from agm.db import get_project_plan_approval, set_project_plan_approval

    with connect() as conn:
        try:
            proj = get_project(conn, name_or_id)
            if not proj:
                raise _not_found("project", name_or_id)

            if reset:
                set_project_plan_approval(conn, proj["id"], None)
                return

            if mode is not None:
                set_project_plan_approval(conn, proj["id"], mode)
                return

            current = get_project_plan_approval(conn, proj["id"])
            click.echo(json.dumps({"plan_approval": current}))
        except ValueError as e:
            raise click.ClickException(str(e)) from e


@project.command("post-merge-command")
@click.argument("name_or_id")
@click.argument("command", required=False)
@click.option("--reset", is_flag=True, help="Clear the post-merge command.")
def project_post_merge_command(name_or_id: str, command: str | None, reset: bool):
    """View or set the post-merge command for a project.

    \b
    When set, this command runs after each successful task merge with the
    project directory as cwd. The merge SHA is provided in `AGM_MERGE_SHA`.
    Failures are logged but never block the pipeline.

    \b
    Examples:
      agm project post-merge-command myproject                            # show current
      agm project post-merge-command myproject "scripts/post-merge.sh"   # set
      agm project post-merge-command myproject --reset                   # clear
    """
    from agm.db import get_project_post_merge_command, set_project_post_merge_command

    with connect() as conn:
        try:
            proj = get_project(conn, name_or_id)
            if not proj:
                raise _not_found("project", name_or_id)

            if reset:
                set_project_post_merge_command(conn, proj["id"], None)
                return

            if command is not None:
                set_project_post_merge_command(conn, proj["id"], command)
                return

            current = get_project_post_merge_command(conn, proj["id"])
            click.echo(json.dumps({"post_merge_command": current}))
        except ValueError as e:
            raise click.ClickException(str(e)) from e


@project.command("app-server-approval")
@click.argument("name_or_id")
@click.option(
    "--set",
    "config_json",
    default=None,
    help="Set approval policy JSON (per app-server request method).",
)
@click.option(
    "--preset",
    type=click.Choice(["allow-all", "deny-all"]),
    default=None,
    help="Apply built-in policy preset.",
)
@click.option("--reset", is_flag=True, help="Reset to default policy.")
def project_app_server_approval(
    name_or_id: str,
    config_json: str | None,
    preset: str | None,
    reset: bool,
):
    """View or set per-request app-server approval decisions for a project."""
    from agm.db import (
        APP_SERVER_APPROVAL_POLICY_DEFAULTS,
        get_project_app_server_approval_policy,
        set_project_app_server_approval_policy,
    )

    deny_all = {
        "item/commandExecution/requestApproval": "decline",
        "item/fileChange/requestApproval": "decline",
        "skill/requestApproval": "decline",
        "execCommandApproval": "denied",
        "applyPatchApproval": "denied",
    }
    preset_map = {
        "allow-all": dict(APP_SERVER_APPROVAL_POLICY_DEFAULTS),
        "deny-all": deny_all,
    }

    update_flags = int(bool(config_json is not None)) + int(bool(preset)) + int(bool(reset))
    if update_flags > 1:
        raise click.ClickException("Choose only one of --set, --preset, or --reset.")

    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)
        try:
            if reset:
                set_project_app_server_approval_policy(conn, proj["id"], None)
                return
            if preset:
                set_project_app_server_approval_policy(conn, proj["id"], preset_map[preset])
                return
            if config_json is not None:
                parsed = json.loads(config_json)
                if not isinstance(parsed, dict):
                    raise click.ClickException("Policy config must be a JSON object.")
                set_project_app_server_approval_policy(conn, proj["id"], parsed)
                return

            current = get_project_app_server_approval_policy(conn, proj["id"])
            click.echo(json.dumps({"app_server_approval_policy": current}, indent=2))
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON: {e}") from e
        except ValueError as e:
            raise click.ClickException(str(e)) from e


@project.command("app-server-ask-for-approval")
@click.argument("name_or_id")
@click.option(
    "--set",
    "config_json",
    default=None,
    help="Set AskForApproval JSON (string enum or reject object).",
)
@click.option(
    "--preset",
    type=click.Choice(["never", "on-request", "on-failure", "untrusted", "reject-all"]),
    default=None,
    help="Apply built-in AskForApproval preset.",
)
@click.option("--reset", is_flag=True, help="Reset to default policy (never).")
def project_app_server_ask_for_approval(
    name_or_id: str,
    config_json: str | None,
    preset: str | None,
    reset: bool,
):
    """View or set app-server AskForApproval policy for a project."""
    from agm.db import (
        get_project_app_server_ask_for_approval,
        set_project_app_server_ask_for_approval,
    )

    preset_map: dict[str, object] = {
        "never": "never",
        "on-request": "on-request",
        "on-failure": "on-failure",
        "untrusted": "untrusted",
        "reject-all": {
            "reject": {
                "mcp_elicitations": True,
                "rules": True,
                "sandbox_approval": True,
            }
        },
    }

    update_flags = int(bool(config_json is not None)) + int(bool(preset)) + int(bool(reset))
    if update_flags > 1:
        raise click.ClickException("Choose only one of --set, --preset, or --reset.")

    with connect() as conn:
        proj = get_project(conn, name_or_id)
        if not proj:
            raise _not_found("project", name_or_id)

        try:
            if reset:
                set_project_app_server_ask_for_approval(conn, proj["id"], None)
                return
            if preset:
                set_project_app_server_ask_for_approval(conn, proj["id"], preset_map[preset])
                return
            if config_json is not None:
                parsed = json.loads(config_json)
                set_project_app_server_ask_for_approval(conn, proj["id"], parsed)
                return

            current = get_project_app_server_ask_for_approval(conn, proj["id"])
            click.echo(json.dumps({"app_server_ask_for_approval": current}, indent=2))
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON: {e}") from e
        except ValueError as e:
            raise click.ClickException(str(e)) from e


@project.command("stats")
@click.argument("name_or_id")
def project_stats(name_or_id: str):
    """Show pipeline analytics for a project."""
    import json as json_mod

    with connect() as conn:
        p = get_project(conn, name_or_id)
        if not p:
            raise _not_found("project", name_or_id)
        plans = list_plan_requests(conn, project_id=p["id"])
        tasks = list_tasks(conn, project_id=p["id"])

    plan_counts = status_counts(plans)
    task_counts = status_counts(tasks)
    tokens = project_token_totals(plans, tasks)

    data = {
        "project": p["name"],
        "total_plans": len(plans),
        "plan_counts": plan_counts,
        "total_tasks": len(tasks),
        "task_counts": task_counts,
        "plan_model_counts": model_usage_counts(plans, "model"),
        "task_model_counts": model_usage_counts(tasks, "model"),
        "tokens": tokens,
    }

    click.echo(json_mod.dumps(data, indent=2, default=str))


# -- plan --


@main.group()
def plan():
    """Request, monitor, and control planning workflows."""


@plan.command("request")
@click.argument("prompt", callback=_validate_prompt)
@click.option("--project", "-p", "project_name", required=True, help="Project name or ID.")
@click.option(
    "--caller",
    default="cli",
    type=CALLER_TYPE,
    help="Client that invoked this.",
)
@click.option(
    "--backend",
    default=None,
    type=click.Choice(sorted(VALID_BACKENDS)),
    help="Backend to produce the plan (default: project default or codex).",
)
def plan_request(prompt: str, project_name: str, caller: str, backend: str | None):
    """Request a new plan — enrichment, planning, then task creation.

    Creates a plan in pending status, enqueues enrichment (prompt refinement),
    then planning (task breakdown), then task creation. The full pipeline runs
    automatically. Monitor with `plan watch PLAN_ID`.
    """
    from agm.queue import enqueue_enrichment

    with connect() as conn:
        try:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)

            backend = resolve_backend(backend)

            p = create_plan_request(
                conn,
                project_id=proj["id"],
                prompt=prompt,
                caller=caller,
                backend=backend,
            )
            session = create_session(
                conn,
                project_id=proj["id"],
                trigger="plan_request",
                trigger_prompt=prompt,
            )
            set_plan_session_id(conn, p["id"], session["id"])
            update_session_status(conn, session["id"], "active")
            _emit_session_event(conn, session["id"], "active", project_name=proj["name"])

            # Warn if quality gate is not configured
            from agm.db import add_channel_message, get_project_quality_gate
            from agm.queue import publish_event

            if not get_project_quality_gate(conn, proj["id"]):
                msg = add_channel_message(
                    conn,
                    session_id=session["id"],
                    kind="steer",
                    sender=f"system:{session['id'][:8]}",
                    content=(
                        "No quality gate configured for this project. "
                        "Agent output will not be automatically validated "
                        "against format, lint, or test checks. "
                        "Run 'agm project setup' to configure."
                    ),
                    metadata=json.dumps(
                        {
                            "phase": "planning",
                            "status": "advisory",
                            "plan_id": p["id"],
                            "project_id": proj["id"],
                        }
                    ),
                )
                publish_event(
                    "session:message",
                    session["id"],
                    "steer",
                    project=proj["name"],
                    plan_id=p["id"],
                    source="cli",
                    extra={
                        "session_id": session["id"],
                        "sender": f"system:{session['id'][:8]}",
                        "kind": "steer",
                        "message_id": msg["id"],
                    },
                )

            try:
                enqueue_enrichment(p["id"])
            except Exception as e:
                # Enqueue failed — mark plan as failed so it doesn't sit as orphaned pending
                from agm.db import update_plan_request_status

                update_plan_request_status(conn, p["id"], "failed")
                raise click.ClickException(f"Failed to enqueue plan {p['id']}: {e}") from e
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


def _plan_list_project_id(conn, project_name: str | None) -> str | None:
    if not project_name:
        return None
    proj = get_project(conn, project_name)
    if not proj:
        raise _not_found("project", project_name)
    return proj["id"]


@plan.command("list")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(sorted(VALID_PLAN_STATUSES), case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--all",
    "-a",
    "show_all",
    is_flag=True,
    help="Include finalized, cancelled, and failed plans.",
)
def plan_list(project_name: str | None, status: str | None, show_all: bool):
    """List plans.

    Shows active plans by default. Filter by project (-p) and/or status (-s).
    Use --all (-a) to include finalized/cancelled/failed history.
    """
    import json

    default_status_filter = (
        None
        if (show_all or status)
        else [
            "pending",
            "running",
            "awaiting_input",
        ]
    )

    with connect() as conn:
        project_id = _plan_list_project_id(conn, project_name)
        plans = list_plan_requests(conn, project_id, status=status, statuses=default_status_filter)
        runtimes = bulk_active_runtime_seconds(conn, "plan", PLAN_ACTIVE_STATUSES)
        enriched_plans = enrich_plan_list_rows(conn, plans, runtimes)
    click.echo(json.dumps(enriched_plans, indent=2, default=str))


@plan.command("failures")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
def plan_failures(project_name: str | None):
    """List recently failed plans with diagnostic snippets."""
    import json as json_mod

    with connect() as conn:
        project_id = None
        if project_name:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)
            project_id = proj["id"]

        plans = list_plan_requests(conn, project_id, status="failed")
        failures = []
        for p in reversed(plans):
            proj = get_project(conn, p["project_id"])
            project = proj["name"] if proj else p["project_id"]
            source, error = plan_failure_diagnostic(conn, p["id"])
            failures.append(
                {
                    "plan_id": p["id"],
                    "project_id": p["project_id"],
                    "project": project,
                    "source": source,
                    "prompt": p.get("prompt", ""),
                    "prompt_snippet": format_plan_failure_prompt(p.get("prompt", "")),
                    "error": error,
                    "error_snippet": format_plan_failure_error(error),
                    "created_at": p.get("created_at"),
                    "updated_at": p.get("updated_at"),
                    "failed": p.get("updated_at"),
                }
            )

    click.echo(json_mod.dumps(failures, indent=2, default=str))


_STALE_RUNNING_SECONDS = 1800


def _collect_troubleshoot_data(conn, project_id) -> dict:
    """Collect all troubleshoot diagnostic data from DB."""
    from agm.queries import elapsed_seconds

    # Failed plans
    plan_issues = []
    for p in reversed(list_plan_requests(conn, project_id, status="failed")):
        proj = get_project(conn, p["project_id"])
        source, error = plan_failure_diagnostic(conn, p["id"])
        plan_issues.append(
            {
                "plan_id": p["id"],
                "project": proj["name"] if proj else p["project_id"],
                "source": source,
                "prompt_snippet": format_plan_failure_prompt(p.get("prompt", "")),
                "error": error,
                "error_snippet": format_plan_failure_error(error),
                "failed": p.get("updated_at"),
            }
        )

    # Failed tasks
    task_issues = []
    for t in reversed(list_tasks(conn, project_id=project_id, status="failed")):
        plan = get_plan_request(conn, t["plan_id"])
        proj = get_project(conn, plan["project_id"]) if plan else None
        source, error = task_failure_diagnostic(conn, t["id"])
        task_issues.append(
            {
                "task_id": t["id"],
                "plan_id": t["plan_id"],
                "project": proj["name"] if proj else "-",
                "title_snippet": watch_truncate(t.get("title", ""), 50),
                "source": source,
                "error": error,
                "error_snippet": format_plan_failure_error(error),
                "failed": t.get("updated_at"),
            }
        )

    # Stale running entities (>{_STALE_RUNNING_SECONDS} seconds)
    def _find_stale(entities, id_key, snippet_fn):
        stale = []
        for e in entities:
            age = elapsed_seconds(e.get("updated_at"))
            if age is not None and age > _STALE_RUNNING_SECONDS:
                stale.append(
                    {
                        id_key: e["id"],
                        "age": format_elapsed(e.get("updated_at")),
                        "snippet": snippet_fn(e),
                    }
                )
        return stale

    stale_plans = _find_stale(
        list_plan_requests(conn, project_id, status="running"),
        "plan_id",
        lambda p: format_plan_failure_prompt(p.get("prompt", "")),
    )
    stale_tasks = _find_stale(
        list_tasks(conn, project_id=project_id, status="running"),
        "task_id",
        lambda t: watch_truncate(t.get("title", ""), 50),
    )
    return {
        "failed_plans": plan_issues,
        "failed_tasks": task_issues,
        "stale_running_plans": stale_plans,
        "stale_running_tasks": stale_tasks,
    }


@plan.command("troubleshoot")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
def plan_troubleshoot(project_name: str | None):
    """Unified failure diagnosis: failed plans + failed tasks + queue health."""
    import json as json_mod

    from agm.queue import get_queue_counts_safe

    with connect() as conn:
        project_id = None
        if project_name:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)
            project_id = proj["id"]
        report = _collect_troubleshoot_data(conn, project_id)

    report["queue"] = get_queue_counts_safe()

    click.echo(json_mod.dumps(report, indent=2, default=str))


@plan.command("show")
@click.argument("plan_id")
@click.option("--tasks", "show_tasks", is_flag=True, help="Include task list in output.")
def plan_show(plan_id: str, show_tasks: bool):
    """Show plan details."""
    import json

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        task_payload_rows: list[dict[str, Any]] = []
        if show_tasks:
            task_rows = list_tasks(conn, plan_id=plan_id)
            task_rows = task_list_filter_rows(task_rows, show_all=False, status=None)
            runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)
            task_payload_rows = enrich_task_list_rows(
                task_rows,
                resolve_project_names_for_tasks(conn, task_rows),
                runtimes,
            )
    if show_tasks:
        payload = dict(p)
        payload["tasks"] = task_payload_rows
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    click.echo(json.dumps(p, indent=2, default=str))


@plan.command("history")
@click.argument("plan_id")
def plan_history(plan_id: str):
    """Show the continuation chain for a plan (latest branch)."""
    import json

    with connect() as conn:
        chain = get_plan_chain(conn, plan_id)
        if not chain:
            raise _not_found("plan", plan_id)
        data = {"plan_id": plan_id, "chain": normalize_plan_chain(chain, plan_id)}
        click.echo(json.dumps(data))


@plan.command("timeline")
@click.argument("plan_id")
def plan_timeline(plan_id: str):
    """Show chronological plan status transitions with durations."""
    import json

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        rows = list_plan_timeline_rows(conn, plan_id)
    click.echo(
        json.dumps(
            {
                "plan_id": plan_id,
                "timeline": normalize_timeline_rows(rows),
            },
            indent=2,
            default=str,
        )
    )


@plan.command("questions")
@click.argument("plan_id")
@click.option("--unanswered", is_flag=True, help="Show only unanswered questions.")
def plan_questions(plan_id: str, unanswered: bool):
    """List questions for a plan."""
    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        questions = list_plan_questions(conn, plan_id, unanswered_only=unanswered)
    click.echo(
        json.dumps(
            {
                "plan_id": plan_id,
                "unanswered_only": unanswered,
                "count": len(questions),
                "questions": normalize_plan_questions(questions),
            },
            indent=2,
            default=str,
        )
    )


@plan.command("answer")
@click.argument("question_id")
@click.argument("answer_text")
def plan_answer(question_id: str, answer_text: str):
    """Answer a plan question. Auto-resumes enrichment when all questions are answered.

    Requires: plan in `awaiting_input` status. Use `plan questions PLAN_ID`
    to list questions and their IDs. When all questions are answered, enrichment
    resumes automatically and the plan transitions to `running`.
    """
    from agm.db import get_plan_question, get_unanswered_question_count

    with connect() as conn:
        if not answer_plan_question(conn, question_id, answer_text):
            raise click.ClickException(f"Question '{question_id}' not found or already answered.")

        # Check if all questions for this plan are now answered
        question = get_plan_question(conn, question_id)
        if not question:
            return
        plan_id = question["plan_id"]
        p = get_plan_request(conn, plan_id)
        if not p or p["status"] != "awaiting_input":
            return

        remaining = get_unanswered_question_count(conn, plan_id)
        if remaining > 0:
            return

        # All answered — resume enrichment
        from agm.db import update_prompt_status
        from agm.queue import enqueue_enrichment

        update_prompt_status(conn, plan_id, "enriching")
        _emit_plan_event(conn, plan_id, "running")
        try:
            enqueue_enrichment(plan_id)
        except Exception as exc:
            update_prompt_status(conn, plan_id, "failed")
            _emit_plan_event(conn, plan_id, "failed")
            raise click.ClickException(
                f"Failed to resume enrichment for plan {plan_id}: {exc}"
            ) from exc


@plan.command("logs")
@click.argument("plan_id")
@click.option("--level", "-l", default=None, help="Filter by log level (e.g. INFO, ERROR).")
@click.option("--tail", "-n", "tail", type=int, default=None, help="Show only the last N logs.")
def plan_logs_cmd(plan_id: str, level: str | None, tail: int | None):
    """Show logs for a plan."""
    import json

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        logs = list_plan_logs(conn, plan_id, level=level)
        if tail is not None:
            logs = logs[-tail:]
    click.echo(
        json.dumps(
            {
                "plan_id": plan_id,
                "level": level,
                "count": len(logs),
                "logs": normalize_logs(logs),
            },
            indent=2,
            default=str,
        )
    )


@plan.command("continue")
@click.argument("plan_id")
@click.argument("prompt", callback=_validate_prompt)
@click.option(
    "--caller",
    default="cli",
    type=CALLER_TYPE,
    help="Client that invoked this.",
)
def plan_continue(plan_id: str, prompt: str, caller: str):
    """Continue a finalized plan with a follow-up prompt.

    Resumes the parent plan's thread so the backend agent has full
    context from the previous conversation.
    """
    from agm.queue import enqueue_enrichment

    with connect() as conn:
        try:
            parent = get_plan_request(conn, plan_id)
            if not parent:
                raise _not_found("plan", plan_id)
            if parent["status"] != "finalized":
                raise click.ClickException(
                    f"Plan '{plan_id}' is '{parent['status']}', not 'finalized'. "
                    f"Only finalized plans can be continued."
                )
            if not parent.get("thread_id"):
                raise click.ClickException(f"Plan '{plan_id}' has no thread_id — cannot resume.")
            p = create_plan_request(
                conn,
                project_id=parent["project_id"],
                prompt=prompt,
                caller=caller,
                backend=parent["backend"],
                parent_id=parent["id"],
            )
            # Reuse the parent plan's session if it has one, otherwise create new
            parent_session_id = parent.get("session_id")
            if parent_session_id and get_session(conn, parent_session_id):
                session_id = parent_session_id
            else:
                session = create_session(
                    conn,
                    project_id=parent["project_id"],
                    trigger="plan_continue",
                    trigger_prompt=prompt,
                )
                session_id = session["id"]
                update_session_status(conn, session_id, "active")
                proj = get_project(conn, parent["project_id"])
                _emit_session_event(
                    conn,
                    session_id,
                    "active",
                    project_name=proj["name"] if proj else "",
                )
            set_plan_session_id(conn, p["id"], session_id)
            try:
                enqueue_enrichment(p["id"])
            except Exception as e:
                from agm.db import update_plan_request_status

                update_plan_request_status(conn, p["id"], "failed")
                raise click.ClickException(f"Failed to enqueue plan {p['id']}: {e}") from e
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@plan.command("retry")
@click.argument("plan_id")
def plan_retry(plan_id: str):
    """Retry a failed plan.

    Requires: plan in `failed` status. Resets the plan to `pending` and
    re-enqueues the full pipeline (enrichment -> planning -> task creation).
    """
    from agm.queue import enqueue_enrichment

    with connect() as conn:
        try:
            p = get_plan_request(conn, plan_id)
            if not p:
                raise _not_found("plan", plan_id)
            if p["status"] != "failed":
                hint = ""
                if p["status"] == "finalized":
                    hint = " Use 'agm plan continue' to build on a finalized plan."
                raise click.ClickException(
                    f"Plan '{plan_id}' is '{p['status']}', not 'failed'. "
                    f"Only failed plans can be retried.{hint}"
                )
            if not reset_plan_for_retry(conn, plan_id):
                raise click.ClickException(
                    f"Failed to reset plan '{plan_id}'. "
                    f"Status may have changed — run 'agm plan show {plan_id}'."
                )
            try:
                enqueue_enrichment(plan_id)
            except Exception as e:
                from agm.db import update_plan_request_status

                update_plan_request_status(conn, plan_id, "failed")
                raise click.ClickException(f"Failed to enqueue plan {plan_id}: {e}") from e
            _emit_plan_event(conn, plan_id, "running")
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@plan.command("cancel")
@click.argument("plan_id")
@click.option("--reason", "-r", default=None, help="Reason for cancellation.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def plan_cancel(plan_id: str, reason: str | None, yes: bool):
    """Cancel a plan and cascade-cancel all non-terminal tasks.

    Works on plans in any status (including finalized and failed). All
    non-terminal tasks under the plan are also cancelled. Use --reason
    to record why.
    """
    from agm.db import add_plan_log, force_cancel_plan, update_plan_request_status

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        if p["status"] == "cancelled":
            raise click.ClickException(f"Plan '{plan_id}' is already cancelled.")
        if not yes and not click.confirm("  Proceed?"):
            return

        if not update_plan_request_status(conn, plan_id, "cancelled"):
            # Plan is in a terminal state — force-cancel it
            old = force_cancel_plan(conn, plan_id)
            if not old:
                raise click.ClickException(
                    f"Plan '{plan_id}' is '{p['status']}' and cannot be cancelled."
                )

        msg = f"Cancelled: {reason}" if reason else "Cancelled via CLI"
        add_plan_log(conn, plan_id=plan_id, level="INFO", message=msg)
        _emit_plan_event(conn, plan_id, "cancelled")

        # Cascade-cancel non-terminal tasks
        tasks = list_tasks(conn, plan_id=plan_id)
        for t in tasks:
            _cancel_task_for_plan(conn, p["project_id"], t, plan_id, msg)


def _task_status_to_job_id(task: TaskRow) -> str | None:
    """Map task status to active worker job id."""
    return {
        "running": f"exec-{task['id']}",
        "review": f"review-{task['id']}",
        "approved": f"merge-{task['id']}",
    }.get(task["status"])


def _cancel_task_for_plan(
    conn,
    project_id: str,
    task: TaskRow,
    plan_id: str,
    message: str,
) -> tuple[bool, list[str]]:
    """Cancel one task during plan cancellation; return whether cancelled and cascaded IDs."""
    from agm.db import add_task_log
    from agm.git_ops import remove_worktree

    if task["status"] in ("completed", "cancelled"):
        return False, []

    job_id = _task_status_to_job_id(task)
    if job_id:
        try:
            from rq.job import Job

            from agm.queue import get_queue

            queue = get_queue()
            job = Job.fetch(job_id, connection=queue.connection)  # type: ignore[assignment]
            try:
                job.cancel()
            except Exception as exc:
                log.debug("Job cancel failed for %s: %s", job_id, exc)
            try:
                job.stop()  # type: ignore[attr-defined]
            except Exception as exc:
                log.debug("Job stop failed for %s: %s", job_id, exc)
        except Exception as exc:
            log.debug("Job fetch/cancel failed for %s: %s", job_id, exc)

    if task.get("worktree") and task.get("branch"):
        proj = get_project(conn, project_id)
        if proj:
            assert task["worktree"] is not None
            assert task["branch"] is not None
            remove_worktree(proj["dir"], task["worktree"], task["branch"])

    update_task_status(conn, task["id"], "cancelled")
    _emit_task_event(conn, task["id"], "cancelled", plan_id)
    add_task_log(conn, task_id=task["id"], level="INFO", message=message, source="cli")
    _promoted, cascaded = resolve_blockers_for_terminal_task(conn, task["id"])
    for cid in cascaded:
        _emit_task_event(conn, cid, "cancelled", plan_id)
    return True, cascaded


@plan.command("retask")
@click.argument("plan_id")
def plan_retask(plan_id: str):
    """Re-trigger task creation for a finalized plan.

    Resets task_creation_status and enqueues a fresh task creation job.
    The task agent sees existing completed tasks and creates only what's missing.
    """
    from agm.db import update_plan_task_creation_status
    from agm.queue import enqueue_task_creation

    with connect() as conn:
        try:
            p = get_plan_request(conn, plan_id)
            if not p:
                raise _not_found("plan", plan_id)
            if p["status"] != "finalized":
                raise click.ClickException(
                    f"Plan '{plan_id}' is '{p['status']}', not 'finalized'. "
                    f"Only finalized plans can have tasks re-created."
                )
            if not p.get("plan"):
                raise click.ClickException(f"Plan '{plan_id}' has no plan text.")
            update_plan_task_creation_status(conn, plan_id, "pending")
            try:
                enqueue_task_creation(plan_id)
            except Exception as e:
                update_plan_task_creation_status(conn, plan_id, "failed")
                raise click.ClickException(
                    f"Failed to enqueue task creation for plan {plan_id}: {e}"
                ) from e
            _emit_plan_event(conn, plan_id, "pending", "plan:task_creation")
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@plan.command("approve")
@click.argument("plan_id")
def plan_approve(plan_id: str):
    """Approve a finalized plan and trigger task creation.

    \b
    When a project has plan_approval set to 'manual', plans wait
    for approval after finalization. This command proceeds with
    task creation.

    Review the plan first with:  agm plan show PLAN_ID --pretty
    """
    from agm.db import update_plan_task_creation_status
    from agm.queue import enqueue_task_creation

    with connect() as conn:
        try:
            p = get_plan_request(conn, plan_id)
            if not p:
                raise _not_found("plan", plan_id)
            if p["status"] != "finalized":
                raise click.ClickException(f"Plan '{plan_id}' is '{p['status']}', not 'finalized'.")
            if not p.get("plan"):
                raise click.ClickException(f"Plan '{plan_id}' has no plan text.")

            tcs = p.get("task_creation_status")
            if tcs not in (None, "awaiting_approval", "failed"):
                raise click.ClickException(
                    f"Plan '{plan_id}' task_creation_status is '{tcs}'. "
                    f"Only plans awaiting approval or failed can be approved."
                )

            update_plan_task_creation_status(conn, plan_id, "pending")
            try:
                enqueue_task_creation(plan_id)
            except Exception as e:
                update_plan_task_creation_status(conn, plan_id, "failed")
                raise click.ClickException(
                    f"Failed to enqueue task creation for plan {plan_id}: {e}"
                ) from e
            _emit_plan_event(conn, plan_id, "pending", "plan:task_creation")
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@plan.command("stats")
@click.argument("plan_id")
def plan_stats(plan_id: str):
    """Show pipeline analytics for a plan and its tasks."""
    import json as json_mod

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        tasks = list_tasks(conn, plan_id=plan_id)
        timing_rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=plan_id)
        data = build_plan_stats_data(conn, plan_id, p, tasks, timing_rows)

    click.echo(json_mod.dumps(data, indent=2, default=str))


@plan.command("trace")
@click.argument("plan_id")
@click.option("--type", "event_type", default=None, help="Filter by event type.")
@click.option("--stage", default=None, help="Filter by pipeline stage.")
@click.option("--summary", is_flag=True, help="Show aggregated summary.")
@click.option("--tail", "-n", "tail", type=int, default=None, help="Show last N events.")
def plan_trace_cmd(
    plan_id: str,
    event_type: str | None,
    stage: str | None,
    summary: bool,
    tail: int | None,
):
    """Show execution trace events for a plan."""
    import json

    from agm.db import get_trace_summary, list_trace_events

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)

        if summary:
            data = get_trace_summary(conn, "plan", plan_id)
            data["plan_id"] = plan_id
            click.echo(json.dumps(data, indent=2, default=str))
            return

        events = list_trace_events(
            conn,
            "plan",
            plan_id,
            event_type=event_type,
            stage=stage,
        )
        if tail is not None:
            events = events[-tail:]
        click.echo(
            json.dumps(
                {"plan_id": plan_id, "count": len(events), "events": events},
                indent=2,
                default=str,
            )
        )


@plan.command("watch")
@click.argument("plan_id")
def plan_watch(plan_id: str):
    """Watch a plan's progress (single JSON snapshot)."""
    _plan_watch_json_loop(plan_id)


def _plan_watch_json_loop(plan_id: str):
    """JSON mode: emit a single snapshot (machine-readable).

    Always exits after one snapshot — consumers (e.g. agm-web) handle
    their own polling cadence.
    """
    import json

    with connect() as conn:
        p = get_plan_request(conn, plan_id)
        if not p:
            raise _not_found("plan", plan_id)
        tasks = list_tasks(conn, plan_id=plan_id)
        recent_events = list_plan_watch_events(
            conn,
            plan_id,
            limit=PLAN_WATCH_RECENT_EVENTS_ROWS,
        )
        timing_rows = list_status_history_timing_rows(conn, entity_type="plan", entity_id=plan_id)
    terminal_state = plan_watch_terminal_state(p, tasks)
    click.echo(
        json.dumps(
            build_plan_watch_snapshot(
                p,
                tasks,
                recent_events,
                terminal_state=terminal_state,
                timing_rows=timing_rows,
            ),
            indent=2,
            default=str,
        )
    )


# -- task --


@main.group()
def task():
    """Run, review, and control task execution."""


@task.command("list")
@click.option("--plan", "plan_id", default=None, help="Filter by plan ID.")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(sorted(VALID_TASK_STATUSES), case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--priority",
    default=None,
    type=click.Choice(sorted(VALID_TASK_PRIORITIES), case_sensitive=False),
    help="Filter by priority.",
)
@click.option(
    "--all",
    "-a",
    "show_all",
    is_flag=True,
    help="Include completed, cancelled, and failed tasks.",
)
def task_list(
    plan_id: str | None,
    project_name: str | None,
    status: str | None,
    priority: str | None,
    show_all: bool,
):
    """List tasks.

    Shows active tasks by default (pending, ready, blocked, running, review,
    approved).  Use --all (-a) to include completed/cancelled/failed history.
    """
    import json

    with connect() as conn:
        project_id = _task_list_resolve_project_id(conn, project_name)
        tasks = list_tasks(
            conn,
            plan_id=plan_id,
            project_id=project_id,
            status=status,
            priority=priority,
        )
        tasks = task_list_filter_rows(tasks, show_all=show_all, status=status)
        runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)
        project_names = resolve_project_names_for_tasks(conn, tasks)
        enriched = enrich_task_list_rows(
            tasks,
            project_names,
            runtimes,
        )
    click.echo(json.dumps(enriched, indent=2, default=str))


def _task_list_resolve_project_id(conn, project_name: str | None) -> str | None:
    if not project_name:
        return None
    proj = get_project(conn, project_name)
    if not proj:
        raise _not_found("project", project_name)
    return proj["id"]


@task.command("failures")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option("--plan", "plan_id", default=None, help="Filter by plan.")
def task_failures(project_name: str | None, plan_id: str | None):
    """List failed tasks with diagnostic snippets."""
    import json as json_mod

    with connect() as conn:
        project_id = None
        if project_name:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)
            project_id = proj["id"]

        tasks = list_tasks(conn, plan_id=plan_id, project_id=project_id, status="failed")
        failures = []
        for t in reversed(tasks):
            plan = get_plan_request(conn, t["plan_id"])
            proj = get_project(conn, plan["project_id"]) if plan else None
            project = proj["name"] if proj else (plan["project_id"] if plan else "-")
            source, error = task_failure_diagnostic(conn, t["id"])
            failures.append(
                {
                    "task_id": t["id"],
                    "plan_id": t["plan_id"],
                    "project_id": plan["project_id"] if plan else None,
                    "project": project,
                    "title": t.get("title", ""),
                    "title_snippet": watch_truncate(t.get("title", ""), 50),
                    "source": source,
                    "error": error,
                    "error_snippet": format_plan_failure_error(error),
                    "priority": effective_task_priority(t.get("priority")),
                    "created_at": t.get("created_at"),
                    "updated_at": t.get("updated_at"),
                    "failed": t.get("updated_at"),
                }
            )

    click.echo(json_mod.dumps(failures, indent=2, default=str))


@task.command("show")
@click.argument("task_id")
def task_show(task_id: str):
    """Show task details."""
    import json

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
    click.echo(json.dumps(t, indent=2, default=str))


@task.command("timeline")
@click.argument("task_id")
def task_timeline(task_id: str):
    """Show chronological task status transitions with durations."""
    import json

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        rows = list_status_history_timing_rows(
            conn,
            entity_type="task",
            entity_id=task_id,
        )
    click.echo(
        json.dumps(
            {
                "task_id": task_id,
                "timeline": normalize_timeline_rows(rows),
            },
            indent=2,
            default=str,
        )
    )


@task.command("blocks")
@click.argument("task_id", required=False)
@click.option("--plan", "plan_id", help="Show blockers for all tasks in a plan.")
@click.option("--project", "project_name", help="Show blockers for all tasks in a project.")
@click.option("--unresolved", is_flag=True, help="Show only unresolved blockers.")
def task_blocks(
    task_id: str | None,
    plan_id: str | None,
    project_name: str | None,
    unresolved: bool,
):
    """Show blockers for a task, plan, or project."""
    import json

    _task_blocks_validate_scope(task_id, plan_id, project_name)
    with connect() as conn:
        _task_blocks_ensure_task_exists(conn, task_id)
        project_id = _task_list_resolve_project_id(conn, project_name)
        blocks = list_task_blocks(
            conn,
            task_id,
            plan_id=plan_id,
            project_id=project_id,
            unresolved_only=unresolved,
        )
    click.echo(
        json.dumps(
            {
                "scope": {
                    "task_id": task_id,
                    "plan_id": plan_id,
                    "project_id": project_id,
                    "project_name": project_name,
                    "unresolved_only": unresolved,
                },
                "count": len(blocks),
                "blocks": normalize_task_blocks(blocks, unresolved),
            },
            indent=2,
            default=str,
        )
    )


def _task_blocks_validate_scope(
    task_id: str | None,
    plan_id: str | None,
    project_name: str | None,
) -> None:
    if not task_id and not plan_id and not project_name:
        raise click.ClickException("Provide a TASK_ID, --plan, or --project.")


def _task_blocks_ensure_task_exists(conn, task_id: str | None) -> None:
    if not task_id:
        return
    t = get_task(conn, task_id)
    if not t:
        raise _not_found("task", task_id)


@task.command("unblock")
@click.argument("block_id")
def task_unblock(block_id: str):
    """Resolve an external blocker (manual blockers, not task-to-task dependencies)."""
    with connect() as conn:
        block = get_task_block(conn, block_id)
        if not block:
            raise _not_found("block", block_id)
        if block["blocked_by_task_id"]:
            raise click.ClickException(
                "Cannot manually resolve internal (task-to-task) blockers. "
                "They resolve when the blocking task completes."
            )
        if block["resolved"]:
            raise click.ClickException(f"Block '{block_id}' is already resolved.")
        if not resolve_task_block(conn, block_id):
            raise click.ClickException(
                f"Failed to resolve block '{block_id}'. It may have been resolved already."
            )


@task.command("set-priority")
@click.argument("task_id")
@click.argument("priority", type=click.Choice(sorted(VALID_TASK_PRIORITIES), case_sensitive=False))
def task_set_priority(task_id: str, priority: str):
    """Set a task's priority."""
    with connect() as conn:
        if not set_task_priority(conn, task_id, priority):
            raise _not_found("task", task_id)


def _create_worktree(
    project_dir: str, task_id: str, title: str, base_branch: str = "main"
) -> tuple[str, str]:
    """Create a git worktree for a task. Returns (branch, worktree_path)."""
    from agm.git_ops import create_worktree

    try:
        return create_worktree(project_dir, task_id, title, base_branch=base_branch)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


@task.command("claim")
@click.argument("task_id")
@click.option("--caller", "-c", default="cli", help="Caller identity.")
def task_claim(task_id: str, caller: str):
    """Claim a ready task and create a worktree for execution.

    Requires: task in `ready` status. Creates a git worktree branch,
    transitions to `running`. Normally called automatically by `task run`
    — use this only for manual worktree setup.
    """
    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        if t["status"] != "ready":
            raise click.ClickException(
                f"Task '{task_id}' is '{t['status']}', not 'ready'. "
                f"Only ready tasks can be claimed."
            )

        # Look up project dir: task -> plan -> project
        plan = get_plan_request(conn, t["plan_id"])
        if not plan:
            raise click.ClickException(
                f"Plan '{t['plan_id']}' not found. Run 'agm doctor' to check."
            )
        proj = get_project(conn, plan["project_id"])
        if not proj:
            raise click.ClickException(
                f"Project '{plan['project_id']}' not found. Run 'agm doctor' to check."
            )
        base_branch = _resolve_project_base_branch(conn, proj["id"])

        branch, worktree_path = _create_worktree(
            proj["dir"],
            task_id,
            t["title"],
            base_branch,
        )

        if not claim_task(
            conn,
            task_id,
            caller=caller,
            branch=branch,
            worktree=worktree_path,
        ):
            _remove_worktree(proj["dir"], worktree_path, branch)
            raise click.ClickException(
                f"Failed to claim task '{task_id}'. "
                f"Task may no longer be ready — run 'agm task show {task_id}'."
            )


def _remove_worktree(project_dir: str, worktree_path: str, branch: str):
    """Remove a git worktree and its branch after merge."""
    from agm.git_ops import remove_worktree

    remove_worktree(project_dir, worktree_path, branch)


def _merge_to_main(
    project_dir: str,
    branch: str,
    task_id: str,
    title: str,
    base_branch: str = "main",
    worktree_path: str | None = None,
) -> str | None:
    """Merge a task branch into main. Returns merge commit SHA."""
    from agm.git_ops import merge_to_main

    try:
        return merge_to_main(
            project_dir,
            branch,
            task_id,
            title,
            base_branch=base_branch,
            worktree_path=worktree_path,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


def _emit_task_event(conn, task_id, status, plan_id):
    """Best-effort event emission for CLI task mutations."""
    try:
        from agm.queue import publish_event

        plan = get_plan_request(conn, plan_id)
        proj = get_project(conn, plan["project_id"]) if plan and plan.get("project_id") else None
        publish_event(
            "task:status",
            task_id,
            status,
            project=proj["name"] if proj else "",
            plan_id=plan_id,
            source="cli",
        )
    except Exception as exc:
        log.debug("Failed to emit task event for task %s: %s", task_id, exc, exc_info=True)


def _emit_plan_event(conn, plan_id, status, event_type="plan:status"):
    """Best-effort event emission for CLI plan mutations."""
    try:
        from agm.queue import publish_event

        plan = get_plan_request(conn, plan_id)
        proj = get_project(conn, plan["project_id"]) if plan and plan.get("project_id") else None
        publish_event(
            event_type, plan_id, status, project=proj["name"] if proj else "", source="cli"
        )
    except Exception as exc:
        log.debug("Failed to emit plan event for plan %s: %s", plan_id, exc, exc_info=True)


def _emit_session_event(conn, session_id, status, project_name=""):
    """Best-effort event emission for CLI session mutations."""
    try:
        from agm.queue import publish_event

        publish_event(
            "session:status",
            session_id,
            status,
            project=project_name,
            source="cli",
        )
    except Exception as exc:
        log.debug(
            "Failed to emit session event for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )


def _transition_task(task_id, from_status, to_status, label):
    """Generic task status transition with validation."""
    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        if t["status"] != from_status:
            raise click.ClickException(f"Task '{task_id}' is '{t['status']}', not '{from_status}'.")
        update_task_status(conn, task_id, to_status)
        _emit_task_event(conn, task_id, to_status, t["plan_id"])
        if to_status == "completed":
            _promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, task_id)
            for cid in cascade_cancelled:
                _emit_task_event(conn, cid, "cancelled", t["plan_id"])
            # Clean up worktree and branch
            if t.get("worktree") and t.get("branch"):
                assert t["worktree"] is not None
                assert t["branch"] is not None
                plan = get_plan_request(conn, t["plan_id"])
                if plan:
                    proj = get_project(conn, plan["project_id"])
                    if proj:
                        _remove_worktree(
                            proj["dir"],
                            t["worktree"],
                            t["branch"],
                        )


@task.command("review")
@click.argument("task_id")
def task_review(task_id: str):
    """Submit a running task for review, or launch reviewer on a task in review.

    If task is `running`: transitions to `review` and enqueues the reviewer agent.
    If task is already in `review`: re-enqueues the reviewer (e.g. after a crash).
    """
    with connect() as conn:
        try:
            t = get_task(conn, task_id)
            if not t:
                raise _not_found("task", task_id)
            if t["status"] == "running":
                update_task_status(conn, task_id, "review")
                _emit_task_event(conn, task_id, "review", t["plan_id"])
            elif t["status"] == "review":
                from agm.queue import enqueue_task_review

                try:
                    enqueue_task_review(task_id)
                except Exception as e:
                    raise click.ClickException(
                        f"Failed to enqueue review for task {task_id}: {e}"
                    ) from e
            else:
                raise click.ClickException(
                    f"Task '{task_id}' is '{t['status']}'. "
                    f"Only 'running' (submit for review) or 'review' "
                    f"(launch reviewer) tasks are valid."
                )
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@task.command("reject")
@click.argument("task_id")
def task_reject(task_id: str):
    """Reject a task in review back to running (reviewer requests fixes).

    Requires: task in `review` status. Transitions to `running` so the
    executor can address findings. After 3 rejections, the task fails.
    """
    _transition_task(task_id, "review", "running", "running (rejected)")


@task.command("approve")
@click.argument("task_id")
def task_approve(task_id: str):
    """Approve a reviewed task (ready for merge).

    Requires: task in `review` status. Transitions to `approved`. The task
    is then eligible for `task merge` (auto-triggered or manual).
    """
    _transition_task(task_id, "review", "approved", "approved")


@task.command("merge")
@click.argument("task_id")
def task_merge(task_id: str):
    """Merge an approved task's branch into main and mark completed.

    Requires: task in `approved` status with a worktree branch. Merges
    the task branch into the project's base branch, removes the worktree,
    transitions to `completed`, and unblocks downstream tasks.
    """
    with connect() as conn:
        t, proj, base_branch = _task_merge_load_context(conn, task_id)
        assert t["branch"] is not None
        assert t["worktree"] is not None

        merge_sha = _merge_to_main(
            proj["dir"],
            t["branch"],
            task_id,
            t["title"],
            base_branch=base_branch,
            worktree_path=t["worktree"],
        )

        if merge_sha and isinstance(merge_sha, str):
            from agm.db import set_task_merge_commit

            set_task_merge_commit(conn, task_id, merge_sha)

        _task_merge_complete(conn, task_id, t, proj["dir"])


def _task_merge_load_context(conn, task_id: str) -> tuple[TaskRow, ProjectRow, str]:
    task_row = get_task(conn, task_id)
    if not task_row:
        raise _not_found("task", task_id)
    if task_row["status"] != "approved":
        raise click.ClickException(f"Task '{task_id}' is '{task_row['status']}', not 'approved'.")
    if not task_row.get("worktree") or not task_row.get("branch"):
        raise click.ClickException(
            f"Task '{task_id}' has no worktree/branch. "
            f"Use 'task complete' to mark it done manually."
        )

    plan = get_plan_request(conn, task_row["plan_id"])
    if not plan:
        raise click.ClickException(
            f"Plan '{task_row['plan_id']}' not found. Run 'agm doctor' to check."
        )
    proj = get_project(conn, plan["project_id"])
    if not proj:
        raise click.ClickException(
            f"Project '{plan['project_id']}' not found. Run 'agm doctor' to check."
        )
    return task_row, proj, _resolve_project_base_branch(conn, proj["id"])


def _task_merge_complete(conn, task_id: str, task_row: TaskRow, project_dir: str) -> None:
    update_task_status(conn, task_id, "completed")
    _emit_task_event(conn, task_id, "completed", task_row["plan_id"])

    _promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, task_id)
    for cid in cascade_cancelled:
        _emit_task_event(conn, cid, "cancelled", task_row["plan_id"])

    assert task_row["worktree"] is not None
    assert task_row["branch"] is not None
    _remove_worktree(project_dir, task_row["worktree"], task_row["branch"])


@task.command("complete")
@click.argument("task_id")
def task_complete(task_id: str):
    """Mark an approved task as completed (already merged to main).

    Requires: task in `approved` status. Use this when the branch was already
    merged outside agm (e.g. manual merge). For normal flow, use `task merge`.
    """
    _transition_task(task_id, "approved", "completed", "completed")


@task.command("fail")
@click.argument("task_id")
@click.option("--reason", required=True, help="Failure reason.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def task_fail(task_id: str, reason: str, yes: bool):
    """Mark a non-terminal active task as failed."""
    from agm.db import add_task_log

    normalized_reason = reason.strip()
    if not normalized_reason:
        raise click.ClickException("Reason cannot be blank.")

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        if t["status"] not in {"running", "review", "approved"}:
            raise click.ClickException(
                f"Task '{task_id}' is '{t['status']}'. "
                "Only 'running', 'review', or 'approved' tasks can fail."
            )
        if not yes and not click.confirm("  Proceed?"):
            return

        add_task_log(
            conn,
            task_id=task_id,
            level="ERROR",
            message=f"Failed: {normalized_reason}",
            source="cli",
        )
        set_task_failure_reason(conn, task_id, normalized_reason)
        if not update_task_status(conn, task_id, "failed", record_history=True):
            raise click.ClickException(
                f"Failed to mark task '{task_id}' as failed. "
                f"Status may have changed — run 'agm task show {task_id}'."
            )
        _emit_task_event(conn, task_id, "failed", t["plan_id"])
        _task_cancel_cleanup_worktree(conn, t)
        clear_task_git_refs(conn, task_id)
        _, cascade_cancelled = resolve_blockers_for_terminal_task(
            conn,
            task_id,
            record_history=True,
        )
        for cid in cascade_cancelled:
            _emit_task_event(conn, cid, "cancelled", t["plan_id"])


@task.command("cancel")
@click.argument("task_id", required=False, default=None)
@click.option("--reason", "-r", default=None, help="Reason for cancellation.")
@click.option(
    "--status",
    "by_status",
    default=None,
    type=click.Choice(["failed", "blocked", "ready", "running", "review", "rejected", "approved"]),
    help="Bulk cancel all tasks with this status.",
)
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def task_cancel(
    task_id: str | None,
    reason: str | None,
    by_status: str | None,
    project_name: str | None,
    yes: bool,
):
    """Cancel a task or bulk cancel tasks by status.

    Single mode: TASK_ID required. Cancels any non-completed/cancelled task
    (including failed), cleans up worktree/branch, cascade-cancels downstream.

    Bulk mode: --status required (e.g. --status failed). Cancels all matching
    tasks. Use -p to scope to a project. Shows preview count before applying.
    """
    if task_id and by_status:
        raise click.UsageError("Cannot combine TASK_ID with --status.")
    if not task_id and not by_status:
        raise click.UsageError("Specify TASK_ID or --status.")

    if by_status:
        _task_cancel_bulk(by_status, project_name, reason, yes)
    else:
        assert task_id is not None
        _task_cancel_single(task_id, reason, yes)


def _task_cancel_single(task_id: str, reason: str | None, yes: bool) -> None:
    """Cancel a single task by ID."""
    from agm.db import add_task_log

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        if t["status"] in {"completed", "cancelled"}:
            raise click.ClickException(f"Task '{task_id}' is already '{t['status']}'.")
        if not yes and not click.confirm("  Proceed?"):
            return

        # Stop the running worker job so the Codex thread is interrupted.
        job_id = _task_status_to_job_id(t)
        if job_id:
            try:
                from rq.job import Job

                from agm.queue import get_queue

                queue = get_queue()
                job = Job.fetch(job_id, connection=queue.connection)  # type: ignore[assignment]
                try:
                    job.cancel()
                except Exception as exc:
                    log.debug("Job cancel failed for %s: %s", job_id, exc)
                try:
                    job.stop()  # type: ignore[attr-defined]
                except Exception as exc:
                    log.debug("Job stop failed for %s: %s", job_id, exc)
            except Exception as exc:
                log.debug("Job fetch/cancel failed for %s: %s", job_id, exc)

        _task_cancel_cleanup_worktree(conn, t)
        update_task_status(conn, task_id, "cancelled")
        _emit_task_event(conn, task_id, "cancelled", t["plan_id"])
        msg = f"Cancelled: {reason}" if reason else "Cancelled via CLI"
        add_task_log(conn, task_id=task_id, level="INFO", message=msg, source="cli")
        _promoted, cascade_cancelled = resolve_blockers_for_terminal_task(conn, task_id)
        _task_cancel_emit_cascade(conn, cascade_cancelled, t["plan_id"])


def _task_cancel_bulk(
    status: str,
    project_name: str | None,
    reason: str | None,
    yes: bool,
) -> None:
    """Bulk cancel tasks matching a status filter."""
    from agm.db import add_task_log, list_tasks

    project_id = _resolve_project_id(project_name)

    with connect() as conn:
        tasks = list_tasks(conn, project_id=project_id, status=status)

    if not tasks:
        click.echo(json.dumps({"status": status, "cancelled": 0}))
        return

    if not yes and not click.confirm("  Proceed?", err=True):
        return

    cancelled = 0
    with connect() as conn:
        for t in tasks:
            if t["status"] in {"completed", "cancelled"}:
                continue
            _task_cancel_cleanup_worktree(conn, t)
            update_task_status(conn, t["id"], "cancelled")
            _emit_task_event(conn, t["id"], "cancelled", t["plan_id"])
            msg = f"Bulk cancelled: {reason}" if reason else "Bulk cancelled via CLI"
            add_task_log(conn, task_id=t["id"], level="INFO", message=msg, source="cli")
            resolve_blockers_for_terminal_task(conn, t["id"])
            cancelled += 1

    click.echo(json.dumps({"status": status, "cancelled": cancelled}))


def _task_cancel_cleanup_worktree(conn, task_row: TaskRow) -> None:
    from agm.git_ops import remove_worktree

    plan = get_plan_request(conn, task_row["plan_id"])
    if not (plan and task_row.get("worktree") and task_row.get("branch")):
        return
    assert task_row["worktree"] is not None
    assert task_row["branch"] is not None
    proj = get_project(conn, plan["project_id"])
    if proj:
        remove_worktree(proj["dir"], task_row["worktree"], task_row["branch"])


def _task_cancel_emit_cascade(conn, cascade_cancelled: list[str], plan_id: str) -> None:
    for cid in cascade_cancelled:
        _emit_task_event(conn, cid, "cancelled", plan_id)


@task.command("retry")
@click.argument("task_id")
@click.option("--run", is_flag=True, help="Immediately transition to ready and launch executor.")
@click.option("--caller", default="cli", hidden=True, help="Override caller identity.")
def task_retry(task_id: str, run: bool, caller: str):
    """Retry a failed task."""
    with connect() as conn:
        t, is_merge_retryable = _task_retry_load_retryable_task(conn, task_id)
        # Clean up old worktree+branch so next executor starts fresh
        plan = get_plan_request(conn, t["plan_id"])
        _task_retry_cleanup_worktree(conn, t, plan)
        if t["status"] == "failed":
            if not reset_task_for_retry(conn, task_id):
                raise click.ClickException(
                    f"Failed to reset task '{task_id}'. "
                    f"Status may have changed — run 'agm task show {task_id}'."
                )
            if run:
                _task_retry_run(conn, task_id, t, plan, caller)
            else:
                _emit_task_event(conn, task_id, "blocked", t["plan_id"])
            return

        if is_merge_retryable and not (
            update_task_status(conn, task_id, "ready")
            and _task_retry_reset_exec_context(conn, task_id)
        ):
            raise click.ClickException(
                f"Failed to reset task '{task_id}'. "
                f"Status may have changed — run 'agm task show {task_id}'."
            )
        clear_task_git_refs(conn, task_id)

        if run:
            _task_retry_run(conn, task_id, t, plan, caller)


def _task_retry_reset_exec_context(conn, task_id: str) -> bool:
    """Reset execution context for merge-failure retries."""
    return set_task_thread_id(conn, task_id, "") and set_task_reviewer_thread_id(
        conn,
        task_id,
        "",
    )


def _task_retry_load_retryable_task(conn, task_id: str) -> tuple[TaskRow, bool]:
    t = get_task(conn, task_id)
    if not t:
        raise _not_found("task", task_id)
    if t["status"] == "failed":
        return t, False
    if t["status"] == "approved":
        has_merge_failure_signal, _ = task_merge_failure_signal(conn, task_id)
        if has_merge_failure_signal:
            return t, True
        raise click.ClickException(
            f"Task '{task_id}' is 'approved' but has no merge-failure signal."
        )
    raise click.ClickException(
        f"Task '{task_id}' is '{t['status']}', not 'failed'. Only failed tasks can be retried."
    )


def _task_retry_cleanup_worktree(conn, task_row: TaskRow, plan: PlanRow | None) -> None:
    from agm.git_ops import remove_worktree

    if not (plan and task_row.get("worktree") and task_row.get("branch")):
        return
    assert task_row["worktree"] is not None
    assert task_row["branch"] is not None
    proj = get_project(conn, plan["project_id"])
    if proj:
        remove_worktree(proj["dir"], task_row["worktree"], task_row["branch"])


def _task_retry_run(
    conn, task_id: str, task_row: TaskRow, plan: PlanRow | None, caller: str
) -> None:
    from agm.git_ops import remove_worktree
    from agm.queue import enqueue_task_execution

    # Transition to ready, claim, and launch executor
    update_task_status(conn, task_id, "ready")
    _emit_task_event(conn, task_id, "running", task_row["plan_id"])
    plan = plan or get_plan_request(conn, task_row["plan_id"])
    if not plan:
        raise click.ClickException(
            f"Plan '{task_row['plan_id']}' not found. Run 'agm doctor' to check."
        )
    proj = get_project(conn, plan["project_id"])
    if not proj:
        raise click.ClickException(
            f"Project '{plan['project_id']}' not found. Run 'agm doctor' to check."
        )
    base_branch = _resolve_project_base_branch(conn, proj["id"])
    branch, worktree_path = _create_worktree(
        proj["dir"],
        task_id,
        task_row["title"],
        base_branch,
    )
    if not claim_task(conn, task_id, caller=caller, branch=branch, worktree=worktree_path):
        _remove_worktree(proj["dir"], worktree_path, branch)
        raise click.ClickException(
            f"Failed to claim task '{task_id}'. "
            f"Task may no longer be ready — run 'agm task show {task_id}'."
        )

    try:
        enqueue_task_execution(task_id)
    except Exception as e:
        from agm.db import add_task_log

        # Clean up the worktree we just created
        remove_worktree(proj["dir"], worktree_path, branch)
        update_task_status(conn, task_id, "failed")
        add_task_log(
            conn,
            task_id=task_id,
            level="ERROR",
            message=f"Enqueue failed: {e}",
            source="cli",
        )
        set_task_failure_reason(conn, task_id, f"Enqueue failed: {e}")
        _emit_task_event(conn, task_id, "failed", task_row["plan_id"])
        resolve_blockers_for_terminal_task(conn, task_id)
        raise click.ClickException(f"Failed to enqueue: {e}") from e


@task.command("cleanup")
@click.option("--project", "-p", "project_name", required=True, help="Project name or ID.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def task_cleanup(project_name: str, yes: bool):
    """Remove orphaned git worktrees and branches for terminal-state tasks."""
    with connect() as conn:
        proj = get_project(conn, project_name)
        if not proj:
            raise _not_found("project", project_name)

        candidates = list_task_cleanup_candidates(conn, proj["id"])
        if not candidates:
            return

        if not yes and not click.confirm("  Proceed?"):
            return

        for t in candidates:
            with contextlib.suppress(Exception):
                assert t["worktree"] is not None
                assert t["branch"] is not None
                _remove_worktree(proj["dir"], t["worktree"], t["branch"])

            with contextlib.suppress(Exception):
                clear_task_git_refs(conn, t["id"])


@task.command("run")
@click.argument("task_id")
@click.option("--caller", "-c", default="cli", help="Caller identity.")
def task_run(task_id: str, caller: str):
    """Launch the executor agent for a task.

    If the task is 'ready', auto-claims it (creates worktree) and enqueues.
    If the task is 'running' (already claimed), enqueues directly.
    """
    with connect() as conn:
        try:
            t = _task_run_prepare(conn, task_id, caller)
            _task_run_enqueue(conn, task_id, t["plan_id"])
            _emit_task_event(conn, task_id, "running", t["plan_id"])
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


def _task_run_prepare(conn, task_id: str, caller: str) -> TaskRow:
    task_row = get_task(conn, task_id)
    if not task_row:
        raise _not_found("task", task_id)

    if task_row["status"] == "ready":
        _task_run_claim_ready(conn, task_id, task_row, caller)
        return task_row

    if task_row["status"] == "running":
        if not task_row.get("worktree"):
            raise click.ClickException(
                f"Task '{task_id}' is running but has no worktree. "
                f"Use 'task retry' then 'task run' to start fresh."
            )
        return task_row

    raise click.ClickException(
        f"Task '{task_id}' is '{task_row['status']}'. Only 'ready' or 'running' tasks can be run."
    )


def _task_run_claim_ready(conn, task_id: str, task_row: TaskRow, caller: str) -> None:
    plan = get_plan_request(conn, task_row["plan_id"])
    if not plan:
        raise click.ClickException(
            f"Plan '{task_row['plan_id']}' not found. Run 'agm doctor' to check."
        )
    proj = get_project(conn, plan["project_id"])
    if not proj:
        raise click.ClickException(
            f"Project '{plan['project_id']}' not found. Run 'agm doctor' to check."
        )
    base_branch = _resolve_project_base_branch(conn, proj["id"])

    branch, worktree_path = _create_worktree(
        proj["dir"],
        task_id,
        task_row["title"],
        base_branch,
    )
    if not claim_task(
        conn,
        task_id,
        caller=caller,
        branch=branch,
        worktree=worktree_path,
    ):
        _remove_worktree(proj["dir"], worktree_path, branch)
        raise click.ClickException(
            f"Failed to claim task '{task_id}'. "
            f"Task may no longer be ready — run 'agm task show {task_id}'."
        )


def _task_run_enqueue(conn, task_id: str, plan_id: str):
    from agm.db import add_task_log
    from agm.queue import enqueue_task_execution

    try:
        return enqueue_task_execution(task_id)
    except Exception as e:
        # Clean up worktree before marking failed
        task = get_task(conn, task_id)
        if task and task.get("worktree") and task.get("branch"):
            assert task["worktree"] is not None
            assert task["branch"] is not None
            plan = get_plan_request(conn, task["plan_id"])
            proj = get_project(conn, plan["project_id"]) if plan else None
            if proj:
                _remove_worktree(
                    proj["dir"],
                    task["worktree"],
                    task["branch"],
                )
        update_task_status(conn, task_id, "failed")
        add_task_log(
            conn,
            task_id=task_id,
            level="ERROR",
            message=f"Enqueue failed: {e}",
            source="cli",
        )
        set_task_failure_reason(conn, task_id, f"Enqueue failed: {e}")
        _emit_task_event(conn, task_id, "failed", plan_id)
        resolve_blockers_for_terminal_task(conn, task_id)
        raise click.ClickException(f"Failed to enqueue task {task_id}: {e}") from e


_MAX_FAILURE_OUTPUT_CHARS = 2000


async def _steer_task_live_turn(
    thread_id: str, active_turn_id: str, content: str
) -> dict[str, Any]:
    return await steer_active_turn(
        thread_id=thread_id,
        active_turn_id=active_turn_id,
        content=content,
    )


@task.command("steer")
@click.argument("task_id")
@click.argument("content")
@click.option("--sender", default=None, help="Sender in role:id format.")
@click.option("--recipient", default=None, help="Recipient in role:id format.")
@click.option(
    "--live/--no-live",
    default=True,
    help="Attempt immediate turn/steer on an active running turn.",
)
@click.option("--metadata", default=None, help="JSON metadata object.")
def task_steer(
    task_id: str,
    content: str,
    sender: str | None,
    recipient: str | None,
    live: bool,
    metadata: str | None,
) -> None:
    """Post steer guidance for a task and optionally apply it mid-turn."""
    normalized_content = content.strip()
    if not normalized_content:
        raise click.ClickException("content must be non-empty")

    with connect() as conn:
        task_row = get_task(conn, task_id)
        if not task_row:
            raise _not_found("task", task_id)
        plan = get_plan_request(conn, task_row["plan_id"])
        if not plan or not plan.get("session_id"):
            raise click.ClickException(f"Task '{task_id}' has no session channel.")
        session_id = cast(str, plan["session_id"])

        from agm.db import add_channel_message

        normalized_sender = _session_normalize_sender(sender, fallback_suffix="cli")
        normalized_recipient = recipient or default_executor_recipient(task_id)
        if metadata is None:
            metadata_json = json.dumps(
                {
                    "phase": "execution",
                    "status": "steer_requested",
                    "task_id": task_id,
                    "live": live,
                },
                sort_keys=True,
            )
        else:
            metadata_json = _session_serialize_metadata(metadata)

        msg = add_channel_message(
            conn,
            session_id=session_id,
            kind="steer",
            sender=normalized_sender,
            recipient=normalized_recipient,
            content=normalized_content,
            metadata=metadata_json,
        )
        _session_emit_message_event(
            conn,
            session_id=session_id,
            kind="steer",
            sender=normalized_sender,
            recipient=normalized_recipient,
            message_id=msg["id"],
            metadata_json=metadata_json,
        )

        steer_record: dict[str, Any] = {
            "task_id": task_id,
            "session_id": session_id,
            "message_id": msg["id"],
            "sender": normalized_sender,
            "recipient": normalized_recipient,
            "content": normalized_content,
            "reason": "manual",
            "metadata": metadata_json,
            "live_requested": live,
            "live_applied": False,
        }

        if not live:
            add_task_steer(conn, **steer_record)
            return
        if task_row.get("status") != "running":
            steer_record["live_error"] = "task is not running"
            add_task_steer(conn, **steer_record)
            return
        thread_id = task_row.get("thread_id")
        active_turn_id = task_row.get("active_turn_id")
        steer_record["thread_id"] = thread_id
        steer_record["expected_turn_id"] = active_turn_id
        if not thread_id or not active_turn_id:
            steer_record["live_error"] = "task has no active turn"
            add_task_steer(conn, **steer_record)
            return
        try:
            response = asyncio.run(
                _steer_task_live_turn(thread_id, active_turn_id, normalized_content)
            )
            steer_record["live_applied"] = True
            steer_record["applied_turn_id"] = response.get("turnId")
        except Exception as exc:
            steer_record["live_error"] = str(exc)
            log.debug("Live steer failed for task %s", task_id, exc_info=True)
        add_task_steer(conn, **steer_record)


@task.command("steer-log")
@click.argument("task_id")
@click.option("--limit", "-n", default=100, type=int, help="Max steer rows.")
@click.option("--offset", default=0, type=int, help="Row offset.")
def task_steer_log(task_id: str, limit: int, offset: int) -> None:
    """List persisted steer history for a task."""
    if limit <= 0:
        raise click.ClickException("limit must be > 0")
    if offset < 0:
        raise click.ClickException("offset must be >= 0")
    with connect() as conn:
        if not get_task(conn, task_id):
            raise _not_found("task", task_id)
        rows = list_task_steers(conn, task_id=task_id, limit=limit, offset=offset)
        click.echo(
            json.dumps(
                {
                    "task_id": task_id,
                    "limit": limit,
                    "offset": offset,
                    "count": len(rows),
                    "items": [dict(row) for row in rows],
                },
                indent=2,
            )
        )


@task.command("check")
@click.argument("task_id")
def task_check(task_id: str):
    """Dry-run quality gate against a task's worktree (does not change task status).

    Uses the project's quality gate config if set, otherwise defaults.
    """
    from agm.db import get_project_quality_gate
    from agm.jobs import _run_quality_checks

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        wt = t.get("worktree")
        if not wt:
            raise click.ClickException(f"Task '{task_id}' has no worktree.")
        if not os.path.isdir(wt):
            raise click.ClickException(f"Worktree not found: {wt}")
        plan = get_plan_request(conn, t["plan_id"])
        quality_gate_json = None
        if plan:
            quality_gate_json = get_project_quality_gate(conn, plan["project_id"])

    from agm.jobs import _serialize_quality_gate_result

    qg_result = _run_quality_checks(wt, quality_gate_json=quality_gate_json)
    click.echo(json.dumps(_serialize_quality_gate_result(qg_result), indent=2))
    if not qg_result.passed:
        raise click.ClickException("Quality checks failed.")


@task.command("diff")
@click.argument("task_id")
def task_diff(task_id: str):
    """Show the git diff for a task's branch against the base branch."""
    import json

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)
        branch = t.get("branch")
        merge_commit = t.get("merge_commit")
        if not branch and not merge_commit:
            raise click.ClickException(f"Task '{task_id}' has no branch or merge commit.")
        plan = get_plan_request(conn, t["plan_id"])
        if not plan:
            raise click.ClickException(f"Plan '{t['plan_id']}' not found.")
        proj = get_project(conn, plan["project_id"])
        if not proj:
            raise click.ClickException(f"Project '{plan['project_id']}' not found.")
        base_branch = get_project_base_branch(conn, proj["id"])
        project_dir = proj["dir"]

    # Try live branch first, fall back to merge commit
    diff_text = _task_diff_resolve_text(project_dir, base_branch, branch, merge_commit)

    click.echo(json.dumps({"task_id": task_id, "diff": diff_text}))


def _task_diff_run_git_diff(project_dir: str, revspec: str) -> str:
    result = subprocess.run(
        ["git", "diff", revspec],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    return ""


def _task_diff_resolve_text(
    project_dir: str,
    base_branch: str,
    branch: str | None,
    merge_commit: str | None,
) -> str:
    if branch:
        diff_text = _task_diff_run_git_diff(project_dir, f"{base_branch}...{branch}")
        if diff_text:
            return diff_text
    if merge_commit:
        # Branch deleted — use the stored merge commit
        return _task_diff_run_git_diff(project_dir, f"{merge_commit}^1..{merge_commit}")
    return ""


@task.command("refresh")
@click.option("--project", "-p", "project_name", required=True, help="Project name or ID.")
@click.option(
    "--backend",
    default=None,
    type=click.Choice(sorted(VALID_BACKENDS)),
    help="Backend (default: project default or codex).",
)
@click.argument("prompt", required=False, default=None)
def task_refresh(project_name: str, backend: str | None, prompt: str | None):
    """Invoke the task agent to review and clean up tasks."""
    from agm.queue import enqueue_task_refresh

    with connect() as conn:
        try:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)

            backend = resolve_backend(backend)

            enqueue_task_refresh(proj["id"], prompt, backend=backend)
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


@task.command("logs")
@click.argument("task_id")
@click.option("--level", "-l", default=None, help="Filter by log level.")
@click.option("--tail", "-n", "tail", type=int, default=None, help="Show only the last N logs.")
@click.option(
    "--worker",
    is_flag=True,
    help="Show executor worker stdout/stderr instead of structured task logs.",
)
def task_logs_cmd(task_id: str, level: str | None, tail: int | None, worker: bool):
    """Show logs for a task."""
    import json

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)

        if worker:
            from agm.queue import LOG_DIR

            # Worker logs use job_id format: exec-{task_id}
            log_file = LOG_DIR / f"exec-{task_id}.log"
            if not log_file.exists():
                click.echo(json.dumps({"task_id": task_id, "worker_log": None}))
                return
            click.echo(json.dumps({"task_id": task_id, "worker_log": log_file.read_text()}))
            return

        logs = list_task_logs(conn, task_id, level=level)
        if tail is not None:
            logs = logs[-tail:]
    click.echo(
        json.dumps(
            {
                "task_id": task_id,
                "level": level,
                "count": len(logs),
                "logs": normalize_logs(logs),
            },
            indent=2,
            default=str,
        )
    )


@task.command("trace")
@click.argument("task_id")
@click.option("--type", "event_type", default=None, help="Filter by event type.")
@click.option("--stage", default=None, help="Filter by pipeline stage.")
@click.option("--summary", is_flag=True, help="Show aggregated summary.")
@click.option("--tail", "-n", "tail", type=int, default=None, help="Show last N events.")
def task_trace_cmd(
    task_id: str,
    event_type: str | None,
    stage: str | None,
    summary: bool,
    tail: int | None,
):
    """Show execution trace events for a task."""
    import json

    from agm.db import get_trace_summary, list_trace_events

    with connect() as conn:
        t = get_task(conn, task_id)
        if not t:
            raise _not_found("task", task_id)

        if summary:
            data = get_trace_summary(conn, "task", task_id)
            data["task_id"] = task_id
            click.echo(json.dumps(data, indent=2, default=str))
            return

        events = list_trace_events(
            conn,
            "task",
            task_id,
            event_type=event_type,
            stage=stage,
        )
        if tail is not None:
            events = events[-tail:]
        click.echo(
            json.dumps(
                {"task_id": task_id, "count": len(events), "events": events},
                indent=2,
                default=str,
            )
        )


@task.command("watch")
@click.argument("task_id", required=False)
@click.option("--plan", "plan_id", default=None, help="Filter by plan ID.")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option(
    "--all",
    "-a",
    "show_all",
    is_flag=True,
    help="Include completed and cancelled tasks.",
)
def task_watch(
    task_id: str | None,
    plan_id: str | None,
    project_name: str | None,
    show_all: bool,
):
    """Watch task progress (single JSON snapshot)."""
    provided_scopes = [
        name
        for name, value in (("task", task_id), ("plan", plan_id), ("project", project_name))
        if value
    ]
    if not provided_scopes:
        raise click.ClickException("Provide TASK_ID, --plan, or --project.")
    if len(provided_scopes) > 1:
        raise click.ClickException("Provide exactly one scope: TASK_ID, --plan, or --project.")

    _task_watch_json_loop(
        task_id,
        plan_id,
        project_name,
        show_all,
    )


def _task_watch_scope_for_task(conn, task_id: str, plan_backends: dict[str, str]):
    task_row = get_task(conn, task_id)
    if not task_row:
        raise _not_found("task", task_id)
    task_plan = get_plan_request(conn, task_row["plan_id"])
    task_backend = (task_plan.get("backend") if task_plan else None) or "codex"
    plan_backends[task_row["plan_id"]] = task_backend
    task_id_short = watch_short_id(task_row["id"])
    scope = {
        "type": "task",
        "task_id": task_row["id"],
        "task_id_short": task_id_short,
        "plan_id": task_row.get("plan_id"),
        "session_id": task_plan.get("session_id") if task_plan else None,
        "backend": task_backend,
    }
    return f"task {task_id_short}", scope, {"task_id": task_row["id"]}


def _task_watch_scope_for_project(conn, project_name: str):
    proj = get_project(conn, project_name)
    if not proj:
        raise _not_found("project", project_name)
    project_id_short = watch_short_id(proj["id"])
    scope = {
        "type": "project",
        "project_id": proj["id"],
        "project_id_short": project_id_short,
        "project_name": proj["name"],
    }
    return f"project {proj['name']} ({project_id_short})", scope, {"project_id": proj["id"]}


def _task_watch_scope_for_plan(conn, plan_id: str | None, plan_backends: dict[str, str]):
    plan_id_value = plan_id or ""
    scope = {
        "type": "plan",
        "plan_id": plan_id,
        "plan_id_short": watch_short_id(plan_id_value),
    }
    if plan_id:
        resolved_plan = get_plan_request(conn, plan_id)
        if not resolved_plan:
            raise _not_found("plan", plan_id)
        backend = resolved_plan.get("backend") or "codex"
        scope["plan_status"] = resolved_plan.get("status")
        scope["session_id"] = resolved_plan.get("session_id")
        scope["backend"] = backend
        plan_backends[plan_id] = backend
        scope["title"] = resolved_plan.get("prompt") or ""
        scope["title_truncated"] = watch_truncate(
            resolved_plan.get("prompt") or "",
            60,
        )
    return f"plan {watch_short_id(plan_id_value)}", scope, {"plan_id": plan_id}


def _task_watch_resolve_scope(conn, task_id, plan_id, project_name):
    """Resolve task watch scope. Returns (scope_context, scope, event_kwargs)."""
    plan_backends: dict[str, str] = {}

    if task_id:
        scope_context, scope, event_kwargs = _task_watch_scope_for_task(
            conn, task_id, plan_backends
        )
        return scope_context, scope, event_kwargs, plan_backends

    if project_name:
        scope_context, scope, event_kwargs = _task_watch_scope_for_project(conn, project_name)
        return scope_context, scope, event_kwargs, plan_backends

    scope_context, scope, event_kwargs = _task_watch_scope_for_plan(conn, plan_id, plan_backends)
    return scope_context, scope, event_kwargs, plan_backends


def _task_watch_enrich_with_rejection_count(conn, task: TaskRow) -> dict:
    enriched = dict(task)
    enriched["rejection_count"] = get_task_rejection_count(conn, task["id"])
    return enriched


def _task_watch_load_tasks(conn, task_id, plan_id, project_name):
    """Load tasks for the given scope."""
    if task_id:
        task_row = get_task(conn, task_id)
        return [_task_watch_enrich_with_rejection_count(conn, task_row)] if task_row else []
    project_id = None
    if project_name:
        proj = get_project(conn, project_name)
        if proj:
            project_id = proj["id"]
    tasks = list_tasks(conn, plan_id=plan_id, project_id=project_id)
    return [_task_watch_enrich_with_rejection_count(conn, task_row) for task_row in tasks]


def _task_watch_fill_plan_backends(conn, tasks, plan_backends):
    for task_row in tasks:
        pid = task_row.get("plan_id")
        if not pid or pid in plan_backends:
            continue
        plan_row = get_plan_request(conn, pid)
        plan_backends[pid] = (plan_row.get("backend") if plan_row else None) or "codex"


def _task_watch_visible_tasks(task_id, tasks, show_all):
    if task_id or show_all:
        return tasks
    return [t for t in tasks if not is_effectively_terminal_task(t)]


def _task_watch_elapsed_runtime(task_runtimes, tasks):
    scope_rt_secs = sum(task_runtimes.get(t["id"], 0) for t in tasks) or None
    return format_duration_seconds(scope_rt_secs) if scope_rt_secs else "-"


def _task_watch_json_loop(task_id, plan_id, project_name, show_all):
    """JSON mode: emit a single snapshot.

    Always exits after one snapshot — consumers (e.g. agm-web) handle
    their own polling cadence.
    """
    import json

    with connect() as conn:
        scope_context, scope, event_kwargs, plan_backends = _task_watch_resolve_scope(
            conn, task_id, plan_id, project_name
        )
        tasks = _task_watch_load_tasks(conn, task_id, plan_id, project_name)
        _task_watch_fill_plan_backends(conn, tasks, plan_backends)
        blocker_counts = {t["id"]: get_unresolved_block_count(conn, t["id"]) for t in tasks}
        events = list_recent_task_events(
            conn,
            **event_kwargs,  # type: ignore[arg-type]
            limit=WATCH_RECENT_EVENT_FETCH_LIMIT,
        )
        t_runtimes = bulk_active_runtime_seconds(conn, "task", TASK_ACTIVE_STATUSES)
        thread_status_by_task = latest_task_thread_statuses(conn, tasks)

    visible = _task_watch_visible_tasks(task_id, tasks, show_all)
    scope_rt = _task_watch_elapsed_runtime(t_runtimes, tasks)

    terminal_state = task_watch_terminal_state(tasks)
    click.echo(
        json.dumps(
            build_task_watch_snapshot(
                scope=scope,
                tasks=tasks,
                visible_tasks=visible,
                recent_events=events,
                blocker_counts=blocker_counts,
                plan_backends=plan_backends,
                watch_elapsed=scope_rt,
                terminal_state=terminal_state,
                thread_status_by_task=thread_status_by_task,
            ),
            indent=2,
            default=str,
        )
    )


# -- queue (monitoring only) --


@main.group()
def queue():
    """Monitor job queue health and clean up failures."""


@queue.command("status")
def queue_status():
    """Show job counts per queue."""
    import json

    from redis.exceptions import RedisError

    from agm.queue import get_queue_counts

    try:
        counts = get_queue_counts()
    except RedisError:
        raise click.ClickException("Redis unavailable — cannot access queue data.") from None

    click.echo(json.dumps(counts, indent=2, default=str))


@queue.command("inspect")
@click.option("--queue-name", "-q", default=None, help="Inspect only one queue.")
@click.option("--limit", "-n", default=None, type=int, help="Limit returned rows.")
def queue_inspect(queue_name: str | None, limit: int | None):
    """Inspect live queued/running jobs with entity linkage and worker heartbeat."""
    import json

    from redis.exceptions import RedisError

    from agm.queue import inspect_queue_jobs

    if limit is not None and limit < 0:
        raise click.ClickException("limit must be >= 0")
    try:
        rows = inspect_queue_jobs(queue_name, limit=limit)
    except RedisError:
        raise click.ClickException("Redis unavailable — cannot access queue data.") from None
    click.echo(json.dumps(rows, indent=2, default=str))


@queue.command("flush")
@click.option(
    "--queue-name",
    "-q",
    default=None,
    help="Queue to flush (default: all queues).",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def queue_flush(queue_name: str | None, yes: bool):
    """Clear failed jobs from queue registries."""
    from redis.exceptions import RedisError

    from agm.queue import flush_failed_jobs, get_queue_counts

    try:
        counts = get_queue_counts()
    except RedisError:
        raise click.ClickException("Redis unavailable — cannot access queue data.") from None
    if queue_name is None:
        total_failed = sum(stats.get("failed", 0) for stats in counts.values())
    else:
        total_failed = counts.get(queue_name, {}).get("failed", 0)

    if total_failed == 0:
        return

    if not yes and not click.confirm("  Proceed?"):
        return
    flush_failed_jobs(queue_name)


@queue.command("clean")
@click.option("--logs", "clean_logs", is_flag=True, help="Delete worker log files.")
@click.option("--finished", "clean_finished", is_flag=True, help="Clear finished job metadata.")
@click.option("--all", "clean_all", is_flag=True, help="Clean both logs and finished jobs.")
@click.option(
    "--older-than",
    default=0,
    type=int,
    metavar="DAYS",
    help="Only delete logs older than DAYS (default: all).",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def queue_clean(
    clean_logs: bool,
    clean_finished: bool,
    clean_all: bool,
    older_than: int,
    yes: bool,
):
    """Clean stale queue artifacts: old logs and finished job metadata.

    Must specify --logs, --finished, or --all.
    Finished job metadata accumulates in Redis after jobs complete.
    Worker logs accumulate in ~/.config/agm/logs/.
    """
    if not clean_logs and not clean_finished and not clean_all:
        raise click.UsageError("Specify --logs, --finished, or --all.")

    do_logs = clean_logs or clean_all
    do_finished = clean_finished or clean_all

    if not yes and not click.confirm("  Proceed?", err=True):
        return

    result: dict = {}

    if do_finished:
        from redis.exceptions import RedisError

        from agm.queue import clean_finished_jobs

        try:
            finished = clean_finished_jobs()
        except RedisError:
            raise click.ClickException("Redis unavailable — cannot clean finished jobs.") from None
        result["finished_jobs"] = finished

    if do_logs:
        from agm.queue import clean_log_files

        result["logs"] = clean_log_files(older_than)

    click.echo(json.dumps(result, indent=2))


@queue.command("failed")
@click.option(
    "--queue-name",
    "-q",
    default=None,
    help="Queue to inspect (default: all queues).",
)
def queue_failed(queue_name: str | None):
    """List failed jobs with error details."""
    from redis.exceptions import RedisError

    from agm.queue import get_failed_jobs

    try:
        jobs = get_failed_jobs(queue_name)
    except RedisError:
        raise click.ClickException("Redis unavailable — cannot access queue data.") from None
    click.echo(json.dumps(jobs))


# -- quick mode --


def _claim_quick_task(conn, proj: ProjectRow, task: TaskRow, caller: str) -> None:
    """Claim quick-mode task with worktree."""
    base_branch = _resolve_project_base_branch(conn, proj["id"])
    branch, worktree_path = _create_worktree(
        proj["dir"],
        task["id"],
        task["title"],
        base_branch,
    )
    if claim_task(
        conn,
        task["id"],
        caller=caller,
        branch=branch,
        worktree=worktree_path,
    ):
        return
    raise click.ClickException(
        f"Failed to claim task '{task['id']}'. "
        f"Task may no longer be ready — run 'agm task show {task['id']}'."
    )


def _enqueue_quick_task_execution(conn, task_id: str, plan_id: str):
    """Enqueue quick task execution or mark task failed with diagnostics."""
    from agm.db import add_task_log
    from agm.queue import enqueue_task_execution

    try:
        return enqueue_task_execution(task_id)
    except Exception as e:
        update_task_status(conn, task_id, "failed")
        add_task_log(
            conn,
            task_id=task_id,
            level="ERROR",
            message=f"Enqueue failed: {e}",
            source="cli",
        )
        set_task_failure_reason(conn, task_id, f"Enqueue failed: {e}")
        _emit_task_event(conn, task_id, "failed", plan_id)
        resolve_blockers_for_terminal_task(conn, task_id)
        raise click.ClickException(f"Failed to enqueue task {task_id}: {e}") from e


def _quick_mode_flags(skip_review: bool, skip_merge: bool) -> list[str]:
    """Return display flags for quick mode invocation."""
    flags: list[str] = []
    if skip_review:
        flags.append("skip-review")
    if skip_merge:
        flags.append("skip-merge")
    return flags


def _quick_mode_expected_terminal_status(skip_merge: bool) -> str:
    """Return eventual terminal status for quick-mode task based on flags."""
    return "approved" if skip_merge else "completed"


@main.command("do")
@click.argument("prompt", callback=_validate_prompt)
@click.option("--project", "-p", "project_name", required=True, help="Project name or ID.")
@click.option("--title", "-t", default=None, help="Task title (default: first 60 chars of prompt).")
@click.option("--files", "-f", multiple=True, help="Files to work on (can be repeated).")
@click.option("--no-review", "skip_review", is_flag=True, help="Skip reviewer agent.")
@click.option("--no-merge", "skip_merge", is_flag=True, help="Stop at approved (don't auto-merge).")
@click.option(
    "--caller",
    default="cli",
    type=CALLER_TYPE,
    help="Client that invoked this.",
)
@click.option(
    "--backend",
    default=None,
    type=click.Choice(sorted(VALID_BACKENDS)),
    help="Backend for execution (default: project default or codex).",
)
def do_quick(
    prompt: str,
    project_name: str,
    title: str | None,
    files: tuple[str, ...],
    skip_review: bool,
    skip_merge: bool,
    caller: str,
    backend: str | None,
):
    """Quick single-task execution — direct executor, reviewer, merger.

    Creates a synthetic plan and a single ready task, then launches
    the executor immediately. Monitor with 'task watch TASK_ID'.
    """
    from agm.db import create_quick_plan_and_task

    with connect() as conn:
        try:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)

            backend = resolve_backend(backend)

            task_title = title or prompt[:60]
            files_json = json.dumps(list(files)) if files else None

            plan, task = create_quick_plan_and_task(
                conn,
                project_id=proj["id"],
                prompt=prompt,
                title=task_title,
                description=prompt,
                caller=caller,
                backend=backend,
                files=files_json,
                skip_review=skip_review,
                skip_merge=skip_merge,
            )
            session = create_session(
                conn,
                project_id=proj["id"],
                trigger="do",
                trigger_prompt=prompt,
            )
            set_plan_session_id(conn, plan["id"], session["id"])
            update_session_status(conn, session["id"], "active")
            _emit_session_event(conn, session["id"], "active", project_name=proj["name"])

            _claim_quick_task(conn, proj, cast(TaskRow, task), caller)
            execution_job = _enqueue_quick_task_execution(conn, task["id"], plan["id"])
            click.echo(
                json.dumps(
                    {
                        "plan_id": plan["id"],
                        "task_id": task["id"],
                        "session_id": session["id"],
                        "execution_job_id": execution_job.id,
                        "title": task["title"],
                        "status": "running",
                        "expected_terminal_status": _quick_mode_expected_terminal_status(
                            skip_merge
                        ),
                        "flags": _quick_mode_flags(skip_review, skip_merge),
                    }
                )
            )
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(str(e)) from e


# -- sessions --


@main.group()
def session():
    """View sessions and inter-agent communication channels."""


@session.command("list")
@click.option("--project", "-p", "project_name", default=None, help="Filter by project.")
@click.option("--status", "-s", default=None, help="Filter by status.")
@click.option("--all", "show_all", is_flag=True, help="Include completed/failed sessions.")
def session_list(project_name: str | None, status: str | None, show_all: bool):
    """List sessions."""
    with connect() as conn:
        reconcile_session_statuses(conn)
        project_id = None
        if project_name:
            proj = get_project(conn, project_name)
            if not proj:
                raise _not_found("project", project_name)
            project_id = proj["id"]

        if status:
            sessions = list_sessions(conn, project_id=project_id, status=status)
        elif show_all:
            sessions = list_sessions(conn, project_id=project_id)
        else:
            sessions = list_sessions(conn, project_id=project_id, statuses=["open", "active"])

        click.echo(json.dumps([dict(s) for s in sessions], indent=2))


@session.command("show")
@click.argument("session_id")
def session_show(session_id: str):
    """Show session details and linked plans."""
    with connect() as conn:
        reconcile_session_statuses(conn, session_id=session_id)
        s = get_session(conn, session_id)
        if not s:
            raise _not_found("session", session_id)

        plans = list_plan_requests(conn, session_id=session_id)
        click.echo(json.dumps({"session": dict(s), "plans": [dict(p) for p in plans]}, indent=2))


def _session_serialize_metadata(metadata: str | None) -> str | None:
    if metadata is None:
        return None
    raw = metadata.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException("metadata must be valid JSON") from exc
    return json.dumps(parsed, sort_keys=True)


def _session_normalize_sender(sender: str | None, *, fallback_suffix: str) -> str:
    raw = (sender or "").strip()
    if not raw:
        return f"operator:{fallback_suffix}"
    if ":" in raw:
        return raw
    return f"{raw}:{fallback_suffix}"


def _session_emit_message_event(
    conn,
    *,
    session_id: str,
    kind: str,
    sender: str,
    recipient: str | None,
    message_id: str,
    metadata_json: str | None,
) -> None:
    from agm.queue import publish_event

    session_row = get_session(conn, session_id)
    project_name = ""
    if session_row:
        project_row = get_project(conn, session_row.get("project_id", ""))
        project_name = project_row["name"] if project_row else ""

    metadata_payload: Any = None
    if metadata_json:
        try:
            metadata_payload = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata_payload = metadata_json
    publish_event(
        "session:message",
        session_id,
        kind,
        project=project_name,
        extra={
            "session_id": session_id,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "message_id": message_id,
            "metadata": metadata_payload,
        },
    )


@session.command("post")
@click.argument("session_id")
@click.argument("content")
@click.option(
    "--kind",
    default="context",
    type=click.Choice(sorted(VALID_MESSAGE_KINDS), case_sensitive=False),
    help="Message kind.",
)
@click.option("--sender", default=None, help="Sender in role:id format.")
@click.option("--recipient", default=None, help="Recipient in role:id format.")
@click.option("--metadata", default=None, help="JSON metadata object.")
def session_post(
    session_id: str,
    content: str,
    kind: str,
    sender: str | None,
    recipient: str | None,
    metadata: str | None,
) -> None:
    """Post a message into a session channel."""
    normalized_content = content.strip()
    if not normalized_content:
        raise click.ClickException("content must be non-empty")
    with connect() as conn:
        reconcile_session_statuses(conn, session_id=session_id)
        session_row = get_session(conn, session_id)
        if not session_row:
            raise _not_found("session", session_id)
        from agm.db import add_channel_message

        normalized_sender = _session_normalize_sender(sender, fallback_suffix="cli")
        metadata_json = _session_serialize_metadata(metadata)
        msg = add_channel_message(
            conn,
            session_id=session_id,
            kind=kind.lower(),
            sender=normalized_sender,
            content=normalized_content,
            recipient=recipient,
            metadata=metadata_json,
        )
        _session_emit_message_event(
            conn,
            session_id=session_id,
            kind=kind.lower(),
            sender=normalized_sender,
            recipient=recipient,
            message_id=msg["id"],
            metadata_json=metadata_json,
        )


@session.command("messages")
@click.argument("session_id")
@click.option("--kind", "-k", default=None, help="Filter by message kind.")
@click.option("--sender", default=None, help="Filter by sender.")
@click.option("--recipient", default=None, help="Filter by recipient.")
@click.option("--limit", "-n", "msg_limit", default=100, type=int, help="Max messages.")
@click.option("--offset", default=0, type=int, help="Starting row offset.")
def session_messages(
    session_id: str,
    kind: str | None,
    sender: str | None,
    recipient: str | None,
    msg_limit: int,
    offset: int,
):
    """List channel messages for a session."""
    if msg_limit < 0:
        raise click.ClickException("limit must be >= 0")
    if offset < 0:
        raise click.ClickException("offset must be >= 0")
    with connect() as conn:
        reconcile_session_statuses(conn, session_id=session_id)
        s = get_session(conn, session_id)
        if not s:
            raise _not_found("session", session_id)

        messages = list_channel_messages(
            conn,
            session_id,
            kind=kind,
            sender=sender,
            recipient=recipient,
            limit=msg_limit,
            offset=offset,
        )
        normalized_messages = [dict(msg) for msg in messages]
        click.echo(
            json.dumps(
                {
                    "session_id": session_id,
                    "kind": kind,
                    "sender": sender,
                    "recipient": recipient,
                    "limit": msg_limit,
                    "offset": offset,
                    "count": len(normalized_messages),
                    "messages": normalized_messages,
                },
                indent=2,
            )
        )


# -- settings --

_SETTINGS_TARGETS = {
    "codex-instructions": {
        "path": Path.home() / ".codex" / "AGENTS.md",
        "description": "Codex CLI global instructions",
    },
    "codex-settings": {
        "path": Path.home() / ".codex" / "config.toml",
        "description": "Codex CLI settings",
    },
}


def _settings_mode_count(
    edit_target: str | None,
    read_target: str | None,
    write_target: str | None,
    show_list: bool,
) -> int:
    """Count enabled mode flags for `agm settings`."""
    return sum(
        [
            bool(edit_target),
            bool(read_target),
            bool(write_target),
            show_list,
        ]
    )


def _settings_targets_data() -> list[dict[str, object]]:
    """Return normalized settings target rows."""
    rows = []
    for name, info in sorted(_SETTINGS_TARGETS.items()):
        rows.append(
            {
                "name": name,
                "path": str(info["path"]),
                "description": info["description"],
                "exists": info["path"].exists(),
            }
        )
    return rows


def _settings_edit_target(target_name: str) -> None:
    target_path = _SETTINGS_TARGETS[target_name]["path"]
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target_path.touch()
    _launch_editor(target_path)


def _settings_read_target(target_name: str, json_dumps) -> None:
    target_path = _SETTINGS_TARGETS[target_name]["path"]
    content = target_path.read_text() if target_path.exists() else ""
    click.echo(json_dumps({"name": target_name, "content": content}))


def _settings_write_target(target_name: str, content: str) -> None:
    target_path = _SETTINGS_TARGETS[target_name]["path"]
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_path.write_text(content)
    click.echo(json.dumps({"name": target_name, "bytes_written": len(content)}))


def _settings_list_targets(json_dumps) -> None:
    targets_data = _settings_targets_data()
    click.echo(json_dumps(targets_data, indent=2))


def _settings_selected_handler(
    edit_target: str | None,
    read_target: str | None,
    write_target: str | None,
    json_dumps,
    stdin,
):
    if edit_target:
        return lambda: _settings_edit_target(edit_target)
    if read_target:
        return lambda: _settings_read_target(read_target, json_dumps)
    if write_target:
        return lambda: _settings_write_target(write_target, stdin.read())
    return None


@main.command()
@click.option(
    "--edit",
    "edit_target",
    default=None,
    type=click.Choice(sorted(_SETTINGS_TARGETS)),
    help="Open a settings file in $EDITOR.",
)
@click.option(
    "--read",
    "read_target",
    default=None,
    type=click.Choice(sorted(_SETTINGS_TARGETS)),
    help="Print file contents (for programmatic access).",
)
@click.option(
    "--write",
    "write_target",
    default=None,
    type=click.Choice(sorted(_SETTINGS_TARGETS)),
    help="Write stdin to the file.",
)
@click.option("--list", "show_list", is_flag=True, help="List all targets with paths.")
def settings(
    edit_target: str | None,
    read_target: str | None,
    write_target: str | None,
    show_list: bool,
):
    """View or edit Codex configuration files."""
    import json as json_mod
    import sys

    mode_count = _settings_mode_count(edit_target, read_target, write_target, show_list)
    if mode_count > 1:
        raise click.ClickException("Choose one mode: --edit, --read, --write, or --list.")

    selected_handler = _settings_selected_handler(
        edit_target=edit_target,
        read_target=read_target,
        write_target=write_target,
        json_dumps=json_mod.dumps,
        stdin=sys.stdin,
    )
    if selected_handler:
        selected_handler()
        return

    _settings_list_targets(json_mod.dumps)


# -- caller --


@main.group()
def caller():
    """Register and remove custom caller identities."""


@caller.command("add")
@click.argument("name")
def caller_add(name: str):
    """Register a custom caller name."""
    try:
        add_caller(name)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@caller.command("remove")
@click.argument("name")
def caller_remove(name: str):
    """Unregister a custom caller name."""
    try:
        remove_caller(name)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@caller.command("list")
def caller_list():
    """Show all registered callers (built-in + custom)."""
    import json

    all_c = sorted(get_all_callers())
    click.echo(
        json.dumps(
            {
                "builtin": sorted(BUILTIN_CALLERS),
                "custom": sorted(get_all_callers() - BUILTIN_CALLERS),
                "all": all_c,
            },
            indent=2,
        )
    )


# -- daemon --


@main.group(invoke_without_command=True)
@click.pass_context
def daemon(ctx: click.Context) -> None:
    """Manage the shared app-server daemon."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _daemon_pid() -> int | None:
    """Read PID from file and verify the process is alive."""
    from agm.daemon import DEFAULT_PID_PATH

    if not DEFAULT_PID_PATH.exists():
        return None
    try:
        pid = int(DEFAULT_PID_PATH.read_text().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


@daemon.command()
def start() -> None:
    """Start the shared app-server daemon in the background."""
    from agm.daemon import DEFAULT_LOG_PATH, DEFAULT_PID_PATH, DEFAULT_SOCKET_PATH

    existing = _daemon_pid()
    if existing:
        click.echo(json.dumps({"ok": True, "status": "already_running", "pid": existing}))
        return

    # Clean stale PID file if process is dead
    if DEFAULT_PID_PATH.exists():
        DEFAULT_PID_PATH.unlink()

    DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(DEFAULT_LOG_PATH, "a")  # noqa: SIM115
    subprocess.Popen(
        [sys.executable, "-m", "agm.daemon"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    # Wait for socket to appear (daemon is ready)
    for _ in range(50):  # 5 seconds max
        time.sleep(0.1)
        if DEFAULT_SOCKET_PATH.exists():
            break

    pid = _daemon_pid()
    if pid:
        click.echo(json.dumps({"ok": True, "pid": pid, "log": str(DEFAULT_LOG_PATH)}))
    else:
        click.echo(json.dumps({"ok": False, "error": "Daemon failed to start"}))
        raise click.ClickException(f"Daemon failed to start. Check logs: {DEFAULT_LOG_PATH}")


@daemon.command()
def stop() -> None:
    """Stop the running app-server daemon."""
    pid = _daemon_pid()
    if not pid:
        click.echo(json.dumps({"ok": True, "status": "not_running"}))
        return

    os.kill(pid, signal.SIGTERM)

    # Wait for process to exit
    for _ in range(30):  # 3 seconds max
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break

    click.echo(json.dumps({"ok": True, "pid": pid}))


@daemon.command("status")
def daemon_status() -> None:
    """Show daemon status."""
    from agm.daemon import DEFAULT_LOG_PATH, DEFAULT_SOCKET_PATH

    pid = _daemon_pid()
    running = pid is not None
    socket_exists = DEFAULT_SOCKET_PATH.exists()

    click.echo(
        json.dumps(
            {
                "running": running,
                "pid": pid,
                "socket": str(DEFAULT_SOCKET_PATH),
                "socket_exists": socket_exists,
                "log": str(DEFAULT_LOG_PATH),
            }
        )
    )


@daemon.command("threads")
@click.option("--search", "search_term", default=None, help="Filter by thread title substring.")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Page size.")
@click.option("--cursor", default=None, help="Pagination cursor.")
@click.option(
    "--archived",
    "archived",
    flag_value=True,
    default=None,
    help="List archived threads only.",
)
@click.option(
    "--non-archived",
    "archived",
    flag_value=False,
    help="List non-archived threads only.",
)
@click.option(
    "--sort-key",
    type=click.Choice(["created_at", "updated_at"]),
    default=None,
    help="Thread sort key.",
)
@click.option("--cwd", default=None, help="Filter to threads created from this cwd.")
def daemon_threads(
    search_term: str | None,
    limit: int | None,
    cursor: str | None,
    archived: bool | None,
    sort_key: str | None,
    cwd: str | None,
) -> None:
    """List app-server threads through the daemon."""
    if _daemon_pid() is None:
        raise click.ClickException("Daemon is not running. Start it with: agm daemon start")

    params: dict[str, Any] = {}
    if search_term:
        params["searchTerm"] = search_term
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if archived is not None:
        params["archived"] = archived
    if sort_key:
        params["sortKey"] = sort_key
    if cwd:
        params["cwd"] = cwd

    async def _list_threads() -> dict[str, Any]:
        from agm.daemon_client import DaemonClient

        async with DaemonClient() as client:
            return await client.request("thread/list", params)

    try:
        result = normalize_daemon_thread_list(asyncio.run(_list_threads()))
    except Exception as exc:
        raise click.ClickException(f"Failed to list daemon threads: {exc}") from exc
    click.echo(json.dumps(result))


@main.command("purge")
@click.option("-p", "--project", "project_name", default=None, help="Scope to a single project.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def purge(project_name: str | None, yes: bool):
    """Delete all operational data (plans, tasks, logs, history) while keeping projects."""
    import json

    project_id = _resolve_project_id(project_name)

    with connect() as conn:
        preview = purge_preview_counts(conn, project_id)

    total = sum(preview.values())
    if total == 0:
        click.echo(json.dumps({"purged": preview, "total": 0}))
        return

    if not yes and not _purge_confirm(preview, project_name):
        return

    with connect() as conn:
        result = purge_data(conn, project_id)

    redis_cleaned = _purge_redis_cleanup(result, project_id)

    click.echo(
        json.dumps(
            {
                "purged": result["counts"],
                "total": sum(result["counts"].values()),
                "redis_jobs_cleaned": redis_cleaned,
            },
            indent=2,
        )
    )


def _purge_confirm(preview: dict[str, int], project_name: str | None) -> bool:
    """Ask user confirmation for purge."""
    return click.confirm("  Proceed?")


def _purge_redis_cleanup(result: dict[str, Any], project_id: str | None) -> int:
    """Best-effort cleanup of purge-related Redis artifacts."""
    redis_cleaned = 0
    try:
        from agm.queue import (
            EVENTS_STREAM,
            clean_finished_jobs,
            flush_failed_jobs,
            get_redis,
            remove_jobs_for_entities,
        )

        plan_ids = result.get("plan_ids")
        task_ids = result.get("task_ids")
        if isinstance(plan_ids, list) and isinstance(task_ids, list):
            redis_cleaned += remove_jobs_for_entities(plan_ids, task_ids)

        if not project_id:
            for counts in flush_failed_jobs().values():
                redis_cleaned += counts
            for counts in clean_finished_jobs().values():
                redis_cleaned += counts
            redis = get_redis()
            redis_cleaned += cast(int, redis.delete(EVENTS_STREAM))
    except Exception as exc:
        log.debug("Failed to clean purge-related redis artifacts: %s", exc, exc_info=True)
    return redis_cleaned


if __name__ == "__main__":
    main()
