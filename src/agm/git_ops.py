"""Git operations shared by CLI and job workers.

Functions raise RuntimeError on failure (not ClickException),
so they can be used from both cli.py and jobs.py.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 40) -> str:
    """Turn a title into a branch-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def create_worktree(
    project_dir: str,
    task_id: str,
    title: str,
    base_branch: str = "main",
) -> tuple[str, str]:
    """Create a git worktree for a task.

    Raises RuntimeError on failure.
    """
    project = Path(project_dir)
    branch = f"agm/{slugify(title)}-{task_id[:8]}"
    worktree_dir = project / ".agm" / "worktrees" / branch.replace("/", "-")

    if worktree_dir.exists():
        # Already exists (e.g. retry) — reuse it
        return branch, str(worktree_dir)

    worktree_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(worktree_dir), base_branch],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create worktree: {e.stderr.strip()}") from None

    # Ensure clean index — prevents stale staged changes from being swept
    # into commits by broad `git add` commands in the executor.
    subprocess.run(
        ["git", "reset", "HEAD"],
        cwd=str(worktree_dir),
        capture_output=True,
        text=True,
    )

    return branch, str(worktree_dir)


def remove_worktree(project_dir: str, worktree_path: str, branch: str) -> None:
    """Remove a git worktree and its branch. Best-effort, logs warnings on failure."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", worktree_path],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning("Failed to remove worktree %s: %s", worktree_path, exc.stderr.strip())

    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning("Failed to delete branch %s: %s", branch, exc.stderr.strip())

    # Prune stale worktree records left by manually deleted directories
    with contextlib.suppress(subprocess.CalledProcessError):
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )


def _format_inspection_error(path: Path, exc: OSError) -> dict[str, str]:
    return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def compute_directory_disk_usage(path: str | Path) -> dict[str, object]:
    """Recursively compute disk usage for a directory without raising.

    Returns structured data suitable for doctor reporting:
    - path: directory path
    - exists: whether the directory exists
    - total_bytes: sum of file sizes discovered during traversal
    - file_count: number of non-directory entries counted
    - dir_count: number of subdirectories discovered
    - errors: list of non-fatal traversal/stat errors
    """
    directory = Path(path)
    errors: list[dict[str, str]] = []
    result: dict[str, object] = {
        "path": str(directory),
        "exists": directory.exists(),
        "total_bytes": 0,
        "file_count": 0,
        "dir_count": 0,
        "errors": errors,
    }

    if not directory.exists():
        return result

    if not directory.is_dir():
        errors.append({"path": str(directory), "error": "Not a directory"})
        return result

    total_bytes = 0
    file_count = 0
    dir_count = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dir_count += 1
                            stack.append(entry_path)
                            continue

                        stat_result = entry.stat(follow_symlinks=False)
                        file_count += 1
                        total_bytes += stat_result.st_size
                    except OSError as exc:
                        errors.append(_format_inspection_error(entry_path, exc))
        except OSError as exc:
            errors.append(_format_inspection_error(current, exc))

    result["total_bytes"] = total_bytes
    result["file_count"] = file_count
    result["dir_count"] = dir_count
    return result


def inspect_worktrees(project_dir: str | Path) -> dict[str, object]:
    """Inspect `.agm/worktrees` and return structured per-worktree usage data.

    This helper is intentionally non-throwing so doctor can degrade to WARNING
    on partial filesystem issues instead of crashing.
    """
    project = Path(project_dir)
    root = project / ".agm" / "worktrees"
    errors: list[dict[str, str]] = []
    result: dict[str, object] = {
        "root_path": str(root),
        "root_exists": root.exists(),
        "worktrees": [],
        "errors": errors,
    }

    if not root.exists():
        return result

    if not root.is_dir():
        errors.append({"path": str(root), "error": "Not a directory"})
        return result

    try:
        root_entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        errors.append(_format_inspection_error(root, exc))
        return result

    worktrees: list[dict[str, object]] = []
    for entry in root_entries:
        try:
            is_dir = entry.is_dir()
        except OSError as exc:
            errors.append(_format_inspection_error(entry, exc))
            continue

        if not is_dir:
            continue

        usage = compute_directory_disk_usage(entry)
        worktrees.append(
            {
                "name": entry.name,
                "path": usage["path"],
                "exists": usage["exists"],
                "total_bytes": usage["total_bytes"],
                "file_count": usage["file_count"],
                "dir_count": usage["dir_count"],
                "errors": usage["errors"],
            }
        )

    result["worktrees"] = worktrees
    return result


def rebase_onto_main(
    worktree_path: str,
    base_branch: str = "main",
) -> bool:
    """Rebase the current branch in a worktree onto base_branch.

    Returns True if rebase was needed and succeeded, False if already up-to-date.
    Raises RuntimeError on conflict (after aborting the rebase).
    """
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Cannot rebase: worktree has uncommitted changes")

    # Check if branch is behind base_branch
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base_branch],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", base_branch],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if merge_base.returncode != 0 or base_sha.returncode != 0:
        return False  # Can't determine — skip rebase

    if merge_base.stdout.strip() == base_sha.stdout.strip():
        return False  # Already up-to-date with base branch

    try:
        subprocess.run(
            ["git", "rebase", base_branch],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        # Abort the failed rebase and raise
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )
        raise RuntimeError(
            f"Rebase conflict (branch is behind {base_branch} and cannot be auto-rebased):\n"
            f"{e.stderr.strip()}"
        ) from None


def check_branch_file_scope(
    project_dir: str,
    branch: str,
    allowed_files: list[str],
    base_branch: str = "main",
) -> list[str]:
    """Check if a branch only touches files in the allowed list.

    Returns a list of out-of-scope files (empty if all files are in scope).
    Compares the branch diff against `base_branch`.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_branch}...{branch}"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []  # Can't determine — don't block

    touched = {f.strip() for f in result.stdout.strip().splitlines() if f.strip()}
    if not touched:
        return []

    allowed = set(allowed_files)
    return sorted(touched - allowed)


def merge_to_main(
    project_dir: str,
    branch: str,
    task_id: str,
    title: str,
    base_branch: str = "main",
    worktree_path: str | None = None,
    on_sync_failure: Callable[[str], None] = lambda _stderr: None,
) -> str:
    """Merge a task branch into base branch via a temporary detached worktree.

    Returns the merge commit SHA on success.

    If worktree_path is provided and the branch is behind base branch, rebases
    onto base branch first. This handles non-conflicting drift from earlier
    merges automatically.

    Always uses a detached HEAD to avoid 'branch already checked out' conflicts.
    Never touches the user's working tree. Raises RuntimeError on conflict or
    any git failure.
    """
    merge_msg = f"Merge task {task_id[:8]}: {title}"

    # Rebase onto base branch if the branch is behind (handles non-conflicting drift)
    if worktree_path:
        rebase_onto_main(worktree_path, base_branch=base_branch)

    # Resolve base to a SHA so we can use --detach
    try:
        base_sha = subprocess.run(
            ["git", "rev-parse", base_branch],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Cannot resolve '{base_branch}' branch: {e.stderr.strip()}") from None

    # Reject no-op merges: branch must have commits ahead of base_branch
    ahead = subprocess.run(
        ["git", "rev-list", "--count", f"{base_branch}..{branch}"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if ahead.returncode == 0 and ahead.stdout.strip() == "0":
        raise RuntimeError(
            f"Branch '{branch}' has no commits ahead of {base_branch} — nothing to merge"
        )

    merge_sha = None
    tmp = tempfile.mkdtemp(prefix="agm-merge-")
    try:
        # Detached HEAD avoids "branch already checked out" conflicts
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", tmp, base_sha],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create merge worktree: {e.stderr.strip()}") from None

        try:
            subprocess.run(
                ["git", "merge", "--no-ff", branch, "-m", merge_msg],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            with contextlib.suppress(subprocess.CalledProcessError):
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
            raise RuntimeError(f"Merge conflict:\n{e.stderr.strip()}") from None

        # Advance the base branch ref to the new merge commit
        new_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        try:
            subprocess.run(
                ["git", "update-ref", f"refs/heads/{base_branch}", new_sha, base_sha],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to update '{base_branch}' ref: {e.stderr.strip()}"
            ) from None

        # Sync the main checkout's working tree and index to match the new HEAD.
        # update-ref only moves the ref pointer; without this the working tree
        # contains pre-merge file contents and git status shows false diffs.
        #
        # Guard: only sync if the user has no uncommitted changes relative to
        # the pre-merge base. Comparing against base_sha (not HEAD) avoids the
        # cascade problem where a prior pipeline merge makes the tree look
        # "dirty" to subsequent merges in the same run.
        user_changes = subprocess.run(
            ["git", "diff", "--quiet", base_sha],
            cwd=project_dir,
            capture_output=True,
        )
        if user_changes.returncode == 0:
            # Remove tracked files that were deleted in the merge, then
            # update remaining files to match HEAD.  `checkout HEAD -- .`
            # alone can't remove files that no longer exist in HEAD.
            deleted = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=D", base_sha, "HEAD"],
                cwd=project_dir,
                capture_output=True,
                text=True,
            )
            if deleted.returncode == 0:
                for f in deleted.stdout.strip().splitlines():
                    f = f.strip()
                    if f:
                        full = Path(project_dir) / f
                        if full.exists():
                            full.unlink()
            checkout_result = subprocess.run(
                ["git", "checkout", "HEAD", "--", "."],
                cwd=project_dir,
                capture_output=True,
                text=True,
            )
            if checkout_result.returncode != 0:
                stderr_message = (
                    checkout_result.stderr.strip() or "(no stderr output from checkout)"
                )
                log.warning(
                    "Failed to sync working tree after merge %s: %s", task_id, stderr_message
                )
                on_sync_failure(stderr_message)
        merge_sha = new_sha
    finally:
        # Clean up temp worktree — try git first, fall back to rm -rf
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(
                ["git", "worktree", "remove", "--force", tmp],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    return merge_sha


def detect_worktree_conflicts(project_dir: str) -> dict:
    """Run ``clash status --json`` and return parsed conflict data.

    Returns a dict with keys:
    - available: bool — whether the clash binary was found
    - worktrees: list of worktree dicts (id, path, branch, status)
    - conflicts: list of conflict dicts (wt1_id, wt2_id, conflicting_files, error)
    - error: optional error string if clash failed

    Gracefully handles: clash not installed, no git repo, subprocess errors.
    """
    result: dict = {
        "available": False,
        "worktrees": [],
        "conflicts": [],
        "error": None,
    }

    # Check if clash is installed
    try:
        subprocess.run(
            ["clash", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        result["error"] = "clash binary not found (install from https://clash.sh)"
        return result
    except subprocess.CalledProcessError as e:
        result["error"] = f"clash --version failed: {e.stderr.strip()}"
        return result

    result["available"] = True

    try:
        proc = subprocess.run(
            ["clash", "status", "--json"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        result["error"] = "clash status timed out after 30s"
        return result
    except Exception as e:
        result["error"] = f"clash status failed: {e}"
        return result

    if proc.returncode not in (0, 2):
        result["error"] = f"clash exited with code {proc.returncode}: {proc.stderr.strip()}"
        return result

    import json

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        result["error"] = f"Failed to parse clash JSON output: {e}"
        return result

    result["worktrees"] = data.get("worktrees", [])
    result["conflicts"] = data.get("conflicts", [])
    return result


def get_real_conflicts(clash_result: dict) -> list[dict]:
    """Filter clash results to only real conflicts (non-empty conflicting_files, no error)."""
    return [
        c
        for c in clash_result.get("conflicts", [])
        if c.get("conflicting_files") and not c.get("error")
    ]
