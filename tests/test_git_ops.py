"""Tests for git_ops module."""

import logging
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import agm.git_ops as git_ops
from agm.git_ops import (
    check_branch_file_scope,
    compute_directory_disk_usage,
    create_worktree,
    detect_worktree_conflicts,
    get_real_conflicts,
    inspect_worktrees,
    merge_to_main,
    rebase_onto_main,
    remove_worktree,
    slugify,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed"),
]


@pytest.fixture(autouse=True)
def git_identity_env(monkeypatch):
    """Ensure commits succeed without relying on global git config."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "agm-tests")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "agm-tests@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "agm-tests")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "agm-tests@example.com")


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert slugify("Fix login/auth bug!") == "fix-login-auth-bug"


def test_slugify_max_len():
    result = slugify("a" * 60, max_len=10)
    assert len(result) <= 10


def test_slugify_strips_trailing_dash():
    result = slugify("hello---", max_len=5)
    assert not result.endswith("-")


def test_create_worktree_success(tmp_path):
    """create_worktree should create a branch and worktree directory."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    branch, wt = create_worktree(str(project), "abc12345", "Test task")
    assert branch.startswith("agm/")
    assert "abc12345" in branch
    assert wt.endswith(branch.replace("/", "-"))


def test_create_worktree_reuses_existing(tmp_path):
    """create_worktree should reuse if the directory already exists."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    branch1, wt1 = create_worktree(str(project), "abc12345", "Test task")
    branch2, wt2 = create_worktree(str(project), "abc12345", "Test task")
    assert branch1 == branch2
    assert wt1 == wt2


def test_create_worktree_uses_selected_base_branch(tmp_path):
    """create_worktree should create a branch from the selected integration branch."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "release"], cwd=str(project), check=True, capture_output=True
    )
    (project / "release.txt").write_text("release")
    subprocess.run(
        ["git", "add", "release.txt"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "release update"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "checkout", "main"], cwd=str(project), check=True, capture_output=True)

    branch, wt = create_worktree(
        str(project),
        "abc12345",
        "Test task",
        base_branch="release",
    )

    assert branch.startswith("agm/")
    assert (Path(wt) / "release.txt").exists()

    release_sha = subprocess.run(
        ["git", "rev-parse", "release"],
        cwd=str(project),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    wt_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=wt,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert wt_sha == release_sha


def test_create_worktree_failure(tmp_path):
    """create_worktree should raise RuntimeError on failure."""
    # tmp_path exists but has no git repo, so git worktree add will fail
    with pytest.raises(RuntimeError, match="Failed to create worktree"):
        create_worktree(str(tmp_path), "abc12345", "Test task")


def test_remove_worktree_best_effort(tmp_path):
    """remove_worktree should not raise on failure (best-effort)."""
    # Should not raise even for nonexistent paths
    remove_worktree(str(tmp_path), "/nonexistent/worktree", "nonexistent-branch")


def test_compute_directory_disk_usage_success(tmp_path):
    usage_root = tmp_path / "usage-root"
    nested = usage_root / "nested"
    nested.mkdir(parents=True)
    (usage_root / "top.txt").write_bytes(b"abcd")
    (nested / "deep.txt").write_bytes(b"xyz")

    usage = compute_directory_disk_usage(usage_root)

    assert usage["exists"] is True
    assert usage["total_bytes"] == 7
    assert usage["file_count"] == 2
    assert usage["dir_count"] == 1
    assert usage["errors"] == []


def test_inspect_worktrees_success(tmp_path):
    project = tmp_path / "project"
    worktrees_root = project / ".agm" / "worktrees"
    wt_a = worktrees_root / "task-a"
    wt_b = worktrees_root / "task-b"
    wt_a.mkdir(parents=True)
    (wt_a / "file.txt").write_bytes(b"abc")
    (wt_b / "nested").mkdir(parents=True)
    (wt_b / "nested" / "file.txt").write_bytes(b"hello")
    (worktrees_root / "README.txt").parent.mkdir(parents=True, exist_ok=True)
    (worktrees_root / "README.txt").write_text("ignore me")

    report = inspect_worktrees(project)

    assert report["root_exists"] is True
    assert report["errors"] == []
    worktrees = report["worktrees"]
    assert [item["name"] for item in worktrees] == ["task-a", "task-b"]
    by_name = {item["name"]: item for item in worktrees}
    assert by_name["task-a"]["total_bytes"] == 3
    assert by_name["task-a"]["file_count"] == 1
    assert by_name["task-a"]["dir_count"] == 0
    assert by_name["task-a"]["errors"] == []
    assert by_name["task-b"]["total_bytes"] == 5
    assert by_name["task-b"]["file_count"] == 1
    assert by_name["task-b"]["dir_count"] == 1
    assert by_name["task-b"]["errors"] == []


def test_inspect_worktrees_missing_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    report = inspect_worktrees(project)

    assert report["root_exists"] is False
    assert report["worktrees"] == []
    assert report["errors"] == []


def test_inspect_worktrees_unreadable_entries_are_non_fatal(tmp_path, monkeypatch):
    project = tmp_path / "project"
    worktree = project / ".agm" / "worktrees" / "task-a"
    blocked_subtree = worktree / "blocked-subtree"
    blocked_subtree.mkdir(parents=True)
    good_file = worktree / "good.txt"
    bad_file = worktree / "bad.txt"
    good_file.write_text("ok")
    bad_file.write_text("cannot-stat")
    (blocked_subtree / "inside.txt").write_text("blocked")

    original_scandir = git_ops.os.scandir

    class _BadStatDirEntry:
        def __init__(self, entry):
            self._entry = entry

        @property
        def path(self):
            return self._entry.path

        def is_dir(self, *, follow_symlinks=True):
            return self._entry.is_dir(follow_symlinks=follow_symlinks)

        def stat(self, *, follow_symlinks=True):
            raise PermissionError("blocked file")

    class _ScandirProxy:
        def __init__(self, entries):
            self._entries = entries

        def __enter__(self):
            self._entries.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._entries.__exit__(exc_type, exc_val, exc_tb)

        def __iter__(self):
            for entry in self._entries:
                if Path(entry.path) == bad_file:
                    yield _BadStatDirEntry(entry)
                    continue
                yield entry

    def fake_scandir(path):
        path_obj = Path(path)
        if path_obj == blocked_subtree:
            raise PermissionError("blocked subtree")

        entries = original_scandir(path)
        if path_obj == worktree:
            return _ScandirProxy(entries)
        return entries

    monkeypatch.setattr(git_ops.os, "scandir", fake_scandir)

    report = inspect_worktrees(project)

    assert report["root_exists"] is True
    assert report["errors"] == []
    assert len(report["worktrees"]) == 1
    usage = report["worktrees"][0]
    assert usage["name"] == "task-a"
    assert usage["total_bytes"] == 2
    assert usage["file_count"] == 1
    assert usage["dir_count"] == 1
    assert len(usage["errors"]) == 2
    assert any(
        err["path"] == str(bad_file) and "blocked file" in err["error"] for err in usage["errors"]
    )
    assert any(
        err["path"] == str(blocked_subtree) and "blocked subtree" in err["error"]
        for err in usage["errors"]
    )


def test_merge_to_main_direct(tmp_path):
    """merge_to_main should merge when project dir is on main."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create a branch with a commit
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "new_file.txt").write_text("hello")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    merge_to_main(str(project), "feature", "abc12345", "Test task")

    # Verify the merge happened
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log.stdout


def test_merge_to_main_via_worktree(tmp_path):
    """merge_to_main should use temp worktree when not on main."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create feature branch with a commit
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "new_file.txt").write_text("hello")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Go to a different branch (not main)
    subprocess.run(
        ["git", "checkout", "-b", "other"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    merge_to_main(str(project), "feature", "abc12345", "Test task")

    # Verify we're still on "other"
    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(project),
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current == "other"

    # Verify main has the merge
    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log.stdout


def test_merge_to_main_conflict(tmp_path):
    """merge_to_main should raise RuntimeError on conflict."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("original")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create conflicting changes
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("feature version")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("main version")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "main change"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="Merge conflict"):
        merge_to_main(str(project), "feature", "abc12345", "Test task")


def test_merge_to_main_checkout_sync_failure_still_returns_merge_sha_and_signals(tmp_path, caplog):
    """merge_to_main should return merge SHA and report sync-failure condition."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "new_file.txt").write_text("hello")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    sync_failures: list[str] = []
    sync_failure_stderr = "manual checkout blocked"
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        result = real_run(cmd, *args, **kwargs)
        if (
            cmd[0] == "git"
            and cmd[1] == "checkout"
            and cmd[2] == "HEAD"
            and kwargs.get("cwd") == str(project)
        ):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout=result.stdout,
                stderr=sync_failure_stderr,
            )
        return result

    caplog.set_level(logging.WARNING)
    with patch("agm.git_ops.subprocess.run", side_effect=fake_run):
        merge_sha = merge_to_main(
            str(project),
            "feature",
            "abc12345",
            "Test task",
            on_sync_failure=sync_failures.append,
        )

    assert merge_sha
    assert sync_failures == [sync_failure_stderr]
    assert "Failed to sync working tree after merge abc12345" in caplog.text
    assert sync_failure_stderr in caplog.text


def test_merge_to_main_checkout_sync_success_does_not_signal(tmp_path):
    """merge_to_main should not report sync-failure when checkout succeeds."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "new_file.txt").write_text("hello")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    sync_failures: list[str] = []
    merge_to_main(
        str(project),
        "feature",
        "abc12345",
        "Test task",
        on_sync_failure=sync_failures.append,
    )
    assert sync_failures == []


def test_merge_to_main_dirty_tree(tmp_path):
    """merge_to_main should succeed even with uncommitted changes in project dir."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create dirty state — should NOT block merge (uses temp worktree)
    (project / "dirty.txt").write_text("uncommitted")

    merge_to_main(str(project), "feature", "abc12345", "Test task")

    # Verify the merge happened on main
    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log.stdout


def test_merge_to_main_main_checked_out_elsewhere(tmp_path):
    """merge_to_main should work even when main is checked out in project dir."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create a task branch with a commit
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "new_file.txt").write_text("hello")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Stay on main (simulates user's checkout — the exact scenario that broke)
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # This is the bug case: main is already checked out in project dir,
    # old code would fail with "already checked out" when creating temp worktree
    merge_to_main(str(project), "feature", "abc12345", "Test task")

    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log.stdout


def test_merge_to_main_concurrent_worktrees(tmp_path):
    """merge_to_main should work when task worktrees exist for other branches."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create two feature branches
    for name in ("feature-a", "feature-b"):
        subprocess.run(
            ["git", "checkout", "-b", name],
            cwd=str(project),
            check=True,
            capture_output=True,
        )
        (project / f"{name}.txt").write_text(name)
        subprocess.run(
            ["git", "add", "."],
            cwd=str(project),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"{name} commit"],
            cwd=str(project),
            check=True,
            capture_output=True,
        )
    # Go to main before creating worktree (can't worktree a checked-out branch)
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create a worktree for feature-b (simulating a running executor)
    wt_dir = project / ".agm" / "worktrees" / "feature-b-wt"
    wt_dir.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "worktree", "add", str(wt_dir), "feature-b"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # Merge feature-a while feature-b worktree exists (and main is checked out)
    merge_to_main(str(project), "feature-a", "aaa11111", "Task A")

    log = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task aaa11111" in log.stdout


def test_merge_to_main_noop_raises(tmp_path):
    """merge_to_main should raise when the branch has no commits ahead of main."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Create a branch at the same point as main (no new commits)
    subprocess.run(
        ["git", "branch", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="no commits ahead of main"):
        merge_to_main(str(project), "feature", "abc12345", "Test task")


def test_merge_to_main_with_non_main_base(tmp_path):
    """merge_to_main should merge into the selected integration branch."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "release"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "release.txt").write_text("release")
    subprocess.run(["git", "add", "release.txt"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "release commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    merge_to_main(str(project), "feature", "abc12345", "Test task", base_branch="release")

    log = subprocess.run(
        ["git", "log", "--oneline", "release"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log.stdout


def test_merge_to_main_noop_raises_non_main_base(tmp_path):
    """merge_to_main should reject no-op merges for the selected integration branch."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "release"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "branch", "feature", "release"], cwd=str(project), check=True, capture_output=True
    )
    with pytest.raises(RuntimeError, match="no commits ahead of release"):
        merge_to_main(str(project), "feature", "abc12345", "Test task", base_branch="release")


def test_merge_to_main_invalid_base_branch_raises(tmp_path):
    """merge_to_main should raise when the base branch doesn't exist."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "feature work"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="Cannot resolve 'nonexistent'"):
        merge_to_main(str(project), "feature", "abc12345", "Test task", base_branch="nonexistent")


def _init_repo(project):
    """Helper: init a git repo with an initial commit."""
    subprocess.run(["git", "init", "-b", "main"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )


def test_rebase_onto_main_already_up_to_date(tmp_path):
    """rebase_onto_main returns False when branch is already at main."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("new")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature"], cwd=str(project), check=True, capture_output=True
    )

    result = rebase_onto_main(str(project))
    assert result is False  # Already based on main tip


def test_rebase_onto_main_raises_with_dirty_worktree(tmp_path):
    """rebase_onto_main should fail fast when the worktree is dirty."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    # Branch behind main to ensure rebase would normally run
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature"], cwd=str(project), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "main.txt").write_text("main change")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main advance"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # Create an uncommitted change to force early failure
    (wt / "dirty.txt").write_text("dirty")

    with pytest.raises(RuntimeError, match=r"^Cannot rebase: worktree has uncommitted changes$"):
        rebase_onto_main(str(wt))


def test_rebase_onto_main_rebases_when_behind(tmp_path):
    """rebase_onto_main should rebase when main has moved forward."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature"], cwd=str(project), check=True, capture_output=True
    )

    # Move main forward (non-conflicting)
    subprocess.run(["git", "checkout", "main"], cwd=str(project), check=True, capture_output=True)
    (project / "main.txt").write_text("main change")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main advance"], cwd=str(project), check=True, capture_output=True
    )

    # Create worktree for feature branch
    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    result = rebase_onto_main(str(wt))
    assert result is True

    # Verify feature is now based on main
    ahead = subprocess.run(
        ["git", "rev-list", "--count", "main..feature"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert ahead.stdout.strip() == "1"  # Still 1 commit ahead, but rebased


def test_rebase_onto_main_rebases_when_behind_non_main(tmp_path):
    """rebase_onto_main should rebase when release has moved forward."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    subprocess.run(
        ["git", "checkout", "-b", "release"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "release.txt").write_text("release")
    subprocess.run(["git", "add", "release.txt"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "release move forward"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # Feature branch from old base (main)
    subprocess.run(
        ["git", "checkout", "-b", "feature", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "checkout", "release"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    result = rebase_onto_main(str(wt), base_branch="release")
    assert result is True

    ahead = subprocess.run(
        ["git", "rev-list", "--count", "release..feature"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert ahead.stdout.strip() == "1"


def test_rebase_onto_main_conflict_raises(tmp_path):
    """rebase_onto_main should abort and raise RuntimeError on conflict."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    # Create initial file
    (project / "file.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add file"], cwd=str(project), check=True, capture_output=True
    )

    # Branch and modify
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("feature version")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature change"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # Main conflicts
    subprocess.run(["git", "checkout", "main"], cwd=str(project), check=True, capture_output=True)
    (project / "file.txt").write_text("main version")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main conflict"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    # Worktree for feature
    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="Rebase conflict"):
        rebase_onto_main(str(wt))


def test_rebase_onto_main_conflict_raises_non_main_base(tmp_path):
    """rebase_onto_main should mention selected base branch in conflict errors."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    (project / "file.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add file"], cwd=str(project), check=True, capture_output=True
    )

    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "file.txt").write_text("feature version")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature change"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "release"], cwd=str(project), check=True, capture_output=True
    )
    (project / "file.txt").write_text("release version")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "release version"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="behind release"):
        rebase_onto_main(str(wt), base_branch="release")


def test_merge_to_main_with_rebase(tmp_path):
    """merge_to_main with worktree_path should rebase before merging."""
    project = tmp_path / "project"
    project.mkdir()
    _init_repo(project)

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    (project / "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature"], cwd=str(project), check=True, capture_output=True
    )

    # Move main forward (non-conflicting)
    subprocess.run(["git", "checkout", "main"], cwd=str(project), check=True, capture_output=True)
    (project / "main.txt").write_text("main change")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main advance"], cwd=str(project), check=True, capture_output=True
    )

    # Create worktree for feature
    wt = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", str(wt), "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    merge_to_main(str(project), "feature", "abc12345", "Test task", worktree_path=str(wt))

    log_result = subprocess.run(
        ["git", "log", "--oneline", "main"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert "Merge task abc12345" in log_result.stdout


# -- clash integration tests --


def test_detect_worktree_conflicts_clash_not_installed(tmp_path):
    """Should return available=False when clash binary is missing."""
    with patch(
        "agm.git_ops.subprocess.run",
        side_effect=FileNotFoundError("clash not found"),
    ):
        result = detect_worktree_conflicts(str(tmp_path))

    assert result["available"] is False
    assert "not found" in result["error"]
    assert result["worktrees"] == []
    assert result["conflicts"] == []


def test_detect_worktree_conflicts_parses_json(tmp_path):
    """Should parse clash JSON output correctly."""
    import json

    clash_output = {
        "worktrees": [
            {"id": "main", "path": "/p", "branch": "main", "status": "clean"},
            {"id": "wt1", "path": "/p/wt1", "branch": "feat/a", "status": "dirty"},
        ],
        "conflicts": [
            {
                "wt1_id": "main",
                "wt2_id": "wt1",
                "conflicting_files": ["src/db.py"],
                "error": None,
            }
        ],
    }

    class FakeProc:
        returncode = 2
        stdout = json.dumps(clash_output)
        stderr = ""

    def fake_run(cmd, **kwargs):
        if cmd[0] == "clash" and cmd[1] == "--version":
            proc = FakeProc()
            proc.returncode = 0
            proc.stdout = "clash 0.2.0"
            return proc
        if cmd[0] == "clash" and cmd[1] == "status":
            return FakeProc()
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch("agm.git_ops.subprocess.run", side_effect=fake_run):
        result = detect_worktree_conflicts(str(tmp_path))

    assert result["available"] is True
    assert result["error"] is None
    assert len(result["worktrees"]) == 2
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["conflicting_files"] == ["src/db.py"]


def test_detect_worktree_conflicts_timeout(tmp_path):
    """Should handle clash timeout gracefully."""

    def fake_run(cmd, **kwargs):
        if cmd[1] == "--version":

            class P:
                returncode = 0
                stdout = "clash 0.2.0"
                stderr = ""

            return P()
        raise subprocess.TimeoutExpired(cmd, 30)

    with patch("agm.git_ops.subprocess.run", side_effect=fake_run):
        result = detect_worktree_conflicts(str(tmp_path))

    assert result["available"] is True
    assert "timed out" in result["error"]


def test_get_real_conflicts_filters():
    """get_real_conflicts should filter out errors and empty file lists."""
    conflicts = [
        {"wt1_id": "a", "wt2_id": "b", "conflicting_files": ["f.py"], "error": None},
        {"wt1_id": "a", "wt2_id": "c", "conflicting_files": [], "error": None},
        {"wt1_id": "b", "wt2_id": "c", "conflicting_files": ["g.py"], "error": "some error"},
        {"wt1_id": "a", "wt2_id": "d", "conflicting_files": ["h.py", "i.py"], "error": None},
    ]
    real = get_real_conflicts({"conflicts": conflicts})
    assert len(real) == 2
    assert real[0]["wt2_id"] == "b"
    assert real[1]["wt2_id"] == "d"


# -- create_worktree index hygiene --


def test_create_worktree_clean_index(tmp_path):
    """create_worktree should reset the index so no stale staged changes leak."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init"], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )

    branch, wt = create_worktree(str(project), "abc12345", "Clean index test")
    # Index should have no staged changes
    status = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=wt,
        capture_output=True,
        text=True,
    )
    assert status.stdout.strip() == ""


# -- check_branch_file_scope --


def _setup_branch_with_files(
    tmp_path,
    files: dict[str, str],
    base_branch: str = "main",
) -> tuple[str, str]:
    """Create a repo with `base_branch` and a feature branch touching files."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    # Initial commit on main with a baseline file
    (project / "baseline.txt").write_text("baseline")
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    if base_branch != "main":
        subprocess.run(
            ["git", "checkout", "-b", base_branch],
            cwd=str(project),
            check=True,
            capture_output=True,
        )

    # Create feature branch and add files
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    for path, content in files.items():
        fpath = project / path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
    subprocess.run(["git", "add", "."], cwd=str(project), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature changes"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", base_branch],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    return str(project), "feature"


def test_check_branch_file_scope_all_in_scope(tmp_path):
    """Should return empty list when all touched files are in scope."""
    project_dir, branch = _setup_branch_with_files(
        tmp_path, {"src/cli.py": "code", "tests/test_cli.py": "tests"}
    )
    out_of_scope = check_branch_file_scope(project_dir, branch, ["src/cli.py", "tests/test_cli.py"])
    assert out_of_scope == []


def test_check_branch_file_scope_detects_extra_files(tmp_path):
    """Should return out-of-scope files when branch touches extra files."""
    project_dir, branch = _setup_branch_with_files(
        tmp_path,
        {"src/cli.py": "code", "src/db.py": "oops", "src/jobs.py": "oops"},
    )
    out_of_scope = check_branch_file_scope(project_dir, branch, ["src/cli.py"])
    assert out_of_scope == ["src/db.py", "src/jobs.py"]


def test_check_branch_file_scope_all_in_scope_non_main_base(tmp_path):
    """Should compare against a non-main base branch."""
    project_dir, branch = _setup_branch_with_files(
        tmp_path, {"src/cli.py": "code"}, base_branch="release"
    )
    out_of_scope = check_branch_file_scope(
        project_dir,
        branch,
        ["src/cli.py"],
        base_branch="release",
    )
    assert out_of_scope == []


def test_check_branch_file_scope_no_files_touched(tmp_path):
    """Should return empty list when branch has no file changes (empty diff)."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "empty-branch"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "empty"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=str(project),
        check=True,
        capture_output=True,
    )
    out_of_scope = check_branch_file_scope(str(project), "empty-branch", ["anything.py"])
    assert out_of_scope == []


def test_check_branch_file_scope_git_failure_returns_empty(tmp_path):
    """Should return empty list (not block) when git command fails."""
    out_of_scope = check_branch_file_scope(str(tmp_path), "nonexistent", ["a.py"])
    assert out_of_scope == []
