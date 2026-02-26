"""Tests for reviewer diff scoping and truncation."""

import subprocess
from pathlib import Path

from agm.jobs import MAX_DIFF_CHARS


def _git(cwd, *args):
    """Run git command in a directory."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


def _make_repo_with_branch(tmp_path):
    """Create a git repo with main and a feature branch with changes."""
    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", repo], capture_output=True, check=True)
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")

    # Initial commit on main
    Path(repo, "README.md").write_text("# Hello\n")
    Path(repo, "src").mkdir(parents=True, exist_ok=True)
    Path(repo, "src/app.py").write_text("def main(): pass\n")
    Path(repo, "src/utils.py").write_text("def helper(): pass\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")

    # Feature branch with changes to multiple files
    _git(repo, "checkout", "-b", "feature")
    Path(repo, "src/app.py").write_text("def main():\n    print('hello')\n")
    Path(repo, "src/utils.py").write_text("def helper():\n    return 42\n")
    Path(repo, "tests").mkdir(parents=True, exist_ok=True)
    Path(repo, "tests/test_app.py").write_text("def test_main(): assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feature changes")

    return repo


def test_scoped_diff_only_shows_task_files(tmp_path):
    """When task has files, git diff is scoped to those files only."""
    repo = _make_repo_with_branch(tmp_path)

    # Scoped diff: only src/app.py
    result = subprocess.run(
        ["git", "diff", "main...HEAD", "--", "src/app.py"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    diff = result.stdout
    assert "src/app.py" in diff
    assert "src/utils.py" not in diff
    assert "tests/test_app.py" not in diff


def test_full_diff_shows_all_files(tmp_path):
    """Without file scoping, all changed files appear in diff."""
    repo = _make_repo_with_branch(tmp_path)

    result = subprocess.run(
        ["git", "diff", "main...HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    diff = result.stdout
    assert "src/app.py" in diff
    assert "src/utils.py" in diff
    assert "tests/test_app.py" in diff


def test_out_of_scope_count(tmp_path):
    """Stat-based out-of-scope file counting works."""
    repo = _make_repo_with_branch(tmp_path)
    task_files = ["src/app.py"]

    full_stat = subprocess.run(
        ["git", "diff", "main...HEAD", "--stat"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    scoped_stat = subprocess.run(
        ["git", "diff", "main...HEAD", "--stat", "--"] + task_files,
        cwd=repo,
        capture_output=True,
        text=True,
    )
    full_lines = [line for line in full_stat.stdout.strip().splitlines() if "|" in line]
    scoped_lines = [line for line in scoped_stat.stdout.strip().splitlines() if "|" in line]
    extra = len(full_lines) - len(scoped_lines)

    # 3 files changed total, 1 in scope → 2 out of scope
    assert len(full_lines) == 3
    assert len(scoped_lines) == 1
    assert extra == 2


def test_max_diff_chars_is_reasonable():
    """MAX_DIFF_CHARS is set to a value that fits in context windows."""
    assert MAX_DIFF_CHARS >= 10_000
    assert MAX_DIFF_CHARS <= 100_000
