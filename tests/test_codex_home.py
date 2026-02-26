"""Tests for CODEX_HOME bootstrap in _ensure_codex_home()."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up isolated filesystem paths for CODEX_HOME tests.

    - ``Path.home()`` → ``tmp_path``
    - ``agm.paths.CODEX_HOME`` → ``tmp_path / "agm" / ".codex"``
    - Creates ``tmp_path / ".codex"`` as the "real" codex home
    """
    codex_home = tmp_path / "agm" / ".codex"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr("agm.paths.CODEX_HOME", codex_home)
    # Create the "real" user codex directory
    (tmp_path / ".codex").mkdir()
    return codex_home


def test_creates_directory_and_skills(isolated_home: Path) -> None:
    from agm.jobs_common import _ensure_codex_home

    result = _ensure_codex_home()

    assert result == isolated_home
    assert isolated_home.is_dir()
    assert (isolated_home / "skills").is_dir()


def test_symlinks_auth_when_source_exists(isolated_home: Path, tmp_path: Path) -> None:
    from agm.jobs_common import _ensure_codex_home

    real_auth = tmp_path / ".codex" / "auth.json"
    real_auth.write_text('{"auth_mode": "test"}')

    _ensure_codex_home()

    link = isolated_home / "auth.json"
    assert link.is_symlink()
    assert link.resolve() == real_auth.resolve()
    assert link.read_text() == '{"auth_mode": "test"}'


def test_skips_symlink_when_source_missing(isolated_home: Path, tmp_path: Path) -> None:
    from agm.jobs_common import _ensure_codex_home

    # No auth.json in the "real" codex home
    assert not (tmp_path / ".codex" / "auth.json").exists()

    _ensure_codex_home()

    assert not (isolated_home / "auth.json").exists()


def test_idempotent(isolated_home: Path, tmp_path: Path) -> None:
    from agm.jobs_common import _ensure_codex_home

    (tmp_path / ".codex" / "auth.json").write_text('{"auth_mode": "test"}')

    result1 = _ensure_codex_home()
    result2 = _ensure_codex_home()

    assert result1 == result2 == isolated_home
    # Symlink created once, not duplicated
    assert (isolated_home / "auth.json").is_symlink()
    assert (isolated_home / "skills").is_dir()
