"""Tests for custom caller management."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agm.callers import (
    BUILTIN_CALLERS,
    _read_custom_callers,
    add_caller,
    get_all_callers,
    remove_caller,
)


def _tmp_callers_file():
    """Return a temporary callers file path for testing."""
    tmp = Path(tempfile.mktemp(suffix=".callers"))
    return tmp


def _patch_callers_file(path: Path):
    return patch("agm.callers._CALLERS_FILE", path)


def test_read_custom_callers_no_file():
    """Returns empty set when the callers file doesn't exist."""
    with _patch_callers_file(Path("/tmp/nonexistent-callers-file-xyz")):
        assert _read_custom_callers() == set()


def test_add_remove_roundtrip():
    """Adding and removing a custom caller round-trips correctly."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path):
            add_caller("my-tool")
            assert "my-tool" in _read_custom_callers()
            assert "my-tool" in get_all_callers()

            remove_caller("my-tool")
            assert "my-tool" not in _read_custom_callers()
            assert "my-tool" not in get_all_callers()
    finally:
        path.unlink(missing_ok=True)


def test_add_invalid_name():
    """Invalid caller names are rejected."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path):
            with pytest.raises(ValueError, match="Invalid caller name"):
                add_caller("UPPER")
            with pytest.raises(ValueError, match="Invalid caller name"):
                add_caller("-starts-with-dash")
            with pytest.raises(ValueError, match="Invalid caller name"):
                add_caller("")
            with pytest.raises(ValueError, match="Invalid caller name"):
                add_caller("has space")
    finally:
        path.unlink(missing_ok=True)


def test_add_builtin_rejected():
    """Cannot add a built-in caller name."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path):
            with pytest.raises(ValueError, match="built-in"):
                add_caller("cli")
            with pytest.raises(ValueError, match="built-in"):
                add_caller("aegis")
    finally:
        path.unlink(missing_ok=True)


def test_add_duplicate_rejected():
    """Cannot add the same custom caller twice."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path):
            add_caller("my-tool")
            with pytest.raises(ValueError, match="already registered"):
                add_caller("my-tool")
    finally:
        path.unlink(missing_ok=True)


def test_remove_builtin_rejected():
    """Cannot remove a built-in caller."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path), pytest.raises(ValueError, match="built-in"):
            remove_caller("cli")
    finally:
        path.unlink(missing_ok=True)


def test_remove_not_found():
    """Cannot remove a caller that isn't registered."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path), pytest.raises(ValueError, match="not registered"):
            remove_caller("ghost")
    finally:
        path.unlink(missing_ok=True)


def test_get_all_callers_includes_both():
    """get_all_callers returns union of built-in and custom callers."""
    path = _tmp_callers_file()
    try:
        with _patch_callers_file(path):
            add_caller("custom1")
            all_c = get_all_callers()
            # All builtins present
            for b in BUILTIN_CALLERS:
                assert b in all_c
            # Custom present
            assert "custom1" in all_c
    finally:
        path.unlink(missing_ok=True)
