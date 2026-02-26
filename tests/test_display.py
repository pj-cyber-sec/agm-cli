"""Tests for display helper functions — pure functions, no mocks needed."""

from __future__ import annotations

import pytest

from agm.queries import format_elapsed, parse_iso_z

# -- parse_iso_z --


def test_parse_iso_z_valid_utc():
    result = parse_iso_z("2025-01-15T10:30:00Z")
    assert result is not None
    assert result.year == 2025
    assert result.month == 1
    assert result.hour == 10


def test_parse_iso_z_valid_offset():
    result = parse_iso_z("2025-01-15T10:30:00+05:00")
    assert result is not None
    assert result.year == 2025


@pytest.mark.parametrize("bad_input", [None, "", "not-a-date", "2025-13-45T99:99:99Z"])
def test_parse_iso_z_rejects_invalid_input(bad_input):
    """None, empty, garbage, and structurally impossible dates all return None."""
    assert parse_iso_z(bad_input) is None


# -- format_elapsed --


@pytest.mark.parametrize("bad_start", [None, "garbage"])
def test_format_elapsed_invalid_start_returns_empty(bad_start):
    assert format_elapsed(bad_start) == ""


def test_format_elapsed_valid_start_returns_duration():
    """A valid timestamp in the past should produce a non-empty duration string."""
    result = format_elapsed("2020-01-01T00:00:00Z")
    assert result != ""


@pytest.mark.parametrize("end_ts", [None, "not-a-date"])
def test_format_elapsed_valid_start_bad_end_still_returns(end_ts):
    """If end_ts is None or unparseable, format_elapsed falls back to now."""
    result = format_elapsed("2020-01-01T00:00:00Z", end_ts=end_ts)
    assert result != ""


def test_format_elapsed_both_valid():
    """Two valid timestamps should produce a duration string."""
    result = format_elapsed("2025-01-01T00:00:00Z", end_ts="2025-01-01T01:30:00Z")
    assert "1h" in result or "90" in result
