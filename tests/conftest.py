"""Shared test fixtures — template DB for fast per-test isolation."""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from agm.db import add_project, get_connection


@pytest.fixture(scope="session")
def _db_template_path() -> Path:
    """Create a single template DB with full schema + a default project.

    Copying this file (~0.2 ms) is vastly cheaper than running all
    migrations from scratch (~141 ms) in every test function.
    """
    fd, path_str = tempfile.mkstemp(suffix=".db")
    path = Path(path_str)
    try:
        conn = get_connection(path)
        add_project(conn, "testproj", "/tmp/testproj")
        conn.close()
        yield path
    finally:
        path.unlink(missing_ok=True)


@pytest.fixture()
def db_conn(tmp_path: Path, _db_template_path: Path) -> sqlite3.Connection:
    """Per-test DB connection with schema + testproj pre-loaded."""
    db_path = tmp_path / "test.db"
    shutil.copy2(_db_template_path, db_path)
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def db_conn_path(tmp_path: Path, _db_template_path: Path) -> tuple[sqlite3.Connection, Path]:
    """Per-test DB connection + path (for tests that re-open the DB)."""
    db_path = tmp_path / "test.db"
    shutil.copy2(_db_template_path, db_path)
    conn = get_connection(db_path)
    try:
        yield conn, db_path
    finally:
        conn.close()
