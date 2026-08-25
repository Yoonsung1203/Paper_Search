from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date

import pytest

from paper_search.models import RoundInput
from paper_search.store import Repository, connect


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def repo(conn: sqlite3.Connection) -> Repository:
    return Repository(conn)


@pytest.fixture
def round_input() -> RoundInput:
    return RoundInput(
        keywords=["single-cell", "spatial transcriptomics"],
        date_from=date(2026, 8, 18),
        date_to=date(2026, 8, 25),
        impact_threshold=10.0,
    )
