"""SQLite 스키마와 마이그레이션.

`PRAGMA user_version` 을 스키마 버전으로 쓴다. 마이그레이션은 앞에서부터 순차 적용된다.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_V1 = """
CREATE TABLE round (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    keywords          TEXT NOT NULL,
    authors           TEXT NOT NULL DEFAULT '[]',
    date_from         TEXT NOT NULL,
    date_to           TEXT NOT NULL,
    impact_threshold  REAL,
    include_preprints INTEGER NOT NULL DEFAULT 1,
    status            TEXT NOT NULL,
    cost_usd          REAL NOT NULL DEFAULT 0.0,
    warnings          TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE paper (
    doi          TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL DEFAULT '',
    authors      TEXT NOT NULL DEFAULT '[]',
    journal      TEXT NOT NULL DEFAULT '',
    issn         TEXT,
    published_at TEXT,
    url          TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL,
    is_preprint  INTEGER NOT NULL DEFAULT 0,
    category     TEXT,
    pmid         TEXT
);

CREATE TABLE journal_metric (
    issn         TEXT PRIMARY KEY,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    source       TEXT NOT NULL,
    year         INTEGER,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE round_paper (
    round_id        INTEGER NOT NULL REFERENCES round(id) ON DELETE CASCADE,
    doi             TEXT NOT NULL REFERENCES paper(doi),
    relevance_score INTEGER,
    evidence        TEXT NOT NULL DEFAULT '[]',
    reason          TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '[]',
    summary_basis   TEXT,
    novelty_claim   TEXT NOT NULL DEFAULT '',
    novelty_category TEXT,
    novelty_confidence TEXT,
    rank            INTEGER,
    PRIMARY KEY (round_id, doi)
);

CREATE TABLE selection (
    round_id    INTEGER NOT NULL REFERENCES round(id) ON DELETE CASCADE,
    doi         TEXT NOT NULL,
    selected    INTEGER NOT NULL,
    selected_at TEXT NOT NULL,
    PRIMARY KEY (round_id, doi)
);

CREATE TABLE criteria (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id INTEGER NOT NULL REFERENCES round(id) ON DELETE CASCADE,
    text     TEXT NOT NULL,
    origin   TEXT NOT NULL DEFAULT 'inferred',
    active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE llm_call (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id      INTEGER REFERENCES round(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    created_at    TEXT NOT NULL
);

CREATE INDEX idx_round_paper_round ON round_paper(round_id);
CREATE INDEX idx_selection_round ON selection(round_id);
CREATE INDEX idx_criteria_round ON criteria(round_id);
CREATE INDEX idx_llm_call_round ON llm_call(round_id);
"""

# v2: KPI 측정을 위한 완료 시각. 라운드 소요 시간(PRD §9)을 재려면 종료 시각이 필요하다.
SCHEMA_V2 = """
ALTER TABLE round ADD COLUMN completed_at TEXT;
"""

MIGRATIONS: list[str] = [SCHEMA_V1, SCHEMA_V2]


def migrate(conn: sqlite3.Connection) -> int:
    """미적용 마이그레이션을 순차 적용하고 최종 스키마 버전을 반환한다."""
    current: int = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, ddl in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript(ddl)
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    return len(MIGRATIONS)


def connect(db_path: Path | str) -> sqlite3.Connection:
    """마이그레이션이 적용된 연결을 돌려준다."""
    path = Path(db_path)
    if path.parent != Path("") and str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), detect_types=0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


@contextmanager
def session(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
