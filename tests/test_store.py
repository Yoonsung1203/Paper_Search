from __future__ import annotations

import sqlite3

import pytest

from paper_search.models import (
    Confidence,
    InferredCriterion,
    JournalMetric,
    NoveltyCategory,
    PaperSummary,
    RelevanceScore,
    RoundInput,
    RoundStatus,
    Source,
    SummaryBasis,
)
from paper_search.store import Repository, connect, migrate
from tests.factories import make_paper


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    before = conn.execute("PRAGMA user_version").fetchone()[0]
    assert migrate(conn) == before
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"round", "paper", "round_paper", "selection", "criteria", "llm_call"} <= tables


def test_round_roundtrip(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    assert repo.get_round_input(rid) == round_input
    assert repo.get_status(rid) is RoundStatus.CREATED


def test_upsert_paper_keeps_longer_abstract(repo: Repository) -> None:
    short = make_paper()
    short.abstract = "short"
    repo.upsert_papers([short])

    long = make_paper()
    long.abstract = "a much longer abstract with real content"
    repo.upsert_papers([long])

    row = repo.conn.execute("SELECT abstract FROM paper WHERE doi = ?", (short.doi,)).fetchone()
    assert row["abstract"] == long.abstract


def test_relevance_and_summary_roundtrip(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    paper = make_paper()
    repo.attach_papers(rid, [paper])
    repo.save_relevance(
        rid, paper.doi, RelevanceScore(score=88, evidence=["spatial atlas"], reason="부합")
    )
    repo.save_summary(
        rid,
        paper.doi,
        PaperSummary(
            summary=["첫 줄", "둘째 줄", "셋째 줄"],
            basis=SummaryBasis.ABSTRACT,
            novelty_claim="새 방법론",
            novelty_category=NoveltyCategory.NEW_METHOD,
            novelty_confidence=Confidence.HIGH,
        ),
    )

    loaded = repo.load_round_papers(rid)
    assert len(loaded) == 1
    sp = loaded[0]
    assert sp.relevance is not None and sp.relevance.score == 88
    assert sp.summary is not None
    assert sp.summary.basis is SummaryBasis.ABSTRACT
    assert sp.summary.novelty_category is NoveltyCategory.NEW_METHOD


def test_selection_records_non_selected_too(repo: Repository, round_input: RoundInput) -> None:
    """F-06은 비선택군도 입력으로 쓴다 — 선택 안 한 논문도 기록되어야 한다."""
    rid = repo.create_round(round_input)
    a = make_paper(doi="10.1/a")
    b = make_paper(doi="10.1/b")
    repo.attach_papers(rid, [a, b])

    repo.save_selection(rid, {a.doi})

    sel = repo.get_selection(rid)
    assert sel == {"10.1/a": True, "10.1/b": False}


def test_selection_is_idempotent(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.attach_papers(rid, [make_paper(doi="10.1/a")])
    repo.save_selection(rid, {"10.1/a"})
    repo.save_selection(rid, set())
    assert repo.get_selection(rid) == {"10.1/a": False}


def test_warnings_are_deduplicated(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.add_warning(rid, "bioRxiv 수집 실패")
    repo.add_warning(rid, "bioRxiv 수집 실패")
    assert repo.get_warnings(rid) == ["bioRxiv 수집 실패"]


def test_cost_accumulates_on_round(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.record_llm_call(rid, "score", "claude-opus-5", 1000, 100, 0.0075)
    repo.record_llm_call(rid, "score", "claude-opus-5", 1000, 100, 0.0075)
    assert repo.round_cost(rid) == pytest.approx(0.015)


def test_cache_hit_ratio(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.record_llm_call(rid, "score", "claude-opus-5", 200, 50, 0.001, cache_read_tokens=800)
    assert repo.cache_hit_ratio(rid) == pytest.approx(0.8)


def test_criteria_inherit_from_previous_round(repo: Repository, round_input: RoundInput) -> None:
    first = repo.create_round(round_input)
    repo.save_criteria(first, [InferredCriterion(text="in vivo 검증 포함")])
    second = repo.create_round(round_input)
    assert repo.latest_criteria_before(second) == ["in vivo 검증 포함"]


def test_deactivated_criterion_is_not_returned(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.save_criteria(rid, [InferredCriterion(text="사람 데이터")])
    cid = repo.list_criteria(rid)[0]["id"]
    repo.set_criterion_active(cid, False)
    assert repo.active_criteria(rid) == []


def test_journal_metric_roundtrip(repo: Repository) -> None:
    repo.upsert_metric(
        JournalMetric(
            issn="1476-4687",
            metric_name="2yr_mean_citedness",
            metric_value=18.5,
            source="OpenAlex",
            year=2026,
        )
    )
    m = repo.get_metric("1476-4687")
    assert m is not None
    assert "OpenAlex" in m.display()
    assert "2yr_mean_citedness" in m.display()


def test_load_result_includes_warnings_and_cost(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.attach_papers(rid, [make_paper(source=Source.BIORXIV, is_preprint=True, issn=None)])
    repo.add_warning(rid, "PubMed 일부 실패")
    repo.record_llm_call(rid, "score", "claude-opus-5", 100, 10, 0.001)
    repo.set_status(rid, RoundStatus.PARTIAL)

    result = repo.load_result(rid)
    assert result.status is RoundStatus.PARTIAL
    assert result.warnings == ["PubMed 일부 실패"]
    assert result.cost_usd > 0
    assert result.papers[0].paper.is_preprint is True


def test_foreign_key_cascade_on_round_delete(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.attach_papers(rid, [make_paper()])
    repo.conn.execute("DELETE FROM round WHERE id = ?", (rid,))
    repo.conn.commit()
    assert repo.load_round_papers(rid) == []


def test_connect_creates_parent_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = tmp_path / "nested" / "dir" / "test.sqlite3"
    conn = connect(db)
    conn.close()
    assert db.exists()


def test_cost_breakdown_groups_by_stage(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    repo.record_llm_call(rid, "score", "claude-opus-5", 100, 20, 0.001, cache_read_tokens=500)
    repo.record_llm_call(rid, "score", "claude-opus-5", 100, 20, 0.001, cache_read_tokens=500)
    repo.record_llm_call(rid, "summarize", "claude-opus-5", 3000, 400, 0.025)

    rows = {r["stage"]: r for r in repo.cost_breakdown(rid)}
    assert rows["score"]["calls"] == 2
    assert rows["score"]["cache_read_tokens"] == 1000
    assert rows["summarize"]["cost_usd"] == pytest.approx(0.025)
    # 비용 내림차순 정렬
    assert [r["stage"] for r in repo.cost_breakdown(rid)][0] == "summarize"
