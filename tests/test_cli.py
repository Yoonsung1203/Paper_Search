"""M1 게이트 검증 — 키워드+기간 입력이 중복 제거된 후보 리스트로 DB에 저장되는가."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from paper_search.__main__ import main
from paper_search.config import get_settings
from paper_search.models import RoundStatus
from paper_search.store import Repository, connect


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "test.sqlite3"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NCBI_API_KEY", "")
    monkeypatch.setenv("BACKOFF_BASE", "0.001")
    monkeypatch.chdir(tmp_path)  # 개발자 .env를 읽지 않도록
    get_settings.cache_clear()
    yield db
    get_settings.cache_clear()


def _mock_all(load_fixture) -> None:  # type: ignore[no-untyped-def]
    respx.get(url__regex=r".*esearch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_esearch.json"))
    )
    respx.get(url__regex=r".*efetch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_efetch.xml"))
    )
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(200, text=load_fixture("biorxiv_page0.json"))
    )
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/medrxiv/.*").mock(
        return_value=httpx.Response(200, json={"messages": [], "collection": []})
    )
    respx.get(url__regex=r"https://api\.crossref\.org/works/.*").mock(
        return_value=httpx.Response(200, text=load_fixture("crossref_work.json"))
    )


def test_init_db(isolated_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init-db"]) == 0
    assert isolated_env.exists()
    assert "스키마 버전 2" in capsys.readouterr().out


def test_search_requires_keyword_or_author(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search"]) == 2
    assert "키워드" in capsys.readouterr().err


@respx.mock
def test_search_stores_deduped_candidates(
    isolated_env: Path,
    load_fixture,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    _mock_all(load_fixture)

    exit_code = main(
        [
            "search",
            "-k",
            "single-cell",
            "-k",
            "spatial transcriptomics",
            "--from",
            "2026-08-18",
            "--to",
            "2026-08-25",
        ]
    )
    assert exit_code == 0

    conn = connect(isolated_env)
    repo = Repository(conn)
    rounds = repo.list_rounds()
    assert len(rounds) == 1
    round_id = rounds[0]["id"]

    papers = repo.load_round_papers(round_id)
    dois = {p.paper.doi for p in papers}

    # PubMed 2건 + bioRxiv 1건(키워드 통과), 그중 cortex atlas는 양쪽에 있어 1건으로 병합
    assert "10.1038/s41586-026-00001-0" in dois
    assert "10.1101/2026.08.19.611111" not in dois, "프리프린트는 저널본으로 병합되어야 한다"

    atlas = next(p for p in papers if p.paper.doi == "10.1038/s41586-026-00001-0")
    assert atlas.paper.is_preprint is False
    assert atlas.paper.issn == "1476-4687"

    assert repo.get_status(round_id) is RoundStatus.AWAITING_SELECTION
    conn.close()

    out = capsys.readouterr().out
    assert "라운드 #1" in out
    assert "pubmed: 2건" in out


@respx.mock
def test_search_records_warning_when_source_fails(
    isolated_env: Path,
    load_fixture,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    _mock_all(load_fixture)
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(503)
    )

    assert main(["search", "-k", "single-cell", "--days", "7"]) == 0

    conn = connect(isolated_env)
    repo = Repository(conn)
    round_id = repo.list_rounds()[0]["id"]
    warnings = repo.get_warnings(round_id)
    conn.close()

    assert any("biorxiv" in w for w in warnings)
    assert "⚠" in capsys.readouterr().out


@respx.mock
def test_rounds_listing(isolated_env: Path, load_fixture, capsys) -> None:  # type: ignore[no-untyped-def]
    _mock_all(load_fixture)
    main(["search", "-k", "single-cell"])
    capsys.readouterr()
    assert main(["rounds"]) == 0
    assert "#  1" in capsys.readouterr().out
