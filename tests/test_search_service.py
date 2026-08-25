from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from paper_search.config import Settings
from paper_search.core.http import SourceUnavailable
from paper_search.core.search import SearchService
from paper_search.models import Paper, RoundInput, Source
from paper_search.sources.base import SearchContext
from tests.factories import make_paper

TITLE = "A single-cell spatial transcriptomic atlas of the human cortex"


class StubSource:
    def __init__(self, name: str, papers: list[Paper], error: Exception | None = None) -> None:
        self.name = name
        self.papers = papers
        self.error = error
        self.calls = 0

    async def search(self, ctx: SearchContext) -> list[Paper]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.papers


class StubEnricher:
    def __init__(self) -> None:
        self.calls = 0

    async def enrich(self, papers: list[Paper], *, limit: int = 100) -> list[Paper]:
        self.calls += 1
        return papers


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(cache_dir=tmp_path / "cache", db_path=tmp_path / "t.sqlite3")


def _service(
    client: httpx.AsyncClient, settings: Settings, sources: list[StubSource]
) -> SearchService:
    return SearchService(
        client,
        settings,
        sources=sources,  # type: ignore[arg-type]
        enricher=StubEnricher(),  # type: ignore[arg-type]
    )


async def test_merges_across_sources(
    client: httpx.AsyncClient, settings: Settings, round_input: RoundInput
) -> None:
    pubmed = StubSource("pubmed", [make_paper(doi="10.1038/a", title=TITLE)])
    biorxiv = StubSource(
        "biorxiv",
        [make_paper(doi="10.1101/b", title=TITLE, source=Source.BIORXIV, is_preprint=True)],
    )

    outcome = await _service(client, settings, [pubmed, biorxiv]).search(round_input)

    assert len(outcome.papers) == 1, "동일 논문은 한 건으로 병합되어야 한다"
    assert outcome.papers[0].doi == "10.1038/a"
    assert outcome.per_source == {"pubmed": 1, "biorxiv": 1}


async def test_one_source_failing_does_not_stop_the_round(
    client: httpx.AsyncClient, settings: Settings, round_input: RoundInput
) -> None:
    """계획 §7.2의 필수 회귀 테스트."""
    ok = StubSource("pubmed", [make_paper(doi="10.1038/a")])
    broken = StubSource("biorxiv", [], error=SourceUnavailable("biorxiv", "HTTP 503"))

    outcome = await _service(client, settings, [ok, broken]).search(round_input)

    assert len(outcome.papers) == 1
    assert any("biorxiv 수집 실패" in w for w in outcome.warnings)
    assert "HTTP 503" in outcome.warnings[0]


async def test_unexpected_exception_is_also_contained(
    client: httpx.AsyncClient, settings: Settings, round_input: RoundInput
) -> None:
    ok = StubSource("pubmed", [make_paper(doi="10.1038/a")])
    broken = StubSource("biorxiv", [], error=ValueError("예상 못한 오류"))

    outcome = await _service(client, settings, [ok, broken]).search(round_input)

    assert len(outcome.papers) == 1
    assert any("ValueError" in w for w in outcome.warnings)


async def test_all_sources_failing_reports_clearly(
    client: httpx.AsyncClient, settings: Settings, round_input: RoundInput
) -> None:
    a = StubSource("pubmed", [], error=SourceUnavailable("pubmed", "HTTP 500"))
    b = StubSource("biorxiv", [], error=SourceUnavailable("biorxiv", "HTTP 500"))

    outcome = await _service(client, settings, [a, b]).search(round_input)

    assert outcome.papers == []
    assert "모든 소스에서 결과를 얻지 못했습니다." in outcome.warnings


async def test_results_are_capped_at_max_candidates(
    client: httpx.AsyncClient, settings: Settings, round_input: RoundInput
) -> None:
    settings.max_candidates = 5
    many = [make_paper(doi=f"10.1/{i}", title=f"Paper {i}") for i in range(20)]
    outcome = await _service(client, settings, [StubSource("pubmed", many)]).search(round_input)
    assert len(outcome.papers) == 5


async def test_ncbi_rate_depends_on_api_key(client: httpx.AsyncClient, settings: Settings) -> None:
    without = SearchService(client, settings)
    assert without.sources[0].fetch.bucket.rate == 3.0  # type: ignore[attr-defined]

    settings.ncbi_api_key = "k"
    with_key = SearchService(client, settings)
    assert with_key.sources[0].fetch.bucket.rate == 10.0  # type: ignore[attr-defined]
