from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from paper_search.core.http import FetchClient
from paper_search.core.impact import ImpactFilter
from paper_search.core.ratelimit import AsyncTokenBucket
from paper_search.models import JournalMetric
from paper_search.sources.openalex import OpenAlexMetrics, parse_source
from paper_search.store import Repository
from tests.factories import make_paper

NATURE = {
    "id": "https://openalex.org/S137773608",
    "display_name": "Nature",
    "issn_l": "0028-0836",
    "issn": ["0028-0836", "1476-4687"],
    "summary_stats": {"2yr_mean_citedness": 18.53, "h_index": 1489, "i10_index": 210000},
}


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture
async def fetch() -> AsyncIterator[FetchClient]:
    async with httpx.AsyncClient() as client:
        yield FetchClient(
            client, source="openalex", bucket=AsyncTokenBucket(rate=1000), sleep=_noop_sleep
        )


def test_parse_source_extracts_metric() -> None:
    metric = parse_source(NATURE, "1476-4687")
    assert metric is not None
    assert metric.metric_name == "2yr_mean_citedness"
    assert metric.metric_value == pytest.approx(18.53)
    assert metric.source == "OpenAlex"


def test_metric_display_never_says_if() -> None:
    """UI에서 'IF'로 단독 표기하지 않기 위한 계약 (PRD §6.2-2)."""
    metric = parse_source(NATURE, "1476-4687")
    assert metric is not None
    text = metric.display()
    assert "2yr_mean_citedness" in text
    assert "OpenAlex" in text
    assert "IF" not in text


def test_parse_source_returns_none_without_stats() -> None:
    assert parse_source({"display_name": "Obscure Journal"}, "1234-5678") is None
    assert parse_source({"summary_stats": {}}, "1234-5678") is None


@respx.mock
async def test_metric_for_returns_none_on_failure(fetch: FetchClient) -> None:
    respx.get(url__regex=r"https://api\.openalex\.org/sources/.*").mock(
        return_value=httpx.Response(404)
    )
    assert await OpenAlexMetrics(fetch).metric_for("0000-0000") is None


@respx.mock
async def test_mailto_is_sent_for_polite_pool(fetch: FetchClient) -> None:
    route = respx.get(url__regex=r"https://api\.openalex\.org/sources/.*").mock(
        return_value=httpx.Response(200, text=json.dumps(NATURE))
    )
    await OpenAlexMetrics(fetch, mailto="me@example.com").metric_for("1476-4687")
    assert route.calls[0].request.url.params["mailto"] == "me@example.com"


# ---------------------------------------------------------------- 필터


class StubMetrics:
    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls: list[str] = []

    async def metric_for(self, issn: str) -> JournalMetric | None:
        self.calls.append(issn)
        value = self.table.get(issn)
        if value is None:
            return None
        return JournalMetric(
            issn=issn, metric_name="2yr_mean_citedness", metric_value=value, source="OpenAlex"
        )


async def test_preprints_are_never_dropped(repo: Repository) -> None:
    """PRD F-05 수용 기준 — 프리프린트는 저널 지표가 없어 필터 대상이 아니다."""
    preprint = make_paper(doi="10.1101/a", is_preprint=True, issn=None, journal="biorxiv")
    low = make_paper(doi="10.1/low", title="Low", issn="1111-1111")

    filt = ImpactFilter(StubMetrics({"1111-1111": 2.0}), repo)  # type: ignore[arg-type]
    outcome = await filt.apply([preprint, low], threshold=10.0)

    dois = {p.doi for p in outcome.papers}
    assert "10.1101/a" in dois
    assert "10.1/low" not in dois
    assert outcome.dropped == 1
    assert outcome.preprints == 1


async def test_unknown_metric_is_kept(repo: Repository) -> None:
    unknown = make_paper(doi="10.1/x", issn="9999-9999")
    filt = ImpactFilter(StubMetrics({}), repo)  # type: ignore[arg-type]
    outcome = await filt.apply([unknown], threshold=10.0)

    assert [p.doi for p in outcome.papers] == ["10.1/x"]
    assert outcome.unknown == 1
    assert outcome.dropped == 0


async def test_above_threshold_passes(repo: Repository) -> None:
    high = make_paper(doi="10.1/high", issn="1476-4687")
    filt = ImpactFilter(StubMetrics({"1476-4687": 18.5}), repo)  # type: ignore[arg-type]
    outcome = await filt.apply([high], threshold=10.0)
    assert len(outcome.papers) == 1


async def test_no_threshold_keeps_everything_but_still_fetches(repo: Repository) -> None:
    """임계값이 없어도 지표는 조회한다 — 화면에 표시해야 하기 때문."""
    low = make_paper(doi="10.1/low", issn="1111-1111")
    metrics = StubMetrics({"1111-1111": 1.0})
    outcome = await ImpactFilter(metrics, repo).apply([low], threshold=None)  # type: ignore[arg-type]

    assert len(outcome.papers) == 1
    assert metrics.calls == ["1111-1111"]
    assert "1111-1111" in outcome.metrics


async def test_metrics_are_cached_in_db(repo: Repository) -> None:
    papers = [make_paper(doi=f"10.1/{i}", title=f"P{i}", issn="1476-4687") for i in range(3)]
    metrics = StubMetrics({"1476-4687": 18.5})
    await ImpactFilter(metrics, repo).apply(papers, threshold=None)  # type: ignore[arg-type]

    assert metrics.calls == ["1476-4687"], "같은 ISSN은 한 번만 조회한다"
    assert repo.get_metric("1476-4687") is not None

    # 두 번째 라운드는 DB 캐시를 쓴다
    metrics2 = StubMetrics({})
    outcome = await ImpactFilter(metrics2, repo).apply(papers, threshold=None)  # type: ignore[arg-type]
    assert metrics2.calls == []
    assert "1476-4687" in outcome.metrics


def test_describe_explains_what_was_kept() -> None:
    from paper_search.core.impact import ImpactOutcome

    outcome = ImpactOutcome(dropped=12, unknown=3, preprints=5)
    note = ImpactFilter.describe(outcome, 10.0)
    assert note is not None
    assert "12건 제외" in note
    assert "미확인 3건은 유지" in note
    assert "프리프린트 5건" in note


def test_describe_is_silent_without_threshold() -> None:
    from paper_search.core.impact import ImpactOutcome

    assert ImpactFilter.describe(ImpactOutcome(), None) is None
