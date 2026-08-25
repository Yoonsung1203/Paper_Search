from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
import respx

from paper_search.core.http import FetchClient
from paper_search.core.ratelimit import AsyncTokenBucket
from paper_search.models import RoundInput, Source
from paper_search.sources.base import SearchContext
from paper_search.sources.biorxiv import (
    BiorxivSource,
    map_categories,
    matches_keywords,
    parse_collection,
)
from paper_search.sources.crossref import CrossrefEnricher, merge_work
from paper_search.sources.pubmed import PubMedSource, build_term, parse_efetch
from tests.factories import make_paper


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture
async def fetch() -> AsyncIterator[FetchClient]:
    async with httpx.AsyncClient() as client:
        yield FetchClient(
            client, source="test", bucket=AsyncTokenBucket(rate=1000), sleep=_noop_sleep
        )


@pytest.fixture
def ctx(round_input: RoundInput) -> SearchContext:
    return SearchContext(spec=round_input, max_results=50)


# ---------------------------------------------------------------- PubMed


def test_build_term_ors_keywords_and_authors() -> None:
    term = build_term(["spatial", "cortex"], ["Kim S"])
    assert term == '("spatial"[Title/Abstract] AND "cortex"[Title/Abstract]) OR ("Kim S"[Author])'


def test_build_term_without_authors() -> None:
    assert build_term(["spatial"], []) == '("spatial"[Title/Abstract])'


def test_parse_efetch_extracts_core_fields(load_fixture) -> None:  # type: ignore[no-untyped-def]
    papers = parse_efetch(load_fixture("pubmed_efetch.xml"))

    # DOI 없는 세 번째 레코드는 버려진다
    assert len(papers) == 2

    first = papers[0]
    assert first.doi == "10.1038/s41586-026-00001-0"
    assert first.journal == "Nature"
    assert first.issn == "1476-4687"
    assert first.published_at == date(2026, 8, 20)
    assert first.authors == ["Kim S", "Lee J"]
    assert first.pmid == "40001111"
    assert first.is_preprint is False
    assert first.source is Source.PUBMED
    # 구조화 초록은 라벨을 살려서 이어붙인다
    assert "BACKGROUND:" in first.abstract
    assert "RESULTS:" in first.abstract


def test_parse_efetch_handles_medline_date_and_collective_author(load_fixture) -> None:  # type: ignore[no-untyped-def]
    papers = parse_efetch(load_fixture("pubmed_efetch.xml"))
    second = papers[1]
    assert second.published_at == date(2026, 1, 1)  # "2026 Aug-Sep" → 연도만 복원
    assert second.authors == ["The Spatial Benchmarking Consortium"]
    assert second.doi == "10.1016/j.crmeth.2026.100999"  # ELocationID 경로


@respx.mock
async def test_pubmed_search_end_to_end(
    fetch: FetchClient,
    ctx: SearchContext,
    load_fixture,  # type: ignore[no-untyped-def]
) -> None:
    respx.get(url__regex=r".*esearch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_esearch.json"))
    )
    respx.get(url__regex=r".*efetch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_efetch.xml"))
    )

    papers = await PubMedSource(fetch).search(ctx)
    assert [p.pmid for p in papers] == ["40001111", "40002222"]


@respx.mock
async def test_pubmed_empty_result_skips_efetch(
    fetch: FetchClient,
    ctx: SearchContext,
    load_fixture,  # type: ignore[no-untyped-def]
) -> None:
    respx.get(url__regex=r".*esearch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_esearch_empty.json"))
    )
    efetch = respx.get(url__regex=r".*efetch\.fcgi.*").mock(
        return_value=httpx.Response(200, text="<PubmedArticleSet/>")
    )
    assert await PubMedSource(fetch).search(ctx) == []
    assert efetch.call_count == 0


@respx.mock
async def test_pubmed_api_key_is_sent_when_configured(
    fetch: FetchClient,
    ctx: SearchContext,
    load_fixture,  # type: ignore[no-untyped-def]
) -> None:
    route = respx.get(url__regex=r".*esearch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=load_fixture("pubmed_esearch_empty.json"))
    )
    await PubMedSource(fetch, api_key="secret").search(ctx)
    assert route.calls[0].request.url.params["api_key"] == "secret"


# ---------------------------------------------------------------- bioRxiv


def test_map_categories_from_keywords() -> None:
    assert "genomics" in map_categories(["spatial transcriptomics", "single-cell"])
    assert "neuroscience" in map_categories(["hippocampus circuits"])


def test_map_categories_returns_empty_when_nothing_matches() -> None:
    """빈 집합은 '필터를 걸지 않는다'는 뜻 — 잘못 좁히는 것보다 안전하다."""
    assert map_categories(["quantum ornithology"]) == set()


def test_matches_keywords_phrase_and_token_forms() -> None:
    paper = make_paper(
        title="Spatial transcriptomics of the mouse cortex",
        abstract="We used single cell resolution imaging.",
    )
    assert matches_keywords(paper, ["spatial transcriptomics"])
    assert matches_keywords(paper, ["single-cell"])  # 토큰 분해 후 모두 일치
    assert not matches_keywords(paper, ["protein folding"])


def test_parse_collection_marks_preprint(load_fixture) -> None:  # type: ignore[no-untyped-def]
    import json

    papers = parse_collection(json.loads(load_fixture("biorxiv_page0.json")), "biorxiv")
    assert len(papers) == 3
    assert all(p.is_preprint for p in papers)
    assert all(p.issn is None for p in papers)
    assert papers[0].authors == ["Kim, S.", "Lee, J.", "Park, H."]
    assert papers[0].published_at == date(2026, 8, 19)


@respx.mock
async def test_biorxiv_search_filters_by_keyword(
    fetch: FetchClient,
    ctx: SearchContext,
    load_fixture,  # type: ignore[no-untyped-def]
) -> None:
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/.*").mock(
        return_value=httpx.Response(200, text=load_fixture("biorxiv_page0.json"))
    )
    papers = await BiorxivSource(fetch).search(ctx)

    # 키워드는 single-cell / spatial transcriptomics → 알파인 토양 균류 논문은 탈락
    dois = {p.doi for p in papers}
    assert "10.1101/2026.08.19.611111" in dois
    assert "10.1101/2026.08.22.633333" not in dois


@respx.mock
async def test_biorxiv_skipped_when_preprints_disabled(
    fetch: FetchClient, round_input: RoundInput
) -> None:
    route = respx.get(url__regex=r"https://api\.biorxiv\.org/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    spec = round_input.model_copy(update={"include_preprints": False})
    assert await BiorxivSource(fetch).search(SearchContext(spec=spec)) == []
    assert route.call_count == 0


def test_biorxiv_rejects_unknown_server(fetch: FetchClient) -> None:
    with pytest.raises(ValueError):
        BiorxivSource(fetch, server="arxiv")


# ---------------------------------------------------------------- Crossref


def test_merge_work_fills_only_missing_fields(load_fixture) -> None:  # type: ignore[no-untyped-def]
    import json

    message = json.loads(load_fixture("crossref_work.json"))["message"]
    paper = make_paper(
        doi="10.1101/2026.08.21.622222",
        title="Deep learning inference of neuronal connectivity from calcium imaging",
        abstract="",
        source=Source.BIORXIV,
        is_preprint=True,
        issn=None,
        journal="biorxiv (preprint)",
    )
    merged = merge_work(paper, message)

    assert merged.issn == "0899-7667"
    assert merged.journal == "Neural Computation"
    assert "transformer" in merged.abstract
    assert "<jats:" not in merged.abstract  # JATS 태그 제거


def test_merge_work_does_not_overwrite_existing_issn(load_fixture) -> None:  # type: ignore[no-untyped-def]
    import json

    message = json.loads(load_fixture("crossref_work.json"))["message"]
    paper = make_paper(issn="1476-4687", journal="Nature")
    assert merge_work(paper, message).issn == "1476-4687"


@respx.mock
async def test_enricher_skips_papers_that_need_nothing(fetch: FetchClient) -> None:
    route = respx.get(url__regex=r"https://api\.crossref\.org/works/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    complete = make_paper()  # issn·journal·abstract 모두 있음
    assert await CrossrefEnricher(fetch).enrich([complete]) == [complete]
    assert route.call_count == 0


@respx.mock
async def test_enricher_survives_individual_failure(fetch: FetchClient) -> None:
    """보강은 부가 기능 — 한 건이 실패해도 원본을 그대로 돌려준다."""
    respx.get(url__regex=r"https://api\.crossref\.org/works/.*").mock(
        return_value=httpx.Response(500)
    )
    paper = make_paper(issn=None, journal="")
    out = await CrossrefEnricher(fetch).enrich([paper])
    assert out == [paper]
