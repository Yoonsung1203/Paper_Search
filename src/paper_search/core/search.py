"""검색 단계 오케스트레이션 (M1).

여러 소스를 병렬로 돌리고, 실패한 소스는 경고로 남긴 뒤 나머지로 계속한다.
한 소스가 죽었다고 라운드 전체를 세우지 않는 것이 이 층의 존재 이유다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from paper_search.config import Settings
from paper_search.core.cache import DiskCache
from paper_search.core.dedupe import dedupe
from paper_search.core.http import FetchClient, SourceUnavailable
from paper_search.core.ratelimit import AsyncTokenBucket
from paper_search.models import Paper, RoundInput
from paper_search.sources.base import PaperSource, SearchContext
from paper_search.sources.biorxiv import BiorxivSource
from paper_search.sources.crossref import CrossrefEnricher
from paper_search.sources.pubmed import PubMedSource

logger = logging.getLogger(__name__)

DAY = 60 * 60 * 24


@dataclass
class SearchOutcome:
    papers: list[Paper] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)


async def _run_source(
    source: PaperSource, ctx: SearchContext
) -> tuple[str, list[Paper], str | None]:
    try:
        return source.name, await source.search(ctx), None
    except SourceUnavailable as exc:
        logger.warning("소스 실패: %s", exc)
        return source.name, [], f"{source.name} 수집 실패 — {exc.detail}"
    except Exception as exc:  # noqa: BLE001 — 어떤 소스 오류도 라운드를 세우면 안 된다
        logger.exception("소스 예외: %s", source.name)
        return source.name, [], f"{source.name} 수집 중 오류 — {type(exc).__name__}"


class SearchService:
    """소스 구성과 실행을 한 곳에서 담당한다."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings,
        *,
        sources: list[PaperSource] | None = None,
        enricher: CrossrefEnricher | None = None,
    ) -> None:
        self.settings = settings
        cache = DiskCache(settings.cache_dir)

        def fetch(name: str, rate: float, ttl: float | None) -> FetchClient:
            return FetchClient(
                client,
                source=name,
                bucket=AsyncTokenBucket(rate=rate),
                cache=cache,
                cache_ttl=ttl,
                backoff_base=settings.backoff_base,
            )

        # NCBI: 키 없으면 3 req/s, 키 발급 시 10 req/s
        ncbi_rate = 10.0 if settings.ncbi_api_key else 3.0

        self.sources: list[PaperSource] = sources or [
            PubMedSource(fetch("pubmed", ncbi_rate, DAY), api_key=settings.ncbi_api_key),
            BiorxivSource(fetch("biorxiv", 2.0, DAY), server="biorxiv"),
            BiorxivSource(fetch("medrxiv", 2.0, DAY), server="medrxiv"),
        ]
        self.enricher = enricher or CrossrefEnricher(fetch("crossref", 5.0, 30 * DAY))

    async def search(self, spec: RoundInput) -> SearchOutcome:
        ctx = SearchContext(spec=spec, max_results=self.settings.max_candidates)
        results = await asyncio.gather(*(_run_source(s, ctx) for s in self.sources))

        outcome = SearchOutcome()
        collected: list[Paper] = []
        for name, papers, warning in results:
            outcome.per_source[name] = len(papers)
            if warning:
                outcome.warnings.append(warning)
            collected.extend(papers)

        if not collected and outcome.warnings:
            outcome.warnings.append("모든 소스에서 결과를 얻지 못했습니다.")

        merged = dedupe(collected)
        merged = await self.enricher.enrich(merged, limit=self.settings.max_candidates)
        merged.sort(key=lambda p: (p.published_at is not None, p.published_at), reverse=True)
        outcome.papers = merged[: self.settings.max_candidates]
        return outcome
