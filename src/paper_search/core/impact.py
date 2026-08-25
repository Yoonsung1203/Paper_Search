"""저널 임팩트 지표 조회와 임계값 필터 (M3).

두 가지 규칙이 핵심이다.
- 프리프린트는 저널 지표가 존재하지 않는다. 필터로 탈락시키지 않는다.
- 지표를 찾지 못한 저널도 탈락시키지 않는다. '미확인'으로 표시할 뿐이다.
둘 다 PRD F-05의 수용 기준이며, 그러지 않으면 재현율을 조용히 잃는다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from paper_search.models import JournalMetric, Paper
from paper_search.sources.openalex import OpenAlexMetrics
from paper_search.store import Repository

logger = logging.getLogger(__name__)


@dataclass
class ImpactOutcome:
    papers: list[Paper] = field(default_factory=list)
    metrics: dict[str, JournalMetric] = field(default_factory=dict)
    dropped: int = 0
    unknown: int = 0
    preprints: int = 0


class ImpactFilter:
    def __init__(self, metrics: OpenAlexMetrics, repo: Repository) -> None:
        self.metrics = metrics
        self.repo = repo

    async def _metric(self, issn: str) -> JournalMetric | None:
        cached = self.repo.get_metric(issn)
        if cached is not None:
            return cached
        metric = await self.metrics.metric_for(issn)
        if metric is not None:
            self.repo.upsert_metric(metric)
        return metric

    async def apply(self, papers: list[Paper], threshold: float | None) -> ImpactOutcome:
        outcome = ImpactOutcome()

        issns = {p.issn for p in papers if p.issn}
        fetched = await asyncio.gather(*(self._metric(issn) for issn in issns))
        outcome.metrics = {m.issn: m for m in fetched if m is not None}

        for paper in papers:
            if paper.is_preprint:
                outcome.preprints += 1
                outcome.papers.append(paper)
                continue

            metric = outcome.metrics.get(paper.issn or "")
            if metric is None:
                outcome.unknown += 1
                outcome.papers.append(paper)
                continue

            if threshold is not None and metric.metric_value < threshold:
                outcome.dropped += 1
                continue

            outcome.papers.append(paper)

        return outcome

    @staticmethod
    def describe(outcome: ImpactOutcome, threshold: float | None) -> str | None:
        """무엇이 걸러졌고 무엇이 통과되었는지 사용자에게 알린다."""
        if threshold is None:
            return None
        parts = [
            f"임팩트 하한 {threshold:g} 적용: {outcome.dropped}건 제외",
        ]
        if outcome.unknown:
            parts.append(f"지표 미확인 {outcome.unknown}건은 유지")
        if outcome.preprints:
            parts.append(f"프리프린트 {outcome.preprints}건은 지표가 없어 필터 대상이 아님")
        return " · ".join(parts) + "."
