"""OpenAlex 어댑터 — 저널 임팩트 지표(IF 대용).

**JCR Impact Factor는 Clarivate 독점이라 무료 API로 얻을 수 없다.** 대신 OpenAlex의
`2yr_mean_citedness`(최근 2년 논문의 평균 피인용)를 쓴다. JIF와 계산 방식이 유사하나
같은 수가 아니므로, UI에서는 반드시 지표명과 출처를 함께 노출한다 (PRD §6.2-2).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from paper_search.core.http import FetchClient, SourceUnavailable
from paper_search.models import JournalMetric

logger = logging.getLogger(__name__)

BASE = "https://api.openalex.org/sources"
METRIC_NAME = "2yr_mean_citedness"
SOURCE_NAME = "OpenAlex"


def parse_source(payload: dict[str, object], issn: str) -> JournalMetric | None:
    stats = payload.get("summary_stats")
    if not isinstance(stats, dict):
        return None
    raw = stats.get(METRIC_NAME)
    if not isinstance(raw, (int, float)):
        return None
    return JournalMetric(
        issn=issn,
        metric_name=METRIC_NAME,
        metric_value=float(raw),
        source=SOURCE_NAME,
        year=None,  # OpenAlex는 summary_stats에 기준연도를 주지 않는다
        fetched_at=datetime.now(UTC),
    )


class OpenAlexMetrics:
    name = "openalex"

    def __init__(self, fetch: FetchClient, *, mailto: str | None = None) -> None:
        self.fetch = fetch
        self.mailto = mailto

    async def metric_for(self, issn: str) -> JournalMetric | None:
        """지표를 찾지 못하면 None. 호출부는 이를 '미확인'으로 표시하되 제외하지 않는다."""
        params = {"mailto": self.mailto} if self.mailto else None
        try:
            payload = await self.fetch.get_json(f"{BASE}/issn:{issn}", params)
        except SourceUnavailable as exc:
            logger.info("저널 지표 조회 실패 (%s): %s", issn, exc.detail)
            return None
        if not isinstance(payload, dict):
            return None
        return parse_source(payload, issn)
