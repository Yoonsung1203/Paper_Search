"""Crossref 어댑터 — 검색이 아니라 **메타데이터 보강**용이다.

bioRxiv 레코드에는 ISSN이 없고, PubMed 레코드도 ISSN이 비는 경우가 있다.
ISSN이 없으면 M3의 저널 지표 조회(OpenAlex)가 아예 불가능하므로 여기서 채운다.
"""

from __future__ import annotations

import re
from datetime import date

from paper_search.core.http import FetchClient, SourceUnavailable
from paper_search.models import Paper

BASE = "https://api.crossref.org/works"


def _clean_abstract(raw: str) -> str:
    """Crossref 초록은 JATS 태그가 섞여 있다."""
    text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(text.split())


def _parse_date(message: dict[str, object]) -> date | None:
    for key in ("published", "published-online", "published-print", "issued"):
        node = message.get(key)
        if not isinstance(node, dict):
            continue
        parts = node.get("date-parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        nums = [int(n) for n in parts[0] if isinstance(n, int)]
        if not nums:
            continue
        year = nums[0]
        month = nums[1] if len(nums) > 1 else 1
        day = nums[2] if len(nums) > 2 else 1
        try:
            return date(year, month, day)
        except ValueError:
            return date(year, 1, 1)
    return None


def merge_work(paper: Paper, message: dict[str, object]) -> Paper:
    """Crossref 응답으로 비어 있는 필드만 채운다. 기존 값은 덮어쓰지 않는다."""
    updated = paper.model_copy(deep=True)

    if not updated.journal or updated.is_preprint:
        titles = message.get("container-title")
        if isinstance(titles, list) and titles:
            updated.journal = str(titles[0])

    if not updated.issn:
        issns = message.get("ISSN")
        if isinstance(issns, list) and issns:
            updated.issn = str(issns[0])

    if not updated.abstract:
        raw = message.get("abstract")
        if isinstance(raw, str):
            updated.abstract = _clean_abstract(raw)

    if updated.published_at is None:
        updated.published_at = _parse_date(message)

    if not updated.title:
        titles = message.get("title")
        if isinstance(titles, list) and titles:
            updated.title = str(titles[0])

    return updated


class CrossrefEnricher:
    name = "crossref"

    def __init__(self, fetch: FetchClient) -> None:
        self.fetch = fetch

    async def enrich(self, papers: list[Paper], *, limit: int = 100) -> list[Paper]:
        """ISSN이 비어 있는 논문만 골라 보강한다.

        개별 DOI 조회가 실패해도 그 논문은 원본 그대로 남긴다 — 보강은 부가 기능이므로
        하나가 실패했다고 라운드를 세우지 않는다.
        """
        out: list[Paper] = []
        budget = limit
        for paper in papers:
            needs = paper.issn is None or not paper.journal or not paper.abstract
            if not needs or budget <= 0:
                out.append(paper)
                continue
            budget -= 1
            try:
                payload = await self.fetch.get_json(f"{BASE}/{paper.doi}")
            except SourceUnavailable:
                out.append(paper)
                continue
            message = payload.get("message") if isinstance(payload, dict) else None
            out.append(merge_work(paper, message) if isinstance(message, dict) else paper)
        return out
