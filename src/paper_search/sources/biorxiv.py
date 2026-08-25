"""bioRxiv / medRxiv 어댑터.

이 API는 **키워드 검색을 지원하지 않는다** (날짜·서버 단위 조회만 가능). 따라서
build_plan §4.1의 4단계를 그대로 구현한다:

1. 사용자 키워드 → 카테고리 매핑
2. 해당 기간 전량 수집
3. 카테고리 + 제목/초록 문자열로 값싼 사전 필터
4. 남은 후보만 LLM 단계로 넘긴다 (전량을 LLM에 넣지 않는다)
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from importlib import resources

from paper_search.core.http import FetchClient
from paper_search.models import Paper, Source, normalize_title
from paper_search.sources.base import SearchContext

BASE = "https://api.biorxiv.org/details"
PAGE_SIZE = 100
MAX_PAGES = 60  # 안전장치: 한 라운드에서 6,000건 이상은 받지 않는다


@lru_cache(maxsize=1)
def _category_map() -> dict[str, dict[str, list[str]]]:
    raw = (
        resources.files("paper_search.data")
        .joinpath("biorxiv_categories.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(raw)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def map_categories(keywords: list[str], server: str = "biorxiv") -> set[str]:
    """키워드가 어떤 카테고리에도 걸리지 않으면 빈 집합을 돌려준다.

    호출부는 빈 집합을 '카테고리 필터를 걸지 않는다'로 해석해야 한다 — 잘못 좁히는 것보다
    전량 수집 후 문자열 필터에 맡기는 편이 재현율에 안전하다.
    """
    table = _category_map().get(server, {})
    haystack = " ".join(keywords).lower()
    return {
        category
        for category, hints in table.items()
        if category in haystack or any(hint in haystack for hint in hints)
    }


def _tokenize(keyword: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9가-힣]+", keyword.lower()) if len(t) > 2]


def matches_keywords(paper: Paper, keywords: list[str]) -> bool:
    """제목+초록에 키워드가 걸리는지 본다 (사전 필터, 3단계).

    구(phrase) 전체가 걸리거나, 구를 이루는 토큰이 모두 걸리면 통과시킨다.
    """
    haystack = normalize_title(f"{paper.title} {paper.abstract}")
    for keyword in keywords:
        needle = normalize_title(keyword)
        if needle and needle in haystack:
            return True
        tokens = _tokenize(keyword)
        if tokens and all(token in haystack for token in tokens):
            return True
    return False


def parse_collection(payload: dict[str, object], server: str) -> list[Paper]:
    source = Source.MEDRXIV if server == "medrxiv" else Source.BIORXIV
    entries = payload.get("collection") or []
    papers: list[Paper] = []
    if not isinstance(entries, list):
        return papers

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doi = str(entry.get("doi", "")).strip()
        if not doi:
            continue
        raw_authors = str(entry.get("authors", ""))
        published: date | None = None
        raw_date = str(entry.get("date", ""))
        try:
            published = date.fromisoformat(raw_date)
        except ValueError:
            published = None
        papers.append(
            Paper(
                doi=doi,
                title=str(entry.get("title", "")).strip(),
                abstract=str(entry.get("abstract", "")).strip(),
                authors=[a.strip() for a in raw_authors.split(";") if a.strip()],
                journal=f"{source.value} (preprint)",
                issn=None,
                published_at=published,
                url=f"https://doi.org/{doi}",
                source=source,
                is_preprint=True,
                category=str(entry.get("category", "")).strip() or None,
            )
        )
    return papers


def total_from_payload(payload: dict[str, object]) -> int:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        try:
            return int(messages[0].get("total", 0))
        except (TypeError, ValueError):
            return 0
    return 0


class BiorxivSource:
    def __init__(self, fetch: FetchClient, *, server: str = "biorxiv") -> None:
        if server not in {"biorxiv", "medrxiv"}:
            raise ValueError("server는 'biorxiv' 또는 'medrxiv'여야 합니다")
        self.fetch = fetch
        self.server = server
        self.name = server

    async def collect(self, start: date, end: date) -> list[Paper]:
        """기간 내 전량 수집 (2단계). 페이지 상한에 걸리면 거기서 멈춘다."""
        papers: list[Paper] = []
        cursor = 0
        for _ in range(MAX_PAGES):
            payload = await self.fetch.get_json(
                f"{BASE}/{self.server}/{start.isoformat()}/{end.isoformat()}/{cursor}"
            )
            page = parse_collection(payload, self.server)
            papers.extend(page)
            total = total_from_payload(payload)
            cursor += PAGE_SIZE
            if len(page) < PAGE_SIZE or cursor >= total:
                break
        return papers

    async def search(self, ctx: SearchContext) -> list[Paper]:
        if not ctx.spec.include_preprints:
            return []

        collected = await self.collect(ctx.spec.date_from, ctx.spec.date_to)
        categories = map_categories(ctx.spec.keywords, self.server)

        if categories:
            collected = [
                p for p in collected if (p.category or "").lower() in categories
            ] or collected

        filtered = [p for p in collected if matches_keywords(p, ctx.spec.keywords)]

        # 저자 이름으로도 한 번 훑는다 (F-01의 연구자 검색 경로)
        if ctx.spec.authors:
            author_needles = [a.lower() for a in ctx.spec.authors]
            seen = {p.doi for p in filtered}
            for paper in collected:
                if paper.doi in seen:
                    continue
                blob = " ".join(paper.authors).lower()
                if any(needle in blob for needle in author_needles):
                    filtered.append(paper)

        filtered.sort(key=lambda p: p.published_at or date.min, reverse=True)
        return filtered[: ctx.max_results]
