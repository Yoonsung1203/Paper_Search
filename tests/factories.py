"""테스트용 도메인 객체 생성기."""

from __future__ import annotations

from datetime import date

from paper_search.models import Paper, Source


def make_paper(
    doi: str = "10.1038/s41586-026-00001-0",
    *,
    title: str = "A spatial atlas of the human brain",
    abstract: str = "We present a spatial transcriptomic atlas.",
    source: Source = Source.PUBMED,
    is_preprint: bool = False,
    issn: str | None = "1476-4687",
    journal: str = "Nature",
) -> Paper:
    return Paper(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=["Kim S", "Lee J"],
        journal=journal,
        issn=issn,
        published_at=date(2026, 8, 20),
        url=f"https://doi.org/{doi}",
        source=source,
        is_preprint=is_preprint,
    )
