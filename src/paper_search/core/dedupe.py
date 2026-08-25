"""중복 제거.

같은 연구가 bioRxiv 프리프린트와 저널 논문으로 각각 잡히는 일이 흔하다.
DOI가 다르므로 DOI만으로는 병합되지 않아, 정규화된 제목까지 본다.
"""

from __future__ import annotations

from paper_search.models import Paper, normalize_title


def _merge(primary: Paper, other: Paper) -> Paper:
    """두 레코드를 하나로 합친다. 피어리뷰본을 대표로 삼는다."""
    if primary.is_preprint and not other.is_preprint:
        primary, other = other, primary

    merged = primary.model_copy(deep=True)
    if len(other.abstract) > len(merged.abstract):
        merged.abstract = other.abstract
    if not merged.issn:
        merged.issn = other.issn
    if not merged.pmid:
        merged.pmid = other.pmid
    if not merged.category:
        merged.category = other.category
    if not merged.journal:
        merged.journal = other.journal
    if len(other.authors) > len(merged.authors):
        merged.authors = other.authors
    if merged.published_at is None:
        merged.published_at = other.published_at
    return merged


def dedupe(papers: list[Paper]) -> list[Paper]:
    """DOI 우선, 제목 보조로 병합한다. 입력 순서를 유지한다."""
    order: list[str] = []
    by_doi: dict[str, Paper] = {}
    title_to_doi: dict[str, str] = {}

    for paper in papers:
        title_key = normalize_title(paper.title)

        target_doi: str | None = None
        if paper.doi in by_doi:
            target_doi = paper.doi
        elif title_key and title_key in title_to_doi:
            target_doi = title_to_doi[title_key]

        if target_doi is None:
            by_doi[paper.doi] = paper
            order.append(paper.doi)
            if title_key:
                title_to_doi[title_key] = paper.doi
            continue

        by_doi[target_doi] = _merge(by_doi[target_doi], paper)

    return [by_doi[doi] for doi in order]
