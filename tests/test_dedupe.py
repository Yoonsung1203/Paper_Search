from __future__ import annotations

from paper_search.core.dedupe import dedupe
from paper_search.models import Source
from tests.factories import make_paper

TITLE = "A single-cell spatial transcriptomic atlas of the human cortex"


def test_same_doi_is_merged() -> None:
    a = make_paper(doi="10.1/x", abstract="short")
    b = make_paper(doi="https://doi.org/10.1/X", abstract="a much longer abstract")
    merged = dedupe([a, b])
    assert len(merged) == 1
    assert merged[0].abstract == "a much longer abstract"


def test_preprint_and_published_version_merge_into_one() -> None:
    """계획 §7.2의 필수 회귀 테스트 — DOI가 달라도 제목이 같으면 한 건이어야 한다."""
    preprint = make_paper(
        doi="10.1101/2026.08.19.611111",
        title=TITLE,
        source=Source.BIORXIV,
        is_preprint=True,
        issn=None,
        journal="biorxiv (preprint)",
        abstract="preprint abstract that happens to be quite long indeed",
    )
    published = make_paper(
        doi="10.1038/s41586-026-00001-0",
        title=f"<i>{TITLE}</i>!",  # 마크업·구두점 차이
        source=Source.PUBMED,
        journal="Nature",
        issn="1476-4687",
        abstract="short",
    )

    merged = dedupe([preprint, published])

    assert len(merged) == 1
    result = merged[0]
    # 피어리뷰본이 대표가 된다
    assert result.doi == "10.1038/s41586-026-00001-0"
    assert result.is_preprint is False
    assert result.issn == "1476-4687"
    # 더 긴 초록은 프리프린트 쪽에서 가져온다
    assert result.abstract == "preprint abstract that happens to be quite long indeed"


def test_published_first_then_preprint_gives_same_result() -> None:
    preprint = make_paper(doi="10.1101/a", title=TITLE, source=Source.BIORXIV, is_preprint=True)
    published = make_paper(doi="10.1038/b", title=TITLE, source=Source.PUBMED)
    assert dedupe([published, preprint])[0].doi == "10.1038/b"
    assert dedupe([preprint, published])[0].doi == "10.1038/b"


def test_distinct_papers_are_kept() -> None:
    a = make_paper(doi="10.1/a", title="Paper A")
    b = make_paper(doi="10.1/b", title="Paper B")
    assert len(dedupe([a, b])) == 2


def test_input_order_is_preserved() -> None:
    papers = [make_paper(doi=f"10.1/{i}", title=f"Title {i}") for i in range(5)]
    assert [p.doi for p in dedupe(papers)] == [p.doi for p in papers]


def test_pmid_is_carried_over_when_missing() -> None:
    a = make_paper(doi="10.1/x", title=TITLE, source=Source.BIORXIV, is_preprint=True)
    b = make_paper(doi="10.1/y", title=TITLE)
    b.pmid = "40001111"
    assert dedupe([a, b])[0].pmid == "40001111"
