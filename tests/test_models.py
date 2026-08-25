from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from paper_search.models import (
    Paper,
    RelevanceScore,
    RoundInput,
    Source,
    normalize_doi,
    normalize_title,
)


@pytest.mark.parametrize(
    "raw",
    [
        "10.1038/S41586-026-1",
        "https://doi.org/10.1038/s41586-026-1",
        "http://dx.doi.org/10.1038/s41586-026-1",
        "doi:10.1038/s41586-026-1",
        "  10.1038/s41586-026-1  ",
    ],
)
def test_doi_normalization_collapses_url_forms(raw: str) -> None:
    assert normalize_doi(raw) == "10.1038/s41586-026-1"


def test_paper_normalizes_doi_on_construction() -> None:
    p = Paper(doi="https://doi.org/10.1101/2026.08.20.999", title="t", source=Source.BIORXIV)
    assert p.doi == "10.1101/2026.08.20.999"


def test_normalize_title_strips_markup_and_punctuation() -> None:
    a = normalize_title("<i>In vivo</i> mapping of neurons: a new approach!")
    b = normalize_title("In vivo mapping of neurons - A New Approach")
    assert a == b


def test_relevance_score_range_is_enforced() -> None:
    with pytest.raises(ValidationError):
        RelevanceScore(score=101)
    with pytest.raises(ValidationError):
        RelevanceScore(score=-1)


def test_round_input_strips_blank_keywords() -> None:
    spec = RoundInput(
        keywords=["  single-cell ", "", "   "],
        date_from=date(2026, 8, 1),
        date_to=date(2026, 8, 25),
    )
    assert spec.keywords == ["single-cell"]


def test_round_input_requires_at_least_one_keyword() -> None:
    with pytest.raises(ValidationError):
        RoundInput(keywords=[], date_from=date(2026, 8, 1), date_to=date(2026, 8, 25))
