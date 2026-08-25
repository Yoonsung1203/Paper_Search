from __future__ import annotations

from paper_search.llm.stages import infer_criteria, score_papers, summarize_papers
from paper_search.models import (
    Confidence,
    CriteriaInference,
    InferredCriterion,
    NoveltyCategory,
    PaperSummary,
    RelevanceScore,
    SummaryBasis,
)
from tests.factories import make_paper
from tests.stubs import StubLlm

SCORE = RelevanceScore(score=85, evidence=["spatial atlas"], reason="부합")
SUMMARY = PaperSummary(
    summary=["a", "b", "c"],
    basis=SummaryBasis.FULLTEXT,  # 모델이 전문이라 주장해도 무시되어야 한다
    novelty_claim="새 방법론",
    novelty_category=NoveltyCategory.NEW_METHOD,
    novelty_confidence=Confidence.HIGH,
)


async def test_score_papers_returns_all() -> None:
    papers = [make_paper(doi=f"10.1/{i}", title=f"P{i}") for i in range(3)]
    llm = StubLlm({"score": SCORE})
    outcome = await score_papers(llm, papers, ["spatial"])  # type: ignore[arg-type]

    assert len(outcome) == 3
    assert outcome.failed == []
    assert outcome.capped is False


async def test_score_papers_uses_low_effort_and_caches_system() -> None:
    """S1은 대량 호출이라 effort를 낮추고 시스템 프롬프트를 캐싱한다."""
    llm = StubLlm({"score": SCORE})
    await score_papers(llm, [make_paper()], ["spatial"], ["in vivo 검증"])  # type: ignore[arg-type]

    system = llm.systems[0]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in system)
    # 이전 라운드 기준이 시스템 프롬프트에 주입된다
    assert "in vivo 검증" in system[1]["text"]


async def test_individual_failure_does_not_stop_the_stage() -> None:
    papers = [make_paper(doi=f"10.1/{i}", title=f"P{i}") for i in range(3)]
    llm = StubLlm({"score": SCORE}, fail_after=1)
    outcome = await score_papers(llm, papers, ["spatial"])  # type: ignore[arg-type]

    assert len(outcome) == 1
    assert len(outcome.failed) == 2
    assert outcome.capped is False


async def test_cost_cap_marks_outcome_capped() -> None:
    papers = [make_paper(doi=f"10.1/{i}", title=f"P{i}") for i in range(5)]
    llm = StubLlm({"score": SCORE}, cap_after=2)
    outcome = await score_papers(llm, papers, ["spatial"])  # type: ignore[arg-type]

    assert outcome.capped is True
    assert len(outcome) <= 2, "상한 이후 결과는 없어야 한다"


async def test_summary_basis_is_forced_to_abstract() -> None:
    """전문을 넣지 않았으므로 모델의 basis 주장은 신뢰하지 않는다 (PRD F-02)."""
    llm = StubLlm({"summarize": SUMMARY})
    outcome = await summarize_papers(llm, [make_paper()])  # type: ignore[arg-type]

    result = next(iter(outcome.results.values()))
    assert result.basis is SummaryBasis.ABSTRACT


async def test_summary_prompt_includes_journal() -> None:
    """차별성 검증(F-04)에는 저널 정보가 필요하다."""
    llm = StubLlm({"summarize": SUMMARY})
    await summarize_papers(llm, [make_paper(journal="Nature")])  # type: ignore[arg-type]
    assert "Nature" in llm.calls[0][1]


async def test_infer_criteria_skipped_when_too_few_selected() -> None:
    llm = StubLlm({"compare": CriteriaInference(criteria=[])})
    assert await infer_criteria(llm, [make_paper()], []) == []  # type: ignore[arg-type]
    assert llm.calls == [], "호출 자체가 일어나지 않아야 한다"


async def test_infer_criteria_returns_criteria() -> None:
    inferred = CriteriaInference(
        criteria=[InferredCriterion(text="in vivo 검증 포함", supporting_dois=["10.1/a"])]
    )
    llm = StubLlm({"compare": inferred})
    selected = [make_paper(doi="10.1/a"), make_paper(doi="10.1/b", title="B")]
    result = await infer_criteria(llm, selected, [make_paper(doi="10.1/c", title="C")])  # type: ignore[arg-type]

    assert [c.text for c in result] == ["in vivo 검증 포함"]
    assert "선택군" in llm.calls[0][1]
    assert "비선택군" in llm.calls[0][1]


async def test_infer_criteria_survives_failure() -> None:
    llm = StubLlm({"compare": CriteriaInference()}, fail_after=0)
    selected = [make_paper(doi="10.1/a"), make_paper(doi="10.1/b", title="B")]
    assert await infer_criteria(llm, selected, []) == []  # type: ignore[arg-type]


async def test_empty_input_makes_no_calls() -> None:
    llm = StubLlm({"score": SCORE})
    assert len(await score_papers(llm, [], ["x"])) == 0  # type: ignore[arg-type]
    assert llm.calls == []
