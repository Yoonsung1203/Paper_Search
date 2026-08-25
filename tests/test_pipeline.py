"""M2 게이트 — 검색부터 Human gate, 재랭킹까지 라운드가 완주하는가."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from paper_search.config import Settings
from paper_search.core.pipeline import Pipeline, PipelineDeps
from paper_search.core.search import SearchOutcome
from paper_search.models import (
    Confidence,
    CriteriaInference,
    InferredCriterion,
    NoveltyCategory,
    Paper,
    PaperSummary,
    RelevanceScore,
    RoundInput,
    RoundStatus,
    SummaryBasis,
)
from paper_search.store import Repository
from tests.factories import make_paper
from tests.stubs import StubLlm

SUMMARY = PaperSummary(
    summary=["요약 1", "요약 2", "요약 3"],
    basis=SummaryBasis.ABSTRACT,
    novelty_claim="새 방법론",
    novelty_category=NoveltyCategory.NEW_METHOD,
    novelty_confidence=Confidence.HIGH,
)


@dataclass
class StubSearch:
    outcome: SearchOutcome

    async def search(self, spec: RoundInput) -> SearchOutcome:
        return self.outcome


def _score_by_title(user: str) -> RelevanceScore:
    """제목 끝 숫자를 점수로 쓴다 — 랭킹 검증을 결정적으로 만들기 위함."""
    digits = "".join(ch for ch in user.split("\n")[0] if ch.isdigit())
    return RelevanceScore(score=int(digits or 0), evidence=["근거"], reason="테스트")


def _papers(n: int) -> list[Paper]:
    return [make_paper(doi=f"10.1/{i}", title=f"Paper {i * 10}") for i in range(1, n + 1)]


def _pipeline(
    repo: Repository, papers: list[Paper], llm: StubLlm, settings: Settings | None = None
) -> Pipeline:
    settings = settings or Settings(summarize_top_n=2, cost_cap_usd=3.0)
    deps = PipelineDeps(
        search=StubSearch(SearchOutcome(papers=papers)),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )
    return Pipeline(repo, deps, settings)


async def test_screening_run_completes(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    result = await _pipeline(repo, _papers(5), llm).run_screening(rid)

    assert result.status is RoundStatus.AWAITING_SELECTION
    assert len(result.papers) == 5
    # 점수 내림차순으로 랭크가 매겨진다
    assert [sp.paper.title for sp in result.papers][:2] == ["Paper 50", "Paper 40"]
    assert result.papers[0].relevance is not None
    assert result.papers[0].relevance.score == 50


async def test_only_top_n_are_summarized(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    result = await _pipeline(repo, _papers(5), llm).run_screening(rid)

    summarized = [sp for sp in result.papers if sp.summary is not None]
    assert len(summarized) == 2  # summarize_top_n=2
    assert {sp.paper.title for sp in summarized} == {"Paper 50", "Paper 40"}


async def test_truncation_is_announced_not_silent(
    repo: Repository, round_input: RoundInput
) -> None:
    """무엇이 잘렸는지 반드시 알린다 (build_plan §9)."""
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    result = await _pipeline(repo, _papers(5), llm).run_screening(rid)

    assert any("상위 2건에만 적용" in w for w in result.warnings)


async def test_too_few_results_advises_user(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    result = await _pipeline(repo, _papers(2), llm).run_screening(rid)

    assert any("키워드를 완화" in w for w in result.warnings)


async def test_no_results_ends_partial(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title})
    result = await _pipeline(repo, [], llm).run_screening(rid)

    assert result.status is RoundStatus.PARTIAL
    assert result.papers == []
    assert llm.calls == []


async def test_cost_cap_returns_partial_results(repo: Repository, round_input: RoundInput) -> None:
    """상한에 걸려도 조용히 자르지 않고, 부분 결과와 사유를 남긴다."""
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY}, cap_after=2)
    result = await _pipeline(repo, _papers(5), llm).run_screening(rid)

    assert result.status is RoundStatus.PARTIAL
    assert any("비용 상한" in w for w in result.warnings)
    scored = [sp for sp in result.papers if sp.relevance is not None]
    assert 0 < len(scored) < 5, "일부는 점수화되어 남아 있어야 한다"
    # 상한에 걸리면 요약 단계는 아예 시작하지 않는다
    assert all(sp.summary is None for sp in result.papers)


async def test_search_warning_is_propagated(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    deps = PipelineDeps(
        search=StubSearch(  # type: ignore[arg-type]
            SearchOutcome(papers=_papers(5), warnings=["biorxiv 수집 실패 — HTTP 503"])
        ),
        llm=llm,  # type: ignore[arg-type]
    )
    result = await Pipeline(repo, deps, Settings(summarize_top_n=2)).run_screening(rid)

    assert result.status is RoundStatus.PARTIAL
    assert any("biorxiv" in w for w in result.warnings)


async def test_selection_infers_criteria_and_reranks(
    repo: Repository, round_input: RoundInput
) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm(
        {
            "score": _score_by_title,
            "summarize": SUMMARY,
            "compare": CriteriaInference(
                criteria=[InferredCriterion(text="in vivo 검증 포함", supporting_dois=["10.1/1"])]
            ),
        }
    )
    pipeline = _pipeline(repo, _papers(5), llm)
    await pipeline.run_screening(rid)

    # 점수가 가장 낮은 두 편을 고른다 — 재랭킹이 실제로 일어나는지 보기 위해
    result = await pipeline.apply_selection(rid, {"10.1/1", "10.1/2"})

    assert result.status is RoundStatus.DONE
    assert [sp.paper.doi for sp in result.papers][:2] == ["10.1/2", "10.1/1"]
    assert [c.text for c in result.criteria] == ["in vivo 검증 포함"]


async def test_criteria_carry_into_next_round(repo: Repository, round_input: RoundInput) -> None:
    """PRD §13-4 — 이전 라운드의 활성 기준을 다음 라운드가 승계한다."""
    first = repo.create_round(round_input)
    llm = StubLlm(
        {
            "score": _score_by_title,
            "summarize": SUMMARY,
            "compare": CriteriaInference(
                criteria=[InferredCriterion(text="사람 데이터를 쓴 연구")]
            ),
        }
    )
    pipeline = _pipeline(repo, _papers(3), llm)
    await pipeline.run_screening(first)
    await pipeline.apply_selection(first, {"10.1/1", "10.1/2"})

    second = repo.create_round(round_input)
    llm2 = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    await _pipeline(repo, _papers(3), llm2).run_screening(second)

    assert "사람 데이터를 쓴 연구" in llm2.systems[0][1]["text"]


async def test_empty_selection_skips_inference(repo: Repository, round_input: RoundInput) -> None:
    """아무것도 고르지 않으면 점수순 리스트를 그대로 준다 (PRD §7)."""
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY})
    pipeline = _pipeline(repo, _papers(3), llm)
    await pipeline.run_screening(rid)
    before = len(llm.calls)

    result = await pipeline.apply_selection(rid, set())

    assert result.status is RoundStatus.DONE
    assert len(llm.calls) == before, "비교 호출이 일어나면 안 된다"
    assert any("선택된 논문이 없어" in w for w in result.warnings)


async def test_selection_records_both_groups(repo: Repository, round_input: RoundInput) -> None:
    rid = repo.create_round(round_input)
    llm = StubLlm({"score": _score_by_title, "summarize": SUMMARY, "compare": CriteriaInference()})
    pipeline = _pipeline(repo, _papers(3), llm)
    await pipeline.run_screening(rid)
    await pipeline.apply_selection(rid, {"10.1/1", "10.1/2"})

    assert repo.get_selection(rid) == {"10.1/1": True, "10.1/2": True, "10.1/3": False}


async def test_unknown_round_raises(repo: Repository) -> None:
    llm = StubLlm({"score": _score_by_title})
    with pytest.raises(ValueError):
        await _pipeline(repo, _papers(1), llm).run_screening(999)
