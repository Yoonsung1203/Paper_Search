"""LLM 단계 실행 (S1 점수화, S2+S3 요약·차별성, S4 그룹 비교).

각 단계는 개별 논문의 실패를 흡수한다 — 한 편이 실패했다고 라운드를 세우지 않는다.
비용 상한(CostCapExceeded)만은 흡수하지 않고 위로 던져서, 파이프라인이 부분 결과를
반환하도록 한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Generic, TypeVar

from paper_search.llm.client import CostCapExceeded, LlmClient
from paper_search.llm.prompts import (
    compare_system,
    compare_user,
    paper_block,
    score_system,
    summary_system,
)
from paper_search.models import (
    CriteriaInference,
    InferredCriterion,
    Paper,
    PaperSummary,
    RelevanceScore,
    SummaryBasis,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StageOutcome(Generic[T]):
    """단계 실행 결과. 상한에 걸려 중단됐는지를 호출부가 알아야 한다."""

    def __init__(self) -> None:
        self.results: dict[str, T] = {}
        self.failed: list[str] = []
        self.capped: bool = False

    def __len__(self) -> int:
        return len(self.results)


async def score_papers(
    llm: LlmClient,
    papers: list[Paper],
    keywords: list[str],
    criteria: list[str] | None = None,
) -> StageOutcome[RelevanceScore]:
    """S1 — 후보 전건을 점수화한다."""
    outcome: StageOutcome[RelevanceScore] = StageOutcome()
    if not papers:
        return outcome

    system = score_system(keywords, criteria or [])
    stop = asyncio.Event()

    async def one(paper: Paper) -> tuple[str, RelevanceScore | None]:
        if stop.is_set():
            return paper.doi, None
        try:
            result = await llm.parse(
                stage="score",
                system=system,
                user=paper_block(paper),
                schema=RelevanceScore,
                effort="low",  # 대량·단순 판단이므로 깊게 생각할 필요가 없다
                max_tokens=1024,
            )
        except CostCapExceeded:
            stop.set()
            return paper.doi, None
        except Exception:
            logger.warning("점수화 실패: %s", paper.doi, exc_info=True)
            return paper.doi, None
        return paper.doi, result

    for doi, result in await asyncio.gather(*(one(p) for p in papers)):
        if result is None:
            outcome.failed.append(doi)
        else:
            outcome.results[doi] = result

    outcome.capped = stop.is_set()
    return outcome


async def summarize_papers(llm: LlmClient, papers: list[Paper]) -> StageOutcome[PaperSummary]:
    """S2+S3 — 요약과 차별성 검증을 한 번의 호출로 받는다.

    같은 본문을 두 번 보내면 입력 토큰이 2배가 되므로 병합한다 (build_plan §3.1).
    """
    outcome: StageOutcome[PaperSummary] = StageOutcome()
    if not papers:
        return outcome

    system = summary_system()
    stop = asyncio.Event()

    async def one(paper: Paper) -> tuple[str, PaperSummary | None]:
        if stop.is_set():
            return paper.doi, None
        try:
            result = await llm.parse(
                stage="summarize",
                system=system,
                user=paper_block(paper, include_journal=True),
                schema=PaperSummary,
                effort="high",
                max_tokens=2048,
            )
        except CostCapExceeded:
            stop.set()
            return paper.doi, None
        except Exception:
            logger.warning("요약 실패: %s", paper.doi, exc_info=True)
            return paper.doi, None

        # 전문을 넣지 않았으므로 근거 범위는 초록으로 고정한다.
        # 모델이 fulltext라고 말해도 신뢰하지 않는다 (PRD F-02 수용 기준).
        result.basis = SummaryBasis.ABSTRACT
        return paper.doi, result

    for doi, result in await asyncio.gather(*(one(p) for p in papers)):
        if result is None:
            outcome.failed.append(doi)
        else:
            outcome.results[doi] = result

    outcome.capped = stop.is_set()
    return outcome


MIN_SELECTED_FOR_INFERENCE = 2


async def infer_criteria(
    llm: LlmClient, selected: list[Paper], rejected: list[Paper]
) -> list[InferredCriterion]:
    """S4 — 선택군과 비선택군을 가르는 기준을 추론한다."""
    if len(selected) < MIN_SELECTED_FOR_INFERENCE:
        logger.info("선택군이 %d편이라 기준 추론을 건너뜁니다", len(selected))
        return []
    try:
        result = await llm.parse(
            stage="compare",
            system=compare_system(),
            user=compare_user(selected, rejected),
            schema=CriteriaInference,
            effort="high",
            max_tokens=4096,
        )
    except CostCapExceeded:
        logger.warning("비용 상한으로 기준 추론을 건너뜁니다")
        return []
    except Exception:
        logger.warning("기준 추론 실패", exc_info=True)
        return []
    return result.criteria
