"""라운드 오케스트레이션.

상태 전이:
    created → searching → scoring → summarizing → awaiting_selection
                                                       │ (Human gate)
                                                       ▼
                                                   reranking → done

비용 상한이나 소스 실패가 나면 라운드를 세우지 않고 `partial`로 마무리한 뒤,
어디까지 됐는지를 경고로 남긴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from anthropic import AsyncAnthropic

from paper_search.config import Settings
from paper_search.core.impact import ImpactFilter
from paper_search.core.search import SearchService
from paper_search.llm.client import LlmClient
from paper_search.llm.stages import infer_criteria, score_papers, summarize_papers
from paper_search.models import RoundResult, RoundStatus
from paper_search.sources.openalex import OpenAlexMetrics
from paper_search.store import Repository

logger = logging.getLogger(__name__)

TOO_FEW = 5
TOO_MANY = 200


@dataclass
class PipelineDeps:
    """외부 의존성을 한 곳에 모아 테스트에서 통째로 갈아끼운다."""

    search: SearchService
    llm: LlmClient
    impact: ImpactFilter | None = None


def build_deps(
    http_client: httpx.AsyncClient,
    anthropic_client: AsyncAnthropic,
    settings: Settings,
    repo: Repository,
    round_id: int,
) -> PipelineDeps:
    def on_usage(
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cost: float,
    ) -> None:
        repo.record_llm_call(round_id, stage, model, input_tokens, output_tokens, cost, cache_read)

    search = SearchService(http_client, settings)
    metrics = OpenAlexMetrics(
        search.fetch_for("openalex", rate=5.0, ttl=30 * 24 * 60 * 60),
        mailto=settings.contact_email,
    )
    return PipelineDeps(
        search=search,
        llm=LlmClient(
            anthropic_client,
            concurrency=settings.llm_concurrency,
            cost_cap_usd=settings.cost_cap_usd,
            on_usage=on_usage,
        ),
        impact=ImpactFilter(metrics, repo),
    )


class Pipeline:
    def __init__(self, repo: Repository, deps: PipelineDeps, settings: Settings) -> None:
        self.repo = repo
        self.deps = deps
        self.settings = settings

    async def run_screening(self, round_id: int) -> RoundResult:
        """검색 → 점수화 → 요약까지. 끝나면 Human gate에서 멈춘다."""
        spec = self.repo.get_round_input(round_id)
        if spec is None:
            raise ValueError(f"라운드 {round_id}를 찾을 수 없습니다")

        degraded = False

        # --- 검색 -----------------------------------------------------
        self.repo.set_status(round_id, RoundStatus.SEARCHING)
        outcome = await self.deps.search.search(spec)
        for warning in outcome.warnings:
            self.repo.add_warning(round_id, warning)
        degraded |= bool(outcome.warnings)

        papers = outcome.papers

        # --- 임팩트 임계값 (M3) ------------------------------------------
        if self.deps.impact is not None and papers:
            impact = await self.deps.impact.apply(papers, spec.impact_threshold)
            papers = impact.papers
            note = ImpactFilter.describe(impact, spec.impact_threshold)
            if note:
                self.repo.add_warning(round_id, note)

        self.repo.attach_papers(round_id, papers)
        self._advise_on_volume(round_id, len(papers))

        if not papers:
            self.repo.set_status(round_id, RoundStatus.PARTIAL)
            return self.repo.load_result(round_id)

        # --- S1 점수화 -------------------------------------------------
        self.repo.set_status(round_id, RoundStatus.SCORING)
        criteria = self.repo.latest_criteria_before(round_id)
        scored = await score_papers(self.deps.llm, papers, spec.keywords, criteria)

        for doi, score in scored.results.items():
            self.repo.save_relevance(round_id, doi, score)
        if scored.failed:
            self.repo.add_warning(round_id, f"{len(scored.failed)}건의 점수화에 실패했습니다.")
            degraded = True
        if scored.capped:
            self._warn_capped(round_id, "점수화")
            degraded = True

        ranked = sorted(
            papers,
            key=lambda p: scored.results[p.doi].score if p.doi in scored.results else -1,
            reverse=True,
        )
        self.repo.save_ranks(round_id, [p.doi for p in ranked])

        # --- S2+S3 요약·차별성 (상위 N건만) ------------------------------
        if not scored.capped:
            self.repo.set_status(round_id, RoundStatus.SUMMARIZING)
            top = ranked[: self.settings.summarize_top_n]
            summarized = await summarize_papers(self.deps.llm, top)
            for doi, summary in summarized.results.items():
                self.repo.save_summary(round_id, doi, summary)
            if summarized.failed:
                self.repo.add_warning(
                    round_id, f"{len(summarized.failed)}건의 요약에 실패했습니다."
                )
                degraded = True
            if summarized.capped:
                self._warn_capped(round_id, "요약")
                degraded = True
            if len(ranked) > self.settings.summarize_top_n:
                # 무엇이 잘렸는지 밝힌다 — 조용한 절단 금지 (build_plan §9)
                self.repo.add_warning(
                    round_id,
                    f"요약은 관련도 상위 {self.settings.summarize_top_n}건에만 적용했습니다 "
                    f"(전체 {len(ranked)}건).",
                )

        self.repo.set_status(
            round_id, RoundStatus.PARTIAL if degraded else RoundStatus.AWAITING_SELECTION
        )
        return self.repo.load_result(round_id)

    async def apply_selection(self, round_id: int, selected_dois: set[str]) -> RoundResult:
        """Human gate 이후 — 선택 기준을 추론하고 재랭킹한다."""
        self.repo.save_selection(round_id, selected_dois)
        self.repo.set_status(round_id, RoundStatus.RERANKING)

        loaded = self.repo.load_round_papers(round_id)
        selected = [sp.paper for sp in loaded if sp.paper.doi in selected_dois]
        rejected = [sp.paper for sp in loaded if sp.paper.doi not in selected_dois]

        if not selected:
            # 아무것도 고르지 않으면 F-06을 건너뛰고 점수순 리스트를 그대로 준다 (PRD §7)
            self.repo.add_warning(
                round_id, "선택된 논문이 없어 관련도 점수순 리스트를 그대로 제시합니다."
            )
            self.repo.set_status(round_id, RoundStatus.DONE)
            return self.repo.load_result(round_id)

        criteria = await infer_criteria(self.deps.llm, selected, rejected)
        if criteria:
            self.repo.save_criteria(round_id, criteria)

        # 선택된 논문을 앞으로 끌어올린다
        ranked = sorted(
            loaded,
            key=lambda sp: (sp.paper.doi in selected_dois, sp.score),
            reverse=True,
        )
        self.repo.save_ranks(round_id, [sp.paper.doi for sp in ranked])
        self.repo.set_status(round_id, RoundStatus.DONE)
        return self.repo.load_result(round_id)

    # ------------------------------------------------------------ 보조

    def _advise_on_volume(self, round_id: int, count: int) -> None:
        """PRD §7의 예외 흐름."""
        if count < TOO_FEW:
            self.repo.add_warning(
                round_id,
                f"후보가 {count}건뿐입니다. 키워드를 완화하거나 기간을 넓혀 보십시오.",
            )
        elif count >= TOO_MANY:
            self.repo.add_warning(
                round_id,
                f"후보가 {count}건입니다. 임팩트 임계값을 올리거나 키워드를 추가해 보십시오.",
            )

    def _warn_capped(self, round_id: int, stage: str) -> None:
        spent = self.repo.round_cost(round_id)
        self.repo.add_warning(
            round_id,
            f"비용 상한(${self.settings.cost_cap_usd:.2f})에 도달해 {stage} 단계를 "
            f"중단했습니다. 현재까지 ${spent:.3f} 사용, 결과는 부분적입니다.",
        )


__all__ = ["Pipeline", "PipelineDeps", "build_deps"]
