"""도메인 모델.

외부 API 응답과 LLM 출력이 모두 이 모델로 정규화된 뒤에야 파이프라인에 들어간다.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 논문


class Source(StrEnum):
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"


class Paper(BaseModel):
    """정규화된 논문 메타데이터. DOI가 동일성의 기준이다."""

    doi: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    issn: str | None = None
    published_at: date | None = None
    url: str = ""
    source: Source
    is_preprint: bool = False
    category: str | None = None
    pmid: str | None = None

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, v: str) -> str:
        return normalize_doi(v)

    @property
    def dedupe_key(self) -> str:
        """DOI가 없는 레코드까지 커버하기 위한 보조 키."""
        return self.doi or normalize_title(self.title)


def normalize_doi(raw: str) -> str:
    """DOI를 비교 가능한 형태로 정규화한다.

    `https://doi.org/10.1/x`, `doi:10.1/X`, `10.1/x` 를 모두 `10.1/x` 로 만든다.
    """
    v = raw.strip().lower()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
    v = re.sub(r"^doi:\s*", "", v)
    return v.strip()


def normalize_title(raw: str) -> str:
    """제목 기반 중복 판정을 위해 구두점·공백·대소문자를 제거한다."""
    v = raw.lower()
    v = re.sub(r"<[^>]+>", " ", v)  # JATS/HTML 태그
    v = re.sub(r"[^a-z0-9가-힣]+", " ", v)
    return " ".join(v.split())


# ---------------------------------------------------------------- 저널 지표


class JournalMetric(BaseModel):
    """IF 대용 지표. 지표명과 출처를 반드시 함께 들고 다닌다 (PRD §6.2-2)."""

    issn: str
    metric_name: str
    metric_value: float
    source: str
    year: int | None = None
    fetched_at: datetime | None = None

    def display(self) -> str:
        year = f", {self.year}" if self.year else ""
        return f"{self.metric_name} {self.metric_value:.2f} ({self.source}{year})"


# ---------------------------------------------------------------- LLM 산출물


class SummaryBasis(StrEnum):
    """요약의 근거 범위. 비-OA 논문은 초록만 볼 수 있다 (PRD §6.2-3)."""

    ABSTRACT = "abstract"
    FULLTEXT = "fulltext"


class NoveltyCategory(StrEnum):
    NEW_METHOD = "new_method"
    LARGE_DATA = "large_data"
    NEW_THEORY = "new_theory"
    CLINICAL_IMPACT = "clinical_impact"
    OTHER = "other"
    UNDETERMINED = "undetermined"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RelevanceScore(BaseModel):
    """F-03 관련도 점수화 결과."""

    score: int = Field(ge=0, le=100)
    evidence: list[str] = Field(default_factory=list, description="논문 원문에서 인용한 근거")
    reason: str = ""


class PaperSummary(BaseModel):
    """F-02 요약 + F-04 차별성 검증. 한 번의 호출로 함께 받는다."""

    summary: list[str] = Field(min_length=1, max_length=6)
    basis: SummaryBasis
    novelty_claim: str = ""
    novelty_category: NoveltyCategory = NoveltyCategory.UNDETERMINED
    novelty_confidence: Confidence = Confidence.LOW


class InferredCriterion(BaseModel):
    """F-06이 추론한 선택 기준."""

    text: str
    supporting_dois: list[str] = Field(default_factory=list)


class CriteriaInference(BaseModel):
    criteria: list[InferredCriterion] = Field(default_factory=list)


# ---------------------------------------------------------------- 라운드


class RoundStatus(StrEnum):
    CREATED = "created"
    SEARCHING = "searching"
    SCORING = "scoring"
    SUMMARIZING = "summarizing"
    AWAITING_SELECTION = "awaiting_selection"  # Human gate
    RERANKING = "reranking"
    DONE = "done"
    FAILED = "failed"
    PARTIAL = "partial"  # 비용 상한/소스 실패로 일부만 완료


class RoundInput(BaseModel):
    """라운드 실행 조건. 그대로 저장되어 재현성을 보장한다 (PRD §10)."""

    keywords: list[str] = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    date_from: date
    date_to: date
    impact_threshold: float | None = None
    include_preprints: bool = True

    @field_validator("keywords", "authors")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s.strip()]


class ScoredPaper(BaseModel):
    """라운드 안에서의 논문 1건 — 원본 메타 + LLM 산출물 + 지표."""

    paper: Paper
    relevance: RelevanceScore | None = None
    summary: PaperSummary | None = None
    metric: JournalMetric | None = None
    selected: bool | None = None
    rank: int | None = None

    @property
    def score(self) -> int:
        return self.relevance.score if self.relevance else 0


class RoundResult(BaseModel):
    round_id: int
    status: RoundStatus
    papers: list[ScoredPaper] = Field(default_factory=list)
    criteria: list[InferredCriterion] = Field(default_factory=list)
    cost_usd: float = 0.0
    warnings: list[str] = Field(default_factory=list, description="부분 실패 안내")
