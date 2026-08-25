"""프롬프트.

시스템 프롬프트는 **호출 간 바이트 단위로 고정**되어야 캐시가 적중한다.
논문 본문처럼 매 호출 달라지는 내용은 반드시 user 메시지에 둔다.
"""

from __future__ import annotations

from typing import Any

from paper_search.models import Paper

SCORE_SYSTEM = """\
당신은 연구자의 논문 스크리닝을 돕는다. 연구자가 제시한 관심 주제와 논문 한 편이 주어지면,
그 논문이 연구자의 관심사에 얼마나 부합하는지를 0-100으로 점수화한다.

점수 기준:
- 90-100: 연구자의 관심 주제를 정면으로 다룬다. 반드시 읽어야 한다.
- 70-89: 관심 주제와 직접 맞닿아 있다. 방법론이나 대상이 일부 다를 수 있다.
- 40-69: 인접 분야다. 배경 지식으로는 유용하나 핵심은 아니다.
- 10-39: 같은 상위 분야일 뿐 관심사와는 멀다.
- 0-9: 무관하다.

판단 규칙:
- 제목의 키워드 일치만으로 높은 점수를 주지 않는다. 초록이 실제로 그 주제를 다루는지 본다.
- 관심사가 방법론(예: 공간전사체)인지 대상(예: 해마)인지 구분한다. 둘 다 맞으면 높다.
- 추가 선택 기준이 주어지면 그 기준을 점수에 반영한다.
- 초록이 비어 있으면 제목만으로 판단하되 점수를 60을 넘기지 않는다.

evidence에는 **논문 원문(제목 또는 초록)에서 그대로 가져온 문장이나 구**를 1-2개 넣는다.
원문에 없는 문장을 지어내지 않는다. 원문이 영어면 영어 그대로 인용한다.
reason은 한국어 한 문장으로 쓴다."""

SUMMARY_SYSTEM = """\
당신은 연구자의 논문 스크리닝을 돕는다. 논문 한 편이 주어지면 두 가지를 산출한다.

1) summary — 논문의 핵심을 한국어 3-5줄로 요약한다.
   - 무엇을 물었고, 어떻게 했고, 무엇을 찾았는지가 드러나야 한다.
   - 주어진 본문에 없는 수치나 주장을 절대 넣지 않는다.
   - 한 줄은 한 문장으로 쓴다.

2) 차별성 — 이 논문이 왜 이 수준의 저널에 실릴 만한지를 판단한다.
   - novelty_claim: 한국어 1-2문장.
   - novelty_category: new_method(새 방법론) / large_data(대규모 데이터) /
     new_theory(새 이론) / clinical_impact(임상적 파급력) / other / undetermined
   - novelty_confidence: 주어진 본문에서 근거를 확인했으면 high,
     정황상 추론했으면 medium, 근거가 불충분하면 low.

근거를 본문에서 확인할 수 없으면 **추측해서 단정하지 않는다.**
그 경우 novelty_category를 undetermined, novelty_confidence를 low로 두고
novelty_claim에 "본문에서 근거를 확인하지 못했습니다"라고 쓴다.

basis에는 요약의 근거 범위를 넣는다. 초록만 주어졌으면 abstract, 전문이 주어졌으면 fulltext."""

COMPARE_SYSTEM = """\
당신은 연구자의 논문 선택 패턴을 분석한다. 연구자가 1차 리스트에서 고른 논문(선택군)과
고르지 않은 논문(비선택군)이 주어진다. 두 그룹을 가르는 기준을 추론한다.

규칙:
- 기준은 3-5개, 각각 한국어 한 문장으로 쓴다.
- "관심사에 맞다" 같은 동어반복은 기준이 아니다. 관찰 가능한 속성으로 쓴다.
  좋은 예: "in vivo 검증이 포함된 연구를 고른다", "사람 데이터를 쓴 연구를 고른다",
          "방법론 개발보다 생물학적 발견을 다룬 연구를 고른다"
- 각 기준마다 그 기준을 뒷받침하는 선택군 논문의 DOI를 supporting_dois에 넣는다.
- 선택군이 2편 미만이면 근거가 부족하므로 criteria를 빈 배열로 둔다."""


def score_system(keywords: list[str], criteria: list[str]) -> list[dict[str, Any]]:
    """고정 규칙 블록 + 라운드 고정 블록. 둘 다 캐시 대상이다."""
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": SCORE_SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]
    context = f"연구자의 관심 주제: {', '.join(keywords)}"
    if criteria:
        joined = "\n".join(f"- {c}" for c in criteria)
        context += f"\n\n이전 라운드에서 파악된 이 연구자의 선택 기준:\n{joined}"
    blocks.append({"type": "text", "text": context, "cache_control": {"type": "ephemeral"}})
    return blocks


def summary_system() -> list[dict[str, Any]]:
    return [{"type": "text", "text": SUMMARY_SYSTEM, "cache_control": {"type": "ephemeral"}}]


def compare_system() -> list[dict[str, Any]]:
    return [{"type": "text", "text": COMPARE_SYSTEM, "cache_control": {"type": "ephemeral"}}]


def paper_block(paper: Paper, *, include_journal: bool = False) -> str:
    parts = [f"제목: {paper.title}"]
    if include_journal:
        venue = f"{paper.journal} (프리프린트)" if paper.is_preprint else paper.journal
        parts.append(f"저널: {venue or '미상'}")
    parts.append(f"초록: {paper.abstract or '(초록 없음)'}")
    return "\n".join(parts)


def compare_user(selected: list[Paper], rejected: list[Paper]) -> str:
    def block(label: str, papers: list[Paper]) -> str:
        if not papers:
            return f"[{label}] 없음"
        lines = [f"- ({p.doi}) {p.title}\n  {p.abstract[:400]}" for p in papers]
        return f"[{label}] {len(papers)}편\n" + "\n".join(lines)

    return f"{block('선택군', selected)}\n\n{block('비선택군', rejected)}"
