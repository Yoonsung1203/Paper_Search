"""M2 게이트 — 웹에서 키워드 입력 → 1차 리스트 선택 → 최종 리스트까지 완주하는가."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paper_search.config import Settings
from paper_search.core.pipeline import PipelineDeps
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
    SummaryBasis,
)
from paper_search.store import Repository, connect
from paper_search.web import create_app
from tests.factories import make_paper
from tests.stubs import StubLlm

SUMMARY = PaperSummary(
    summary=["핵심 요약 한 줄", "두 번째 줄", "세 번째 줄"],
    basis=SummaryBasis.ABSTRACT,
    novelty_claim="대규모 코호트를 처음으로 다룬다",
    novelty_category=NoveltyCategory.LARGE_DATA,
    novelty_confidence=Confidence.HIGH,
)


@dataclass
class StubSearch:
    outcome: SearchOutcome

    async def search(self, spec: RoundInput) -> SearchOutcome:
        return self.outcome


def _score(user: str) -> RelevanceScore:
    digits = "".join(ch for ch in user.split("\n")[0] if ch.isdigit())
    return RelevanceScore(score=int(digits or 0), evidence=["spatial atlas"], reason="부합")


def _papers(n: int = 3) -> list[Paper]:
    return [
        make_paper(doi=f"10.1/{i}", title=f"Paper {i * 10}", journal="Nature")
        for i in range(1, n + 1)
    ]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "web.sqlite3",
        cache_dir=tmp_path / "cache",
        summarize_top_n=2,
        cost_cap_usd=3.0,
    )


@pytest.fixture
def llm() -> StubLlm:
    return StubLlm(
        {
            "score": _score,
            "summarize": SUMMARY,
            "compare": CriteriaInference(
                criteria=[
                    InferredCriterion(text="사람 데이터를 쓴 연구", supporting_dois=["10.1/1"])
                ]
            ),
        }
    )


@pytest.fixture
def client(settings: Settings, llm: StubLlm) -> Iterator[TestClient]:
    search = StubSearch(SearchOutcome(papers=_papers()))

    def deps_factory(repo: Repository, round_id: int) -> PipelineDeps:
        return PipelineDeps(search=search, llm=llm)  # type: ignore[arg-type]

    with TestClient(create_app(settings, deps_factory=deps_factory)) as c:
        yield c


def _start_round(client: TestClient) -> int:
    response = client.post(
        "/rounds",
        data={
            "keywords": "single-cell, spatial transcriptomics",
            "date_from": "2026-08-18",
            "date_to": "2026-08-25",
            "include_preprints": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    return int(response.headers["location"].rsplit("/", 1)[1])


def test_index_renders_form(client: TestClient) -> None:
    body = client.get("/").text
    assert "관심 주제" in body
    assert "스크리닝 시작" in body


def test_round_requires_keyword_or_author(client: TestClient) -> None:
    response = client.post(
        "/rounds", data={"keywords": "", "date_from": "2026-08-18", "date_to": "2026-08-25"}
    )
    assert response.status_code == 400


def test_full_flow_input_to_final_list(
    client: TestClient, settings: Settings, llm: StubLlm
) -> None:
    """M2 게이트 그 자체."""
    round_id = _start_round(client)

    # 백그라운드 파이프라인이 끝날 때까지 진행 화면이 폴링된다
    for _ in range(50):
        progress = client.get(f"/rounds/{round_id}/progress")
        if "HX-Redirect" in progress.headers:
            break
    else:
        pytest.fail("파이프라인이 끝나지 않았습니다")

    # 1차 리스트 — 선택 가능한 체크박스가 있어야 한다
    screen = client.get(f"/rounds/{round_id}").text
    assert "1차 리스트" in screen
    assert 'name="selected"' in screen
    assert "Paper 30" in screen
    assert "관련도 30" in screen
    assert "핵심 요약 한 줄" in screen  # 상위 2건은 요약이 붙는다

    # Human gate
    # 기준 추론에는 선택군이 2편 이상 필요하다 (llm/stages.py MIN_SELECTED_FOR_INFERENCE)
    response = client.post(
        f"/rounds/{round_id}/selection",
        data={"selected": ["10.1/1", "10.1/2"]},
        follow_redirects=False,
    )
    assert response.status_code == 303

    final = client.get(f"/rounds/{round_id}").text
    assert "최종 리스트" in final
    assert "선택 2건" in final
    assert "파악된 선택 기준" in final
    assert "사람 데이터를 쓴 연구" in final

    conn = connect(settings.db_path)
    repo = Repository(conn)
    assert repo.get_selection(round_id) == {"10.1/1": True, "10.1/2": True, "10.1/3": False}
    conn.close()


def test_screen_shows_warnings(client: TestClient, settings: Settings) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break

    body = client.get(f"/rounds/{round_id}").text
    # 후보 3건 → "키워드를 완화" 안내 + 요약 절단 안내가 화면에 보여야 한다
    assert "키워드를 완화" in body
    assert "상위 2건에만 적용" in body


def test_preprint_badge_is_shown(settings: Settings, llm: StubLlm) -> None:
    preprint = make_paper(
        doi="10.1101/2026.08.20.1",
        title="Paper 90",
        is_preprint=True,
        issn=None,
        journal="biorxiv (preprint)",
    )
    search = StubSearch(SearchOutcome(papers=[preprint]))

    def deps_factory(repo: Repository, round_id: int) -> PipelineDeps:
        return PipelineDeps(search=search, llm=llm)  # type: ignore[arg-type]

    with TestClient(create_app(settings, deps_factory=deps_factory)) as client:
        round_id = _start_round(client)
        for _ in range(50):
            if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
                break
        body = client.get(f"/rounds/{round_id}").text

    assert "preprint" in body
    assert "지표 미확인" not in body, "프리프린트에는 지표 미확인 배지를 붙이지 않는다"


def test_empty_selection_still_produces_final_list(client: TestClient) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break

    client.post(f"/rounds/{round_id}/selection", data={}, follow_redirects=False)
    body = client.get(f"/rounds/{round_id}").text

    assert "최종 리스트" in body
    assert "선택된 논문이 없어" in body


def test_unknown_round_returns_404(client: TestClient) -> None:
    assert client.get("/rounds/9999").status_code == 404
    assert client.get("/rounds/9999/progress").status_code == 404


def test_criterion_can_be_disabled(client: TestClient, settings: Settings) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break
    client.post(f"/rounds/{round_id}/selection", data={"selected": ["10.1/1", "10.1/2"]})

    conn = connect(settings.db_path)
    repo = Repository(conn)
    criterion_id = repo.list_criteria(round_id)[0]["id"]
    conn.close()

    client.post(f"/rounds/{round_id}/criteria/{criterion_id}", data={"active": ""})

    conn = connect(settings.db_path)
    assert Repository(conn).active_criteria(round_id) == []
    conn.close()


def test_kpi_page_is_empty_before_any_round(client: TestClient) -> None:
    body = client.get("/kpi").text
    assert "아직 완료된 라운드가 없습니다" in body


def test_kpi_page_shows_selection_rate(client: TestClient) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break
    client.post(f"/rounds/{round_id}/selection", data={"selected": ["10.1/1", "10.1/2"]})

    body = client.get("/kpi").text
    assert "평균 1차 선택률" in body
    assert "67%" in body  # 3편 중 2편 선택
    assert "목표 70%" in body


def test_criterion_text_can_be_edited(client: TestClient, settings: Settings) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break
    client.post(f"/rounds/{round_id}/selection", data={"selected": ["10.1/1", "10.1/2"]})

    conn = connect(settings.db_path)
    criterion_id = Repository(conn).list_criteria(round_id)[0]["id"]
    conn.close()

    client.post(
        f"/rounds/{round_id}/criteria/{criterion_id}",
        data={"action": "save", "text": "직접 고쳐 쓴 기준"},
    )

    conn = connect(settings.db_path)
    repo = Repository(conn)
    assert "직접 고쳐 쓴 기준" in repo.active_criteria(round_id)
    assert repo.list_criteria(round_id)[0]["origin"] == "user_edited"
    conn.close()


def test_blank_criterion_text_is_ignored(client: TestClient, settings: Settings) -> None:
    round_id = _start_round(client)
    for _ in range(50):
        if "HX-Redirect" in client.get(f"/rounds/{round_id}/progress").headers:
            break
    client.post(f"/rounds/{round_id}/selection", data={"selected": ["10.1/1", "10.1/2"]})

    conn = connect(settings.db_path)
    repo = Repository(conn)
    criterion_id = repo.list_criteria(round_id)[0]["id"]
    before = repo.list_criteria(round_id)[0]["text"]
    conn.close()

    client.post(
        f"/rounds/{round_id}/criteria/{criterion_id}", data={"action": "save", "text": "   "}
    )

    conn = connect(settings.db_path)
    assert Repository(conn).list_criteria(round_id)[0]["text"] == before
    conn.close()
