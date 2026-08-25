"""웹 UI — Human gate가 실제로 일어나는 곳.

파이프라인은 수 분이 걸리므로 백그라운드 태스크로 돌리고, 화면은 HTMX로 상태를 폴링한다.
SQLite 연결은 요청/태스크마다 새로 연다(스레드 간 공유 금지).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from paper_search.config import Settings, get_settings
from paper_search.core.pipeline import Pipeline, PipelineDeps, build_deps
from paper_search.models import RoundInput, RoundStatus
from paper_search.store import Repository, connect

logger = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"

DepsFactory = Callable[[Repository, int], PipelineDeps]

RUNNING_STATUSES = {
    RoundStatus.CREATED,
    RoundStatus.SEARCHING,
    RoundStatus.SCORING,
    RoundStatus.SUMMARIZING,
    RoundStatus.RERANKING,
}

STATUS_LABEL = {
    RoundStatus.CREATED: "준비 중",
    RoundStatus.SEARCHING: "논문 검색 중",
    RoundStatus.SCORING: "관련도 점수화 중",
    RoundStatus.SUMMARIZING: "요약·차별성 검증 중",
    RoundStatus.AWAITING_SELECTION: "선택 대기",
    RoundStatus.RERANKING: "선택 기준 분석 중",
    RoundStatus.DONE: "완료",
    RoundStatus.PARTIAL: "부분 완료",
    RoundStatus.FAILED: "실패",
}


def create_app(
    settings: Settings | None = None, *, deps_factory: DepsFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, Any] = {"tasks": {}}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if deps_factory is None:
            state["http"] = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": settings.user_agent},
                follow_redirects=True,
            )
            state["anthropic"] = AsyncAnthropic(api_key=settings.require_anthropic_key())
        yield
        client = state.get("http")
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Paper Search", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(TEMPLATES))

    def open_repo() -> tuple[sqlite3.Connection, Repository]:
        conn = connect(settings.db_path)
        return conn, Repository(conn)

    def make_deps(repo: Repository, round_id: int) -> PipelineDeps:
        if deps_factory is not None:
            return deps_factory(repo, round_id)
        return build_deps(state["http"], state["anthropic"], settings, repo, round_id)

    async def run_round(round_id: int) -> None:
        conn, repo = open_repo()
        try:
            await Pipeline(repo, make_deps(repo, round_id), settings).run_screening(round_id)
        except Exception:
            logger.exception("라운드 %s 실패", round_id)
            repo.add_warning(round_id, "파이프라인이 예기치 않게 중단되었습니다.")
            repo.set_status(round_id, RoundStatus.FAILED)
        finally:
            conn.close()
            state["tasks"].pop(round_id, None)

    # ------------------------------------------------------------ 화면

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        conn, repo = open_repo()
        rounds = repo.list_rounds(10)
        conn.close()
        today = date.today()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "rounds": rounds,
                "default_from": (today - timedelta(days=7)).isoformat(),
                "default_to": today.isoformat(),
                "status_label": {s.value: label for s, label in STATUS_LABEL.items()},
            },
        )

    @app.post("/rounds")
    async def create_round(
        keywords: str = Form(""),
        authors: str = Form(""),
        date_from: str = Form(...),
        date_to: str = Form(...),
        impact_threshold: str = Form(""),
        include_preprints: str = Form(""),
    ) -> Any:
        keyword_list = [k.strip() for k in keywords.replace("\n", ",").split(",") if k.strip()]
        author_list = [a.strip() for a in authors.replace("\n", ",").split(",") if a.strip()]
        if not keyword_list and not author_list:
            raise HTTPException(400, "키워드 또는 연구자를 하나 이상 입력하십시오.")

        spec = RoundInput(
            keywords=keyword_list or ["*"],
            authors=author_list,
            date_from=date.fromisoformat(date_from),
            date_to=date.fromisoformat(date_to),
            impact_threshold=float(impact_threshold) if impact_threshold.strip() else None,
            include_preprints=bool(include_preprints),
        )

        conn, repo = open_repo()
        round_id = repo.create_round(spec)
        conn.close()

        state["tasks"][round_id] = asyncio.create_task(run_round(round_id))
        return RedirectResponse(f"/rounds/{round_id}", status_code=303)

    @app.get("/rounds/{round_id}", response_class=HTMLResponse)
    async def show_round(request: Request, round_id: int) -> Any:
        conn, repo = open_repo()
        try:
            status = repo.get_status(round_id)
            if status is None:
                raise HTTPException(404, "라운드를 찾을 수 없습니다.")
            result = repo.load_result(round_id)
            spec = repo.get_round_input(round_id)
            criteria_rows = repo.list_criteria(round_id)
        finally:
            conn.close()

        if status in RUNNING_STATUSES:
            template = "progress.html"
        elif status is RoundStatus.DONE:
            template = "final.html"
        else:
            template = "screen.html"

        return templates.TemplateResponse(
            request,
            template,
            {
                "result": result,
                "spec": spec,
                "status": status,
                "status_label": STATUS_LABEL.get(status, status.value),
                "criteria_rows": criteria_rows,
                "running": status in RUNNING_STATUSES,
            },
        )

    @app.get("/rounds/{round_id}/progress", response_class=HTMLResponse)
    async def progress(request: Request, round_id: int) -> Any:
        """HTMX 폴링 대상. 진행이 끝나면 브라우저를 결과 화면으로 보낸다."""
        conn, repo = open_repo()
        try:
            status = repo.get_status(round_id)
            if status is None:
                raise HTTPException(404, "라운드를 찾을 수 없습니다.")
            cost = repo.round_cost(round_id)
            counted = len(repo.load_round_papers(round_id))
        finally:
            conn.close()

        response = templates.TemplateResponse(
            request,
            "_progress.html",
            {
                "round_id": round_id,
                "status": status,
                "status_label": STATUS_LABEL.get(status, status.value),
                "cost": cost,
                "counted": counted,
                "running": status in RUNNING_STATUSES,
            },
        )
        if status not in RUNNING_STATUSES:
            response.headers["HX-Redirect"] = f"/rounds/{round_id}"
        return response

    @app.post("/rounds/{round_id}/selection")
    async def apply_selection(request: Request, round_id: int) -> Any:
        form = await request.form()
        selected = {str(v) for v in form.getlist("selected")}

        conn, repo = open_repo()
        try:
            if repo.get_status(round_id) is None:
                raise HTTPException(404, "라운드를 찾을 수 없습니다.")
            await Pipeline(repo, make_deps(repo, round_id), settings).apply_selection(
                round_id, selected
            )
        finally:
            conn.close()
        return RedirectResponse(f"/rounds/{round_id}", status_code=303)

    @app.post("/rounds/{round_id}/criteria/{criterion_id}")
    async def toggle_criterion(round_id: int, criterion_id: int, active: str = Form("")) -> Any:
        conn, repo = open_repo()
        try:
            repo.set_criterion_active(criterion_id, bool(active))
        finally:
            conn.close()
        return RedirectResponse(f"/rounds/{round_id}", status_code=303)

    return app
