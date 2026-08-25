"""CLI 진입점."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta

import httpx

from paper_search.config import Settings, get_settings
from paper_search.core.search import SearchService
from paper_search.models import RoundInput, RoundStatus
from paper_search.store import Repository, connect


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-search", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="디버그 로그 출력")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="SQLite 스키마를 생성/마이그레이션한다")

    search = sub.add_parser("search", help="후보 논문을 검색해 라운드로 저장한다 (M1)")
    search.add_argument(
        "-k", "--keyword", action="append", default=[], help="검색 키워드 (반복 가능)"
    )
    search.add_argument(
        "-a", "--author", action="append", default=[], help="관심 연구자 (반복 가능)"
    )
    search.add_argument("--days", type=int, default=7, help="최근 N일 (기본 7)")
    search.add_argument("--from", dest="date_from", help="시작일 YYYY-MM-DD")
    search.add_argument("--to", dest="date_to", help="종료일 YYYY-MM-DD")
    search.add_argument("--no-preprints", action="store_true", help="프리프린트 제외")

    rounds = sub.add_parser("rounds", help="저장된 라운드 목록")
    rounds.add_argument("--limit", type=int, default=10)

    return parser


def _resolve_dates(args: argparse.Namespace) -> tuple[date, date]:
    end = date.fromisoformat(args.date_to) if args.date_to else date.today()
    start = (
        date.fromisoformat(args.date_from) if args.date_from else end - timedelta(days=args.days)
    )
    return start, end


async def _run_search(spec: RoundInput, settings: Settings) -> int:
    conn = connect(settings.db_path)
    repo = Repository(conn)
    round_id = repo.create_round(spec)
    repo.set_status(round_id, RoundStatus.SEARCHING)

    try:
        async with httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": settings.user_agent}, follow_redirects=True
        ) as client:
            outcome = await SearchService(client, settings).search(spec)

        repo.attach_papers(round_id, outcome.papers)
        for warning in outcome.warnings:
            repo.add_warning(round_id, warning)
        repo.set_status(
            round_id,
            RoundStatus.PARTIAL if outcome.warnings else RoundStatus.AWAITING_SELECTION,
        )

        print(f"라운드 #{round_id} — 후보 {len(outcome.papers)}건")
        for name, count in sorted(outcome.per_source.items()):
            print(f"  {name}: {count}건 수집")
        for warning in outcome.warnings:
            print(f"  ⚠ {warning}")
        if len(outcome.papers) < 5:
            print("  → 결과가 적습니다. 키워드를 완화하거나 기간을 늘려 보십시오.")
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if args.command == "init-db":
        conn = connect(settings.db_path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        print(f"{settings.db_path} — 스키마 버전 {version}")
        return 0

    if args.command == "rounds":
        conn = connect(settings.db_path)
        rows = Repository(conn).list_rounds(args.limit)
        conn.close()
        if not rows:
            print("저장된 라운드가 없습니다.")
        for row in rows:
            print(
                f"#{row['id']:>3}  {row['created_at'][:19]}  {row['status']:<18} "
                f"${row['cost_usd']:.3f}  {row['keywords']}"
            )
        return 0

    if args.command == "search":
        if not args.keyword and not args.author:
            print("키워드(-k) 또는 연구자(-a)를 하나 이상 지정해야 합니다.", file=sys.stderr)
            return 2
        start, end = _resolve_dates(args)
        spec = RoundInput(
            keywords=args.keyword or ["*"],
            authors=args.author,
            date_from=start,
            date_to=end,
            include_preprints=not args.no_preprints,
        )
        return asyncio.run(_run_search(spec, settings))

    return 1


if __name__ == "__main__":
    sys.exit(main())
