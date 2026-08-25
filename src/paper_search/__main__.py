"""CLI 진입점."""

from __future__ import annotations

import argparse
import sys

from paper_search.config import get_settings
from paper_search.store import connect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper-search", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="SQLite 스키마를 생성/마이그레이션한다")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "init-db":
        conn = connect(settings.db_path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        print(f"{settings.db_path} — 스키마 버전 {version}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
