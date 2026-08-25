"""모든 논문 소스가 따르는 계약.

소스를 추가할 때 파이프라인을 건드리지 않도록, 반환 타입을 `list[Paper]`로 고정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from paper_search.models import Paper, RoundInput


@dataclass(frozen=True)
class SearchContext:
    """소스 공통 실행 조건."""

    spec: RoundInput
    max_results: int = 200


@runtime_checkable
class PaperSource(Protocol):
    name: str

    async def search(self, ctx: SearchContext) -> list[Paper]:
        """조건에 맞는 논문을 반환한다.

        실패 시 `SourceUnavailable`을 던진다. 파이프라인이 이를 잡아
        경고로 기록하고 나머지 소스로 라운드를 계속한다.
        """
        ...
