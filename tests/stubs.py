"""LLM 스텁 — 테스트가 실제 API를 치지 않도록 한다."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from paper_search.llm.client import CostCapExceeded, LlmUsage


class StubLlm:
    """`LlmClient.parse`와 같은 모양의 스텁.

    `responses`에 단계별 응답을 넣어두고, 호출 순서대로 꺼내 쓴다.
    `fail_on`에 든 순번은 예외를 던진다.
    """

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        fail_after: int | None = None,
        cap_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_after = fail_after
        self.cap_after = cap_after
        self.error = error or RuntimeError("stub failure")
        self.calls: list[tuple[str, str]] = []
        self.systems: list[Any] = []
        self.usage = LlmUsage()

    async def parse(
        self,
        *,
        stage: str,
        system: Any,
        user: str,
        schema: type[BaseModel],
        effort: str = "high",
        max_tokens: int = 4096,
    ) -> Any:
        self.calls.append((stage, user))
        self.systems.append(system)
        index = len(self.calls)

        if self.cap_after is not None and index > self.cap_after:
            raise CostCapExceeded(3.0, 3.0)
        if self.fail_after is not None and index > self.fail_after:
            raise self.error

        value = self.responses.get(stage)
        if value is None:
            raise AssertionError(f"스텁에 '{stage}' 단계 응답이 없습니다")
        if callable(value):
            return value(user)
        return value
