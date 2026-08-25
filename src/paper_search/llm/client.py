"""Anthropic 호출 공통 계층.

세 가지를 여기서만 처리한다.
1. 동시성 제한 — 라운드당 200건을 한꺼번에 던지지 않는다
2. 비용 집계와 상한 집행 — 상한에 닿으면 조용히 자르지 않고 예외로 알린다
3. 구조화 출력 — 파싱 실패를 애플리케이션 코드에서 다루지 않는다
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"

# USD per 1M tokens. 캐시 읽기는 입력가의 10%로 잡는다(보수적 추정).
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_DISCOUNT = 0.1

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class CostCapExceeded(RuntimeError):
    """라운드 비용 상한 초과. 파이프라인은 부분 결과를 반환해야 한다."""

    def __init__(self, spent: float, cap: float) -> None:
        super().__init__(f"비용 상한 초과: ${spent:.3f} / 상한 ${cap:.2f}")
        self.spent = spent
        self.cap = cap


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float:
    in_price, out_price = PRICING.get(model, PRICING[MODEL])
    return (
        input_tokens * in_price
        + output_tokens * out_price
        + cache_read_tokens * in_price * CACHE_READ_DISCOUNT
    ) / 1_000_000


@dataclass
class LlmUsage:
    """라운드 단위 사용량 집계."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    per_stage: dict[str, float] = field(default_factory=dict)

    def add(
        self,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cost: float,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cost_usd += cost
        self.calls += 1
        self.per_stage[stage] = self.per_stage.get(stage, 0.0) + cost

    @property
    def cache_hit_ratio(self) -> float:
        total = self.input_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total else 0.0


class LlmClient:
    """구조화 출력 전용 래퍼. 자유 형식 응답은 이 계층을 쓰지 않는다."""

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str = MODEL,
        concurrency: int = 8,
        cost_cap_usd: float | None = None,
        on_usage: Any = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cost_cap_usd = cost_cap_usd
        self.usage = LlmUsage()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._on_usage = on_usage
        self._lock = asyncio.Lock()

    def _check_cap(self) -> None:
        if self.cost_cap_usd is not None and self.usage.cost_usd >= self.cost_cap_usd:
            raise CostCapExceeded(self.usage.cost_usd, self.cost_cap_usd)

    async def parse(
        self,
        *,
        stage: str,
        system: list[dict[str, Any]] | str,
        user: str,
        schema: type[T],
        effort: Effort = "high",
        max_tokens: int = 4096,
    ) -> T:
        """구조화 출력을 파싱된 Pydantic 객체로 받는다.

        `system`을 리스트로 넘기고 앞쪽 블록에 `cache_control`을 걸면 프롬프트 캐시가
        적용된다. 가변 내용(논문 본문)은 반드시 `user`에 둔다.
        """
        self._check_cap()

        async with self._semaphore:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,  # type: ignore[arg-type]
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                output_format=schema,
            )

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = estimate_cost(self.model, usage.input_tokens, usage.output_tokens, cache_read)

        async with self._lock:
            self.usage.add(stage, usage.input_tokens, usage.output_tokens, cache_read, cost)
            if self._on_usage is not None:
                self._on_usage(
                    stage, self.model, usage.input_tokens, usage.output_tokens, cache_read, cost
                )

        parsed = response.parsed_output
        if parsed is None:
            raise ValueError(f"{stage} 단계에서 구조화 출력을 받지 못했습니다")
        return parsed
