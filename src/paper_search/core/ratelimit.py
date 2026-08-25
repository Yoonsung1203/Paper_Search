"""소스별 요청 속도 제어.

PubMed는 API 키 없이 3 req/s, 키 발급 시 10 req/s를 넘으면 차단된다.
어댑터마다 따로 세는 것이 아니라, 소스 단위로 하나의 버킷을 공유해야 의미가 있다.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class AsyncTokenBucket:
    """초당 `rate` 개의 토큰이 채워지는 버킷. 토큰이 없으면 대기한다."""

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate는 0보다 커야 합니다")
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError("요청 토큰이 버킷 용량보다 큽니다")
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(wait)

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


def backoff_delays(retries: int = 4, base: float = 2.0, factor: float = 2.0) -> list[float]:
    """지수 백오프 지연 시간 (기본 2s → 4s → 8s → 16s)."""
    return [base * (factor**i) for i in range(retries)]
