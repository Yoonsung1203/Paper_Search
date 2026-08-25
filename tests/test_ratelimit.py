from __future__ import annotations

import asyncio

import pytest

from paper_search.core.ratelimit import AsyncTokenBucket, backoff_delays


class FakeClock:
    """실제로 기다리지 않고 토큰버킷 동작을 검증하기 위한 가짜 시계."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def test_bucket_allows_burst_up_to_capacity() -> None:
    clock = FakeClock()
    bucket = AsyncTokenBucket(rate=3.0, clock=clock)
    for _ in range(3):
        await bucket.acquire()
    assert bucket.available == pytest.approx(0.0)


async def test_bucket_refills_over_time() -> None:
    clock = FakeClock()
    bucket = AsyncTokenBucket(rate=3.0, clock=clock)
    for _ in range(3):
        await bucket.acquire()
    clock.advance(1.0)
    assert bucket.available == pytest.approx(3.0)


async def test_bucket_blocks_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    bucket = AsyncTokenBucket(rate=3.0, clock=clock)
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    for _ in range(4):
        await bucket.acquire()

    assert slept, "네 번째 요청은 대기해야 한다"
    assert slept[0] == pytest.approx(1 / 3)


async def test_rate_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AsyncTokenBucket(rate=0)


def test_backoff_delays_are_exponential() -> None:
    assert backoff_delays() == [2.0, 4.0, 8.0, 16.0]
