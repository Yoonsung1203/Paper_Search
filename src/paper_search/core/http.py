"""외부 API 호출 공통 계층 — 속도 제어, 캐시, 재시도를 한 곳에서 처리한다."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from paper_search.core.cache import DiskCache
from paper_search.core.ratelimit import AsyncTokenBucket, backoff_delays

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class SourceUnavailable(RuntimeError):
    """소스 호출이 재시도 후에도 실패. 라운드는 이 소스 없이 계속되어야 한다."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source} 호출 실패: {detail}")
        self.source = source
        self.detail = detail


class FetchClient:
    """`httpx.AsyncClient` 위에 속도 제어·캐시·백오프를 얹은 얇은 래퍼."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        source: str,
        bucket: AsyncTokenBucket,
        cache: DiskCache | None = None,
        cache_ttl: float | None = None,
        retries: int = 4,
        backoff_base: float = 2.0,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.client = client
        self.source = source
        self.bucket = bucket
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.retries = retries
        self.backoff_base = backoff_base
        self._sleep = sleep

    async def get_text(
        self, url: str, params: dict[str, Any] | None = None, *, use_cache: bool = True
    ) -> str:
        cache_key = DiskCache.key(self.source, url, sorted((params or {}).items()))
        if use_cache and self.cache is not None:
            hit = self.cache.get(cache_key, ttl=self.cache_ttl)
            if isinstance(hit, str):
                logger.debug("cache hit %s %s", self.source, url)
                return hit

        text = await self._request(url, params)

        if use_cache and self.cache is not None:
            self.cache.set(cache_key, text)
        return text

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None, *, use_cache: bool = True
    ) -> Any:
        import json

        raw = await self.get_text(url, params, use_cache=use_cache)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable(self.source, f"JSON 파싱 실패: {exc}") from exc

    async def _request(self, url: str, params: dict[str, Any] | None) -> str:
        delays = backoff_delays(self.retries, base=self.backoff_base)
        last_detail = "알 수 없는 오류"

        for attempt in range(self.retries + 1):
            await self.bucket.acquire()
            try:
                response = await self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return response.text
                last_detail = f"HTTP {response.status_code}"
                if response.status_code not in RETRY_STATUS:
                    raise SourceUnavailable(self.source, last_detail)

            if attempt < self.retries:
                delay = delays[attempt]
                logger.warning(
                    "%s 재시도 %d/%d (%s) — %.0fs 대기",
                    self.source,
                    attempt + 1,
                    self.retries,
                    last_detail,
                    delay,
                )
                await self._sleep(delay)

        raise SourceUnavailable(self.source, f"{self.retries}회 재시도 후 실패 ({last_detail})")
