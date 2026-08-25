from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from paper_search.core.cache import DiskCache
from paper_search.core.http import FetchClient, SourceUnavailable
from paper_search.core.ratelimit import AsyncTokenBucket

URL = "https://api.test/thing"


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as c:
        yield c


def _fetch(client: httpx.AsyncClient, **kwargs: object) -> FetchClient:
    return FetchClient(
        client,
        source="test",
        bucket=AsyncTokenBucket(rate=1000),
        sleep=_noop_sleep,
        **kwargs,  # type: ignore[arg-type]
    )


@respx.mock
async def test_get_json(client: httpx.AsyncClient) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    assert await _fetch(client).get_json(URL) == {"ok": True}


@respx.mock
async def test_retries_then_succeeds(client: httpx.AsyncClient) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, text="fine"),
        ]
    )
    assert await _fetch(client).get_text(URL) == "fine"
    assert route.call_count == 3


@respx.mock
async def test_gives_up_after_retries(client: httpx.AsyncClient) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(SourceUnavailable) as exc:
        await _fetch(client).get_text(URL)
    assert exc.value.source == "test"
    assert route.call_count == 5  # 최초 1회 + 재시도 4회


@respx.mock
async def test_non_retryable_status_fails_immediately(client: httpx.AsyncClient) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(SourceUnavailable):
        await _fetch(client).get_text(URL)
    assert route.call_count == 1


@respx.mock
async def test_connection_error_is_retried(client: httpx.AsyncClient) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, text="ok")]
    )
    assert await _fetch(client).get_text(URL) == "ok"
    assert route.call_count == 2


@respx.mock
async def test_cache_prevents_second_request(client: httpx.AsyncClient, tmp_path) -> None:  # type: ignore[no-untyped-def]
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="cached"))
    fetch = _fetch(client, cache=DiskCache(tmp_path))
    assert await fetch.get_text(URL) == "cached"
    assert await fetch.get_text(URL) == "cached"
    assert route.call_count == 1


@respx.mock
async def test_invalid_json_raises_source_unavailable(client: httpx.AsyncClient) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(SourceUnavailable):
        await _fetch(client).get_json(URL)
