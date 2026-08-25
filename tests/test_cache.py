from __future__ import annotations

from pathlib import Path

from paper_search.core.cache import DiskCache


def test_roundtrip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = DiskCache.key("pubmed", "https://example.test", {"a": 1})
    cache.set(key, "hello")
    assert cache.get(key) == "hello"


def test_miss_returns_none(tmp_path: Path) -> None:
    assert DiskCache(tmp_path).get("nope") is None


def test_ttl_expiry(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("k", "v")
    assert cache.get("k", ttl=-1) is None


def test_key_is_stable_and_distinct() -> None:
    a = DiskCache.key("pubmed", "url", {"x": 1})
    b = DiskCache.key("pubmed", "url", {"x": 1})
    c = DiskCache.key("pubmed", "url", {"x": 2})
    assert a == b
    assert a != c
    assert a.startswith("pubmed-")


def test_corrupt_file_is_treated_as_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("k", "v")
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get("k") is None
