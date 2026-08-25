"""외부 API 응답 디스크 캐시.

같은 라운드를 다시 돌리거나 개발 중 반복 실행할 때 외부 API를 다시 치지 않기 위한 것이다.
TTL이 지난 항목은 없는 것으로 취급한다.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, root: Path | str, *, default_ttl: float | None = None) -> None:
        self.root = Path(root)
        self.default_ttl = default_ttl

    @staticmethod
    def key(namespace: str, *parts: Any) -> str:
        raw = "|".join([namespace, *(str(p) for p in parts)])
        return f"{namespace}-{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, *, ttl: float | None = None) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if effective_ttl is not None and time.time() - payload.get("stored_at", 0) > effective_ttl:
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"stored_at": time.time(), "value": value}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    def clear(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.rglob("*.json"):
            path.unlink()
