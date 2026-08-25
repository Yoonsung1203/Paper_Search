"""환경변수 기반 설정.

키가 없을 때 런타임 깊은 곳에서 터지지 않도록, 필요한 시점에 명확한 에러를 낸다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingCredentialError(RuntimeError):
    """필요한 자격증명이 설정되지 않았을 때."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    ncbi_api_key: str | None = None
    contact_email: str | None = None

    cost_cap_usd: float = 3.0
    db_path: Path = Path("./paper_search.sqlite3")
    cache_dir: Path = Path("./.cache/http")

    # 파이프라인 튜닝
    max_candidates: int = Field(default=200, description="LLM 점수화에 태울 최대 후보 수")
    summarize_top_n: int = Field(default=30, description="요약·차별성 검증 대상 상위 건수")
    llm_concurrency: int = Field(default=8, description="동시 LLM 호출 수")

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise MissingCredentialError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 추가하거나 환경변수로 지정하십시오 (.env.example 참고)."
            )
        return self.anthropic_api_key

    @property
    def user_agent(self) -> str:
        """Crossref/OpenAlex polite pool 용 User-Agent."""
        base = "paper-search/0.1 (https://github.com/Yoonsung1203/Paper_Search)"
        if self.contact_email:
            return f"{base}; mailto:{self.contact_email}"
        return base


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
