# Paper Search — 신규 논문 스크리닝

최근 공개된 논문 중 내 관심사에 맞는 것만 주 1회 걸러내는 도구입니다.

- **무엇을 만드는가** → [PRD.md](./PRD.md)
- **어떻게 만드는가** → [build_plan.md](./build_plan.md)

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # ANTHROPIC_API_KEY 등을 채운다
```

## 개발

```bash
ruff check src tests      # 린트
ruff format src tests     # 포맷
mypy                      # 타입 체크
pytest                    # 테스트 (외부 API 호출 테스트는 기본 제외)
pytest -m network         # 실제 외부 API를 치는 테스트만 실행
```

## 현재 상태

| 마일스톤 | 범위 | 상태 |
| --- | --- | --- |
| M0 | 프로젝트 골격, 설정, 도메인 모델, SQLite, CI | 완료 |
| M1 | 검색 파이프라인 (PubMed / bioRxiv / Crossref, 중복 제거) | 예정 |
| M2 | LLM 스크리닝 + 웹 UI (Human gate) | 예정 |
| M3 | 저널 임팩트 지표, 차별성 검증 노출 | 예정 |
| M4 | 선택 기준 추론 및 재랭킹 | 예정 |
