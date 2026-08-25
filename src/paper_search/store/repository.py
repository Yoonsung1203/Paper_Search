"""도메인 모델 ↔ SQLite 사이의 유일한 통로."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime

from paper_search.models import (
    Confidence,
    InferredCriterion,
    JournalMetric,
    NoveltyCategory,
    Paper,
    PaperSummary,
    RelevanceScore,
    RoundInput,
    RoundResult,
    RoundStatus,
    ScoredPaper,
    Source,
    SummaryBasis,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _as_date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


class CostCapExceeded(RuntimeError):
    """라운드 비용 상한 초과. 파이프라인은 부분 결과를 반환해야 한다."""

    def __init__(self, spent: float, cap: float) -> None:
        super().__init__(f"라운드 비용 상한 초과: ${spent:.2f} / 상한 ${cap:.2f}")
        self.spent = spent
        self.cap = cap


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------ round

    def create_round(self, spec: RoundInput) -> int:
        cur = self.conn.execute(
            """INSERT INTO round
               (created_at, keywords, authors, date_from, date_to,
                impact_threshold, include_preprints, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now(),
                json.dumps(spec.keywords, ensure_ascii=False),
                json.dumps(spec.authors, ensure_ascii=False),
                spec.date_from.isoformat(),
                spec.date_to.isoformat(),
                spec.impact_threshold,
                int(spec.include_preprints),
                RoundStatus.CREATED.value,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def get_round_input(self, round_id: int) -> RoundInput | None:
        row = self.conn.execute("SELECT * FROM round WHERE id = ?", (round_id,)).fetchone()
        if row is None:
            return None
        return RoundInput(
            keywords=json.loads(row["keywords"]),
            authors=json.loads(row["authors"]),
            date_from=date.fromisoformat(row["date_from"]),
            date_to=date.fromisoformat(row["date_to"]),
            impact_threshold=row["impact_threshold"],
            include_preprints=bool(row["include_preprints"]),
        )

    def set_status(self, round_id: int, status: RoundStatus) -> None:
        self.conn.execute("UPDATE round SET status = ? WHERE id = ?", (status.value, round_id))
        self.conn.commit()

    def get_status(self, round_id: int) -> RoundStatus | None:
        row = self.conn.execute("SELECT status FROM round WHERE id = ?", (round_id,)).fetchone()
        return RoundStatus(row["status"]) if row else None

    def add_warning(self, round_id: int, message: str) -> None:
        """부분 실패를 기록한다. 조용히 삼키지 않기 위한 장치 (계획 §4)."""
        row = self.conn.execute("SELECT warnings FROM round WHERE id = ?", (round_id,)).fetchone()
        warnings: list[str] = json.loads(row["warnings"]) if row else []
        if message not in warnings:
            warnings.append(message)
        self.conn.execute(
            "UPDATE round SET warnings = ? WHERE id = ?",
            (json.dumps(warnings, ensure_ascii=False), round_id),
        )
        self.conn.commit()

    def get_warnings(self, round_id: int) -> list[str]:
        row = self.conn.execute("SELECT warnings FROM round WHERE id = ?", (round_id,)).fetchone()
        return list(json.loads(row["warnings"])) if row else []

    def list_rounds(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM round ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        )

    # ------------------------------------------------------------ paper

    def upsert_papers(self, papers: list[Paper]) -> None:
        self.conn.executemany(
            """INSERT INTO paper
               (doi, title, abstract, authors, journal, issn, published_at,
                url, source, is_preprint, category, pmid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doi) DO UPDATE SET
                 title = excluded.title,
                 abstract = CASE WHEN length(excluded.abstract) > length(paper.abstract)
                                 THEN excluded.abstract ELSE paper.abstract END,
                 journal = COALESCE(NULLIF(excluded.journal, ''), paper.journal),
                 issn = COALESCE(excluded.issn, paper.issn),
                 pmid = COALESCE(excluded.pmid, paper.pmid)""",
            [
                (
                    p.doi,
                    p.title,
                    p.abstract,
                    json.dumps(p.authors, ensure_ascii=False),
                    p.journal,
                    p.issn,
                    p.published_at.isoformat() if p.published_at else None,
                    p.url,
                    p.source.value,
                    int(p.is_preprint),
                    p.category,
                    p.pmid,
                )
                for p in papers
            ],
        )
        self.conn.commit()

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]),
            journal=row["journal"],
            issn=row["issn"],
            published_at=_as_date(row["published_at"]),
            url=row["url"],
            source=Source(row["source"]),
            is_preprint=bool(row["is_preprint"]),
            category=row["category"],
            pmid=row["pmid"],
        )

    # ------------------------------------------------------------ round_paper

    def attach_papers(self, round_id: int, papers: list[Paper]) -> None:
        self.upsert_papers(papers)
        self.conn.executemany(
            "INSERT OR IGNORE INTO round_paper (round_id, doi) VALUES (?, ?)",
            [(round_id, p.doi) for p in papers],
        )
        self.conn.commit()

    def save_relevance(self, round_id: int, doi: str, score: RelevanceScore) -> None:
        self.conn.execute(
            """UPDATE round_paper
               SET relevance_score = ?, evidence = ?, reason = ?
               WHERE round_id = ? AND doi = ?""",
            (
                score.score,
                json.dumps(score.evidence, ensure_ascii=False),
                score.reason,
                round_id,
                doi,
            ),
        )
        self.conn.commit()

    def save_summary(self, round_id: int, doi: str, summary: PaperSummary) -> None:
        self.conn.execute(
            """UPDATE round_paper
               SET summary = ?, summary_basis = ?, novelty_claim = ?,
                   novelty_category = ?, novelty_confidence = ?
               WHERE round_id = ? AND doi = ?""",
            (
                json.dumps(summary.summary, ensure_ascii=False),
                summary.basis.value,
                summary.novelty_claim,
                summary.novelty_category.value,
                summary.novelty_confidence.value,
                round_id,
                doi,
            ),
        )
        self.conn.commit()

    def save_ranks(self, round_id: int, ranked_dois: list[str]) -> None:
        self.conn.executemany(
            "UPDATE round_paper SET rank = ? WHERE round_id = ? AND doi = ?",
            [(i, round_id, doi) for i, doi in enumerate(ranked_dois, start=1)],
        )
        self.conn.commit()

    def load_round_papers(self, round_id: int) -> list[ScoredPaper]:
        rows = self.conn.execute(
            """SELECT rp.*, p.*, jm.metric_name, jm.metric_value, jm.source AS metric_source,
                      jm.year AS metric_year, s.selected
               FROM round_paper rp
               JOIN paper p ON p.doi = rp.doi
               LEFT JOIN journal_metric jm ON jm.issn = p.issn
               LEFT JOIN selection s ON s.round_id = rp.round_id AND s.doi = rp.doi
               WHERE rp.round_id = ?
               ORDER BY COALESCE(rp.rank, 9999), COALESCE(rp.relevance_score, -1) DESC""",
            (round_id,),
        ).fetchall()

        out: list[ScoredPaper] = []
        for row in rows:
            relevance = (
                RelevanceScore(
                    score=row["relevance_score"],
                    evidence=json.loads(row["evidence"]),
                    reason=row["reason"],
                )
                if row["relevance_score"] is not None
                else None
            )
            summary_list = json.loads(row["summary"])
            summary = (
                PaperSummary(
                    summary=summary_list,
                    basis=SummaryBasis(row["summary_basis"]),
                    novelty_claim=row["novelty_claim"],
                    novelty_category=NoveltyCategory(
                        row["novelty_category"] or NoveltyCategory.UNDETERMINED.value
                    ),
                    novelty_confidence=Confidence(row["novelty_confidence"] or "low"),
                )
                if summary_list and row["summary_basis"]
                else None
            )
            metric = (
                JournalMetric(
                    issn=row["issn"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    source=row["metric_source"],
                    year=row["metric_year"],
                )
                if row["metric_name"] is not None
                else None
            )
            out.append(
                ScoredPaper(
                    paper=self._row_to_paper(row),
                    relevance=relevance,
                    summary=summary,
                    metric=metric,
                    selected=None if row["selected"] is None else bool(row["selected"]),
                    rank=row["rank"],
                )
            )
        return out

    # ------------------------------------------------------------ selection

    def save_selection(self, round_id: int, selected_dois: set[str]) -> None:
        """Human gate 결과. 화면에 있던 전 논문에 대해 선택/비선택을 모두 기록한다."""
        dois = [
            r["doi"]
            for r in self.conn.execute(
                "SELECT doi FROM round_paper WHERE round_id = ?", (round_id,)
            ).fetchall()
        ]
        now = _now()
        self.conn.executemany(
            """INSERT INTO selection (round_id, doi, selected, selected_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(round_id, doi) DO UPDATE SET
                 selected = excluded.selected, selected_at = excluded.selected_at""",
            [(round_id, doi, int(doi in selected_dois), now) for doi in dois],
        )
        self.conn.commit()

    def get_selection(self, round_id: int) -> dict[str, bool]:
        return {
            r["doi"]: bool(r["selected"])
            for r in self.conn.execute(
                "SELECT doi, selected FROM selection WHERE round_id = ?", (round_id,)
            ).fetchall()
        }

    # ------------------------------------------------------------ criteria

    def save_criteria(self, round_id: int, criteria: list[InferredCriterion]) -> None:
        self.conn.execute(
            "DELETE FROM criteria WHERE round_id = ? AND origin = 'inferred'", (round_id,)
        )
        self.conn.executemany(
            "INSERT INTO criteria (round_id, text, origin, active) VALUES (?, ?, 'inferred', 1)",
            [(round_id, c.text) for c in criteria],
        )
        self.conn.commit()

    def active_criteria(self, round_id: int) -> list[str]:
        return [
            r["text"]
            for r in self.conn.execute(
                "SELECT text FROM criteria WHERE round_id = ? AND active = 1 ORDER BY id",
                (round_id,),
            ).fetchall()
        ]

    def set_criterion_active(self, criterion_id: int, active: bool) -> None:
        self.conn.execute(
            "UPDATE criteria SET active = ?, origin = 'user_edited' WHERE id = ?",
            (int(active), criterion_id),
        )
        self.conn.commit()

    def update_criterion_text(self, criterion_id: int, text: str) -> None:
        self.conn.execute(
            "UPDATE criteria SET text = ?, origin = 'user_edited' WHERE id = ?",
            (text, criterion_id),
        )
        self.conn.commit()

    def list_criteria(self, round_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM criteria WHERE round_id = ? ORDER BY id", (round_id,)
            ).fetchall()
        )

    def latest_criteria_before(self, round_id: int) -> list[str]:
        """직전 라운드의 활성 기준을 승계한다 (PRD §13-4)."""
        row = self.conn.execute(
            """SELECT round_id FROM criteria
               WHERE round_id < ? AND active = 1
               ORDER BY round_id DESC LIMIT 1""",
            (round_id,),
        ).fetchone()
        return self.active_criteria(row["round_id"]) if row else []

    # ------------------------------------------------------------ journal metric

    def upsert_metric(self, metric: JournalMetric) -> None:
        self.conn.execute(
            """INSERT INTO journal_metric
               (issn, metric_name, metric_value, source, year, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(issn) DO UPDATE SET
                 metric_name = excluded.metric_name,
                 metric_value = excluded.metric_value,
                 source = excluded.source,
                 year = excluded.year,
                 fetched_at = excluded.fetched_at""",
            (
                metric.issn,
                metric.metric_name,
                metric.metric_value,
                metric.source,
                metric.year,
                (metric.fetched_at or datetime.now(UTC)).isoformat(),
            ),
        )
        self.conn.commit()

    def get_metric(self, issn: str) -> JournalMetric | None:
        row = self.conn.execute("SELECT * FROM journal_metric WHERE issn = ?", (issn,)).fetchone()
        if row is None:
            return None
        return JournalMetric(
            issn=row["issn"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            source=row["source"],
            year=row["year"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )

    # ------------------------------------------------------------ 비용

    def record_llm_call(
        self,
        round_id: int | None,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cache_read_tokens: int = 0,
    ) -> None:
        self.conn.execute(
            """INSERT INTO llm_call
               (round_id, stage, model, input_tokens, output_tokens,
                cache_read_tokens, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                round_id,
                stage,
                model,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cost_usd,
                _now(),
            ),
        )
        self.conn.execute(
            "UPDATE round SET cost_usd = cost_usd + ? WHERE id = ?", (cost_usd, round_id)
        )
        self.conn.commit()

    def round_cost(self, round_id: int) -> float:
        row = self.conn.execute("SELECT cost_usd FROM round WHERE id = ?", (round_id,)).fetchone()
        return float(row["cost_usd"]) if row else 0.0

    def cost_breakdown(self, round_id: int) -> list[sqlite3.Row]:
        """단계별 토큰·비용. T2-9(비용 실측)의 입력이다."""
        return list(
            self.conn.execute(
                """SELECT stage, model, COUNT(*) AS calls,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(cache_read_tokens) AS cache_read_tokens,
                          SUM(cost_usd) AS cost_usd
                   FROM llm_call WHERE round_id = ?
                   GROUP BY stage, model ORDER BY cost_usd DESC""",
                (round_id,),
            ).fetchall()
        )

    def cache_hit_ratio(self, round_id: int) -> float:
        """프롬프트 캐시가 실제로 적중하는지 확인하기 위한 지표 (계획 §7.2)."""
        row = self.conn.execute(
            """SELECT SUM(input_tokens) AS inp, SUM(cache_read_tokens) AS cached
               FROM llm_call WHERE round_id = ?""",
            (round_id,),
        ).fetchone()
        total = (row["inp"] or 0) + (row["cached"] or 0)
        return (row["cached"] or 0) / total if total else 0.0

    # ------------------------------------------------------------ 결과 조립

    def load_result(self, round_id: int) -> RoundResult:
        status = self.get_status(round_id) or RoundStatus.FAILED
        return RoundResult(
            round_id=round_id,
            status=status,
            papers=self.load_round_papers(round_id),
            criteria=[
                InferredCriterion(text=r["text"])
                for r in self.list_criteria(round_id)
                if r["active"]
            ],
            cost_usd=self.round_cost(round_id),
            warnings=self.get_warnings(round_id),
        )
