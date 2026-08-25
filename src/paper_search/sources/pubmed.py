"""PubMed E-utilities 어댑터.

`esearch`로 PMID를 모으고 `efetch`(XML)로 상세를 받는다. `esummary`(JSON)에는 초록이
없어서, 초록과 DOI를 한 번에 얻으려면 efetch XML이 필요하다.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from xml.etree import ElementTree as ET

from paper_search.core.http import FetchClient
from paper_search.models import Paper, Source
from paper_search.sources.base import SearchContext

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_BATCH = 200

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def build_term(keywords: list[str], authors: list[str]) -> str:
    """키워드는 AND, 저자는 OR로 묶고 둘은 OR로 결합한다.

    저자를 AND로 걸면 "그 저자가 쓴, 그 키워드의 논문"만 남아 F-01의
    "연구자의 연구분야 논문" 요구를 만족하지 못한다.
    """
    parts: list[str] = []
    if keywords:
        kw = " AND ".join(f'"{k}"[Title/Abstract]' for k in keywords)
        parts.append(f"({kw})")
    if authors:
        au = " OR ".join(f'"{a}"[Author]' for a in authors)
        parts.append(f"({au})")
    return " OR ".join(parts)


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _parse_pubdate(article: ET.Element) -> date | None:
    """ArticleDate(전자 출판일)를 우선한다. 없으면 Journal의 PubDate."""
    for path in ("ArticleDate", "Journal/JournalIssue/PubDate"):
        node = article.find(path)
        if node is None:
            continue
        year = _text(node.find("Year"))
        if not year.isdigit():
            medline = _text(node.find("MedlineDate"))
            match = re.search(r"\d{4}", medline)
            if not match:
                continue
            return date(int(match.group()), 1, 1)
        raw_month = _text(node.find("Month"))
        month = int(raw_month) if raw_month.isdigit() else _MONTHS.get(raw_month[:3].lower(), 1)
        raw_day = _text(node.find("Day"))
        day = int(raw_day) if raw_day.isdigit() else 1
        try:
            return date(int(year), month, day)
        except ValueError:
            return date(int(year), 1, 1)
    return None


def _parse_abstract(article: ET.Element) -> str:
    """구조화 초록(Label 포함)은 라벨을 살려서 이어붙인다."""
    chunks: list[str] = []
    for node in article.findall("Abstract/AbstractText"):
        body = _text(node)
        if not body:
            continue
        label = node.get("Label")
        chunks.append(f"{label}: {body}" if label else body)
    return "\n".join(chunks)


def _parse_authors(article: ET.Element) -> list[str]:
    out: list[str] = []
    for node in article.findall("AuthorList/Author"):
        collective = _text(node.find("CollectiveName"))
        if collective:
            out.append(collective)
            continue
        last = _text(node.find("LastName"))
        initials = _text(node.find("Initials"))
        if last:
            out.append(f"{last} {initials}".strip())
    return out


def _parse_doi(entry: ET.Element, article: ET.Element) -> str:
    for node in entry.findall("PubmedData/ArticleIdList/ArticleId"):
        if node.get("IdType") == "doi":
            return _text(node)
    for node in article.findall("ELocationID"):
        if node.get("EIdType") == "doi":
            return _text(node)
    return ""


def parse_efetch(xml: str) -> list[Paper]:
    """efetch XML을 `Paper` 목록으로 변환한다. DOI가 없는 레코드는 버린다."""
    root = ET.fromstring(xml)
    papers: list[Paper] = []

    for entry in root.findall("PubmedArticle"):
        article = entry.find("MedlineCitation/Article")
        if article is None:
            continue
        doi = _parse_doi(entry, article)
        if not doi:
            # DOI가 동일성의 기준이므로, 없으면 중복 제거와 링크가 성립하지 않는다.
            continue
        pmid = _text(entry.find("MedlineCitation/PMID"))
        journal = _text(article.find("Journal/Title")) or _text(
            article.find("Journal/ISOAbbreviation")
        )
        issn_node = article.find("Journal/ISSN")
        papers.append(
            Paper(
                doi=doi,
                title=_text(article.find("ArticleTitle")),
                abstract=_parse_abstract(article),
                authors=_parse_authors(article),
                journal=journal,
                issn=_text(issn_node) or None,
                published_at=_parse_pubdate(article),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                if pmid
                else f"https://doi.org/{doi}",
                source=Source.PUBMED,
                is_preprint=False,
                pmid=pmid or None,
            )
        )
    return papers


class PubMedSource:
    name = "pubmed"

    def __init__(self, fetch: FetchClient, *, api_key: str | None = None) -> None:
        self.fetch = fetch
        self.api_key = api_key

    def _params(self, **kwargs: object) -> dict[str, object]:
        params = dict(kwargs)
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def esearch(self, term: str, ctx: SearchContext) -> list[str]:
        payload = await self.fetch.get_json(
            f"{BASE}/esearch.fcgi",
            self._params(
                db="pubmed",
                term=term,
                retmode="json",
                retmax=ctx.max_results,
                sort="date",
                datetype="pdat",
                mindate=ctx.spec.date_from.strftime("%Y/%m/%d"),
                maxdate=ctx.spec.date_to.strftime("%Y/%m/%d"),
            ),
        )
        return list(payload.get("esearchresult", {}).get("idlist", []))

    async def efetch(self, pmids: list[str]) -> list[Paper]:
        papers: list[Paper] = []
        for i in range(0, len(pmids), EFETCH_BATCH):
            batch = pmids[i : i + EFETCH_BATCH]
            xml = await self.fetch.get_text(
                f"{BASE}/efetch.fcgi",
                self._params(db="pubmed", id=",".join(batch), retmode="xml"),
            )
            papers.extend(parse_efetch(xml))
        return papers

    async def search(self, ctx: SearchContext) -> list[Paper]:
        term = build_term(ctx.spec.keywords, ctx.spec.authors)
        if not term:
            return []
        pmids = await self.esearch(term, ctx)
        if not pmids:
            return []
        return await self.efetch(pmids[: ctx.max_results])

    async def author_topics(self, author: str, ctx: SearchContext, top_k: int = 5) -> list[str]:
        """저자의 최근 논문 제목에서 빈출 용어를 뽑아 '연구분야'를 추정한다 (T1-6).

        MeSH를 쓰는 편이 정확하지만 신규 논문에는 아직 색인되지 않는 경우가 많아,
        제목 기반으로 근사한다.
        """
        pmids = await self.esearch(f'"{author}"[Author]', ctx)
        if not pmids:
            return []
        papers = await self.efetch(pmids[:40])
        counter: Counter[str] = Counter()
        for paper in papers:
            for token in re.findall(r"[a-z][a-z\-]{4,}", paper.title.lower()):
                if token not in _STOPWORDS:
                    counter[token] += 1
        return [word for word, count in counter.most_common(top_k) if count > 1]


_STOPWORDS = {
    "using",
    "based",
    "study",
    "novel",
    "human",
    "cells",
    "cell",
    "analysis",
    "between",
    "during",
    "through",
    "these",
    "their",
    "which",
    "against",
    "reveals",
    "identifies",
    "associated",
    "effects",
    "role",
    "roles",
    "with",
}
