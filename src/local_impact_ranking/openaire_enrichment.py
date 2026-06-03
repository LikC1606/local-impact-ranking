from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.enrichment_resume import (
    resume_rows_by_paper_id,
    write_jsonl_atomic,
)
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_doi
from local_impact_ranking.openalex_enrichment import paper_doi


OPENAIRE_RESEARCH_PRODUCTS_URL = "https://api.openaire.eu/graph/v2/researchProducts"


def enrich_papers_with_openaire(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    timeout: float = 30.0,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    destination = Path(output_path)
    resumed_rows = resume_rows_by_paper_id(destination) if resume else {}
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows: list[dict[str, Any]] = []
    summary = {
        "source": "openaire",
        "source_url": OPENAIRE_RESEARCH_PRODUCTS_URL,
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "doi_query_count": 0,
        "query_error_count": 0,
        "last_error": "",
        "resume": resume,
        "resumed_count": 0,
    }
    try:
        for paper in papers:
            resumed_row = resumed_rows.get(paper.paper_id)
            if resumed_row is not None:
                rows.append(resumed_row)
                summary["resumed_count"] += 1
                continue
            record = paper.model_dump(mode="json", exclude_none=True)
            doi = _normalize_doi(paper_doi(paper) or "")
            if not doi:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            summary["queryable_count"] += 1
            summary["doi_query_count"] += 1
            try:
                result = _fetch_openaire_result(http_client, doi)
            except Exception as exc:
                result = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
            if result is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            rows.append(_merge_openaire_result(record, result))
            summary["matched_count"] += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    write_jsonl_atomic(destination, rows)
    summary["output_path"] = str(destination)
    if summary_output is not None:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def _fetch_openaire_result(client: Any, doi: str) -> dict[str, Any] | None:
    response = client.get(
        OPENAIRE_RESEARCH_PRODUCTS_URL,
        params={"pid": doi, "pageSize": 1},
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    candidate = results[0]
    return candidate if isinstance(candidate, dict) else None


def _merge_openaire_result(
    record: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    openaire_id = _text(result.get("id"))
    if openaire_id:
        metadata["openaire_id"] = openaire_id
    title = _text(result.get("mainTitle"))
    if title:
        metadata["openaire_title"] = title
    result_type = _text(result.get("type"))
    if result_type:
        metadata["openaire_type"] = result_type
    publisher = _text(result.get("publisher"))
    if publisher:
        metadata["openaire_publisher"] = publisher
    year = _publication_year_from_date(result.get("publicationDate"))
    if year is not None and not record.get("year"):
        record["year"] = year

    impact = _citation_impact(result)
    external_impact = dict(metadata.get("external_impact") or {})
    citation_count = _non_negative_number(impact.get("citationCount"))
    if citation_count is not None:
        external_impact["openaire_citation_count"] = citation_count
        existing = _non_negative_number(record.get("citation_count"))
        if existing is None or citation_count > existing:
            record["citation_count"] = citation_count
            metadata["citation_count"] = citation_count
    for source_key, output_key in (
        ("influence", "openaire_influence"),
        ("popularity", "openaire_popularity"),
        ("impulse", "openaire_impulse"),
    ):
        value = _non_negative_number(impact.get(source_key))
        if value is not None:
            external_impact[output_key] = value
    citation_class = _text(impact.get("citationClass"))
    if citation_class:
        external_impact["openaire_citation_class"] = citation_class
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("openaire")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["openaire_enriched"] = True
    record["metadata"] = metadata
    return record


def _citation_impact(result: dict[str, Any]) -> dict[str, Any]:
    indicators = result.get("indicators")
    if not isinstance(indicators, dict):
        return {}
    citation_impact = indicators.get("citationImpact")
    return citation_impact if isinstance(citation_impact, dict) else {}


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _publication_year_from_date(value: Any) -> int | None:
    text = _text(value)
    match = re.match(r"^(\d{4})", text)
    if not match:
        return None
    return _publication_year(match.group(1))


def _publication_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year < 1800 or year > 2100:
        return None
    return year


def _non_negative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
