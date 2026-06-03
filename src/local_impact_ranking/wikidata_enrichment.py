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


WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"


def enrich_papers_with_wikidata(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    timeout: float = 30.0,
    client: Any | None = None,
    user_agent: str = "local-impact-ranking/0.1",
    resume: bool = False,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    destination = Path(output_path)
    resumed_rows = resume_rows_by_paper_id(destination) if resume else {}
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows: list[dict[str, Any]] = []
    summary = {
        "source": "wikidata",
        "source_url": WIKIDATA_SPARQL_URL,
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "doi_query_count": 0,
        "positive_count": 0,
        "zero_count": 0,
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
                result = _fetch_wikidata_result(http_client, doi, user_agent=user_agent)
            except Exception as exc:
                result = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
            if result is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            enriched = _merge_wikidata_result(record, result)
            citation_count = _non_negative_number(
                enriched.get("metadata", {})
                .get("external_impact", {})
                .get("wikidata_cited_by_count")
            )
            summary["matched_count"] += 1
            if citation_count and citation_count > 0.0:
                summary["positive_count"] += 1
            else:
                summary["zero_count"] += 1
            rows.append(enriched)
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


def _fetch_wikidata_result(
    client: Any,
    doi: str,
    *,
    user_agent: str,
) -> dict[str, Any] | None:
    response = client.get(
        WIKIDATA_SPARQL_URL,
        params={"query": _wikidata_citation_query(doi), "format": "json"},
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": user_agent,
        },
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    bindings = payload.get("results", {}).get("bindings")
    if not isinstance(bindings, list) or not bindings:
        return None
    first = bindings[0]
    return first if isinstance(first, dict) else None


def _merge_wikidata_result(
    record: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    entity = _wikidata_entity_id(result.get("work", {}).get("value"))
    if entity:
        metadata["wikidata_entity"] = entity
        external_ids["wikidata"] = entity
    title = _text(result.get("workLabel", {}).get("value"))
    if title and not title.startswith("Q"):
        metadata["wikidata_title"] = title
    if external_ids:
        metadata["external_ids"] = external_ids

    citation_count = _non_negative_number(result.get("count", {}).get("value"))
    external_impact = dict(metadata.get("external_impact") or {})
    if citation_count is not None:
        external_impact["wikidata_cited_by_count"] = citation_count
        existing = _non_negative_number(record.get("citation_count"))
        if citation_count > 0.0 and (existing is None or citation_count > existing):
            record["citation_count"] = citation_count
            metadata["citation_count"] = citation_count
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("wikidata")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["wikidata_enriched"] = True
    record["metadata"] = metadata
    return record


def _wikidata_citation_query(doi: str) -> str:
    escaped_doi = doi.replace("\\", "\\\\").replace('"', '\\"')
    return f"""
SELECT ?work ?workLabel (COUNT(DISTINCT ?citing) AS ?count) WHERE {{
  ?work wdt:P356 ?doi .
  FILTER(LCASE(STR(?doi)) = LCASE("{escaped_doi}"))
  OPTIONAL {{ ?citing wdt:P2860 ?work . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]". }}
}}
GROUP BY ?work ?workLabel
LIMIT 1
""".strip()


def _wikidata_entity_id(value: Any) -> str:
    text = _text(value)
    match = re.search(r"/entity/(Q\d+)$", text)
    return match.group(1) if match else text


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


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
