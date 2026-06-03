from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from local_impact_ranking.enrichment_resume import (
    resume_rows_by_paper_id,
    write_jsonl_atomic,
)
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_doi
from local_impact_ranking.openalex_enrichment import paper_doi


DATACITE_DOIS_URL = "https://api.datacite.org/dois"


def enrich_papers_with_datacite(
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
        "source": "datacite",
        "source_url": DATACITE_DOIS_URL,
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
                attributes = _fetch_datacite_attributes(http_client, doi)
            except Exception as exc:
                attributes = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
            if attributes is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            rows.append(_merge_datacite_attributes(record, attributes))
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


def _fetch_datacite_attributes(client: Any, doi: str) -> dict[str, Any] | None:
    response = client.get(f"{DATACITE_DOIS_URL}/{quote(doi, safe='')}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    attributes = data.get("attributes")
    return attributes if isinstance(attributes, dict) else None


def _merge_datacite_attributes(
    record: dict[str, Any],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    doi = _text(attributes.get("doi"))
    if doi:
        external_ids["doi"] = doi
        if not record.get("doi"):
            record["doi"] = doi
    if external_ids:
        metadata["external_ids"] = external_ids

    title = _title(attributes.get("titles"))
    if title:
        metadata["datacite_title"] = title
    publisher = _text(attributes.get("publisher"))
    if publisher:
        metadata["datacite_publisher"] = publisher
    resource_type = _resource_type(attributes.get("types"))
    if resource_type:
        metadata["datacite_resource_type"] = resource_type
    year = _publication_year(attributes.get("publicationYear"))
    if year is not None and not record.get("year"):
        record["year"] = year

    citation_count = _non_negative_number(attributes.get("citationCount"))
    external_impact = dict(metadata.get("external_impact") or {})
    if citation_count is not None:
        external_impact["datacite_citation_count"] = citation_count
        existing = _non_negative_number(record.get("citation_count"))
        if existing is None or citation_count > existing:
            record["citation_count"] = citation_count
            metadata["citation_count"] = citation_count
    view_count = _non_negative_number(attributes.get("viewCount"))
    if view_count is not None:
        external_impact["datacite_view_count"] = view_count
    download_count = _non_negative_number(attributes.get("downloadCount"))
    if download_count is not None:
        external_impact["datacite_download_count"] = download_count
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("datacite")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["datacite_enriched"] = True
    record["metadata"] = metadata
    return record


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _title(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    first = value[0]
    if isinstance(first, dict):
        return _text(first.get("title"))
    return _text(first)


def _resource_type(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return _text(value.get("resourceTypeGeneral") or value.get("resourceType"))


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
