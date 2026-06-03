from __future__ import annotations

import json
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


DATACITE_EVENTS_URL = "https://api.datacite.org/events"
DATACITE_EVENTS_CITATION_RELATION_TYPES = ("references", "cites")


def enrich_papers_with_datacite_events(
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
        "source": "datacite_events",
        "source_url": DATACITE_EVENTS_URL,
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
            obj_id = _doi_obj_id(doi)
            try:
                meta = _fetch_datacite_events_meta(http_client, obj_id)
            except Exception as exc:
                meta = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
            if meta is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            enriched = _merge_datacite_events_meta(record, obj_id, meta)
            citation_count = _non_negative_number(
                enriched.get("metadata", {})
                .get("external_impact", {})
                .get("datacite_events_cited_by_count")
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


def _fetch_datacite_events_meta(client: Any, obj_id: str) -> dict[str, Any] | None:
    response = client.get(
        DATACITE_EVENTS_URL,
        params={"obj-id": obj_id, "page[size]": 0},
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else None


def _merge_datacite_events_meta(
    record: dict[str, Any],
    obj_id: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_impact = dict(metadata.get("external_impact") or {})
    relation_counts = _count_items_by_id(meta.get("relation-types"))
    source_counts = _count_items_by_id(meta.get("sources"))
    citation_count = sum(
        relation_counts.get(relation_type, 0.0)
        for relation_type in DATACITE_EVENTS_CITATION_RELATION_TYPES
    )

    external_impact["datacite_events_cited_by_count"] = citation_count
    external_impact["datacite_events_total_event_count"] = _non_negative_number(
        meta.get("total")
    ) or 0.0
    for relation_type in DATACITE_EVENTS_CITATION_RELATION_TYPES:
        external_impact[f"datacite_events_{_field_token(relation_type)}_count"] = (
            relation_counts.get(relation_type, 0.0)
        )
    for source_id, count in source_counts.items():
        external_impact[f"datacite_events_{_field_token(source_id)}_count"] = count

    existing = _non_negative_number(record.get("citation_count"))
    if citation_count > 0.0 and (existing is None or citation_count > existing):
        record["citation_count"] = citation_count
        metadata["citation_count"] = citation_count
    metadata["external_impact"] = external_impact
    metadata["datacite_events_enriched"] = True
    metadata["datacite_events_obj_id"] = obj_id
    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("datacite_events")
    metadata["external_evidence_sources"] = sorted(sources)
    record["metadata"] = metadata
    return record


def _count_items_by_id(value: Any) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        item_id = _text(item.get("id"))
        count = _non_negative_number(item.get("count"))
        if item_id and count is not None:
            counts[item_id] = count
    return counts


def _doi_obj_id(doi: str) -> str:
    return f"https://doi.org/{doi}"


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _field_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in value.lower())
    return "_".join(part for part in token.split("_") if part)


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
