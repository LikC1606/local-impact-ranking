from __future__ import annotations

import json
import re
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


OPENCITATIONS_INDEX_URL = "https://api.opencitations.net/index/v2"
DOI_PID_RE = re.compile(r"\bdoi:([^\s;]+)", re.IGNORECASE)
DEFAULT_MAX_INCOMING_EDGES_PER_PAPER = 500


def enrich_papers_with_opencitations(
    input_path: str | Path,
    output_path: str | Path,
    *,
    token: str | None = None,
    timeout: float = 30.0,
    include_edges: bool = True,
    max_incoming_edges_per_paper: int = DEFAULT_MAX_INCOMING_EDGES_PER_PAPER,
    materialize_internal_edges: bool = False,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    if max_incoming_edges_per_paper < 0:
        raise ValueError("max_incoming_edges_per_paper must be at least 0")
    papers = load_papers_jsonl(input_path)
    destination = Path(output_path)
    resumed_rows = resume_rows_by_paper_id(destination) if resume else {}
    doi_to_paper_id = _doi_to_paper_id(papers)
    paper_lookup = {paper.paper_id: paper for paper in papers}
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows = []
    pending_materialized_edges: dict[str, list[dict[str, Any]]] = {}
    summary = {
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "internal_incoming_edge_count": 0,
        "external_incoming_edge_count": 0,
        "stored_incoming_edge_count": 0,
        "truncated_incoming_edge_count": 0,
        "papers_with_truncated_incoming_edges": 0,
        "materialized_internal_edge_count": 0,
        "source": "opencitations",
        "api": "index/v2",
        "include_edges": include_edges,
        "max_incoming_edges_per_paper": max_incoming_edges_per_paper,
        "materialize_internal_edges": materialize_internal_edges,
        "resume": resume,
        "resumed_count": 0,
    }

    try:
        for paper in papers:
            resumed_row = resumed_rows.get(paper.paper_id)
            if resumed_row is not None:
                rows.append(resumed_row)
                summary["resumed_count"] += 1
                if materialize_internal_edges:
                    for edge in _stored_opencitations_incoming_edges(resumed_row):
                        reference = _materialized_reference(
                            edge,
                            paper_lookup=paper_lookup,
                        )
                        if reference is None:
                            continue
                        pending_materialized_edges.setdefault(
                            edge["source_paper_id"],
                            [],
                        ).append(reference)
                        summary["materialized_internal_edge_count"] += 1
                continue
            record = paper.model_dump(mode="json", exclude_none=True)
            doi = paper_doi(paper)
            if not doi:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue

            normalized_doi = _normalize_doi(doi)
            summary["queryable_count"] += 1
            citation_count = _fetch_count(
                http_client,
                "citation-count",
                normalized_doi,
                token=token,
            )
            reference_count = _fetch_count(
                http_client,
                "reference-count",
                normalized_doi,
                token=token,
            )
            if citation_count is None and reference_count is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue

            stored_incoming_edges = []
            truncated_count = 0
            if include_edges:
                citation_rows = _fetch_citation_rows(
                    http_client,
                    "citations",
                    normalized_doi,
                    token=token,
                )
                incoming_edges = _internal_incoming_edges(
                    citation_rows,
                    target_paper_id=paper.paper_id,
                    target_doi=normalized_doi,
                    doi_to_paper_id=doi_to_paper_id,
                )
                summary["internal_incoming_edge_count"] += len(incoming_edges)
                summary["external_incoming_edge_count"] += max(
                    len(citation_rows) - len(incoming_edges),
                    0,
                )
                stored_incoming_edges = incoming_edges[
                    :max_incoming_edges_per_paper
                ]
                summary["stored_incoming_edge_count"] += len(stored_incoming_edges)
                truncated_count = len(incoming_edges) - len(stored_incoming_edges)
                if truncated_count > 0:
                    summary["truncated_incoming_edge_count"] += truncated_count
                    summary["papers_with_truncated_incoming_edges"] += 1
                if materialize_internal_edges:
                    for edge in incoming_edges:
                        reference = _materialized_reference(
                            edge,
                            paper_lookup=paper_lookup,
                        )
                        if reference is None:
                            continue
                        pending_materialized_edges.setdefault(
                            edge["source_paper_id"],
                            [],
                        ).append(reference)
                        summary["materialized_internal_edge_count"] += 1

            rows.append(
                _merge_opencitations_evidence(
                    record,
                    doi=normalized_doi,
                    citation_count=citation_count,
                    reference_count=reference_count,
                    incoming_edges=stored_incoming_edges,
                    truncated_incoming_edge_count=truncated_count,
                )
            )
            summary["matched_count"] += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    if materialize_internal_edges and pending_materialized_edges:
        rows = [
            _merge_materialized_edges(
                row,
                pending_materialized_edges.get(str(row.get("paper_id")), []),
            )
            for row in rows
        ]

    write_jsonl_atomic(destination, rows)
    summary["output_path"] = str(destination)
    return summary


def _fetch_count(
    client: Any,
    operation: str,
    doi: str,
    *,
    token: str | None,
) -> float | None:
    payload = _fetch_json(client, operation, doi, token=token)
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    return _non_negative_number(first.get("count"))


def _fetch_citation_rows(
    client: Any,
    operation: str,
    doi: str,
    *,
    token: str | None,
) -> list[dict[str, Any]]:
    payload = _fetch_json(client, operation, doi, token=token)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _fetch_json(
    client: Any,
    operation: str,
    doi: str,
    *,
    token: str | None,
) -> Any | None:
    headers = {"authorization": token} if token else None
    response = client.get(_opencitations_url(operation, doi), headers=headers)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _opencitations_url(operation: str, doi: str) -> str:
    return f"{OPENCITATIONS_INDEX_URL}/{operation}/doi:{quote(_normalize_doi(doi), safe='')}"


def _merge_opencitations_evidence(
    record: dict[str, Any],
    *,
    doi: str,
    citation_count: float | None,
    reference_count: float | None,
    incoming_edges: list[dict[str, Any]],
    truncated_incoming_edge_count: int,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    external_ids["doi"] = doi
    metadata["external_ids"] = external_ids

    external_impact = dict(metadata.get("external_impact") or {})
    if citation_count is not None:
        record["citation_count"] = citation_count
        metadata["citation_count"] = citation_count
        external_impact["opencitations_cited_by_count"] = citation_count
    if reference_count is not None:
        external_impact["opencitations_reference_count"] = reference_count
    metadata["external_impact"] = external_impact
    metadata["opencitations_enriched"] = True
    if incoming_edges:
        metadata["opencitations_incoming_edges"] = incoming_edges
    if truncated_incoming_edge_count > 0:
        metadata["opencitations_incoming_edges_truncated_count"] = (
            truncated_incoming_edge_count
        )
    record["metadata"] = metadata
    return record


def _stored_opencitations_incoming_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return []
    edges = metadata.get("opencitations_incoming_edges")
    if not isinstance(edges, list):
        return []
    return [edge for edge in edges if isinstance(edge, dict)]


def _internal_incoming_edges(
    rows: list[dict[str, Any]],
    *,
    target_paper_id: str,
    target_doi: str,
    doi_to_paper_id: dict[str, str],
) -> list[dict[str, Any]]:
    edges = []
    target_doi = _normalize_doi(target_doi)
    for row in rows:
        citing_doi = _first_doi(row.get("citing"))
        if not citing_doi:
            continue
        source_paper_id = doi_to_paper_id.get(citing_doi)
        if not source_paper_id:
            continue
        edges.append(
            {
                "source_paper_id": source_paper_id,
                "source_doi": citing_doi,
                "target_paper_id": target_paper_id,
                "target_doi": target_doi,
                "oci": str(row.get("oci", "")),
                "creation": str(row.get("creation", "")),
                "journal_self_citation": _yes_no(row.get("journal_sc")),
                "author_self_citation": _yes_no(row.get("author_sc")),
            }
        )
    return edges


def _materialized_reference(
    edge: dict[str, Any],
    *,
    paper_lookup: dict[str, Paper],
) -> dict[str, Any] | None:
    target = paper_lookup.get(str(edge.get("target_paper_id")))
    if target is None:
        return None
    target_doi = str(edge.get("target_doi", ""))
    return {
        "reference_id": target.paper_id,
        "title": target.title,
        "doi": target_doi,
        "external_ids": {"doi": target_doi},
        "raw_citation": "OpenCitations internal DOI edge",
    }


def _merge_materialized_edges(
    record: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    if not references:
        return record
    existing_references = list(record.get("references") or [])
    existing_ids = {
        reference.get("reference_id")
        for reference in existing_references
        if isinstance(reference, dict)
    }
    added = []
    for reference in references:
        reference_id = reference.get("reference_id")
        if not reference_id or reference_id in existing_ids:
            continue
        existing_references.append(reference)
        existing_ids.add(reference_id)
        added.append(reference)
    if not added:
        return record
    metadata = dict(record.get("metadata") or {})
    metadata["opencitations_materialized_edge_count"] = (
        float(metadata.get("opencitations_materialized_edge_count") or 0.0) + len(added)
    )
    record["references"] = existing_references
    record["metadata"] = metadata
    return record


def _doi_to_paper_id(papers: list[Paper]) -> dict[str, str]:
    mapping = {}
    for paper in papers:
        doi = paper_doi(paper)
        if not doi:
            continue
        mapping[_normalize_doi(doi)] = paper.paper_id
    return mapping


def _first_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = DOI_PID_RE.search(value)
    if match is None:
        return None
    return _normalize_doi(match.group(1))


def _normalize_doi(value: str) -> str:
    return normalize_doi(value).lower()


def _yes_no(value: Any) -> bool | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None


def _non_negative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
