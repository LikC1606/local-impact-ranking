from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.openalex_enrichment import paper_doi


DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
MIN_TITLE_MATCH_LENGTH = 20
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


@dataclass(frozen=True)
class InternalCitationMatch:
    target_paper_id: str
    target_title: str
    target_doi: str | None
    method: str
    confidence: float


def materialize_internal_citation_edges(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    title_threshold: float = 0.95,
    title_margin: float = 0.03,
    openalex_references: bool = False,
    openalex_email: str | None = None,
    openalex_timeout: float = 30.0,
    openalex_client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    paper_ids = {paper.paper_id for paper in papers}
    doi_index = _doi_index(papers)
    title_entries = _title_entries(papers)
    summary: dict[str, Any] = {
        "paper_count": len(papers),
        "reference_count": 0,
        "existing_internal_edge_count": 0,
        "materialized_internal_edge_count": 0,
        "doi_match_count": 0,
        "title_match_count": 0,
        "self_reference_skipped_count": 0,
        "duplicate_internal_edge_skipped_count": 0,
        "ambiguous_title_match_count": 0,
        "unmatched_reference_count": 0,
        "openalex_references_enabled": openalex_references,
        "openalex_reference_query_count": 0,
        "openalex_reference_match_count": 0,
        "openalex_reference_error_count": 0,
        "title_threshold": title_threshold,
        "title_margin": title_margin,
        "source": "dataset_internal_references",
    }
    output_rows = []

    for paper in papers:
        row = paper.model_dump(mode="json", exclude_none=True)
        references = [
            reference.model_dump(mode="json", exclude_none=True)
            for reference in paper.references
        ]
        metadata = dict(row.get("metadata") or {})
        source_seen_targets: set[str] = set()
        new_edges: list[dict[str, Any]] = []
        materialized_count = 0
        existing_count = 0
        unmatched_count = 0

        for reference in references:
            summary["reference_count"] += 1
            reference_id = reference.get("reference_id")
            if isinstance(reference_id, str) and reference_id in paper_ids:
                existing_count += 1
                source_seen_targets.add(reference_id)
                continue

            match = _match_reference(
                source_paper=paper,
                reference=reference,
                doi_index=doi_index,
                title_entries=title_entries,
                title_threshold=title_threshold,
                title_margin=title_margin,
            )
            if match == "self":
                summary["self_reference_skipped_count"] += 1
                continue
            if match == "ambiguous":
                summary["ambiguous_title_match_count"] += 1
                continue
            if match is None:
                summary["unmatched_reference_count"] += 1
                unmatched_count += 1
                continue
            if match.target_paper_id in source_seen_targets:
                summary["duplicate_internal_edge_skipped_count"] += 1
                continue

            reference["reference_id"] = match.target_paper_id
            reference["internal_citation_match_method"] = match.method
            reference["internal_citation_match_confidence"] = round(match.confidence, 6)
            source_seen_targets.add(match.target_paper_id)
            materialized_count += 1
            summary["materialized_internal_edge_count"] += 1
            if match.method == "doi":
                summary["doi_match_count"] += 1
            else:
                summary["title_match_count"] += 1
            new_edges.append(
                {
                    "source_paper_id": paper.paper_id,
                    "target_paper_id": match.target_paper_id,
                    "target_title": match.target_title,
                    "target_doi": match.target_doi,
                    "match_method": match.method,
                    "confidence": round(match.confidence, 6),
                    "reference_title": str(reference.get("title") or ""),
                    "reference_doi": _reference_doi(reference),
                }
            )

        summary["existing_internal_edge_count"] += existing_count
        metadata["internal_citation_matching_attempted"] = True
        metadata["internal_citation_openalex_references_checked"] = openalex_references
        metadata["internal_citation_reference_count"] = len(references)
        metadata["internal_citation_unmatched_reference_count"] = unmatched_count
        metadata["internal_citation_edge_count"] = existing_count + materialized_count
        if materialized_count:
            metadata["internal_citation_materialized_edge_count"] = materialized_count
        if new_edges:
            existing_edges = metadata.get("internal_citation_edges")
            if not isinstance(existing_edges, list):
                existing_edges = []
            metadata["internal_citation_edges"] = [*existing_edges, *new_edges]
        row["references"] = references
        row["metadata"] = metadata
        output_rows.append(row)

    if openalex_references:
        _materialize_openalex_reference_edges(
            papers,
            output_rows,
            paper_ids=paper_ids,
            summary=summary,
            email=openalex_email,
            timeout=openalex_timeout,
            client=openalex_client,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["output_path"] = str(destination)
    if summary_output is not None:
        summary_destination = Path(summary_output)
        summary_destination.parent.mkdir(parents=True, exist_ok=True)
        summary_destination.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def _materialize_openalex_reference_edges(
    papers: list[Paper],
    rows: list[dict[str, Any]],
    *,
    paper_ids: set[str],
    summary: dict[str, Any],
    email: str | None,
    timeout: float,
    client: Any | None,
) -> None:
    openalex_to_paper = {
        openalex_id: paper
        for paper in papers
        if (openalex_id := _paper_openalex_id(paper)) is not None
    }
    row_by_paper_id = {str(row.get("paper_id")): row for row in rows}
    source_targets = {
        paper_id: _internal_reference_targets(row, paper_ids)
        for paper_id, row in row_by_paper_id.items()
    }
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        for source_paper in papers:
            source_openalex_id = _paper_openalex_id(source_paper)
            if source_openalex_id is None:
                continue
            summary["openalex_reference_query_count"] += 1
            referenced_works = _fetch_openalex_referenced_works(
                http_client,
                source_openalex_id,
                email=email,
            )
            if referenced_works is None:
                summary["openalex_reference_error_count"] += 1
                continue
            source_row = row_by_paper_id.get(source_paper.paper_id)
            if source_row is None:
                continue
            seen_targets = source_targets.setdefault(source_paper.paper_id, set())
            for referenced_work in referenced_works:
                target = openalex_to_paper.get(_normalize_openalex_id(referenced_work))
                if target is None or target.paper_id == source_paper.paper_id:
                    continue
                if target.paper_id in seen_targets:
                    summary["duplicate_internal_edge_skipped_count"] += 1
                    continue
                _append_materialized_reference(
                    source_row,
                    source_paper_id=source_paper.paper_id,
                    target_paper=target,
                    method="openalex_referenced_work",
                    confidence=1.0,
                    raw_citation="OpenAlex referenced_works internal edge",
                )
                seen_targets.add(target.paper_id)
                summary["materialized_internal_edge_count"] += 1
                summary["openalex_reference_match_count"] += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()


def _fetch_openalex_referenced_works(
    client: Any,
    openalex_id: str,
    *,
    email: str | None,
) -> list[str] | None:
    params: dict[str, Any] = {"select": "id,referenced_works"}
    if email:
        params["mailto"] = email
    response = client.get(
        f"{OPENALEX_WORKS_URL}/{quote(openalex_id, safe='')}",
        params=params,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    referenced_works = payload.get("referenced_works")
    if not isinstance(referenced_works, list):
        return []
    return [str(value) for value in referenced_works if value]


def _append_materialized_reference(
    row: dict[str, Any],
    *,
    source_paper_id: str,
    target_paper: Paper,
    method: str,
    confidence: float,
    raw_citation: str,
) -> None:
    target_doi = _paper_normalized_doi(target_paper)
    reference: dict[str, Any] = {
        "reference_id": target_paper.paper_id,
        "title": target_paper.title,
        "raw_citation": raw_citation,
        "internal_citation_match_method": method,
        "internal_citation_match_confidence": round(confidence, 6),
    }
    if target_doi:
        reference["doi"] = target_doi
        reference["external_ids"] = {"doi": target_doi}
    references = list(row.get("references") or [])
    references.append(reference)
    row["references"] = references

    metadata = dict(row.get("metadata") or {})
    metadata["internal_citation_edge_count"] = (
        int(metadata.get("internal_citation_edge_count") or 0) + 1
    )
    metadata["internal_citation_materialized_edge_count"] = (
        int(metadata.get("internal_citation_materialized_edge_count") or 0) + 1
    )
    existing_edges = metadata.get("internal_citation_edges")
    if not isinstance(existing_edges, list):
        existing_edges = []
    metadata["internal_citation_edges"] = [
        *existing_edges,
        {
            "source_paper_id": source_paper_id,
            "target_paper_id": target_paper.paper_id,
            "target_title": target_paper.title,
            "target_doi": target_doi,
            "match_method": method,
            "confidence": round(confidence, 6),
            "reference_title": target_paper.title,
            "reference_doi": target_doi,
        },
    ]
    row["metadata"] = metadata


def _internal_reference_targets(
    row: dict[str, Any],
    paper_ids: set[str],
) -> set[str]:
    targets = set()
    for reference in row.get("references") or []:
        if not isinstance(reference, dict):
            continue
        reference_id = reference.get("reference_id")
        if isinstance(reference_id, str) and reference_id in paper_ids:
            targets.add(reference_id)
    return targets


def _match_reference(
    *,
    source_paper: Paper,
    reference: dict[str, Any],
    doi_index: dict[str, Paper],
    title_entries: list[tuple[str, Paper]],
    title_threshold: float,
    title_margin: float,
) -> InternalCitationMatch | str | None:
    reference_doi = _reference_doi(reference)
    if reference_doi:
        target = doi_index.get(reference_doi)
        if target is not None:
            if target.paper_id == source_paper.paper_id:
                return "self"
            return InternalCitationMatch(
                target_paper_id=target.paper_id,
                target_title=target.title,
                target_doi=_paper_normalized_doi(target),
                method="doi",
                confidence=1.0,
            )

    reference_title = str(reference.get("title") or "")
    normalized_reference_title = _normalize_title(reference_title)
    if len(normalized_reference_title) < MIN_TITLE_MATCH_LENGTH:
        return None

    scored = []
    for normalized_title, target in title_entries:
        score = _title_similarity(normalized_reference_title, normalized_title)
        scored.append((score, target))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < title_threshold:
        return None

    best_score, best_target = scored[0]
    if best_target.paper_id == source_paper.paper_id:
        return "self"
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if second_score >= title_threshold and best_score - second_score < title_margin:
        return "ambiguous"
    return InternalCitationMatch(
        target_paper_id=best_target.paper_id,
        target_title=best_target.title,
        target_doi=_paper_normalized_doi(best_target),
        method="title",
        confidence=best_score,
    )


def _doi_index(papers: list[Paper]) -> dict[str, Paper]:
    index = {}
    for paper in papers:
        doi = _paper_normalized_doi(paper)
        if doi and doi not in index:
            index[doi] = paper
    return index


def _title_entries(papers: list[Paper]) -> list[tuple[str, Paper]]:
    entries = []
    for paper in papers:
        normalized = _normalize_title(paper.title)
        if len(normalized) >= MIN_TITLE_MATCH_LENGTH:
            entries.append((normalized, paper))
    return entries


def _reference_doi(reference: dict[str, Any]) -> str | None:
    for value in (
        reference.get("doi"),
        _external_id(reference, "doi"),
        _external_id(reference, "DOI"),
        _doi_from_text(reference.get("raw_citation")),
    ):
        if value:
            return _normalize_doi(str(value))
    return None


def _external_id(record: dict[str, Any], key: str) -> str | None:
    external_ids = record.get("external_ids")
    if not isinstance(external_ids, dict):
        return None
    value = external_ids.get(key)
    return str(value) if value else None


def _paper_normalized_doi(paper: Paper) -> str | None:
    doi = paper_doi(paper)
    return _normalize_doi(doi) if doi else None


def _paper_openalex_id(paper: Paper) -> str | None:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get("openalex"):
        return _normalize_openalex_id(external_ids["openalex"])
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    value = extra.get("openalex")
    return _normalize_openalex_id(value) if value else None


def _normalize_openalex_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().rstrip("/")
    if not normalized:
        return None
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized


def _doi_from_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = DOI_RE.search(value)
    if match is None:
        return None
    return match.group(1)


def _normalize_doi(value: str) -> str:
    normalized = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    return normalized.rstrip(".,);]")


def _title_similarity(normalized_left: str, normalized_right: str) -> float:
    if normalized_left == normalized_right:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _normalize_title(title: str) -> str:
    normalized = title.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())
