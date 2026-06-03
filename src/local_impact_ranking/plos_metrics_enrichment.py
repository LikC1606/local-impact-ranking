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


PLOS_SEARCH_URL = "https://api.plos.org/search"


def enrich_papers_with_plos_metrics(
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
        "source": "plos_metrics",
        "source_url": PLOS_SEARCH_URL,
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
                result = _fetch_plos_result(http_client, doi)
            except Exception as exc:
                result = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
            if result is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            enriched = _merge_plos_result(record, result)
            usage_count = _non_negative_number(
                enriched.get("metadata", {})
                .get("external_impact", {})
                .get("article_usage_count")
            )
            summary["matched_count"] += 1
            if usage_count and usage_count > 0.0:
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


def _fetch_plos_result(client: Any, doi: str) -> dict[str, Any] | None:
    response = client.get(
        PLOS_SEARCH_URL,
        params={
            "q": f'id:"{doi}"',
            "fl": (
                "id,title,title_display,journal,publication_date,"
                "counter_total_all,counter_total_month"
            ),
            "rows": 1,
            "wt": "json",
        },
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    response_body = payload.get("response")
    if not isinstance(response_body, dict):
        return None
    docs = response_body.get("docs")
    if not isinstance(docs, list) or not docs:
        return None
    first = docs[0]
    return first if isinstance(first, dict) else None


def _merge_plos_result(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    title = _title(result.get("title_display") or result.get("title"))
    if title:
        metadata["plos_title"] = title
    journal = _text(result.get("journal"))
    if journal:
        metadata["plos_journal"] = journal
    publication_date = _text(result.get("publication_date"))
    if publication_date:
        metadata["plos_publication_date"] = publication_date

    external_impact = dict(metadata.get("external_impact") or {})
    total_all = _non_negative_number(result.get("counter_total_all"))
    if total_all is not None:
        external_impact["plos_counter_total_all"] = total_all
        external_impact["plos_usage_count"] = total_all
        external_impact["article_usage_count"] = total_all
    total_month = _non_negative_number(result.get("counter_total_month"))
    if total_month is not None:
        external_impact["plos_counter_total_month"] = total_month
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("plos_metrics")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["plos_metrics_enriched"] = True
    record["metadata"] = metadata
    return record


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _title(value: Any) -> str:
    if isinstance(value, list) and value:
        return _text(value[0])
    return _text(value)


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
