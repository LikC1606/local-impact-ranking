from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def enrich_papers_with_openalex(
    input_path: str | Path,
    output_path: str | Path,
    *,
    email: str | None = None,
    timeout: float = 30.0,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    destination = Path(output_path)
    resumed_rows = resume_rows_by_paper_id(destination) if resume else {}
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows = []
    summary = {
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "title_query_count": 0,
        "title_match_count": 0,
        "title_rejected_count": 0,
        "resume": resume,
        "resumed_count": 0,
        "source": "openalex",
    }
    try:
        for paper in papers:
            resumed_row = resumed_rows.get(paper.paper_id)
            if resumed_row is not None:
                rows.append(resumed_row)
                summary["resumed_count"] += 1
                continue
            record = paper.model_dump(mode="json", exclude_none=True)
            query = _openalex_query_for_paper(paper)
            if query is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            summary["queryable_count"] += 1
            if query["kind"] == "title":
                summary["title_query_count"] += 1
            work = _fetch_openalex_work(http_client, query, email=email)
            if work is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                if query["kind"] == "title":
                    summary["title_rejected_count"] += 1
                continue
            rows.append(_merge_openalex_work(record, work))
            summary["matched_count"] += 1
            if query["kind"] == "title":
                summary["title_match_count"] += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    write_jsonl_atomic(destination, rows)
    summary["output_path"] = str(destination)
    return summary


def _openalex_query_for_paper(paper: Paper) -> dict[str, str] | None:
    openalex_id = _paper_identifier(paper, "openalex")
    if openalex_id:
        return {"kind": "openalex_id", "value": openalex_id}
    doi = paper_doi(paper)
    if doi:
        return {"kind": "doi", "value": doi}
    if _is_title_queryable(paper):
        return {
            "kind": "title",
            "value": paper.title,
            "search": _title_search_text(paper.title),
        }
    return None


def paper_doi(paper: Paper) -> str | None:
    doi = _paper_identifier(paper, "doi")
    if doi:
        return doi
    return _paper_extra(paper, "doi")


def _paper_identifier(paper: Paper, key: str) -> str | None:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get(key):
        return str(external_ids[key])
    value = _paper_extra(paper, key)
    return str(value) if value else None


def _paper_extra(paper: Paper, key: str) -> str | None:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    value = extra.get(key)
    return str(value) if value else None


def _fetch_openalex_work(
    client: Any,
    query: dict[str, str],
    *,
    email: str | None,
) -> dict[str, Any] | None:
    params: dict[str, Any] = {
        "select": (
            "id,doi,display_name,publication_year,primary_topic,cited_by_count,counts_by_year,"
            "citation_normalized_percentile"
        )
    }
    if email:
        params["mailto"] = email
    if query["kind"] == "openalex_id":
        url = f"{OPENALEX_WORKS_URL}/{_normalize_openalex_work_id(query['value'])}"
    elif query["kind"] == "doi":
        url = f"{OPENALEX_WORKS_URL}/doi:{quote(_normalize_doi(query['value']), safe='')}"
    else:
        url = OPENALEX_WORKS_URL
        params["filter"] = f"title.search:{query.get('search') or query['value']}"
        params["per-page"] = 5
    response = client.get(url, params=params)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    if query["kind"] == "title":
        return _best_title_match(payload, query["value"])
    return payload


def _merge_openalex_work(record: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    if work.get("id"):
        external_ids["openalex"] = str(work["id"])
    if work.get("doi"):
        external_ids["doi"] = str(work["doi"])
    metadata["external_ids"] = external_ids
    year = _publication_year(work.get("publication_year"))
    if year is not None and not record.get("year"):
        record["year"] = year
    field = _primary_field(work.get("primary_topic"))
    if field is not None and not record.get("field"):
        record["field"] = field
        metadata["field_source"] = "openalex_primary_topic"
    citation_count = _non_negative_number(work.get("cited_by_count"))
    if citation_count is not None:
        metadata["citation_count"] = citation_count
        record["citation_count"] = citation_count
    counts_by_year = _counts_by_year(work.get("counts_by_year"))
    if counts_by_year:
        metadata["citation_counts_by_year"] = counts_by_year
    external_impact = dict(metadata.get("external_impact") or {})
    percentile = _citation_percentile(work.get("citation_normalized_percentile"))
    if percentile is not None:
        external_impact["field_year_percentile"] = percentile
    external_impact["openalex_cited_by_count"] = citation_count
    metadata["external_impact"] = external_impact
    metadata["openalex_enriched"] = True
    record["metadata"] = metadata
    return record


def _normalize_openalex_work_id(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _is_title_queryable(paper: Paper) -> bool:
    if _is_supplement_record(paper):
        return False
    title = _normalize_title(paper.title)
    if len(title) < 20:
        return False
    if re.match(r"^(fig(?:ure)?\s*s?\d+|figs\d+|table\s*s?\d+)\b", title):
        return False
    if title in {"untitled paper", "supplementary material", "supporting information"}:
        return False
    if len(_title_search_text(paper.title)) < 20:
        return False
    return True


def _is_supplement_record(paper: Paper) -> bool:
    source_kind = str(paper.metadata.get("source_kind", "") or "")
    if source_kind == "supplement_or_extracted_supplement":
        return True
    return _normalize_title(paper.title).startswith(
        (
            "supplementary ",
            "supplemental ",
            "supporting information",
            "supplementary online content",
            "supplemental online content",
        )
    )


def _best_title_match(payload: dict[str, Any], target_title: str) -> dict[str, Any] | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_title = item.get("display_name") or item.get("title")
        if not candidate_title:
            continue
        score = _title_similarity(target_title, str(candidate_title))
        if best is None or score > best[0]:
            best = (score, item)
    if best is None or best[0] < 0.92:
        return None
    return best[1]


def _title_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_title(left)
    normalized_right = _normalize_title(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if len(shorter) >= 30 and shorter in longer:
        return 0.96
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_search_text(title: str) -> str:
    normalized = _normalize_title(title)
    if len(normalized) <= 180:
        return normalized
    truncated = normalized[:180].rsplit(" ", 1)[0]
    return truncated or normalized[:180]


def _counts_by_year(value: Any) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    parsed: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        count = _non_negative_number(item.get("cited_by_count"))
        if year is None or count is None:
            continue
        parsed[str(year)] = count
    return parsed


def _citation_percentile(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    percentile = _non_negative_number(value.get("value"))
    if percentile is None:
        return None
    if percentile <= 1.0:
        return percentile
    return min(percentile / 100.0, 1.0)


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


def _primary_field(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("field", "subfield", "domain"):
        item = value.get(key)
        if isinstance(item, dict) and item.get("display_name"):
            return str(item["display_name"])
    return None


def _non_negative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
