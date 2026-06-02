from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.openalex_enrichment import paper_doi


SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,corpusId,externalIds,title,year,citationCount,"
    "influentialCitationCount,fieldsOfStudy"
)
RATE_LIMITED = object()


def enrich_papers_with_semantic_scholar(
    input_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows = []
    summary = {
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "doi_query_count": 0,
        "semantic_scholar_id_query_count": 0,
        "title_query_count": 0,
        "title_match_count": 0,
        "title_rejected_count": 0,
        "batch_query_count": 0,
        "batch_size": 0,
        "rate_limited_count": 0,
        "rate_limit_stopped_count": 0,
        "source": "semantic_scholar",
    }
    prepared = [
        (paper, paper.model_dump(mode="json", exclude_none=True), _semantic_scholar_query_for_paper(paper))
        for paper in papers
    ]
    batch_indices = [
        index
        for index, (_, _, query) in enumerate(prepared)
        if query is not None and query["kind"] in {"doi", "semantic_scholar"}
    ]
    batch_results: dict[int, dict[str, Any] | None] = {}
    batch_rate_limited = False
    if len(batch_indices) > 1 and hasattr(http_client, "post"):
        batch_queries = [prepared[index][2] for index in batch_indices]
        batch_payload = _fetch_semantic_scholar_batch(
            http_client,
            [query for query in batch_queries if query is not None],
            api_key=api_key,
        )
        summary["batch_query_count"] = 1
        summary["batch_size"] = len(batch_indices)
        if batch_payload is RATE_LIMITED:
            batch_rate_limited = True
        else:
            for index, payload in zip(batch_indices, batch_payload):
                batch_results[index] = payload if isinstance(payload, dict) else None

    rate_limit_reached = False
    try:
        for index, (paper, record, query) in enumerate(prepared):
            if rate_limit_reached:
                rows.append(record)
                summary["unmatched_count"] += 1
                summary["rate_limit_stopped_count"] += 1
                continue
            if query is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            summary["queryable_count"] += 1
            if query["kind"] == "doi":
                summary["doi_query_count"] += 1
            elif query["kind"] == "semantic_scholar":
                summary["semantic_scholar_id_query_count"] += 1
            elif query["kind"] == "title":
                summary["title_query_count"] += 1
            if batch_rate_limited and index in batch_indices:
                paper_payload = RATE_LIMITED
            elif index in batch_results:
                paper_payload = batch_results[index]
            else:
                paper_payload = _fetch_semantic_scholar_paper(
                    http_client,
                    query,
                    api_key=api_key,
                )
            if paper_payload is RATE_LIMITED:
                rows.append(record)
                summary["unmatched_count"] += 1
                summary["rate_limited_count"] += 1
                rate_limit_reached = True
                continue
            if paper_payload is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                if query["kind"] == "title":
                    summary["title_rejected_count"] += 1
                continue
            rows.append(_merge_semantic_scholar_paper(record, paper_payload))
            summary["matched_count"] += 1
            if query["kind"] == "title":
                summary["title_match_count"] += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["output_path"] = str(destination)
    return summary


def _fetch_semantic_scholar_batch(
    client: Any,
    queries: list[dict[str, str]],
    *,
    api_key: str | None,
) -> list[Any] | object:
    headers = {"x-api-key": api_key} if api_key else None
    response = client.post(
        f"{SEMANTIC_SCHOLAR_API_URL}/paper/batch",
        json={"ids": [_semantic_scholar_batch_id(query) for query in queries]},
        params={"fields": S2_FIELDS},
        headers=headers,
    )
    if response.status_code == 429:
        return RATE_LIMITED
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _semantic_scholar_query_for_paper(paper: Paper) -> dict[str, str] | None:
    semantic_scholar_id = _paper_identifier(paper, "semantic_scholar")
    if semantic_scholar_id:
        return {"kind": "semantic_scholar", "value": semantic_scholar_id}
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


def _fetch_semantic_scholar_paper(
    client: Any,
    query: dict[str, str],
    *,
    api_key: str | None,
) -> dict[str, Any] | object | None:
    headers = {"x-api-key": api_key} if api_key else None
    params = {"fields": S2_FIELDS}
    if query["kind"] == "doi":
        identifier = "DOI:" + quote(_normalize_doi(query["value"]), safe="")
        url = f"{SEMANTIC_SCHOLAR_API_URL}/paper/{identifier}"
    elif query["kind"] == "semantic_scholar":
        url = f"{SEMANTIC_SCHOLAR_API_URL}/paper/{quote(query['value'], safe='')}"
    else:
        url = f"{SEMANTIC_SCHOLAR_API_URL}/paper/search/match"
        params["query"] = query.get("search") or query["value"]
    response = client.get(url, params=params, headers=headers)
    if response.status_code == 404:
        return None
    if response.status_code == 429:
        return RATE_LIMITED
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    if query["kind"] == "title" and not _title_is_match(
        query["value"],
        str(payload.get("title") or ""),
    ):
        return None
    return payload


def _semantic_scholar_batch_id(query: dict[str, str]) -> str:
    if query["kind"] == "doi":
        return "DOI:" + _normalize_doi(query["value"])
    return query["value"]


def _merge_semantic_scholar_paper(
    record: dict[str, Any],
    paper_payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    if paper_payload.get("paperId"):
        external_ids["semantic_scholar"] = str(paper_payload["paperId"])
    if paper_payload.get("corpusId") is not None:
        external_ids["semantic_scholar_corpus_id"] = str(paper_payload["corpusId"])
    external_doi = _external_doi(paper_payload.get("externalIds"))
    if external_doi and not (record.get("doi") or external_ids.get("doi")):
        external_ids["doi"] = external_doi
        record["doi"] = external_doi
    if external_ids:
        metadata["external_ids"] = external_ids

    year = _publication_year(paper_payload.get("year"))
    if year is not None and not record.get("year"):
        record["year"] = year
    field = _field_of_study(paper_payload.get("fieldsOfStudy"))
    if field is not None and not record.get("field"):
        record["field"] = field
        metadata["field_source"] = "semantic_scholar_fields_of_study"

    citation_count = _non_negative_number(paper_payload.get("citationCount"))
    if citation_count is not None:
        metadata["semantic_scholar_citation_count"] = citation_count
        existing_count = _non_negative_number(record.get("citation_count"))
        if existing_count is None or (existing_count <= 0.0 and citation_count > 0.0):
            record["citation_count"] = citation_count
            metadata["citation_count"] = citation_count

    external_impact = dict(metadata.get("external_impact") or {})
    if citation_count is not None:
        external_impact["semantic_scholar_citation_count"] = citation_count
    influential = _non_negative_number(paper_payload.get("influentialCitationCount"))
    if influential is not None:
        external_impact["semantic_scholar_influential_citation_count"] = influential
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("semantic_scholar")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["semantic_scholar_enriched"] = True
    record["metadata"] = metadata
    return record


def _paper_identifier(paper: Paper, key: str) -> str | None:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict):
        for candidate in (key, "semantic_scholar_id", "s2_id", "paperId"):
            value = external_ids.get(candidate)
            if value:
                return str(value)
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    for candidate in (key, "semantic_scholar_id", "s2_id", "paperId"):
        value = extra.get(candidate)
        if value:
            return str(value)
    return None


def _normalize_doi(value: str) -> str:
    normalized = value.strip()
    normalized = normalized.removeprefix("https://doi.org/")
    normalized = normalized.removeprefix("http://doi.org/")
    normalized = normalized.removeprefix("doi:")
    return normalized


def _external_doi(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    doi = value.get("DOI") or value.get("doi")
    return str(doi) if doi else None


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


def _field_of_study(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for item in value:
        if item:
            return str(item)
    return None


def _is_title_queryable(paper: Paper) -> bool:
    source_kind = str(paper.metadata.get("source_kind", "") or "")
    title = _normalize_title(paper.title)
    if source_kind == "supplement_or_extracted_supplement":
        return False
    if len(title) < 20:
        return False
    if re.match(r"^(fig(?:ure)?\s*s?\d+|figs\d+|table\s*s?\d+)\b", title):
        return False
    if title in {"untitled paper", "supplementary material", "supporting information"}:
        return False
    return True


def _title_is_match(left: str, right: str) -> bool:
    normalized_left = _normalize_title(left)
    normalized_right = _normalize_title(right)
    if normalized_left == normalized_right:
        return True
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    return len(shorter) >= 30 and shorter in longer


def _title_search_text(title: str) -> str:
    normalized = _normalize_title(title)
    if len(normalized) <= 180:
        return normalized
    truncated = normalized[:180].rsplit(" ", 1)[0]
    return truncated or normalized[:180]


def _normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _non_negative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
