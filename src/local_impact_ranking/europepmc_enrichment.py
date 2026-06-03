from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def enrich_papers_with_europepmc(
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
        "source": "europepmc",
        "source_url": EUROPEPMC_SEARCH_URL,
        "paper_count": len(papers),
        "matched_count": 0,
        "unmatched_count": 0,
        "queryable_count": 0,
        "doi_query_count": 0,
        "pmid_query_count": 0,
        "pmcid_query_count": 0,
        "title_query_count": 0,
        "title_match_count": 0,
        "title_rejected_count": 0,
        "pmid_replaced_count": 0,
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
            queries = _europepmc_queries_for_paper(paper)
            if not queries:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            summary["queryable_count"] += 1
            match: dict[str, Any] | None = None
            match_kind = ""
            for query in queries:
                summary[f"{query['kind']}_query_count"] += 1
                try:
                    candidate = _fetch_europepmc_result(http_client, query)
                except Exception as exc:
                    summary["query_error_count"] += 1
                    summary["last_error"] = str(exc)
                    continue
                if candidate is None:
                    if query["kind"] == "title":
                        summary["title_rejected_count"] += 1
                    continue
                match = candidate
                match_kind = query["kind"]
                break
            if match is None:
                rows.append(record)
                summary["unmatched_count"] += 1
                continue
            merged, replaced_pmid = _merge_europepmc_result(
                record,
                match,
                match_method=match_kind,
            )
            rows.append(merged)
            summary["matched_count"] += 1
            if match_kind == "title":
                summary["title_match_count"] += 1
            if replaced_pmid:
                summary["pmid_replaced_count"] += 1
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


def _europepmc_queries_for_paper(paper: Paper) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    doi = paper_doi(paper)
    if doi:
        queries.append({"kind": "doi", "search": f'DOI:"{_normalize_doi(doi)}"'})
    pmcid = _paper_identifier(paper, "pmcid") or _paper_identifier(paper, "pmc")
    if pmcid:
        queries.append({"kind": "pmcid", "search": f"PMCID:{pmcid}"})
    pmid = (
        _paper_identifier(paper, "pmid")
        or _paper_identifier(paper, "pubmed")
        or _paper_identifier(paper, "pubmed_id")
    )
    if pmid:
        queries.append({"kind": "pmid", "search": f"EXT_ID:{pmid} AND SRC:MED"})
    if _is_title_queryable(paper):
        queries.append({"kind": "title", "search": f'TITLE:"{paper.title}"', "title": paper.title})
    return queries


def _fetch_europepmc_result(
    client: Any,
    query: dict[str, str],
) -> dict[str, Any] | None:
    response = client.get(
        EUROPEPMC_SEARCH_URL,
        params={
            "query": query["search"],
            "format": "json",
            "pageSize": 1,
            "resultType": "core",
        },
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    results = payload.get("resultList", {}).get("result", [])
    if not isinstance(results, list) or not results:
        return None
    candidate = results[0]
    if not isinstance(candidate, dict):
        return None
    if query["kind"] == "title" and not _title_is_match(
        query.get("title", ""),
        str(candidate.get("title") or ""),
    ):
        return None
    return candidate


def _merge_europepmc_result(
    record: dict[str, Any],
    result: dict[str, Any],
    *,
    match_method: str,
) -> tuple[dict[str, Any], bool]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    source = _text(result.get("source"))
    result_id = _text(result.get("id"))
    if source and result_id:
        external_ids["europepmc"] = f"{source}:{result_id}"
    elif result_id:
        external_ids["europepmc"] = result_id

    replaced_pmid = False
    pmid = _text(result.get("pmid"))
    old_pmid = _text(external_ids.get("pmid"))
    if pmid:
        if old_pmid and old_pmid != pmid:
            metadata["europepmc_replaced_pmid"] = {"old": old_pmid, "new": pmid}
            replaced_pmid = True
        external_ids["pmid"] = pmid
    pmcid = _text(result.get("pmcid"))
    if pmcid:
        external_ids["pmcid"] = pmcid
    doi = _text(result.get("doi"))
    if doi:
        external_ids["doi"] = doi
        if not record.get("doi"):
            record["doi"] = doi
    if external_ids:
        metadata["external_ids"] = external_ids

    year = _publication_year(result.get("pubYear"))
    if year is not None and not record.get("year"):
        record["year"] = year
    journal = _text(result.get("journalTitle"))
    if journal and not record.get("venue"):
        record["venue"] = journal

    external_impact = dict(metadata.get("external_impact") or {})
    cited_by_count = _non_negative_number(result.get("citedByCount"))
    if cited_by_count is not None:
        external_impact["europepmc_cited_by_count"] = cited_by_count
        external_impact["europepmc_citation_count"] = cited_by_count
        existing = _non_negative_number(record.get("citation_count"))
        if existing is None or cited_by_count > existing:
            record["citation_count"] = cited_by_count
            metadata["citation_count"] = cited_by_count

    if external_impact:
        metadata["external_impact"] = external_impact
    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("europepmc")
    metadata["external_evidence_sources"] = sorted(sources)
    metadata["europepmc_enriched"] = True
    metadata["europepmc_match_method"] = match_method
    if result.get("title"):
        metadata["europepmc_title"] = str(result["title"])
    record["metadata"] = metadata
    return record, replaced_pmid


def _paper_identifier(paper: Paper, key: str) -> str | None:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get(key):
        return str(external_ids[key])
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    value = extra.get(key)
    return str(value) if value else None


def _normalize_doi(value: str) -> str:
    return normalize_doi(value)


def _is_title_queryable(paper: Paper) -> bool:
    source_kind = str(paper.metadata.get("source_kind", "") or "")
    title = _normalize_title(paper.title)
    if source_kind == "supplement_or_extracted_supplement":
        return False
    if len(title) < 20:
        return False
    if re.match(r"^(fig(?:ure)?\s*s?\d+|figs\d+|table\s*s?\d+)\b", title):
        return False
    return title not in {
        "untitled paper",
        "supplementary material",
        "supporting information",
    }


def _title_is_match(left: str, right: str) -> bool:
    normalized_left = _normalize_title(left)
    normalized_right = _normalize_title(right)
    if normalized_left == normalized_right:
        return True
    if not normalized_left or not normalized_right:
        return False
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if len(shorter) >= 30 and shorter in longer:
        return True
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.92


def _normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
