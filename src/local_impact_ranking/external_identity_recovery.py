from __future__ import annotations

import html
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
from local_impact_ranking.openalex_enrichment import (
    _fetch_openalex_work,
    _merge_openalex_work,
    _title_search_text,
)


IDENTITY_MATCH_THRESHOLD = 0.70
RECOVERY_TITLE_MATCH_THRESHOLD = 0.92

EXTERNAL_TITLE_KEYS = (
    "europepmc_title",
    "crossref_title",
    "datacite_title",
    "openaire_title",
    "wikidata_title",
    "plos_title",
    "openalex_title",
    "openalex_display_name",
    "semantic_scholar_title",
)

IDENTITY_EXTERNAL_ID_KEYS = {
    "doi",
    "openalex",
    "semantic_scholar",
    "semantic_scholar_corpus_id",
    "pmid",
    "pmcid",
    "europepmc",
}

IDENTITY_METADATA_KEYS = {
    "citation_count",
    "citation_counts_by_year",
    "field_source",
    "europepmc_title",
    "europepmc_enriched",
    "europepmc_match_method",
    "europepmc_replaced_pmid",
    "icite_enriched",
    "ncbi_citation_enriched",
    "opencitations_enriched",
    "semantic_scholar_enriched",
    "crossref_cited_by_enriched",
    "crossref_cited_by_error",
    "crossref_publisher",
    "datacite_enriched",
    "datacite_title",
    "datacite_publisher",
    "datacite_resource_type",
    "openaire_enriched",
    "openaire_id",
    "openaire_title",
    "openaire_type",
    "openaire_publisher",
    "wikidata_enriched",
    "wikidata_entity",
    "wikidata_title",
    "datacite_events_enriched",
    "datacite_events_obj_id",
    "crossref_data_citations_enriched",
    "plos_metrics_enriched",
    "plos_title",
    "plos_journal",
    "plos_publication_date",
    "openalex_enriched",
    "openalex_title",
    "openalex_display_name",
    "semantic_scholar_title",
}

IDENTITY_IMPACT_KEYS = {
    "field_year_percentile",
    "openalex_cited_by_count",
    "openalex_citation_count",
    "semantic_scholar_citation_count",
    "semantic_scholar_influential_citation_count",
    "s2_citation_count",
    "pubmed_citation_count",
    "pmid_citation_count",
    "pmc_citation_count",
    "icite_citation_count",
    "nih_icite_citation_count",
    "icite_total_citations",
    "nih_icite_citations",
    "citations_per_year",
    "relative_citation_ratio",
    "nih_percentile",
    "expected_citation_rate",
    "field_citation_rate",
    "apt",
    "human",
    "animal",
    "molecular_cellular",
    "is_research_article",
    "icite_provisional",
    "cited_by_clinical_article",
    "opencitations_cited_by_count",
    "opencitations_citation_count",
    "opencitations_reference_count",
    "crossref_cited_by_count",
    "crossref_is_referenced_by_count",
    "is_referenced_by_count",
    "crossref_data_citation_count",
    "crossref_data_citations_total_results",
    "europepmc_cited_by_count",
    "europepmc_citation_count",
    "ncbi_pubmed_citedin_count",
    "ncbi_pubmed_cited_by_count",
    "ncbi_pmc_cited_by_count",
    "datacite_citation_count",
    "datacite_cited_by_count",
    "datacite_view_count",
    "datacite_download_count",
    "datacite_events_cited_by_count",
    "datacite_events_citation_count",
    "datacite_events_total_event_count",
    "datacite_events_references_count",
    "datacite_events_cites_count",
    "openaire_citation_count",
    "openaire_cited_by_count",
    "openaire_influence",
    "openaire_popularity",
    "openaire_impulse",
    "openaire_citation_class",
    "wikidata_cited_by_count",
    "wikidata_citation_count",
    "plos_counter_total_all",
    "plos_counter_total_month",
    "plos_usage_count",
    "article_usage_count",
}


def recover_external_identities(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    min_title_similarity: float = RECOVERY_TITLE_MATCH_THRESHOLD,
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
        "source": "external_identity_recovery",
        "paper_count": len(papers),
        "mismatch_count": 0,
        "title_query_count": 0,
        "recovered_count": 0,
        "unresolved_count": 0,
        "preserved_count": 0,
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
            mismatch = _identity_mismatch(paper)
            if mismatch is None:
                rows.append(record)
                summary["preserved_count"] += 1
                continue
            summary["mismatch_count"] += 1
            summary["title_query_count"] += 1
            try:
                work = _recover_openalex_work(
                    http_client,
                    paper,
                    min_title_similarity=min_title_similarity,
                )
            except Exception as exc:
                work = None
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)

            quarantined = _quarantine_identity(record, mismatch)
            if work is None:
                rows.append(_mark_unresolved(quarantined, mismatch))
                summary["unresolved_count"] += 1
                continue

            rows.append(_merge_recovered_work(quarantined, work, mismatch))
            summary["recovered_count"] += 1
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


def _identity_mismatch(paper: Paper) -> dict[str, Any] | None:
    external_title = _external_title(paper.metadata)
    if not external_title:
        return None
    similarity = _title_similarity(paper.title, external_title)
    if similarity >= IDENTITY_MATCH_THRESHOLD:
        return None
    return {
        "status": "mismatch",
        "local_title": paper.title,
        "external_title": external_title,
        "title_similarity": round(similarity, 6),
        "reason": "External title is inconsistent with the local paper title.",
    }


def _recover_openalex_work(
    client: Any,
    paper: Paper,
    *,
    min_title_similarity: float,
) -> dict[str, Any] | None:
    work = _fetch_openalex_work(
        client,
        {
            "kind": "title",
            "value": paper.title,
            "search": _title_search_text(paper.title),
        },
        email=None,
    )
    if work is None:
        return None
    candidate_title = str(work.get("display_name") or work.get("title") or "")
    if _title_similarity(paper.title, candidate_title) < min_title_similarity:
        return None
    return work


def _quarantine_identity(record: dict[str, Any], mismatch: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    stale_field_from_openalex = metadata.get("field_source") == "openalex_primary_topic"
    old_external_ids = dict(metadata.get("external_ids") or {})
    old_external_impact = dict(metadata.get("external_impact") or {})
    rejected = list(metadata.get("rejected_external_ids") or [])
    if old_external_ids or old_external_impact:
        rejected.append(
            {
                "reason": "external_identity_mismatch",
                "identity_mismatch": mismatch,
                "external_ids": old_external_ids,
                "external_impact": old_external_impact,
            }
        )
    metadata["rejected_external_ids"] = rejected
    metadata["external_ids"] = {
        key: value
        for key, value in old_external_ids.items()
        if key not in IDENTITY_EXTERNAL_ID_KEYS
    }
    metadata["external_impact"] = {
        key: value
        for key, value in old_external_impact.items()
        if key not in IDENTITY_IMPACT_KEYS
    }
    for key in IDENTITY_METADATA_KEYS:
        metadata.pop(key, None)
    record["metadata"] = metadata
    record.pop("citation_count", None)
    if stale_field_from_openalex:
        record.pop("field", None)
    return record


def _merge_recovered_work(
    record: dict[str, Any],
    work: dict[str, Any],
    mismatch: dict[str, Any],
) -> dict[str, Any]:
    merged = _merge_openalex_work(record, work)
    metadata = dict(merged.get("metadata") or {})
    candidate_title = str(work.get("display_name") or work.get("title") or "")
    metadata["openalex_title"] = candidate_title
    metadata["external_identity_recovery"] = {
        "status": "recovered",
        "source": "openalex_title_search",
        "previous_identity": mismatch,
        "recovered_title": candidate_title,
        "title_similarity": round(_title_similarity(merged.get("title", ""), candidate_title), 6),
    }
    merged["metadata"] = metadata
    return merged


def _mark_unresolved(record: dict[str, Any], mismatch: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    metadata["external_identity_recovery"] = {
        "status": "unresolved",
        "source": "openalex_title_search",
        "previous_identity": mismatch,
        "reason": "No high-confidence title match was found.",
    }
    record["metadata"] = metadata
    return record


def _external_title(metadata: dict[str, Any]) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in EXTERNAL_TITLE_KEYS:
        value = metadata.get(key)
        if value:
            return _clean_title(str(value))
    return None


def _clean_title(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _normalize_title(value: str) -> str:
    text = html.unescape(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
