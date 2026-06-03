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


NCBI_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
NCBI_ELINK_BATCH_SIZE = 5


def enrich_papers_with_ncbi_citations(
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
        "source": "ncbi_elink",
        "source_url": NCBI_ELINK_URL,
        "paper_count": len(papers),
        "matched_count": 0,
        "skipped_count": 0,
        "pubmed_query_count": 0,
        "pubmed_nonzero_count": 0,
        "pmc_query_count": 0,
        "pmc_nonzero_count": 0,
        "query_error_count": 0,
        "last_error": "",
        "resume": resume,
        "resumed_count": 0,
    }
    try:
        prepared = [
            (
                paper,
                paper.model_dump(mode="json", exclude_none=True),
                _paper_identifier(paper, "pmid")
                or _paper_identifier(paper, "pubmed_id"),
                _paper_identifier(paper, "pmcid") or _paper_identifier(paper, "pmc"),
            )
            for paper in papers
        ]
        unique_pmids = sorted(
            {
                pmid
                for paper, _, pmid, _ in prepared
                if paper.paper_id not in resumed_rows and pmid
            }
        )
        unique_pmcids = sorted(
            {
                normalized
                for paper, _, _, pmcid in prepared
                if paper.paper_id not in resumed_rows
                and pmcid
                and (normalized := _normalize_pmcid_for_elink(pmcid))
            }
        )
        pubmed_counts: dict[str, float] = {}
        pmc_counts: dict[str, float] = {}
        if unique_pmids:
            summary["pubmed_query_count"] += 1
            try:
                pubmed_counts = _fetch_elink_counts(
                    http_client,
                    dbfrom="pubmed",
                    linkname="pubmed_pubmed_citedin",
                    identifiers=unique_pmids,
                )
            except Exception as exc:
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)
        if unique_pmcids:
            summary["pmc_query_count"] += 1
            try:
                pmc_counts = _fetch_elink_counts(
                    http_client,
                    dbfrom="pmc",
                    linkname="pmc_pmc_citedby",
                    identifiers=unique_pmcids,
                )
            except Exception as exc:
                summary["query_error_count"] += 1
                summary["last_error"] = str(exc)

        for paper, record, pmid, pmcid in prepared:
            resumed_row = resumed_rows.get(paper.paper_id)
            if resumed_row is not None:
                rows.append(resumed_row)
                summary["resumed_count"] += 1
                continue
            if not pmid and not pmcid:
                rows.append(record)
                summary["skipped_count"] += 1
                continue

            pubmed_count = pubmed_counts.get(pmid) if pmid else None
            pmc_identifier = _normalize_pmcid_for_elink(pmcid) if pmcid else None
            pmc_count = pmc_counts.get(pmc_identifier) if pmc_identifier else None
            if pubmed_count and pubmed_count > 0:
                summary["pubmed_nonzero_count"] += 1
            if pmc_count and pmc_count > 0:
                summary["pmc_nonzero_count"] += 1

            if pubmed_count is None and pmc_count is None:
                rows.append(record)
                continue
            rows.append(
                _merge_ncbi_counts(
                    record,
                    pubmed_count=pubmed_count,
                    pmc_count=pmc_count,
                )
            )
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


def _fetch_elink_counts(
    client: Any,
    *,
    dbfrom: str,
    linkname: str,
    identifiers: list[str],
) -> dict[str, float]:
    if not identifiers:
        return {}
    counts = {identifier: 0.0 for identifier in identifiers}
    errors: list[Exception] = []
    successful_chunks = 0
    for chunk in _chunks(identifiers, NCBI_ELINK_BATCH_SIZE):
        try:
            response = client.get(
                NCBI_ELINK_URL,
                params={
                    "dbfrom": dbfrom,
                    "linkname": linkname,
                    "id": chunk,
                    "retmode": "json",
                },
            )
            if response.status_code == 404:
                successful_chunks += 1
                continue
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            errors.append(exc)
            continue
        if not isinstance(payload, dict):
            continue
        successful_chunks += 1
        counts.update(_elink_counts_by_id(payload, linkname, chunk))
    if successful_chunks == 0 and errors:
        raise errors[-1]
    return counts


def _chunks(values: list[str], size: int) -> list[list[str]]:
    size = max(int(size), 1)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _elink_counts_by_id(
    payload: dict[str, Any],
    linkname: str,
    identifiers: list[str],
) -> dict[str, float]:
    counts = {identifier: 0.0 for identifier in identifiers}
    linksets = payload.get("linksets", [])
    if not isinstance(linksets, list):
        return counts
    for index, linkset in enumerate(linksets):
        if not isinstance(linkset, dict):
            continue
        ids = linkset.get("ids")
        identifier = None
        if isinstance(ids, list) and len(ids) == 1:
            identifier = str(ids[0])
        elif len(identifiers) == 1:
            identifier = identifiers[0]
        elif len(linksets) == len(identifiers):
            identifier = identifiers[index]
        if identifier is None:
            continue
        counts[identifier] = float(len(set(_elink_links_from_linkset(linkset, linkname))))
    return counts


def _elink_links_from_linkset(linkset: dict[str, Any], linkname: str) -> list[str]:
    links: list[str] = []
    linkset_dbs = linkset.get("linksetdbs", [])
    if not isinstance(linkset_dbs, list):
        return links
    for linkset_db in linkset_dbs:
        if not isinstance(linkset_db, dict):
            continue
        if linkset_db.get("linkname") != linkname:
            continue
        raw_links = linkset_db.get("links", [])
        if isinstance(raw_links, list):
            links.extend(str(link) for link in raw_links if link is not None)
    return links


def _merge_ncbi_counts(
    record: dict[str, Any],
    *,
    pubmed_count: float | None,
    pmc_count: float | None,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_impact = dict(metadata.get("external_impact") or {})
    if pubmed_count is not None:
        external_impact["ncbi_pubmed_citedin_count"] = pubmed_count
    if pmc_count is not None:
        external_impact["ncbi_pmc_cited_by_count"] = pmc_count
    if pubmed_count is not None:
        existing = _non_negative_number(record.get("citation_count"))
        if existing is None or pubmed_count > existing:
            record["citation_count"] = pubmed_count
            metadata["citation_count"] = pubmed_count
    metadata["external_impact"] = external_impact
    metadata["ncbi_citation_enriched"] = True
    sources = set(metadata.get("external_evidence_sources") or [])
    sources.add("ncbi")
    metadata["external_evidence_sources"] = sorted(sources)
    record["metadata"] = metadata
    return record


def _paper_identifier(paper: Paper, key: str) -> str | None:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get(key):
        return str(external_ids[key])
    metadata = paper.metadata if isinstance(paper.metadata, dict) else {}
    if metadata.get(key):
        return str(metadata[key])
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    value = extra.get(key)
    return str(value) if value else None


def _normalize_pmcid_for_elink(value: str) -> str | None:
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else None


def _non_negative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
