from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.current_impact_credibility import (
    write_current_impact_credibility,
)
from local_impact_ranking.enrichment_resume import (
    resume_rows_by_paper_id,
    write_jsonl_atomic,
)
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper


ICITE_API_URL = "https://icite.od.nih.gov/api/pubs"

ICITE_CURRENT_METRIC_FIELDS = [
    "paper_id",
    "pmid",
    "doi",
    "title",
    "year",
    "journal",
    "rcr",
    "citations_per_year",
    "apt",
    "nih_percentile",
    "icite_citation_count",
    "expected_citations_per_year",
    "field_citation_rate",
    "human",
    "animal",
    "molecular_cellular",
    "is_research_article",
    "provisional",
    "cited_by_clinical_article",
    "benchmark_source",
]


def build_icite_current_metrics(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    batch_size: int = 200,
    timeout: float = 30.0,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    paper_pmids = {paper.paper_id: _paper_pmid(paper) for paper in papers}
    unique_pmids = sorted({pmid for pmid in paper_pmids.values() if pmid})
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        payload_by_pmid = _fetch_icite_payloads(
            http_client,
            unique_pmids,
            batch_size=batch_size,
        )
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    rows = [
        _icite_label_row(paper, paper_pmids[paper.paper_id], payload_by_pmid)
        for paper in papers
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ICITE_CURRENT_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(sanitize_csv_row(row) for row in rows)

    matched_count = sum(1 for row in rows if row["benchmark_source"] == "nih_icite")
    label_coverage = {
        column: sum(1 for row in rows if row.get(column) != "")
        for column in [
            "rcr",
            "citations_per_year",
            "apt",
            "nih_percentile",
            "icite_citation_count",
        ]
    }
    summary = {
        "source": "nih_icite",
        "source_url": ICITE_API_URL,
        "paper_count": len(papers),
        "pmid_query_count": len(unique_pmids),
        "matched_count": matched_count,
        "unmatched_count": len(papers) - matched_count,
        "label_coverage": label_coverage,
        "output_path": str(destination),
        "primary_label_column": "rcr",
        "label_columns": [
            "rcr",
            "citations_per_year",
            "apt",
            "nih_percentile",
            "icite_citation_count",
        ],
    }
    if summary_output is not None:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def enrich_papers_with_icite(
    input_path: str | Path,
    output_path: str | Path,
    *,
    summary_output: str | Path | None = None,
    batch_size: int = 200,
    timeout: float = 30.0,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    destination = Path(output_path)
    resumed_rows = resume_rows_by_paper_id(destination) if resume else {}
    paper_pmids = {paper.paper_id: _paper_pmid(paper) for paper in papers}
    unique_pmids = sorted(
        {
            pmid
            for paper in papers
            if paper.paper_id not in resumed_rows
            for pmid in [paper_pmids[paper.paper_id]]
            if pmid
        }
    )
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        payload_by_pmid = _fetch_icite_payloads(
            http_client,
            unique_pmids,
            batch_size=batch_size,
        )
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    rows: list[dict[str, Any]] = []
    matched_count = 0
    unmatched_count = 0
    resumed_count = 0
    for paper in papers:
        resumed_row = resumed_rows.get(paper.paper_id)
        if resumed_row is not None:
            rows.append(resumed_row)
            resumed_count += 1
            continue
        record = paper.model_dump(mode="json", exclude_none=True)
        pmid = paper_pmids[paper.paper_id]
        payload = payload_by_pmid.get(pmid or "")
        if payload:
            rows.append(_merge_icite_payload(record, pmid or "", payload))
            matched_count += 1
        else:
            rows.append(record)
            unmatched_count += 1

    write_jsonl_atomic(destination, rows)

    summary = {
        "source": "nih_icite",
        "source_url": ICITE_API_URL,
        "paper_count": len(papers),
        "pmid_query_count": len(unique_pmids),
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "output_path": str(destination),
        "resume": resume,
        "resumed_count": resumed_count,
    }
    if summary_output is not None:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_icite_current_benchmark(
    papers_path: str | Path,
    scores_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 200,
    timeout: float = 30.0,
    top_fraction: float = 0.1,
    bootstrap_samples: int = 100,
    permutation_samples: int = 100,
    random_seed: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels_path = destination / "icite_current_metrics.csv"
    summary_path = destination / "icite_current_metrics_summary.json"
    summary = build_icite_current_metrics(
        papers_path,
        labels_path,
        summary_output=summary_path,
        batch_size=batch_size,
        timeout=timeout,
        client=client,
    )
    label_columns = _available_label_columns(labels_path, summary["label_columns"])
    primary_label, fallback_reason = _primary_icite_label(label_columns)
    credibility = write_current_impact_credibility(
        scores_path,
        labels_path,
        destination,
        label_columns=label_columns,
        primary_label_column=primary_label,
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    report = {
        "summary": {
            "benchmark_source": "nih_icite",
            "papers_path": str(papers_path),
            "scores_path": str(scores_path),
            "labels_path": str(labels_path),
            "matched_count": summary["matched_count"],
            "paper_count": summary["paper_count"],
            "primary_label_column": primary_label,
            "primary_label_fallback_reason": fallback_reason,
        },
        "icite_metrics": summary,
        "credibility": credibility,
    }
    (destination / "icite_current_benchmark_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _available_label_columns(
    labels_path: Path,
    candidates: list[str],
) -> list[str]:
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    available = []
    for column in candidates:
        if any(_is_numeric(row.get(column)) for row in rows):
            available.append(column)
    if not available:
        raise ValueError("No iCite benchmark label columns contain numeric values.")
    return available


def _primary_icite_label(label_columns: list[str]) -> tuple[str, str]:
    if "rcr" in label_columns:
        return "rcr", ""
    primary = label_columns[0]
    return (
        primary,
        "NIH iCite RCR is unavailable for these papers; using the first available current iCite metric as a disclosed fallback.",
    )


def _fetch_icite_payloads(
    client: Any,
    pmids: list[str],
    *,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    payload_by_pmid: dict[str, dict[str, Any]] = {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        if not batch:
            continue
        response = client.get(ICITE_API_URL, params={"pmids": ",".join(batch)})
        response.raise_for_status()
        payload = response.json()
        for row in payload.get("data", []):
            pmid = _normalize_pmid(row.get("pmid"))
            if pmid:
                payload_by_pmid[pmid] = row
    return payload_by_pmid


def _icite_label_row(
    paper: Paper,
    pmid: str | None,
    payload_by_pmid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = payload_by_pmid.get(pmid or "", {})
    return {
        "paper_id": paper.paper_id,
        "pmid": pmid or "",
        "doi": _text(payload.get("doi")),
        "title": _text(payload.get("title")),
        "year": _value(payload.get("year")),
        "journal": _text(payload.get("journal")),
        "rcr": _value(payload.get("relative_citation_ratio")),
        "citations_per_year": _value(payload.get("citations_per_year")),
        "apt": _value(payload.get("apt")),
        "nih_percentile": _value(payload.get("nih_percentile")),
        "icite_citation_count": _value(payload.get("citation_count")),
        "expected_citations_per_year": _value(payload.get("expected_citations_per_year")),
        "field_citation_rate": _value(payload.get("field_citation_rate")),
        "human": _value(payload.get("human")),
        "animal": _value(payload.get("animal")),
        "molecular_cellular": _value(payload.get("molecular_cellular")),
        "is_research_article": _bool_text(payload.get("is_research_article")),
        "provisional": _bool_text(payload.get("provisional")),
        "cited_by_clinical_article": _bool_text(payload.get("citedByClinicalArticle")),
        "benchmark_source": "nih_icite" if payload else "",
    }


def _merge_icite_payload(
    record: dict[str, Any],
    pmid: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    external_ids["pmid"] = pmid
    if payload.get("doi"):
        external_ids["doi"] = _text(payload.get("doi"))
    metadata["external_ids"] = external_ids

    year = _numeric_or_none(payload.get("year"))
    if year is not None and not record.get("year"):
        record["year"] = int(year)
    if payload.get("journal") and not record.get("venue"):
        record["venue"] = _text(payload.get("journal"))

    external_impact = dict(metadata.get("external_impact") or {})
    citation_count = _numeric_or_none(payload.get("citation_count"))
    if citation_count is not None:
        record["citation_count"] = citation_count
        metadata["citation_count"] = citation_count
        external_impact["pubmed_citation_count"] = citation_count
        external_impact["icite_citation_count"] = citation_count

    numeric_fields = {
        "relative_citation_ratio": "relative_citation_ratio",
        "citations_per_year": "citations_per_year",
        "nih_percentile": "nih_percentile",
        "apt": "apt",
        "human": "human",
        "animal": "animal",
        "molecular_cellular": "molecular_cellular",
        "expected_citations_per_year": "expected_citation_rate",
        "field_citation_rate": "field_citation_rate",
    }
    for source_key, target_key in numeric_fields.items():
        value = _numeric_or_none(payload.get(source_key))
        if value is not None:
            external_impact[target_key] = value

    if payload.get("is_research_article") is not None:
        external_impact["is_research_article"] = bool(payload.get("is_research_article"))
    if payload.get("provisional") is not None:
        external_impact["icite_provisional"] = bool(payload.get("provisional"))
    if payload.get("citedByClinicalArticle") is not None:
        external_impact["cited_by_clinical_article"] = bool(
            payload.get("citedByClinicalArticle")
        )

    metadata["external_impact"] = external_impact
    metadata["icite_enriched"] = True
    record["metadata"] = metadata
    return record


def _paper_pmid(paper: Paper) -> str | None:
    for value in _candidate_pmid_values(paper):
        pmid = _normalize_pmid(value)
        if pmid:
            return pmid
    return None


def _candidate_pmid_values(paper: Paper) -> list[Any]:
    values: list[Any] = []
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    metadata = paper.metadata if isinstance(paper.metadata, dict) else {}
    external_ids = metadata.get("external_ids")
    if isinstance(external_ids, dict):
        values.extend(
            external_ids.get(key)
            for key in ("pmid", "pubmed", "pubmed_id", "PMID")
        )
    values.extend(metadata.get(key) for key in ("pmid", "pubmed_id", "PMID"))
    values.extend(extra.get(key) for key in ("pmid", "pubmed_id", "PMID"))
    return values


def _normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if match is None:
        return None
    return match.group(1)


def _value(value: Any) -> Any:
    return "" if value is None else value


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _is_numeric(value: Any) -> bool:
    if value in {None, ""}:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
