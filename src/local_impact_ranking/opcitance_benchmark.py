from __future__ import annotations

from collections import defaultdict
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
from local_impact_ranking.icite_benchmark import ICITE_API_URL
from local_impact_ranking.impact_fusion import compute_impact_index_v2
from local_impact_ranking.model_impact_index import _write_impact_scores_csv
from local_impact_ranking.models import Paper


OPCITANCE_SCORE_COLUMNS = [
    "final_impact_score",
    "true_citation_score",
    "effective_citation_score",
    "score_100_true_00_effective",
    "score_95_true_05_effective",
    "score_90_true_10_effective",
    "score_80_true_20_effective",
    "score_70_true_30_effective",
    "score_50_true_50_effective",
    "score_30_true_70_effective",
]

OPCITANCE_LABEL_COLUMNS = [
    "rcr",
    "citations_per_year",
    "apt",
    "nih_percentile",
    "icite_citation_count",
]


def build_opcitance_icite_current_benchmark(
    opcitance_path: str | Path,
    output_dir: str | Path,
    *,
    max_papers: int = 1000,
    max_rows: int | None = None,
    candidate_limit: int | None = None,
    batch_size: int = 200,
    timeout: float = 30.0,
    top_fraction: float = 0.1,
    bootstrap_samples: int = 100,
    permutation_samples: int = 100,
    random_seed: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build an OpCitance-derived biomedical current-impact benchmark.

    OpCitance supplies citation contexts, while NIH iCite supplies public current-impact
    labels. The generated paper JSONL is intentionally local: effective citations are
    only those citation-context edges visible inside the selected OpCitance subgraph.
    """
    if max_papers <= 0:
        raise ValueError("max_papers must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    source = Path(opcitance_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    graph = _read_opcitance_graph(source, max_rows=max_rows)
    if not graph["edges"]:
        raise ValueError("No PMID-to-PMID citation edges were found in OpCitance input.")

    candidate_pmids = _rank_candidate_pmids(graph, candidate_limit, max_papers)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        payload_by_pmid = _fetch_icite_payloads(
            http_client,
            candidate_pmids,
            batch_size=batch_size,
        )
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    selected_pmids = _select_paper_pmids(
        graph,
        payload_by_pmid,
        max_papers=max_papers,
    )
    if not selected_pmids:
        raise ValueError("No OpCitance PMID candidates matched NIH iCite.")

    papers = _build_papers(graph, selected_pmids, payload_by_pmid)
    papers_path = destination / "opcitance_icite_papers.jsonl"
    _write_papers_jsonl(papers_path, papers)

    labels_path = destination / "icite_current_metrics.csv"
    _write_icite_labels(labels_path, papers, payload_by_pmid)

    rankings = compute_impact_index_v2(
        papers,
        _neutral_pairwise_rankings(papers),
    )["rankings"]
    scores_path = destination / "impact_scores.csv"
    _write_impact_scores_csv(scores_path, rankings)

    credibility = write_current_impact_credibility(
        scores_path,
        labels_path,
        destination / "current_impact_credibility",
        score_columns=OPCITANCE_SCORE_COLUMNS,
        label_columns=OPCITANCE_LABEL_COLUMNS,
        primary_label_column="rcr",
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    summary = _summary(
        source,
        graph,
        papers,
        rankings,
        labels_path,
        scores_path,
        papers_path,
        credibility,
        max_rows=max_rows,
        candidate_count=len(candidate_pmids),
        output_dir=destination,
    )
    (destination / "opcitance_icite_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "opcitance_icite_benchmark_summary.md").write_text(
        render_opcitance_benchmark_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def render_opcitance_benchmark_summary_markdown(summary: dict[str, Any]) -> str:
    alignment = summary.get("primary_alignment", {})
    baseline = summary.get("baseline_comparison", {})
    lines = [
        "# OpCitance + iCite Current Impact Benchmark",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {summary['paper_count']} |",
        f"| icite_matched_paper_count | {summary['icite_matched_paper_count']} |",
        f"| opcitance_rows_read | {summary['opcitance_rows_read']} |",
        f"| internal_reference_edges | {summary['internal_reference_edges']} |",
        f"| citation_context_count | {summary['citation_context_count']} |",
        f"| effective_citation_coverage_rate | {summary['effective_citation_coverage_rate']:.3f} |",
        f"| credibility_status | {summary['credibility_status']} |",
        f"| credibility_grade | {summary['credibility_grade']} |",
        "",
        "## Primary RCR Alignment",
        "",
        "| Score | Spearman | Kendall | Pairwise Acc | NDCG@K |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for score_name, metrics in alignment.items():
        lines.append(
            f"| {score_name} | {_fmt(metrics.get('spearman_correlation'))} | "
            f"{_fmt(metrics.get('kendall_tau_b'))} | "
            f"{_fmt(metrics.get('pairwise_accuracy'))} | "
            f"{_fmt(metrics.get('ndcg_at_top_k'))} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| final_minus_true_spearman | {_fmt(baseline.get('final_minus_true_spearman'))} |",
            f"| final_minus_true_pairwise_accuracy | {_fmt(baseline.get('final_minus_true_pairwise_accuracy'))} |",
            f"| effective_minus_true_spearman | {_fmt(baseline.get('effective_minus_true_spearman'))} |",
            "",
            "Interpretation: this benchmark validates current biomedical citation impact. "
            "It tests whether effective citation context adds useful signal beyond true citation count, "
            "but it is not a future-impact prediction benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_opcitance_graph(
    path: Path,
    *,
    max_rows: int | None,
) -> dict[str, Any]:
    contexts: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    rows_read = 0
    skipped_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if max_rows is not None and rows_read >= max_rows:
                break
            rows_read += 1
            citing = _normalize_pmid(row.get("pmid"))
            cited = _cited_pmid(row)
            sentence = str(row.get("citation") or "").strip()
            if not citing or not cited or citing == cited or not sentence:
                skipped_rows += 1
                continue
            contexts[(citing, cited)].append(
                {
                    "sentence": sentence,
                    "section": _section(row),
                    "location": str(row.get("location") or ""),
                    "imrad": str(row.get("IMRaD") or ""),
                }
            )
    return {
        "edges": contexts,
        "rows_read": rows_read,
        "skipped_rows": skipped_rows,
    }


def _rank_candidate_pmids(
    graph: dict[str, Any],
    candidate_limit: int | None,
    max_papers: int,
) -> list[str]:
    incoming_contexts: dict[str, int] = defaultdict(int)
    outgoing_contexts: dict[str, int] = defaultdict(int)
    for (citing, cited), contexts in graph["edges"].items():
        incoming_contexts[cited] += len(contexts)
        outgoing_contexts[citing] += len(contexts)
    all_pmids = set(incoming_contexts) | set(outgoing_contexts)
    limit = candidate_limit or max(max_papers * 5, max_papers)
    ranked = sorted(
        all_pmids,
        key=lambda pmid: (
            -(incoming_contexts.get(pmid, 0) + outgoing_contexts.get(pmid, 0)),
            -incoming_contexts.get(pmid, 0),
            pmid,
        ),
    )
    return ranked[:limit]


def _fetch_icite_payloads(
    client: Any,
    pmids: list[str],
    *,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    payload_by_pmid: dict[str, dict[str, Any]] = {}
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        if not batch:
            continue
        response = client.get(ICITE_API_URL, params={"pmids": ",".join(batch)})
        response.raise_for_status()
        for row in response.json().get("data", []):
            pmid = _normalize_pmid(row.get("pmid"))
            if pmid:
                payload_by_pmid[pmid] = row
    return payload_by_pmid


def _select_paper_pmids(
    graph: dict[str, Any],
    payload_by_pmid: dict[str, dict[str, Any]],
    *,
    max_papers: int,
) -> list[str]:
    matched = set(payload_by_pmid)
    internal_contexts: dict[str, int] = defaultdict(int)
    incoming_contexts: dict[str, int] = defaultdict(int)
    outgoing_contexts: dict[str, int] = defaultdict(int)
    for (citing, cited), contexts in graph["edges"].items():
        if citing not in matched or cited not in matched:
            continue
        weight = len(contexts)
        internal_contexts[citing] += weight
        internal_contexts[cited] += weight
        incoming_contexts[cited] += weight
        outgoing_contexts[citing] += weight

    ranked = sorted(
        matched,
        key=lambda pmid: (
            -internal_contexts.get(pmid, 0),
            -incoming_contexts.get(pmid, 0),
            -outgoing_contexts.get(pmid, 0),
            -_numeric(payload_by_pmid.get(pmid, {}).get("citation_count")),
            pmid,
        ),
    )
    return ranked[:max_papers]


def _build_papers(
    graph: dict[str, Any],
    selected_pmids: list[str],
    payload_by_pmid: dict[str, dict[str, Any]],
) -> list[Paper]:
    selected = set(selected_pmids)
    references_by_paper: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    contexts_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (citing, cited), contexts in graph["edges"].items():
        if citing not in selected or cited not in selected:
            continue
        cited_payload = payload_by_pmid.get(cited, {})
        references_by_paper[citing][cited] = {
            "reference_id": cited,
            "title": _title(cited_payload, cited),
            "year": _int_or_none(cited_payload.get("year")),
            "venue": _text(cited_payload.get("journal")),
            "field": "biomedicine",
            "external_ids": {"pmid": cited},
        }
        for context in contexts:
            contexts_by_paper[citing].append(
                {
                    "reference_id": cited,
                    "sentence": context["sentence"],
                    "section": context["section"],
                }
            )

    papers = []
    for pmid in selected_pmids:
        payload = payload_by_pmid[pmid]
        papers.append(
            Paper.model_validate(
                {
                    "paper_id": pmid,
                    "title": _title(payload, pmid),
                    "abstract": _text(payload.get("abstract")) or None,
                    "year": _int_or_none(payload.get("year")),
                    "venue": _text(payload.get("journal")) or None,
                    "field": "biomedicine",
                    "authors": [],
                    "references": list(references_by_paper.get(pmid, {}).values()),
                    "citation_contexts": contexts_by_paper.get(pmid, []),
                    "metadata": {
                        "external_ids": {"pmid": pmid, "doi": _text(payload.get("doi"))},
                        "external_impact": {
                            "pubmed_citation_count": _value(payload.get("citation_count")),
                            "relative_citation_ratio": _value(
                                payload.get("relative_citation_ratio")
                            ),
                            "expected_citation_rate": _value(
                                payload.get("expected_citations_per_year")
                            ),
                            "field_citation_rate": _value(
                                payload.get("field_citation_rate")
                            ),
                        },
                        "benchmark_source": "opcitance_icite",
                    },
                    "icite": {
                        "relative_citation_ratio": _value(
                            payload.get("relative_citation_ratio")
                        ),
                        "citations_per_year": _value(payload.get("citations_per_year")),
                        "apt": _value(payload.get("apt")),
                        "nih_percentile": _value(payload.get("nih_percentile")),
                        "citation_count": _value(payload.get("citation_count")),
                    },
                }
            )
        )
    return papers


def _write_papers_jsonl(path: Path, papers: list[Paper]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.model_dump(), ensure_ascii=False) + "\n")


def _write_icite_labels(
    path: Path,
    papers: list[Paper],
    payload_by_pmid: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
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
        "benchmark_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            payload = payload_by_pmid.get(paper.paper_id, {})
            writer.writerow(
                sanitize_csv_row(
                    {
                        "paper_id": paper.paper_id,
                        "pmid": paper.paper_id,
                        "doi": _text(payload.get("doi")),
                        "title": _title(payload, paper.paper_id),
                        "year": _value(payload.get("year")),
                        "journal": _text(payload.get("journal")),
                        "rcr": _value(payload.get("relative_citation_ratio")),
                        "citations_per_year": _value(payload.get("citations_per_year")),
                        "apt": _value(payload.get("apt")),
                        "nih_percentile": _value(payload.get("nih_percentile")),
                        "icite_citation_count": _value(payload.get("citation_count")),
                        "expected_citations_per_year": _value(
                            payload.get("expected_citations_per_year")
                        ),
                        "field_citation_rate": _value(
                            payload.get("field_citation_rate")
                        ),
                        "benchmark_source": "nih_icite",
                    }
                )
            )


def _neutral_pairwise_rankings(papers: list[Paper]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]


def _summary(
    source: Path,
    graph: dict[str, Any],
    papers: list[Paper],
    rankings: list[dict[str, Any]],
    labels_path: Path,
    scores_path: Path,
    papers_path: Path,
    credibility: dict[str, Any],
    *,
    max_rows: int | None,
    candidate_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    paper_ids = {paper.paper_id for paper in papers}
    internal_edges = [
        (citing, cited, contexts)
        for (citing, cited), contexts in graph["edges"].items()
        if citing in paper_ids and cited in paper_ids
    ]
    citation_context_count = sum(len(contexts) for _, _, contexts in internal_edges)
    effective_count = sum(
        1 for row in rankings if int(row.get("effective_citation_edge_count") or 0) > 0
    )
    alignment = credibility.get("alignment", {})
    primary_alignment = {
        score: labels.get("rcr", {})
        for score, labels in alignment.items()
        if "rcr" in labels
    }
    baseline = _baseline_comparison(primary_alignment)
    summary = {
        "source_file": str(source),
        "source_file_partial": max_rows is not None,
        "opcitance_rows_read": graph["rows_read"],
        "opcitance_skipped_rows": graph["skipped_rows"],
        "candidate_pmid_count": candidate_count,
        "paper_count": len(papers),
        "icite_matched_paper_count": len(papers),
        "papers_with_references": sum(1 for paper in papers if paper.references),
        "internal_reference_edges": len(internal_edges),
        "citation_context_count": citation_context_count,
        "effective_citation_paper_count": effective_count,
        "effective_citation_coverage_rate": round(
            effective_count / len(papers) if papers else 0.0,
            6,
        ),
        "credibility_status": credibility["summary"]["credibility_status"],
        "credibility_grade": credibility["summary"]["credibility_grade"],
        "primary_alignment": primary_alignment,
        "baseline_comparison": baseline,
        "outputs": {
            "papers_jsonl": str(papers_path),
            "scores_csv": str(scores_path),
            "labels_csv": str(labels_path),
            "current_impact_credibility": str(output_dir / "current_impact_credibility"),
        },
    }
    return summary


def _baseline_comparison(
    primary_alignment: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    final = primary_alignment.get("final_impact_score", {})
    true = primary_alignment.get("true_citation_score", {})
    effective = primary_alignment.get("effective_citation_score", {})
    return {
        "final_minus_true_spearman": _delta(
            final.get("spearman_correlation"),
            true.get("spearman_correlation"),
        ),
        "final_minus_true_pairwise_accuracy": _delta(
            final.get("pairwise_accuracy"),
            true.get("pairwise_accuracy"),
        ),
        "effective_minus_true_spearman": _delta(
            effective.get("spearman_correlation"),
            true.get("spearman_correlation"),
        ),
    }


def _cited_pmid(row: dict[str, str]) -> str | None:
    best = _normalize_pmid(row.get("best_id"))
    if best and str(row.get("best_id_diff") or "").upper() != "NONE":
        return best
    return _normalize_pmid(row.get("intxt_pmid"))


def _normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    if re.fullmatch(r"\d+", text):
        return text
    return None


def _section(row: dict[str, str]) -> str:
    location = str(row.get("location") or "").strip()
    imrad = str(row.get("IMRaD") or "").strip()
    if location and imrad:
        return f"{location}:{imrad}"
    return location or imrad or "unknown"


def _title(payload: dict[str, Any], pmid: str) -> str:
    title = _text(payload.get("title"))
    return title or f"PMID {pmid}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _numeric(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
