from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import scoring_edges


def paper_influence_explanations(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    credit_mode: str,
    top_n: int = 20,
    max_edges_per_paper: int = 5,
) -> dict[str, Any]:
    references = {
        reference.reference_id: reference
        for paper in papers
        for reference in paper.references
        if reference.reference_id
    }
    papers_by_id = {paper.paper_id: paper for paper in papers}
    grouped_edges: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in scoring_edges(papers, judge_results, credit_mode=credit_mode):
        grouped_edges[edge["reference_id"]].append(edge)

    rows = []
    for reference_id, edges in grouped_edges.items():
        reference = references.get(reference_id)
        sorted_edges = sorted(
            edges,
            key=lambda edge: (-edge["credit"], edge["target_paper_id"], edge["rank"]),
        )
        rows.append(
            {
                "reference_id": reference_id,
                "reference_title": reference.title if reference else "",
                "score": sum(edge["credit"] for edge in edges),
                "source_edge_count": len(edges),
                "top_source_edges": [
                    _source_edge_row(edge, papers_by_id)
                    for edge in sorted_edges[:max_edges_per_paper]
                ],
            }
        )

    return {
        "credit_mode": credit_mode,
        "top_n": top_n,
        "max_edges_per_paper": max_edges_per_paper,
        "papers": sorted(
            rows,
            key=lambda row: (-row["score"], row["reference_id"]),
        )[:top_n],
        "limitations": [
            "Explanations trace local selected-reference credit, not global scientific impact.",
            "A source edge is only as reliable as the judge selection and available metadata/context.",
        ],
    }


def paper_evidence_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    paper_score_details: dict[str, dict[str, Any]] | None,
    paper_score_uncertainty: dict[str, Any] | None = None,
    credit_mode: str,
    top_n: int = 100,
    max_edges_per_paper: int = 5,
) -> dict[str, Any]:
    references = {
        reference.reference_id: reference
        for paper in papers
        for reference in paper.references
        if reference.reference_id
    }
    papers_by_id = {paper.paper_id: paper for paper in papers}
    grouped_edges: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in scoring_edges(papers, judge_results, credit_mode=credit_mode):
        grouped_edges[edge["reference_id"]].append(edge)

    score_details = paper_score_details or {}
    uncertainty_by_paper = _paper_score_uncertainty_index(paper_score_uncertainty)
    rows = []
    for reference_id, details in score_details.items():
        reference = references.get(reference_id)
        edges = grouped_edges.get(reference_id, [])
        uncertainty = uncertainty_by_paper.get(reference_id, {})
        sorted_edges = sorted(
            edges,
            key=lambda edge: (-edge["credit"], edge["target_paper_id"], edge["rank"]),
        )
        rows.append(
            {
                "paper_id": reference_id,
                "paper_title": reference.title if reference else "",
                "final_score": details.get("final_score", 0.0),
                "confidence_flag": details.get("confidence_flag", ""),
                "max_single_edge_share": details.get("max_single_edge_share", 0.0),
                "support_band": _support_band(
                    details.get("max_single_edge_share", 0.0),
                    len(edges),
                ),
                "score_ci_low": uncertainty.get("score_ci_low", 0.0),
                "score_ci_high": uncertainty.get("score_ci_high", 0.0),
                "score_ci_width": uncertainty.get("score_ci_width", 0.0),
                "rank_span": uncertainty.get("rank_span", 0),
                "zero_score_rate": uncertainty.get("zero_score_rate", 0.0),
                "uncertainty_band": uncertainty.get("uncertainty_band", ""),
                "source_edge_count": len(edges),
                "top_source_edges": [
                    _evidence_edge_row(edge, papers_by_id)
                    for edge in sorted_edges[:max_edges_per_paper]
                ],
            }
        )

    return {
        "credit_mode": credit_mode,
        "top_n": top_n,
        "max_edges_per_paper": max_edges_per_paper,
        "papers": sorted(
            rows,
            key=lambda row: (-float(row["final_score"]), row["paper_id"]),
        )[:top_n],
        "limitations": [
            "Evidence rows explain local selected-reference score contributions, not global scientific impact.",
            "Low evidence coverage or high single-edge dominance should be reported with the final score.",
        ],
    }


def export_paper_evidence_csv(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "final_score",
        "confidence_flag",
        "support_band",
        "max_single_edge_share",
        "score_ci_low",
        "score_ci_high",
        "score_ci_width",
        "rank_span",
        "zero_score_rate",
        "uncertainty_band",
        "top_edge_target_paper_id",
        "top_edge_credit",
        "top_edge_rank",
        "top_edge_context_count",
        "top_edge_context_weight",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paper in report.get("papers", []):
            top_edges = paper.get("top_source_edges", [])
            if not top_edges:
                writer.writerow(sanitize_csv_row(_paper_evidence_csv_row(paper, {})))
                continue
            for edge in top_edges:
                writer.writerow(sanitize_csv_row(_paper_evidence_csv_row(paper, edge)))


def entity_influence_explanations(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    credit_mode: str,
    top_n: int = 20,
    max_edges_per_entity: int = 5,
) -> dict[str, Any]:
    references = {
        reference.reference_id: reference
        for paper in papers
        for reference in paper.references
        if reference.reference_id
    }
    papers_by_id = {paper.paper_id: paper for paper in papers}
    author_edges: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    institution_edges: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in scoring_edges(papers, judge_results, credit_mode=credit_mode):
        reference = references.get(edge["reference_id"])
        if reference is None or not reference.authors:
            continue
        author_credit = edge["credit"] / len(reference.authors)
        for author in reference.authors:
            author_edges[author.name].append(
                _entity_source_edge_row(
                    edge,
                    papers_by_id,
                    references,
                    author_credit,
                )
            )
            if not author.affiliations:
                continue
            institution_credit = author_credit / len(author.affiliations)
            for affiliation in author.affiliations:
                institution_edges[affiliation].append(
                    _entity_source_edge_row(
                        edge,
                        papers_by_id,
                        references,
                        institution_credit,
                    )
                )

    return {
        "credit_mode": credit_mode,
        "top_n": top_n,
        "max_edges_per_entity": max_edges_per_entity,
        "authors": _entity_rows(author_edges, top_n, max_edges_per_entity),
        "institutions": _entity_rows(institution_edges, top_n, max_edges_per_entity),
        "limitations": [
            "Author and institution explanations inherit the coauthor and affiliation credit-splitting assumptions.",
            "Missing or unnormalized author and affiliation metadata directly reduces explanation reliability.",
        ],
    }


def _source_edge_row(edge: dict[str, Any], papers_by_id: dict[str, Paper]) -> dict[str, Any]:
    target = papers_by_id.get(edge["target_paper_id"])
    return {
        "target_paper_id": edge["target_paper_id"],
        "target_title": target.title if target else "",
        "rank": edge["rank"],
        "credit": edge["credit"],
        "context_count": edge["context_count"],
    }


def _evidence_edge_row(edge: dict[str, Any], papers_by_id: dict[str, Paper]) -> dict[str, Any]:
    target = papers_by_id.get(edge["target_paper_id"])
    return {
        "target_paper_id": edge["target_paper_id"],
        "target_title": target.title if target else "",
        "rank": edge["rank"],
        "credit": edge["credit"],
        "context_count": edge["context_count"],
        "context_weight": edge["context_weight"],
    }


def _paper_evidence_csv_row(
    paper: dict[str, Any],
    edge: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paper_id", ""),
        "final_score": f"{float(paper.get('final_score', 0.0)):.6f}",
        "confidence_flag": paper.get("confidence_flag", ""),
        "support_band": paper.get("support_band", ""),
        "max_single_edge_share": f"{float(paper.get('max_single_edge_share', 0.0)):.6f}",
        "score_ci_low": f"{float(paper.get('score_ci_low', 0.0)):.6f}",
        "score_ci_high": f"{float(paper.get('score_ci_high', 0.0)):.6f}",
        "score_ci_width": f"{float(paper.get('score_ci_width', 0.0)):.6f}",
        "rank_span": paper.get("rank_span", 0),
        "zero_score_rate": f"{float(paper.get('zero_score_rate', 0.0)):.6f}",
        "uncertainty_band": paper.get("uncertainty_band", ""),
        "top_edge_target_paper_id": edge.get("target_paper_id", ""),
        "top_edge_credit": f"{float(edge.get('credit', 0.0)):.6f}",
        "top_edge_rank": edge.get("rank", ""),
        "top_edge_context_count": edge.get("context_count", ""),
        "top_edge_context_weight": (
            f"{float(edge.get('context_weight', 0.0)):.6f}"
            if edge
            else ""
        ),
    }


def _support_band(max_single_edge_share: Any, source_edge_count: int) -> str:
    try:
        share = float(max_single_edge_share)
    except (TypeError, ValueError):
        share = 0.0
    if source_edge_count <= 1 or share >= 0.75:
        return "concentrated"
    if share >= 0.4:
        return "mixed"
    return "distributed"


def _paper_score_uncertainty_index(
    report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    return {
        str(row.get("paper_id")): row
        for row in report.get("papers", [])
        if row.get("paper_id")
    }


def _entity_source_edge_row(
    edge: dict[str, Any],
    papers_by_id: dict[str, Paper],
    references: dict[str, Any],
    credit: float,
) -> dict[str, Any]:
    target = papers_by_id.get(edge["target_paper_id"])
    reference = references.get(edge["reference_id"])
    return {
        "target_paper_id": edge["target_paper_id"],
        "target_title": target.title if target else "",
        "reference_id": edge["reference_id"],
        "reference_title": reference.title if reference else "",
        "rank": edge["rank"],
        "credit": credit,
        "context_count": edge["context_count"],
    }


def _entity_rows(
    grouped_edges: dict[str, list[dict[str, Any]]],
    top_n: int,
    max_edges_per_entity: int,
) -> list[dict[str, Any]]:
    rows = []
    for entity_id, edges in grouped_edges.items():
        sorted_edges = sorted(
            edges,
            key=lambda edge: (
                -edge["credit"],
                edge["target_paper_id"],
                edge["reference_id"],
                edge["rank"],
            ),
        )
        rows.append(
            {
                "entity_id": entity_id,
                "score": sum(edge["credit"] for edge in edges),
                "source_edge_count": len(edges),
                "top_source_edges": sorted_edges[:max_edges_per_entity],
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["entity_id"]))[:top_n]
