from __future__ import annotations

from collections import defaultdict
from typing import Any

from local_impact_ranking.models import Paper
from local_impact_ranking.judge import JudgeResult
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
