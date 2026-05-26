from __future__ import annotations

from collections import defaultdict
from typing import Any

from local_impact_ranking.contexts import citation_contexts_by_reference
from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import InfluenceScores, selection_credit


def anomaly_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    scores: InfluenceScores,
    *,
    credit_mode: str,
    top_n: int = 20,
) -> dict[str, Any]:
    return {
        "top_scores": {
            "papers": _top_scores(scores.paper_scores, top_n),
            "authors": _top_scores(scores.author_scores, top_n),
            "institutions": _top_scores(scores.institution_scores, top_n),
        },
        "missing_metadata": _missing_metadata(papers),
        "score_dominance": _score_dominance(papers, judge_results, credit_mode, top_n),
        "limitations": [
            "High dominance can indicate a real concentrated influence pattern or a fragile ranking driven by one citation edge.",
            "Missing metadata directly reduces reliability for author and institution rankings.",
        ],
    }


def _missing_metadata(papers: list[Paper]) -> dict[str, int]:
    references = [reference for paper in papers for reference in paper.references]
    return {
        "papers_missing_title_count": sum(1 for paper in papers if not paper.title),
        "papers_missing_year_count": sum(1 for paper in papers if paper.year is None),
        "papers_missing_field_count": sum(1 for paper in papers if paper.field is None),
        "references_missing_title_count": sum(
            1 for reference in references if not reference.title
        ),
        "references_missing_author_count": sum(
            1 for reference in references if not reference.authors
        ),
        "references_missing_affiliation_count": sum(
            1
            for reference in references
            if not any(author.affiliations for author in reference.authors)
        ),
    }


def _score_dominance(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    credit_mode: str,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    reference_totals: defaultdict[str, float] = defaultdict(float)
    reference_max_edge: defaultdict[str, float] = defaultdict(float)
    author_totals: defaultdict[str, float] = defaultdict(float)
    author_max_edge: defaultdict[str, float] = defaultdict(float)
    institution_totals: defaultdict[str, float] = defaultdict(float)
    institution_max_edge: defaultdict[str, float] = defaultdict(float)

    references = {
        reference.reference_id: reference
        for paper in papers
        for reference in paper.references
        if reference.reference_id
    }
    papers_by_id = {paper.paper_id: paper for paper in papers}
    for result in judge_results:
        paper = papers_by_id.get(result.target_paper_id)
        contexts_by_reference = citation_contexts_by_reference(paper) if paper else {}
        for selection in result.selected_references:
            if not selection.valid:
                continue
            reference = references.get(selection.reference_id)
            if reference is None:
                continue
            credit = selection_credit(
                selection,
                credit_mode,
                contexts=contexts_by_reference.get(selection.reference_id, []),
            )
            reference_totals[selection.reference_id] += credit
            reference_max_edge[selection.reference_id] = max(
                reference_max_edge[selection.reference_id],
                credit,
            )
            if not reference.authors:
                continue
            author_credit = credit / len(reference.authors)
            for author in reference.authors:
                author_totals[author.name] += author_credit
                author_max_edge[author.name] = max(author_max_edge[author.name], author_credit)
                if not author.affiliations:
                    continue
                institution_credit = author_credit / len(author.affiliations)
                for affiliation in author.affiliations:
                    institution_totals[affiliation] += institution_credit
                    institution_max_edge[affiliation] = max(
                        institution_max_edge[affiliation],
                        institution_credit,
                    )

    return {
        "paper_single_edge_dominance": _dominance_rows(
            reference_totals,
            reference_max_edge,
            top_n,
        ),
        "author_single_edge_dominance": _dominance_rows(
            author_totals,
            author_max_edge,
            top_n,
        ),
        "institution_single_edge_dominance": _dominance_rows(
            institution_totals,
            institution_max_edge,
            top_n,
        ),
    }


def _dominance_rows(
    totals: dict[str, float],
    max_edges: dict[str, float],
    top_n: int,
) -> list[dict[str, Any]]:
    rows = []
    for identifier, total in totals.items():
        max_edge = max_edges.get(identifier, 0.0)
        rows.append(
            {
                "id": identifier,
                "score": total,
                "max_single_edge_score": max_edge,
                "max_single_edge_share": max_edge / total if total else 0.0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["max_single_edge_share"], -row["score"], row["id"]),
    )[:top_n]


def _top_scores(scores: dict[str, float], top_n: int) -> list[dict[str, Any]]:
    return [
        {"id": identifier, "score": score}
        for identifier, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]
