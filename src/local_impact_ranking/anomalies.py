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
        "self_citation_risk": _self_citation_risk(
            papers,
            judge_results,
            credit_mode,
            top_n,
        ),
        "limitations": [
            "High dominance can indicate a real concentrated influence pattern or a fragile ranking driven by one citation edge.",
            "Self-citation risk is a disclosure signal, not an automatic penalty; some self-citation can be methodologically appropriate.",
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


def _self_citation_risk(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    credit_mode: str,
    top_n: int,
) -> dict[str, Any]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    selected_edge_count = 0
    self_citation_edge_count = 0
    total_credit = 0.0
    self_citation_credit = 0.0
    by_reference: dict[str, dict[str, Any]] = {}

    for result in judge_results:
        paper = papers_by_id.get(result.target_paper_id)
        if paper is None:
            continue
        references = {
            reference.reference_id: reference
            for reference in paper.references
            if reference.reference_id
        }
        contexts_by_reference = citation_contexts_by_reference(paper)
        target_authors = _author_identity_set(paper.authors)
        for selection in result.selected_references:
            if not selection.valid:
                continue
            reference = references.get(selection.reference_id)
            if reference is None:
                continue
            selected_edge_count += 1
            credit = selection_credit(
                selection,
                credit_mode,
                contexts=contexts_by_reference.get(selection.reference_id, []),
            )
            total_credit += credit
            overlap = sorted(target_authors & _author_identity_set(reference.authors))
            if not overlap:
                continue
            self_citation_edge_count += 1
            self_citation_credit += credit
            row = by_reference.setdefault(
                selection.reference_id,
                {
                    "reference_id": selection.reference_id,
                    "title": reference.title,
                    "self_citation_credit": 0.0,
                    "total_credit": 0.0,
                    "self_citation_edge_count": 0,
                    "overlapping_authors": set(),
                },
            )
            row["self_citation_credit"] += credit
            row["total_credit"] += credit
            row["self_citation_edge_count"] += 1
            row["overlapping_authors"].update(overlap)

    rows = []
    for row in by_reference.values():
        overlap = sorted(row["overlapping_authors"])
        rows.append(
            {
                "reference_id": row["reference_id"],
                "title": row["title"],
                "self_citation_credit": round(float(row["self_citation_credit"]), 6),
                "total_credit": round(float(row["total_credit"]), 6),
                "self_citation_credit_share": _safe_divide(
                    float(row["self_citation_credit"]),
                    float(row["total_credit"]),
                ),
                "self_citation_edge_count": int(row["self_citation_edge_count"]),
                "overlapping_authors": overlap,
            }
        )

    edge_rate = _safe_divide(self_citation_edge_count, selected_edge_count)
    credit_share = _safe_divide(self_citation_credit, total_credit)
    return {
        "selected_edge_count": selected_edge_count,
        "self_citation_edge_count": self_citation_edge_count,
        "self_citation_edge_rate": round(edge_rate, 6),
        "total_credit": round(total_credit, 6),
        "self_citation_credit": round(self_citation_credit, 6),
        "self_citation_credit_share": round(credit_share, 6),
        "risk_level": _self_citation_risk_level(edge_rate, credit_share),
        "top_self_cited_references": sorted(
            rows,
            key=lambda item: (
                -float(item["self_citation_credit"]),
                item["reference_id"],
            ),
        )[:top_n],
    }


def _author_identity_set(authors: list[Any]) -> set[str]:
    identities: set[str] = set()
    for author in authors:
        name = str(getattr(author, "name", "") or "").strip()
        if name:
            identities.add(name)
        ids = getattr(author, "ids", {}) or {}
        if isinstance(ids, dict):
            for key, value in ids.items():
                if value:
                    identities.add(f"{key}:{value}")
    return identities


def _self_citation_risk_level(edge_rate: float, credit_share: float) -> str:
    if edge_rate >= 0.5 or credit_share >= 0.5:
        return "high"
    if edge_rate >= 0.2 or credit_share >= 0.2:
        return "moderate"
    return "low"


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _top_scores(scores: dict[str, float], top_n: int) -> list[dict[str, Any]]:
    return [
        {"id": identifier, "score": score}
        for identifier, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]
