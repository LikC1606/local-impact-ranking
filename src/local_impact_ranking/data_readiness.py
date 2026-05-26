from __future__ import annotations

from typing import Any

from local_impact_ranking.models import Paper


def data_readiness_report(papers: list[Paper]) -> dict[str, Any]:
    paper_count = len(papers)
    references = [reference for paper in papers for reference in paper.references]
    reference_count = len(references)

    coverage = {
        "paper_id_coverage": _coverage(papers, lambda paper: bool(paper.paper_id)),
        "target_title_coverage": _coverage(papers, lambda paper: bool(paper.title)),
        "candidate_reference_list_coverage": _coverage(
            papers,
            lambda paper: bool(paper.references),
        ),
        "reference_id_coverage": _coverage(
            references,
            lambda reference: bool(reference.reference_id),
        ),
        "reference_title_coverage": _coverage(
            references,
            lambda reference: bool(reference.title),
        ),
        "referenced_author_coverage": _coverage(
            references,
            lambda reference: bool(reference.authors),
        ),
        "referenced_affiliation_coverage": _coverage(
            references,
            lambda reference: any(author.affiliations for author in reference.authors),
        ),
        "citation_context_coverage": _coverage(
            papers,
            lambda paper: bool(paper.citation_contexts),
        ),
    }
    return {
        "paper_count": paper_count,
        "reference_count": reference_count,
        "coverage": coverage,
        "ranking_scope": {
            "paper_ranking": _scope(_paper_gate(coverage), "exploratory"),
            "author_ranking": _scope(_author_gate(coverage), "exploratory"),
            "institution_ranking": _scope(_institution_gate(coverage), "exploratory"),
            "context_aware_ranking": _scope(_context_gate(coverage), "metadata-only"),
        },
        "thresholds": {
            "target_title_coverage": 0.95,
            "candidate_reference_list_coverage": 0.95,
            "reference_title_coverage": 0.90,
            "referenced_author_coverage": 0.70,
            "referenced_affiliation_coverage": 0.70,
        },
        "limitations": [
            "Failed gates do not stop the run, but affected rankings should be treated as exploratory.",
            "Context-aware metrics require citation context coverage and should not be interpreted as correctness when coverage is incomplete.",
        ],
    }


def _coverage(items, predicate) -> float:
    total = len(items)
    if total == 0:
        return 0.0
    return sum(1 for item in items if predicate(item)) / total


def _paper_gate(coverage: dict[str, float]) -> bool:
    return (
        coverage["paper_id_coverage"] == 1.0
        and coverage["target_title_coverage"] >= 0.95
        and coverage["candidate_reference_list_coverage"] >= 0.95
        and coverage["reference_id_coverage"] == 1.0
        and coverage["reference_title_coverage"] >= 0.90
    )


def _author_gate(coverage: dict[str, float]) -> bool:
    return _paper_gate(coverage) and coverage["referenced_author_coverage"] >= 0.70


def _institution_gate(coverage: dict[str, float]) -> bool:
    return (
        _author_gate(coverage)
        and coverage["referenced_affiliation_coverage"] >= 0.70
    )


def _context_gate(coverage: dict[str, float]) -> bool:
    return coverage["citation_context_coverage"] > 0.0


def _scope(passes: bool, failed_scope: str) -> str:
    return "formal" if passes else failed_scope
