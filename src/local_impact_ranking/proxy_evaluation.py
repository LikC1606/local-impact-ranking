from __future__ import annotations

from typing import Any

from local_impact_ranking.baselines import BibliometricBaselines
from local_impact_ranking.contexts import citation_contexts_by_reference
from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import CitationContext, Paper

METHOD_CUE_TERMS = {
    "build on",
    "builds on",
    "based on",
    "extend",
    "extends",
    "architecture",
    "method",
    "algorithm",
    "dataset",
    "benchmark",
    "baseline",
    "compare",
    "comparison",
}


def evaluate_proxy_quality(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    baselines: BibliometricBaselines,
    *,
    top_n: int = 100,
) -> dict[str, Any]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    baseline_top_ids = _baseline_top_ids(baselines, top_n)
    evidence_rows: list[dict[str, Any]] = []

    for result in judge_results:
        paper = papers_by_id.get(result.target_paper_id)
        contexts_by_reference = citation_contexts_by_reference(paper) if paper else {}
        for selection in result.selected_references:
            if not selection.valid:
                continue
            contexts = contexts_by_reference.get(selection.reference_id, [])
            context_supported = bool(contexts)
            method_context_supported = any(_has_method_cue(context) for context in contexts)
            bibliometric_support = any(
                selection.reference_id in ids for ids in baseline_top_ids.values()
            )
            evidence_rows.append(
                {
                    "target_paper_id": result.target_paper_id,
                    "reference_id": selection.reference_id,
                    "rank": selection.rank,
                    "context_supported": context_supported,
                    "method_context_supported": method_context_supported,
                    "bibliometric_top_n_supported": bibliometric_support,
                    "citation_count_top_n_supported": selection.reference_id
                    in baseline_top_ids["citation_count"],
                    "pagerank_top_n_supported": selection.reference_id
                    in baseline_top_ids["pagerank"],
                    "field_year_normalized_top_n_supported": selection.reference_id
                    in baseline_top_ids["field_year_normalized_citation"],
                    "unsupported": not context_supported and not bibliometric_support,
                }
            )

    selected_count = len(evidence_rows)
    return {
        "selected_references": selected_count,
        "top_n": top_n,
        "context_supported_selection_rate": _rate(
            evidence_rows,
            "context_supported",
        ),
        "method_context_selection_rate": _rate(
            evidence_rows,
            "method_context_supported",
        ),
        "bibliometric_top_n_support_rate": _rate(
            evidence_rows,
            "bibliometric_top_n_supported",
        ),
        "unsupported_selection_rate": _rate(evidence_rows, "unsupported"),
        "selected_reference_evidence": evidence_rows,
        "limitations": [
            "Proxy evaluation is not a human correctness label.",
            "Low proxy support can indicate either a weak selection or missing metadata/context coverage.",
        ],
    }


def _baseline_top_ids(baselines: BibliometricBaselines, top_n: int) -> dict[str, set[str]]:
    return {
        "citation_count": set(_top_ids(baselines.citation_count, top_n)),
        "pagerank": set(_top_ids(baselines.pagerank, top_n)),
        "field_year_normalized_citation": set(
            _top_ids(baselines.field_year_normalized_citation, top_n)
        ),
    }


def _top_ids(scores: dict[str, float], top_n: int) -> list[str]:
    return [
        identifier
        for identifier, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]


def _has_method_cue(context: CitationContext) -> bool:
    text = f"{context.section or ''} {context.sentence} {context.paragraph or ''}".lower()
    return any(term in text for term in METHOD_CUE_TERMS)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)
