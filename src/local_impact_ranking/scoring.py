from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from local_impact_ranking.contexts import citation_contexts_by_reference
from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import CitationContext, Paper, Reference


RANK_WEIGHTS = {
    1: 1.0,
    2: 0.8,
    3: 0.6,
    4: 0.4,
    5: 0.2,
}

SCORING_VERSION_BY_CREDIT_MODE = {
    "rank_weighted": "score_v1_rank_weighted",
    "flat": "score_v1_flat",
    "context_weighted": "score_v2_context_weighted",
}

METHOD_CUE_TERMS = {
    "build on",
    "builds on",
    "based on",
    "extend",
    "extends",
    "method",
    "algorithm",
    "dataset",
    "benchmark",
    "baseline",
    "compare",
    "comparison",
}

CONTEXT_WEIGHT_NO_CONTEXT = 1.0
CONTEXT_WEIGHT_CONTEXT_SUPPORTED = 1.2
CONTEXT_WEIGHT_METHOD_SUPPORTED = 1.5


@dataclass(frozen=True)
class InfluenceScores:
    paper_scores: dict[str, float]
    author_scores: dict[str, float]
    institution_scores: dict[str, float]
    field_scores: dict[str, float]
    year_scores: dict[str, float]


def score_influence(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    credit_mode: str = "rank_weighted",
) -> InfluenceScores:
    references = _reference_index(papers)
    papers_by_id = {paper.paper_id: paper for paper in papers}
    paper_scores: defaultdict[str, float] = defaultdict(float)
    author_scores: defaultdict[str, float] = defaultdict(float)
    institution_scores: defaultdict[str, float] = defaultdict(float)
    field_scores: defaultdict[str, float] = defaultdict(float)
    year_scores: defaultdict[str, float] = defaultdict(float)

    for result in judge_results:
        for selection in result.selected_references:
            if not selection.valid:
                continue
            reference = references.get(selection.reference_id)
            if reference is None:
                continue
            paper = papers_by_id.get(result.target_paper_id)
            contexts = []
            if paper is not None:
                contexts = citation_contexts_by_reference(paper).get(
                    selection.reference_id,
                    [],
                )
            credit = selection_credit(selection, credit_mode, contexts=contexts)
            paper_scores[selection.reference_id] += credit
            if reference.field:
                field_scores[reference.field] += credit
            if reference.year is not None:
                year_scores[str(reference.year)] += credit
            _assign_author_and_institution_credit(
                reference,
                credit,
                author_scores,
                institution_scores,
            )

    return InfluenceScores(
        paper_scores=dict(paper_scores),
        author_scores=dict(author_scores),
        institution_scores=dict(institution_scores),
        field_scores=dict(field_scores),
        year_scores=dict(year_scores),
    )


def scoring_edges(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    credit_mode: str = "rank_weighted",
) -> list[dict]:
    references = _reference_index(papers)
    papers_by_id = {paper.paper_id: paper for paper in papers}
    rows = []
    for result in judge_results:
        paper = papers_by_id.get(result.target_paper_id)
        contexts_by_reference = citation_contexts_by_reference(paper) if paper else {}
        for selection in result.selected_references:
            if not selection.valid:
                continue
            if selection.reference_id not in references:
                continue
            contexts = contexts_by_reference.get(selection.reference_id, [])
            rows.append(
                {
                    "target_paper_id": result.target_paper_id,
                    "reference_id": selection.reference_id,
                    "rank": selection.rank,
                    "rank_weight": rank_weight(selection, credit_mode),
                    "context_count": len(contexts),
                    "context_weight": context_weight(contexts),
                    "credit": selection_credit(
                        selection,
                        credit_mode,
                        contexts=contexts,
                    ),
                }
            )
    return rows


def _reference_index(papers: list[Paper]) -> dict[str, Reference]:
    index: dict[str, Reference] = {}
    for paper in papers:
        for reference in paper.references:
            if reference.reference_id:
                index[reference.reference_id] = reference
    return index


def selection_credit(
    selection: SelectedReference,
    credit_mode: str,
    contexts: list[CitationContext] | None = None,
) -> float:
    if credit_mode == "flat":
        return 1.0
    if credit_mode == "rank_weighted":
        return RANK_WEIGHTS.get(selection.rank, 0.0)
    if credit_mode == "context_weighted":
        return RANK_WEIGHTS.get(selection.rank, 0.0) * context_weight(contexts or [])
    raise ValueError(f"Unsupported credit_mode: {credit_mode}")


def rank_weight(selection: SelectedReference, credit_mode: str) -> float:
    if credit_mode == "flat":
        return 1.0
    if credit_mode in {"rank_weighted", "context_weighted"}:
        return RANK_WEIGHTS.get(selection.rank, 0.0)
    raise ValueError(f"Unsupported credit_mode: {credit_mode}")


def context_weight(contexts: list[CitationContext]) -> float:
    if not contexts:
        return CONTEXT_WEIGHT_NO_CONTEXT
    if any(_has_method_cue(context) for context in contexts):
        return CONTEXT_WEIGHT_METHOD_SUPPORTED
    return CONTEXT_WEIGHT_CONTEXT_SUPPORTED


def scoring_config(credit_mode: str) -> dict:
    if credit_mode == "rank_weighted":
        return {
            "version": SCORING_VERSION_BY_CREDIT_MODE[credit_mode],
            "credit_mode": credit_mode,
            "rank_weights": {str(rank): weight for rank, weight in RANK_WEIGHTS.items()},
            "normalization": "none",
            "context_weighting": "none",
            "confidence_weighting": "none",
            "contribution_cap": None,
        }
    if credit_mode == "flat":
        return {
            "version": SCORING_VERSION_BY_CREDIT_MODE[credit_mode],
            "credit_mode": credit_mode,
            "rank_weights": {},
            "normalization": "none",
            "context_weighting": "none",
            "confidence_weighting": "none",
            "contribution_cap": None,
        }
    if credit_mode == "context_weighted":
        return {
            "version": SCORING_VERSION_BY_CREDIT_MODE[credit_mode],
            "credit_mode": credit_mode,
            "rank_weights": {str(rank): weight for rank, weight in RANK_WEIGHTS.items()},
            "normalization": "none",
            "context_weighting": {
                "no_context": CONTEXT_WEIGHT_NO_CONTEXT,
                "context_supported": CONTEXT_WEIGHT_CONTEXT_SUPPORTED,
                "method_or_data_or_baseline_supported": CONTEXT_WEIGHT_METHOD_SUPPORTED,
                "cue_terms": sorted(METHOD_CUE_TERMS),
            },
            "confidence_weighting": "none",
            "contribution_cap": None,
        }
    raise ValueError(f"Unsupported credit_mode: {credit_mode}")


def _has_method_cue(context: CitationContext) -> bool:
    text = f"{context.section or ''} {context.sentence} {context.paragraph or ''}".lower()
    return any(term in text for term in METHOD_CUE_TERMS)


def _assign_author_and_institution_credit(
    reference: Reference,
    credit: float,
    author_scores: defaultdict[str, float],
    institution_scores: defaultdict[str, float],
) -> None:
    if not reference.authors:
        return
    author_credit = credit / len(reference.authors)
    for author in reference.authors:
        author_scores[author.name] += author_credit
        if not author.affiliations:
            continue
        affiliation_credit = author_credit / len(author.affiliations)
        for affiliation in author.affiliations:
            institution_scores[affiliation] += affiliation_credit
