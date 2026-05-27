from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

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

FUSED_SCORE_WEIGHTS = {
    "core_score": 0.40,
    "expanded_score": 0.25,
    "context_score": 0.20,
    "graph_score": 0.15,
}

FIELD_YEAR_FUSED_SCORE_WEIGHTS = {
    "core_score": 0.36,
    "expanded_score": 0.225,
    "context_score": 0.18,
    "graph_score": 0.135,
    "field_year_percentile": 0.10,
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
EXPANDED_SCORE_MAX_RANK = 20


@dataclass(frozen=True)
class InfluenceScores:
    paper_scores: dict[str, float]
    author_scores: dict[str, float]
    institution_scores: dict[str, float]
    field_scores: dict[str, float]
    year_scores: dict[str, float]
    paper_score_details: dict[str, dict[str, float]] | None = None


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
    paper_core_scores: defaultdict[str, float] = defaultdict(float)
    paper_expanded_scores: defaultdict[str, float] = defaultdict(float)
    paper_context_scores: defaultdict[str, float] = defaultdict(float)
    paper_max_edge_scores: defaultdict[str, float] = defaultdict(float)

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
            expanded_credit = expanded_selection_credit(selection, contexts=contexts)
            if credit > 0:
                paper_scores[selection.reference_id] += credit
                paper_max_edge_scores[selection.reference_id] = max(
                    paper_max_edge_scores[selection.reference_id],
                    credit,
                )
            core_credit = RANK_WEIGHTS.get(selection.rank, 0.0)
            if core_credit > 0:
                paper_core_scores[selection.reference_id] += core_credit
            if expanded_credit > 0:
                paper_expanded_scores[selection.reference_id] += expanded_credit
            context_credit = context_selection_credit(selection, contexts=contexts)
            if context_credit > 0:
                paper_context_scores[selection.reference_id] += context_credit
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
        paper_score_details=_paper_score_details(
            paper_scores,
            paper_core_scores,
            paper_expanded_scores,
            paper_context_scores,
            paper_max_edge_scores,
        ),
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


def expanded_selection_credit(
    selection: SelectedReference,
    contexts: list[CitationContext] | None = None,
) -> float:
    return expanded_rank_weight(selection.rank) * context_weight(contexts or [])


def context_selection_credit(
    selection: SelectedReference,
    contexts: list[CitationContext] | None = None,
) -> float:
    weight = context_weight(contexts or [])
    if weight <= CONTEXT_WEIGHT_NO_CONTEXT:
        return 0.0
    return expanded_rank_weight(selection.rank) * (weight - CONTEXT_WEIGHT_NO_CONTEXT)


def add_fused_paper_scores(
    scores: InfluenceScores,
    *,
    graph_scores: dict[str, float] | None = None,
    paper_metadata: dict[str, dict[str, object]] | None = None,
) -> InfluenceScores:
    details = {
        identifier: dict(values)
        for identifier, values in (scores.paper_score_details or {}).items()
    }
    identifiers = set(scores.paper_scores) | set(details)
    graph_scores = graph_scores or {}
    field_year_percentiles = _field_year_percentiles(
        scores.paper_scores,
        paper_metadata or {},
    )
    fusion_weights = (
        FIELD_YEAR_FUSED_SCORE_WEIGHTS if field_year_percentiles else FUSED_SCORE_WEIGHTS
    )

    component_values = {
        "core_score": {
            identifier: details.get(identifier, {}).get("core_score", 0.0)
            for identifier in identifiers
        },
        "expanded_score": {
            identifier: details.get(identifier, {}).get("expanded_score", 0.0)
            for identifier in identifiers
        },
        "context_score": {
            identifier: details.get(identifier, {}).get("context_score", 0.0)
            for identifier in identifiers
        },
        "graph_score": {
            identifier: graph_scores.get(identifier, 0.0) for identifier in identifiers
        },
        "field_year_percentile": {
            identifier: field_year_percentiles.get(identifier, 0.0)
            for identifier in identifiers
        },
    }
    normalized_components = {
        name: _normalize_to_100(values) for name, values in component_values.items()
    }

    for identifier in identifiers:
        details.setdefault(identifier, {})
        details[identifier]["graph_score"] = graph_scores.get(identifier, 0.0)
        details[identifier]["field_year_percentile"] = field_year_percentiles.get(
            identifier,
            0.0,
        )
        final_score = sum(
            fusion_weights[name] * normalized_components[name].get(identifier, 0.0)
            for name in fusion_weights
        )
        details[identifier]["final_score"] = final_score
        confidence_flag, confidence_reason = _paper_confidence(
            raw_score=scores.paper_scores.get(identifier, 0.0),
            details=details[identifier],
        )
        details[identifier]["confidence_flag"] = confidence_flag
        details[identifier]["confidence_reason"] = confidence_reason

    return InfluenceScores(
        paper_scores=scores.paper_scores,
        author_scores=scores.author_scores,
        institution_scores=scores.institution_scores,
        field_scores=scores.field_scores,
        year_scores=scores.year_scores,
        paper_score_details=details,
    )


def rank_weight(selection: SelectedReference, credit_mode: str) -> float:
    if credit_mode == "flat":
        return 1.0
    if credit_mode in {"rank_weighted", "context_weighted"}:
        return RANK_WEIGHTS.get(selection.rank, 0.0)
    raise ValueError(f"Unsupported credit_mode: {credit_mode}")


def expanded_rank_weight(rank: int) -> float:
    if rank < 1 or rank > EXPANDED_SCORE_MAX_RANK:
        return 0.0
    return 1.0 / math.log2(rank + 1)


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
            "derived_scores": {
                "core_score": "top_5_rank_weighted",
                "expanded_score": "log_discounted_top_20",
                "context_score": "log_discounted_top_20_context_bonus",
                "final_score": "weighted_normalized_fusion_v1",
                "final_score_weights": FUSED_SCORE_WEIGHTS,
                "field_year_percentile": "within_field_year_raw_score_percentile_when_metadata_available",
                "field_year_final_score_weights": FIELD_YEAR_FUSED_SCORE_WEIGHTS,
            },
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
            "derived_scores": {
                "core_score": "top_5_rank_weighted",
                "expanded_score": "log_discounted_top_20",
                "context_score": "log_discounted_top_20_context_bonus",
                "final_score": "weighted_normalized_fusion_v1",
                "final_score_weights": FUSED_SCORE_WEIGHTS,
                "field_year_percentile": "within_field_year_raw_score_percentile_when_metadata_available",
                "field_year_final_score_weights": FIELD_YEAR_FUSED_SCORE_WEIGHTS,
            },
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
            "derived_scores": {
                "core_score": "top_5_rank_weighted",
                "expanded_score": "log_discounted_top_20_context_weighted",
                "expanded_max_rank": EXPANDED_SCORE_MAX_RANK,
                "context_score": "log_discounted_top_20_context_bonus",
                "final_score": "weighted_normalized_fusion_v1",
                "final_score_weights": FUSED_SCORE_WEIGHTS,
                "field_year_percentile": "within_field_year_raw_score_percentile_when_metadata_available",
                "field_year_final_score_weights": FIELD_YEAR_FUSED_SCORE_WEIGHTS,
            },
        }
    raise ValueError(f"Unsupported credit_mode: {credit_mode}")


def _paper_score_details(
    paper_scores: dict[str, float],
    paper_core_scores: dict[str, float],
    paper_expanded_scores: dict[str, float],
    paper_context_scores: dict[str, float],
    paper_max_edge_scores: dict[str, float],
) -> dict[str, dict[str, float]]:
    identifiers = (
        set(paper_scores)
        | set(paper_core_scores)
        | set(paper_expanded_scores)
        | set(paper_context_scores)
        | set(paper_max_edge_scores)
    )
    return {
        identifier: {
            "core_score": paper_core_scores.get(identifier, 0.0),
            "expanded_score": paper_expanded_scores.get(identifier, 0.0),
            "context_score": paper_context_scores.get(identifier, 0.0),
            "max_single_edge_share": (
                paper_max_edge_scores.get(identifier, 0.0) / paper_scores[identifier]
                if paper_scores.get(identifier, 0.0) > 0
                else 0.0
            ),
        }
        for identifier in identifiers
    }


def _normalize_to_100(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values(), default=0.0)
    if max_score <= 0:
        return {identifier: 0.0 for identifier in scores}
    return {identifier: score / max_score * 100.0 for identifier, score in scores.items()}


def _field_year_percentiles(
    paper_scores: dict[str, float],
    paper_metadata: dict[str, dict[str, object]],
) -> dict[str, float]:
    buckets: defaultdict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for identifier, score in paper_scores.items():
        metadata = paper_metadata.get(identifier, {})
        field = metadata.get("field")
        year = metadata.get("year")
        if not field or year is None:
            continue
        try:
            year_value = int(year)
        except (TypeError, ValueError):
            continue
        buckets[(str(field), year_value)].append((identifier, score))

    percentiles: dict[str, float] = {}
    for rows in buckets.values():
        ordered = sorted(rows, key=lambda item: (-item[1], item[0]))
        total = len(ordered)
        for rank, (identifier, _score) in enumerate(ordered, start=1):
            percentiles[identifier] = (total - rank + 1) / total * 100.0
    return percentiles


def _paper_confidence(raw_score: float, details: dict[str, float]) -> tuple[str, str]:
    core_score = details.get("core_score", 0.0)
    expanded_score = details.get("expanded_score", 0.0)
    context_score = details.get("context_score", 0.0)
    graph_score = details.get("graph_score", 0.0)

    if raw_score <= 0 or (core_score <= 0 and expanded_score <= 0):
        return "low_evidence", "no positive core or expanded influence score"
    if core_score <= 0 and context_score <= 0 and graph_score <= 0:
        return "low_evidence", "expanded-only influence with no context or graph support"
    if context_score <= 0 and graph_score <= 0:
        return "fragile", "low context or graph support"
    if expanded_score > 0 and core_score / expanded_score >= 0.95 and context_score <= 0:
        return "fragile", "score is concentrated in strict top-k credit without context support"
    return "stable", "context or graph support present"


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
