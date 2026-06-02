from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from local_impact_ranking.baselines import compute_bibliometric_baselines
from local_impact_ranking.contexts import citation_contexts_by_reference
from local_impact_ranking.models import CitationContext, Paper
from local_impact_ranking.scoring import context_weight


IMPACT_INDEX_V2_WEIGHTS = {
    "model_pairwise": 0.30,
    "field_year_normalized_citation": 0.25,
    "citation_graph": 0.20,
    "citation_context_quality": 0.15,
    "temporal_momentum": 0.10,
}


EVIDENCE_FIRST_WEIGHTS = {
    "field_year_normalized_citation": 0.18,
    "relative_citation_ratio": 0.14,
    "citation_graph": 0.14,
    "citation_context_quality": 0.14,
    "influential_citation": 0.10,
    "temporal_momentum": 0.10,
    "external_adoption": 0.20,
}


IMPACT_INDEX_V3_WEIGHTS = {
    "evidence_supported": 0.80,
    "model_supported": 0.20,
}


COMPONENT_SCORE_FIELDS = {
    "model_pairwise": "model_pairwise_score",
    "field_year_normalized_citation": "field_year_normalized_citation_score",
    "relative_citation_ratio": "relative_citation_ratio_score",
    "citation_graph": "citation_graph_score",
    "citation_context_quality": "citation_context_quality_score",
    "influential_citation": "influential_citation_score",
    "temporal_momentum": "temporal_momentum_score",
    "external_adoption": "external_adoption_score",
}


EFFECTIVE_CITATION_ROLE_WEIGHTS = {
    "guideline": 2.0,
    "clinical": 1.8,
    "method": 1.6,
    "dataset_or_tool": 1.5,
    "negative_or_contrasting": 1.1,
    "supporting": 1.0,
    "background": 0.35,
    "unknown": 0.1,
}

STRONG_EFFECTIVE_CITATION_ROLES = {
    "guideline",
    "clinical",
    "method",
    "dataset_or_tool",
    "negative_or_contrasting",
    "supporting",
}


CALIBRATABLE_V2_COMPONENT_SCORE_FIELDS = {
    name: COMPONENT_SCORE_FIELDS[name] for name in IMPACT_INDEX_V2_WEIGHTS
}

CALIBRATABLE_COMPONENT_SCORE_FIELDS = {
    "model_pairwise": COMPONENT_SCORE_FIELDS["model_pairwise"],
    **{name: COMPONENT_SCORE_FIELDS[name] for name in EVIDENCE_FIRST_WEIGHTS},
}


def compute_impact_index_v2(
    papers: list[Paper],
    pairwise_rankings: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    evidence_weights: dict[str, float] | None = None,
    impact_index_v3_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fuse model and bibliometric signals into a dataset-local 0-100 score."""
    weights = dict(weights or IMPACT_INDEX_V2_WEIGHTS)
    evidence_weights = dict(evidence_weights or _evidence_first_weights_from_fusion(weights))
    impact_index_v3_weights = dict(impact_index_v3_weights or IMPACT_INDEX_V3_WEIGHTS)
    pairwise_by_id = {row["paper_id"]: dict(row) for row in pairwise_rankings}
    components = _impact_components(papers, pairwise_by_id)
    rows: list[dict[str, Any]] = []

    for paper in papers:
        paper_id = paper.paper_id
        base_row = pairwise_by_id.get(
            paper_id,
            {
                "paper_id": paper_id,
                "title": paper.title,
                "impact_index": 50.0,
                "comparisons": 0,
                "confidence": 0.0,
            },
        )
        row = dict(base_row)
        row.setdefault("paper_id", paper_id)
        row.setdefault("title", paper.title)
        row["model_pairwise_rank"] = base_row.get("rank")
        row["model_pairwise_score"] = components["model_pairwise"]["scores"].get(
            paper_id,
            0.0,
        )
        row["field_year_normalized_citation_score"] = components[
            "field_year_normalized_citation"
        ]["scores"].get(paper_id, 0.0)
        row["relative_citation_ratio_score"] = components[
            "relative_citation_ratio"
        ]["scores"].get(paper_id, 0.0)
        row["citation_graph_score"] = components["citation_graph"]["scores"].get(
            paper_id,
            0.0,
        )
        row["citation_context_quality_score"] = components[
            "citation_context_quality"
        ]["scores"].get(paper_id, 0.0)
        row["influential_citation_score"] = components[
            "influential_citation"
        ]["scores"].get(paper_id, 0.0)
        row["temporal_momentum_score"] = components["temporal_momentum"]["scores"].get(
            paper_id,
            0.0,
        )
        row["external_adoption_score"] = components["external_adoption"]["scores"].get(
            paper_id,
            0.0,
        )
        _attach_raw_component_values(row, paper_id, components)
        _attach_external_component_values(row, paper_id, components)

        available_components = [
            name
            for name in weights
            if paper_id in components.get(name, {}).get("available_ids", set())
        ]
        available_weight = sum(weights[name] for name in available_components)
        if available_weight > 0:
            normalized_weights = {
                name: (weights[name] / available_weight if name in available_components else 0.0)
                for name in weights
            }
            fused_score = sum(
                normalized_weights[name] * row[COMPONENT_SCORE_FIELDS[name]]
                for name in weights
            )
        else:
            normalized_weights = {name: 0.0 for name in weights}
            fused_score = 0.0

        row["impact_index_v2"] = round(fused_score, 6)
        row["final_impact_score"] = row["impact_index_v2"]
        row["fusion_signal_count"] = len(available_components)
        row["fusion_available_weight"] = round(available_weight, 6)
        for name in weights:
            row[f"fusion_weight_{name}"] = round(normalized_weights[name], 6)
        _attach_quality_layer(
            row,
            available_components=available_components,
            normalized_weights=normalized_weights,
        )
        quality_flag, quality_reason = _evidence_quality(
            signal_count=len(available_components),
            available_weight=available_weight,
            pairwise_comparisons=int(row.get("comparisons") or 0),
            uncertainty_band=row["uncertainty_band"],
        )
        row["evidence_quality_flag"] = quality_flag
        row["evidence_quality_reason"] = quality_reason
        reportability_status, reportability_reason = _reportability_status(
            signal_count=len(available_components),
            pairwise_comparisons=int(row.get("comparisons") or 0),
            uncertainty_band=row["uncertainty_band"],
        )
        row["legacy_reportability_status"] = reportability_status
        row["legacy_reportability_reason"] = reportability_reason
        row["reportability_status"] = reportability_status
        row["reportability_reason"] = reportability_reason
        _attach_evidence_first_layer(
            row,
            components=components,
            paper_id=paper_id,
            evidence_weights=evidence_weights,
            impact_index_v3_weights=impact_index_v3_weights,
        )
        row["reportability_status"], row["reportability_reason"] = (
            _v3_reportability_status(row)
        )
        row["reportability_basis"] = "impact_index_v3_evidence_first"
        _attach_credibility_adjusted_score(row)
        _attach_source_quality_gate(row, paper)
        _attach_reportability_gaps(row)
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["final_impact_score"]), row["paper_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    summary = _fusion_summary(
        papers,
        components,
        weights,
        evidence_weights=evidence_weights,
        impact_index_v3_weights=impact_index_v3_weights,
    )
    summary["reportability"] = _reportability_summary(rows)
    summary["evidence_first"] = _evidence_first_summary(rows)
    return {
        "summary": summary,
        "rankings": rows,
    }


def _impact_components(
    papers: list[Paper],
    pairwise_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baselines = compute_bibliometric_baselines(papers)
    internal_edge_count = _internal_reference_edge_count(papers)
    raw_citation_counts = {
        paper.paper_id: count
        for paper in papers
        if (count := _citation_count(paper)) is not None
    }
    raw_field_year = _field_year_normalized_citations(papers)
    relative_citation = _relative_citation_ratios(papers)
    citation_outliers = _citation_outliers(papers)
    effective_citations = _effective_citation_components(papers)
    raw_context = effective_citations["raw"]
    raw_influential = _influential_citation_counts(papers)
    raw_momentum = _temporal_momentum(papers)
    external = _external_impact_components(papers)

    pairwise_available_ids = {
        paper_id
        for paper_id, row in pairwise_by_id.items()
        if int(row.get("comparisons") or 0) > 0 and row.get("impact_index") is not None
    }
    citation_available_ids = set(raw_field_year)
    relative_citation_available_ids = set(relative_citation["ratio"])
    graph_available_ids = (
        {paper.paper_id for paper in papers} if internal_edge_count > 0 else set()
    )
    context_available_ids = (
        {paper.paper_id for paper in papers} if _corpus_context_count(papers) > 0 else set()
    )
    influential_available_ids = set(raw_influential)
    momentum_available_ids = set(raw_momentum)
    external_available_ids = set().union(
        external["field_year_percentile"].keys(),
        external["patent"].keys(),
        external["policy"].keys(),
        external["artifact"].keys(),
        external["media"].keys(),
        external["award"].keys(),
        external["bip"].keys(),
        external["life_science_translation"].keys(),
    )
    effective_citation_scores = _normalize_to_100(
        {
            paper_id: _robust_citation_value(value)
            for paper_id, value in raw_context.items()
        },
        papers,
    )

    return {
        "model_pairwise": {
            "scores": {
                paper.paper_id: _bounded_score(
                    pairwise_by_id.get(paper.paper_id, {}).get("impact_index"),
                )
                for paper in papers
            },
            "raw": {
                paper.paper_id: _bounded_score(
                    pairwise_by_id.get(paper.paper_id, {}).get("impact_index"),
                )
                for paper in papers
            },
            "available_ids": pairwise_available_ids,
            "reason": "Pairwise local model score is available when the paper has at least one decisive comparison; model-score-dataset uses order-independent Rank Centrality by default and retains Elo as an audit field.",
        },
        "field_year_normalized_citation": {
            "scores": _normalize_to_100(raw_field_year, papers),
            "raw": raw_field_year,
            "citation_counts": raw_citation_counts,
            "available_ids": citation_available_ids,
            "reason": "Citation counts are normalized inside each field/year bucket when citation_count, field, and year are available.",
        },
        "citation_outliers": {
            "rows": citation_outliers,
            "available_ids": set(citation_outliers),
            "reason": "Citation outlier diagnostics compare raw citation counts against the local field/year median.",
        },
        "relative_citation_ratio": {
            "scores": _normalize_to_100(relative_citation["ratio"], papers),
            "raw": relative_citation["ratio"],
            "citation_rates": relative_citation["citation_rate"],
            "expected_rates": relative_citation["expected_rate"],
            "available_ids": relative_citation_available_ids,
            "reason": "RCR-like local signal: annual citation rate divided by a supplied expected citation rate, falling back to the local field/year expected rate.",
        },
        "citation_graph": {
            "scores": _normalize_to_100(baselines.pagerank, papers)
            if internal_edge_count > 0
            else {paper.paper_id: 0.0 for paper in papers},
            "raw": baselines.pagerank if internal_edge_count > 0 else {},
            "available_ids": graph_available_ids,
            "reason": "Local PageRank is available when the supplied references contain edges between papers in the same dataset.",
        },
        "citation_context_quality": {
            "scores": effective_citation_scores,
            "raw": raw_context,
            "available_ids": context_available_ids,
            "reason": "Citation-context quality uses auditable effective-citation roles; life-science clinical, guideline, method, and data/tool reuse contexts receive stronger support than background mentions.",
        },
        "effective_citation": {
            "scores": effective_citation_scores,
            "raw": raw_context,
            "available_ids": {
                paper_id
                for paper_id, count in effective_citations["edge_count"].items()
                if count > 0
            },
            "reason": "Effective citations are incoming dataset citation edges classified by explicit citation contexts and biomedical role cues.",
            **effective_citations,
        },
        "influential_citation": {
            "scores": _normalize_to_100(
                {
                    paper_id: _robust_citation_value(value)
                    for paper_id, value in raw_influential.items()
                },
                papers,
            ),
            "raw": raw_influential,
            "available_ids": influential_available_ids,
            "reason": "Influential-citation evidence uses Semantic Scholar highly influential citation counts or compatible supplied fields.",
        },
        "temporal_momentum": {
            "scores": _normalize_to_100(raw_momentum, papers),
            "raw": raw_momentum,
            "available_ids": momentum_available_ids,
            "reason": "Temporal momentum uses recent yearly citation history when metadata contains citation_counts_by_year or compatible fields.",
        },
        "external_adoption": {
            "scores": _external_adoption_scores(external, papers),
            "raw": external["combined_raw"],
            "available_ids": external_available_ids,
            "reason": "External adoption uses optional field/year percentile, patent citations, policy citations, and software/data/model adoption metadata when supplied.",
            "field_year_percentile": external["field_year_percentile_scores"],
            "patent": external["patent_scores"],
            "policy": external["policy_scores"],
            "artifact": external["artifact_scores"],
            "media": external["media_scores"],
            "award": external["award_scores"],
            "bip": external["bip_scores"],
            "life_science_translation": external["life_science_translation_scores"],
            "raw_field_year_percentile": external["field_year_percentile"],
            "raw_patent": external["patent"],
            "raw_policy": external["policy"],
            "raw_artifact": external["artifact"],
            "raw_media": external["media"],
            "raw_award": external["award"],
            "raw_bip": external["bip"],
            "raw_life_science_translation": external["life_science_translation"],
        },
    }


def _attach_raw_component_values(
    row: dict[str, Any],
    paper_id: str,
    components: dict[str, dict[str, Any]],
) -> None:
    row["model_pairwise_raw"] = components["model_pairwise"]["raw"].get(paper_id, 0.0)
    row["field_year_normalized_citation_raw"] = components[
        "field_year_normalized_citation"
    ]["raw"].get(paper_id, 0.0)
    row["relative_citation_ratio_raw"] = components["relative_citation_ratio"][
        "raw"
    ].get(paper_id, 0.0)
    row["relative_citation_rate_raw"] = components["relative_citation_ratio"][
        "citation_rates"
    ].get(paper_id)
    row["relative_expected_citation_rate_raw"] = components["relative_citation_ratio"][
        "expected_rates"
    ].get(paper_id)
    row["citation_graph_raw"] = components["citation_graph"]["raw"].get(paper_id, 0.0)
    row["citation_context_quality_raw"] = components["citation_context_quality"][
        "raw"
    ].get(paper_id, 0.0)
    effective = components["effective_citation"]
    row["effective_citation_score"] = effective["scores"].get(paper_id, 0.0)
    row["effective_citation_raw"] = effective["raw"].get(paper_id, 0.0)
    row["effective_citation_edge_count"] = effective["edge_count"].get(paper_id, 0)
    row["effective_citation_context_count"] = effective["context_count"].get(paper_id, 0)
    row["effective_citation_count"] = effective["effective_count"].get(paper_id, 0)
    row["effective_citation_guideline_count"] = effective["guideline_count"].get(
        paper_id,
        0,
    )
    row["effective_citation_clinical_count"] = effective["clinical_count"].get(
        paper_id,
        0,
    )
    row["effective_citation_method_count"] = effective["method_count"].get(paper_id, 0)
    row["effective_citation_dataset_or_tool_count"] = effective[
        "dataset_or_tool_count"
    ].get(paper_id, 0)
    row["effective_citation_supporting_count"] = effective["supporting_count"].get(
        paper_id,
        0,
    )
    row["effective_citation_negative_count"] = effective[
        "negative_or_contrasting_count"
    ].get(paper_id, 0)
    row["effective_citation_background_count"] = effective["background_count"].get(
        paper_id,
        0,
    )
    row["effective_citation_unknown_count"] = effective["unknown_count"].get(
        paper_id,
        0,
    )
    row["effective_citation_coverage"] = effective["coverage"].get(
        paper_id,
        "missing",
    )
    row["effective_citation_evidence_level"] = effective["evidence_level"].get(
        paper_id,
        "missing",
    )
    row["influential_citation_count_raw"] = components["influential_citation"][
        "raw"
    ].get(paper_id)
    row["temporal_momentum_raw"] = components["temporal_momentum"]["raw"].get(
        paper_id,
        0.0,
    )
    row["external_adoption_raw"] = components["external_adoption"]["raw"].get(
        paper_id,
        0.0,
    )
    row["citation_count_raw"] = components["field_year_normalized_citation"][
        "citation_counts"
    ].get(paper_id)
    citation_outlier = components["citation_outliers"]["rows"].get(paper_id, {})
    row["citation_outlier_flag"] = bool(citation_outlier.get("citation_outlier_flag"))
    row["citation_outlier_ratio"] = citation_outlier.get("citation_outlier_ratio", 0.0)
    row["citation_outlier_basis"] = citation_outlier.get("citation_outlier_basis", "")


def _attach_external_component_values(
    row: dict[str, Any],
    paper_id: str,
    components: dict[str, dict[str, Any]],
) -> None:
    external = components["external_adoption"]
    row["external_field_year_percentile_score"] = external[
        "field_year_percentile"
    ].get(paper_id, 0.0)
    row["patent_impact_score"] = external["patent"].get(paper_id, 0.0)
    row["policy_impact_score"] = external["policy"].get(paper_id, 0.0)
    row["artifact_adoption_score"] = external["artifact"].get(paper_id, 0.0)
    row["media_impact_score"] = external["media"].get(paper_id, 0.0)
    row["award_impact_score"] = external["award"].get(paper_id, 0.0)
    row["bip_impact_score"] = external["bip"].get(paper_id, 0.0)
    row["external_field_year_percentile_raw"] = external[
        "raw_field_year_percentile"
    ].get(paper_id)
    row["patent_citations_raw"] = external["raw_patent"].get(paper_id)
    row["policy_citations_raw"] = external["raw_policy"].get(paper_id)
    row["artifact_adoption_raw"] = external["raw_artifact"].get(paper_id)
    row["media_mentions_raw"] = external["raw_media"].get(paper_id)
    row["award_count_raw"] = external["raw_award"].get(paper_id)
    row["bip_impact_raw"] = external["raw_bip"].get(paper_id)
    row["life_science_translation_score"] = external[
        "life_science_translation"
    ].get(paper_id, 0.0)
    row["life_science_translation_raw"] = external[
        "raw_life_science_translation"
    ].get(paper_id)


def _fusion_summary(
    papers: list[Paper],
    components: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    evidence_weights: dict[str, float],
    impact_index_v3_weights: dict[str, float],
) -> dict[str, Any]:
    paper_count = len(papers)
    return {
        "version": "impact_index_v3_evidence_first",
        "score_range": "0-100 dataset-local fused impact score",
        "base_weights": weights,
        "evidence_first_weights": evidence_weights,
        "impact_index_v3_weights": impact_index_v3_weights,
        "component_score_fields": COMPONENT_SCORE_FIELDS,
        "signal_availability": {
            name: {
                "weight": weights.get(name, EVIDENCE_FIRST_WEIGHTS.get(name, 0.0)),
                "available": bool(values["available_ids"]),
                "available_paper_count": len(values["available_ids"]),
                "paper_count": paper_count,
                "coverage_rate": round(
                    len(values["available_ids"]) / paper_count if paper_count else 0.0,
                    6,
                ),
                "reason": values["reason"],
            }
            for name, values in components.items()
            if name in set(weights) | set(EVIDENCE_FIRST_WEIGHTS)
        },
        "missing_signal_policy": "Per paper, missing signal weights are redistributed across available signals instead of being treated as zero.",
        "claim_scope": "dataset-local influence inside the supplied corpus; not global scientific impact unless external graph calibration is supplied.",
    }


def _attach_evidence_first_layer(
    row: dict[str, Any],
    *,
    components: dict[str, dict[str, Any]],
    paper_id: str,
    evidence_weights: dict[str, float],
    impact_index_v3_weights: dict[str, float],
) -> None:
    available_evidence = [
        name
        for name, weight in evidence_weights.items()
        if paper_id in components.get(name, {}).get("available_ids", set())
        and weight > 0.0
    ]
    evidence_supported = _weighted_score(
        row,
        weights=evidence_weights,
        available_components=available_evidence,
    )
    model_supported = float(row.get("model_pairwise_score", 0.0))
    v3_score = (
        impact_index_v3_weights.get("evidence_supported", 0.0) * evidence_supported
        + impact_index_v3_weights.get("model_supported", 0.0) * model_supported
    )
    if not available_evidence:
        v3_score = model_supported

    row["evidence_supported_score"] = round(evidence_supported, 6)
    row["model_supported_score"] = round(model_supported, 6)
    row["impact_index_v3"] = round(v3_score, 6)
    row["final_impact_score"] = row["impact_index_v3"]
    row["evidence_first_signal_count"] = len(available_evidence)
    positive_signal_count = _positive_evidence_signal_count(row, available_evidence)
    row["positive_evidence_signal_count"] = positive_signal_count
    row["zero_evidence_signal_count"] = _zero_evidence_signal_count(
        row,
        available_evidence,
    )
    for name, weight in evidence_weights.items():
        row[f"evidence_first_weight_{name}"] = round(float(weight), 6)
    for name, weight in impact_index_v3_weights.items():
        row[f"impact_index_v3_weight_{name}"] = round(float(weight), 6)
    row["credible_evidence_tier"] = _credible_evidence_tier(row, available_evidence)
    row["dominant_score_family"] = (
        "evidence_supported"
        if evidence_supported >= model_supported
        else "model_assisted"
    )
    row["missing_high_value_evidence"] = _missing_high_value_evidence(row)


def _evidence_first_weights_from_fusion(
    fusion_weights: dict[str, float],
) -> dict[str, float]:
    default_v2_profile = (
        set(fusion_weights) == set(IMPACT_INDEX_V2_WEIGHTS)
        and all(
            math.isclose(
                float(fusion_weights.get(name, -1.0)),
                float(default_weight),
            )
            for name, default_weight in IMPACT_INDEX_V2_WEIGHTS.items()
        )
    )
    evidence_weights: dict[str, float] = {}
    for name, default_weight in EVIDENCE_FIRST_WEIGHTS.items():
        if name == "external_adoption":
            continue
        if name in fusion_weights:
            evidence_weights[name] = float(fusion_weights.get(name, 0.0))
        else:
            evidence_weights[name] = float(default_weight) if default_v2_profile else 0.0
    evidence_weights["external_adoption"] = EVIDENCE_FIRST_WEIGHTS["external_adoption"]
    return evidence_weights


def _weighted_score(
    row: dict[str, Any],
    *,
    weights: dict[str, float],
    available_components: list[str],
) -> float:
    available_weight = sum(weights[component] for component in available_components)
    if available_weight <= 0.0:
        return 0.0
    return sum(
        weights[component] / available_weight * float(row[COMPONENT_SCORE_FIELDS[component]])
        for component in available_components
    )


def _credible_evidence_tier(row: dict[str, Any], available_evidence: list[str]) -> str:
    external_count = _positive_external_evidence_count(row)
    positive_signal_count = _positive_evidence_signal_count(row, available_evidence)
    if positive_signal_count >= 4 and external_count >= 3:
        return "very_high"
    if external_count >= 3 and positive_signal_count >= 2:
        return "high"
    if positive_signal_count >= 3 and external_count >= 1:
        return "high"
    if positive_signal_count >= 2:
        return "moderate"
    if positive_signal_count == 1 and external_count >= 1:
        return "moderate"
    if positive_signal_count == 1:
        return "low"
    return "model_only"


def _positive_evidence_signal_count(
    row: dict[str, Any],
    available_evidence: list[str],
) -> int:
    return sum(
        1
        for component in available_evidence
        if _numeric_or_none(row.get(COMPONENT_SCORE_FIELDS[component])) is not None
        and float(row.get(COMPONENT_SCORE_FIELDS[component]) or 0.0) > 0.0
    )


def _zero_evidence_signal_count(
    row: dict[str, Any],
    available_evidence: list[str],
) -> int:
    return sum(
        1
        for component in available_evidence
        if _numeric_or_none(row.get(COMPONENT_SCORE_FIELDS[component])) is not None
        and float(row.get(COMPONENT_SCORE_FIELDS[component]) or 0.0) == 0.0
    )


def _positive_external_evidence_count(row: dict[str, Any]) -> int:
    return sum(
        1
        for field in (
            "external_field_year_percentile_raw",
            "patent_citations_raw",
            "policy_citations_raw",
            "artifact_adoption_raw",
            "media_mentions_raw",
            "award_count_raw",
            "life_science_translation_raw",
        )
        if (value := _numeric_or_none(row.get(field))) is not None and value > 0.0
    )


def _missing_high_value_evidence(row: dict[str, Any]) -> list[str]:
    missing = []
    if row.get("external_field_year_percentile_raw") is None:
        missing.append("external_field_year_percentile")
    if _numeric_or_none(row.get("patent_citations_raw")) is None or float(row.get("patent_citations_raw") or 0.0) <= 0.0:
        missing.append("patent")
    if _numeric_or_none(row.get("policy_citations_raw")) is None or float(row.get("policy_citations_raw") or 0.0) <= 0.0:
        missing.append("policy")
    if _numeric_or_none(row.get("artifact_adoption_raw")) is None or float(row.get("artifact_adoption_raw") or 0.0) <= 0.0:
        missing.append("artifact_adoption")
    return missing


def _v3_reportability_status(row: dict[str, Any]) -> tuple[str, str]:
    tier = str(row.get("credible_evidence_tier", "model_only"))
    dominant_family = str(row.get("dominant_score_family", ""))
    missing = row.get("missing_high_value_evidence", [])
    blocking_missing = _blocking_reportability_missing_evidence(missing)

    if tier == "very_high":
        return (
            "reportable",
            "Very high evidence tier under the evidence-first impact index.",
        )
    if tier == "high" and dominant_family == "evidence_supported" and not blocking_missing:
        if missing:
            return (
                "reportable",
                "High evidence tier and evidence-supported final score; disclose optional missing adoption evidence.",
            )
        return (
            "reportable",
            "High evidence tier, evidence-supported final score, and no missing high-value external evidence.",
        )
    if tier in {"high", "moderate"} and dominant_family == "evidence_supported":
        return (
            "exploratory",
            "Evidence supports the score, but the evidence tier or missing high-value evidence requires disclosure.",
        )
    if tier in {"high", "moderate"}:
        return (
            "exploratory",
            "Some evidence is available, but the final score is still model-assisted or incomplete.",
        )
    return (
        "low_confidence",
        "Too little positive evidence is available for a reportable impact score.",
    )


def _blocking_reportability_missing_evidence(missing: Any) -> list[str]:
    if not isinstance(missing, list):
        return []
    optional_adoption_evidence = {"patent", "policy", "artifact_adoption"}
    return [str(item) for item in missing if str(item) not in optional_adoption_evidence]


def _attach_credibility_adjusted_score(row: dict[str, Any]) -> None:
    factor, reason = _credibility_adjustment(row)
    adjusted = float(row.get("impact_index_v3", 0.0)) * factor
    row["credibility_adjusted_impact_score"] = round(adjusted, 6)
    row["credibility_adjustment_factor"] = factor
    row["credibility_adjustment_reason"] = reason
    row["final_impact_score"] = row["credibility_adjusted_impact_score"]


def _credibility_adjustment(row: dict[str, Any]) -> tuple[float, str]:
    status = str(row.get("reportability_status", "low_confidence"))
    tier = str(row.get("credible_evidence_tier", "model_only"))
    dominant_family = str(row.get("dominant_score_family", "model_assisted"))
    missing = row.get("missing_high_value_evidence", [])
    missing_count = len(missing) if isinstance(missing, list) else 0

    if status == "reportable":
        return 1.0, "Reportable evidence-first score; no credibility discount applied."
    if status == "exploratory" and tier == "high":
        return 0.85, "High evidence tier but still exploratory, so a modest credibility discount is applied."
    if status == "exploratory" and tier == "moderate":
        return 0.70, "Moderate evidence tier; final score is discounted for incomplete support."
    if dominant_family == "model_assisted" or tier == "model_only":
        return 0.45, "Model-assisted or model-only score; strong credibility discount applied."
    if missing_count >= 3:
        return 0.55, "Multiple high-value evidence families are missing."
    return 0.60, "Low-confidence evidence profile; conservative credibility discount applied."


def _attach_source_quality_gate(row: dict[str, Any], paper: Paper) -> None:
    source_kind = str(paper.metadata.get("source_kind", "") or "")
    row["source_kind"] = source_kind
    if _is_non_article_source(paper, source_kind):
        row["source_quality_flag"] = "non_article"
        row["source_quality_adjustment_factor"] = 0.0
        row["source_quality_reason"] = (
            "Record appears to be supplementary or non-article material; excluded "
            "from reportable paper impact ranking."
        )
        row["pre_source_quality_final_impact_score"] = row.get("final_impact_score", 0.0)
        row["final_impact_score"] = 0.0
        row["reportability_status"] = "excluded_non_article"
        row["reportability_reason"] = row["source_quality_reason"]
        return

    row["source_quality_flag"] = "article"
    row["source_quality_adjustment_factor"] = 1.0
    row["source_quality_reason"] = "Record is treated as a candidate article."
    row["pre_source_quality_final_impact_score"] = row.get("final_impact_score", 0.0)


def _attach_reportability_gaps(row: dict[str, Any]) -> None:
    actions = _reportability_gap_actions(row)
    row["reportability_gap_actions"] = actions
    row["reportability_gap_count"] = len(actions)


def _reportability_gap_actions(row: dict[str, Any]) -> list[str]:
    status = str(row.get("reportability_status", ""))
    if status == "reportable":
        return []
    if status == "excluded_non_article":
        return [
            "Use article/full-paper records; remove supplementary or non-article records from public ranking."
        ]

    actions: list[str] = []
    tier = str(row.get("credible_evidence_tier", "model_only"))
    dominant_family = str(row.get("dominant_score_family", "model_assisted"))
    positive_count = int(row.get("positive_evidence_signal_count") or 0)
    if positive_count < 2:
        actions.append("Add at least two positive non-model evidence signals.")
    if tier not in {"high", "very_high"}:
        actions.append("Strengthen evidence tier to high or very_high.")
    if dominant_family != "evidence_supported":
        actions.append("Increase evidence-supported signals so the final score is not model-assisted.")

    missing = row.get("missing_high_value_evidence", [])
    if not isinstance(missing, list):
        missing = []
    missing_actions = {
        "external_field_year_percentile": "Add field/year percentile evidence.",
        "patent": "Add patent impact evidence.",
        "policy": "Add policy impact evidence.",
        "artifact_adoption": "Add artifact or dataset/software adoption evidence.",
    }
    for item in missing:
        action = missing_actions.get(str(item))
        if action and action not in actions:
            actions.append(action)
    return actions


def _is_non_article_source(paper: Paper, source_kind: str) -> bool:
    title = str(paper.title or "").strip().lower()
    if source_kind == "supplement_or_extracted_supplement":
        return True
    return title.startswith(
        (
            "supplementary ",
            "supplemental ",
            "supporting information",
            "supplementary online content",
            "supplemental online content",
        )
    )


def _field_year_normalized_citations(papers: list[Paper]) -> dict[str, float]:
    counts = {
        paper.paper_id: count
        for paper in papers
        if (count := _citation_count(paper)) is not None
        and paper.field is not None
        and paper.year is not None
    }
    buckets: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for paper in papers:
        count = counts.get(paper.paper_id)
        if count is None or paper.field is None or paper.year is None:
            continue
        buckets[(paper.field, int(paper.year))].append(_robust_citation_value(count))

    means = {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if values and sum(values) > 0
    }
    scores: dict[str, float] = {}
    for paper in papers:
        count = counts.get(paper.paper_id)
        if count is None or paper.field is None or paper.year is None:
            continue
        mean = means.get((paper.field, int(paper.year)))
        if mean is None or mean <= 0.0:
            continue
        scores[paper.paper_id] = _robust_citation_value(count) / mean
    return scores


def _relative_citation_ratios(papers: list[Paper]) -> dict[str, dict[str, float]]:
    citation_rates: dict[str, float] = {}
    supplied_ratios: dict[str, float] = {}
    supplied_expected_rates: dict[str, float] = {}
    buckets: defaultdict[tuple[str, int], list[float]] = defaultdict(list)

    for paper in papers:
        supplied_ratio = _external_number(
            paper,
            "relative_citation_ratio",
            "nih_rcr",
            "rcr",
            "icite_rcr",
            "relative_citation_ratio_value",
        )
        if supplied_ratio is not None:
            supplied_ratios[paper.paper_id] = supplied_ratio
        count = _citation_count(paper)
        if count is None or paper.year is None:
            continue
        observed_year = _citation_observation_year(paper, papers)
        age_years = max(observed_year - int(paper.year) + 1, 1)
        citation_rate = float(count) / age_years
        citation_rates[paper.paper_id] = citation_rate
        expected_rate = _external_number(
            paper,
            "expected_citation_rate",
            "rcr_expected_citation_rate",
            "field_expected_citation_rate",
            "field_citation_rate",
        )
        if expected_rate is not None and expected_rate > 0.0:
            supplied_expected_rates[paper.paper_id] = expected_rate
        if paper.field is not None:
            buckets[(str(paper.field), int(paper.year))].append(citation_rate)

    local_expected_rates = {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if len(values) >= 5 and sum(values) > 0.0
    }

    ratios: dict[str, float] = {}
    expected_rates: dict[str, float] = {}
    for paper in papers:
        supplied_ratio = supplied_ratios.get(paper.paper_id)
        if supplied_ratio is not None:
            ratios[paper.paper_id] = supplied_ratio
            continue
        citation_rate = citation_rates.get(paper.paper_id)
        if citation_rate is None:
            continue
        expected_rate = supplied_expected_rates.get(paper.paper_id)
        if expected_rate is None and paper.field is not None and paper.year is not None:
            expected_rate = local_expected_rates.get((str(paper.field), int(paper.year)))
        if expected_rate is None or expected_rate <= 0.0:
            continue
        expected_rates[paper.paper_id] = expected_rate
        ratios[paper.paper_id] = citation_rate / expected_rate

    return {
        "ratio": ratios,
        "citation_rate": citation_rates,
        "expected_rate": expected_rates,
    }


def _citation_outliers(papers: list[Paper]) -> dict[str, dict[str, Any]]:
    counts = {
        paper.paper_id: count
        for paper in papers
        if (count := _citation_count(paper)) is not None
        and paper.field is not None
        and paper.year is not None
    }
    buckets: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for paper in papers:
        count = counts.get(paper.paper_id)
        if count is None or paper.field is None or paper.year is None:
            continue
        buckets[(str(paper.field), int(paper.year))].append(float(count))

    medians = {
        key: _median(values)
        for key, values in buckets.items()
        if len(values) >= 3 and _median(values) > 0.0
    }
    rows: dict[str, dict[str, Any]] = {}
    for paper in papers:
        count = counts.get(paper.paper_id)
        if count is None or paper.field is None or paper.year is None:
            continue
        median = medians.get((str(paper.field), int(paper.year)))
        if median is None or median <= 0.0:
            continue
        ratio = float(count) / median
        rows[paper.paper_id] = {
            "citation_outlier_flag": ratio >= 10.0 and float(count) >= 100.0,
            "citation_outlier_ratio": round(ratio, 6),
            "citation_outlier_basis": "field_year_median_citation_count",
        }
    return rows


def _influential_citation_counts(papers: list[Paper]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for paper in papers:
        count = _external_number(
            paper,
            "semantic_scholar_influential_citation_count",
            "s2_influential_citation_count",
            "influential_citation_count",
            "highly_influential_citation_count",
        )
        if count is not None:
            counts[paper.paper_id] = count
    return counts


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    length = len(ordered)
    midpoint = length // 2
    if length == 0:
        return 0.0
    if length % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _citation_observation_year(paper: Paper, papers: list[Paper]) -> int:
    explicit = _external_number(
        paper,
        "citation_observation_year",
        "observation_year",
        "citation_year",
    )
    if explicit is not None and explicit > 0:
        return int(explicit)
    years = [int(candidate.year) for candidate in papers if candidate.year is not None]
    if years:
        return max(years)
    return int(paper.year or 0)


def _external_impact_components(papers: list[Paper]) -> dict[str, dict[str, float]]:
    field_year_percentile: dict[str, float] = {}
    patent: dict[str, float] = {}
    policy: dict[str, float] = {}
    artifact: dict[str, float] = {}
    media: dict[str, float] = {}
    award: dict[str, float] = {}
    bip: dict[str, float] = {}
    life_science_translation: dict[str, float] = {}

    for paper in papers:
        percentile = _external_number(
            paper,
            "field_year_percentile",
            "field_year_citation_percentile",
            "citation_percentile",
            "openalex_percentile",
        )
        if percentile is not None:
            field_year_percentile[paper.paper_id] = _percentile_to_100(percentile)

        patent_count = _external_number(
            paper,
            "patent_citations",
            "patent_citation_count",
            "patent_count",
        )
        if patent_count is not None:
            patent[paper.paper_id] = patent_count

        policy_count = _external_number(
            paper,
            "policy_citations",
            "policy_citation_count",
            "policy_mentions",
            "guideline_citations",
            "clinical_guideline_citations",
            "practice_guideline_citations",
            "clinical_guideline_mentions",
            "guideline_mentions",
            "consensus_guideline_citations",
        )
        if policy_count is not None:
            policy[paper.paper_id] = policy_count

        artifact_count = _artifact_adoption_count(paper)
        if artifact_count is not None:
            artifact[paper.paper_id] = artifact_count

        translation_count = _life_science_translation_count(paper)
        if translation_count is not None:
            life_science_translation[paper.paper_id] = translation_count

        media_count = _external_number(
            paper,
            "media_mentions",
            "news_mentions",
            "press_mentions",
            "altmetric_mentions",
        )
        if media_count is not None:
            media[paper.paper_id] = media_count

        award_count = _external_number(
            paper,
            "award_count",
            "awards",
            "best_paper_awards",
            "prize_count",
        )
        if award_count is not None:
            award[paper.paper_id] = award_count

        bip_percentile = _external_number(
            paper,
            "bip_impact_percentile",
            "bip_percentile",
            "bip_field_year_percentile",
        )
        if bip_percentile is not None:
            bip[paper.paper_id] = _percentile_to_100(bip_percentile)
        else:
            bip_score = _external_number(
                paper,
                "bip_impact_score",
                "bip_score",
                "bip_influence_score",
                "bip_impact_class",
                "bip_class",
                "bip_pagerank",
                "bip_page_rank",
                "bip_att_rank",
                "bip_attrank",
                "bip_ram",
                "bip_ram_score",
            )
            if bip_score is not None:
                bip[paper.paper_id] = bip_score

    field_year_percentile_scores = dict(field_year_percentile)
    patent_scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in patent.items()},
        papers,
    )
    policy_scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in policy.items()},
        papers,
    )
    artifact_scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in artifact.items()},
        papers,
    )
    media_scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in media.items()},
        papers,
    )
    award_scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in award.items()},
        papers,
    )
    bip_scores = dict(bip)
    life_science_translation_scores = _normalize_to_100(
        {
            paper_id: _robust_citation_value(value)
            for paper_id, value in life_science_translation.items()
        },
        papers,
    )
    combined_raw = {
        paper.paper_id: sum(
            value
            for value in (
                field_year_percentile.get(paper.paper_id),
                patent.get(paper.paper_id),
                policy.get(paper.paper_id),
                artifact.get(paper.paper_id),
                media.get(paper.paper_id),
                award.get(paper.paper_id),
                bip.get(paper.paper_id),
                life_science_translation.get(paper.paper_id),
            )
            if value is not None
        )
        for paper in papers
    }
    return {
        "field_year_percentile": field_year_percentile,
        "patent": patent,
        "policy": policy,
        "artifact": artifact,
        "media": media,
        "award": award,
        "bip": bip,
        "life_science_translation": life_science_translation,
        "field_year_percentile_scores": field_year_percentile_scores,
        "patent_scores": patent_scores,
        "policy_scores": policy_scores,
        "artifact_scores": artifact_scores,
        "media_scores": media_scores,
        "award_scores": award_scores,
        "bip_scores": bip_scores,
        "life_science_translation_scores": life_science_translation_scores,
        "combined_raw": combined_raw,
    }


def _external_adoption_scores(
    external: dict[str, dict[str, float]],
    papers: list[Paper],
) -> dict[str, float]:
    weights = {
        "field_year_percentile_scores": 0.30,
        "patent_scores": 0.15,
        "policy_scores": 0.15,
        "artifact_scores": 0.20,
        "life_science_translation_scores": 0.25,
        "media_scores": 0.10,
        "award_scores": 0.10,
        "bip_scores": 0.20,
    }
    raw_sources = {
        "field_year_percentile_scores": "field_year_percentile",
        "patent_scores": "patent",
        "policy_scores": "policy",
        "artifact_scores": "artifact",
        "life_science_translation_scores": "life_science_translation",
        "media_scores": "media",
        "award_scores": "award",
        "bip_scores": "bip",
    }
    scores: dict[str, float] = {}
    for paper in papers:
        available = [
            name
            for name in weights
            if paper.paper_id in external.get(raw_sources[name], {})
        ]
        total_weight = sum(weights[name] for name in available)
        if total_weight <= 0.0:
            continue
        scores[paper.paper_id] = sum(
            weights[name] / total_weight * external[name].get(paper.paper_id, 0.0)
            for name in available
        )
    return scores


def _artifact_adoption_count(paper: Paper) -> float | None:
    keys = (
        "artifact_adoption",
        "artifact_adoption_count",
        "software_mentions",
        "software_citations",
        "github_stars",
        "repository_stars",
        "dataset_reuses",
        "dataset_citations",
        "model_downloads",
        "dataset_downloads",
        "huggingface_downloads",
        "protocol_reuses",
        "protocol_citations",
        "assay_reuses",
        "assay_citations",
        "reagent_reuses",
        "reagent_citations",
        "reagent_or_assay_reuses",
        "antibody_reuses",
        "plasmid_reuses",
        "cell_line_reuses",
        "mouse_model_reuses",
        "biobank_reuses",
    )
    values = [
        value
        for key in keys
        if (value := _external_number(paper, key)) is not None
    ]
    if not values:
        return None
    return sum(values)


def _life_science_translation_count(paper: Paper) -> float | None:
    keys = (
        "life_science_translation",
        "life_science_translation_count",
        "biomedical_translation_count",
        "clinical_trial_citations",
        "clinical_trial_mentions",
        "clinical_study_citations",
        "trial_registry_citations",
        "guideline_citations",
        "clinical_guideline_citations",
        "practice_guideline_citations",
        "clinical_guideline_mentions",
        "guideline_mentions",
        "consensus_guideline_citations",
        "drug_approval_mentions",
        "fda_label_mentions",
        "ema_label_mentions",
        "diagnostic_guideline_mentions",
        "therapeutic_guideline_mentions",
    )
    values = [
        value
        for key in keys
        if (value := _external_number(paper, key)) is not None
    ]
    if not values:
        return None
    return sum(values)


def _external_number(paper: Paper, *keys: str) -> float | None:
    sources = [
        paper.metadata.get("external_impact"),
        paper.metadata.get("impact"),
        paper.metadata.get("metrics"),
        paper.metadata,
        getattr(paper, "__pydantic_extra__", None) or {},
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = _numeric_or_none(source.get(key))
            if value is not None:
                return value
    return None


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number, 0.0)


def _percentile_to_100(value: float) -> float:
    if value <= 1.0:
        return value * 100.0
    return min(value, 100.0)


def _robust_citation_value(count: float) -> float:
    return math.log1p(max(float(count), 0.0))


def _effective_citation_components(papers: list[Paper]) -> dict[str, dict[str, Any]]:
    paper_ids = {paper.paper_id for paper in papers}
    raw: defaultdict[str, float] = defaultdict(float)
    edge_count: defaultdict[str, int] = defaultdict(int)
    context_count: defaultdict[str, int] = defaultdict(int)
    effective_count: defaultdict[str, int] = defaultdict(int)
    role_counts = {
        role: defaultdict(int)
        for role in EFFECTIVE_CITATION_ROLE_WEIGHTS
    }

    for citing_paper in papers:
        contexts_by_reference = citation_contexts_by_reference(citing_paper)
        seen_targets: set[str] = set()
        for reference in citing_paper.references:
            target_id = reference.reference_id
            if target_id not in paper_ids or target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            edge_count[target_id] += 1
            contexts = contexts_by_reference.get(target_id, [])
            context_count[target_id] += len(contexts)
            if not contexts:
                role_counts["unknown"][target_id] += 1
                raw[target_id] += EFFECTIVE_CITATION_ROLE_WEIGHTS["unknown"]
                continue

            edge_roles: set[str] = set()
            edge_weight = 0.0
            for context in contexts:
                roles = _life_science_context_roles(context)
                for role in roles:
                    role_counts[role][target_id] += 1
                    edge_roles.add(role)
                    edge_weight = max(
                        edge_weight,
                        EFFECTIVE_CITATION_ROLE_WEIGHTS[role],
                    )
            raw[target_id] += edge_weight
            if edge_roles & STRONG_EFFECTIVE_CITATION_ROLES:
                effective_count[target_id] += 1

    paper_id_list = [paper.paper_id for paper in papers]
    scores = _normalize_to_100(
        {paper_id: _robust_citation_value(value) for paper_id, value in raw.items()},
        papers,
    )
    return {
        "raw": {paper_id: round(float(raw.get(paper_id, 0.0)), 6) for paper_id in paper_id_list},
        "scores": scores,
        "edge_count": {paper_id: int(edge_count.get(paper_id, 0)) for paper_id in paper_id_list},
        "context_count": {
            paper_id: int(context_count.get(paper_id, 0)) for paper_id in paper_id_list
        },
        "effective_count": {
            paper_id: int(effective_count.get(paper_id, 0)) for paper_id in paper_id_list
        },
        **{
            f"{role}_count": {
                paper_id: int(counts.get(paper_id, 0))
                for paper_id in paper_id_list
            }
            for role, counts in role_counts.items()
        },
        "coverage": {
            paper_id: _effective_citation_coverage(
                edge_count.get(paper_id, 0),
                context_count.get(paper_id, 0),
            )
            for paper_id in paper_id_list
        },
        "evidence_level": {
            paper_id: _effective_citation_evidence_level(
                edge_count=edge_count.get(paper_id, 0),
                effective_count=effective_count.get(paper_id, 0),
                guideline_count=role_counts["guideline"].get(paper_id, 0),
                clinical_count=role_counts["clinical"].get(paper_id, 0),
                method_count=role_counts["method"].get(paper_id, 0),
                dataset_count=role_counts["dataset_or_tool"].get(paper_id, 0),
            )
            for paper_id in paper_id_list
        },
    }


def _life_science_context_roles(context: CitationContext) -> list[str]:
    explicit = _explicit_context_role(context)
    if explicit is not None:
        return [explicit]

    text = f"{context.section or ''} {context.sentence} {context.paragraph or ''}".lower()
    if _contains_any(
        text,
        (
            "guideline",
            "guidelines",
            "practice recommendation",
            "consensus statement",
            "nccn",
            "asco",
            "esmo",
            "nice",
            "cdc",
            "who recommendation",
            "fda guidance",
        ),
    ):
        return ["guideline"]
    if _contains_any(
        text,
        (
            "contradict",
            "contrary",
            "inconsistent with",
            "failed to reproduce",
            "failed to replicate",
            "not reproduce",
            "not replicated",
            "refute",
            "conflicting",
            "no association",
        ),
    ):
        return ["negative_or_contrasting"]

    roles: list[str] = []
    if _contains_any(
        text,
        (
            "clinical trial",
            "randomized",
            "randomised",
            "phase 1",
            "phase i",
            "phase 2",
            "phase ii",
            "phase 3",
            "phase iii",
            "patient",
            "patients",
            "cohort",
            "therapy",
            "treatment",
            "diagnostic",
            "diagnosis",
            "prognostic",
            "survival",
        ),
    ):
        roles.append("clinical")
    if _contains_any(
        text,
        (
            "method",
            "methods",
            "protocol",
            "assay",
            "reagent",
            "antibody",
            "plasmid",
            "cell line",
            "mouse model",
            "crispr",
            "sequencing",
            "pcr",
            "elisa",
            "western blot",
            "flow cytometry",
            "staining",
        ),
    ):
        roles.append("method")
    if _contains_any(
        text,
        (
            "dataset",
            "database",
            "software",
            "pipeline",
            "tool",
            "code",
            "resource",
            "benchmark",
            "repository",
            "accession",
            "downloaded from",
        ),
    ):
        roles.append("dataset_or_tool")
    if _contains_any(
        text,
        (
            "support",
            "supports",
            "consistent with",
            "confirmed",
            "demonstrated",
            "showed",
            "shown",
            "reported",
            "found",
            "evidence",
        ),
    ):
        roles.append("supporting")

    return roles or ["background"]


def _explicit_context_role(context: CitationContext) -> str | None:
    extra = getattr(context, "__pydantic_extra__", None) or {}
    value = None
    for key in ("role", "label", "intent", "citation_intent", "category"):
        if extra.get(key):
            value = str(extra[key]).strip().lower()
            break
    if not value:
        return None
    if value in {"method", "methods", "uses", "extends", "reuse"}:
        return "method"
    if value in {"result", "results", "supporting", "support"}:
        return "supporting"
    if value in {"background", "literature", "review"}:
        return "background"
    if value in {"clinical", "clinical_trial", "trial", "translational"}:
        return "clinical"
    if value in {"guideline", "policy", "recommendation"}:
        return "guideline"
    if value in {"dataset", "data", "tool", "software", "resource"}:
        return "dataset_or_tool"
    if value in {"negative", "contrasting", "disputing", "refuting"}:
        return "negative_or_contrasting"
    return None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _effective_citation_coverage(edge_count: int, context_count: int) -> str:
    if edge_count <= 0:
        return "missing"
    if context_count <= 0:
        return "edge_only"
    if context_count >= edge_count:
        return "contextualized"
    return "partial_context"


def _effective_citation_evidence_level(
    *,
    edge_count: int,
    effective_count: int,
    guideline_count: int,
    clinical_count: int,
    method_count: int,
    dataset_count: int,
) -> str:
    if edge_count <= 0:
        return "missing"
    if guideline_count > 0 or clinical_count > 0 or method_count > 0 or dataset_count > 0:
        return "high"
    if effective_count >= 2:
        return "moderate"
    if effective_count == 1:
        return "low"
    return "weak_background_or_unknown"


def _incoming_context_quality(papers: list[Paper]) -> dict[str, float]:
    paper_ids = {paper.paper_id for paper in papers}
    scores: defaultdict[str, float] = defaultdict(float)
    for citing_paper in papers:
        for reference_id, contexts in citation_contexts_by_reference(citing_paper).items():
            if reference_id not in paper_ids:
                continue
            scores[reference_id] += _context_quality_credit(contexts)
    return dict(scores)


def _context_quality_credit(contexts: list[CitationContext]) -> float:
    if not contexts:
        return 0.0
    return sum(max(context_weight([context]) - 1.0, 0.0) for context in contexts)


def _attach_quality_layer(
    row: dict[str, Any],
    *,
    available_components: list[str],
    normalized_weights: dict[str, float],
) -> None:
    contributions = {
        name: normalized_weights[name] * float(row[COMPONENT_SCORE_FIELDS[name]])
        for name in normalized_weights
    }
    total = sum(contributions.values())
    if contributions:
        dominant_signal, dominant_contribution = max(
            contributions.items(),
            key=lambda item: (item[1], item[0]),
        )
    else:
        dominant_signal, dominant_contribution = "", 0.0
    row["dominant_signal"] = dominant_signal
    row["max_signal_contribution_share"] = round(
        dominant_contribution / total if total > 0.0 else 0.0,
        6,
    )

    leave_one_scores = [
        _fused_score_without_component(
            row,
            available_components=available_components,
            base_weights=normalized_weights,
            omitted_component=component,
        )
        for component in available_components
    ]
    interval_values = [float(row["impact_index_v2"]), *leave_one_scores]
    ci_low = min(interval_values) if interval_values else float(row["impact_index_v2"])
    ci_high = max(interval_values) if interval_values else float(row["impact_index_v2"])
    ci_width = ci_high - ci_low
    row["score_ci_low"] = round(ci_low, 6)
    row["score_ci_high"] = round(ci_high, 6)
    row["score_ci_width"] = round(ci_width, 6)
    row["score_interval_method"] = "leave_one_signal_out"
    row["uncertainty_band"] = _uncertainty_band(
        signal_count=len(available_components),
        ci_width=ci_width,
        max_signal_contribution_share=float(row["max_signal_contribution_share"]),
    )


def _fused_score_without_component(
    row: dict[str, Any],
    *,
    available_components: list[str],
    base_weights: dict[str, float],
    omitted_component: str,
) -> float:
    remaining = [
        component
        for component in available_components
        if component != omitted_component and base_weights.get(component, 0.0) > 0.0
    ]
    total_weight = sum(base_weights[component] for component in remaining)
    if total_weight <= 0.0:
        return 0.0
    return sum(
        (base_weights[component] / total_weight)
        * float(row[COMPONENT_SCORE_FIELDS[component]])
        for component in remaining
    )


def _uncertainty_band(
    *,
    signal_count: int,
    ci_width: float,
    max_signal_contribution_share: float,
) -> str:
    if signal_count <= 1 or max_signal_contribution_share >= 0.85 or ci_width >= 50.0:
        return "high"
    if signal_count < 4 or max_signal_contribution_share >= 0.65 or ci_width >= 25.0:
        return "medium"
    return "low"


def _reportability_status(
    *,
    signal_count: int,
    pairwise_comparisons: int,
    uncertainty_band: str,
) -> tuple[str, str]:
    if signal_count <= 1 or uncertainty_band == "high":
        return (
            "low_confidence",
            "Too few independent signals or too much leave-one-signal-out uncertainty.",
        )
    if signal_count >= 4 and pairwise_comparisons > 0 and uncertainty_band == "low":
        return (
            "reportable",
            "Multiple independent signals, pairwise model evidence, and low sensitivity.",
        )
    return (
        "exploratory",
        "Some supporting signals are available, but the score still needs disclosure.",
    )


def _reportability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("reportability_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    paper_count = len(rows)
    return {
        "paper_count": paper_count,
        "status_counts": dict(sorted(counts.items())),
        "reportable_rate": round(counts.get("reportable", 0) / paper_count, 6)
        if paper_count
        else 0.0,
        "low_confidence_rate": round(counts.get("low_confidence", 0) / paper_count, 6)
        if paper_count
        else 0.0,
    }


def _evidence_first_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paper_count = len(rows)
    tier_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    positive_signal_counts: dict[str, int] = {}
    zero_signal_counts: dict[str, int] = {}
    available_signal_counts: dict[str, int] = {}
    external_count = 0
    positive_signal_total = 0
    zero_signal_total = 0
    zero_only_count = 0
    for row in rows:
        tier = str(row.get("credible_evidence_tier", "unknown"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        family = str(row.get("dominant_score_family", "unknown"))
        family_counts[family] = family_counts.get(family, 0) + 1
        positive_count = _safe_int(row.get("positive_evidence_signal_count"))
        zero_count = _safe_int(row.get("zero_evidence_signal_count"))
        available_count = _safe_int(row.get("evidence_first_signal_count"))
        positive_signal_counts[str(positive_count)] = (
            positive_signal_counts.get(str(positive_count), 0) + 1
        )
        zero_signal_counts[str(zero_count)] = (
            zero_signal_counts.get(str(zero_count), 0) + 1
        )
        available_signal_counts[str(available_count)] = (
            available_signal_counts.get(str(available_count), 0) + 1
        )
        positive_signal_total += positive_count
        zero_signal_total += zero_count
        if positive_count == 0 and zero_count > 0:
            zero_only_count += 1
        if row.get("external_adoption_score") is not None and float(row.get("external_adoption_score") or 0.0) > 0.0:
            external_count += 1
    return {
        "paper_count": paper_count,
        "tier_counts": dict(sorted(tier_counts.items())),
        "dominant_score_family_counts": dict(sorted(family_counts.items())),
        "external_adoption_coverage_rate": round(
            external_count / paper_count if paper_count else 0.0,
            6,
        ),
        "high_or_better_rate": round(
            (
                tier_counts.get("high", 0)
                + tier_counts.get("very_high", 0)
            )
            / paper_count
            if paper_count
            else 0.0,
            6,
        ),
        "credibility_adjustment": {
            "adjusted_sorting": True,
            "description": "final_impact_score applies a credibility adjustment on top of impact_index_v3",
        },
        "evidence_signal_audit": {
            "positive_signal_count_distribution": _sort_numeric_keyed_counts(
                positive_signal_counts
            ),
            "zero_signal_count_distribution": _sort_numeric_keyed_counts(
                zero_signal_counts
            ),
            "available_signal_count_distribution": _sort_numeric_keyed_counts(
                available_signal_counts
            ),
            "mean_positive_evidence_signal_count": round(
                positive_signal_total / paper_count if paper_count else 0.0,
                6,
            ),
            "mean_zero_evidence_signal_count": round(
                zero_signal_total / paper_count if paper_count else 0.0,
                6,
            ),
            "papers_with_no_positive_evidence": positive_signal_counts.get("0", 0),
            "papers_with_zero_only_evidence": zero_only_count,
            "interpretation": (
                "Positive evidence signals can raise the credible evidence tier; "
                "available zero-valued signals are audited as conservative evidence "
                "but do not upgrade the tier."
            ),
        },
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sort_numeric_keyed_counts(counts: dict[str, int]) -> dict[str, int]:
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda value: int(value))
    }


def _temporal_momentum(papers: list[Paper], *, recent_years: int = 2) -> dict[str, float]:
    scores: dict[str, float] = {}
    for paper in papers:
        history = _citation_history_by_year(paper)
        if not history:
            citation_count = _citation_count(paper)
            if citation_count == 0.0:
                scores[paper.paper_id] = 0.0
            continue
        ordered_years = sorted(history)
        recent = sum(history[year] for year in ordered_years[-recent_years:])
        total = sum(history.values())
        if total <= 0.0:
            scores[paper.paper_id] = 0.0
        else:
            scores[paper.paper_id] = recent / total
    return scores


def _citation_history_by_year(paper: Paper) -> dict[int, float]:
    candidates = [
        _paper_extra(paper, "citation_counts_by_year"),
        _paper_extra(paper, "citations_by_year"),
        _paper_extra(paper, "yearly_citation_counts"),
        paper.metadata.get("citation_counts_by_year"),
        paper.metadata.get("citations_by_year"),
        paper.metadata.get("yearly_citation_counts"),
        paper.metadata.get("citation_history"),
    ]
    for candidate in candidates:
        parsed = _parse_yearly_counts(candidate)
        if parsed:
            return parsed
    return {}


def _parse_yearly_counts(value: Any) -> dict[int, float]:
    if isinstance(value, dict):
        parsed: dict[int, float] = {}
        for year, count in value.items():
            try:
                parsed[int(year)] = float(count)
            except (TypeError, ValueError):
                continue
        return parsed
    if isinstance(value, list):
        parsed = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            year = item.get("year")
            count = (
                item.get("citation_count")
                if item.get("citation_count") is not None
                else item.get("count")
            )
            try:
                parsed[int(year)] = float(count)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def _internal_reference_edge_count(papers: list[Paper]) -> int:
    paper_ids = {paper.paper_id for paper in papers}
    return sum(
        1
        for paper in papers
        for reference in paper.references
        if reference.reference_id in paper_ids
    )


def _corpus_context_count(papers: list[Paper]) -> int:
    return sum(len(paper.citation_contexts) for paper in papers)


def _citation_count(paper: Paper | None) -> float | None:
    if paper is None:
        return None
    value = _paper_extra(paper, "citation_count")
    if value is None:
        value = paper.metadata.get("citation_count")
    if value is None:
        value = _external_number(
            paper,
            "citation_count",
            "cited_by_count",
            "pubmed_citation_count",
            "pmc_citation_count",
            "openalex_cited_by_count",
            "semantic_scholar_citation_count",
            "opencitations_cited_by_count",
        )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paper_extra(paper: Paper, key: str) -> Any | None:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    return extra.get(key)


def _normalize_to_100(values: dict[str, float], papers: list[Paper]) -> dict[str, float]:
    paper_ids = [paper.paper_id for paper in papers]
    max_value = max(values.values(), default=0.0)
    if max_value <= 0.0:
        return {paper_id: 0.0 for paper_id in paper_ids}
    present_values = {
        paper_id: float(value)
        for paper_id, value in values.items()
        if paper_id in set(paper_ids)
    }
    if len(present_values) == 1:
        present_id = next(iter(present_values))
        return {
            paper_id: 100.0 if paper_id == present_id else 0.0
            for paper_id in paper_ids
        }

    ordered = sorted(
        present_values.items(),
        key=lambda item: (item[1], item[0]),
    )
    denominator = max(len(ordered) - 1, 1)
    scores: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        value = ordered[index][1]
        while end + 1 < len(ordered) and math.isclose(ordered[end + 1][1], value):
            end += 1
        if end == len(ordered) - 1:
            percentile_score = 100.0
        elif index == 0:
            percentile_score = 0.0
        else:
            average_rank = (index + end) / 2.0
            percentile_score = average_rank / denominator * 100.0
        for offset in range(index, end + 1):
            scores[ordered[offset][0]] = percentile_score
        index = end + 1

    return {paper_id: scores.get(paper_id, 0.0) for paper_id in paper_ids}


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 100.0)


def _evidence_quality(
    *,
    signal_count: int,
    available_weight: float,
    pairwise_comparisons: int,
    uncertainty_band: str,
) -> tuple[str, str]:
    if signal_count >= 4 and available_weight >= 0.75 and pairwise_comparisons > 0 and uncertainty_band != "high":
        return "strong", "At least four impact signals are available, including model pairwise evidence."
    if signal_count >= 2 and available_weight >= 0.5:
        return "moderate", "Multiple impact signals are available, but the evidence is not complete."
    signal_word = "signal" if signal_count == 1 else "signals"
    return (
        "weak",
        f"Only {signal_count} impact {signal_word} available; treat this score as exploratory.",
    )
