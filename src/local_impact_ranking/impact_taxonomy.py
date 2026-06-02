from __future__ import annotations

from typing import Any


def impact_dimension_taxonomy() -> dict[str, dict[str, Any]]:
    return {
        "citation": {
            "claim_type": "citation-impact",
            "score_fields": [
                "field_year_normalized_citation_score",
                "relative_citation_ratio_score",
                "citation_graph_score",
                "influential_citation_score",
                "bip_impact_score",
                "temporal_momentum_score",
            ],
            "raw_fields": [
                "citation_count_raw",
                "relative_citation_ratio_raw",
                "influential_citation_count_raw",
                "bip_impact_raw",
            ],
            "claim_boundary": "Scholarly citation impact, field/year normalized when metadata supports it.",
        },
        "patent": {
            "claim_type": "translational-impact",
            "score_fields": ["patent_impact_score"],
            "raw_fields": ["patent_citations_raw"],
            "claim_boundary": "Patent-linked translational impact only when patent citation evidence is supplied.",
        },
        "policy": {
            "claim_type": "policy-impact",
            "score_fields": ["policy_impact_score"],
            "raw_fields": ["policy_citations_raw"],
            "claim_boundary": "Policy or guideline impact only when policy citation evidence is supplied.",
        },
        "media": {
            "claim_type": "public-attention-impact",
            "score_fields": ["media_impact_score"],
            "raw_fields": ["media_mentions_raw"],
            "claim_boundary": "Public/media attention, not a substitute for scholarly quality.",
        },
        "award": {
            "claim_type": "recognition-impact",
            "score_fields": ["award_impact_score"],
            "raw_fields": ["award_count_raw"],
            "claim_boundary": "Prize or award recognition when explicit award evidence is supplied.",
        },
        "downloads": {
            "claim_type": "artifact-adoption-impact",
            "score_fields": ["artifact_adoption_score"],
            "raw_fields": ["artifact_adoption_raw"],
            "claim_boundary": "Dataset/model/software adoption only when usage, reuse, stars, or download evidence is supplied.",
        },
        "stars": {
            "claim_type": "artifact-adoption-impact",
            "score_fields": ["artifact_adoption_score"],
            "raw_fields": ["artifact_adoption_raw"],
            "claim_boundary": "Repository or artifact popularity; report separately from citation impact.",
        },
    }


def ranking_dimension_coverage(rankings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    taxonomy = impact_dimension_taxonomy()
    paper_count = len(rankings)
    coverage: dict[str, dict[str, Any]] = {}
    for dimension, spec in taxonomy.items():
        raw_fields = list(spec.get("raw_fields", []))
        score_fields = list(spec.get("score_fields", []))
        covered = [
            row
            for row in rankings
            if _row_has_positive_evidence(row, raw_fields, score_fields)
        ]
        coverage[dimension] = {
            "claim_type": spec["claim_type"],
            "covered_paper_count": len(covered),
            "paper_count": paper_count,
            "coverage_rate": round(len(covered) / paper_count if paper_count else 0.0, 6),
            "score_fields": score_fields,
            "raw_fields": raw_fields,
            "claim_boundary": spec["claim_boundary"],
        }
    return coverage


def _row_has_positive_evidence(
    row: dict[str, Any],
    raw_fields: list[str],
    score_fields: list[str],
) -> bool:
    for field in raw_fields:
        value = _numeric_or_none(row.get(field))
        if value is not None and value > 0.0:
            return True
    for field in score_fields:
        value = _numeric_or_none(row.get(field))
        if value is not None and value > 0.0:
            return True
    return False


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
