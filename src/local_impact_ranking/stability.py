from __future__ import annotations

from typing import Any

from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import score_influence


DEFAULT_COMPARISON_CREDIT_MODES = ["rank_weighted", "flat"]


def stability_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    primary_credit_mode: str,
    comparison_credit_modes: list[str] | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    comparison_modes = [
        mode
        for mode in (comparison_credit_modes or DEFAULT_COMPARISON_CREDIT_MODES)
        if mode != primary_credit_mode
    ]
    primary_scores = score_influence(
        papers,
        judge_results,
        credit_mode=primary_credit_mode,
    ).paper_scores
    return {
        "primary_credit_mode": primary_credit_mode,
        "top_n": top_n,
        "ranked_item_count": len(primary_scores),
        "mode_comparison": {
            mode: _score_comparison(
                primary_scores,
                score_influence(papers, judge_results, credit_mode=mode).paper_scores,
                top_n,
            )
            for mode in comparison_modes
        },
        "leave_one_target_out": _leave_one_target_out(
            papers,
            judge_results,
            primary_credit_mode,
            top_n,
        ),
        "interpretation": {
            "scope": "dataset-local robustness check, not a proof of true global impact",
            "top_n_overlap_rate": "Share of primary top-N papers retained under a comparison scoring mode.",
            "kendall_tau": "Pairwise rank agreement over papers scored by either ranking; missing papers are ranked after present papers.",
            "leave_one_target_out": "Average retention of the primary top-N after removing one target paper's judge result at a time.",
        },
        "limitations": [
            "This report measures ranking sensitivity to scoring and sample perturbation, not correctness against human labels.",
            "Small datasets can have low retention because one target paper contributes a large share of all selected edges.",
            "Stable rankings can still be wrong if the judge systematically selects weak references.",
        ],
    }


def _score_comparison(
    primary_scores: dict[str, float],
    comparison_scores: dict[str, float],
    top_n: int,
) -> dict[str, Any]:
    primary_top = _top_ids(primary_scores, top_n)
    comparison_top = _top_ids(comparison_scores, top_n)
    primary_set = set(primary_top)
    comparison_set = set(comparison_top)
    overlap = len(primary_set & comparison_set)
    return {
        "ranked_item_count": len(comparison_scores),
        "top_n_overlap": overlap,
        "top_n_overlap_rate": overlap / len(primary_set) if primary_set else 0.0,
        "kendall_tau": _kendall_tau(primary_scores, comparison_scores),
    }


def _leave_one_target_out(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    credit_mode: str,
    top_n: int,
) -> dict[str, Any]:
    primary_scores = score_influence(papers, judge_results, credit_mode=credit_mode).paper_scores
    primary_top = set(_top_ids(primary_scores, top_n))
    if not primary_top or not judge_results:
        return {
            "sample_count": 0,
            "average_top_n_retention_rate": 0.0,
            "min_top_n_retention_rate": 0.0,
            "max_top_n_retention_rate": 0.0,
        }

    retention_rates = []
    for removed_index, _result in enumerate(judge_results):
        perturbed_results = [
            result for index, result in enumerate(judge_results) if index != removed_index
        ]
        perturbed_scores = score_influence(
            papers,
            perturbed_results,
            credit_mode=credit_mode,
        ).paper_scores
        perturbed_top = set(_top_ids(perturbed_scores, top_n))
        retention_rates.append(len(primary_top & perturbed_top) / len(primary_top))

    return {
        "sample_count": len(retention_rates),
        "average_top_n_retention_rate": sum(retention_rates) / len(retention_rates),
        "min_top_n_retention_rate": min(retention_rates),
        "max_top_n_retention_rate": max(retention_rates),
    }


def _kendall_tau(
    first_scores: dict[str, float],
    second_scores: dict[str, float],
) -> float:
    identifiers = sorted(set(first_scores) | set(second_scores))
    if len(identifiers) < 2:
        return 1.0

    first_ranks = _rank_map(first_scores, identifiers)
    second_ranks = _rank_map(second_scores, identifiers)
    concordant = 0
    discordant = 0
    for left_index, left_id in enumerate(identifiers):
        for right_id in identifiers[left_index + 1 :]:
            first_delta = first_ranks[left_id] - first_ranks[right_id]
            second_delta = second_ranks[left_id] - second_ranks[right_id]
            product = first_delta * second_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def _rank_map(scores: dict[str, float], identifiers: list[str]) -> dict[str, int]:
    ordered = _top_ids(scores, len(scores))
    missing_rank = len(ordered) + 1
    ranks = {identifier: rank for rank, identifier in enumerate(ordered, start=1)}
    return {identifier: ranks.get(identifier, missing_rank) for identifier in identifiers}


def _top_ids(scores: dict[str, float], top_n: int) -> list[str]:
    return [
        identifier
        for identifier, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]
