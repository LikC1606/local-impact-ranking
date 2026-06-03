from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
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


def paper_stability_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    credit_mode: str,
    top_n: int = 100,
) -> dict[str, Any]:
    base_scores = score_influence(papers, judge_results, credit_mode=credit_mode).paper_scores
    base_rank_ids = _top_ids(base_scores, len(base_scores))
    base_ranks = {identifier: rank for rank, identifier in enumerate(base_rank_ids, start=1)}
    tracked_ids = _top_ids(base_scores, top_n)
    if not tracked_ids:
        return {
            "credit_mode": credit_mode,
            "top_n": top_n,
            "sample_count": 0,
            "papers": [],
            "limitations": _paper_stability_limitations(),
        }

    missing_rank = len(base_rank_ids) + 1
    perturbed_ranks: dict[str, list[int]] = {identifier: [] for identifier in tracked_ids}
    dropped_counts: dict[str, int] = {identifier: 0 for identifier in tracked_ids}

    for removed_index, _result in enumerate(judge_results):
        perturbed_results = [
            result for index, result in enumerate(judge_results) if index != removed_index
        ]
        perturbed_scores = score_influence(
            papers,
            perturbed_results,
            credit_mode=credit_mode,
        ).paper_scores
        perturbed_rank_ids = _top_ids(perturbed_scores, len(perturbed_scores))
        perturbed_rank_map = {
            identifier: rank for rank, identifier in enumerate(perturbed_rank_ids, start=1)
        }
        for identifier in tracked_ids:
            rank = perturbed_rank_map.get(identifier, missing_rank)
            perturbed_ranks[identifier].append(rank)
            if identifier not in perturbed_rank_map:
                dropped_counts[identifier] += 1

    rows = []
    for identifier in tracked_ids:
        ranks = perturbed_ranks[identifier]
        best_rank = min(ranks) if ranks else base_ranks[identifier]
        worst_rank = max(ranks) if ranks else base_ranks[identifier]
        rank_span = worst_rank - best_rank
        rows.append(
            {
                "paper_id": identifier,
                "base_rank": base_ranks[identifier],
                "base_score": base_scores[identifier],
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "rank_span": rank_span,
                "dropped_count": dropped_counts[identifier],
                "stability_band": _paper_stability_band(
                    dropped_counts[identifier],
                    rank_span,
                    len(judge_results),
                ),
            }
        )

    return {
        "credit_mode": credit_mode,
        "top_n": top_n,
        "sample_count": len(judge_results),
        "papers": rows,
        "limitations": _paper_stability_limitations(),
    }


def export_paper_stability_csv(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "base_rank",
        "base_score",
        "best_rank",
        "worst_rank",
        "rank_span",
        "dropped_count",
        "stability_band",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("papers", []):
            writer.writerow(
                sanitize_csv_row(
                    {
                        "paper_id": row["paper_id"],
                        "base_rank": row["base_rank"],
                        "base_score": f"{float(row['base_score']):.6f}",
                        "best_rank": row["best_rank"],
                        "worst_rank": row["worst_rank"],
                        "rank_span": row["rank_span"],
                        "dropped_count": row["dropped_count"],
                        "stability_band": row["stability_band"],
                    }
                )
            )


def bootstrap_paper_score_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    credit_mode: str,
    top_n: int = 100,
    samples: int = 200,
    seed: int = 13,
) -> dict[str, Any]:
    base_scores = score_influence(papers, judge_results, credit_mode=credit_mode).paper_scores
    base_rank_ids = _top_ids(base_scores, len(base_scores))
    base_ranks = {identifier: rank for rank, identifier in enumerate(base_rank_ids, start=1)}
    tracked_ids = _top_ids(base_scores, top_n)
    if not tracked_ids:
        return {
            "method": "target_bootstrap",
            "score_name": "raw_influence_score",
            "credit_mode": credit_mode,
            "top_n": top_n,
            "samples": samples,
            "seed": seed,
            "sample_count": 0,
            "papers": [],
            "limitations": _bootstrap_score_limitations(),
        }

    if not judge_results or samples <= 0:
        return {
            "method": "target_bootstrap",
            "score_name": "raw_influence_score",
            "credit_mode": credit_mode,
            "top_n": top_n,
            "samples": max(samples, 0),
            "seed": seed,
            "sample_count": len(judge_results),
            "papers": [
                {
                    "paper_id": identifier,
                    "base_rank": base_ranks[identifier],
                    "base_score": base_scores[identifier],
                    "score_mean": base_scores[identifier],
                    "score_ci_low": base_scores[identifier],
                    "score_ci_high": base_scores[identifier],
                    "score_ci_width": 0.0,
                    "best_rank": base_ranks[identifier],
                    "worst_rank": base_ranks[identifier],
                    "rank_span": 0,
                    "zero_score_rate": 0.0,
                    "uncertainty_band": "insufficient_samples",
                }
                for identifier in tracked_ids
            ],
            "limitations": _bootstrap_score_limitations(),
        }

    rng = random.Random(seed)
    score_samples: dict[str, list[float]] = {identifier: [] for identifier in tracked_ids}
    rank_samples: dict[str, list[int]] = {identifier: [] for identifier in tracked_ids}
    missing_rank = len(base_rank_ids) + 1

    for _sample_index in range(samples):
        sampled_results = [rng.choice(judge_results) for _ in judge_results]
        sampled_scores = score_influence(
            papers,
            sampled_results,
            credit_mode=credit_mode,
        ).paper_scores
        sampled_rank_ids = _top_ids(sampled_scores, len(sampled_scores))
        sampled_ranks = {
            identifier: rank for rank, identifier in enumerate(sampled_rank_ids, start=1)
        }
        for identifier in tracked_ids:
            score_samples[identifier].append(sampled_scores.get(identifier, 0.0))
            rank_samples[identifier].append(sampled_ranks.get(identifier, missing_rank))

    rows = []
    for identifier in tracked_ids:
        scores = sorted(score_samples[identifier])
        ranks = rank_samples[identifier]
        score_ci_low = _quantile(scores, 0.025)
        score_ci_high = _quantile(scores, 0.975)
        score_mean = sum(scores) / len(scores) if scores else base_scores[identifier]
        best_rank = min(ranks) if ranks else base_ranks[identifier]
        worst_rank = max(ranks) if ranks else base_ranks[identifier]
        rank_span = worst_rank - best_rank
        zero_score_rate = (
            sum(1 for score in scores if score <= 0.0) / len(scores) if scores else 0.0
        )
        rows.append(
            {
                "paper_id": identifier,
                "base_rank": base_ranks[identifier],
                "base_score": base_scores[identifier],
                "score_mean": score_mean,
                "score_ci_low": score_ci_low,
                "score_ci_high": score_ci_high,
                "score_ci_width": score_ci_high - score_ci_low,
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "rank_span": rank_span,
                "zero_score_rate": zero_score_rate,
                "uncertainty_band": _bootstrap_uncertainty_band(
                    samples,
                    rank_span,
                    zero_score_rate,
                    score_ci_high - score_ci_low,
                    base_scores[identifier],
                ),
            }
        )

    return {
        "method": "target_bootstrap",
        "score_name": "raw_influence_score",
        "credit_mode": credit_mode,
        "top_n": top_n,
        "samples": samples,
        "seed": seed,
        "sample_count": len(judge_results),
        "papers": rows,
        "limitations": _bootstrap_score_limitations(),
    }


def export_bootstrap_paper_score_csv(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "base_rank",
        "base_score",
        "score_mean",
        "score_ci_low",
        "score_ci_high",
        "score_ci_width",
        "best_rank",
        "worst_rank",
        "rank_span",
        "zero_score_rate",
        "uncertainty_band",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("papers", []):
            writer.writerow(
                sanitize_csv_row(
                    {
                        "paper_id": row["paper_id"],
                        "base_rank": row["base_rank"],
                        "base_score": f"{float(row['base_score']):.6f}",
                        "score_mean": f"{float(row['score_mean']):.6f}",
                        "score_ci_low": f"{float(row['score_ci_low']):.6f}",
                        "score_ci_high": f"{float(row['score_ci_high']):.6f}",
                        "score_ci_width": f"{float(row['score_ci_width']):.6f}",
                        "best_rank": row["best_rank"],
                        "worst_rank": row["worst_rank"],
                        "rank_span": row["rank_span"],
                        "zero_score_rate": f"{float(row['zero_score_rate']):.6f}",
                        "uncertainty_band": row["uncertainty_band"],
                    }
                )
            )


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


def _paper_stability_band(dropped_count: int, rank_span: int, sample_count: int) -> str:
    if sample_count <= 1:
        return "insufficient_samples"
    if dropped_count > 0 or rank_span > 5:
        return "unstable"
    if rank_span > 1:
        return "moderate"
    return "stable"


def _paper_stability_limitations() -> list[str]:
    return [
        "Leave-one-target-out stability measures sensitivity to input target coverage, not semantic correctness.",
        "Small datasets can make otherwise valid papers appear unstable because every target contributes large mass.",
    ]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = position - lower_index
    return values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction


def _bootstrap_uncertainty_band(
    samples: int,
    rank_span: int,
    zero_score_rate: float,
    score_ci_width: float,
    base_score: float,
) -> str:
    if samples < 30:
        return "insufficient_samples"
    relative_width = score_ci_width / base_score if base_score > 0 else 1.0
    if zero_score_rate >= 0.25 or rank_span > 10 or relative_width >= 1.5:
        return "unstable"
    if zero_score_rate > 0.0 or rank_span > 3 or relative_width >= 0.75:
        return "moderate"
    return "stable"


def _bootstrap_score_limitations() -> list[str]:
    return [
        "Bootstrap intervals measure sensitivity to target-paper sampling, not external truth.",
        "Intervals are dataset-local and can be narrow even when the judge is systematically biased.",
        "Small datasets produce coarse uncertainty estimates; increase target count before making strong claims.",
    ]
