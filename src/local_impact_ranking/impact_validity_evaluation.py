from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SCORE_COLUMNS = (
    "final_impact_score",
    "impact_index_v3",
    "credibility_adjusted_impact_score",
    "impact_index",
    "score",
)


def evaluate_impact_validity(
    rankings_path: str | Path,
    labels_path: str | Path,
    *,
    score_column: str | None = None,
    label_column: str = "future_impact",
    top_fraction: float = 0.1,
    min_label_delta: float = 0.0,
    bootstrap_samples: int = 200,
    permutation_samples: int = 200,
    random_seed: int = 0,
) -> dict[str, Any]:
    scores, selected_score_column = _load_scores(rankings_path, score_column=score_column)
    labels = _load_labels(labels_path, label_column=label_column)
    rows = [
        {
            "paper_id": paper_id,
            "score": score,
            "label": label["label"],
            "field": label.get("field", "unknown"),
        }
        for paper_id, score in scores.items()
        if paper_id in labels
        for label in [labels[paper_id]]
    ]
    summary = _summary(
        rows,
        score_column=selected_score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        ranking_paper_count=len(scores),
        label_paper_count=len(labels),
    )
    bootstrap = _bootstrap_evidence(
        rows,
        score_column=selected_score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        sample_count=bootstrap_samples,
        random_seed=random_seed,
    )
    permutation = _permutation_baseline(
        rows,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        sample_count=permutation_samples,
        random_seed=random_seed + 7919,
    )
    significance = _statistical_significance(
        summary,
        bootstrap,
        permutation,
    )
    summary["decision"] = _decision(
        summary["matched_paper_count"],
        summary["spearman_correlation"],
        summary["pairwise_accuracy"],
        summary["ndcg_at_top_k"],
        significance,
    )
    by_field = _group_summaries(
        rows,
        group_key="field",
        score_column=selected_score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
    )
    robustness = _robustness_checks(
        rows,
        score_column=selected_score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
    )
    scorecard = _credibility_scorecard(
        summary,
        significance=significance,
        by_field=by_field,
        robustness=robustness,
    )
    return {
        "summary": summary,
        "credibility_scorecard": scorecard,
        "bootstrap": bootstrap,
        "permutation_baseline": permutation,
        "statistical_significance": significance,
        "robustness": robustness,
        "coverage_examples": _coverage_examples(scores, labels),
        "by_field": by_field,
        "examples": _examples(rows),
        "interpretation": {
            "claim": "This evaluates whether impact scores align with external continuous labels or a future-window outcome.",
            "limits": [
                "A high score here supports validity only for the label source and time window used.",
                "Use field/year normalized labels when comparing across fields or publication years.",
                "Low coverage means the evaluation is mostly an entity-resolution or data-coverage audit.",
            ],
        },
    }


def write_impact_validity_evaluation(
    rankings_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    score_column: str | None = None,
    label_column: str = "future_impact",
    top_fraction: float = 0.1,
    min_label_delta: float = 0.0,
    bootstrap_samples: int = 200,
    permutation_samples: int = 200,
    random_seed: int = 0,
) -> dict[str, Any]:
    report = evaluate_impact_validity(
        rankings_path,
        labels_path,
        score_column=score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "impact_validity_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "impact_validity_evaluation.md").write_text(
        render_impact_validity_evaluation_markdown(report),
        encoding="utf-8",
    )
    (destination / "impact_credibility_scorecard.json").write_text(
        json.dumps(report["credibility_scorecard"], indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "impact_credibility_scorecard.md").write_text(
        render_impact_credibility_scorecard_markdown(report["credibility_scorecard"]),
        encoding="utf-8",
    )
    return report


def render_impact_validity_evaluation_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    scorecard = report.get("credibility_scorecard", {})
    lines = [
        "# Impact Validity Evaluation",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if key == "decision":
            continue
        lines.append(f"| {key} | {_format_value(value)} |")
    decision = summary.get("decision", {})
    significance = report.get("statistical_significance", {})
    lines.extend(
        [
            f"| decision_status | {_text(decision.get('status'))} |",
            f"| decision_reason | {_text(decision.get('reason'))} |",
            f"| significant_vs_random | {_yes_no(significance.get('significant_vs_random'))} |",
            f"| significance_p_value | {_format_value(significance.get('p_value'))} |",
            f"| significance_status | {_text(significance.get('status'))} |",
            f"| credibility_status | {_text(scorecard.get('status'))} |",
            f"| credibility_grade | {_text(scorecard.get('grade'))} |",
            "",
            "## Credibility Scorecard",
            "",
            "| Check | Status | Value | Reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {_text(name)} | {_text(check.get('status'))} | "
            f"{_format_value(check.get('value'))} | {_text(check.get('reason'))} |"
        )
    lines.extend(
        [
            "",
            "## Statistical Evidence",
            "",
            "| Metric | Observed | Bootstrap CI95 | Permutation mean | Permutation p-value |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    bootstrap_metrics = report.get("bootstrap", {}).get("metrics", {})
    permutation_metrics = report.get("permutation_baseline", {}).get("metrics", {})
    for metric in (
        "spearman_correlation",
        "kendall_tau_b",
        "pairwise_accuracy",
        "ndcg_at_top_k",
    ):
        boot = bootstrap_metrics.get(metric, {})
        perm = permutation_metrics.get(metric, {})
        lines.append(
            f"| {metric} | {_format_value(summary.get(metric))} | "
            f"{_ci(boot.get('ci95'))} | "
            f"{_format_value(perm.get('null_mean'))} | "
            f"{_format_value(perm.get('p_value'))} |"
        )
    lines.extend(
        [
            "",
            "## Coverage Examples",
            "",
            "| Type | Paper IDs |",
            "| --- | --- |",
            f"| unmatched_score_ids | {_joined_ids(report.get('coverage_examples', {}).get('unmatched_score_ids'))} |",
            f"| unmatched_label_ids | {_joined_ids(report.get('coverage_examples', {}).get('unmatched_label_ids'))} |",
            "",
            "## By Field",
            "",
            "| Field | Matched | Spearman | Kendall Tau-b | Pairwise Accuracy | NDCG@K | Top-K Precision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field, field_summary in sorted(report.get("by_field", {}).items()):
        lines.append(
            f"| {_text(field)} | {field_summary['matched_paper_count']} | "
            f"{_format_value(field_summary['spearman_correlation'])} | "
            f"{_format_value(field_summary['kendall_tau_b'])} | "
            f"{_format_value(field_summary['pairwise_accuracy'])} | "
            f"{_format_value(field_summary['ndcg_at_top_k'])} | "
            f"{_format_value(field_summary['top_k_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _text(report.get("interpretation", {}).get("claim")),
            "",
        ]
    )
    for limit in report.get("interpretation", {}).get("limits", []):
        lines.append(f"- {_text(limit)}")
    lines.append("")
    return "\n".join(lines)


def render_impact_credibility_scorecard_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Impact Credibility Scorecard",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {_text(scorecard.get('status'))} |",
        f"| grade | {_text(scorecard.get('grade'))} |",
        f"| can_claim_high_validation_credibility | {_yes_no(scorecard.get('can_claim_high_validation_credibility'))} |",
        f"| reason | {_text(scorecard.get('reason'))} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {_text(name)} | {_text(check.get('status'))} | "
            f"{_format_value(check.get('value'))} | {_text(check.get('reason'))} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    blockers = scorecard.get("blocking_reasons", [])
    lines.extend(f"- {_text(item)}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = scorecard.get("warnings", [])
    lines.extend(f"- {_text(item)}" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Recommended Next Actions", ""])
    actions = scorecard.get("recommended_next_actions", [])
    lines.extend(f"- {_text(item)}" for item in actions) if actions else lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _load_scores(
    path: str | Path,
    *,
    score_column: str | None,
) -> tuple[dict[str, float], str]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Ranking file has no header: {source}")
        selected_column = _select_score_column(reader.fieldnames, score_column)
        scores: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            paper_id = row.get("paper_id") or row.get("id")
            if not paper_id:
                raise ValueError(f"Ranking row {row_number} is missing paper_id/id")
            raw_score = row.get(selected_column)
            if raw_score in (None, ""):
                continue
            scores[str(paper_id)] = float(raw_score)
    return scores, selected_column


def _coverage_examples(
    scores: dict[str, float],
    labels: dict[str, dict[str, Any]],
    *,
    limit: int = 50,
) -> dict[str, Any]:
    score_ids = set(scores)
    label_ids = set(labels)
    return {
        "unmatched_score_count": len(score_ids - label_ids),
        "unmatched_label_count": len(label_ids - score_ids),
        "unmatched_score_ids": sorted(score_ids - label_ids)[:limit],
        "unmatched_label_ids": sorted(label_ids - score_ids)[:limit],
    }


def _select_score_column(fieldnames: list[str], score_column: str | None) -> str:
    if score_column is not None:
        if score_column not in fieldnames:
            raise ValueError(f"Score column not found in ranking file: {score_column}")
        return score_column
    for candidate in DEFAULT_SCORE_COLUMNS:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Ranking file needs a score column such as final_impact_score or impact_index_v3"
    )


def _load_labels(path: str | Path, *, label_column: str) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Label CSV has no header: {source}")
            for row_number, row in enumerate(reader, start=2):
                _add_label(labels, row, row_number=row_number, label_column=label_column)
        return labels
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            _add_label(
                labels,
                json.loads(stripped),
                row_number=line_number,
                label_column=label_column,
            )
    return labels


def _add_label(
    labels: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    row_number: int,
    label_column: str,
) -> None:
    paper_id = row.get("paper_id") or row.get("id")
    if not paper_id:
        raise ValueError(f"Label row {row_number} is missing paper_id/id")
    raw_label = row.get(label_column)
    if raw_label in (None, ""):
        raise ValueError(f"Label row {row_number} is missing {label_column}")
    label = float(raw_label)
    if not math.isfinite(label):
        raise ValueError(f"Label row {row_number} has non-finite {label_column}")
    labels[str(paper_id)] = {
        "label": label,
        "field": str(row.get("field") or "unknown"),
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    score_column: str,
    label_column: str,
    top_fraction: float,
    min_label_delta: float,
    ranking_paper_count: int | None = None,
    label_paper_count: int | None = None,
) -> dict[str, Any]:
    matched = len(rows)
    ranking_count = matched if ranking_paper_count is None else ranking_paper_count
    label_count = matched if label_paper_count is None else label_paper_count
    top_k = _top_k(matched, top_fraction)
    spearman = _spearman(rows)
    kendall = _kendall_tau_b(rows)
    pairwise = _pairwise_accuracy(rows, min_label_delta=min_label_delta)
    ndcg = _ndcg_at_k(rows, top_k)
    precision, recall = _top_k_precision_recall(rows, top_k)
    return {
        "matched_paper_count": matched,
        "ranking_paper_count": ranking_count,
        "label_paper_count": label_count,
        "score_label_match_rate": _round(matched / ranking_count)
        if ranking_count
        else 0.0,
        "label_match_rate": _round(matched / label_count) if label_count else 0.0,
        "score_column": score_column,
        "label_column": label_column,
        "top_fraction": top_fraction,
        "top_k": top_k,
        "spearman_correlation": spearman,
        "kendall_tau_b": kendall,
        "pairwise_accuracy": pairwise["accuracy"],
        "pairwise_comparable_pairs": pairwise["comparable_pairs"],
        "pairwise_correct_pairs": pairwise["correct_pairs"],
        "pairwise_tie_pairs": pairwise["tie_pairs"],
        "ndcg_at_top_k": ndcg,
        "top_k_precision": precision,
        "top_k_recall": recall,
        "decision": _decision(matched, spearman, pairwise["accuracy"], ndcg, None),
    }


def _group_summaries(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    score_column: str,
    label_column: str,
    top_fraction: float,
    min_label_delta: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(row)
    return {
        group: _summary(
            group_rows,
            score_column=score_column,
            label_column=label_column,
            top_fraction=top_fraction,
            min_label_delta=min_label_delta,
        )
        for group, group_rows in grouped.items()
    }


def _bootstrap_evidence(
    rows: list[dict[str, Any]],
    *,
    score_column: str,
    label_column: str,
    top_fraction: float,
    min_label_delta: float,
    sample_count: int,
    random_seed: int,
) -> dict[str, Any]:
    metrics = (
        "spearman_correlation",
        "kendall_tau_b",
        "pairwise_accuracy",
        "ndcg_at_top_k",
    )
    if len(rows) < 2 or sample_count <= 0:
        return {
            "sample_count": 0,
            "random_seed": random_seed,
            "metrics": {
                metric: {"mean": 0.0, "ci95": [0.0, 0.0]} for metric in metrics
            },
        }
    rng = random.Random(random_seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(sample_count):
        sampled_rows = [rows[rng.randrange(len(rows))] for _ in rows]
        summary = _summary(
            sampled_rows,
            score_column=score_column,
            label_column=label_column,
            top_fraction=top_fraction,
            min_label_delta=min_label_delta,
        )
        for metric in metrics:
            samples[metric].append(float(summary.get(metric, 0.0) or 0.0))
    return {
        "sample_count": sample_count,
        "random_seed": random_seed,
        "metrics": {
            metric: {
                "mean": _round(sum(values) / len(values)) if values else 0.0,
                "ci95": _ci95(values),
            }
            for metric, values in samples.items()
        },
    }


def _permutation_baseline(
    rows: list[dict[str, Any]],
    *,
    top_fraction: float,
    min_label_delta: float,
    sample_count: int,
    random_seed: int,
) -> dict[str, Any]:
    metrics = (
        "spearman_correlation",
        "kendall_tau_b",
        "pairwise_accuracy",
        "ndcg_at_top_k",
    )
    observed = _metric_values(rows, top_fraction, min_label_delta)
    if len(rows) < 2 or sample_count <= 0:
        return {
            "sample_count": 0,
            "random_seed": random_seed,
            "metrics": {
                metric: {
                    "observed": observed.get(metric, 0.0),
                    "null_mean": 0.0,
                    "p_value": 1.0,
                }
                for metric in metrics
            },
        }
    rng = random.Random(random_seed)
    labels = [row["label"] for row in rows]
    null_samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(sample_count):
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        shuffled_rows = [
            {**row, "label": shuffled_labels[index]} for index, row in enumerate(rows)
        ]
        values = _metric_values(shuffled_rows, top_fraction, min_label_delta)
        for metric in metrics:
            null_samples[metric].append(values[metric])
    return {
        "sample_count": sample_count,
        "random_seed": random_seed,
        "metrics": {
            metric: {
                "observed": observed.get(metric, 0.0),
                "null_mean": _round(sum(values) / len(values)) if values else 0.0,
                "p_value": _permutation_p_value(
                    observed.get(metric, 0.0),
                    values,
                ),
            }
            for metric, values in null_samples.items()
        },
    }


def _robustness_checks(
    rows: list[dict[str, Any]],
    *,
    score_column: str,
    label_column: str,
    top_fraction: float,
    min_label_delta: float,
) -> dict[str, Any]:
    if len(rows) < 20:
        return {
            "status": "insufficient_validation_data",
            "reason": "Fewer than 20 matched papers; outlier robustness is not claimed.",
            "minimum_required_papers": 20,
            "matched_paper_count": len(rows),
            "checks": {},
        }
    observed = _metric_values(rows, top_fraction, min_label_delta)
    variants = {
        "leave_top_label_out": _drop_top_row(rows, key="label"),
        "leave_top_score_out": _drop_top_row(rows, key="score"),
        "drop_top_label_5_percent": _drop_top_fraction(rows, key="label", fraction=0.05),
    }
    checks = {
        name: _robustness_check(
            observed,
            variant_rows,
            top_fraction=top_fraction,
            min_label_delta=min_label_delta,
        )
        for name, variant_rows in variants.items()
        if len(variant_rows) >= 2
    }
    statuses = {check["status"] for check in checks.values()}
    if "fail" in statuses:
        status = "outlier_sensitive"
        reason = "Validity metrics change materially after removing extreme score or label rows."
    elif "warn" in statuses:
        status = "moderately_stable"
        reason = "Validity metrics are usable but show some sensitivity to extreme rows."
    else:
        status = "stable"
        reason = "Validity metrics remain stable after removing extreme score or label rows."
    return {
        "status": status,
        "reason": reason,
        "observed": observed,
        "score_column": score_column,
        "label_column": label_column,
        "checks": checks,
    }


def _robustness_check(
    observed: dict[str, float],
    variant_rows: list[dict[str, Any]],
    *,
    top_fraction: float,
    min_label_delta: float,
) -> dict[str, Any]:
    variant = _metric_values(variant_rows, top_fraction, min_label_delta)
    deltas = {
        metric: _round(float(variant.get(metric, 0.0)) - float(observed.get(metric, 0.0)))
        for metric in (
            "spearman_correlation",
            "kendall_tau_b",
            "pairwise_accuracy",
            "ndcg_at_top_k",
        )
    }
    max_rank_delta = max(
        abs(deltas["spearman_correlation"]),
        abs(deltas["kendall_tau_b"]),
    )
    max_accuracy_delta = abs(deltas["pairwise_accuracy"])
    max_top_delta = abs(deltas["ndcg_at_top_k"])
    if max_rank_delta > 0.15 or max_accuracy_delta > 0.10 or max_top_delta > 0.15:
        status = "fail"
        reason = "Removing an extreme row changes rank, pairwise, or top-k validity too much."
    elif max_rank_delta > 0.08 or max_accuracy_delta > 0.05 or max_top_delta > 0.08:
        status = "warn"
        reason = "Removing an extreme row changes validity metrics enough to disclose."
    else:
        status = "pass"
        reason = "Metrics are stable after removing this extreme row set."
    return {
        "status": status,
        "reason": reason,
        "matched_paper_count": len(variant_rows),
        "metrics": variant,
        "deltas": deltas,
        "max_abs_delta": _round(max(max_rank_delta, max_accuracy_delta, max_top_delta)),
    }


def _drop_top_row(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    top_index = max(range(len(rows)), key=lambda index: rows[index][key])
    return [row for index, row in enumerate(rows) if index != top_index]


def _drop_top_fraction(
    rows: list[dict[str, Any]],
    *,
    key: str,
    fraction: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    drop_count = max(1, math.ceil(len(rows) * fraction))
    top_ids = {
        row["paper_id"]
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[:drop_count]
    }
    return [row for row in rows if row["paper_id"] not in top_ids]


def _credibility_scorecard(
    summary: dict[str, Any],
    *,
    significance: dict[str, Any],
    by_field: dict[str, dict[str, Any]],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "coverage": _coverage_check(summary),
        "statistical_support": _statistical_support_check(significance),
        "rank_alignment": _rank_alignment_check(summary),
        "top_rank_quality": _top_rank_quality_check(summary),
        "slice_stability": _slice_stability_check(by_field),
        "outlier_robustness": _outlier_robustness_check(robustness),
    }
    blocking = [
        check["reason"]
        for name, check in checks.items()
        if check["status"] == "fail"
        and name in {
            "coverage",
            "statistical_support",
            "rank_alignment",
            "outlier_robustness",
        }
    ]
    warnings = [
        check["reason"]
        for check in checks.values()
        if check["status"] == "warn"
    ]
    pass_count = sum(1 for check in checks.values() if check["status"] == "pass")
    fail_count = sum(1 for check in checks.values() if check["status"] == "fail")
    if int(summary.get("matched_paper_count", 0) or 0) < 20:
        status = "insufficient_validation_data"
        grade = "D"
        reason = "Fewer than 20 matched papers; use as a smoke check only."
    elif blocking:
        status = "not_credible" if checks["statistical_support"]["status"] == "fail" else "low_credibility"
        grade = "D" if status == "not_credible" else "C"
        reason = blocking[0]
    elif pass_count == len(checks):
        status = "high_credibility"
        grade = "A"
        reason = "Validation passes coverage, statistical, alignment, top-k, slice, and robustness checks."
    elif fail_count == 0 and checks["statistical_support"]["status"] == "pass":
        status = "moderate_credibility"
        grade = "B"
        reason = "Validation is statistically supported, with warnings that must be disclosed."
    else:
        status = "low_credibility"
        grade = "C"
        reason = "Validation has partial signal but is not strong enough for a high-credibility claim."
    return {
        "status": status,
        "grade": grade,
        "can_claim_high_validation_credibility": status == "high_credibility",
        "reason": reason,
        "checks": checks,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "recommended_next_actions": _scorecard_actions(checks, status),
    }


def _coverage_check(summary: dict[str, Any]) -> dict[str, Any]:
    matched = int(summary.get("matched_paper_count", 0) or 0)
    score_match = float(summary.get("score_label_match_rate", 0.0) or 0.0)
    label_match = float(summary.get("label_match_rate", 0.0) or 0.0)
    value = min(score_match, label_match)
    if matched < 20:
        return _check(
            "fail",
            value,
            "Fewer than 20 matched papers; validation coverage is too small.",
        )
    if value >= 0.8:
        return _check("pass", value, "At least 80% of score and label rows are matched.")
    if value >= 0.5:
        return _check("warn", value, "Matched coverage is usable but should be improved.")
    return _check("fail", value, "Matched coverage is below 50%; entity coverage may dominate metrics.")


def _statistical_support_check(significance: dict[str, Any]) -> dict[str, Any]:
    value = float(significance.get("observed", 0.0) or 0.0)
    if significance.get("significant_vs_random"):
        return _check("pass", value, "Pairwise validity beats the random-label baseline.")
    return _check(
        "fail",
        value,
        significance.get(
            "reason",
            "Validity is not statistically supported against the random-label baseline.",
        ),
    )


def _rank_alignment_check(summary: dict[str, Any]) -> dict[str, Any]:
    spearman = float(summary.get("spearman_correlation", 0.0) or 0.0)
    kendall = float(summary.get("kendall_tau_b", 0.0) or 0.0)
    pairwise = float(summary.get("pairwise_accuracy", 0.0) or 0.0)
    value = min(spearman, kendall, pairwise)
    if spearman >= 0.5 and kendall >= 0.35 and pairwise >= 0.65:
        return _check("pass", value, "Rank, tie-aware rank, and pairwise metrics are all strong.")
    if spearman >= 0.25 or kendall >= 0.2 or pairwise >= 0.58:
        return _check("warn", value, "Some rank-alignment signal exists but it is not strong.")
    return _check("fail", value, "Rank-alignment metrics are weak.")


def _top_rank_quality_check(summary: dict[str, Any]) -> dict[str, Any]:
    ndcg = float(summary.get("ndcg_at_top_k", 0.0) or 0.0)
    precision = float(summary.get("top_k_precision", 0.0) or 0.0)
    value = min(ndcg, precision)
    if ndcg >= 0.75 and precision >= 0.5:
        return _check("pass", value, "Top-ranked papers agree with the external/future label.")
    if ndcg >= 0.6 or precision >= 0.3:
        return _check("warn", value, "Top-ranked quality is usable but not strong.")
    return _check("fail", value, "Top-ranked papers do not align well with the label.")


def _slice_stability_check(by_field: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = {
        field: summary
        for field, summary in by_field.items()
        if int(summary.get("matched_paper_count", 0) or 0) >= 10
    }
    if not eligible:
        return _check(
            "warn",
            0.0,
            "No field slice has at least 10 matched papers; slice stability is not established.",
        )
    worst_pairwise = min(
        float(summary.get("pairwise_accuracy", 0.0) or 0.0)
        for summary in eligible.values()
    )
    worst_spearman = min(
        float(summary.get("spearman_correlation", 0.0) or 0.0)
        for summary in eligible.values()
    )
    value = min(worst_pairwise, worst_spearman)
    if worst_pairwise >= 0.58 and worst_spearman >= 0.1:
        return _check("pass", value, "Eligible field slices show positive validity signal.")
    if worst_pairwise >= 0.5 and worst_spearman >= 0.0:
        return _check("warn", value, "At least one field slice has weak validity signal.")
    return _check("fail", value, "At least one field slice is random-like or negative.")


def _outlier_robustness_check(robustness: dict[str, Any]) -> dict[str, Any]:
    status = robustness.get("status")
    value = max(
        (
            float(check.get("max_abs_delta", 0.0) or 0.0)
            for check in robustness.get("checks", {}).values()
        ),
        default=0.0,
    )
    if status == "stable":
        return _check("pass", value, "Validity is stable after removing extreme rows.")
    if status in {"moderately_stable", "insufficient_validation_data"}:
        return _check("warn", value, robustness.get("reason", "Robustness is limited."))
    return _check("fail", value, robustness.get("reason", "Validity is outlier sensitive."))


def _scorecard_actions(checks: dict[str, dict[str, Any]], status: str) -> list[str]:
    actions: list[str] = []
    if checks["coverage"]["status"] != "pass":
        actions.append("Improve paper ID/entity matching or label coverage before claiming validated impact accuracy.")
    if checks["statistical_support"]["status"] != "pass":
        actions.append("Use more labels, a larger future window, or a stronger external benchmark until the score beats a random baseline.")
    if checks["rank_alignment"]["status"] != "pass":
        actions.append("Recalibrate fusion weights on a development split and re-run validity evaluation.")
    if checks["top_rank_quality"]["status"] != "pass":
        actions.append("Inspect top-ranked false positives and strengthen evidence requirements for top-paper claims.")
    if checks["slice_stability"]["status"] != "pass":
        actions.append("Report field-slice limitations and collect more labels for weak slices.")
    if checks["outlier_robustness"]["status"] != "pass":
        actions.append("Audit extreme citation/adoption rows and prefer rank-normalized or winsorized labels for validation.")
    if not actions and status == "high_credibility":
        actions.append("Use the scorecard, validity report, and claim certificate together in the final report.")
    return actions


def _check(status: str, value: float, reason: str) -> dict[str, Any]:
    return {"status": status, "value": _round(float(value)), "reason": reason}


def _metric_values(
    rows: list[dict[str, Any]],
    top_fraction: float,
    min_label_delta: float,
) -> dict[str, float]:
    top_k = _top_k(len(rows), top_fraction)
    return {
        "spearman_correlation": _spearman(rows),
        "kendall_tau_b": _kendall_tau_b(rows),
        "pairwise_accuracy": _pairwise_accuracy(
            rows,
            min_label_delta=min_label_delta,
        )["accuracy"],
        "ndcg_at_top_k": _ndcg_at_k(rows, top_k),
    }


def _statistical_significance(
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    metric = "pairwise_accuracy"
    permutation_metric = permutation.get("metrics", {}).get(metric, {})
    bootstrap_metric = bootstrap.get("metrics", {}).get(metric, {})
    p_value = float(permutation_metric.get("p_value", 1.0) or 1.0)
    ci95 = bootstrap_metric.get("ci95", [0.0, 0.0])
    observed = float(summary.get(metric, 0.0) or 0.0)
    significant = (
        int(summary.get("matched_paper_count", 0) or 0) >= 20
        and observed > 0.5
        and _ci_lower(ci95) > 0.5
        and p_value <= 0.05
    )
    if significant:
        status = "statistically_supported"
        reason = "Pairwise accuracy is above chance with bootstrap and permutation support."
    elif int(summary.get("matched_paper_count", 0) or 0) < 20:
        status = "insufficient_validation_data"
        reason = "Fewer than 20 matched papers; statistical support is not claimed."
    else:
        status = "not_statistically_supported"
        reason = "Pairwise accuracy is not significantly above a random-label baseline."
    return {
        "status": status,
        "reason": reason,
        "tested_metric": metric,
        "observed": observed,
        "bootstrap_ci95": ci95,
        "permutation_null_mean": permutation_metric.get("null_mean", 0.0),
        "p_value": p_value,
        "significant_vs_random": significant,
    }


def _top_k(count: int, top_fraction: float) -> int:
    if count <= 0:
        return 0
    fraction = min(max(top_fraction, 0.0), 1.0)
    return max(1, math.ceil(count * fraction))


def _spearman(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    score_ranks = _average_ranks([row["score"] for row in rows])
    label_ranks = _average_ranks([row["label"] for row in rows])
    return _round(_pearson(score_ranks, label_ranks))


def _kendall_tau_b(rows: list[dict[str, Any]]) -> float:
    concordant = 0
    discordant = 0
    score_ties = 0
    label_ties = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            score_cmp = _compare(left["score"], right["score"])
            label_cmp = _compare(left["label"], right["label"])
            if score_cmp == 0 and label_cmp == 0:
                continue
            if score_cmp == 0:
                score_ties += 1
            elif label_cmp == 0:
                label_ties += 1
            elif score_cmp == label_cmp:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + score_ties)
        * (concordant + discordant + label_ties)
    )
    return _round((concordant - discordant) / denominator if denominator else 0.0)


def _pairwise_accuracy(
    rows: list[dict[str, Any]],
    *,
    min_label_delta: float,
) -> dict[str, Any]:
    comparable = 0
    correct = 0
    ties = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            label_delta = left["label"] - right["label"]
            if abs(label_delta) <= min_label_delta:
                continue
            comparable += 1
            score_delta = left["score"] - right["score"]
            if score_delta == 0:
                ties += 1
            elif (score_delta > 0 and label_delta > 0) or (
                score_delta < 0 and label_delta < 0
            ):
                correct += 1
    return {
        "accuracy": correct / comparable if comparable else 0.0,
        "comparable_pairs": comparable,
        "correct_pairs": correct,
        "tie_pairs": ties,
    }


def _ndcg_at_k(rows: list[dict[str, Any]], k: int) -> float:
    if k <= 0 or not rows:
        return 0.0
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:k]
    ideal = sorted(rows, key=lambda row: row["label"], reverse=True)[:k]
    dcg = _dcg([row["label"] for row in ranked])
    ideal_dcg = _dcg([row["label"] for row in ideal])
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _dcg(relevances: list[float]) -> float:
    return sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def _top_k_precision_recall(rows: list[dict[str, Any]], k: int) -> tuple[float, float]:
    if k <= 0 or not rows:
        return 0.0, 0.0
    predicted = {
        row["paper_id"]
        for row in sorted(rows, key=lambda item: item["score"], reverse=True)[:k]
    }
    truth = {
        row["paper_id"]
        for row in sorted(rows, key=lambda item: item["label"], reverse=True)[:k]
    }
    overlap = len(predicted & truth)
    return overlap / len(predicted), overlap / len(truth)


def _examples(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": row["paper_id"],
            "score": row["score"],
            "label": row["label"],
            "field": row["field"],
        }
        for row in sorted(
            rows,
            key=lambda row: abs(row["score"] - row["label"]),
            reverse=True,
        )[:limit]
    ]


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        rank = (index + 1 + next_index) / 2
        for original_index, _ in indexed[index:next_index]:
            ranks[original_index] = rank
        index = next_index
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_denominator = sum((value - left_mean) ** 2 for value in left)
    right_denominator = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_denominator * right_denominator)
    return numerator / denominator if denominator else 0.0


def _decision(
    matched: int,
    spearman: float,
    pairwise_accuracy: float,
    ndcg: float,
    significance: dict[str, Any] | None,
) -> dict[str, Any]:
    if matched < 20:
        return {
            "status": "insufficient_validation_data",
            "reason": "Fewer than 20 matched papers; use this as a smoke check only.",
        }
    significant = True if significance is None else bool(
        significance.get("significant_vs_random")
    )
    if significance is not None and not significant:
        return {
            "status": "not_validated",
            "reason": "Point metrics are not statistically supported against a random-label baseline.",
        }
    if spearman >= 0.5 and pairwise_accuracy >= 0.65 and ndcg >= 0.75:
        return {
            "status": "externally_supported",
            "reason": "Scores align with external/future labels across rank, pairwise, and top-k metrics.",
        }
    if spearman >= 0.25 or pairwise_accuracy >= 0.55 or ndcg >= 0.6:
        return {
            "status": "usable_with_disclosure",
            "reason": "Scores show some validity signal but need stronger benchmark or future-window support.",
        }
    return {
        "status": "not_validated",
        "reason": "Scores do not align strongly enough with the supplied external/future labels.",
    }


def _compare(left: float, right: float) -> int:
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return _text(value)


def _ci95(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    sorted_values = sorted(values)
    lower_index = max(0, math.floor((len(sorted_values) - 1) * 0.025))
    upper_index = min(len(sorted_values) - 1, math.ceil((len(sorted_values) - 1) * 0.975))
    return [_round(sorted_values[lower_index]), _round(sorted_values[upper_index])]


def _ci_lower(value: Any) -> float:
    if not isinstance(value, list | tuple) or not value:
        return 0.0
    try:
        return float(value[0])
    except (TypeError, ValueError):
        return 0.0


def _permutation_p_value(observed: float, null_values: list[float]) -> float:
    if not null_values:
        return 1.0
    exceedances = sum(1 for value in null_values if value >= observed)
    return _round((exceedances + 1) / (len(null_values) + 1))


def _ci(value: Any) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_format_value(value[0])}, {_format_value(value[1])}]"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _joined_ids(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    return ", ".join(_text(item) for item in value[:20])


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _round(value: float) -> float:
    return round(value, 6)
