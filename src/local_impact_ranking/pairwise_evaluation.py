from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def evaluate_pairwise_labels(
    rankings_path: str | Path,
    labels_path: str | Path,
    *,
    score_column: str | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    scores = _load_scores(rankings_path, score_column=score_column)
    labels = _load_labels(labels_path)
    examples: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    for label in labels:
        winner_id = str(label["winner_id"])
        loser_id = str(label["loser_id"])
        winner_score = scores.get(winner_id)
        loser_score = scores.get(loser_id)
        covered = winner_score is not None and loser_score is not None
        if not covered:
            outcome = "uncovered"
        elif winner_score > loser_score:
            outcome = "correct"
        elif winner_score < loser_score:
            outcome = "wrong"
        else:
            outcome = "tie"
        row = {
            "pair_id": label.get("pair_id", ""),
            "winner_id": winner_id,
            "loser_id": loser_id,
            "winner_score": winner_score,
            "loser_score": loser_score,
            "impact_dimension": label.get("impact_dimension", "unknown"),
            "field": label.get("field", "unknown"),
            "outcome": outcome,
        }
        outcomes.append(row)
        if len(examples) < max_examples:
            examples.append(row)

    summary = _summarize(outcomes)
    by_dimension = _summarize_groups(outcomes, "impact_dimension")
    by_field = _summarize_groups(outcomes, "field")
    significance = _statistical_significance(summary)
    scorecard = _credibility_scorecard(
        summary,
        significance=significance,
        by_dimension=by_dimension,
        by_field=by_field,
    )
    return {
        "summary": summary,
        "statistical_significance": significance,
        "credibility_scorecard": scorecard,
        "by_dimension": by_dimension,
        "by_field": by_field,
        "examples": examples,
    }


def write_pairwise_evaluation(
    rankings_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    score_column: str | None = None,
) -> dict[str, Any]:
    report = evaluate_pairwise_labels(
        rankings_path,
        labels_path,
        score_column=score_column,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pairwise_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "pairwise_evaluation.md").write_text(
        render_pairwise_evaluation_markdown(report),
        encoding="utf-8",
    )
    (destination / "pairwise_credibility_scorecard.json").write_text(
        json.dumps(report["credibility_scorecard"], indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "pairwise_credibility_scorecard.md").write_text(
        render_pairwise_credibility_scorecard_markdown(
            report["credibility_scorecard"]
        ),
        encoding="utf-8",
    )
    return report


def render_pairwise_evaluation_markdown(report: dict[str, Any]) -> str:
    scorecard = report.get("credibility_scorecard", {})
    significance = report.get("statistical_significance", {})
    lines = [
        "# Pairwise Impact Evaluation",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            f"| significant_vs_random | {_yes_no(significance.get('significant_vs_random'))} |",
            f"| significance_p_value | {_format_value(significance.get('p_value'))} |",
            f"| credibility_status | {scorecard.get('status', '')} |",
            f"| credibility_grade | {scorecard.get('grade', '')} |",
            "",
            "## Credibility Scorecard",
            "",
            "| Check | Status | Value | Reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', '')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## By Impact Dimension",
            "",
            "| Group | Total | Covered | Pairwise Accuracy | Coverage | Tie Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_dimension"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## By Field",
            "",
            "| Group | Total | Covered | Pairwise Accuracy | Coverage | Tie Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_field"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Pairwise accuracy measures whether the local score ranks the benchmark "
                "winner above the benchmark loser. Uncovered pairs are reported separately "
                "instead of counted as wrong, because they indicate entity-resolution or "
                "dataset-overlap gaps."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_pairwise_credibility_scorecard_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Credibility Scorecard",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {scorecard.get('status', '')} |",
        f"| grade | {scorecard.get('grade', '')} |",
        f"| can_claim_high_benchmark_credibility | {_yes_no(scorecard.get('can_claim_high_benchmark_credibility'))} |",
        f"| reason | {scorecard.get('reason', '')} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', '')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    blockers = scorecard.get("blocking_reasons", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = scorecard.get("warnings", [])
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _load_scores(path: str | Path, *, score_column: str | None) -> dict[str, float]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Ranking file has no header: {source}")
        selected_column = _select_score_column(reader.fieldnames, score_column)
        scores: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            identifier = row.get("id") or row.get("paper_id")
            if identifier is None or identifier == "":
                raise ValueError(f"Ranking row {row_number} is missing id/paper_id")
            raw_score = row.get(selected_column)
            if raw_score is None or raw_score == "":
                raise ValueError(
                    f"Ranking row {row_number} is missing score column {selected_column}"
                )
            scores[identifier] = float(raw_score)
    return scores


def _select_score_column(fieldnames: list[str], score_column: str | None) -> str:
    if score_column is not None:
        if score_column not in fieldnames:
            raise ValueError(f"Score column not found in ranking file: {score_column}")
        return score_column
    for candidate in ("final_score", "score", "normalized_score"):
        if candidate in fieldnames:
            return candidate
    raise ValueError("Ranking file needs one of: final_score, score, normalized_score")


def _load_labels(path: str | Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "winner_id" not in record or "loser_id" not in record:
                raise ValueError(
                    f"Pairwise label on line {line_number} needs winner_id and loser_id"
                )
            labels.append(record)
    return labels


def _summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {group: _summarize(group_rows) for group, group_rows in grouped.items()}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(row["outcome"] != "uncovered" for row in rows)
    uncovered = total - covered
    correct = sum(row["outcome"] == "correct" for row in rows)
    wrong = sum(row["outcome"] == "wrong" for row in rows)
    ties = sum(row["outcome"] == "tie" for row in rows)
    non_tie = correct + wrong
    return {
        "total_pairs": total,
        "covered_pairs": covered,
        "uncovered_pairs": uncovered,
        "correct_pairs": correct,
        "wrong_pairs": wrong,
        "tie_pairs": ties,
        "non_tie_pairs": non_tie,
        "coverage_rate": _safe_divide(covered, total),
        "pairwise_accuracy": _safe_divide(correct, covered),
        "non_tie_pairwise_accuracy": _safe_divide(correct, non_tie),
        "tie_rate": _safe_divide(ties, covered),
    }


def _statistical_significance(summary: dict[str, Any]) -> dict[str, Any]:
    trials = int(summary.get("non_tie_pairs", 0) or 0)
    successes = int(summary.get("correct_pairs", 0) or 0)
    observed = _safe_divide(successes, trials)
    ci95 = _wilson_ci(successes, trials)
    p_value = _binomial_greater_equal_p_value(successes, trials)
    significant = trials >= 20 and observed > 0.5 and ci95[0] > 0.5 and p_value <= 0.05
    if significant:
        status = "statistically_supported"
        reason = "Non-tie pairwise accuracy is significantly above a random 0.5 baseline."
    elif trials < 20:
        status = "insufficient_validation_data"
        reason = "Fewer than 20 non-tie covered pairs; benchmark significance is not claimed."
    else:
        status = "not_statistically_supported"
        reason = "Pairwise benchmark accuracy is not significantly above random."
    return {
        "status": status,
        "reason": reason,
        "tested_metric": "non_tie_pairwise_accuracy",
        "successes": successes,
        "trials": trials,
        "observed": observed,
        "wilson_ci95": ci95,
        "p_value": p_value,
        "significant_vs_random": significant,
    }


def _credibility_scorecard(
    summary: dict[str, Any],
    *,
    significance: dict[str, Any],
    by_dimension: dict[str, dict[str, Any]],
    by_field: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "coverage": _coverage_check(summary),
        "statistical_support": _statistical_support_check(significance),
        "benchmark_accuracy": _benchmark_accuracy_check(summary),
        "tie_rate": _tie_rate_check(summary),
        "slice_stability": _slice_stability_check(by_dimension, by_field),
    }
    blocking = [
        check["reason"]
        for name, check in checks.items()
        if check["status"] == "fail"
        and name in {"coverage", "statistical_support", "benchmark_accuracy"}
    ]
    warnings = [
        check["reason"] for check in checks.values() if check["status"] == "warn"
    ]
    fail_count = sum(1 for check in checks.values() if check["status"] == "fail")
    pass_count = sum(1 for check in checks.values() if check["status"] == "pass")
    if int(summary.get("non_tie_pairs", 0) or 0) < 20:
        status = "insufficient_validation_data"
        grade = "D"
        reason = "Fewer than 20 non-tie covered pairs; use this as a smoke check only."
    elif blocking:
        status = "not_credible"
        grade = "D"
        reason = blocking[0]
    elif pass_count == len(checks):
        status = "high_credibility"
        grade = "A"
        reason = "Pairwise benchmark validity passes coverage, significance, accuracy, tie, and slice checks."
    elif fail_count == 0 and checks["statistical_support"]["status"] == "pass":
        status = "moderate_credibility"
        grade = "B"
        reason = "Pairwise benchmark validity is supported, with warnings to disclose."
    else:
        status = "low_credibility"
        grade = "C"
        reason = "Pairwise benchmark validity has partial signal but does not meet high-credibility thresholds."
    return {
        "status": status,
        "grade": grade,
        "can_claim_high_benchmark_credibility": status == "high_credibility",
        "reason": reason,
        "checks": checks,
        "blocking_reasons": blocking,
        "warnings": warnings,
    }


def _coverage_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("coverage_rate", 0.0) or 0.0)
    if int(summary.get("total_pairs", 0) or 0) < 20:
        return _check("fail", value, "Fewer than 20 benchmark pairs were supplied.")
    if value >= 0.9:
        return _check("pass", value, "At least 90% of benchmark pairs are covered.")
    if value >= 0.75:
        return _check("warn", value, "Benchmark coverage is usable but incomplete.")
    return _check("fail", value, "Benchmark coverage is too low for a credible claim.")


def _statistical_support_check(significance: dict[str, Any]) -> dict[str, Any]:
    value = float(significance.get("observed", 0.0) or 0.0)
    if significance.get("significant_vs_random"):
        return _check("pass", value, "Accuracy is significantly above the random baseline.")
    return _check("fail", value, significance.get("reason", "Benchmark significance failed."))


def _benchmark_accuracy_check(summary: dict[str, Any]) -> dict[str, Any]:
    pairwise = float(summary.get("pairwise_accuracy", 0.0) or 0.0)
    non_tie = float(summary.get("non_tie_pairwise_accuracy", 0.0) or 0.0)
    value = min(pairwise, non_tie)
    if pairwise >= 0.6 and non_tie >= 0.65:
        return _check("pass", value, "Pairwise and non-tie accuracy are both strong.")
    if pairwise >= 0.55 or non_tie >= 0.58:
        return _check("warn", value, "Benchmark accuracy has signal but is not strong.")
    return _check("fail", value, "Benchmark accuracy is too close to random.")


def _tie_rate_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("tie_rate", 0.0) or 0.0)
    if value <= 0.05:
        return _check("pass", value, "Tie rate is low.")
    if value <= 0.15:
        return _check("warn", value, "Tie rate is elevated and should be disclosed.")
    return _check("fail", value, "Tie rate is high enough to weaken pairwise conclusions.")


def _slice_stability_check(
    by_dimension: dict[str, dict[str, Any]],
    by_field: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        (f"dimension:{name}", summary)
        for name, summary in by_dimension.items()
        if int(summary.get("non_tie_pairs", 0) or 0) >= 10
    ] + [
        (f"field:{name}", summary)
        for name, summary in by_field.items()
        if int(summary.get("non_tie_pairs", 0) or 0) >= 10
    ]
    if not eligible:
        return _check(
            "warn",
            0.0,
            "No dimension or field slice has at least 10 non-tie covered pairs.",
        )
    worst = min(
        float(summary.get("non_tie_pairwise_accuracy", 0.0) or 0.0)
        for _, summary in eligible
    )
    if worst >= 0.58:
        return _check("pass", worst, "Eligible benchmark slices remain above random.")
    if worst >= 0.5:
        return _check("warn", worst, "At least one benchmark slice is weak.")
    return _check("fail", worst, "At least one benchmark slice is below random.")


def _check(status: str, value: float, reason: str) -> dict[str, Any]:
    return {"status": status, "value": round(float(value), 6), "reason": reason}


def _wilson_ci(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p_hat = successes / trials
    denominator = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _binomial_greater_equal_p_value(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    if successes <= trials / 2:
        return 1.0
    if trials <= 1000:
        probability = sum(math.comb(trials, k) for k in range(successes, trials + 1))
        return round(probability * (0.5 ** trials), 6)
    z = (successes - trials * 0.5 - 0.5) / math.sqrt(trials * 0.25)
    return round(0.5 * math.erfc(z / math.sqrt(2)), 6)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_row(group: str, summary: dict[str, Any]) -> str:
    return (
        f"| {group} | {summary['total_pairs']} | {summary['covered_pairs']} | "
        f"{summary['pairwise_accuracy']:.3f} | {summary['coverage_rate']:.3f} | "
        f"{summary['tie_rate']:.3f} |"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
