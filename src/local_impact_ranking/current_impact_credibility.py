from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.impact_validity_evaluation import evaluate_impact_validity


DEFAULT_CURRENT_LABEL_COLUMNS = (
    "rcr",
    "citations_per_year",
    "apt",
    "nih_percentile",
    "icite_citation_count",
)

DEFAULT_CURRENT_SCORE_COLUMNS = (
    "final_impact_score",
    "score_95_true_05_effective",
    "score_90_true_10_effective",
    "score_80_true_20_effective",
    "score_70_true_30_effective",
    "score_50_true_50_effective",
    "score_30_true_70_effective",
)


def evaluate_current_impact_credibility(
    scores_path: str | Path,
    labels_path: str | Path,
    *,
    score_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
    primary_label_column: str | None = None,
    top_fraction: float = 0.1,
    bootstrap_samples: int = 100,
    permutation_samples: int = 100,
    random_seed: int = 0,
) -> dict[str, Any]:
    scores_path = Path(scores_path)
    labels_path = Path(labels_path)
    score_columns = score_columns or list(DEFAULT_CURRENT_SCORE_COLUMNS)
    label_columns = label_columns or list(DEFAULT_CURRENT_LABEL_COLUMNS)
    score_rows = _read_table(scores_path)
    label_rows = _read_table(labels_path)
    available_score_columns = [
        column for column in score_columns if _column_has_numeric(score_rows, column)
    ]
    available_label_columns = [
        column for column in label_columns if _column_has_numeric(label_rows, column)
    ]
    if not available_score_columns:
        raise ValueError("No requested score columns contain numeric values.")
    if not available_label_columns:
        raise ValueError("No requested label columns contain numeric values.")
    primary_label = _select_primary_label_column(
        available_label_columns,
        primary_label_column,
    )

    alignment: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="current-impact-labels-") as temp_dir:
        temp_root = Path(temp_dir)
        filtered_label_paths = {
            label_column: _write_filtered_label_table(
                label_rows,
                label_column,
                temp_root / f"{label_column}.csv",
            )
            for label_column in available_label_columns
        }
        for score_column in available_score_columns:
            alignment[score_column] = {}
            for label_column in available_label_columns:
                report = evaluate_impact_validity(
                    scores_path,
                    filtered_label_paths[label_column],
                    score_column=score_column,
                    label_column=label_column,
                    top_fraction=top_fraction,
                    bootstrap_samples=bootstrap_samples,
                    permutation_samples=permutation_samples,
                    random_seed=random_seed,
                )
                summary = report["summary"]
                alignment[score_column][label_column] = {
                    "matched_paper_count": summary["matched_paper_count"],
                    "score_label_match_rate": summary["score_label_match_rate"],
                    "spearman_correlation": summary["spearman_correlation"],
                    "kendall_tau_b": summary["kendall_tau_b"],
                    "pairwise_accuracy": summary["pairwise_accuracy"],
                    "ndcg_at_top_k": summary["ndcg_at_top_k"],
                    "decision_status": summary["decision"]["status"],
                    "credibility_status": report["credibility_scorecard"]["status"],
                }

    coverage = _coverage_summary(score_rows, label_rows, available_label_columns)
    sensitivity = _sensitivity_summary(score_rows, available_score_columns)
    scorecard = _current_credibility_scorecard(
        coverage,
        alignment,
        sensitivity,
        primary_label_column=primary_label,
    )
    return {
        "summary": {
            "score_path": str(scores_path),
            "labels_path": str(labels_path),
            "score_columns": available_score_columns,
            "label_columns": available_label_columns,
            "primary_label_column": primary_label,
            "top_fraction": top_fraction,
            "credibility_status": scorecard["status"],
            "credibility_grade": scorecard["grade"],
            "can_claim_current_impact_credible": scorecard[
                "can_claim_current_impact_credible"
            ],
        },
        "coverage": coverage,
        "alignment": alignment,
        "sensitivity": sensitivity,
        "credibility_scorecard": scorecard,
        "interpretation": {
            "claim": "This evaluates whether current final_impact_score aligns with a preselected primary current external impact metric, is corroborated by secondary current metrics, and remains stable under true/effective citation weight changes.",
            "not_a_prediction": "This is not a future-impact prediction benchmark.",
            "limits": [
                "Citation-derived labels such as iCite citation_count are not independent of true_citation_score.",
                "RCR, NIH percentile, APT, and influential-citation labels are better current external validity checks when available.",
                "Low effective-citation coverage means sensitivity fields should be disclosed with the final score.",
            ],
        },
    }


def write_current_impact_credibility(
    scores_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    score_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
    primary_label_column: str | None = None,
    top_fraction: float = 0.1,
    bootstrap_samples: int = 100,
    permutation_samples: int = 100,
    random_seed: int = 0,
) -> dict[str, Any]:
    report = evaluate_current_impact_credibility(
        scores_path,
        labels_path,
        score_columns=score_columns,
        label_columns=label_columns,
        primary_label_column=primary_label_column,
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "current_impact_credibility.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "current_impact_credibility.md").write_text(
        render_current_impact_credibility_markdown(report),
        encoding="utf-8",
    )
    return report


def render_current_impact_credibility_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    scorecard = report["credibility_scorecard"]
    coverage = report["coverage"]
    lines = [
        "# Current Impact Credibility",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| credibility_status | {scorecard['status']} |",
        f"| credibility_grade | {scorecard['grade']} |",
        f"| can_claim_current_impact_credible | {_yes_no(scorecard['can_claim_current_impact_credible'])} |",
        f"| primary_label_column | {summary['primary_label_column']} |",
        f"| scored_paper_count | {coverage['scored_paper_count']} |",
        f"| current_label_paper_count | {coverage['label_paper_count']} |",
        f"| true_citation_coverage_rate | {_number(coverage['true_citation_coverage_rate'])} |",
        f"| effective_citation_coverage_rate | {_number(coverage['effective_citation_coverage_rate'])} |",
        f"| both_signal_coverage_rate | {_number(coverage['both_signal_coverage_rate'])} |",
        "",
        "## Scorecard",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in scorecard["checks"].items():
        lines.append(
            f"| {name} | {check['status']} | {_number(check['value'])} | {check['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Current Metric Alignment",
            "",
            "| Score | Label | Matched | Spearman | Kendall | Pairwise Acc | NDCG@K | Credibility |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for score_column, labels in report["alignment"].items():
        for label_column, metrics in labels.items():
            lines.append(
                f"| {score_column} | {label_column} | "
                f"{metrics['matched_paper_count']} | "
                f"{_number(metrics['spearman_correlation'])} | "
                f"{_number(metrics['kendall_tau_b'])} | "
                f"{_number(metrics['pairwise_accuracy'])} | "
                f"{_number(metrics['ndcg_at_top_k'])} | "
                f"{metrics['credibility_status']} |"
            )
    lines.extend(
        [
            "",
            "## Weight Sensitivity",
            "",
            "| Score | Spearman vs Final | Top-K Overlap |",
            "| --- | ---: | ---: |",
        ]
    )
    for score_column, metrics in report["sensitivity"]["by_score"].items():
        lines.append(
            f"| {score_column} | "
            f"{_number(metrics['spearman_vs_final'])} | "
            f"{_number(metrics['top_k_overlap_with_final'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"]["claim"],
            report["interpretation"]["not_a_prediction"],
            "",
        ]
    )
    for limit in report["interpretation"]["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    return "\n".join(lines)


def _coverage_summary(
    score_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    label_columns: list[str],
) -> dict[str, Any]:
    score_ids = {row.get("paper_id", "") for row in score_rows if row.get("paper_id")}
    label_ids = {row.get("paper_id", "") for row in label_rows if row.get("paper_id")}
    matched_ids = score_ids & label_ids
    true_count = sum(1 for row in score_rows if _numeric(row.get("true_citation_raw")) is not None)
    effective_count = sum(
        1
        for row in score_rows
        if (_numeric(row.get("effective_citation_edge_count")) or 0.0) > 0.0
    )
    both_count = sum(
        1
        for row in score_rows
        if _numeric(row.get("true_citation_raw")) is not None
        and (_numeric(row.get("effective_citation_edge_count")) or 0.0) > 0.0
    )
    label_coverage = {
        column: sum(1 for row in label_rows if _numeric(row.get(column)) is not None)
        for column in label_columns
    }
    paper_count = len(score_rows)
    return {
        "scored_paper_count": paper_count,
        "label_paper_count": len(label_ids),
        "matched_paper_count": len(matched_ids),
        "score_label_match_rate": round(len(matched_ids) / paper_count if paper_count else 0.0, 6),
        "true_citation_paper_count": true_count,
        "effective_citation_paper_count": effective_count,
        "both_signal_paper_count": both_count,
        "true_citation_coverage_rate": round(true_count / paper_count if paper_count else 0.0, 6),
        "effective_citation_coverage_rate": round(effective_count / paper_count if paper_count else 0.0, 6),
        "both_signal_coverage_rate": round(both_count / paper_count if paper_count else 0.0, 6),
        "label_column_coverage": label_coverage,
    }


def _sensitivity_summary(
    rows: list[dict[str, str]],
    score_columns: list[str],
    *,
    final_column: str = "final_impact_score",
    top_fraction: float = 0.1,
) -> dict[str, Any]:
    final_scores = _score_values(rows, final_column)
    by_score = {}
    for column in score_columns:
        if column == final_column:
            continue
        values = _score_values(rows, column)
        common = sorted(set(final_scores) & set(values))
        if not common:
            continue
        final_common = {paper_id: final_scores[paper_id] for paper_id in common}
        values_common = {paper_id: values[paper_id] for paper_id in common}
        by_score[column] = {
            "spearman_vs_final": _spearman(final_common, values_common),
            "top_k_overlap_with_final": _top_k_overlap(
                final_common,
                values_common,
                top_fraction=top_fraction,
            ),
            "matched_paper_count": len(common),
        }
    gate_columns = [
        column for column in by_score if _is_gated_sensitivity_score_column(column)
    ]
    gate_metrics = {
        column: by_score[column]
        for column in gate_columns
    } or by_score
    min_spearman = min(
        (metrics["spearman_vs_final"] for metrics in gate_metrics.values()),
        default=1.0,
    )
    min_top_overlap = min(
        (metrics["top_k_overlap_with_final"] for metrics in gate_metrics.values()),
        default=1.0,
    )
    all_min_spearman = min(
        (metrics["spearman_vs_final"] for metrics in by_score.values()),
        default=1.0,
    )
    all_min_top_overlap = min(
        (metrics["top_k_overlap_with_final"] for metrics in by_score.values()),
        default=1.0,
    )
    return {
        "final_score_column": final_column,
        "by_score": by_score,
        "gate_score_columns": gate_columns,
        "min_spearman_vs_final": round(min_spearman, 6),
        "min_top_k_overlap_with_final": round(min_top_overlap, 6),
        "all_score_min_spearman_vs_final": round(all_min_spearman, 6),
        "all_score_min_top_k_overlap_with_final": round(all_min_top_overlap, 6),
    }


def _is_gated_sensitivity_score_column(column: str) -> bool:
    match = re.fullmatch(r"score_(\d+)_true_(\d+)_effective", column)
    if not match:
        return False
    true_weight = int(match.group(1))
    effective_weight = int(match.group(2))
    return true_weight + effective_weight == 100 and true_weight >= 80


def _current_credibility_scorecard(
    coverage: dict[str, Any],
    alignment: dict[str, dict[str, Any]],
    sensitivity: dict[str, Any],
    *,
    primary_label_column: str,
) -> dict[str, Any]:
    best_alignment = _best_alignment(alignment)
    primary_alignment = _primary_alignment(alignment, primary_label_column)
    checks = {
        "coverage": _coverage_check(coverage),
        "primary_external_alignment": _alignment_check(
            primary_alignment,
            label="Primary current external metric",
        ),
        "external_alignment": _alignment_check(best_alignment),
        "weight_sensitivity": _sensitivity_check(sensitivity),
    }
    statuses = {check["status"] for check in checks.values()}
    if "fail" in statuses:
        status = "not_credible"
        grade = "D"
        reason = "At least one current-impact credibility gate failed."
    elif "warn" in statuses:
        status = "moderate_credibility"
        grade = "B"
        reason = "Current-impact evidence is usable but requires disclosure."
    else:
        status = "high_credibility"
        grade = "A"
        reason = "Current-impact scores align with external metrics and are stable under weight sensitivity checks."
    return {
        "status": status,
        "grade": grade,
        "reason": reason,
        "can_claim_current_impact_credible": status in {
            "high_credibility",
            "moderate_credibility",
        },
        "primary_label_column": primary_label_column,
        "primary_alignment": primary_alignment,
        "best_alignment": best_alignment,
        "checks": checks,
        "recommended_next_actions": _recommended_actions(status, checks),
    }


def _coverage_check(coverage: dict[str, Any]) -> dict[str, Any]:
    true_rate = float(coverage.get("true_citation_coverage_rate") or 0.0)
    effective_rate = float(coverage.get("effective_citation_coverage_rate") or 0.0)
    match_rate = float(coverage.get("score_label_match_rate") or 0.0)
    value = min(true_rate, match_rate)
    if true_rate >= 0.8 and match_rate >= 0.8 and effective_rate >= 0.3:
        return _check("pass", value, "True citation, current-label, and effective-citation coverage are sufficient.")
    if true_rate >= 0.6 and match_rate >= 0.6:
        return _check("warn", value, "Coverage is usable, but effective-citation or label coverage requires disclosure.")
    return _check("fail", value, "Coverage is too low for a credible current-impact claim.")


def _alignment_check(
    best: dict[str, Any],
    *,
    label: str = "Current score",
) -> dict[str, Any]:
    value = float(best.get("spearman_correlation") or 0.0)
    pairwise = float(best.get("pairwise_accuracy") or 0.0)
    if value >= 0.75 and pairwise >= 0.8:
        return _check("pass", value, f"{label} aligns strongly with a current external metric.")
    if value >= 0.5 and pairwise >= 0.7:
        return _check("warn", value, f"{label} has moderate external alignment; disclose the benchmark source.")
    return _check("fail", value, f"{label} does not align strongly enough with external metrics.")


def _sensitivity_check(sensitivity: dict[str, Any]) -> dict[str, Any]:
    value = float(sensitivity.get("min_spearman_vs_final") or 0.0)
    overlap = float(sensitivity.get("min_top_k_overlap_with_final") or 0.0)
    if value >= 0.85 and overlap >= 0.7:
        return _check("pass", value, "Top ranks are stable across true/effective citation weight variants.")
    if value >= 0.65 and overlap >= 0.5:
        return _check("warn", value, "Weight variants are moderately stable; show sensitivity fields in reports.")
    return _check("fail", value, "Ranking is too sensitive to true/effective citation weights.")


def _best_alignment(alignment: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {
        "score_column": "",
        "label_column": "",
        "spearman_correlation": 0.0,
        "pairwise_accuracy": 0.0,
        "ndcg_at_top_k": 0.0,
    }
    for score_column, labels in alignment.items():
        for label_column, metrics in labels.items():
            candidate = {
                "score_column": score_column,
                "label_column": label_column,
                **metrics,
            }
            if (
                float(candidate.get("spearman_correlation") or 0.0),
                float(candidate.get("pairwise_accuracy") or 0.0),
                float(candidate.get("ndcg_at_top_k") or 0.0),
            ) > (
                float(best.get("spearman_correlation") or 0.0),
                float(best.get("pairwise_accuracy") or 0.0),
                float(best.get("ndcg_at_top_k") or 0.0),
            ):
                best = candidate
    return best


def _primary_alignment(
    alignment: dict[str, dict[str, Any]],
    primary_label_column: str,
) -> dict[str, Any]:
    final_metrics = alignment.get("final_impact_score", {}).get(primary_label_column)
    if final_metrics is not None:
        return {
            "score_column": "final_impact_score",
            "label_column": primary_label_column,
            **final_metrics,
        }
    best: dict[str, Any] = {
        "score_column": "",
        "label_column": primary_label_column,
        "spearman_correlation": 0.0,
        "pairwise_accuracy": 0.0,
        "ndcg_at_top_k": 0.0,
    }
    for score_column, labels in alignment.items():
        metrics = labels.get(primary_label_column)
        if metrics is None:
            continue
        candidate = {
            "score_column": score_column,
            "label_column": primary_label_column,
            **metrics,
        }
        if (
            float(candidate.get("spearman_correlation") or 0.0),
            float(candidate.get("pairwise_accuracy") or 0.0),
            float(candidate.get("ndcg_at_top_k") or 0.0),
        ) > (
            float(best.get("spearman_correlation") or 0.0),
            float(best.get("pairwise_accuracy") or 0.0),
            float(best.get("ndcg_at_top_k") or 0.0),
        ):
            best = candidate
    return best


def _recommended_actions(status: str, checks: dict[str, dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if checks["coverage"]["status"] != "pass":
        actions.append("Increase true citation/current metric coverage and disclose effective-citation coverage.")
    if checks["primary_external_alignment"]["status"] != "pass":
        actions.append("Choose a primary current external metric before evaluation and improve alignment with it.")
    if checks["external_alignment"]["status"] != "pass":
        actions.append("Evaluate against additional current external metrics such as RCR, NIH percentile, or influential citations.")
    if checks["weight_sensitivity"]["status"] != "pass":
        actions.append("Report sensitivity score columns and avoid overclaiming unstable top ranks.")
    if not actions and status == "high_credibility":
        actions.append("Use the current-impact credibility report with the score cards and claim certificate.")
    return actions


def _read_table(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_filtered_label_table(
    rows: list[dict[str, str]],
    label_column: str,
    output_path: Path,
) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", label_column, "field"])
        writer.writeheader()
        for row in rows:
            paper_id = row.get("paper_id") or row.get("id")
            value = row.get(label_column)
            if not paper_id or _numeric(value) is None:
                continue
            writer.writerow(
                sanitize_csv_row(
                    {
                        "paper_id": paper_id,
                        label_column: value,
                        "field": row.get("field") or "unknown",
                    }
                )
            )
    return output_path


def _select_primary_label_column(
    available_label_columns: list[str],
    requested_primary: str | None,
) -> str:
    if requested_primary:
        if requested_primary not in available_label_columns:
            raise ValueError(
                f"Primary label column '{requested_primary}' is unavailable or non-numeric."
            )
        return requested_primary
    return available_label_columns[0]


def _column_has_numeric(rows: list[dict[str, str]], column: str) -> bool:
    return any(_numeric(row.get(column)) is not None for row in rows)


def _score_values(rows: list[dict[str, str]], column: str) -> dict[str, float]:
    values = {}
    for row in rows:
        paper_id = row.get("paper_id")
        value = _numeric(row.get(column))
        if paper_id and value is not None:
            values[str(paper_id)] = value
    return values


def _spearman(left: dict[str, float], right: dict[str, float]) -> float:
    ids = sorted(set(left) & set(right))
    if len(ids) < 2:
        return 0.0
    left_ranks = _rank_map({paper_id: left[paper_id] for paper_id in ids})
    right_ranks = _rank_map({paper_id: right[paper_id] for paper_id in ids})
    return round(_pearson([left_ranks[paper_id] for paper_id in ids], [right_ranks[paper_id] for paper_id in ids]), 6)


def _rank_map(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        value = ordered[index][1]
        while end + 1 < len(ordered) and math.isclose(ordered[end + 1][1], value):
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for offset in range(index, end + 1):
            ranks[ordered[offset][0]] = rank
        index = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _top_k_overlap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    top_fraction: float,
) -> float:
    ids = sorted(set(left) & set(right))
    if not ids:
        return 0.0
    top_k = max(1, math.ceil(len(ids) * top_fraction))
    left_top = {
        paper_id
        for paper_id, _ in sorted(
            ((paper_id, left[paper_id]) for paper_id in ids),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    }
    right_top = {
        paper_id
        for paper_id, _ in sorted(
            ((paper_id, right[paper_id]) for paper_id in ids),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    }
    return round(len(left_top & right_top) / top_k, 6)


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _check(status: str, value: float, reason: str) -> dict[str, Any]:
    return {"status": status, "value": round(float(value), 6), "reason": reason}


def _number(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
