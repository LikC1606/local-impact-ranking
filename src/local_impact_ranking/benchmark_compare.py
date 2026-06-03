from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_impact_ranking.audit import load_judge_results
from local_impact_ranking.csv_safety import sanitize_csv_row


FIELDNAMES = [
    "run",
    "paper_count",
    "reference_count",
    "judge_results",
    "parse_success_rate",
    "invalid_selection_rate",
    "empty_selection_rate",
    "top_k_compliance_rate",
    "labeled_targets",
    "hit_rate",
    "selection_precision",
    "selection_recall",
    "mean_reciprocal_rank",
    "mean_ndcg",
    "usable_for_benchmark_report",
    "needs_prompt_or_model_tuning",
]


def benchmark_comparison_rows(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    rows = [_row_from_run_dir(Path(run_dir)) for run_dir in run_dirs]
    return sorted(rows, key=lambda row: (int(row["paper_count"]), str(row["run"])))


def write_benchmark_comparison(run_dirs: list[str | Path], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = benchmark_comparison_rows(run_dirs)
    with (destination / "benchmark_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sanitize_csv_row(row) for row in rows)
    (destination / "benchmark_comparison.md").write_text(
        render_benchmark_comparison_markdown(rows),
        encoding="utf-8",
    )


def write_model_eval(
    base_run_dir: str | Path,
    candidate_run_dir: str | Path,
    output_dir: str | Path,
    *,
    max_broken_rate: float = 0.05,
    min_ndcg_delta: float = 0.0,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_benchmark_comparison([base_run_dir, candidate_run_dir], destination)
    write_gold_target_comparison(base_run_dir, candidate_run_dir, destination)
    write_model_selection_decision(
        destination,
        max_broken_rate=max_broken_rate,
        min_ndcg_delta=min_ndcg_delta,
    )
    benchmark_rows = benchmark_comparison_rows([base_run_dir, candidate_run_dir])
    target_comparison = json.loads(
        (destination / "target_comparison.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (destination / "model_selection_decision.json").read_text(encoding="utf-8")
    )
    report = {
        "base_run": Path(base_run_dir).name,
        "candidate_run": Path(candidate_run_dir).name,
        "output_dir": str(destination),
        "benchmark_comparison": benchmark_rows,
        "target_comparison_summary": target_comparison.get("summary", {}),
        "decision": decision,
        "artifacts": [
            "benchmark_comparison.csv",
            "benchmark_comparison.md",
            "target_comparison.json",
            "target_changes.csv",
            "target_comparison.md",
            "model_selection_decision.json",
            "model_selection_decision.md",
            "model_eval.json",
            "model_eval.md",
        ],
    }
    (destination / "model_eval.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "model_eval.md").write_text(
        render_model_eval_markdown(report),
        encoding="utf-8",
    )
    return report


def write_gold_target_comparison(
    base_run_dir: str | Path,
    candidate_run_dir: str | Path,
    output_dir: str | Path,
) -> None:
    base_dir = Path(base_run_dir)
    candidate_dir = Path(candidate_run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_report = json.loads(
        (base_dir / "reports" / "gold_evaluation.json").read_text(encoding="utf-8")
    )
    candidate_report = json.loads(
        (candidate_dir / "reports" / "gold_evaluation.json").read_text(encoding="utf-8")
    )
    candidate_ids_by_target = _candidate_ids_by_target(base_dir / "audit" / "judge_outputs.jsonl")
    report = compare_gold_target_reports(
        base_report,
        candidate_report,
        base_name=base_dir.name,
        candidate_name=candidate_dir.name,
        candidate_ids_by_target=candidate_ids_by_target,
    )
    (destination / "target_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_target_change_csv(report, destination / "target_changes.csv")
    (destination / "target_comparison.md").write_text(
        render_gold_target_comparison_markdown(report),
        encoding="utf-8",
    )


def write_model_selection_decision(
    comparison_dir: str | Path,
    *,
    max_broken_rate: float = 0.05,
    min_ndcg_delta: float = 0.0,
) -> None:
    destination = Path(comparison_dir)
    report = json.loads(
        (destination / "target_comparison.json").read_text(encoding="utf-8")
    )
    candidate_benchmark_summary = _read_candidate_benchmark_summary(destination, report)
    decision = model_selection_decision(
        report,
        candidate_benchmark_summary=candidate_benchmark_summary,
        max_broken_rate=max_broken_rate,
        min_ndcg_delta=min_ndcg_delta,
    )
    (destination / "model_selection_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "model_selection_decision.md").write_text(
        render_model_selection_decision_markdown(decision),
        encoding="utf-8",
    )


def compare_gold_target_reports(
    base_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    base_name: str = "base",
    candidate_name: str = "candidate",
    candidate_ids_by_target: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    base_targets = _targets_by_id(base_report)
    candidate_targets = _targets_by_id(candidate_report)
    common_ids = sorted(set(base_targets) & set(candidate_targets))
    candidate_ids_by_target = candidate_ids_by_target or {}

    buckets: dict[str, list[dict[str, Any]]] = {
        "fixed": [],
        "broken": [],
        "rank_improved": [],
        "rank_worsened": [],
        "hit_quality_changed": [],
        "unchanged_hits": [],
        "unchanged_misses": [],
    }
    all_changes = []

    for target_id in common_ids:
        base = base_targets[target_id]
        candidate = candidate_targets[target_id]
        row = _target_change_row(
            target_id,
            base,
            candidate,
            candidate_ids=candidate_ids_by_target.get(target_id, []),
        )
        all_changes.append(row)
        change_type = row["change_type"]
        if change_type == "fixed":
            buckets["fixed"].append(row)
        elif change_type == "broken":
            buckets["broken"].append(row)
        elif change_type == "rank_improved":
            buckets["rank_improved"].append(row)
        elif change_type == "rank_worsened":
            buckets["rank_worsened"].append(row)
        elif change_type == "hit_quality_changed":
            buckets["hit_quality_changed"].append(row)
        elif change_type == "unchanged_hit":
            buckets["unchanged_hits"].append(row)
        else:
            buckets["unchanged_misses"].append(row)

    return {
        "base_run": base_name,
        "candidate_run": candidate_name,
        "summary": {
            "base_targets": len(base_targets),
            "candidate_targets": len(candidate_targets),
            "common_targets": len(common_ids),
            "base_only_targets": len(set(base_targets) - set(candidate_targets)),
            "candidate_only_targets": len(set(candidate_targets) - set(base_targets)),
            "fixed_count": len(buckets["fixed"]),
            "broken_count": len(buckets["broken"]),
            "rank_improved_count": len(buckets["rank_improved"]),
            "rank_worsened_count": len(buckets["rank_worsened"]),
            "hit_quality_changed_count": len(buckets["hit_quality_changed"]),
            "unchanged_hit_count": len(buckets["unchanged_hits"]),
            "unchanged_miss_count": len(buckets["unchanged_misses"]),
            "net_hit_change": len(buckets["fixed"]) - len(buckets["broken"]),
            "mean_selection_jaccard": _mean(
                row["selection_jaccard"] for row in all_changes
            ),
            "mean_reciprocal_rank_delta": _mean(
                row["reciprocal_rank_delta"] for row in all_changes
            ),
            "mean_ndcg_delta": _mean(row["ndcg_delta"] for row in all_changes),
        },
        "fixed": buckets["fixed"],
        "broken": buckets["broken"],
        "rank_improved": buckets["rank_improved"],
        "rank_worsened": buckets["rank_worsened"],
        "hit_quality_changed": buckets["hit_quality_changed"],
        "unchanged_hits": buckets["unchanged_hits"],
        "unchanged_misses": buckets["unchanged_misses"],
        "all_changes": all_changes,
    }


def model_selection_decision(
    target_comparison_report: dict[str, Any],
    *,
    candidate_benchmark_summary: dict[str, Any] | None = None,
    max_broken_rate: float = 0.05,
    min_ndcg_delta: float = 0.0,
) -> dict[str, Any]:
    summary = target_comparison_report.get("summary", {})
    common_targets = int(summary.get("common_targets", 0))
    fixed_count = int(summary.get("fixed_count", 0))
    broken_count = int(summary.get("broken_count", 0))
    net_hit_change = int(summary.get("net_hit_change", 0))
    mean_ndcg_delta = _float(summary.get("mean_ndcg_delta"))
    broken_rate = broken_count / common_targets if common_targets else 0.0
    baseline_quality = _candidate_baseline_quality(candidate_benchmark_summary)

    checks = {
        "has_common_targets": common_targets > 0,
        "net_hit_change_positive": net_hit_change > 0,
        "mean_ndcg_delta_non_negative": mean_ndcg_delta >= min_ndcg_delta,
        "broken_rate_within_threshold": broken_rate <= max_broken_rate,
    }
    if baseline_quality["available"]:
        checks["judge_beats_all_baselines"] = baseline_quality[
            "judge_beats_all_baselines"
        ]
        checks["judge_baseline_delta_statistically_positive"] = baseline_quality[
            "statistically_positive_ndcg_delta"
        ]
    passed_checks = [name for name, passed in checks.items() if passed]
    failed_checks = [name for name, passed in checks.items() if not passed]
    passed = not failed_checks
    return {
        "recommendation": "promote_to_test" if passed else "keep_base_or_iterate",
        "pass": passed,
        "thresholds": {
            "max_broken_rate": max_broken_rate,
            "min_ndcg_delta": min_ndcg_delta,
        },
        "metrics": {
            "common_targets": common_targets,
            "fixed_count": fixed_count,
            "broken_count": broken_count,
            "net_hit_change": net_hit_change,
            "mean_ndcg_delta": mean_ndcg_delta,
            "broken_rate": broken_rate,
            "judge_beats_all_baselines": baseline_quality[
                "judge_beats_all_baselines"
            ],
            "judge_mean_ndcg_delta_vs_best_baseline": baseline_quality[
                "judge_mean_ndcg_delta_vs_best_baseline"
            ],
            "judge_ndcg_delta_ci95": baseline_quality["judge_ndcg_delta_ci95"],
            "statistically_positive_ndcg_delta": baseline_quality[
                "statistically_positive_ndcg_delta"
            ],
        },
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
    }


def render_benchmark_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        "| Run | Papers | Parse | Invalid | Hit rate | NDCG | Usable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("|  |  |  |  |  |  |  |")
    else:
        for row in rows:
            lines.append(
                f"| {row['run']} | {row['paper_count']} | "
                f"{_number(row['parse_success_rate'])} | "
                f"{_number(row['invalid_selection_rate'])} | "
                f"{_number(row['hit_rate'])} | "
                f"{_number(row['mean_ndcg'])} | "
                f"{_yes_no(row['usable_for_benchmark_report'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_model_selection_decision_markdown(decision: dict[str, Any]) -> str:
    metrics = decision.get("metrics", {})
    lines = [
        "# Model Selection Decision",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| recommendation | {decision.get('recommendation', '')} |",
        f"| pass | {_yes_no(decision.get('pass'))} |",
    ]
    for key in [
        "common_targets",
        "fixed_count",
        "broken_count",
        "net_hit_change",
        "mean_ndcg_delta",
        "broken_rate",
        "judge_beats_all_baselines",
        "judge_mean_ndcg_delta_vs_best_baseline",
        "judge_ndcg_delta_ci95",
        "statistically_positive_ndcg_delta",
    ]:
        lines.append(f"| {key} | {_format_summary_value(metrics.get(key))} |")
    lines.extend(
        [
            f"| passed_checks | {_join(decision.get('passed_checks', []))} |",
            f"| failed_checks | {_join(decision.get('failed_checks', []))} |",
            "",
        ]
    )
    return "\n".join(lines)


def render_model_eval_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    target_summary = report.get("target_comparison_summary", {})
    lines = [
        "# Model Eval",
        "",
        f"Base run: `{report.get('base_run', '')}`",
        f"Candidate run: `{report.get('candidate_run', '')}`",
        "",
        "## Decision",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| recommendation | {decision.get('recommendation', '')} |",
        f"| pass | {_yes_no(decision.get('pass'))} |",
        f"| failed_checks | {_join(decision.get('failed_checks', []))} |",
        "",
        "## Target Delta",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "common_targets",
        "fixed_count",
        "broken_count",
        "net_hit_change",
        "mean_ndcg_delta",
    ]:
        lines.append(f"| {key} | {_format_summary_value(target_summary.get(key))} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact in report.get("artifacts", []):
        lines.append(f"- `{artifact}`")
    lines.append("")
    return "\n".join(lines)


def _candidate_baseline_quality(
    candidate_benchmark_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not candidate_benchmark_summary:
        return {
            "available": False,
            "judge_beats_all_baselines": None,
            "judge_mean_ndcg_delta_vs_best_baseline": None,
            "judge_ndcg_delta_ci95": None,
            "statistically_positive_ndcg_delta": None,
        }
    comparison = candidate_benchmark_summary.get("gold_baseline_comparison", {})
    return {
        "available": bool(comparison),
        "judge_beats_all_baselines": bool(
            comparison.get("judge_beats_all_baselines", False)
        ),
        "judge_mean_ndcg_delta_vs_best_baseline": comparison.get(
            "judge_mean_ndcg_delta_vs_best_baseline"
        ),
        "judge_ndcg_delta_ci95": comparison.get("judge_ndcg_delta_ci95"),
        "statistically_positive_ndcg_delta": bool(
            comparison.get("statistically_positive_ndcg_delta", False)
        ),
    }


def _read_candidate_benchmark_summary(
    comparison_dir: Path,
    target_comparison_report: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_run = target_comparison_report.get("candidate_run")
    if not candidate_run:
        return None
    summary_path = comparison_dir.parent / str(candidate_run) / "reports" / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def render_gold_target_comparison_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Gold Target Comparison",
        "",
        f"{report.get('base_run', 'base')} -> {report.get('candidate_run', 'candidate')}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "common_targets",
        "fixed_count",
        "broken_count",
        "net_hit_change",
        "rank_improved_count",
        "rank_worsened_count",
        "hit_quality_changed_count",
        "unchanged_hit_count",
        "unchanged_miss_count",
        "mean_selection_jaccard",
        "mean_reciprocal_rank_delta",
        "mean_ndcg_delta",
    ]:
        lines.append(f"| {key} | {_format_summary_value(summary.get(key))} |")
    lines.extend(
        [
            "",
            "## Fixed Base Misses",
            "",
            "| Target | Gold | Base selected | Candidate selected | Gold candidate positions |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("fixed", [])[:20]:
        lines.append(_target_markdown_row(row))
    if not report.get("fixed"):
        lines.append("|  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Broken Base Hits",
            "",
            "| Target | Gold | Base selected | Candidate selected | Gold candidate positions |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("broken", [])[:20]:
        lines.append(_target_markdown_row(row))
    if not report.get("broken"):
        lines.append("|  |  |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def _row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "reports" / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark = summary.get("benchmark", {})
    judge_quality = summary.get("judge_quality", {})
    gold_quality = summary.get("gold_quality", {})
    decision = summary.get("decision", {})
    return {
        "run": run_dir.name,
        "paper_count": benchmark.get("paper_count", 0),
        "reference_count": benchmark.get("reference_count", 0),
        "judge_results": benchmark.get("judge_results", 0),
        "parse_success_rate": judge_quality.get("parse_success_rate", 0.0),
        "invalid_selection_rate": judge_quality.get("invalid_selection_rate", 0.0),
        "empty_selection_rate": judge_quality.get("empty_selection_rate", 0.0),
        "top_k_compliance_rate": judge_quality.get("top_k_compliance_rate", 0.0),
        "labeled_targets": gold_quality.get("labeled_targets", 0),
        "hit_rate": gold_quality.get("hit_rate", 0.0),
        "selection_precision": gold_quality.get("selection_precision", 0.0),
        "selection_recall": gold_quality.get("selection_recall", 0.0),
        "mean_reciprocal_rank": gold_quality.get("mean_reciprocal_rank", 0.0),
        "mean_ndcg": gold_quality.get("mean_ndcg", 0.0),
        "usable_for_benchmark_report": decision.get("usable_for_benchmark_report", False),
        "needs_prompt_or_model_tuning": decision.get("needs_prompt_or_model_tuning", True),
    }


def _targets_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["target_paper_id"]): row for row in report.get("targets", [])}


def _target_change_row(
    target_id: str,
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_ids: list[str],
) -> dict[str, Any]:
    base_hit = bool(base.get("hit"))
    candidate_hit = bool(candidate.get("hit"))
    reciprocal_rank_delta = _float(candidate.get("reciprocal_rank")) - _float(
        base.get("reciprocal_rank")
    )
    ndcg_delta = _float(candidate.get("ndcg")) - _float(base.get("ndcg"))
    selected_positive_delta = len(candidate.get("selected_positive_reference_ids", [])) - len(
        base.get("selected_positive_reference_ids", [])
    )
    if not base_hit and candidate_hit:
        change_type = "fixed"
    elif base_hit and not candidate_hit:
        change_type = "broken"
    elif base_hit and candidate_hit and reciprocal_rank_delta > 0:
        change_type = "rank_improved"
    elif base_hit and candidate_hit and reciprocal_rank_delta < 0:
        change_type = "rank_worsened"
    elif base_hit and candidate_hit and (
        selected_positive_delta != 0 or ndcg_delta != 0
    ):
        change_type = "hit_quality_changed"
    elif base_hit and candidate_hit:
        change_type = "unchanged_hit"
    else:
        change_type = "unchanged_miss"

    positive_ids = [str(value) for value in base.get("positive_reference_ids", [])]
    return {
        "target_paper_id": target_id,
        "change_type": change_type,
        "positive_reference_ids": positive_ids,
        "base_selected_reference_ids": [
            str(value) for value in base.get("selected_reference_ids", [])
        ],
        "candidate_selected_reference_ids": [
            str(value) for value in candidate.get("selected_reference_ids", [])
        ],
        "base_hit": base_hit,
        "candidate_hit": candidate_hit,
        "base_reciprocal_rank": _float(base.get("reciprocal_rank")),
        "candidate_reciprocal_rank": _float(candidate.get("reciprocal_rank")),
        "reciprocal_rank_delta": reciprocal_rank_delta,
        "base_selected_positive_count": len(base.get("selected_positive_reference_ids", [])),
        "candidate_selected_positive_count": len(
            candidate.get("selected_positive_reference_ids", [])
        ),
        "selected_positive_delta": selected_positive_delta,
        "base_ndcg": _float(base.get("ndcg")),
        "candidate_ndcg": _float(candidate.get("ndcg")),
        "ndcg_delta": ndcg_delta,
        "selection_jaccard": _jaccard(
            base.get("selected_reference_ids", []),
            candidate.get("selected_reference_ids", []),
        ),
        "gold_candidate_positions": _candidate_positions(positive_ids, candidate_ids),
    }


def _candidate_ids_by_target(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    return {
        result.target_paper_id: list(result.candidate_reference_ids)
        for result in load_judge_results(path)
    }


def _write_target_change_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "target_paper_id",
        "change_type",
        "positive_reference_ids",
        "base_selected_reference_ids",
        "candidate_selected_reference_ids",
        "base_hit",
        "candidate_hit",
        "base_reciprocal_rank",
        "candidate_reciprocal_rank",
        "reciprocal_rank_delta",
        "base_selected_positive_count",
        "candidate_selected_positive_count",
        "selected_positive_delta",
        "base_ndcg",
        "candidate_ndcg",
        "ndcg_delta",
        "selection_jaccard",
        "gold_candidate_positions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("all_changes", []):
            writer.writerow(
                sanitize_csv_row(
                    {key: _csv_value(row.get(key)) for key in fieldnames}
                )
            )


def _target_markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row.get('target_paper_id', '')} | "
        f"{_join(row.get('positive_reference_ids', []))} | "
        f"{_join(row.get('base_selected_reference_ids', []))} | "
        f"{_join(row.get('candidate_selected_reference_ids', []))} | "
        f"{_join(row.get('gold_candidate_positions', []))} |"
    )


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return _number(value)
    if value is None:
        return "0"
    return str(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _jaccard(left: Any, right: Any) -> float:
    left_set = {str(value) for value in left}
    right_set = {str(value) for value in right}
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _candidate_positions(positive_ids: list[str], candidate_ids: list[str]) -> list[int]:
    positions = {identifier: index for index, identifier in enumerate(candidate_ids, start=1)}
    return [positions[identifier] for identifier in positive_ids if identifier in positions]


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def _join(values: Any) -> str:
    return ", ".join(str(value) for value in values)
