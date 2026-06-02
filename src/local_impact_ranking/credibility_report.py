from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "This system reports dataset-local paper influence. It does not prove global "
    "scientific impact without an external, calibrated citation/adoption graph."
)


def credibility_report(
    *,
    baseline_runs: list[str | Path],
    candidate_runs: list[str | Path],
    baseline_model: str,
    candidate_model: str,
    seed_labels: list[str] | None = None,
    min_selected_delta: float = 0.0,
    min_seed_count: int = 1,
    allowed_seed_selected_regression: float = 0.0,
    allowed_slice_regression: float | None = None,
) -> dict[str, Any]:
    if not baseline_runs:
        raise ValueError("at least one baseline run is required")
    if len(baseline_runs) != len(candidate_runs):
        raise ValueError("baseline_runs and candidate_runs must have the same length")
    if seed_labels is not None and len(seed_labels) != len(baseline_runs):
        raise ValueError("seed_labels must match the number of run pairs")

    baseline_rows = [
        _load_run_summary(run, model=baseline_model) for run in baseline_runs
    ]
    candidate_rows = [
        _load_run_summary(run, model=candidate_model) for run in candidate_runs
    ]
    seed_rows = []
    for index, (baseline, candidate) in enumerate(
        zip(baseline_rows, candidate_rows, strict=True),
        start=1,
    ):
        seed = (
            seed_labels[index - 1]
            if seed_labels is not None
            else candidate["seed"] or baseline["seed"] or f"run{index}"
        )
        seed_rows.append(_seed_comparison_row(str(seed), baseline, candidate))

    aggregate = {
        "baseline": _aggregate_rows(baseline_rows),
        "candidate": _aggregate_rows(candidate_rows),
    }
    aggregate_delta = _aggregate_delta(aggregate["baseline"], aggregate["candidate"])
    decision = _decision(
        seed_rows,
        aggregate_delta,
        min_selected_delta,
        min_seed_count=min_seed_count,
        allowed_seed_selected_regression=allowed_seed_selected_regression,
        allowed_slice_regression=allowed_slice_regression,
    )
    report = {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "seed_count": len(seed_rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "decision": decision,
        "aggregate": aggregate,
        "aggregate_delta": aggregate_delta,
        "seed_comparisons": seed_rows,
        "candidate_strengths": _top_group_deltas(seed_rows, positive=True),
        "candidate_regressions": _top_group_deltas(seed_rows, positive=False),
        "evidence_files": _evidence_files(baseline_rows, candidate_rows),
        "limitations": [
            "SciImpact is an external benchmark for pairwise scientific-impact preference, not a perfect ground truth for all paper influence.",
            "Reported gains are benchmark evidence; deploy reports should still expose score uncertainty, signal coverage, and dataset-local claim scope.",
            "Field/year citation normalization, graph signals, and citation-context quality remain necessary when the user dataset contains those signals.",
        ],
    }
    return report


def write_credibility_report(
    *,
    baseline_runs: list[str | Path],
    candidate_runs: list[str | Path],
    output_dir: str | Path,
    baseline_model: str,
    candidate_model: str,
    seed_labels: list[str] | None = None,
    min_selected_delta: float = 0.0,
    min_seed_count: int = 1,
    allowed_seed_selected_regression: float = 0.0,
    allowed_slice_regression: float | None = None,
) -> dict[str, Any]:
    report = credibility_report(
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        seed_labels=seed_labels,
        min_selected_delta=min_selected_delta,
        min_seed_count=min_seed_count,
        allowed_seed_selected_regression=allowed_seed_selected_regression,
        allowed_slice_regression=allowed_slice_regression,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "credibility_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "credibility_report.md").write_text(
        render_credibility_report_markdown(report),
        encoding="utf-8",
    )
    return report


def render_credibility_report_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    delta = report.get("aggregate_delta", {})
    aggregate = report.get("aggregate", {})
    baseline = aggregate.get("baseline", {})
    candidate = aggregate.get("candidate", {})
    lines = [
        "# Impact System Credibility Report",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| baseline_model | {_text(report.get('baseline_model'))} |",
        f"| candidate_model | {_text(report.get('candidate_model'))} |",
        f"| recommendation | {_text(decision.get('recommendation'))} |",
        f"| pass | {_yes_no(decision.get('pass'))} |",
        f"| seed_count | {report.get('seed_count', 0)} |",
        f"| passed_checks | {_join(decision.get('passed_checks', []))} |",
        f"| failed_checks | {_join(decision.get('failed_checks', []))} |",
        "",
        "## Quality Gate",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| min_seed_count | {decision.get('thresholds', {}).get('min_seed_count', 1)} |",
        f"| min_selected_delta | {_number(decision.get('thresholds', {}).get('min_selected_delta'))} |",
        f"| allowed_seed_selected_regression | {_number(decision.get('thresholds', {}).get('allowed_seed_selected_regression'))} |",
        f"| allowed_slice_regression | {_number(decision.get('thresholds', {}).get('allowed_slice_regression'))} |",
        f"| min_seed_selected_accuracy_delta | {_number(delta.get('min_seed_selected_accuracy_delta'))} |",
        f"| worst_group_delta | {_number(delta.get('worst_group_delta'))} |",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        _metric_row(
            "mean_direct_accuracy",
            baseline.get("mean_direct_accuracy"),
            candidate.get("mean_direct_accuracy"),
            delta.get("mean_direct_accuracy_delta"),
        ),
        _metric_row(
            "mean_selected_accuracy",
            baseline.get("mean_selected_accuracy"),
            candidate.get("mean_selected_accuracy"),
            delta.get("mean_selected_accuracy_delta"),
        ),
        _metric_row(
            "mean_consistency_rate",
            baseline.get("mean_consistency_rate"),
            candidate.get("mean_consistency_rate"),
            delta.get("mean_consistency_rate_delta"),
        ),
        "",
        "## Aggregate Deltas",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| mean_direct_accuracy_delta | {_number(delta.get('mean_direct_accuracy_delta'))} |",
        f"| mean_selected_accuracy_delta | {_number(delta.get('mean_selected_accuracy_delta'))} |",
        f"| mean_consistency_rate_delta | {_number(delta.get('mean_consistency_rate_delta'))} |",
        "",
        "## Per-Seed Evidence",
        "",
        "| Seed | Baseline direct | Candidate direct | Direct delta | Baseline selected | Candidate selected | Selected delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("seed_comparisons", []):
        lines.append(
            "| "
            f"{_text(row.get('seed'))} | "
            f"{_number(row.get('baseline_direct_accuracy'))} | "
            f"{_number(row.get('candidate_direct_accuracy'))} | "
            f"{_number(row.get('direct_accuracy_delta'))} | "
            f"{_number(row.get('baseline_selected_accuracy'))} | "
            f"{_number(row.get('candidate_selected_accuracy'))} | "
            f"{_number(row.get('selected_accuracy_delta'))} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Regressions",
            "",
            "| Group | Metric | Delta |",
            "| --- | --- | ---: |",
        ]
    )
    regressions = report.get("candidate_regressions", [])
    if regressions:
        for row in regressions:
            lines.append(
                f"| {_text(row.get('group'))} | {_text(row.get('metric'))} | {_number(row.get('delta'))} |"
            )
    else:
        lines.append("|  |  |  |")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            report.get("claim_boundary", CLAIM_BOUNDARY),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(["", "## Evidence Files", ""])
    for path in report.get("evidence_files", []):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def _load_run_summary(run_dir: str | Path, *, model: str) -> dict[str, Any]:
    root = Path(run_dir)
    eval_path = _first_existing(
        [
            root / "eval" / "sciimpact_model_eval.json",
            root / "selected_eval" / "sciimpact_model_eval.json",
            root / "selected_eval_margin005" / "sciimpact_model_eval.json",
        ]
    )
    flip_path = _first_existing(
        [
            root / "flip_eval_margin005" / "sciimpact_flip_consistency.json",
            root / "flip_consistency_margin005" / "sciimpact_flip_consistency.json",
            root / "flip" / "sciimpact_flip_consistency.json",
            root / "flip_consistency" / "sciimpact_flip_consistency.json",
        ]
    )
    eval_report = _read_json(eval_path) if eval_path else {}
    flip_report = _read_json(flip_path) if flip_path else {}
    summary = eval_report.get("summary", {})
    flip_summary = flip_report.get("summary", {})
    direct_accuracy = flip_summary.get("original_accuracy")
    if direct_accuracy is None:
        direct_accuracy = summary.get("accuracy")
    return {
        "model": model,
        "run_dir": str(root),
        "seed": _infer_seed(root),
        "direct_accuracy": _float(direct_accuracy),
        "coverage_rate": _float(summary.get("coverage_rate")),
        "selected_accuracy": _float(flip_summary.get("selected_accuracy")),
        "flipped_mapped_accuracy": _float(flip_summary.get("flipped_mapped_accuracy")),
        "consistency_rate": _float(flip_summary.get("consistency_rate")),
        "confidence_margin": _float(flip_summary.get("confidence_margin")),
        "by_dimension": _group_selected_accuracy(flip_report.get("by_dimension", {})),
        "by_item_type": _group_selected_accuracy(flip_report.get("by_item_type", {})),
        "evidence_files": [str(path) for path in [eval_path, flip_path] if path],
    }


def _seed_comparison_row(
    seed: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_run_dir": baseline["run_dir"],
        "candidate_run_dir": candidate["run_dir"],
        "baseline_direct_accuracy": baseline["direct_accuracy"],
        "candidate_direct_accuracy": candidate["direct_accuracy"],
        "direct_accuracy_delta": candidate["direct_accuracy"] - baseline["direct_accuracy"],
        "baseline_selected_accuracy": baseline["selected_accuracy"],
        "candidate_selected_accuracy": candidate["selected_accuracy"],
        "selected_accuracy_delta": candidate["selected_accuracy"] - baseline["selected_accuracy"],
        "baseline_consistency_rate": baseline["consistency_rate"],
        "candidate_consistency_rate": candidate["consistency_rate"],
        "consistency_rate_delta": candidate["consistency_rate"] - baseline["consistency_rate"],
        "by_dimension_delta": _group_delta(
            baseline.get("by_dimension", {}),
            candidate.get("by_dimension", {}),
        ),
        "by_item_type_delta": _group_delta(
            baseline.get("by_item_type", {}),
            candidate.get("by_item_type", {}),
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_direct_accuracy": _mean(row["direct_accuracy"] for row in rows),
        "mean_selected_accuracy": _mean(row["selected_accuracy"] for row in rows),
        "mean_consistency_rate": _mean(row["consistency_rate"] for row in rows),
        "min_selected_accuracy": min(row["selected_accuracy"] for row in rows),
        "max_selected_accuracy": max(row["selected_accuracy"] for row in rows),
    }


def _aggregate_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mean_direct_accuracy_delta": _rounded(
            candidate["mean_direct_accuracy"] - baseline["mean_direct_accuracy"]
        ),
        "mean_selected_accuracy_delta": _rounded(
            candidate["mean_selected_accuracy"] - baseline["mean_selected_accuracy"]
        ),
        "mean_consistency_rate_delta": _rounded(
            candidate["mean_consistency_rate"] - baseline["mean_consistency_rate"]
        ),
    }


def _decision(
    seed_rows: list[dict[str, Any]],
    aggregate_delta: dict[str, Any],
    min_selected_delta: float,
    *,
    min_seed_count: int,
    allowed_seed_selected_regression: float,
    allowed_slice_regression: float | None,
) -> dict[str, Any]:
    min_seed_delta = min(row["selected_accuracy_delta"] for row in seed_rows)
    worst_group_delta = _worst_group_delta(seed_rows)
    aggregate_delta["min_seed_selected_accuracy_delta"] = _rounded(min_seed_delta)
    aggregate_delta["worst_group_delta"] = _rounded(worst_group_delta)
    checks = {
        "minimum_seed_count_met": len(seed_rows) >= min_seed_count,
        "candidate_mean_selected_accuracy_improves": aggregate_delta[
            "mean_selected_accuracy_delta"
        ]
        > min_selected_delta,
        "seed_selected_regression_within_tolerance": min_seed_delta
        >= -allowed_seed_selected_regression,
        "candidate_direct_accuracy_not_worse_on_average": aggregate_delta[
            "mean_direct_accuracy_delta"
        ]
        >= 0.0,
    }
    if allowed_slice_regression is not None:
        checks["slice_regression_within_tolerance"] = (
            worst_group_delta >= -allowed_slice_regression
        )
    if allowed_seed_selected_regression == 0.0:
        checks["candidate_selected_accuracy_non_decreasing"] = checks[
            "seed_selected_regression_within_tolerance"
        ]
    passed_checks = [name for name, passed in checks.items() if passed]
    failed_checks = [name for name, passed in checks.items() if not passed]
    passed = not failed_checks
    return {
        "recommendation": "promote_candidate_for_bidirectional_scoring"
        if passed
        else "keep_baseline_or_collect_more_evidence",
        "pass": passed,
        "thresholds": {
            "min_selected_delta": min_selected_delta,
            "min_seed_count": min_seed_count,
            "allowed_seed_selected_regression": allowed_seed_selected_regression,
            "allowed_slice_regression": allowed_slice_regression,
        },
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
    }


def _worst_group_delta(seed_rows: list[dict[str, Any]]) -> float:
    deltas: list[float] = []
    for row in seed_rows:
        for metric_name in ["by_dimension_delta", "by_item_type_delta"]:
            deltas.extend(float(delta) for delta in row.get(metric_name, {}).values())
    return min(deltas) if deltas else 0.0


def _group_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    groups = sorted(set(baseline) | set(candidate))
    return {
        group: candidate.get(group, 0.0) - baseline.get(group, 0.0)
        for group in groups
    }


def _group_selected_accuracy(groups: dict[str, Any]) -> dict[str, float]:
    selected: dict[str, float] = {}
    for group, summary in groups.items():
        selected[str(group)] = _float(summary.get("selected_accuracy"))
    return selected


def _top_group_deltas(
    seed_rows: list[dict[str, Any]],
    *,
    positive: bool,
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], list[float]] = {}
    for row in seed_rows:
        for metric_name in ["by_dimension_delta", "by_item_type_delta"]:
            for group, delta in row.get(metric_name, {}).items():
                totals.setdefault((metric_name, group), []).append(float(delta))
    rows = [
        {
            "metric": metric_name.replace("_delta", ""),
            "group": group,
            "delta": _mean(values),
        }
        for (metric_name, group), values in totals.items()
        if (sum(values) / len(values) > 0.0 if positive else sum(values) / len(values) < 0.0)
    ]
    rows.sort(key=lambda row: row["delta"], reverse=positive)
    return rows[:10]


def _evidence_files(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[str]:
    paths: list[str] = []
    for row in [*baseline_rows, *candidate_rows]:
        paths.extend(row.get("evidence_files", []))
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _infer_seed(path: Path) -> str | None:
    for part in reversed(path.parts):
        lowered = part.lower()
        if "seed" in lowered:
            return lowered
    return None


def _metric_row(label: str, baseline: Any, candidate: Any, delta: Any) -> str:
    return (
        f"| {label} | {_number(baseline)} | {_number(candidate)} | {_number(delta)} |"
    )


def _mean(values: Any) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _join(values: Any) -> str:
    return ", ".join(str(value) for value in values)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
