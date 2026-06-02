from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_impact_ranking.impact_fusion import (
    CALIBRATABLE_COMPONENT_SCORE_FIELDS,
    EVIDENCE_FIRST_WEIGHTS,
    IMPACT_INDEX_V3_WEIGHTS,
)


def _default_v3_component_weights() -> dict[str, float]:
    weights = {"model_pairwise": IMPACT_INDEX_V3_WEIGHTS["model_supported"]}
    weights.update(
        {
            signal: IMPACT_INDEX_V3_WEIGHTS["evidence_supported"] * weight
            for signal, weight in EVIDENCE_FIRST_WEIGHTS.items()
        }
    )
    return weights


def calibrate_fusion_weights(
    rankings_path: str | Path,
    labels_path: str | Path,
    *,
    grid_step: float = 0.1,
    top_k: int = 10,
) -> dict[str, Any]:
    if grid_step <= 0.0 or grid_step > 1.0:
        raise ValueError("grid_step must be in the range (0, 1]")
    rows = _load_ranking_rows(rankings_path)
    component_coverage = _component_coverage(rows)
    calibration_quality = _calibration_quality(component_coverage)
    labels = _load_labels(labels_path)
    default_component_weights = _default_v3_component_weights()
    candidates = []
    for weights in _weight_grid(list(CALIBRATABLE_COMPONENT_SCORE_FIELDS), step=grid_step):
        scores = _fused_scores(rows, weights)
        evaluation = _evaluate_scores(scores, labels)
        candidates.append(
            {
                "weights": weights,
                **evaluation,
            }
        )
    default_evaluation = _evaluate_scores(
        _fused_scores(rows, default_component_weights),
        labels,
    )
    candidates.sort(
        key=lambda item: (
            -float(item["pairwise_accuracy"]),
            -float(item["coverage_rate"]),
            float(item["tie_rate"]),
            -float(item["mean_correct_margin"]),
            _weight_distance(item["weights"], default_component_weights),
        )
    )
    best = candidates[0] if candidates else {"weights": dict(default_component_weights)}
    best_scores = _fused_scores(rows, best["weights"])
    best_diagnostics = _diagnose_scores(best_scores, labels)
    best_layer_weights, best_evidence_weights = _v3_layer_weights_from_components(
        best["weights"]
    )
    return {
        "summary": {
            "method": "grid_search_pairwise_accuracy",
            "weight_profile": "impact_index_v3_final_components",
            "grid_step": grid_step,
            "candidate_count": len(candidates),
            "ranking_row_count": len(rows),
            "label_count": len(labels),
            "component_coverage": component_coverage,
            **calibration_quality,
            "covered_pairs": best.get("covered_pairs", 0),
            "best_pairwise_accuracy": best.get("pairwise_accuracy", 0.0),
            "default_pairwise_accuracy": default_evaluation["pairwise_accuracy"],
            "accuracy_delta_vs_default": round(
                best.get("pairwise_accuracy", 0.0)
                - default_evaluation["pairwise_accuracy"],
                6,
            ),
        },
        "best_component_weights": best["weights"],
        "default_component_weights": default_component_weights,
        "best_impact_index_v3_weights": best_layer_weights,
        "best_evidence_first_weights": best_evidence_weights,
        "best_weights": best_evidence_weights,
        "default_weights": dict(EVIDENCE_FIRST_WEIGHTS),
        "default_evaluation": default_evaluation,
        "best_evaluation": {
            key: value for key, value in best.items() if key != "weights"
        },
        "best_diagnostics": best_diagnostics,
        "top_candidates": candidates[:top_k],
        "metric_contract": {
            "task": "Calibrate ImpactIndex v3 final component weights against pairwise impact labels.",
            "dataset": str(rankings_path),
            "labels": str(labels_path),
            "metric": "pairwise_accuracy",
            "metric_direction": "higher_is_better",
            "known_limitations": [
                "Calibration quality depends on label coverage and benchmark-domain match.",
                "Grid search optimizes ranking agreement, not a universal scientific impact truth.",
                calibration_quality["known_limitation"],
            ],
        },
    }


def write_fusion_calibration(
    rankings_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    grid_step: float = 0.1,
) -> dict[str, Any]:
    report = calibrate_fusion_weights(
        rankings_path,
        labels_path,
        grid_step=grid_step,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "fusion_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "fusion_calibration.md").write_text(
        render_fusion_calibration_markdown(report),
        encoding="utf-8",
    )
    return report


def render_fusion_calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fusion Weight Calibration",
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
            "",
            "## Best Weights",
            "",
            "| Signal | Weight |",
            "| --- | ---: |",
        ]
    )
    for signal, weight in report["best_weights"].items():
        lines.append(f"| {signal} | {weight:.6f} |")
    if "best_component_weights" in report:
        lines.extend(
            [
                "",
                "## Best Final Component Weights",
                "",
                "| Signal | Weight |",
                "| --- | ---: |",
            ]
        )
        for signal, weight in report["best_component_weights"].items():
            lines.append(f"| {signal} | {weight:.6f} |")
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Rank | Pairwise Accuracy | Coverage | Tie Rate | Weights |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, candidate in enumerate(report.get("top_candidates", []), start=1):
        lines.append(
            f"| {rank} | {candidate['pairwise_accuracy']:.6f} | "
            f"{candidate['coverage_rate']:.6f} | {candidate['tie_rate']:.6f} | "
            f"{json.dumps(candidate['weights'], sort_keys=True)} |"
        )
    diagnostics = report.get("best_diagnostics", {})
    lines.extend(
        [
            "",
            "## By Impact Dimension",
            "",
            "| Dimension | Total | Covered | Pairwise Accuracy | Wrong | Tie |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dimension, summary in sorted(diagnostics.get("by_dimension", {}).items()):
        lines.append(
            f"| {dimension} | {summary['total_pairs']} | {summary['covered_pairs']} | "
            f"{summary['pairwise_accuracy']:.6f} | {summary['wrong_pairs']} | "
            f"{summary['tie_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Wrong Examples",
            "",
            "| Pair | Dimension | Winner Score | Loser Score | Margin |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for example in diagnostics.get("wrong_examples", [])[:10]:
        lines.append(
            f"| {example.get('pair_id', '')} | {example.get('impact_dimension', '')} | "
            f"{_format_value(example.get('winner_score'))} | "
            f"{_format_value(example.get('loser_score'))} | "
            f"{_format_value(example.get('score_margin'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this file as a calibration profile for datasets with similar label semantics. "
            "It is a benchmark fit, not proof of global scientific impact.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_ranking_rows(path: str | Path) -> dict[str, dict[str, float]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required_fields = set(CALIBRATABLE_COMPONENT_SCORE_FIELDS.values())
        missing_fields = sorted(required_fields - fieldnames)
        if missing_fields:
            raise ValueError(
                "Ranking file is missing required component score columns: "
                + ", ".join(missing_fields)
            )
        rows: dict[str, dict[str, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            paper_id = row.get("paper_id") or row.get("id")
            if not paper_id:
                raise ValueError(f"Ranking row {row_number} is missing paper_id/id")
            rows[paper_id] = {
                component: _float_or_zero(row.get(field))
                for component, field in CALIBRATABLE_COMPONENT_SCORE_FIELDS.items()
            }
    if not rows:
        raise ValueError("Ranking file contains no ranking rows")
    return rows


def _component_coverage(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    row_count = len(rows)
    return {
        component: {
            "non_zero_count": sum(
                1 for values in rows.values() if values.get(component, 0.0) != 0.0
            ),
            "row_count": row_count,
            "non_zero_rate": _safe_divide(
                sum(1 for values in rows.values() if values.get(component, 0.0) != 0.0),
                row_count,
            ),
        }
        for component in CALIBRATABLE_COMPONENT_SCORE_FIELDS
    }


def _calibration_quality(
    component_coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    effective_signals = [
        signal
        for signal, coverage in component_coverage.items()
        if int(coverage.get("non_zero_count", 0)) > 0
    ]
    if len(effective_signals) >= 2:
        return {
            "effective_signal_count": len(effective_signals),
            "effective_signals": effective_signals,
            "calibration_status": "calibrated",
            "calibration_status_reason": (
                "At least two component signals have non-zero scores, so fusion "
                "weight calibration can compare signal trade-offs."
            ),
            "known_limitation": (
                "Calibration only tests signals with non-zero component coverage."
            ),
        }
    if len(effective_signals) == 1:
        return {
            "effective_signal_count": 1,
            "effective_signals": effective_signals,
            "calibration_status": "single_signal_diagnostic",
            "calibration_status_reason": (
                "At least two component signals need non-zero scores for real "
                "fusion calibration; this run only diagnoses one active signal."
            ),
            "known_limitation": (
                "This calibration has a single active signal and cannot validate "
                "multi-signal fusion weights."
            ),
        }
    return {
        "effective_signal_count": 0,
        "effective_signals": [],
        "calibration_status": "no_active_signal",
        "calibration_status_reason": (
            "No component signal has non-zero scores, so fusion calibration is not meaningful."
        ),
        "known_limitation": (
            "This calibration has no active component signal and should not be used."
        ),
    }


def _load_labels(path: str | Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
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


def _v3_layer_weights_from_components(
    component_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    model_weight = float(component_weights.get("model_pairwise", 0.0))
    evidence_weight = sum(
        float(component_weights.get(signal, 0.0))
        for signal in EVIDENCE_FIRST_WEIGHTS
    )
    layer_weights = {
        "evidence_supported": round(evidence_weight, 6),
        "model_supported": round(model_weight, 6),
    }
    if evidence_weight > 0.0:
        evidence_weights = {
            signal: round(float(component_weights.get(signal, 0.0)) / evidence_weight, 6)
            for signal in EVIDENCE_FIRST_WEIGHTS
        }
    else:
        evidence_weights = {signal: 0.0 for signal in EVIDENCE_FIRST_WEIGHTS}
    return layer_weights, evidence_weights


def _weight_grid(signals: list[str], *, step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    if not math_is_close(units * step, 1.0):
        raise ValueError("grid_step must evenly divide 1.0")
    rows = []
    for allocation in _integer_simplex(len(signals), units):
        rows.append(
            {
                signal: round(allocation[index] / units, 6)
                for index, signal in enumerate(signals)
            }
        )
    return rows


def _integer_simplex(length: int, total: int) -> list[list[int]]:
    if length == 1:
        return [[total]]
    rows: list[list[int]] = []
    for value in range(total + 1):
        for tail in _integer_simplex(length - 1, total - value):
            rows.append([value, *tail])
    return rows


def _fused_scores(
    rows: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    return {
        paper_id: sum(
            weights.get(component, 0.0) * components.get(component, 0.0)
            for component in CALIBRATABLE_COMPONENT_SCORE_FIELDS
        )
        for paper_id, components in rows.items()
    }


def _evaluate_scores(
    scores: dict[str, float],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(labels)
    covered = correct = wrong = ties = 0
    correct_margin_sum = 0.0
    for label in labels:
        winner_score = scores.get(str(label["winner_id"]))
        loser_score = scores.get(str(label["loser_id"]))
        if winner_score is None or loser_score is None:
            continue
        covered += 1
        if winner_score > loser_score:
            correct += 1
            correct_margin_sum += winner_score - loser_score
        elif winner_score < loser_score:
            wrong += 1
        else:
            ties += 1
    non_tie = correct + wrong
    return {
        "total_pairs": total,
        "covered_pairs": covered,
        "correct_pairs": correct,
        "wrong_pairs": wrong,
        "tie_pairs": ties,
        "coverage_rate": _safe_divide(covered, total),
        "pairwise_accuracy": _safe_divide(correct, covered),
        "non_tie_pairwise_accuracy": _safe_divide(correct, non_tie),
        "tie_rate": _safe_divide(ties, covered),
        "mean_correct_margin": correct_margin_sum / correct if correct else 0.0,
    }


def _diagnose_scores(
    scores: dict[str, float],
    labels: list[dict[str, Any]],
    *,
    max_wrong_examples: int = 20,
) -> dict[str, Any]:
    outcomes = [_label_outcome(scores, label) for label in labels]
    by_dimension = _summarize_groups(outcomes, "impact_dimension")
    by_field = _summarize_groups(outcomes, "field")
    wrong_examples = [
        outcome for outcome in outcomes if outcome["outcome"] == "wrong"
    ][:max_wrong_examples]
    weakest_dimensions = [
        {"dimension": dimension, **summary}
        for dimension, summary in by_dimension.items()
        if summary["covered_pairs"] > 0
    ]
    weakest_dimensions.sort(
        key=lambda row: (
            float(row["pairwise_accuracy"]),
            -int(row["covered_pairs"]),
            str(row["dimension"]),
        )
    )
    return {
        "summary": _summarize_outcomes(outcomes),
        "by_dimension": by_dimension,
        "by_field": by_field,
        "weakest_dimensions": weakest_dimensions[:5],
        "wrong_examples": wrong_examples,
    }


def _label_outcome(
    scores: dict[str, float],
    label: dict[str, Any],
) -> dict[str, Any]:
    winner_id = str(label["winner_id"])
    loser_id = str(label["loser_id"])
    winner_score = scores.get(winner_id)
    loser_score = scores.get(loser_id)
    if winner_score is None or loser_score is None:
        outcome = "uncovered"
        margin = None
    else:
        margin = winner_score - loser_score
        if winner_score > loser_score:
            outcome = "correct"
        elif winner_score < loser_score:
            outcome = "wrong"
        else:
            outcome = "tie"
    return {
        "pair_id": label.get("pair_id", ""),
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_score": winner_score,
        "loser_score": loser_score,
        "score_margin": round(margin, 6) if margin is not None else None,
        "impact_dimension": label.get("impact_dimension", "unknown"),
        "field": label.get("field", "unknown"),
        "outcome": outcome,
    }


def _summarize_groups(
    outcomes: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        groups.setdefault(str(outcome.get(key) or "unknown"), []).append(outcome)
    return {
        group: _summarize_outcomes(group_outcomes)
        for group, group_outcomes in groups.items()
    }


def _summarize_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(outcomes)
    covered = sum(outcome["outcome"] != "uncovered" for outcome in outcomes)
    correct = sum(outcome["outcome"] == "correct" for outcome in outcomes)
    wrong = sum(outcome["outcome"] == "wrong" for outcome in outcomes)
    ties = sum(outcome["outcome"] == "tie" for outcome in outcomes)
    non_tie = correct + wrong
    return {
        "total_pairs": total,
        "covered_pairs": covered,
        "uncovered_pairs": total - covered,
        "correct_pairs": correct,
        "wrong_pairs": wrong,
        "tie_pairs": ties,
        "coverage_rate": _safe_divide(covered, total),
        "pairwise_accuracy": _safe_divide(correct, covered),
        "non_tie_pairwise_accuracy": _safe_divide(correct, non_tie),
        "tie_rate": _safe_divide(ties, covered),
    }


def _weight_distance(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    return sum(
        abs(first.get(signal, 0.0) - second.get(signal, 0.0))
        for signal in CALIBRATABLE_COMPONENT_SCORE_FIELDS
    )


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def math_is_close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9
