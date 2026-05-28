from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from local_impact_ranking.sciimpact_adapter import _parse_pair


def evaluate_sciimpact_predictions(
    sft_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    predictions = _load_predictions(predictions_path)
    rows = []
    skipped = []
    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = _parse_pair(json.loads(stripped), line_number)
                expected_side = _side_from_winner(parsed, line_number)
            except ValueError as exc:
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            prediction = predictions.get(line_number)
            predicted_side = _side_from_prediction(prediction) if prediction else None
            if predicted_side is None:
                outcome = "uncovered"
            elif predicted_side == expected_side:
                outcome = "correct"
            else:
                outcome = "wrong"
            rows.append(
                {
                    "source_line": line_number,
                    "expected_side": expected_side.upper(),
                    "predicted_side": predicted_side.upper() if predicted_side else None,
                    "impact_dimension": parsed.get("impact_dimension", "unknown"),
                    "item_type": parsed.get("item_type", "unknown"),
                    "outcome": outcome,
                }
            )

    report = {
        "summary": _summarize(rows),
        "by_dimension": _summarize_groups(rows, "impact_dimension"),
        "by_item_type": _summarize_groups(rows, "item_type"),
        "skipped_source_count": len(skipped),
        "skipped_sources": skipped[:20],
        "examples": rows[:20],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_model_eval.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_eval.md").write_text(
        render_sciimpact_model_eval_markdown(report),
        encoding="utf-8",
    )
    return report


def render_sciimpact_model_eval_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SciImpact Model Evaluation",
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
            "## By Impact Dimension",
            "",
            "| Group | Total | Covered | Accuracy | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_dimension"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## By Item Type",
            "",
            "| Group | Total | Covered | Accuracy | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_item_type"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Accuracy is computed over covered examples only. Coverage reports how many source lines had a parseable A/B prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_predictions(path: str | Path) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            source_line = record.get("source_line", line_number)
            prediction = record.get("prediction") or record.get("answer") or record.get("text")
            if prediction is None:
                continue
            predictions[int(source_line)] = str(prediction)
    return predictions


def _side_from_winner(parsed: dict[str, Any], line_number: int) -> str:
    source_side = parsed["winner"].get("metadata", {}).get("sciimpact_source_side")
    if source_side not in {"a", "b"}:
        raise ValueError(f"SciImpact record on line {line_number} has no winner side")
    return source_side


def _side_from_prediction(prediction: str | None) -> str | None:
    if prediction is None:
        return None
    match = re.search(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+([AB])\b", prediction, re.I)
    if match is None:
        return None
    return match.group(1).lower()


def _summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {group: _summarize(group_rows) for group, group_rows in grouped.items()}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(row["outcome"] != "uncovered" for row in rows)
    correct = sum(row["outcome"] == "correct" for row in rows)
    wrong = sum(row["outcome"] == "wrong" for row in rows)
    uncovered = total - covered
    return {
        "total_examples": total,
        "covered_examples": covered,
        "uncovered_examples": uncovered,
        "correct_examples": correct,
        "wrong_examples": wrong,
        "coverage_rate": _safe_divide(covered, total),
        "accuracy": _safe_divide(correct, covered),
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_row(group: str, summary: dict[str, Any]) -> str:
    return (
        f"| {group} | {summary['total_examples']} | {summary['covered_examples']} | "
        f"{summary['accuracy']:.3f} | {summary['coverage_rate']:.3f} |"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
