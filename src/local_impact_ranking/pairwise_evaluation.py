from __future__ import annotations

import csv
import json
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

    return {
        "summary": _summarize(outcomes),
        "by_dimension": _summarize_groups(outcomes, "impact_dimension"),
        "by_field": _summarize_groups(outcomes, "field"),
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
    return report


def render_pairwise_evaluation_markdown(report: dict[str, Any]) -> str:
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
        "coverage_rate": _safe_divide(covered, total),
        "pairwise_accuracy": _safe_divide(correct, covered),
        "non_tie_pairwise_accuracy": _safe_divide(correct, non_tie),
        "tie_rate": _safe_divide(ties, covered),
    }


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
