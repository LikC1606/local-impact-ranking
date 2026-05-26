from __future__ import annotations

from typing import Any

from local_impact_ranking.judge import JudgeResult


def evaluate_judge_results(results: list[JudgeResult], top_k: int | None = None) -> dict[str, float]:
    total_results = len(results)
    if total_results == 0:
        return {
            "judge_results": 0,
            "parse_success_rate": 0.0,
            "invalid_selection_rate": 0.0,
            "average_selected_references": 0.0,
            "empty_selection_rate": 0.0,
            "duplicate_selection_rate": 0.0,
            "top_k_compliance_rate": 0.0,
        }

    parse_successes = sum(1 for result in results if result.parse_error is None)
    error_count = sum(1 for result in results if result.error is not None)
    selected_count = sum(len(result.selected_references) for result in results)
    invalid_count = sum(result.invalid_selection_count for result in results)
    empty_count = sum(1 for result in results if len(result.selected_references) == 0)
    duplicate_count = sum(1 for result in results if _has_duplicate_selection(result))
    top_k_compliant_count = sum(
        1
        for result in results
        if top_k is None or len(result.selected_references) <= top_k
    )
    return {
        "judge_results": total_results,
        "parse_success_rate": parse_successes / total_results,
        "error_rate": error_count / total_results,
        "invalid_selection_rate": invalid_count / selected_count if selected_count else 0.0,
        "average_selected_references": selected_count / total_results,
        "empty_selection_rate": empty_count / total_results,
        "duplicate_selection_rate": duplicate_count / total_results,
        "top_k_compliance_rate": top_k_compliant_count / total_results,
    }


def _has_duplicate_selection(result: JudgeResult) -> bool:
    reference_ids = [selection.reference_id for selection in result.selected_references]
    return len(reference_ids) != len(set(reference_ids))


def render_evaluation_markdown(metrics: dict[str, Any]) -> str:
    rows = [
        ("judge_results", _integer(metrics.get("judge_results"))),
        ("parse_success_rate", _number(metrics.get("parse_success_rate"))),
        ("error_rate", _number(metrics.get("error_rate"))),
        ("invalid_selection_rate", _number(metrics.get("invalid_selection_rate"))),
        ("average_selected_references", _number(metrics.get("average_selected_references"))),
        ("empty_selection_rate", _number(metrics.get("empty_selection_rate"))),
        ("duplicate_selection_rate", _number(metrics.get("duplicate_selection_rate"))),
        ("top_k_compliance_rate", _number(metrics.get("top_k_compliance_rate"))),
    ]
    lines = [
        "# Judge Evaluation Report",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(
        [
            "",
            "This report measures judge output reliability and constrained-selection hygiene; it is not a human correctness label.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _integer(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"
