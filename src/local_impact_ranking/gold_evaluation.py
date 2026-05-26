from __future__ import annotations

import math
from typing import Any

from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import Paper


def evaluate_gold_reference_selection(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    top_k: int,
) -> dict[str, Any]:
    results_by_id = {result.target_paper_id: result for result in judge_results}
    rows = []
    selected_count = 0
    selected_positive_count = 0
    positive_count = 0
    hit_count = 0
    reciprocal_ranks = []
    ndcgs = []

    for paper in papers:
        positive_ids = _positive_reference_ids(paper)
        if not positive_ids:
            continue
        result = results_by_id.get(paper.paper_id)
        selected_ids = _selected_reference_ids(result, top_k=top_k) if result else []
        selected_positive_ids = [identifier for identifier in selected_ids if identifier in positive_ids]
        selected_count += len(selected_ids)
        selected_positive_count += len(selected_positive_ids)
        positive_count += len(positive_ids)
        hit = bool(selected_positive_ids)
        hit_count += int(hit)
        reciprocal_ranks.append(_reciprocal_rank(selected_ids, positive_ids))
        ndcgs.append(_ndcg(selected_ids, positive_ids, top_k=top_k))
        rows.append(
            {
                "target_paper_id": paper.paper_id,
                "positive_reference_ids": sorted(positive_ids),
                "selected_reference_ids": selected_ids,
                "selected_positive_reference_ids": selected_positive_ids,
                "hit": hit,
                "reciprocal_rank": reciprocal_ranks[-1],
                "ndcg": ndcgs[-1],
            }
        )

    labeled_targets = len(rows)
    return {
        "label_source": "paper.metadata.prescience_positive_reference_ids",
        "labeled_targets": labeled_targets,
        "top_k": top_k,
        "selected_references": selected_count,
        "positive_references": positive_count,
        "selected_positive_references": selected_positive_count,
        "hit_rate": hit_count / labeled_targets if labeled_targets else 0.0,
        "selection_precision": selected_positive_count / selected_count
        if selected_count
        else 0.0,
        "selection_recall": selected_positive_count / positive_count if positive_count else 0.0,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / labeled_targets
        if labeled_targets
        else 0.0,
        "mean_ndcg": sum(ndcgs) / labeled_targets if labeled_targets else 0.0,
        "targets": rows,
        "limitations": [
            "Gold labels are benchmark-specific and only evaluate reference selection for labeled targets.",
            "A strong gold score does not prove global scientific impact.",
        ],
    }


def render_gold_evaluation_markdown(report: dict[str, Any]) -> str:
    rows = [
        ("labeled_targets", _integer(report.get("labeled_targets"))),
        ("hit_rate", _number(report.get("hit_rate"))),
        ("selection_precision", _number(report.get("selection_precision"))),
        ("selection_recall", _number(report.get("selection_recall"))),
        ("mean_reciprocal_rank", _number(report.get("mean_reciprocal_rank"))),
        ("mean_ndcg", _number(report.get("mean_ndcg"))),
    ]
    lines = [
        "# Gold Reference Evaluation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(["", "## Limitations", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _positive_reference_ids(paper: Paper) -> set[str]:
    values = paper.metadata.get("prescience_positive_reference_ids") or paper.metadata.get(
        "gold_reference_ids",
    )
    if not values:
        return set()
    return {str(value) for value in values if value is not None}


def _selected_reference_ids(result: JudgeResult, *, top_k: int) -> list[str]:
    return [
        selection.reference_id
        for selection in sorted(result.selected_references, key=lambda item: item.rank)
        if selection.valid and selection.rank <= top_k
    ]


def _reciprocal_rank(selected_ids: list[str], positive_ids: set[str]) -> float:
    for index, identifier in enumerate(selected_ids, start=1):
        if identifier in positive_ids:
            return 1 / index
    return 0.0


def _ndcg(selected_ids: list[str], positive_ids: set[str], *, top_k: int) -> float:
    dcg = 0.0
    for index, identifier in enumerate(selected_ids[:top_k], start=1):
        if identifier in positive_ids:
            dcg += 1 / math.log2(index + 1)
    ideal_hits = min(len(positive_ids), top_k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


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
