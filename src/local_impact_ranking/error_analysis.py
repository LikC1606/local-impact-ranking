from __future__ import annotations

from typing import Any


def gold_error_analysis_report(
    gold_report: dict[str, Any],
    *,
    candidate_ids_by_target: dict[str, list[Any]] | None = None,
    max_failures: int = 50,
) -> dict[str, Any]:
    targets = gold_report.get("targets", [])
    candidate_ids_by_target = _normalize_candidate_ids_by_target(candidate_ids_by_target)
    labeled_targets = len(targets)
    first_positive_ranks = [_first_positive_rank(row) for row in targets]
    failures = []
    miss_decomposition = {
        "miss_count": 0,
        "miss_gold_absent_from_candidates": 0,
        "miss_gold_present_but_not_selected": 0,
        "miss_candidate_status_unknown": 0,
    }
    for row, first_rank in zip(targets, first_positive_ranks):
        if first_rank is not None:
            continue
        miss_decomposition["miss_count"] += 1
        miss_reason = _miss_reason(row, candidate_ids_by_target)
        if miss_reason == "gold_absent_from_candidates":
            miss_decomposition["miss_gold_absent_from_candidates"] += 1
        elif miss_reason == "gold_present_but_not_selected":
            miss_decomposition["miss_gold_present_but_not_selected"] += 1
        else:
            miss_decomposition["miss_candidate_status_unknown"] += 1
        if len(failures) >= max_failures:
            continue
        target_id = str(row.get("target_paper_id"))
        failure = {
            "target_paper_id": row.get("target_paper_id"),
            "positive_reference_ids": row.get("positive_reference_ids", []),
            "selected_reference_ids": row.get("selected_reference_ids", []),
            "miss_reason": miss_reason,
        }
        if target_id in candidate_ids_by_target:
            failure["candidate_reference_ids"] = candidate_ids_by_target[target_id]
            failure["gold_candidate_positions"] = _gold_candidate_positions(
                row,
                candidate_ids_by_target[target_id],
            )
        failures.append(failure)
    return {
        "labeled_targets": labeled_targets,
        "hit_at": {
            "hit@1": _hit_at(first_positive_ranks, 1),
            "hit@3": _hit_at(first_positive_ranks, 3),
            "hit@5": _hit_at(first_positive_ranks, 5),
        },
        "first_positive_rank_buckets": {
            "rank_1": sum(1 for rank in first_positive_ranks if rank == 1),
            "rank_2_to_3": sum(1 for rank in first_positive_ranks if rank in {2, 3}),
            "rank_4_to_5": sum(1 for rank in first_positive_ranks if rank in {4, 5}),
            "miss": sum(1 for rank in first_positive_ranks if rank is None),
        },
        "candidate_coverage": _candidate_coverage(targets, candidate_ids_by_target),
        "miss_decomposition": miss_decomposition,
        "candidate_position_buckets": _candidate_position_buckets(
            targets,
            candidate_ids_by_target,
        ),
        "miss_candidate_position_buckets": _candidate_position_buckets(
            [
                row
                for row, first_rank in zip(targets, first_positive_ranks)
                if first_rank is None
            ],
            candidate_ids_by_target,
        ),
        "failures": failures,
        "limitations": [
            "Failure rows show benchmark misses, not necessarily true scientific non-impact.",
            "Gold labels evaluate candidate reference selection only.",
            "Candidate coverage is computed only for targets present in the judge audit.",
        ],
    }


def render_gold_error_analysis_markdown(report: dict[str, Any]) -> str:
    hit_at = report.get("hit_at", {})
    buckets = report.get("first_positive_rank_buckets", {})
    lines = [
        "# Gold Error Analysis",
        "",
        "## Hit At K",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| hit@1 | {_number(hit_at.get('hit@1'))} |",
        f"| hit@3 | {_number(hit_at.get('hit@3'))} |",
        f"| hit@5 | {_number(hit_at.get('hit@5'))} |",
        "",
        "## First Positive Rank",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
        f"| rank_1 | {buckets.get('rank_1', 0)} |",
        f"| rank_2_to_3 | {buckets.get('rank_2_to_3', 0)} |",
        f"| rank_4_to_5 | {buckets.get('rank_4_to_5', 0)} |",
        f"| miss | {buckets.get('miss', 0)} |",
        "",
        "## Candidate Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    coverage = report.get("candidate_coverage", {})
    lines.extend(
        [
            f"| available | {_text(coverage.get('available', False))} |",
            f"| targets_with_candidate_data | {coverage.get('targets_with_candidate_data', 0)} |",
            f"| gold_references | {coverage.get('gold_references', 0)} |",
            f"| gold_in_candidates | {coverage.get('gold_in_candidates', 0)} |",
            f"| gold_candidate_recall | {_number(coverage.get('gold_candidate_recall'))} |",
            "",
            "## Miss Decomposition",
            "",
            "| Cause | Count |",
            "| --- | ---: |",
        ]
    )
    decomposition = report.get("miss_decomposition", {})
    lines.extend(
        [
            f"| miss_count | {decomposition.get('miss_count', 0)} |",
            f"| miss_gold_absent_from_candidates | {decomposition.get('miss_gold_absent_from_candidates', 0)} |",
            f"| miss_gold_present_but_not_selected | {decomposition.get('miss_gold_present_but_not_selected', 0)} |",
            f"| miss_candidate_status_unknown | {decomposition.get('miss_candidate_status_unknown', 0)} |",
            "",
            "## Gold Candidate Positions",
            "",
            "| Bucket | All gold refs | Miss gold refs |",
            "| --- | ---: | ---: |",
        ]
    )
    position_buckets = report.get("candidate_position_buckets", {})
    miss_position_buckets = report.get("miss_candidate_position_buckets", {})
    for bucket in _POSITION_BUCKET_KEYS:
        lines.append(
            f"| {bucket} | {position_buckets.get(bucket, 0)} | "
            f"{miss_position_buckets.get(bucket, 0)} |"
        )
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(
        [
            "## Failure Examples",
            "",
            "| Target | Gold positives | Selected | Candidates | Miss reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    failures = report.get("failures", [])
    if not failures:
        lines.append("|  |  |  |  |  |")
    for row in failures:
        lines.append(
            f"| {_text(row.get('target_paper_id'))} | "
            f"{_join(row.get('positive_reference_ids', []))} | "
            f"{_join(row.get('selected_reference_ids', []))} | "
            f"{_join(row.get('candidate_reference_ids', []))} | "
            f"{_text(row.get('miss_reason'))} |"
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {_text(limitation)}")
    lines.append("")
    return "\n".join(lines)


def _first_positive_rank(row: dict[str, Any]) -> int | None:
    positives = {str(value) for value in row.get("positive_reference_ids", [])}
    for index, reference_id in enumerate(row.get("selected_reference_ids", []), start=1):
        if str(reference_id) in positives:
            return index
    return None


def _normalize_candidate_ids_by_target(
    candidate_ids_by_target: dict[str, list[Any]] | None,
) -> dict[str, list[str]]:
    if not candidate_ids_by_target:
        return {}
    return {
        str(target_id): [str(value) for value in values if value is not None]
        for target_id, values in candidate_ids_by_target.items()
    }


def _candidate_coverage(
    targets: list[dict[str, Any]],
    candidate_ids_by_target: dict[str, list[str]],
) -> dict[str, Any]:
    gold_references = 0
    gold_in_candidates = 0
    targets_with_candidate_data = 0
    for row in targets:
        target_id = str(row.get("target_paper_id"))
        if target_id not in candidate_ids_by_target:
            continue
        targets_with_candidate_data += 1
        positives = _positive_reference_ids(row)
        candidates = set(candidate_ids_by_target[target_id])
        gold_references += len(positives)
        gold_in_candidates += len(positives & candidates)
    return {
        "available": bool(candidate_ids_by_target),
        "targets_with_candidate_data": targets_with_candidate_data,
        "gold_references": gold_references,
        "gold_in_candidates": gold_in_candidates,
        "gold_candidate_recall": gold_in_candidates / gold_references
        if gold_references
        else 0.0,
    }


_POSITION_BUCKET_KEYS = (
    "gold_position_1",
    "gold_position_2_to_5",
    "gold_position_6_to_10",
    "gold_position_11_to_20",
    "gold_position_21_plus",
    "gold_position_missing",
)


def _candidate_position_buckets(
    targets: list[dict[str, Any]],
    candidate_ids_by_target: dict[str, list[str]],
) -> dict[str, int]:
    buckets = {key: 0 for key in _POSITION_BUCKET_KEYS}
    for row in targets:
        target_id = str(row.get("target_paper_id"))
        if target_id not in candidate_ids_by_target:
            continue
        positions = _gold_candidate_positions(row, candidate_ids_by_target[target_id])
        for position in positions.values():
            buckets[_position_bucket(position)] += 1
    return buckets


def _gold_candidate_positions(
    row: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, int | None]:
    candidate_positions = {
        reference_id: index for index, reference_id in enumerate(candidate_ids, start=1)
    }
    return {
        positive_id: candidate_positions.get(positive_id)
        for positive_id in sorted(_positive_reference_ids(row))
    }


def _position_bucket(position: int | None) -> str:
    if position is None:
        return "gold_position_missing"
    if position == 1:
        return "gold_position_1"
    if 2 <= position <= 5:
        return "gold_position_2_to_5"
    if 6 <= position <= 10:
        return "gold_position_6_to_10"
    if 11 <= position <= 20:
        return "gold_position_11_to_20"
    return "gold_position_21_plus"


def _miss_reason(
    row: dict[str, Any],
    candidate_ids_by_target: dict[str, list[str]],
) -> str:
    target_id = str(row.get("target_paper_id"))
    if target_id not in candidate_ids_by_target:
        return "candidate_status_unknown"
    positives = _positive_reference_ids(row)
    candidates = set(candidate_ids_by_target[target_id])
    if positives & candidates:
        return "gold_present_but_not_selected"
    return "gold_absent_from_candidates"


def _positive_reference_ids(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("positive_reference_ids", []) if value is not None}


def _hit_at(first_positive_ranks: list[int | None], k: int) -> float:
    if not first_positive_ranks:
        return 0.0
    return sum(1 for rank in first_positive_ranks if rank is not None and rank <= k) / len(
        first_positive_ranks
    )


def _join(values: list[Any]) -> str:
    return ", ".join(_text(value) for value in values)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
