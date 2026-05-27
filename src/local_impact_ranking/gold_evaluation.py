from __future__ import annotations

import math
import random
from typing import Any

from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper, Reference


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


def gold_candidate_position_report(
    papers: list[Paper],
    *,
    front_position_threshold: int = 1,
) -> dict[str, Any]:
    rows = []
    all_positions: list[int] = []
    best_positions: list[int] = []
    ordering_values = set()

    for paper in papers:
        positive_ids = _positive_reference_ids(paper)
        if not positive_ids:
            continue
        reference_positions = {
            reference.reference_id: index
            for index, reference in enumerate(paper.references, start=1)
            if reference.reference_id
        }
        positions = sorted(
            reference_positions[positive_id]
            for positive_id in positive_ids
            if positive_id in reference_positions
        )
        if not positions:
            continue
        ordering = str(paper.metadata.get("candidate_ordering") or "input_order")
        ordering_values.add(ordering)
        all_positions.extend(positions)
        best_positions.append(positions[0])
        rows.append(
            {
                "target_paper_id": paper.paper_id,
                "candidate_count": len(paper.references),
                "positive_reference_count": len(positive_ids),
                "positive_reference_positions": positions,
                "best_gold_position": positions[0],
                "candidate_ordering": ordering,
            }
        )

    labeled_targets = len(rows)
    front_count = sum(
        1 for position in best_positions if position <= front_position_threshold
    )
    front_share = front_count / labeled_targets if labeled_targets else 0.0
    mean_best = sum(best_positions) / labeled_targets if labeled_targets else 0.0
    mean_all = sum(all_positions) / len(all_positions) if all_positions else 0.0
    return {
        "label_source": "paper.metadata.prescience_positive_reference_ids",
        "labeled_targets": labeled_targets,
        "gold_reference_count": len(all_positions),
        "front_position_threshold": front_position_threshold,
        "front_position_count": front_count,
        "front_position_share": front_share,
        "mean_best_gold_position": mean_best,
        "mean_gold_position": mean_all,
        "candidate_ordering": _candidate_ordering_summary(ordering_values),
        "position_bias_risk": _position_bias_risk(front_share, ordering_values),
        "targets": rows,
        "limitations": [
            "This report detects candidate-order leakage risk; it does not evaluate judge quality by itself.",
            "A low position-bias risk does not remove semantic or metadata leakage in the benchmark.",
        ],
    }


def render_gold_candidate_position_markdown(report: dict[str, Any]) -> str:
    rows = [
        ("labeled_targets", _integer(report.get("labeled_targets"))),
        ("gold_reference_count", _integer(report.get("gold_reference_count"))),
        ("front_position_threshold", _integer(report.get("front_position_threshold"))),
        ("front_position_share", _number(report.get("front_position_share"))),
        ("mean_best_gold_position", _number(report.get("mean_best_gold_position"))),
        ("mean_gold_position", _number(report.get("mean_gold_position"))),
        ("candidate_ordering", str(report.get("candidate_ordering") or "")),
        ("position_bias_risk", str(report.get("position_bias_risk") or "")),
    ]
    lines = [
        "# Gold Candidate Position",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(["", "## Limitations", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def gold_baseline_comparison_report(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    top_k: int,
    random_seed: int = 13,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    methods = [
        ("judge", judge_results),
        ("candidate_order", _candidate_order_results(papers, top_k=top_k)),
        ("title_overlap", _title_overlap_results(papers, top_k=top_k)),
        ("random", _random_results(papers, top_k=top_k, seed=random_seed)),
    ]
    method_rows = []
    for name, results in methods:
        evaluation = evaluate_gold_reference_selection(papers, results, top_k=top_k)
        method_rows.append(
            {
                "method": name,
                "labeled_targets": evaluation["labeled_targets"],
                "hit_rate": evaluation["hit_rate"],
                "selection_precision": evaluation["selection_precision"],
                "selection_recall": evaluation["selection_recall"],
                "mean_reciprocal_rank": evaluation["mean_reciprocal_rank"],
                "mean_ndcg": evaluation["mean_ndcg"],
            }
        )
    best_method = max(
        method_rows,
        key=lambda row: (row["mean_ndcg"], row["hit_rate"], row["method"] == "judge"),
    )
    judge = next(row for row in method_rows if row["method"] == "judge")
    baselines = [row for row in method_rows if row["method"] != "judge"]
    best_baseline = max(
        baselines,
        key=lambda row: (row["mean_ndcg"], row["hit_rate"]),
    )
    judge_delta = judge["mean_ndcg"] - best_baseline["mean_ndcg"]
    target_rows = _baseline_target_delta_rows(
        papers,
        judge_results,
        _results_for_method(methods, best_baseline["method"]),
        top_k=top_k,
    )
    bootstrap = _paired_bootstrap_delta(
        [row["ndcg_delta"] for row in target_rows],
        samples=bootstrap_samples,
        seed=random_seed,
    )
    statistically_positive_delta = bootstrap["ci95"][0] > 0
    judge_beats_baselines = judge_delta > 0
    return {
        "top_k": top_k,
        "random_seed": random_seed,
        "methods": method_rows,
        "best_method_by_mean_ndcg": best_method["method"],
        "best_baseline_by_mean_ndcg": best_baseline["method"],
        "judge_vs_best_baseline": {
            "best_baseline": best_baseline["method"],
            "hit_rate_delta": judge["hit_rate"] - best_baseline["hit_rate"],
            "mean_reciprocal_rank_delta": (
                judge["mean_reciprocal_rank"]
                - best_baseline["mean_reciprocal_rank"]
            ),
            "mean_ndcg_delta": judge_delta,
            "mean_ndcg_delta_ci95": bootstrap["ci95"],
            "paired_bootstrap_samples": bootstrap_samples,
            "statistically_positive_ndcg_delta": statistically_positive_delta,
        },
        "targets": target_rows,
        "decision": {
            "judge_beats_all_baselines": judge_beats_baselines,
            "judge_baseline_delta_statistically_positive": statistically_positive_delta,
            "needs_model_or_prompt_improvement": not (
                judge_beats_baselines and statistically_positive_delta
            ),
        },
        "limitations": [
            "Baselines are computed only over candidate references available in this run.",
            "Random baseline is seed dependent.",
            "A judge that beats these baselines is better than simple candidate heuristics, not proven globally valid.",
        ],
    }


def render_gold_baseline_comparison_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    delta = report.get("judge_vs_best_baseline", {})
    lines = [
        "# Gold Baseline Comparison",
        "",
        "## Methods",
        "",
        "| Method | Labeled targets | Hit rate | Precision | Recall | MRR | NDCG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("methods", []):
        lines.append(
            "| "
            f"{row.get('method')} | "
            f"{_integer(row.get('labeled_targets'))} | "
            f"{_number(row.get('hit_rate'))} | "
            f"{_number(row.get('selection_precision'))} | "
            f"{_number(row.get('selection_recall'))} | "
            f"{_number(row.get('mean_reciprocal_rank'))} | "
            f"{_number(row.get('mean_ndcg'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| best_method_by_mean_ndcg | {report.get('best_method_by_mean_ndcg', '')} |",
            f"| best_baseline_by_mean_ndcg | {report.get('best_baseline_by_mean_ndcg', '')} |",
            f"| judge_mean_ndcg_delta_vs_best_baseline | {_number(delta.get('mean_ndcg_delta'))} |",
            f"| judge_ndcg_delta_ci95 | {_ci(delta.get('mean_ndcg_delta_ci95'))} |",
            f"| statistically_positive_ndcg_delta | {_yes_no(delta.get('statistically_positive_ndcg_delta'))} |",
            f"| judge_beats_all_baselines | {_yes_no(decision.get('judge_beats_all_baselines'))} |",
            f"| judge_baseline_delta_statistically_positive | {_yes_no(decision.get('judge_baseline_delta_statistically_positive', delta.get('statistically_positive_ndcg_delta')))} |",
            f"| needs_model_or_prompt_improvement | {_yes_no(decision.get('needs_model_or_prompt_improvement'))} |",
            "",
            "## Limitations",
            "",
        ]
    )
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


def _candidate_ordering_summary(ordering_values: set[str]) -> str:
    if not ordering_values:
        return "none"
    if len(ordering_values) == 1:
        return next(iter(ordering_values))
    return "mixed"


def _position_bias_risk(front_share: float, ordering_values: set[str]) -> str:
    if "deterministic_shuffle" in ordering_values and front_share < 0.8:
        return "low"
    if front_share >= 0.8:
        return "high"
    if front_share >= 0.5:
        return "medium"
    return "low"


def _candidate_order_results(papers: list[Paper], *, top_k: int) -> list[JudgeResult]:
    return [
        _result_from_references(
            paper,
            paper.references[:top_k],
            rationale="Selected by input candidate order baseline.",
        )
        for paper in papers
    ]


def _title_overlap_results(papers: list[Paper], *, top_k: int) -> list[JudgeResult]:
    results = []
    for paper in papers:
        query_terms = _tokenize(" ".join([paper.title, paper.abstract or ""]))
        references = sorted(
            paper.references,
            key=lambda reference: (
                -len(query_terms & _tokenize(reference.title)),
                reference.reference_id or "",
            ),
        )
        results.append(
            _result_from_references(
                paper,
                references[:top_k],
                rationale="Selected by title/abstract lexical-overlap baseline.",
            )
        )
    return results


def _random_results(
    papers: list[Paper],
    *,
    top_k: int,
    seed: int,
) -> list[JudgeResult]:
    results = []
    for paper in papers:
        references = list(paper.references)
        random.Random(f"{seed}:{paper.paper_id}").shuffle(references)
        results.append(
            _result_from_references(
                paper,
                references[:top_k],
                rationale="Selected by seeded random baseline.",
            )
        )
    return results


def _results_for_method(
    methods: list[tuple[str, list[JudgeResult]]],
    method_name: str,
) -> list[JudgeResult]:
    for name, results in methods:
        if name == method_name:
            return results
    return []


def _baseline_target_delta_rows(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    baseline_results: list[JudgeResult],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    judge_by_id = {result.target_paper_id: result for result in judge_results}
    baseline_by_id = {result.target_paper_id: result for result in baseline_results}
    rows = []
    for paper in papers:
        positive_ids = _positive_reference_ids(paper)
        if not positive_ids:
            continue
        judge_selected = _selected_reference_ids(
            judge_by_id.get(paper.paper_id),
            top_k=top_k,
        )
        baseline_selected = _selected_reference_ids(
            baseline_by_id.get(paper.paper_id),
            top_k=top_k,
        )
        judge_ndcg = _ndcg(judge_selected, positive_ids, top_k=top_k)
        baseline_ndcg = _ndcg(baseline_selected, positive_ids, top_k=top_k)
        rows.append(
            {
                "target_paper_id": paper.paper_id,
                "judge_ndcg": judge_ndcg,
                "best_baseline_ndcg": baseline_ndcg,
                "ndcg_delta": judge_ndcg - baseline_ndcg,
            }
        )
    return rows


def _paired_bootstrap_delta(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not deltas:
        return {"ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    sample_means = []
    count = len(deltas)
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        sample_means.append(total / count)
    sample_means.sort()
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples) - 1)
    return {"ci95": [sample_means[lower_index], sample_means[upper_index]]}


def _result_from_references(
    paper: Paper,
    references: list[Reference],
    *,
    rationale: str,
) -> JudgeResult:
    selections = [
        SelectedReference(
            reference.reference_id or "",
            rank=index,
            confidence=0.0,
            rationale=rationale,
            valid=bool(reference.reference_id),
        )
        for index, reference in enumerate(references, start=1)
    ]
    return JudgeResult(
        target_paper_id=paper.paper_id,
        selected_references=selections,
        raw_output="baseline",
        candidate_reference_ids=tuple(
            reference.reference_id or "" for reference in paper.references
        ),
    )


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.replace("-", " ").split() if len(token) > 3}


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _ci(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"
