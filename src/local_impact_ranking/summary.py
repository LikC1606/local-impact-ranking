from __future__ import annotations

from typing import Any


def benchmark_summary_report(
    *,
    evaluation: dict[str, Any],
    gold_evaluation: dict[str, Any],
    gold_baseline_comparison: dict[str, Any] | None = None,
    data_readiness: dict[str, Any],
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    gold_baseline_comparison = gold_baseline_comparison or {}
    parse_success_rate = evaluation.get("parse_success_rate", 0.0)
    invalid_selection_rate = evaluation.get("invalid_selection_rate", 1.0)
    hit_rate = gold_evaluation.get("hit_rate", 0.0)
    judge_vs_best = gold_baseline_comparison.get("judge_vs_best_baseline", {})
    baseline_decision = gold_baseline_comparison.get("decision", {})
    has_baseline_decision = bool(gold_baseline_comparison)
    judge_beats_baselines = bool(
        baseline_decision.get("judge_beats_all_baselines", False)
    )
    baseline_delta_is_stable = bool(
        judge_vs_best.get("statistically_positive_ndcg_delta", False)
    )
    return {
        "benchmark": {
            "paper_count": data_readiness.get("paper_count", 0),
            "reference_count": data_readiness.get("reference_count", 0),
            "judge_results": evaluation.get("judge_results", 0),
            "credit_mode": scoring_config.get("credit_mode"),
        },
        "judge_quality": {
            "parse_success_rate": parse_success_rate,
            "invalid_selection_rate": invalid_selection_rate,
            "empty_selection_rate": evaluation.get("empty_selection_rate", 0.0),
            "top_k_compliance_rate": evaluation.get("top_k_compliance_rate", 0.0),
        },
        "gold_quality": {
            "label_source": gold_evaluation.get("label_source"),
            "labeled_targets": gold_evaluation.get("labeled_targets", 0),
            "hit_rate": hit_rate,
            "selection_precision": gold_evaluation.get("selection_precision", 0.0),
            "selection_recall": gold_evaluation.get("selection_recall", 0.0),
            "mean_reciprocal_rank": gold_evaluation.get("mean_reciprocal_rank", 0.0),
            "mean_ndcg": gold_evaluation.get("mean_ndcg", 0.0),
        },
        "gold_baseline_comparison": {
            "best_baseline_by_mean_ndcg": gold_baseline_comparison.get(
                "best_baseline_by_mean_ndcg"
            ),
            "judge_mean_ndcg_delta_vs_best_baseline": judge_vs_best.get(
                "mean_ndcg_delta",
                0.0,
            ),
            "judge_ndcg_delta_ci95": judge_vs_best.get(
                "mean_ndcg_delta_ci95",
                [0.0, 0.0],
            ),
            "statistically_positive_ndcg_delta": judge_vs_best.get(
                "statistically_positive_ndcg_delta",
                False,
            ),
            "judge_hit_rate_delta_vs_best_baseline": judge_vs_best.get(
                "hit_rate_delta",
                0.0,
            ),
            "judge_beats_all_baselines": baseline_decision.get(
                "judge_beats_all_baselines",
                False,
            ),
        },
        "decision": {
            "usable_for_benchmark_report": (
                parse_success_rate >= 0.98
                and invalid_selection_rate == 0.0
                and gold_evaluation.get("labeled_targets", 0) > 0
            ),
            "needs_prompt_or_model_tuning": (
                parse_success_rate < 0.98
                or invalid_selection_rate > 0.0
                or hit_rate < 0.5
                or (
                    has_baseline_decision
                    and not (judge_beats_baselines and baseline_delta_is_stable)
                )
            ),
        },
        "limitations": [
            "Gold metrics evaluate reference selection, not absolute global impact.",
            "Benchmark quality depends on the candidate references and labels available in the input data.",
            "This report supports local pipeline evaluation; it does not prove external field-wide impact.",
        ],
    }


def render_benchmark_summary_markdown(summary: dict[str, Any]) -> str:
    benchmark = summary.get("benchmark", {})
    judge_quality = summary.get("judge_quality", {})
    gold_quality = summary.get("gold_quality", {})
    gold_baseline = summary.get("gold_baseline_comparison", {})
    decision = summary.get("decision", {})
    limitations = summary.get("limitations", [])

    lines = [
        "# Benchmark Summary",
        "",
        "## Run",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {benchmark.get('paper_count', 0)} |",
        f"| reference_count | {benchmark.get('reference_count', 0)} |",
        f"| judge_results | {benchmark.get('judge_results', 0)} |",
        f"| credit_mode | {_text(benchmark.get('credit_mode'))} |",
        "",
        "## Judge Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| parse_success_rate | {_number(judge_quality.get('parse_success_rate'))} |",
        f"| invalid_selection_rate | {_number(judge_quality.get('invalid_selection_rate'))} |",
        f"| empty_selection_rate | {_number(judge_quality.get('empty_selection_rate'))} |",
        f"| top_k_compliance_rate | {_number(judge_quality.get('top_k_compliance_rate'))} |",
        "",
        "## Gold Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| label_source | {_text(gold_quality.get('label_source'))} |",
        f"| labeled_targets | {gold_quality.get('labeled_targets', 0)} |",
        f"| hit_rate | {_number(gold_quality.get('hit_rate'))} |",
        f"| selection_precision | {_number(gold_quality.get('selection_precision'))} |",
        f"| selection_recall | {_number(gold_quality.get('selection_recall'))} |",
        f"| mean_reciprocal_rank | {_number(gold_quality.get('mean_reciprocal_rank'))} |",
        f"| mean_ndcg | {_number(gold_quality.get('mean_ndcg'))} |",
        "",
        "## Gold Baseline Comparison",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| best_baseline_by_mean_ndcg | {_text(gold_baseline.get('best_baseline_by_mean_ndcg'))} |",
        f"| judge_mean_ndcg_delta_vs_best_baseline | {_number(gold_baseline.get('judge_mean_ndcg_delta_vs_best_baseline'))} |",
        f"| judge_ndcg_delta_ci95 | {_ci(gold_baseline.get('judge_ndcg_delta_ci95'))} |",
        f"| statistically_positive_ndcg_delta | {_yes_no(gold_baseline.get('statistically_positive_ndcg_delta'))} |",
        f"| judge_hit_rate_delta_vs_best_baseline | {_number(gold_baseline.get('judge_hit_rate_delta_vs_best_baseline'))} |",
        f"| judge_beats_all_baselines | {_yes_no(gold_baseline.get('judge_beats_all_baselines'))} |",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| usable_for_benchmark_report | {_yes_no(decision.get('usable_for_benchmark_report'))} |",
        f"| needs_prompt_or_model_tuning | {_yes_no(decision.get('needs_prompt_or_model_tuning'))} |",
        "",
        "## Limitations",
        "",
    ]
    for limitation in limitations:
        lines.append(f"- {_text(limitation)}")
    lines.append("")
    return "\n".join(lines)


def mvp_summary_report(
    *,
    data_readiness: dict[str, Any],
    evaluation: dict[str, Any],
    proxy_evaluation: dict[str, Any],
    anomaly: dict[str, Any],
    stability: dict[str, Any],
    scoring_config: dict[str, Any],
    paper_stability: dict[str, Any] | None = None,
    paper_score_uncertainty: dict[str, Any] | None = None,
    context_coverage: dict[str, Any] | None = None,
    ranking_preview: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    context_coverage = context_coverage or {}
    paper_stability = paper_stability or {}
    paper_score_uncertainty = paper_score_uncertainty or {}
    missing_metadata = anomaly.get("missing_metadata", {})
    dominance_rows = anomaly.get("score_dominance", {}).get(
        "paper_single_edge_dominance",
        [],
    )
    strongest_dominance = dominance_rows[0] if dominance_rows else {}
    leave_one_out = stability.get("leave_one_target_out", {})
    mode_comparison = stability.get("mode_comparison", {})
    confidence = _confidence_report(
        evaluation=evaluation,
        proxy_evaluation=proxy_evaluation,
        anomaly=anomaly,
        stability=stability,
        context_coverage=context_coverage,
    )
    return {
        "claim_scope": "dataset-local content-aware impact ranking",
        "data": {
            "paper_count": data_readiness.get("paper_count", 0),
            "reference_count": data_readiness.get("reference_count", 0),
            "ranking_scope": data_readiness.get("ranking_scope", {}),
            "context_coverage": {
                "papers_with_contexts": context_coverage.get("papers_with_contexts", 0),
                "paper_context_coverage_rate": context_coverage.get(
                    "paper_context_coverage_rate",
                    0.0,
                ),
                "references_with_contexts": context_coverage.get(
                    "references_with_contexts",
                    0,
                ),
                "reference_context_coverage_rate": context_coverage.get(
                    "reference_context_coverage_rate",
                    0.0,
                ),
                "citation_contexts": context_coverage.get("citation_contexts", 0),
            },
        },
        "scoring": {
            "version": scoring_config.get("version"),
            "credit_mode": scoring_config.get("credit_mode"),
        },
        "ranking_preview": ranking_preview or {
            "papers": [],
            "authors": [],
            "institutions": [],
            "fields": [],
            "years": [],
        },
        "judge_quality": {
            "parse_success_rate": evaluation.get("parse_success_rate", 0.0),
            "invalid_selection_rate": evaluation.get("invalid_selection_rate", 0.0),
            "empty_selection_rate": evaluation.get("empty_selection_rate", 0.0),
            "top_k_compliance_rate": evaluation.get("top_k_compliance_rate", 0.0),
        },
        "proxy_quality": {
            "unsupported_selection_rate": proxy_evaluation.get(
                "unsupported_selection_rate",
                0.0,
            ),
            "context_supported_selection_rate": proxy_evaluation.get(
                "context_supported_selection_rate",
                0.0,
            ),
            "method_context_selection_rate": proxy_evaluation.get(
                "method_context_selection_rate",
                0.0,
            ),
            "bibliometric_top_n_support_rate": proxy_evaluation.get(
                "bibliometric_top_n_support_rate",
                0.0,
            ),
        },
        "anomaly_flags": {
            "references_missing_title_count": missing_metadata.get(
                "references_missing_title_count",
                0,
            ),
            "references_missing_author_count": missing_metadata.get(
                "references_missing_author_count",
                0,
            ),
            "references_missing_affiliation_count": missing_metadata.get(
                "references_missing_affiliation_count",
                0,
            ),
            "highest_paper_single_edge_dominance_id": strongest_dominance.get("id"),
            "highest_paper_single_edge_dominance_share": strongest_dominance.get(
                "max_single_edge_share",
                0.0,
            ),
        },
        "stability": {
            "leave_one_target_out_average_retention_rate": leave_one_out.get(
                "average_top_n_retention_rate",
                0.0,
            ),
            "leave_one_target_out_min_retention_rate": leave_one_out.get(
                "min_top_n_retention_rate",
                0.0,
            ),
            "mode_comparison": _compact_mode_comparison(mode_comparison),
        },
        "paper_stability": _compact_paper_stability(paper_stability),
        "paper_score_uncertainty": _compact_paper_score_uncertainty(
            paper_score_uncertainty
        ),
        "decision_support": {
            "ready_for_larger_pilot": _ready_for_larger_pilot(
                data_readiness,
                evaluation,
                proxy_evaluation,
                stability,
            ),
            "must_not_claim_global_impact": True,
            "requires_human_labels_for_human_validated_correctness": True,
        },
        "confidence": confidence,
        "limitations": [
            "This summary aggregates proxy and robustness evidence; it is not a ground-truth impact label.",
            "The valid claim is limited to the input dataset and available metadata coverage.",
            "Human-validated correctness requires human labels or an accepted external gold standard.",
        ],
    }


def render_mvp_summary_markdown(summary: dict[str, Any]) -> str:
    data = summary.get("data", {})
    ranking_scope = data.get("ranking_scope", {})
    scoring = summary.get("scoring", {})
    ranking_preview = summary.get("ranking_preview", {})
    judge_quality = summary.get("judge_quality", {})
    proxy_quality = summary.get("proxy_quality", {})
    anomaly_flags = summary.get("anomaly_flags", {})
    stability = summary.get("stability", {})
    paper_stability = summary.get("paper_stability", {})
    paper_score_uncertainty = summary.get("paper_score_uncertainty", {})
    confidence = summary.get("confidence", {})
    context_coverage = data.get("context_coverage", {})
    decision_support = summary.get("decision_support", {})
    limitations = summary.get("limitations", [])

    lines = [
        "# Local Impact Ranking MVP Summary",
        "",
        "## Scope",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Claim scope | {_text(summary.get('claim_scope'))} |",
        f"| Ready for larger pilot | {_yes_no(decision_support.get('ready_for_larger_pilot'))} |",
        f"| Must not claim global impact | {_yes_no(decision_support.get('must_not_claim_global_impact'))} |",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {data.get('paper_count', 0)} |",
        f"| reference_count | {data.get('reference_count', 0)} |",
        "",
        "## Context Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| papers_with_contexts | {context_coverage.get('papers_with_contexts', 0)} |",
        f"| paper_context_coverage_rate | {_number(context_coverage.get('paper_context_coverage_rate'))} |",
        f"| references_with_contexts | {context_coverage.get('references_with_contexts', 0)} |",
        f"| reference_context_coverage_rate | {_number(context_coverage.get('reference_context_coverage_rate'))} |",
        f"| citation_contexts | {context_coverage.get('citation_contexts', 0)} |",
        "",
        "## Ranking Scope",
        "",
        "| Ranking | Scope |",
        "| --- | --- |",
        f"| paper_ranking | {_text(ranking_scope.get('paper_ranking'))} |",
        f"| author_ranking | {_text(ranking_scope.get('author_ranking'))} |",
        f"| institution_ranking | {_text(ranking_scope.get('institution_ranking'))} |",
        f"| context_aware_ranking | {_text(ranking_scope.get('context_aware_ranking'))} |",
        "",
        "## Scoring",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| version | {_text(scoring.get('version'))} |",
        f"| credit_mode | {_text(scoring.get('credit_mode'))} |",
        "",
        "## Top Papers",
        "",
        "| Rank | ID | Score |",
        "| ---: | --- | ---: |",
        *_ranking_preview_rows(ranking_preview.get("papers", [])),
        "",
        "## Top Authors",
        "",
        "| Rank | ID | Score |",
        "| ---: | --- | ---: |",
        *_ranking_preview_rows(ranking_preview.get("authors", [])),
        "",
        "## Top Institutions",
        "",
        "| Rank | ID | Score |",
        "| ---: | --- | ---: |",
        *_ranking_preview_rows(ranking_preview.get("institutions", [])),
        "",
        "## Top Fields",
        "",
        "| Rank | ID | Score |",
        "| ---: | --- | ---: |",
        *_ranking_preview_rows(ranking_preview.get("fields", [])),
        "",
        "## Top Years",
        "",
        "| Rank | ID | Score |",
        "| ---: | --- | ---: |",
        *_ranking_preview_rows(ranking_preview.get("years", [])),
        "",
        "## Judge Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| parse_success_rate | {_number(judge_quality.get('parse_success_rate'))} |",
        f"| invalid_selection_rate | {_number(judge_quality.get('invalid_selection_rate'))} |",
        f"| empty_selection_rate | {_number(judge_quality.get('empty_selection_rate'))} |",
        f"| top_k_compliance_rate | {_number(judge_quality.get('top_k_compliance_rate'))} |",
        "",
        "## Proxy Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| unsupported_selection_rate | {_number(proxy_quality.get('unsupported_selection_rate'))} |",
        f"| context_supported_selection_rate | {_number(proxy_quality.get('context_supported_selection_rate'))} |",
        f"| method_context_selection_rate | {_number(proxy_quality.get('method_context_selection_rate'))} |",
        f"| bibliometric_top_n_support_rate | {_number(proxy_quality.get('bibliometric_top_n_support_rate'))} |",
        "",
        "## Anomaly Flags",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| references_missing_title_count | {anomaly_flags.get('references_missing_title_count', 0)} |",
        f"| references_missing_author_count | {anomaly_flags.get('references_missing_author_count', 0)} |",
        f"| references_missing_affiliation_count | {anomaly_flags.get('references_missing_affiliation_count', 0)} |",
        f"| highest_paper_single_edge_dominance_id | {_text(anomaly_flags.get('highest_paper_single_edge_dominance_id'))} |",
        f"| highest_paper_single_edge_dominance_share | {_number(anomaly_flags.get('highest_paper_single_edge_dominance_share'))} |",
        "",
        "## Stability",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| leave_one_target_out_average_retention_rate | {_number(stability.get('leave_one_target_out_average_retention_rate'))} |",
        f"| leave_one_target_out_min_retention_rate | {_number(stability.get('leave_one_target_out_min_retention_rate'))} |",
        "",
        "| Comparison mode | top_n_overlap_rate | kendall_tau |",
        "| --- | ---: | ---: |",
    ]
    for mode, metrics in stability.get("mode_comparison", {}).items():
        lines.append(
            f"| {_text(mode)} | {_number(metrics.get('top_n_overlap_rate'))} | {_number(metrics.get('kendall_tau'))} |"
        )
    lines.extend(
        [
            "",
            "## Paper Stability",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| stable_papers | {paper_stability.get('stable_papers', 0)} |",
            f"| moderate_papers | {paper_stability.get('moderate_papers', 0)} |",
            f"| unstable_papers | {paper_stability.get('unstable_papers', 0)} |",
            f"| insufficient_sample_papers | {paper_stability.get('insufficient_sample_papers', 0)} |",
            "",
            "| Paper ID | Base rank | Rank span | Dropped count | Stability band |",
            "| --- | ---: | ---: | ---: | --- |",
            *_paper_stability_rows(paper_stability.get("most_unstable_papers", [])),
            "",
            "## Paper Score Uncertainty",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| stable_papers | {paper_score_uncertainty.get('stable_papers', 0)} |",
            f"| moderate_papers | {paper_score_uncertainty.get('moderate_papers', 0)} |",
            f"| unstable_papers | {paper_score_uncertainty.get('unstable_papers', 0)} |",
            f"| insufficient_sample_papers | {paper_score_uncertainty.get('insufficient_sample_papers', 0)} |",
            "",
            "| Paper ID | Base rank | Score CI width | Rank span | Zero score rate | Uncertainty band |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *_paper_score_uncertainty_rows(
                paper_score_uncertainty.get("highest_uncertainty_papers", [])
            ),
            "",
            "## Confidence",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| confidence_band | {_text(confidence.get('band'))} |",
            f"| coverage_band | {_text(confidence.get('coverage_band'))} |",
            f"| dominance_risk_band | {_text(confidence.get('dominance_risk_band'))} |",
            f"| stability_band | {_text(confidence.get('stability_band'))} |",
            f"| max_single_edge_share | {_number(confidence.get('max_single_edge_share'))} |",
            f"| leave_one_target_out_average_retention_rate | {_number(confidence.get('leave_one_target_out_average_retention_rate'))} |",
            "",
            "### Confidence Reasons",
            "",
            *_confidence_reason_rows(confidence.get("reasons", [])),
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in limitations:
        lines.append(f"- {_text(limitation)}")
    lines.append("")
    return "\n".join(lines)


def ranking_preview_from_scores(
    *,
    paper_scores: dict[str, float],
    author_scores: dict[str, float],
    institution_scores: dict[str, float],
    field_scores: dict[str, float] | None = None,
    year_scores: dict[str, float] | None = None,
    top_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "papers": _top_score_rows(paper_scores, top_n),
        "authors": _top_score_rows(author_scores, top_n),
        "institutions": _top_score_rows(institution_scores, top_n),
        "fields": _top_score_rows(field_scores or {}, top_n),
        "years": _top_score_rows(year_scores or {}, top_n),
    }


def _compact_mode_comparison(mode_comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        mode: {
            "top_n_overlap_rate": metrics.get("top_n_overlap_rate", 0.0),
            "kendall_tau": metrics.get("kendall_tau", 0.0),
        }
        for mode, metrics in mode_comparison.items()
    }


def _compact_paper_stability(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("papers", []))
    band_counts = {
        "stable": 0,
        "moderate": 0,
        "unstable": 0,
        "insufficient_samples": 0,
    }
    for row in rows:
        band = row.get("stability_band")
        if band in band_counts:
            band_counts[band] += 1

    most_unstable = sorted(
        rows,
        key=lambda row: (
            -int(row.get("dropped_count", 0) or 0),
            -int(row.get("rank_span", 0) or 0),
            int(row.get("base_rank", 0) or 0),
            str(row.get("paper_id", "")),
        ),
    )[:10]
    return {
        "top_n": report.get("top_n", 0),
        "sample_count": report.get("sample_count", 0),
        "stable_papers": band_counts["stable"],
        "moderate_papers": band_counts["moderate"],
        "unstable_papers": band_counts["unstable"],
        "insufficient_sample_papers": band_counts["insufficient_samples"],
        "most_unstable_papers": [
            {
                "paper_id": row.get("paper_id"),
                "base_rank": row.get("base_rank", 0),
                "rank_span": row.get("rank_span", 0),
                "dropped_count": row.get("dropped_count", 0),
                "stability_band": row.get("stability_band", ""),
            }
            for row in most_unstable
        ],
    }


def _compact_paper_score_uncertainty(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("papers", []))
    band_counts = {
        "stable": 0,
        "moderate": 0,
        "unstable": 0,
        "insufficient_samples": 0,
    }
    for row in rows:
        band = row.get("uncertainty_band")
        if band in band_counts:
            band_counts[band] += 1

    highest_uncertainty = sorted(
        rows,
        key=lambda row: (
            -float(row.get("zero_score_rate", 0.0) or 0.0),
            -float(row.get("score_ci_width", 0.0) or 0.0),
            -int(row.get("rank_span", 0) or 0),
            int(row.get("base_rank", 0) or 0),
            str(row.get("paper_id", "")),
        ),
    )[:10]
    return {
        "top_n": report.get("top_n", 0),
        "samples": report.get("samples", 0),
        "sample_count": report.get("sample_count", 0),
        "stable_papers": band_counts["stable"],
        "moderate_papers": band_counts["moderate"],
        "unstable_papers": band_counts["unstable"],
        "insufficient_sample_papers": band_counts["insufficient_samples"],
        "highest_uncertainty_papers": [
            {
                "paper_id": row.get("paper_id"),
                "base_rank": row.get("base_rank", 0),
                "score_ci_width": row.get("score_ci_width", 0.0),
                "rank_span": row.get("rank_span", 0),
                "zero_score_rate": row.get("zero_score_rate", 0.0),
                "uncertainty_band": row.get("uncertainty_band", ""),
            }
            for row in highest_uncertainty
        ],
    }


def _top_score_rows(scores: dict[str, float], top_n: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "id": identifier,
            "score": score,
        }
        for rank, (identifier, score) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n],
            start=1,
        )
    ]


def _ranking_preview_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["|  |  |  |"]
    return [
        f"| {row.get('rank', '')} | {_text(row.get('id'))} | {_number(row.get('score'))} |"
        for row in rows
    ]


def _confidence_reason_rows(reasons: list[Any]) -> list[str]:
    if not reasons:
        return ["- no confidence reasons recorded"]
    return [f"- {_text(reason)}" for reason in reasons]


def _paper_stability_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["|  |  |  |  |  |"]
    return [
        f"| {_text(row.get('paper_id'))} | {row.get('base_rank', 0)} | {row.get('rank_span', 0)} | {row.get('dropped_count', 0)} | {_text(row.get('stability_band'))} |"
        for row in rows
    ]


def _paper_score_uncertainty_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["|  |  |  |  |  |  |"]
    return [
        f"| {_text(row.get('paper_id'))} | {row.get('base_rank', 0)} | {_number(row.get('score_ci_width'))} | {row.get('rank_span', 0)} | {_number(row.get('zero_score_rate'))} | {_text(row.get('uncertainty_band'))} |"
        for row in rows
    ]


def _ready_for_larger_pilot(
    data_readiness: dict[str, Any],
    evaluation: dict[str, Any],
    proxy_evaluation: dict[str, Any],
    stability: dict[str, Any],
) -> bool:
    ranking_scope = data_readiness.get("ranking_scope", {})
    leave_one_out = stability.get("leave_one_target_out", {})
    return (
        ranking_scope.get("paper_ranking") == "formal"
        and ranking_scope.get("author_ranking") == "formal"
        and ranking_scope.get("institution_ranking") == "formal"
        and evaluation.get("parse_success_rate", 0.0) >= 0.98
        and evaluation.get("invalid_selection_rate", 1.0) == 0.0
        and evaluation.get("top_k_compliance_rate", 0.0) == 1.0
        and proxy_evaluation.get("unsupported_selection_rate", 1.0) <= 0.5
        and leave_one_out.get("average_top_n_retention_rate", 0.0) >= 0.5
    )


def _confidence_report(
    *,
    evaluation: dict[str, Any],
    proxy_evaluation: dict[str, Any],
    anomaly: dict[str, Any],
    stability: dict[str, Any],
    context_coverage: dict[str, Any],
) -> dict[str, Any]:
    parse_success_rate = evaluation.get("parse_success_rate", 0.0)
    invalid_selection_rate = evaluation.get("invalid_selection_rate", 1.0)
    top_k_compliance_rate = evaluation.get("top_k_compliance_rate", 0.0)
    unsupported_selection_rate = proxy_evaluation.get("unsupported_selection_rate", 1.0)
    leave_one_out = stability.get("leave_one_target_out", {})
    average_retention_rate = leave_one_out.get("average_top_n_retention_rate", 0.0)
    dominance_rows = anomaly.get("score_dominance", {}).get(
        "paper_single_edge_dominance",
        [],
    )
    max_single_edge_share = (
        dominance_rows[0].get("max_single_edge_share", 0.0) if dominance_rows else 0.0
    )
    paper_context_rate = context_coverage.get("paper_context_coverage_rate", 0.0)
    reference_context_rate = context_coverage.get("reference_context_coverage_rate", 0.0)
    min_context_rate = min(paper_context_rate, reference_context_rate)

    coverage_band = _three_level_band(min_context_rate, high=0.5, medium=0.2)
    stability_band = _three_level_band(average_retention_rate, high=0.75, medium=0.5)
    dominance_risk_band = (
        "high"
        if max_single_edge_share >= 0.75
        else "medium"
        if max_single_edge_share >= 0.5
        else "low"
    )

    if (
        parse_success_rate >= 0.98
        and invalid_selection_rate == 0.0
        and top_k_compliance_rate >= 1.0
        and coverage_band == "high"
        and stability_band == "high"
        and dominance_risk_band == "low"
        and unsupported_selection_rate <= 0.25
    ):
        band = "high"
    elif (
        parse_success_rate >= 0.95
        and invalid_selection_rate <= 0.02
        and top_k_compliance_rate >= 0.9
        and coverage_band != "low"
        and stability_band != "low"
        and dominance_risk_band != "high"
        and unsupported_selection_rate <= 0.5
    ):
        band = "medium"
    else:
        band = "low"

    reasons = [
        f"parse_success_rate={_number(parse_success_rate)}",
        f"invalid_selection_rate={_number(invalid_selection_rate)}",
        f"top_k_compliance_rate={_number(top_k_compliance_rate)}",
        f"paper_context_coverage_rate={_number(paper_context_rate)}",
        f"reference_context_coverage_rate={_number(reference_context_rate)}",
        f"leave_one_target_out_average_retention_rate={_number(average_retention_rate)}",
        f"max_single_edge_share={_number(max_single_edge_share)}",
        f"unsupported_selection_rate={_number(unsupported_selection_rate)}",
    ]
    if coverage_band == "low":
        reasons.append("low citation-context coverage")
    if stability_band == "low":
        reasons.append("unstable under leave-one-out perturbation")
    if dominance_risk_band == "high":
        reasons.append("single-edge dominance is high")
    if unsupported_selection_rate > 0.5:
        reasons.append("many selected references lack proxy support")

    return {
        "band": band,
        "coverage_band": coverage_band,
        "dominance_risk_band": dominance_risk_band,
        "stability_band": stability_band,
        "max_single_edge_share": max_single_edge_share,
        "leave_one_target_out_average_retention_rate": average_retention_rate,
        "paper_context_coverage_rate": paper_context_rate,
        "reference_context_coverage_rate": reference_context_rate,
        "unsupported_selection_rate": unsupported_selection_rate,
        "reasons": reasons,
        "flags": {
            "low_context_coverage": coverage_band == "low",
            "high_single_edge_dominance": dominance_risk_band == "high",
            "unstable_leave_one_out": stability_band == "low",
            "high_unsupported_selection_rate": unsupported_selection_rate > 0.5,
        },
    }


def _three_level_band(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _ci(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
