from __future__ import annotations

from typing import Any


def benchmark_summary_report(
    *,
    evaluation: dict[str, Any],
    gold_evaluation: dict[str, Any],
    data_readiness: dict[str, Any],
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    parse_success_rate = evaluation.get("parse_success_rate", 0.0)
    invalid_selection_rate = evaluation.get("invalid_selection_rate", 1.0)
    hit_rate = gold_evaluation.get("hit_rate", 0.0)
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
        "decision": {
            "usable_for_benchmark_report": (
                parse_success_rate >= 0.98
                and invalid_selection_rate == 0.0
                and gold_evaluation.get("labeled_targets", 0) > 0
            ),
            "needs_prompt_or_model_tuning": (
                parse_success_rate < 0.98 or invalid_selection_rate > 0.0 or hit_rate < 0.5
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
    ranking_preview: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    missing_metadata = anomaly.get("missing_metadata", {})
    dominance_rows = anomaly.get("score_dominance", {}).get(
        "paper_single_edge_dominance",
        [],
    )
    strongest_dominance = dominance_rows[0] if dominance_rows else {}
    leave_one_out = stability.get("leave_one_target_out", {})
    mode_comparison = stability.get("mode_comparison", {})
    return {
        "claim_scope": "dataset-local content-aware impact ranking",
        "data": {
            "paper_count": data_readiness.get("paper_count", 0),
            "reference_count": data_readiness.get("reference_count", 0),
            "ranking_scope": data_readiness.get("ranking_scope", {}),
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


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
