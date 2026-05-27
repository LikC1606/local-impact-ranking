from local_impact_ranking.summary import (
    benchmark_summary_report,
    mvp_summary_report,
    render_benchmark_summary_markdown,
    render_mvp_summary_markdown,
)


def test_mvp_summary_report_collects_key_quality_and_scope_fields():
    summary = mvp_summary_report(
        data_readiness={
            "paper_count": 10,
            "reference_count": 100,
            "ranking_scope": {
                "paper_ranking": "formal",
                "author_ranking": "exploratory",
                "institution_ranking": "exploratory",
                "context_aware_ranking": "metadata-only",
            },
        },
        evaluation={
            "parse_success_rate": 1.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.1,
            "top_k_compliance_rate": 1.0,
        },
        proxy_evaluation={
            "unsupported_selection_rate": 0.2,
            "context_supported_selection_rate": 0.5,
            "method_context_selection_rate": 0.3,
            "bibliometric_top_n_support_rate": 0.6,
        },
        anomaly={
            "missing_metadata": {
                "references_missing_title_count": 2,
                "references_missing_author_count": 3,
                "references_missing_affiliation_count": 4,
            },
            "score_dominance": {
                "paper_single_edge_dominance": [
                    {"id": "r1", "max_single_edge_share": 1.0}
                ],
            },
        },
        context_coverage={
            "papers_with_contexts": 8,
            "paper_context_coverage_rate": 0.8,
            "references_with_contexts": 60,
            "reference_context_coverage_rate": 0.6,
            "citation_contexts": 120,
        },
        stability={
            "leave_one_target_out": {
                "average_top_n_retention_rate": 0.8,
                "min_top_n_retention_rate": 0.6,
            },
            "mode_comparison": {
                "rank_weighted": {"top_n_overlap_rate": 0.9, "kendall_tau": 0.7}
            },
        },
        paper_stability={
            "top_n": 100,
            "sample_count": 10,
            "papers": [
                {
                    "paper_id": "r1",
                    "base_rank": 1,
                    "rank_span": 0,
                    "dropped_count": 0,
                    "stability_band": "stable",
                },
                {
                    "paper_id": "r2",
                    "base_rank": 2,
                    "rank_span": 6,
                    "dropped_count": 1,
                    "stability_band": "unstable",
                },
            ],
        },
        paper_score_uncertainty={
            "top_n": 100,
            "samples": 200,
            "papers": [
                {
                    "paper_id": "r1",
                    "base_rank": 1,
                    "score_ci_width": 0.2,
                    "rank_span": 0,
                    "zero_score_rate": 0.0,
                    "uncertainty_band": "stable",
                },
                {
                    "paper_id": "r3",
                    "base_rank": 3,
                    "score_ci_width": 1.5,
                    "rank_span": 8,
                    "zero_score_rate": 0.3,
                    "uncertainty_band": "unstable",
                },
            ],
        },
        scoring_config={
            "version": "score_v2_context_weighted",
            "credit_mode": "context_weighted",
        },
        ranking_preview={
            "papers": [{"rank": 1, "id": "r1", "score": 2.0}],
            "authors": [{"rank": 1, "id": "Author A", "score": 1.0}],
            "institutions": [{"rank": 1, "id": "Institution X", "score": 1.0}],
        },
    )

    assert summary["claim_scope"] == "dataset-local content-aware impact ranking"
    assert summary["data"]["paper_count"] == 10
    assert summary["scoring"]["version"] == "score_v2_context_weighted"
    assert summary["judge_quality"]["parse_success_rate"] == 1.0
    assert summary["proxy_quality"]["unsupported_selection_rate"] == 0.2
    assert summary["anomaly_flags"]["references_missing_author_count"] == 3
    assert summary["stability"]["leave_one_target_out_average_retention_rate"] == 0.8
    assert summary["paper_stability"]["stable_papers"] == 1
    assert summary["paper_stability"]["unstable_papers"] == 1
    assert summary["paper_stability"]["most_unstable_papers"][0]["paper_id"] == "r2"
    assert summary["paper_score_uncertainty"]["stable_papers"] == 1
    assert summary["paper_score_uncertainty"]["unstable_papers"] == 1
    assert summary["paper_score_uncertainty"]["highest_uncertainty_papers"][0]["paper_id"] == "r3"
    assert summary["confidence"]["band"] == "low"
    assert summary["ranking_preview"]["papers"][0]["id"] == "r1"
    assert summary["decision_support"]["ready_for_larger_pilot"] is False


def test_render_mvp_summary_markdown_contains_report_ready_sections():
    summary = {
        "claim_scope": "dataset-local content-aware impact ranking",
        "data": {
            "paper_count": 10,
            "reference_count": 100,
            "ranking_scope": {
                "paper_ranking": "formal",
                "author_ranking": "formal",
                "institution_ranking": "formal",
                "context_aware_ranking": "metadata-only",
            },
            "context_coverage": {
                "papers_with_contexts": 8,
                "paper_context_coverage_rate": 0.8,
                "references_with_contexts": 60,
                "reference_context_coverage_rate": 0.6,
                "citation_contexts": 120,
            },
        },
        "scoring": {
            "version": "score_v2_context_weighted",
            "credit_mode": "context_weighted",
        },
        "ranking_preview": {
            "papers": [{"rank": 1, "id": "r1", "score": 2.0}],
            "authors": [{"rank": 1, "id": "Author A", "score": 1.0}],
            "institutions": [{"rank": 1, "id": "Institution X", "score": 1.0}],
        },
        "judge_quality": {
            "parse_success_rate": 1.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.0,
            "top_k_compliance_rate": 1.0,
        },
        "proxy_quality": {
            "unsupported_selection_rate": 0.2,
            "context_supported_selection_rate": 0.5,
            "method_context_selection_rate": 0.3,
            "bibliometric_top_n_support_rate": 0.6,
        },
        "anomaly_flags": {
            "references_missing_title_count": 0,
            "references_missing_author_count": 0,
            "references_missing_affiliation_count": 0,
            "highest_paper_single_edge_dominance_id": "r1",
            "highest_paper_single_edge_dominance_share": 1.0,
        },
        "stability": {
            "leave_one_target_out_average_retention_rate": 0.8,
            "leave_one_target_out_min_retention_rate": 0.6,
            "mode_comparison": {
                "rank_weighted": {"top_n_overlap_rate": 0.9, "kendall_tau": 0.7},
            },
        },
        "paper_stability": {
            "stable_papers": 8,
            "moderate_papers": 1,
            "unstable_papers": 1,
            "insufficient_sample_papers": 0,
            "most_unstable_papers": [
                {
                    "paper_id": "r2",
                    "base_rank": 2,
                    "rank_span": 6,
                    "dropped_count": 1,
                    "stability_band": "unstable",
                }
            ],
        },
        "paper_score_uncertainty": {
            "stable_papers": 7,
            "moderate_papers": 2,
            "unstable_papers": 1,
            "insufficient_sample_papers": 0,
            "highest_uncertainty_papers": [
                {
                    "paper_id": "r3",
                    "base_rank": 3,
                    "score_ci_width": 1.5,
                    "rank_span": 8,
                    "zero_score_rate": 0.3,
                    "uncertainty_band": "unstable",
                }
            ],
        },
        "confidence": {
            "band": "medium",
            "coverage_band": "high",
            "dominance_risk_band": "low",
            "stability_band": "medium",
            "max_single_edge_share": 0.2,
            "leave_one_target_out_average_retention_rate": 0.8,
            "reasons": ["parse_success_rate=1.000"],
        },
        "decision_support": {
            "ready_for_larger_pilot": True,
            "must_not_claim_global_impact": True,
            "requires_human_labels_for_human_validated_correctness": True,
        },
        "limitations": [
            "This summary aggregates proxy and robustness evidence.",
        ],
    }

    markdown = render_mvp_summary_markdown(summary)

    assert markdown.startswith("# Local Impact Ranking MVP Summary\n")
    assert "| Claim scope | dataset-local content-aware impact ranking |" in markdown
    assert "| Ready for larger pilot | yes |" in markdown
    assert "| parse_success_rate | 1.000 |" in markdown
    assert "## Confidence" in markdown
    assert "| unstable_papers | 1 |" in markdown
    assert "| r2 | 2 | 6 | 1 | unstable |" in markdown
    assert "## Paper Score Uncertainty" in markdown
    assert "| r3 | 3 | 1.500 | 8 | 0.300 | unstable |" in markdown
    assert "| confidence_band | medium |" in markdown
    assert "| 1 | r1 | 2.000 |" in markdown
    assert "| rank_weighted | 0.900 | 0.700 |" in markdown
    assert "This summary aggregates proxy and robustness evidence." in markdown


def test_benchmark_summary_report_collects_report_ready_metrics():
    summary = benchmark_summary_report(
        evaluation={
            "judge_results": 100,
            "parse_success_rate": 1.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.01,
            "top_k_compliance_rate": 1.0,
        },
        gold_evaluation={
            "label_source": "paper.metadata.prescience_positive_reference_ids",
            "labeled_targets": 100,
            "hit_rate": 0.86,
            "selection_precision": 0.36,
            "selection_recall": 0.75,
            "mean_reciprocal_rank": 0.73,
            "mean_ndcg": 0.72,
        },
        gold_baseline_comparison={
            "best_baseline_by_mean_ndcg": "title_overlap",
            "judge_vs_best_baseline": {
                "hit_rate_delta": 0.1,
                "mean_ndcg_delta": 0.2,
                "mean_ndcg_delta_ci95": [0.05, 0.3],
                "statistically_positive_ndcg_delta": True,
            },
            "decision": {"judge_beats_all_baselines": True},
        },
        data_readiness={"paper_count": 100, "reference_count": 3200},
        scoring_config={"credit_mode": "rank_weighted"},
    )

    assert summary["benchmark"]["paper_count"] == 100
    assert summary["judge_quality"]["parse_success_rate"] == 1.0
    assert summary["gold_quality"]["hit_rate"] == 0.86
    assert summary["gold_baseline_comparison"]["judge_beats_all_baselines"] is True
    assert summary["gold_baseline_comparison"]["judge_mean_ndcg_delta_vs_best_baseline"] == 0.2
    assert summary["gold_baseline_comparison"]["judge_ndcg_delta_ci95"] == [0.05, 0.3]
    assert summary["gold_baseline_comparison"]["statistically_positive_ndcg_delta"] is True
    assert summary["decision"]["usable_for_benchmark_report"] is True
    assert summary["decision"]["needs_prompt_or_model_tuning"] is False
    assert summary["limitations"][0].startswith("Gold metrics evaluate")


def test_benchmark_summary_recommends_tuning_when_judge_does_not_beat_baselines():
    summary = benchmark_summary_report(
        evaluation={
            "judge_results": 100,
            "parse_success_rate": 1.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.0,
            "top_k_compliance_rate": 1.0,
        },
        gold_evaluation={
            "label_source": "paper.metadata.prescience_positive_reference_ids",
            "labeled_targets": 100,
            "hit_rate": 0.84,
            "selection_precision": 0.31,
            "selection_recall": 0.65,
            "mean_reciprocal_rank": 0.59,
            "mean_ndcg": 0.60,
        },
        gold_baseline_comparison={
            "best_baseline_by_mean_ndcg": "title_overlap",
            "judge_vs_best_baseline": {
                "hit_rate_delta": -0.01,
                "mean_ndcg_delta": -0.012,
                "mean_ndcg_delta_ci95": [-0.05, 0.02],
                "statistically_positive_ndcg_delta": False,
            },
            "decision": {"judge_beats_all_baselines": False},
        },
        data_readiness={"paper_count": 100, "reference_count": 1145},
        scoring_config={"credit_mode": "rank_weighted"},
    )

    assert summary["decision"]["usable_for_benchmark_report"] is True
    assert summary["decision"]["needs_prompt_or_model_tuning"] is True


def test_benchmark_summary_recommends_tuning_when_baseline_win_is_not_statistically_stable():
    summary = benchmark_summary_report(
        evaluation={
            "judge_results": 100,
            "parse_success_rate": 1.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.0,
            "top_k_compliance_rate": 1.0,
        },
        gold_evaluation={
            "label_source": "paper.metadata.prescience_positive_reference_ids",
            "labeled_targets": 100,
            "hit_rate": 0.89,
            "selection_precision": 0.32,
            "selection_recall": 0.66,
            "mean_reciprocal_rank": 0.66,
            "mean_ndcg": 0.63,
        },
        gold_baseline_comparison={
            "best_baseline_by_mean_ndcg": "title_overlap",
            "judge_vs_best_baseline": {
                "hit_rate_delta": 0.04,
                "mean_ndcg_delta": 0.021592,
                "mean_ndcg_delta_ci95": [-0.023404, 0.064868],
                "statistically_positive_ndcg_delta": False,
            },
            "decision": {"judge_beats_all_baselines": True},
        },
        data_readiness={"paper_count": 100, "reference_count": 1800},
        scoring_config={"credit_mode": "rank_weighted"},
    )

    assert summary["gold_baseline_comparison"]["judge_beats_all_baselines"] is True
    assert summary["gold_baseline_comparison"]["statistically_positive_ndcg_delta"] is False
    assert summary["decision"]["usable_for_benchmark_report"] is True
    assert summary["decision"]["needs_prompt_or_model_tuning"] is True


def test_render_benchmark_summary_markdown_contains_key_tables():
    markdown = render_benchmark_summary_markdown(
        {
            "benchmark": {
                "paper_count": 100,
                "reference_count": 3200,
                "judge_results": 100,
                "credit_mode": "rank_weighted",
            },
            "judge_quality": {
                "parse_success_rate": 1.0,
                "invalid_selection_rate": 0.0,
                "empty_selection_rate": 0.01,
                "top_k_compliance_rate": 1.0,
            },
            "gold_quality": {
                "label_source": "paper.metadata.prescience_positive_reference_ids",
                "labeled_targets": 100,
                "hit_rate": 0.86,
                "selection_precision": 0.36,
                "selection_recall": 0.75,
                "mean_reciprocal_rank": 0.73,
                "mean_ndcg": 0.72,
            },
            "gold_baseline_comparison": {
                "best_baseline_by_mean_ndcg": "title_overlap",
                "judge_mean_ndcg_delta_vs_best_baseline": 0.2,
                "judge_hit_rate_delta_vs_best_baseline": 0.1,
                "judge_ndcg_delta_ci95": [0.05, 0.3],
                "statistically_positive_ndcg_delta": True,
                "judge_beats_all_baselines": True,
            },
            "decision": {
                "usable_for_benchmark_report": True,
                "needs_prompt_or_model_tuning": False,
            },
            "limitations": [
                "Gold metrics evaluate reference selection, not absolute global impact.",
            ],
        }
    )

    assert markdown.startswith("# Benchmark Summary\n")
    assert "| paper_count | 100 |" in markdown
    assert "| parse_success_rate | 1.000 |" in markdown
    assert "| hit_rate | 0.860 |" in markdown
    assert "| judge_beats_all_baselines | yes |" in markdown
    assert "| judge_mean_ndcg_delta_vs_best_baseline | 0.200 |" in markdown
    assert "| statistically_positive_ndcg_delta | yes |" in markdown
    assert "| usable_for_benchmark_report | yes |" in markdown
