import json

from local_impact_ranking.artifacts import RUN_ARTIFACTS
from local_impact_ranking.project_readiness import (
    project_readiness_report,
    render_project_readiness_markdown,
    required_artifacts_for_readiness,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_minimal_reportable_run(run_dir):
    for relative_path in RUN_ARTIFACTS:
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        elif path.suffix == ".jsonl":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("placeholder\n", encoding="utf-8")

    _write_json(
        run_dir / "reports" / "validation.json",
        {"summary": {"can_run": True, "issue_count": 0}},
    )
    _write_json(
        run_dir / "reports" / "evaluation.json",
        {
            "judge_results": 10,
            "parse_success_rate": 1.0,
            "error_rate": 0.0,
            "invalid_selection_rate": 0.0,
            "empty_selection_rate": 0.0,
            "duplicate_selection_rate": 0.0,
            "top_k_compliance_rate": 1.0,
        },
    )
    _write_json(
        run_dir / "reports" / "data_readiness.json",
        {
            "paper_count": 10,
            "reference_count": 100,
            "ranking_scope": {
                "paper_ranking": "formal",
                "author_ranking": "formal",
                "institution_ranking": "formal",
                "context_aware_ranking": "formal",
            },
        },
    )
    _write_json(
        run_dir / "reports" / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "ready_for_high_credibility_scoring",
                "can_claim_high_credibility": True,
                "reason": "External evidence coverage is strong.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.9},
            "source_coverage": {
                "openalex_enriched_coverage": 0.8,
                "opencitations_enriched_coverage": 0.8,
            },
            "signal_coverage": {
                "citation_count_coverage": 0.9,
                "field_year_percentile_coverage": 0.8,
            },
            "graph_evidence": {
                "internal_reference_edge_count": 5,
                "materialized_internal_edge_count": 2,
            },
        },
    )
    _write_json(
        run_dir / "reports" / "proxy_evaluation.json",
        {
            "selected_references": 20,
            "unsupported_selection_rate": 0.2,
            "context_supported_selection_rate": 0.6,
            "method_context_selection_rate": 0.3,
            "bibliometric_top_n_support_rate": 0.5,
        },
    )
    _write_json(
        run_dir / "reports" / "stability_report.json",
        {
            "leave_one_target_out": {
                "average_top_n_retention_rate": 0.8,
                "min_top_n_retention_rate": 0.7,
            }
        },
    )
    _write_json(
        run_dir / "reports" / "anomaly_report.json",
        {
            "missing_metadata": {
                "references_missing_title_count": 0,
                "references_missing_author_count": 0,
                "references_missing_affiliation_count": 0,
            },
            "score_dominance": {
                "paper_single_edge_dominance": [
                    {"id": "r1", "max_single_edge_share": 0.3}
                ]
            },
        },
    )
    _write_json(
        run_dir / "reports" / "gold_evaluation.json",
        {
            "labeled_targets": 10,
            "hit_rate": 0.7,
            "mean_ndcg": 0.6,
        },
    )
    _write_json(
        run_dir / "reports" / "gold_candidate_position.json",
        {
            "labeled_targets": 10,
            "candidate_ordering": "deterministic_shuffle",
            "front_position_share": 0.2,
            "mean_best_gold_position": 4.0,
            "mean_gold_position": 5.0,
            "position_bias_risk": "low",
        },
    )
    _write_json(
        run_dir / "reports" / "gold_baseline_comparison.json",
        {
            "best_baseline_by_mean_ndcg": "title_overlap",
            "judge_vs_best_baseline": {
                "hit_rate_delta": 0.1,
                "mean_ndcg_delta": 0.2,
                "mean_ndcg_delta_ci95": [0.05, 0.3],
                "statistically_positive_ndcg_delta": True,
            },
            "decision": {
                "judge_beats_all_baselines": True,
                "needs_model_or_prompt_improvement": False,
            },
        },
    )
    _write_json(
        run_dir / "reports" / "baseline_comparison.json",
        {
            "baselines": {
                "citation_count": {"final_spearman": 0.4},
                "pagerank": {"final_spearman": 0.3},
            }
        },
    )
    _write_json(
        run_dir / "reports" / "impact_validity_evaluation.json",
        {
            "summary": {
                "matched_paper_count": 100,
                "ranking_paper_count": 100,
                "label_paper_count": 100,
                "score_label_match_rate": 1.0,
                "label_match_rate": 1.0,
                "spearman_correlation": 0.52,
                "kendall_tau_b": 0.39,
                "pairwise_accuracy": 0.68,
                "ndcg_at_top_k": 0.81,
                "top_k_precision": 0.7,
                "decision": {
                    "status": "externally_supported",
                    "reason": "Scores align with future labels.",
                },
            },
            "statistical_significance": {
                "status": "statistically_supported",
                "significant_vs_random": True,
                "tested_metric": "pairwise_accuracy",
                "p_value": 0.01,
            },
        },
    )
    _write_json(
        run_dir / "reports" / "citation_cutoff_audit.json",
        {
            "summary": {
                "paper_count": 100,
                "cutoff_year": 2022,
                "papers_with_future_years": 0,
                "future_year_count": 0,
                "citation_cutoff_year_coverage": 1.0,
            },
            "decision": {
                "status": "passed",
                "reason": "No future-year citation leakage was found.",
            },
        },
    )
    (run_dir / "reports" / "citation_cutoff_audit.md").write_text(
        "# Citation Cutoff Audit\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "finetuning" / "finetuning_preflight.json",
        {
            "decision": {
                "recommended_next_step": "dpo_smoke",
                "can_start_sft_smoke": True,
                "can_start_dpo_smoke": True,
                "can_start_full_dpo": False,
            }
        },
    )
    _write_json(
        run_dir / "reports" / "scoring_config.json",
        {"version": "score_v2_context_weighted", "credit_mode": "context_weighted"},
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "created_at": "2026-05-27T00:00:00+00:00",
            "input": {
                "path": "data/papers.jsonl",
                "sha256": "input-sha",
                "normalized_sha256": "normalized-sha",
                "normalized_record_count": 10,
            },
            "run": {
                "top_k": 5,
                "limit": 10,
                "credit_mode": "context_weighted",
                "judge_audit": None,
            },
            "judge": {
                "name": "openai-compatible",
                "model": "local-impact-judge",
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "prompt": {"version": "impact-reference-topk-v1"},
            "scoring": {
                "version": "score_v2_context_weighted",
                "credit_mode": "context_weighted",
            },
            "code": {
                "git_revision": "abc123",
                "git_dirty": True,
            },
        },
    )
    (run_dir / "rankings" / "papers.csv").write_text(
        "rank,id,score,final_score\n1,r1,2.0,100.0\n",
        encoding="utf-8",
    )


def test_project_readiness_report_marks_reportable_run(tmp_path):
    _write_minimal_reportable_run(tmp_path)

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "reportable"
    assert report["decision"]["can_report"] is True
    assert "required_artifacts_present" in report["passed_checks"]
    assert report["failed_checks"] == []
    assert report["claim_scope"] == "dataset-local content-aware impact ranking"
    assert report["quality"]["judge"]["parse_success_rate"] == 1.0
    assert report["quality"]["evidence_level"] == "gold_validated"
    assert report["quality"]["gold_candidate_position"]["position_bias_risk"] == "low"
    assert report["quality"]["gold_baseline_comparison"]["judge_beats_all_baselines"] is True
    assert report["quality"]["external_evidence"]["status"] == "ready_for_high_credibility_scoring"
    assert report["quality"]["external_evidence"]["can_claim_high_credibility"] is True
    assert report["quality"]["impact_validity"]["status"] == "externally_supported"
    assert report["quality"]["impact_validity"]["spearman_correlation"] == 0.52
    assert report["quality"]["impact_validity"]["significant_vs_random"] is True
    assert report["quality"]["impact_validity"]["p_value"] == 0.01
    assert report["quality"]["citation_cutoff_audit"]["status"] == "passed"
    assert report["quality"]["citation_cutoff_audit"]["future_year_count"] == 0
    assert report["quality"]["gold_baseline_comparison"]["mean_ndcg_delta"] == 0.2
    assert report["quality"]["gold_baseline_comparison"]["mean_ndcg_delta_ci95"] == [0.05, 0.3]
    assert report["quality"]["gold_baseline_comparison"]["statistically_positive_ndcg_delta"] is True
    assert report["reproducibility"]["input_sha256"] == "input-sha"
    assert report["reproducibility"]["normalized_sha256"] == "normalized-sha"
    assert report["reproducibility"]["judge_model"] == "local-impact-judge"
    assert report["reproducibility"]["prompt_version"] == "impact-reference-topk-v1"
    assert report["reproducibility"]["git_dirty"] is True
    assert report["evidence_files"]["paper_ranking"] == "rankings/papers.csv"
    assert report["evidence_files"]["gold_candidate_position"] == "reports/gold_candidate_position.json"
    assert report["evidence_files"]["gold_baseline_comparison"] == "reports/gold_baseline_comparison.json"
    assert report["evidence_files"]["external_evidence_readiness"] == "reports/external_evidence_readiness.json"
    assert report["evidence_files"]["external_evidence_paper_coverage"] == "reports/external_evidence_paper_coverage.csv"
    assert report["evidence_files"]["impact_validity_evaluation"] == "reports/impact_validity_evaluation.json"
    assert report["evidence_files"]["citation_cutoff_audit"] == "reports/citation_cutoff_audit.json"


def test_project_readiness_required_artifacts_exclude_self_outputs_only():
    required = required_artifacts_for_readiness()

    assert "reports/artifact_index.json" in required
    assert "reports/impact_validity_evaluation.json" not in required
    assert "reports/impact_validity_evaluation.md" not in required
    assert "reports/citation_cutoff_audit.json" not in required
    assert "reports/citation_cutoff_audit.md" not in required
    assert "reports/project_readiness.json" not in required
    assert "reports/project_readiness.md" not in required


def test_project_readiness_report_blocks_missing_or_invalid_run(tmp_path):
    (tmp_path / "reports").mkdir()
    _write_json(
        tmp_path / "reports" / "validation.json",
        {"summary": {"can_run": False, "issue_count": 2}},
    )
    _write_json(
        tmp_path / "reports" / "evaluation.json",
        {
            "judge_results": 2,
            "parse_success_rate": 0.5,
            "invalid_selection_rate": 0.2,
            "top_k_compliance_rate": 0.5,
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "blocked"
    assert report["decision"]["can_report"] is False
    assert "required_artifacts_present" in report["failed_checks"]
    assert "validation_can_run" in report["failed_checks"]
    assert any("Missing required artifacts" in item for item in report["blockers"])
    assert any("Validation failed" in item for item in report["blockers"])


def test_project_readiness_report_warns_when_gold_labels_are_absent(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "gold_evaluation.json",
        {"labeled_targets": 0, "hit_rate": 0.0, "mean_ndcg": 0.0},
    )
    _write_json(
        tmp_path / "reports" / "proxy_evaluation.json",
        {"selected_references": 10, "unsupported_selection_rate": 0.45},
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert report["decision"]["can_report"] is True
    assert "gold_or_proxy_evidence_available" in report["passed_checks"]
    assert "proxy_support_reasonable" in report["warning_checks"]
    assert report["quality"]["evidence_level"] == "proxy_only_needs_review"
    assert report["recommended_next_actions"][0].startswith("Add benchmark gold labels")


def test_project_readiness_warns_when_external_evidence_is_not_high_credibility(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "usable_with_disclosure",
                "can_claim_high_credibility": False,
                "reason": "External evidence is usable but incomplete.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.45},
            "source_coverage": {
                "openalex_enriched_coverage": 0.2,
                "opencitations_enriched_coverage": 0.1,
            },
            "signal_coverage": {
                "citation_count_coverage": 0.4,
                "field_year_percentile_coverage": 0.1,
            },
            "graph_evidence": {
                "internal_reference_edge_count": 0,
                "materialized_internal_edge_count": 0,
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert report["decision"]["can_report"] is True
    assert "external_evidence_ready_for_high_credibility" in report["warning_checks"]
    assert report["quality"]["external_evidence"]["status"] == "usable_with_disclosure"
    assert report["quality"]["external_evidence"]["external_query_id_coverage"] == 0.45
    assert any("External evidence coverage" in warning for warning in report["warnings"])
    assert any("Run enrich-openalex" in action for action in report["recommended_next_actions"])


def test_project_readiness_warns_when_impact_validity_lacks_cutoff_audit(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    (tmp_path / "reports" / "citation_cutoff_audit.json").unlink()
    (tmp_path / "reports" / "citation_cutoff_audit.md").unlink()

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert "citation_cutoff_audit_available_for_impact_validity" in report["warning_checks"]
    assert report["quality"]["citation_cutoff_audit"]["status"] == "not_available"
    assert any("citation cutoff audit" in warning.lower() for warning in report["warnings"])
    assert any("audit-citation-cutoff" in action for action in report["recommended_next_actions"])


def test_project_readiness_blocks_failed_cutoff_audit_for_impact_validity(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "citation_cutoff_audit.json",
        {
            "summary": {
                "paper_count": 100,
                "cutoff_year": 2022,
                "papers_with_future_years": 3,
                "future_year_count": 8,
                "citation_cutoff_year_coverage": 1.0,
            },
            "decision": {
                "status": "failed",
                "reason": "Citation histories contain years after the requested cutoff.",
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "blocked"
    assert "citation_cutoff_audit_no_future_leakage" in report["failed_checks"]
    assert report["quality"]["citation_cutoff_audit"]["status"] == "failed"
    assert report["quality"]["citation_cutoff_audit"]["future_year_count"] == 8
    assert any("post-cutoff" in blocker for blocker in report["blockers"])


def test_project_readiness_warns_when_impact_validity_is_not_statistically_supported(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "impact_validity_evaluation.json",
        {
            "summary": {
                "matched_paper_count": 100,
                "ranking_paper_count": 100,
                "label_paper_count": 100,
                "score_label_match_rate": 1.0,
                "label_match_rate": 1.0,
                "spearman_correlation": 0.08,
                "kendall_tau_b": 0.04,
                "pairwise_accuracy": 0.51,
                "ndcg_at_top_k": 0.55,
                "top_k_precision": 0.2,
                "decision": {
                    "status": "not_validated",
                    "reason": "Point metrics are not statistically supported.",
                },
            },
            "statistical_significance": {
                "status": "not_statistically_supported",
                "significant_vs_random": False,
                "tested_metric": "pairwise_accuracy",
                "p_value": 0.62,
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert "impact_validity_statistically_supported" in report["warning_checks"]
    assert report["quality"]["impact_validity"]["status"] == "not_validated"
    assert report["quality"]["impact_validity"]["significant_vs_random"] is False
    assert any("random baseline" in warning.lower() for warning in report["warnings"])
    assert any("not claim validated impact accuracy" in action for action in report["recommended_next_actions"])


def test_project_readiness_warns_when_impact_validity_match_coverage_is_low(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "impact_validity_evaluation.json",
        {
            "summary": {
                "matched_paper_count": 25,
                "ranking_paper_count": 100,
                "label_paper_count": 25,
                "score_label_match_rate": 0.25,
                "label_match_rate": 1.0,
                "spearman_correlation": 0.9,
                "kendall_tau_b": 0.8,
                "pairwise_accuracy": 0.9,
                "ndcg_at_top_k": 0.95,
                "top_k_precision": 0.9,
                "decision": {
                    "status": "externally_supported",
                    "reason": "Strong point metrics.",
                },
            },
            "statistical_significance": {
                "status": "statistically_supported",
                "significant_vs_random": True,
                "tested_metric": "pairwise_accuracy",
                "p_value": 0.01,
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert "impact_validity_match_coverage_reasonable" in report["warning_checks"]
    assert report["quality"]["impact_validity"]["score_label_match_rate"] == 0.25
    assert any("coverage" in warning.lower() for warning in report["warnings"])
    assert any("entity matching" in action.lower() for action in report["recommended_next_actions"])


def test_project_readiness_warns_when_self_citation_risk_is_high(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "anomaly_report.json",
        {
            "missing_metadata": {
                "references_missing_title_count": 0,
                "references_missing_author_count": 0,
                "references_missing_affiliation_count": 0,
            },
            "score_dominance": {
                "paper_single_edge_dominance": [
                    {"id": "r1", "max_single_edge_share": 0.3}
                ]
            },
            "self_citation_risk": {
                "risk_level": "high",
                "self_citation_edge_rate": 0.55,
                "self_citation_credit_share": 0.62,
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert "self_citation_risk_not_high" in report["warning_checks"]
    assert report["quality"]["anomaly"]["self_citation_risk_level"] == "high"
    assert report["quality"]["anomaly"]["self_citation_credit_share"] == 0.62


def test_project_readiness_does_not_require_cutoff_audit_without_impact_validity(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    (tmp_path / "reports" / "impact_validity_evaluation.json").unlink()
    (tmp_path / "reports" / "impact_validity_evaluation.md").unlink()
    (tmp_path / "reports" / "citation_cutoff_audit.json").unlink()
    (tmp_path / "reports" / "citation_cutoff_audit.md").unlink()

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "reportable"
    assert "citation_cutoff_audit_available_for_impact_validity" not in report["warning_checks"]
    assert report["quality"]["impact_validity"]["status"] == "not_available"
    assert report["quality"]["citation_cutoff_audit"]["status"] == "not_available"


def test_project_readiness_accepts_proxy_evidence_when_gold_labels_are_absent(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "gold_evaluation.json",
        {"labeled_targets": 0, "hit_rate": 0.0, "mean_ndcg": 0.0},
    )
    _write_json(
        tmp_path / "reports" / "proxy_evaluation.json",
        {
            "selected_references": 10,
            "unsupported_selection_rate": 0.2,
            "context_supported_selection_rate": 0.7,
            "method_context_selection_rate": 0.4,
            "bibliometric_top_n_support_rate": 0.5,
        },
    )

    report = project_readiness_report(tmp_path)

    assert "gold_or_proxy_evidence_available" in report["passed_checks"]
    assert "gold_or_proxy_evidence_available" not in report["warning_checks"]
    assert report["quality"]["evidence_level"] == "proxy_supported"
    assert any(action.startswith("Add benchmark gold labels") for action in report["recommended_next_actions"])


def test_project_readiness_warns_when_gold_baseline_win_is_not_statistically_stable(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    _write_json(
        tmp_path / "reports" / "gold_baseline_comparison.json",
        {
            "best_baseline_by_mean_ndcg": "title_overlap",
            "judge_vs_best_baseline": {
                "hit_rate_delta": 0.04,
                "mean_ndcg_delta": 0.021592,
                "mean_ndcg_delta_ci95": [-0.023404, 0.064868],
                "statistically_positive_ndcg_delta": False,
            },
            "decision": {
                "judge_beats_all_baselines": True,
                "judge_baseline_delta_statistically_positive": False,
                "needs_model_or_prompt_improvement": True,
            },
        },
    )

    report = project_readiness_report(tmp_path)

    assert report["decision"]["status"] == "needs_improvement"
    assert "gold_baseline_delta_statistically_positive" in report["warning_checks"]
    assert report["quality"]["gold_baseline_comparison"]["judge_beats_all_baselines"] is True
    assert report["quality"]["gold_baseline_comparison"]["statistically_positive_ndcg_delta"] is False
    assert any("bootstrap" in warning.lower() for warning in report["warnings"])


def test_render_project_readiness_markdown_contains_decision_and_claim_boundary(tmp_path):
    _write_minimal_reportable_run(tmp_path)
    report = project_readiness_report(tmp_path)

    markdown = render_project_readiness_markdown(report)

    assert markdown.startswith("# Project Readiness\n")
    assert "| status | reportable |" in markdown
    assert "| claim_scope | dataset-local content-aware impact ranking |" in markdown
    assert "## Quality Signals" in markdown
    assert "| evidence_level | gold_validated |" in markdown
    assert "| external_evidence_status | ready_for_high_credibility_scoring |" in markdown
    assert "| impact_validity_status | externally_supported |" in markdown
    assert "| impact_validity_match_rate | 1.000 |" in markdown
    assert "| impact_validity_significant_vs_random | yes |" in markdown
    assert "| impact_validity_p_value | 0.010 |" in markdown
    assert "| citation_cutoff_audit_status | passed |" in markdown
    assert "| citation_cutoff_future_year_count | 0 |" in markdown
    assert "| external_query_id_coverage | 0.900 |" in markdown
    assert "## Reproducibility" in markdown
    assert "| input_sha256 | input-sha |" in markdown
    assert "| judge_model | local-impact-judge |" in markdown
    assert "## Evidence Files" in markdown
    assert "| external_evidence_paper_coverage | `reports/external_evidence_paper_coverage.csv` |" in markdown
    assert "| citation_cutoff_audit | `reports/citation_cutoff_audit.json` |" in markdown
