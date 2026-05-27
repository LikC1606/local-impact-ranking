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


def test_project_readiness_required_artifacts_exclude_self_outputs_only():
    required = required_artifacts_for_readiness()

    assert "reports/artifact_index.json" in required
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
    assert "## Reproducibility" in markdown
    assert "| input_sha256 | input-sha |" in markdown
    assert "| judge_model | local-impact-judge |" in markdown
    assert "## Evidence Files" in markdown
