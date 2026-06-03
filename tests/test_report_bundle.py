import json

from local_impact_ranking.report_bundle import (
    REPORT_BUNDLE_FILES,
    render_report_bundle_readme,
    write_report_bundle,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_write_report_bundle_copies_report_ready_artifacts_and_manifest(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "bundle"
    _write(run_dir / "rankings" / "papers.csv", "rank,id,final_score\n1,p1,100\n")
    _write(run_dir / "rankings" / "authors.csv", "rank,id,final_score\n1,a1,100\n")
    _write(run_dir / "reports" / "mvp_summary.md", "# MVP Summary\n")
    _write(run_dir / "reports" / "benchmark_summary.md", "# Benchmark Summary\n")
    _write(run_dir / "reports" / "project_readiness.md", "# Project Readiness\n")
    _write(run_dir / "reports" / "external_evidence_readiness.md", "# External Evidence Readiness\n")
    _write(run_dir / "reports" / "external_evidence_paper_coverage.csv", "paper_id,missing_evidence\np1,\n")
    _write(run_dir / "reports" / "citation_completeness_audit.md", "# Citation Completeness Audit\n")
    _write(run_dir / "reports" / "citation_completeness_audit.json", '{"summary":{"paper_count":1}}\n')
    _write(run_dir / "reports" / "citation_completeness_papers.csv", "paper_id,completeness_grade\np1,multi_source_compatible\n")
    _write(run_dir / "reports" / "impact_validity_evaluation.md", "# Impact Validity Evaluation\n")
    _write(run_dir / "reports" / "impact_credibility_scorecard.md", "# Impact Credibility Scorecard\n")
    _write(run_dir / "reports" / "citation_cutoff_audit.md", "# Citation Cutoff Audit\n")
    _write(run_dir / "reports" / "citation_cutoff_audit.json", '{"decision":{"status":"passed"}}\n')
    _write(
        run_dir / "reports" / "impact_claim_certificate.json",
        '{"claim":{"status":"evidence_supported","can_claim_validated_impact_accuracy":false,"can_claim_high_credibility":false,"confidence":0.25}}\n',
    )
    _write(
        run_dir / "reports" / "impact_claim_certificate.md",
        "# Impact Claim Certificate\n",
    )
    _write(run_dir / "reports" / "paper_evidence_report.csv", "paper_id,final_score\np1,100\n")
    _write(run_dir / "reports" / "score_cards.md", "# Paper Score Cards\n")
    _write(run_dir / "reports" / "entity_resolution_report.md", "# Entity Resolution Report\n")
    _write(run_dir / "reports" / "scoring_config.json", '{"version":"score_v2"}\n')
    _write(run_dir / "run_manifest.json", '{"created_at":"2026-05-27T00:00:00+00:00"}\n')
    _write(
        run_dir / "reports" / "project_readiness.json",
        json.dumps(
            {
                "claim_scope": "dataset-local content-aware impact ranking",
                "decision": {
                    "status": "needs_improvement",
                    "can_report": True,
                    "reason": "Warnings must be disclosed.",
                },
                "quality": {"evidence_level": "gold_validated"},
                "warnings": ["Institution ranking is exploratory."],
            }
        )
        + "\n",
    )
    _write(
        run_dir / "reports" / "benchmark_summary.json",
        json.dumps(
            {
                "benchmark": {"paper_count": 300},
                "gold_quality": {"hit_rate": 0.83, "mean_ndcg": 0.59},
                "gold_baseline_comparison": {
                    "judge_mean_ndcg_delta_vs_best_baseline": 0.028,
                    "judge_ndcg_delta_ci95": [0.004, 0.052],
                    "statistically_positive_ndcg_delta": True,
                },
                "decision": {"needs_prompt_or_model_tuning": False},
            }
        )
        + "\n",
    )

    manifest = write_report_bundle(run_dir, output_dir)

    assert (output_dir / "README.md").exists()
    assert (output_dir / "bundle_manifest.json").exists()
    assert (output_dir / "rankings" / "papers.csv").read_text() == "rank,id,final_score\n1,p1,100\n"
    assert (output_dir / "reports" / "project_readiness.md").exists()
    assert (output_dir / "reports" / "external_evidence_readiness.md").exists()
    assert (output_dir / "reports" / "external_evidence_paper_coverage.csv").exists()
    assert (output_dir / "reports" / "citation_completeness_audit.md").exists()
    assert (output_dir / "reports" / "citation_completeness_audit.json").exists()
    assert (output_dir / "reports" / "citation_completeness_papers.csv").exists()
    assert (output_dir / "reports" / "impact_validity_evaluation.md").exists()
    assert (output_dir / "reports" / "impact_credibility_scorecard.md").exists()
    assert (output_dir / "reports" / "citation_cutoff_audit.md").exists()
    assert (output_dir / "reports" / "citation_cutoff_audit.json").exists()
    assert (output_dir / "reports" / "impact_claim_certificate.md").exists()
    assert (output_dir / "reports" / "score_cards.md").exists()
    assert (output_dir / "reports" / "entity_resolution_report.md").exists()
    assert "reports/project_readiness.json" in manifest["copied_files"]
    assert "reports/external_evidence_readiness.md" in manifest["copied_files"]
    assert "reports/external_evidence_paper_coverage.csv" in manifest["copied_files"]
    assert "reports/citation_completeness_audit.md" in manifest["copied_files"]
    assert "reports/citation_completeness_audit.json" in manifest["copied_files"]
    assert "reports/citation_completeness_papers.csv" in manifest["copied_files"]
    assert "reports/impact_validity_evaluation.md" in manifest["copied_files"]
    assert "reports/impact_credibility_scorecard.md" in manifest["copied_files"]
    assert "reports/citation_cutoff_audit.md" in manifest["copied_files"]
    assert "reports/impact_claim_certificate.md" in manifest["copied_files"]
    assert "reports/gold_evaluation.md" in manifest["missing_optional_files"]
    assert manifest["readiness"]["status"] == "needs_improvement"
    assert manifest["impact_claim"]["status"] == "evidence_supported"
    assert manifest["benchmark"]["paper_count"] == 300
    assert manifest["claim_scope"] == "dataset-local content-aware impact ranking"
    assert set(manifest["bundle_files"]) == set(REPORT_BUNDLE_FILES)

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "# Local Impact Ranking Report Bundle" in readme
    assert "| readiness_status | needs_improvement |" in readme
    assert "| impact_claim_status | evidence_supported |" in readme
    assert "| statistically_positive_ndcg_delta | yes |" in readme
    assert "- Institution ranking is exploratory." in readme


def test_write_report_bundle_supports_model_score_dataset_outputs(tmp_path):
    run_dir = tmp_path / "model-impact"
    output_dir = tmp_path / "bundle"
    _write(
        run_dir / "impact_scores.csv",
        (
            "paper_id,title,final_impact_score,validity_status,validity_confidence\n"
            "p1,Paper One,88.5,unvalidated,0.25\n"
        ),
    )
    _write(run_dir / "impact_scores.jsonl", '{"paper_id":"p1"}\n')
    _write(run_dir / "pairwise_judgments.jsonl", '{"paper_a_id":"p1","paper_b_id":"p2"}\n')
    _write(run_dir / "model_impact_report.md", "# Model Impact Report\n")
    _write(
        run_dir / "model_impact_report.json",
        json.dumps(
            {
                "summary": {
                    "paper_count": 4,
                    "judgment_count": 6,
                    "parseable_rate": 1.0,
                    "scoring_method": "impact_index_v3_evidence_first",
                    "ranking_quality": {"status": "usable"},
                    "validated_impact_claim": {
                        "status": "unvalidated",
                        "can_claim_validated_impact_accuracy": False,
                    },
                }
            }
        )
        + "\n",
    )
    _write(run_dir / "impact_score_cards.md", "# Impact Score Cards\n")
    _write(run_dir / "impact_score_cards.json", "[]\n")
    _write(run_dir / "external_evidence_readiness.md", "# External Evidence Readiness\n")
    _write(
        run_dir / "external_evidence_readiness.json",
        '{"decision":{"status":"usable_with_disclosure","can_claim_high_credibility":false}}\n',
    )
    _write(run_dir / "external_evidence_paper_coverage.csv", "paper_id,missing_evidence\np1,\n")
    _write(
        run_dir / "impact_claim_certificate.json",
        '{"claim":{"status":"evidence_supported","can_claim_validated_impact_accuracy":false,"can_claim_high_credibility":false,"confidence":0.25}}\n',
    )
    _write(run_dir / "impact_claim_certificate.md", "# Impact Claim Certificate\n")
    _write(
        run_dir / "reports" / "project_readiness.json",
        '{"decision":{"status":"blocked","can_report":false,"reason":"Full pipeline artifacts are missing."},"quality":{"evidence_level":"no_external_or_proxy_evidence"}}\n',
    )
    _write(
        run_dir / "reports" / "impact_claim_certificate.json",
        '{"claim":{"status":"blocked","can_claim_validated_impact_accuracy":false,"can_claim_high_credibility":false,"confidence":0.0}}\n',
    )
    _write(
        run_dir / "model_credibility_card.json",
        json.dumps(
            {
                "decision": {
                    "status": "moderate_credibility",
                    "grade": "B",
                    "model_identity_match": True,
                    "profile_model_names": ["local-impact-judge-source-task-dpo"],
                    "can_use_for_report_with_disclosure": True,
                    "can_claim_high_model_credibility": False,
                },
                "served_model": "local-impact-judge-source-task-dpo",
                "summary": {
                    "pooled_accuracy": 0.658889,
                    "min_seed_accuracy": 0.646667,
                    "total_examples": 900,
                },
            }
        )
        + "\n",
    )
    _write(run_dir / "model_credibility_card.md", "# Model Credibility Card\n")
    _write(
        run_dir / "calibration_profile_card.json",
        json.dumps(
            {
                "decision": {
                    "status": "calibration_supported",
                    "model_identity_match": True,
                    "compatible_models": ["local-impact-judge-source-task-dpo"],
                    "can_use_calibrated_confidence": True,
                },
                "served_model": "local-impact-judge-source-task-dpo",
                "summary": {
                    "scored_examples": 150,
                    "ece": 0.086021,
                    "mce": 0.160367,
                    "brier_score": 0.221465,
                },
            }
        )
        + "\n",
    )
    _write(run_dir / "calibration_profile_card.md", "# Calibration Profile Card\n")
    _write(
        run_dir / "impact_validity" / "impact_credibility_scorecard.md",
        "# Impact Credibility Scorecard\n",
    )
    _write(
        run_dir / "impact_validity" / "impact_validity_evaluation.json",
        json.dumps(
            {
                "credibility_scorecard": {
                    "status": "moderate_credibility",
                    "grade": "B",
                }
            }
        )
        + "\n",
    )

    manifest = write_report_bundle(run_dir, output_dir)

    assert (output_dir / "impact_scores.csv").exists()
    assert (output_dir / "model_impact_report.md").exists()
    assert (output_dir / "impact_score_cards.md").exists()
    assert (output_dir / "pairwise_judgments.jsonl").exists()
    assert (output_dir / "external_evidence_readiness.md").exists()
    assert (output_dir / "impact_claim_certificate.md").exists()
    assert (output_dir / "model_credibility_card.md").exists()
    assert (output_dir / "calibration_profile_card.md").exists()
    assert (output_dir / "impact_validity" / "impact_credibility_scorecard.md").exists()
    assert "impact_scores.csv" in manifest["copied_files"]
    assert "model_impact_report.md" in manifest["copied_files"]
    assert "model_credibility_card.md" in manifest["copied_files"]
    assert "calibration_profile_card.md" in manifest["copied_files"]
    assert manifest["model_score"]["paper_count"] == 4
    assert manifest["model_score"]["judgment_count"] == 6
    assert manifest["model_score"]["scoring_method"] == "impact_index_v3_evidence_first"
    assert manifest["model_score"]["impact_credibility_status"] == "moderate_credibility"
    assert manifest["impact_claim"]["status"] == "evidence_supported"
    assert manifest["bundle_decision"]["status"] == "evidence_supported"
    assert manifest["bundle_decision"]["basis"] == "model_score_dataset"
    assert manifest["model_credibility"]["status"] == "moderate_credibility"
    assert manifest["model_credibility"]["served_model"] == "local-impact-judge-source-task-dpo"
    assert manifest["model_credibility"]["model_identity_match"] is True
    assert manifest["model_credibility"]["can_use_for_report_with_disclosure"] is True
    assert manifest["calibration_profile"]["status"] == "calibration_supported"
    assert manifest["calibration_profile"]["can_use_calibrated_confidence"] is True

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## Model Score Dataset" in readme
    assert "| model_paper_count | 4 |" in readme
    assert "| model_scoring_method | impact_index_v3_evidence_first |" in readme
    assert "| impact_credibility_status | moderate_credibility |" in readme
    assert "| readiness_status | evidence_supported |" in readme
    assert "model_score_dataset" in readme
    assert "| impact_claim_status | evidence_supported |" in readme
    assert "## Model Credibility" in readme
    assert "| model_credibility_status | moderate_credibility |" in readme
    assert "| served_model | local-impact-judge-source-task-dpo |" in readme
    assert "| model_identity_match | yes |" in readme
    assert "| model_can_use_for_report_with_disclosure | yes |" in readme
    assert "## Calibration Profile" in readme
    assert "| calibration_status | calibration_supported |" in readme
    assert "| can_use_calibrated_confidence | yes |" in readme


def test_render_report_bundle_readme_keeps_claim_boundary_when_reports_are_missing():
    readme = render_report_bundle_readme(
        {
            "run_dir": "runs/example",
            "claim_scope": "dataset-local impact ranking",
            "readiness": {"status": "blocked", "can_report": False},
            "benchmark": {},
            "warnings": [],
            "copied_files": [],
            "missing_optional_files": ["reports/mvp_summary.md"],
        }
    )

    assert "dataset-local impact ranking" in readme
    assert "| readiness_status | blocked |" in readme
    assert "- `reports/mvp_summary.md`" in readme
