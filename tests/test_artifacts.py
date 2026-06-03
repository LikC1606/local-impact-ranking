from local_impact_ranking.artifacts import artifact_index


def test_artifact_index_reports_existing_and_missing_files(tmp_path):
    existing = tmp_path / "reports" / "evaluation.json"
    existing.parent.mkdir()
    existing.write_text("{}\n", encoding="utf-8")

    index = artifact_index(
        tmp_path,
        ["reports/evaluation.json", "rankings/papers.csv"],
    )

    assert index == {
        "artifact_count": 2,
        "required_artifact_count": 2,
        "optional_artifact_count": 0,
        "missing_count": 1,
        "missing_required_count": 1,
        "missing_optional_count": 0,
        "artifacts": [
            {
                "path": "reports/evaluation.json",
                "exists": True,
                "size_bytes": 3,
                "required": True,
            },
            {
                "path": "rankings/papers.csv",
                "exists": False,
                "size_bytes": 0,
                "required": True,
            },
        ],
    }


def test_artifact_index_excludes_missing_optional_files_from_missing_count(tmp_path):
    index = artifact_index(
        tmp_path,
        [
            "reports/evaluation.json",
            "reports/impact_validity_evaluation.json",
            "reports/citation_cutoff_audit.json",
        ],
        optional_paths={
            "reports/impact_validity_evaluation.json",
            "reports/citation_cutoff_audit.json",
        },
    )

    assert index["missing_count"] == 1
    assert index["missing_required_count"] == 1
    assert index["missing_optional_count"] == 2
    by_path = {artifact["path"]: artifact for artifact in index["artifacts"]}
    assert by_path["reports/evaluation.json"]["required"] is True
    assert by_path["reports/impact_validity_evaluation.json"]["required"] is False
    assert by_path["reports/citation_cutoff_audit.json"]["required"] is False


def test_run_artifacts_include_benchmark_summary_reports():
    from local_impact_ranking.artifacts import RUN_ARTIFACTS

    assert "reports/benchmark_summary.json" in RUN_ARTIFACTS
    assert "reports/benchmark_summary.md" in RUN_ARTIFACTS
    assert "reports/gold_candidate_position.json" in RUN_ARTIFACTS
    assert "reports/gold_candidate_position.md" in RUN_ARTIFACTS
    assert "reports/gold_baseline_comparison.json" in RUN_ARTIFACTS
    assert "reports/gold_baseline_comparison.md" in RUN_ARTIFACTS
    assert "reports/external_evidence_readiness.json" in RUN_ARTIFACTS
    assert "reports/external_evidence_readiness.md" in RUN_ARTIFACTS
    assert "reports/external_evidence_paper_coverage.csv" in RUN_ARTIFACTS
    assert "reports/citation_completeness_audit.json" in RUN_ARTIFACTS
    assert "reports/citation_completeness_audit.md" in RUN_ARTIFACTS
    assert "reports/citation_completeness_papers.csv" in RUN_ARTIFACTS
    assert "reports/impact_validity_evaluation.json" in RUN_ARTIFACTS
    assert "reports/impact_validity_evaluation.md" in RUN_ARTIFACTS
    assert "reports/impact_credibility_scorecard.json" in RUN_ARTIFACTS
    assert "reports/impact_credibility_scorecard.md" in RUN_ARTIFACTS
    assert "reports/citation_cutoff_audit.json" in RUN_ARTIFACTS
    assert "reports/citation_cutoff_audit.md" in RUN_ARTIFACTS
    assert "reports/score_cards.json" in RUN_ARTIFACTS
    assert "reports/score_cards.md" in RUN_ARTIFACTS
    assert "reports/entity_resolution_report.json" in RUN_ARTIFACTS
    assert "reports/entity_resolution_report.md" in RUN_ARTIFACTS
