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
        "missing_count": 1,
        "artifacts": [
            {
                "path": "reports/evaluation.json",
                "exists": True,
                "size_bytes": 3,
            },
            {
                "path": "rankings/papers.csv",
                "exists": False,
                "size_bytes": 0,
            },
        ],
    }


def test_run_artifacts_include_benchmark_summary_reports():
    from local_impact_ranking.artifacts import RUN_ARTIFACTS

    assert "reports/benchmark_summary.json" in RUN_ARTIFACTS
    assert "reports/benchmark_summary.md" in RUN_ARTIFACTS
    assert "reports/gold_candidate_position.json" in RUN_ARTIFACTS
    assert "reports/gold_candidate_position.md" in RUN_ARTIFACTS
    assert "reports/gold_baseline_comparison.json" in RUN_ARTIFACTS
    assert "reports/gold_baseline_comparison.md" in RUN_ARTIFACTS
