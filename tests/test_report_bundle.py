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
    _write(run_dir / "reports" / "paper_evidence_report.csv", "paper_id,final_score\np1,100\n")
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
    assert "reports/project_readiness.json" in manifest["copied_files"]
    assert "reports/gold_evaluation.md" in manifest["missing_optional_files"]
    assert manifest["readiness"]["status"] == "needs_improvement"
    assert manifest["benchmark"]["paper_count"] == 300
    assert manifest["claim_scope"] == "dataset-local content-aware impact ranking"
    assert set(manifest["bundle_files"]) == set(REPORT_BUNDLE_FILES)

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "# Local Impact Ranking Report Bundle" in readme
    assert "| readiness_status | needs_improvement |" in readme
    assert "| statistically_positive_ndcg_delta | yes |" in readme
    assert "- Institution ranking is exploratory." in readme


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
