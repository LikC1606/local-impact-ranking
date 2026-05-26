import csv
import json

from local_impact_ranking.benchmark_compare import (
    benchmark_comparison_rows,
    compare_gold_target_reports,
    render_gold_target_comparison_markdown,
    write_gold_target_comparison,
    render_benchmark_comparison_markdown,
    write_benchmark_comparison,
)


def write_summary(run_dir, *, paper_count, hit_rate, ndcg):
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "benchmark": {
                    "paper_count": paper_count,
                    "reference_count": paper_count * 10,
                    "judge_results": paper_count,
                    "credit_mode": "rank_weighted",
                },
                "judge_quality": {
                    "parse_success_rate": 1.0,
                    "invalid_selection_rate": 0.0,
                    "empty_selection_rate": 0.0,
                    "top_k_compliance_rate": 1.0,
                },
                "gold_quality": {
                    "labeled_targets": paper_count,
                    "hit_rate": hit_rate,
                    "selection_precision": 0.3,
                    "selection_recall": 0.7,
                    "mean_reciprocal_rank": 0.6,
                    "mean_ndcg": ndcg,
                },
                "decision": {
                    "usable_for_benchmark_report": True,
                    "needs_prompt_or_model_tuning": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_benchmark_comparison_rows_flatten_summary_metrics(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_summary(run_a, paper_count=32, hit_rate=0.81, ndcg=0.66)
    write_summary(run_b, paper_count=100, hit_rate=0.86, ndcg=0.72)

    rows = benchmark_comparison_rows([run_b, run_a])

    assert [row["run"] for row in rows] == ["run-a", "run-b"]
    assert rows[0]["paper_count"] == 32
    assert rows[1]["hit_rate"] == 0.86
    assert rows[1]["mean_ndcg"] == 0.72
    assert rows[1]["usable_for_benchmark_report"] is True


def test_write_benchmark_comparison_exports_csv_and_markdown(tmp_path):
    run_dir = tmp_path / "run-a"
    output_dir = tmp_path / "comparison"
    write_summary(run_dir, paper_count=32, hit_rate=0.81, ndcg=0.66)

    write_benchmark_comparison([run_dir], output_dir)

    csv_path = output_dir / "benchmark_comparison.csv"
    md_path = output_dir / "benchmark_comparison.md"
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    markdown = md_path.read_text(encoding="utf-8")
    assert rows[0]["run"] == "run-a"
    assert rows[0]["paper_count"] == "32"
    assert "# Benchmark Comparison" in markdown
    assert "| run-a | 32 | 1.000 | 0.000 | 0.810 | 0.660 | yes |" in markdown


def test_render_benchmark_comparison_markdown_handles_empty_rows():
    markdown = render_benchmark_comparison_markdown([])

    assert markdown.startswith("# Benchmark Comparison\n")
    assert "|  |  |  |  |  |  |  |" in markdown


def test_compare_gold_target_reports_finds_fixed_broken_and_rank_changes():
    base_report = {
        "targets": [
            {
                "target_paper_id": "fixed",
                "positive_reference_ids": ["g1"],
                "selected_reference_ids": ["r1", "r2"],
                "selected_positive_reference_ids": [],
                "hit": False,
                "reciprocal_rank": 0.0,
                "ndcg": 0.0,
            },
            {
                "target_paper_id": "broken",
                "positive_reference_ids": ["g2"],
                "selected_reference_ids": ["g2", "r3"],
                "selected_positive_reference_ids": ["g2"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 1.0,
            },
            {
                "target_paper_id": "ranked",
                "positive_reference_ids": ["g3"],
                "selected_reference_ids": ["r4", "g3", "r5"],
                "selected_positive_reference_ids": ["g3"],
                "hit": True,
                "reciprocal_rank": 0.5,
                "ndcg": 0.63,
            },
            {
                "target_paper_id": "same",
                "positive_reference_ids": ["g4"],
                "selected_reference_ids": ["g4", "r6"],
                "selected_positive_reference_ids": ["g4"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 1.0,
            },
        ],
    }
    candidate_ids = {
        "fixed": ["r1", "g1", "r2"],
        "broken": ["g2", "r3"],
        "ranked": ["r4", "g3", "r5"],
        "same": ["g4", "r6"],
    }
    dpo_report = {
        "targets": [
            {
                "target_paper_id": "fixed",
                "positive_reference_ids": ["g1"],
                "selected_reference_ids": ["g1", "r2"],
                "selected_positive_reference_ids": ["g1"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 1.0,
            },
            {
                "target_paper_id": "broken",
                "positive_reference_ids": ["g2"],
                "selected_reference_ids": ["r3"],
                "selected_positive_reference_ids": [],
                "hit": False,
                "reciprocal_rank": 0.0,
                "ndcg": 0.0,
            },
            {
                "target_paper_id": "ranked",
                "positive_reference_ids": ["g3"],
                "selected_reference_ids": ["g3", "r4", "r5"],
                "selected_positive_reference_ids": ["g3"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 1.0,
            },
            {
                "target_paper_id": "same",
                "positive_reference_ids": ["g4"],
                "selected_reference_ids": ["g4", "r6"],
                "selected_positive_reference_ids": ["g4"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 1.0,
            },
        ],
    }

    report = compare_gold_target_reports(
        base_report,
        dpo_report,
        base_name="base",
        candidate_ids_by_target=candidate_ids,
    )

    assert report["summary"]["common_targets"] == 4
    assert report["summary"]["fixed_count"] == 1
    assert report["summary"]["broken_count"] == 1
    assert report["summary"]["rank_improved_count"] == 1
    assert report["summary"]["hit_quality_changed_count"] == 0
    assert report["summary"]["unchanged_hit_count"] == 1
    assert report["summary"]["net_hit_change"] == 0
    assert report["fixed"][0]["target_paper_id"] == "fixed"
    assert report["fixed"][0]["gold_candidate_positions"] == [2]
    assert report["broken"][0]["target_paper_id"] == "broken"
    assert report["rank_improved"][0]["target_paper_id"] == "ranked"
    assert report["rank_improved"][0]["reciprocal_rank_delta"] == 0.5
    assert report["rank_improved"][0]["selection_jaccard"] == 1.0


def test_compare_gold_target_reports_classifies_ndcg_only_hit_quality_change():
    base_report = {
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["g1", "g2"],
                "selected_reference_ids": ["g1", "r1", "r2"],
                "selected_positive_reference_ids": ["g1"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 0.61,
            }
        ]
    }
    candidate_report = {
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["g1", "g2"],
                "selected_reference_ids": ["g1", "r1", "g2"],
                "selected_positive_reference_ids": ["g1", "g2"],
                "hit": True,
                "reciprocal_rank": 1.0,
                "ndcg": 0.92,
            }
        ]
    }

    report = compare_gold_target_reports(base_report, candidate_report)

    assert report["summary"]["hit_quality_changed_count"] == 1
    assert report["summary"]["unchanged_hit_count"] == 0
    assert report["hit_quality_changed"][0]["target_paper_id"] == "p1"
    assert report["hit_quality_changed"][0]["selected_positive_delta"] == 1


def test_write_gold_target_comparison_exports_json_csv_and_markdown(tmp_path):
    base_dir = tmp_path / "base"
    candidate_path = base_dir / "audit" / "judge_outputs.jsonl"
    dpo_dir = tmp_path / "dpo"
    output_dir = tmp_path / "comparison"
    (base_dir / "reports").mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    (dpo_dir / "reports").mkdir(parents=True)
    (base_dir / "reports" / "gold_evaluation.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["g1"],
                        "selected_reference_ids": ["r1"],
                        "selected_positive_reference_ids": [],
                        "hit": False,
                        "reciprocal_rank": 0.0,
                        "ndcg": 0.0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p1",
                "candidate_reference_ids": ["r1", "g1"],
                "selected_references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dpo_dir / "reports" / "gold_evaluation.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["g1"],
                        "selected_reference_ids": ["g1"],
                        "selected_positive_reference_ids": ["g1"],
                        "hit": True,
                        "reciprocal_rank": 1.0,
                        "ndcg": 1.0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    write_gold_target_comparison(base_dir, dpo_dir, output_dir)

    assert (output_dir / "target_comparison.json").exists()
    rows = list(csv.DictReader((output_dir / "target_changes.csv").read_text().splitlines()))
    assert rows[0]["target_paper_id"] == "p1"
    assert rows[0]["change_type"] == "fixed"
    markdown = (output_dir / "target_comparison.md").read_text(encoding="utf-8")
    assert "# Gold Target Comparison" in markdown
    assert "| fixed_count | 1 |" in markdown


def test_render_gold_target_comparison_markdown_handles_empty_report():
    markdown = render_gold_target_comparison_markdown(
        {"base_run": "base", "candidate_run": "dpo", "summary": {}, "fixed": []}
    )

    assert markdown.startswith("# Gold Target Comparison\n")
    assert "base -> dpo" in markdown
