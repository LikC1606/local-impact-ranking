import csv
import json

from local_impact_ranking.benchmark_compare import (
    benchmark_comparison_rows,
    compare_gold_target_reports,
    write_model_eval,
    model_selection_decision,
    render_model_selection_decision_markdown,
    render_gold_target_comparison_markdown,
    write_gold_target_comparison,
    write_model_selection_decision,
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


def test_model_selection_decision_recommends_test_when_candidate_improves_safely():
    report = {
        "summary": {
            "common_targets": 100,
            "fixed_count": 8,
            "broken_count": 2,
            "net_hit_change": 6,
            "mean_ndcg_delta": 0.02,
        }
    }

    candidate_summary = {
        "gold_baseline_comparison": {
            "judge_beats_all_baselines": True,
            "judge_mean_ndcg_delta_vs_best_baseline": 0.02,
            "judge_ndcg_delta_ci95": [0.01, 0.04],
            "statistically_positive_ndcg_delta": True,
        }
    }

    decision = model_selection_decision(
        report,
        candidate_benchmark_summary=candidate_summary,
        max_broken_rate=0.05,
    )

    assert decision["recommendation"] == "promote_to_test"
    assert decision["pass"] is True
    assert decision["metrics"]["broken_rate"] == 0.02
    assert decision["metrics"]["judge_beats_all_baselines"] is True
    assert decision["metrics"]["statistically_positive_ndcg_delta"] is True
    assert "net_hit_change_positive" in decision["passed_checks"]
    assert "judge_beats_all_baselines" in decision["passed_checks"]
    assert "judge_baseline_delta_statistically_positive" in decision["passed_checks"]


def test_model_selection_decision_blocks_test_when_breakage_is_high():
    report = {
        "summary": {
            "common_targets": 100,
            "fixed_count": 10,
            "broken_count": 8,
            "net_hit_change": 2,
            "mean_ndcg_delta": 0.01,
        }
    }

    decision = model_selection_decision(report, max_broken_rate=0.05)

    assert decision["recommendation"] == "keep_base_or_iterate"
    assert decision["pass"] is False
    assert "broken_rate_within_threshold" in decision["failed_checks"]


def test_model_selection_decision_blocks_when_candidate_does_not_beat_baselines():
    report = {
        "summary": {
            "common_targets": 100,
            "fixed_count": 8,
            "broken_count": 2,
            "net_hit_change": 6,
            "mean_ndcg_delta": 0.02,
        }
    }
    candidate_summary = {
        "gold_baseline_comparison": {
            "judge_beats_all_baselines": False,
            "judge_mean_ndcg_delta_vs_best_baseline": -0.012,
            "judge_ndcg_delta_ci95": [-0.04, 0.01],
            "statistically_positive_ndcg_delta": False,
        }
    }

    decision = model_selection_decision(
        report,
        candidate_benchmark_summary=candidate_summary,
        max_broken_rate=0.05,
    )

    assert decision["recommendation"] == "keep_base_or_iterate"
    assert decision["pass"] is False
    assert decision["metrics"]["judge_beats_all_baselines"] is False
    assert decision["metrics"]["judge_mean_ndcg_delta_vs_best_baseline"] == -0.012
    assert "judge_beats_all_baselines" in decision["failed_checks"]


def test_model_selection_decision_blocks_when_baseline_win_is_not_statistically_stable():
    report = {
        "summary": {
            "common_targets": 100,
            "fixed_count": 8,
            "broken_count": 2,
            "net_hit_change": 6,
            "mean_ndcg_delta": 0.02,
        }
    }
    candidate_summary = {
        "gold_baseline_comparison": {
            "judge_beats_all_baselines": True,
            "judge_mean_ndcg_delta_vs_best_baseline": 0.021592,
            "judge_ndcg_delta_ci95": [-0.023404, 0.064868],
            "statistically_positive_ndcg_delta": False,
        }
    }

    decision = model_selection_decision(
        report,
        candidate_benchmark_summary=candidate_summary,
        max_broken_rate=0.05,
    )

    assert decision["recommendation"] == "keep_base_or_iterate"
    assert decision["pass"] is False
    assert decision["metrics"]["judge_beats_all_baselines"] is True
    assert decision["metrics"]["statistically_positive_ndcg_delta"] is False
    assert decision["metrics"]["judge_ndcg_delta_ci95"] == [-0.023404, 0.064868]
    assert "judge_beats_all_baselines" in decision["passed_checks"]
    assert "judge_baseline_delta_statistically_positive" in decision["failed_checks"]


def test_write_model_selection_decision_exports_json_and_markdown(tmp_path):
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    (comparison / "target_comparison.json").write_text(
        json.dumps(
            {
                "summary": {
                    "common_targets": 50,
                    "fixed_count": 5,
                    "broken_count": 1,
                    "net_hit_change": 4,
                    "mean_ndcg_delta": 0.03,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    write_model_selection_decision(comparison)

    payload = json.loads((comparison / "model_selection_decision.json").read_text())
    markdown = (comparison / "model_selection_decision.md").read_text(encoding="utf-8")
    assert payload["recommendation"] == "promote_to_test"
    assert "# Model Selection Decision" in markdown
    assert "| recommendation | promote_to_test |" in markdown


def test_write_model_selection_decision_uses_candidate_benchmark_baseline_gate(tmp_path):
    comparison = tmp_path / "comparison"
    candidate = tmp_path / "candidate"
    comparison.mkdir()
    (candidate / "reports").mkdir(parents=True)
    (comparison / "target_comparison.json").write_text(
        json.dumps(
            {
                "candidate_run": "candidate",
                "summary": {
                    "common_targets": 50,
                    "fixed_count": 5,
                    "broken_count": 1,
                    "net_hit_change": 4,
                    "mean_ndcg_delta": 0.03,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate / "reports" / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "gold_baseline_comparison": {
                    "judge_beats_all_baselines": False,
                    "judge_mean_ndcg_delta_vs_best_baseline": -0.01,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    write_model_selection_decision(comparison)

    payload = json.loads((comparison / "model_selection_decision.json").read_text())
    assert payload["recommendation"] == "keep_base_or_iterate"
    assert "judge_beats_all_baselines" in payload["failed_checks"]
    assert payload["metrics"]["judge_mean_ndcg_delta_vs_best_baseline"] == -0.01


def test_write_model_eval_runs_full_promotion_pipeline(tmp_path):
    base_dir = tmp_path / "base"
    candidate_dir = tmp_path / "candidate"
    output_dir = tmp_path / "model-eval"
    write_summary(base_dir, paper_count=2, hit_rate=0.5, ndcg=0.5)
    write_summary(candidate_dir, paper_count=2, hit_rate=1.0, ndcg=0.75)
    (base_dir / "audit").mkdir()
    (base_dir / "audit" / "judge_outputs.jsonl").write_text(
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
    (candidate_dir / "reports" / "gold_evaluation.json").write_text(
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

    report = write_model_eval(base_dir, candidate_dir, output_dir)

    assert report["base_run"] == "base"
    assert report["candidate_run"] == "candidate"
    assert report["decision"]["recommendation"] == "promote_to_test"
    assert (output_dir / "benchmark_comparison.csv").exists()
    assert (output_dir / "target_comparison.json").exists()
    assert (output_dir / "model_selection_decision.json").exists()
    assert (output_dir / "model_eval.json").exists()
    assert "# Model Eval" in (output_dir / "model_eval.md").read_text(encoding="utf-8")


def test_render_model_selection_decision_markdown_lists_failed_checks():
    markdown = render_model_selection_decision_markdown(
        {
            "recommendation": "keep_base_or_iterate",
            "pass": False,
            "metrics": {"net_hit_change": -1, "broken_rate": 0.1},
            "passed_checks": [],
            "failed_checks": ["net_hit_change_positive"],
        }
    )

    assert "| failed_checks | net_hit_change_positive |" in markdown
