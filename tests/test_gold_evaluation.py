from local_impact_ranking.gold_evaluation import (
    evaluate_gold_reference_selection,
    gold_baseline_comparison_report,
    gold_candidate_position_report,
    render_gold_baseline_comparison_markdown,
    render_gold_candidate_position_markdown,
    render_gold_evaluation_markdown,
)
from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper


def test_evaluate_gold_reference_selection_reports_hit_precision_recall_and_mrr():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {"reference_id": "r1", "title": "Positive"},
                {"reference_id": "r2", "title": "Negative"},
            ],
            "metadata": {
                "prescience_positive_reference_ids": ["r1"],
            },
        }
    )
    result = JudgeResult(
        target_paper_id="p1",
        selected_references=[
            SelectedReference("r2", rank=1, confidence=0.8, rationale=""),
            SelectedReference("r1", rank=2, confidence=0.7, rationale=""),
        ],
        raw_output="{}",
    )

    report = evaluate_gold_reference_selection([paper], [result], top_k=2)

    assert report["labeled_targets"] == 1
    assert report["selection_precision"] == 0.5
    assert report["selection_recall"] == 1.0
    assert report["hit_rate"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.5
    assert report["targets"][0]["selected_positive_reference_ids"] == ["r1"]


def test_evaluate_gold_reference_selection_handles_runs_without_gold_labels():
    report = evaluate_gold_reference_selection(
        [Paper.model_validate({"paper_id": "p1", "title": "No Gold", "references": []})],
        [],
        top_k=5,
    )

    assert report["labeled_targets"] == 0
    assert report["hit_rate"] == 0.0
    assert report["limitations"]


def test_render_gold_evaluation_markdown_contains_report_ready_metrics():
    markdown = render_gold_evaluation_markdown(
        {
            "labeled_targets": 2,
            "hit_rate": 0.5,
            "selection_precision": 0.25,
            "selection_recall": 0.4,
            "mean_reciprocal_rank": 0.3,
            "mean_ndcg": 0.2,
        }
    )

    assert markdown.startswith("# Gold Reference Evaluation\n")
    assert "| hit_rate | 0.500 |" in markdown
    assert "| labeled_targets | 2 |" in markdown


def test_gold_candidate_position_report_flags_front_loaded_gold_candidates():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target 1",
                "references": [
                    {"reference_id": "r1", "title": "Positive"},
                    {"reference_id": "r2", "title": "Negative"},
                ],
                "metadata": {"prescience_positive_reference_ids": ["r1"]},
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Target 2",
                "references": [
                    {"reference_id": "r3", "title": "Positive"},
                    {"reference_id": "r4", "title": "Negative"},
                ],
                "metadata": {"prescience_positive_reference_ids": ["r3"]},
            }
        ),
    ]

    report = gold_candidate_position_report(papers, front_position_threshold=1)

    assert report["labeled_targets"] == 2
    assert report["gold_reference_count"] == 2
    assert report["mean_best_gold_position"] == 1.0
    assert report["front_position_share"] == 1.0
    assert report["position_bias_risk"] == "high"


def test_gold_candidate_position_report_accepts_shuffled_candidates():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {"reference_id": "r2", "title": "Negative"},
                {"reference_id": "r1", "title": "Positive"},
                {"reference_id": "r3", "title": "Negative"},
            ],
            "metadata": {
                "prescience_positive_reference_ids": ["r1"],
                "candidate_ordering": "deterministic_shuffle",
                "candidate_shuffle_seed": 13,
            },
        }
    )

    report = gold_candidate_position_report([paper], front_position_threshold=1)
    markdown = render_gold_candidate_position_markdown(report)

    assert report["candidate_ordering"] == "deterministic_shuffle"
    assert report["position_bias_risk"] == "low"
    assert "| position_bias_risk | low |" in markdown


def test_gold_baseline_comparison_report_compares_judge_against_baselines():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target Learning",
                "abstract": "This paper studies representation learning.",
                "references": [
                    {"reference_id": "r2", "title": "Unrelated Systems"},
                    {"reference_id": "r1", "title": "Representation Learning"},
                    {"reference_id": "r3", "title": "Other Work"},
                ],
                "metadata": {"prescience_positive_reference_ids": ["r1"]},
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Target Graph",
                "abstract": "Graph neural network methods.",
                "references": [
                    {"reference_id": "r4", "title": "Graph Neural Networks"},
                    {"reference_id": "r5", "title": "Dataset Paper"},
                ],
                "metadata": {"prescience_positive_reference_ids": ["r4"]},
            }
        ),
    ]
    judge_results = [
        JudgeResult(
            target_paper_id="p1",
            selected_references=[
                SelectedReference("r1", rank=1, confidence=0.9, rationale="")
            ],
            raw_output="{}",
        ),
        JudgeResult(
            target_paper_id="p2",
            selected_references=[
                SelectedReference("r5", rank=1, confidence=0.8, rationale="")
            ],
            raw_output="{}",
        ),
    ]

    report = gold_baseline_comparison_report(
        papers,
        judge_results,
        top_k=1,
        random_seed=1,
    )

    assert report["top_k"] == 1
    assert [row["method"] for row in report["methods"]] == [
        "judge",
        "candidate_order",
        "title_overlap",
        "random",
    ]
    assert report["methods"][0]["hit_rate"] == 0.5
    assert report["methods"][1]["hit_rate"] == 0.5
    assert report["methods"][2]["hit_rate"] == 1.0
    assert report["best_method_by_mean_ndcg"] == "title_overlap"
    assert report["judge_vs_best_baseline"]["mean_ndcg_delta"] < 0
    assert "mean_ndcg_delta_ci95" in report["judge_vs_best_baseline"]
    assert report["judge_vs_best_baseline"]["paired_bootstrap_samples"] == 2000
    assert report["judge_vs_best_baseline"]["statistically_positive_ndcg_delta"] is False
    assert report["targets"][0]["target_paper_id"] == "p1"
    assert "judge_ndcg" in report["targets"][0]
    assert "best_baseline_ndcg" in report["targets"][0]
    assert report["decision"]["judge_beats_all_baselines"] is False
    assert report["decision"]["judge_baseline_delta_statistically_positive"] is False
    assert report["decision"]["needs_model_or_prompt_improvement"] is True


def test_render_gold_baseline_comparison_markdown_contains_method_table():
    markdown = render_gold_baseline_comparison_markdown(
        {
            "methods": [
                {
                    "method": "judge",
                    "labeled_targets": 2,
                    "hit_rate": 0.5,
                    "selection_precision": 0.5,
                    "selection_recall": 0.5,
                    "mean_reciprocal_rank": 0.5,
                    "mean_ndcg": 0.5,
                }
            ],
            "best_method_by_mean_ndcg": "judge",
            "judge_vs_best_baseline": {
                "mean_ndcg_delta": 0.1,
                "mean_ndcg_delta_ci95": [0.01, 0.2],
                "statistically_positive_ndcg_delta": True,
            },
            "decision": {"judge_beats_all_baselines": True},
            "limitations": ["Random baseline is seed dependent."],
        }
    )

    assert markdown.startswith("# Gold Baseline Comparison\n")
    assert "| judge | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |" in markdown
    assert "| judge_beats_all_baselines | yes |" in markdown
    assert "| judge_ndcg_delta_ci95 | [0.010, 0.200] |" in markdown
    assert "| judge_baseline_delta_statistically_positive | yes |" in markdown
