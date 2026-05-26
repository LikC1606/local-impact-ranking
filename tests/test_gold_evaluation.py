from local_impact_ranking.gold_evaluation import (
    evaluate_gold_reference_selection,
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
