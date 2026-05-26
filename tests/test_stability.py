from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper
from local_impact_ranking.stability import stability_report


def test_stability_report_compares_scoring_modes_and_leave_one_target_out():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r2", "title": "Background B"},
                ],
                "citation_contexts": [
                    {
                        "reference_id": "r1",
                        "sentence": "We build on this method.",
                        "section": "Methods",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Target Two",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r3", "title": "Dataset C"},
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Target Three",
                "references": [
                    {"reference_id": "r2", "title": "Background B"},
                    {"reference_id": "r3", "title": "Dataset C"},
                ],
            }
        ),
    ]
    judge_results = [
        JudgeResult(
            target_paper_id="p1",
            selected_references=[
                SelectedReference("r1", rank=1, confidence=1.0, rationale=""),
                SelectedReference("r2", rank=2, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
        JudgeResult(
            target_paper_id="p2",
            selected_references=[
                SelectedReference("r1", rank=1, confidence=1.0, rationale=""),
                SelectedReference("r3", rank=2, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
        JudgeResult(
            target_paper_id="p3",
            selected_references=[
                SelectedReference("r2", rank=1, confidence=1.0, rationale=""),
                SelectedReference("r3", rank=2, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
    ]

    report = stability_report(
        papers,
        judge_results,
        primary_credit_mode="context_weighted",
        comparison_credit_modes=["rank_weighted"],
        top_n=2,
    )

    assert report["primary_credit_mode"] == "context_weighted"
    assert report["top_n"] == 2
    assert report["mode_comparison"]["rank_weighted"]["top_n_overlap_rate"] == 1.0
    assert 0.0 <= report["mode_comparison"]["rank_weighted"]["kendall_tau"] <= 1.0
    assert report["leave_one_target_out"]["sample_count"] == 3
    assert report["leave_one_target_out"]["average_top_n_retention_rate"] < 1.0
    assert "dataset-local" in report["interpretation"]["scope"]
