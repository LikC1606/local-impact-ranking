from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper
from local_impact_ranking.stability import (
    bootstrap_paper_score_report,
    export_bootstrap_paper_score_csv,
    export_paper_stability_csv,
    paper_stability_report,
    stability_report,
)


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


def test_paper_stability_report_tracks_leave_one_target_rank_sensitivity(tmp_path):
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r2", "title": "Method B"},
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Target Two",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r2", "title": "Method B"},
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
            ],
            raw_output="{}",
        ),
    ]

    report = paper_stability_report(
        papers,
        judge_results,
        credit_mode="rank_weighted",
        top_n=2,
    )

    assert report["papers"][0]["paper_id"] == "r1"
    assert report["papers"][0]["base_rank"] == 1
    assert report["papers"][0]["worst_rank"] == 1
    assert report["papers"][0]["rank_span"] == 0
    assert report["papers"][1]["paper_id"] == "r2"
    assert report["papers"][1]["dropped_count"] == 1
    assert report["papers"][1]["stability_band"] == "unstable"

    output = tmp_path / "paper_stability_report.csv"
    export_paper_stability_csv(report, output)
    text = output.read_text(encoding="utf-8")
    assert "paper_id,base_rank,base_score,best_rank,worst_rank,rank_span,dropped_count,stability_band" in text
    assert "r2,2,0.800000,2,3,1,1,unstable" in text


def test_bootstrap_paper_score_report_exports_score_and_rank_intervals(tmp_path):
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r2", "title": "Method B"},
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Target Two",
                "references": [
                    {"reference_id": "r1", "title": "Method A"},
                    {"reference_id": "r2", "title": "Method B"},
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Target Three",
                "references": [
                    {"reference_id": "r2", "title": "Method B"},
                    {"reference_id": "r3", "title": "Method C"},
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

    report = bootstrap_paper_score_report(
        papers,
        judge_results,
        credit_mode="rank_weighted",
        top_n=3,
        samples=40,
        seed=7,
    )

    assert report["method"] == "target_bootstrap"
    assert report["samples"] == 40
    assert report["score_name"] == "raw_influence_score"
    assert report["papers"][0]["paper_id"] == "r1"
    assert report["papers"][0]["base_rank"] == 1
    assert report["papers"][0]["score_ci_low"] <= report["papers"][0]["base_score"]
    assert report["papers"][0]["score_ci_high"] >= report["papers"][0]["base_score"]
    assert report["papers"][1]["score_ci_width"] > 0
    assert report["papers"][2]["zero_score_rate"] > 0
    assert report["papers"][2]["uncertainty_band"] in {
        "stable",
        "moderate",
        "unstable",
        "insufficient_samples",
    }

    output = tmp_path / "paper_score_uncertainty_report.csv"
    export_bootstrap_paper_score_csv(report, output)
    text = output.read_text(encoding="utf-8")
    assert (
        "paper_id,base_rank,base_score,score_mean,score_ci_low,score_ci_high,"
        "score_ci_width,best_rank,worst_rank,rank_span,zero_score_rate,uncertainty_band"
    ) in text
    assert "r1,1,2.000000" in text
