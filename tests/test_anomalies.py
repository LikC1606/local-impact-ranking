from local_impact_ranking.anomalies import anomaly_report
from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import score_influence


def test_anomaly_report_flags_selected_reference_self_citation_risk():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "target",
                "title": "Target Paper",
                "authors": [{"name": "Alice Smith"}, {"name": "Bob Chen"}],
                "references": [
                    {
                        "reference_id": "self_ref",
                        "title": "Prior Work By Alice",
                        "authors": [{"name": "Alice Smith"}],
                    },
                    {
                        "reference_id": "other_ref",
                        "title": "Independent Work",
                        "authors": [{"name": "Dana Lee"}],
                    },
                ],
            }
        )
    ]
    judge_results = [
        JudgeResult(
            target_paper_id="target",
            selected_references=[
                SelectedReference(
                    reference_id="self_ref",
                    rank=1,
                    confidence=0.9,
                    rationale="Used directly.",
                ),
                SelectedReference(
                    reference_id="other_ref",
                    rank=2,
                    confidence=0.8,
                    rationale="Independent baseline.",
                ),
            ],
            raw_output="{}",
        )
    ]
    scores = score_influence(papers, judge_results, credit_mode="rank_weighted")

    report = anomaly_report(
        papers,
        judge_results,
        scores,
        credit_mode="rank_weighted",
    )

    risk = report["self_citation_risk"]
    assert risk["selected_edge_count"] == 2
    assert risk["self_citation_edge_count"] == 1
    assert risk["self_citation_edge_rate"] == 0.5
    assert risk["risk_level"] == "high"
    assert risk["top_self_cited_references"][0]["reference_id"] == "self_ref"
    assert risk["top_self_cited_references"][0]["overlapping_authors"] == [
        "Alice Smith"
    ]
