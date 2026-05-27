import pytest

from local_impact_ranking.explanations import (
    entity_influence_explanations,
    export_paper_evidence_csv,
    paper_influence_explanations,
    paper_evidence_report,
)
from local_impact_ranking.judge import JudgeResult, SelectedReference
from local_impact_ranking.models import Paper


def test_paper_influence_explanations_group_top_source_edges_by_reference():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {"reference_id": "r1", "title": "Shared Method"},
                    {"reference_id": "r2", "title": "Other Method"},
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
                    {"reference_id": "r1", "title": "Shared Method"},
                    {"reference_id": "r2", "title": "Other Method"},
                ],
            }
        ),
    ]
    judge_results = [
        JudgeResult(
            target_paper_id="p1",
            selected_references=[
                SelectedReference("r1", rank=1, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
        JudgeResult(
            target_paper_id="p2",
            selected_references=[
                SelectedReference("r2", rank=1, confidence=1.0, rationale=""),
                SelectedReference("r1", rank=2, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
    ]

    report = paper_influence_explanations(
        papers,
        judge_results,
        credit_mode="context_weighted",
        top_n=1,
        max_edges_per_paper=2,
    )

    assert report["top_n"] == 1
    assert report["max_edges_per_paper"] == 2
    assert report["papers"][0]["reference_id"] == "r1"
    assert report["papers"][0]["reference_title"] == "Shared Method"
    assert report["papers"][0]["score"] == pytest.approx(2.3)
    assert report["papers"][0]["source_edge_count"] == 2
    assert report["papers"][0]["top_source_edges"] == [
        {
            "target_paper_id": "p1",
            "target_title": "Target One",
            "rank": 1,
            "credit": 1.5,
            "context_count": 1,
        },
        {
            "target_paper_id": "p2",
            "target_title": "Target Two",
            "rank": 2,
            "credit": 0.8,
            "context_count": 0,
        },
    ]


def test_entity_influence_explanations_split_credit_across_authors_and_institutions():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Shared Method",
                        "authors": [
                            {
                                "name": "Author A",
                                "affiliations": ["Institution X", "Institution Y"],
                            },
                            {"name": "Author B", "affiliations": ["Institution X"]},
                        ],
                    }
                ],
            }
        )
    ]
    judge_results = [
        JudgeResult(
            target_paper_id="p1",
            selected_references=[
                SelectedReference("r1", rank=1, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        )
    ]

    report = entity_influence_explanations(
        papers,
        judge_results,
        credit_mode="rank_weighted",
        top_n=5,
        max_edges_per_entity=5,
    )

    assert report["authors"] == [
        {
            "entity_id": "Author A",
            "score": 0.5,
            "source_edge_count": 1,
            "top_source_edges": [
                {
                    "target_paper_id": "p1",
                    "target_title": "Target One",
                    "reference_id": "r1",
                    "reference_title": "Shared Method",
                    "rank": 1,
                    "credit": 0.5,
                    "context_count": 0,
                }
            ],
        },
        {
            "entity_id": "Author B",
            "score": 0.5,
            "source_edge_count": 1,
            "top_source_edges": [
                {
                    "target_paper_id": "p1",
                    "target_title": "Target One",
                    "reference_id": "r1",
                    "reference_title": "Shared Method",
                    "rank": 1,
                    "credit": 0.5,
                    "context_count": 0,
                }
            ],
        },
    ]
    assert report["institutions"] == [
        {
            "entity_id": "Institution X",
            "score": 0.75,
            "source_edge_count": 2,
            "top_source_edges": [
                {
                    "target_paper_id": "p1",
                    "target_title": "Target One",
                    "reference_id": "r1",
                    "reference_title": "Shared Method",
                    "rank": 1,
                    "credit": 0.5,
                    "context_count": 0,
                },
                {
                    "target_paper_id": "p1",
                    "target_title": "Target One",
                    "reference_id": "r1",
                    "reference_title": "Shared Method",
                    "rank": 1,
                    "credit": 0.25,
                    "context_count": 0,
                },
            ],
        },
        {
            "entity_id": "Institution Y",
            "score": 0.25,
            "source_edge_count": 1,
            "top_source_edges": [
                {
                    "target_paper_id": "p1",
                    "target_title": "Target One",
                    "reference_id": "r1",
                    "reference_title": "Shared Method",
                    "rank": 1,
                    "credit": 0.25,
                    "context_count": 0,
                }
            ],
        },
    ]


def test_paper_evidence_report_combines_scores_and_top_edges(tmp_path):
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target One",
                "references": [
                    {"reference_id": "r1", "title": "Shared Method"},
                    {"reference_id": "r2", "title": "Other Method"},
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
                "references": [{"reference_id": "r1", "title": "Shared Method"}],
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
                SelectedReference("r1", rank=2, confidence=1.0, rationale=""),
            ],
            raw_output="{}",
        ),
    ]
    score_details = {
        "r1": {
            "final_score": 90.0,
            "confidence_flag": "stable",
            "max_single_edge_share": 0.65,
        },
        "r2": {
            "final_score": 10.0,
            "confidence_flag": "fragile",
            "max_single_edge_share": 1.0,
        },
    }

    report = paper_evidence_report(
        papers,
        judge_results,
        paper_score_details=score_details,
        paper_score_uncertainty={
            "papers": [
                {
                    "paper_id": "r1",
                    "score_ci_low": 80.0,
                    "score_ci_high": 95.0,
                    "score_ci_width": 15.0,
                    "rank_span": 2,
                    "zero_score_rate": 0.0,
                    "uncertainty_band": "moderate",
                },
                {
                    "paper_id": "r2",
                    "score_ci_low": 0.0,
                    "score_ci_high": 12.0,
                    "score_ci_width": 12.0,
                    "rank_span": 4,
                    "zero_score_rate": 0.25,
                    "uncertainty_band": "unstable",
                },
            ],
        },
        credit_mode="context_weighted",
        top_n=2,
        max_edges_per_paper=2,
    )

    assert report["papers"][0]["paper_id"] == "r1"
    assert report["papers"][0]["final_score"] == 90.0
    assert report["papers"][0]["confidence_flag"] == "stable"
    assert report["papers"][0]["support_band"] == "mixed"
    assert report["papers"][0]["score_ci_low"] == 80.0
    assert report["papers"][0]["score_ci_high"] == 95.0
    assert report["papers"][0]["score_ci_width"] == 15.0
    assert report["papers"][0]["rank_span"] == 2
    assert report["papers"][0]["zero_score_rate"] == 0.0
    assert report["papers"][0]["uncertainty_band"] == "moderate"
    assert report["papers"][0]["top_source_edges"][0] == {
        "target_paper_id": "p1",
        "target_title": "Target One",
        "rank": 1,
        "credit": 1.5,
        "context_count": 1,
        "context_weight": 1.5,
    }

    output = tmp_path / "paper_evidence_report.csv"
    export_paper_evidence_csv(report, output)
    text = output.read_text(encoding="utf-8")
    assert "paper_id,final_score,confidence_flag,support_band,max_single_edge_share,score_ci_low,score_ci_high,score_ci_width,rank_span,zero_score_rate,uncertainty_band" in text
    assert "r1,90.000000,stable,mixed,0.650000,80.000000,95.000000,15.000000,2,0.000000,moderate,p1,1.500000,1,1,1.500000" in text
