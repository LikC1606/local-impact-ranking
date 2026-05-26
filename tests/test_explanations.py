import pytest

from local_impact_ranking.explanations import (
    entity_influence_explanations,
    paper_influence_explanations,
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
