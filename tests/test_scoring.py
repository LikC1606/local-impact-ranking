import json
import math

from local_impact_ranking.judge import parse_judge_output
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import (
    expanded_rank_weight,
    scoring_edges,
    score_influence,
)


def sample_paper():
    return Paper.model_validate(
        {
            "paper_id": "target",
            "title": "Target",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "Influential",
                    "year": 2020,
                    "field": "Machine Learning",
                    "authors": [
                        {"name": "Author A", "affiliations": ["Institution X"]},
                        {"name": "Author B", "affiliations": ["Institution Y"]},
                    ],
                },
                {
                    "reference_id": "r2",
                    "title": "Second",
                    "year": 2021,
                    "field": "Machine Learning",
                    "authors": [
                        {"name": "Author C", "affiliations": ["Institution X", "Institution Z"]}
                    ],
                },
            ],
        }
    )


def judge_result(paper):
    return parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "target",
                "selected_references": [
                    {"reference_id": "r1", "rank": 1, "confidence": 0.9, "rationale": ""},
                    {"reference_id": "r2", "rank": 2, "confidence": 0.8, "rationale": ""},
                    {"reference_id": "bad", "rank": 3, "confidence": 0.1, "rationale": ""},
                ],
            }
        ),
        paper,
    )


def test_rank_weighted_credit_splits_across_authors_and_institutions():
    paper = sample_paper()
    scores = score_influence([paper], [judge_result(paper)], credit_mode="rank_weighted")

    assert scores.paper_scores["r1"] == 1.0
    assert scores.paper_scores["r2"] == 0.8
    assert scores.author_scores["Author A"] == 0.5
    assert scores.author_scores["Author B"] == 0.5
    assert scores.author_scores["Author C"] == 0.8
    assert scores.institution_scores["Institution X"] == 0.9
    assert scores.institution_scores["Institution Y"] == 0.5
    assert scores.institution_scores["Institution Z"] == 0.4
    assert scores.field_scores["Machine Learning"] == 1.8
    assert scores.year_scores["2020"] == 1.0
    assert scores.year_scores["2021"] == 0.8


def test_flat_credit_mode_assigns_one_point_per_valid_selection():
    paper = sample_paper()
    scores = score_influence([paper], [judge_result(paper)], credit_mode="flat")

    assert scores.paper_scores["r1"] == 1.0
    assert scores.paper_scores["r2"] == 1.0


def test_expanded_rank_weight_uses_log_discount_past_top_five():
    assert expanded_rank_weight(1) == 1.0
    assert expanded_rank_weight(5) == 1 / math.log2(6)
    assert expanded_rank_weight(20) == 1 / math.log2(21)
    assert expanded_rank_weight(21) == 0.0


def test_score_influence_tracks_core_and_expanded_paper_scores():
    paper = Paper.model_validate(
        {
            "paper_id": "target",
            "title": "Target",
            "references": [
                {"reference_id": f"r{i}", "title": f"Reference {i}"}
                for i in range(1, 7)
            ],
        }
    )
    result = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "target",
                "selected_references": [
                    {
                        "reference_id": f"r{i}",
                        "rank": i,
                        "confidence": 0.9,
                        "rationale": "",
                    }
                    for i in range(1, 7)
                ],
            }
        ),
        paper,
    )

    scores = score_influence([paper], [result], credit_mode="rank_weighted")

    assert scores.paper_scores["r1"] == 1.0
    assert scores.paper_scores["r5"] == 0.2
    assert "r6" not in scores.paper_scores
    assert scores.paper_score_details["r1"]["core_score"] == 1.0
    assert scores.paper_score_details["r1"]["expanded_score"] == 1.0
    assert scores.paper_score_details["r5"]["core_score"] == 0.2
    assert scores.paper_score_details["r5"]["expanded_score"] == 1 / math.log2(6)
    assert scores.paper_score_details["r6"]["core_score"] == 0.0
    assert scores.paper_score_details["r6"]["expanded_score"] == 1 / math.log2(7)


def test_context_weighted_credit_rewards_context_supported_method_references():
    paper = Paper.model_validate(
        {
            "paper_id": "target",
            "title": "Target",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "Method Reference",
                    "authors": [{"name": "Author A", "affiliations": ["Institution X"]}],
                },
                {
                    "reference_id": "r2",
                    "title": "Background Reference",
                    "authors": [{"name": "Author B", "affiliations": ["Institution Y"]}],
                },
            ],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We build on this method for our model.",
                    "section": "Methods",
                },
                {
                    "reference_id": "r2",
                    "sentence": "This area has been studied before.",
                    "section": "Introduction",
                },
            ],
        }
    )

    scores = score_influence([paper], [judge_result(paper)], credit_mode="context_weighted")

    assert scores.paper_scores["r1"] == 1.5
    assert scores.paper_scores["r2"] == 0.96


def test_scoring_edges_explain_each_valid_selection_credit():
    paper = Paper.model_validate(
        {
            "paper_id": "target",
            "title": "Target",
            "references": [
                {"reference_id": "r1", "title": "Method Reference"},
                {"reference_id": "r2", "title": "Background Reference"},
            ],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We build on this method for our model.",
                    "section": "Methods",
                }
            ],
        }
    )

    edges = scoring_edges([paper], [judge_result(paper)], credit_mode="context_weighted")

    assert edges == [
        {
            "target_paper_id": "target",
            "reference_id": "r1",
            "rank": 1,
            "rank_weight": 1.0,
            "context_count": 1,
            "context_weight": 1.5,
            "credit": 1.5,
        },
        {
            "target_paper_id": "target",
            "reference_id": "r2",
            "rank": 2,
            "rank_weight": 0.8,
            "context_count": 0,
            "context_weight": 1.0,
            "credit": 0.8,
        },
    ]
