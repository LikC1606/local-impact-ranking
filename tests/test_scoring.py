import json
import math

from local_impact_ranking.judge import parse_judge_output
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import (
    add_fused_paper_scores,
    expanded_rank_weight,
    scoring_edges,
    score_influence,
)
from local_impact_ranking.scoring import InfluenceScores


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


def test_score_influence_tracks_max_single_edge_share():
    paper1 = Paper.model_validate(
        {
            "paper_id": "target1",
            "title": "Target 1",
            "references": [{"reference_id": "r1", "title": "Reference 1"}],
        }
    )
    paper2 = Paper.model_validate(
        {
            "paper_id": "target2",
            "title": "Target 2",
            "references": [{"reference_id": "r1", "title": "Reference 1"}],
        }
    )
    result1 = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "target1",
                "selected_references": [
                    {"reference_id": "r1", "rank": 1, "confidence": 0.9, "rationale": ""}
                ],
            }
        ),
        paper1,
    )
    result2 = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "target2",
                "selected_references": [
                    {"reference_id": "r1", "rank": 2, "confidence": 0.9, "rationale": ""}
                ],
            }
        ),
        paper2,
    )

    scores = score_influence(
        [paper1, paper2],
        [result1, result2],
        credit_mode="rank_weighted",
    )

    assert scores.paper_scores["r1"] == 1.8
    assert math.isclose(scores.paper_score_details["r1"]["max_single_edge_share"], 1.0 / 1.8)


def test_score_influence_tracks_context_score_separately():
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
                },
                {
                    "reference_id": "r2",
                    "sentence": "This was discussed in previous work.",
                    "section": "Introduction",
                },
            ],
        }
    )

    scores = score_influence([paper], [judge_result(paper)], credit_mode="rank_weighted")

    assert scores.paper_score_details["r1"]["context_score"] == 0.5
    assert math.isclose(
        scores.paper_score_details["r2"]["context_score"],
        (1 / math.log2(3)) * 0.2,
    )


def test_add_fused_paper_scores_combines_normalized_components():
    scores = InfluenceScores(
        paper_scores={"p1": 10.0, "p2": 5.0},
        author_scores={},
        institution_scores={},
        field_scores={},
        year_scores={},
        paper_score_details={
            "p1": {"core_score": 2.0, "expanded_score": 2.0, "context_score": 0.0},
            "p2": {"core_score": 1.0, "expanded_score": 4.0, "context_score": 2.0},
        },
    )

    fused = add_fused_paper_scores(scores, graph_scores={"p1": 50.0, "p2": 100.0})

    assert fused.paper_score_details["p1"]["graph_score"] == 50.0
    assert fused.paper_score_details["p2"]["graph_score"] == 100.0
    assert fused.paper_score_details["p1"]["final_score"] == 60.0
    assert fused.paper_score_details["p2"]["final_score"] == 80.0


def test_add_fused_paper_scores_adds_field_year_percentile_component():
    scores = InfluenceScores(
        paper_scores={"p1": 10.0, "p2": 5.0, "p3": 1.0},
        author_scores={},
        institution_scores={},
        field_scores={},
        year_scores={},
        paper_score_details={
            "p1": {"core_score": 2.0, "expanded_score": 2.0, "context_score": 0.0},
            "p2": {"core_score": 1.0, "expanded_score": 4.0, "context_score": 2.0},
            "p3": {"core_score": 0.5, "expanded_score": 1.0, "context_score": 0.0},
        },
    )

    fused = add_fused_paper_scores(
        scores,
        graph_scores={"p1": 50.0, "p2": 100.0, "p3": 0.0},
        paper_metadata={
            "p1": {"field": "AI", "year": 2024},
            "p2": {"field": "AI", "year": 2024},
            "p3": {"field": "AI", "year": 2025},
        },
    )

    assert fused.paper_score_details["p1"]["field_year_percentile"] == 100.0
    assert fused.paper_score_details["p2"]["field_year_percentile"] == 50.0
    assert fused.paper_score_details["p3"]["field_year_percentile"] == 100.0
    assert fused.paper_score_details["p1"]["final_score"] == 64.0
    assert fused.paper_score_details["p2"]["final_score"] == 77.0


def test_add_fused_paper_scores_adds_paper_confidence_flags():
    scores = InfluenceScores(
        paper_scores={"stable": 10.0, "low": 1.0, "fragile": 10.0},
        author_scores={},
        institution_scores={},
        field_scores={},
        year_scores={},
        paper_score_details={
            "stable": {"core_score": 2.0, "expanded_score": 3.0, "context_score": 1.0},
            "low": {"core_score": 0.0, "expanded_score": 0.2, "context_score": 0.0},
            "fragile": {"core_score": 10.0, "expanded_score": 10.0, "context_score": 0.0},
        },
    )

    fused = add_fused_paper_scores(scores, graph_scores={"stable": 0.5, "fragile": 0.0})

    assert fused.paper_score_details["stable"]["confidence_flag"] == "stable"
    assert fused.paper_score_details["low"]["confidence_flag"] == "low_evidence"
    assert fused.paper_score_details["fragile"]["confidence_flag"] == "fragile"
    assert "low context or graph support" in fused.paper_score_details["fragile"]["confidence_reason"]


def test_add_fused_paper_scores_does_not_add_graph_only_papers():
    scores = InfluenceScores(
        paper_scores={"p1": 1.0},
        author_scores={},
        institution_scores={},
        field_scores={},
        year_scores={},
        paper_score_details={"p1": {"core_score": 1.0, "expanded_score": 1.0}},
    )

    fused = add_fused_paper_scores(scores, graph_scores={"p1": 0.1, "target": 0.9})

    assert "target" not in fused.paper_score_details


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
