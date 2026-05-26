import json

from local_impact_ranking.baselines import compute_bibliometric_baselines
from local_impact_ranking.judge import parse_judge_output
from local_impact_ranking.models import Paper
from local_impact_ranking.proxy_evaluation import evaluate_proxy_quality


def sample_papers():
    return [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target",
                "references": [
                    {"reference_id": "p2", "title": "Method Reference"},
                    {"reference_id": "p3", "title": "Background Reference"},
                ],
                "citation_contexts": [
                    {
                        "reference_id": "p2",
                        "sentence": "We build on Method Reference for our architecture.",
                        "section": "Method",
                    },
                    {
                        "reference_id": "p3",
                        "sentence": "Background Reference is related prior work.",
                        "section": "Related Work",
                    },
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Method Reference",
                "year": 2020,
                "field": "AI",
                "citation_count": 100,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Background Reference",
                "year": 2020,
                "field": "AI",
                "citation_count": 1,
                "references": [],
            }
        ),
    ]


def sample_result(paper):
    return parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {"reference_id": "p2", "rank": 1, "confidence": 0.9, "rationale": ""},
                    {"reference_id": "p3", "rank": 2, "confidence": 0.8, "rationale": ""},
                ],
            }
        ),
        paper,
    )


def test_evaluate_proxy_quality_reports_context_and_baseline_support():
    papers = sample_papers()
    baselines = compute_bibliometric_baselines(papers)

    report = evaluate_proxy_quality(papers, [sample_result(papers[0])], baselines, top_n=1)

    assert report["selected_references"] == 2
    assert report["context_supported_selection_rate"] == 1.0
    assert report["method_context_selection_rate"] == 0.5
    assert report["bibliometric_top_n_support_rate"] == 0.5
    assert report["unsupported_selection_rate"] == 0.0
    assert report["selected_reference_evidence"][0]["reference_id"] == "p2"
    assert report["selected_reference_evidence"][0]["method_context_supported"] is True


def test_evaluate_proxy_quality_counts_unsupported_selected_references():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [{"reference_id": "p2", "title": "No Evidence"}],
        }
    )
    result = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {"reference_id": "p2", "rank": 1, "confidence": 0.9, "rationale": ""}
                ],
            }
        ),
        paper,
    )

    report = evaluate_proxy_quality(
        [paper],
        [result],
        compute_bibliometric_baselines([paper]),
        top_n=1,
    )

    assert report["unsupported_selection_rate"] == 1.0
