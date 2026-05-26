from local_impact_ranking.contexts import (
    citation_contexts_by_reference,
    context_coverage_report,
)
from local_impact_ranking.models import Paper


def sample_paper():
    return Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {"reference_id": "r1", "title": "Reference One"},
                {"reference_id": "r2", "title": "Reference Two"},
            ],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We build directly on Reference One.",
                    "paragraph": "We build directly on Reference One for the core method.",
                    "section": "Method",
                },
                {
                    "reference_id": "r1",
                    "sentence": "Reference One is our main baseline.",
                    "section": "Experiments",
                },
            ],
        }
    )


def test_citation_contexts_by_reference_groups_contexts():
    grouped = citation_contexts_by_reference(sample_paper())

    assert set(grouped) == {"r1"}
    assert len(grouped["r1"]) == 2
    assert grouped["r1"][0].section == "Method"


def test_context_coverage_report_counts_reference_and_paper_coverage():
    report = context_coverage_report([sample_paper()])

    assert report == {
        "papers": 1,
        "papers_with_contexts": 1,
        "paper_context_coverage_rate": 1.0,
        "references": 2,
        "references_with_contexts": 1,
        "reference_context_coverage_rate": 0.5,
        "citation_contexts": 2,
    }
