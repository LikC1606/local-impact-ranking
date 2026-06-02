from local_impact_ranking.impact_taxonomy import (
    impact_dimension_taxonomy,
    ranking_dimension_coverage,
)


def test_impact_dimension_taxonomy_maps_sciimpact_dimensions_to_score_fields():
    taxonomy = impact_dimension_taxonomy()

    assert taxonomy["citation"]["claim_type"] == "citation-impact"
    assert "relative_citation_ratio_score" in taxonomy["citation"]["score_fields"]
    assert "patent_impact_score" in taxonomy["patent"]["score_fields"]
    assert "media_impact_score" in taxonomy["media"]["score_fields"]
    assert "artifact_adoption_score" in taxonomy["downloads"]["score_fields"]
    assert "artifact_adoption_score" in taxonomy["stars"]["score_fields"]
    assert taxonomy["award"]["claim_type"] == "recognition-impact"


def test_ranking_dimension_coverage_reports_supported_dimensions():
    coverage = ranking_dimension_coverage(
        [
            {
                "paper_id": "p1",
                "field_year_normalized_citation_score": 80.0,
                "patent_citations_raw": 2,
                "media_mentions_raw": 5,
                "artifact_adoption_raw": None,
            },
            {
                "paper_id": "p2",
                "field_year_normalized_citation_score": 0.0,
                "patent_citations_raw": None,
                "media_mentions_raw": 0,
                "artifact_adoption_raw": 3,
            },
        ]
    )

    assert coverage["citation"]["covered_paper_count"] == 1
    assert coverage["patent"]["covered_paper_count"] == 1
    assert coverage["media"]["covered_paper_count"] == 1
    assert coverage["downloads"]["covered_paper_count"] == 1
    assert coverage["stars"]["covered_paper_count"] == 1
