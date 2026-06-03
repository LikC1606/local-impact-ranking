import csv
import json

from local_impact_ranking.citation_completeness_audit import (
    citation_completeness_audit_report,
    render_citation_completeness_audit_markdown,
    write_citation_completeness_audit,
)
from local_impact_ranking.models import Paper


def test_citation_completeness_audit_grades_missing_single_compatible_conflict_and_zero():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "no_id",
                "title": "No external identifier",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "single",
                "title": "Single source only",
                "doi": "10.1000/single",
                "references": [],
                "metadata": {
                    "external_impact": {"openalex_cited_by_count": 12},
                },
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "compatible",
                "title": "Compatible multi source",
                "doi": "10.1000/compatible",
                "references": [],
                "metadata": {
                    "external_ids": {"semantic_scholar": "S2-1"},
                    "external_impact": {
                        "openalex_cited_by_count": 10,
                        "semantic_scholar_citation_count": 11,
                    },
                },
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "conflict",
                "title": "Conflicting source counts",
                "doi": "10.1000/conflict",
                "references": [],
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 0,
                        "semantic_scholar_citation_count": 50,
                    },
                },
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "zero",
                "title": "Confirmed zero",
                "doi": "10.1000/zero",
                "references": [],
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 0,
                        "semantic_scholar_citation_count": 0,
                    },
                },
            }
        ),
    ]

    report = citation_completeness_audit_report(papers)
    rows = {row["paper_id"]: row for row in report["papers"]}

    assert report["summary"]["paper_count"] == 5
    assert report["summary"]["papers_without_query_ids"] == 1
    assert report["summary"]["papers_with_single_source_only"] == 1
    assert report["summary"]["papers_with_source_conflict"] == 1
    assert report["summary"]["papers_with_confirmed_multi_source_zero"] == 1
    assert report["summary"]["citation_evidence_coverage"] == 0.8
    assert report["summary"]["multi_source_coverage"] == 0.6
    assert report["summary"]["mean_completeness_score"] == 0.55
    assert report["summary"]["coverage_confidence_counts"] == {
        "high": 2,
        "low": 2,
        "moderate": 1,
    }
    assert report["summary"]["recommended_next_source_action_counts"][
        "resolve_query_identifier"
    ] == 1
    assert report["summary"]["recommended_next_source_action_counts"][
        "query_high_recall_sources"
    ] == 1

    assert rows["no_id"]["completeness_grade"] == "no_query_id"
    assert rows["no_id"]["preferred_count"] is None
    assert rows["no_id"]["can_treat_zero_as_confirmed"] is False
    assert "doi" in rows["no_id"]["missing_query_ids"]
    assert rows["no_id"]["completeness_score"] == 0.05
    assert rows["no_id"]["coverage_confidence"] == "low"
    assert rows["no_id"]["recommended_next_source_actions"] == [
        "resolve_query_identifier"
    ]

    assert rows["single"]["completeness_grade"] == "single_source_only"
    assert rows["single"]["preferred_source"] == "openalex"
    assert rows["single"]["preferred_count"] == 12.0
    assert rows["single"]["source_counts"] == {"openalex": 12.0}
    assert rows["single"]["completeness_score"] == 0.45
    assert rows["single"]["coverage_confidence"] == "low"
    assert "add_second_independent_source" in rows["single"][
        "recommended_next_source_actions"
    ]

    assert rows["compatible"]["completeness_grade"] == "multi_source_compatible"
    assert rows["compatible"]["source_conflict_flag"] is False
    assert rows["compatible"]["source_spread"] == 1.0
    assert rows["compatible"]["completeness_score"] == 0.9
    assert rows["compatible"]["coverage_confidence"] == "high"

    assert rows["conflict"]["completeness_grade"] == "source_conflict"
    assert rows["conflict"]["source_conflict_flag"] is True
    assert rows["conflict"]["preferred_source"] == "semantic_scholar"
    assert rows["conflict"]["preferred_count"] == 50.0
    assert rows["conflict"]["completeness_score"] == 0.55
    assert rows["conflict"]["coverage_confidence"] == "moderate"
    assert rows["conflict"]["recommended_next_source_actions"] == [
        "inspect_source_conflict",
        "query_high_recall_sources",
    ]

    assert rows["zero"]["completeness_grade"] == "multi_source_zero"
    assert rows["zero"]["zero_citation_status"] == "confirmed_multi_source_zero"
    assert rows["zero"]["can_treat_zero_as_confirmed"] is True
    assert rows["zero"]["completeness_score"] == 0.8
    assert rows["zero"]["coverage_confidence"] == "high"


def test_write_citation_completeness_audit_writes_json_markdown_and_csv(tmp_path):
    output_dir = tmp_path / "audit"
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Paper",
                "doi": "10.1000/a",
                "references": [],
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 3,
                        "semantic_scholar_citation_count": 3,
                    }
                },
            }
        )
    ]

    report = write_citation_completeness_audit(papers, output_dir)

    assert report["summary"]["paper_count"] == 1
    assert (output_dir / "citation_completeness_audit.json").exists()
    assert (output_dir / "citation_completeness_audit.md").exists()
    assert (output_dir / "citation_completeness_papers.csv").exists()
    assert "# Citation Completeness Audit" in (
        output_dir / "citation_completeness_audit.md"
    ).read_text(encoding="utf-8")
    csv_rows = list(
        csv.DictReader(
            (output_dir / "citation_completeness_papers.csv").open(encoding="utf-8")
        )
    )
    assert csv_rows[0]["paper_id"] == "p1"
    assert csv_rows[0]["completeness_grade"] == "multi_source_compatible"
    assert csv_rows[0]["coverage_confidence"] == "high"
    assert csv_rows[0]["completeness_score"] == "0.9"
    assert json.loads(csv_rows[0]["source_counts"]) == {
        "openalex": 3.0,
        "semantic_scholar": 3.0,
    }


def test_render_citation_completeness_audit_markdown_explains_no_exhaustive_claim():
    report = citation_completeness_audit_report(
        [
            Paper.model_validate(
                {
                    "paper_id": "p1",
                    "title": "Sparse",
                    "references": [],
                }
            )
        ]
    )

    markdown = render_citation_completeness_audit_markdown(report)

    assert "does not prove exhaustive citation coverage" in markdown
    assert "no_query_id" in markdown


def test_citation_completeness_audit_recommends_life_science_union_sources():
    report = citation_completeness_audit_report(
        [
            Paper.model_validate(
                {
                    "paper_id": "bio",
                    "title": "Biomedical paper",
                    "field": "life sciences",
                    "doi": "10.1000/bio",
                    "pmid": "12345",
                    "references": [],
                    "metadata": {
                        "external_impact": {"openalex_cited_by_count": 4},
                    },
                }
            )
        ]
    )

    row = report["papers"][0]

    assert row["is_life_science_record"] is True
    assert "query_pubmed_union_sources" in row["recommended_next_source_actions"]
    assert report["summary"]["life_science_record_count"] == 1
    assert report["summary"]["recommended_next_source_action_counts"][
        "query_pubmed_union_sources"
    ] == 1


def test_citation_completeness_audit_reports_high_recall_source_coverage():
    report = citation_completeness_audit_report(
        [
            Paper.model_validate(
                {
                    "paper_id": "high-recall",
                    "title": "High recall citation evidence",
                    "doi": "10.1000/high",
                    "references": [],
                    "metadata": {
                        "external_impact": {
                            "ucite_citation_count": 10,
                            "nih_occ_citation_count": 10,
                            "dimensions_citation_count": 12,
                            "scopus_citation_count": 12,
                            "web_of_science_citation_count": 11,
                            "lens_citation_count": 12,
                            "google_scholar_citation_count": 14,
                        }
                    },
                }
            )
        ]
    )

    row = report["papers"][0]

    assert row["preferred_count"] == 14.0
    assert row["preferred_source"] == "google_scholar"
    assert row["source_count"] == 7
    assert row["source_counts"] == {
        "ucite": 10.0,
        "nih_occ": 10.0,
        "dimensions": 12.0,
        "scopus": 12.0,
        "web_of_science": 11.0,
        "lens": 12.0,
        "google_scholar": 14.0,
    }
    assert report["summary"]["source_coverage"]["ucite"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["nih_occ"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["dimensions"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["scopus"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["web_of_science"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["lens"]["paper_count"] == 1
    assert report["summary"]["source_coverage"]["google_scholar"]["paper_count"] == 1
