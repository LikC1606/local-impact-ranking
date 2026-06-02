import json

from local_impact_ranking.external_evidence_readiness import (
    external_evidence_paper_coverage_rows,
    external_evidence_readiness_report,
    render_external_evidence_readiness_markdown,
    write_external_evidence_paper_coverage_csv,
)
from local_impact_ranking.io import load_papers_jsonl


def test_external_evidence_readiness_reports_identifier_signal_and_graph_coverage(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "OpenAlex and OpenCitations paper",
                        "doi": "10.1000/a",
                        "citation_count": 12,
                        "references": [],
                        "metadata": {
                            "external_ids": {"openalex": "https://openalex.org/W1"},
                            "openalex_enriched": True,
                            "opencitations_enriched": True,
                            "opencitations_incoming_edges": [
                                {"source_paper_id": "p2", "target_paper_id": "p1"}
                            ],
                            "external_impact": {
                                "field_year_percentile": 0.91,
                                "bip_impact_percentile": 0.86,
                                "semantic_scholar_influential_citation_count": 4,
                                "opencitations_cited_by_count": 12,
                                "opencitations_reference_count": 20,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Materialized citing paper",
                        "references": [
                            {
                                "reference_id": "p1",
                                "title": "OpenAlex and OpenCitations paper",
                                "doi": "10.1000/a",
                                "raw_citation": "OpenCitations internal DOI edge",
                            }
                        ],
                        "metadata": {
                            "citation_count": 0,
                            "external_ids": {
                                "doi": "10.1000/b",
                                "openalex": "https://openalex.org/W2",
                            },
                            "opencitations_materialized_edge_count": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p3",
                        "title": "Adoption evidence only",
                        "references": [],
                        "metadata": {
                            "external_impact": {
                                "patent_citations": 2,
                                "policy_citations": 0,
                                "software_mentions": 5,
                                "media_mentions": 3,
                                "award_count": 1,
                            }
                        },
                    }
                ),
                json.dumps({"paper_id": "p4", "title": "Sparse", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = external_evidence_readiness_report(load_papers_jsonl(input_path))

    assert report["paper_count"] == 4
    assert report["identifier_coverage"]["doi_coverage"] == 0.5
    assert report["identifier_coverage"]["openalex_id_coverage"] == 0.5
    assert report["identifier_coverage"]["external_query_id_coverage"] == 0.5
    assert report["source_coverage"]["openalex_enriched_coverage"] == 0.25
    assert report["source_coverage"]["opencitations_enriched_coverage"] == 0.25
    assert report["source_coverage"]["bip_enriched_coverage"] == 0.25
    assert report["source_coverage"]["semantic_scholar_enriched_coverage"] == 0.25
    assert report["signal_coverage"]["citation_count_coverage"] == 0.5
    assert report["signal_coverage"]["yearly_citation_history_coverage"] == 0.25
    assert report["signal_coverage"]["field_year_percentile_coverage"] == 0.25
    assert report["signal_coverage"]["bip_impact_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["influential_citation_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["patent_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["policy_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["positive_policy_evidence_coverage"] == 0.0
    assert report["signal_coverage"]["artifact_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["media_evidence_coverage"] == 0.25
    assert report["signal_coverage"]["award_evidence_coverage"] == 0.25
    assert report["graph_evidence"]["internal_reference_edge_count"] == 1
    assert report["graph_evidence"]["materialized_internal_edge_count"] == 1
    assert report["graph_evidence"]["opencitations_incoming_edge_paper_coverage"] == 0.25
    assert report["paper_coverage_summary"]["paper_count"] == 4
    assert report["paper_coverage_summary"]["papers_with_any_missing_evidence"] == 4
    assert report["paper_coverage_summary"]["missing_evidence_counts"]["yearly_citation_history"] == 3
    assert report["paper_coverage_summary"]["missing_evidence_rates"]["yearly_citation_history"] == 0.75
    assert report["decision"]["status"] == "usable_with_disclosure"
    assert report["decision"]["can_claim_high_credibility"] is False
    assert "Increase DOI/OpenAlex ID coverage" in report["recommendations"]


def test_external_evidence_readiness_treats_known_zero_citations_as_known_yearly_history():
    papers = load_papers_jsonl(
        _write_jsonl(
            [
                {
                    "paper_id": "zero",
                    "title": "Known zero citations",
                    "references": [],
                    "metadata": {"citation_count": 0},
                },
                {
                    "paper_id": "positive",
                    "title": "Positive citations without yearly history",
                    "references": [],
                    "metadata": {"citation_count": 5},
                },
                {
                    "paper_id": "history",
                    "title": "Positive citations with yearly history",
                    "references": [],
                    "metadata": {
                        "citation_count": 5,
                        "citation_counts_by_year": {"2025": 5},
                    },
                },
            ]
        )
    )

    report = external_evidence_readiness_report(papers)
    rows = {
        row["paper_id"]: row
        for row in external_evidence_paper_coverage_rows(papers)
    }

    assert report["signal_coverage"]["yearly_citation_history_coverage"] == 0.666667
    assert rows["zero"]["has_yearly_citation_history"] is True
    assert rows["positive"]["has_yearly_citation_history"] is False
    assert rows["history"]["has_yearly_citation_history"] is True
    assert "yearly_citation_history" not in rows["zero"]["missing_evidence"]
    assert "yearly_citation_history" in rows["positive"]["missing_evidence"]


def test_external_evidence_readiness_counts_local_materialized_internal_edges():
    papers = load_papers_jsonl(
        _write_jsonl(
            [
                {"paper_id": "p1", "title": "Target", "references": []},
                {
                    "paper_id": "p2",
                    "title": "Citing",
                    "references": [{"reference_id": "p1", "title": "Target"}],
                    "metadata": {"internal_citation_materialized_edge_count": 1},
                },
            ]
        )
    )

    report = external_evidence_readiness_report(papers)

    assert report["graph_evidence"]["internal_reference_edge_count"] == 1
    assert report["graph_evidence"]["materialized_internal_edge_count"] == 1


def test_external_evidence_readiness_reports_completed_empty_internal_graph_audit():
    papers = load_papers_jsonl(
        _write_jsonl(
            [
                {
                    "paper_id": "p1",
                    "title": "One",
                    "references": [],
                    "metadata": {"internal_citation_matching_attempted": True},
                },
                {
                    "paper_id": "p2",
                    "title": "Two",
                    "references": [],
                    "metadata": {"internal_citation_matching_attempted": True},
                },
            ]
        )
    )

    report = external_evidence_readiness_report(papers)

    assert report["graph_evidence"]["internal_citation_audit_coverage"] == 1.0
    assert (
        "Internal citation audit found no dataset-internal edges"
        in report["recommendations"]
    )
    assert "Add or materialize dataset-internal citation edges" not in report[
        "recommendations"
    ]


def test_external_evidence_readiness_marks_strong_coverage_as_ready():
    papers = load_papers_jsonl(
        _write_jsonl(
            [
                {
                    "paper_id": "p1",
                    "title": "One",
                    "doi": "10.1000/1",
                    "citation_count": 10,
                    "references": [],
                    "metadata": {
                        "openalex_enriched": True,
                        "opencitations_enriched": True,
                        "external_ids": {"openalex": "W1"},
                        "citation_counts_by_year": {"2024": 2},
                            "external_impact": {
                                "field_year_percentile": 0.9,
                                "bip_impact_percentile": 0.9,
                                "semantic_scholar_influential_citation_count": 3,
                                "patent_citations": 1,
                                "policy_citations": 1,
                                "software_mentions": 1,
                        },
                    },
                },
                {
                    "paper_id": "p2",
                    "title": "Two",
                    "doi": "10.1000/2",
                    "citation_count": 5,
                    "references": [
                        {
                            "reference_id": "p1",
                            "title": "One",
                            "raw_citation": "OpenCitations internal DOI edge",
                        }
                    ],
                    "metadata": {
                        "openalex_enriched": True,
                        "opencitations_enriched": True,
                        "external_ids": {"openalex": "W2"},
                        "citation_counts_by_year": {"2024": 1},
                        "opencitations_materialized_edge_count": 1,
                            "external_impact": {
                                "field_year_percentile": 0.8,
                                "bip_impact_percentile": 0.8,
                                "semantic_scholar_influential_citation_count": 2,
                                "patent_citations": 1,
                                "policy_citations": 1,
                                "software_mentions": 1,
                        },
                    },
                },
            ]
        )
    )

    report = external_evidence_readiness_report(papers)

    assert report["decision"]["status"] == "ready_for_high_credibility_scoring"
    assert report["decision"]["can_claim_high_credibility"] is True
    assert report["recommendations"] == []


def test_render_external_evidence_readiness_markdown_contains_decision():
    markdown = render_external_evidence_readiness_markdown(
        {
            "paper_count": 2,
            "decision": {
                "status": "usable_with_disclosure",
                "can_claim_high_credibility": False,
                "reason": "External evidence is usable but incomplete.",
            },
            "identifier_coverage": {"doi_coverage": 0.5},
            "source_coverage": {"openalex_enriched_coverage": 0.5},
            "signal_coverage": {"citation_count_coverage": 0.5},
            "graph_evidence": {"internal_reference_edge_count": 1},
            "paper_coverage_summary": {
                "paper_count": 2,
                "fully_covered_paper_count": 1,
                "fully_covered_paper_coverage": 0.5,
                "papers_with_any_missing_evidence": 1,
                "papers_with_any_missing_evidence_coverage": 0.5,
                "missing_evidence_counts": {
                    "doi_or_openalex_id": 1,
                    "field_year_percentile": 1,
                },
                "missing_evidence_rates": {
                    "doi_or_openalex_id": 0.5,
                    "field_year_percentile": 0.5,
                },
                "top_missing_evidence": [
                    {"label": "doi_or_openalex_id", "paper_count": 1, "paper_coverage": 0.5},
                    {"label": "field_year_percentile", "paper_count": 1, "paper_coverage": 0.5},
                ],
            },
            "recommendations": ["Increase DOI/OpenAlex ID coverage"],
            "thresholds": {"strong_external_query_id_coverage": 0.8},
        }
    )

    assert markdown.startswith("# External Evidence Readiness\n")
    assert "| Status | usable_with_disclosure |" in markdown
    assert "| Can claim high credibility | no |" in markdown
    assert "| doi_coverage | 0.500 |" in markdown
    assert "## Per-Paper Evidence Gaps" in markdown
    assert "| fully_covered_paper_coverage | 0.500 |" in markdown
    assert "| doi_or_openalex_id | 1 | 0.500 |" in markdown
    assert "- Increase DOI/OpenAlex ID coverage" in markdown


def test_external_evidence_paper_coverage_rows_explain_missing_evidence(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Well covered",
                        "doi": "10.1000/a",
                        "citation_count": 12,
                        "references": [],
                        "metadata": {
                            "external_ids": {"openalex": "W1"},
                            "openalex_enriched": True,
                            "opencitations_enriched": True,
                            "citation_counts_by_year": {"2025": 3},
                            "opencitations_incoming_edges": [
                                {"source_paper_id": "p2", "target_paper_id": "p1"}
                            ],
                            "external_impact": {
                                "field_year_percentile": 0.9,
                                "bip_impact_percentile": 0.9,
                                "semantic_scholar_influential_citation_count": 2,
                                "patent_citations": 1,
                                "policy_citations": 1,
                                "software_mentions": 4,
                            },
                        },
                    }
                ),
                json.dumps({"paper_id": "p2", "title": "Sparse", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = external_evidence_paper_coverage_rows(load_papers_jsonl(input_path))

    assert rows[0]["paper_id"] == "p1"
    assert rows[0]["has_doi"] is True
    assert rows[0]["has_openalex_id"] is True
    assert rows[0]["has_positive_policy_evidence"] is True
    assert rows[0]["has_bip_impact_evidence"] is True
    assert rows[0]["has_influential_citation_evidence"] is True
    assert rows[0]["has_opencitations_incoming_edges"] is True
    assert rows[0]["missing_evidence"] == ""
    assert rows[1]["paper_id"] == "p2"
    assert rows[1]["has_doi"] is False
    assert rows[1]["has_citation_count"] is False
    assert "doi_or_openalex_id" in rows[1]["missing_evidence"]
    assert "field_year_percentile" in rows[1]["missing_evidence"]


def test_write_external_evidence_paper_coverage_csv(tmp_path):
    path = tmp_path / "coverage.csv"
    rows = [
        {
            "paper_id": "p1",
            "title": "Paper",
            "has_doi": True,
            "has_openalex_id": False,
            "missing_evidence": "openalex_id",
        }
    ]

    write_external_evidence_paper_coverage_csv(rows, path)

    text = path.read_text(encoding="utf-8")
    assert "paper_id,title,has_doi" in text
    assert "p1,Paper,yes,no" in text


def _write_jsonl(records, tmp_path=None):
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    if tmp_path is not None:
        path = Path(tmp_path) / "papers.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path
    handle = NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
    with handle:
        handle.write("\n".join(json.dumps(record) for record in records) + "\n")
    return Path(handle.name)
