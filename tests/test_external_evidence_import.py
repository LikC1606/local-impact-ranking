import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.external_evidence_import import (
    merge_external_impact_evidence,
    write_external_evidence_template,
)
from local_impact_ranking.cli import app


def test_merge_external_impact_evidence_matches_by_doi_and_s2_id(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "papers.evidence.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "OpenAlex matched paper",
                        "doi": "10.1000/ABC",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Semantic Scholar matched paper",
                        "references": [],
                        "metadata": {
                            "external_ids": {
                                "semantic_scholar": "S2-123",
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doi",
                "semantic_scholar_id",
                "openalex_field_year_percentile",
                "openalex_fwci",
                "bip_impact_percentile",
                "bip_impact_class",
                "semantic_scholar_influential_citation_count",
                "nih_rcr",
                "pubmed_citation_count",
                "clinical_trial_citations",
                "guideline_citations",
                "drug_approval_mentions",
                "protocol_reuses",
                "reagent_or_assay_reuses",
                "citation_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doi": "HTTPS://DOI.ORG/10.1000/abc.",
                "semantic_scholar_id": "",
                "openalex_field_year_percentile": "0.91",
                "openalex_fwci": "2.4",
                "bip_impact_percentile": "88",
                "bip_impact_class": "4",
                "semantic_scholar_influential_citation_count": "7",
                "nih_rcr": "2.5",
                "pubmed_citation_count": "123",
                "clinical_trial_citations": "3",
                "guideline_citations": "2",
                "drug_approval_mentions": "1",
                "protocol_reuses": "4",
                "reagent_or_assay_reuses": "5",
                "citation_count": "123",
            }
        )
        writer.writerow(
            {
                "doi": "",
                "semantic_scholar_id": "S2-123",
                "openalex_field_year_percentile": "",
                "openalex_fwci": "",
                "bip_impact_percentile": "0.73",
                "bip_impact_class": "3",
                "semantic_scholar_influential_citation_count": "5",
                "nih_rcr": "1.1",
                "pubmed_citation_count": "45",
                "clinical_trial_citations": "0",
                "guideline_citations": "0",
                "drug_approval_mentions": "0",
                "protocol_reuses": "1",
                "reagent_or_assay_reuses": "0",
                "citation_count": "45",
            }
        )

    summary = merge_external_impact_evidence(input_path, evidence_path, output_path)

    assert summary["paper_count"] == 2
    assert summary["matched_count"] == 2
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {record["paper_id"]: record for record in records}
    p1_impact = by_id["p1"]["metadata"]["external_impact"]
    p2_impact = by_id["p2"]["metadata"]["external_impact"]
    assert by_id["p1"]["citation_count"] == 123.0
    assert p1_impact["field_year_percentile"] == 0.91
    assert p1_impact["openalex_fwci"] == 2.4
    assert p1_impact["bip_impact_percentile"] == 88.0
    assert p1_impact["semantic_scholar_influential_citation_count"] == 7.0
    assert p1_impact["relative_citation_ratio"] == 2.5
    assert p1_impact["pubmed_citation_count"] == 123.0
    assert p1_impact["clinical_trial_citations"] == 3.0
    assert p1_impact["drug_approval_mentions"] == 1.0
    assert p1_impact["policy_citations"] == 2.0
    assert p1_impact["protocol_reuses"] == 4.0
    assert p1_impact["reagent_or_assay_reuses"] == 5.0
    assert p2_impact["bip_impact_percentile"] == 0.73
    assert p2_impact["semantic_scholar_influential_citation_count"] == 5.0
    assert by_id["p1"]["metadata"]["openalex_enriched"] is True
    assert by_id["p1"]["metadata"]["bip_enriched"] is True
    assert by_id["p2"]["metadata"]["semantic_scholar_enriched"] is True
    assert by_id["p1"]["metadata"]["icite_enriched"] is True
    assert by_id["p1"]["metadata"]["pubmed_enriched"] is True
    assert by_id["p1"]["metadata"]["life_science_translation_enriched"] is True


def test_cli_import_external_evidence_writes_summary(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    output_path = tmp_path / "papers.evidence.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "One", "references": []}) + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "bip_impact_score": 42,
                "influential_citation_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "import-external-evidence",
            "--input",
            str(input_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    assert result.exit_code == 0
    assert "matched=1" in result.output
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_cli_external_evidence_template_writes_csv(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "external_template.csv"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "doi": "10.1000/a",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "external-evidence-template",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "papers=1" in result.output
    assert "missing_query_ids=0" in result.output
    assert output_path.exists()
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert rows[0]["paper_id"] == "p1"
    assert "scopus_citation_count" in rows[0]


def test_merge_external_impact_evidence_imports_high_recall_citation_sources(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    evidence_path = tmp_path / "high_recall.csv"
    output_path = tmp_path / "papers.high_recall.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "bio-1",
                "title": "Biomedical paper with high recall citation exports",
                "references": [],
                "metadata": {"external_ids": {"pmid": "123456"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pmid",
                "ucite_citation_count",
                "nih_occ_citation_count",
                "dimensions_citation_count",
                "scopus_citation_count",
                "web_of_science_citation_count",
                "lens_citation_count",
                "google_scholar_citation_count",
                "scopus_title",
                "dimensions_title",
                "external_identity_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pmid": "123456",
                "ucite_citation_count": "31",
                "nih_occ_citation_count": "29",
                "dimensions_citation_count": "42",
                "scopus_citation_count": "40",
                "web_of_science_citation_count": "38",
                "lens_citation_count": "41",
                "google_scholar_citation_count": "57",
                "scopus_title": "Biomedical paper with high recall citation exports",
                "dimensions_title": "Biomedical paper with high recall citation exports",
                "external_identity_status": "verified",
            }
        )

    summary = merge_external_impact_evidence(input_path, evidence_path, output_path)

    assert summary["matched_count"] == 1
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    impact = record["metadata"]["external_impact"]
    assert impact["ucite_citation_count"] == 31.0
    assert impact["nih_occ_citation_count"] == 29.0
    assert impact["dimensions_citation_count"] == 42.0
    assert impact["scopus_citation_count"] == 40.0
    assert impact["web_of_science_citation_count"] == 38.0
    assert impact["lens_citation_count"] == 41.0
    assert impact["google_scholar_citation_count"] == 57.0
    assert record["metadata"]["scopus_title"] == (
        "Biomedical paper with high recall citation exports"
    )
    assert record["metadata"]["dimensions_title"] == (
        "Biomedical paper with high recall citation exports"
    )
    assert record["metadata"]["external_identity_status"] == "verified"
    assert record["metadata"]["high_recall_citation_enriched"] is True
    assert record["metadata"]["ucite_enriched"] is True
    assert record["metadata"]["nih_occ_enriched"] is True
    assert record["metadata"]["dimensions_enriched"] is True
    assert record["metadata"]["scopus_enriched"] is True
    assert record["metadata"]["web_of_science_enriched"] is True
    assert record["metadata"]["lens_enriched"] is True
    assert record["metadata"]["google_scholar_enriched"] is True
    assert record["metadata"]["external_evidence_sources"] == [
        "dimensions",
        "google_scholar",
        "high_recall_citation",
        "lens",
        "nih_occ",
        "scopus",
        "ucite",
        "web_of_science",
    ]


def test_merge_external_impact_evidence_reports_evidence_quality_issues(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    evidence_path = tmp_path / "evidence_quality.csv"
    output_path = tmp_path / "papers.evidence.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "doi": "10.1000/quality",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doi",
                "scopus_citation_count",
                "dimensions_citation_count",
                "source_note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doi": "10.1000/quality",
                "scopus_citation_count": "not-a-number",
                "dimensions_citation_count": "-3",
                "source_note": "first",
            }
        )
        writer.writerow(
            {
                "doi": "10.1000/quality",
                "scopus_citation_count": "5",
                "dimensions_citation_count": "4",
                "source_note": "duplicate",
            }
        )

    summary = merge_external_impact_evidence(input_path, evidence_path, output_path)

    validation = summary["evidence_validation"]
    assert validation["evidence_row_count"] == 2
    assert validation["rows_without_match_key_count"] == 0
    assert validation["duplicate_match_key_count"] == 1
    assert validation["invalid_numeric_cell_count"] == 1
    assert validation["negative_numeric_cell_count"] == 1
    assert validation["invalid_numeric_cells"][0]["field"] == "scopus_citation_count"
    assert validation["negative_numeric_cells"][0]["field"] == (
        "dimensions_citation_count"
    )
    assert validation["duplicate_match_keys"][0]["match_key"] == "doi:10.1000/quality"


def test_write_external_evidence_template_includes_identifiers_and_high_recall_columns(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "external_template.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Paper with DOI",
                        "doi": "https://doi.org/10.1000/ABC",
                        "references": [],
                        "metadata": {
                            "external_ids": {
                                "openalex": "https://openalex.org/W123",
                                "semantic_scholar": "S2-1",
                                "pmid": "111",
                                "pmcid": "PMC222",
                            },
                            "scopus_title": "Paper with DOI",
                            "external_impact": {
                                "openalex_cited_by_count": 5,
                                "scopus_citation_count": 7,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Paper without query id",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = write_external_evidence_template(input_path, output_path)

    assert summary["paper_count"] == 2
    assert summary["papers_with_any_query_id"] == 1
    assert summary["papers_without_query_id"] == 1
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert rows[0]["paper_id"] == "p1"
    assert rows[0]["title"] == "Paper with DOI"
    assert rows[0]["doi"] == "10.1000/ABC"
    assert rows[0]["openalex"] == "https://openalex.org/W123"
    assert rows[0]["semantic_scholar_id"] == "S2-1"
    assert rows[0]["pmid"] == "111"
    assert rows[0]["pmcid"] == "PMC222"
    assert rows[0]["openalex_cited_by_count"] == "5"
    assert rows[0]["scopus_citation_count"] == "7"
    assert rows[0]["ucite_citation_count"] == ""
    assert rows[0]["nih_occ_citation_count"] == ""
    assert rows[0]["dimensions_citation_count"] == ""
    assert rows[0]["web_of_science_citation_count"] == ""
    assert rows[0]["lens_citation_count"] == ""
    assert rows[0]["google_scholar_citation_count"] == ""
    assert rows[0]["scopus_title"] == "Paper with DOI"
    assert rows[0]["external_identity_status"] == ""
    assert rows[1]["paper_id"] == "p2"
