import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.external_evidence_import import merge_external_impact_evidence
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
                "doi": "https://doi.org/10.1000/abc",
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
