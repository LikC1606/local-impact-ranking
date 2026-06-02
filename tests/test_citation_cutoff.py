import json

from typer.testing import CliRunner

from local_impact_ranking.citation_cutoff import (
    apply_citation_cutoff,
    audit_citation_cutoff,
    write_citation_cutoff,
    write_citation_cutoff_audit,
)
from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl


def test_apply_citation_cutoff_removes_future_yearly_counts_and_updates_totals(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "citation_count": 30,
                "references": [],
                "metadata": {
                    "citation_counts_by_year": {
                        "2020": 3,
                        "2021": 5,
                        "2022": 7,
                        "2023": 11,
                    },
                    "external_impact": {
                        "openalex_cited_by_count": 30,
                        "field_year_percentile": 0.9,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = apply_citation_cutoff(input_path, cutoff_year=2021)

    assert report["summary"] == {
        "paper_count": 1,
        "cutoff_year": 2021,
        "updated_citation_history_count": 1,
        "updated_citation_count_count": 1,
    }
    paper = report["papers"][0]
    assert paper.metadata["citation_counts_by_year"] == {"2020": 3.0, "2021": 5.0}
    assert paper.metadata["citation_count"] == 8.0
    assert getattr(paper, "__pydantic_extra__", {})["citation_count"] == 8.0
    assert paper.metadata["external_impact"]["openalex_cited_by_count"] == 8.0
    assert paper.metadata["citation_cutoff_year"] == 2021


def test_write_citation_cutoff_outputs_jsonl_and_manifest(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "cutoff.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "references": [],
                "metadata": {"citation_counts_by_year": {"2021": 5, "2022": 7}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_citation_cutoff(input_path, output_path, cutoff_year=2021)

    assert report["summary"]["updated_citation_history_count"] == 1
    assert output_path.exists()
    assert (tmp_path / "cutoff.manifest.json").exists()
    paper = load_papers_jsonl(output_path)[0]
    assert paper.metadata["citation_counts_by_year"] == {"2021": 5.0}


def test_cli_apply_citation_cutoff_writes_outputs(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "cutoff.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "references": [],
                "metadata": {"citation_counts_by_year": {"2021": 5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "apply-citation-cutoff",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cutoff-year",
            "2021",
        ],
    )

    assert result.exit_code == 0
    assert "updated_histories=1" in result.output
    assert output_path.exists()


def test_audit_citation_cutoff_detects_future_year_leakage(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Leaky",
                        "references": [],
                        "metadata": {
                            "citation_cutoff_year": 2022,
                            "citation_counts_by_year": {"2021": 5, "2023": 7},
                        },
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Clean",
                        "references": [],
                        "metadata": {
                            "citation_cutoff_year": 2022,
                            "citation_counts_by_year": {"2021": 2},
                        },
                    }
                ),
                json.dumps({"paper_id": "p3", "title": "No history", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_citation_cutoff(input_path, cutoff_year=2022)

    assert report["summary"]["paper_count"] == 3
    assert report["summary"]["papers_with_yearly_citation_history"] == 2
    assert report["summary"]["papers_with_future_years"] == 1
    assert report["summary"]["future_year_count"] == 1
    assert report["summary"]["citation_cutoff_year_coverage"] == 2 / 3
    assert report["decision"]["status"] == "failed"
    assert report["future_year_examples"][0]["paper_id"] == "p1"


def test_write_citation_cutoff_audit_outputs_json_and_markdown(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "audit"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Clean",
                "references": [],
                "metadata": {
                    "citation_cutoff_year": 2022,
                    "citation_counts_by_year": {"2021": 5},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_citation_cutoff_audit(input_path, output_dir, cutoff_year=2022)

    assert report["decision"]["status"] == "passed"
    assert (output_dir / "citation_cutoff_audit.json").exists()
    markdown = (output_dir / "citation_cutoff_audit.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Citation Cutoff Audit\n")
    assert "| papers_with_future_years | 0 |" in markdown


def test_cli_audit_citation_cutoff_writes_report(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "audit"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Clean",
                "references": [],
                "metadata": {"citation_counts_by_year": {"2021": 5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "audit-citation-cutoff",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--cutoff-year",
            "2022",
        ],
    )

    assert result.exit_code == 0
    assert "status=warning" in result.output
    assert (output_dir / "citation_cutoff_audit.json").exists()
