import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.impact_backtest_workflow import write_impact_backtest


def _write_enriched_papers(path):
    rows = [
        {
            "paper_id": "p1",
            "title": "A",
            "field": "AI",
            "references": [],
            "metadata": {
                "citation_cutoff_year": 2022,
                "citation_counts_by_year": {"2021": 3, "2022": 4, "2023": 10},
            },
        },
        {
            "paper_id": "p2",
            "title": "B",
            "field": "AI",
            "references": [],
            "metadata": {
                "citation_cutoff_year": 2022,
                "citation_counts_by_year": {"2021": 7, "2022": 8, "2023": 2},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_cutoff_papers(path):
    rows = [
        {
            "paper_id": "p1",
            "title": "A",
            "field": "AI",
            "references": [],
            "metadata": {
                "citation_cutoff_year": 2022,
                "citation_counts_by_year": {"2021": 3, "2022": 4},
            },
        },
        {
            "paper_id": "p2",
            "title": "B",
            "field": "AI",
            "references": [],
            "metadata": {
                "citation_cutoff_year": 2022,
                "citation_counts_by_year": {"2021": 7, "2022": 8},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_rankings(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        writer.writerow({"paper_id": "p1", "final_impact_score": 90})
        writer.writerow({"paper_id": "p2", "final_impact_score": 10})


def test_write_impact_backtest_creates_auditable_evidence_bundle(tmp_path):
    enriched = tmp_path / "papers.openalex.jsonl"
    cutoff = tmp_path / "papers.cutoff.jsonl"
    run_dir = tmp_path / "run"
    _write_enriched_papers(enriched)
    _write_cutoff_papers(cutoff)
    _write_rankings(run_dir / "impact_scores.csv")

    manifest = write_impact_backtest(
        enriched_input=enriched,
        cutoff_input=cutoff,
        rankings=run_dir / "impact_scores.csv",
        output_dir=run_dir,
        cutoff_year=2022,
        future_start_year=2023,
        future_end_year=2023,
        label_normalization="field_year_percentile",
        bootstrap_samples=10,
        permutation_samples=10,
        random_seed=3,
    )

    assert manifest["summary"]["cutoff_audit_status"] == "passed"
    assert manifest["summary"]["label_normalization"] == "field_year_percentile"
    assert manifest["summary"]["future_label_count"] == 2
    assert manifest["summary"]["validity_matched_paper_count"] == 2
    assert (run_dir / "reports" / "citation_cutoff_audit.json").exists()
    assert (run_dir / "reports" / "impact_validity_evaluation.json").exists()
    assert (run_dir / "reports" / "impact_credibility_scorecard.json").exists()
    assert (run_dir / "future-labels" / "future_impact_labels.jsonl").exists()
    assert (run_dir / "impact_backtest_manifest.json").exists()
    assert (
        manifest["outputs"]["impact_credibility_scorecard_md"]
        == "reports/impact_credibility_scorecard.md"
    )


def test_cli_backtest_impact_validity_writes_workflow_outputs(tmp_path):
    enriched = tmp_path / "papers.openalex.jsonl"
    cutoff = tmp_path / "papers.cutoff.jsonl"
    run_dir = tmp_path / "run"
    _write_enriched_papers(enriched)
    _write_cutoff_papers(cutoff)
    _write_rankings(run_dir / "impact_scores.csv")

    result = CliRunner().invoke(
        app,
        [
            "backtest-impact-validity",
            "--enriched-input",
            str(enriched),
            "--cutoff-input",
            str(cutoff),
            "--rankings",
            str(run_dir / "impact_scores.csv"),
            "--output",
            str(run_dir),
            "--cutoff-year",
            "2022",
            "--future-start-year",
            "2023",
            "--future-end-year",
            "2023",
            "--label-normalization",
            "field-year-percentile",
            "--bootstrap-samples",
            "10",
            "--permutation-samples",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert "label_normalization=field_year_percentile" in result.output
    assert "cutoff_audit=passed" in result.output
    assert (run_dir / "reports" / "citation_cutoff_audit.md").exists()
    assert (run_dir / "reports" / "impact_validity_evaluation.md").exists()
    assert (run_dir / "reports" / "impact_credibility_scorecard.md").exists()
