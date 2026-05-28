import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.sciimpact_adapter import convert_sciimpact_jsonl


def write_sciimpact_fixture(path):
    rows = [
        {
            "id": "pair-1",
            "chosen": {
                "paper_id": "p-strong",
                "title": "Strong Impact Paper",
                "abstract": "Introduces a broadly reused method.",
                "field": "AI",
                "year": 2022,
            },
            "rejected": {
                "paper_id": "p-weak",
                "title": "Weak Impact Paper",
                "abstract": "Has narrower follow-on use.",
                "field": "AI",
                "year": 2022,
            },
            "impact_dimension": "citation",
            "field": "AI",
        },
        {
            "paper_a": {"id": "p-social", "title": "Policy Impact Paper"},
            "paper_b": {"id": "p-method", "title": "Method Impact Paper"},
            "winner": "paper_b",
            "dimension": "methodological",
            "domain": "Public Health",
        },
        {
            "positive": {"doi": "10.1234/high", "title": "DOI High Impact"},
            "negative": {"doi": "10.1234/low", "title": "DOI Low Impact"},
            "impact_type": "societal",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_convert_sciimpact_jsonl_writes_papers_labels_and_manifest(tmp_path):
    source = tmp_path / "sciimpact.jsonl"
    output = tmp_path / "sciimpact"
    write_sciimpact_fixture(source)

    manifest = convert_sciimpact_jsonl(source, output)

    assert manifest["source_benchmark"] == "sciimpact"
    assert manifest["pair_count"] == 3
    assert manifest["paper_count"] == 6
    assert (output / "manifest.json").exists()

    papers = load_papers_jsonl(output / "papers.jsonl")
    assert {paper.paper_id for paper in papers} >= {"p-strong", "p-weak", "p-method"}
    strong = next(paper for paper in papers if paper.paper_id == "p-strong")
    assert strong.metadata["source_benchmark"] == "sciimpact"
    assert strong.metadata["sciimpact_pair_ids"] == ["pair-1"]
    doi_paper = next(paper for paper in papers if paper.title == "DOI High Impact")
    assert doi_paper.paper_id == "doi:10.1234/high"

    labels = [
        json.loads(line)
        for line in (output / "pairwise_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert labels[0]["pair_id"] == "pair-1"
    assert labels[0]["winner_id"] == "p-strong"
    assert labels[0]["loser_id"] == "p-weak"
    assert labels[0]["impact_dimension"] == "citation"
    assert labels[1]["winner_id"] == "p-method"
    assert labels[1]["field"] == "Public Health"
    assert labels[2]["impact_dimension"] == "societal"
    assert {label["source_benchmark"] for label in labels} == {"sciimpact"}


def test_cli_benchmarks_convert_sciimpact_writes_local_pairwise_files(tmp_path):
    source = tmp_path / "sciimpact.jsonl"
    output = tmp_path / "sciimpact"
    write_sciimpact_fixture(source)

    result = CliRunner().invoke(
        app,
        [
            "benchmarks",
            "convert-sciimpact",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 3 SciImpact pairs" in result.output
    assert (output / "papers.jsonl").exists()
    assert (output / "pairwise_labels.jsonl").exists()
