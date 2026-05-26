import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.scicite_adapter import convert_scicite_jsonl


def write_scicite_fixture(path):
    rows = [
        {
            "citingPaperId": "citing-1",
            "citedPaperId": "cited-1",
            "context": "We use the cited method as a baseline.",
            "sectionName": "Experiment",
            "label": "method",
        },
        {
            "citingPaperId": "citing-1",
            "citedPaperId": "cited-2",
            "context": "This is related background.",
            "sectionName": "Introduction",
            "label": "background",
        },
        {
            "citingPaperId": "citing-2",
            "citedPaperId": "cited-1",
            "string": "We compare against this result.",
            "sectionName": float("nan"),
            "label": "result",
        },
        {
            "citingPaperId": "citing-3",
            "citedPaperId": "None",
            "string": "This row has no linked cited paper.",
            "sectionName": "Discussion",
            "label": "background",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_convert_scicite_jsonl_groups_rows_by_citing_paper(tmp_path):
    source = tmp_path / "scicite.jsonl"
    output = tmp_path / "papers.jsonl"
    write_scicite_fixture(source)

    count = convert_scicite_jsonl(source, output)

    assert count == 2
    papers = load_papers_jsonl(output)
    assert papers[0].paper_id == "citing-1"
    assert {reference.reference_id for reference in papers[0].references} == {
        "cited-1",
        "cited-2",
    }
    assert papers[0].citation_contexts[0].reference_id == "cited-1"
    assert papers[0].citation_contexts[0].section == "Experiment"
    assert papers[0].citation_contexts[0].label == "method"
    assert papers[0].metadata["source_benchmark"] == "scicite"
    second_context = load_papers_jsonl(output)[1].citation_contexts[0]
    assert second_context.sentence == "We compare against this result."
    assert second_context.section is None


def test_cli_benchmarks_convert_scicite_writes_local_schema(tmp_path):
    source = tmp_path / "scicite.jsonl"
    output = tmp_path / "papers.jsonl"
    write_scicite_fixture(source)

    result = CliRunner().invoke(
        app,
        [
            "benchmarks",
            "convert-scicite",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 converted papers" in result.output
    assert load_papers_jsonl(output)[0].references[0].title.startswith("SciCite cited paper")
