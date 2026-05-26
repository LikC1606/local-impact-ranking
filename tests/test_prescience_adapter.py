import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.prescience_adapter import convert_prescience_file


def write_prescience_fixture(path):
    rows = [
        {
            "arxiv_id": "2401.00001",
            "corpus_id": "target-1",
            "title": "Target AI Paper",
            "abstract": "We introduce a ranking model.",
            "date": "2024-01-15",
            "categories": ["cs.AI"],
            "roles": ["target"],
            "citation_trajectory": [1, 2, 3],
            "key_references": [{"corpus_id": "ref-positive", "num_citations": 9}],
            "authors": [
                {
                    "author_id": "ada_1",
                    "name": "Ada",
                    "publication_history": None,
                }
            ],
        },
        {
            "corpus_id": "ref-positive",
            "arxiv_id": "2301.00001",
            "title": "Important Prior Work",
            "abstract": "A strong prior method.",
            "date": "2023-01-01",
            "categories": ["cs.LG"],
            "roles": ["target.key_reference"],
            "authors": [{"name": "Grace"}],
        },
        {
            "corpus_id": "ref-negative",
            "arxiv_id": "2201.00001",
            "title": "Less Relevant Prior Work",
            "date": "2022-01-01",
            "categories": ["cs.CL"],
            "roles": ["target.author.publication_history"],
            "authors": [{"author_id": "ada_1", "name": "Alan"}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_convert_prescience_file_builds_candidates_and_positive_metadata(tmp_path):
    source = tmp_path / "prescience.jsonl"
    output = tmp_path / "papers.jsonl"
    write_prescience_fixture(source)

    count = convert_prescience_file(source, output, max_candidates=10)

    papers = load_papers_jsonl(output)
    assert count == 1
    assert papers[0].paper_id == "target-1"
    assert papers[0].year == 2024
    assert {reference.reference_id for reference in papers[0].references} == {
        "ref-positive",
        "ref-negative",
    }
    assert papers[0].references[0].title == "Important Prior Work"
    assert papers[0].references[0].field == "cs.LG"
    assert papers[0].metadata["source_benchmark"] == "prescience"
    assert papers[0].metadata["prescience_positive_reference_ids"] == ["ref-positive"]
    assert papers[0].metadata["prescience_key_reference_citations"]["ref-positive"] == 9


def test_cli_benchmarks_convert_prescience_writes_local_schema(tmp_path):
    source = tmp_path / "prescience.jsonl"
    output = tmp_path / "papers.jsonl"
    write_prescience_fixture(source)

    result = CliRunner().invoke(
        app,
        [
            "benchmarks",
            "convert-prescience",
            "--input",
            str(source),
            "--output",
            str(output),
            "--max-candidates",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 converted papers" in result.output
    assert load_papers_jsonl(output)[0].metadata["source_benchmark"] == "prescience"
