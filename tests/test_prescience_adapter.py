import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.prescience_adapter import (
    convert_prescience_file,
    write_prescience_splits,
)


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
    assert papers[0].metadata["candidate_ordering"] == "deterministic_shuffle"
    assert papers[0].metadata["candidate_shuffle_seed"] == 13


def test_convert_prescience_file_shuffles_candidates_deterministically(tmp_path):
    source = tmp_path / "prescience.jsonl"
    output_a = tmp_path / "papers-a.jsonl"
    output_b = tmp_path / "papers-b.jsonl"
    write_prescience_fixture(source)

    convert_prescience_file(source, output_a, max_candidates=10, shuffle_seed=7)
    convert_prescience_file(source, output_b, max_candidates=10, shuffle_seed=7)

    paper_a = load_papers_jsonl(output_a)[0]
    paper_b = load_papers_jsonl(output_b)[0]
    ids_a = [reference.reference_id for reference in paper_a.references]
    ids_b = [reference.reference_id for reference in paper_b.references]
    assert ids_a == ids_b
    assert ids_a == ["ref-negative", "ref-positive"]


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


def test_write_prescience_splits_exports_fixed_protocol_manifest(tmp_path):
    source = tmp_path / "prescience.jsonl"
    output_dir = tmp_path / "splits"
    rows = []
    for index in range(1, 5):
        rows.extend(
            [
                {
                    "corpus_id": f"target-{index}",
                    "title": f"Target {index}",
                    "date": "2024-01-01",
                    "roles": ["target"],
                    "key_references": [
                        {"corpus_id": f"positive-{index}", "num_citations": index}
                    ],
                    "authors": [
                        {
                            "author_id": f"author-{index}",
                            "name": f"Author {index}",
                            "publication_history": None,
                        }
                    ],
                },
                {
                    "corpus_id": f"positive-{index}",
                    "title": f"Positive {index}",
                    "date": "2023-01-01",
                    "roles": ["target.key_reference"],
                    "authors": [{"name": f"Prior {index}"}],
                },
                {
                    "corpus_id": f"negative-{index}",
                    "title": f"Negative {index}",
                    "date": "2022-01-01",
                    "roles": ["target.author.publication_history"],
                    "authors": [
                        {"author_id": f"author-{index}", "name": f"History {index}"}
                    ],
                },
            ]
        )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = write_prescience_splits(
        source,
        output_dir,
        train_count=2,
        dev_count=1,
        test_count=1,
        max_candidates=10,
        shuffle_seed=7,
    )

    assert [paper.paper_id for paper in load_papers_jsonl(output_dir / "train.papers.jsonl")] == [
        "target-1",
        "target-2",
    ]
    assert [paper.paper_id for paper in load_papers_jsonl(output_dir / "dev.papers.jsonl")] == [
        "target-3",
    ]
    assert [paper.paper_id for paper in load_papers_jsonl(output_dir / "test.papers.jsonl")] == [
        "target-4",
    ]
    written_manifest = json.loads((output_dir / "split_manifest.json").read_text())
    assert manifest == written_manifest
    assert manifest["benchmark"] == "prescience"
    assert manifest["method"]["type"] == "ordered_prefix_after_conversion"
    assert manifest["method"]["max_candidates"] == 10
    assert manifest["method"]["shuffle_seed"] == 7
    assert manifest["splits"]["train"]["candidate_reference_count"] == 4
    assert manifest["splits"]["train"]["gold_positive_reference_count"] == 2
    assert manifest["splits"]["dev"]["paper_ids"] == ["target-3"]
    assert manifest["splits"]["test"]["gold_positive_reference_count"] == 1
