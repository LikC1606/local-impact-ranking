import json

import pytest
from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.sciimpact_adapter import (
    convert_sciimpact_jsonl,
    export_sciimpact_dpo_jsonl,
)


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


def test_convert_sciimpact_sharegpt_sft_writes_pairwise_labels(tmp_path):
    source = tmp_path / "sciimpact_sft.jsonl"
    output = tmp_path / "sciimpact_sft"
    rows = [
        {
            "conversations": [
                {
                    "from": "system",
                    "value": (
                        "You are an impartial paper reviewer. Given the titles and abstracts "
                        "of two papers, identify which paper won the Best Paper award."
                    ),
                },
                {
                    "from": "human",
                    "value": (
                        "Two papers are nominated for the Best Paper award. "
                        "Paper A: Title: Alpha System Abstract: Alpha introduces a broad system. "
                        "Paper B: Title: Beta Baseline Abstract: Beta is a narrower baseline. "
                        "Which paper should win the Best Paper award?"
                    ),
                },
                {"from": "gpt", "value": "Paper A won the best paper award."},
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": (
                        "You are an impartial paper reviewer. Given two papers, identify "
                        "which paper has more citations."
                    ),
                },
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Low Cited Paper Abstract: Low cited abstract. "
                        "Paper B: Title: High Cited Paper Abstract: High cited abstract. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = convert_sciimpact_jsonl(source, output)

    assert manifest["pair_count"] == 2
    assert manifest["paper_count"] == 4
    papers = load_papers_jsonl(output / "papers.jsonl")
    papers_by_title = {paper.title: paper for paper in papers}
    assert papers_by_title["Alpha System"].abstract == "Alpha introduces a broad system."
    assert papers_by_title["High Cited Paper"].abstract == "High cited abstract."
    assert papers_by_title["Alpha System"].paper_id.startswith("sciimpact:")

    labels = [
        json.loads(line)
        for line in (output / "pairwise_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert labels[0]["winner_id"] == papers_by_title["Alpha System"].paper_id
    assert labels[0]["loser_id"] == papers_by_title["Beta Baseline"].paper_id
    assert labels[0]["impact_dimension"] == "award"
    assert labels[1]["winner_id"] == papers_by_title["High Cited Paper"].paper_id
    assert labels[1]["loser_id"] == papers_by_title["Low Cited Paper"].paper_id
    assert labels[1]["impact_dimension"] == "citation"


def test_convert_sciimpact_sharegpt_uses_content_stable_ids(tmp_path):
    source = tmp_path / "sciimpact_repeated_sft.jsonl"
    output = tmp_path / "sciimpact_repeated_sft"
    repeated = "Title: Shared Paper Abstract: Same text appears in two pairs."
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "Which paper has more citations?"},
                {
                    "from": "human",
                    "value": (
                        f"Paper A: {repeated} "
                        "Paper B: Title: Other One Abstract: Other text. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper A has more citations."},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "Which paper has more citations?"},
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Other Two Abstract: Other text. "
                        f"Paper B: {repeated} "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = convert_sciimpact_jsonl(source, output)

    assert manifest["pair_count"] == 2
    assert manifest["paper_count"] == 3
    papers = load_papers_jsonl(output / "papers.jsonl")
    shared = [paper for paper in papers if paper.title == "Shared Paper"]
    assert len(shared) == 1
    assert shared[0].paper_id.startswith("sciimpact:paper:")
    assert shared[0].metadata["sciimpact_pair_ids"] == ["sciimpact-1", "sciimpact-2"]


def test_export_sciimpact_dpo_jsonl_writes_chosen_and_rejected_answers(tmp_path):
    source = tmp_path / "sciimpact_sft.jsonl"
    output = tmp_path / "sciimpact_dpo.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "from": "system",
                        "value": (
                            "You are an impartial bibliometrics analyst. Your reply must be "
                            "either 'Paper A has more citations.' or "
                            "'Paper B has more citations.'"
                        ),
                    },
                    {
                        "from": "human",
                        "value": (
                            "Paper A: Title: Low Impact Abstract: Narrow. "
                            "Paper B: Title: High Impact Abstract: Broad. "
                            "Which paper has more citations?"
                        ),
                    },
                    {"from": "gpt", "value": "Paper B has more citations."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_sciimpact_dpo_jsonl(source, output)

    assert manifest["source_record_count"] == 1
    assert manifest["dpo_record_count"] == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["label_source"] == "sciimpact_sft"
    assert record["chosen"]["value"] == "Paper B has more citations."
    assert record["rejected"]["value"] == "Paper A has more citations."
    assert record["winner_id"].startswith("sciimpact:paper:")
    assert record["loser_id"].startswith("sciimpact:paper:")
    assert record["impact_dimension"] == "citation"
    assert record["conversations"][0]["from"] == "system"
    assert record["conversations"][1]["from"] == "human"


def test_export_sciimpact_dpo_jsonl_skips_invalid_rows_by_default(tmp_path):
    source = tmp_path / "sciimpact_invalid.jsonl"
    output = tmp_path / "sciimpact_dpo.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Paper A: Title: Broken Abstract: Missing B."},
                    {"from": "gpt", "value": "Paper B has more citations."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_sciimpact_dpo_jsonl(source, output, skip_invalid=True)

    assert manifest["source_record_count"] == 1
    assert manifest["dpo_record_count"] == 0
    assert manifest["skipped_record_count"] == 1
    assert output.read_text(encoding="utf-8") == ""


def test_convert_sciimpact_can_skip_invalid_sft_rows_with_manifest_report(tmp_path):
    source = tmp_path / "sciimpact_sft_invalid.jsonl"
    output = tmp_path / "sciimpact_sft_invalid"
    rows = [
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "Given two papers, identify which paper has more citations.",
                },
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Broken Row Abstract: This row is missing Paper B. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "Given two papers, identify which paper has more citations.",
                },
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Lower Impact Abstract: Lower impact text. "
                        "Paper B: Title: Higher Impact Abstract: Higher impact text. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = convert_sciimpact_jsonl(source, output, skip_invalid=True)

    assert manifest["source_record_count"] == 2
    assert manifest["pair_count"] == 1
    assert manifest["skipped_pair_count"] == 1
    assert manifest["skipped_pairs"][0]["source_line"] == 1
    assert "no supported A/B markers" in manifest["skipped_pairs"][0]["error"]

    labels = [
        json.loads(line)
        for line in (output / "pairwise_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output / "papers.jsonl")}
    assert papers[labels[0]["winner_id"]].title == "Higher Impact"


def test_convert_sciimpact_is_strict_by_default_for_invalid_rows(tmp_path):
    source = tmp_path / "sciimpact_sft_invalid.jsonl"
    output = tmp_path / "sciimpact_sft_invalid"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Paper A: Title: Broken Abstract: Missing B."},
                    {"from": "gpt", "value": "Paper B has more citations."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no supported A/B markers"):
        convert_sciimpact_jsonl(source, output)


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


def test_cli_benchmarks_convert_sciimpact_skips_invalid_rows_by_default(tmp_path):
    source = tmp_path / "sciimpact_invalid.jsonl"
    output = tmp_path / "sciimpact_invalid"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Paper A: Title: Broken Abstract: Missing B."},
                    {"from": "gpt", "value": "Paper B has more citations."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
    assert "Skipped 1 invalid SciImpact rows" in result.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_record_count"] == 1
    assert manifest["pair_count"] == 0
    assert manifest["skipped_pair_count"] == 1


def test_cli_benchmarks_export_sciimpact_dpo_writes_preference_file(tmp_path):
    source = tmp_path / "sciimpact_sft.jsonl"
    output = tmp_path / "sciimpact_dpo.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "system", "value": "Which paper has more citations?"},
                    {
                        "from": "human",
                        "value": (
                            "Paper A: Title: Low Impact Abstract: Narrow. "
                            "Paper B: Title: High Impact Abstract: Broad. "
                            "Which paper has more citations?"
                        ),
                    },
                    {"from": "gpt", "value": "Paper B has more citations."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmarks",
            "export-sciimpact-dpo",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 SciImpact DPO preference records" in result.output
    assert output.exists()
    assert (tmp_path / "sciimpact_dpo_manifest.json").exists()
