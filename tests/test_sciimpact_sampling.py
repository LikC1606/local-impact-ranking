import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_model_eval import write_sciimpact_stratified_sample


def sft_row(dimension: str, answer: str = "Paper A") -> dict:
    return {
        "conversations": [
            {"from": "system", "value": f"Which paper has more {dimension}?"},
            {
                "from": "human",
                "value": (
                    "Paper A: Title: High Year: 2024 Abstract: Strong. "
                    "Paper B: Title: Low Year: 2024 Abstract: Weak. "
                    f"Which paper has more {dimension}?"
                ),
            },
            {"from": "gpt", "value": f"{answer} has more {dimension}."},
        ]
    }


def sft_item_row(item_type: str, dimension: str, answer: str = "A") -> dict:
    labels = {
        "paper": ("Paper A:", "Paper B:", "Paper"),
        "dataset": ("Dataset A:", "Dataset B:", "Dataset"),
        "github_repo": ("GitHub repo A README:", "GitHub repo B README:", "GitHub repo"),
        "model": ("Model A:", "Model B:", "Model"),
    }
    marker_a, marker_b, answer_label = labels[item_type]
    return {
        "conversations": [
            {"from": "system", "value": f"Which {answer_label.lower()} has more {dimension}?"},
            {
                "from": "human",
                "value": (
                    f"{marker_a} Name: High {item_type} for {dimension}. "
                    f"{marker_b} Name: Low {item_type} for {dimension}. "
                    f"Which {answer_label.lower()} has more {dimension}?"
                ),
            },
            {"from": "gpt", "value": f"{answer_label} {answer} has more {dimension}."},
        ]
    }


def test_write_sciimpact_stratified_sample_balances_dimensions_and_records_lineage(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    rows = [
        sft_row("citation"),
        sft_row("citation", answer="Paper B"),
        sft_row("citation"),
        sft_row("patent"),
        sft_row("patent", answer="Paper B"),
        sft_row("download"),
        sft_row("download", answer="Paper B"),
        sft_row("download"),
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_stratified_sample(
        source,
        output,
        per_dimension=2,
        seed=7,
    )

    sampled_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sampled_rows) == 6
    assert manifest["sample_count"] == 6
    assert manifest["by_dimension"]["citation"]["sampled_count"] == 2
    assert manifest["by_dimension"]["patent"]["sampled_count"] == 2
    assert manifest["by_dimension"]["downloads"]["sampled_count"] == 2
    assert manifest["sampled_records"][0]["sample_line"] == 1
    assert "source_line" in manifest["sampled_records"][0]


def test_write_sciimpact_stratified_sample_can_filter_dimensions(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_row("citation")),
                json.dumps(sft_row("patent")),
                json.dumps(sft_row("download")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_stratified_sample(
        source,
        output,
        per_dimension=5,
        dimensions=["citation", "downloads"],
    )

    assert manifest["sample_count"] == 2
    assert sorted(manifest["by_dimension"]) == ["citation", "downloads"]


def test_write_sciimpact_stratified_sample_accepts_exported_pairwise_sft_metadata(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    rows = [
        {
            "messages": [{"role": "user", "content": "Prompt without original markers"}],
            "target": "Answer: A",
            "impact_dimension": "citation",
            "item_type": "paper",
        },
        {
            "messages": [{"role": "user", "content": "Prompt without original markers"}],
            "target": "Answer: B",
            "impact_dimension": "citation",
            "item_type": "dataset",
        },
        {
            "messages": [{"role": "user", "content": "Prompt without original markers"}],
            "target": "Answer: A",
            "impact_dimension": "citation",
            "item_type": "dataset",
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_stratified_sample(
        source,
        output,
        per_dimension=1,
        item_type_weights={"dataset": 2},
        seed=13,
    )

    assert manifest["sample_count"] == 3
    assert manifest["skipped_invalid_count"] == 0
    assert manifest["by_dimension"]["citation"]["sampled_count"] == 3
    assert manifest["by_item_type"]["paper"]["sampled_count"] == 1
    assert manifest["by_item_type"]["dataset"]["sampled_count"] == 2


def test_write_sciimpact_stratified_sample_can_oversample_weak_dimensions(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_row("citation")),
                json.dumps(sft_row("citation", answer="Paper B")),
                json.dumps(sft_row("citation")),
                json.dumps(sft_row("patent")),
                json.dumps(sft_row("patent", answer="Paper B")),
                json.dumps(sft_row("patent")),
                json.dumps(sft_row("download")),
                json.dumps(sft_row("download", answer="Paper B")),
                json.dumps(sft_row("download")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_stratified_sample(
        source,
        output,
        per_dimension=1,
        dimension_weights={"patent": 3, "downloads": 2},
        seed=5,
    )

    assert manifest["sample_count"] == 6
    assert manifest["dimension_weights"] == {"patent": 3, "downloads": 2}
    assert manifest["by_dimension"]["citation"]["sampled_count"] == 1
    assert manifest["by_dimension"]["patent"]["target_count"] == 3
    assert manifest["by_dimension"]["patent"]["sampled_count"] == 3
    assert manifest["by_dimension"]["downloads"]["target_count"] == 2
    assert manifest["by_dimension"]["downloads"]["sampled_count"] == 2


def test_write_sciimpact_stratified_sample_can_oversample_item_types(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_item_row("paper", "citation")),
                json.dumps(sft_item_row("paper", "citation", answer="B")),
                json.dumps(sft_item_row("dataset", "citation")),
                json.dumps(sft_item_row("dataset", "citation", answer="B")),
                json.dumps(sft_item_row("dataset", "citation")),
                json.dumps(sft_item_row("github_repo", "stars")),
                json.dumps(sft_item_row("github_repo", "stars", answer="B")),
                json.dumps(sft_item_row("github_repo", "stars")),
                json.dumps(sft_item_row("paper", "stars")),
                json.dumps(sft_item_row("paper", "stars", answer="B")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_stratified_sample(
        source,
        output,
        per_dimension=1,
        item_type_weights={"dataset": 2, "github_repo": 3},
        seed=11,
    )

    assert manifest["sample_count"] == 7
    assert manifest["item_type_weights"] == {"dataset": 2, "github_repo": 3}
    assert manifest["by_dimension"]["citation"]["sampled_count"] == 3
    assert manifest["by_dimension"]["stars"]["sampled_count"] == 4
    assert manifest["by_item_type"]["paper"]["sampled_count"] == 2
    assert manifest["by_item_type"]["dataset"]["target_count"] == 2
    assert manifest["by_item_type"]["dataset"]["sampled_count"] == 2
    assert manifest["by_item_type"]["github_repo"]["target_count"] == 3
    assert manifest["by_item_type"]["github_repo"]["sampled_count"] == 3
    assert {row["item_type"] for row in manifest["sampled_records"]} == {
        "paper",
        "dataset",
        "github_repo",
    }


def test_cli_sample_sciimpact_sft_writes_stratified_sample(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_row("citation")),
                json.dumps(sft_row("citation", answer="Paper B")),
                json.dumps(sft_row("patent")),
                json.dumps(sft_row("download")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sample-sciimpact-sft",
            "--sft",
            str(source),
            "--output",
            str(output),
            "--per-dimension",
            "1",
            "--seed",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 3 stratified SciImpact examples" in result.output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_sample_sciimpact_sft_accepts_dimension_weights(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_row("citation")),
                json.dumps(sft_row("patent")),
                json.dumps(sft_row("patent", answer="Paper B")),
                json.dumps(sft_row("download")),
                json.dumps(sft_row("download", answer="Paper B")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sample-sciimpact-sft",
            "--sft",
            str(source),
            "--output",
            str(output),
            "--per-dimension",
            "1",
            "--dimension-weight",
            "patent=2",
            "--dimension-weight",
            "downloads=2",
        ],
    )

    assert result.exit_code == 0
    manifest = json.loads(
        (tmp_path / "sciimpact_stratified_sample_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["sample_count"] == 5
    assert manifest["by_dimension"]["patent"]["sampled_count"] == 2
    assert manifest["by_dimension"]["downloads"]["sampled_count"] == 2


def test_cli_sample_sciimpact_sft_accepts_item_type_weights(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_item_row("paper", "citation")),
                json.dumps(sft_item_row("dataset", "citation")),
                json.dumps(sft_item_row("dataset", "citation", answer="B")),
                json.dumps(sft_item_row("github_repo", "stars")),
                json.dumps(sft_item_row("github_repo", "stars", answer="B")),
                json.dumps(sft_item_row("github_repo", "stars")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sample-sciimpact-sft",
            "--sft",
            str(source),
            "--output",
            str(output),
            "--per-dimension",
            "1",
            "--item-type-weight",
            "dataset=2",
            "--item-type-weight",
            "github=3",
        ],
    )

    assert result.exit_code == 0
    manifest = json.loads(
        (tmp_path / "sciimpact_stratified_sample_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["sample_count"] == 6
    assert manifest["item_type_weights"] == {"dataset": 2, "github_repo": 3}
    assert manifest["by_item_type"]["dataset"]["sampled_count"] == 2
    assert manifest["by_item_type"]["github_repo"]["sampled_count"] == 3
