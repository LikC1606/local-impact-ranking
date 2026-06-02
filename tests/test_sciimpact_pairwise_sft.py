import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_pairwise_sft import (
    export_sciimpact_pairwise_dpo,
    export_sciimpact_pairwise_sft,
    mine_sciimpact_hard_pairwise_examples,
)


def sft_row(answer: str = "A") -> dict:
    return {
        "conversations": [
            {"from": "system", "value": "Which paper has more citations?"},
            {
                "from": "human",
                "value": (
                    "Paper A: Title: Alpha Year: 2024 Abstract: Alpha abstract. "
                    "Paper B: Title: Beta Year: 2024 Abstract: Beta abstract. "
                    "Which paper has more citations?"
                ),
            },
            {"from": "gpt", "value": f"Paper {answer} has more citations."},
        ]
    }


def test_export_sciimpact_pairwise_sft_writes_prompt_aligned_records(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "pairwise_sft.jsonl"
    source.write_text(json.dumps(sft_row("B")) + "\n", encoding="utf-8")

    manifest = export_sciimpact_pairwise_sft(source, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["record_count"] == 1
    assert rows[0]["messages"][0]["role"] == "user"
    assert rows[0]["messages"][0]["content"].startswith(
        "Which paper has higher scholarly impact within this dataset?"
    )
    assert rows[0]["conversations"][0]["from"] == "human"
    assert rows[0]["conversations"][0]["value"] == rows[0]["messages"][0]["content"]
    assert rows[0]["conversations"][1] == {"from": "gpt", "value": "Answer: B"}
    assert "Paper A ID: sciimpact:paper:" in rows[0]["messages"][0]["content"]
    assert "Paper B ID: sciimpact:paper:" in rows[0]["messages"][0]["content"]
    assert rows[0]["target"] == "Answer: B"
    assert rows[0]["source_line"] == 1
    assert rows[0]["impact_dimension"] == "citation"
    assert rows[0]["item_type"] == "paper"


def test_export_sciimpact_pairwise_sft_can_preserve_source_task_prompt(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "pairwise_sft.jsonl"
    source.write_text(json.dumps(sft_row("B")) + "\n", encoding="utf-8")

    manifest = export_sciimpact_pairwise_sft(
        source,
        output,
        prompt_style="source_task",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    messages = rows[0]["messages"]
    assert manifest["record_count"] == 1
    assert manifest["prompt_family"] == "sciimpact_source_task_answer_v1"
    assert messages[0] == {"role": "system", "content": "Which paper has more citations?"}
    assert messages[1]["role"] == "user"
    assert "Which paper has more citations?" in messages[1]["content"]
    assert "Output exactly: Answer: A or Answer: B. Do not explain." in messages[1][
        "content"
    ]
    assert "Which paper has higher scholarly impact within this dataset?" not in messages[1][
        "content"
    ]
    assert rows[0]["conversations"] == [
        {"from": "system", "value": messages[0]["content"]},
        {"from": "human", "value": messages[1]["content"]},
        {"from": "gpt", "value": "Answer: B"},
    ]
    assert rows[0]["target"] == "Answer: B"
    assert rows[0]["impact_dimension"] == "citation"
    assert rows[0]["item_type"] == "paper"


def test_cli_export_sciimpact_pairwise_sft_writes_records(monkeypatch, tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "pairwise_sft.jsonl"
    source.write_text(json.dumps(sft_row("A")) + "\n", encoding="utf-8")

    captured = {}

    def fake_export(*args, **kwargs):
        captured.update(kwargs)
        output.write_text(json.dumps({"messages": [], "target": "Answer: A"}) + "\n")
        return {"record_count": 1, "skipped_record_count": 0}

    monkeypatch.setattr(cli, "export_sciimpact_pairwise_sft", fake_export)

    result = CliRunner().invoke(
        app,
        [
            "export-sciimpact-pairwise-sft",
            "--input",
            str(source),
            "--output",
            str(output),
            "--prompt-style",
            "source-task",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 SciImpact pairwise SFT records" in result.output
    assert captured["prompt_style"] == "source_task"


def test_cli_export_sciimpact_pairwise_dpo_writes_records(monkeypatch, tmp_path):
    source = tmp_path / "pairwise_sft.jsonl"
    output = tmp_path / "pairwise_dpo.jsonl"
    source.write_text(json.dumps({"messages": [], "target": "Answer: A"}) + "\n")

    def fake_export(*args, **kwargs):
        output.write_text(json.dumps({"chosen": {"value": "Answer: A"}}) + "\n")
        return {"record_count": 1, "skipped_record_count": 0}

    monkeypatch.setattr(cli, "export_sciimpact_pairwise_dpo", fake_export)

    result = CliRunner().invoke(
        app,
        [
            "export-sciimpact-pairwise-dpo",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 SciImpact pairwise DPO records" in result.output


def test_export_sciimpact_pairwise_dpo_flips_answer_target(tmp_path):
    source = tmp_path / "pairwise_sft.jsonl"
    output = tmp_path / "pairwise_dpo.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Prompt"}],
                "target": "Answer: B",
                "source_line": 9,
                "impact_dimension": "stars",
                "item_type": "github_repo",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = export_sciimpact_pairwise_dpo(source, output)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert row["conversations"] == [{"from": "human", "value": "Prompt"}]
    assert row["chosen"] == {"from": "gpt", "value": "Answer: B"}
    assert row["rejected"] == {"from": "gpt", "value": "Answer: A"}
    assert row["target_paper_id"] == "sciimpact-pairwise:9"
    assert row["impact_dimension"] == "stars"
    assert row["item_type"] == "github_repo"


def test_export_sciimpact_pairwise_dpo_preserves_source_task_system_prompt(tmp_path):
    source = tmp_path / "pairwise_sft.jsonl"
    output = tmp_path / "pairwise_dpo.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "System instruction"},
                    {"role": "user", "content": "User prompt"},
                ],
                "target": "Answer: A",
                "source_line": 11,
                "impact_dimension": "award",
                "item_type": "paper",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_sciimpact_pairwise_dpo(source, output)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["conversations"] == [
        {"from": "system", "value": "System instruction"},
        {"from": "human", "value": "User prompt"},
    ]
    assert row["chosen"] == {"from": "gpt", "value": "Answer: A"}
    assert row["rejected"] == {"from": "gpt", "value": "Answer: B"}


def test_mine_sciimpact_hard_pairwise_examples_writes_wrong_and_conflict_records(
    tmp_path,
):
    source = tmp_path / "sft.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(sft_row("A")),
                json.dumps(sft_row("B")),
                json.dumps(sft_row("A")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    flip_rows = tmp_path / "flip_rows.jsonl"
    flip_rows.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_line": 1,
                        "selected_outcome": "wrong",
                        "consistent": True,
                        "original_outcome": "wrong",
                        "flipped_mapped_outcome": "wrong",
                        "selected_source": "consistent",
                        "impact_dimension": "citation",
                        "item_type": "paper",
                    }
                ),
                json.dumps(
                    {
                        "source_line": 2,
                        "selected_outcome": "correct",
                        "consistent": False,
                        "original_outcome": "correct",
                        "flipped_mapped_outcome": "wrong",
                        "selected_source": "original_confidence",
                        "impact_dimension": "citation",
                        "item_type": "paper",
                    }
                ),
                json.dumps(
                    {
                        "source_line": 3,
                        "selected_outcome": "correct",
                        "consistent": True,
                        "original_outcome": "correct",
                        "flipped_mapped_outcome": "correct",
                        "selected_source": "consistent",
                        "impact_dimension": "citation",
                        "item_type": "paper",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "hard"

    manifest = mine_sciimpact_hard_pairwise_examples(source, flip_rows, output)

    sft_rows = [
        json.loads(line)
        for line in (output / "train_pairwise_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    dpo_rows = [
        json.loads(line)
        for line in (output / "train_pairwise_dpo.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["record_count"] == 2
    assert manifest["reason_counts"] == {
        "selected_wrong": 1,
        "position_inconsistent": 1,
    }
    assert [row["source_line"] for row in sft_rows] == [1, 2]
    assert sft_rows[0]["hard_example_reasons"] == ["selected_wrong"]
    assert sft_rows[1]["hard_example_reasons"] == ["position_inconsistent"]
    assert dpo_rows[0]["chosen"]["value"] == "Answer: A"
    assert dpo_rows[1]["chosen"]["value"] == "Answer: B"
    assert dpo_rows[1]["label_source"] == "sciimpact_hard_pairwise_mining"


def test_mine_sciimpact_hard_pairwise_examples_warns_for_test_source_manifest(
    tmp_path,
):
    work = Path(tempfile.mkdtemp(prefix="li-hard-manifest-"))
    sample_dir = work / "sample"
    sample_dir.mkdir()
    source = sample_dir / "sciimpact_sft_stratified.jsonl"
    source.write_text(json.dumps(sft_row("A")) + "\n", encoding="utf-8")
    (sample_dir / "sciimpact_stratified_sample_manifest.json").write_text(
        json.dumps({"source_path": "data/sciimpact/paper_impact_sft_test_clean.jsonl"})
        + "\n",
        encoding="utf-8",
    )
    flip_rows = work / "flip_rows.jsonl"
    flip_rows.write_text(
        json.dumps(
            {
                "source_line": 1,
                "selected_outcome": "wrong",
                "consistent": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = mine_sciimpact_hard_pairwise_examples(
        source,
        flip_rows,
        work / "hard",
    )

    assert "test split" in manifest["training_leakage_warning"]


def test_cli_mine_sciimpact_hard_examples_writes_manifest(monkeypatch, tmp_path):
    source = tmp_path / "sft.jsonl"
    flip_rows = tmp_path / "flip_rows.jsonl"
    output = tmp_path / "hard"
    source.write_text(json.dumps(sft_row("A")) + "\n", encoding="utf-8")
    flip_rows.write_text(json.dumps({"source_line": 1}) + "\n", encoding="utf-8")

    def fake_mine(*args, **kwargs):
        output.mkdir()
        return {"record_count": 3, "skipped_record_count": 0}

    monkeypatch.setattr(cli, "mine_sciimpact_hard_pairwise_examples", fake_mine)

    result = CliRunner().invoke(
        app,
        [
            "mine-sciimpact-hard-examples",
            "--sft",
            str(source),
            "--flip-rows",
            str(flip_rows),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 3 SciImpact hard pairwise examples" in result.output
