import json

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.llamafactory_export import (
    export_llamafactory_configs,
    export_sciimpact_pairwise_sft_llamafactory_configs,
)


def test_export_llamafactory_configs_writes_dataset_info_and_yaml_files(tmp_path):
    (tmp_path / "format_sft.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "proxy_positive_sft.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "dpo_preferences.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "context_hard_negative_dpo.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "method_context_contrastive_dpo.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "gold_hard_negative_dpo.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "merged_dpo_preferences.jsonl").write_text("{}\n", encoding="utf-8")

    export_llamafactory_configs(
        tmp_path,
        model_name_or_path="/models/local-impact-judge",
        output_root=tmp_path / "checkpoints",
    )

    dataset_info = json.loads((tmp_path / "llamafactory_dataset_info.json").read_text())
    assert dataset_info["impact_format_sft"]["file_name"] == "format_sft.jsonl"
    assert dataset_info["impact_proxy_positive_sft"]["formatting"] == "sharegpt"
    assert dataset_info["impact_dpo_preferences"]["ranking"] is True
    assert dataset_info["impact_context_hard_negative_dpo"]["file_name"] == (
        "context_hard_negative_dpo.jsonl"
    )
    assert dataset_info["impact_method_context_contrastive_dpo"]["ranking"] is True
    assert dataset_info["impact_gold_hard_negative_dpo"]["file_name"] == (
        "gold_hard_negative_dpo.jsonl"
    )
    assert dataset_info["impact_merged_dpo_preferences"]["ranking"] is True

    sft_yaml = (tmp_path / "llamafactory_sft.yaml").read_text(encoding="utf-8")
    assert "dataset: impact_format_sft,impact_proxy_positive_sft" in sft_yaml
    assert "model_name_or_path: /models/local-impact-judge" in sft_yaml

    dpo_yaml = (tmp_path / "llamafactory_dpo.yaml").read_text(encoding="utf-8")
    assert "stage: dpo" in dpo_yaml
    assert "dataset: impact_merged_dpo_preferences" in dpo_yaml


def test_export_sciimpact_pairwise_sft_llamafactory_configs(tmp_path):
    train_file = tmp_path / "train_pairwise_sft.jsonl"
    train_file.write_text(
        '{"messages":[{"role":"user","content":"Prompt"}],"target":"Answer: A"}\n',
        encoding="utf-8",
    )

    manifest = export_sciimpact_pairwise_sft_llamafactory_configs(
        train_file,
        tmp_path / "llamafactory",
        model_name_or_path="models/qwen",
        output_dir=tmp_path / "checkpoints" / "sciimpact-qwen-lora",
        max_samples=128,
    )

    dataset_info = json.loads(
        (tmp_path / "llamafactory" / "dataset_info.json").read_text(encoding="utf-8")
    )
    assert dataset_info["sciimpact_pairwise_sft"]["file_name"] == str(train_file)
    assert dataset_info["sciimpact_pairwise_sft"]["columns"] == {
        "messages": "conversations",
    }
    assert dataset_info["sciimpact_pairwise_sft"]["tags"] == {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
    }
    train_yaml = (tmp_path / "llamafactory" / "sciimpact_pairwise_sft_lora.yaml").read_text(
        encoding="utf-8"
    )
    assert "model_name_or_path: models/qwen" in train_yaml
    assert "dataset: sciimpact_pairwise_sft" in train_yaml
    assert "max_samples: 128" in train_yaml
    script = (tmp_path / "llamafactory" / "run_sciimpact_pairwise_sft_lora.sh").read_text(
        encoding="utf-8"
    )
    assert "llamafactory-cli train sciimpact_pairwise_sft_lora.yaml" in script
    assert manifest["train_file"] == str(train_file)


def test_cli_export_sciimpact_pairwise_sft_llamafactory_configs(monkeypatch, tmp_path):
    train_file = tmp_path / "train.jsonl"
    output = tmp_path / "configs"
    train_file.write_text('{"messages":[],"target":"Answer: A"}\n', encoding="utf-8")

    def fake_export(*args, **kwargs):
        output.mkdir(parents=True)
        return {"train_config_path": str(output / "config.yaml")}

    monkeypatch.setattr(
        cli,
        "export_sciimpact_pairwise_sft_llamafactory_configs",
        fake_export,
    )

    result = CliRunner().invoke(
        app,
        [
            "export-sciimpact-pairwise-sft-lora-config",
            "--train-file",
            str(train_file),
            "--output",
            str(output),
            "--model-name-or-path",
            "models/qwen",
            "--checkpoint-output",
            str(tmp_path / "ckpt"),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote SciImpact pairwise SFT LoRA config" in result.output
