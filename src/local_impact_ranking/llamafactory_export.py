from __future__ import annotations

import json
from pathlib import Path


def export_llamafactory_configs(
    finetuning_dir: str | Path,
    *,
    model_name_or_path: str,
    output_root: str | Path,
) -> None:
    destination = Path(finetuning_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "impact_format_sft": _sharegpt_dataset("format_sft.jsonl"),
        "impact_proxy_positive_sft": _sharegpt_dataset("proxy_positive_sft.jsonl"),
        "impact_dpo_preferences": {
            **_sharegpt_dataset("dpo_preferences.jsonl"),
            "ranking": True,
        },
        "impact_gold_hard_negative_dpo": {
            **_sharegpt_dataset("gold_hard_negative_dpo.jsonl"),
            "ranking": True,
        },
    }
    (destination / "llamafactory_dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "llamafactory_sft.yaml").write_text(
        _sft_yaml(model_name_or_path, Path(output_root)),
        encoding="utf-8",
    )
    (destination / "llamafactory_dpo.yaml").write_text(
        _dpo_yaml(model_name_or_path, Path(output_root)),
        encoding="utf-8",
    )


def _sharegpt_dataset(file_name: str) -> dict:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "conversations": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "human_tag": "human",
            "gpt_tag": "gpt",
        },
    }


def _sft_yaml(model_name_or_path: str, output_root: Path) -> str:
    return (
        "### Local Impact Ranking SFT config for LLaMA-Factory\n"
        "stage: sft\n"
        "do_train: true\n"
        f"model_name_or_path: {model_name_or_path}\n"
        "dataset_dir: .\n"
        "dataset: impact_format_sft,impact_proxy_positive_sft\n"
        "template: qwen\n"
        "finetuning_type: lora\n"
        f"output_dir: {output_root / 'impact-sft'}\n"
        "overwrite_cache: true\n"
        "overwrite_output_dir: true\n"
        "per_device_train_batch_size: 1\n"
        "gradient_accumulation_steps: 8\n"
        "learning_rate: 2.0e-5\n"
        "num_train_epochs: 1.0\n"
        "lr_scheduler_type: cosine\n"
        "warmup_ratio: 0.03\n"
        "bf16: true\n"
        "logging_steps: 10\n"
        "save_steps: 200\n"
    )


def _dpo_yaml(model_name_or_path: str, output_root: Path) -> str:
    return (
        "### Local Impact Ranking DPO config for LLaMA-Factory\n"
        "stage: dpo\n"
        "do_train: true\n"
        f"model_name_or_path: {model_name_or_path}\n"
        "dataset_dir: .\n"
        "dataset: impact_dpo_preferences,impact_gold_hard_negative_dpo\n"
        "template: qwen\n"
        "finetuning_type: lora\n"
        f"output_dir: {output_root / 'impact-dpo'}\n"
        "overwrite_cache: true\n"
        "overwrite_output_dir: true\n"
        "per_device_train_batch_size: 1\n"
        "gradient_accumulation_steps: 8\n"
        "learning_rate: 1.0e-6\n"
        "num_train_epochs: 1.0\n"
        "lr_scheduler_type: cosine\n"
        "warmup_ratio: 0.03\n"
        "bf16: true\n"
        "logging_steps: 10\n"
        "save_steps: 200\n"
    )
