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
        "impact_context_hard_negative_dpo": {
            **_sharegpt_dataset("context_hard_negative_dpo.jsonl"),
            "ranking": True,
        },
        "impact_method_context_contrastive_dpo": {
            **_sharegpt_dataset("method_context_contrastive_dpo.jsonl"),
            "ranking": True,
        },
        "impact_gold_hard_negative_dpo": {
            **_sharegpt_dataset("gold_hard_negative_dpo.jsonl"),
            "ranking": True,
        },
        "impact_merged_dpo_preferences": {
            **_sharegpt_dataset("merged_dpo_preferences.jsonl"),
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


def export_sciimpact_pairwise_sft_llamafactory_configs(
    train_file: str | Path,
    config_dir: str | Path,
    *,
    model_name_or_path: str,
    output_dir: str | Path,
    max_samples: int | None = None,
) -> dict:
    train_path = Path(train_file)
    destination = Path(config_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "sciimpact_pairwise_sft": {
            "file_name": str(train_path),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    (destination / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n",
        encoding="utf-8",
    )
    yaml_text = _sciimpact_pairwise_sft_yaml(
        model_name_or_path=model_name_or_path,
        output_dir=Path(output_dir),
        max_samples=max_samples,
    )
    (destination / "sciimpact_pairwise_sft_lora.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )
    script_path = destination / "run_sciimpact_pairwise_sft_lora.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "llamafactory-cli train sciimpact_pairwise_sft_lora.yaml\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    manifest = {
        "train_file": str(train_path),
        "output_dir": str(destination),
        "model_name_or_path": model_name_or_path,
        "checkpoint_output_dir": str(output_dir),
        "max_samples": max_samples,
        "dataset_info_path": str(destination / "dataset_info.json"),
        "train_config_path": str(destination / "sciimpact_pairwise_sft_lora.yaml"),
        "run_script_path": str(script_path),
    }
    (destination / "sciimpact_pairwise_sft_lora_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
        "dataset: impact_merged_dpo_preferences\n"
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


def _sciimpact_pairwise_sft_yaml(
    *,
    model_name_or_path: str,
    output_dir: Path,
    max_samples: int | None,
) -> str:
    lines = [
        "### SciImpact pairwise judge SFT config for LLaMA-Factory",
        "stage: sft",
        "do_train: true",
        f"model_name_or_path: {model_name_or_path}",
        "dataset_dir: .",
        "dataset: sciimpact_pairwise_sft",
        "template: qwen",
        "finetuning_type: lora",
        f"output_dir: {output_dir}",
        "overwrite_cache: true",
        "overwrite_output_dir: true",
        "cutoff_len: 4096",
        "preprocessing_num_workers: 8",
        "per_device_train_batch_size: 1",
        "gradient_accumulation_steps: 16",
        "learning_rate: 2.0e-5",
        "num_train_epochs: 1.0",
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.03",
        "bf16: true",
        "logging_steps: 10",
        "save_steps: 500",
        "lora_rank: 16",
        "lora_alpha: 32",
        "lora_dropout: 0.05",
        "lora_target: all",
    ]
    if max_samples is not None:
        lines.append(f"max_samples: {max_samples}")
    return "\n".join(lines) + "\n"
