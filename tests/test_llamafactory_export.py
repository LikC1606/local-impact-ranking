import json

from local_impact_ranking.llamafactory_export import export_llamafactory_configs


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
