import json

from local_impact_ranking.baselines import compute_bibliometric_baselines
from local_impact_ranking.finetune_export import (
    export_finetuning_datasets,
    finetuning_readiness_report,
    merge_dpo_preference_records,
    make_context_hard_negative_dpo_records,
    make_format_sft_records,
    make_gold_hard_negative_dpo_records,
    make_method_context_contrastive_dpo_records,
    make_pairwise_preference_records,
    make_proxy_positive_sft_records,
    render_finetuning_readiness_markdown,
)
from local_impact_ranking.judge import parse_judge_output
from local_impact_ranking.models import Paper
from local_impact_ranking.proxy_evaluation import evaluate_proxy_quality


def sample_papers():
    return [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Target",
                "abstract": "Target abstract",
                "references": [
                    {"reference_id": "p2", "title": "Supported Reference"},
                    {"reference_id": "p3", "title": "Unsupported Reference"},
                ],
                "citation_contexts": [
                    {
                        "reference_id": "p2",
                        "sentence": "We build on Supported Reference for the method.",
                        "section": "Method",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Supported Reference",
                "citation_count": 100,
                "references": [],
            }
        ),
    ]


def sample_extra_target_paper():
    return Paper.model_validate(
        {
            "paper_id": "p4",
            "title": "Second Target",
            "abstract": "Second abstract",
            "references": [
                {"reference_id": "p5", "title": "Second Supported Reference"},
                {"reference_id": "p6", "title": "Second Unsupported Reference"},
            ],
            "citation_contexts": [
                {
                    "reference_id": "p5",
                    "sentence": "We build on Second Supported Reference for the method.",
                    "section": "Method",
                }
            ],
        }
    )


def sample_result(paper):
    return parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {"reference_id": "p2", "rank": 1, "confidence": 0.9, "rationale": "supported"},
                    {
                        "reference_id": "p3",
                        "rank": 2,
                        "confidence": 0.8,
                        "rationale": "unsupported",
                    },
                ],
            }
        ),
        paper,
    )


def proxy_report(papers, result):
    return evaluate_proxy_quality(
        papers,
        [result],
        compute_bibliometric_baselines(papers),
        top_n=1,
    )


def extra_result(paper):
    return parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p4",
                "selected_references": [
                    {
                        "reference_id": "p5",
                        "rank": 1,
                        "confidence": 0.9,
                        "rationale": "supported",
                    },
                    {
                        "reference_id": "p6",
                        "rank": 2,
                        "confidence": 0.8,
                        "rationale": "unsupported",
                    },
                ],
            }
        ),
        paper,
    )


def sft_record(target_paper_id, *, label_source="model_audit"):
    return {
        "target_paper_id": target_paper_id,
        "messages": [
            {"role": "user", "content": "prompt"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "target_paper_id": target_paper_id,
                        "selected_references": [
                            {"reference_id": f"{target_paper_id}-r1", "rank": 1}
                        ],
                    }
                ),
            },
        ],
        "label_source": label_source,
    }


def dpo_record(target_paper_id, *, label_source="proxy_supported_vs_unsupported"):
    return {
        "target_paper_id": target_paper_id,
        "chosen_reference_id": f"{target_paper_id}-chosen",
        "rejected_reference_id": f"{target_paper_id}-rejected",
        "label_source": label_source,
    }


def test_make_format_sft_records_exports_strict_json_targets():
    papers = sample_papers()
    result = sample_result(papers[0])

    records = make_format_sft_records(papers, [result], top_k=2)

    assert len(records) == 1
    assert records[0]["task"] == "format_sft"
    assert "Candidate references" in records[0]["messages"][0]["content"]
    target = json.loads(records[0]["messages"][1]["content"])
    assert target["selected_references"][0]["reference_id"] == "p2"


def test_make_format_sft_records_skips_empty_valid_selection_targets():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [{"reference_id": "p2", "title": "Candidate"}],
        }
    )
    result = parse_judge_output(
        json.dumps({"target_paper_id": "p1", "selected_references": []}),
        paper,
    )

    records = make_format_sft_records([paper], [result], top_k=2)

    assert records == []


def test_make_proxy_positive_sft_records_filters_unsupported_selections():
    papers = sample_papers()
    result = sample_result(papers[0])

    records = make_proxy_positive_sft_records(
        papers,
        [result],
        proxy_report(papers, result),
        top_k=2,
    )

    target = json.loads(records[0]["messages"][1]["content"])
    assert [item["reference_id"] for item in target["selected_references"]] == ["p2"]


def test_make_pairwise_preference_records_prefers_proxy_supported_selection():
    papers = sample_papers()
    result = sample_result(papers[0])

    records = make_pairwise_preference_records(
        papers,
        [result],
        proxy_report(papers, result),
    )

    assert len(records) == 1
    assert records[0]["task"] == "pairwise_preference"
    assert records[0]["chosen_reference_id"] == "p2"
    assert records[0]["rejected_reference_id"] == "p3"


def test_make_gold_hard_negative_dpo_records_prefers_gold_over_model_miss():
    papers = sample_papers()
    result = sample_result(papers[0])
    gold_report = {
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["p2"],
                "selected_reference_ids": ["p3"],
                "selected_positive_reference_ids": [],
                "hit": False,
            }
        ]
    }

    records = make_gold_hard_negative_dpo_records(
        papers,
        [result],
        gold_report,
    )

    assert len(records) == 1
    assert records[0]["target_paper_id"] == "p1"
    assert records[0]["chosen_reference_id"] == "p2"
    assert records[0]["rejected_reference_id"] == "p3"
    assert records[0]["label_source"] == "gold_hard_negative"
    assert records[0]["chosen"]["value"] == "p2"
    assert records[0]["rejected"]["value"] == "p3"
    assert "Candidate A" in records[0]["conversations"][0]["value"]


def test_make_context_hard_negative_dpo_records_prefers_context_supported_missed_reference():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "abstract": "Target abstract",
            "references": [
                {"reference_id": "p2", "title": "Context Supported Method"},
                {"reference_id": "p3", "title": "Unsupported Background"},
            ],
            "citation_contexts": [
                {
                    "reference_id": "p2",
                    "sentence": "We build on this method for our model.",
                    "section": "Method",
                }
            ],
        }
    )
    result = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {
                        "reference_id": "p3",
                        "rank": 1,
                        "confidence": 0.9,
                        "rationale": "generic background",
                    }
                ],
            }
        ),
        paper,
    )

    records = make_context_hard_negative_dpo_records([paper], [result])

    assert len(records) == 1
    assert records[0]["target_paper_id"] == "p1"
    assert records[0]["chosen_reference_id"] == "p2"
    assert records[0]["rejected_reference_id"] == "p3"
    assert records[0]["label_source"] == "context_hard_negative"
    assert records[0]["chosen"]["value"] == "p2"
    assert records[0]["rejected"]["value"] == "p3"
    assert "Context evidence" in records[0]["conversations"][0]["value"]


def test_make_method_context_contrastive_dpo_records_prefers_method_context_over_no_context():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "abstract": "Target abstract",
            "references": [
                {"reference_id": "p2", "title": "Method Dataset"},
                {"reference_id": "p3", "title": "Background Survey"},
            ],
            "citation_contexts": [
                {
                    "reference_id": "p2",
                    "sentence": "We compare against this baseline dataset.",
                    "section": "Experiments",
                }
            ],
        }
    )

    records = make_method_context_contrastive_dpo_records([paper])

    assert len(records) == 1
    assert records[0]["target_paper_id"] == "p1"
    assert records[0]["chosen_reference_id"] == "p2"
    assert records[0]["rejected_reference_id"] == "p3"
    assert records[0]["label_source"] == "method_context_contrastive"
    assert "Context evidence" in records[0]["conversations"][0]["value"]


def test_export_finetuning_datasets_writes_jsonl_files(tmp_path):
    papers = sample_papers()
    result = sample_result(papers[0])
    gold_report = {
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["p2"],
                "selected_reference_ids": ["p3"],
                "selected_positive_reference_ids": [],
                "hit": False,
            }
        ]
    }

    export_finetuning_datasets(
        papers,
        [result],
        proxy_report(papers, result),
        tmp_path,
        top_k=2,
        gold_report=gold_report,
    )

    assert (tmp_path / "format_sft.jsonl").exists()
    assert (tmp_path / "proxy_positive_sft.jsonl").exists()
    assert (tmp_path / "pairwise_preferences.jsonl").exists()
    assert (tmp_path / "dpo_preferences.jsonl").exists()
    assert (tmp_path / "context_hard_negative_dpo.jsonl").exists()
    assert (tmp_path / "method_context_contrastive_dpo.jsonl").exists()
    assert (tmp_path / "gold_hard_negative_dpo.jsonl").exists()
    assert (tmp_path / "merged_dpo_preferences.jsonl").exists()
    first = json.loads((tmp_path / "format_sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["task"] == "format_sft"
    dpo = json.loads((tmp_path / "dpo_preferences.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert dpo["conversations"][0]["from"] == "human"
    assert dpo["chosen"]["from"] == "gpt"
    assert dpo["rejected"]["from"] == "gpt"
    hard_negative = json.loads(
        (tmp_path / "gold_hard_negative_dpo.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert hard_negative["chosen_reference_id"] == "p2"
    readiness = json.loads(
        (tmp_path / "finetuning_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["dataset_counts"]["format_sft"] == 1
    assert readiness["dataset_counts"]["gold_hard_negative_dpo"] == 1
    assert "merged_dpo_preferences" in readiness["dataset_counts"]
    assert "context_hard_negative_dpo" in readiness["dataset_counts"]
    assert "method_context_contrastive_dpo" in readiness["dataset_counts"]
    assert readiness["recommendation"]["next_step"] in {
        "collect_more_examples",
        "sft_smoke",
        "dpo_smoke",
        "dpo_train",
    }
    readiness_md = (tmp_path / "finetuning_readiness.md").read_text(encoding="utf-8")
    assert "# Finetuning Readiness" in readiness_md


def test_export_finetuning_datasets_can_filter_target_paper_ids(tmp_path):
    papers = sample_papers() + [sample_extra_target_paper()]
    result = sample_result(papers[0])
    extra = extra_result(papers[-1])
    proxy = proxy_report(papers, result)
    proxy_extra = evaluate_proxy_quality(
        papers,
        [result, extra],
        compute_bibliometric_baselines(papers),
        top_n=1,
    )

    export_finetuning_datasets(
        papers,
        [result, extra],
        proxy_extra,
        tmp_path,
        top_k=2,
        target_paper_ids={"p1"},
    )

    format_records = [
        json.loads(line)
        for line in (tmp_path / "format_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dpo_records = [
        json.loads(line)
        for line in (tmp_path / "dpo_preferences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [record["target_paper_id"] for record in format_records] == ["p1"]
    assert [record["target_paper_id"] for record in dpo_records] == ["p1"]
    assert not (tmp_path / "finetuning_manifest.json").exists()


def test_finetuning_readiness_report_recommends_training_stage_by_counts():
    report = finetuning_readiness_report(
        format_sft_records=[sft_record(f"p{i}") for i in range(80)],
        proxy_positive_sft_records=[
            sft_record(f"p{i}", label_source="proxy_positive") for i in range(50)
        ],
        pairwise_preference_records=[{"target_paper_id": f"p{i}"} for i in range(20)],
        dpo_preference_records=[dpo_record(f"p{i}") for i in range(20)],
        context_hard_negative_dpo_records=[
            dpo_record(f"c{i}", label_source="context_hard_negative")
            for i in range(10)
        ],
        method_context_contrastive_dpo_records=[
            dpo_record(f"m{i}", label_source="method_context_contrastive")
            for i in range(15)
        ],
        gold_hard_negative_dpo_records=[
            dpo_record(f"g{i}", label_source="gold_hard_negative")
            for i in range(5)
        ],
        merged_dpo_preference_records=[
            dpo_record(f"x{i}", label_source="context_hard_negative")
            for i in range(40)
        ],
    )

    assert report["dataset_counts"]["format_sft"] == 80
    assert report["dataset_counts"]["proxy_positive_sft"] == 50
    assert report["summary"]["sft_record_count"] == 130
    assert report["summary"]["effective_sft_record_count"] == 130
    assert report["summary"]["dpo_record_count"] == 50
    assert report["summary"]["merged_dpo_record_count"] == 40
    assert report["summary"]["effective_dpo_record_count"] == 40
    assert report["summary"]["unique_training_targets"] == 110
    assert report["readiness"]["ready_for_sft_smoke"] is True
    assert report["readiness"]["ready_for_dpo_smoke"] is True
    assert report["readiness"]["ready_for_full_dpo"] is False
    assert report["recommendation"]["next_step"] == "dpo_smoke"

    markdown = render_finetuning_readiness_markdown(report)
    assert "| dpo_record_count | 50 |" in markdown
    assert "| merged_dpo_record_count | 40 |" in markdown
    assert "| effective_dpo_record_count | 40 |" in markdown
    assert "| next_step | dpo_smoke |" in markdown


def test_finetuning_readiness_uses_merged_dpo_count_for_dpo_gates():
    report = finetuning_readiness_report(
        format_sft_records=[sft_record(f"p{i}") for i in range(80)],
        proxy_positive_sft_records=[],
        pairwise_preference_records=[],
        dpo_preference_records=[dpo_record(f"raw{i}") for i in range(50)],
        context_hard_negative_dpo_records=[],
        method_context_contrastive_dpo_records=[],
        gold_hard_negative_dpo_records=[],
        merged_dpo_preference_records=[
            dpo_record(f"m{i}", label_source="context_hard_negative")
            for i in range(5)
        ],
    )

    assert report["summary"]["dpo_record_count"] == 50
    assert report["summary"]["merged_dpo_record_count"] == 5
    assert report["summary"]["effective_dpo_record_count"] == 5
    assert report["readiness"]["ready_for_sft_smoke"] is True
    assert report["readiness"]["ready_for_dpo_smoke"] is False
    assert report["recommendation"]["next_step"] == "sft_smoke"


def test_finetuning_readiness_uses_effective_nonempty_sft_count_for_sft_gate():
    empty_sft_records = [
        {
            "target_paper_id": f"p{i}",
            "messages": [
                {"role": "user", "content": "prompt"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"target_paper_id": f"p{i}", "selected_references": []}
                    ),
                },
            ],
            "label_source": "model_audit",
        }
        for i in range(25)
    ]
    report = finetuning_readiness_report(
        format_sft_records=empty_sft_records,
        proxy_positive_sft_records=[],
        pairwise_preference_records=[],
        dpo_preference_records=[],
        context_hard_negative_dpo_records=[],
        method_context_contrastive_dpo_records=[],
        gold_hard_negative_dpo_records=[],
        merged_dpo_preference_records=[],
    )

    assert report["summary"]["sft_record_count"] == 25
    assert report["summary"]["effective_sft_record_count"] == 0
    assert report["readiness"]["ready_for_sft_smoke"] is False
    assert report["recommendation"]["next_step"] == "fix_training_data"


def test_merge_dpo_preference_records_deduplicates_and_caps_per_target():
    proxy = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r1",
            "rejected_reference_id": "r2",
            "label_source": "proxy_supported_vs_unsupported",
        },
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r3",
            "rejected_reference_id": "r4",
            "label_source": "proxy_supported_vs_unsupported",
        },
    ]
    context = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r1",
            "rejected_reference_id": "r2",
            "label_source": "context_hard_negative",
        }
    ]
    method = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r5",
            "rejected_reference_id": "r6",
            "label_source": "method_context_contrastive",
        }
    ]
    gold = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r7",
            "rejected_reference_id": "r8",
            "label_source": "gold_hard_negative",
        }
    ]

    merged = merge_dpo_preference_records(
        proxy,
        context_hard_negative_records=context,
        method_context_contrastive_records=method,
        gold_hard_negative_records=gold,
        max_pairs_per_target=3,
    )

    assert [record["label_source"] for record in merged] == [
        "gold_hard_negative",
        "context_hard_negative",
        "method_context_contrastive",
    ]
    assert len(merged) == 3


def test_merge_dpo_preference_records_drops_lower_priority_reverse_conflicts():
    proxy = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r2",
            "rejected_reference_id": "r1",
            "label_source": "proxy_supported_vs_unsupported",
        }
    ]
    context = [
        {
            "target_paper_id": "p1",
            "chosen_reference_id": "r1",
            "rejected_reference_id": "r2",
            "label_source": "context_hard_negative",
        }
    ]

    merged = merge_dpo_preference_records(
        proxy,
        context_hard_negative_records=context,
    )

    assert [(record["chosen_reference_id"], record["rejected_reference_id"]) for record in merged] == [
        ("r1", "r2")
    ]


def test_merge_dpo_preference_records_skips_self_pairs():
    merged = merge_dpo_preference_records(
        [
            {
                "target_paper_id": "p1",
                "chosen_reference_id": "r1",
                "rejected_reference_id": "r1",
                "label_source": "proxy_supported_vs_unsupported",
            },
            {
                "target_paper_id": "p1",
                "chosen_reference_id": "r2",
                "rejected_reference_id": "r3",
                "label_source": "proxy_supported_vs_unsupported",
            },
        ]
    )

    assert [(record["chosen_reference_id"], record["rejected_reference_id"]) for record in merged] == [
        ("r2", "r3")
    ]


def test_finetuning_readiness_reports_preference_training_data_quality():
    report = finetuning_readiness_report(
        format_sft_records=[{"target_paper_id": f"sft{i}"} for i in range(20)],
        proxy_positive_sft_records=[],
        pairwise_preference_records=[],
        dpo_preference_records=[
            {
                "target_paper_id": "p1",
                "chosen_reference_id": "r1",
                "rejected_reference_id": "r2",
                "label_source": "proxy_supported_vs_unsupported",
            },
            {
                "target_paper_id": "p1",
                "chosen_reference_id": "r2",
                "rejected_reference_id": "r1",
                "label_source": "proxy_supported_vs_unsupported",
            },
            {
                "target_paper_id": "p1",
                "chosen_reference_id": "r3",
                "rejected_reference_id": "r3",
                "label_source": "proxy_supported_vs_unsupported",
            },
        ],
        context_hard_negative_dpo_records=[],
        method_context_contrastive_dpo_records=[],
        gold_hard_negative_dpo_records=[],
        merged_dpo_preference_records=[
            {
                "target_paper_id": "p1",
                "chosen_reference_id": f"r{i}",
                "rejected_reference_id": f"n{i}",
                "label_source": "context_hard_negative",
            }
            for i in range(8)
        ],
    )

    raw_quality = report["training_data_quality"]["raw_dpo"]
    effective_quality = report["training_data_quality"]["effective_dpo"]
    assert raw_quality["reverse_conflict_count"] == 1
    assert raw_quality["invalid_pair_count"] == 1
    assert effective_quality["record_count"] == 8
    assert effective_quality["unique_targets"] == 1
    assert effective_quality["max_target_share"] == 1.0
    assert "target_concentrated" in effective_quality["quality_flags"]

    markdown = render_finetuning_readiness_markdown(report)
    assert "## Training Data Quality" in markdown
    assert "| effective_dpo.max_target_share | 1.0000 |" in markdown


def test_finetuning_readiness_reports_sft_training_data_quality():
    report = finetuning_readiness_report(
        format_sft_records=[
            {
                "target_paper_id": "p1",
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "target_paper_id": "p1",
                                "selected_references": [],
                            }
                        ),
                    },
                ],
                "label_source": "model_audit",
            },
            {
                "target_paper_id": "p1",
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "{bad json"},
                ],
                "label_source": "model_audit",
            },
            {
                "target_paper_id": "p2",
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "target_paper_id": "p2",
                                "selected_references": [
                                    {"reference_id": "r2", "rank": 1}
                                ],
                            }
                        ),
                    },
                ],
                "label_source": "model_audit",
            },
        ],
        proxy_positive_sft_records=[],
        pairwise_preference_records=[],
        dpo_preference_records=[],
        context_hard_negative_dpo_records=[],
        method_context_contrastive_dpo_records=[],
        gold_hard_negative_dpo_records=[],
        merged_dpo_preference_records=[],
    )

    sft_quality = report["training_data_quality"]["sft"]
    assert sft_quality["record_count"] == 3
    assert sft_quality["parse_error_count"] == 1
    assert sft_quality["empty_selection_count"] == 1
    assert sft_quality["empty_selection_rate"] == 0.5
    assert sft_quality["unique_targets"] == 2
    assert "parse_errors" in sft_quality["quality_flags"]
    assert "empty_selections" in sft_quality["quality_flags"]

    markdown = render_finetuning_readiness_markdown(report)
    assert "| sft.parse_error_count | 1 |" in markdown
    assert "| sft.empty_selection_rate | 0.5000 |" in markdown
    assert "| sft.quality_flags | parse_errors, empty_selections |" in markdown


def test_finetuning_readiness_quality_issues_block_training_recommendation():
    report = finetuning_readiness_report(
        format_sft_records=[sft_record(f"p{i}") for i in range(25)]
        + [
            {
                "target_paper_id": "bad",
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "{bad json"},
                ],
                "label_source": "model_audit",
            }
        ],
        proxy_positive_sft_records=[],
        pairwise_preference_records=[],
        dpo_preference_records=[],
        context_hard_negative_dpo_records=[],
        method_context_contrastive_dpo_records=[],
        gold_hard_negative_dpo_records=[],
        merged_dpo_preference_records=[],
    )

    assert report["readiness"]["has_training_data_quality_blockers"] is True
    assert report["readiness"]["ready_for_sft_smoke"] is False
    assert report["recommendation"]["next_step"] == "fix_training_data"
    assert "sft_parse_errors" in report["readiness"]["training_data_quality_blockers"]
    markdown = render_finetuning_readiness_markdown(report)
    assert "| training_data_quality_blockers | sft_parse_errors |" in markdown
