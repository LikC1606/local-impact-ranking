import json

from local_impact_ranking.baselines import compute_bibliometric_baselines
from local_impact_ranking.finetune_export import (
    export_finetuning_datasets,
    make_format_sft_records,
    make_gold_hard_negative_dpo_records,
    make_pairwise_preference_records,
    make_proxy_positive_sft_records,
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


def test_make_format_sft_records_exports_strict_json_targets():
    papers = sample_papers()
    result = sample_result(papers[0])

    records = make_format_sft_records(papers, [result], top_k=2)

    assert len(records) == 1
    assert records[0]["task"] == "format_sft"
    assert "Candidate references" in records[0]["messages"][0]["content"]
    target = json.loads(records[0]["messages"][1]["content"])
    assert target["selected_references"][0]["reference_id"] == "p2"


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
    assert (tmp_path / "gold_hard_negative_dpo.jsonl").exists()
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
