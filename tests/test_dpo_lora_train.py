import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from local_impact_ranking.dpo_lora_train import (
    dpo_loss,
    load_preference_examples,
    tokenize_prompt_response,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_load_preference_examples_reads_sharegpt_dpo_jsonl(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "conversations": [{"from": "human", "value": "Prompt"}],
                "chosen": {"from": "gpt", "value": "r1"},
                "rejected": {"from": "gpt", "value": "r2"},
                "target_paper_id": "p1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_preference_examples([dataset], limit=10)

    assert len(examples) == 1
    assert examples[0].prompt == "Prompt"
    assert examples[0].chosen == "r1"
    assert examples[0].rejected == "r2"
    assert examples[0].target_paper_id == "p1"


def test_tokenize_prompt_response_masks_prompt_tokens_and_keeps_response_labels():
    encoded = tokenize_prompt_response(
        FakeTokenizer(),
        prompt="abc",
        response="xy",
        max_length=10,
    )

    assert encoded["input_ids"] == [97, 98, 99, 120, 121, 60, 101, 111, 115, 62]
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:] == [120, 121, 60, 101, 111, 115, 62]
    assert encoded["attention_mask"] == [1] * 10


def test_dpo_loss_is_lower_when_policy_prefers_chosen_more_than_reference():
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])

    good_loss = dpo_loss(
        policy_chosen_logps=torch.tensor([2.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=1.0,
    )
    bad_loss = dpo_loss(
        policy_chosen_logps=torch.tensor([0.0]),
        policy_rejected_logps=torch.tensor([2.0]),
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=1.0,
    )

    assert good_loss < bad_loss
