import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from local_impact_ranking.dpo_lora_train import (
    dpo_loss,
    load_preference_examples,
    load_preference_examples_with_report,
    run_dpo_lora_training,
    sample_preference_examples,
    tokenize_prompt_response,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class FakeChatTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)


class TinyTokenizer(FakeTokenizer):
    pad_token = None

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) % 32 for char in text])

    def save_pretrained(self, output_dir):
        (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.proj = torch.nn.Linear(1, 64)
        self.active_adapter = "default"
        self.disabled_adapter_calls = 0
        self.reference_adapter_calls = 0

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, input_ids, attention_mask=None):
        logits = self.proj(input_ids.float().unsqueeze(-1) / 32.0)
        if self.active_adapter == "reference":
            self.reference_adapter_calls += 1
            logits = logits.detach()
        return SimpleNamespace(logits=logits)

    def disable_adapter(self):
        model = self

        class DisabledAdapter:
            def __enter__(self):
                model.disabled_adapter_calls += 1

            def __exit__(self, exc_type, exc, tb):
                return False

        return DisabledAdapter()

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def save_pretrained(self, output_dir):
        (output_dir / "adapter_model.safetensors").write_text("tiny", encoding="utf-8")


class TinyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return TinyTokenizer()


class TinyAutoModelForCausalLM:
    latest_model = None

    @staticmethod
    def from_pretrained(*args, **kwargs):
        TinyAutoModelForCausalLM.latest_model = TinyCausalLM()
        return TinyAutoModelForCausalLM.latest_model


class TinyLoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TinyTaskType:
    CAUSAL_LM = "CAUSAL_LM"


class TinyPeftModel:
    @staticmethod
    def from_pretrained(model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        model.loaded_adapter = {
            "model_id": str(model_id),
            "adapter_name": adapter_name,
            "is_trainable": is_trainable,
        }
        model.active_adapter = adapter_name
        return model


def install_tiny_training_modules(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=TinyAutoModelForCausalLM,
            AutoTokenizer=TinyAutoTokenizer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=TinyLoraConfig,
            PeftModel=TinyPeftModel,
            TaskType=TinyTaskType,
            get_peft_model=lambda model, config: model,
        ),
    )


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


def test_load_preference_examples_keeps_multi_message_prompt(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "system", "value": "Reply with Paper A or Paper B."},
                    {"from": "human", "value": "Paper A: Alpha. Paper B: Beta."},
                ],
                "chosen": {"from": "gpt", "value": "Paper A."},
                "rejected": {"from": "gpt", "value": "Paper B."},
                "target_paper_id": "pair-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_preference_examples([dataset], limit=10)

    assert len(examples) == 1
    assert examples[0].prompt == (
        "System: Reply with Paper A or Paper B.\n"
        "Human: Paper A: Alpha. Paper B: Beta."
    )


def test_load_preference_examples_filters_invalid_and_reverse_conflicting_pairs(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    records = [
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r1"},
            "rejected": {"from": "gpt", "value": "r2"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r2"},
            "rejected": {"from": "gpt", "value": "r1"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r3"},
            "rejected": {"from": "gpt", "value": "r3"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [],
            "chosen": {"from": "gpt", "value": "r4"},
            "rejected": {"from": "gpt", "value": "r5"},
            "target_paper_id": "p2",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    examples = load_preference_examples([dataset], limit=10)

    assert [(example.chosen, example.rejected) for example in examples] == [("r1", "r2")]


def test_load_preference_examples_with_report_counts_filtered_records(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    records = [
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r1"},
            "rejected": {"from": "gpt", "value": "r2"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r1"},
            "rejected": {"from": "gpt", "value": "r2"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r2"},
            "rejected": {"from": "gpt", "value": "r1"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [{"from": "human", "value": "Prompt"}],
            "chosen": {"from": "gpt", "value": "r3"},
            "rejected": {"from": "gpt", "value": "r3"},
            "target_paper_id": "p1",
        },
        {
            "conversations": [],
            "chosen": {"from": "gpt", "value": "r4"},
            "rejected": {"from": "gpt", "value": "r5"},
            "target_paper_id": "p2",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_preference_examples_with_report([dataset], limit=10)

    assert [(example.chosen, example.rejected) for example in loaded.examples] == [
        ("r1", "r2")
    ]
    assert loaded.report == {
        "raw_record_count": 5,
        "loaded_example_count": 1,
        "invalid_record_count": 2,
        "duplicate_pair_count": 1,
        "reverse_conflict_count": 1,
        "limit": 10,
    }


def test_sample_preference_examples_reads_all_records_before_seeded_sampling(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    records = [
        {
            "conversations": [{"from": "human", "value": f"Prompt {index}"}],
            "chosen": {"from": "gpt", "value": "Answer: A"},
            "rejected": {"from": "gpt", "value": "Answer: B"},
            "target_paper_id": f"pair-{index}",
        }
        for index in range(10)
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    sampled = sample_preference_examples([dataset], max_examples=3, seed=7)

    assert len(sampled.examples) == 3
    assert sampled.report["raw_record_count"] == 10
    assert sampled.report["loaded_example_count"] == 10
    assert sampled.report["sampled_example_count"] == 3
    assert {example.target_paper_id for example in sampled.examples} != {
        "pair-0",
        "pair-1",
        "pair-2",
    }


def test_sample_preference_examples_can_balance_metadata_and_answer_side(tmp_path):
    dataset = tmp_path / "preferences.jsonl"
    records = []
    for dimension, answer, count in [
        ("citation", "A", 8),
        ("citation", "B", 2),
        ("stars", "A", 8),
        ("stars", "B", 2),
    ]:
        for index in range(count):
            records.append(
                {
                    "conversations": [
                        {"from": "human", "value": f"{dimension} prompt {answer} {index}"}
                    ],
                    "chosen": {"from": "gpt", "value": f"Answer: {answer}"},
                    "rejected": {
                        "from": "gpt",
                        "value": "Answer: B" if answer == "A" else "Answer: A",
                    },
                    "target_paper_id": f"{dimension}-{answer}-{index}",
                    "impact_dimension": dimension,
                    "item_type": "paper",
                }
            )
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    sampled = sample_preference_examples(
        [dataset],
        max_examples=8,
        seed=7,
        sampling_strategy="balanced_metadata",
    )

    groups = {}
    for example in sampled.examples:
        key = (example.metadata["impact_dimension"], example.metadata["target_side"])
        groups[key] = groups.get(key, 0) + 1
    assert groups == {
        ("citation", "A"): 2,
        ("citation", "B"): 2,
        ("stars", "A"): 2,
        ("stars", "B"): 2,
    }
    assert sampled.report["sampling_strategy"] == "balanced_metadata"
    assert sampled.report["sampling_group_count"] == 4


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


def test_tokenize_prompt_response_applies_chat_template_to_prompt_tokens():
    encoded = tokenize_prompt_response(
        FakeChatTokenizer(),
        prompt="abc",
        response="xy",
        max_length=20,
    )

    chat_prompt = "<user>abc"
    assert encoded["input_ids"][: len(chat_prompt)] == [ord(char) for char in chat_prompt]
    assert encoded["labels"][: len(chat_prompt)] == [-100] * len(chat_prompt)


def test_tokenize_prompt_response_truncates_long_prompt_without_dropping_prefix_or_response():
    encoded = tokenize_prompt_response(
        FakeTokenizer(),
        prompt="START-" + ("m" * 40) + "-END",
        response="A",
        max_length=24,
    )
    text = "".join(chr(token_id) for token_id in encoded["input_ids"])

    assert text.startswith("START-")
    assert "-END" in text
    assert text.endswith("A<eos>")
    assert encoded["labels"][: len(encoded["input_ids"]) - len("A<eos>")] == [
        -100
    ] * (len(encoded["input_ids"]) - len("A<eos>"))
    assert encoded["labels"][-len("A<eos>") :] == [ord(char) for char in "A<eos>"]


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


def test_run_dpo_lora_training_uses_optimizer_step_semantics_and_adapter_reference(
    monkeypatch,
    tmp_path,
):
    install_tiny_training_modules(monkeypatch)
    train_file = tmp_path / "preferences.jsonl"
    train_file.write_text(
        "".join(
            json.dumps(
                {
                    "conversations": [{"from": "human", "value": f"Prompt {index}"}],
                    "chosen": {"from": "gpt", "value": "Answer: A"},
                    "rejected": {"from": "gpt", "value": "Answer: B"},
                    "target_paper_id": f"pair-{index}",
                }
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "adapter"

    run_dpo_lora_training(
        model_name_or_path="tiny",
        train_files=[train_file],
        output_dir=output_dir,
        max_steps=3,
        max_examples=8,
        max_length=32,
        beta=0.1,
        learning_rate=1e-3,
        seed=7,
        gradient_accumulation_steps=2,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        device_map="cpu",
        base_adapter=tmp_path / "sft-adapter",
    )

    config = json.loads((output_dir / "train_config.json").read_text(encoding="utf-8"))
    log_rows = [
        json.loads(line)
        for line in (output_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    model = TinyAutoModelForCausalLM.latest_model

    assert config["max_steps_semantics"] == "optimizer_updates"
    assert config["planned_micro_steps"] == 6
    assert config["reference_policy"] == "base_adapter"
    assert [row["step"] for row in log_rows] == [1, 2, 3]
    assert [row["micro_step"] for row in log_rows] == [2, 4, 6]
    assert model.reference_adapter_calls == 12
    assert model.disabled_adapter_calls == 0
