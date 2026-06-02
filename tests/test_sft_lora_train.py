import json
import sys
from types import SimpleNamespace

import pytest

from local_impact_ranking.sft_lora_train import (
    chat_prompt_text,
    inspect_sft_tokenization,
    load_sft_examples,
    load_sft_examples_with_report,
    run_sft_lora_training,
    sample_sft_examples,
    tokenize_sft_example,
)

torch = pytest.importorskip("torch")


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

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_grads_enabled = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.proj(input_ids.float().unsqueeze(-1) / 32.0)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, output_dir):
        (output_dir / "adapter_model.safetensors").write_text("tiny", encoding="utf-8")


class TinyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return TinyTokenizer()


class TinyAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return TinyCausalLM()


class TinyLoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TinyTaskType:
    CAUSAL_LM = "CAUSAL_LM"


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
            TaskType=TinyTaskType,
            get_peft_model=lambda model, config: model,
        ),
    )


def test_chat_prompt_text_uses_tokenizer_chat_template_when_available():
    prompt = chat_prompt_text(FakeChatTokenizer(), "Prompt")

    assert prompt == "<user>Prompt"


def test_load_sft_examples_reads_sharegpt_conversations(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Prompt"},
                    {"from": "gpt", "value": "Answer: A"},
                ],
                "source_line": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_sft_examples([dataset], limit=10)

    assert len(examples) == 1
    assert examples[0].prompt == "Prompt"
    assert examples[0].response == "Answer: A"
    assert examples[0].source_line == 7


def test_load_sft_examples_reads_messages_target_compatibility_format(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Prompt"}],
                "target": "Answer: B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_sft_examples([dataset], limit=10)

    assert len(examples) == 1
    assert examples[0].prompt == "Prompt"
    assert examples[0].response == "Answer: B"


def test_load_sft_examples_with_report_counts_invalid_records(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    records = [
        {
            "conversations": [
                {"from": "human", "value": "Prompt"},
                {"from": "gpt", "value": "Answer: A"},
            ],
        },
        {
            "conversations": [
                {"from": "human", "value": "Prompt without assistant"},
            ],
        },
        {"messages": [{"role": "system", "content": "No user"}], "target": "Answer: A"},
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_sft_examples_with_report([dataset], limit=10)

    assert len(loaded.examples) == 1
    assert loaded.report == {
        "raw_record_count": 3,
        "loaded_example_count": 1,
        "invalid_record_count": 2,
        "limit": 10,
    }


def test_sample_sft_examples_reads_all_records_before_seeded_sampling(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    records = [
        {
            "conversations": [
                {"from": "human", "value": f"Prompt {index}"},
                {"from": "gpt", "value": "Answer: A"},
            ],
            "source_line": index,
        }
        for index in range(10)
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    sampled = sample_sft_examples([dataset], max_examples=3, seed=7)

    assert len(sampled.examples) == 3
    assert sampled.report["raw_record_count"] == 10
    assert sampled.report["loaded_example_count"] == 10
    assert sampled.report["sampled_example_count"] == 3
    assert {example.source_line for example in sampled.examples} != {0, 1, 2}


def test_sample_sft_examples_can_balance_metadata_and_answer_side(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    records = []
    source_line = 1
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
                        {"from": "human", "value": f"{dimension} prompt {answer} {index}"},
                        {"from": "gpt", "value": f"Answer: {answer}"},
                    ],
                    "source_line": source_line,
                    "impact_dimension": dimension,
                    "item_type": "paper",
                }
            )
            source_line += 1
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    sampled = sample_sft_examples(
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


def test_tokenize_sft_example_masks_prompt_tokens_and_keeps_response_labels():
    encoded = tokenize_sft_example(
        FakeTokenizer(),
        prompt="abc",
        response="xy",
        max_length=10,
    )

    assert encoded["input_ids"] == [97, 98, 99, 120, 121, 60, 101, 111, 115, 62]
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:] == [120, 121, 60, 101, 111, 115, 62]
    assert encoded["attention_mask"] == [1] * 10


def test_tokenize_sft_example_applies_chat_template_to_prompt_tokens():
    encoded = tokenize_sft_example(
        FakeChatTokenizer(),
        prompt="abc",
        response="xy",
        max_length=20,
    )

    chat_prompt = "<user>abc"
    assert encoded["input_ids"][: len(chat_prompt)] == [ord(char) for char in chat_prompt]
    assert encoded["labels"][: len(chat_prompt)] == [-100] * len(chat_prompt)


def test_tokenize_sft_example_truncates_long_prompt_without_dropping_prefix_or_response():
    encoded = tokenize_sft_example(
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


def test_inspect_sft_tokenization_reports_prompt_retention(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "START-" + ("m" * 40) + "-END"},
                    {"from": "gpt", "value": "Answer: A"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_sft_tokenization(
        [dataset],
        tokenizer=FakeTokenizer(),
        max_length=32,
        max_examples=10,
    )

    assert report["summary"]["loaded_example_count"] == 1
    assert report["summary"]["truncated_example_count"] == 1
    assert report["summary"]["prefix_retained_rate"] == 1.0
    assert report["summary"]["suffix_retained_rate"] == 1.0
    assert report["summary"]["response_fully_retained_rate"] == 1.0
    assert report["examples"][0]["retained_prompt_token_count"] == 18


def test_run_sft_lora_training_treats_max_steps_as_optimizer_updates(monkeypatch, tmp_path):
    install_tiny_training_modules(monkeypatch)
    train_file = tmp_path / "sft.jsonl"
    train_file.write_text(
        "".join(
            json.dumps(
                {
                    "conversations": [
                        {"from": "human", "value": f"Prompt {index}"},
                        {"from": "gpt", "value": "Answer: A" if index % 2 else "Answer: B"},
                    ],
                    "source_line": index,
                }
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "adapter"

    run_sft_lora_training(
        model_name_or_path="tiny",
        train_files=[train_file],
        output_dir=output_dir,
        max_steps=3,
        max_examples=8,
        max_length=32,
        learning_rate=1e-3,
        seed=7,
        gradient_accumulation_steps=2,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        device_map="cpu",
    )

    config = json.loads((output_dir / "train_config.json").read_text(encoding="utf-8"))
    log_rows = [
        json.loads(line)
        for line in (output_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert config["max_steps_semantics"] == "optimizer_updates"
    assert config["planned_micro_steps"] == 6
    assert [row["step"] for row in log_rows] == [1, 2, 3]
    assert [row["micro_step"] for row in log_rows] == [2, 4, 6]
