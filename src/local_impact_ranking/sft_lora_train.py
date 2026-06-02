from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI envs.
    torch = None


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    response: str
    source_line: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SFTLoadResult:
    examples: list[SFTExample]
    report: dict[str, int | None]


def load_sft_examples(
    paths: list[str | Path],
    *,
    limit: int | None = None,
) -> list[SFTExample]:
    return load_sft_examples_with_report(paths, limit=limit).examples


def load_sft_examples_with_report(
    paths: list[str | Path],
    *,
    limit: int | None = None,
) -> SFTLoadResult:
    examples: list[SFTExample] = []
    raw_record_count = 0
    invalid_record_count = 0
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw_record_count += 1
                record = json.loads(stripped)
                example = _sft_example_from_record(record, line_number)
                if example is None:
                    invalid_record_count += 1
                    continue
                examples.append(example)
                if limit is not None and len(examples) >= limit:
                    return SFTLoadResult(
                        examples=examples,
                        report={
                            "raw_record_count": raw_record_count,
                            "loaded_example_count": len(examples),
                            "invalid_record_count": invalid_record_count,
                            "limit": limit,
                        },
                    )
    return SFTLoadResult(
        examples=examples,
        report={
            "raw_record_count": raw_record_count,
            "loaded_example_count": len(examples),
            "invalid_record_count": invalid_record_count,
            "limit": limit,
        },
    )


def sample_sft_examples(
    paths: list[str | Path],
    *,
    max_examples: int | None,
    seed: int,
    sampling_strategy: str = "random",
) -> SFTLoadResult:
    if sampling_strategy not in {"random", "balanced_label", "balanced_metadata"}:
        raise ValueError(
            "sampling_strategy must be 'random', 'balanced_label', or 'balanced_metadata'"
        )
    loaded = load_sft_examples_with_report(paths)
    examples = list(loaded.examples)
    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        if sampling_strategy == "random":
            examples = rng.sample(examples, max_examples)
        else:
            examples = _balanced_sample_examples(
                examples,
                max_examples=max_examples,
                seed=seed,
                group_by_metadata=sampling_strategy == "balanced_metadata",
            )
    return SFTLoadResult(
        examples=examples,
        report={
            **loaded.report,
            "sampled_example_count": len(examples),
            "sample_seed": seed,
            "sampling_strategy": sampling_strategy,
            "sampling_group_count": _sampling_group_count(
                examples,
                group_by_metadata=sampling_strategy == "balanced_metadata",
            )
            if sampling_strategy != "random"
            else None,
        },
    )


def inspect_sft_tokenization(
    paths: list[str | Path],
    *,
    tokenizer: Any,
    max_length: int,
    max_examples: int | None = None,
) -> dict[str, Any]:
    loaded = load_sft_examples_with_report(paths, limit=max_examples)
    rows = []
    for index, example in enumerate(loaded.examples, start=1):
        prompt_text = chat_prompt_text(tokenizer, example.prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_text = example.response + (tokenizer.eos_token or "")
        response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
        input_ids, response_start = truncate_prompt_response_tokens(
            prompt_ids,
            response_ids,
            max_length=max_length,
        )
        retained_prompt_ids = input_ids[:response_start]
        row = {
            "example_index": index,
            "source_line": example.source_line,
            "prompt_token_count": len(prompt_ids),
            "response_token_count": len(response_ids),
            "total_token_count": len(prompt_ids) + len(response_ids),
            "max_length": max_length,
            "truncated": len(prompt_ids) + len(response_ids) > max_length,
            "retained_prompt_token_count": len(retained_prompt_ids),
            "retained_response_token_count": len(input_ids) - response_start,
            "prefix_retained": _starts_with_tokens(retained_prompt_ids, prompt_ids),
            "suffix_retained": _ends_with_tokens(retained_prompt_ids, prompt_ids),
            "response_fully_retained": len(input_ids) - response_start
            == min(len(response_ids), max_length),
        }
        row["prompt_retention_rate"] = round(
            row["retained_prompt_token_count"] / len(prompt_ids) if prompt_ids else 1.0,
            6,
        )
        rows.append(row)
    return {
        "summary": _tokenization_inspection_summary(rows, loaded.report),
        "examples": rows[:20],
    }


def tokenize_sft_example(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text = chat_prompt_text(tokenizer, prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_text = response + (tokenizer.eos_token or "")
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    input_ids, response_start = truncate_prompt_response_tokens(
        prompt_ids,
        response_ids,
        max_length=max_length,
    )
    labels = list(input_ids)
    for index in range(response_start):
        labels[index] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def truncate_prompt_response_tokens(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    max_length: int,
) -> tuple[list[int], int]:
    """Keep prompt prefix and suffix while preserving response labels."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if len(response_ids) >= max_length:
        input_ids = response_ids[-max_length:]
        return input_ids, 0
    prompt_budget = max_length - len(response_ids)
    if len(prompt_ids) <= prompt_budget:
        kept_prompt = list(prompt_ids)
    elif prompt_budget <= 0:
        kept_prompt = []
    elif prompt_budget == 1:
        kept_prompt = [prompt_ids[0]]
    else:
        head_count = max(1, prompt_budget // 2)
        tail_count = prompt_budget - head_count
        kept_prompt = list(prompt_ids[:head_count]) + list(prompt_ids[-tail_count:])
    input_ids = kept_prompt + list(response_ids)
    return input_ids, len(kept_prompt)


def _starts_with_tokens(candidate: list[int], original: list[int]) -> bool:
    if not original:
        return True
    if not candidate:
        return False
    anchor = min(8, len(candidate), len(original))
    return candidate[:anchor] == original[:anchor]


def _ends_with_tokens(candidate: list[int], original: list[int]) -> bool:
    if not original:
        return True
    if not candidate:
        return False
    anchor = min(8, len(candidate), len(original))
    return candidate[-anchor:] == original[-anchor:]


def _tokenization_inspection_summary(
    rows: list[dict[str, Any]],
    load_report: dict[str, int | None],
) -> dict[str, Any]:
    count = len(rows)
    truncated_count = sum(1 for row in rows if row["truncated"])
    return {
        **load_report,
        "inspected_example_count": count,
        "truncated_example_count": truncated_count,
        "truncated_example_rate": round(truncated_count / count if count else 0.0, 6),
        "mean_prompt_retention_rate": round(
            sum(float(row["prompt_retention_rate"]) for row in rows) / count
            if count
            else 0.0,
            6,
        ),
        "prefix_retained_rate": round(
            sum(1 for row in rows if row["prefix_retained"]) / count if count else 0.0,
            6,
        ),
        "suffix_retained_rate": round(
            sum(1 for row in rows if row["suffix_retained"]) / count if count else 0.0,
            6,
        ),
        "response_fully_retained_rate": round(
            sum(1 for row in rows if row["response_fully_retained"]) / count
            if count
            else 0.0,
            6,
        ),
    }


def chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt
    try:
        return str(
            apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except (TypeError, ValueError):
        return prompt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal local SFT LoRA trainer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "balanced_label", "balanced_metadata"],
        default="random",
    )
    args = parser.parse_args(argv)

    run_sft_lora_training(
        model_name_or_path=args.model,
        train_files=[Path(path) for path in args.train_file],
        output_dir=Path(args.output),
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device_map=args.device_map,
        sampling_strategy=args.sampling_strategy,
    )


def run_sft_lora_training(
    *,
    model_name_or_path: str,
    train_files: list[Path],
    output_dir: Path,
    max_steps: int,
    max_examples: int | None,
    max_length: int,
    learning_rate: float,
    seed: int,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device_map: str,
    sampling_strategy: str = "random",
) -> None:
    _require_torch()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = sample_sft_examples(
        train_files,
        max_examples=max_examples,
        seed=seed,
        sampling_strategy=sampling_strategy,
    )
    examples = loaded.examples
    if not examples:
        raise ValueError("No SFT examples loaded.")
    random.shuffle(examples)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        ),
    )
    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metadata = {
        "model_name_or_path": model_name_or_path,
        "train_files": [str(path) for path in train_files],
        "example_count": len(examples),
        "sft_load_report": loaded.report,
        "max_steps": max_steps,
        "max_steps_semantics": "optimizer_updates",
        "planned_micro_steps": max_steps * gradient_accumulation_steps,
        "max_examples": max_examples,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "device_map": device_map,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = output_dir / "train_log.jsonl"
    optimizer_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        while optimizer_step < max_steps:
            accum_loss = 0.0
            actual_accumulation = 0
            for _ in range(gradient_accumulation_steps):
                example = examples[micro_step % len(examples)]
                encoded = tokenize_sft_example(
                    tokenizer,
                    prompt=example.prompt,
                    response=example.response,
                    max_length=max_length,
                )
                batch = _single_example_batch(encoded, model)
                outputs = model(**batch)
                loss = outputs.loss
                (loss / gradient_accumulation_steps).backward()
                accum_loss += float(loss.detach().cpu())
                actual_accumulation += 1
                micro_step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            mean_loss = accum_loss / max(1, actual_accumulation)
            log_handle.write(
                json.dumps(
                    {
                        "step": optimizer_step,
                        "micro_step": micro_step,
                        "loss": mean_loss,
                        "finite_loss": math.isfinite(mean_loss),
                    }
                )
                + "\n"
            )
            log_handle.flush()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _sft_example_from_record(record: dict[str, Any], line_number: int) -> SFTExample | None:
    metadata = _metadata_from_record(record)
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        example = _sft_example_from_conversations(
            conversations,
            source_line=_source_line(record, line_number),
            metadata=metadata,
        )
        if example is not None:
            return example
    return _sft_example_from_messages_target(
        record,
        source_line=_source_line(record, line_number),
        metadata=metadata,
    )


def _sft_example_from_conversations(
    conversations: list[Any],
    *,
    source_line: int | None,
    metadata: dict[str, Any],
) -> SFTExample | None:
    assistant_index = None
    for index in range(len(conversations) - 1, -1, -1):
        message = conversations[index]
        if not isinstance(message, dict):
            return None
        if message.get("from") in {"gpt", "assistant"}:
            assistant_index = index
            break
    if assistant_index is None:
        return None
    response = conversations[assistant_index].get("value")
    if not isinstance(response, str) or not response.strip():
        return None
    prompt = _prompt_from_tagged_messages(
        conversations[:assistant_index],
        role_key="from",
        content_key="value",
        user_roles={"human", "user"},
    )
    if prompt is None:
        return None
    return SFTExample(
        prompt=prompt,
        response=response.strip(),
        source_line=source_line,
        metadata={**metadata, "target_side": _target_side(response)},
    )


def _sft_example_from_messages_target(
    record: dict[str, Any],
    *,
    source_line: int | None,
    metadata: dict[str, Any],
) -> SFTExample | None:
    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    prompt = _prompt_from_tagged_messages(
        messages,
        role_key="role",
        content_key="content",
        user_roles={"user", "human"},
    )
    if prompt is None:
        return None
    return SFTExample(
        prompt=prompt,
        response=target.strip(),
        source_line=source_line,
        metadata={**metadata, "target_side": _target_side(target)},
    )


def _metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "impact_dimension": str(record.get("impact_dimension") or "unknown"),
        "item_type": str(record.get("item_type") or "unknown"),
    }


def _target_side(response: str) -> str:
    normalized = response.strip().upper()
    if normalized.endswith("A") or "ANSWER: A" in normalized:
        return "A"
    if normalized.endswith("B") or "ANSWER: B" in normalized:
        return "B"
    return "unknown"


def _balanced_sample_examples(
    examples: list[SFTExample],
    *,
    max_examples: int,
    seed: int,
    group_by_metadata: bool,
) -> list[SFTExample]:
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[SFTExample]] = {}
    for example in examples:
        groups.setdefault(_sampling_key(example, group_by_metadata=group_by_metadata), []).append(
            example
        )
    for group_examples in groups.values():
        rng.shuffle(group_examples)
    ordered_keys = sorted(groups)
    rng.shuffle(ordered_keys)
    selected: list[SFTExample] = []
    while len(selected) < max_examples and ordered_keys:
        next_keys: list[tuple[str, ...]] = []
        for key in ordered_keys:
            group_examples = groups[key]
            if group_examples and len(selected) < max_examples:
                selected.append(group_examples.pop())
            if group_examples:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def _sampling_group_count(
    examples: list[SFTExample],
    *,
    group_by_metadata: bool,
) -> int:
    return len(
        {
            _sampling_key(example, group_by_metadata=group_by_metadata)
            for example in examples
        }
    )


def _sampling_key(example: SFTExample, *, group_by_metadata: bool) -> tuple[str, ...]:
    metadata = example.metadata or {}
    target_side = str(metadata.get("target_side") or _target_side(example.response))
    if not group_by_metadata:
        return (target_side,)
    return (
        str(metadata.get("impact_dimension") or "unknown"),
        str(metadata.get("item_type") or "unknown"),
        target_side,
    )


def _prompt_from_tagged_messages(
    messages: list[Any],
    *,
    role_key: str,
    content_key: str,
    user_roles: set[str],
) -> str | None:
    tagged: list[tuple[str, str]] = []
    has_user_message = False
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get(role_key)
        value = message.get(content_key)
        if not isinstance(role, str) or not isinstance(value, str) or not value.strip():
            return None
        if role in {"gpt", "assistant"}:
            return None
        if role in user_roles:
            has_user_message = True
        tagged.append((role, value.strip()))
    if not tagged or not has_user_message:
        return None
    if len(tagged) == 1 and tagged[0][0] in user_roles:
        return tagged[0][1]
    return "\n".join(f"{role.title()}: {value}" for role, value in tagged)


def _single_example_batch(encoded: dict[str, list[int]], model: Any) -> dict[str, Any]:
    _require_torch()
    device = next(model.parameters()).device
    return {
        "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [encoded["attention_mask"]],
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.tensor([encoded["labels"]], dtype=torch.long, device=device),
    }


def _source_line(record: dict[str, Any], fallback: int) -> int | None:
    source_line = record.get("source_line", fallback)
    return source_line if isinstance(source_line, int) else None


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for SFT LoRA training. Install the training "
            "environment dependencies before running this command."
        )


if __name__ == "__main__":
    main()
