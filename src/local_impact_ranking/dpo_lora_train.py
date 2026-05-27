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
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI envs.
    torch = None
    F = None


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    target_paper_id: str | None = None


@dataclass(frozen=True)
class PreferenceLoadResult:
    examples: list[PreferenceExample]
    report: dict[str, int | None]


def load_preference_examples(
    paths: list[str | Path],
    *,
    limit: int | None = None,
) -> list[PreferenceExample]:
    return load_preference_examples_with_report(paths, limit=limit).examples


def load_preference_examples_with_report(
    paths: list[str | Path],
    *,
    limit: int | None = None,
) -> PreferenceLoadResult:
    examples: list[PreferenceExample] = []
    raw_record_count = 0
    invalid_record_count = 0
    duplicate_pair_count = 0
    reverse_conflict_count = 0
    seen_oriented_pairs: set[tuple[str | None, str, str]] = set()
    seen_unordered_pairs: set[tuple[str | None, frozenset[str]]] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw_record_count += 1
                record = json.loads(stripped)
                example = _preference_example_from_record(record)
                if example is None:
                    invalid_record_count += 1
                    continue
                oriented_pair = (
                    example.target_paper_id,
                    example.chosen,
                    example.rejected,
                )
                unordered_pair = (
                    example.target_paper_id,
                    frozenset({example.chosen, example.rejected}),
                )
                if oriented_pair in seen_oriented_pairs:
                    duplicate_pair_count += 1
                    continue
                if unordered_pair in seen_unordered_pairs:
                    reverse_conflict_count += 1
                    continue
                seen_oriented_pairs.add(oriented_pair)
                seen_unordered_pairs.add(unordered_pair)
                examples.append(example)
                if limit is not None and len(examples) >= limit:
                    return PreferenceLoadResult(
                        examples=examples,
                        report={
                            "raw_record_count": raw_record_count,
                            "loaded_example_count": len(examples),
                            "invalid_record_count": invalid_record_count,
                            "duplicate_pair_count": duplicate_pair_count,
                            "reverse_conflict_count": reverse_conflict_count,
                            "limit": limit,
                        },
                    )
    return PreferenceLoadResult(
        examples=examples,
        report={
            "raw_record_count": raw_record_count,
            "loaded_example_count": len(examples),
            "invalid_record_count": invalid_record_count,
            "duplicate_pair_count": duplicate_pair_count,
            "reverse_conflict_count": reverse_conflict_count,
            "limit": limit,
        },
    )


def _preference_example_from_record(record: dict[str, Any]) -> PreferenceExample | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    first_message = conversations[0]
    if not isinstance(first_message, dict):
        return None
    prompt = first_message.get("value")
    chosen = record.get("chosen", {}).get("value")
    rejected = record.get("rejected", {}).get("value")
    if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    if not prompt.strip() or not chosen.strip() or not rejected.strip():
        return None
    if chosen == rejected:
        return None
    target_paper_id = record.get("target_paper_id")
    return PreferenceExample(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        target_paper_id=str(target_paper_id) if target_paper_id is not None else None,
    )


def tokenize_prompt_response(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_text = response + (tokenizer.eos_token or "")
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    input_ids = (prompt_ids + response_ids)[-max_length:]
    kept_prompt_tokens = min(len(prompt_ids), len(input_ids))
    prompt_start = len(input_ids) - len(response_ids)
    labels = list(input_ids)
    for index in range(max(0, prompt_start)):
        labels[index] = -100
    if kept_prompt_tokens == len(input_ids):
        labels = [-100] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def dpo_loss(
    *,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    beta: float,
) -> Any:
    _require_torch()
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    return -F.logsigmoid(logits).mean()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal local DPO LoRA trainer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)

    run_dpo_lora_training(
        model_name_or_path=args.model,
        train_files=[Path(path) for path in args.train_file],
        output_dir=Path(args.output),
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        max_length=args.max_length,
        beta=args.beta,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device_map=args.device_map,
    )


def run_dpo_lora_training(
    *,
    model_name_or_path: str,
    train_files: list[Path],
    output_dir: Path,
    max_steps: int,
    max_examples: int | None,
    max_length: int,
    beta: float,
    learning_rate: float,
    seed: int,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device_map: str,
) -> None:
    _require_torch()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_preferences = load_preference_examples_with_report(
        train_files,
        limit=max_examples,
    )
    examples = loaded_preferences.examples
    if not examples:
        raise ValueError("No preference examples loaded.")
    random.shuffle(examples)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
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
    log_path = output_dir / "train_log.jsonl"
    metadata = {
        "model_name_or_path": model_name_or_path,
        "train_files": [str(path) for path in train_files],
        "example_count": len(examples),
        "preference_load_report": loaded_preferences.report,
        "max_steps": max_steps,
        "max_examples": max_examples,
        "max_length": max_length,
        "beta": beta,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "device_map": device_map,
        "seed": seed,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        while step < max_steps:
            accum_loss = 0.0
            for _ in range(gradient_accumulation_steps):
                example = examples[step % len(examples)]
                chosen = tokenize_prompt_response(
                    tokenizer,
                    prompt=example.prompt,
                    response=example.chosen,
                    max_length=max_length,
                )
                rejected = tokenize_prompt_response(
                    tokenizer,
                    prompt=example.prompt,
                    response=example.rejected,
                    max_length=max_length,
                )
                policy_chosen = _sequence_logprob(model, chosen)
                policy_rejected = _sequence_logprob(model, rejected)
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_chosen = _sequence_logprob(model, chosen)
                        ref_rejected = _sequence_logprob(model, rejected)
                loss = dpo_loss(
                    policy_chosen_logps=policy_chosen,
                    policy_rejected_logps=policy_rejected,
                    reference_chosen_logps=ref_chosen,
                    reference_rejected_logps=ref_rejected,
                    beta=beta,
                )
                (loss / gradient_accumulation_steps).backward()
                accum_loss += float(loss.detach().cpu())
                step += 1
                if step >= max_steps:
                    break
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log_handle.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": accum_loss / gradient_accumulation_steps,
                        "finite_loss": math.isfinite(accum_loss),
                    }
                )
                + "\n"
            )
            log_handle.flush()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _sequence_logprob(model: Any, encoded: dict[str, list[int]]) -> Any:
    _require_torch()
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoded["attention_mask"]],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * valid).sum(dim=-1)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for DPO LoRA training. Install the training "
            "environment dependencies before running this command."
        )


if __name__ == "__main__":
    main()
