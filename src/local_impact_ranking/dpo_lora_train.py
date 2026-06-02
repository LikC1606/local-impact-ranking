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

from local_impact_ranking.sft_lora_train import truncate_prompt_response_tokens


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    target_paper_id: str | None = None
    metadata: dict[str, Any] | None = None


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


def sample_preference_examples(
    paths: list[str | Path],
    *,
    max_examples: int | None,
    seed: int,
    sampling_strategy: str = "random",
) -> PreferenceLoadResult:
    if sampling_strategy not in {"random", "balanced_label", "balanced_metadata"}:
        raise ValueError(
            "sampling_strategy must be 'random', 'balanced_label', or 'balanced_metadata'"
        )
    loaded = load_preference_examples_with_report(paths)
    examples = list(loaded.examples)
    if max_examples is not None and len(examples) > max_examples:
        if sampling_strategy == "random":
            examples = random.Random(seed).sample(examples, max_examples)
        else:
            examples = _balanced_sample_examples(
                examples,
                max_examples=max_examples,
                seed=seed,
                group_by_metadata=sampling_strategy == "balanced_metadata",
            )
    return PreferenceLoadResult(
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


def _preference_example_from_record(record: dict[str, Any]) -> PreferenceExample | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    prompt = _prompt_from_conversations(conversations)
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
        metadata=_metadata_from_record(record, chosen),
    )


def _metadata_from_record(record: dict[str, Any], chosen: str) -> dict[str, Any]:
    return {
        "impact_dimension": str(record.get("impact_dimension") or "unknown"),
        "item_type": str(record.get("item_type") or "unknown"),
        "target_side": _target_side(chosen),
    }


def _target_side(response: str) -> str:
    normalized = response.strip().upper()
    if normalized.endswith("A") or "ANSWER: A" in normalized:
        return "A"
    if normalized.endswith("B") or "ANSWER: B" in normalized:
        return "B"
    return "unknown"


def _balanced_sample_examples(
    examples: list[PreferenceExample],
    *,
    max_examples: int,
    seed: int,
    group_by_metadata: bool,
) -> list[PreferenceExample]:
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(_sampling_key(example, group_by_metadata=group_by_metadata), []).append(
            example
        )
    for group_examples in groups.values():
        rng.shuffle(group_examples)
    ordered_keys = sorted(groups)
    rng.shuffle(ordered_keys)
    selected: list[PreferenceExample] = []
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
    examples: list[PreferenceExample],
    *,
    group_by_metadata: bool,
) -> int:
    return len(
        {
            _sampling_key(example, group_by_metadata=group_by_metadata)
            for example in examples
        }
    )


def _sampling_key(example: PreferenceExample, *, group_by_metadata: bool) -> tuple[str, ...]:
    metadata = example.metadata or {}
    target_side = str(metadata.get("target_side") or _target_side(example.chosen))
    if not group_by_metadata:
        return (target_side,)
    return (
        str(metadata.get("impact_dimension") or "unknown"),
        str(metadata.get("item_type") or "unknown"),
        target_side,
    )


def _prompt_from_conversations(conversations: list[Any]) -> str | None:
    messages: list[tuple[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            return None
        role = message.get("from")
        value = message.get("value")
        if not isinstance(role, str) or not isinstance(value, str) or not value.strip():
            return None
        messages.append((role, value.strip()))
    if len(messages) == 1:
        return messages[0][1]
    return "\n".join(f"{role.title()}: {value}" for role, value in messages)


def tokenize_prompt_response(
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
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--base-adapter", default=None)
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "balanced_label", "balanced_metadata"],
        default="random",
    )
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
        base_adapter=args.base_adapter,
        sampling_strategy=args.sampling_strategy,
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
    base_adapter: str | Path | None = None,
    sampling_strategy: str = "random",
) -> None:
    _require_torch()
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_preferences = sample_preference_examples(
        train_files,
        max_examples=max_examples,
        seed=seed,
        sampling_strategy=sampling_strategy,
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
    policy_adapter_name = "default"
    model.gradient_checkpointing_enable()
    if base_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            str(base_adapter),
            adapter_name=policy_adapter_name,
            is_trainable=True,
        )
        load_adapter = getattr(model, "load_adapter", None)
        if load_adapter is not None:
            load_adapter(
                str(base_adapter),
                adapter_name="reference",
                is_trainable=False,
            )
    else:
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
        "max_steps_semantics": "optimizer_updates",
        "planned_micro_steps": max_steps * gradient_accumulation_steps,
        "max_examples": max_examples,
        "max_length": max_length,
        "beta": beta,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "device_map": device_map,
        "base_adapter": str(base_adapter) if base_adapter is not None else None,
        "reference_policy": "base_adapter" if base_adapter is not None else "base_model",
        "reference_logps_precomputed": True,
        "policy_adapter_name": policy_adapter_name,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    planned_micro_steps = max_steps * gradient_accumulation_steps
    reference_logps = _precompute_reference_logps(
        model,
        tokenizer,
        examples=examples,
        planned_micro_steps=planned_micro_steps,
        max_length=max_length,
        reference_policy="base_adapter" if base_adapter is not None else "base_model",
        policy_adapter_name=policy_adapter_name,
    )

    optimizer_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        while optimizer_step < max_steps:
            accum_loss = 0.0
            actual_accumulation = 0
            for _ in range(gradient_accumulation_steps):
                example = examples[micro_step % len(examples)]
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
                ref_chosen_value, ref_rejected_value = reference_logps[micro_step]
                ref_chosen = policy_chosen.detach().new_tensor(ref_chosen_value).view_as(
                    policy_chosen
                )
                ref_rejected = policy_rejected.detach().new_tensor(
                    ref_rejected_value
                ).view_as(policy_rejected)
                loss = dpo_loss(
                    policy_chosen_logps=policy_chosen,
                    policy_rejected_logps=policy_rejected,
                    reference_chosen_logps=ref_chosen,
                    reference_rejected_logps=ref_rejected,
                    beta=beta,
                )
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

    _save_policy_adapter(model, output_dir, policy_adapter_name=policy_adapter_name)
    tokenizer.save_pretrained(output_dir)


def _precompute_reference_logps(
    model: Any,
    tokenizer: Any,
    *,
    examples: list[PreferenceExample],
    planned_micro_steps: int,
    max_length: int,
    reference_policy: str,
    policy_adapter_name: str,
) -> list[tuple[float, float]]:
    _require_torch()
    reference_logps: list[tuple[float, float]] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        if reference_policy == "base_adapter":
            model.set_adapter("reference")
            for micro_step in range(planned_micro_steps):
                reference_logps.append(
                    _reference_logprob_pair(
                        model,
                        tokenizer,
                        examples[micro_step % len(examples)],
                        max_length=max_length,
                    )
                )
            model.set_adapter(policy_adapter_name)
        elif reference_policy == "base_model":
            with model.disable_adapter():
                for micro_step in range(planned_micro_steps):
                    reference_logps.append(
                        _reference_logprob_pair(
                            model,
                            tokenizer,
                            examples[micro_step % len(examples)],
                            max_length=max_length,
                        )
                    )
        else:
            raise ValueError("reference_policy must be 'base_adapter' or 'base_model'")
    if was_training:
        model.train()
    return reference_logps


def _reference_logprob_pair(
    model: Any,
    tokenizer: Any,
    example: PreferenceExample,
    *,
    max_length: int,
) -> tuple[float, float]:
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
    return (
        float(_sequence_logprob(model, chosen).detach().cpu()[0]),
        float(_sequence_logprob(model, rejected).detach().cpu()[0]),
    )


def _save_policy_adapter(model: Any, output_dir: Path, *, policy_adapter_name: str) -> None:
    try:
        model.save_pretrained(output_dir, selected_adapters=[policy_adapter_name])
    except TypeError:
        model.save_pretrained(output_dir)


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
