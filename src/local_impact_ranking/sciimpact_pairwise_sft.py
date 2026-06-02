from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from local_impact_ranking.model_impact_index import build_pairwise_impact_prompt
from local_impact_ranking.models import Paper
from local_impact_ranking.sciimpact_adapter import _parse_pair


def export_sciimpact_pairwise_sft(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    skip_invalid: bool = True,
    prompt_style: str = "generic_impact",
) -> dict[str, Any]:
    prompt_style = _normalize_prompt_style(prompt_style)
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_record_count = 0
    record_count = 0
    skipped = []

    with source.open("r", encoding="utf-8") as input_handle, destination.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if limit is not None and source_record_count >= limit:
                break
            source_record_count += 1
            try:
                source_record = json.loads(stripped)
                parsed = _parse_pair(source_record, line_number)
                record = _pairwise_sft_record(
                    source_record,
                    parsed,
                    line_number,
                    prompt_style=prompt_style,
                )
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_count += 1

    manifest = {
        "source_path": str(source),
        "output_path": str(destination),
        "source_record_count": source_record_count,
        "record_count": record_count,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped[:20],
        "format": "chat_messages_with_target",
        "prompt_family": _prompt_family(prompt_style),
        "prompt_style": prompt_style,
    }
    (destination.parent / "sciimpact_pairwise_sft_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_sciimpact_pairwise_dpo(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    skip_invalid: bool = True,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_record_count = 0
    record_count = 0
    skipped = []

    with source.open("r", encoding="utf-8") as input_handle, destination.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if limit is not None and source_record_count >= limit:
                break
            source_record_count += 1
            try:
                record = _pairwise_dpo_record(json.loads(stripped), line_number)
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_count += 1

    manifest = {
        "source_path": str(source),
        "output_path": str(destination),
        "source_record_count": source_record_count,
        "record_count": record_count,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped[:20],
        "format": "sharegpt_dpo",
        "prompt_family": "model_impact_index_pairwise_v1",
    }
    (destination.parent / "sciimpact_pairwise_dpo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def mine_sciimpact_hard_pairwise_examples(
    sft_path: str | Path,
    flip_rows_path: str | Path,
    output_dir: str | Path,
    *,
    max_examples: int | None = None,
    include_wrong: bool = True,
    include_inconsistent: bool = True,
    skip_invalid: bool = True,
) -> dict[str, Any]:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    if not include_wrong and not include_inconsistent:
        raise ValueError("At least one hard-example reason must be enabled")

    source = Path(sft_path)
    rows_source = Path(flip_rows_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sft_output = destination / "train_pairwise_sft.jsonl"
    dpo_output = destination / "train_pairwise_dpo.jsonl"

    hard_rows = _load_hard_flip_rows(
        rows_source,
        include_wrong=include_wrong,
        include_inconsistent=include_inconsistent,
    )
    requested_source_lines = set(hard_rows)
    reason_counts: Counter[str] = Counter()
    by_dimension: Counter[str] = Counter()
    by_item_type: Counter[str] = Counter()
    skipped = []
    record_count = 0

    with source.open("r", encoding="utf-8") as input_handle, sft_output.open(
        "w",
        encoding="utf-8",
    ) as sft_handle, dpo_output.open("w", encoding="utf-8") as dpo_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if max_examples is not None and record_count >= max_examples:
                break
            hard_row = hard_rows.get(line_number)
            if hard_row is None:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                source_record = json.loads(stripped)
                parsed = _parse_pair(source_record, line_number)
                sft_record = _pairwise_sft_record(
                    source_record,
                    parsed,
                    line_number,
                    prompt_style="generic_impact",
                )
                _annotate_hard_record(sft_record, hard_row)
                dpo_record = _pairwise_dpo_record(sft_record, line_number)
                _annotate_hard_record(dpo_record, hard_row)
                dpo_record["label_source"] = "sciimpact_hard_pairwise_mining"
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue

            sft_handle.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            dpo_handle.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
            record_count += 1
            for reason in hard_row["hard_example_reasons"]:
                reason_counts[reason] += 1
            by_dimension[str(sft_record.get("impact_dimension") or "unknown")] += 1
            by_item_type[str(sft_record.get("item_type") or "unknown")] += 1
            requested_source_lines.discard(line_number)

    manifest = {
        "source_path": str(source),
        "flip_rows_path": str(rows_source),
        "output_dir": str(destination),
        "sft_output_path": str(sft_output),
        "dpo_output_path": str(dpo_output),
        "candidate_hard_row_count": len(hard_rows),
        "record_count": record_count,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped[:20],
        "unmatched_source_line_count": len(requested_source_lines),
        "unmatched_source_lines": sorted(requested_source_lines)[:20],
        "reason_counts": dict(reason_counts),
        "by_dimension": dict(by_dimension),
        "by_item_type": dict(by_item_type),
        "max_examples": max_examples,
        "include_wrong": include_wrong,
        "include_inconsistent": include_inconsistent,
        "training_leakage_warning": _training_leakage_warning(source),
        "format": "hard_pairwise_sft_and_dpo",
    }
    (destination / "sciimpact_hard_pairwise_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _normalize_prompt_style(prompt_style: str) -> str:
    normalized = prompt_style.strip().lower().replace("-", "_")
    if normalized not in {"generic_impact", "source_task"}:
        raise ValueError("prompt_style must be 'generic_impact' or 'source_task'")
    return normalized


def _prompt_family(prompt_style: str) -> str:
    if prompt_style == "source_task":
        return "sciimpact_source_task_answer_v1"
    return "model_impact_index_pairwise_v1"


def _pairwise_sft_record(
    source_record: dict[str, Any],
    parsed: dict[str, Any],
    line_number: int,
    *,
    prompt_style: str,
) -> dict[str, Any]:
    if prompt_style == "source_task":
        return _source_task_pairwise_sft_record(source_record, parsed, line_number)
    winner_side = _winner_side(parsed, line_number)
    paper_a = parsed["winner"] if winner_side == "a" else parsed["loser"]
    paper_b = parsed["winner"] if winner_side == "b" else parsed["loser"]
    prompt = build_pairwise_impact_prompt(
        _paper_from_sciimpact_record(paper_a),
        _paper_from_sciimpact_record(paper_b),
    )
    target = f"Answer: {winner_side.upper()}"
    return {
        "messages": [{"role": "user", "content": prompt}],
        "target": target,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target},
        ],
        "source_line": line_number,
        "impact_dimension": parsed.get("impact_dimension", "unknown"),
        "item_type": parsed.get("item_type", "unknown"),
    }


def _source_task_pairwise_sft_record(
    source_record: dict[str, Any],
    parsed: dict[str, Any],
    line_number: int,
) -> dict[str, Any]:
    winner_side = _winner_side(parsed, line_number)
    messages = _source_task_messages(source_record, line_number)
    target = f"Answer: {winner_side.upper()}"
    return {
        "messages": messages,
        "target": target,
        "conversations": _sharegpt_prompt_conversations(messages)
        + [{"from": "gpt", "value": target}],
        "source_line": line_number,
        "impact_dimension": parsed.get("impact_dimension", "unknown"),
        "item_type": parsed.get("item_type", "unknown"),
        "prompt_style": "source_task",
    }


def _source_task_messages(
    source_record: dict[str, Any],
    line_number: int,
) -> list[dict[str, str]]:
    conversations = source_record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(
            f"SciImpact SFT record on line {line_number} is missing conversations"
        )
    messages: list[dict[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        value = message.get("value")
        if role == "gpt":
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if role == "system":
            messages.append({"role": "system", "content": value.strip()})
        elif role == "human":
            messages.append({"role": "user", "content": value.strip()})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(
            f"SciImpact SFT record on line {line_number} has no human prompt"
        )
    messages[-1]["content"] = _append_strict_answer_instruction(
        messages[-1]["content"]
    )
    return messages


def _append_strict_answer_instruction(prompt: str) -> str:
    instruction = "Output exactly: Answer: A or Answer: B. Do not explain."
    stripped = prompt.rstrip()
    if instruction in stripped:
        return stripped
    return f"{stripped}\n\n{instruction}"


def _load_hard_flip_rows(
    flip_rows_path: Path,
    *,
    include_wrong: bool,
    include_inconsistent: bool,
) -> dict[int, dict[str, Any]]:
    hard_rows: dict[int, dict[str, Any]] = {}
    with flip_rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source_line = int(row["source_line"])
            if source_line in hard_rows:
                continue
            reasons = _hard_example_reasons(
                row,
                include_wrong=include_wrong,
                include_inconsistent=include_inconsistent,
            )
            if not reasons:
                continue
            hard_row = dict(row)
            hard_row["hard_example_reasons"] = reasons
            hard_rows[source_line] = hard_row
    return hard_rows


def _hard_example_reasons(
    row: dict[str, Any],
    *,
    include_wrong: bool,
    include_inconsistent: bool,
) -> list[str]:
    reasons = []
    if include_wrong and row.get("selected_outcome") == "wrong":
        reasons.append("selected_wrong")
    if include_inconsistent and row.get("consistent") is False:
        reasons.append("position_inconsistent")
    return reasons


def _annotate_hard_record(record: dict[str, Any], hard_row: dict[str, Any]) -> None:
    record["hard_example_reasons"] = list(hard_row["hard_example_reasons"])
    record["mining_source"] = "sciimpact_flip_consistency"
    for key in (
        "selected_outcome",
        "original_outcome",
        "flipped_mapped_outcome",
        "selected_source",
        "consistent",
    ):
        if key in hard_row:
            record[key] = hard_row[key]


def _training_leakage_warning(source: Path) -> str | None:
    source_texts = [source.name.lower()]
    for manifest_name in (
        "sciimpact_stratified_sample_manifest.json",
        "sciimpact_flipped_sft_manifest.json",
    ):
        manifest_path = source.parent / manifest_name
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        source_path = manifest.get("source_path")
        if isinstance(source_path, str):
            source_texts.append(source_path.lower())

    if not any("test" in source_text for source_text in source_texts):
        return None
    return (
        "Source path appears to be a test split. Use mined records from train/val "
        "splits for model training; test-derived hard examples are diagnostic only."
    )


def _pairwise_dpo_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    conversations = _prompt_conversations_from_pairwise_record(record, line_number)
    target = str(record.get("target") or "").strip()
    if target not in {"Answer: A", "Answer: B"}:
        raise ValueError(f"Pairwise SFT record on line {line_number} has invalid target")
    rejected = "Answer: B" if target == "Answer: A" else "Answer: A"
    source_line = record.get("source_line", line_number)
    return {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": target},
        "rejected": {"from": "gpt", "value": rejected},
        "target_paper_id": f"sciimpact-pairwise:{source_line}",
        "impact_dimension": str(record.get("impact_dimension") or "unknown"),
        "item_type": str(record.get("item_type") or "unknown"),
        "label_source": "sciimpact_pairwise_sft",
        "source_line": source_line,
    }


def _prompt_conversations_from_pairwise_record(
    record: dict[str, Any],
    line_number: int,
) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        conversations = _sharegpt_prompt_conversations(messages)
        if conversations:
            return conversations
    prompt = _prompt_from_pairwise_record(record, line_number)
    return [{"from": "human", "value": prompt}]


def _sharegpt_prompt_conversations(messages: list[Any]) -> list[dict[str, str]]:
    conversations: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            conversations.append({"from": "system", "value": content.strip()})
        elif role in {"user", "human"}:
            conversations.append({"from": "human", "value": content.strip()})
    return conversations


def _prompt_from_pairwise_record(record: dict[str, Any], line_number: int) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") in {"user", "human"} and isinstance(
                message.get("content"), str
            ):
                prompt = message["content"].strip()
                if prompt:
                    return prompt
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if message.get("from") in {"human", "user"} and isinstance(
                message.get("value"), str
            ):
                prompt = message["value"].strip()
                if prompt:
                    return prompt
    raise ValueError(f"Pairwise SFT record on line {line_number} has no user prompt")


def _winner_side(parsed: dict[str, Any], line_number: int) -> str:
    side = parsed["winner"].get("metadata", {}).get("sciimpact_source_side")
    if side not in {"a", "b"}:
        raise ValueError(f"SciImpact record on line {line_number} has no source side")
    return side


def _paper_from_sciimpact_record(record: dict[str, Any]) -> Paper:
    return Paper.model_validate(
        {
            "paper_id": record["paper_id"],
            "title": record.get("title") or "Untitled SciImpact item",
            "abstract": record.get("abstract"),
            "year": record.get("year"),
            "references": [],
            "metadata": record.get("metadata", {}),
        }
    )
