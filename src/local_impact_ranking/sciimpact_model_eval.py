from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.sciimpact_adapter import _parse_pair


def generate_sciimpact_predictions(
    sft_path: str | Path,
    output_path: str | Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int | None = 32,
    limit: int | None = None,
    resume: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_source_lines(destination) if resume else set()
    http_client = client or httpx.Client()
    written = 0
    skipped_existing = 0
    skipped_invalid = []
    source_record_count = 0

    with Path(sft_path).open("r", encoding="utf-8") as input_handle, destination.open(
        "a" if resume else "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            source_record_count += 1
            if limit is not None and source_record_count > limit:
                break
            if line_number in completed:
                skipped_existing += 1
                continue
            record = json.loads(stripped)
            try:
                _parse_pair(record, line_number)
                messages = _sciimpact_prompt_messages(record, line_number)
            except ValueError as exc:
                skipped_invalid.append({"source_line": line_number, "error": str(exc)})
                continue
            prediction = _complete_sciimpact_prompt(
                messages,
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                client=http_client,
            )
            output_handle.write(
                json.dumps(
                    {
                        "source_line": line_number,
                        "prediction": prediction,
                        "model": model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_handle.flush()
            written += 1

    manifest = {
        "source_path": str(sft_path),
        "output_path": str(destination),
        "model": model,
        "source_record_count": source_record_count,
        "written_prediction_count": written,
        "skipped_existing_count": skipped_existing,
        "skipped_invalid_count": len(skipped_invalid),
        "skipped_invalid": skipped_invalid[:20],
        "limit": limit,
    }
    (destination.parent / "sciimpact_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def evaluate_sciimpact_predictions(
    sft_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    predictions = _load_predictions(predictions_path)
    rows = []
    skipped = []
    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = _parse_pair(json.loads(stripped), line_number)
                expected_side = _side_from_winner(parsed, line_number)
            except ValueError as exc:
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            prediction = predictions.get(line_number)
            predicted_side = _side_from_prediction(prediction) if prediction else None
            if predicted_side is None:
                outcome = "uncovered"
            elif predicted_side == expected_side:
                outcome = "correct"
            else:
                outcome = "wrong"
            rows.append(
                {
                    "source_line": line_number,
                    "expected_side": expected_side.upper(),
                    "predicted_side": predicted_side.upper() if predicted_side else None,
                    "impact_dimension": parsed.get("impact_dimension", "unknown"),
                    "item_type": parsed.get("item_type", "unknown"),
                    "outcome": outcome,
                }
            )

    report = {
        "summary": _summarize(rows),
        "by_dimension": _summarize_groups(rows, "impact_dimension"),
        "by_item_type": _summarize_groups(rows, "item_type"),
        "skipped_source_count": len(skipped),
        "skipped_sources": skipped[:20],
        "examples": rows[:20],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_model_eval.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_eval.md").write_text(
        render_sciimpact_model_eval_markdown(report),
        encoding="utf-8",
    )
    return report


def _completed_source_lines(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            source_line = record.get("source_line")
            if source_line is not None:
                completed.add(int(source_line))
    return completed


def _sciimpact_prompt_messages(record: dict[str, Any], line_number: int) -> list[dict[str, str]]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"SciImpact record on line {line_number} is missing conversations")
    messages = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from")
        value = message.get("value")
        if role == "gpt":
            continue
        if not isinstance(role, str) or not isinstance(value, str) or not value.strip():
            continue
        if role == "system":
            openai_role = "system"
        elif role == "human":
            openai_role = "user"
        else:
            continue
        messages.append({"role": openai_role, "content": value})
    if not messages:
        raise ValueError(f"SciImpact record on line {line_number} has no prompt messages")
    return messages


def _complete_sciimpact_prompt(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    client: Any,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_payload = response.json()
    return str(response_payload["choices"][0]["message"]["content"]).strip()


def render_sciimpact_model_eval_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SciImpact Model Evaluation",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## By Impact Dimension",
            "",
            "| Group | Total | Covered | Accuracy | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_dimension"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## By Item Type",
            "",
            "| Group | Total | Covered | Accuracy | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_item_type"].items()):
        lines.append(_group_row(group, summary))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Accuracy is computed over covered examples only. Coverage reports how many source lines had a parseable A/B prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_predictions(path: str | Path) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            source_line = record.get("source_line", line_number)
            prediction = record.get("prediction") or record.get("answer") or record.get("text")
            if prediction is None:
                continue
            predictions[int(source_line)] = str(prediction)
    return predictions


def _side_from_winner(parsed: dict[str, Any], line_number: int) -> str:
    source_side = parsed["winner"].get("metadata", {}).get("sciimpact_source_side")
    if source_side not in {"a", "b"}:
        raise ValueError(f"SciImpact record on line {line_number} has no winner side")
    return source_side


def _side_from_prediction(prediction: str | None) -> str | None:
    if prediction is None:
        return None
    match = re.search(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+([AB])\b", prediction, re.I)
    if match is None:
        return None
    return match.group(1).lower()


def _summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {group: _summarize(group_rows) for group, group_rows in grouped.items()}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(row["outcome"] != "uncovered" for row in rows)
    correct = sum(row["outcome"] == "correct" for row in rows)
    wrong = sum(row["outcome"] == "wrong" for row in rows)
    uncovered = total - covered
    return {
        "total_examples": total,
        "covered_examples": covered,
        "uncovered_examples": uncovered,
        "correct_examples": correct,
        "wrong_examples": wrong,
        "coverage_rate": _safe_divide(covered, total),
        "accuracy": _safe_divide(correct, covered),
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_row(group: str, summary: dict[str, Any]) -> str:
    return (
        f"| {group} | {summary['total_examples']} | {summary['covered_examples']} | "
        f"{summary['accuracy']:.3f} | {summary['coverage_rate']:.3f} |"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
