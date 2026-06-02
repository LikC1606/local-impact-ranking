from __future__ import annotations

import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.sciimpact_adapter import _flip_sft_answer, _parse_pair
from local_impact_ranking.sciimpact_pairwise_sft import (
    _pairwise_dpo_record,
    _pairwise_sft_record,
)

_SFT_ITEM_MARKERS = (
    ("paper", "Paper A:", "Paper B:"),
    ("dataset", "Dataset A:", "Dataset B:"),
    ("github_repo", "GitHub repo A README:", "GitHub repo B README:"),
    ("model", "Model A:", "Model B:"),
)

_SFT_QUESTION_SUFFIX_PATTERNS = (
    r"\s+Which paper\b.*$",
    r"\s+Which dataset\b.*$",
    r"\s+Which repository\b.*$",
    r"\s+Which model\b.*$",
    r"\s+Reply with exactly one sentence\b.*$",
)


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
    concurrency: int = 1,
    answer_format: str = "freeform",
    client: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    answer_format = answer_format.replace("-", "_")
    if answer_format not in {
        "freeform",
        "final_answer",
        "strict_answer",
        "constrained_answer",
        "choice_logprob",
        "structured_choice",
    }:
        raise ValueError(
            "answer_format must be 'freeform', 'final_answer', "
            "'strict_answer', 'constrained_answer', 'choice_logprob', "
            "or 'structured_choice'"
        )
    if answer_format == "structured_choice":
        max_tokens = 4
    elif answer_format == "choice_logprob":
        max_tokens = 1
    elif answer_format == "constrained_answer":
        max_tokens = 3
    elif answer_format == "strict_answer" and max_tokens is None:
        max_tokens = 4
    elif answer_format == "strict_answer":
        max_tokens = min(max_tokens, 4)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_source_lines(destination) if resume else set()
    owns_client = client is None
    http_client = client or httpx.Client()
    written = 0
    skipped_existing = 0
    skipped_invalid = []
    source_record_count = 0
    pending: list[dict[str, Any]] = []

    with Path(sft_path).open("r", encoding="utf-8") as input_handle:
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if limit is not None and source_record_count >= limit:
                break
            source_record_count += 1
            record = json.loads(stripped)
            if line_number in completed:
                skipped_existing += 1
                continue
            try:
                _expected_side_for_record(record, line_number)
                messages = _sciimpact_prompt_messages(record, line_number)
                if answer_format == "final_answer":
                    messages = _with_final_answer_instruction(messages)
                elif answer_format == "choice_logprob":
                    messages = _with_choice_only_instruction(messages)
                elif answer_format in {
                    "strict_answer",
                    "constrained_answer",
                    "structured_choice",
                }:
                    messages = _with_strict_answer_instruction(messages)
            except ValueError as exc:
                skipped_invalid.append({"source_line": line_number, "error": str(exc)})
                continue
            pending.append({"source_line": line_number, "messages": messages})

    confidence_enabled = answer_format in {
        "constrained_answer",
        "choice_logprob",
        "structured_choice",
    }

    def complete(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        prediction = _complete_sciimpact_prompt(
            task["messages"],
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            allowed_token_ids=_allowed_answer_token_ids(answer_format),
            structured_choice=_structured_answer_choice(answer_format),
            request_logprobs=confidence_enabled,
            client=http_client,
        )
        return int(task["source_line"]), prediction

    try:
        with destination.open("a" if resume else "w", encoding="utf-8") as output_handle:
            if concurrency == 1:
                for task in pending:
                    source_line, prediction = complete(task)
                    _write_prediction(output_handle, source_line, prediction, model)
                    written += 1
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(complete, task) for task in pending]
                    for future in as_completed(futures):
                        source_line, prediction = future.result()
                        _write_prediction(output_handle, source_line, prediction, model)
                        written += 1
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    manifest = {
        "source_path": str(sft_path),
        "output_path": str(destination),
        "model": model,
        "source_record_count": source_record_count,
        "queued_prediction_count": len(pending),
        "written_prediction_count": written,
        "skipped_existing_count": skipped_existing,
        "skipped_invalid_count": len(skipped_invalid),
        "skipped_invalid": skipped_invalid[:20],
        "limit": limit,
        "concurrency": concurrency,
        "answer_format": answer_format,
        "confidence_enabled": confidence_enabled,
    }
    (destination.parent / "sciimpact_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_sciimpact_stratified_sample(
    sft_path: str | Path,
    output_path: str | Path,
    *,
    per_dimension: int,
    seed: int = 0,
    dimensions: list[str] | None = None,
    dimension_weights: dict[str, int] | None = None,
    item_type_weights: dict[str, int] | None = None,
    skip_invalid: bool = True,
) -> dict[str, Any]:
    if per_dimension < 1:
        raise ValueError("per_dimension must be at least 1")
    dimension_weights = _normalize_dimension_weights(dimension_weights)
    item_type_weights = _normalize_item_type_weights(item_type_weights)
    source = Path(sft_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_dimensions = set(dimensions) if dimensions else None
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = []
    source_record_count = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            source_record_count += 1
            record = json.loads(stripped)
            try:
                dimension, item_type = _sampling_record_dimension_and_item_type(
                    record,
                    line_number,
                )
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            if target_dimensions is not None and dimension not in target_dimensions:
                continue
            buckets[dimension].append(
                {
                    "source_line": line_number,
                    "dimension": dimension,
                    "item_type": item_type,
                    "raw": stripped,
                }
            )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    by_dimension: dict[str, dict[str, int]] = {}
    by_item_type: dict[str, dict[str, int]] = {}
    dimension_names = sorted(target_dimensions if target_dimensions is not None else buckets)
    for dimension in dimension_names:
        records = buckets.get(dimension, [])
        if item_type_weights:
            dimension_target_count = 0
            dimension_sampled_count = 0
            records_by_item_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                records_by_item_type[str(record["item_type"])].append(record)
            for item_type, item_records in sorted(records_by_item_type.items()):
                target_count = (
                    per_dimension
                    * dimension_weights.get(dimension, 1)
                    * item_type_weights.get(item_type, 1)
                )
                sampled = _sample_records(item_records, target_count, rng)
                selected.extend(sampled)
                sample_size = len(sampled)
                dimension_target_count += target_count
                dimension_sampled_count += sample_size
                _add_sampling_counts(
                    by_item_type,
                    item_type,
                    available_count=len(item_records),
                    target_count=target_count,
                    sampled_count=sample_size,
                )
            by_dimension[dimension] = {
                "available_count": len(records),
                "target_count": dimension_target_count,
                "sampled_count": dimension_sampled_count,
            }
        else:
            target_count = per_dimension * dimension_weights.get(dimension, 1)
            sampled = _sample_records(records, target_count, rng)
            selected.extend(sampled)
            sample_size = len(sampled)
            by_dimension[dimension] = {
                "available_count": len(records),
                "target_count": target_count,
                "sampled_count": sample_size,
            }
            for item_type, item_records in _group_records(sampled, "item_type").items():
                _add_sampling_counts(
                    by_item_type,
                    item_type,
                    available_count=sum(
                        1 for record in records if record["item_type"] == item_type
                    ),
                    target_count=0,
                    sampled_count=len(item_records),
                )

    selected.sort(key=lambda record: int(record["source_line"]))
    with destination.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(str(record["raw"]) + "\n")

    sampled_records = [
        {
            "sample_line": index,
            "source_line": int(record["source_line"]),
            "impact_dimension": str(record["dimension"]),
            "item_type": str(record["item_type"]),
        }
        for index, record in enumerate(selected, start=1)
    ]
    manifest = {
        "source_path": str(source),
        "output_path": str(destination),
        "source_record_count": source_record_count,
        "sample_count": len(selected),
        "per_dimension": per_dimension,
        "seed": seed,
        "dimensions": dimension_names,
        "dimension_weights": dimension_weights,
        "item_type_weights": item_type_weights,
        "by_dimension": by_dimension,
        "by_item_type": dict(sorted(by_item_type.items())),
        "skipped_invalid_count": len(skipped),
        "skipped_invalid": skipped[:20],
        "sampled_records": sampled_records,
    }
    (destination.parent / "sciimpact_stratified_sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sampling_record_dimension_and_item_type(
    record: dict[str, Any],
    line_number: int,
) -> tuple[str, str]:
    try:
        parsed = _parse_pair(record, line_number)
        dimension = str(parsed.get("impact_dimension") or "unknown")
        item_type = str(parsed.get("item_type") or "unknown")
        return _normalize_sciimpact_dimension(dimension), _normalize_sciimpact_item_type(
            item_type
        )
    except ValueError as exc:
        if not _is_exported_pairwise_sft_record(record):
            raise exc
        dimension = str(record.get("impact_dimension") or "unknown")
        item_type = str(record.get("item_type") or "unknown")
        return _normalize_sciimpact_dimension(dimension), _normalize_sciimpact_item_type(
            item_type
        )


def _is_exported_pairwise_sft_record(record: dict[str, Any]) -> bool:
    target = str(record.get("target") or "").strip()
    return target in {"Answer: A", "Answer: B"} and isinstance(record.get("messages"), list)


def _sample_records(
    records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    sample_size = min(target_count, len(records))
    return rng.sample(records, sample_size) if sample_size < len(records) else list(records)


def _add_sampling_counts(
    summaries: dict[str, dict[str, int]],
    key: str,
    *,
    available_count: int,
    target_count: int,
    sampled_count: int,
) -> None:
    row = summaries.setdefault(
        key,
        {
            "available_count": 0,
            "target_count": 0,
            "sampled_count": 0,
        },
    )
    row["available_count"] += available_count
    row["target_count"] += target_count
    row["sampled_count"] += sampled_count


def _normalize_dimension_weights(
    dimension_weights: dict[str, int] | None,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for dimension, weight in (dimension_weights or {}).items():
        normalized_dimension = _normalize_sciimpact_dimension(str(dimension))
        numeric_weight = int(weight)
        if numeric_weight < 1:
            raise ValueError("dimension weights must be positive integers")
        normalized[normalized_dimension] = numeric_weight
    return normalized


def _normalize_sciimpact_dimension(dimension: str) -> str:
    lowered = dimension.strip().lower().replace("_", "-")
    aliases = {
        "download": "downloads",
        "downloads": "downloads",
        "dataset-download": "downloads",
        "dataset-downloads": "downloads",
        "star": "stars",
        "stars": "stars",
        "github-star": "stars",
        "github-stars": "stars",
        "media": "media",
        "media-mention": "media",
        "media-mentions": "media",
        "patent": "patent",
        "patents": "patent",
        "citation": "citation",
        "citations": "citation",
        "award": "award",
        "awards": "award",
    }
    return aliases.get(lowered, lowered)


def _normalize_item_type_weights(
    item_type_weights: dict[str, int] | None,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for item_type, weight in (item_type_weights or {}).items():
        normalized_item_type = _normalize_sciimpact_item_type(str(item_type))
        numeric_weight = int(weight)
        if numeric_weight < 1:
            raise ValueError("item type weights must be positive integers")
        normalized[normalized_item_type] = numeric_weight
    return normalized


def _normalize_sciimpact_item_type(item_type: str) -> str:
    lowered = item_type.strip().lower().replace("-", "_")
    aliases = {
        "dataset": "dataset",
        "datasets": "dataset",
        "github": "github_repo",
        "github_repo": "github_repo",
        "github_repos": "github_repo",
        "repo": "github_repo",
        "repos": "github_repo",
        "repository": "github_repo",
        "repositories": "github_repo",
        "model": "model",
        "models": "model",
        "paper": "paper",
        "papers": "paper",
    }
    return aliases.get(lowered, lowered)


def write_sciimpact_flipped_sft(
    sft_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    skip_invalid: bool = True,
) -> dict[str, Any]:
    source = Path(sft_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_record_count = 0
    flipped_record_count = 0
    skipped: list[dict[str, Any]] = []

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
            record = json.loads(stripped)
            try:
                flipped = _flip_sciimpact_sft_record(record, line_number)
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            output_handle.write(json.dumps(flipped, ensure_ascii=False) + "\n")
            flipped_record_count += 1

    manifest = {
        "source_path": str(source),
        "output_path": str(destination),
        "source_record_count": source_record_count,
        "flipped_record_count": flipped_record_count,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped[:20],
        "limit": limit,
    }
    (destination.parent / "sciimpact_flipped_sft_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def evaluate_sciimpact_predictions(
    sft_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    calibration_report_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions = _load_predictions(predictions_path)
    calibration_model = _load_empirical_calibration(calibration_report_path)
    rows = []
    skipped = []
    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if limit is not None and line_number > limit:
                break
            try:
                record = json.loads(stripped)
                parsed = _parse_pair_metadata(record, line_number)
                expected_side = _expected_side_for_record(record, line_number)
            except ValueError as exc:
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            prediction = predictions.get(line_number)
            predicted_text = _prediction_text(prediction)
            predicted_side = _side_from_prediction(predicted_text) if prediction else None
            if predicted_side is None:
                outcome = "uncovered"
            elif predicted_side == expected_side:
                outcome = "correct"
            else:
                outcome = "wrong"
            confidence = _prediction_confidence(prediction)
            calibrated_confidence = _calibrated_confidence(
                confidence,
                calibration_model,
            )
            rows.append(
                {
                    "source_line": line_number,
                    "expected_side": expected_side.upper(),
                    "predicted_side": predicted_side.upper() if predicted_side else None,
                    "impact_dimension": parsed.get("impact_dimension", "unknown"),
                    "item_type": parsed.get("item_type", "unknown"),
                    "outcome": outcome,
                    "prediction": predicted_text,
                    "confidence": confidence,
                    "confidence_bucket": _confidence_bucket(confidence),
                    "calibrated_confidence": calibrated_confidence,
                    "calibrated_confidence_method": calibration_model["method"]
                    if calibrated_confidence is not None
                    else None,
                    "confidence_method": _prediction_field(prediction, "confidence_method"),
                    "choice_logprob_margin": _prediction_field(
                        prediction,
                        "choice_logprob_margin",
                    ),
                }
            )

    summary = _summarize(rows)
    by_dimension = _summarize_groups(rows, "impact_dimension")
    by_item_type = _summarize_groups(rows, "item_type")
    significance = _model_statistical_significance(summary)
    scorecard = _model_credibility_scorecard(
        summary,
        significance=significance,
        by_dimension=by_dimension,
        by_item_type=by_item_type,
    )
    report = {
        "summary": summary,
        "statistical_significance": significance,
        "credibility_scorecard": scorecard,
        "by_dimension": by_dimension,
        "by_item_type": by_item_type,
        "confidence_summary": _confidence_summary(rows),
        "calibrated_confidence_summary": _confidence_summary(
            _rows_with_confidence_alias(rows, "calibrated_confidence")
        )
        if calibration_model
        else None,
        "calibrated_calibration": _calibration_report(
            _rows_with_confidence_alias(rows, "calibrated_confidence")
        )
        if calibration_model
        else None,
        "calibration_model": _calibration_model_report(calibration_model),
        "calibration": _calibration_report(rows),
        "by_confidence_bucket": _summarize_groups(
            [row for row in rows if row["confidence_bucket"] != "unscored"],
            "confidence_bucket",
        ),
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
    (destination / "sciimpact_model_credibility_scorecard.json").write_text(
        json.dumps(report["credibility_scorecard"], indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_credibility_scorecard.md").write_text(
        render_sciimpact_model_credibility_scorecard_markdown(
            report["credibility_scorecard"]
        ),
        encoding="utf-8",
    )
    _write_jsonl(destination / "sciimpact_model_eval_rows.jsonl", rows)
    _write_jsonl(
        destination / "sciimpact_model_eval_wrong.jsonl",
        [row for row in rows if row["outcome"] == "wrong"],
    )
    error_analysis = sciimpact_model_error_analysis(rows)
    (destination / "sciimpact_model_error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_error_analysis.md").write_text(
        render_sciimpact_model_error_analysis_markdown(error_analysis),
        encoding="utf-8",
    )
    return report


def write_sciimpact_model_credibility_summary(
    evaluation_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    reports = [_load_sciimpact_model_eval_report(path) for path in evaluation_paths]
    if not reports:
        raise ValueError("at least one SciImpact model evaluation is required")

    evaluations = []
    total_examples = 0
    covered_examples = 0
    correct_examples = 0
    for index, report in enumerate(reports, start=1):
        summary = report["report"].get("summary", {})
        total = int(summary.get("total_examples", 0) or 0)
        covered = int(summary.get("covered_examples", 0) or 0)
        correct = int(summary.get("correct_examples", 0) or 0)
        total_examples += total
        covered_examples += covered
        correct_examples += correct
        evaluations.append(
            {
                "label": _model_eval_label(report["path"], index),
                "path": str(report["path"]),
                "total_examples": total,
                "covered_examples": covered,
                "correct_examples": correct,
                "coverage_rate": round(
                    float(summary.get("coverage_rate", 0.0) or 0.0),
                    6,
                ),
                "accuracy": round(float(summary.get("accuracy", 0.0) or 0.0), 6),
                "scorecard_status": report["report"]
                .get("credibility_scorecard", {})
                .get("status", "unknown"),
            }
        )

    pooled_summary = {
        "total_examples": total_examples,
        "covered_examples": covered_examples,
        "uncovered_examples": max(0, total_examples - covered_examples),
        "correct_examples": correct_examples,
        "wrong_examples": max(0, covered_examples - correct_examples),
        "coverage_rate": round(_safe_divide(covered_examples, total_examples), 6),
        "accuracy": round(_safe_divide(correct_examples, covered_examples), 6),
    }
    accuracies = [row["accuracy"] for row in evaluations]
    coverages = [row["coverage_rate"] for row in evaluations]
    slice_rows = _model_eval_summary_slices(reports)
    weak_slices = [
        row
        for row in slice_rows
        if row["pooled_accuracy"] < 0.60 or row["min_seed_accuracy"] < 0.60
    ]
    significance = _model_statistical_significance(pooled_summary)
    summary = {
        "evaluation_count": len(evaluations),
        "total_examples": total_examples,
        "covered_examples": covered_examples,
        "correct_examples": correct_examples,
        "pooled_accuracy": pooled_summary["accuracy"],
        "mean_accuracy": _mean(row["accuracy"] for row in evaluations),
        "min_accuracy": round(min(accuracies), 6) if accuracies else 0.0,
        "max_accuracy": round(max(accuracies), 6) if accuracies else 0.0,
        "accuracy_range": round(max(accuracies) - min(accuracies), 6)
        if accuracies
        else 0.0,
        "mean_coverage_rate": _mean(row["coverage_rate"] for row in evaluations),
        "min_coverage_rate": round(min(coverages), 6) if coverages else 0.0,
    }
    scorecard = _model_credibility_summary_scorecard(
        summary,
        significance=significance,
        slice_rows=slice_rows,
    )
    report = {
        "summary": summary,
        "pooled_statistical_significance": significance,
        "credibility_scorecard": scorecard,
        "evaluations": evaluations,
        "weak_slices": weak_slices[:30],
        "all_slices": slice_rows,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_model_credibility_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_credibility_summary.md").write_text(
        render_sciimpact_model_credibility_summary_markdown(report),
        encoding="utf-8",
    )
    return report


def write_sciimpact_model_candidate_gate(
    baseline_summary_path: str | Path,
    candidate_summary_path: str | Path,
    output_dir: str | Path,
    *,
    min_pooled_accuracy_delta: float = 0.005,
    min_evaluations: int = 3,
    max_min_seed_accuracy_regression: float = 0.0,
    max_accuracy_range_increase: float = 0.02,
    max_weak_slice_count_increase: int = 0,
    max_worst_slice_regression: float = 0.02,
) -> dict[str, Any]:
    baseline = _load_sciimpact_model_credibility_summary(baseline_summary_path)
    candidate = _load_sciimpact_model_credibility_summary(candidate_summary_path)
    baseline_summary = baseline["report"].get("summary", {})
    candidate_summary = candidate["report"].get("summary", {})
    baseline_scorecard = baseline["report"].get("credibility_scorecard", {})
    candidate_scorecard = candidate["report"].get("credibility_scorecard", {})
    baseline_weak_count = _model_summary_weak_slice_count(baseline["report"])
    candidate_weak_count = _model_summary_weak_slice_count(candidate["report"])
    baseline_worst_slice = _model_summary_worst_slice_accuracy(baseline["report"])
    candidate_worst_slice = _model_summary_worst_slice_accuracy(candidate["report"])

    comparison = {
        "baseline_status": baseline_scorecard.get("status", "unknown"),
        "candidate_status": candidate_scorecard.get("status", "unknown"),
        "baseline_evaluation_count": int(
            baseline_summary.get("evaluation_count", 0) or 0
        ),
        "candidate_evaluation_count": int(
            candidate_summary.get("evaluation_count", 0) or 0
        ),
        "baseline_pooled_accuracy": float(
            baseline_summary.get("pooled_accuracy", 0.0) or 0.0
        ),
        "candidate_pooled_accuracy": float(
            candidate_summary.get("pooled_accuracy", 0.0) or 0.0
        ),
        "baseline_min_accuracy": float(baseline_summary.get("min_accuracy", 0.0) or 0.0),
        "candidate_min_accuracy": float(
            candidate_summary.get("min_accuracy", 0.0) or 0.0
        ),
        "baseline_accuracy_range": float(
            baseline_summary.get("accuracy_range", 0.0) or 0.0
        ),
        "candidate_accuracy_range": float(
            candidate_summary.get("accuracy_range", 0.0) or 0.0
        ),
        "baseline_weak_slice_count": baseline_weak_count,
        "candidate_weak_slice_count": candidate_weak_count,
        "baseline_worst_slice_min_accuracy": baseline_worst_slice,
        "candidate_worst_slice_min_accuracy": candidate_worst_slice,
    }
    comparison.update(
        {
            "pooled_accuracy_delta": round(
                comparison["candidate_pooled_accuracy"]
                - comparison["baseline_pooled_accuracy"],
                6,
            ),
            "min_seed_accuracy_delta": round(
                comparison["candidate_min_accuracy"]
                - comparison["baseline_min_accuracy"],
                6,
            ),
            "accuracy_range_delta": round(
                comparison["candidate_accuracy_range"]
                - comparison["baseline_accuracy_range"],
                6,
            ),
            "weak_slice_count_delta": (
                comparison["candidate_weak_slice_count"]
                - comparison["baseline_weak_slice_count"]
            ),
            "worst_slice_min_accuracy_delta": round(
                comparison["candidate_worst_slice_min_accuracy"]
                - comparison["baseline_worst_slice_min_accuracy"],
                6,
            ),
        }
    )
    thresholds = {
        "min_pooled_accuracy_delta": min_pooled_accuracy_delta,
        "min_evaluations": min_evaluations,
        "max_min_seed_accuracy_regression": max_min_seed_accuracy_regression,
        "max_accuracy_range_increase": max_accuracy_range_increase,
        "max_weak_slice_count_increase": max_weak_slice_count_increase,
        "max_worst_slice_regression": max_worst_slice_regression,
    }
    checks = _model_candidate_gate_checks(
        comparison,
        thresholds,
        baseline_status=str(baseline_scorecard.get("status", "unknown")),
        candidate_status=str(candidate_scorecard.get("status", "unknown")),
    )
    blocking = [check["reason"] for check in checks.values() if check["status"] == "fail"]
    decision = "promote" if not blocking else "do_not_promote"
    reason = (
        "Candidate improves cross-seed model evidence without gate regressions."
        if decision == "promote"
        else blocking[0]
    )
    report = {
        "decision": decision,
        "reason": reason,
        "baseline_summary_path": str(baseline["path"]),
        "candidate_summary_path": str(candidate["path"]),
        "thresholds": thresholds,
        "comparison": comparison,
        "checks": checks,
        "blocking_reasons": blocking,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_model_candidate_gate.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_candidate_gate.md").write_text(
        render_sciimpact_model_candidate_gate_markdown(report),
        encoding="utf-8",
    )
    return report


def write_sciimpact_model_weak_slice_error_pack(
    evaluation_paths: list[str | Path],
    output_dir: str | Path,
    *,
    weak_slice_threshold: float = 0.60,
    training_source: bool = False,
    sft_paths: list[str | Path] | None = None,
    prompt_style: str = "source_task",
) -> dict[str, Any]:
    reports = [_load_sciimpact_model_eval_report(path) for path in evaluation_paths]
    if not reports:
        raise ValueError("at least one SciImpact model evaluation is required")
    if sft_paths is not None and len(sft_paths) != len(reports):
        raise ValueError("sft_paths must match the number of evaluation paths")
    prompt_style = prompt_style.replace("-", "_")
    slice_rows = _model_eval_summary_slices(reports)
    weak_slices = {
        row["slice"]: row
        for row in slice_rows
        if row["pooled_accuracy"] < weak_slice_threshold
        or row["min_seed_accuracy"] < weak_slice_threshold
    }
    error_rows = []
    training_sft_rows = []
    training_dpo_rows = []
    training_source_rows = []
    skipped_training_rows = []
    by_weak_slice: dict[str, int] = defaultdict(int)
    for index, report in enumerate(reports, start=1):
        label = _model_eval_label(report["path"], index)
        rows_path = report["path"].parent / "sciimpact_model_eval_rows.jsonl"
        if not rows_path.exists():
            raise FileNotFoundError(f"SciImpact model rows not found: {rows_path}")
        sft_records = (
            _load_sft_records_by_line(sft_paths[index - 1])
            if sft_paths is not None
            else None
        )
        for row in _read_jsonl(rows_path):
            if row.get("outcome") not in {"wrong", "uncovered"}:
                continue
            matched = _matched_weak_slices(row, weak_slices)
            if not matched:
                continue
            for slice_name in matched:
                by_weak_slice[slice_name] += 1
            error_rows.append(
                {
                    "evaluation_label": label,
                    "evaluation_path": str(report["path"]),
                    "source_line": row.get("source_line"),
                    "matched_weak_slices": matched,
                    "impact_dimension": row.get("impact_dimension"),
                    "item_type": row.get("item_type"),
                    "outcome": row.get("outcome"),
                    "expected_side": row.get("expected_side"),
                    "predicted_side": row.get("predicted_side"),
                    "confidence": row.get("confidence"),
                    "confidence_bucket": row.get("confidence_bucket"),
                    "prediction": row.get("prediction"),
                }
            )
            error_record = error_rows[-1]
            if sft_records is not None:
                source_line = error_record.get("source_line")
                source_record = sft_records.get(int(source_line or 0))
                if source_record is None:
                    skipped_training_rows.append(
                        {
                            "evaluation_label": label,
                            "source_line": source_line,
                            "reason": "source line not found in SFT path",
                        }
                    )
                    continue
                try:
                    parsed = _parse_pair(source_record, int(source_line))
                    sft_record = _pairwise_sft_record(
                        source_record,
                        parsed,
                        int(source_line),
                        prompt_style=prompt_style,
                    )
                    _annotate_weak_slice_training_record(sft_record, error_record)
                    dpo_record = _pairwise_dpo_record(sft_record, int(source_line))
                    _annotate_weak_slice_training_record(dpo_record, error_record)
                    dpo_record["label_source"] = (
                        "sciimpact_model_weak_slice_error_pack"
                    )
                except ValueError as exc:
                    skipped_training_rows.append(
                        {
                            "evaluation_label": label,
                            "source_line": source_line,
                            "reason": str(exc),
                        }
                    )
                    continue
                training_sft_rows.append(sft_record)
                training_dpo_rows.append(dpo_record)
                training_source_rows.append(
                    {
                        "source_line": source_line,
                        "weak_slice_error": error_record,
                        "source_record": source_record,
                    }
                )

    recommended_use = "training_candidate" if training_source else "diagnostic_only"
    leakage_warning = None
    if not training_source:
        leakage_warning = (
            "Defaulting to diagnostic_only. Do not train on held-out/test "
            "evaluation errors; regenerate this pack from train-split predictions "
            "before using rows as training data."
        )
    manifest = {
        "evaluation_count": len(reports),
        "weak_slice_threshold": weak_slice_threshold,
        "weak_slice_count": len(weak_slices),
        "error_count": len(error_rows),
        "high_confidence_error_count": sum(
            float(row.get("confidence") or 0.0) >= 0.80 for row in error_rows
        ),
        "by_weak_slice": dict(sorted(by_weak_slice.items())),
        "recommended_use": recommended_use,
        "training_source": training_source,
        "leakage_warning": leakage_warning,
        "weak_slices": list(weak_slices.values()),
        "sft_paths": [str(path) for path in sft_paths] if sft_paths else [],
        "prompt_style": prompt_style,
        "training_record_count": len(training_sft_rows),
        "skipped_training_record_count": len(skipped_training_rows),
        "skipped_training_records": skipped_training_rows[:20],
    }
    report = {
        "manifest": manifest,
        "errors_preview": error_rows[:20],
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_model_weak_slice_error_pack.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_model_weak_slice_error_pack.md").write_text(
        render_sciimpact_model_weak_slice_error_pack_markdown(report),
        encoding="utf-8",
    )
    _write_jsonl(
        destination / "sciimpact_model_weak_slice_errors.jsonl",
        error_rows,
    )
    if sft_paths is not None:
        _write_jsonl(destination / "train_pairwise_sft.jsonl", training_sft_rows)
        _write_jsonl(destination / "train_pairwise_dpo.jsonl", training_dpo_rows)
        _write_jsonl(
            destination / "weak_slice_source_records.jsonl",
            training_source_rows,
        )
    return report


def evaluate_sciimpact_flip_consistency(
    sft_path: str | Path,
    predictions_path: str | Path,
    flipped_predictions_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    confidence_margin: float = 0.0,
) -> dict[str, Any]:
    if confidence_margin < 0.0:
        raise ValueError("confidence_margin must be non-negative")
    predictions = _load_predictions(predictions_path)
    flipped_predictions = _load_predictions(flipped_predictions_path)
    rows = []
    skipped = []
    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if limit is not None and line_number > limit:
                break
            try:
                record = json.loads(stripped)
                parsed = _parse_pair_metadata(record, line_number)
                expected_side = _expected_side_for_record(record, line_number)
            except ValueError as exc:
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue

            original_prediction = predictions.get(line_number)
            flipped_prediction = flipped_predictions.get(line_number)
            original_text = _prediction_text(original_prediction)
            flipped_text = _prediction_text(flipped_prediction)
            original_side = _side_from_prediction(original_text) if original_prediction else None
            flipped_side = _side_from_prediction(flipped_text) if flipped_prediction else None
            flipped_mapped_side = _opposite_side(flipped_side)
            original_confidence = _prediction_confidence(original_prediction)
            flipped_confidence = _prediction_confidence(flipped_prediction)
            selected_side, selected_source = _select_flip_consistency_side(
                original_side,
                flipped_mapped_side,
                original_confidence,
                flipped_confidence,
                confidence_margin=confidence_margin,
            )
            rows.append(
                {
                    "source_line": line_number,
                    "expected_side": expected_side.upper(),
                    "original_side": original_side.upper() if original_side else None,
                    "flipped_side": flipped_side.upper() if flipped_side else None,
                    "flipped_mapped_side": flipped_mapped_side.upper()
                    if flipped_mapped_side
                    else None,
                    "selected_side": selected_side.upper() if selected_side else None,
                    "impact_dimension": parsed.get("impact_dimension", "unknown"),
                    "item_type": parsed.get("item_type", "unknown"),
                    "original_outcome": _side_outcome(original_side, expected_side),
                    "flipped_mapped_outcome": _side_outcome(
                        flipped_mapped_side,
                        expected_side,
                    ),
                    "selected_outcome": _side_outcome(selected_side, expected_side),
                    "consistent": original_side is not None
                    and flipped_mapped_side is not None
                    and original_side == flipped_mapped_side,
                    "selected_source": selected_source,
                    "original_prediction": original_text,
                    "flipped_prediction": flipped_text,
                    "original_confidence": original_confidence,
                    "flipped_confidence": flipped_confidence,
                    "original_confidence_method": _prediction_field(
                        original_prediction,
                        "confidence_method",
                    ),
                    "flipped_confidence_method": _prediction_field(
                        flipped_prediction,
                        "confidence_method",
                    ),
                    "original_choice_logprob_margin": _prediction_field(
                        original_prediction,
                        "choice_logprob_margin",
                    ),
                    "flipped_choice_logprob_margin": _prediction_field(
                        flipped_prediction,
                        "choice_logprob_margin",
                    ),
                }
            )

    report = {
        "summary": _flip_consistency_summary(rows),
        "by_dimension": _flip_consistency_groups(rows, "impact_dimension"),
        "by_item_type": _flip_consistency_groups(rows, "item_type"),
        "selection_summary": _selection_summary(rows),
        "skipped_source_count": len(skipped),
        "skipped_sources": skipped[:20],
        "selection_config": {"confidence_margin": confidence_margin},
        "examples": rows[:20],
    }
    report["summary"]["confidence_margin"] = confidence_margin
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sciimpact_flip_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_flip_consistency.md").write_text(
        render_sciimpact_flip_consistency_markdown(report),
        encoding="utf-8",
    )
    _write_jsonl(destination / "sciimpact_flip_consistency_rows.jsonl", rows)
    _write_jsonl(
        destination / "sciimpact_flip_consistency_inconsistent.jsonl",
        [
            row
            for row in rows
            if row["original_side"] is not None
            and row["flipped_mapped_side"] is not None
            and not row["consistent"]
        ],
    )
    _write_jsonl(
        destination / "sciimpact_flip_selected_predictions.jsonl",
        _flip_selected_predictions(rows),
    )
    return report


def write_sciimpact_calibration_split(
    sft_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    predictions = _load_predictions(predictions_path)
    records: list[dict[str, Any]] = []
    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                parsed = _parse_pair_metadata(record, line_number)
            except ValueError:
                parsed = {
                    "impact_dimension": record.get("impact_dimension", "unknown"),
                    "item_type": record.get("item_type", "unknown"),
                }
            records.append(
                {
                    "source_line": line_number,
                    "record": record,
                    "prediction": predictions.get(line_number),
                    "impact_dimension": str(parsed.get("impact_dimension", "unknown")),
                }
            )

    rng = random.Random(seed)
    calibration_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for dimension, group in sorted(_group_records(records, "impact_dimension").items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        calibration_count = round(len(shuffled) * calibration_fraction)
        calibration_count = min(max(calibration_count, 1), len(shuffled) - 1)
        calibration_records.extend(shuffled[:calibration_count])
        validation_records.extend(shuffled[calibration_count:])

    calibration_records.sort(key=lambda row: int(row["source_line"]))
    validation_records.sort(key=lambda row: int(row["source_line"]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_split_files(destination, "calibration", calibration_records)
    _write_split_files(destination, "validation", validation_records)
    by_dimension = {}
    for dimension in sorted({row["impact_dimension"] for row in records}):
        by_dimension[dimension] = {
            "calibration_count": sum(
                row["impact_dimension"] == dimension for row in calibration_records
            ),
            "validation_count": sum(
                row["impact_dimension"] == dimension for row in validation_records
            ),
        }
    manifest = {
        "source_sft_path": str(sft_path),
        "source_predictions_path": str(predictions_path),
        "output_dir": str(destination),
        "source_record_count": len(records),
        "calibration_count": len(calibration_records),
        "validation_count": len(validation_records),
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "by_dimension": by_dimension,
    }
    (destination / "calibration_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def sciimpact_model_error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong_rows = [row for row in rows if row["outcome"] == "wrong"]
    return {
        "summary": _error_group_summary(rows),
        "by_dimension": {
            group: _error_group_summary(group_rows)
            for group, group_rows in sorted(_group_rows(rows, "impact_dimension").items())
        },
        "by_item_type": {
            group: _error_group_summary(group_rows)
            for group, group_rows in sorted(_group_rows(rows, "item_type").items())
        },
        "top_failure_dimensions": _top_failure_dimensions(rows),
        "wrong_examples": wrong_rows[:20],
    }


def render_sciimpact_model_error_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# SciImpact Model Error Analysis",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in analysis["summary"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## By Impact Dimension",
            "",
            "| Dimension | Total | Wrong | Wrong Rate | Predicted A | Predicted B | Wrong Predicted A | Wrong Predicted B |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dimension, summary in sorted(analysis["by_dimension"].items()):
        predicted = summary["predicted_side_counts"]
        wrong_predicted = summary["wrong_predicted_side_counts"]
        lines.append(
            "| "
            f"{dimension} | "
            f"{summary['total_examples']} | "
            f"{summary['wrong_examples']} | "
            f"{summary['wrong_rate']:.3f} | "
            f"{predicted.get('A', 0)} | "
            f"{predicted.get('B', 0)} | "
            f"{wrong_predicted.get('A', 0)} | "
            f"{wrong_predicted.get('B', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Side counts help detect A/B selection bias. Wrong-side counts show which side the model chose when it failed.",
            "",
        ]
    )
    return "\n".join(lines)


def render_sciimpact_flip_consistency_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SciImpact A/B Flip Consistency",
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
            "## Selection Summary",
            "",
            "| Source | Count | Accuracy |",
            "| --- | ---: | ---: |",
        ]
    )
    for source, summary in sorted(report["selection_summary"].items()):
        lines.append(
            f"| {source} | {summary['count']} | {_format_value(summary['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## By Impact Dimension",
            "",
            "| Group | Total | Original Acc. | Flipped-Mapped Acc. | Selected Acc. | Consistency |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report["by_dimension"].items()):
        lines.append(_flip_consistency_group_row(group, summary))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The flipped-mapped result swaps Paper A/B in the prompt, then maps the answer back to the original orientation. "
            "Disagreements indicate position sensitivity; selected accuracy uses the higher-confidence side when confidence is available.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _write_prediction(
    output_handle: Any,
    source_line: int,
    prediction: dict[str, Any],
    model: str,
) -> None:
    record = {
        "source_line": source_line,
        "prediction": _normalized_prediction_content(prediction["content"]),
        "model": model,
    }
    record.update(_prediction_confidence_info(prediction))
    output_handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )
    output_handle.flush()


def _normalized_prediction_content(content: str) -> str:
    stripped = str(content).strip()
    if stripped in {"A", "B"}:
        return stripped
    return stripped


def _prediction_text(prediction: dict[str, Any] | None) -> str | None:
    if prediction is None:
        return None
    text = prediction.get("prediction") or prediction.get("answer") or prediction.get("text")
    return str(text) if text is not None else None


def _prediction_field(prediction: dict[str, Any] | None, field: str) -> Any:
    if prediction is None:
        return None
    return prediction.get(field)


def _prediction_confidence(prediction: dict[str, Any] | None) -> float | None:
    value = _prediction_field(prediction, "confidence")
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _prediction_confidence_info(prediction: dict[str, Any]) -> dict[str, Any]:
    margin = _choice_logprob_margin(prediction.get("logprobs"))
    if margin is None:
        return {}
    return {
        "confidence": round(_logistic(margin), 6),
        "confidence_method": "choice_logprob_margin",
        "choice_logprob_margin": round(margin, 6),
    }


def _choice_logprob_margin(logprobs: Any) -> float | None:
    best_a = _best_choice_logprob(logprobs, "A")
    best_b = _best_choice_logprob(logprobs, "B")
    if best_a is None or best_b is None:
        return None
    return abs(best_a - best_b)


def _best_choice_logprob(logprobs: Any, label: str) -> float | None:
    candidates: list[float] = []
    for item in _logprob_content_items(logprobs):
        token = str(item.get("token", "")).strip()
        if token == label and item.get("logprob") is not None:
            candidates.append(float(item["logprob"]))
        for top_item in item.get("top_logprobs") or []:
            top_token = str(top_item.get("token", "")).strip()
            if top_token == label and top_item.get("logprob") is not None:
                candidates.append(float(top_item["logprob"]))
    if not candidates:
        return None
    return max(candidates)


def _logprob_content_items(logprobs: Any) -> list[dict[str, Any]]:
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _confidence_bucket(confidence: float | None) -> str:
    if confidence is None:
        return "unscored"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_split_files(
    destination: Path,
    split_name: str,
    rows: list[dict[str, Any]],
) -> None:
    with (destination / f"{split_name}_sft.jsonl").open("w", encoding="utf-8") as sft_handle, (
        destination / f"{split_name}_predictions.jsonl"
    ).open("w", encoding="utf-8") as prediction_handle:
        for new_source_line, row in enumerate(rows, start=1):
            sft_handle.write(json.dumps(row["record"], ensure_ascii=False) + "\n")
            prediction = row.get("prediction")
            if prediction is None:
                continue
            rewritten_prediction = dict(prediction)
            rewritten_prediction["source_line"] = new_source_line
            prediction_handle.write(
                json.dumps(rewritten_prediction, ensure_ascii=False) + "\n"
            )


def _flip_sciimpact_sft_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    if not isinstance(record.get("conversations"), list):
        raise ValueError(f"SciImpact SFT record on line {line_number} is missing conversations")
    conversations = record["conversations"]
    human_index = _conversation_index(conversations, "human")
    gpt_index = _conversation_index(conversations, "gpt")
    if human_index is None or gpt_index is None:
        raise ValueError(f"SciImpact SFT record on line {line_number} is missing human/gpt text")
    human_text = conversations[human_index].get("value")
    answer = conversations[gpt_index].get("value")
    if not isinstance(human_text, str) or not isinstance(answer, str):
        raise ValueError(f"SciImpact SFT record on line {line_number} has non-string text")
    flipped_prompt = _flip_sft_human_prompt(human_text, line_number)
    original_expected_side = _expected_side_for_record(record, line_number)
    flipped = json.loads(json.dumps(record, ensure_ascii=False))
    flipped["conversations"][human_index]["value"] = flipped_prompt
    flipped["conversations"][gpt_index]["value"] = _flip_sft_answer(answer, line_number)
    flipped["flip_metadata"] = {
        "mode": "swap_ab",
        "source_line": line_number,
        "original_expected_side": original_expected_side.upper(),
        "flipped_expected_side": _opposite_side(original_expected_side).upper(),
    }
    return flipped


def _conversation_index(conversations: list[Any], role: str) -> int | None:
    for index, message in enumerate(conversations):
        if isinstance(message, dict) and message.get("from") == role:
            return index
    return None


def _flip_sft_human_prompt(human_prompt: str, line_number: int) -> str:
    for _, marker_a, marker_b in _SFT_ITEM_MARKERS:
        if marker_a not in human_prompt or marker_b not in human_prompt:
            continue
        prefix, raw_a_with_rest = human_prompt.split(marker_a, 1)
        raw_a, raw_b_with_question = raw_a_with_rest.split(marker_b, 1)
        raw_b, question_suffix = _split_sft_question_suffix(raw_b_with_question)
        return (
            f"{prefix}{marker_a} {raw_b.strip()} "
            f"{marker_b} {raw_a.strip()}{question_suffix}"
        )
    raise ValueError(f"SciImpact SFT record on line {line_number} has no supported A/B markers")


def _split_sft_question_suffix(raw_b_with_question: str) -> tuple[str, str]:
    earliest_match: re.Match[str] | None = None
    for pattern in _SFT_QUESTION_SUFFIX_PATTERNS:
        match = re.search(pattern, raw_b_with_question, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            continue
        if earliest_match is None or match.start() < earliest_match.start():
            earliest_match = match
    if earliest_match is None:
        return raw_b_with_question, ""
    return (
        raw_b_with_question[: earliest_match.start()],
        raw_b_with_question[earliest_match.start() :],
    )


def _group_records(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key) or "unknown")].append(record)
    return dict(grouped)


def _with_final_answer_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = (
        "Think through the comparison, then end with exactly one final line. "
        "Final output format: Answer: A or Answer: B"
    )
    if not messages:
        return [{"role": "user", "content": instruction}]
    updated = [dict(message) for message in messages]
    updated[-1]["content"] = f"{updated[-1]['content'].rstrip()}\n\n{instruction}"
    return updated


def _with_strict_answer_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = "Output exactly: Answer: A or Answer: B. Do not explain."
    if not messages:
        return [{"role": "user", "content": instruction}]
    updated = [dict(message) for message in messages]
    updated[-1]["content"] = f"{updated[-1]['content'].rstrip()}\n\n{instruction}"
    return updated


def _with_choice_only_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = "Output only A or B. Do not explain."
    if not messages:
        return [{"role": "user", "content": instruction}]
    updated = [dict(message) for message in messages]
    updated[-1]["content"] = f"{updated[-1]['content'].rstrip()}\n\n{instruction}"
    return updated


def _sciimpact_prompt_messages(record: dict[str, Any], line_number: int) -> list[dict[str, str]]:
    messages = _chat_messages(record.get("messages"))
    if messages:
        return messages
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


def _chat_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
            if role == "assistant":
                continue
            converted.append({"role": role, "content": content})
    return converted


def _expected_side_for_record(record: dict[str, Any], line_number: int) -> str:
    target_side = _side_from_prediction(_optional_prediction_text(record.get("target")))
    if target_side is not None:
        return target_side
    parsed = _parse_pair(record, line_number)
    return _side_from_winner(parsed, line_number)


def _parse_pair_metadata(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    try:
        return _parse_pair(record, line_number)
    except ValueError:
        expected_side = _expected_side_for_record_from_target_only(record, line_number)
        return {
            "winner": {"metadata": {"sciimpact_source_side": expected_side}},
            "loser": {},
            "impact_dimension": record.get("impact_dimension", "unknown"),
            "item_type": record.get("item_type", "unknown"),
        }


def _expected_side_for_record_from_target_only(
    record: dict[str, Any],
    line_number: int,
) -> str:
    target_side = _side_from_prediction(_optional_prediction_text(record.get("target")))
    if target_side is None:
        raise ValueError(f"SciImpact record on line {line_number} has no parseable target")
    return target_side


def _optional_prediction_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _complete_sciimpact_prompt(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    allowed_token_ids: list[int] | None,
    structured_choice: list[str] | None,
    request_logprobs: bool = False,
    client: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if allowed_token_ids is not None:
        payload["allowed_token_ids"] = allowed_token_ids
    if structured_choice is not None:
        payload["structured_outputs"] = {"choice": structured_choice}
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 2
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_payload = response.json()
    choice = response_payload["choices"][0]
    return {
        "content": str(choice["message"]["content"]).strip(),
        "logprobs": choice.get("logprobs"),
    }


def _allowed_answer_token_ids(answer_format: str) -> list[int] | None:
    if answer_format == "choice_logprob":
        # Qwen-family tokenizer ids for " A", " B".
        return [362, 425]
    if answer_format != "constrained_answer":
        return None
    # Qwen-family tokenizer ids for "Answer", ":", " A", " B".
    return [16141, 25, 362, 425]


def _structured_answer_choice(answer_format: str) -> list[str] | None:
    if answer_format != "structured_choice":
        return None
    return ["Answer: A", "Answer: B"]


def render_sciimpact_model_eval_markdown(report: dict[str, Any]) -> str:
    scorecard = report.get("credibility_scorecard", {})
    significance = report.get("statistical_significance", {})
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
            f"| significant_vs_random | {_yes_no(significance.get('significant_vs_random'))} |",
            f"| significance_p_value | {_format_value(significance.get('p_value'))} |",
            f"| credibility_status | {scorecard.get('status', '')} |",
            f"| credibility_grade | {scorecard.get('grade', '')} |",
            "",
            "## Credibility Scorecard",
            "",
            "| Check | Status | Value | Reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', '')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    confidence_summary = report.get("confidence_summary", {})
    if confidence_summary:
        lines.extend(
            [
                "",
                "## Confidence Summary",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in confidence_summary.items():
            lines.append(f"| {key} | {_format_value(value)} |")
    calibrated_summary = report.get("calibrated_confidence_summary")
    if calibrated_summary:
        lines.extend(
            [
                "",
                "## Calibrated Confidence Summary",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in calibrated_summary.items():
            lines.append(f"| {key} | {_format_value(value)} |")
    calibrated_calibration = report.get("calibrated_calibration")
    if calibrated_calibration:
        lines.extend(
            [
                "",
                "## Calibrated Calibration",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Expected Calibration Error | {_format_value(calibrated_calibration.get('ece'))} |",
                f"| Maximum Calibration Error | {_format_value(calibrated_calibration.get('mce'))} |",
                f"| Brier Score | {_format_value(calibrated_calibration.get('brier_score'))} |",
                f"| Overconfidence Gap | {_format_value(calibrated_calibration.get('overconfidence_gap'))} |",
            ]
        )
    calibration = report.get("calibration", {})
    if calibration:
        lines.extend(
            [
                "",
                "## Calibration",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Expected Calibration Error | {_format_value(calibration.get('ece'))} |",
                f"| Maximum Calibration Error | {_format_value(calibration.get('mce'))} |",
                f"| Brier Score | {_format_value(calibration.get('brier_score'))} |",
                f"| Overconfidence Gap | {_format_value(calibration.get('overconfidence_gap'))} |",
                "",
                "| Bin | Count | Accuracy | Mean Confidence | Gap |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in calibration.get("bins", []):
            lines.append(
                f"| {row['bin']} | {row['count']} | "
                f"{row['accuracy']:.3f} | {row['mean_confidence']:.3f} | "
                f"{row['calibration_gap']:.3f} |"
            )
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
    if report.get("by_confidence_bucket"):
        lines.extend(
            [
                "",
                "## By Confidence Bucket",
                "",
                "| Group | Total | Covered | Accuracy | Coverage | Mean Confidence |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, summary in sorted(report["by_confidence_bucket"].items()):
            lines.append(_confidence_group_row(group, summary))
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


def render_sciimpact_model_credibility_scorecard_markdown(
    scorecard: dict[str, Any],
) -> str:
    lines = [
        "# SciImpact Model Credibility Scorecard",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {scorecard.get('status', '')} |",
        f"| grade | {scorecard.get('grade', '')} |",
        f"| can_claim_high_model_credibility | {_yes_no(scorecard.get('can_claim_high_model_credibility'))} |",
        f"| reason | {scorecard.get('reason', '')} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', '')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    blockers = scorecard.get("blocking_reasons", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = scorecard.get("warnings", [])
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def render_sciimpact_model_credibility_summary_markdown(
    report: dict[str, Any],
) -> str:
    summary = report.get("summary", {})
    scorecard = report.get("credibility_scorecard", {})
    lines = [
        "# SciImpact Model Credibility Summary",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {scorecard.get('status', 'unknown')} |",
        f"| grade | {scorecard.get('grade', 'unknown')} |",
        f"| can_claim_high_model_credibility | {_yes_no(scorecard.get('can_claim_high_model_credibility'))} |",
        f"| reason | {scorecard.get('reason', '')} |",
        f"| evaluation_count | {summary.get('evaluation_count', 0)} |",
        f"| pooled_accuracy | {_format_value(summary.get('pooled_accuracy'))} |",
        f"| mean_accuracy | {_format_value(summary.get('mean_accuracy'))} |",
        f"| min_accuracy | {_format_value(summary.get('min_accuracy'))} |",
        f"| accuracy_range | {_format_value(summary.get('accuracy_range'))} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in scorecard.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', 'unknown')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Evaluations",
            "",
            "| Evaluation | Accuracy | Coverage | Scorecard | Source |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("evaluations", []):
        lines.append(
            f"| {row.get('label', '')} | {_format_value(row.get('accuracy'))} | "
            f"{_format_value(row.get('coverage_rate'))} | "
            f"{row.get('scorecard_status', 'unknown')} | `{row.get('path', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Weak Slices",
            "",
            "| Slice | Pooled Accuracy | Min Seed Accuracy | Covered | Evaluations |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    weak_slices = report.get("weak_slices", [])
    if weak_slices:
        for row in weak_slices:
            lines.append(
                f"| {row.get('slice', '')} | {_format_value(row.get('pooled_accuracy'))} | "
                f"{_format_value(row.get('min_seed_accuracy'))} | "
                f"{row.get('covered_examples', 0)} | {row.get('evaluation_count', 0)} |"
            )
    else:
        lines.append("| None |  |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def render_sciimpact_model_candidate_gate_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison", {})
    lines = [
        "# SciImpact Model Candidate Gate",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| decision | {report.get('decision', 'unknown')} |",
        f"| reason | {report.get('reason', '')} |",
        f"| baseline_status | {comparison.get('baseline_status', 'unknown')} |",
        f"| candidate_status | {comparison.get('candidate_status', 'unknown')} |",
        f"| baseline_summary | `{report.get('baseline_summary_path', '')}` |",
        f"| candidate_summary | `{report.get('candidate_summary_path', '')}` |",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        _candidate_gate_metric_row(
            "pooled_accuracy",
            comparison.get("baseline_pooled_accuracy"),
            comparison.get("candidate_pooled_accuracy"),
            comparison.get("pooled_accuracy_delta"),
        ),
        _candidate_gate_metric_row(
            "min_seed_accuracy",
            comparison.get("baseline_min_accuracy"),
            comparison.get("candidate_min_accuracy"),
            comparison.get("min_seed_accuracy_delta"),
        ),
        _candidate_gate_metric_row(
            "accuracy_range",
            comparison.get("baseline_accuracy_range"),
            comparison.get("candidate_accuracy_range"),
            comparison.get("accuracy_range_delta"),
        ),
        _candidate_gate_metric_row(
            "weak_slice_count",
            comparison.get("baseline_weak_slice_count"),
            comparison.get("candidate_weak_slice_count"),
            comparison.get("weak_slice_count_delta"),
        ),
        _candidate_gate_metric_row(
            "worst_slice_min_accuracy",
            comparison.get("baseline_worst_slice_min_accuracy"),
            comparison.get("candidate_worst_slice_min_accuracy"),
            comparison.get("worst_slice_min_accuracy_delta"),
        ),
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in report.get("checks", {}).items():
        lines.append(
            f"| {name} | {check.get('status', 'unknown')} | "
            f"{_format_value(check.get('value'))} | {check.get('reason', '')} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    blockers = report.get("blocking_reasons", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def render_sciimpact_model_weak_slice_error_pack_markdown(
    report: dict[str, Any],
) -> str:
    manifest = report.get("manifest", {})
    lines = [
        "# SciImpact Model Weak-Slice Error Pack",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| recommended_use | {manifest.get('recommended_use', '')} |",
        f"| evaluation_count | {manifest.get('evaluation_count', 0)} |",
        f"| weak_slice_threshold | {_format_value(manifest.get('weak_slice_threshold'))} |",
        f"| weak_slice_count | {manifest.get('weak_slice_count', 0)} |",
        f"| error_count | {manifest.get('error_count', 0)} |",
        f"| high_confidence_error_count | {manifest.get('high_confidence_error_count', 0)} |",
        f"| training_record_count | {manifest.get('training_record_count', 0)} |",
        f"| skipped_training_record_count | {manifest.get('skipped_training_record_count', 0)} |",
    ]
    if manifest.get("leakage_warning"):
        lines.extend(["", "## Leakage Warning", "", manifest["leakage_warning"]])
    lines.extend(
        [
            "",
            "## Errors By Weak Slice",
            "",
            "| Slice | Error Rows |",
            "| --- | ---: |",
        ]
    )
    by_slice = manifest.get("by_weak_slice", {})
    if by_slice:
        for name, count in by_slice.items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| None | 0 |")
    lines.extend(
        [
            "",
            "## Error Preview",
            "",
            "| Evaluation | Source Line | Slices | Expected | Predicted | Confidence |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    preview = report.get("errors_preview", [])
    if preview:
        for row in preview:
            lines.append(
                f"| {row.get('evaluation_label', '')} | {row.get('source_line', '')} | "
                f"{', '.join(row.get('matched_weak_slices', []))} | "
                f"{row.get('expected_side', '')} | {row.get('predicted_side', '')} | "
                f"{_format_value(row.get('confidence'))} |"
            )
    else:
        lines.append("| None |  |  |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def _load_predictions(path: str | Path) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
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
            predictions[int(source_line)] = {
                **record,
                "prediction": str(prediction),
            }
    return predictions


def _side_from_winner(parsed: dict[str, Any], line_number: int) -> str:
    source_side = parsed["winner"].get("metadata", {}).get("sciimpact_source_side")
    if source_side not in {"a", "b"}:
        raise ValueError(f"SciImpact record on line {line_number} has no winner side")
    return source_side


def _side_from_prediction(prediction: str | None) -> str | None:
    if prediction is None:
        return None
    bare_side = re.fullmatch(r"\s*(?:Answer\s*:\s*)?([AB])\s*\.?\s*", prediction, re.I)
    if bare_side:
        return bare_side.group(1).lower()
    final_answer_matches = list(
        re.finditer(r"\bAnswer\s*:\s*([AB])\b", prediction, re.IGNORECASE)
    )
    if final_answer_matches:
        return final_answer_matches[-1].group(1).lower()
    matches = list(
        re.finditer(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+([AB])\b", prediction, re.I)
    )
    if not matches:
        return None
    return matches[-1].group(1).lower()


def _opposite_side(side: str | None) -> str | None:
    if side is None:
        return None
    lowered = side.lower()
    if lowered == "a":
        return "b"
    if lowered == "b":
        return "a"
    return None


def _side_outcome(side: str | None, expected_side: str) -> str:
    if side is None:
        return "uncovered"
    return "correct" if side == expected_side else "wrong"


def _select_flip_consistency_side(
    original_side: str | None,
    flipped_mapped_side: str | None,
    original_confidence: float | None,
    flipped_confidence: float | None,
    *,
    confidence_margin: float = 0.0,
) -> tuple[str | None, str]:
    if original_side is not None and original_side == flipped_mapped_side:
        return original_side, "consistent"
    if original_side is None and flipped_mapped_side is None:
        return None, "uncovered"
    if original_side is None:
        return flipped_mapped_side, "flipped_mapped_only"
    if flipped_mapped_side is None:
        return original_side, "original_only"
    if original_confidence is not None and flipped_confidence is not None:
        if flipped_confidence > original_confidence + confidence_margin:
            return flipped_mapped_side, "flipped_mapped_confidence"
        if original_confidence > flipped_confidence + confidence_margin:
            return original_side, "original_confidence"
        return original_side, "original_margin_default"
    if flipped_confidence is not None and original_confidence is None:
        return flipped_mapped_side, "flipped_mapped_confidence"
    return original_side, "original_default"


def _flip_consistency_groups(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    return {
        group: _flip_consistency_summary(group_rows)
        for group, group_rows in _group_rows(rows, key).items()
    }


def _flip_consistency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    original_covered = [row for row in rows if row["original_outcome"] != "uncovered"]
    flipped_covered = [row for row in rows if row["flipped_mapped_outcome"] != "uncovered"]
    selected_covered = [row for row in rows if row["selected_outcome"] != "uncovered"]
    comparable = [
        row
        for row in rows
        if row["original_side"] is not None and row["flipped_mapped_side"] is not None
    ]
    both_correct = [
        row
        for row in rows
        if row["original_outcome"] == "correct"
        and row["flipped_mapped_outcome"] == "correct"
    ]
    original_only = [
        row
        for row in rows
        if row["original_outcome"] == "correct"
        and row["flipped_mapped_outcome"] == "wrong"
    ]
    flipped_only = [
        row
        for row in rows
        if row["original_outcome"] == "wrong"
        and row["flipped_mapped_outcome"] == "correct"
    ]
    both_wrong = [
        row
        for row in rows
        if row["original_outcome"] == "wrong"
        and row["flipped_mapped_outcome"] == "wrong"
    ]
    return {
        "total_examples": total,
        "original_covered_examples": len(original_covered),
        "flipped_mapped_covered_examples": len(flipped_covered),
        "selected_covered_examples": len(selected_covered),
        "original_accuracy": _outcome_accuracy(original_covered, "original_outcome"),
        "flipped_mapped_accuracy": _outcome_accuracy(
            flipped_covered,
            "flipped_mapped_outcome",
        ),
        "selected_accuracy": _outcome_accuracy(selected_covered, "selected_outcome"),
        "consistency_rate": _safe_divide(
            sum(row["consistent"] for row in comparable),
            len(comparable),
        ),
        "comparable_examples": len(comparable),
        "both_correct_examples": len(both_correct),
        "original_only_correct_examples": len(original_only),
        "flipped_only_correct_examples": len(flipped_only),
        "both_wrong_examples": len(both_wrong),
    }


def _selection_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for source, source_rows in sorted(_group_rows(rows, "selected_source").items()):
        covered = [row for row in source_rows if row["selected_outcome"] != "uncovered"]
        summary[source] = {
            "count": len(source_rows),
            "accuracy": _outcome_accuracy(covered, "selected_outcome"),
        }
    return summary


def _flip_selected_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        selected_side = row.get("selected_side")
        prediction = f"Answer: {selected_side}" if selected_side in {"A", "B"} else None
        selected_source = row.get("selected_source")
        confidence = None
        if selected_source == "original_confidence":
            confidence = row.get("original_confidence")
        elif selected_source == "flipped_mapped_confidence":
            confidence = row.get("flipped_confidence")
        elif selected_source == "consistent":
            confidence = _max_optional(
                row.get("original_confidence"),
                row.get("flipped_confidence"),
            )
        else:
            confidence = _max_optional(
                row.get("original_confidence"),
                row.get("flipped_confidence"),
            )
        record = {
            "source_line": row["source_line"],
            "prediction": prediction,
            "confidence": confidence,
            "confidence_method": "flip_consistency",
            "selected_source": selected_source,
            "original_prediction": row.get("original_prediction"),
            "flipped_prediction": row.get("flipped_prediction"),
            "flipped_mapped_side": row.get("flipped_mapped_side"),
        }
        predictions.append(record)
    return predictions


def _max_optional(*values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return max(numbers)


def _outcome_accuracy(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(row[key] == "correct" for row in rows) / len(rows)


def _summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {group: _summarize(group_rows) for group, group_rows in _group_rows(rows, key).items()}


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
        "mean_confidence": _mean(
            row["confidence"] for row in rows if row.get("confidence") is not None
        ),
    }


def _model_statistical_significance(summary: dict[str, Any]) -> dict[str, Any]:
    trials = int(summary.get("covered_examples", 0) or 0)
    successes = int(summary.get("correct_examples", 0) or 0)
    observed = _safe_divide(successes, trials)
    ci95 = _wilson_ci(successes, trials)
    p_value = _binomial_greater_equal_p_value(successes, trials)
    significant = trials >= 20 and observed > 0.5 and ci95[0] > 0.5 and p_value <= 0.05
    if significant:
        status = "statistically_supported"
        reason = "SciImpact A/B accuracy is significantly above a random 0.5 baseline."
    elif trials < 20:
        status = "insufficient_validation_data"
        reason = "Fewer than 20 covered SciImpact examples; model significance is not claimed."
    else:
        status = "not_statistically_supported"
        reason = "SciImpact A/B accuracy is not significantly above random."
    return {
        "status": status,
        "reason": reason,
        "tested_metric": "accuracy",
        "successes": successes,
        "trials": trials,
        "observed": observed,
        "wilson_ci95": ci95,
        "p_value": p_value,
        "significant_vs_random": significant,
    }


def _model_credibility_scorecard(
    summary: dict[str, Any],
    *,
    significance: dict[str, Any],
    by_dimension: dict[str, dict[str, Any]],
    by_item_type: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "coverage": _model_coverage_check(summary),
        "statistical_support": _model_statistical_support_check(significance),
        "benchmark_accuracy": _model_accuracy_check(summary),
        "slice_stability": _model_slice_stability_check(by_dimension, by_item_type),
    }
    blocking = [
        check["reason"]
        for name, check in checks.items()
        if check["status"] == "fail"
        and name in {"coverage", "statistical_support", "benchmark_accuracy"}
    ]
    warnings = [
        check["reason"] for check in checks.values() if check["status"] == "warn"
    ]
    pass_count = sum(1 for check in checks.values() if check["status"] == "pass")
    fail_count = sum(1 for check in checks.values() if check["status"] == "fail")
    if int(summary.get("covered_examples", 0) or 0) < 20:
        status = "insufficient_validation_data"
        grade = "D"
        reason = "Fewer than 20 covered examples; use this as a smoke check only."
    elif blocking:
        status = "not_credible"
        grade = "D"
        reason = blocking[0]
    elif pass_count == len(checks):
        status = "high_credibility"
        grade = "A"
        reason = "SciImpact model evaluation passes coverage, significance, accuracy, and slice checks."
    elif fail_count == 0 and checks["statistical_support"]["status"] == "pass":
        status = "moderate_credibility"
        grade = "B"
        reason = "SciImpact model evaluation is supported, with warnings to disclose."
    else:
        status = "low_credibility"
        grade = "C"
        reason = "SciImpact model evaluation has partial signal but does not meet high-credibility thresholds."
    return {
        "status": status,
        "grade": grade,
        "can_claim_high_model_credibility": status == "high_credibility",
        "reason": reason,
        "checks": checks,
        "blocking_reasons": blocking,
        "warnings": warnings,
    }


def _load_sciimpact_model_eval_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "sciimpact_model_eval.json"
    if not source.exists():
        raise FileNotFoundError(f"SciImpact model evaluation not found: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if "summary" not in report:
        raise ValueError(f"SciImpact model evaluation lacks summary: {source}")
    return {"path": source, "report": report}


def _load_sciimpact_model_credibility_summary(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "sciimpact_model_credibility_summary.json"
    if not source.exists():
        raise FileNotFoundError(f"SciImpact model credibility summary not found: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if "summary" not in report or "credibility_scorecard" not in report:
        raise ValueError(f"SciImpact model credibility summary is invalid: {source}")
    return {"path": source, "report": report}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_sft_records_by_line(path: str | Path) -> dict[int, dict[str, Any]]:
    records = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                records[line_number] = json.loads(stripped)
    return records


def _annotate_weak_slice_training_record(
    record: dict[str, Any],
    error_record: dict[str, Any],
) -> None:
    record["mining_source"] = "sciimpact_model_weak_slice_error_pack"
    record["matched_weak_slices"] = list(error_record.get("matched_weak_slices", []))
    record["model_eval_label"] = error_record.get("evaluation_label")
    record["model_eval_path"] = error_record.get("evaluation_path")
    record["model_outcome"] = error_record.get("outcome")
    record["model_expected_side"] = error_record.get("expected_side")
    record["model_predicted_side"] = error_record.get("predicted_side")
    record["model_confidence"] = error_record.get("confidence")


def _matched_weak_slices(
    row: dict[str, Any],
    weak_slices: dict[str, dict[str, Any]],
) -> list[str]:
    candidates = [
        f"dimension:{row.get('impact_dimension')}",
        f"item_type:{row.get('item_type')}",
    ]
    return [name for name in candidates if name in weak_slices]


def _model_summary_weak_slice_count(report: dict[str, Any]) -> int:
    return sum(
        1
        for row in report.get("all_slices", [])
        if float(row.get("pooled_accuracy", 0.0) or 0.0) < 0.60
        or float(row.get("min_seed_accuracy", 0.0) or 0.0) < 0.60
    )


def _model_summary_worst_slice_accuracy(report: dict[str, Any]) -> float:
    eligible = [
        float(row.get("min_seed_accuracy", 0.0) or 0.0)
        for row in report.get("all_slices", [])
        if int(row.get("covered_examples", 0) or 0) >= 10
    ]
    if not eligible:
        return 0.0
    return round(min(eligible), 6)


def _model_candidate_gate_checks(
    comparison: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    baseline_status: str,
    candidate_status: str,
) -> dict[str, dict[str, Any]]:
    status_delta = _model_credibility_status_rank(candidate_status) - _model_credibility_status_rank(
        baseline_status
    )
    return {
        "candidate_evaluation_count": _model_candidate_min_evaluations_check(
            comparison["candidate_evaluation_count"],
            int(thresholds["min_evaluations"]),
        ),
        "pooled_accuracy_delta": _model_candidate_min_delta_check(
            comparison["pooled_accuracy_delta"],
            float(thresholds["min_pooled_accuracy_delta"]),
            "Pooled accuracy must improve enough to justify promotion.",
        ),
        "min_seed_accuracy_delta": _model_candidate_max_regression_check(
            comparison["min_seed_accuracy_delta"],
            float(thresholds["max_min_seed_accuracy_regression"]),
            "Worst-seed accuracy must not regress.",
        ),
        "seed_stability_delta": _model_candidate_max_increase_check(
            comparison["accuracy_range_delta"],
            float(thresholds["max_accuracy_range_increase"]),
            "Seed-to-seed accuracy range must not become materially less stable.",
        ),
        "weak_slice_count_delta": _model_candidate_max_count_increase_check(
            comparison["weak_slice_count_delta"],
            int(thresholds["max_weak_slice_count_increase"]),
            "Candidate must not add weak dimension or item-type slices.",
        ),
        "worst_slice_min_accuracy_delta": _model_candidate_max_regression_check(
            comparison["worst_slice_min_accuracy_delta"],
            float(thresholds["max_worst_slice_regression"]),
            "Worst eligible slice accuracy must not materially regress.",
        ),
        "credibility_status_delta": _model_candidate_status_check(
            status_delta,
            candidate_status,
        ),
    }


def _model_candidate_min_evaluations_check(
    value: int,
    minimum: int,
) -> dict[str, Any]:
    if value >= minimum:
        return _model_check("pass", value, f"Candidate has at least {minimum} evaluations.")
    return _model_check("fail", value, f"Candidate needs at least {minimum} evaluations.")


def _model_candidate_min_delta_check(
    value: float,
    minimum: float,
    reason: str,
) -> dict[str, Any]:
    if value >= minimum:
        return _model_check("pass", value, reason)
    return _model_check("fail", value, reason)


def _model_candidate_max_regression_check(
    value: float,
    maximum_regression: float,
    reason: str,
) -> dict[str, Any]:
    if value >= -maximum_regression:
        return _model_check("pass", value, reason)
    return _model_check("fail", value, reason)


def _model_candidate_max_increase_check(
    value: float,
    maximum: float,
    reason: str,
) -> dict[str, Any]:
    if value <= maximum:
        return _model_check("pass", value, reason)
    return _model_check("fail", value, reason)


def _model_candidate_max_count_increase_check(
    value: int,
    maximum: int,
    reason: str,
) -> dict[str, Any]:
    if value <= maximum:
        return _model_check("pass", value, reason)
    return _model_check("fail", value, reason)


def _model_candidate_status_check(
    status_delta: int,
    candidate_status: str,
) -> dict[str, Any]:
    blocked_statuses = {"insufficient_validation_data", "not_credible", "low_credibility"}
    if candidate_status in blocked_statuses:
        return _model_check(
            "fail",
            status_delta,
            "Candidate credibility status is not strong enough for promotion.",
        )
    if status_delta >= 0:
        return _model_check(
            "pass",
            status_delta,
            "Candidate credibility status is not below the baseline.",
        )
    return _model_check(
        "fail",
        status_delta,
        "Candidate credibility status is below the baseline.",
    )


def _model_credibility_status_rank(status: str) -> int:
    return {
        "unknown": 0,
        "insufficient_validation_data": 0,
        "not_credible": 1,
        "low_credibility": 2,
        "moderate_credibility": 3,
        "high_credibility": 4,
    }.get(status, 0)


def _candidate_gate_metric_row(
    name: str,
    baseline: Any,
    candidate: Any,
    delta: Any,
) -> str:
    return (
        f"| {name} | {_format_value(baseline)} | {_format_value(candidate)} | "
        f"{_format_value(delta)} |"
    )


def _model_eval_label(path: Path, index: int) -> str:
    generic_names = {"eval", "evaluation", "model_eval", "model_eval_eval"}
    for parent in path.parents:
        name = parent.name
        if name and name != "." and name not in generic_names:
            return name
    return f"evaluation_{index}"


def _model_eval_summary_slices(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for report in reports:
        for group_name, prefix in (
            ("by_dimension", "dimension"),
            ("by_item_type", "item_type"),
        ):
            for name, summary in report["report"].get(group_name, {}).items():
                key = f"{prefix}:{name}"
                row = grouped.setdefault(
                    key,
                    {
                        "slice": key,
                        "covered_examples": 0,
                        "correct_examples": 0,
                        "total_examples": 0,
                        "seed_accuracies": [],
                        "evaluation_count": 0,
                    },
                )
                covered = int(summary.get("covered_examples", 0) or 0)
                correct = int(summary.get("correct_examples", 0) or 0)
                total = int(summary.get("total_examples", covered) or 0)
                row["covered_examples"] += covered
                row["correct_examples"] += correct
                row["total_examples"] += total
                if covered > 0:
                    row["seed_accuracies"].append(
                        round(float(summary.get("accuracy", 0.0) or 0.0), 6)
                    )
                    row["evaluation_count"] += 1
    rows = []
    for row in grouped.values():
        accuracies = row.pop("seed_accuracies")
        rows.append(
            {
                **row,
                "pooled_accuracy": round(
                    _safe_divide(
                        int(row["correct_examples"]),
                        int(row["covered_examples"]),
                    ),
                    6,
                ),
                "min_seed_accuracy": round(min(accuracies), 6) if accuracies else 0.0,
                "mean_seed_accuracy": _mean(accuracies),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["min_seed_accuracy"],
            row["pooled_accuracy"],
            row["slice"],
        ),
    )


def _model_credibility_summary_scorecard(
    summary: dict[str, Any],
    *,
    significance: dict[str, Any],
    slice_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "evaluation_count": _model_eval_count_check(summary),
        "coverage": _model_summary_coverage_check(summary),
        "statistical_support": _model_statistical_support_check(significance),
        "mean_accuracy": _model_summary_accuracy_check(summary),
        "seed_stability": _model_seed_stability_check(summary),
        "slice_stability": _model_summary_slice_stability_check(slice_rows),
    }
    blocking_names = {
        "evaluation_count",
        "coverage",
        "statistical_support",
        "mean_accuracy",
    }
    blocking = [
        check["reason"]
        for name, check in checks.items()
        if check["status"] == "fail" and name in blocking_names
    ]
    warnings = [
        check["reason"] for check in checks.values() if check["status"] == "warn"
    ]
    pass_count = sum(1 for check in checks.values() if check["status"] == "pass")
    fail_count = sum(1 for check in checks.values() if check["status"] == "fail")
    if int(summary.get("evaluation_count", 0) or 0) < 2:
        status = "insufficient_validation_data"
        grade = "D"
        reason = "Fewer than two SciImpact model evaluations were supplied."
    elif blocking:
        status = "not_credible"
        grade = "D"
        reason = blocking[0]
    elif pass_count == len(checks):
        status = "high_credibility"
        grade = "A"
        reason = "Cross-seed SciImpact model evidence passes all credibility checks."
    elif fail_count == 0 and checks["statistical_support"]["status"] == "pass":
        status = "moderate_credibility"
        grade = "B"
        reason = "Cross-seed SciImpact model evidence is supported, with warnings to disclose."
    else:
        status = "low_credibility"
        grade = "C"
        reason = "Cross-seed SciImpact model evidence has partial signal but remains unstable."
    return {
        "status": status,
        "grade": grade,
        "can_claim_high_model_credibility": status == "high_credibility",
        "reason": reason,
        "checks": checks,
        "blocking_reasons": blocking,
        "warnings": warnings,
    }


def _model_eval_count_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = int(summary.get("evaluation_count", 0) or 0)
    if value >= 3:
        return _model_check("pass", value, "At least three model evaluations are included.")
    if value >= 2:
        return _model_check("warn", value, "Only two model evaluations are included.")
    return _model_check("fail", value, "At least two model evaluations are required.")


def _model_summary_coverage_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("min_coverage_rate", 0.0) or 0.0)
    if value >= 0.90:
        return _model_check("pass", value, "Every included evaluation has at least 90% coverage.")
    if value >= 0.75:
        return _model_check("warn", value, "At least one evaluation has incomplete coverage.")
    return _model_check("fail", value, "At least one evaluation has too little coverage.")


def _model_summary_accuracy_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("mean_accuracy", 0.0) or 0.0)
    if value >= 0.65:
        return _model_check("pass", value, "Mean SciImpact A/B accuracy is strong.")
    if value >= 0.58:
        return _model_check(
            "warn",
            value,
            "Mean SciImpact A/B accuracy has signal but is not strong.",
        )
    return _model_check("fail", value, "Mean SciImpact A/B accuracy is too close to random.")


def _model_seed_stability_check(summary: dict[str, Any]) -> dict[str, Any]:
    min_accuracy = float(summary.get("min_accuracy", 0.0) or 0.0)
    accuracy_range = float(summary.get("accuracy_range", 0.0) or 0.0)
    if min_accuracy >= 0.65 and accuracy_range <= 0.07:
        return _model_check("pass", min_accuracy, "All seed accuracies are strong and stable.")
    if min_accuracy >= 0.58:
        return _model_check(
            "warn",
            min_accuracy,
            "At least one seed is weaker than the high-credibility threshold.",
        )
    return _model_check("fail", min_accuracy, "At least one seed is too close to random.")


def _model_summary_slice_stability_check(
    slice_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        row
        for row in slice_rows
        if int(row.get("covered_examples", 0) or 0) >= 10
    ]
    if not eligible:
        return _model_check(
            "warn",
            0.0,
            "No dimension or item-type slice has at least 10 covered examples.",
        )
    worst = min(float(row.get("min_seed_accuracy", 0.0) or 0.0) for row in eligible)
    if worst >= 0.60:
        return _model_check("pass", worst, "All eligible slices remain strong across seeds.")
    if worst >= 0.50:
        return _model_check("warn", worst, "At least one eligible slice is weak across seeds.")
    return _model_check("fail", worst, "At least one eligible slice is below random.")


def _model_coverage_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("coverage_rate", 0.0) or 0.0)
    if int(summary.get("total_examples", 0) or 0) < 20:
        return _model_check("fail", value, "Fewer than 20 SciImpact examples were supplied.")
    if value >= 0.9:
        return _model_check("pass", value, "At least 90% of SciImpact examples are covered.")
    if value >= 0.75:
        return _model_check("warn", value, "SciImpact coverage is usable but incomplete.")
    return _model_check("fail", value, "SciImpact coverage is too low for a credible claim.")


def _model_statistical_support_check(significance: dict[str, Any]) -> dict[str, Any]:
    value = float(significance.get("observed", 0.0) or 0.0)
    if significance.get("significant_vs_random"):
        return _model_check("pass", value, "Accuracy is significantly above random.")
    return _model_check(
        "fail",
        value,
        significance.get("reason", "SciImpact significance failed."),
    )


def _model_accuracy_check(summary: dict[str, Any]) -> dict[str, Any]:
    value = float(summary.get("accuracy", 0.0) or 0.0)
    if value >= 0.65:
        return _model_check("pass", value, "SciImpact A/B accuracy is strong.")
    if value >= 0.58:
        return _model_check("warn", value, "SciImpact A/B accuracy has signal but is not strong.")
    return _model_check("fail", value, "SciImpact A/B accuracy is too close to random.")


def _model_slice_stability_check(
    by_dimension: dict[str, dict[str, Any]],
    by_item_type: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        (f"dimension:{name}", summary)
        for name, summary in by_dimension.items()
        if int(summary.get("covered_examples", 0) or 0) >= 10
    ] + [
        (f"item_type:{name}", summary)
        for name, summary in by_item_type.items()
        if int(summary.get("covered_examples", 0) or 0) >= 10
    ]
    if not eligible:
        return _model_check(
            "warn",
            0.0,
            "No dimension or item-type slice has at least 10 covered examples.",
        )
    worst = min(float(summary.get("accuracy", 0.0) or 0.0) for _, summary in eligible)
    if worst >= 0.58:
        return _model_check("pass", worst, "Eligible SciImpact slices remain above random.")
    if worst >= 0.5:
        return _model_check("warn", worst, "At least one SciImpact slice is weak.")
    return _model_check("fail", worst, "At least one SciImpact slice is below random.")


def _model_check(status: str, value: float, reason: str) -> dict[str, Any]:
    return {"status": status, "value": round(float(value), 6), "reason": reason}


def _wilson_ci(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p_hat = successes / trials
    denominator = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _binomial_greater_equal_p_value(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    if successes <= trials / 2:
        return 1.0
    if trials <= 1000:
        probability = sum(math.comb(trials, k) for k in range(successes, trials + 1))
        return round(probability * (0.5**trials), 6)
    z = (successes - trials * 0.5 - 0.5) / math.sqrt(trials * 0.25)
    return round(0.5 * math.erfc(z / math.sqrt(2)), 6)


def _confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("confidence") is not None]
    correct_rows = [row for row in scored_rows if row["outcome"] == "correct"]
    wrong_rows = [row for row in scored_rows if row["outcome"] == "wrong"]
    high_rows = [row for row in scored_rows if row["confidence_bucket"] == "high"]
    low_rows = [row for row in scored_rows if row["confidence_bucket"] == "low"]
    return {
        "scored_examples": len(scored_rows),
        "mean_confidence": _mean(row["confidence"] for row in scored_rows),
        "correct_mean_confidence": _mean(row["confidence"] for row in correct_rows),
        "wrong_mean_confidence": _mean(row["confidence"] for row in wrong_rows),
        "high_confidence_accuracy": _accuracy(high_rows),
        "low_confidence_accuracy": _accuracy(low_rows),
    }


def _rows_with_confidence_alias(
    rows: list[dict[str, Any]],
    confidence_key: str,
) -> list[dict[str, Any]]:
    aliased = []
    for row in rows:
        copied = dict(row)
        copied["confidence"] = row.get(confidence_key)
        copied["confidence_bucket"] = _confidence_bucket(copied["confidence"])
        aliased.append(copied)
    return aliased


def _load_empirical_calibration(
    calibration_report_path: str | Path | None,
) -> dict[str, Any] | None:
    if calibration_report_path is None:
        return None
    source = Path(calibration_report_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    bins = []
    for row in report.get("calibration", {}).get("bins", []):
        if row.get("count", 0) <= 0:
            continue
        if row.get("accuracy") is None:
            continue
        bins.append(
            {
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "accuracy": float(row["accuracy"]),
                "count": int(row["count"]),
            }
        )
    return {
        "method": "empirical_bin_accuracy",
        "source_path": str(source),
        "bins": bins,
    }


def _calibrated_confidence(
    confidence: float | None,
    calibration_model: dict[str, Any] | None,
) -> float | None:
    if confidence is None or calibration_model is None:
        return None
    bins = calibration_model.get("bins", [])
    for index, row in enumerate(bins):
        lower = float(row["lower"])
        upper = float(row["upper"])
        is_last = index == len(bins) - 1
        if lower <= confidence < upper or (is_last and lower <= confidence <= upper):
            return round(float(row["accuracy"]), 6)
    return None


def _calibration_model_report(
    calibration_model: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if calibration_model is None:
        return None
    return {
        "method": calibration_model["method"],
        "source_path": calibration_model["source_path"],
        "bin_count": len(calibration_model.get("bins", [])),
    }


def _calibration_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [
        row
        for row in rows
        if row.get("confidence") is not None and row["outcome"] != "uncovered"
    ]
    if not scored_rows:
        return {
            "scored_examples": 0,
            "bin_count": 5,
            "ece": None,
            "mce": None,
            "brier_score": None,
            "overconfidence_gap": None,
            "bins": [],
        }
    bins = _calibration_bins(scored_rows, bin_count=5)
    total = len(scored_rows)
    ece = sum(row["count"] / total * abs(row["calibration_gap"]) for row in bins)
    mce = max(abs(row["calibration_gap"]) for row in bins)
    brier = sum(
        (float(row["confidence"]) - _correctness(row)) ** 2 for row in scored_rows
    ) / total
    mean_confidence = sum(float(row["confidence"]) for row in scored_rows) / total
    accuracy = sum(_correctness(row) for row in scored_rows) / total
    return {
        "scored_examples": total,
        "bin_count": 5,
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier_score": round(brier, 6),
        "overconfidence_gap": round(mean_confidence - accuracy, 6),
        "bins": bins,
    }


def _calibration_bins(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    bins = []
    width = 1.0 / bin_count
    for index in range(bin_count):
        lower = round(index * width, 10)
        upper = round((index + 1) * width, 10)
        if index == bin_count - 1:
            members = [
                row
                for row in rows
                if lower <= float(row["confidence"]) <= upper
            ]
        else:
            members = [
                row
                for row in rows
                if lower <= float(row["confidence"]) < upper
            ]
        if not members:
            continue
        accuracy = sum(_correctness(row) for row in members) / len(members)
        mean_confidence = sum(float(row["confidence"]) for row in members) / len(members)
        bins.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": len(members),
                "accuracy": round(accuracy, 6),
                "mean_confidence": round(mean_confidence, 6),
                "calibration_gap": round(mean_confidence - accuracy, 6),
            }
        )
    return bins


def _correctness(row: dict[str, Any]) -> float:
    return 1.0 if row["outcome"] == "correct" else 0.0


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return dict(grouped)


def _error_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wrong_rows = [row for row in rows if row["outcome"] == "wrong"]
    uncovered_rows = [row for row in rows if row["outcome"] == "uncovered"]
    return {
        "total_examples": total,
        "wrong_examples": len(wrong_rows),
        "uncovered_examples": len(uncovered_rows),
        "wrong_rate": _safe_divide(len(wrong_rows), total),
        "uncovered_rate": _safe_divide(len(uncovered_rows), total),
        "predicted_side_counts": _side_counts(rows, "predicted_side"),
        "expected_side_counts": _side_counts(rows, "expected_side"),
        "wrong_predicted_side_counts": _side_counts(wrong_rows, "predicted_side"),
        "wrong_expected_side_counts": _side_counts(wrong_rows, "expected_side"),
        "predictions_mentioning_both_sides": sum(
            _mentions_both_sides(row.get("prediction")) for row in rows
        ),
        "wrong_predictions_mentioning_both_sides": sum(
            _mentions_both_sides(row.get("prediction")) for row in wrong_rows
        ),
    }


def _top_failure_dimensions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for dimension, group_rows in _group_rows(rows, "impact_dimension").items():
        summary = _error_group_summary(group_rows)
        failures.append(
            {
                "impact_dimension": dimension,
                "wrong_examples": summary["wrong_examples"],
                "wrong_rate": summary["wrong_rate"],
                "total_examples": summary["total_examples"],
            }
        )
    return sorted(failures, key=lambda row: (-row["wrong_rate"], -row["wrong_examples"], row["impact_dimension"]))


def _side_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        side = row.get(key)
        if side is None:
            continue
        counts[str(side)] += 1
    return dict(sorted(counts.items()))


def _mentions_both_sides(prediction: Any) -> bool:
    if not isinstance(prediction, str):
        return False
    return bool(
        re.search(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+A\b", prediction, re.I)
        and re.search(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+B\b", prediction, re.I)
    )


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 6)


def _accuracy(rows: list[dict[str, Any]]) -> float:
    covered = [row for row in rows if row["outcome"] != "uncovered"]
    if not covered:
        return 0.0
    return sum(row["outcome"] == "correct" for row in covered) / len(covered)


def _group_row(group: str, summary: dict[str, Any]) -> str:
    return (
        f"| {group} | {summary['total_examples']} | {summary['covered_examples']} | "
        f"{summary['accuracy']:.3f} | {summary['coverage_rate']:.3f} |"
    )


def _confidence_group_row(group: str, summary: dict[str, Any]) -> str:
    confidence = summary.get("mean_confidence")
    confidence_text = "" if confidence is None else f"{float(confidence):.3f}"
    return (
        f"| {group} | {summary['total_examples']} | {summary['covered_examples']} | "
        f"{summary['accuracy']:.3f} | {summary['coverage_rate']:.3f} | "
        f"{confidence_text} |"
    )


def _flip_consistency_group_row(group: str, summary: dict[str, Any]) -> str:
    return (
        f"| {group} | {summary['total_examples']} | "
        f"{summary['original_accuracy']:.3f} | "
        f"{summary['flipped_mapped_accuracy']:.3f} | "
        f"{summary['selected_accuracy']:.3f} | "
        f"{summary['consistency_rate']:.3f} |"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
