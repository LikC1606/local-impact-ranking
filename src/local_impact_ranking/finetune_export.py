from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_impact_ranking.judge import JudgeResult, SelectedReference, build_judge_prompt
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import context_weight, CONTEXT_WEIGHT_METHOD_SUPPORTED


def make_format_sft_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    top_k: int,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    records = []
    for result in judge_results:
        if not _target_is_in_scope(result.target_paper_id, target_paper_ids):
            continue
        paper = papers_by_id.get(result.target_paper_id)
        if paper is None or result.error is not None or result.parse_error is not None:
            continue
        selections = [selection for selection in result.selected_references if selection.valid]
        if not selections:
            continue
        records.append(_sft_record("format_sft", paper, selections, top_k=top_k))
    return records


def make_proxy_positive_sft_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
    *,
    top_k: int,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    supported_ids = _supported_selection_ids(proxy_report)
    records = []
    for result in judge_results:
        if not _target_is_in_scope(result.target_paper_id, target_paper_ids):
            continue
        paper = papers_by_id.get(result.target_paper_id)
        if paper is None or result.error is not None or result.parse_error is not None:
            continue
        selections = [
            selection
            for selection in result.selected_references
            if selection.valid
            and (result.target_paper_id, selection.reference_id) in supported_ids
        ]
        if selections:
            records.append(_sft_record("proxy_positive_sft", paper, selections, top_k=top_k))
    return records


def make_pairwise_preference_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
    *,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    evidence_by_pair = {
        (row["target_paper_id"], row["reference_id"]): row
        for row in proxy_report.get("selected_reference_evidence", [])
    }
    records: list[dict[str, Any]] = []
    for result in judge_results:
        if not _target_is_in_scope(result.target_paper_id, target_paper_ids):
            continue
        paper = papers_by_id.get(result.target_paper_id)
        if paper is None:
            continue
        valid_selections = [selection for selection in result.selected_references if selection.valid]
        for chosen in valid_selections:
            chosen_evidence = evidence_by_pair.get((result.target_paper_id, chosen.reference_id))
            if not chosen_evidence or not _is_supported(chosen_evidence):
                continue
            for rejected in valid_selections:
                if rejected.reference_id == chosen.reference_id:
                    continue
                rejected_evidence = evidence_by_pair.get(
                    (result.target_paper_id, rejected.reference_id)
                )
                if rejected_evidence and _is_supported(rejected_evidence):
                    continue
                records.append(
                    {
                        "task": "pairwise_preference",
                        "target_paper_id": paper.paper_id,
                        "prompt": _pairwise_prompt(paper, chosen, rejected),
                        "chosen_reference_id": chosen.reference_id,
                        "rejected_reference_id": rejected.reference_id,
                        "chosen": _selection_payload(chosen),
                        "rejected": _selection_payload(rejected),
                        "label_source": "proxy_supported_vs_unsupported",
                    }
                )
    return records


def export_finetuning_datasets(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
    output_dir: str | Path,
    *,
    top_k: int,
    gold_report: dict[str, Any] | None = None,
    target_paper_ids: set[str] | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    format_sft_records = make_format_sft_records(
        papers,
        judge_results,
        top_k=top_k,
        target_paper_ids=target_paper_ids,
    )
    proxy_positive_sft_records = make_proxy_positive_sft_records(
        papers,
        judge_results,
        proxy_report,
        top_k=top_k,
        target_paper_ids=target_paper_ids,
    )
    pairwise_preference_records = make_pairwise_preference_records(
        papers,
        judge_results,
        proxy_report,
        target_paper_ids=target_paper_ids,
    )
    context_hard_negative_dpo_records = make_context_hard_negative_dpo_records(
        papers,
        judge_results,
        target_paper_ids=target_paper_ids,
    )
    method_context_contrastive_dpo_records = make_method_context_contrastive_dpo_records(
        papers,
        target_paper_ids=target_paper_ids,
    )
    dpo_preference_records = make_dpo_preference_records(
        papers,
        judge_results,
        proxy_report,
        target_paper_ids=target_paper_ids,
    )
    gold_hard_negative_dpo_records = make_gold_hard_negative_dpo_records(
        papers,
        judge_results,
        gold_report or {},
        target_paper_ids=target_paper_ids,
    )
    merged_dpo_preference_records = merge_dpo_preference_records(
        dpo_preference_records,
        context_hard_negative_records=context_hard_negative_dpo_records,
        method_context_contrastive_records=method_context_contrastive_dpo_records,
        gold_hard_negative_records=gold_hard_negative_dpo_records,
    )
    _write_jsonl(
        destination / "format_sft.jsonl",
        format_sft_records,
    )
    _write_jsonl(
        destination / "proxy_positive_sft.jsonl",
        proxy_positive_sft_records,
    )
    _write_jsonl(
        destination / "pairwise_preferences.jsonl",
        pairwise_preference_records,
    )
    _write_jsonl(
        destination / "dpo_preferences.jsonl",
        dpo_preference_records,
    )
    _write_jsonl(
        destination / "context_hard_negative_dpo.jsonl",
        context_hard_negative_dpo_records,
    )
    _write_jsonl(
        destination / "method_context_contrastive_dpo.jsonl",
        method_context_contrastive_dpo_records,
    )
    _write_jsonl(
        destination / "gold_hard_negative_dpo.jsonl",
        gold_hard_negative_dpo_records,
    )
    _write_jsonl(
        destination / "merged_dpo_preferences.jsonl",
        merged_dpo_preference_records,
    )
    readiness = finetuning_readiness_report(
        format_sft_records=format_sft_records,
        proxy_positive_sft_records=proxy_positive_sft_records,
        pairwise_preference_records=pairwise_preference_records,
        dpo_preference_records=dpo_preference_records,
        context_hard_negative_dpo_records=context_hard_negative_dpo_records,
        method_context_contrastive_dpo_records=method_context_contrastive_dpo_records,
        gold_hard_negative_dpo_records=gold_hard_negative_dpo_records,
        merged_dpo_preference_records=merged_dpo_preference_records,
    )
    (destination / "finetuning_readiness.json").write_text(
        json.dumps(readiness, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "finetuning_readiness.md").write_text(
        render_finetuning_readiness_markdown(readiness),
        encoding="utf-8",
    )


def finetuning_readiness_report(
    *,
    format_sft_records: list[dict[str, Any]],
    proxy_positive_sft_records: list[dict[str, Any]],
    pairwise_preference_records: list[dict[str, Any]],
    dpo_preference_records: list[dict[str, Any]],
    context_hard_negative_dpo_records: list[dict[str, Any]] | None = None,
    method_context_contrastive_dpo_records: list[dict[str, Any]] | None = None,
    gold_hard_negative_dpo_records: list[dict[str, Any]],
    merged_dpo_preference_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    has_merged_dpo_preferences = merged_dpo_preference_records is not None
    context_hard_negative_dpo_records = context_hard_negative_dpo_records or []
    method_context_contrastive_dpo_records = (
        method_context_contrastive_dpo_records or []
    )
    merged_dpo_preference_records = merged_dpo_preference_records or []
    sft_record_count = len(format_sft_records) + len(proxy_positive_sft_records)
    sft_quality = _sft_training_quality(format_sft_records + proxy_positive_sft_records)
    effective_sft_record_count = (
        sft_quality["parsed_record_count"] - sft_quality["empty_selection_count"]
    )
    dpo_record_count = (
        len(dpo_preference_records)
        + len(context_hard_negative_dpo_records)
        + len(method_context_contrastive_dpo_records)
        + len(gold_hard_negative_dpo_records)
    )
    raw_dpo_preference_records = (
        dpo_preference_records
        + context_hard_negative_dpo_records
        + method_context_contrastive_dpo_records
        + gold_hard_negative_dpo_records
    )
    effective_dpo_record_count = (
        len(merged_dpo_preference_records)
        if has_merged_dpo_preferences
        else dpo_record_count
    )
    effective_dpo_quality = _preference_training_quality(
        merged_dpo_preference_records
        if has_merged_dpo_preferences
        else raw_dpo_preference_records
    )
    training_data_quality_blockers = _training_data_quality_blockers(
        sft_quality=sft_quality,
        effective_dpo_quality=effective_dpo_quality,
    )
    unique_targets = _unique_training_targets(
        format_sft_records,
        proxy_positive_sft_records,
        pairwise_preference_records,
        dpo_preference_records,
        context_hard_negative_dpo_records,
        method_context_contrastive_dpo_records,
        gold_hard_negative_dpo_records,
    )
    ready_for_sft_smoke = (
        effective_sft_record_count >= 20 and not training_data_quality_blockers
    )
    ready_for_dpo_smoke = (
        effective_dpo_record_count >= 10 and not training_data_quality_blockers
    )
    ready_for_full_dpo = (
        effective_dpo_record_count >= 200 and len(gold_hard_negative_dpo_records) >= 25
        and not training_data_quality_blockers
    )
    return {
        "dataset_counts": {
            "format_sft": len(format_sft_records),
            "proxy_positive_sft": len(proxy_positive_sft_records),
            "pairwise_preferences": len(pairwise_preference_records),
            "dpo_preferences": len(dpo_preference_records),
            "context_hard_negative_dpo": len(context_hard_negative_dpo_records),
            "method_context_contrastive_dpo": len(
                method_context_contrastive_dpo_records
            ),
            "gold_hard_negative_dpo": len(gold_hard_negative_dpo_records),
            "merged_dpo_preferences": len(merged_dpo_preference_records),
        },
        "summary": {
            "sft_record_count": sft_record_count,
            "effective_sft_record_count": effective_sft_record_count,
            "dpo_record_count": dpo_record_count,
            "merged_dpo_record_count": len(merged_dpo_preference_records),
            "effective_dpo_record_count": effective_dpo_record_count,
            "gold_dpo_record_count": len(gold_hard_negative_dpo_records),
            "unique_training_targets": len(unique_targets),
        },
        "readiness": {
            "ready_for_sft_smoke": ready_for_sft_smoke,
            "ready_for_dpo_smoke": ready_for_dpo_smoke,
            "ready_for_full_dpo": ready_for_full_dpo,
            "has_training_data_quality_blockers": bool(
                training_data_quality_blockers
            ),
            "training_data_quality_blockers": training_data_quality_blockers,
        },
        "training_data_quality": {
            "sft": sft_quality,
            "raw_dpo": _preference_training_quality(raw_dpo_preference_records),
            "effective_dpo": effective_dpo_quality,
        },
        "recommendation": _finetuning_recommendation(
            sft_record_count=sft_record_count,
            dpo_record_count=effective_dpo_record_count,
            gold_dpo_record_count=len(gold_hard_negative_dpo_records),
            ready_for_sft_smoke=ready_for_sft_smoke,
            ready_for_dpo_smoke=ready_for_dpo_smoke,
            ready_for_full_dpo=ready_for_full_dpo,
            training_data_quality_blockers=training_data_quality_blockers,
        ),
        "limitations": [
            "Proxy-labeled examples improve format and weak preference behavior but are not human labels.",
            "Gold hard-negative DPO is the strongest signal when benchmark positives are available.",
            "Use dev comparison before promoting a fine-tuned judge to test or production runs.",
        ],
    }


def render_finetuning_readiness_markdown(report: dict[str, Any]) -> str:
    counts = report.get("dataset_counts", {})
    summary = report.get("summary", {})
    readiness = report.get("readiness", {})
    recommendation = report.get("recommendation", {})
    training_data_quality = report.get("training_data_quality", {})
    lines = [
        "# Finetuning Readiness",
        "",
        "## Dataset Counts",
        "",
        "| Dataset | Count |",
        "| --- | ---: |",
    ]
    for key in [
        "format_sft",
        "proxy_positive_sft",
        "pairwise_preferences",
        "dpo_preferences",
        "context_hard_negative_dpo",
        "method_context_contrastive_dpo",
        "gold_hard_negative_dpo",
        "merged_dpo_preferences",
    ]:
        lines.append(f"| {key} | {counts.get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| sft_record_count | {summary.get('sft_record_count', 0)} |",
            f"| effective_sft_record_count | {summary.get('effective_sft_record_count', summary.get('sft_record_count', 0))} |",
            f"| dpo_record_count | {summary.get('dpo_record_count', 0)} |",
            f"| merged_dpo_record_count | {summary.get('merged_dpo_record_count', 0)} |",
            f"| effective_dpo_record_count | {summary.get('effective_dpo_record_count', summary.get('dpo_record_count', 0))} |",
            f"| gold_dpo_record_count | {summary.get('gold_dpo_record_count', 0)} |",
            f"| unique_training_targets | {summary.get('unique_training_targets', 0)} |",
            "",
            "## Readiness",
            "",
            "| Gate | Value |",
            "| --- | --- |",
            f"| ready_for_sft_smoke | {_yes_no(readiness.get('ready_for_sft_smoke'))} |",
            f"| ready_for_dpo_smoke | {_yes_no(readiness.get('ready_for_dpo_smoke'))} |",
            f"| ready_for_full_dpo | {_yes_no(readiness.get('ready_for_full_dpo'))} |",
            f"| has_training_data_quality_blockers | {_yes_no(readiness.get('has_training_data_quality_blockers'))} |",
            f"| training_data_quality_blockers | {_format_list(readiness.get('training_data_quality_blockers', []))} |",
            "",
            "## Training Data Quality",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[
                f"| {prefix}.{metric} | {_format_quality_value(value)} |"
                for prefix, quality in [
                    ("sft", training_data_quality.get("sft", {})),
                    ("raw_dpo", training_data_quality.get("raw_dpo", {})),
                    ("effective_dpo", training_data_quality.get("effective_dpo", {})),
                ]
                for metric, value in [
                    ("record_count", quality.get("record_count", 0)),
                    ("unique_targets", quality.get("unique_targets", 0)),
                    ("max_target_share", quality.get("max_target_share", 0.0)),
                    ("parse_error_count", quality.get("parse_error_count", 0)),
                    (
                        "empty_selection_rate",
                        quality.get("empty_selection_rate", 0.0),
                    ),
                    ("invalid_pair_count", quality.get("invalid_pair_count", 0)),
                    (
                        "reverse_conflict_count",
                        quality.get("reverse_conflict_count", 0),
                    ),
                    ("quality_flags", quality.get("quality_flags", [])),
                ]
            ],
            "",
            "## Recommendation",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| next_step | {recommendation.get('next_step', '')} |",
            f"| reason | {recommendation.get('reason', '')} |",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def make_dpo_preference_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
    *,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for record in make_pairwise_preference_records(
        papers,
        judge_results,
        proxy_report,
        target_paper_ids=target_paper_ids,
    ):
        records.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": record["prompt"],
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": record["chosen_reference_id"],
                },
                "rejected": {
                    "from": "gpt",
                    "value": record["rejected_reference_id"],
                },
                "target_paper_id": record["target_paper_id"],
                "chosen_reference_id": record["chosen_reference_id"],
                "rejected_reference_id": record["rejected_reference_id"],
                "label_source": record["label_source"],
            }
        )
    return records


def merge_dpo_preference_records(
    dpo_preference_records: list[dict[str, Any]],
    *,
    context_hard_negative_records: list[dict[str, Any]] | None = None,
    method_context_contrastive_records: list[dict[str, Any]] | None = None,
    gold_hard_negative_records: list[dict[str, Any]] | None = None,
    max_pairs_per_target: int = 12,
) -> list[dict[str, Any]]:
    ordered_groups = [
        gold_hard_negative_records or [],
        context_hard_negative_records or [],
        method_context_contrastive_records or [],
        dpo_preference_records,
    ]
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_unordered_pairs: set[tuple[str, str, frozenset[str]]] = set()
    per_target_counts: dict[str, int] = {}
    merged: list[dict[str, Any]] = []
    for records in ordered_groups:
        for record in records:
            target_id = str(record.get("target_paper_id", ""))
            chosen_id = str(record.get("chosen_reference_id", ""))
            rejected_id = str(record.get("rejected_reference_id", ""))
            if not target_id or not chosen_id or not rejected_id:
                continue
            if chosen_id == rejected_id:
                continue
            key = (target_id, chosen_id, rejected_id)
            unordered_key = (target_id, "pair", frozenset({chosen_id, rejected_id}))
            if key in seen_pairs:
                continue
            if unordered_key in seen_unordered_pairs:
                continue
            if per_target_counts.get(target_id, 0) >= max_pairs_per_target:
                continue
            seen_pairs.add(key)
            seen_unordered_pairs.add(unordered_key)
            per_target_counts[target_id] = per_target_counts.get(target_id, 0) + 1
            merged.append(record)
    return merged


def make_context_hard_negative_dpo_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    results_by_id = {result.target_paper_id: result for result in judge_results}
    records: list[dict[str, Any]] = []
    for paper in papers:
        if not _target_is_in_scope(paper.paper_id, target_paper_ids):
            continue
        result = results_by_id.get(paper.paper_id)
        if result is None:
            continue
        selected_ids = [
            selection.reference_id
            for selection in sorted(result.selected_references, key=lambda item: item.rank)
            if selection.valid
        ]
        supported_ids = _context_supported_reference_ids(paper)
        if not supported_ids or not selected_ids:
            continue
        rejected_ids = [
            reference_id for reference_id in selected_ids if reference_id not in supported_ids
        ]
        if not rejected_ids:
            continue
        for chosen_id in sorted(supported_ids):
            if chosen_id in selected_ids:
                continue
            for rejected_id in rejected_ids:
                records.append(
                    _sharegpt_preference_record(
                        paper=paper,
                        chosen_reference_id=chosen_id,
                        rejected_reference_id=rejected_id,
                        label_source="context_hard_negative",
                        prompt=_context_pairwise_prompt(
                            paper,
                            chosen_id,
                            rejected_id,
                        ),
                    )
                )
    return records


def make_method_context_contrastive_dpo_records(
    papers: list[Paper],
    *,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for paper in papers:
        if not _target_is_in_scope(paper.paper_id, target_paper_ids):
            continue
        reference_ids = [
            reference.reference_id for reference in paper.references if reference.reference_id
        ]
        contexts_by_reference = {
            reference_id: [
                context
                for context in paper.citation_contexts
                if context.reference_id == reference_id
            ]
            for reference_id in reference_ids
        }
        chosen_ids = [
            reference_id
            for reference_id, contexts in contexts_by_reference.items()
            if contexts and context_weight(contexts) >= CONTEXT_WEIGHT_METHOD_SUPPORTED
        ]
        rejected_ids = [
            reference_id
            for reference_id, contexts in contexts_by_reference.items()
            if not contexts
        ]
        for chosen_id in sorted(chosen_ids):
            for rejected_id in sorted(rejected_ids):
                records.append(
                    _sharegpt_preference_record(
                        paper=paper,
                        chosen_reference_id=chosen_id,
                        rejected_reference_id=rejected_id,
                        label_source="method_context_contrastive",
                        prompt=_context_pairwise_prompt(
                            paper,
                            chosen_id,
                            rejected_id,
                        ),
                    )
                )
    return records


def make_gold_hard_negative_dpo_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    gold_report: dict[str, Any],
    *,
    target_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    results_by_id = {result.target_paper_id: result for result in judge_results}
    records: list[dict[str, Any]] = []
    for row in gold_report.get("targets", []):
        if row.get("hit"):
            continue
        target_id = str(row.get("target_paper_id"))
        if not _target_is_in_scope(target_id, target_paper_ids):
            continue
        paper = papers_by_id.get(target_id)
        result = results_by_id.get(target_id)
        if paper is None or result is None:
            continue
        positive_ids = [str(value) for value in row.get("positive_reference_ids", [])]
        selected_ids = [
            selection.reference_id
            for selection in sorted(result.selected_references, key=lambda item: item.rank)
            if selection.valid
        ]
        for chosen_id in positive_ids:
            if chosen_id not in {reference.reference_id for reference in paper.references}:
                continue
            for rejected_id in selected_ids:
                if rejected_id == chosen_id:
                    continue
                records.append(
                    _sharegpt_preference_record(
                        paper=paper,
                        chosen_reference_id=chosen_id,
                        rejected_reference_id=rejected_id,
                        label_source="gold_hard_negative",
                        prompt=_gold_pairwise_prompt(
                            paper,
                            chosen_id,
                            rejected_id,
                        ),
                    )
                )
    return records


def _sft_record(
    task: str,
    paper: Paper,
    selections: list[SelectedReference],
    *,
    top_k: int,
) -> dict[str, Any]:
    target = {
        "target_paper_id": paper.paper_id,
        "selected_references": [_selection_payload(selection) for selection in selections],
    }
    return {
        "task": task,
        "target_paper_id": paper.paper_id,
        "messages": [
            {"role": "user", "content": build_judge_prompt(paper, top_k=top_k)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "label_source": "model_audit" if task == "format_sft" else "proxy_positive",
    }


def _selection_payload(selection: SelectedReference) -> dict[str, Any]:
    return {
        "reference_id": selection.reference_id,
        "rank": selection.rank,
        "confidence": selection.confidence,
        "rationale": selection.rationale,
    }


def _supported_selection_ids(proxy_report: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (row["target_paper_id"], row["reference_id"])
        for row in proxy_report.get("selected_reference_evidence", [])
        if _is_supported(row)
    }


def _target_is_in_scope(
    target_paper_id: str,
    target_paper_ids: set[str] | None,
) -> bool:
    return target_paper_ids is None or target_paper_id in target_paper_ids


def _is_supported(evidence_row: dict[str, Any]) -> bool:
    return bool(
        evidence_row.get("context_supported")
        or evidence_row.get("bibliometric_top_n_supported")
    )


def _pairwise_prompt(
    paper: Paper,
    chosen: SelectedReference,
    rejected: SelectedReference,
) -> str:
    references_by_id = {reference.reference_id: reference for reference in paper.references}
    chosen_reference = references_by_id.get(chosen.reference_id)
    rejected_reference = references_by_id.get(rejected.reference_id)
    return (
        "Choose which candidate reference is more influential for the target paper. "
        "Return only the winning reference_id.\n\n"
        f"Target paper ID: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract or ''}\n\n"
        "Candidate A:\n"
        f"reference_id: {chosen.reference_id}\n"
        f"title: {chosen_reference.title if chosen_reference else ''}\n\n"
        "Candidate B:\n"
        f"reference_id: {rejected.reference_id}\n"
        f"title: {rejected_reference.title if rejected_reference else ''}"
    )


def _gold_pairwise_prompt(
    paper: Paper,
    chosen_reference_id: str,
    rejected_reference_id: str,
) -> str:
    references_by_id = {reference.reference_id: reference for reference in paper.references}
    chosen_reference = references_by_id.get(chosen_reference_id)
    rejected_reference = references_by_id.get(rejected_reference_id)
    return (
        "Choose which candidate reference is more influential for the target paper. "
        "Return only the winning reference_id.\n\n"
        f"Target paper ID: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract or ''}\n\n"
        "Candidate A:\n"
        f"reference_id: {chosen_reference_id}\n"
        f"title: {chosen_reference.title if chosen_reference else ''}\n\n"
        "Candidate B:\n"
        f"reference_id: {rejected_reference_id}\n"
        f"title: {rejected_reference.title if rejected_reference else ''}"
    )


def _context_pairwise_prompt(
    paper: Paper,
    chosen_reference_id: str,
    rejected_reference_id: str,
) -> str:
    base_prompt = _gold_pairwise_prompt(
        paper,
        chosen_reference_id,
        rejected_reference_id,
    )
    evidence = [
        context.sentence
        for context in paper.citation_contexts
        if context.reference_id == chosen_reference_id
    ]
    return (
        base_prompt
        + "\n\nContext evidence for Candidate A:\n"
        + "\n".join(f"- {sentence}" for sentence in evidence[:3])
    )


def _sharegpt_preference_record(
    *,
    paper: Paper,
    chosen_reference_id: str,
    rejected_reference_id: str,
    label_source: str,
    prompt: str,
) -> dict[str, Any]:
    return {
        "conversations": [
            {
                "from": "human",
                "value": prompt,
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen_reference_id,
        },
        "rejected": {
            "from": "gpt",
            "value": rejected_reference_id,
        },
        "target_paper_id": paper.paper_id,
        "chosen_reference_id": chosen_reference_id,
        "rejected_reference_id": rejected_reference_id,
        "label_source": label_source,
    }


def _context_supported_reference_ids(paper: Paper) -> set[str]:
    return {
        context.reference_id
        for context in paper.citation_contexts
        if context.reference_id
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _unique_training_targets(*record_groups: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["target_paper_id"])
        for records in record_groups
        for record in records
        if record.get("target_paper_id")
    }


def _preference_training_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_pair_count = 0
    invalid_pair_count = 0
    duplicate_pair_count = 0
    reverse_conflict_count = 0
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_pair_orientation: dict[tuple[str, frozenset[str]], tuple[str, str]] = {}
    target_counts: dict[str, int] = {}
    label_source_counts: dict[str, int] = {}

    for record in records:
        target_id = str(record.get("target_paper_id", ""))
        chosen_id = str(record.get("chosen_reference_id", ""))
        rejected_id = str(record.get("rejected_reference_id", ""))
        label_source = str(record.get("label_source") or "unknown")
        label_source_counts[label_source] = label_source_counts.get(label_source, 0) + 1
        if not target_id or not chosen_id or not rejected_id or chosen_id == rejected_id:
            invalid_pair_count += 1
            continue

        valid_pair_count += 1
        target_counts[target_id] = target_counts.get(target_id, 0) + 1
        pair_key = (target_id, chosen_id, rejected_id)
        unordered_key = (target_id, frozenset({chosen_id, rejected_id}))
        previous_orientation = seen_pair_orientation.get(unordered_key)
        if pair_key in seen_pairs:
            duplicate_pair_count += 1
        elif previous_orientation and previous_orientation != (chosen_id, rejected_id):
            reverse_conflict_count += 1
        else:
            seen_pair_orientation[unordered_key] = (chosen_id, rejected_id)
        seen_pairs.add(pair_key)

    max_target_pair_count = max(target_counts.values(), default=0)
    max_target_share = (
        max_target_pair_count / valid_pair_count if valid_pair_count else 0.0
    )
    quality_flags: list[str] = []
    if not records:
        quality_flags.append("empty")
    if invalid_pair_count:
        quality_flags.append("invalid_pairs")
    if reverse_conflict_count:
        quality_flags.append("reverse_conflicts")
    if duplicate_pair_count:
        quality_flags.append("duplicate_pairs")
    if valid_pair_count >= 5 and max_target_share > 0.5:
        quality_flags.append("target_concentrated")

    return {
        "record_count": len(records),
        "valid_pair_count": valid_pair_count,
        "invalid_pair_count": invalid_pair_count,
        "duplicate_pair_count": duplicate_pair_count,
        "reverse_conflict_count": reverse_conflict_count,
        "unique_targets": len(target_counts),
        "max_target_pair_count": max_target_pair_count,
        "max_target_share": max_target_share,
        "label_source_counts": label_source_counts,
        "quality_flags": quality_flags,
    }


def _sft_training_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    parse_error_count = 0
    parsed_record_count = 0
    empty_selection_count = 0
    selected_reference_count = 0
    target_counts: dict[str, int] = {}
    label_source_counts: dict[str, int] = {}

    for record in records:
        target_id = str(record.get("target_paper_id", ""))
        if target_id:
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
        label_source = str(record.get("label_source") or "unknown")
        label_source_counts[label_source] = label_source_counts.get(label_source, 0) + 1
        assistant_content = _assistant_message_content(record)
        if assistant_content is None:
            parse_error_count += 1
            continue
        try:
            payload = json.loads(assistant_content)
        except json.JSONDecodeError:
            parse_error_count += 1
            continue
        selections = payload.get("selected_references")
        if not isinstance(selections, list):
            parse_error_count += 1
            continue
        parsed_record_count += 1
        selected_reference_count += len(selections)
        if not selections:
            empty_selection_count += 1

    max_target_record_count = max(target_counts.values(), default=0)
    max_target_share = (
        max_target_record_count / len(records) if records else 0.0
    )
    empty_selection_rate = (
        empty_selection_count / parsed_record_count if parsed_record_count else 0.0
    )
    average_selected_references = (
        selected_reference_count / parsed_record_count if parsed_record_count else 0.0
    )
    quality_flags: list[str] = []
    if not records:
        quality_flags.append("empty")
    if parse_error_count:
        quality_flags.append("parse_errors")
    if empty_selection_count:
        quality_flags.append("empty_selections")
    if len(records) >= 5 and max_target_share > 0.5:
        quality_flags.append("target_concentrated")

    return {
        "record_count": len(records),
        "parsed_record_count": parsed_record_count,
        "parse_error_count": parse_error_count,
        "empty_selection_count": empty_selection_count,
        "empty_selection_rate": empty_selection_rate,
        "average_selected_references": average_selected_references,
        "unique_targets": len(target_counts),
        "max_target_record_count": max_target_record_count,
        "max_target_share": max_target_share,
        "label_source_counts": label_source_counts,
        "quality_flags": quality_flags,
    }


def _assistant_message_content(record: dict[str, Any]) -> str | None:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _training_data_quality_blockers(
    *,
    sft_quality: dict[str, Any],
    effective_dpo_quality: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if sft_quality.get("parse_error_count", 0) > 0:
        blockers.append("sft_parse_errors")
    if sft_quality.get("empty_selection_count", 0) > 0:
        blockers.append("sft_empty_selections")
    if effective_dpo_quality.get("invalid_pair_count", 0) > 0:
        blockers.append("dpo_invalid_pairs")
    if effective_dpo_quality.get("reverse_conflict_count", 0) > 0:
        blockers.append("dpo_reverse_conflicts")
    return blockers


def _finetuning_recommendation(
    *,
    sft_record_count: int,
    dpo_record_count: int,
    gold_dpo_record_count: int,
    ready_for_sft_smoke: bool,
    ready_for_dpo_smoke: bool,
    ready_for_full_dpo: bool,
    training_data_quality_blockers: list[str] | None = None,
) -> dict[str, str]:
    if training_data_quality_blockers:
        return {
            "next_step": "fix_training_data",
            "reason": "training data has quality blockers: "
            + ", ".join(training_data_quality_blockers),
        }
    if ready_for_full_dpo:
        return {
            "next_step": "dpo_train",
            "reason": "enough DPO records and gold hard negatives for a larger preference run",
        }
    if ready_for_dpo_smoke:
        reason = "enough preference records for a short DPO smoke run"
        if gold_dpo_record_count == 0:
            reason += "; add benchmark gold labels before trusting the adapter"
        return {"next_step": "dpo_smoke", "reason": reason}
    if ready_for_sft_smoke:
        return {
            "next_step": "sft_smoke",
            "reason": "enough SFT records to improve output format before preference training",
        }
    return {
        "next_step": "collect_more_examples",
        "reason": (
            "need at least 20 SFT records or 10 DPO records before training; "
            f"currently sft={sft_record_count}, dpo={dpo_record_count}"
        ),
    }


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_quality_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return _format_list(value)
    return str(value)


def _format_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "none"
    return ", ".join(str(value) for value in values)
