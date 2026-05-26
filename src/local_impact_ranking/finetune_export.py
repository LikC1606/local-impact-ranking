from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_impact_ranking.judge import JudgeResult, SelectedReference, build_judge_prompt
from local_impact_ranking.models import Paper


def make_format_sft_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    records = []
    for result in judge_results:
        paper = papers_by_id.get(result.target_paper_id)
        if paper is None or result.error is not None or result.parse_error is not None:
            continue
        selections = [selection for selection in result.selected_references if selection.valid]
        records.append(_sft_record("format_sft", paper, selections, top_k=top_k))
    return records


def make_proxy_positive_sft_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    supported_ids = _supported_selection_ids(proxy_report)
    records = []
    for result in judge_results:
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
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    evidence_by_pair = {
        (row["target_paper_id"], row["reference_id"]): row
        for row in proxy_report.get("selected_reference_evidence", [])
    }
    records: list[dict[str, Any]] = []
    for result in judge_results:
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
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        destination / "format_sft.jsonl",
        make_format_sft_records(papers, judge_results, top_k=top_k),
    )
    _write_jsonl(
        destination / "proxy_positive_sft.jsonl",
        make_proxy_positive_sft_records(
            papers,
            judge_results,
            proxy_report,
            top_k=top_k,
        ),
    )
    _write_jsonl(
        destination / "pairwise_preferences.jsonl",
        make_pairwise_preference_records(papers, judge_results, proxy_report),
    )
    _write_jsonl(
        destination / "dpo_preferences.jsonl",
        make_dpo_preference_records(papers, judge_results, proxy_report),
    )
    _write_jsonl(
        destination / "gold_hard_negative_dpo.jsonl",
        make_gold_hard_negative_dpo_records(
            papers,
            judge_results,
            gold_report or {},
        ),
    )


def make_dpo_preference_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    proxy_report: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for record in make_pairwise_preference_records(papers, judge_results, proxy_report):
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


def make_gold_hard_negative_dpo_records(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    gold_report: dict[str, Any],
) -> list[dict[str, Any]]:
    papers_by_id = {paper.paper_id: paper for paper in papers}
    results_by_id = {result.target_paper_id: result for result in judge_results}
    records: list[dict[str, Any]] = []
    for row in gold_report.get("targets", []):
        if row.get("hit"):
            continue
        target_id = str(row.get("target_paper_id"))
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
                    {
                        "conversations": [
                            {
                                "from": "human",
                                "value": _gold_pairwise_prompt(
                                    paper,
                                    chosen_id,
                                    rejected_id,
                                ),
                            }
                        ],
                        "chosen": {
                            "from": "gpt",
                            "value": chosen_id,
                        },
                        "rejected": {
                            "from": "gpt",
                            "value": rejected_id,
                        },
                        "target_paper_id": target_id,
                        "chosen_reference_id": chosen_id,
                        "rejected_reference_id": rejected_id,
                        "label_source": "gold_hard_negative",
                    }
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
