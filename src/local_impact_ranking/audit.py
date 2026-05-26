from __future__ import annotations

import json
import hashlib
from pathlib import Path

from local_impact_ranking.judge import JudgeResult, SelectedReference, build_judge_prompt
from local_impact_ranking.models import Paper


def judge_result_to_record(result: JudgeResult) -> dict:
    return {
        "target_paper_id": result.target_paper_id,
        "parse_error": result.parse_error,
        "error": result.error,
        "prompt_sha256": result.prompt_sha256,
        "candidate_reference_ids": list(result.candidate_reference_ids),
        "invalid_selection_count": result.invalid_selection_count,
        "selected_references": [
            {
                "reference_id": selection.reference_id,
                "rank": selection.rank,
                "confidence": selection.confidence,
                "rationale": selection.rationale,
                "valid": selection.valid,
            }
            for selection in result.selected_references
        ],
        "raw_output": result.raw_output,
    }


def judge_result_from_record(record: dict) -> JudgeResult:
    return JudgeResult(
        target_paper_id=str(record["target_paper_id"]),
        selected_references=[
            SelectedReference(
                reference_id=str(item["reference_id"]),
                rank=int(item["rank"]),
                confidence=float(item["confidence"]),
                rationale=str(item.get("rationale", "")),
                valid=bool(item.get("valid", True)),
            )
            for item in record.get("selected_references", [])
        ],
        raw_output=str(record.get("raw_output", "")),
        parse_error=record.get("parse_error"),
        error=record.get("error"),
        prompt_sha256=record.get("prompt_sha256"),
        candidate_reference_ids=tuple(record.get("candidate_reference_ids", [])),
    )


def append_judge_result(path: str | Path, result: JudgeResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judge_result_to_record(result)) + "\n")


def prompt_audit_record(paper: Paper, top_k: int) -> dict:
    prompt = build_judge_prompt(paper, top_k=top_k)
    return {
        "target_paper_id": paper.paper_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "candidate_reference_ids": [
            reference.reference_id or "" for reference in paper.references
        ],
        "prompt": prompt,
    }


def write_prompt_audit(papers: list[Paper], path: str | Path, top_k: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(prompt_audit_record(paper, top_k=top_k)) + "\n")


def load_judge_results(path: str | Path) -> list[JudgeResult]:
    source = Path(path)
    if not source.exists():
        return []
    results: list[JudgeResult] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                results.append(judge_result_from_record(json.loads(stripped)))
    return results


def completed_paper_ids(path: str | Path) -> set[str]:
    return {result.target_paper_id for result in load_judge_results(path)}
