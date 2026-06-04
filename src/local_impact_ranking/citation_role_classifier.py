from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.enrichment_resume import write_jsonl_atomic
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import CitationContext, Paper


ALLOWED_CITATION_ROLES = (
    "guideline",
    "clinical",
    "method",
    "dataset_or_tool",
    "negative_or_contrasting",
    "supporting",
    "background",
    "unknown",
)

MODEL_CITATION_ROLE_PROMPT_VERSION = "citation_role_classifier_v1"


ROLE_DESCRIPTIONS = {
    "guideline": "guideline, consensus, policy, or clinical recommendation",
    "clinical": "clinical trial, patient cohort, real-world clinical evidence, diagnosis, treatment, or translational clinical use",
    "method": "protocol, assay, reagent, antibody, plasmid, cell line, animal model, sequencing, PCR, ELISA, CRISPR, or other method reuse",
    "dataset_or_tool": "dataset, database, software, tool, platform, benchmark, repository, or reusable resource",
    "negative_or_contrasting": "contradiction, failed replication, limitation, refutation, or contrasting evidence",
    "supporting": "evidence that supports or confirms a claim",
    "background": "related work, literature framing, general introduction, or ordinary background mention",
    "unknown": "not enough context to classify reliably",
}


def classify_citation_contexts(
    input_path: str | Path,
    output_path: str | Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int | None = 256,
    journal_path: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    papers = load_papers_jsonl(input_path)
    replayed_results = _load_role_journal(journal_path)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    rows: list[dict[str, Any]] = []
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "journal_path": str(journal_path) if journal_path is not None else None,
        "model": model,
        "paper_count": len(papers),
        "citation_context_count": 0,
        "classified_context_count": 0,
        "replayed_context_count": 0,
        "empty_context_count": 0,
        "model_request_count": 0,
    }

    try:
        for paper in papers:
            row = paper.model_dump(mode="json", exclude_none=True)
            context_rows = row.get("citation_contexts", [])
            for context_index, context_row in enumerate(context_rows):
                context = paper.citation_contexts[context_index]
                report["citation_context_count"] += 1
                source_hint = _source_role_hint_from_record(context_row)
                if source_hint:
                    context_row["source_role_hint"] = source_hint
                key = _context_key(
                    paper.paper_id,
                    context_index,
                    context.reference_id,
                    context.sentence,
                )
                replayed = replayed_results.get(key)
                if replayed is not None:
                    _apply_role_result(context_row, replayed)
                    report["replayed_context_count"] += 1
                    continue
                if not context.sentence.strip():
                    _apply_role_result(
                        context_row,
                        {
                            "role": "unknown",
                            "confidence": 0.0,
                            "rationale": "Citation context sentence was empty.",
                            "validation_status": "empty_sentence",
                            "model": model,
                            "prompt_sha256": "",
                            "raw_output": "",
                        },
                    )
                    report["empty_context_count"] += 1
                    continue
                prompt = build_citation_role_prompt(
                    paper,
                    context,
                    context_index=context_index,
                )
                digest = prompt_sha256(prompt)
                raw_output = _complete_role_prompt(
                    prompt,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    client=http_client,
                )
                report["model_request_count"] += 1
                parsed = parse_citation_role_output(raw_output)
                result = {
                    **parsed,
                    "model": model,
                    "prompt_sha256": digest,
                    "raw_output": raw_output,
                }
                _apply_role_result(context_row, result)
                _append_role_journal(
                    journal_path,
                    {
                        "paper_id": paper.paper_id,
                        "context_index": context_index,
                        "reference_id": context.reference_id,
                        "sentence": context.sentence,
                        "role": result["role"],
                        "confidence": result["confidence"],
                        "rationale": result["rationale"],
                        "validation_status": result["validation_status"],
                        "model": model,
                        "prompt_sha256": digest,
                        "raw_output": raw_output,
                    },
                )
                report["classified_context_count"] += 1
            rows.append(row)
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()

    _write_jsonl(output_path, rows)
    return report


def build_citation_role_prompt(
    paper: Paper,
    context: CitationContext,
    *,
    context_index: int = 0,
) -> str:
    reference_title = _reference_title(paper, context.reference_id)
    source_hint = _source_role_hint(context)
    citation_data = {
        "citing_paper_id": paper.paper_id,
        "citing_title": paper.title,
        "citing_abstract": paper.abstract or "",
        "context_index": context_index,
        "reference_id": context.reference_id,
        "reference_title": reference_title,
        "section": context.section or "",
        "sentence": context.sentence,
        "paragraph": context.paragraph or "",
        "source_role_hint_non_authoritative": source_hint or "",
    }
    citation_data_json = json.dumps(citation_data, ensure_ascii=False, indent=2)
    role_lines = "\n".join(
        f"- {role}: {ROLE_DESCRIPTIONS[role]}" for role in ALLOWED_CITATION_ROLES
    )
    return (
        "You classify citation context roles for a reproducible scholarly impact scorer. "
        "Return JSON only. Choose exactly one role from the allowed list. "
        "Do not invent roles. If the sentence is insufficient, choose unknown.\n\n"
        f"Prompt version: {MODEL_CITATION_ROLE_PROMPT_VERSION}\n"
        f"Context index: {context_index}\n"
        f"Citation data:\n{citation_data_json}\n\n"
        f"Allowed roles:\n{role_lines}\n\n"
        "Return this JSON shape exactly: "
        '{"role":"method","confidence":0.0,"rationale":"short reason"}'
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_citation_role_output(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return _unknown_parse_result(
            "invalid_json",
            "The raw output was not valid JSON.",
        )
    if not isinstance(payload, dict):
        return _unknown_parse_result(
            "invalid_schema",
            "The JSON output must be an object with role, confidence, and rationale.",
        )
    role = str(payload.get("role", "")).strip().lower()
    if role not in ALLOWED_CITATION_ROLES:
        return _unknown_parse_result("invalid_role", f"Illegal role: {role}")
    confidence = _clamp_confidence(payload.get("confidence"))
    if confidence is None:
        return _unknown_parse_result(
            "invalid_confidence",
            "The confidence value was not numeric.",
        )
    return {
        "role": role,
        "confidence": confidence,
        "rationale": str(payload.get("rationale", "")),
        "validation_status": "valid",
    }


def _unknown_parse_result(validation_status: str, rationale: str) -> dict[str, Any]:
    return {
        "role": "unknown",
        "confidence": 0.0,
        "rationale": rationale,
        "validation_status": validation_status,
    }


def _complete_role_prompt(
    prompt: str,
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
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
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
    completion = response.json()
    return str(completion["choices"][0]["message"]["content"])


def _apply_role_result(
    context_record: dict[str, Any],
    result: dict[str, Any],
) -> None:
    confidence = result.get("confidence", 0.0)
    context_record["role"] = str(result.get("role", "unknown"))
    context_record["role_confidence"] = float(
        confidence if confidence is not None else 0.0
    )
    context_record["role_rationale"] = str(result.get("rationale", ""))
    context_record["role_source"] = "model"
    context_record["role_model"] = str(result.get("model", ""))
    context_record["role_prompt_sha256"] = str(result.get("prompt_sha256", ""))
    context_record["role_raw_output"] = str(result.get("raw_output", ""))
    context_record["role_validation_status"] = str(
        result.get("validation_status", "unknown")
    )


def _context_key(
    paper_id: str,
    context_index: int,
    reference_id: str,
    sentence: str,
) -> tuple[str, int, str, str]:
    return (paper_id, context_index, reference_id, sentence)


def _load_role_journal(
    journal_path: str | Path | None,
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    if journal_path is None:
        return {}
    source = Path(journal_path)
    if not source.exists():
        return {}
    results: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            try:
                key = _context_key(
                    str(record.get("paper_id", "")),
                    int(record.get("context_index", 0)),
                    str(record.get("reference_id", "")),
                    str(record.get("sentence", "")),
                )
            except (TypeError, ValueError):
                continue
            results[key] = _normalize_role_journal_record(record)
    return results


def _append_role_journal(
    journal_path: str | Path | None,
    record: dict[str, Any],
) -> None:
    if journal_path is None:
        return
    destination = Path(journal_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _source_role_hint_from_record(record: dict[str, Any]) -> str | None:
    existing_hint = record.get("source_role_hint")
    if existing_hint:
        return str(existing_hint)
    for key in ("role", "label", "intent", "citation_intent", "category"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _normalize_role_journal_record(record: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_citation_role_output(
        json.dumps(
            {
                "role": record.get("role", ""),
                "confidence": record.get("confidence"),
                "rationale": _string_or_empty(record.get("rationale")),
            },
            ensure_ascii=False,
        )
    )
    validation_status = parsed["validation_status"]
    journal_status = _string_or_empty(record.get("validation_status"))
    if validation_status == "valid" and journal_status not in {"", "valid"}:
        validation_status = journal_status
    if validation_status != "valid":
        return {
            "role": "unknown",
            "confidence": 0.0,
            "rationale": parsed["rationale"]
            or f"Cached journal record was marked {validation_status}.",
            "validation_status": validation_status,
            "model": _string_or_empty(record.get("model")),
            "prompt_sha256": _string_or_empty(record.get("prompt_sha256")),
            "raw_output": _string_or_empty(record.get("raw_output")),
        }
    return {
        "role": parsed["role"],
        "confidence": parsed["confidence"],
        "rationale": parsed["rationale"],
        "validation_status": validation_status,
        "model": _string_or_empty(record.get("model")),
        "prompt_sha256": _string_or_empty(record.get("prompt_sha256")),
        "raw_output": _string_or_empty(record.get("raw_output")),
    }


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(path, rows)


def _clamp_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


def _reference_title(paper: Paper, reference_id: str) -> str:
    for reference in paper.references:
        if reference.reference_id == reference_id:
            return reference.title
    return ""


def _source_role_hint(context: CitationContext) -> str | None:
    extra = getattr(context, "__pydantic_extra__", None) or {}
    existing_hint = extra.get("source_role_hint")
    if existing_hint:
        return str(existing_hint)
    for key in ("role", "label", "intent", "citation_intent", "category"):
        value = extra.get(key)
        if value:
            return str(value)
    return None
