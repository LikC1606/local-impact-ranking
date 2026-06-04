from __future__ import annotations

import hashlib
import json
import math
from typing import Any

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
        payload = json.loads(_extract_json_object(raw_output))
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


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _clamp_confidence(value: Any) -> float | None:
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
    for key in ("role", "label", "intent", "citation_intent", "category"):
        value = extra.get(key)
        if value:
            return str(value)
    return None
