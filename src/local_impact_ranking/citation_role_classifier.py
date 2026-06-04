from __future__ import annotations

import hashlib
import json
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
    context_index: int,
) -> str:
    reference_title = _reference_title(paper, context.reference_id)
    source_hint = _source_role_hint(context)
    role_lines = "\n".join(
        f"- {role}: {ROLE_DESCRIPTIONS[role]}" for role in ALLOWED_CITATION_ROLES
    )
    return (
        "You classify citation context roles for a reproducible scholarly impact scorer. "
        "Return JSON only. Choose exactly one role from the allowed list. "
        "Do not invent roles. If the sentence is insufficient, choose unknown.\n\n"
        f"Prompt version: {MODEL_CITATION_ROLE_PROMPT_VERSION}\n"
        f"Citing paper ID: {paper.paper_id}\n"
        f"Citing title: {paper.title}\n"
        f"Citing abstract: {paper.abstract or ''}\n"
        f"Context index: {context_index}\n"
        f"Reference ID: {context.reference_id}\n"
        f"Reference title: {reference_title}\n"
        f"Section: {context.section or ''}\n"
        f"Sentence: {context.sentence}\n"
        f"Paragraph: {context.paragraph or ''}\n"
        f"Source role hint, non-authoritative: {source_hint or ''}\n\n"
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
        return {
            "role": "unknown",
            "confidence": 0.0,
            "rationale": "The raw output was not valid JSON.",
            "validation_status": "invalid_json",
        }
    role = str(payload.get("role", "")).strip().lower()
    if role not in ALLOWED_CITATION_ROLES:
        return {
            "role": "unknown",
            "confidence": 0.0,
            "rationale": f"Illegal role: {role}",
            "validation_status": "invalid_role",
        }
    return {
        "role": role,
        "confidence": _clamp_confidence(payload.get("confidence")),
        "rationale": str(payload.get("rationale", "")),
        "validation_status": "valid",
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


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
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
