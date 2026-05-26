from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from local_impact_ranking.contexts import citation_contexts_by_reference
from local_impact_ranking.models import Paper


@dataclass(frozen=True)
class SelectedReference:
    reference_id: str
    rank: int
    confidence: float
    rationale: str
    valid: bool = True


@dataclass(frozen=True)
class JudgeResult:
    target_paper_id: str
    selected_references: list[SelectedReference]
    raw_output: str
    parse_error: str | None = None
    error: str | None = None
    prompt_sha256: str | None = None
    candidate_reference_ids: tuple[str, ...] = ()

    @property
    def invalid_selection_count(self) -> int:
        return sum(1 for selection in self.selected_references if not selection.valid)


def error_result(target_paper_id: str, error: str) -> JudgeResult:
    return JudgeResult(
        target_paper_id=target_paper_id,
        selected_references=[],
        raw_output="",
        parse_error=None,
        error=error,
    )


def with_judge_audit_metadata(
    result: JudgeResult,
    paper: Paper,
    *,
    top_k: int,
    prompt: str | None = None,
) -> JudgeResult:
    prompt_text = prompt if prompt is not None else build_judge_prompt(paper, top_k=top_k)
    return replace(
        result,
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        candidate_reference_ids=tuple(
            reference.reference_id or "" for reference in paper.references
        ),
    )


class Judge(Protocol):
    def judge(self, paper: Paper, top_k: int = 5) -> JudgeResult:
        ...


def build_judge_prompt(paper: Paper, top_k: int = 5) -> str:
    contexts_by_reference = citation_contexts_by_reference(paper)
    single_candidate_rule = ""
    if len(paper.references) == 1:
        only_reference_id = paper.references[0].reference_id or ""
        single_candidate_rule = (
            "\nThere is exactly one single candidate reference. You must select that candidate "
            f'with reference_id {json.dumps(only_reference_id)} if it is present.'
        )
    references = "\n".join(
        f"- reference_id: {reference.reference_id}\n"
        f"  title: {reference.title}\n"
        f"  year: {reference.year or ''}\n"
        f"  venue: {reference.venue or ''}"
        f"{_format_reference_contexts(reference.reference_id or '', contexts_by_reference)}"
        for reference in paper.references
    )
    return (
        "You are ranking scholarly influence. Choose only from the candidate "
        "references listed below. Do not invent paper IDs. Return JSON only; "
        "do not include reasoning, markdown, or commentary before or after the JSON.\n\n"
        f"{single_candidate_rule}\n"
        f"Target paper ID: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract or ''}\n\n"
        f"Candidate references:\n{references}\n\n"
        f"Return JSON with target_paper_id exactly equal to {json.dumps(paper.paper_id)} "
        f"and at most {top_k} selected references:\n"
        '{"target_paper_id":"...",'
        '"selected_references":[{"reference_id":"...","rank":1,'
        '"confidence":0.0,"rationale":"..."}]}'
    )


def _format_reference_contexts(
    reference_id: str,
    contexts_by_reference: dict[str, list],
    max_contexts: int = 3,
) -> str:
    contexts = contexts_by_reference.get(reference_id, [])[:max_contexts]
    if not contexts:
        return ""
    lines = ["\n  Citation contexts:"]
    for context in contexts:
        section = context.section or "unknown"
        lines.append(f"\n    - section: {section}")
        lines.append(f"\n      sentence: {context.sentence}")
    return "".join(lines)


def build_repair_prompt(paper: Paper, raw_output: str, top_k: int = 5) -> str:
    candidate_ids = ", ".join(reference.reference_id or "" for reference in paper.references)
    single_candidate_rule = ""
    if len(paper.references) == 1:
        only_reference_id = paper.references[0].reference_id or ""
        single_candidate_rule = (
            " There is exactly one single candidate reference. You must select that candidate "
            f'with reference_id {json.dumps(only_reference_id)}.'
        )
    return (
        "Repair the model output into valid JSON only. Do not add commentary. "
        "Use only these reference_id values: "
        f"{candidate_ids}. Keep at most {top_k} selected references. "
        f"{single_candidate_rule}"
        "Return exactly this JSON shape: "
        '{"target_paper_id":"...",'
        '"selected_references":[{"reference_id":"...","rank":1,'
        '"confidence":0.0,"rationale":"..."}]}'
        "\n\nBroken output:\n"
        f"{raw_output}"
    )


def parse_judge_output(raw_output: str, paper: Paper) -> JudgeResult:
    candidate_ids = {reference.reference_id for reference in paper.references}
    try:
        payload = json.loads(_extract_json_object(raw_output))
    except json.JSONDecodeError as exc:
        return JudgeResult(
            target_paper_id=paper.paper_id,
            selected_references=[],
            raw_output=raw_output,
            parse_error=f"Invalid JSON: {exc.msg}",
        )

    selections: list[SelectedReference] = []
    for item in payload.get("selected_references", []):
        reference_id = str(item.get("reference_id", ""))
        selections.append(
            SelectedReference(
                reference_id=reference_id,
                rank=int(item.get("rank", len(selections) + 1)),
                confidence=float(item.get("confidence", 0.0)),
                rationale=str(item.get("rationale", "")),
                valid=reference_id in candidate_ids,
            )
        )

    return JudgeResult(
        target_paper_id=paper.paper_id,
        selected_references=selections,
        raw_output=raw_output,
    )


class HeuristicJudge:
    """Deterministic smoke-test judge; real runs should use a local LLM adapter."""

    def judge(self, paper: Paper, top_k: int = 5) -> JudgeResult:
        abstract_terms = _tokenize(paper.abstract or paper.title)
        scored: list[tuple[int, str]] = []
        for reference in paper.references:
            title_terms = _tokenize(reference.title)
            overlap = len(abstract_terms & title_terms)
            scored.append((overlap, reference.reference_id or ""))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [
            {
                "reference_id": reference_id,
                "rank": rank,
                "confidence": 1.0 if overlap > 0 else 0.1,
                "rationale": "Selected by deterministic lexical overlap baseline.",
            }
            for rank, (overlap, reference_id) in enumerate(scored[:top_k], start=1)
        ]
        result = parse_judge_output(
            json.dumps({"target_paper_id": paper.paper_id, "selected_references": selected}),
            paper,
        )
        return with_judge_audit_metadata(result, paper, top_k=top_k)


class OpenAICompatibleJudge:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
        repair_attempts: int = 0,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.client = client or httpx.Client()
        self.repair_attempts = repair_attempts
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    def judge(self, paper: Paper, top_k: int = 5) -> JudgeResult:
        prompt = build_judge_prompt(paper, top_k=top_k)
        raw = self._complete(prompt)
        result = parse_judge_output(raw, paper)
        attempts_left = self.repair_attempts
        while attempts_left > 0 and _needs_repair(result):
            raw = self._complete(build_repair_prompt(paper, result.raw_output, top_k=top_k))
            result = parse_judge_output(raw, paper)
            attempts_left -= 1
        return with_judge_audit_metadata(result, paper, top_k=top_k, prompt=prompt)

    def _complete(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload["choices"][0]["message"]["content"])


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.replace("-", " ").split() if len(token) > 3}


def _needs_repair(result: JudgeResult) -> bool:
    return result.parse_error is not None or result.invalid_selection_count > 0


def _extract_json_object(raw_output: str) -> str:
    stripped = raw_output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    starts = [index for index, char in enumerate(raw_output) if char == "{"]
    valid_candidates: list[tuple[bool, str]] = []
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw_output)):
            char = raw_output[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_output[start : index + 1]
                    payload = json.loads(candidate)
                    is_result_object = (
                        isinstance(payload, dict)
                        and (
                            "selected_references" in payload
                            or "target_paper_id" in payload
                        )
                    )
                    valid_candidates.append((is_result_object, candidate))
                    break
    for is_result_object, candidate in reversed(valid_candidates):
        if is_result_object:
            return candidate
    if valid_candidates:
        return valid_candidates[-1][1]
    return raw_output
