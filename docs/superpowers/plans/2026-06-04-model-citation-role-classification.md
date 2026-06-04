# Model Citation Role Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable model-backed classifier that assigns every existing citation context one canonical effective-citation role before scoring.

**Architecture:** Create a focused `citation_role_classifier` module for prompt construction, output validation, journal replay, and JSONL enrichment. Add a CLI command that calls the module. Update scoring so model-written `unknown` roles stay `unknown` instead of falling back to keyword inference.

**Tech Stack:** Python, Typer CLI, Pydantic models, httpx-compatible client interface, pytest.

---

## File Structure

- Create `src/local_impact_ranking/citation_role_classifier.py`
  - Owns allowed roles, prompt version, prompt building, model calls, JSON parsing, journal replay, and enriched JSONL writing.
- Modify `src/local_impact_ranking/cli.py`
  - Imports the classifier and exposes `classify-citation-contexts`.
- Modify `src/local_impact_ranking/impact_fusion.py`
  - Treats model-written `unknown` as a real role and prevents keyword fallback for model-classified contexts.
- Create `tests/test_citation_role_classifier.py`
  - Covers parsing, invalid outputs, model enrichment, source-role preservation, and journal replay.
- Modify `tests/test_impact_fusion.py`
  - Covers scoring behavior for model-written `unknown`.
- Modify `tests/test_cli.py`
  - Covers the new CLI wrapper.
- Modify `docs/input-schema.md`
  - Documents model role fields.
- Modify `docs/scoring-method.md`
  - Explains that model-classified roles are read from cached context fields.

---

### Task 1: Classifier Parsing And Prompt Unit

**Files:**
- Create: `src/local_impact_ranking/citation_role_classifier.py`
- Test: `tests/test_citation_role_classifier.py`

- [ ] **Step 1: Write failing tests for prompt and output parsing**

Add this file:

```python
import json

from local_impact_ranking.citation_role_classifier import (
    ALLOWED_CITATION_ROLES,
    build_citation_role_prompt,
    parse_citation_role_output,
)
from local_impact_ranking.models import CitationContext, Paper


def test_build_citation_role_prompt_contains_fixed_role_schema():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Citing paper",
            "abstract": "Uses a protocol.",
            "references": [{"reference_id": "r1", "title": "Protocol paper"}],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We used this protocol for the assay.",
                    "section": "Methods",
                }
            ],
        }
    )

    prompt = build_citation_role_prompt(
        paper,
        paper.citation_contexts[0],
        context_index=0,
    )

    assert "Return JSON only" in prompt
    for role in ALLOWED_CITATION_ROLES:
        assert role in prompt
    assert "We used this protocol" in prompt
    assert "Protocol paper" in prompt


def test_parse_citation_role_output_accepts_valid_role():
    parsed = parse_citation_role_output(
        json.dumps(
            {
                "role": "method",
                "confidence": 0.86,
                "rationale": "The sentence says a protocol was used.",
            }
        )
    )

    assert parsed["role"] == "method"
    assert parsed["confidence"] == 0.86
    assert parsed["validation_status"] == "valid"


def test_parse_citation_role_output_invalid_json_becomes_unknown():
    parsed = parse_citation_role_output("not json")

    assert parsed["role"] == "unknown"
    assert parsed["confidence"] == 0.0
    assert parsed["validation_status"] == "invalid_json"
    assert "raw output was not valid JSON" in parsed["rationale"]


def test_parse_citation_role_output_illegal_role_becomes_unknown():
    parsed = parse_citation_role_output(
        json.dumps({"role": "important", "confidence": 0.7, "rationale": "bad"})
    )

    assert parsed["role"] == "unknown"
    assert parsed["confidence"] == 0.0
    assert parsed["validation_status"] == "invalid_role"


def test_parse_citation_role_output_clamps_confidence():
    parsed = parse_citation_role_output(
        json.dumps({"role": "supporting", "confidence": 7, "rationale": "ok"})
    )

    assert parsed["confidence"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_citation_role_classifier.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'local_impact_ranking.citation_role_classifier'`.

- [ ] **Step 3: Implement parsing and prompt functions**

Create `src/local_impact_ranking/citation_role_classifier.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_citation_role_classifier.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_impact_ranking/citation_role_classifier.py tests/test_citation_role_classifier.py
git commit -m "Add citation role classifier prompt parsing"
```

---

### Task 2: Model Enrichment And Journal Replay

**Files:**
- Modify: `src/local_impact_ranking/citation_role_classifier.py`
- Modify: `tests/test_citation_role_classifier.py`

- [ ] **Step 1: Add failing tests for model calls, source hint preservation, empty contexts, and replay**

Append to `tests/test_citation_role_classifier.py`:

```python
from pathlib import Path

from local_impact_ranking.citation_role_classifier import classify_citation_contexts


class RoleClient:
    def __init__(self):
        self.payloads = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"role":"method","confidence":0.91,"rationale":"protocol reuse"}'
                            }
                        }
                    ]
                }

        return Response()


class FailingClient:
    def post(self, url, headers, json, timeout):
        raise AssertionError("replay should not call the model")


def write_jsonl(path: Path, records: list[dict]):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_classify_citation_contexts_writes_model_roles_and_preserves_source_hint(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.roles.jsonl"
    journal_path = tmp_path / "role_journal.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Citing paper",
                "references": [{"reference_id": "r1", "title": "Protocol paper"}],
                "citation_contexts": [
                    {
                        "reference_id": "r1",
                        "sentence": "We used this protocol.",
                        "section": "Methods",
                        "role": "background",
                    }
                ],
            }
        ],
    )
    client = RoleClient()

    report = classify_citation_contexts(
        input_path,
        output_path,
        base_url="http://localhost:8000/v1",
        model="local-impact-role",
        journal_path=journal_path,
        client=client,
    )

    rows = read_jsonl(output_path)
    context = rows[0]["citation_contexts"][0]
    assert report["classified_context_count"] == 1
    assert len(client.payloads) == 1
    assert context["role"] == "method"
    assert context["source_role_hint"] == "background"
    assert context["role_source"] == "model"
    assert context["role_model"] == "local-impact-role"
    assert context["role_validation_status"] == "valid"
    assert context["role_prompt_sha256"]
    assert journal_path.exists()


def test_classify_citation_contexts_empty_sentence_becomes_unknown_without_model_call(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.roles.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Citing paper",
                "references": [{"reference_id": "r1", "title": "Protocol paper"}],
                "citation_contexts": [{"reference_id": "r1", "sentence": ""}],
            }
        ],
    )

    report = classify_citation_contexts(
        input_path,
        output_path,
        base_url="http://localhost:8000/v1",
        model="local-impact-role",
        client=FailingClient(),
    )

    context = read_jsonl(output_path)[0]["citation_contexts"][0]
    assert report["model_request_count"] == 0
    assert context["role"] == "unknown"
    assert context["role_validation_status"] == "empty_sentence"


def test_classify_citation_contexts_replays_journal_without_model_call(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.roles.jsonl"
    journal_path = tmp_path / "role_journal.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Citing paper",
                "references": [{"reference_id": "r1", "title": "Protocol paper"}],
                "citation_contexts": [
                    {
                        "reference_id": "r1",
                        "sentence": "We used this protocol.",
                        "section": "Methods",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        journal_path,
        [
            {
                "paper_id": "p1",
                "context_index": 0,
                "reference_id": "r1",
                "sentence": "We used this protocol.",
                "role": "method",
                "confidence": 0.8,
                "rationale": "cached",
                "validation_status": "valid",
                "model": "cached-model",
                "prompt_sha256": "cached-hash",
                "raw_output": "{}",
            }
        ],
    )

    report = classify_citation_contexts(
        input_path,
        output_path,
        base_url="http://localhost:8000/v1",
        model="local-impact-role",
        journal_path=journal_path,
        client=FailingClient(),
    )

    context = read_jsonl(output_path)[0]["citation_contexts"][0]
    assert report["replayed_context_count"] == 1
    assert report["model_request_count"] == 0
    assert context["role"] == "method"
    assert context["role_model"] == "cached-model"
    assert context["role_prompt_sha256"] == "cached-hash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_citation_role_classifier.py -q`

Expected: FAIL with `ImportError: cannot import name 'classify_citation_contexts'`.

- [ ] **Step 3: Implement model enrichment and replay**

Add these imports to `citation_role_classifier.py`:

```python
from pathlib import Path

import httpx

from local_impact_ranking.io import load_papers_jsonl
```

Add these functions:

```python
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
    journal = _load_role_journal(journal_path)
    http_client = client or httpx.Client()
    owns_client = client is None
    classified_count = 0
    replayed_count = 0
    model_request_count = 0
    output_records: list[dict[str, Any]] = []
    try:
        for paper in papers:
            record = paper.model_dump(mode="json")
            contexts = list(record.get("citation_contexts") or [])
            for index, context_record in enumerate(contexts):
                key = _context_key(paper.paper_id, index, context_record)
                cached = journal.get(key)
                if cached is not None:
                    _apply_role_result(
                        context_record,
                        cached,
                        role_model=str(cached.get("model") or model),
                        prompt_hash=str(cached.get("prompt_sha256") or ""),
                        raw_output=str(cached.get("raw_output") or ""),
                    )
                    replayed_count += 1
                    classified_count += 1
                    continue
                context = paper.citation_contexts[index]
                if not context.sentence.strip():
                    result = {
                        "role": "unknown",
                        "confidence": 0.0,
                        "rationale": "The citation context sentence is empty.",
                        "validation_status": "empty_sentence",
                    }
                    _apply_role_result(
                        context_record,
                        result,
                        role_model=model,
                        prompt_hash="",
                        raw_output="",
                    )
                    classified_count += 1
                    continue
                prompt = build_citation_role_prompt(paper, context, context_index=index)
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
                model_request_count += 1
                parsed = parse_citation_role_output(raw_output)
                prompt_hash = prompt_sha256(prompt)
                _apply_role_result(
                    context_record,
                    parsed,
                    role_model=model,
                    prompt_hash=prompt_hash,
                    raw_output=raw_output,
                )
                _append_role_journal(
                    journal_path,
                    paper_id=paper.paper_id,
                    context_index=index,
                    context_record=context_record,
                )
                classified_count += 1
            record["citation_contexts"] = contexts
            output_records.append(record)
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()
    _write_jsonl(output_records, Path(output_path))
    return {
        "paper_count": len(papers),
        "classified_context_count": classified_count,
        "replayed_context_count": replayed_count,
        "model_request_count": model_request_count,
        "output_path": str(output_path),
        "journal_path": str(journal_path) if journal_path is not None else None,
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
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def _apply_role_result(
    context_record: dict[str, Any],
    result: dict[str, Any],
    *,
    role_model: str,
    prompt_hash: str,
    raw_output: str,
) -> None:
    source_hint = _source_role_hint_from_record(context_record)
    if source_hint is not None and "source_role_hint" not in context_record:
        context_record["source_role_hint"] = source_hint
    context_record["role"] = result["role"]
    context_record["role_confidence"] = result["confidence"]
    context_record["role_rationale"] = result["rationale"]
    context_record["role_source"] = "model"
    context_record["role_model"] = role_model
    context_record["role_prompt_sha256"] = prompt_hash
    context_record["role_raw_output"] = raw_output
    context_record["role_validation_status"] = result["validation_status"]


def _context_key(
    paper_id: str,
    context_index: int,
    context_record: dict[str, Any],
) -> tuple[str, int, str, str]:
    return (
        str(paper_id),
        int(context_index),
        str(context_record.get("reference_id") or ""),
        str(context_record.get("sentence") or ""),
    )


def _load_role_journal(
    journal_path: str | Path | None,
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    if journal_path is None:
        return {}
    path = Path(journal_path)
    if not path.exists():
        return {}
    rows: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            rows[
                (
                    str(row.get("paper_id") or ""),
                    int(row.get("context_index") or 0),
                    str(row.get("reference_id") or ""),
                    str(row.get("sentence") or ""),
                )
            ] = row
    return rows


def _append_role_journal(
    journal_path: str | Path | None,
    *,
    paper_id: str,
    context_index: int,
    context_record: dict[str, Any],
) -> None:
    if journal_path is None:
        return
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "paper_id": paper_id,
        "context_index": context_index,
        "reference_id": context_record.get("reference_id"),
        "sentence": context_record.get("sentence"),
        "role": context_record.get("role"),
        "confidence": context_record.get("role_confidence"),
        "rationale": context_record.get("role_rationale"),
        "validation_status": context_record.get("role_validation_status"),
        "model": context_record.get("role_model"),
        "prompt_sha256": context_record.get("role_prompt_sha256"),
        "raw_output": context_record.get("role_raw_output"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _source_role_hint_from_record(context_record: dict[str, Any]) -> str | None:
    for key in ("role", "label", "intent", "citation_intent", "category"):
        value = context_record.get(key)
        if value:
            return str(value)
    return None


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_citation_role_classifier.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_impact_ranking/citation_role_classifier.py tests/test_citation_role_classifier.py
git commit -m "Add model citation role enrichment"
```

---

### Task 3: Scoring Integration For Model Unknown

**Files:**
- Modify: `src/local_impact_ranking/impact_fusion.py`
- Modify: `tests/test_impact_fusion.py`

- [ ] **Step 1: Add failing scoring test**

Append to `tests/test_impact_fusion.py`:

```python
def test_model_unknown_role_does_not_fall_back_to_keyword_rules():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "cited",
                "title": "Cited method paper",
                "citation_count": 0,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "citing",
                "title": "Citing paper",
                "citation_count": 0,
                "references": [{"reference_id": "cited", "title": "Cited method paper"}],
                "citation_contexts": [
                    {
                        "reference_id": "cited",
                        "sentence": "We used this method as the assay protocol.",
                        "section": "Methods",
                        "role": "unknown",
                        "role_source": "model",
                        "role_validation_status": "valid",
                    }
                ],
            }
        ),
    ]

    result = compute_impact_index_v2(papers, [])
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["cited"]["effective_citation_method_count"] == 0
    assert by_id["cited"]["effective_citation_unknown_count"] == 1
    assert by_id["cited"]["effective_citation_raw"] == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_impact_fusion.py::test_model_unknown_role_does_not_fall_back_to_keyword_rules -q`

Expected: FAIL because `effective_citation_method_count` is `1`.

- [ ] **Step 3: Update explicit role mapping**

Modify `_explicit_context_role` in `src/local_impact_ranking/impact_fusion.py` so `unknown` is accepted and model-invalid values do not fall through to keywords:

```python
def _explicit_context_role(context: CitationContext) -> str | None:
    extra = getattr(context, "__pydantic_extra__", None) or {}
    value = None
    for key in ("role", "label", "intent", "citation_intent", "category"):
        if extra.get(key):
            value = str(extra[key]).strip().lower()
            break
    if not value:
        return None
    if value in {"unknown", "uncertain", "unclassified"}:
        return "unknown"
    if value in {"method", "methods", "uses", "extends", "reuse"}:
        return "method"
    if value in {"result", "results", "supporting", "support"}:
        return "supporting"
    if value in {"background", "literature", "review"}:
        return "background"
    if value in {"clinical", "clinical_trial", "trial", "translational"}:
        return "clinical"
    if value in {"guideline", "policy", "recommendation"}:
        return "guideline"
    if value in {"dataset", "data", "tool", "software", "resource"}:
        return "dataset_or_tool"
    if value in {"negative", "contrasting", "disputing", "refuting"}:
        return "negative_or_contrasting"
    if str(extra.get("role_source") or "").strip().lower() == "model":
        return "unknown"
    return None
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
pytest tests/test_impact_fusion.py::test_model_unknown_role_does_not_fall_back_to_keyword_rules \
  tests/test_impact_fusion.py::test_final_impact_score_uses_only_true_and_effective_citations -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_impact_ranking/impact_fusion.py tests/test_impact_fusion.py
git commit -m "Use cached model citation roles in scoring"
```

---

### Task 4: CLI Command

**Files:**
- Modify: `src/local_impact_ranking/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI test**

Append to `tests/test_cli.py`:

```python
def test_cli_classify_citation_contexts_writes_report(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.roles.jsonl"
    journal_path = tmp_path / "role_journal.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "references": [{"reference_id": "r1", "title": "Ref"}],
                "citation_contexts": [
                    {"reference_id": "r1", "sentence": "We used the protocol."}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_classify(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["base_url"] == "http://localhost:8000/v1"
        assert kwargs["model"] == "local-role"
        assert kwargs["journal_path"] == journal_path
        output_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "paper_count": 1,
            "classified_context_count": 1,
            "replayed_context_count": 0,
            "model_request_count": 1,
            "output_path": str(output_path),
            "journal_path": str(journal_path),
        }

    monkeypatch.setattr(cli, "classify_citation_contexts", fake_classify)

    result = CliRunner().invoke(
        app,
        [
            "classify-citation-contexts",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-role",
            "--journal",
            str(journal_path),
        ],
    )

    assert result.exit_code == 0
    assert "Classified citation contexts: 1" in result.stdout
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_classify_citation_contexts_writes_report -q`

Expected: FAIL because `classify-citation-contexts` is not registered.

- [ ] **Step 3: Add CLI import and command**

Add import near other local imports in `src/local_impact_ranking/cli.py`:

```python
from local_impact_ranking.citation_role_classifier import classify_citation_contexts
```

Add command near other enrichment-style commands:

```python
@app.command("classify-citation-contexts")
def classify_citation_contexts_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--base-url"),
    model: str = typer.Option(..., "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    timeout: float = typer.Option(120.0, "--timeout"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int | None = typer.Option(256, "--max-tokens", min=1),
    journal: Path | None = typer.Option(None, "--journal"),
) -> None:
    report = classify_citation_contexts(
        input,
        output,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        journal_path=journal,
    )
    typer.echo(
        "Classified citation contexts: "
        f"{report['classified_context_count']} "
        f"(model_requests={report['model_request_count']}, "
        f"replayed={report['replayed_context_count']}) -> {output}"
    )
```

- [ ] **Step 4: Run CLI test**

Run: `pytest tests/test_cli.py::test_cli_classify_citation_contexts_writes_report -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_impact_ranking/cli.py tests/test_cli.py
git commit -m "Add citation context role classification CLI"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/input-schema.md`
- Modify: `docs/scoring-method.md`

- [ ] **Step 1: Update input schema documentation**

In `docs/input-schema.md`, after the citation-context example, add:

````markdown
When using model-classified citation roles, each `citation_contexts[]` item should include the cached model decision:

```json
{
  "reference_id": "cited_paper",
  "sentence": "We used this method as the assay protocol.",
  "section": "Methods",
  "role": "method",
  "role_confidence": 0.86,
  "role_source": "model",
  "role_model": "local-impact-citation-role-classifier",
  "role_prompt_sha256": "..."
}
```

The scoring step reads the saved `role`. It does not call the model during score fusion.
````

- [ ] **Step 2: Update scoring method documentation**

In `docs/scoring-method.md`, near the effective citation role section, add:

```markdown
In the model-classified workflow, citation roles are assigned before scoring by `classify-citation-contexts`. The model receives the fixed role rubric and writes a cached `role` plus audit fields onto each `citation_contexts[]` item. `model-score-dataset` and `impact_fusion` then read the saved role. They do not ask the model again during scoring.

If a model-classified context has `role=unknown`, the scorer keeps it as `unknown` even when the sentence contains keywords that older heuristic rules would have matched.
```

- [ ] **Step 3: Run docs-related tests**

Run: `pytest tests/test_docs.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/input-schema.md docs/scoring-method.md
git commit -m "Document model-classified citation roles"
```

---

### Task 6: Final Verification

**Files:**
- No source edits unless a test failure exposes a defect in the files modified above.

- [ ] **Step 1: Run targeted feature tests**

Run:

```bash
pytest tests/test_citation_role_classifier.py tests/test_impact_fusion.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 3: Inspect git status**

Run: `git status --short`

Expected: no output.

- [ ] **Step 4: Report outcome**

Summarize:

- new command name.
- scoring behavior for model `unknown`.
- tests run.
- any tests not run.

---

## Self-Review

- Spec coverage: Task 1 covers role schema and strict parsing. Task 2 covers model classification, cached outputs, audit fields, empty contexts, and replay. Task 3 covers scoring integration and keyword fallback prevention. Task 4 exposes CLI. Task 5 documents input and scoring behavior. Task 6 verifies the change.
- Red-flag scan: no incomplete markers are present.
- Type consistency: the plan consistently uses `role`, `role_confidence`, `role_rationale`, `role_source`, `role_model`, `role_prompt_sha256`, `role_raw_output`, and `role_validation_status`.
