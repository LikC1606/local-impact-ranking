# Local Impact Ranking MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal local ImpactRankings-style pipeline that can load a JSONL paper dataset, validate and normalize it, parse local judge outputs, score selected references, and export rankings with audit records.

**Architecture:** Implement a Python package with small modules for schema models, IO, normalization, judge parsing, scoring, evaluation, and CLI orchestration. The first version includes a deterministic heuristic judge for local smoke tests and an OpenAI-compatible client adapter for local vLLM/SGLang servers.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, pytest, httpx.

---

### Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/local_impact_ranking/__init__.py`
- Create: `tests/test_import.py`

- [ ] Add package metadata and test dependencies.
- [ ] Add a README with the MVP command shape.
- [ ] Add a smoke import test.
- [ ] Run `pytest tests/test_import.py -v`.

### Task 2: Schemas and JSONL Loading

**Files:**
- Create: `src/local_impact_ranking/models.py`
- Create: `src/local_impact_ranking/io.py`
- Create: `tests/test_io_models.py`

- [ ] Test that valid JSONL records load into typed paper objects.
- [ ] Test that missing `paper_id`, `title`, or `references` fails validation.
- [ ] Implement Pydantic models for `Author`, `Reference`, and `Paper`.
- [ ] Implement JSONL loader.
- [ ] Run `pytest tests/test_io_models.py -v`.

### Task 3: Normalization

**Files:**
- Create: `src/local_impact_ranking/normalize.py`
- Create: `tests/test_normalize.py`

- [ ] Test text whitespace normalization and stable generated reference IDs.
- [ ] Implement normalization helpers and record normalization.
- [ ] Run `pytest tests/test_normalize.py -v`.

### Task 4: Judge Output Parsing

**Files:**
- Create: `src/local_impact_ranking/judge.py`
- Create: `tests/test_judge.py`

- [ ] Test strict JSON parsing of valid selected references.
- [ ] Test that selected reference IDs not present in candidates are marked invalid.
- [ ] Implement prompt builder, parser, heuristic judge, and OpenAI-compatible client adapter.
- [ ] Run `pytest tests/test_judge.py -v`.

### Task 5: Scoring

**Files:**
- Create: `src/local_impact_ranking/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] Test rank-weighted credit split across paper, author, and institution.
- [ ] Test flat credit mode.
- [ ] Implement scoring aggregation.
- [ ] Run `pytest tests/test_scoring.py -v`.

### Task 6: Evaluation and Export

**Files:**
- Create: `src/local_impact_ranking/evaluate.py`
- Create: `src/local_impact_ranking/export.py`
- Create: `tests/test_evaluate_export.py`

- [ ] Test evaluation metrics from parsed judge records.
- [ ] Test CSV exports for paper, author, and institution rankings.
- [ ] Implement evaluation and export helpers.
- [ ] Run `pytest tests/test_evaluate_export.py -v`.

### Task 7: CLI Pipeline

**Files:**
- Create: `src/local_impact_ranking/cli.py`
- Create: `tests/test_cli.py`

- [ ] Test a full smoke run on a tiny JSONL dataset using the heuristic judge.
- [ ] Implement `validate`, `judge`, `score`, and `run` commands.
- [ ] Run `pytest tests/test_cli.py -v`.

### Task 8: Verification

**Files:**
- Modify: `README.md`

- [ ] Add end-to-end usage instructions.
- [ ] Run `pytest -v`.
- [ ] Run `python -m local_impact_ranking.cli --help`.
- [ ] Run the smoke pipeline on a fixture dataset.

