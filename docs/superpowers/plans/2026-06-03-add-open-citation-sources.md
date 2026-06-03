# Add Open Citation Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three additional no-key, auditable citation-evidence sources: Crossref, DataCite, and OpenAIRE.

**Architecture:** Implement one focused enrichment module per source, each reading local JSONL papers and merging source-specific evidence into `metadata.external_impact`. Add Typer CLI commands, wire the sources into `scripts/run_user_impact.sh`, expose raw source columns in score outputs, and keep final scoring conservative by selecting one trusted true-citation value rather than summing databases.

**Tech Stack:** Python, Typer, httpx, JSONL, pytest, existing enrichment/fusion patterns.

---

### Task 1: Crossref Enrichment

**Files:**
- Create: `src/local_impact_ranking/crossref_enrichment.py`
- Modify: `src/local_impact_ranking/cli.py`
- Test: `tests/test_crossref_enrichment.py`

- [x] Write tests for DOI lookup, title/count merge, missing DOI skip, and CLI output.
- [x] Run `PYTHONPATH=src pytest -q tests/test_crossref_enrichment.py` and confirm it fails because the module is missing.
- [x] Implement `enrich_papers_with_crossref(input_path, output_path, summary_output=None, timeout=30.0, client=None)`.
- [x] Add CLI command `enrich-crossref`.
- [x] Re-run `PYTHONPATH=src pytest -q tests/test_crossref_enrichment.py` and confirm it passes.

### Task 2: DataCite Enrichment

**Files:**
- Create: `src/local_impact_ranking/datacite_enrichment.py`
- Modify: `src/local_impact_ranking/cli.py`
- Test: `tests/test_datacite_enrichment.py`

- [x] Write tests for DOI lookup, `citationCount`/`viewCount`/`downloadCount` merge, 404 skip, and CLI output.
- [x] Run `PYTHONPATH=src pytest -q tests/test_datacite_enrichment.py` and confirm it fails because the module is missing.
- [x] Implement `enrich_papers_with_datacite(input_path, output_path, summary_output=None, timeout=30.0, client=None)`.
- [x] Add CLI command `enrich-datacite`.
- [x] Re-run `PYTHONPATH=src pytest -q tests/test_datacite_enrichment.py` and confirm it passes.

### Task 3: OpenAIRE Enrichment

**Files:**
- Create: `src/local_impact_ranking/openaire_enrichment.py`
- Modify: `src/local_impact_ranking/cli.py`
- Test: `tests/test_openaire_enrichment.py`

- [x] Write tests for DOI lookup, `indicators.citationImpact` merge, no-result skip, and CLI output.
- [x] Run `PYTHONPATH=src pytest -q tests/test_openaire_enrichment.py` and confirm it fails because the module is missing.
- [x] Implement `enrich_papers_with_openaire(input_path, output_path, summary_output=None, timeout=30.0, client=None)`.
- [x] Add CLI command `enrich-openaire`.
- [x] Re-run `PYTHONPATH=src pytest -q tests/test_openaire_enrichment.py` and confirm it passes.

### Task 4: Fusion And Report Columns

**Files:**
- Modify: `src/local_impact_ranking/impact_fusion.py`
- Modify: `src/local_impact_ranking/model_impact_index.py`
- Test: `tests/test_impact_fusion.py`, `tests/test_model_impact_index.py`

- [x] Write tests showing Crossref, DataCite, and OpenAIRE raw values appear in fused rows and can be selected as true-citation evidence.
- [x] Run the focused tests and confirm they fail before implementation.
- [x] Add source specs for `datacite` and `openaire`; keep existing Crossref support and expose the new raw columns.
- [x] Re-run focused tests and confirm they pass.

### Task 5: User Workflow And Documentation

**Files:**
- Modify: `scripts/run_user_impact.sh`
- Modify: `README.md`
- Modify: `docs/citation-data-sources.md`
- Modify: `docs/input-schema.md`
- Modify: `docs/scoring-method.md`
- Modify: `scripts/check_publish_ready.sh`
- Test: `tests/test_scripts.py`, `tests/test_docs.py`

- [x] Add `crossref`, `datacite`, and `openaire` to supported `AUTO_ENRICH_SOURCES`.
- [x] Update the default order to `openalex,europepmc,identity_recovery,europepmc,crossref,icite,ncbi_citations,opencitations,semantic_scholar,datacite,openaire`.
- [x] Update docs and publish-readiness checks for the new commands and fields.
- [x] Run `PYTHONPATH=src pytest -q tests/test_scripts.py tests/test_docs.py`.

### Task 6: Verification

**Files:**
- Modify: `progress.md`
- Optional output: `runs/data-100-verification/...`

- [x] Run focused enrichment/fusion/script/docs tests.
- [x] Run `PYTHONPATH=src pytest -q`.
- [x] Run `scripts/check_publish_ready.sh`.
- [x] Record the final verification result in `progress.md`.

## Final Result

- Added `enrich-crossref`, `enrich-datacite`, and `enrich-openaire`.
- Added auto-enrichment sources `crossref`, `datacite`, and `openaire`.
- Added true-citation raw fields `true_citation_datacite_raw` and `true_citation_openaire_raw`.
- Extended identity quarantine/title checks to DataCite and OpenAIRE fields.
- User-data replay: Crossref matched `84/85`, DataCite matched `1/85`, OpenAIRE matched `12/85`; final nonzero count stayed `3/85`.
- Verification: `PYTHONPATH=src pytest -q` reported `573 passed`; `scripts/check_publish_ready.sh` reported `573 passed` and `Publish readiness checks passed.`
