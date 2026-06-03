# Extra Citation Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add free, credentialless biomedical citation sources and prevent false-zero citation counts caused by invalid PMID inference.

**Architecture:** Add Europe PMC enrichment for DOI-to-PMID/PMCID resolution and cited-by counts, add NCBI ELink cited-in enrichment for resolved PubMed IDs, and extend the true-citation resolver to audit both sources. Keep existing max-source scoring policy and source-specific raw outputs.

**Tech Stack:** Python, Typer CLI, httpx clients, pytest, JSONL paper records.

---

### Task 1: Stop Treating Local Numeric Paper IDs As PMIDs

**Files:**
- Modify: `src/local_impact_ranking/icite_benchmark.py`
- Test: `tests/test_icite_benchmark.py`

- [x] Add a test showing `_paper_pmid` ignores numeric `paper_id` when no explicit PMID field exists.
- [x] Run `PYTHONPATH=src pytest -q tests/test_icite_benchmark.py::test_paper_pmid_does_not_infer_local_numeric_paper_id` and confirm it fails.
- [x] Remove the numeric `paper.paper_id` fallback from `_candidate_pmid_values`.
- [x] Re-run the focused test and `PYTHONPATH=src pytest -q tests/test_icite_benchmark.py`.

### Task 2: Add Europe PMC Enrichment

**Files:**
- Create: `src/local_impact_ranking/europepmc_enrichment.py`
- Modify: `src/local_impact_ranking/cli.py`
- Test: `tests/test_europepmc_enrichment.py`

- [x] Add tests for DOI match merging `europepmc_cited_by_count`, PMID, and PMCID.
- [x] Add tests for no-match records preserving the original record and reporting unmatched status.
- [x] Implement `enrich_papers_with_europepmc(input_path, output_path, summary_output=None, timeout=30.0, client=None)`.
- [x] Add CLI command `enrich-europepmc`.
- [x] Run focused tests.

### Task 3: Add NCBI ELink Enrichment

**Files:**
- Create: `src/local_impact_ranking/ncbi_citation_enrichment.py`
- Modify: `src/local_impact_ranking/cli.py`
- Test: `tests/test_ncbi_citation_enrichment.py`

- [x] Add tests for `pubmed_pubmed_citedin` response parsing into `ncbi_pubmed_citedin_count`.
- [x] Add tests for missing PMID preserving the record and reporting skipped status.
- [x] Implement `enrich_papers_with_ncbi_citations(input_path, output_path, summary_output=None, timeout=30.0, client=None)`.
- [x] Add CLI command `enrich-ncbi-citations`.
- [x] Run focused tests.
- [x] Add a regression test so PMC cited-by counts are audit-only and cannot overwrite generic `citation_count`.

### Task 4: Extend True-Citation Resolver

**Files:**
- Modify: `src/local_impact_ranking/impact_fusion.py`
- Modify: `src/local_impact_ranking/model_impact_index.py`
- Test: `tests/test_impact_fusion.py`, `tests/test_model_impact_index.py`

- [x] Add failing resolver tests for Europe PMC and NCBI PubMed counts beating OpenAlex zero.
- [x] Add CSV schema assertions for new source raw fields.
- [x] Add `europepmc` and `ncbi_pubmed` to source collection and CSV output.
- [x] Keep `ncbi_pmc_cited_by_count` as audit-only because the user-data run exposed an implausible PMC-only count.
- [x] Run focused tests.

### Task 5: Wire User Auto-Enrichment And Re-run User Data

**Files:**
- Modify: `scripts/run_user_impact.sh`
- Modify: docs and publish readiness checks if needed.
- Output: `runs/data-100-verification/...`

- [x] Add `europepmc` and `ncbi_citations` to supported `AUTO_ENRICH_SOURCES`.
- [x] Run source-specific enrichment on `runs/data-100-verification/papers.enriched.multi_source.jsonl`.
- [x] Replay saved 200 pairwise judgments without GPU.
- [x] Compare nonzero citation count before/after.
- [x] Run `PYTHONPATH=src pytest -q` and `scripts/check_publish_ready.sh`.

### Task 6: Manual Spot-Check And Identity Guard

**Files:**
- Modify: `src/local_impact_ranking/impact_fusion.py`
- Modify: `src/local_impact_ranking/model_impact_index.py`
- Test: `tests/test_impact_fusion.py`, `tests/test_model_impact_index.py`

- [x] Spot-check high, conflicting, and zero-citation papers against public APIs.
- [x] Fix NCBI ELink per-ID mapping by using repeated `id` parameters and chunked requests.
- [x] Add title-identity guarding so wrong-paper DOI/PMID citation counts are audited but excluded from final true-citation scoring.
- [x] Expose `true_citation_identity_*` fields in JSON/CSV outputs.

## User-Data Result

- Europe PMC matched 85/85 curated user papers and corrected false PMID evidence for records whose `paper_id` had been mistaken for a PMID.
- iCite after Europe PMC recovered nonzero life-science citation evidence for 3 papers.
- NCBI PubMed cited-in has 2 nonzero corroborating records after fixing ELink batch mapping; NCBI PMC cited-by is retained only as audit evidence.
- Initial extra-source replay output: `runs/data-100-verification/user-impact-model-p200-extra-sources-replay/model-impact`.
- Final identity-recovered replay output: `runs/data-100-verification/user-impact-model-p200-identity-recovery/model-impact`.
- Final guarded nonzero true-citation papers are 3/85. Identity recovery resolved 2 wrong-paper DOI/PMID matches back to trusted 0-score rows (`42030147`, `42169456`) and left 1 unresolved extract-only row (`42124699`).
- The main improvement is credibility and false-zero/false-high/wrong-paper prevention, not a large increase in nonzero coverage.
- Verification passed: `PYTHONPATH=src pytest -q` -> `563 passed`; `scripts/check_publish_ready.sh` -> `563 passed`, `Publish readiness checks passed.`
