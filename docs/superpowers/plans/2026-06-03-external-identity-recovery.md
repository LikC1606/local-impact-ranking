# External Identity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover correct DOI/OpenAlex identity for records whose existing external IDs fail title-identity checks, so correct citation evidence is found instead of only blocking bad evidence.

**Architecture:** Add a conservative enrichment command that scans paper records for external-title mismatches, searches OpenAlex by the local title, accepts only high-similarity candidates, clears stale wrong-identity evidence, merges the recovered OpenAlex work, and records all rejected/recovered IDs for audit. The normal enrichment chain can then run downstream sources again from the recovered DOI/OpenAlex ID.

**Tech Stack:** Python, Typer CLI, httpx, JSONL, pytest, existing OpenAlex enrichment helpers.

---

### Task 1: Recovery Module

**Files:**
- Create: `src/local_impact_ranking/external_identity_recovery.py`
- Test: `tests/test_external_identity_recovery.py`

- [x] Add tests for recovering a mismatched paper from OpenAlex title search.
- [x] Add tests for preserving an unresolved mismatch without overwriting IDs.
- [x] Add tests for clearing stale DOI/PMID-derived citation evidence before merging the recovered work.
- [x] Add tests for removing stale OpenAlex-derived field labels from unresolved mismatches.
- [x] Implement `recover_external_identities(input_path, output_path, summary_output=None, min_title_similarity=0.92, timeout=30.0, client=None)`.

### Task 2: CLI And Workflow

**Files:**
- Modify: `src/local_impact_ranking/cli.py`
- Modify: `scripts/run_user_impact.sh`
- Test: `tests/test_external_identity_recovery.py`, `tests/test_scripts.py`

- [x] Add CLI command `recover-external-identities`.
- [x] Add `identity_recovery` to `AUTO_ENRICH_SOURCES`.
- [x] Set the default auto-enrichment order to run identity recovery after Europe PMC and before downstream citation sources.

### Task 3: User Data Replay

**Files:**
- Output: `runs/data-100-verification/extra-sources/...`

- [x] Run identity recovery on the user's 85-paper enriched input.
- [x] Re-run downstream enrichment from the recovered identity file.
- [x] Replay saved pairwise judgments without GPU.
- [x] Compare nonzero counts, mismatch counts, and top-ranked papers.
- [x] Run `PYTHONPATH=src pytest -q` and `scripts/check_publish_ready.sh`.

## Final Result

- Recovery input: `runs/data-100-verification/extra-sources/papers.europepmc.icite.ncbi.jsonl`.
- Recovery output: `runs/data-100-verification/extra-sources/papers.identity_recovered.jsonl`.
- Recovery summary: `mismatch_count=3`, `recovered_count=2`, `unresolved_count=1`.
- Replayed model-score output: `runs/data-100-verification/user-impact-model-p200-identity-recovery/model-impact`.
- Final scored identity state: 85 papers, 84 matched rows, 1 unresolved extract-only row (`42124699`).
- Final nonzero true-citation papers: `42125479=24`, `42032168=6`, and `42108843=1`.
- Two previous false high-score rows, `42030147` and `42169456`, are no longer allowed to use wrong DOI/PMID citation evidence and score as 0 after recovery.
- Verification: `PYTHONPATH=src pytest -q` reported `563 passed`; `scripts/check_publish_ready.sh` reported `563 passed` and `Publish readiness checks passed.`
