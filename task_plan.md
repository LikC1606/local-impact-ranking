# Local Impact Ranking Full Optimization Plan

## Goal

Turn the current local paper-impact ranking MVP into a complete, defensible project deliverable: analyze current weaknesses, ground the design in public methods and benchmarks, implement the missing engineering pieces, verify with tests and smoke runs, and leave clear report-ready documentation.

## Current Baseline

- The project already runs a dataset-local, content-aware impact ranking pipeline.
- It exports rankings, score evidence, baselines, stability/uncertainty reports, validation reports, model-comparison reports, fine-tuning datasets, LLaMAFactory configs, DPO loader validation, and no-GPU fine-tuning preflight reports.
- Current verified state before this full optimization pass: full test suite passed with `170 passed`; local smoke passed at `/tmp/local-impact-local-smoke-preflight-final`.
- Worktree contains many intentional project changes from prior implementation phases. Do not revert unrelated changes.

## Optimization Phases

- [x] Phase 1: Audit current project capabilities and remaining shortcomings.
  - Compare implementation against the project goal: local score, ranking, confidence, benchmark, model improvement, and reportability.
  - Identify concrete gaps that can still weaken credibility or usability.

- [x] Phase 2: Research evidence base.
  - Search current and method-defining papers for scientific impact prediction, influential citation classification, citation-context modeling, bibliometric baselines, graph ranking, uncertainty/robustness, and benchmark datasets.
  - Search GitHub for reusable benchmark/code assets.
  - Store decision-relevant findings in `findings.md`.

- [x] Phase 3: Produce an optimization design.
  - Write a concise design doc covering scoring, evaluation, training, benchmarks, confidence, CLI workflow, and final deliverables.
  - Decide what is required for the finished project versus deferred research work.

- [x] Phase 4: Implement the selected product gaps.
  - Prioritize features that improve reproducibility, credibility, and ease of use without requiring private data or GPU.
  - Keep tests close to each behavioral change.

- [x] Phase 5: Final verification and handoff.
  - Run targeted tests, full tests, and smoke scripts.
  - Produce a final project summary with commands, artifacts, limitations, and next optional GPU/model steps.

## Design Constraints

- The valid claim remains dataset-local impact inside the user-provided corpus unless an external calibrated citation graph is added.
- Human labels are not available in the short term, so benchmark/proxy evidence must be labeled as such.
- Do not start GPU-serving or training jobs unless explicitly requested.
- Prefer no-GPU checks and benchmark smokes for final project completeness.
- Keep implementation auditable: every score should trace to selected citation edges, prompt/audit records, and scoring configuration.

## Errors Encountered

| Date | Error | Attempt | Resolution |
|---|---|---|---|
| 2026-05-27 | Existing planning files described an older completion phase. | Replaced `task_plan.md` with this full optimization plan. | Continue using `findings.md` and `progress.md` as persistent logs. |
| 2026-05-27 | New readiness tests failed at collection because `project_readiness.py` did not exist yet. | Ran target tests before implementation. | Implemented the module, CLI command, auto-run integration, and artifact index entries; target tests passed. |
