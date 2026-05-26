# Local Impact Ranking Completion Plan

## Goal

Finish the local paper-impact ranking MVP so it can run on a user paper dataset, produce dataset-local paper/author/institution/field/year influence rankings, and emit auditable reports for validation, judging, scoring, proxy quality, anomalies, stability, and reproducibility.

## Current Baseline

- CLI supports `validate`, `judge`, `run`, and `benchmarks`.
- `run` writes rankings, audit logs, baselines, validation reports, proxy reports, anomaly/stability reports, explanations, fine-tuning data, LLaMAFactory configs, artifact index, and manifest.
- Full tests passed before this plan file was created: `76 passed`.
- Local smoke passed without starting a model server or using GPU.

## Remaining Completion Phases

- [x] Phase 1: Fill reporting-output gaps from the design docs.
  - Add field and year ranking exports.
  - Add a Markdown evaluation report alongside `evaluation.json`.
  - Include new artifacts in artifact index, scripts, README, and tests.

- [x] Phase 2: Strengthen data-ingestion readiness.
  - Review whether JSONL-only input is sufficient for the user's expected data handoff.
  - If needed, add a simple CSV-to-JSONL converter rather than expanding all commands to mixed formats.

- [x] Phase 3: Strengthen benchmark path.
  - Ensure SciCite conversion and benchmark registry are documented and smoke-tested enough to demo without private data.
  - Add a tiny fixture-based benchmark run if missing.

- [x] Phase 4: Final project audit.
  - Run full tests.
  - Run local smoke.
  - Check CLI help.
  - Update project summary with final capability boundary and commands.
  - Add an explicit no-model `score` command for replaying judge audit files.

## Decisions

- Keep the project dataset-local. Do not claim global impact without external graph calibration.
- Keep human review optional. Reports must distinguish proxy quality from human-validated correctness.
- Do not start GPU model serving unless explicitly requested.

## Errors Encountered

| Date | Error | Resolution |
|---|---|---|
| 2026-05-25 | Root `task_plan.md`, `findings.md`, and `progress.md` were absent. | Created persistent planning files in the project root. |
