# Progress Log

## 2026-05-25

- Created persistent planning files.
- Selected next completion increment: field/year rankings and Markdown evaluation report.
- No GPU-serving process started.
- Implemented field/year ranking exports and `reports/evaluation.md`.
- Updated artifact index, scripts, README, and tests.
- Verification: targeted tests passed, full test suite reported `77 passed`, and local smoke completed at `/tmp/local-impact-ranking-complete-smoke-1`.
- Implemented `convert-csv` to convert tabular paper data into canonical JSONL before validation/ranking.
- Verification: `PYTHONPATH=src pytest -q tests/test_io_models.py tests/test_cli.py -q` passed.
- Added `scripts/run_benchmark_smoke.sh` and README instructions for a no-private-data SciCite smoke run.
- Verification: `PYTHONPATH=src pytest -q tests/test_scripts.py -q` passed and benchmark smoke completed at `/tmp/local-impact-ranking-benchmark-smoke`.
- Added `score` command as a no-model replay stage for existing judge audit files.
- Verification: `PYTHONPATH=src pytest -q tests/test_cli.py -q` passed.
- Updated `docs/planning/project-summary-report.md` with the current report-ready project summary.
- Final verification: `PYTHONPATH=src pytest -q` reported `82 passed`.
- Final smoke checks completed:
  - `scripts/run_local_smoke.sh /tmp/local-impact-ranking-final-local-smoke`
  - `scripts/run_benchmark_smoke.sh /tmp/local-impact-ranking-final-benchmark-smoke`
- Artifact index checks reported `missing_count = 0` for both final smoke outputs.
- Added PreScience JSONL/Parquet adapter. It builds candidates from key references plus author publication history and preserves gold positives in metadata.
- Added gold reference evaluation reports for benchmark datasets with positive reference IDs.

## 2026-05-26

- Added target-level gold comparison for same-input benchmark runs via `compare-gold-targets`.
- Generated `runs/benchmark-comparison-base-vs-dpo-same100/target_comparison.json`, `target_changes.csv`, and `target_comparison.md`.
- Same100 target-level result: DPO pilot fixed 2 base misses, broke 1 base hit, worsened rank for 2 still-hit targets, changed hit quality for 2 multi-positive targets, and left 93 targets in the same hit/miss category.
- Verification: `PYTHONNOUSERSITE=1 PYTHONPATH=src /root/miniconda3/envs/air/bin/python3.11 -m pytest -q` reported `121 passed`.
- Fixed `judge --limit` resume semantics so benchmark limits apply to the input prefix before skipping completed audit records.
- Ran strict same300 comparison:
  - Base: `runs/prescience-base-same300`
  - DPO: `runs/prescience-dpo-pilot-300`
  - Comparison: `runs/benchmark-comparison-base-vs-dpo-same300`
- Same300 result: DPO hit_rate `0.753333` vs base `0.746667`; mean_ndcg `0.583968` vs `0.576676`; target-level net hit change `+2`.
- Verification: strict audit check reported 300 unique records, no outside-first300 records, no missing-first300 records.
- Verification: `PYTHONNOUSERSITE=1 PYTHONPATH=src /root/miniconda3/envs/air/bin/python3.11 -m pytest -q` reported `122 passed`.
