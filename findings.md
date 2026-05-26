# Findings

## Project State

- Existing design docs already define the intended MVP scope: dataset-local content-aware impact ranking, not global scientific impact.
- Current implementation is stronger than the original MVP plan in several areas: validation reports, prompt/output audit, proxy evaluation, anomaly report, stability report, explanations, fine-tuning exports, and manifest.
- The previous mismatch with missing `rankings/fields.csv`, missing `rankings/years.csv`, and missing `reports/evaluation.md` has been resolved.
- Field and year rankings are aggregated from selected referenced-paper metadata. Missing field/year metadata is left absent rather than imputed.
- The main pipeline remains JSONL-only internally. CSV support is provided as a `convert-csv` preprocessing command with JSON columns for nested authors, references, citation contexts, and metadata.
- No-private-data demo path is now explicit: `scripts/run_benchmark_smoke.sh` runs a 5-paper SciCite prefix through the full no-GPU heuristic pipeline.
- `score` is now an explicit no-model stage that replays `audit/judge_outputs.jsonl` into the same complete outputs as `run --judge-audit`.
- Final artifact index checks on local context smoke and benchmark smoke both reported `missing_count = 0` across 35 key artifacts.

## Verification History

- `PYTHONPATH=src pytest -q` reported `76 passed`.
- `scripts/run_local_smoke.sh /tmp/local-impact-ranking-prompt-audit-smoke` completed and generated `audit/prompts.jsonl` for both tiny and context examples.
- After adding field/year rankings and `evaluation.md`, `PYTHONPATH=src pytest -q` reported `77 passed`.
- `scripts/run_local_smoke.sh /tmp/local-impact-ranking-complete-smoke-1` completed.
- Targeted CSV ingestion tests passed with `PYTHONPATH=src pytest -q tests/test_io_models.py tests/test_cli.py -q`.
- `scripts/run_benchmark_smoke.sh /tmp/local-impact-ranking-benchmark-smoke` completed.
- Final verification: `PYTHONPATH=src pytest -q` reported `82 passed`.
- Final local smoke completed at `/tmp/local-impact-ranking-final-local-smoke`.
- Final benchmark smoke completed at `/tmp/local-impact-ranking-final-benchmark-smoke`.

## Constraints

- The user does not currently have private data.
- No human annotation is available in the short term.
- GPU/model server should stay off unless explicitly requested.

## Latest Model Comparison Finding

- The 20-step DPO pilot has a small positive same-input signal on the 100-paper PreScience slice, but the improvement is weak.
- Target-level comparison shows the net hit gain is +1: fixed targets `2883238` and `3265742`, broken target `3264213`.
- Most outputs did not change materially: mean selection Jaccard is `0.928`, with 69 unchanged hits and 24 unchanged misses.
- Two still-hit targets worsened in gold rank (`14636714`, `14930700`), while two multi-positive targets changed hit quality without changing hit status (`1695826`, `376086`).
- This supports continuing error analysis or a modest 300-sample resumable evaluation, but it is not enough to claim the DPO model is definitively better.
- The strict same300 evaluation preserved the same direction but remained weak: DPO fixed 3 base misses, broke 1 base hit, and improved hit_rate by `+0.006667`.
- Same300 mean_ndcg improved by `+0.007293`, mean reciprocal rank by `+0.004389`, and mean selection Jaccard stayed high at `0.927`, meaning most decisions were unchanged.
- The DPO pilot should still be treated as an early positive signal, not a production-quality tuned judge. A stronger next training run should use more hard negatives and evaluate on a held-out slice.
