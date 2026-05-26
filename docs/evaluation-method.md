# Evaluation Method

The project separates three questions:

1. Did the judge produce valid constrained output?
2. Did the judge select benchmark-positive references when gold labels exist?
3. Did a prompt/model/training change improve or degrade target-level behavior?

These reports evaluate the ranking pipeline. They are not global scientific impact labels.

## Judge Reliability

Files:

- `reports/evaluation.json`
- `reports/evaluation.md`

Metrics:

- `parse_success_rate`: fraction of judge outputs parsed as JSON.
- `error_rate`: fraction of target papers that ended in per-paper judge errors.
- `invalid_selection_rate`: fraction of selected references whose `reference_id` was not in the target paper's candidate references.
- `average_selected_references`: average selected reference count per target.
- `empty_selection_rate`: fraction of target papers with no selected references.
- `duplicate_selection_rate`: fraction of target papers with duplicate selected reference IDs.
- `top_k_compliance_rate`: fraction of outputs with no more than the configured `top_k` selections.

These metrics measure output hygiene and constrained-selection reliability. They do not measure semantic correctness.

## Gold Reference Evaluation

Files:

- `reports/gold_evaluation.json`
- `reports/gold_evaluation.md`

This report is available when target papers include benchmark-positive references in:

- `metadata.prescience_positive_reference_ids`
- `metadata.gold_reference_ids`

Metrics:

- `labeled_targets`: target papers with at least one gold positive reference.
- `selected_references`: valid selected references among labeled targets.
- `positive_references`: total gold positive references.
- `selected_positive_references`: selected references that are also gold positives.
- `hit_rate`: fraction of labeled targets where at least one selected reference is gold-positive.
- `selection_precision`: selected-positive references divided by selected references.
- `selection_recall`: selected-positive references divided by positive references.
- `mean_reciprocal_rank`: average reciprocal rank of the first selected positive reference.
- `mean_ndcg`: average NDCG over selected references, treating benchmark positives as relevant.

Interpretation:

- `hit_rate` is useful for "did the model find at least one known important reference?"
- `selection_precision` penalizes extra non-gold selections.
- `selection_recall` rewards recovering multiple positives.
- `mean_reciprocal_rank` rewards putting the first positive high.
- `mean_ndcg` rewards ranking all positives high.

Gold evaluation is benchmark-specific reference-selection evaluation, not global scientific impact.

## Gold Error Analysis

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli analyze-gold-errors \
  --gold-evaluation runs/example/reports/gold_evaluation.json \
  --judge-audit runs/example/audit/judge_outputs.jsonl \
  --output runs/example/error-analysis
```

Files:

- `gold_error_analysis.json`
- `gold_error_analysis.md`

Metrics:

- `hit@1`, `hit@3`, `hit@5`: hit rate when considering only the first K selected references.
- `first_positive_rank_buckets`: counts for rank 1, ranks 2-3, ranks 4-5, and misses.
- `gold_candidate_recall`: fraction of gold references that were present in the candidate list.
- `miss_gold_absent_from_candidates`: misses caused by candidate construction.
- `miss_gold_present_but_not_selected`: misses caused by judge ranking/selection.
- `gold_candidate_positions`: where gold references appeared in candidate order.

This separates candidate-coverage failures from model-selection failures. If `gold_candidate_recall` is low, improving the judge will not fix all misses.

## Benchmark Comparison

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli compare-benchmarks \
  runs/base \
  runs/candidate \
  --output runs/comparison
```

Files:

- `benchmark_comparison.csv`
- `benchmark_comparison.md`

The comparison flattens each run's benchmark summary into a table with judge reliability, gold quality, and report-readiness fields.

## Target-Level Gold Comparison

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli compare-gold-targets \
  runs/base \
  runs/candidate \
  --output runs/comparison
```

Files:

- `target_comparison.json`
- `target_comparison.md`
- `target_changes.csv`

Key fields:

- `fixed_count`: base missed the gold positive, candidate hit it.
- `broken_count`: base hit the gold positive, candidate missed it.
- `net_hit_change`: `fixed_count - broken_count`.
- `rank_improved_count`: both hit, candidate placed the first positive higher.
- `rank_worsened_count`: both hit, candidate placed the first positive lower.
- `hit_quality_changed_count`: both hit, but NDCG or selected-positive count changed.
- `mean_selection_jaccard`: average overlap between base and candidate selected reference sets.
- `mean_reciprocal_rank_delta`: candidate MRR minus base MRR at target level.
- `mean_ndcg_delta`: candidate NDCG minus base NDCG at target level.

Target-level comparison is the preferred way to inspect whether a prompt or fine-tuned model changed behavior in useful ways or merely shuffled mostly unchanged outputs.

## Reporting Limits

- Strong benchmark metrics do not prove global scientific impact.
- Weak benchmark metrics may reflect missing candidates rather than poor reasoning.
- Benchmarks usually evaluate reference selection, not downstream author or institution rankings.
- Results are sensitive to target-paper coverage, candidate construction, metadata quality, and citation-context availability.
- Report every run with its `run_manifest.json`, `scoring_config.json`, judge audit, and input hash.
