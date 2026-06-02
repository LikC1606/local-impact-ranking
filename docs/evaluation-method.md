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

## Pairwise Impact Benchmarks

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-pairwise-impact \
  --rankings runs/example/rankings/papers.csv \
  --labels data/sciimpact/train/pairwise_labels.jsonl \
  --output runs/example/pairwise_eval
```

Files:

- `pairwise_evaluation.json`
- `pairwise_evaluation.md`
- `pairwise_credibility_scorecard.json`
- `pairwise_credibility_scorecard.md`

Use this for SciImpact-style labels where each benchmark row says paper A should outrank paper B for a stated impact dimension. The report separates uncovered pairs from wrong pairs so identifier coverage is visible instead of silently reducing accuracy.

Key fields:

- `pairwise_accuracy`: correct winner-over-loser pairs divided by covered pairs.
- `non_tie_pairwise_accuracy`: correct pairs divided by correct plus wrong pairs, excluding score ties.
- `coverage_rate`: covered benchmark pairs divided by total benchmark pairs.
- `tie_rate`: tied covered pairs divided by covered pairs.
- `statistical_significance`: one-sided binomial evidence that non-tie accuracy is above a random 0.5 baseline.
- `credibility_scorecard`: coverage, statistical support, benchmark accuracy, tie-rate, and slice-stability gates. `high_credibility` requires strong accuracy, low ties, adequate coverage, statistically significant performance, and no weak eligible dimension/field slice.

## SciImpact Model Answer Evaluation

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-sciimpact-predictions \
  --sft data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl \
  --predictions runs/my-sciimpact-model/predictions.jsonl \
  --output runs/my-sciimpact-model/eval
```

Files:

- `sciimpact_model_eval.json`
- `sciimpact_model_eval.md`
- `sciimpact_model_credibility_scorecard.json`
- `sciimpact_model_credibility_scorecard.md`
- `sciimpact_model_eval_rows.jsonl`
- `sciimpact_model_eval_wrong.jsonl`
- `sciimpact_model_error_analysis.json`
- `sciimpact_model_error_analysis.md`

Use this for local model A/B answers on the official SciImpact SFT format. The scorecard uses coverage, one-sided binomial significance against a random 0.5 A/B baseline, overall accuracy, and dimension/item-type slice stability. `high_credibility` requires enough covered examples, significant above-random accuracy, useful absolute accuracy, and no weak eligible slice.

For multi-seed model evidence, run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli summarize-sciimpact-model-credibility \
  --eval runs/model/seed23/eval \
  --eval runs/model/seed47/eval \
  --eval runs/model/seed71/eval \
  --output runs/model/credibility_summary
```

Files:

- `sciimpact_model_credibility_summary.json`
- `sciimpact_model_credibility_summary.md`

This summary pools direct accuracy across evaluations, reports mean/min seed accuracy, checks seed stability, lists weak dimension/item-type slices, and emits a single model-credibility decision. A model should not be promoted from one favorable seed when the cross-seed summary remains weak or slice-unstable.

For user-facing runs, `scripts/run_user_impact.sh` converts the selected model-credibility profile into `model_credibility_card.json/md`. The card is copied into the report bundle and gates `report_decision`: missing, low, not-credible, or mismatched model evidence downgrades otherwise reportable model-supported scores to `exploratory_only`; moderate evidence is reportable only with disclosure. The card records `served_model`, profile model names, and `model_identity_match`, so benchmark evidence for one local model cannot be silently reused for another served model.

The same workflow writes `calibration_profile_card.json/md` for the confidence calibration profile. It checks calibration sample size, ECE/MCE/Brier thresholds, `compatible_models`, and `calibration_model_identity_match`. Missing, weak, or mismatched calibration evidence also downgrades `report_decision` to `exploratory_only`, because calibrated confidence affects the report-facing confidence fields and should not be reused across unrelated local model services.

Candidate promotion gate:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli compare-sciimpact-model-candidate \
  --baseline-summary runs/current-default/credibility_summary \
  --candidate-summary runs/candidate-model/credibility_summary \
  --output runs/candidate-model/promotion_gate
```

Files:

- `sciimpact_model_candidate_gate.json`
- `sciimpact_model_candidate_gate.md`

This gate compares the candidate against the current recommended model instead of judging it in isolation. The default decision is `do_not_promote` unless the candidate has at least three evaluations, improves pooled accuracy, avoids worst-seed regression, avoids material seed-stability regression, does not add weak slices, avoids material worst-slice regression, and keeps credibility status at least as strong as the baseline.

Weak-slice error export:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli export-sciimpact-model-weak-slice-errors \
  --eval runs/model/seed23/eval \
  --eval runs/model/seed47/eval \
  --eval runs/model/seed71/eval \
  --output runs/model/weak_slice_errors
```

Files:

- `sciimpact_model_weak_slice_error_pack.json`
- `sciimpact_model_weak_slice_error_pack.md`
- `sciimpact_model_weak_slice_errors.jsonl`

The error pack collects wrong/uncovered rows whose dimension or item type appears in a weak slice. It is marked `diagnostic_only` unless `--training-source` is passed, so held-out SciImpact errors are not accidentally reused for fine-tuning.

If the aligned SFT source is provided with one `--sft` per `--eval`, the command also writes:

- `train_pairwise_sft.jsonl`
- `train_pairwise_dpo.jsonl`
- `weak_slice_source_records.jsonl`

Use this only for train-split predictions. The exported training rows preserve `matched_weak_slices`, model expected/predicted sides, confidence, and source-line metadata so later fine-tuning can be audited against the weak-slice diagnosis.

The prepared continuation script is:

```bash
scripts/train_sciimpact_weak_slice_dpo.sh
```

It validates the DPO candidate file before training. Its output still requires held-out SciImpact re-evaluation, cross-seed credibility summary, and weak-slice error-pack comparison before any model promotion claim.

## Gold Baseline Comparison

Files:

- `reports/gold_baseline_comparison.json`
- `reports/gold_baseline_comparison.md`

This report compares the judge against simple candidate-only baselines on the same labeled targets:

- `candidate_order`: select the first `top_k` converted candidates.
- `title_overlap`: rank candidates by lexical overlap with the target title/abstract.
- `random`: seeded random candidate selection.

Key fields:

- `judge_beats_all_baselines`: true only when the judge beats the strongest baseline by mean NDCG.
- `judge_mean_ndcg_delta_vs_best_baseline`: positive means the judge is better than the strongest simple baseline.
- `judge_ndcg_delta_ci95`: paired bootstrap 95% interval for the judge's per-target NDCG delta versus the strongest baseline.
- `statistically_positive_ndcg_delta`: true only when the bootstrap interval lower bound is above zero.
- `needs_model_or_prompt_improvement`: true when the judge does not beat these baselines or the bootstrap NDCG-delta interval still includes zero.

This check prevents reporting a model as useful when it is only matching or underperforming simple retrieval heuristics.

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

## Continuous Impact Validity

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli apply-citation-cutoff \
  --input papers.openalex.jsonl \
  --output papers.cutoff-2022.jsonl \
  --cutoff-year 2022

PYTHONPATH=src python -m local_impact_ranking.cli audit-citation-cutoff \
  --input papers.cutoff-2022.jsonl \
  --output runs/example/reports \
  --cutoff-year 2022

PYTHONPATH=src python -m local_impact_ranking.cli derive-future-impact-labels \
  --input papers.openalex.jsonl \
  --output runs/example/future-labels \
  --start-year 2023 \
  --end-year 2025 \
  --normalization field-year-percentile

PYTHONPATH=src python -m local_impact_ranking.cli evaluate-impact-validity \
  --rankings runs/example/impact_scores.csv \
  --labels runs/example/future-labels/future_impact_labels.jsonl \
  --label-column future_impact \
  --top-fraction 0.1 \
  --bootstrap-samples 1000 \
  --permutation-samples 1000 \
  --output runs/example/impact_validity
```

The same evidence bundle can be generated after scoring the cutoff dataset with:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli backtest-impact-validity \
  --enriched-input papers.openalex.jsonl \
  --cutoff-input papers.cutoff-2022.jsonl \
  --rankings runs/example/impact_scores.csv \
  --output runs/example \
  --cutoff-year 2022 \
  --future-start-year 2023 \
  --future-end-year 2025 \
  --label-normalization field-year-percentile \
  --bootstrap-samples 1000 \
  --permutation-samples 1000
```

Files:

- `impact_validity_evaluation.json`
- `impact_validity_evaluation.md`
- `impact_credibility_scorecard.json`
- `impact_credibility_scorecard.md`
- `citation_cutoff_audit.json`
- `citation_cutoff_audit.md`

Use this when the evaluation target is a continuous or ordinal external impact label rather than pairwise winners. Examples include future-window field/year-normalized citations, BIP! impact classes, future patent or policy citations, artifact adoption counts, or a combined held-out impact target.

`apply-citation-cutoff` prevents time leakage in historical backtests. It removes citation-history years after the cutoff and rewrites citation totals in the scoring input. `audit-citation-cutoff` writes `citation_cutoff_audit.json/md` and fails the audit when any citation-history year is greater than the cutoff. Put those files under the run's `reports/` directory so `project-readiness` and `report-bundle` can carry the leakage-safety evidence. When `impact_validity_evaluation.json` exists, missing cutoff audit evidence becomes a readiness warning and a failed cutoff audit becomes a blocking condition. The intended workflow is to score the cutoff dataset, then evaluate the resulting scores against a later window generated from the uncropped enriched dataset.

`derive-future-impact-labels` is a convenience command for OpenAlex-enriched data. It reads `metadata.citation_counts_by_year`, `metadata.citations_by_year`, `metadata.yearly_citation_counts`, or compatible top-level fields and writes `future_impact_labels.jsonl`, `future_impact_labels.csv`, and `future_impact_labels_manifest.json`. This is useful for historical backtesting: score papers using evidence available up to a cutoff, then validate against citation counts in a later window. For mixed-field or mixed-year datasets, prefer `field_year_percentile` via `--normalization field-year-percentile` or `--label-normalization field-year-percentile`; the raw future count is preserved as `future_impact_raw`, while `future_impact` is normalized inside each field/year group.

`evaluate-impact-validity` also writes bootstrap confidence intervals, a permutation random baseline, and an `impact_credibility_scorecard`. The report's `significant_vs_random` flag is true only when pairwise accuracy is above chance, the bootstrap interval lower bound is above `0.5`, and the permutation p-value is at most `0.05`. The scorecard combines coverage, statistical support, rank alignment, top-rank quality, field-slice stability, and outlier robustness into `high_credibility`, `moderate_credibility`, `low_credibility`, `not_credible`, or `insufficient_validation_data`. This prevents a point estimate that is close to random or driven by one extreme row from being reported as validated impact accuracy.

`project-readiness` consumes the same report. If `impact_validity_evaluation.json` is present but the result is not statistically supported, readiness adds a warning and the run must not be described as validated impact accuracy. The dataset-local ranking can still be reported with disclosure when the rest of the evidence is usable.

The report also records label coverage: `ranking_paper_count`, `label_paper_count`, `matched_paper_count`, `score_label_match_rate`, and `label_match_rate`. It includes `coverage_examples` with `unmatched_score_ids` and `unmatched_label_ids` so low coverage can be traced back to entity IDs. `project-readiness` warns when `score_label_match_rate` is below `0.5`, because strong metrics on a small matched subset may reflect entity matching or coverage bias rather than trustworthy impact validity.

Key fields:

- `spearman_correlation`: rank correlation between the local score and the external/future label.
- `kendall_tau_b`: pairwise rank agreement with tie correction.
- `pairwise_accuracy`: fraction of comparable paper pairs where the higher score also has the higher external/future label.
- `ndcg_at_top_k`: top-ranked-score quality using the external/future label as graded relevance.
- `top_k_precision` and `top_k_recall`: overlap between the score's top-k papers and the label's top-k papers.
- `score_label_match_rate`: matched labeled papers divided by scored ranking rows.
- `label_match_rate`: matched labeled papers divided by label rows.
- `coverage_examples`: sampled unmatched IDs from the score table and label table for entity-matching audit.
- `bootstrap.metrics.*.ci95`: bootstrap 95% confidence interval for each validity metric.
- `permutation_baseline.metrics.*.p_value`: random-label baseline p-value for each validity metric.
- `significant_vs_random`: conservative flag used by the decision text to distinguish real signal from random-like scores.
- `robustness`: leave-top-label, leave-top-score, and top-label-5-percent exclusion checks that expose outlier-sensitive validity metrics.
- `credibility_scorecard`: coverage, statistical support, rank alignment, top rank, slice stability, and outlier-robustness gates for report-facing credibility.
- `by_field`: the same validity metrics split by `field` when the label file provides it.

The command returns `insufficient_validation_data` for tiny samples, `externally_supported` only when rank, pairwise, and top-k metrics are all strong, and `usable_with_disclosure` for partial signal. This should be the preferred report for future-window validation because it directly tests whether `final_impact_score` tracks observed downstream impact.

## Model Selection Decision

Files:

- `model_selection_decision.json`
- `model_selection_decision.md`

The dev-set decision recommends `promote_to_test` only when:

- common labeled targets exist;
- `net_hit_change` is positive;
- `mean_ndcg_delta` is at least the configured minimum;
- `broken_count / common_targets` is below the configured maximum;
- when the candidate run's `reports/benchmark_summary.json` is available, `judge_beats_all_baselines=true` and `statistically_positive_ndcg_delta=true`.

This prevents promoting a candidate model that improves over the previous model but still fails to beat simple candidate baselines such as title overlap, or only beats them by an unstable mean delta.

## Baseline Alignment

File:

- `reports/baseline_comparison.json`

This report compares the local impact scores against traditional bibliometric baselines:

- `citation_count`
- `pagerank`
- `field_year_normalized_citation`

For each baseline it reports:

- `raw_top_n_overlap_rate`: overlap between raw selected-reference score top-N and baseline top-N.
- `final_top_n_overlap_rate`: overlap between fused `final_score` top-N and baseline top-N.
- `final_spearman`: Spearman rank correlation between `final_score` and the baseline.
- `final_kendall_tau`: pairwise rank agreement between `final_score` and the baseline.

Interpretation:

- High alignment with citation count or PageRank means the local model is broadly consistent with conventional citation signals.
- Low alignment is not automatically wrong, but it must be explained by citation-context evidence, field/year normalization, or known dataset scope.
- Negative correlation is a warning that the final score may be ranking against the graph/citation signal and should be inspected through `paper_evidence_report.csv`.

## Reporting Limits

- Strong benchmark metrics do not prove global scientific impact.
- Weak benchmark metrics may reflect missing candidates rather than poor reasoning.
- Benchmarks usually evaluate reference selection, not downstream author or institution rankings.
- Results are sensitive to target-paper coverage, candidate construction, metadata quality, and citation-context availability.
- Report every run with its `run_manifest.json`, `scoring_config.json`, judge audit, and input hash.

## Project Readiness Gate

Command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli project-readiness runs/example --markdown
```

Files:

- `reports/project_readiness.json`
- `reports/project_readiness.md`

This is the final reportability gate. It aggregates artifact completeness, validation, judge reliability, scoring config, ranking scope, gold/proxy evidence, stability, anomaly risk, fine-tuning preflight, and claim boundaries.

It also records `evidence_level`:

- `gold_validated`: benchmark gold labels exist, such as PreScience positive references.
- `proxy_supported`: no gold labels exist, but selected references have proxy support from citation contexts or bibliometric baselines.
- `proxy_only_needs_review`: proxy evidence exists, but unsupported-selection risk is high.
- `no_external_or_proxy_evidence`: neither gold labels nor proxy support are available.

The report includes a reproducibility summary copied from `run_manifest.json`, including input hashes, normalized record count, top-k, judge/model, prompt version, scoring version, git revision, and whether the worktree was dirty when the run manifest was created.

Statuses:

- `reportable`: hard gates pass and no warning gates fail.
- `needs_improvement`: hard gates pass, but warnings such as missing gold labels, exploratory ranking scope, high unsupported-selection rate, weak stability, or single-edge dominance must be disclosed.
- `blocked`: missing required artifacts, validation failure, missing judge results, parse reliability failure, invalid selections, top-k violations, missing scoring config, or missing paper ranking.

The readiness gate does not create a new impact metric. It decides whether the existing run evidence is strong enough to report under the dataset-local claim boundary.
