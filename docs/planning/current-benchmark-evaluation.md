# Current Benchmark Evaluation

Date: 2026-05-28

## What Was Evaluated

Benchmark: SciImpact official SFT test split.

Local file:

```text
data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl
```

Dataset size:

- Source rows: 63,207
- Parsed pairwise examples: 62,811
- Skipped malformed rows: 396
- Unique converted items: 69,184

## Current System Under Test

The evaluated system is the current no-GPU `sciimpact_text_prior` baseline. It scores each converted SciImpact item from text cues in title/abstract and then evaluates whether the benchmark winner is ranked above the loser.

This is not a trained local model result. It is the current deterministic baseline and lower bound for future model fine-tuning.

## Reproduction Command

```bash
LIMIT=63207 scripts/run_sciimpact_sft_baseline.sh runs/sciimpact-sft-baseline-full-test
```

Generated artifacts:

```text
runs/sciimpact-sft-baseline-full-test/converted/manifest.json
runs/sciimpact-sft-baseline-full-test/rankings/sciimpact_text_prior.csv
runs/sciimpact-sft-baseline-full-test/pairwise_eval/pairwise_evaluation.json
runs/sciimpact-sft-baseline-full-test/pairwise_eval/pairwise_evaluation.md
```

## Results

Overall:

| Metric | Value |
| --- | ---: |
| total_pairs | 62,811 |
| coverage_rate | 1.000 |
| pairwise_accuracy | 0.494 |
| non_tie_pairwise_accuracy | 0.507 |
| tie_rate | 0.025 |

By impact dimension:

| Dimension | Pairs | Pairwise Accuracy | Non-tie Accuracy | Tie Rate |
| --- | ---: | ---: | ---: | ---: |
| citation | 13,340 | 0.540 | 0.541 | 0.001 |
| award | 10,766 | 0.524 | 0.526 | 0.004 |
| patent | 13,657 | 0.502 | 0.502 | 0.001 |
| stars | 2,681 | 0.493 | 0.519 | 0.050 |
| media | 15,789 | 0.464 | 0.465 | 0.002 |
| downloads | 6,578 | 0.408 | 0.509 | 0.199 |

## Interpretation

The deterministic text-prior baseline is approximately random on the full SciImpact test split. It is slightly positive for citation and award tasks, neutral for patent/stars, and weak for media/downloads.

This means the current baseline is useful mainly as a sanity-check floor. A trained local impact model should be evaluated against the same test split and should materially exceed this baseline before being treated as useful.

## Next Evaluation Target

The next meaningful benchmark is a real local model prediction run:

1. Export train DPO data:

```bash
LIMIT=88515 scripts/run_sciimpact_dpo_export_smoke.sh runs/sciimpact-dpo-export
```

2. Train or load a local model.

3. Generate predictions JSONL on SciImpact test:

```json
{"source_line": 1, "prediction": "Paper A won the best paper award."}
```

With an already running OpenAI-compatible local model server:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge LIMIT=200 \
  scripts/run_sciimpact_model_eval.sh runs/sciimpact-model-eval
```

4. Evaluate model answers:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-sciimpact-predictions \
  --sft data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl \
  --predictions runs/my-sciimpact-model/predictions.jsonl \
  --output runs/my-sciimpact-model/eval
```

The current target to beat is:

```text
pairwise_accuracy > 0.494
non_tie_pairwise_accuracy > 0.507
```
