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

## Local Model Pilot

Model server:

```text
local-impact-judge
models/deepseek-r1-distill-qwen-32b
```

Important implementation detail: DeepSeek-R1-Distill needs enough generation budget to reach a final answer. `max_tokens=32` produced reasoning fragments and zero parseable coverage. The current script uses `MAX_TOKENS=512` by default.

Serial 32-example pilot:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge \
  LIMIT=32 MAX_TOKENS=512 \
  scripts/run_sciimpact_model_eval.sh runs/sciimpact-model-eval-32-mt512
```

Result:

| Split Prefix | Dimension | Coverage | Accuracy |
| --- | --- | ---: | ---: |
| 32 examples | award | 1.000 | 0.844 |

Concurrent 100-example pilot:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge \
  LIMIT=100 MAX_TOKENS=512 CONCURRENCY=4 \
  scripts/run_sciimpact_model_eval.sh runs/sciimpact-model-eval-100-c4-mt512
```

Result:

| Split Prefix | Dimension | Coverage | Accuracy |
| --- | --- | ---: | ---: |
| 100 examples | award | 1.000 | 0.810 |

Generated artifacts:

```text
runs/sciimpact-model-eval-32-mt512/eval/sciimpact_model_eval.json
runs/sciimpact-model-eval-100-c4-mt512/eval/sciimpact_model_eval.json
```

These are pilot results only. The first 100 test examples are all `award` tasks, so this is evidence that the local model can beat the text-prior floor on this dimension, not evidence of full SciImpact performance across citation, patent, stars, media, and downloads.

## Stratified Local Model Pilot

To avoid prefix bias in the SFT test file, the project now supports stratified SFT sampling:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge \
  PER_DIMENSION=20 SEED=7 MAX_TOKENS=512 CONCURRENCY=4 \
  scripts/run_sciimpact_stratified_model_eval.sh \
  runs/sciimpact-model-eval-stratified-p20-c4-mt512
```

Sample:

| Dimension | Available | Sampled |
| --- | ---: | ---: |
| award | 10,766 | 20 |
| citation | 13,340 | 20 |
| downloads | 6,578 | 20 |
| media | 15,789 | 20 |
| patent | 13,657 | 20 |
| stars | 2,681 | 20 |

Overall result:

| Examples | Coverage | Accuracy |
| ---: | ---: | ---: |
| 120 | 1.000 | 0.558 |

By dimension:

| Dimension | Accuracy |
| --- | ---: |
| media | 0.700 |
| stars | 0.650 |
| award | 0.600 |
| downloads | 0.600 |
| citation | 0.400 |
| patent | 0.400 |

Generated artifacts:

```text
runs/sciimpact-model-eval-stratified-p20-c4-mt512/sample/sciimpact_sft_stratified.jsonl
runs/sciimpact-model-eval-stratified-p20-c4-mt512/sample/sciimpact_stratified_sample_manifest.json
runs/sciimpact-model-eval-stratified-p20-c4-mt512/model_eval/eval/sciimpact_model_eval.json
runs/sciimpact-model-eval-stratified-p20-c4-mt512/model_eval/eval/sciimpact_model_eval_rows.jsonl
runs/sciimpact-model-eval-stratified-p20-c4-mt512/model_eval/eval/sciimpact_model_eval_wrong.jsonl
runs/sciimpact-model-eval-stratified-p20-c4-mt512/model_eval/eval/sciimpact_model_error_analysis.json
runs/sciimpact-model-eval-stratified-p20-c4-mt512/model_eval/eval/sciimpact_model_error_analysis.md
```

Interpretation: the local zero-shot 32B model is above the deterministic text-prior floor on this balanced pilot, but the margin is modest and uneven. Citation and patent are the immediate failure dimensions. The award-only prefix result should not be used as the main quality claim.

## Final-Answer Prompt Experiment

The evaluator now prefers an explicit final answer line when present:

```text
Answer: A
Answer: B
```

This was tested as a prompt-format intervention:

```bash
INPUT=runs/sciimpact-model-eval-stratified-p20-final-answer-c4-mt512/sample/sciimpact_sft_stratified.jsonl \
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge \
LIMIT=120 MAX_TOKENS=512 CONCURRENCY=4 ANSWER_FORMAT=final_answer \
scripts/run_sciimpact_model_eval.sh \
  runs/sciimpact-model-eval-stratified-p20-final-answer-c4-mt512/model_eval
```

Result:

| Prompt | Examples | Coverage | Accuracy | Explicit `Answer:` Count |
| --- | ---: | ---: | ---: | ---: |
| freeform | 120 | 1.000 | 0.558 | n/a |
| final_answer | 120 | 1.000 | 0.475 | 45 |

By dimension:

| Dimension | Freeform | Final Answer |
| --- | ---: | ---: |
| award | 0.600 | 0.500 |
| citation | 0.400 | 0.350 |
| downloads | 0.600 | 0.450 |
| media | 0.700 | 0.500 |
| patent | 0.400 | 0.400 |
| stars | 0.650 | 0.650 |

Interpretation: this prompt intervention did not improve the model. It reduced accuracy and the model only followed the explicit `Answer:` format in 45 of 120 cases. Keep the parser support because it is useful for stricter future models, but do not make `ANSWER_FORMAT=final_answer` the default for DeepSeek-R1-Distill-32B.

## Next Evaluation Target

The next meaningful benchmark is a larger local model prediction run with balanced coverage across dimensions:

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
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge \
  LIMIT=200 MAX_TOKENS=512 CONCURRENCY=4 \
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

For local model answer evaluation, use these equivalent thresholds as a minimum floor:

```text
accuracy > 0.494
coverage_rate close to 1.000
```
