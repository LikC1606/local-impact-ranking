# Qwen32B SciImpact Evaluation Notes

Date: 2026-05-28

## Setup

- Served model: `local-impact-judge`
- Model path: `models/deepseek-r1-distill-qwen-32b`
- Benchmark sample: SciImpact test split, stratified `PER_DIMENSION=50`, `SEED=23`
- Inference: `MAX_TOKENS=512`, `CONCURRENCY=4`

## Direct Pairwise Result

| Metric | Value |
| --- | ---: |
| total_examples | 300 |
| covered_examples | 299 |
| correct_examples | 152 |
| wrong_examples | 147 |
| coverage_rate | 0.997 |
| accuracy | 0.508 |

## By Dimension

| Dimension | Accuracy | Correct | Wrong | Coverage |
| --- | ---: | ---: | ---: | ---: |
| award | 0.560 | 28 | 22 | 1.000 |
| citation | 0.500 | 25 | 25 | 1.000 |
| downloads | 0.560 | 28 | 22 | 1.000 |
| media | 0.400 | 20 | 30 | 1.000 |
| patent | 0.560 | 28 | 22 | 1.000 |
| stars | 0.469 | 23 | 26 | 0.980 |

## Impact Index Evaluation

| Metric | Value |
| --- | ---: |
| total_pairs | 300 |
| covered_pairs | 300 |
| correct_pairs | 152 |
| wrong_pairs | 147 |
| pairwise_accuracy | 0.507 |
| coverage_rate | 1.000 |

## Main Weaknesses

- `media`: wrong_rate=0.600 (30/50)
- `stars`: wrong_rate=0.520 (26/50)
- `citation`: wrong_rate=0.500 (25/50)
- `award`: wrong_rate=0.440 (22/50)
- `downloads`: wrong_rate=0.440 (22/50)
- `patent`: wrong_rate=0.440 (22/50)

## Interpretation

The current local Qwen-series judge is only slightly above random on this balanced SciImpact sample. The 120-example pilot was optimistic; the 300-example stratified run is a better signal. Media, stars, and citation are the most important weaknesses in this run. The impact-index aggregation path is wired correctly, but this SciImpact sample has mostly one-off pair entities, so aggregation cannot rescue weak pairwise judgments.

## Next Improvement Target

Use SciImpact train data to fine-tune or preference-tune the local judge, then rerun the same `PER_DIMENSION=50` evaluation. For immediate non-training improvement, evaluate a stronger local Qwen model if available; otherwise the bottleneck is model judgment quality, not the aggregation code.

## LoRA SFT Pilot: Random 1024 / 64 Steps

- Adapter: `runs/sciimpact-pairwise-sft-pilot/checkpoints/qwen32b-sft-random1024-64step`
- Training data: `runs/sciimpact-pairwise-sft-full/train_pairwise_sft.jsonl`
- Loaded clean examples: `87984`
- Sampled examples: `1024`, random seed `23`
- Training: `max_steps=64`, `max_length=1024`, `gradient_accumulation_steps=4`, `lora_r=8`, `lora_alpha=16`
- Decoding for benchmark: constrained Qwen token IDs for exactly `Answer: A` / `Answer: B`

| Run | Direct accuracy | Index pairwise accuracy | Coverage |
| --- | ---: | ---: | ---: |
| Base Qwen32B, p50 | 0.508 | 0.507 | 0.997 / 1.000 |
| SFT random1024-64, constrained p50 | 0.637 | 0.637 | 1.000 |

By dimension for `SFT random1024-64`:

| Dimension | Accuracy |
| --- | ---: |
| award | 0.660 |
| citation | 0.660 |
| downloads | 0.660 |
| media | 0.640 |
| patent | 0.640 |
| stars | 0.560 |

Interpretation: the useful gain comes from combining small LoRA SFT with constrained answer decoding. Free generation still produces reasoning text and is not the preferred production/evaluation mode for pairwise judging.

## LoRA SFT Pilot: Balanced Metadata 4096 / 128 Steps

- Adapter: `runs/sciimpact-pairwise-sft-pilot/checkpoints/qwen32b-sft-balanced4096-128step`
- Training data: `runs/sciimpact-pairwise-sft-full/train_pairwise_sft.jsonl`
- Loaded clean examples: `87984`
- Sampled examples: `4096`, seed `29`
- Sampling strategy: `balanced_metadata`, grouped by `impact_dimension + item_type + target_side`
- Sampling groups: `14`
- Training: `max_steps=128`, `max_length=1024`, `gradient_accumulation_steps=4`, `lora_r=8`, `lora_alpha=16`
- Final logged loss: `0.143`
- Evaluation run: `runs/sciimpact-qwen32b-sft-balanced128-constrained-answer3-stratified-p50-c4`
- Index evaluation: `runs/sciimpact-qwen32b-sft-balanced128-constrained-answer3-index-eval-p50-c4`

| Run | Direct accuracy | Index pairwise accuracy | Coverage | Predicted A/B |
| --- | ---: | ---: | ---: | --- |
| SFT random1024-64, constrained p50 | 0.637 | 0.637 | 1.000 | 179 / 121 |
| SFT balanced4096-128, constrained p50 | 0.623 | 0.623 | 1.000 | 189 / 111 |

By dimension for `SFT balanced4096-128`:

| Dimension | Accuracy | Change vs random1024-64 |
| --- | ---: | ---: |
| award | 0.640 | -0.020 |
| citation | 0.600 | -0.060 |
| downloads | 0.640 | -0.020 |
| media | 0.680 | +0.040 |
| patent | 0.600 | -0.040 |
| stars | 0.580 | +0.020 |

Interpretation: balanced metadata sampling improved `media` and slightly improved `stars`, but reduced `citation`, `patent`, `award`, and `downloads`; it also increased A-side bias on this p50 sample. It should be kept as a negative/diagnostic experiment, not promoted over `random1024-64`.

## Current Best

Use the DPO adapter `runs/sciimpact-pairwise-dpo-pilot/checkpoints/qwen32b-sft1024-dpo-random4096-64step` with structured or constrained `Answer: A/B` decoding.

Best measured inference recipe on the original 300-example stratified p50 benchmark (`SEED=23`):

1. Run the original A/B prompt.
2. Run a flipped prompt with Paper A and Paper B swapped.
3. Map the flipped answer back to the original orientation.
4. If the two mapped answers agree, use the agreed side.
5. If they disagree, use the higher-confidence side only when its raw logprob confidence is ahead by at least `0.05`; otherwise keep the original answer.

| Run | Direct accuracy | Index pairwise accuracy | Correct / total | Coverage |
| --- | ---: | ---: | ---: | ---: |
| DPO64 constrained answer | 0.653 | 0.653 | 196 / 300 | 1.000 |
| DPO64 structured choice | 0.653 | 0.653 | 196 / 300 | 1.000 |
| DPO64 flip consistency, margin 0.00 | 0.677 | 0.677 | 203 / 300 | 1.000 |
| DPO64 flip consistency, margin 0.05 | 0.690 | 0.690 | 207 / 300 | 1.000 |

Flip-consistency diagnostic for margin `0.05`:

- Original accuracy: `0.653333`
- Flipped-mapped accuracy: `0.636667`
- Selected accuracy: `0.690000`
- Consistency rate: `0.636667`
- Both-correct examples: `139`
- Original-only correct examples: `57`
- Flipped-only correct examples: `52`
- Both-wrong examples: `52`

Interpretation on `SEED=23`: this is an inference-time robustness improvement, not a new trained model. It reduces A/B position sensitivity and uses confidence only when the confidence gap is large enough to justify overriding the original orientation. The `0.05` margin was selected on this benchmark sample, so it should not be treated as a general default without held-out validation.

Held-out seed check (`SEED=47`):

- Original structured-choice accuracy: `0.620` (`186/300`)
- Flipped-only mapped accuracy: `0.633` (`190/300`)
- Flip consistency, margin `0.00`: `0.620` (`186/300`)
- Flip consistency, margin `0.05`: `0.623` (`187/300`)
- Best margin sweep on this seed: about `0.21-0.22`, `0.637` (`191/300`)
- Position consistency rate: `0.653`

Interpretation after `SEED=47`: the fixed `0.05` margin does not generalize strongly. Bidirectional judging should be promoted in production as a robustness/confidence signal and a way to mark unresolved position conflicts non-decisive, not as a fixed benchmark-tuned answer-selection rule.

## Next Model Improvements

1. Keep constrained `Answer: A/B` decoding for benchmark and production pairwise judging.
2. Use A/B flip consistency as the default robustness and confidence diagnostic for pairwise judging.
3. Do not promote the `0.05` confidence margin as a universal default; multi-seed validation shows it is sample-sensitive.
4. Try longer random or lightly stratified training instead of fully balanced metadata sampling, because the balanced128 run worsened the main p50 score.
5. Calibrate logprob/classification-style confidence on a larger held-out SciImpact sample before exposing confidence as a trust score.
6. Investigate `stars`, `dataset`, and `github_repo` failure modes separately; these remain weaker than paper/model item types.

## Weak-Dimension SFT Diagnostic

- Adapter: `runs/sciimpact-weak-dimension-sample/checkpoints/qwen32b-sft-weakdims-64step`
- Training data: `runs/sciimpact-weak-dimension-sample/train_pairwise_sft.jsonl`
- Sampling: `2000` examples, oversampling `patent` and `downloads` threefold.
- Training: `max_steps=64`, `max_length=1024`, `gradient_accumulation_steps=4`, `lora_r=8`, `lora_alpha=16`, seed `53`.
- Evaluation sample: same SciImpact stratified p50 sample as the current-best comparison.

| Run | Direct accuracy | Index pairwise accuracy | Correct / total | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Weakdims64 constrained answer | 0.633 | 0.633 | 190 / 300 | 1.000 |
| Weakdims64 flip consistency, margin 0.05 | 0.657 | 0.657 | 197 / 300 | 1.000 |
| Current best DPO64 flip consistency, margin 0.05 | 0.690 | 0.690 | 207 / 300 | 1.000 |

By dimension for `Weakdims64 constrained answer`:

| Dimension | Accuracy |
| --- | ---: |
| award | 0.680 |
| citation | 0.720 |
| downloads | 0.640 |
| media | 0.520 |
| patent | 0.680 |
| stars | 0.560 |

By dimension for `Weakdims64 flip consistency, margin 0.05`:

| Dimension | Accuracy |
| --- | ---: |
| award | 0.680 |
| citation | 0.700 |
| downloads | 0.620 |
| media | 0.680 |
| patent | 0.700 |
| stars | 0.560 |

Complementarity against current-best DPO64 selected predictions was weak: `weakdims64` fixed only `3` DPO errors while breaking `13` DPO correct cases, and an oracle that could choose the correct answer between the two models reached only `210/300 = 0.700`.

Interpretation: oversampling weak dimensions in SFT moved some dimensions, but it did not produce a stronger judge. It should stay as a negative diagnostic. The next training route should be DPO continuation from the current best DPO adapter on weak-dimension preference data, not promotion of this adapter or a simple ensemble.

## Weak-Dimension DPO Continuation Diagnostic

- Adapter: `runs/sciimpact-weak-dimension-sample/checkpoints/qwen32b-dpo-weakdims-from-dpo64-32step`
- Base adapter/reference policy: `runs/sciimpact-pairwise-dpo-pilot/checkpoints/qwen32b-sft1024-dpo-random4096-64step`
- Training data: `runs/sciimpact-weak-dimension-sample/train_pairwise_dpo.jsonl`
- Loaded examples: `2000`
- Training: `max_steps=32`, `max_length=1024`, `gradient_accumulation_steps=4`, `learning_rate=1e-6`, `lora_r=8`, `lora_alpha=16`, seed `61`
- Final logged DPO loss: `0.6875`

| Run | Direct accuracy | Index pairwise accuracy | Correct / total | Coverage |
| --- | ---: | ---: | ---: | ---: |
| DPO weakdims32 constrained answer | 0.647 | 0.647 | 194 / 300 | 1.000 |
| DPO weakdims32 flip consistency, margin 0.05 | 0.687 | 0.687 | 206 / 300 | 1.000 |
| DPO weakdims32 flip consistency, best swept margin 0.04 | 0.690 | not rerun as index | 207 / 300 | 1.000 |
| Current best DPO64 flip consistency, margin 0.05 | 0.690 | 0.690 | 207 / 300 | 1.000 |

By dimension for `DPO weakdims32 flip consistency, margin 0.05`:

| Dimension | Accuracy | Change vs current-best DPO64 margin 0.05 |
| --- | ---: | ---: |
| award | 0.680 | 0.000 |
| citation | 0.720 | +0.020 |
| downloads | 0.680 | 0.000 |
| media | 0.700 | 0.000 |
| patent | 0.740 | -0.020 |
| stars | 0.600 | -0.020 |

Margin sweep over `0.00-0.30` found the best result at margin `0.04`, tying the current best at `207/300 = 0.690`. This is not a held-out improvement and should not be promoted as a new default. The current best remains `DPO64 + flip-consistency margin 0.05`; the weak-dimension DPO continuation is useful as evidence that weak-dimension oversampling alone is not enough.

## Item-Type Targeted Next Route

The next useful model route is item-type targeted, not another broad dimension-only oversampling run.

Evidence:

- Current best raw DPO64 by item type:
  - `paper`: `0.67`
  - `model`: `0.80`
  - `dataset`: `0.52`
  - `github_repo`: `0.58`
- Current best flip-consistency selected accuracy by item type:
  - `paper`: `0.71`
  - `model`: `0.80`
  - `dataset`: `0.56`
  - `github_repo`: `0.62`
- Weak-dimension SFT improved `dataset` to `0.60` but did not improve `github_repo` and hurt `model`, so broad weak-dimension oversampling is not enough.

Implementation completed for the next training input:

- `sample-sciimpact-sft` now supports `--item-type-weight ITEM=WEIGHT`.
- The sampler now supports both original SciImpact ShareGPT records and project-exported pairwise SFT records with `messages + target + impact_dimension + item_type`.
- Generated item-type targeted bundle:
  - directory: `runs/sciimpact-itemtype-targeted-sample`
  - SFT records: `1400`
  - DPO records: `1400`
  - skipped DPO records: `0`
  - sampling config: `per_dimension=100`, `seed=71`, `stars=2`, `dataset=3`, `github=3`
  - composition: `paper=400`, `dataset=300`, `github_repo=600`, `model=100`

Next experiment:

Train a short DPO continuation from the current best DPO64 adapter on `runs/sciimpact-itemtype-targeted-sample/train_pairwise_dpo.jsonl`, then evaluate on the same stratified p50 `SEED=23` and held-out `SEED=47` samples. Promote only if it improves `dataset/github_repo/stars` without reducing total p50 accuracy below the current best envelope.

## Item-Type Targeted DPO Continuation Diagnostic

- Adapter: `runs/sciimpact-itemtype-targeted-sample/checkpoints/qwen32b-dpo-itemtype-from-dpo64-32step`
- Base adapter/reference policy: `runs/sciimpact-pairwise-dpo-pilot/checkpoints/qwen32b-sft1024-dpo-random4096-64step`
- Training data: `runs/sciimpact-itemtype-targeted-sample/train_pairwise_dpo.jsonl`
- Loaded examples: `1400`
- Training: `max_steps=32`, `max_length=1024`, `gradient_accumulation_steps=4`, `learning_rate=1e-6`, `lora_r=8`, `lora_alpha=16`, seed `73`
- Final logged DPO loss: `0.693359`

Same p50 benchmark (`SEED=23`):

| Run | Direct accuracy | Flip consistency margin 0.05 | Correct / total | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Current best DPO64 | 0.653 | 0.690 | 207 / 300 | 1.000 |
| Item-type DPO32 | 0.643 | 0.683 | 205 / 300 | 1.000 |

By item type for `Item-type DPO32` at margin `0.05`:

| Item type | Selected accuracy | Current best DPO64 |
| --- | ---: | ---: |
| paper | 0.700 | 0.710 |
| dataset | 0.560 | 0.560 |
| github_repo | 0.620 | 0.620 |
| model | 0.800 | 0.800 |

By dimension for `Item-type DPO32` at margin `0.05`:

| Dimension | Selected accuracy |
| --- | ---: |
| award | 0.640 |
| citation | 0.700 |
| downloads | 0.680 |
| stars | 0.620 |
| media | 0.700 |
| patent | 0.760 |

Held-out p50 check (`SEED=47`):

- Direct structured-choice accuracy: `0.610` (`183/300`)
- Flipped-mapped accuracy: `0.627` (`188/300`)
- Flip consistency, margin `0.00`: `0.617` (`185/300`)
- Flip consistency, margin `0.05`: `0.613` (`184/300`)
- Best seed-specific margin sweep over `0.00-0.30`: `0.640` (`192/300`) at margin `0.20`

Interpretation: item-type targeted continuation did not solve the weak-type bottleneck. On the original p50 sample it exactly tied current DPO64 on `dataset`, `github_repo`, and `model`, while reducing `paper`; on held-out `SEED=47` it underperformed the current best fixed-margin recipe. This adapter should stay as a negative diagnostic and should not be promoted.

## Next Route: Error-Driven Hard Examples

Simple item-type oversampling did not move the targeted weak types, so the next route should train on model-observed hard cases rather than hand-designed strata.

Implemented tooling:

```bash
python -m local_impact_ranking.cli mine-sciimpact-hard-examples \
  --sft SAMPLE_OR_TRAIN_SFT.jsonl \
  --flip-rows sciimpact_flip_consistency_rows.jsonl \
  --output runs/sciimpact-hard-examples
```

The command writes:

- `train_pairwise_sft.jsonl`
- `train_pairwise_dpo.jsonl`
- `sciimpact_hard_pairwise_manifest.json`

Selection rules:

- include examples where the selected bidirectional answer is wrong (`selected_wrong`);
- include examples where original and flipped-mapped answers disagree (`position_inconsistent`);
- deduplicate by original `source_line`;
- preserve impact dimension, item type, selected outcome, selected source, and consistency metadata;
- warn when the source ultimately comes from a test split.

Diagnostic-only run on current-best DPO64 p50:

- Output: `runs/sciimpact-hard-example-diagnostic-dpo64-p50`
- Hard examples: `161`
- Reason counts: `position_inconsistent=109`, `selected_wrong=93`
- By dimension: patent `35`, award `29`, stars `29`, citation `24`, downloads `22`, media `22`
- By item type: paper `110`, github_repo `29`, dataset `13`, model `9`
- Leakage warning: present, because this run is derived from the SciImpact test split.

Interpretation: the diagnostic shows that the largest remaining failure source is not only item type. Position inconsistency is broader than `dataset/github_repo`, and patent/award/stars remain prominent. For actual training, run current-best DPO64 on a train/validation sample, mine hard examples there, then do a short DPO continuation from DPO64.

## Train-Derived Hard-Example DPO Diagnostic

Train hard mining run:

- Source split: SciImpact train
- Sample: `runs/sciimpact-hard-example-train-p100-dpo64/sample/sciimpact_sft_stratified.jsonl`
- Sampling: `per_dimension=100`, seed `83`, `600` examples
- Current-best DPO64 on this train sample:
  - direct accuracy: `363/600 = 0.605`
  - flip-consistency margin `0.05`: `367/600 = 0.612`
- Mined train hard examples: `347`
- Reason counts: `selected_wrong=233`, `position_inconsistent=210`
- By dimension: patent `77`, downloads `60`, media `57`, stars `56`, award `51`, citation `46`
- By item type: paper `231`, github_repo `56`, model `35`, dataset `25`
- Test leakage warning: none

Adapter:

- `runs/sciimpact-hard-example-train-p100-dpo64/checkpoints/qwen32b-dpo-hardmine-from-dpo64-32step`
- Base adapter/reference policy: `runs/sciimpact-pairwise-dpo-pilot/checkpoints/qwen32b-sft1024-dpo-random4096-64step`
- Training: `max_steps=32`, `max_length=1024`, `gradient_accumulation_steps=4`, `learning_rate=1e-6`, `lora_r=8`, `lora_alpha=16`, seed `89`

Evaluation:

| Run | Direct accuracy | Flip consistency margin 0.05 | Correct / total | Note |
| --- | ---: | ---: | ---: | --- |
| DPO64, p50 seed23 | 0.653 | 0.690 | 207 / 300 | current best |
| Hardmine32, p50 seed23 | 0.653 | 0.683 | 205 / 300 | worse selected |
| DPO64, p50 seed47 | 0.620 | 0.623 | 187 / 300 | held-out fixed margin |
| Hardmine32, p50 seed47 | 0.620 | 0.623 | 187 / 300 | tie |

Margin sweep:

- seed23 hardmine32 best: `206/300 = 0.686667`
- seed47 hardmine32 best: `191/300 = 0.636667`

Interpretation: hard-example mining is useful as infrastructure and produces a cleaner train-derived DPO set than hand-designed item-type oversampling, but this first small p100/32-step continuation did not improve the accepted benchmark envelope. It preserved direct accuracy but did not improve bidirectional selected accuracy, so it should not be promoted.

## Larger SFT+DPO Candidate

The small targeted runs did not materially improve the benchmark. The next useful route was to stop using tiny subsets and train on a larger random slice of the official train split.

Training:

- SFT adapter: `runs/sciimpact-large-sft-dpo/checkpoints/qwen32b-sft-random16k-512step`
- SFT data: `runs/sciimpact-pairwise-sft-full/train_pairwise_sft.jsonl`
- SFT config: `max_examples=16000`, `max_steps=512`, `max_length=1024`, `gradient_accumulation_steps=4`, `learning_rate=1e-5`, `lora_r=8`, `lora_alpha=16`, seed `101`
- DPO adapter: `runs/sciimpact-large-sft-dpo/checkpoints/qwen32b-sft16k512-dpo-random16k-256step`
- DPO data: `runs/sciimpact-pairwise-dpo-full/train_pairwise_dpo.jsonl`
- DPO config: `max_examples=16000`, `max_steps=256`, `max_length=1024`, `gradient_accumulation_steps=4`, `learning_rate=1e-6`, `lora_r=8`, `lora_alpha=16`, seed `103`
- DPO base/reference adapter: the new 16k SFT adapter

Main comparison with the current DPO64 baseline:

| Run | Direct accuracy | Flip consistency margin 0.05 | Correct / total | Status |
| --- | ---: | ---: | ---: | --- |
| DPO64, p50 seed23 | 0.653 | 0.690 | 207 / 300 | current best baseline |
| SFT16k+DPO16k, p50 seed23 | 0.663 | 0.693 | 208 / 300 | small improvement |
| DPO64, p50 seed47 | 0.620 | 0.623 | 187 / 300 | held-out baseline |
| SFT16k+DPO16k, p50 seed47 | 0.617 | 0.633 | 190 / 300 | selected improvement, direct slight drop |

Item-type effects at margin `0.05`:

| Seed | Item type | DPO64 | SFT16k+DPO16k |
| --- | --- | ---: | ---: |
| 23 | paper | 0.710 | 0.710 |
| 23 | dataset | 0.560 | 0.560 |
| 23 | github_repo | 0.620 | 0.660 |
| 23 | model | 0.800 | 0.760 |
| 47 | paper | 0.680 | 0.680 |
| 47 | dataset | 0.579 | 0.579 |
| 47 | github_repo | 0.480 | 0.540 |
| 47 | model | 0.516 | 0.516 |

Margin sweep diagnostic:

- seed23 best: `209/300 = 0.696667` at margin `0.03`
- seed47 best: `196/300 = 0.653333` at margin `0.13`

Third held-out seed check (`SEED=71`):

| Run | Direct accuracy | Flip consistency margin 0.05 | Correct / total | Status |
| --- | ---: | ---: | ---: | --- |
| DPO64, p50 seed71 | 0.617 | 0.653 | 196 / 300 | baseline |
| SFT16k+DPO16k, p50 seed71 | 0.623 | 0.660 | 198 / 300 | small improvement |

Three-seed credibility report:

- Report: `runs/impact-system-credibility-qwen32b-sft16k-dpo16k/credibility_report.md`
- Recommendation: `promote_candidate_for_bidirectional_scoring`
- Mean direct accuracy: DPO64 `0.630`, SFT16k+DPO16k `0.634`, delta `+0.004`
- Mean bidirectional selected accuracy: DPO64 `0.656`, SFT16k+DPO16k `0.662`, delta `+0.007`
- Mean consistency rate: DPO64 `0.646`, SFT16k+DPO16k `0.631`, delta `-0.014`
- Candidate strengths: `stars`, `github_repo`, `award`, `patent`, and `paper`
- Candidate regressions: `model`, `downloads`, `citation`, `dataset`, and `media`

Interpretation at the time: the third seed confirmed a small but consistent bidirectional-scoring gain over DPO64, so `runs/sciimpact-large-sft-dpo/checkpoints/qwen32b-sft16k512-dpo-random16k-256step` became the first recommended production-style bidirectional adapter. It is now treated as the baseline for newer candidates, not the active default.

Promotion gate for future adapters:

- Require at least three held-out SciImpact p50 seeds.
- Require positive mean selected-accuracy delta against the current active adapter.
- Require non-negative mean direct-accuracy delta.
- Allow at most `0.01` selected-accuracy regression on any seed.
- Allow at most `0.05` average regression on any impact-dimension or item-type slice.
- Use the margin `0.05` flip-consistency report when both margin `0.00` and `0.05` reports exist.

The clean 16k-token SFT+DPO run is the first diagnostic through this stricter gate:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli credibility-report \
  --baseline-run runs/sciimpact-qwen32b-sft16k-dpo16k-flip-consistency-stratified-p50-c4/model_eval \
  --baseline-run runs/sciimpact-qwen32b-sft16k-dpo16k-flip-consistency-stratified-p50-seed47-c4/model_eval \
  --baseline-run runs/sciimpact-qwen32b-thirdseed-p50-seed71/sft16k_dpo16k \
  --candidate-run runs/sciimpact-long-context/eval/sft16k-dpo16k-16k-256step-seed23-p50 \
  --candidate-run runs/sciimpact-long-context/eval/sft16k-dpo16k-16k-256step-seed47-p50 \
  --candidate-run runs/sciimpact-long-context/eval/sft16k-dpo16k-16k-256step-seed71-p50 \
  --seed-label seed23 --seed-label seed47 --seed-label seed71 \
  --baseline-model recommended-sft16k512-dpo16k256 \
  --candidate-model clean-16k-token-sft-dpo \
  --min-seed-count 3 \
  --allowed-seed-selected-regression 0.01 \
  --allowed-slice-regression 0.05 \
  --output runs/impact-system-credibility-clean16k-token-diagnostic
```

Gate result for clean 16k-token SFT+DPO: `keep_baseline_or_collect_more_evidence`. It fails mean selected accuracy (`-0.009`), mean direct accuracy (`-0.013`), worst seed selected regression (`-0.017`), and worst slice regression (`-0.060`). This makes the long-context experiment useful as evidence, but not as a replacement for the then-current adapter.

## Source-Task SFT+DPO Result

The largest confirmed bottleneck after the long-context run was prompt/task mismatch. Earlier exports rewrote every SciImpact training example into a generic "which paper has higher scholarly impact" prompt. The held-out SciImpact tasks are dimension-specific: award, citation, downloads, media, patent, and stars. That means the training objective was partially misaligned with the benchmark even when the answer format was correct.

Implementation update:

- `export-sciimpact-pairwise-sft --prompt-style source-task` preserves the original SciImpact `system` and `human` task prompts.
- Targets are normalized to `Answer: A` / `Answer: B` so constrained local decoding remains compatible with the production pairwise judge.
- DPO export keeps the source system prompt instead of dropping it.
- Full source-task tokenization at `max_length=4096` truncated only `4/87984` records, with task prefix, prompt suffix, and answer retained.

Training:

- SFT data: `runs/sciimpact-source-task-sft-dpo/train_pairwise_sft.jsonl`
- DPO data: `runs/sciimpact-source-task-sft-dpo/train_pairwise_dpo.jsonl`
- SFT adapter: `runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k-4096-512step`
- DPO adapter: `runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k4096-dpo16k1024-128step`
- SFT config: `max_examples=16000`, `max_steps=512`, `max_length=4096`, `learning_rate=1e-5`, LoRA r8/alpha16
- DPO config: base/reference is the source-task SFT adapter, `max_examples=16000`, `max_steps=128`, `max_length=1024`, `learning_rate=1e-6`, LoRA r8/alpha16

Three-seed comparison against the previous recommended adapter:

| Run | Direct accuracy | Selected accuracy, margin 0.05 | Consistency | Status |
| --- | ---: | ---: | ---: | --- |
| Previous recommended SFT16k512+DPO16k256 | 0.634 | 0.662 | 0.631 | baseline |
| Source-task SFT+DPO | 0.659 | 0.669 | 0.789 | active default |

Per-seed source-task DPO results:

| Seed | Direct accuracy | Selected accuracy, margin 0.05 | Consistency |
| --- | ---: | ---: | ---: |
| 23 | 0.680 | 0.703 | 0.763 |
| 47 | 0.647 | 0.640 | 0.817 |
| 71 | 0.650 | 0.663 | 0.787 |

Credibility reports:

- Strict slice gate: `runs/impact-system-credibility-source-task-dpo/credibility_report.md`
- Disclosed slice-tolerance gate: `runs/impact-system-credibility-source-task-dpo-slice-tolerance006/credibility_report.md`

Decision: use `runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k4096-dpo16k1024-128step` as the active default local judge. The evidence is stronger than the previous adapter on mean direct accuracy and A/B consistency, with a modest selected-accuracy gain. The caveat is explicit: the stricter `0.05` slice-regression gate fails because seed71 `patent` regresses by `0.060`, while the disclosed `0.06` threshold passes. Reports must keep confidence, uncertainty, ranking-quality, and the dataset-local claim boundary visible.

Negative follow-ups:

- Slice-targeted DPO continuation on patent/downloads worsened mean direct and selected accuracy. Report: `runs/impact-system-credibility-source-task-slice-dpo/credibility_report.md`
- 4096-length DPO continuation did not beat the 1024-DPO source-task run on the first seed and had worse consistency, so it was not expanded to three seeds.
- Larger balanced source-task training also did not clear the promotion gate:
  - Strong SFT adapter: `runs/sciimpact-source-task-strong/checkpoints/qwen32b-source-task-sft32k-4096-1024step`
  - Strong SFT seed23 p50: direct `0.673`, selected `0.710`, consistency `0.730`
  - Strong SFT+DPO adapter: `runs/sciimpact-source-task-strong/checkpoints/qwen32b-source-task-sft32k4096-dpo32k1024-128step`
  - Strong SFT+DPO seed23 p50: direct `0.670`, selected `0.713`, consistency `0.730`
  - Active default on the same seed23 p50 sample: direct `0.680`, selected `0.703`, consistency `0.763`
  - Decision: do not expand to three seeds. The candidate gains only selected accuracy and regresses direct accuracy plus A/B consistency, so it is not a more credible default.

## Long-Context Training Result

The project tested whether the weak SciImpact quality was mainly caused by the old 1024-token training setup.

Finding:

- Prior Qwen32B LoRA SFT/DPO runs used `max_length=1024`.
- The released SciImpact LLaMA-Factory recipe uses `cutoff_len=4096`.
- Artifact tasks such as GitHub README, Hugging Face dataset cards, and model cards need longer context.
- The old local tokenization kept only the final `max_length` tokens, which could drop the task prefix and the beginning of Paper/Artifact A.

Implementation update:

- SFT and DPO tokenization now use head+tail prompt truncation while always preserving the response tokens.
- The retained prompt contains the task/pair prefix and the prompt suffix near the answer.
- `inspect-sft-tokenization` writes `sft_tokenization_report.json/md` with prefix/suffix/response retention rates before training.

The 16k-token preflight used for the clean run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli inspect-sft-tokenization \
  --model models/deepseek-r1-distill-qwen-32b \
  --train-file runs/sciimpact-pairwise-sft-full/train_pairwise_sft.jsonl \
  --output runs/sciimpact-long-context/tokenization-16384 \
  --max-length 16384 \
  --max-examples 5000
```

Tokenization result on the full SciImpact SFT training file:

| max_length | Truncated examples | Mean prompt retention | Interpretation |
| --- | ---: | ---: | --- |
| 1024 | 12718 / 87984 | 0.913004 | too much evidence loss |
| 4096 | 2461 / 87984 | 0.997316 | close to complete |
| 8192 | 4 / 87984 | 0.999996 | effectively complete |
| 16384 | 0 / 87984 | 1.000000 | no truncation |

Clean 16k-token training/evaluation:

| Run | Adapter | Mean direct accuracy | Mean selected accuracy, margin 0.05 | Mean consistency | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Clean 16k SFT | `runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step` | 0.634 | 0.643 | 0.673 | diagnostic |
| Clean 16k SFT+DPO | `runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-dpo16k-256step` | 0.621 | 0.653 | 0.613 | diagnostic |
| Previous recommended SFT16k512+DPO16k256 | `runs/sciimpact-large-sft-dpo/checkpoints/qwen32b-sft16k512-dpo-random16k-256step` | 0.634 | 0.662 | 0.631 | baseline |
| Source-task SFT+DPO | `runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k4096-dpo16k1024-128step` | 0.659 | 0.669 | 0.789 | active default |

Per-seed clean 16k-token results:

| Run | Seed | Direct accuracy | Selected accuracy, margin 0.00 | Selected accuracy, margin 0.05 |
| --- | ---: | ---: | ---: | ---: |
| Clean 16k SFT | 23 | 0.637 | 0.657 | 0.643 |
| Clean 16k SFT | 47 | 0.640 | 0.630 | 0.640 |
| Clean 16k SFT | 71 | 0.627 | 0.637 | 0.647 |
| Clean 16k SFT+DPO | 23 | 0.637 | 0.687 | 0.683 |
| Clean 16k SFT+DPO | 47 | 0.610 | 0.627 | 0.633 |
| Clean 16k SFT+DPO | 71 | 0.617 | 0.660 | 0.643 |

Decision: keep the 16k-token scripts and tokenization fix because they remove a real confound, but do not promote the clean 16k-token adapters. The available H100 budget is enough for 16k context, yet the benchmark bottleneck is not context length by itself. The next improvement route should keep sufficient context while changing the data/objective strategy: better preference data, hard-but-clean training examples, stronger local teacher/judge comparison, and evidence extraction for citation/artifact-adoption signals.

## Local Dataset Scoring Update

The project now supports a hybrid scoring path for user datasets:

```bash
python -m local_impact_ranking.cli model-score-dataset \
  --input papers.jsonl \
  --output runs/my-impact \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge-source-task-dpo \
  --direct-score \
  --pair-sampling coverage \
  --pair-budget auto \
  --target-paper-coverage 0.25 \
  --bidirectional-pairwise \
  --bidirectional-confidence-margin 0.05 \
  --calibration-report runs/sciimpact-qwen32b-confidence-heldout-p50-c4/calibration_eval/sciimpact_model_eval.json
```

Outputs:

- `direct_scores.jsonl`
- `direct_scores.csv`
- `pairwise_judgments.jsonl`
- `impact_scores.jsonl`
- `impact_scores.csv`
- `model_impact_report.json/md`

Implementation details:

- Direct per-paper score is constrained with vLLM `structured_outputs.choice` to calibrated bins: `Score: 0/20/40/60/80/100`.
- Pairwise comparison is constrained with vLLM `structured_outputs.choice` to `Answer: A` or `Answer: B`.
- Pairwise comparison requests choice logprobs when the local vLLM service supports them.
- `--pair-sampling coverage` is recommended for budget-limited local scoring because it spreads early comparisons across papers instead of only comparing the first records.
- `--pair-budget auto --target-paper-coverage X` derives `max_pairs` from the dataset size and records the effective budget in the report.
- `--bidirectional-pairwise` runs both A/B and B/A orientations, records `position_consistency`, and treats unresolved low-confidence conflicts as non-decisive.
- `ranking_quality` is written to the report and CSV output; sparse comparison budgets are marked `exploratory` with paper-coverage counts.
- Final `impact_index` is now produced from order-independent Rank Centrality aggregation by default, while `elo_rating` remains available for audit and legacy comparison.
- Direct score is written as a diagnostic/initial-prior column, not as the final calibrated score.
- `direct_score_quality` flags degenerate direct-score runs so a collapsed score distribution is not mistaken for a usable prior.
- `pairwise_judgments.jsonl` includes `confidence`, `confidence_method`, and `choice_logprob_margin`.
- `impact_scores.csv/jsonl` includes `coverage_confidence`, `mean_pairwise_confidence`, and combined `confidence`.
- With `--calibration-report`, `pairwise_judgments.jsonl` also includes `calibrated_confidence` and `calibrated_confidence_method`.
- With `--calibration-report`, `impact_scores.csv/jsonl` includes `mean_calibrated_pairwise_confidence` and `confidence_source`; final `confidence` uses calibrated pairwise confidence when available.

SciImpact benchmark confidence evaluation:

- `generate-sciimpact-predictions --answer-format constrained-answer` writes `confidence`, `confidence_method`, and `choice_logprob_margin`.
- `evaluate-sciimpact-predictions` writes `confidence_summary` and `by_confidence_bucket` into `sciimpact_model_eval.json/md`.
- The evaluator accepts both original ShareGPT `conversations` records and project-exported `messages + target` pairwise SFT records.
- Tiny real smoke: `/tmp/sciimpact-confidence-smoke`
  - examples: `8`
  - accuracy: `0.750`
  - coverage: `1.000`
  - scored examples: `8/8`
  - mean confidence: `0.655636`
  - correct mean confidence: `0.647991`
  - wrong mean confidence: `0.678570`
- Interpretation: the smoke verifies wiring, but raw logprob confidence is not yet calibrated. On this tiny sample, wrong predictions were slightly more confident than correct predictions.

Calibration metrics added:

- `sciimpact_model_eval.json/md` now includes:
  - expected calibration error (`ece`)
  - maximum calibration error (`mce`)
  - Brier score
  - overconfidence gap
  - five-bin calibration table
- Re-evaluated smoke: `/tmp/sciimpact-confidence-smoke/eval-calibrated`
  - ECE: `0.360095`
  - MCE: `0.453307`
  - Brier score: `0.228986`
  - overconfidence gap: `-0.094364`
- Interpretation: these metrics are now available, but the tiny smoke is not sufficient to calibrate thresholds. Next useful run is a larger stratified p50/p100 confidence evaluation.

Larger p50 confidence calibration run:

- Run: `runs/sciimpact-qwen32b-confidence-calibration-p50-c4`
- Sample: `runs/sciimpact-qwen32b-sft64-constrained-answer3-stratified-p50-c4/sample/sciimpact_sft_stratified.jsonl`
- Examples: `300`
- Accuracy: `0.643333`
- Coverage: `1.000`
- Mean confidence: `0.750796`
- ECE: `0.107463`
- MCE: `0.156286`
- Brier score: `0.233059`
- Overconfidence gap: `0.107463`
- Confidence buckets:
  - high: accuracy `0.756522`, mean confidence `0.912808`, count `115`
  - medium: accuracy `0.600000`, mean confidence `0.702178`, count `120`
  - low: accuracy `0.523077`, mean confidence `0.553916`, count `65`
- Interpretation: confidence is useful for ranking/selective filtering because high-confidence predictions are more accurate than low-confidence predictions, but raw probabilities are overconfident. Use calibrated thresholds or temperature scaling before exposing confidence as a trust score.

Empirical calibration layer:

- `evaluate-sciimpact-predictions` supports `--calibration-report PATH`.
- The calibration report supplies empirical bins from a prior evaluation.
- Each prediction row gets:
  - `calibrated_confidence`
  - `calibrated_confidence_method=empirical_bin_accuracy`
- Report-level additions:
  - `calibration_model`
  - `calibrated_confidence_summary`
- Same-sample implementation check:
  - output: `runs/sciimpact-qwen32b-confidence-calibration-p50-c4/model_eval/eval-calibrated`
  - raw mean confidence: `0.750796`
  - calibrated mean confidence: `0.643333`
  - same-sample accuracy: `0.643333`
- Interpretation: this proves the calibration layer works, but it is not a valid generalization estimate because the calibration source and evaluation target are the same sample. Next step should split calibration and validation samples.

Held-out calibration validation:

- Run: `runs/sciimpact-qwen32b-confidence-heldout-p50-c4`
- Split command: `split-sciimpact-calibration`
- Split: `150` calibration / `150` validation examples, stratified by impact dimension with `25` validation examples per dimension.
- Calibration split:
  - accuracy: `0.660`
  - raw ECE: `0.086021`
  - raw Brier: `0.221465`
- Validation split before calibration:
  - accuracy: `0.626667`
  - raw mean confidence: `0.755572`
  - raw ECE: `0.128905`
  - raw Brier: `0.244653`
- Validation split after applying empirical bin calibration:
  - calibrated mean confidence: `0.665748`
  - calibrated ECE: `0.039081`
  - calibrated Brier: `0.225445`
- Interpretation: held-out validation shows empirical bin calibration materially improves confidence calibration. For reporting, expose calibrated confidence as the trust score and keep raw confidence as a diagnostic/model signal.

Production scoring calibration smoke:

- Command: `model-score-dataset --calibration-report runs/sciimpact-qwen32b-confidence-heldout-p50-c4/calibration_eval/sciimpact_model_eval.json`
- Input: three SciCite records in `/tmp/local-impact-three-papers.jsonl`
- Output: `/tmp/local-impact-calibrated-confidence-smoke-small`
- Pairwise parseable count: `3/3`
- Output fields verified:
  - per-judgment `calibrated_confidence`
  - per-ranking `mean_calibrated_pairwise_confidence`
  - per-ranking `confidence_source=calibrated_pairwise`
- Interpretation: the formal local dataset scoring path now exposes calibrated confidence for reporting. The calibration source is SciImpact-derived, so it should be disclosed as an approximate external trust calibration until the user has domain-specific validation data.

Source-task DPO cross-check:

- Run: `runs/sciimpact-source-task-dpo-confidence-heldout-p50`
- Source predictions: `runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-seed23-p50/model_eval/predictions.jsonl`
- Split: `150` calibration / `150` validation examples, stratified by impact dimension with `25` validation examples per dimension.
- Raw validation:
  - accuracy: `0.633333`
  - raw ECE: `0.045373`
  - raw Brier: `0.217641`
- Same-run empirical bins overfit this split and worsen validation calibration:
  - calibrated ECE: `0.093333`
  - calibrated Brier: `0.227989`
- The older bundled SciImpact trust-prior profile remains useful for conservative reporting on the source-task DPO validation half:
  - calibrated ECE: `0.029138`
  - calibrated Brier: `0.219402`
- Interpretation: do not call the bundled profile a model-specific calibration for the source-task DPO adapter. It should remain documented as a reusable SciImpact-derived trust prior. For a new user corpus, domain-specific calibration labels are still the stronger evidence path.

Real local smoke:

- Input: three SciCite records in `/tmp/local-impact-three-papers.jsonl`
- Output: `/tmp/local-impact-direct-score-smoke-small`
- Direct score parseable count: `3/3`
- Pairwise parseable count: `3/3`
- Direct score quality: `degenerate`
- Observation: all direct scores were `0.0`, so direct scoring is structurally wired but not yet semantically useful with the current local R1/Qwen prompt/model. Pairwise constrained scoring is the reliable path for final `impact_index`.

Confidence smoke:

- Output: `/tmp/local-impact-confidence-smoke-small`
- Pairwise confidence method: `choice_logprob_margin` for all three judgments
- Judgment confidence range: about `0.706-0.755`
- `model_impact_report.md` now exposes both coverage confidence and mean pairwise confidence.

Bidirectional production scoring smoke:

- Command: `model-score-dataset --pair-sampling coverage --bidirectional-pairwise --bidirectional-confidence-margin 0.05 --calibration-report configs/calibration/sciimpact-qwen32b-heldout-p50.json`
- Served model: `local-impact-judge-sft1024-dpo64`
- Input: `runs/prescience-gpu5-1000/prescience.papers.jsonl`
- Output: `/tmp/local-impact-bidirectional-coverage-smoke`
- Pair budget: `6`
- Unique compared papers: `12`
- Position consistency counts:
  - consistent: `4`
  - conflict: `2`
- Decisive judgments: `6/6`
- Interpretation: the production local dataset path now supports the same position-sensitivity check used in SciImpact benchmarking, while `--pair-sampling coverage` prevents tiny pair budgets from only scoring the first few input records.

Ranking quality smoke:

- Command: `model-score-dataset --pair-sampling coverage --max-pairs 4 --calibration-report configs/calibration/sciimpact-qwen32b-heldout-p50.json`
- Served model: `local-impact-judge`
- Input: `runs/prescience-gpu5-1000/prescience.papers.jsonl`
- Output: `/tmp/local-impact-ranking-quality-smoke`
- Result:
  - `ranking_quality.status`: `exploratory`
  - compared papers: `8/1000`
  - paper coverage rate: `0.008`
  - decisive judgments: `4`
- Interpretation: the system now explicitly warns when the produced score table is only a smoke/partial ranking and not a reportable full-dataset influence index.

Auto pair-budget smoke:

- Command: `model-score-dataset --pair-budget auto --target-paper-coverage 0.01 --pair-sampling coverage --calibration-report configs/calibration/sciimpact-qwen32b-heldout-p50.json`
- Served model: `local-impact-judge`
- Input: `runs/prescience-gpu5-1000/prescience.papers.jsonl`
- Output: `/tmp/local-impact-auto-budget-smoke`
- Effective budget:
  - `effective_max_pairs`: `5`
  - target paper coverage: `0.01`
  - actual compared papers: `10/1000`
- Interpretation: users can now request an approximate dataset coverage target instead of hand-tuning `--max-pairs`.
