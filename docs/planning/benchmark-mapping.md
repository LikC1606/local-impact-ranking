# Benchmark Mapping for Local Impact Ranking

## Goal

Provide public benchmark options when the user has no private dataset, and map each benchmark to the correct part of the local pipeline.

## Recommended Order

Use the CLI to print the current machine-readable plan:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan --markdown
```

The JSON output includes `recommended_order`, `ready_now`, source URLs, local commands, metrics, and transfer-risk notes. Use it as the benchmark checklist before running a larger model comparison or `project-readiness`.

### 1. PreScience

Use for:

- Prior work selection.
- Impact prediction / reference importance ranking.
- A large AI-paper citation graph with realistic scale.

Map to pipeline:

- `metadata.prescience_positive_reference_ids` for gold reference-selection evaluation.
- Author `publication_history` as negative/neutral candidate references.
- `citation_contexts` only if full text or citation spans are available.
- `judge` for top-k influential references.
- `baselines` for citation count and graph metrics.
- Deterministically shuffle converted candidate references with `--shuffle-seed` to reduce gold-position leakage.

Why first:

- Closest to the current project goal.
- Already aligned with AI papers and citation graph structure.
- Public source: https://huggingface.co/datasets/allenai/prescience
- Current project status: ready.
- Metrics: `hit@k`, `mrr`, `ndcg@k`, `recall@k`.
- Smoke script: `scripts/run_prescience_smoke.sh` checks the cached sample target count and redownloads when it is smaller than `TARGETS`, so `TARGETS=100 LIMIT=100` cannot silently reuse a 30-target cache.

Current local benchmark observations:

- 30-target heuristic smoke: `hit_rate=0.933333`, `mean_ndcg=0.657533`, `position_bias_risk=low`.
- 100-target heuristic smoke: `hit_rate=0.84`, `mean_ndcg=0.601007`, `position_bias_risk=low`.
- 100-target baseline comparison: title-overlap baseline reaches `mean_ndcg=0.612931`, while the current heuristic judge reaches `mean_ndcg=0.601007`; `judge_beats_all_baselines=false`.
- After changing the heuristic judge to score `title + abstract`, the 100-target heuristic reaches `mean_ndcg=0.612931` and `hit_rate=0.85`, matching title-overlap but still not beating it.
- After changing the heuristic judge to IDF-weighted title/abstract overlap, the 100-target heuristic reaches `hit_rate=0.89`, `mean_ndcg=0.634523`, and `judge_beats_all_baselines=true`.
- Paired bootstrap check on the same 100-target run: judge vs title-overlap NDCG delta is `+0.021592`, but the 95% CI is `[-0.023404, 0.064868]`, so `statistically_positive_ndcg_delta=false`.
- 300-target IDF heuristic smoke: `hit_rate=0.836667`, `mean_ndcg=0.589646`, judge vs title-overlap NDCG delta `+0.028453`, 95% CI `[0.004613, 0.052724]`, and `statistically_positive_ndcg_delta=true`.
- Interpretation: the 30-target result was optimistic; the 100-target IDF result is better in mean but not statistically stable, while the 300-target result passes the current bootstrap stability gate.
- Current action from benchmark evidence: use the IDF heuristic as the no-GPU baseline, then require any local LLM/prompt/fine-tuned judge to beat it on held-out PreScience targets before promotion.

### 2. SCIDOCA 2025

Use for:

- Citation discovery.
- Citation sentence prediction.
- Masked citation prediction.

Map to pipeline:

- `judge` for candidate reference selection.
- `contexts` for citation-sentence evidence.
- `finetuning` for proxy-positive and pairwise examples.

Why second:

- Good fit for citation-context-aware judging.
- Useful for testing whether the model can identify missing or likely citations.
- Current project status: planned.
- Metrics: `recall@k`, `mrr`, `masked_citation_accuracy`.

### 3. SciCite / ACL-ARC / MultiCite

Use for:

- Citation intent classification.
- Background vs method vs result-comparison labeling.

Map to pipeline:

- `proxy_evaluation` for method-context support.
- `finetuning` for auxiliary intent classifier data.
- `judge` prompt conditioning.

Why third:

- Small, clean, and useful for training the model to tell meaningful citations from background citations.
- Public source: https://github.com/allenai/scicite
- Current project status: ready.
- Metrics: `context_support_rate`, `method_context_rate`, `label_coverage`.

### 4. SciImpact

Use for:

- Broader scientific impact evaluation.
- Pairwise comparison supervision.

Map to pipeline:

- `finetuning` for pairwise preference or ranking data.
- `reports` for pairwise impact evaluation by dimension and field.

Why fourth:

- Useful, but broader than citation-only influence.
- Public source: https://flypig23.github.io/sciimpact-homepage/
- Current project status: adapter_ready.
- Metrics: `pairwise_accuracy`, `non_tie_pairwise_accuracy`, `coverage_rate`, `tie_rate`, `field_macro_accuracy`.
- Local smoke: `scripts/run_sciimpact_smoke.sh runs/sciimpact-smoke`.
- Official SFT smoke: `scripts/run_sciimpact_sft_smoke.sh runs/sciimpact-sft-smoke`.
- No-GPU baseline: `LIMIT=63207 scripts/run_sciimpact_sft_baseline.sh runs/sciimpact-sft-baseline`; current full downloaded test result is `pairwise_accuracy=0.494`, `non_tie_pairwise_accuracy=0.507`, `coverage_rate=1.000` after skipping 396 malformed rows.
- DPO export: `LIMIT=88515 scripts/run_sciimpact_dpo_export_smoke.sh runs/sciimpact-dpo-export`; current downloaded train split yields 87,984 clean DPO preference records after skipping 531 malformed rows.
- Model-answer evaluation: `evaluate-sciimpact-predictions` compares local-model A/B answers against official SFT answers; `scripts/run_sciimpact_prediction_eval_smoke.sh` verifies the evaluator with oracle predictions.
- Full-data workflow: convert the downloaded pairwise JSONL or official ShareGPT/SFT file with `benchmarks convert-sciimpact`, run the normal ranking pipeline over the converted `papers.jsonl` or an ID-aligned private ranking, then run `evaluate-pairwise-impact` with `pairwise_labels.jsonl`.
- Data-quality note: the official SFT files contain a small number of malformed A/B prompts; the CLI skips them by default and reports the count in `manifest.json`.

### 5. BIP! DB / OpenCitations

Use for:

- Citation count baseline.
- PageRank baseline.
- Citation graph construction.

Map to pipeline:

- `baselines`.

Why fifth:

- They are not benchmark tasks by themselves, but they are the best comparison anchors.
- Public source: https://zenodo.org/records/10804822
- Current project status: external baseline.
- Metrics: `spearman`, `kendall_tau`, `top_n_overlap`, `field_year_normalized_correlation`.

## Task-to-Pipeline Matrix

| Benchmark | judge | contexts | baselines | proxy_eval | finetuning |
| --- | --- | --- | --- | --- | --- |
| PreScience | yes | optional | yes | yes | yes |
| SCIDOCA 2025 | yes | yes | yes | yes | yes |
| SciCite / ACL-ARC / MultiCite | limited | yes | limited | yes | yes |
| SciImpact | yes | optional | limited | yes | yes |
| BIP! DB / OpenCitations | no | no | yes | yes | yes |

## Practical Dataset Choice

If the user needs one dataset to start:

1. PreScience
2. SCIDOCA 2025
3. SciCite

If the user wants a baseline-only sanity check:

1. BIP! DB
2. OpenCitations
