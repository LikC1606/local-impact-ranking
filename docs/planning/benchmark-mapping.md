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

### 1. NIH iCite / RCR Current Impact

Use for:

- Public current-impact credibility evaluation.
- PubMed-indexed life-science datasets.
- Field/time-normalized article-level validation through Relative Citation Ratio.

Map to pipeline:

- Input papers need PMID values under `metadata.external_ids.pmid`, `metadata.pmid`, or top-level `pmid`.
- Run `evaluate-icite-current-benchmark` after generating `impact_scores.csv`.
- Use iCite `rcr` as the primary label and `citations_per_year`, `apt`, `nih_percentile`, and `icite_citation_count` as corroborating labels.
- Report PMID match coverage and do not claim high credibility when coverage is weak.

Why first:

- It is public, authoritative, and directly current-impact oriented for biomedical papers.
- It evaluates the actual `final_impact_score`, not only pairwise model answers or future prediction.
- Public source: https://icite.od.nih.gov/
- Method paper: https://doi.org/10.1371/journal.pbio.1002541
- Current project status: ready.
- Metrics: `spearman`, `kendall_tau`, `pairwise_accuracy`, `ndcg@k`, `coverage_rate`, `weight_sensitivity`.
- Local command: `PYTHONPATH=src python -m local_impact_ranking.cli evaluate-icite-current-benchmark --papers papers.jsonl --scores runs/my-impact/impact_scores.csv --output runs/my-impact/icite_public_benchmark`.

Limits:

- Biomedical/PubMed scope only.
- RCR and citation-count labels are not fully independent of true-citation evidence, so they validate current citation-normalized impact, not all social or translational impact.

### 2. OpCitance + NIH iCite Current Impact

Use for:

- Reproducing a public current-impact benchmark when the user has no private dataset.
- Testing both true citation and effective citation-context evidence in one biomedical corpus.
- Checking whether the final two-category score stays aligned with NIH iCite/RCR.

Map to pipeline:

- OpCitance rows provide citing PMID, cited PMID, and citation context sentences.
- The builder creates local `papers.jsonl`, internal reference edges, citation contexts, `impact_scores.csv`, iCite labels, and credibility reports.
- Use the generated `opcitance_icite_benchmark_summary.md` as the benchmark handoff artifact.

Why second:

- It is the first fully executable public benchmark for the current final score, including effective-citation coverage.
- It uses public citation contexts instead of only current citation labels.
- Public source: https://databank.illinois.edu/datasets/IDB-7312599
- Method paper: https://doi.org/10.1038/s41597-023-02134-x
- Current project status: ready.
- Metrics: `spearman`, `kendall_tau`, `pairwise_accuracy`, `ndcg@k`, `effective_citation_coverage`, `citation_context_count`, `weight_sensitivity`.
- Local command: `PYTHONPATH=src python -m local_impact_ranking.cli build-opcitance-icite-current-benchmark --input data/external/opcitance/opcitance_Q.txt --output runs/opcitance-icite-current-expanded --max-papers 1000 --candidate-limit 5000`.

Current local benchmark observations:

- 1000 papers, 998 RCR labels, 1224 internal reference edges, and 5442 citation contexts.
- `effective_citation_coverage_rate=0.713`.
- `final_impact_score` uses `0.95 true + 0.05 effective` after public weight calibration.
- `final_impact_score` vs RCR: Spearman `0.985`, Kendall `0.898`, pairwise accuracy `0.948`, NDCG@K `0.950`.
- Final score multi-metric report: `high_credibility`, grade `A`.
- Pure true citation remains the strongest RCR baseline, so effective citation should be reported as a conservative context adjustment and explanation signal, not as a standalone impact target.

Limits:

- Biomedical/PubMed/PMC scope only.
- iCite labels validate current citation-normalized impact, not future impact.
- RCR and citation-count labels are not fully independent of true-citation evidence.

### 3. PreScience

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

Why third:

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

### 4. SciCite / ACL-ARC / MultiCite

Use for:

- Citation intent classification.
- Background vs method vs result-comparison labeling.

Map to pipeline:

- `proxy_evaluation` for method-context support.
- `finetuning` for auxiliary intent classifier data.
- `judge` prompt conditioning.

Why fourth:

- Small, clean, and useful for training the model to tell meaningful citations from background citations.
- Public source: https://github.com/allenai/scicite
- Current project status: ready.
- Metrics: `context_support_rate`, `method_context_rate`, `label_coverage`.

### 5. SCIDOCA 2025

Use for:

- Citation discovery.
- Citation sentence prediction.
- Masked citation prediction.

Map to pipeline:

- `judge` for candidate reference selection.
- `contexts` for citation-sentence evidence.
- `finetuning` for proxy-positive and pairwise examples.

Why fifth:

- Good fit for citation-context-aware judging.
- Useful for testing whether the model can identify missing or likely citations.
- Current project status: planned.
- Metrics: `recall@k`, `mrr`, `masked_citation_accuracy`.

### 6. SciImpact

Use for:

- Broader scientific impact evaluation.
- Pairwise comparison supervision.

Map to pipeline:

- `finetuning` for pairwise preference or ranking data.
- `reports` for pairwise impact evaluation by dimension and field.

Why sixth:

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

### 7. BIP! DB / OpenCitations

Use for:

- Citation count baseline.
- PageRank baseline.
- Citation graph construction.

Map to pipeline:

- `baselines`.

Why seventh:

- They are not benchmark tasks by themselves, but they are the best comparison anchors.
- Public source: https://zenodo.org/records/10804822
- Current project status: external baseline.
- Metrics: `spearman`, `kendall_tau`, `top_n_overlap`, `field_year_normalized_correlation`.

## Task-to-Pipeline Matrix

| Benchmark | judge | contexts | baselines | proxy_eval | finetuning |
| --- | --- | --- | --- | --- | --- |
| NIH iCite / RCR Current Impact | no | no | yes | yes | no |
| OpCitance + NIH iCite Current Impact | no | yes | yes | yes | no |
| PreScience | yes | optional | yes | yes | yes |
| SCIDOCA 2025 | yes | yes | yes | yes | yes |
| SciCite / ACL-ARC / MultiCite | limited | yes | limited | yes | yes |
| SciImpact | yes | optional | limited | yes | yes |
| BIP! DB / OpenCitations | no | no | yes | yes | yes |

## Practical Dataset Choice

If the user needs one dataset to start:

1. NIH iCite / RCR for biomedical current-impact credibility.
2. OpCitance + NIH iCite when no private biomedical dataset is available.
3. PreScience for AI-paper reference-selection/judge evaluation.
4. SciCite
5. SCIDOCA 2025

If the user wants a baseline-only sanity check:

1. NIH iCite / RCR for PubMed-matched data.
2. BIP! DB
3. OpenCitations
