# Local Impact Ranking

Local, auditable ImpactRankings-style pipeline for ranking papers, authors, and institutions from a paper dataset with references.

## Repository Scope

This repository contains the source code, tests, examples, scripts, and planning documents needed to reproduce the local impact-ranking pipeline.

It intentionally does not include local datasets, downloaded model weights, or generated experiment outputs. Those paths are ignored by git:

- `data/`
- `models/`
- `runs/`

The core package and CI test suite do not require GPU libraries. Local LLM serving and DPO LoRA training require a separate machine environment with model weights and training dependencies such as `torch`, `transformers`, `peft`, and `vllm`.

Before publishing or pushing to GitHub, run the checks in [docs/publish-checklist.md](docs/publish-checklist.md).

## Install

```bash
python -m pip install -e ".[dev]"
```

For direct development without installing:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli --help
```

Check the local project environment before running, publishing, or debugging GitHub upload issues:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli doctor --markdown
PYTHONPATH=src python -m local_impact_ranking.cli doctor --output runs/doctor
```

The second command writes `doctor.json` and `doctor.md` with Python, Git, optional `gh`, required-file, and publish-safety checks.

Run the no-GPU quickstart to verify the full local example pipeline:

```bash
scripts/run_quickstart.sh runs/quickstart
```

It runs `doctor`, the context-weighted heuristic example, score cards, entity-resolution diagnostics, a report bundle, and a runs index.

## Input

Use JSONL with one paper per line:

```json
{"paper_id":"p1","title":"Target Representation Paper","abstract":"representation learning","references":[{"reference_id":"r1","title":"Representation Learning","authors":[{"name":"Author A","affiliations":["Institution X"]}]}]}
```

References should include stable `reference_id` values when available. Missing reference IDs are filled during normalization.

For the full schema, citation-context format, and benchmark gold-label conventions, see [docs/input-schema.md](docs/input-schema.md).

If your data starts as CSV, convert it to the canonical JSONL format first:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli convert-csv \
  --input papers.csv \
  --output papers.jsonl
```

Supported CSV columns are `paper_id`, `title`, `abstract`, `year`, `venue`, `field`, `authors_json`, `references_json`, `citation_contexts_json`, and `metadata_json`. Nested columns must contain JSON arrays or objects using the same schema as the JSONL input.

Validate a dataset before running the judge:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli validate \
  --input papers.jsonl \
  --output runs/validation
```

This writes `validation.json` and `validation.md`, flagging duplicate paper IDs, empty reference lists, duplicate reference IDs within a paper, missing titles, and out-of-range years.

Before model improvement or benchmark reporting, cut a held-out split:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli split \
  --input papers.jsonl \
  --output data/my-dataset/splits \
  --train-count 700 \
  --dev-count 150 \
  --test-count 150
```

This writes `train.papers.jsonl`, `dev.papers.jsonl`, `test.papers.jsonl`, and `split_manifest.json`. The split is an ordered prefix split by explicit counts, so it is deterministic and auditable. Use train only for SFT/DPO data generation, dev for prompt/model selection, and test only for the final reported metrics.

Minimal model-improvement loop:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input data/my-dataset/splits/train.papers.jsonl \
  --output runs/my-dataset-train \
  --top-k 5 \
  --judge openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge

PYTHONPATH=src python -m local_impact_ranking.cli validate-dpo-data \
  --train-file runs/my-dataset-train/finetuning/merged_dpo_preferences.jsonl \
  --output runs/my-dataset-train/finetuning

PYTHONPATH=src python -m local_impact_ranking.cli finetuning-preflight \
  runs/my-dataset-train

PYTHONPATH=src python -m local_impact_ranking.cli train-dpo \
  --model models/deepseek-r1-distill-qwen-32b \
  --train-file runs/my-dataset-train/finetuning/merged_dpo_preferences.jsonl \
  --output runs/my-dataset-train/checkpoints/dpo
```

After training, serve the adapter or merged model separately and compare it against the base judge on `dev.papers.jsonl`. Only use `test.papers.jsonl` for the final reported comparison.

Each run also writes `finetuning/context_hard_negative_dpo.jsonl`, `finetuning/method_context_contrastive_dpo.jsonl`, `finetuning/merged_dpo_preferences.jsonl`, `finetuning/finetuning_readiness.json`, `finetuning/finetuning_readiness.md`, `finetuning/finetuning_preflight.json`, and `finetuning/finetuning_preflight.md`. Context hard negatives prefer references with citation-context evidence over selected references without context support. Method-context contrastive examples prefer method/data/baseline-context references over no-context references from the same target paper. `merged_dpo_preferences.jsonl` deduplicates all DPO sources, drops reversed pair conflicts and self-pairs, and caps pairs per target to reduce target-level overfitting; use it as the default DPO training file. Check readiness before training: it reports raw and effective SFT/DPO counts, SFT parse and empty-selection quality, DPO invalid-pair and reverse-conflict quality, target concentration, and recommends `collect_more_examples`, `sft_smoke`, `dpo_smoke`, or `dpo_train`. The preflight report combines readiness gates with DPO JSONL validation and says whether SFT smoke, DPO smoke, or full DPO can start without using GPU.

```bash
BASE_MODEL=local-impact-judge \
CANDIDATE_MODEL=local-impact-judge-candidate \
LIMIT=150 TOP_K=5 \
  scripts/compare_model_judges.sh \
  data/my-dataset/splits/dev.papers.jsonl \
  runs/my-dataset-dev-comparison
```

The comparison writes `benchmark_comparison.md`, `target_comparison.md`, `target_comparison.json`, `target_changes.csv`, `model_selection_decision.json`, and `model_selection_decision.md`. The default decision rule recommends moving to test only when dev has common labeled targets, positive `net_hit_change`, non-negative `mean_ndcg_delta`, `broken_count / common_targets <= 0.05`, and, when candidate `benchmark_summary.json` is available, both `judge_beats_all_baselines=true` and `statistically_positive_ndcg_delta=true`. Run the same command on `test.papers.jsonl` once for the final report only after dev says `promote_to_test`.

If base and candidate runs already exist, run the pure post-processing evaluator:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli model-eval \
  runs/my-dataset-dev-comparison/base \
  runs/my-dataset-dev-comparison/candidate \
  --output runs/my-dataset-dev-comparison/comparison
```

Index existing experiments into one matrix:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli runs-index \
  runs \
  --output runs/index
```

## Public Benchmarks

When you do not have a private dataset yet, list candidate public benchmarks and their pipeline mappings:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks list
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks show prescience
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan --markdown
```

The current registry covers PreScience, SCIDOCA 2025, SciCite/ACL-ARC/MultiCite, SciImpact, and BIP! DB/OpenCitations. `benchmarks plan` prints `recommended_order`, `ready_now`, metrics, source URLs, transfer risks, and local commands. See [docs/planning/benchmark-mapping.md](docs/planning/benchmark-mapping.md) for the mapping plan.

Convert a locally downloaded PreScience JSONL or Parquet shard into the local paper schema:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks convert-prescience \
  --input data/prescience/train-00000-of-00001.parquet \
  --output data/prescience/papers.jsonl \
  --limit 100 \
  --max-candidates 50 \
  --shuffle-seed 13
```

The adapter builds each target paper's candidate references from PreScience key references plus author publication history, stores gold positives in `metadata.prescience_positive_reference_ids`, and deterministically shuffles candidate order by default to reduce gold-position leakage in benchmark runs.

Create fixed PreScience train/dev/test splits directly from a downloaded shard:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks split-prescience \
  --input data/prescience/train-00000-of-00001.parquet \
  --output data/prescience/splits \
  --train-count 700 \
  --dev-count 150 \
  --test-count 150 \
  --max-candidates 50
```

This writes `train.papers.jsonl`, `dev.papers.jsonl`, `test.papers.jsonl`, and `split_manifest.json` with source hash, split IDs, candidate counts, gold-positive counts, max-candidate setting, and shuffle seed.

To fetch a bounded PreScience sample through the HuggingFace mirror and run a no-GPU benchmark pilot:

```bash
HF_ENDPOINT=https://hf-mirror.com TARGETS=100 LIMIT=100 \
  scripts/run_prescience_smoke.sh runs/prescience-smoke
```

Convert SciCite JSONL into the local paper schema:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks convert-scicite \
  --input data/scicite/train.jsonl \
  --output data/scicite/papers.jsonl
```

The adapter groups rows by `citingPaperId`, converts each `citedPaperId` into a candidate reference, and stores `context`, `sectionName`, and `label` as citation-context metadata.

Download and convert the official SciCite splits in one step:

```bash
scripts/download_scicite.sh data/scicite
```

This writes:

- `data/scicite/train.papers.jsonl`
- `data/scicite/dev.papers.jsonl`
- `data/scicite/test.papers.jsonl`

Run a no-GPU benchmark smoke on a small SciCite prefix:

```bash
scripts/run_benchmark_smoke.sh runs/benchmark-smoke
```

By default this uses `data/scicite/dev.papers.jsonl`, `LIMIT=5`, the heuristic judge, and `context_weighted` scoring.

## Run With The Heuristic Smoke Judge

The MVP uses a deterministic heuristic judge for smoke tests:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input papers.jsonl \
  --output runs/example \
  --top-k 5
```

Use strict validation to stop before judging if validation errors are found:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input papers.jsonl \
  --output runs/example \
  --top-k 5 \
  --strict-validation
```

Use `--limit N` for a quick prefix run without cutting a separate input file:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input papers.jsonl \
  --output runs/example-100 \
  --top-k 5 \
  --limit 100
```

The first implementation prioritizes reproducible intermediate artifacts, strict judge output parsing, and transparent scoring.

Scoring modes:

- `rank_weighted`: default `score_v1_rank_weighted`; selected references receive rank weights 1.0, 0.8, 0.6, 0.4, and 0.2.
- `context_weighted`: `score_v2_context_weighted`; rank weights are multiplied by citation-context support weights when citation contexts are available.
- `flat`: every valid selected reference receives one point.

Ranking CSV files include `score`, `core_score`, `expanded_score`, `context_score`, `graph_score`, `field_year_percentile`, `max_single_edge_share`, `final_score`, `confidence_flag`, `confidence_reason`, `normalized_score`, and `percentile`. `score` is the raw accumulated dataset-local influence credit, `core_score` is the strict top-5 signal, `expanded_score` uses log-discounted top-20 credit to reduce hard cutoff effects, `context_score` captures citation-context support, `graph_score` records the local PageRank prior, `field_year_percentile` reports within-field/year standing when metadata is available, `max_single_edge_share` flags scores dominated by one citation edge, `final_score` is the 0-100 fused paper influence score used to sort `papers.csv`, `confidence_flag` labels each paper as `stable`, `fragile`, or `low_evidence`, `normalized_score` rescales the raw score table's top score to 100, and `percentile` reports where the item sits within that ranking table.

For the scoring formulas, aggregation rules, audit artifacts, and limitations, see [docs/scoring-method.md](docs/scoring-method.md).

To see context weighting on a minimal fixture:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input examples/context_papers.jsonl \
  --output runs/context-example \
  --top-k 1 \
  --credit-mode context_weighted
```

The selected reference `r1` receives score `1.500000` because its citation context contains a method-support cue.

Run a no-GPU local smoke check for both built-in examples:

```bash
scripts/run_local_smoke.sh runs/local-smoke
```

This uses the heuristic judge, writes full run artifacts, and checks that the context example produces score `1.500000`.

Outputs:

- `runs/example/rankings/papers.csv`
- `runs/example/rankings/authors.csv`
- `runs/example/rankings/institutions.csv`
- `runs/example/rankings/fields.csv`
- `runs/example/rankings/years.csv`
- `runs/example/audit/judge_outputs.jsonl`
- `runs/example/audit/prompts.jsonl`
- `runs/example/audit/scoring_edges.csv`
- `runs/example/reports/validation.json`
- `runs/example/reports/validation.md`
- `runs/example/reports/artifact_index.json`
- `runs/example/reports/evaluation.json`
- `runs/example/reports/evaluation.md`
- `runs/example/reports/gold_evaluation.json`
- `runs/example/reports/gold_evaluation.md`
- `runs/example/reports/stability_report.json`
- `runs/example/reports/paper_stability_report.json`
- `runs/example/reports/paper_stability_report.csv`
- `runs/example/reports/paper_score_uncertainty_report.json`
- `runs/example/reports/paper_score_uncertainty_report.csv`
- `runs/example/reports/paper_explanations.json`
- `runs/example/reports/entity_explanations.json`
- `runs/example/reports/paper_evidence_report.json`
- `runs/example/reports/paper_evidence_report.csv`
- `runs/example/reports/mvp_summary.json`
- `runs/example/reports/mvp_summary.md`
- `runs/example/reports/project_readiness.json`
- `runs/example/reports/project_readiness.md`

`stability_report.json` checks whether the dataset-local paper ranking is sensitive to scoring choices or to one target paper's selections. It reports top-N overlap across scoring modes, rank agreement, and leave-one-target-out retention. This is a robustness signal, not human-validated ground truth.

`paper_stability_report.json` and `paper_stability_report.csv` show per-paper rank sensitivity under leave-one-target-out perturbation. `stability_band` is `stable`, `moderate`, `unstable`, or `insufficient_samples`; use it to spot papers whose influence score is dominated by narrow target coverage.

`paper_score_uncertainty_report.json` and `paper_score_uncertainty_report.csv` use target-level bootstrap resampling to estimate each top paper's local score interval, rank interval, zero-score rate, and uncertainty band. These are confidence diagnostics for the dataset-local score, not proof of true external impact.

`validation.json` and `validation.md` are also written by `run`, so every ranking output carries the same data-quality gate report used by the standalone `validate` command.

`run_manifest.json` records the validation summary (`can_run`, issue count, error count, and warning count) together with input hashes, judge settings, scoring config, environment, and hardware.

`artifact_index.json` lists the key run artifacts, whether each file exists, and its byte size.

`mvp_summary.json` is the machine-readable one-page summary. `mvp_summary.md` is the report-ready Markdown version. They collect claim scope, data readiness, scoring version, top paper/author/institution/field/year previews, judge reliability, proxy quality, anomaly flags, run-level stability, per-paper stability counts, bootstrap score-uncertainty counts, and a conservative larger-pilot readiness flag.

`project_readiness.json` and `project_readiness.md` are the final quality-gate reports. They combine artifact completeness, validation, judge reliability, scoring config, ranking scope, gold/proxy evidence, stability, anomaly risk, fine-tuning preflight, and claim boundaries into a single decision: `reportable`, `needs_improvement`, or `blocked`. They also report `evidence_level`: `gold_validated`, `proxy_supported`, `proxy_only_needs_review`, or `no_external_or_proxy_evidence`. The report includes a reproducibility summary from `run_manifest.json`: input hashes, normalized record count, top-k, judge/model, prompt version, scoring version, git revision, and dirty-worktree flag.

You can regenerate or print that decision for an existing run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli project-readiness runs/example --markdown
```

Create a compact report bundle for upload or sharing:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli report-bundle \
  runs/example \
  --output runs/example/report_bundle
```

The bundle writes `README.md` and `bundle_manifest.json`, then copies the main report-facing artifacts: paper/entity rankings, `mvp_summary.md`, `benchmark_summary.md`, `project_readiness.md`, evaluation reports, evidence CSVs, `scoring_config.json`, `artifact_index.json`, and `run_manifest.json`.

`evaluation.json` and `evaluation.md` report judge reliability metrics such as parse success, invalid selections, empty selections, duplicate selections, and top-k compliance. They are reliability checks, not human correctness labels.

`gold_evaluation.json` and `gold_evaluation.md` report hit rate, precision, recall, MRR, and NDCG when benchmark gold labels are available, such as PreScience `prescience_positive_reference_ids`.

For evaluation metric definitions, error analysis, benchmark comparison, and target-level comparison, see [docs/evaluation-method.md](docs/evaluation-method.md).

`baseline_comparison.json` compares raw and fused paper impact scores against citation count, PageRank, and field/year-normalized citation baselines. It reports top-N overlap plus Spearman and Kendall alignment for `final_score`.

`audit/judge_outputs.jsonl` stores each judge result with `prompt_sha256` and the candidate `reference_id` list, so model outputs can be traced back to the prompt and candidate set used for that paper.

`audit/prompts.jsonl` stores the exact prompt sent for each target paper, its `prompt_sha256`, and the candidate `reference_id` list. This makes prompt construction auditable even when judge results are replayed from an existing audit file.

`audit/scoring_edges.csv` is the score ledger. Each row records one valid target-paper to selected-reference edge, including rank, rank weight, context count, context weight, and final credit.

`paper_explanations.json` groups the score ledger by high-ranking referenced paper and lists the top target-paper edges that contributed to each score.

`paper_evidence_report.json` and `paper_evidence_report.csv` combine each scored paper's `final_score`, `confidence_flag`, `support_band`, `max_single_edge_share`, bootstrap score uncertainty (`score_ci_low`, `score_ci_high`, `score_ci_width`, `rank_span`, `zero_score_rate`, `uncertainty_band`), and top contributing citation edges. Use this as the main report-facing paper audit table.

`entity_explanations.json` explains author and institution rankings using the same score ledger, with credit split across reference authors and affiliations.

`score_cards.json` and `score_cards.md` turn the top paper evidence rows into report-facing cards with score interval, credibility level, source-edge count, concentration risk, and top supporting target papers.

`entity_resolution_report.json` and `entity_resolution_report.md` quantify author/institution metadata risks: reference-author ID coverage, affiliation coverage, same-name multiple-ID flags, same-ID multiple-name flags, institution variants, and whether entity rankings are `reportable` or only `exploratory`.

You can regenerate these helpers for existing artifacts:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli score-card \
  runs/example \
  --output runs/example/score_cards

PYTHONPATH=src python -m local_impact_ranking.cli entity-resolution-report \
  --input papers.jsonl \
  --output runs/example/entity_resolution
```

## Run With A Local LLM Server

Start a local OpenAI-compatible server with vLLM or SGLang.

For a smaller first pass, download and serve a 32B model:

```bash
scripts/download_model.sh \
  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  models/deepseek-r1-distill-qwen-32b
```

```bash
TENSOR_PARALLEL_SIZE=1 PORT=8000 \
  scripts/serve_vllm.sh models/deepseek-r1-distill-qwen-32b local-impact-judge
```

If the machine has several jobs already using GPUs, select a free GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=5 TENSOR_PARALLEL_SIZE=1 PORT=8000 \
  scripts/serve_vllm.sh models/deepseek-r1-distill-qwen-32b local-impact-judge
```

For 8x H100, download and serve a larger FP8 judge:

```bash
scripts/download_model.sh \
  Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  models/qwen3-235b-a22b-instruct-fp8
```

```bash
TENSOR_PARALLEL_SIZE=8 PORT=8000 \
  scripts/serve_vllm.sh models/qwen3-235b-a22b-instruct-fp8 local-impact-judge
```

The script uses the repository-tested vLLM launch path with `PYTHONNOUSERSITE=1`, `VLLM_USE_DEEP_GEMM=0`, `VLLM_DEEP_GEMM_WARMUP=skip`, `--max-model-len 8192`, and `--enforce-eager` by default. These defaults avoid local Python package leakage and DeepGEMM warmup failures observed in the shared `air` environment.

Check the server:

```bash
scripts/check_model_server.sh http://localhost:8000/v1
```

Then point the pipeline at the server:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli run \
  --input papers.jsonl \
  --output runs/qwen3 \
  --top-k 5 \
  --judge openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge \
  --temperature 0 \
  --repair-attempts 1
```

The local model must return strict JSON in the shape requested by the prompt. If the first output is malformed JSON or selects a `reference_id` outside the candidate list, `--repair-attempts` sends a repair prompt asking the model to rewrite the output as valid constrained JSON. Any remaining invalid selected `reference_id` values are recorded in `audit/judge_outputs.jsonl` and excluded from scoring.

## Large Runs With Resume

For larger datasets, run judging as its own resumable stage. The command appends to an audit JSONL file and skips paper IDs already present in that file:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli judge \
  --input papers.jsonl \
  --audit-output runs/qwen3/audit/judge_outputs.jsonl \
  --top-k 5 \
  --judge openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge \
  --temperature 0 \
  --repair-attempts 1 \
  --concurrency 16 \
  --max-retries 2
```

Use `--limit N` for a small pilot batch:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli judge \
  --input papers.jsonl \
  --audit-output runs/qwen3/audit/judge_outputs.jsonl \
  --top-k 5 \
  --limit 100
```

Each paper is isolated: if the model server fails for one paper, the command writes an audit record with an `error` field and continues. Re-running the same command skips completed paper IDs in the audit file.

After judging completes, score from the audit file without calling the model again:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli score \
  --input papers.jsonl \
  --output runs/qwen3 \
  --judge-audit runs/qwen3/audit/judge_outputs.jsonl
```

`run --judge-audit ...` is still supported, but `score` is the explicit no-model stage for replaying an existing audit file.

## Standard 100-Paper Pilot

When a local OpenAI-compatible model server is already running, use the pilot script to run a bounded judge batch, score from the resumable audit file, and verify that the key reports were written:

```bash
LIMIT=100 TOP_K=5 CONCURRENCY=4 \
  scripts/run_pilot.sh papers.jsonl runs/pilot-100
```

Defaults:

- `BASE_URL=http://localhost:8000/v1`
- `MODEL=local-impact-judge`
- `LIMIT=100`
- `TOP_K=5`
- `CONCURRENCY=1`
- `PYTHON_BIN=/root/miniconda3/envs/air/bin/python3.11`

The script writes:

- `runs/pilot-100/audit/judge_outputs.jsonl`
- `runs/pilot-100/audit/prompts.jsonl`
- `runs/pilot-100/audit/scoring_edges.csv`
- `runs/pilot-100/rankings/fields.csv`
- `runs/pilot-100/rankings/years.csv`
- `runs/pilot-100/reports/validation.json`
- `runs/pilot-100/reports/validation.md`
- `runs/pilot-100/reports/artifact_index.json`
- `runs/pilot-100/reports/evaluation.json`
- `runs/pilot-100/reports/evaluation.md`
- `runs/pilot-100/reports/proxy_evaluation.json`
- `runs/pilot-100/reports/baseline_comparison.json`
- `runs/pilot-100/reports/context_coverage.json`
- `runs/pilot-100/reports/data_readiness.json`
- `runs/pilot-100/reports/anomaly_report.json`
- `runs/pilot-100/reports/scoring_config.json`
- `runs/pilot-100/reports/stability_report.json`
- `runs/pilot-100/reports/paper_stability_report.json`
- `runs/pilot-100/reports/paper_stability_report.csv`
- `runs/pilot-100/reports/paper_score_uncertainty_report.json`
- `runs/pilot-100/reports/paper_score_uncertainty_report.csv`
- `runs/pilot-100/reports/paper_explanations.json`
- `runs/pilot-100/reports/entity_explanations.json`
- `runs/pilot-100/reports/mvp_summary.json`
- `runs/pilot-100/reports/mvp_summary.md`
- `runs/pilot-100/finetuning/*.jsonl`
- `runs/pilot-100/finetuning/context_hard_negative_dpo.jsonl`
- `runs/pilot-100/finetuning/method_context_contrastive_dpo.jsonl`
- `runs/pilot-100/finetuning/merged_dpo_preferences.jsonl`
- `runs/pilot-100/finetuning/finetuning_readiness.json`
- `runs/pilot-100/finetuning/finetuning_readiness.md`
- `runs/pilot-100/finetuning/finetuning_preflight.json`
- `runs/pilot-100/finetuning/finetuning_preflight.md`
- `runs/pilot-100/finetuning/llamafactory_*.yaml`
- `runs/pilot-100/run_manifest.json`

Verified local example:

- Input: `data/scicite/dev.papers.jsonl`
- Output: `runs/scicite-dev-deepseek32b-100c`
- Model: `local-impact-judge` served from `http://localhost:8000/v1`
- Metrics: `parse_success_rate=1.0`, `invalid_selection_rate=0.0`, `empty_selection_rate=0.0`
