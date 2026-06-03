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

## Reproduce From A Fresh Clone

The public repository is reproducible without private datasets, generated `runs/`, model weights, a GPU, or a model server:

```bash
git clone https://github.com/LikC1606/local-impact-ranking.git
cd local-impact-ranking

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

scripts/reproduce_no_gpu.sh runs/reproduce-no-gpu
```

The script installs the package, runs the full test suite and publish checks, then runs the no-GPU deliverable workflow. Expected key outputs include:

- `runs/reproduce-no-gpu/reproduce_status.json`
- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/impact_scores.csv`
- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/impact_claim_certificate.md`
- `runs/reproduce-no-gpu/deliverable-check/user-impact/user_impact_summary.md`
- `runs/reproduce-no-gpu/deliverable-check/user-impact/model-impact/report_bundle/`

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact no-GPU, optional local-model, and optional training reproduction paths.

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

Run the no-GPU impact demo to see the project goal end to end:

```bash
scripts/run_impact_demo.sh runs/impact-demo
```

This uses `examples/impact_demo_papers.jsonl` plus offline replayed pairwise judgments, so it does not call a model server or use GPU. It writes `impact_scores.csv`, `model_impact_report.md`, `impact_score_cards.md`, `impact_claim_certificate.md`, `impact_validity/impact_credibility_scorecard.json`, `report_bundle/`, and `demo_status.json`. Treat it as the canonical local demo for "given a paper dataset, produce dataset-local impact scores with evidence and claim boundaries."

For a full handoff check, run the deliverable script:

```bash
scripts/run_deliverable_check.sh runs/deliverable-check
```

It runs the no-GPU quickstart, the SciImpact pairwise fixture, validates report bundles, and writes `deliverable_status.json`. If a local OpenAI-compatible model server is already running, include the optional model-score smoke:

```bash
CHECK_MODEL=1 \
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_deliverable_check.sh runs/deliverable-check-model
```

The optional model smoke runs `model-score-dataset` on the built-in context fixture with coverage sampling, bidirectional pairwise judging, calibrated confidence, and a report bundle.

Run your own paper dataset through the recommended local model path:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact
```

This validates the input, writes a model-score preflight estimate, checks the OpenAI-compatible local model server, runs `model-score-dataset` with coverage sampling, bidirectional pairwise judging, calibrated confidence, a pairwise checkpoint journal, and a report bundle, then writes `user_impact_status.json` and `user_impact_summary.md`. The summary includes claim boundaries, validation issue counts and examples, model benchmark credibility (`model_credibility_status` from `configs/model_credibility/local-impact-judge-source-task-dpo.json`), calibration credibility (`calibration_status` from `configs/calibration/sciimpact-qwen32b-heldout-p50.json`), `model_credibility_card.md`, `calibration_profile_card.md`, `model_identity_match`, `calibration_model_identity_match`, `report_decision`, citation completeness, top scores, evidence gaps, recommended next actions, and artifact paths. If the supplied model profile is missing, below reportable credibility, or does not match the served `MODEL`, `report_decision` is downgraded to `exploratory_only`; the same downgrade applies when the calibration profile is missing, too weak, or incompatible with the served model. The current recommended model profile is only moderate credibility, so it can be used with disclosure but not as a high-credibility standalone impact oracle. Defaults are conservative: `PAIR_BUDGET=auto`, `TARGET_PAPER_COVERAGE=0.25`, `BIDIRECTIONAL_PAIRWISE=1`, `AUTO_ENRICH=0`, and `CALIBRATION_REPORT=configs/calibration/sciimpact-qwen32b-heldout-p50.json`.

To let the user workflow look up missing citation evidence before scoring, enable automatic enrichment:

```bash
AUTO_ENRICH=1 \
AUTO_ENRICH_SOURCES=openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire \
OPENALEX_EMAIL=you@example.org \
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.jsonl runs/my-impact
```

`AUTO_ENRICH=1` runs the selected enrichment commands before validation, preflight, and scoring, then uses the final enriched JSONL as `scoring_input`. `europepmc` can recover PMID/PMCID from DOI and adds Europe PMC cited-by counts; `identity_recovery` checks whether external titles match the local paper title, quarantines wrong-paper DOI/PMID evidence, and tries to recover a high-confidence OpenAlex title match; `crossref` adds DOI-level `is-referenced-by-count`; `crossref_data_citations` adds Crossref Data Citations counts for data/software DOI impact evidence; `icite` uses explicit PMID values from `metadata.external_ids.pmid`, `metadata.pmid`, or top-level `pmid` and writes PubMed/iCite citation counts, RCR, APT, NIH percentile, and translation-related iCite fields; `ncbi_citations` uses resolved PMID/PMCID values to query NCBI cited-in links; `datacite` adds DOI-level citation, view, and download counts for DataCite-registered outputs such as Zenodo records; `datacite_events` adds DOI-to-DOI incoming event counts from DataCite Event Data and only treats `references`/`cites` relations as citation evidence; `openaire` adds OpenAIRE citation-impact indicators. Numeric local `paper_id` values are not treated as PMIDs. Repeated sources in the chain use numbered intermediate files after the first occurrence, so resume mode does not accidentally reuse an earlier stage. The optional `enrich-wikidata` command can add public Wikidata citation counts, but it is not in the default chain because Wikidata Query Service can be slow or unavailable. The optional `enrich-plos-metrics` command adds PLOS article usage metrics for audit/external-adoption evidence and does not directly change the final true/effective citation score. The run writes `auto-enrich/auto_enrich_status.json`, `auto-enrich/auto_enrich_summary.md`, and reports `auto_enrich_enabled`, `auto_enrich_sources`, `auto_enrich_output`, and `scoring_input` in `user_impact_status.json` and `user_impact_summary.md`. The default remains `AUTO_ENRICH=0` so public no-GPU reproduction and replay runs stay offline and deterministic.

Useful overrides:

```bash
RUN_PRESET=balanced \
MAX_PAIRS=200 \
TARGET_PAPER_COVERAGE=1.0 \
CONCURRENCY=4 \
VALIDITY_LABELS=runs/my-impact/future-labels/future_impact_labels.jsonl \
RESUME_PAIRWISE_JUDGMENTS=runs/my-impact/model-impact/pairwise_judgments.jsonl \
PAIRWISE_JOURNAL=runs/my-impact/model-impact/pairwise_judgments.jsonl \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact-resumed
```

Use `PAIRWISE_JUDGMENTS=...` to replay saved model judgments without calling the pairwise model again. The claim boundary remains dataset-local paper influence unless external calibrated evidence and validity labels support stronger claims.

`RUN_PRESET=quick|balanced|full|custom` provides convenience defaults for model-scoring scale. `custom` preserves explicit environment variables and is the default; `quick` is for cheap preflight/smoke runs, `balanced` is for ordinary pilot runs, and `full` targets full paper coverage unless `MAX_PAIRS` or `TARGET_PAPER_COVERAGE` are set manually.

## Input

Use JSONL with one paper per line:

```json
{"paper_id":"p1","title":"Target Representation Paper","abstract":"representation learning","references":[{"reference_id":"r1","title":"Representation Learning","authors":[{"name":"Author A","affiliations":["Institution X"]}]}]}
```

References should include stable `reference_id` values when available. Missing reference IDs are filled during normalization. During `run` and `judge`, normalization also deduplicates repeated references within the same paper by DOI, OpenAlex/Semantic Scholar/PMID IDs, title-year-authors, or repeated `reference_id`; citation contexts pointing at removed duplicate IDs are remapped to the kept reference ID, source provenance is retained as `deduplicated_reference_evidence`, and the mapping is recorded in `metadata.reference_deduplication`. Citations from different target papers are not deduplicated, because they are real incoming impact evidence.

For the full schema, citation-context format, and benchmark gold-label conventions, see [docs/input-schema.md](docs/input-schema.md). For the scoring formulas and credibility boundaries, see [docs/scoring-method.md](docs/scoring-method.md). For the citation data source policy, see [docs/citation-data-sources.md](docs/citation-data-sources.md). For the exact life-science effective-citation calculation, see [docs/effective-citation-calculation.md](docs/effective-citation-calculation.md).

To add external citation evidence from OpenAlex before scoring:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-openalex \
  --input papers.jsonl \
  --output papers.openalex.jsonl \
  --email you@example.com \
  --resume
```

Use `papers.openalex.jsonl` as the input to `model-score-dataset`. The enrichment command uses DOI or OpenAlex IDs and writes `citation_count`, yearly citation counts, OpenAlex IDs, and `metadata.external_impact.field_year_percentile` when a work is matched. `--resume` reuses rows already present in the output JSONL, so interrupted large runs skip papers that were already enriched.

To add NIH iCite evidence for life-science records with PMID values:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-icite \
  --input papers.openalex.jsonl \
  --output papers.icite.jsonl \
  --summary-output papers.icite.summary.json \
  --resume
```

Use `papers.icite.jsonl` when PMID coverage is strong. The command writes `metadata.external_impact.pubmed_citation_count`, `metadata.external_impact.icite_citation_count`, `metadata.external_impact.relative_citation_ratio`, `metadata.external_impact.apt`, `metadata.external_impact.nih_percentile`, and compatible iCite translation fields. `--resume` reuses rows already present in the output JSONL and excludes their PMIDs from new iCite requests.

To recover PMID/PMCID values and add Europe PMC cited-by counts before iCite:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-europepmc \
  --input papers.openalex.jsonl \
  --output papers.europepmc.jsonl \
  --summary-output papers.europepmc.summary.json \
  --resume
```

To add NCBI ELink cited-in counts after PMID/PMCID recovery:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-ncbi-citations \
  --input papers.europepmc.jsonl \
  --output papers.ncbi-citations.jsonl \
  --summary-output papers.ncbi-citations.summary.json \
  --resume
```

`--resume` reuses rows already present in the output JSONL and excludes their PMID/PMCID identifiers from new NCBI ELink requests.

To add Crossref DOI-level cited-by counts:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-crossref \
  --input papers.europepmc.jsonl \
  --output papers.crossref.jsonl \
  --summary-output papers.crossref.summary.json \
  --resume
```

To add Semantic Scholar citation counts and influential-citation evidence:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-semantic-scholar \
  --input papers.openalex.jsonl \
  --output papers.semantic-scholar.jsonl \
  --batch-size 100 \
  --resume
```

Use Semantic Scholar as a source-specific signal, especially for `metadata.external_impact.semantic_scholar_influential_citation_count`; do not average it with OpenAlex or OpenCitations counts. Identifier lookups are split into `--batch-size` chunks so large datasets do not send one oversized request, and `--resume` reuses rows already present in the output JSONL. See [docs/citation-data-sources.md](docs/citation-data-sources.md) for the source comparison and reconciliation policy.

To add DataCite evidence for DataCite-registered DOIs such as Zenodo records:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-datacite \
  --input papers.semantic-scholar.jsonl \
  --output papers.datacite.jsonl \
  --summary-output papers.datacite.summary.json \
  --resume
```

To add OpenAIRE citation-impact indicators:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-openaire \
  --input papers.datacite.jsonl \
  --output papers.openaire.jsonl \
  --summary-output papers.openaire.summary.json \
  --resume
```

`--resume` reuses rows already present in the output JSONL, so interrupted large OpenAIRE enrichment runs skip papers that were already processed.

To add incoming DOI citation events from DataCite Event Data:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-datacite-events \
  --input papers.datacite.jsonl \
  --output papers.datacite-events.jsonl \
  --summary-output papers.datacite-events.summary.json \
  --resume
```

This command writes `metadata.external_impact.datacite_events_cited_by_count`, `datacite_events_references_count`, `datacite_events_cites_count`, and source-specific audit counts. `--resume` reuses rows already present in the output JSONL. Only `references` and `cites` relation types feed true-citation scoring.

To add Crossref Data Citations evidence for data/software DOI reuse:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-crossref-data-citations \
  --input papers.datacite-events.jsonl \
  --output papers.crossref-data-citations.jsonl \
  --summary-output papers.crossref-data-citations.summary.json \
  --resume
```

This command writes `metadata.external_impact.crossref_data_citation_count` and `true_citation_crossref_data_citations_raw`. It is most useful for dataset/software DOI records, and usually returns zero for ordinary article DOIs. `--resume` reuses rows already present in the output JSONL.

To optionally add Wikidata citation counts:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-wikidata \
  --input papers.datacite-events.jsonl \
  --output papers.wikidata.jsonl \
  --summary-output papers.wikidata.summary.json \
  --resume
```

Wikidata evidence writes `metadata.wikidata_entity`, `metadata.wikidata_title`, and `metadata.external_impact.wikidata_cited_by_count`. It is kept out of the default auto-enrichment chain because the public query endpoint can be rate-limited or slow. `--resume` reuses rows already present in the output JSONL.

To optionally add PLOS article usage metrics:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-plos-metrics \
  --input papers.wikidata.jsonl \
  --output papers.plos-metrics.jsonl \
  --summary-output papers.plos-metrics.summary.json \
  --resume
```

PLOS metrics write `metadata.plos_title`, `metadata.plos_journal`, `metadata.external_impact.plos_counter_total_all`, and `metadata.external_impact.article_usage_count`. These fields support audit/external-adoption diagnostics, not the final true/effective citation score. `--resume` reuses rows already present in the output JSONL.

To add DOI-level citation counts and recover dataset-internal citation graph edges from OpenCitations:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-opencitations \
  --input papers.openalex.jsonl \
  --output papers.opencitations.jsonl \
  --max-incoming-edges-per-paper 500 \
  --materialize-internal-edges \
  --resume
```

Use `papers.opencitations.jsonl` as the scoring input when DOI coverage is good. The command writes `metadata.external_impact.opencitations_cited_by_count`, `metadata.external_impact.opencitations_reference_count`, and capped `metadata.opencitations_incoming_edges` for audit. If incoming edges are truncated, it records `metadata.opencitations_incoming_edges_truncated_count`; with `--materialize-internal-edges`, DOI-matched internal references are still added so `citation_graph_score` and PageRank can use OpenCitations evidence. `--resume` reuses rows already present in the output JSONL and can materialize saved internal incoming edges without refetching that DOI.

To merge external evidence you already exported from OpenAlex, BIP!, Semantic Scholar, uCite, NIH-OCC, Scopus, Web of Science, Dimensions, Lens, Google Scholar, patent, policy, media, award, or artifact-adoption sources:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-template \
  --input papers.opencitations.jsonl \
  --output external_evidence_template.csv
```

The generated CSV pre-fills any citation counts already present in the input and leaves missing source columns blank. Fill the blank columns with exported counts such as `scopus_citation_count`, `web_of_science_citation_count`, `dimensions_citation_count`, `ucite_citation_count`, or `nih_occ_citation_count`. Keep the identifier columns unchanged when possible; they are used for deterministic matching.

```bash
PYTHONPATH=src python -m local_impact_ranking.cli import-external-evidence \
  --input papers.opencitations.jsonl \
  --evidence external_evidence_template.csv \
  --output papers.external.jsonl \
  --summary-output runs/external-import-summary.json
```

The importer matches by `paper_id`, DOI, OpenAlex ID, Semantic Scholar/S2 ID, PMID, or PMCID and writes compatible fields under `metadata.external_impact`. Supported high-value fields include OpenAlex field/year percentile, BIP impact evidence, Semantic Scholar influential citation counts, uCite/NIH-OCC biomedical citation counts, Scopus/Web of Science/Dimensions/Lens/Google Scholar exported citation counts, patent and policy citations, media and award counts, GitHub/software adoption, dataset reuse, and model/download evidence. These imported citation counts remain source-specific audit evidence; the scorer takes the maximum compatible true-citation source and never sums overlapping databases.
The import summary includes `evidence_validation`, which reports evidence rows without match keys, duplicate match keys, invalid numeric cells, and negative numeric cells before scoring.

For a filled external-evidence CSV, run the import, citation completeness audit, and optional no-GPU score replay in one step:

```bash
PAIRWISE_JUDGMENTS=runs/my-impact/pairwise_judgments.jsonl \
  scripts/apply_high_recall_evidence.sh \
    papers.opencitations.jsonl \
    external_evidence_template.csv \
    runs/high-recall-evidence
```

When `PAIRWISE_JUDGMENTS` is set, the script replays existing judgments and rewrites `impact_scores.csv`, score cards, the claim certificate, citation completeness reports, and a report bundle without calling the local model. It also writes before/after citation-completeness audits and `high_recall_external_evidence_status.json`, whose `citation_completeness_delta` field shows whether the filled evidence improved coverage or reduced risk.

Before claiming high-credibility impact scores, check external evidence coverage:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-readiness \
  --input papers.external.jsonl \
  --output runs/external-readiness \
  --markdown
```

This writes `external_evidence_readiness.json`, `external_evidence_readiness.md`, and `external_evidence_paper_coverage.csv` with DOI/OpenAlex ID coverage, OpenAlex/OpenCitations/BIP/Semantic Scholar enrichment coverage, citation and field/year percentile coverage, influential-citation coverage, patent/policy/artifact/media/award evidence coverage, materialized graph-edge counts, per-paper missing evidence labels, a missing-evidence count summary, and a decision of `ready_for_high_credibility_scoring`, `usable_with_disclosure`, or `insufficient_external_evidence`. The same readiness artifacts are also generated automatically by `run` under `reports/` and by `model-score-dataset` in its output directory.

To audit whether true citation counts may be incomplete, run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli citation-completeness-audit \
  --input papers.external.jsonl \
  --output runs/citation-completeness \
  --markdown
```

This writes `citation_completeness_audit.json`, `citation_completeness_audit.md`, and `citation_completeness_papers.csv`. It reports DOI/OpenAlex/Semantic Scholar/PMID/PMCID availability, source-specific citation counts, source spread/conflict, single-source-only papers, papers without query identifiers, a 0-1 `completeness_score`, `coverage_confidence`, recommended next source actions, and whether zero citations are supported by multiple sources. It does not prove all citations were found; it separates supported zeros from missing-source and single-source cases. The same artifacts are generated automatically by `model-score-dataset` and `run_user_impact.sh`.

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

This writes `validation.json` and `validation.md`, flagging duplicate paper IDs, empty reference lists, duplicate reference IDs within a paper, missing titles, and out-of-range years. The validate command reports the raw input; the scoring workflow normalizes and deduplicates same-paper duplicate references before judging and scoring.

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

PYTHONPATH=src python -m local_impact_ranking.cli inspect-sft-tokenization \
  --model models/deepseek-r1-distill-qwen-32b \
  --train-file runs/my-dataset-train/finetuning/format_sft.jsonl \
  --output runs/my-dataset-train/finetuning/tokenization-16384 \
  --max-length 16384

PYTHONPATH=src python -m local_impact_ranking.cli finetuning-preflight \
  runs/my-dataset-train

PYTHONPATH=src python -m local_impact_ranking.cli train-dpo \
  --model models/deepseek-r1-distill-qwen-32b \
  --train-file runs/my-dataset-train/finetuning/merged_dpo_preferences.jsonl \
  --output runs/my-dataset-train/checkpoints/dpo
```

`inspect-sft-tokenization` is a no-GPU training-readiness check. It reports whether long SFT prompts preserve the task prefix, the prompt suffix, and the full answer tokens at the selected `--max-length`. On 8x H100, prefer checking `8192` and `16384` before training; earlier 1024-token LoRA experiments are context-limited diagnostics. A later clean 16k-token SFT/DPO run removed truncation but did not beat the current recommended adapter, so context length alone is not treated as the active quality bottleneck.

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

## Local Model Impact Index

For a local Qwen-compatible OpenAI API server, score a paper dataset directly with pairwise model judgments:

Recommended local judge for the current project state:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8000 \
  scripts/serve_recommended_impact_judge.sh
```

This serves `local-impact-judge-source-task-dpo`, the current recommended LoRA adapter. It is trained from the SciImpact source-task prompts instead of rewriting all examples into a generic "higher scholarly impact" prompt. That keeps the benchmark task semantics visible during training: award, citation, downloads, media, patent, and stars questions are learned as distinct impact dimensions while the output is normalized to `Answer: A/B`.

Three held-out SciImpact p50 seeds show a stronger and more stable judge than the previous recommended SFT16k512+DPO16k256 adapter: mean direct accuracy `0.634 -> 0.659`, mean bidirectional selected accuracy `0.662 -> 0.669`, and mean A/B consistency `0.631 -> 0.789`. The improvement is still modest on selected accuracy, and the report discloses small `dataset` and `patent` slice regressions. Treat the adapter as the default for bidirectional scoring with uncertainty/reportability fields visible, not as a claim of global scientific impact.

The evidence bundle is written by:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli credibility-report \
  --baseline-run runs/sciimpact-qwen32b-sft16k-dpo16k-flip-consistency-stratified-p50-c4/model_eval \
  --baseline-run runs/sciimpact-qwen32b-sft16k-dpo16k-flip-consistency-stratified-p50-seed47-c4/model_eval \
  --baseline-run runs/sciimpact-qwen32b-thirdseed-p50-seed71/sft16k_dpo16k \
  --candidate-run runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-seed23-p50 \
  --candidate-run runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-seed47-p50 \
  --candidate-run runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-seed71-p50 \
  --seed-label seed23 --seed-label seed47 --seed-label seed71 \
  --baseline-model recommended-sft16k512-dpo-random16k256 \
  --candidate-model source-task-sft16k4096-dpo16k1024-128 \
  --min-seed-count 3 \
  --allowed-seed-selected-regression 0.01 \
  --allowed-slice-regression 0.06 \
  --output runs/impact-system-credibility-source-task-dpo-slice-tolerance006
```

Stricter `0.05` slice-regression gating keeps the previous adapter because seed71 `patent` regresses by `0.06`. The published default uses the source-task adapter with that explicit caveat because it improves every seed, improves mean direct accuracy more substantially, and greatly reduces A/B position sensitivity. New candidate adapters should still be judged with at least three held-out seeds, positive mean selected-accuracy delta, no direct-accuracy drop on average, no seed selected-accuracy regression worse than `0.01`, and a disclosed dimension/item-type slice-regression threshold.

Before a large model-scoring run, estimate the work without calling the model:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli model-score-preflight \
  --input papers.external.jsonl \
  --output runs/my-impact-preflight \
  --pair-sampling coverage \
  --pair-budget auto \
  --target-paper-coverage 0.25 \
  --bidirectional-pairwise \
  --direct-score
```

This writes `model_score_preflight.json/md` with `planned_pair_count`, resume coverage when `--resume-pairwise-judgments` is supplied, `missing_pair_count`, `pairwise_model_request_count`, `direct_model_request_count`, `total_model_request_count`, and a compact external-evidence readiness summary. Use it to size runtime and decide whether to reduce pair budget before starting GPU inference.

```bash
PYTHONPATH=src python -m local_impact_ranking.cli model-score-dataset \
  --input papers.external.jsonl \
  --output runs/my-impact \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge-source-task-dpo \
  --direct-score \
  --pair-sampling coverage \
  --pair-budget auto \
  --target-paper-coverage 0.25 \
  --bidirectional-pairwise \
  --bidirectional-confidence-margin 0.05 \
  --calibration-report configs/calibration/sciimpact-qwen32b-heldout-p50.json \
  --fusion-weights runs/my-calibration/fusion_calibration.json \
  --validity-labels runs/my-impact/future-labels/future_impact_labels.jsonl \
  --validity-label-column future_impact \
  --report-bundle
```

This writes `pairwise_judgments.jsonl`, `impact_scores.csv`, `impact_scores.jsonl`, `impact_score_cards.json/md`, `external_evidence_readiness.json/md`, `external_evidence_paper_coverage.csv`, `citation_completeness_audit.json/md`, `citation_completeness_papers.csv`, `impact_claim_certificate.json/md`, `impact_validity/impact_validity_evaluation.json/md` when `--validity-labels` is supplied, `direct_scores.csv/jsonl` when direct scoring is enabled, and `model_impact_report.json/md`.
With `--report-bundle`, it also writes a shareable bundle at `runs/my-impact/report_bundle` by default; use `--report-bundle-output PATH` to choose another location.

To reproduce or resume a long model-scoring run without calling the model again, replay the saved pairwise judgments:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli model-score-dataset \
  --input papers.external.jsonl \
  --output runs/my-impact-replayed \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge-source-task-dpo \
  --pairwise-judgments runs/my-impact/pairwise_judgments.jsonl \
  --report-bundle
```

Replay mode still recomputes `impact_scores.csv`, score cards, external-evidence readiness, the impact claim certificate, and the optional report bundle. The model report records `pairwise_judgment_source.type=file` and the judgment file path for audit.

For interrupted long runs, use resume mode instead of replay mode:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli model-score-dataset \
  --input papers.external.jsonl \
  --output runs/my-impact-resumed \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge-source-task-dpo \
  --resume-pairwise-judgments runs/my-impact/pairwise_judgments.jsonl \
  --pairwise-journal runs/my-impact/pairwise_judgments.jsonl \
  --pair-sampling coverage \
  --pair-budget auto \
  --report-bundle
```

Resume mode loads the existing judgments, skips already completed planned pairs, calls the model only for missing pairs, and writes a merged `pairwise_judgments.jsonl`. `--pairwise-journal` writes each newly generated judgment to a JSONL checkpoint immediately after completion; when the journal path is the same as the resume file, missing judgments are appended instead of truncating the file. The model report records `pairwise_judgment_source.type=resume`, `existing_count`, `generated_count`, and `pairwise_journal`.

The final report score is `final_impact_score`, a dataset-local 0-100 score using only two categories:

```text
final_impact_score =
  0.95 * true_citation_score
+ 0.05 * effective_citation_score
```

`true_citation_score` is the tied-rank percentile of `log1p(true_citation_raw)`. The raw count is selected from one disclosed source per paper across OpenAlex, Semantic Scholar, PubMed/iCite, uCite, NIH-OCC, Dimensions, Scopus, Web of Science, Lens, Google Scholar, OpenCitations, Crossref, Europe PMC, NCBI ELink, DataCite, OpenAIRE, Wikidata, or supplied `citation_count`. Source counts are never summed, because the same citing article can appear in multiple databases; the scorer uses the maximum compatible non-null source after identity checks and keeps every raw source field for audit. The citation completeness audit records whether the selected count is multi-source-compatible, single-source-only, conflicted, or unsupported by query identifiers. `effective_citation_score` is the tied-rank percentile of weighted incoming effective citations inside the supplied dataset. Effective citations use internal reference edges plus citation-context roles such as guideline, clinical, method/protocol, dataset/tool, supporting, negative/contrasting, background, or unknown. They can also use constrained ImpactRankings-style local-model selections from `metadata.impact_selected_references`; selected IDs must be real references of the citing paper and are fused with context evidence by a per-edge max rule instead of being added on top. The default `0.95/0.05` weighting is intentionally conservative: expanded OpCitance + NIH iCite checks showed effective-citation context is useful for explanation and sensitivity analysis, but heavier effective-citation weights reduce alignment with current RCR when context coverage is incomplete. The output also includes `score_100_true_00_effective`, `score_95_true_05_effective`, `score_90_true_10_effective`, `score_80_true_20_effective`, `score_70_true_30_effective`, `score_50_true_50_effective`, and `score_30_true_70_effective` for sensitivity analysis.

The local Qwen pairwise score is still available in `model_supported_score` / `impact_index` for diagnostics. `impact_index_v2` and `impact_index_v3` are retained as legacy ablation scores with graph, context, momentum, external-adoption, and model components, but they do not directly change `final_impact_score`. Legacy `impact_index_v2` uses `0.30 model_pairwise + 0.25 field_year_normalized_citation + 0.20 citation_graph + 0.15 citation_context_quality + 0.10 temporal_momentum`. Each row writes `true_citation_raw`, `true_citation_source`, `true_citation_score`, `effective_citation_raw`, `effective_citation_count`, `effective_citation_score`, `final_score_signal_count`, `evidence_quality_flag`, `credible_evidence_tier`, `missing_high_value_evidence`, `reportability_status`, and missing-signal fields so sparse data are visible instead of silently imputed.

`model_impact_report.md` includes **Impact Dimension Coverage**, mapping SciImpact-style dimensions to supported evidence: citation, patent, policy, media, award, downloads, and stars/artifact adoption. `impact_score_cards.md` gives per-paper score cards with component scores, raw evidence, citation outlier diagnostics, missing high-value evidence, confidence, and reportability.

`impact_claim_certificate.md` is the report-facing claim gate. It summarizes whether the run is `validated`, `evidence_supported`, `exploratory`, or `blocked`; whether validated impact accuracy can be claimed; whether high-credibility external evidence is present; the score table coverage; and the next actions needed to strengthen the claim. Use it as the first page of a results handoff before showing detailed score cards.

To fit legacy diagnostic fusion weights against a labeled benchmark or held-out development split, run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli calibrate-fusion-weights \
  --rankings runs/my-impact/impact_scores.csv \
  --labels data/pairwise_labels.jsonl \
  --output runs/my-calibration \
  --grid-step 0.1
```

Then pass `runs/my-calibration/fusion_calibration.json` back to `model-score-dataset` with `--fusion-weights`. The run report records `fusion.weight_source`, so calibrated and default diagnostic runs remain auditable. These weights affect legacy diagnostic scores, not the two-category `final_impact_score`.

`calibrate-fusion-weights` now fits `impact_index_v3_final_components`: model pairwise, field/year citation, RCR-like citation ratio, graph, citation context, influential citation, temporal momentum, and external adoption components are all eligible for calibration. It expects a `model-score-dataset` `impact_scores.csv` with v3 component score columns such as `model_pairwise_score`, `relative_citation_ratio_score`, `influential_citation_score`, and `external_adoption_score`. Legacy SciImpact index-eval CSV files with only `impact_index` are intentionally rejected because they cannot calibrate fusion weights.

The quality layer uses log-scaled citation normalization plus rank-percentile component scaling, leave-one-signal-out score intervals (`score_ci_low`, `score_ci_high`, `score_ci_width`), dominant-signal contribution share, and a per-paper `reportability_status` of `reportable`, `exploratory`, or `low_confidence`. Rank-percentile scaling keeps extreme maxima from compressing the rest of a component while preserving tied top evidence at 100. Final reportability uses `reportability_basis=impact_index_v3_evidence_first`; `legacy_reportability_status` keeps the earlier v2 signal-count decision for audit. `validity_confidence` is reported separately from `final_impact_score`: without future-window or external labels, the run-level `validated_impact_claim` remains `unvalidated`, so scores can be described only as evidence-supported estimates, not validated impact accuracy.

`confidence` combines comparison coverage with pairwise model confidence. When the built-in SciImpact-derived calibration profile is supplied, `confidence` uses empirical calibrated confidence, while `mean_pairwise_confidence` remains the raw logprob diagnostic. The bundled profile is an approximate trust prior from SciImpact held-out validation, not domain-specific ground truth for every dataset. Its profile lists `compatible_models`, and user runs generate `calibration_profile_card.md` with `calibration_model_identity_match`; an incompatible calibration profile downgrades `report_decision` to `exploratory_only`. On a held-out source-task DPO validation split, the same profile improved ECE from `0.045` to `0.029` with nearly unchanged Brier score (`0.218` to `0.219`), so it is useful as a conservative reporting prior but still not a substitute for validating your own corpus.

For budget-limited runs, prefer `--pair-sampling coverage`; it spreads early pairwise comparisons across papers instead of spending the first comparisons only on the first few records. `--bidirectional-pairwise` asks each pair in both A/B and B/A order, records position consistency, and treats low-confidence conflicts as non-decisive so they do not distort the pairwise score. The default `--pairwise-aggregation rank_centrality` is order-independent; `--pairwise-aggregation elo` is still available for legacy comparisons.

Use `--pair-budget auto --target-paper-coverage <0-1>` when you want the command to choose a pair budget from the dataset size. For example, `--target-paper-coverage 0.25` on 1000 papers selects about 125 pairwise comparisons and covers about 250 papers under coverage sampling.

`model_impact_report.json/md` and `impact_scores.csv` include `ranking_quality`. Sparse runs are marked `exploratory` with paper-coverage counts, so a small smoke run is not mistaken for a complete impact ranking.

To evaluate whether the current impact score is credible against current external metrics such as iCite RCR, citations/year, APT, NIH percentile, or current citation counts, run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-current-impact-credibility \
  --scores runs/my-impact/impact_scores.csv \
  --labels current_external_metrics.csv \
  --label-columns rcr,citations_per_year,apt,nih_percentile,icite_citation_count \
  --primary-label-column rcr \
  --top-fraction 0.1 \
  --bootstrap-samples 1000 \
  --permutation-samples 1000 \
  --output runs/my-impact/current_impact_credibility
```

This writes `current_impact_credibility.json/md`. It is not a future-impact prediction benchmark. The primary label is the main credibility gate; other current metrics are corroborating evidence, so a run cannot claim high credibility by cherry-picking the best-looking external metric after evaluation. The report checks current external-metric alignment, true/effective citation coverage, and weight sensitivity across `score_100_true_00_effective`, `score_95_true_05_effective`, `score_90_true_10_effective`, `score_80_true_20_effective`, `score_70_true_30_effective`, `score_50_true_50_effective`, and `score_30_true_70_effective`.

For a public high-credibility benchmark on life-science data, use NIH iCite/RCR directly. The input papers must contain PMID values under `metadata.external_ids.pmid`, `metadata.pmid`, or top-level `pmid`.

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-icite-current-benchmark \
  --papers papers.jsonl \
  --scores runs/my-impact/impact_scores.csv \
  --output runs/my-impact/icite_public_benchmark \
  --bootstrap-samples 1000 \
  --permutation-samples 1000
```

This writes `icite_current_metrics.csv`, `icite_current_metrics_summary.json`, `current_impact_credibility.json/md`, and `icite_current_benchmark_report.json`. The benchmark uses NIH iCite `relative_citation_ratio` as the primary current-impact gate and checks corroborating iCite metrics such as citations/year, APT, NIH percentile, and current citation count.

When you do not yet have private biomedical data, reproduce the public OpCitance + NIH iCite benchmark. OpCitance supplies citation-context edges, and iCite supplies current-impact labels:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli build-opcitance-icite-current-benchmark \
  --input data/external/opcitance/opcitance_Q.txt \
  --output runs/opcitance-icite-current-expanded \
  --max-papers 1000 \
  --candidate-limit 5000 \
  --bootstrap-samples 20 \
  --permutation-samples 20
```

The benchmark writes `opcitance_icite_papers.jsonl`, `impact_scores.csv`, `icite_current_metrics.csv`, `current_impact_credibility/`, and `opcitance_icite_benchmark_summary.md`. The current local 1000-paper run reports `high_credibility` / grade `A`, with `effective_citation_coverage_rate=0.713` and `final_impact_score` vs RCR Spearman `0.985`. The result supports the conservative default `0.95 true + 0.05 effective` score and treats effective citation as a context adjustment rather than a standalone impact target.

When you have future-window labels or an external continuous impact target, validate that the score is actually aligned with that target:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli apply-citation-cutoff \
  --input papers.openalex.jsonl \
  --output papers.cutoff-2022.jsonl \
  --cutoff-year 2022

PYTHONPATH=src python -m local_impact_ranking.cli audit-citation-cutoff \
  --input papers.cutoff-2022.jsonl \
  --output runs/my-impact/reports \
  --cutoff-year 2022

PYTHONPATH=src python -m local_impact_ranking.cli derive-future-impact-labels \
  --input papers.openalex.jsonl \
  --output runs/my-impact/future-labels \
  --start-year 2023 \
  --end-year 2025 \
  --normalization field-year-percentile

PYTHONPATH=src python -m local_impact_ranking.cli evaluate-impact-validity \
  --rankings runs/my-impact/impact_scores.csv \
  --labels runs/my-impact/future-labels/future_impact_labels.jsonl \
  --label-column future_impact \
  --top-fraction 0.1 \
  --bootstrap-samples 1000 \
  --permutation-samples 1000 \
  --output runs/my-impact/impact_validity
```

Or run the evidence-building steps together after scoring the cutoff dataset:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli backtest-impact-validity \
  --enriched-input papers.openalex.jsonl \
  --cutoff-input papers.cutoff-2022.jsonl \
  --rankings runs/my-impact/impact_scores.csv \
  --output runs/my-impact \
  --cutoff-year 2022 \
  --future-start-year 2023 \
  --future-end-year 2025 \
  --label-normalization field-year-percentile \
  --bootstrap-samples 1000 \
  --permutation-samples 1000
```

`apply-citation-cutoff` removes citation-history years after the cutoff and rewrites citation totals so the scoring input cannot leak future citation evidence. `audit-citation-cutoff` checks that the scoring input contains no post-cutoff citation years before you run the model. Writing it to `runs/my-impact/reports` makes `citation_cutoff_audit.json/md` part of `project-readiness` and `report-bundle`: if future-window validity exists but the audit is missing, readiness warns; if the audit fails, readiness blocks future-window validity claims. Score `papers.cutoff-2022.jsonl`, then validate its `impact_scores.csv` against labels from 2023-2025. `derive-future-impact-labels` reads `metadata.citation_counts_by_year` or compatible yearly citation-history fields and writes a future-window citation label. Prefer `--normalization field-year-percentile` or `--label-normalization field-year-percentile` for cross-field datasets; this keeps `future_impact_raw` while evaluating `future_impact` as a field/year percentile so large fields and older cohorts do not dominate the benchmark target. The labels file can also be a manually supplied JSONL or CSV with `paper_id` and a numeric label such as future normalized citations, BIP! impact class, future patent/policy counts, or a combined benchmark target. The report writes `impact_validity_evaluation.json/md` and `impact_credibility_scorecard.json/md` with `ranking_paper_count`, `label_paper_count`, `score_label_match_rate`, `coverage_examples` (`unmatched_score_ids` and `unmatched_label_ids`), `spearman_correlation`, `kendall_tau_b`, `pairwise_accuracy`, `ndcg_at_top_k`, top-k precision/recall, field slices, bootstrap confidence intervals, a permutation random baseline, `significant_vs_random`, outlier robustness checks, and a conservative credibility scorecard. The scorecard separates `high_credibility`, `moderate_credibility`, `low_credibility`, `not_credible`, and `insufficient_validation_data`, so a high point estimate cannot hide low coverage, random-like significance, weak field slices, or outlier-sensitive top-k quality. The project-readiness warns when a validity report is present but not statistically supported or has low `score_label_match_rate`, so a random-like or sparse-label score cannot be presented as validated impact accuracy. This is the preferred way to claim that `final_impact_score` has external or future-window validity; without such labels, the system should report evidence coverage and benchmark proxies, not validated impact accuracy.

## Public Benchmarks

When you do not have a private dataset yet, list candidate public benchmarks and their pipeline mappings:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks list
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks show prescience
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks plan --markdown
```

The current registry covers NIH iCite/RCR, OpCitance + NIH iCite, PreScience, SCIDOCA 2025, SciCite/ACL-ARC/MultiCite, SciImpact, and BIP! DB/OpenCitations. `benchmarks plan` prints `recommended_order`, `ready_now`, metrics, source URLs, transfer risks, and local commands. See [docs/planning/benchmark-mapping.md](docs/planning/benchmark-mapping.md) for the mapping plan.

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

Convert SciImpact-style pairwise impact JSONL into local benchmark files:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli benchmarks convert-sciimpact \
  --input data/sciimpact/train.jsonl \
  --output data/sciimpact/train
```

The adapter writes `papers.jsonl`, `pairwise_labels.jsonl`, and `manifest.json`. It accepts common pair formats such as `chosen/rejected`, `positive/negative`, and `paper_a/paper_b + winner`. It also parses the official SciImpact ShareGPT/SFT files, including:

- `data/sciimpact/source/sft_data/paper_impact_sft_train_clean.jsonl`
- `data/sciimpact/source/sft_data/paper_impact_sft_val_clean.jsonl`
- `data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl`

The CLI skips malformed SciImpact rows by default and records `skipped_pair_count` plus sample errors in `manifest.json`. Use `--strict` when you want conversion to fail on the first malformed row.

Run a small official-SFT conversion smoke after downloading SciImpact:

```bash
scripts/run_sciimpact_sft_smoke.sh runs/sciimpact-sft-smoke
```

Run a no-GPU text-prior baseline on an official SciImpact SFT prefix:

```bash
LIMIT=1000 scripts/run_sciimpact_sft_baseline.sh runs/sciimpact-sft-baseline
```

This converts the SFT prefix, writes `rankings/sciimpact_text_prior.csv`, and evaluates it with `evaluate-pairwise-impact`. On the full downloaded SciImpact test split, this simple baseline reached `pairwise_accuracy=0.494`, `non_tie_pairwise_accuracy=0.507`, and `coverage_rate=1.000` after skipping 396 malformed rows; treat it as a lower bound for local model fine-tuning, not as the final model result.

The current benchmark result is summarized in `docs/planning/current-benchmark-evaluation.md`.

Export official SciImpact SFT rows into DPO preference data for local fine-tuning:

```bash
scripts/run_sciimpact_dpo_export_smoke.sh runs/sciimpact-dpo-export
```

On the downloaded SciImpact train split, this writes 87,984 clean DPO preference records and skips 531 malformed rows. The output file is `finetuning/sciimpact_dpo_preferences.jsonl`, which can be passed to `validate-dpo-data` and then to `train-dpo`.

Evaluate local-model predictions against SciImpact SFT answers:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-sciimpact-predictions \
  --sft data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl \
  --predictions runs/my-sciimpact-model/predictions.jsonl \
  --output runs/my-sciimpact-model/eval
```

The predictions file is JSONL with `source_line` and `prediction`. Run the oracle smoke to verify the evaluator:

```bash
scripts/run_sciimpact_prediction_eval_smoke.sh runs/sciimpact-prediction-eval-smoke
```

If a local OpenAI-compatible model server is already running, generate predictions and evaluate them in one step:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=local-impact-judge LIMIT=200 \
  scripts/run_sciimpact_model_eval.sh runs/sciimpact-model-eval
```

`evaluate-sciimpact-predictions` writes `sciimpact_model_eval.json/md`, `sciimpact_model_credibility_scorecard.json/md`, row-level outputs, wrong examples, and error analysis. The scorecard gates direct model credibility on coverage, one-sided binomial significance against a random 0.5 A/B baseline, overall SciImpact accuracy, and dimension/item-type slice stability, so a model cannot be promoted from a small smoke or random-like slice result.

After evaluating multiple held-out seeds, summarize them into one report-facing gate:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli summarize-sciimpact-model-credibility \
  --eval runs/my-sciimpact-model/seed23/eval \
  --eval runs/my-sciimpact-model/seed47/eval \
  --eval runs/my-sciimpact-model/seed71/eval \
  --output runs/my-sciimpact-model/credibility_summary
```

This writes `sciimpact_model_credibility_summary.json/md` with pooled accuracy, mean/min seed accuracy, seed stability, weak dimension/item-type slices, and a final `high_credibility` / `moderate_credibility` / `not_credible` decision. Use the summary, not a single seed, for model-promotion claims.

Compare a candidate model against the current default before changing deployment recommendations:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli compare-sciimpact-model-candidate \
  --baseline-summary runs/current-default/credibility_summary \
  --candidate-summary runs/candidate-model/credibility_summary \
  --output runs/candidate-model/promotion_gate
```

This writes `sciimpact_model_candidate_gate.json/md`. The default gate requires at least three candidate evaluations, a pooled accuracy gain, no worst-seed accuracy regression, no material seed-stability regression, no additional weak slices, no material worst-slice regression, and no drop in cross-seed credibility status. Use `decision=promote` before replacing the recommended local judge; otherwise keep the current model and report the candidate as a negative experiment.

Export the weak-slice errors that explain the credibility warnings:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli export-sciimpact-model-weak-slice-errors \
  --eval runs/my-sciimpact-model/seed23/eval \
  --eval runs/my-sciimpact-model/seed47/eval \
  --eval runs/my-sciimpact-model/seed71/eval \
  --output runs/my-sciimpact-model/weak_slice_errors
```

This writes `sciimpact_model_weak_slice_error_pack.json/md` and `sciimpact_model_weak_slice_errors.jsonl`. By default the pack is marked `diagnostic_only`, because held-out/test errors should not be used as training data. Regenerate it from train-split predictions and pass `--training-source` before using rows as training candidates.

When the matching SFT files are available, add one `--sft` per `--eval` to export directly trainable rows:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli export-sciimpact-model-weak-slice-errors \
  --eval runs/my-train-eval/model_eval/eval \
  --sft runs/my-train-eval/sample/sciimpact_sft_stratified.jsonl \
  --training-source \
  --output runs/my-train-eval/weak_slice_training_candidates
```

This additionally writes `train_pairwise_sft.jsonl`, `train_pairwise_dpo.jsonl`, and `weak_slice_source_records.jsonl`, each annotated with `matched_weak_slices` and the model's wrong prediction metadata.

Run the small weak-slice DPO continuation only after the candidate data is prepared and the target GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/train_sciimpact_weak_slice_dpo.sh
```

The script validates `train_pairwise_dpo.jsonl` before training and defaults to `MAX_STEPS=32`, `MAX_LENGTH=4096`, and the current source-task DPO adapter as `BASE_ADAPTER`. Treat this as a controlled candidate experiment; promotion still requires held-out three-seed evaluation and the cross-seed credibility summary.

Evaluate an existing paper ranking against pairwise impact labels:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli evaluate-pairwise-impact \
  --rankings runs/my-run/rankings/papers.csv \
  --labels data/sciimpact/train/pairwise_labels.jsonl \
  --output runs/my-run/sciimpact_eval
```

This writes `pairwise_evaluation.json/md` and `pairwise_credibility_scorecard.json/md`. The evaluation reports pairwise accuracy, non-tie accuracy, coverage rate, tie rate, per-dimension/per-field metrics, and a scorecard based on coverage, one-sided binomial significance versus a random 0.5 baseline, benchmark accuracy, tie rate, and slice stability. Run the fixture smoke without downloading the full benchmark:

```bash
scripts/run_sciimpact_smoke.sh runs/sciimpact-smoke
```

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

`project_readiness.json` and `project_readiness.md` are the final quality-gate reports. They combine artifact completeness, validation, judge reliability, scoring config, ranking scope, gold/proxy evidence, external evidence readiness, stability, anomaly risk, fine-tuning preflight, and claim boundaries into a single decision: `reportable`, `needs_improvement`, or `blocked`. If external evidence is usable but not strong enough for high-credibility claims, project readiness remains reportable only with disclosure and is marked `needs_improvement`. They also report `evidence_level`: `gold_validated`, `proxy_supported`, `proxy_only_needs_review`, or `no_external_or_proxy_evidence`. The report includes a reproducibility summary from `run_manifest.json`: input hashes, normalized record count, top-k, judge/model, prompt version, scoring version, git revision, and dirty-worktree flag.

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

The bundle writes `README.md` and `bundle_manifest.json`, then copies the main report-facing artifacts. For full pipeline runs, it includes paper/entity rankings, `mvp_summary.md`, `benchmark_summary.md`, `project_readiness.md`, `impact_claim_certificate.md`, evaluation reports, evidence CSVs, `scoring_config.json`, `artifact_index.json`, and `run_manifest.json`. For `model-score-dataset` directories, it also includes `impact_scores.csv/jsonl`, `model_impact_report.json/md`, `impact_score_cards.json/md`, `model_credibility_card.json/md`, `calibration_profile_card.json/md`, `pairwise_judgments.jsonl`, external evidence readiness files, validity reports when present, and the claim certificate, with **Model Score Dataset**, **Model Credibility**, and **Calibration Profile** summaries in the bundle README. The manifest includes `bundle_decision`; for model-score runs this uses the model-score impact claim certificate instead of full-pipeline readiness, so a valid model-score handoff is not mislabeled as blocked because author/institution artifacts are absent. The Model Credibility and Calibration Profile sections include `served_model`, `model_identity_match`, and `calibration_model_identity_match`, preventing one model's benchmark or calibration profile from being reused for another local service.

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

To feed those constrained top-k key-reference selections into the two-category `final_impact_score`, attach the audit selections back to the paper JSONL and use that enriched file for `model-score-dataset`:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli attach-judge-selections \
  --input papers.jsonl \
  --judge-audit runs/qwen3/audit/judge_outputs.jsonl \
  --output papers.with-impact-selections.jsonl

PYTHONPATH=src python -m local_impact_ranking.cli model-score-dataset \
  --input papers.with-impact-selections.jsonl \
  --output runs/my-impact-with-effective-selections \
  --pairwise-judgments runs/my-impact/pairwise_judgments.jsonl \
  --report-bundle
```

The attached selections are stored under `metadata.impact_selected_references`. The effective-citation scorer accepts only selected IDs that are real references of the citing paper and papers in the supplied dataset. If citation-context evidence and selected-reference evidence both exist on the same edge, the scorer uses the stronger per-edge weight rather than adding both.

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
