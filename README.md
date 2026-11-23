<!-- README_SYNC: source=c1cf7fddfa49326c543bacbe665fed5e7e1d03d0; updated=2026-11-23 -->
<p align="center">English · <a href="./README_CN.md" lang="zh-CN">简体中文</a></p>

# Local Impact Ranking

Turn a corpus of papers and references into an auditable ranking. The pipeline estimates influence **within the supplied corpus**, records the citation evidence behind each score, and flags papers whose evidence is too thin to support a firm conclusion.

<a href="https://github.com/LikC1606/local-impact-ranking"><img alt="GitHub stars" src="https://img.shields.io/github/stars/LikC1606/local-impact-ranking"></a>
<a href="https://github.com/LikC1606/local-impact-ranking/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/LikC1606/local-impact-ranking/actions/workflows/tests.yml/badge.svg"></a>
<img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
<a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2f855a"></a>

[Star this repository](https://github.com/LikC1606/local-impact-ranking) if you work on auditable bibliometrics or evidence-grounded paper evaluation.

## Run the offline demo

Use Python 3.11 or newer and Bash from the repository root. After installing the package, the demo needs no network, GPU, model weights, or model server. It uses tracked paper records and offline replayed pairwise judgments. The demo recreates its output directory, so give it a path you can overwrite.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
scripts/run_impact_demo.sh runs/impact-demo
```

Open `runs/impact-demo/demo_report.md` first. `demo_status.json` records whether the replay completed. Under `model-impact/`, these files explain the result:

| File | What to look for |
| --- | --- |
| `impact_scores.csv` | Paper rankings, `final_impact_score`, and the evidence behind each score |
| `impact_score_cards.md` | Per-paper evidence, confidence, and **Impact Dimension Coverage** |
| `impact_claim_certificate.md` | What the result supports and which claims still need evidence |
| `report_bundle/` | The scores, report, and provenance in one place |

For the complete offline workflow, follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md). `scripts/reproduce_no_gpu.sh` runs the tests, publication checks, and local examples. `scripts/run_quickstart.sh` shows the heuristic path without a model, and `scripts/run_deliverable_check.sh` checks the generated reports and bundles. The optional `CHECK_MODEL=1` path requires a separately configured model server.

## Use your own papers

Supply JSONL with one record per paper, including a stable `paper_id`, a title, and its references. Add citation contexts and source identifiers when available. Read [docs/input-schema.md](docs/input-schema.md) for the complete format. The normalizer deduplicates references within each citing paper and records the mapping in `metadata.reference_deduplication`.

With a local OpenAI-compatible model server already running, use the command below. It checks the input, estimates the number of model requests, scores the corpus, and produces a report bundle. Set `MODEL` to the model you actually serve. The repository does not include the model weights.

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact
```

Read `user_impact_summary.md` alongside `user_impact_status.json`. To reuse judgments for the same input and pair plan without another model call, set `PAIRWISE_JUDGMENTS` to the saved JSONL. The reporting workflow keeps corpus coverage, model compatibility, and validation evidence visible.

## Understand the score

`model-score-dataset` calculates a dataset-local `final_impact_score` from the rank-percentile signals for true citation counts and effective within-corpus citations. The default weights are 95% and 5%, respectively. True citations use one compatible count per paper after checking the paper's identity. Effective citations weight incoming edges by their context, including guideline, clinical, and method usage. A per-edge max rule keeps context and model evidence for the same edge from being counted twice.

Older `impact_index_v2` and `impact_index_v3` values, `relative_citation_ratio_score`, and citation outlier diagnostics remain available for auditing. The heuristic workflow also aggregates credit by author and institution. These scores depend on the input's coverage and are not global scientific impact measures. Agreement with a current external metric and validity on a future time window are separate questions. See [docs/scoring-method.md](docs/scoring-method.md), [docs/effective-citation-calculation.md](docs/effective-citation-calculation.md), and [docs/evaluation-method.md](docs/evaluation-method.md).

<details>
<summary>Command and evidence reference</summary>

- **Prepare and audit a run.** Use `doctor --markdown` or save `doctor.json` with `doctor --output`. `model-score-preflight` estimates `planned_pair_count` and `total_model_request_count`. `external-evidence-readiness` summarizes coverage in `external_evidence_readiness.json`; `citation-completeness-audit` writes `citation_completeness_audit.json` to distinguish supported zeros from missing or conflicting sources. `score-card`, `entity-resolution-report`, and `runs-index` inspect outputs; `report-bundle` assembles them.
- **Replay or resume comparisons.** `--pairwise-judgments` reads a complete saved set and records `pairwise_judgment_source.type=file`. `--resume-pairwise-judgments` fills missing pairs and reports `pairwise_judgment_source.type=resume`, `existing_count`, and `generated_count`. `--pairwise-journal` writes each new judgment to a JSONL checkpoint as it completes. `rank_centrality` is the default aggregation for the pairwise signal.
- **Add citation evidence.** Use `enrich-openalex` or `enrich-opencitations` to produce files such as `papers.openalex.jsonl` and `papers.opencitations.jsonl`. Other connectors include `enrich-semantic-scholar`, `enrich-europepmc`, `enrich-icite`, `enrich-ncbi-citations`, `enrich-crossref`, `enrich-crossref-data-citations`, `enrich-datacite`, `enrich-datacite-events`, `enrich-openaire`, `enrich-wikidata`, and `enrich-plos-metrics`. `AUTO_ENRICH=1` enables network lookups in the user workflow, which records `auto_enrich_status.json` and the final `scoring_input`. See [docs/citation-data-sources.md](docs/citation-data-sources.md).
- **Import existing evidence.** `external-evidence-template` and `import-external-evidence` support compatible citation records identified by paper ID, DOI, PMID, or PMCID, among other IDs. The source fields can retain uCite/NIH-OCC biomedical citation counts and Scopus/Web of Science/Dimensions/Lens/Google Scholar export counts. `scripts/apply_high_recall_evidence.sh` assembles an audit of imported evidence, including `citation_completeness_delta` and `evidence_validation`.
- **Check what can be reported.** Per-paper fields `evidence_quality_flag`, `credible_evidence_tier`, and `missing_high_value_evidence` expose evidence gaps. Compare `reportability_status` and its `reportability_basis` with `legacy_reportability_status` to see which decision rule was applied. Read `model_credibility_card.md` for `model_credibility_status` and `model_identity_match`, and `calibration_profile_card.md` for `calibration_status` and `calibration_model_identity_match`. The SciImpact-derived profile lists `compatible_models` and applies only to matching models. A mismatch or weak evidence may leave the `report_decision` at `exploratory_only`. The claim certificate summarizes the scope as `validated`, `evidence_supported`, `exploratory`, or `blocked`.
- **Read the report bundle.** Pass `--report-bundle` and `--report-bundle-output` to `model-score-dataset`. The main files are `bundle_manifest.json`, `model_impact_report.json/md`, and `impact_scores.csv/jsonl`. The **Model Score Dataset** report can include **Model Credibility** (`model_credibility_card.json/md`) and **Calibration Profile** (`calibration_profile_card.json/md`) sections and a `bundle_decision`.

The optional enrichment sequence below is the existing script default when enrichment is enabled. Its second Europe PMC pass runs after identity recovery.

```bash
AUTO_ENRICH_SOURCES=openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire
```

The bundled profiles are [configs/model_credibility/local-impact-judge-source-task-dpo.json](configs/model_credibility/local-impact-judge-source-task-dpo.json) and [configs/calibration/sciimpact-qwen32b-heldout-p50.json](configs/calibration/sciimpact-qwen32b-heldout-p50.json).

</details>

<details>
<summary>Scoring and sensitivity fields</summary>

```text
final_impact_score =
  0.95 * true_citation_score
+ 0.05 * effective_citation_score
```

Each component applies `log1p` to its count, then converts it to a rank percentile that preserves ties. Missing signals contribute zero and are listed in `final_score_missing_signals`. The selected count's source is recorded in `true_citation_source`. Compare `score_95_true_05_effective` with sensitivity columns such as `score_90_true_10_effective`, `score_70_true_30_effective`, and `score_50_true_50_effective` to see how different weights affect the ranking.

`impact_index_v2` and `impact_index_v3` are legacy ablation scores; the v2 recipe starts with `0.30 model_pairwise`. Use `calibrate-fusion-weights` with a v3 component table and pass its output back with `--fusion-weights` for diagnostic runs. Legacy SciImpact index-eval CSV files with only `impact_index` lack the components needed for calibration. Diagnostic weights do not affect `final_impact_score`.

</details>

<details>
<summary>Benchmarks and training</summary>

- **Current impact.** `benchmarks plan` lists `ready_now` benchmark options. `build-opcitance-icite-current-benchmark` joins citation context to NIH iCite labels and writes `opcitance_icite_benchmark_summary.md`. `evaluate-current-impact-credibility` writes `current_impact_credibility.json/md` for comparison against external current-impact metrics.
- **Future-window validity.** Run `apply-citation-cutoff` and `audit-citation-cutoff` before scoring. If a validity report lacks `citation_cutoff_audit.json`, project-readiness warns; when the audit fails, readiness blocks future-window validity claims. Use `derive-future-impact-labels` to prepare labels and `evaluate-impact-validity` or `backtest-impact-validity` to evaluate them. For comparisons across fields, use `--label-normalization field-year-percentile` in the backtest. `--validity-labels` supplies labels when scoring; the `impact_credibility_scorecard` reports coverage (`score_label_match_rate`, `coverage_examples`), bootstrap confidence intervals, a permutation random baseline, and outlier robustness checks.
- **Local judge.** [scripts/serve_recommended_impact_judge.sh](scripts/serve_recommended_impact_judge.sh) serves the `local-impact-judge-source-task-dpo` adapter with a separately installed base model, adapter weights, and vLLM. The recorded `credibility-report` comparison used `--min-seed-count 3`, `--allowed-seed-selected-regression 0.01`, and `--allowed-slice-regression 0.06`. On those held-out seeds, mean direct accuracy went from 0.634 to 0.659 and mean bidirectional selected accuracy from 0.662 to 0.669. The stricter 0.05 slice tolerance would have kept the previous adapter because of a patent-slice regression.
- **Evaluate a judge.** `split-prescience` prepares dataset splits, and `model-eval` evaluates the judge. Inspect the `pairwise_credibility_scorecard` and the `sciimpact_model_credibility_scorecard`, `sciimpact_model_credibility_summary`, `sciimpact_model_candidate_gate`, and `sciimpact_model_weak_slice_error_pack` outputs before choosing a candidate.
- **Optional training.** Use `inspect-sft-tokenization` to compare `8192` tokens with `--max-length 16384` if the model supports it. The earlier 1024-token LoRA experiments were context-limited diagnostics. For weak-slice training, the error pack can export `train_pairwise_sft.jsonl`, `train_pairwise_dpo.jsonl`, and `weak_slice_source_records.jsonl`. See [scripts/train_sciimpact_weak_slice_dpo.sh](scripts/train_sciimpact_weak_slice_dpo.sh) for the continuation script.

</details>

## Release boundary and license

Local datasets, model weights, and generated runs are excluded from Git. Review [docs/publish-checklist.md](docs/publish-checklist.md) before sharing a report or release, and see the [original technical guide at the source revision](https://github.com/LikC1606/local-impact-ranking/blob/c1cf7fddfa49326c543bacbe665fed5e7e1d03d0/README.md) for detailed examples and experiment records.

Licensed under the [MIT License](./LICENSE). [Open an issue](https://github.com/LikC1606/local-impact-ranking/issues) for reproducible bugs or evaluation feedback, and [star the repository](https://github.com/LikC1606/local-impact-ranking) to keep it handy.
