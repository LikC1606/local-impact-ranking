# Local Impact Ranking Full Optimization Plan

## Goal

Turn the current local paper-impact ranking MVP into a complete, defensible project deliverable: analyze current weaknesses, ground the design in public methods and benchmarks, implement the missing engineering pieces, verify with tests and smoke runs, and leave clear report-ready documentation.

## Current Baseline

- The project already runs a dataset-local, content-aware impact ranking pipeline.
- It exports rankings, score evidence, baselines, stability/uncertainty reports, validation reports, model-comparison reports, fine-tuning datasets, LLaMAFactory configs, DPO loader validation, and no-GPU fine-tuning preflight reports.
- Current verified state before this full optimization pass: full test suite passed with `170 passed`; local smoke passed at `/tmp/local-impact-local-smoke-preflight-final`.
- Worktree contains many intentional project changes from prior implementation phases. Do not revert unrelated changes.

## Optimization Phases

- [x] Phase 1: Audit current project capabilities and remaining shortcomings.
  - Compare implementation against the project goal: local score, ranking, confidence, benchmark, model improvement, and reportability.
  - Identify concrete gaps that can still weaken credibility or usability.

- [x] Phase 2: Research evidence base.
  - Search current and method-defining papers for scientific impact prediction, influential citation classification, citation-context modeling, bibliometric baselines, graph ranking, uncertainty/robustness, and benchmark datasets.
  - Search GitHub for reusable benchmark/code assets.
  - Store decision-relevant findings in `findings.md`.

- [x] Phase 3: Produce an optimization design.
  - Write a concise design doc covering scoring, evaluation, training, benchmarks, confidence, CLI workflow, and final deliverables.
  - Decide what is required for the finished project versus deferred research work.

- [x] Phase 4: Implement the selected product gaps.
  - Prioritize features that improve reproducibility, credibility, and ease of use without requiring private data or GPU.
  - Keep tests close to each behavioral change.

- [x] Phase 5: Final verification and handoff.
  - Run targeted tests, full tests, and smoke scripts.
  - Produce a final project summary with commands, artifacts, limitations, and next optional GPU/model steps.

- [x] Phase 6: Model credibility hardening.
  - Use the source-task-aligned SciImpact export as the active training format.
  - Train a stronger Qwen32B local impact judge candidate with larger balanced source-task SFT/DPO runs.
  - Evaluate candidates against the active default using the same three held-out SciImpact seeds and credibility gate.
  - Promote only if the candidate improves selected accuracy and direct accuracy without undisclosed slice regressions; otherwise keep the current default and report the failed experiment.
  - Final non-GPU hardening: default pairwise aggregation now uses confidence-weighted Rank Centrality to remove Elo order sensitivity, with `elo_rating` retained for audit.

- [x] Phase 7: Final deliverable verification after model hardening.
  - Run full tests, publish readiness, no-model deliverable check, and model-backed deliverable smoke against the active recommended service.
  - Update README, planning docs, `findings.md`, and `progress.md` with the final adapter decision and evidence.
  - Leave a concise handoff summary with commands, artifacts, model limitations, and claim boundaries.
  - Verified no-GPU final hardening with full tests, publish readiness, and deliverable check; model-backed smoke remains optional and was not run because no model service was requested for this pass.

- [x] Phase 8: Evidence-first final scoring hardening.
  - Promote the final user-facing score from model-primary v2 fusion to evidence-first `impact_index_v3`.
  - Add optional external evidence signals: external field/year percentile, patent citations, policy citations, and software/data/model adoption.
  - Keep `impact_index_v2` for compatibility and audit, but sort/report by `impact_index_v3`.
  - Update schema, scoring docs, README, tests, and final deliverable checks.

- [x] Phase 9: External citation evidence ingestion.
  - Add OpenAlex enrichment for citation counts, yearly citation history, and field/year percentile evidence.
  - Add OpenCitations enrichment for DOI-level cited-by/reference counts and incoming citation rows.
  - Materialize DOI-matched OpenCitations edges into dataset-internal `references` so `citation_graph_score` and PageRank become evidence-backed when DOI coverage is available.
  - Document the enrichment order and output fields.

- [x] Phase 10: External evidence readiness gate.
  - Add a no-model readiness command before scoring/reporting.
  - Quantify identifier, enrichment-source, citation, percentile, patent, policy, artifact, and graph-edge coverage.
  - Return a conservative decision so high-credibility claims require enough external evidence rather than only a model score.

- [x] Phase 11: Responsible-metrics credibility upgrade.
  - Added an RCR-like relative citation signal using article citation rate divided by a supplied expected citation rate, with local field/year fallback only for sufficiently sized buckets.
  - Added self-citation and extreme citation outlier disclosure so high scores caused by abnormal evidence are visible.
  - Added tied-rank percentile scaling for fused component scores so extreme maxima do not compress all other papers in the same component.
  - Added a reportable impact-dimension taxonomy that maps SciImpact-style dimensions to local score signals and claim boundaries.
  - Added model-score evidence cards so the model-backed path has score, confidence, evidence, and missing-data disclosure.
  - Updated readiness/docs/publish checks and verified the focused responsible-metrics test set.

- [x] Phase 12: External evidence and validity integration hardening.
  - Added CSV/JSONL `import-external-evidence` so user-supplied OpenAlex, BIP!, Semantic Scholar, patent, policy, media, award, and artifact-adoption exports can be merged into paper metadata by `paper_id`, DOI, OpenAlex ID, or S2/Semantic Scholar ID.
  - Added Semantic Scholar-compatible influential citation scoring and BIP-compatible external impact scoring to `impact_index_v3`, score cards, readiness coverage, taxonomy, docs, and publish checks.
  - Added optional `model-score-dataset --validity-labels --validity-label-column` so supplied future-window or external labels produce `impact_validity/impact_validity_evaluation.json/md` inside the model-scoring run.
  - Updated README, scoring docs, input schema, and release gates so these evidence-first features are required documentation before publishing.
  - Final no-GPU verification passed with full tests, publish readiness, and deliverable check; model smoke remained skipped because no local model server was requested for this pass.

- [x] Phase 13: Validated-impact credibility gate.
  - Upgraded `calibrate-fusion-weights` from legacy v2-only calibration to `impact_index_v3_final_components`, fitting final weights across model pairwise, field/year citation, RCR-like citation ratio, citation graph, citation context, influential citation, temporal momentum, and external adoption components.
  - Kept legacy v2 calibration files loadable while allowing new calibration files to set `impact_index_v3_weights` and `evidence_first_weights` separately.
  - Added `validated_impact_claim` and per-paper `validity_confidence` so numeric scores are separated from validation evidence. Without future-window or external validity labels, model-scored rows are explicitly `validity_status=unvalidated`.
  - Updated README, scoring docs, tests, and publish checks so full v3 calibration and validation-confidence reporting are required for release readiness.
  - Verified with focused regression, full tests, publish readiness, and no-GPU deliverable check.

- [x] Phase 14: Final delivery gate closure.
  - Extended the optional model branch of `scripts/run_deliverable_check.sh` so model-backed deliverable smoke must prove `validated_impact_claim.status` appears in the model report and `validity_confidence` appears in `impact_scores.csv`.
  - Updated `calibrate-fusion-weights` CLI output to print the active calibration profile, so logs distinguish `impact_index_v3_final_components` from legacy/unknown profiles.
  - Re-ran focused regression, full tests, publish readiness, and no-GPU deliverable check after the final gate changes.

- [x] Phase 15: Report-facing impact claim certificate.
  - Added `impact_claim_certificate.json/md` as the first-page claim artifact for model-score and full-run outputs.
  - The certificate combines score coverage, external evidence readiness, validated-impact evidence, and project readiness into `validated`, `evidence_supported`, `exploratory`, or `blocked`.
  - Added the certificate to report bundles, deliverable checks, docs, and publish readiness so final reports cannot silently overclaim validated impact accuracy.

- [x] Phase 16: Model-score report-bundle hardening.
  - Extended `report-bundle` so `model-score-dataset` output directories are first-class deliverables, not only full pipeline runs.
  - The bundle now copies model score tables, model reports, impact score cards, pairwise judgments, external evidence readiness, optional validity reports, and the impact claim certificate.
  - The bundle README now summarizes the model-score run separately from benchmark/readiness summaries.

- [x] Phase 17: One-command model-score delivery.
  - Added `model-score-dataset --report-bundle` and `--report-bundle-output`.
  - This lets the primary user workflow produce scores, score cards, external-evidence readiness, claim certificate, and report bundle in one command.
  - Updated deliverable checks and README to document and exercise the one-command path.

- [x] Phase 18: Offline replay and reproducible rescoring.
  - Added `model-score-dataset --pairwise-judgments PATH`.
  - Existing pairwise model judgments can now be replayed without calling the model again, enabling long-run recovery, audit replay, and recalibration/fusion reruns.
  - The report records the pairwise judgment source so replayed runs remain distinguishable from fresh model inference.

- [x] Phase 19: Pairwise judgment resume mode.
  - Added `model-score-dataset --resume-pairwise-judgments PATH`.
  - Interrupted pairwise model-scoring runs can now continue from existing `pairwise_judgments.jsonl` files by generating only missing planned comparisons.
  - The merged judgment output and report source summary make resumed inference auditable.

- [x] Phase 20: Pairwise judgment journaling.
  - Added `model-score-dataset --pairwise-journal PATH`.
  - Newly generated model judgments are checkpointed immediately, so interrupted long runs leave usable recovery state.
  - Journal and resume mode can share the same path to append missing judgments and continue safely.

- [x] Phase 21: Model-score preflight sizing.
  - Added `model-score-preflight`.
  - The preflight report estimates planned pairwise comparisons, missing resume comparisons, bidirectional/direct model request counts, and external-evidence readiness without calling the model.
  - This lets users size GPU work before running local inference.

- [x] Phase 22: Impact validity credibility scorecard.
  - Added report-facing `impact_credibility_scorecard.json/md` next to every impact-validity evaluation.
  - The scorecard combines label coverage, statistical support, rank alignment, top-rank quality, field-slice stability, and outlier robustness into `high_credibility`, `moderate_credibility`, `low_credibility`, `not_credible`, or `insufficient_validation_data`.
  - Added leave-top-label, leave-top-score, and top-label-5-percent robustness checks so a high nDCG or point estimate cannot hide extreme-row sensitivity.
  - Integrated the scorecard into backtest outputs, artifact indexing, report bundles, project readiness, claim certificates, docs, and publish readiness.

- [x] Phase 23: Pairwise benchmark credibility scorecard.
  - Added `pairwise_credibility_scorecard.json/md` for SciImpact-style pairwise impact labels.
  - The scorecard uses coverage, one-sided binomial significance against a random 0.5 non-tie baseline, benchmark accuracy, tie rate, and dimension/field slice stability.
  - Integrated the scorecard into `evaluate-pairwise-impact`, SciImpact smoke/baseline scripts, docs, publish checks, and the deliverable check.
  - Re-ran the existing full SciImpact text-prior baseline with the new scorecard; it is correctly marked `not_credible` despite full coverage because pairwise accuracy is close to random and at least one slice is below random.

- [x] Phase 24: Direct model-answer credibility scorecard.
  - Added `sciimpact_model_credibility_scorecard.json/md` for `evaluate-sciimpact-predictions`.
  - The scorecard gates local model answer credibility on coverage, one-sided binomial significance against the random A/B baseline, overall SciImpact accuracy, and dimension/item-type slice stability.
  - Integrated the scorecard into SciImpact model-eval scripts, docs, publish checks, and smoke coverage.
  - Re-ran the three existing source-task DPO p50 prediction sets without GPU: seed23 accuracy `0.680`, seed47 `0.647`, seed71 `0.650`; all are `moderate_credibility`, because weak slices prevent a high-credibility model claim.
  - Verified with focused tests, full tests, publish readiness, and no-GPU deliverable check.

- [x] Phase 25: Cross-seed model credibility summary.
  - Added `summarize-sciimpact-model-credibility` to combine multiple `sciimpact_model_eval.json` reports into one report-facing model-evidence gate.
  - The summary reports pooled accuracy, mean/min seed accuracy, seed stability, weak dimension/item-type slices, and a final credibility decision.
  - Generated the real three-seed source-task DPO summary at `runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-three-seed-model-credibility`.
  - Result: pooled accuracy `0.659`, coverage `1.000`, status `moderate_credibility`; weak slices include `dataset`, `stars`, `github_repo`, `model`, `downloads`, and `patent`.
  - Updated README, evaluation docs, tests, and publish readiness so model-promotion claims use cross-seed evidence rather than a single favorable seed.

- [x] Phase 26: Weak-slice model error pack.
  - Added `export-sciimpact-model-weak-slice-errors` to collect wrong/uncovered rows from weak dimension and item-type slices across one or more model evaluations.
  - The pack writes `sciimpact_model_weak_slice_error_pack.json/md` and `sciimpact_model_weak_slice_errors.jsonl`.
  - Default use is `diagnostic_only`; users must pass `--training-source` for train-split predictions, preventing held-out/test errors from being silently reused for fine-tuning.
  - Generated the real source-task DPO three-seed diagnostic pack at `runs/sciimpact-source-task-sft-dpo/eval/source-task-dpo-three-seed-weak-slice-errors`.
  - Result: `178` weak-slice errors, including `13` high-confidence errors; biggest error buckets are `stars/github_repo`, `downloads`, `patent`, `model`, and `dataset`.

- [x] Phase 27: Train-split weak-slice training candidate export.
  - Extended `export-sciimpact-model-weak-slice-errors` with aligned `--sft` inputs.
  - When `--sft` is supplied, the command also writes `train_pairwise_sft.jsonl`, `train_pairwise_dpo.jsonl`, and `weak_slice_source_records.jsonl`.
  - Generated a real train-split candidate pack from `runs/sciimpact-hard-example-train-p100-dpo64`.
  - Result: `139` SFT records and `139` DPO records, all annotated with weak-slice and model-error metadata.
  - Validation passed: SFT `loaded=139 invalid=0`; DPO `loaded=139 invalid=0 duplicates=0 reverse_conflicts=0`.

- [x] Phase 28: Weak-slice DPO continuation script.
  - Added `scripts/train_sciimpact_weak_slice_dpo.sh`.
  - The script validates `train_pairwise_dpo.jsonl` before training, then runs a conservative 32-step DPO continuation from the current source-task DPO adapter.
  - Defaults: `MAX_EXAMPLES=139`, `MAX_STEPS=32`, `MAX_LENGTH=4096`, `LEARNING_RATE=5e-7`, `BETA=0.1`.
  - The script was syntax-checked and the DPO candidate data was validated without starting GPU training.
  - Promotion remains gated on held-out three-seed evaluation and cross-seed model credibility summary.

- [x] Phase 29: Weak-slice DPO continuation experiment.
  - Ran the 32-step weak-slice DPO continuation on GPUs `0,1,2,3`.
  - Adapter output: `runs/sciimpact-weak-slice-candidate/checkpoints/qwen32b-source-task-dpo-weak-slice-32step`.
  - Training completed with finite losses through step `32`; the recurring bitsandbytes GLIBC warning remained non-blocking.
  - Served the candidate adapter on GPU `6` at port `8020`, evaluated held-out p50 seeds `23/47/71`, then stopped the candidate service and released GPU `6`.
  - Candidate direct accuracies: seed23 `0.667`, seed47 `0.657`, seed71 `0.627`.
  - Candidate three-seed pooled accuracy `0.650`, below the source-task DPO baseline `0.658889`; do not promote this adapter.

- [x] Phase 30: Model candidate promotion gate.
  - Added `compare-sciimpact-model-candidate` to compare a candidate cross-seed credibility summary against the current recommended model summary.
  - The gate requires at least three candidate evaluations, a pooled accuracy gain, no worst-seed regression, no material seed-stability regression, no extra weak slices, no material worst-slice regression, and no credibility-status downgrade.
  - Generated a real gate report for the weak-slice DPO continuation at `runs/sciimpact-weak-slice-candidate/eval/weak-slice-dpo-promotion-gate`.
  - Decision: `do_not_promote`; pooled accuracy delta `-0.008889`, min seed accuracy delta `-0.020`.
  - This closes the model-improvement loop by making adapter promotion/rejection auditable rather than relying on training completion or one favorable seed.

- [x] Phase 31: Self-mined train-split weak-slice DPO experiment.
  - Used the current source-task DPO model itself to predict a 1200-row stratified train-split SciImpact sample (`PER_DIMENSION=200`, seed `20260531`) instead of mining from held-out errors.
  - The train-split diagnostic found 259 train-safe weak-slice errors, with valid SFT/DPO exports and no duplicate or reverse-conflict DPO pairs.
  - Weak train-split slices matched the held-out bottleneck family: `downloads`, `patent`, `stars`, `github_repo`, and `model`.
  - Trained `runs/sciimpact-source-task-selfmine-train-p200/checkpoints/qwen32b-source-task-dpo-selfmine-p200-64step` for 64 DPO steps on GPUs `0,1,2,3`; all logged losses were finite.
  - Served the candidate on GPU `6` at port `8021`, evaluated held-out p50 seeds `23/47/71`, then stopped the candidate service and released GPU `6`.
  - Candidate direct accuracies: seed23 `0.663`, seed47 `0.650`, seed71 `0.633`.
  - Candidate three-seed pooled accuracy `0.648889`, below the source-task DPO baseline `0.658889`; promotion gate decision `do_not_promote`.
  - Conclusion: targeted weak-slice DPO improves the worst eligible slice but lowers overall and worst-seed accuracy, so the current source-task DPO adapter remains the recommended local judge.

- [x] Phase 32: Refocus on product deliverable instead of more model tuning.
  - Added a no-GPU end-to-end impact demo: `scripts/run_impact_demo.sh`.
  - Added `examples/impact_demo_papers.jsonl`, `examples/impact_demo_pairwise_judgments.jsonl`, and `examples/impact_demo_future_labels.jsonl`.
  - The demo uses offline pairwise replay, not a model server, and writes `impact_scores.csv`, `model_impact_report.md`, `impact_score_cards.md`, `impact_claim_certificate.md`, `impact_validity/`, `report_bundle/`, and `demo_status.json`.
  - Added `demo_report.md` as a concise human-readable handoff report; it includes claim status, score boundaries, top scores, validation boundary, and key artifact paths.
  - The demo script now fails if `impact_claim_certificate.json["claim"]` lacks `status` or `reason`, preventing blank claim rows in handoff reports.
  - Verified the optional real-model smoke path with `CHECK_MODEL=1` against the running GPU4 `local-impact-judge-source-task-dpo` service; the deliverable check passed with `model_smoke=passed`.
  - Added `scripts/run_user_impact.sh`, the direct user-dataset entrypoint. It runs validation, model-score preflight, model scoring, report bundling, `user_impact_status.json`, and `user_impact_summary.md`; it supports replay, resume, pairwise journaling, validity labels, fusion-weight overrides, and `RUN_PRESET=quick|balanced|full|custom`. The summary now includes validation issues, `report_decision`, evidence gaps, and recommended next actions, not only top scores.
  - Added the publishable model credibility profile `configs/model_credibility/local-impact-judge-source-task-dpo.json` and a user-run `model_credibility_card.json/md`. The card exposes cross-seed SciImpact accuracy, weak slices, reportability with disclosure, and the standalone-model claim boundary; missing or low model credibility downgrades `report_decision` to `exploratory_only`.
  - Hardened the model credibility card with served-model identity matching. If `MODEL` does not match the supplied credibility profile, the card reports `mismatched_profile`, sets `model_identity_match=false`, and the user workflow downgrades `report_decision` to `exploratory_only`. This prevents one local model's benchmark evidence from being reused for another model service.
  - Added `model_credibility_card.json/md` to model-score report bundles so the delivered artifact contains the model evidence gate alongside scores and claim certificates.
  - Added `calibration_profile_card.json/md` for the confidence-calibration profile. The card checks sample size, ECE/MCE/Brier quality thresholds, `compatible_models`, and `calibration_model_identity_match`; missing, weak, or mismatched calibration evidence downgrades `report_decision` to `exploratory_only`.
  - Added `compatible_models` to the built-in SciImpact confidence calibration profile so the recommended `local-impact-judge-source-task-dpo` can use it explicitly rather than inheriting an old `source_model` name silently.
  - Integrated `scripts/run_user_impact.sh` into `scripts/run_deliverable_check.sh` so the release smoke now checks the real user-data entrypoint, not only the built-in demo script.
  - Updated `docs/planning/project-summary-report.md` as the current report-facing summary of method, implementation, validation, and claim boundaries.
  - This becomes the canonical local proof that the project satisfies the original workflow: given a paper dataset, produce dataset-local impact scores with evidence, uncertainty, validation boundaries, and a handoff bundle.

- [x] Phase 33: Public current-impact benchmark and conservative final-score weighting.
  - Added a reusable OpCitance + NIH iCite benchmark builder for biomedical current-impact validation.
  - Built a 1000-paper public benchmark from `data/external/opcitance/opcitance_Q.txt` with 1224 internal reference edges and 5442 citation contexts.
  - Validated the final score against NIH iCite RCR and related current-impact metrics.
  - Adjusted the default final score from `0.80 true + 0.20 effective` to `0.95 true + 0.05 effective`, because expanded public validation showed heavier effective-citation weights reduce RCR alignment when citation-context coverage is incomplete.
  - Kept `score_100_true_00_effective`, `score_95_true_05_effective`, `score_90_true_10_effective`, `score_80_true_20_effective`, `score_70_true_30_effective`, `score_50_true_50_effective`, and `score_30_true_70_effective` as sensitivity fields.
  - Updated the current-impact credibility gate so component diagnostic columns such as `effective_citation_score` are disclosed but do not by themselves fail weight-sensitivity gating.
  - Formalized OpCitance + NIH iCite as benchmark registry item 2 and documented the reproducible local command.
  - Final verification passed after registry/docs integration: full test suite `536 passed`; publish readiness `536 passed`, `Publish readiness checks passed.`

- [x] Phase 34: Public no-GPU reproduction verification.
  - Ran `scripts/reproduce_no_gpu.sh runs/reproduce-no-gpu-fresh-20260603` without skipping installation.
  - Verified the public clone-style path: editable install, publish readiness, quickstart, impact demo, user-impact workflow, SciImpact fixture, and backtest evidence.
  - `reproduce_status.json` reports `status=passed`, `publish_ready=passed`, and `deliverable_check=passed`.
  - `deliverable_status.json` reports `quickstart=passed`, `impact_demo=passed`, `user_impact=passed`, `sciimpact_fixture=passed`, and `backtest_evidence=passed`; `model_smoke=skipped` because this was the no-GPU path.

- [x] Phase 35: Automatic citation-evidence lookup in user workflow.
  - Added `AUTO_ENRICH=1` to `scripts/run_user_impact.sh`.
  - Supported sources are `openalex`, `icite`, `opencitations`, and `semantic_scholar`, reusing existing enrichment commands instead of duplicating API clients.
  - The final enriched JSONL becomes `SCORING_INPUT` for validation, preflight, and scoring.
  - Added auto-enrichment status artifacts under `auto-enrich/` and surfaced them in `user_impact_status.json` plus `user_impact_summary.md`.
  - Kept `AUTO_ENRICH=0` as the default so offline reproducibility remains deterministic.
  - Verified default replay and real OpenAlex auto-enrichment smoke paths.

- [x] Phase 36: NIH iCite enrichment for life-science citation evidence.
  - Added `enrich_papers_with_icite` and CLI command `enrich-icite`.
  - The enrichment maps PMID records to PubMed/iCite citation count, RCR, APT, NIH percentile, expected citation rate, field citation rate, Human/Animal/Molecular scores, and clinical-citation flags.
  - Integrated `icite` into `AUTO_ENRICH_SOURCES`, so user datasets with PMIDs can automatically fetch life-science current-impact evidence before scoring.
  - Real iCite API smoke matched `3/3` OpCitance PMIDs.
  - User workflow smoke with `AUTO_ENRICH_SOURCES=icite` passed.

- [x] Phase 37: Same-paper duplicate reference deduplication.
  - Added normalization-time deduplication for repeated references within the same target paper.
  - Deduplication keys are DOI, external IDs, title/year/authors, then repeated `reference_id`.
  - Citation context IDs are normalized, and contexts pointing to removed duplicate IDs are remapped to the kept reference ID.
  - Source provenance from removed duplicates is retained as `deduplicated_reference_evidence`.
  - Deduplication audit is written to `metadata.reference_deduplication`.
  - Cross-paper citations are not deduplicated, because they are valid incoming impact evidence.
  - Added a scoring regression test that counts a cross-source duplicate edge once.
  - Verified with full test suite: `544 passed`.
  - Publish readiness passed.

- [x] Phase 38: User 100-paper dataset verification.
  - Converted and enriched `/home/wyx/zh/local-impact-ranking/data_100.zip`.
  - Curated the MinerU output from 100 records to 85 article records by excluding supplements and extraction artifacts.
  - Repaired 3 bad extracted titles with OpenAlex display names.
  - Audited duplicate/self DOI evidence: 3820 references after curation, 1522 reference DOIs, 2 self DOI matches, and 0 cross-paper internal DOI matches.
  - Re-ran validation, external-evidence readiness, preflight, and no-model score bundle on the curated input.
  - Current claim: reportable dataset-local score with disclosure, not validated/high-credibility impact accuracy.

- [x] Phase 39: User 100-paper model-backed scoring.
  - Served `local-impact-judge-source-task-dpo` locally with vLLM on GPU5, scored the curated 85-paper dataset, then stopped the server and released GPU memory.
  - The first auto-budget run produced only 43 pairwise comparisons and was exploratory because the graph was disconnected.
  - Re-ran with `PAIR_BUDGET=manual MAX_PAIRS=200 TARGET_PAPER_COVERAGE=1.0`, resuming the 43 existing judgments and generating 157 additional judgments.
  - Final output: `runs/data-100-verification/user-impact-model-p200`.
  - Pair plan quality is reportable: all papers covered, one connected component, min 4 comparisons per paper.
  - Model pairwise evidence is usable with disclosure: 200 judgments, 190 parseable, pairwise stability `moderate`.
  - Final citation-impact score remains stable and citation-dominant; only 3 final scores are nonzero because the curated dataset has sparse current citations and no internal/effective citation edges.

- [x] Phase 40: Multi-source citation resolution for user-data scoring.
  - Added a true-citation resolver that reads OpenAlex, Semantic Scholar, PubMed/iCite, OpenCitations, Crossref, and supplied citation counts.
  - The scoring count uses the maximum available non-null source value to avoid false zero overrides, while all source-specific raw values are retained in `impact_scores.csv/jsonl`.
  - Added audit fields: `true_citation_*_raw`, `true_citation_source_count`, `true_citation_positive_source_count`, `true_citation_source_agreement`, and `true_citation_source_conflict_flag`.
  - `supplied_citation_count` is treated as a fallback or higher-than-source override, not as a duplicate independent source when it merely repeats a named source.
  - Merged the 85 curated user records with Semantic Scholar, NIH iCite, OpenCitations/OpenAlex, and Crossref cited-by evidence into `runs/data-100-verification/papers.enriched.multi_source.jsonl`.
  - Replayed the existing 200 model judgments without GPU at `runs/data-100-verification/user-impact-model-p200-multisource-replay/model-impact`.
  - Multi-source lookup initially found 5 nonzero true-citation papers instead of the earlier OpenAlex-priority 3; follow-up identity guarding later showed that 2 of those 5 were wrong-paper DOI/PMID matches.

- [x] Phase 41: Extra biomedical citation sources and false-zero hardening.
  - Stopped treating local numeric `paper_id` values as PMIDs, preventing iCite/NCBI false-zero lookups when the dataset uses numeric local IDs.
  - Added Europe PMC enrichment to resolve DOI/PMID/PMCID and write `europepmc_cited_by_count` plus corrected life-science identifiers.
  - Added NCBI ELink PubMed cited-in enrichment and exported `true_citation_ncbi_pubmed_raw` in model score CSVs.
  - Fixed NCBI ELink batching to use repeated `id` parameters in small chunks, preventing merged multi-ID linksets from assigning one paper's citations to another paper.
  - Kept NCBI PMC cited-by as audit-only evidence; it no longer overwrites generic `citation_count` or final true-citation scoring.
  - Updated `AUTO_ENRICH_SOURCES` to support `europepmc` and `ncbi_citations` in addition to OpenAlex, iCite, OpenCitations, and Semantic Scholar.
  - Added title-identity guarding to true-citation scoring. If an external source title clearly mismatches the local paper title, raw source counts stay visible but are excluded from `true_citation_raw`.
  - Replayed the 85-paper user dataset without GPU at `runs/data-100-verification/user-impact-model-p200-identity-recovery/model-impact`.
  - Final recovery replay leaves 3 true nonzero papers, 84 matched rows, and 1 unresolved extract-only record. The improvement is stronger source corroboration, corrected PMIDs, wrong-paper citation leakage prevention, and recovery of two previously false high-score rows back to 0 rather than a larger nonzero count.

- [x] Phase 42: Additional no-key open citation sources.
  - Added `enrich-crossref`, `enrich-datacite`, and `enrich-openaire`.
  - `crossref` adds DOI-level `is-referenced-by-count` evidence.
  - `datacite` adds DataCite citation, view, and download counts for repository/data/software DOI records.
  - `openaire` adds OpenAIRE citation-impact indicators, including citation count, influence, popularity, and impulse.
  - The user auto-enrichment chain now supports `crossref`, `datacite`, and `openaire` in addition to the existing OpenAlex, Europe PMC, iCite, NCBI, OpenCitations, Semantic Scholar, and identity-recovery steps.
  - Fusion exposes `true_citation_datacite_raw` and `true_citation_openaire_raw`, and identity quarantine removes these source fields when external titles mismatch.
  - Verification: focused enrichment/fusion/docs/script tests passed; full test suite reported `573 passed` before final publish readiness.

- [x] Phase 43: More public citation-source coverage.
  - Added `enrich-datacite-events` for incoming DOI event evidence from DataCite Event Data.
  - Only DataCite Event `references` and `cites` relation types are counted as citation evidence; total events and source facets are retained as audit fields.
  - Added `enrich-wikidata` as an optional public citation source using DOI `P356` and cites-work `P2860`.
  - Kept Wikidata out of the default auto-enrichment chain because the public endpoint can be slow or unavailable.
  - Added default auto-enrichment support for `datacite_events`.
  - Fusion exports `true_citation_datacite_events_raw` and `true_citation_wikidata_raw`.
  - Replayed the 85-paper user dataset after DataCite Events enrichment; DataCite Events matched all 85 DOIs but found 0 positive incoming event counts, so the final nonzero count remained 3/85.
  - Verification: focused tests reported `54 passed`; full suite reported `579 passed`.

- [x] Phase 44: Crossref Data Citations and PLOS usage source expansion.
  - Added `enrich-crossref-data-citations` for Crossref beta Data Citations counts.
  - Added `true_citation_crossref_data_citations_raw` to the score outputs.
  - Added `enrich-plos-metrics` for PLOS article usage counters as optional audit/external-adoption evidence.
  - Integrated `crossref_data_citations` into the default auto-enrichment chain; left `plos_metrics` optional because it is PLOS-only and does not directly feed the final true/effective citation score.
  - Replayed the 85-paper user dataset: Crossref Data Citations matched 85/85 but found 0 positive counts; PLOS Metrics matched 0/85; final nonzero count remained 3/85.
  - Verification: focused tests reported `60 passed`; full suite reported `585 passed`.

## Design Constraints

- The valid claim remains dataset-local impact inside the user-provided corpus unless an external calibrated citation graph is added.
- Human labels are not available in the short term, so benchmark/proxy evidence must be labeled as such.
- Do not start GPU-serving or training jobs unless explicitly requested.
- Prefer no-GPU checks and benchmark smokes for final project completeness.
- Keep implementation auditable: every score should trace to selected citation edges, prompt/audit records, and scoring configuration.

## Errors Encountered

| Date | Error | Attempt | Resolution |
|---|---|---|---|
| 2026-05-27 | Existing planning files described an older completion phase. | Replaced `task_plan.md` with this full optimization plan. | Continue using `findings.md` and `progress.md` as persistent logs. |
| 2026-05-27 | New readiness tests failed at collection because `project_readiness.py` did not exist yet. | Ran target tests before implementation. | Implemented the module, CLI command, auto-run integration, and artifact index entries; target tests passed. |
| 2026-05-29 | `bitsandbytes` optional CUDA128 library warning reported missing `GLIBC_2.34` during SFT startup. | Checked process, GPU use, and `train_log.jsonl`. | Training continued with `torch.optim.AdamW`; treat the warning as non-blocking unless the process exits. |
| 2026-06-02 | 1000-paper OpCitance full multi-score/multi-label credibility run took too long after scores and labels had already been generated. | Stopped the specific long local Python process and reused generated papers/scores/labels. | Re-ran focused RCR and final-score multi-metric reports with narrower score columns. |
| 2026-06-02 | One-off recomputation script failed with `ModuleNotFoundError: local_impact_ranking`. | Ran Python without `PYTHONPATH=src`. | Re-ran with `PYTHONPATH=src`; score recomputation succeeded. |
| 2026-06-03 | First plan update patch targeted the parent working directory context rather than the repository file. | Used `apply_patch` without the repo-relative path. | Re-ran with the absolute repository path and verified `Phase 38` in `task_plan.md`. |
| 2026-06-03 | The curated validation/readiness/preflight commands were triggered more than once. | Polled all session IDs and checked for leftover scoring processes. | All duplicate sessions exited; final summaries were read from the last on-disk curated artifacts. |
