# Findings

## 2026-06-03 Extra Citation Source Finding

- Added two more credentialless biomedical citation sources to the user-data pipeline: Europe PMC and NCBI ELink PubMed cited-in.
- Europe PMC matched 85/85 curated user papers and fixed an important identifier problem: several records had numeric local `paper_id` values that looked like PMIDs but were not true PMIDs. Europe PMC DOI matching recovered real PMIDs such as `17555991` for `42030147` and `32058000` for `42169456`.
- iCite after Europe PMC recovered life-science evidence for several DOI/PMID-linked records, but later title-identity auditing showed that some of those external IDs belonged to cited references rather than the local target paper.
- NCBI PubMed cited-in now uses repeated `id` parameters and small chunks, preventing NCBI ELink from returning one merged multi-ID linkset and assigning the total to the wrong paper. In the inspected records, `42030147` has NCBI PubMed cited-in `248`, `42169456` has `34`, and `42032168` has `0`.
- The final replay with extra sources is at `runs/data-100-verification/user-impact-model-p200-extra-sources-replay/model-impact`.
- Manual spot checks exposed a more important issue: some external DOI/PMID values were attached to references inside the PDF, not to the target article. The final scoring now blocks those external citation counts when the external title is clearly inconsistent with the local title.
- Final true-citation nonzero count is 3/85 after identity guarding and recovery. The recovery command found 3 title mismatches, recovered 2 by high-confidence local-title OpenAlex search, and left 1 unresolved extract-only row.
- Identity-recovery examples:
  - `42030147`: the previous external title was `Initial T cell receptor transgenic cell precursor frequency dictates critical aspects of the CD8(+) T cell response to infection.` rather than local title `Design principles of the cytotoxic CD8+ T cell response`; wrong DOI/PMID evidence is quarantined and the final score is 0.
  - `42169456`: the previous external title was `TbCAPs: A toolbox for co-activation pattern analysis.` rather than local title `Resting-State Coactivation Patterns of Language Reorganization in Brain Tumors`; wrong DOI/PMID evidence is quarantined and the final score is 0.
  - `42124699`: local title `Main Text Results & Discussion` remains unresolved because it is an extraction artifact; it is scored as 0 and retained only for audit.
- The current top true-citation papers after recovery are `42125479` (`24`, OpenCitations), `42032168` (`6`, Crossref), and `42108843` (`1`, Semantic Scholar). The remaining 81 matched rows and the 1 unresolved row are zero across trusted/matched open citation sources.
- Manual zero-citation audit sampled 13 zero-score rows, including 2020/2024/2025/2026 records and the 2 recovered wrong-identity cases. OpenAlex, Crossref, Semantic Scholar, and Europe PMC all returned 0 citations for the matched DOI-indexed samples, with only the recovered Zenodo DOI missing from Crossref/Semantic Scholar. The audit report is `runs/data-100-verification/manual-zero-citation-audit.md`.

## 2026-06-03 User 100-Paper Dataset Verification Finding

- `/home/wyx/zh/local-impact-ranking/data_100.zip` is usable after curation, but it is not a high-credibility validation dataset by itself.
- The MinerU conversion produced 100 records; after excluding supplements and extraction artifacts, the curated article set contains 85 papers.
- OpenAlex repaired 3 clearly bad extracted titles. The curated dataset still has 3820 references, including 1522 reference DOIs.
- DOI matching found 2 self DOI matches and 0 cross-paper internal DOI matches. This explains why effective/internal citation evidence remains zero: the papers do not cite each other inside this 85-paper corpus.
- External evidence coverage is strong enough for disclosed current-impact scoring: DOI coverage `0.929`, citation-count coverage `0.929`, yearly citation-history coverage `0.918`, and field/year percentile coverage `0.859`.
- The score distribution is dominated by paper age and current citation sparsity: only 3 papers have nonzero final citation-impact scores, while 82 have zero final scores. This is expected for mostly 2026 papers, not a scoring-runtime failure.
- The claim certificate is `evidence_supported` with confidence `0.25`: reportable as dataset-local current citation impact with disclosure, but not validated/high-credibility impact accuracy without future-window labels, internal citation edges, or an external calibrated benchmark label for these papers.
- A model-backed p200 run with `local-impact-judge-source-task-dpo` is now available at `runs/data-100-verification/user-impact-model-p200`.
- The p200 pairwise graph is reportable: 200 planned comparisons, all 85 papers covered, one connected component, and at least 4 comparisons per paper. The model produced 200 judgments with 190 parseable outputs.
- The local model contributes a useful diagnostic/model-supported signal, but it does not change the final reportable score boundary. The final score remains `0.95 true citation + 0.05 effective citation`; because effective/internal citation evidence is zero and true current citations are sparse, only 3 final scores are nonzero.
- Model evidence is only moderate credibility: the supplied model profile has pooled SciImpact accuracy `0.658889`, weak slices, and `can_claim_high_model_credibility=false`. Therefore model-assisted fields should be disclosed as auxiliary evidence, not used as a high-credibility standalone impact score.
- Stability is mixed: final citation score top-20 retention is `1.0`, but pairwise model-ranking top-20 retention is `0.80375` with 32 unstable papers. This reinforces the decision to keep the final score citation-dominant for current-impact reporting.

## Project State

- Existing design docs already define the intended MVP scope: dataset-local content-aware impact ranking, not global scientific impact.
- Current implementation is stronger than the original MVP plan in several areas: validation reports, prompt/output audit, proxy evaluation, anomaly report, stability report, explanations, fine-tuning exports, and manifest.
- The previous mismatch with missing `rankings/fields.csv`, missing `rankings/years.csv`, and missing `reports/evaluation.md` has been resolved.
- Field and year rankings are aggregated from selected referenced-paper metadata. Missing field/year metadata is left absent rather than imputed.
- The main pipeline remains JSONL-only internally. CSV support is provided as a `convert-csv` preprocessing command with JSON columns for nested authors, references, citation contexts, and metadata.
- No-private-data demo path is now explicit: `scripts/run_benchmark_smoke.sh` runs a 5-paper SciCite prefix through the full no-GPU heuristic pipeline.
- `score` is now an explicit no-model stage that replays `audit/judge_outputs.jsonl` into the same complete outputs as `run --judge-audit`.
- Final artifact index checks on local context smoke and benchmark smoke both reported `missing_count = 0` across 35 key artifacts.

## 2026-05-30 Evidence-First Credibility Upgrade

- The main credibility gap is not another prompt or larger local model. A public-facing impact score needs auditable evidence families: field/year-normalized citation impact, citation graph structure, citation-context quality, temporal momentum, and external adoption signals.
- The project now treats the local Qwen pairwise judge as a model-assisted signal rather than the sole authority for the final score.
- The final scoring target is `impact_index_v3`: `0.80 evidence_supported_score + 0.20 model_supported_score`. The compatibility score `impact_index_v2` remains available for ablation and audit.
- `evidence_supported_score` gives high weight to optional external adoption evidence: external field/year percentile, patent citations, policy citations, and software/data/model adoption.
- This design follows the responsible-metrics direction already recorded in this file: use multiple contextualized indicators, avoid a single opaque metric, expose limitations, and avoid claiming global impact unless the input corpus supplies calibrated global citation/adoption evidence.
- Practical data priority for users:
  1. Required: paper IDs, titles, year/field, references.
  2. Strongly recommended: citation counts, yearly citation counts, citation contexts, DOI/OpenAlex/Semantic Scholar IDs.
  3. High-credibility external evidence: field/year percentile, patent citations, policy citations, GitHub/software adoption, dataset reuse, model/download evidence.

## 2026-05-30 Responsible-Metrics Upgrade Findings

- Literature review supports a multi-indicator design rather than a single opaque score: field/year normalized citation impact, article-level relative citation influence, citation-network prestige, citation-context quality, and broader adoption signals should be separated and then fused with explicit coverage and limitations.
- The project already has evidence-first `impact_index_v3`, external evidence readiness, future-window backtesting, bootstrap/permutation validity checks, PageRank, citation-context quality, and external adoption fields.
- Remaining engineering gaps before calling this optimization complete:
  - an RCR-like relative citation component to make article citation rate comparable to expected field/year citation rate;
  - explicit self-citation and abnormal citation-risk disclosure;
  - a stable taxonomy mapping SciImpact dimensions (`citation`, `award`, `patent`, `media`, `stars`, `downloads`, artifact adoption) to local evidence fields and claim boundaries;
  - model-score evidence cards for the `model-score-dataset` path, not only the classic selected-reference path.
- Final implementation added two more credibility controls:
  - Tied-rank percentile component scaling replaces max-value scaling, so a single extreme paper no longer compresses all other component scores while tied top evidence still receives 100.
  - Confidence-weighted Rank Centrality is now the default pairwise aggregation method, removing Elo's dependence on the order in which pairwise judgments are processed. Elo remains as an audit field.

## 2026-05-30 External Evidence Compatibility Findings

- The highest-leverage credibility improvement is to let the user bring calibrated external evidence into the same local scoring run, rather than forcing all influence evidence through prompt judgments.
- The project now supports a practical evidence import path for OpenAlex-compatible field/year evidence, BIP!-style impact indicators, Semantic Scholar highly influential citation counts, patent and policy citations, media and award evidence, and artifact adoption. This keeps the final claim dataset-local unless the imported evidence itself is globally calibrated.
- BIP and Semantic Scholar signals are treated as compatible external evidence, not as guaranteed live API truth. The system records coverage and missing fields, so a dataset with no such evidence remains reportable only with weaker evidence tiers.
- Optional validity labels are now integrated into `model-score-dataset`. This makes future-window or external-label evaluation part of the deliverable run instead of a separate afterthought. If labels are absent, the system reports evidence coverage and benchmark proxies, not validated impact accuracy.

## 2026-05-30 Impact Evaluation Literature Refresh

- The strongest current answer is a layered, evidence-first evaluation contract, not a single model-generated impact score.
- Mature bibliometric sources support field/time/type normalization as the citation backbone:
  - NIH iCite RCR defines article influence through field- and time-normalized citation rate using co-citation fields.
  - OpenAlex FWCI normalizes citations by publication type, subfield, and publication year, and exposes FWCI plus normalized citation percentile fields.
  - Waltman and van Eck argue that field-normalized comparisons require careful counting choices, especially fractional counting for aggregate comparisons.
- Citation context literature supports down-weighting incidental references:
  - `Measuring academic influence: Not all citations are equal` uses author-identified key references and finds repeated in-text mentions especially useful.
  - `Identifying Meaningful Citations` defines important citations as cases where the cited work is used or extended, and shows section/context features matter.
  - Semantic Scholar highly influential citations operationalize this idea with a machine-learning model over citation metadata and surrounding context, with full-text coverage limitations.
- Newer benchmark work says impact is multi-dimensional:
  - SciImpact covers 19 fields and dimensions including citations, awards, media, patents, and artifact adoption.
  - PreScience includes time-aligned prior-work and impact-prediction tasks over AI papers and citation histories.
  - CausalCite frames paper impact as a counterfactual causal effect on follow-up papers, useful as a research direction but heavier than the current product contract.
- Open external evidence sources are practical:
  - BIP! DB provides citation count, 3-year citation count, PageRank, RAM, and AttRank-style indicators over large PID/work collections.
  - OpenCitations offers open citation dumps and citation-count data; it is useful for auditable DOI-level graph evidence.
- Responsible-metrics guidance sets the claim boundary:
  - DORA rejects journal-level metrics as surrogates for article quality and recommends considering datasets, software, policy/practice influence, and multiple indicators.
  - The Leiden Manifesto supports contextualized, transparent metrics rather than opaque one-number evaluation.
- Implementation consequence: the next credibility jump should prioritize external calibrated labels and validation, full v3 component calibration, and strict reportability tiers. More prompt engineering or a larger local judge alone is a false-progress signal unless it improves held-out SciImpact/future-window validity and calibration.

## 2026-05-30 Credibility Recommendation Adopted

- The `impact_index_v3_final_components` calibration path implements the main literature-refresh recommendation: the final score can now be calibrated against pairwise or future/external labels across all v3 components, not only legacy v2 model/citation/graph/context/momentum signals.
- `validated_impact_claim` and per-paper `validity_confidence` enforce the responsible-metrics boundary: a score can be high without being validated. The project now makes that distinction explicit in the model-scoring report and CSV/JSONL outputs.
- Remaining claim boundary: if the user supplies no future-window labels, external continuous impact labels, or calibrated benchmark labels, the system should still be described as an evidence-supported dataset-local index, not validated real-world impact accuracy.

## 2026-05-30 Impact Validity Scorecard Finding

- The credibility bottleneck is no longer whether the pipeline can compute a score; it is whether the validation evidence can survive coverage, random-baseline, slice, and outlier checks.
- Literature-backed impact evaluation favors rank/field-normalized and multi-indicator evidence, but report audiences still need a simple gate. The new `impact_credibility_scorecard` makes that gate explicit without replacing the underlying metrics.
- The scorecard intentionally treats `high_credibility` as a stricter state than `externally_supported`: it requires coverage, statistical support, rank alignment, top-rank quality, field-slice stability, and robustness after removing extreme label/score rows.
- This directly addresses the user's concern about abnormal values: a score that looks good mainly because one extreme citation/adoption row dominates top-k quality is marked outlier-sensitive and cannot be presented as high-credibility validation.

## 2026-05-30 Pairwise Benchmark Scorecard Finding

- SciImpact-style pairwise labels need their own credibility gate because they evaluate direct pairwise impact ordering rather than continuous future/adoption labels.
- A large benchmark can produce statistically significant but practically weak results. The existing full SciImpact text-prior baseline shows this: non-tie accuracy is slightly above 0.5 and statistically significant due to sample size, but pairwise accuracy is `0.494` and a slice is below random.
- The new `pairwise_credibility_scorecard` therefore does not let p-value alone define credibility. It also requires useful benchmark accuracy, low tie rate, high coverage, and stable eligible dimensions/fields.
- This makes the public benchmark story stronger: a model or score must beat random in a practically meaningful way and avoid hidden slice failures before being presented as high credibility.

## 2026-05-31 Direct Model-Answer Scorecard Finding

- A direct model-answer benchmark needs a separate credibility gate from both pairwise ranking evaluation and continuous validity-label evaluation. It answers the narrower question: can the local model choose the higher-impact A/B item on SciImpact-style prompts?
- The current source-task DPO judge is not random: the three existing p50 held-out prediction sets have direct accuracies `0.680`, `0.647`, and `0.650` with full coverage and significant support against a random A/B baseline.
- It is still not high-credibility as a standalone impact oracle. Each seed has at least one weak slice below `0.60`, especially `dataset`, `github_repo`, `model`, `downloads`, `stars`, and `patent` slices.
- Practical consequence: the project should present the model as one auditable signal inside an evidence-first impact index. High-confidence reporting should come from external evidence coverage, future-window/backtest labels, calibration, outlier checks, and the claim certificate, not from model-answer accuracy alone.

## 2026-05-31 Cross-Seed Model Summary Finding

- Single-seed SciImpact model accuracy is too easy to overread. A reportable model-quality claim should aggregate multiple held-out seeds and expose the weakest seed and weakest slices.
- The new three-seed source-task DPO summary gives a cleaner claim: pooled direct accuracy `0.658889` over `900` covered examples, mean seed accuracy `0.658889`, and min seed accuracy `0.646667`.
- This is credible enough to say the local judge has above-random benchmark signal, but not enough for a high-credibility standalone model claim. The summary remains `moderate_credibility` because seed stability and slice stability are warnings.
- The highest-priority model-improvement targets are now explicit and measurable: `dataset`, `github_repo`, `model`, `stars`, `downloads`, and `patent` slices.

## 2026-05-31 Weak-Slice Error Pack Finding

- The model-quality bottleneck is now concrete rather than abstract: the three-seed source-task DPO run has `178` wrong/uncovered rows in weak slices and `13` high-confidence weak-slice errors.
- Error volume is concentrated in `stars/github_repo`, `downloads`, `patent`, `model`, and `dataset`, matching the cross-seed summary's weak-slice diagnosis.
- These held-out errors are useful for qualitative diagnosis and prompt/data-design inspection, but they should not be used directly as fine-tuning data. The error pack is therefore `diagnostic_only` by default.
- The next credible model-improvement experiment should regenerate the same weak-slice pack from train-split predictions, then use those rows as targeted hard examples while evaluating only on held-out seeds.

## 2026-05-31 Train-Split Weak-Slice Candidate Finding

- A train-split weak-slice candidate pack now exists from `runs/sciimpact-hard-example-train-p100-dpo64`: `139` SFT and `139` DPO records, with no invalid rows and no DPO duplicate/reverse-conflict issues.
- This is not yet proof of model improvement. It is a training candidate set that should be used for a controlled continuation run and judged against the held-out source-task DPO three-seed summary.
- The candidate set targets the same family of failures as the held-out diagnosis, especially `downloads`, `media`, `stars/github_repo`, `model`, and `dataset`.
- The next GPU experiment should be small and gated: train on this candidate set or a larger source-task train-split variant, then promote only if cross-seed direct accuracy, seed stability, and weak-slice stability improve without new regressions.

## 2026-05-31 Weak-Slice DPO Continuation Finding

- The next model-improvement step is now executable but not yet run: a 32-step weak-slice DPO continuation from the current source-task DPO adapter.
- The continuation is deliberately conservative because previous small targeted runs often traded gains in one slice for regressions in another. A low learning rate and short run reduce the chance of overwriting broad source-task behavior.
- The experiment should be treated as a candidate only. It must be evaluated on held-out seeds `23/47/71`, summarized with `summarize-sciimpact-model-credibility`, and rejected if direct accuracy, seed stability, or weak-slice stability regresses.

## 2026-05-31 Weak-Slice DPO Experiment Result

- The 32-step weak-slice DPO continuation trained successfully but did not improve model credibility.
- It improved seed47 slightly (`0.647 -> 0.657`) but regressed seed23 (`0.680 -> 0.667`) and seed71 (`0.650 -> 0.627`).
- Three-seed pooled direct accuracy fell from baseline `0.658889` to candidate `0.650`; min seed accuracy fell from `0.646667` to `0.626667`.
- The experiment confirms that simply training on a small weak-slice error pack is not enough. The next route should either enlarge train-split weak-slice coverage from the source-task DPO model itself or improve evidence extraction/calibration rather than promoting this adapter.

## 2026-05-31 Model Candidate Promotion Gate Finding

- Model improvement needs an explicit promotion gate, not only a candidate training run and a separate benchmark table.
- The new gate compares candidate and baseline cross-seed credibility summaries. By default it blocks promotion unless the candidate has at least three evaluations, improves pooled accuracy, does not regress the weakest seed, does not materially worsen seed stability, does not add weak slices, does not materially regress the worst eligible slice, and keeps credibility status at least as strong as the baseline.
- Applied to the weak-slice DPO continuation, the gate correctly returns `do_not_promote`: pooled accuracy delta `-0.008889` and min seed accuracy delta `-0.020`.
- This improves project credibility by making negative model experiments first-class evidence. Future adapters should be promoted only through this gate or an explicitly documented stricter replacement.

## 2026-05-31 Self-Mined Weak-Slice DPO Result

- The improved mining route used the current recommended source-task DPO model on a train-split stratified sample, so it avoided training on held-out/test errors and avoided mining from an older model's failures.
- The 1200-row train diagnostic produced 259 valid train-safe DPO records, concentrated in the same weak-slice family seen on held-out data: downloads, patent, stars/GitHub repos, and models.
- A 64-step DPO continuation trained successfully with finite losses, but held-out evaluation did not improve the judge:
  - seed23 `0.680 -> 0.663`
  - seed47 `0.647 -> 0.650`
  - seed71 `0.650 -> 0.633`
  - three-seed pooled accuracy `0.658889 -> 0.648889`
- The candidate did improve the worst eligible slice minimum (`0.520 -> 0.548`) and kept weak-slice count unchanged, but this came at the cost of pooled and worst-seed accuracy.
- Decision: do not promote. The repeated failure of targeted weak-slice DPO suggests the next credible route is not another small DPO continuation; it should be stronger teacher relabeling, better source evidence extraction, or larger preference-data quality redesign with held-out promotion gates.

## 2026-05-31 Product Demo Refocus Finding

- The most useful next deliverable is not another model-tuning loop. It is a short, no-GPU, reproducible demonstration of the user's original workflow: provide a paper dataset and receive impact scores, evidence, validity boundaries, and a handoff bundle.
- `scripts/run_impact_demo.sh` now uses offline pairwise replay with `examples/impact_demo_papers.jsonl` and does not call a model server. This keeps the demo deterministic and runnable in CI-like environments.
- The demo output is intentionally honest: it can show dataset-local evidence-supported scores, but the small four-paper validation fixture remains `insufficient_validation_data`, so the claim certificate does not overclaim validated impact accuracy.
- This is a better product artifact than another weak-slice adapter because it makes the scoring system inspectable and immediately usable with the user's eventual data.

## 2026-06-02 OpCitance + iCite Current-Impact Finding

- A 1000-paper public biomedical benchmark was built from OpCitance citation contexts and NIH iCite labels:
  - `runs/opcitance-icite-current-expanded/opcitance_icite_papers.jsonl`
  - `runs/opcitance-icite-current-expanded/impact_scores.csv`
  - `runs/opcitance-icite-current-expanded/icite_current_metrics.csv`
  - `runs/opcitance-icite-current-expanded/opcitance_icite_benchmark_summary.md`
- Coverage is strong enough for a current-impact check: 1000 scored papers, 998 RCR labels, 1224 internal reference edges, 5442 citation contexts, and effective-citation coverage `0.713`.
- The expanded benchmark shows the most defensible current-impact score is citation-dominant:
  - `score_100_true_00_effective` vs RCR: Spearman `0.987`, pairwise accuracy `0.952`.
  - `score_95_true_05_effective` / new `final_impact_score` vs RCR: Spearman `0.985`, pairwise accuracy `0.948`.
  - `score_80_true_20_effective` vs RCR: Spearman `0.961`, pairwise accuracy `0.912`.
  - `score_50_true_50_effective` vs RCR: Spearman `0.816`, pairwise accuracy `0.818`.
  - `effective_citation_score` alone vs RCR: Spearman `0.482`, pairwise accuracy `0.564`.
- Conclusion: effective citation is useful as an explanation/context adjustment and sensitivity signal, but public iCite validation does not support giving it a large default weight for current impact. The final score was therefore changed to `0.95 * true_citation_score + 0.05 * effective_citation_score`.
- The final 95/5 score is high-credibility on current iCite metrics: RCR Spearman `0.985`, citations/year `0.975`, APT `0.830`, NIH percentile `0.985`, and iCite citation count `0.999`.
- This benchmark is now part of the formal benchmark plan as a reproducible, no-private-data current-impact check. It is appropriate for validating the project's current-impact score in life-science/PubMed-like data, but it should not be presented as a future-impact or all-field social-impact benchmark.

## Verification History

- `PYTHONPATH=src pytest -q` reported `76 passed`.
- `scripts/run_local_smoke.sh /tmp/local-impact-ranking-prompt-audit-smoke` completed and generated `audit/prompts.jsonl` for both tiny and context examples.
- After adding field/year rankings and `evaluation.md`, `PYTHONPATH=src pytest -q` reported `77 passed`.
- `scripts/run_local_smoke.sh /tmp/local-impact-ranking-complete-smoke-1` completed.
- Targeted CSV ingestion tests passed with `PYTHONPATH=src pytest -q tests/test_io_models.py tests/test_cli.py -q`.
- `scripts/run_benchmark_smoke.sh /tmp/local-impact-ranking-benchmark-smoke` completed.
- Final verification: `PYTHONPATH=src pytest -q` reported `82 passed`.
- Final local smoke completed at `/tmp/local-impact-ranking-final-local-smoke`.
- Final benchmark smoke completed at `/tmp/local-impact-ranking-final-benchmark-smoke`.

## Constraints

- The user does not currently have private data.
- No human annotation is available in the short term.
- GPU/model server should stay off unless explicitly requested.

## Latest Model Comparison Finding

- The 20-step DPO pilot has a small positive same-input signal on the 100-paper PreScience slice, but the improvement is weak.
- Target-level comparison shows the net hit gain is +1: fixed targets `2883238` and `3265742`, broken target `3264213`.
- Most outputs did not change materially: mean selection Jaccard is `0.928`, with 69 unchanged hits and 24 unchanged misses.
- Two still-hit targets worsened in gold rank (`14636714`, `14930700`), while two multi-positive targets changed hit quality without changing hit status (`1695826`, `376086`).
- This supports continuing error analysis or a modest 300-sample resumable evaluation, but it is not enough to claim the DPO model is definitively better.
- The strict same300 evaluation preserved the same direction but remained weak: DPO fixed 3 base misses, broke 1 base hit, and improved hit_rate by `+0.006667`.
- Same300 mean_ndcg improved by `+0.007293`, mean reciprocal rank by `+0.004389`, and mean selection Jaccard stayed high at `0.927`, meaning most decisions were unchanged.
- The DPO pilot should still be treated as an early positive signal, not a production-quality tuned judge. A stronger next training run should use more hard negatives and evaluate on a held-out slice.

## 2026-05-27 Full Optimization Audit

- Current implementation already covers the main product pipeline: validation, constrained judging, replay scoring, fused scores, rankings, baselines, proxy/gold evaluation, anomaly reports, stability/uncertainty reports, explanations, fine-tuning exports, LLaMAFactory configs, and artifact indexing.
- The key remaining product gap is not another scoring component; it is an end-user quality gate. Evidence is spread across `mvp_summary`, `benchmark_summary`, `artifact_index`, `finetuning_preflight`, and baseline/stability reports. A report-ready project needs one command/report that says whether a run is `reportable`, `needs_improvement`, or `blocked`, with concrete blockers and evidence paths.
- Existing docs correctly constrain the claim to dataset-local impact. The final project should keep that boundary visible in the top-level readiness report so users do not overclaim global scientific impact.
- Current `project-summary-report.md` still records an old `82 passed` state. It must be updated after final verification if it is used as a deliverable.
- Follow-up audit found one readiness semantics issue: a check named `gold_or_proxy_evidence_available` only accepted gold labels. For no-human-label use, proxy evidence should count as evidence but be labeled lower confidence. This is fixed by adding `evidence_level`.
- Follow-up audit found an artifact dependency issue: `project-readiness` was generated before `artifact_index.json`, while readiness requires artifact completeness. `RUN_ARTIFACTS` also did not include `reports/artifact_index.json`. The fix makes readiness required artifacts explicit, includes artifact index in `RUN_ARTIFACTS`, and writes artifact index before and after readiness so the final index is self-contained.
- A second artifact audit found that the standalone `project-readiness RUN_DIR` command also needed to refresh `artifact_index.json`; otherwise old runs or manually deleted indexes would leave inconsistent reports. The CLI now uses a shared helper that writes artifact index, writes readiness, then refreshes artifact index again.
- Readiness was still not fully one-page reportable because reproducibility details lived only in `run_manifest.json`. The readiness report now includes a compact reproducibility summary: input hashes, normalized record count, top-k, judge/model, prompt version, scoring version, git revision, and dirty flag.

## 2026-05-27 Research Findings For Optimization

- ImpactRankings public repo is a method inspiration, not a reproducible protocol. Its README says it asks DeepSeek-R1-Distill-Llama-8B for the five most important references per AI paper, runs on top AI conference papers from 2020-2025, maps authors to CSRankings affiliations, and splits credit by coauthors. The repo mainly exposes a website/static artifacts, with only 4 commits, 13 stars, and no releases at the time of review. Missing reproducibility pieces remain prompt, candidate construction, inference parameters, matching failures, evaluation protocol, and training data.
  - Source: https://github.com/AI-Research-Impact-Ranking/ImpactRankings
- `Measuring academic influence: Not all citations are equal` directly supports key-reference selection before aggregation. It argues that equal citation counting ignores citation functions, asks authors to identify key references, and finds mention-frequency features especially strong for predicting central academic influence.
  - Source: https://arxiv.org/abs/1501.06587
- Semantic Scholar's highly influential citations provide an external design analogue: they identify citations where the cited paper significantly affects the citing paper using citation metadata and citation context, while noting full-text coverage limitations.
  - Source: https://www.semanticscholar.org/faq/influential-citations
- `Identifying Meaningful Citations` supports citation-context and section/placement features for separating important from incidental citations. It frames important citations as those where the cited work is used or extended.
  - Source: https://ai2-website.s3.amazonaws.com/publications/ValenzuelaHaMeaningfulCitations.pdf
- SciCite is a reusable public citation-intent dataset and codebase. Its GitHub repo provides JSONL data with `label` and `context`, suitable for evaluating or training method/background/result citation distinctions.
  - Source: https://github.com/allenai/scicite
- PreScience is the closest current public benchmark for this project. It has 98K AI-related papers, disambiguated author identities, temporal metadata, companion author publication histories/citation graph over 502K papers, and tasks including prior-work selection and impact prediction.
  - Source: https://arxiv.org/abs/2602.20459
- SciImpact is a new direct scientific-impact benchmark. It covers 19 fields, 215,928 contrastive paper pairs, and impact dimensions beyond citation count, including awards, media attention, patent references, and artifact adoption. It is useful as an optional pairwise/general-impact evaluation, but broader than this repo's default citation-derived local influence.
  - Source: https://arxiv.org/abs/2604.17141
- SCIDOCA 2025 provides citation discovery, masked citation prediction, and citation sentence prediction. It is useful for testing citation modeling/candidate selection, but not itself an impact-ranking benchmark.
  - Source: https://arxiv.org/abs/2509.24283
- Network baselines remain necessary for credibility. CiteRank/PageRank-style methods model citation-network traffic and address citation aging; they should remain baseline/prior signals, not replace semantic edge judging.
  - Source: https://doi.org/10.1088/1742-5468/2007/06/P06010
- Field/year normalization and fractional attribution are required when comparing across fields or organizations. Waltman and van Eck argue field-normalized results are not proper under full counting and recommend fractional counting for field-normalized organizational studies.
  - Source: https://arxiv.org/abs/1501.04431
- BIP! DB is a reusable external baseline/calibration asset. The Zenodo record describes citation-based impact indicators over persistent identifiers for research products and cites the WWW 2021 BIP! DB paper.
  - Source: https://zenodo.org/records/13642957
- Responsible metrics guidance supports explicit claim boundaries. DORA warns against journal-level metrics as substitutes for article or researcher quality; the Leiden Manifesto calls for principles in metrics-based research evaluation, including contextualization and field variation. This supports reporting multiple signals, limitations, and no single absolute truth claim.
  - Sources: https://sfdora.org/read/ and https://www.nature.com/articles/520429a

## 2026-05-29 Quality Bottleneck Diagnosis

- The current deliverable system is structurally complete, but the accepted Qwen32B LoRA judge is only moderately credible on SciImpact: three p50 held-out seeds give mean direct accuracy `0.634` and mean bidirectional selected accuracy `0.662`; the improvement over the earlier DPO64 adapter is only `+0.007` selected accuracy and consistency is worse by `-0.014`.
- The major failure modes are not random engineering bugs. Persistent errors concentrate in evidence-hungry dimensions and item types: citation, downloads, media, patent, dataset cards, and GitHub README examples. Many wrong examples are high-confidence, so raw logprob confidence alone cannot fix quality.
- Small targeted SFT/DPO experiments have already been tried and are mostly negative: weak-dimension oversampling, item-type oversampling, hard-example DPO, and larger 16k SFT+DPO produced at most small gains. Two-adapter complementarity was weak, so a simple ensemble of current 32B adapters has a low ceiling.
- Context length was a plausible bottleneck because current 32B LoRA training used `max_length=1024`, while the SciImpact released LLaMA-Factory recipe uses `cutoff_len=4096` and official task scripts preserve longer README/card contexts. Follow-up 16k-token SFT/DPO removed truncation as a confound, but did not improve benchmark accuracy enough to replace the recommended adapter.
- The product scoring path remains more defensible than the benchmark-only judge because ImpactIndex v2 fuses model pairwise Elo with field/year-normalized citations, local citation PageRank, citation-context quality, and temporal momentum. But if a user dataset lacks citation counts, internal references, citation contexts, year/field metadata, and citation history, the score necessarily degrades into a weak model-only estimate.
- Next high-leverage route: evaluate a stronger local teacher/judge, improve preference data quality, add multi-dimensional evidence extraction, and make final reports distinguish `evidence-supported impact score` from `model-only exploratory score`.

## 2026-05-29 Long-Context Tokenization Finding

- A concrete training-quality issue was fixed before further model work: previous SFT/DPO tokenization kept the sequence tail when examples exceeded `max_length`, which can remove the task instruction and the beginning of Paper/Artifact A.
- The new truncation keeps prompt head + prompt tail + full response labels. This is important for pairwise impact judging because the prompt prefix contains the task and side labels, while the suffix is closest to the answer.
- Full SciImpact SFT inspection showed why 4096 context matters: `max_length=1024` truncated `12718/87984` examples (`14.4549%`, mean prompt retention `0.913004`); `max_length=4096` truncated `2461/87984` (`2.7971%`, mean prompt retention `0.997316`).
- Prefix, suffix, and response retention are now all `1.000` in the inspection reports after the fix, so the next 4096-context training run has a cleaner input pipeline.
- Because the available hardware is 8x H100, the practical next run should not stop at 4096. Full SciImpact SFT inspection shows `max_length=8192` truncates only `4/87984` examples (`0.0045%`, mean prompt retention `0.999996`), and `max_length=16384` truncates `0/87984` examples. This removes tokenization loss as a major confound for the next Qwen32B local impact judge experiment; the remaining question is benchmark accuracy after training.

## 2026-05-29 Clean 16k-Token Training Result

- The clean long-context route was completed with 16k-token SFT and DPO training on 16k SciImpact examples:
  - SFT adapter: `runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step`
  - DPO adapter: `runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-dpo16k-256step`
  - Both used `max_length=16384`, `max_examples=16000`, and `max_steps=256`.
- Three-seed SciImpact p50 evaluation shows that longer context fixes truncation but not the main model-quality bottleneck:
  - Clean 16k SFT mean direct accuracy: `0.634`; mean bidirectional selected accuracy at margin `0.05`: `0.643`.
  - Clean 16k SFT+DPO mean direct accuracy: `0.621`; mean bidirectional selected accuracy at margin `0.05`: `0.653`.
  - Existing recommended adapter `runs/sciimpact-large-sft-dpo/checkpoints/qwen32b-sft16k512-dpo-random16k-256step` remains stronger, with mean direct accuracy `0.634` and mean bidirectional selected accuracy `0.662`.
- Conclusion: GPU capacity is sufficient for 16k context, but context length alone is not the route to a credible step-change. The next improvement should focus on preference data quality, evidence extraction, stronger local teacher/judge comparisons, and held-out calibration rather than simply increasing `max_length`.
- The stricter three-seed promotion gate confirms the clean 16k-token SFT+DPO adapter should not replace the recommended adapter: mean selected delta `-0.009`, mean direct delta `-0.013`, worst seed selected delta `-0.017`, and worst slice delta `-0.060`. Future candidate adapters should be judged against the current recommended adapter with the same gate before any deployment recommendation.

## 2026-05-29 Source-Task Alignment Result

- The strongest confirmed training bottleneck after long-context diagnostics was not raw context length but prompt/task mismatch. Earlier SciImpact exports rewrote all tasks into a generic "higher scholarly impact" pairwise prompt, while the benchmark asks dimension-specific questions about awards, citations, downloads, media attention, patents, and artifact stars.
- Implemented source-task-aligned SciImpact export:
  - `export-sciimpact-pairwise-sft --prompt-style source-task` preserves the original SciImpact `system` and `human` prompts.
  - Targets are normalized to `Answer: A/B` for constrained local decoding.
  - DPO export preserves the source system prompt instead of dropping it.
- Generated `87984` source-task SFT/DPO training records in `runs/sciimpact-source-task-sft-dpo/`. Full 4096-token inspection truncated only `4/87984` examples and retained task prefix, suffix, and answer tokens.
- Source-task SFT alone gave seed23 direct accuracy `0.670` and selected accuracy `0.690`, a positive but insufficient single-seed signal.
- Source-task SFT+DPO improved the previous recommended adapter over three SciImpact p50 seeds:
  - Mean direct accuracy: `0.634 -> 0.659`
  - Mean selected accuracy at margin `0.05`: `0.662 -> 0.669`
  - Mean A/B consistency: `0.631 -> 0.789`
  - Per-seed source-task DPO: seed23 `0.680/0.703/0.763`, seed47 `0.647/0.640/0.817`, seed71 `0.650/0.663/0.787` for direct/selected/consistency.
- The active default adapter is now `runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k4096-dpo16k1024-128step`, served as `local-impact-judge-source-task-dpo`.
- Credibility caveat: strict `0.05` slice-regression gating fails because seed71 `patent` regresses by `0.060`; disclosed `0.06` gating passes. The project should present this as a more defensible default, not a perfect impact oracle.
- Negative follow-ups: patent/downloads slice DPO continuation worsened mean metrics, and 4096-length DPO did not beat 1024-length source-task DPO on the first seed, so neither should be promoted.

## 2026-05-29 Confidence Calibration Cross-Check

- The bundled `configs/calibration/sciimpact-qwen32b-heldout-p50.json` profile was originally derived from the older `local-impact-judge` run, so it must not be described as source-task-DPO-specific calibration.
- A source-task DPO seed23 p50 split check showed that the adapter's raw confidence is already reasonably calibrated on that slice: validation accuracy `0.633333`, raw ECE `0.045373`, raw Brier `0.217641`.
- Same-run empirical bins overfit this split and made validation calibration worse: calibrated ECE `0.093333`, calibrated Brier `0.227989`.
- The older bundled SciImpact trust-prior profile still improved ECE on the source-task DPO validation half, from `0.045373` to `0.029138`, with nearly unchanged Brier (`0.217641` to `0.219402`).
- Decision: keep the bundled calibration profile as an approximate reusable trust prior and disclose that it is not domain-specific ground truth. Do not promote a newly generated source-task calibration profile from this small split.

## 2026-05-30 Strong Source-Task Candidate Result

- Tested a larger source-task route to see whether more balanced source-task training could beat the active default:
  - SFT adapter: `runs/sciimpact-source-task-strong/checkpoints/qwen32b-source-task-sft32k-4096-1024step`
  - DPO adapter: `runs/sciimpact-source-task-strong/checkpoints/qwen32b-source-task-sft32k4096-dpo32k1024-128step`
  - SFT used `32000` balanced-metadata examples, `max_length=4096`, `1024` optimizer steps, and zero tokenization truncation.
  - DPO used the strong SFT adapter as base/reference, `32000` balanced-metadata preference examples, `max_length=1024`, `128` optimizer steps, and `learning_rate=5e-7`.
- Seed23 p50 quick gate against the active source-task DPO default:
  - Active default: direct `0.680`, selected `0.703`, consistency `0.763`.
  - Strong SFT: direct `0.673`, selected `0.710`, consistency `0.730`.
  - Strong SFT+DPO: direct `0.670`, selected `0.713`, consistency `0.730`.
- Decision: do not promote the strong SFT/DPO route. It has a small selected-accuracy gain on seed23, but direct accuracy and A/B consistency regress. That violates the project gate requiring no average direct-accuracy drop and stable position consistency before replacing the default. Three-seed expansion is not justified unless a future variant fixes those regressions.

## 2026-06-03 Multi-Source Citation Resolution Finding

- The previous true-citation resolver used fixed source priority. In the user 85-paper dataset, that meant an OpenAlex value of `0` could override nonzero evidence from Semantic Scholar, NIH iCite, OpenCitations, or Crossref.
- A multi-source audit initially found 5 papers with at least one nonzero open citation source:
  - `42030147`: OpenAlex `410`, OpenCitations `363`, Crossref `334`, Semantic Scholar `493`.
  - `42169456`: OpenAlex `3`, OpenCitations `61`, Crossref `75`, Semantic Scholar `64`.
  - `42125479`: OpenAlex/Semantic Scholar/iCite `0`, OpenCitations `24`.
  - `42032168`: OpenAlex `5`, OpenCitations `2`, Crossref `6`.
  - `42108843`: OpenAlex/OpenCitations/Crossref `0`, Semantic Scholar/iCite `1`.
- Implemented current-impact true citation policy: collect all named open-source counts, select the maximum non-null count for scoring, and retain source-specific raw values plus conflict diagnostics for audit.
- `supplied_citation_count` is no longer double-counted when it only repeats a named source; it is used when no named source is available or when it is higher than all named source counts.
- Replayed the 200 saved model judgments without GPU using the merged input:
  - Output: `runs/data-100-verification/user-impact-model-p200-multisource-replay/model-impact`.
  - Initial `true_nonzero=5`, `all_zero=80`, `source_conflict_flag=True` for all 5 nonzero papers because at least one source disagrees sharply or returns zero.
  - Rank changes from old to new: `42125479` moves `40 -> 3`; `42108843` moves `29 -> 5`; `42169456` moves `3 -> 2`.
- Follow-up manual spot checks showed that `42030147` and `42169456` were false positives caused by external identity mismatch. After identity recovery, the final replay has `true_nonzero=3`, 84 matched rows, and 1 unresolved extract-only row (`42124699`).
- Claim boundary is unchanged: the system now has better open-source citation evidence and auditability, but the user dataset still lacks external/future-window ground-truth labels and internal/effective citation edges, so it cannot claim validated high-credibility impact accuracy.
