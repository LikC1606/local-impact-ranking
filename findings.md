# Findings

## Project State

- Existing design docs already define the intended MVP scope: dataset-local content-aware impact ranking, not global scientific impact.
- Current implementation is stronger than the original MVP plan in several areas: validation reports, prompt/output audit, proxy evaluation, anomaly report, stability report, explanations, fine-tuning exports, and manifest.
- The previous mismatch with missing `rankings/fields.csv`, missing `rankings/years.csv`, and missing `reports/evaluation.md` has been resolved.
- Field and year rankings are aggregated from selected referenced-paper metadata. Missing field/year metadata is left absent rather than imputed.
- The main pipeline remains JSONL-only internally. CSV support is provided as a `convert-csv` preprocessing command with JSON columns for nested authors, references, citation contexts, and metadata.
- No-private-data demo path is now explicit: `scripts/run_benchmark_smoke.sh` runs a 5-paper SciCite prefix through the full no-GPU heuristic pipeline.
- `score` is now an explicit no-model stage that replays `audit/judge_outputs.jsonl` into the same complete outputs as `run --judge-audit`.
- Final artifact index checks on local context smoke and benchmark smoke both reported `missing_count = 0` across 35 key artifacts.

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
