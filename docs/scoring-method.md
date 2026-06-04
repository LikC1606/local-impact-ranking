# Scoring Method

This project computes dataset-local influence. It does not claim global scientific impact unless the input dataset already represents a calibrated global citation graph.

The pipeline has two separate stages:

1. A judge selects influential references for each target paper.
2. The scoring module converts those selections into paper, author, institution, field, and year scores.

## Judge Output Contract

For each target paper, the judge returns selected candidate references:

```json
{
  "target_paper_id": "p1",
  "selected_references": [
    {
      "reference_id": "r1",
      "rank": 1,
      "confidence": 0.9,
      "rationale": "Foundational method used by the target paper."
    }
  ]
}
```

Only valid `reference_id` values present in the target paper's `references` list receive credit. Invalid or invented IDs are recorded in `audit/judge_outputs.jsonl` but excluded from scoring.

## Credit Modes

### `rank_weighted`

Version: `score_v1_rank_weighted`

This is the default mode. The selected reference receives credit by rank:

```text
rank 1 -> 1.0
rank 2 -> 0.8
rank 3 -> 0.6
rank 4 -> 0.4
rank 5 -> 0.2
```

In short: `1.0, 0.8, 0.6, 0.4, 0.2`.

Formula:

```text
credit(target -> reference) = rank_weight(rank)
```

### `flat`

Version: `score_v1_flat`

Every valid selected reference receives one point:

```text
credit(target -> reference) = 1.0
```

This is useful as an ablation because it removes rank sensitivity.

### `context_weighted`

Version: `score_v2_context_weighted`

This mode starts from `rank_weighted` and multiplies by citation-context support:

```text
no citation context                -> 1.0
context present                    -> 1.2
method/data/baseline cue present   -> 1.5
```

Formula:

```text
credit(target -> reference) = rank_weight(rank) * context_weight(contexts)
```

The method/data/baseline cue list is intentionally simple and auditable. Current cue terms include words and phrases such as `build on`, `based on`, `method`, `algorithm`, `dataset`, `benchmark`, `baseline`, `compare`, and `comparison`.

## Effective Citations For Life Sciences

For paper-level `impact_index_v3`, the scorer now records an effective-citation audit layer for every paper. This layer is designed for biomedical and life-science corpora, where a citation in a clinical guideline, trial, assay protocol, reagent reuse, or dataset/tool reuse is usually stronger evidence than a generic background citation.

The detailed reproducible specification is in [effective-citation-calculation.md](effective-citation-calculation.md). Source-specific citation evidence and reconciliation rules are in [citation-data-sources.md](citation-data-sources.md).

Model-classified roles are materialized before scoring. `classify-citation-contexts` runs over the existing `citation_contexts[]` items, sends each context to a model with the fixed canonical-role rubric, and writes the cached `role` plus audit fields such as `role_confidence`, `role_source`, `role_model`, `role_prompt_sha256`, `role_rationale`, `role_raw_output`, `role_validation_status`, and `source_role_hint`. `model-score-dataset` and the underlying `impact_fusion` scorer then read the cached `role`; scoring does not ask the model to classify citation roles. If a cached context has `role_source=model` and `role=unknown`, it remains `unknown` even when the sentence contains older keyword cues.

The unit is an incoming internal citation edge: a paper in the supplied dataset cites another paper in the same dataset through `references[].reference_id`. Repeated reference entries inside the same citing paper are deduplicated during normalization, and citation contexts are remapped to the kept reference ID, so PDF extraction duplicates do not inflate `effective_citation_*` counts. Each source-paper to cited-paper edge receives at most one primary weighted credit, based on the stronger of explicit citation-context roles and constrained ImpactRankings-style local-model key-reference evidence:

```text
guideline                 -> 2.00
clinical                  -> 1.80
method                    -> 1.60
dataset_or_tool           -> 1.50
negative_or_contrasting   -> 1.10
supporting                -> 1.00
background                -> 0.35
unknown                   -> 0.10
impact_selected_rank_1    -> 1.00
impact_selected_rank_2    -> 0.80
impact_selected_rank_3    -> 0.60
impact_selected_rank_4    -> 0.40
impact_selected_rank_5    -> 0.20
```

Local-model key-reference selections are read from `metadata.impact_selected_references` and compatible metadata keys. They are accepted only when the selected `reference_id` is a real reference of the citing paper and a paper in the supplied dataset. If both context and model-selection evidence exist for the same edge, the scorer uses `max(context_role_weight, impact_selected_reference_weight)` rather than adding both.

`effective_citation_raw` is the sum of those per-edge max weights. `effective_citation_score` is a 0-100 tied-rank percentile computed from `log1p(effective_citation_raw)` inside the supplied dataset. The same raw signal backs `citation_context_quality_score`, so life-science citation roles and constrained key-reference selections affect the evidence-supported impact score without creating an opaque extra model signal.

The output records both the score and the audit counts:

- `effective_citation_edge_count`: incoming internal citation edges.
- `effective_citation_context_count`: explicit citation contexts attached to those edges.
- `effective_citation_count`: incoming edges with a strong role or a constrained top-5 local-model key-reference selection.
- `effective_citation_context_role_raw`, `effective_citation_impact_selected_raw`, `effective_citation_impact_selected_count`, `effective_citation_impact_selected_top5_count`: audit fields separating explicit context evidence from ImpactRankings-style model-selected evidence.
- `effective_citation_guideline_count`, `effective_citation_clinical_count`, `effective_citation_method_count`, `effective_citation_dataset_or_tool_count`, `effective_citation_supporting_count`, `effective_citation_negative_count`, `effective_citation_background_count`, `effective_citation_unknown_count`: role observations from contexts or edge-only records.
- `effective_citation_coverage`: `contextualized`, `partial_context`, `edge_only`, or `missing`.
- `effective_citation_evidence_level`: `high`, `moderate`, `low`, `weak_background_or_unknown`, or `missing`.

Important interpretation boundary: `effective_citation_count` is not the same as global citation count. It only describes role-qualified incoming citations visible inside the supplied dataset. Background and unknown edge-only citations remain visible, but they do not by themselves create a strong effective-citation count.

Life-science external evidence can also be supplied under `metadata.external_impact`. The scorer accepts PubMed/PMC citation count fields (`pubmed_citation_count`, `pmc_citation_count`), direct RCR-like fields (`nih_rcr`, `rcr`, `icite_rcr`, `relative_citation_ratio`), clinical/guideline/approval fields (`clinical_trial_citations`, `clinical_guideline_citations`, `guideline_citations`, `drug_approval_mentions`, `fda_label_mentions`), and protocol/reagent reuse fields (`protocol_reuses`, `assay_reuses`, `reagent_or_assay_reuses`, `antibody_reuses`, `plasmid_reuses`, `cell_line_reuses`). These feed existing auditable components:

- PubMed/PMC counts feed `citation_count_raw` and field/year citation components.
- Direct RCR-like fields feed `relative_citation_ratio_raw` and `relative_citation_ratio_score`.
- Clinical trial, guideline, and approval fields feed `life_science_translation_raw` and `life_science_translation_score`, which are included inside `external_adoption_score`.
- Guideline fields also count as policy/guideline evidence in `policy_citations_raw`.
- Protocol, assay, reagent, and biological-resource reuse fields count as artifact adoption in `artifact_adoption_raw`.

Clinical impact should only be claimed when clinical, guideline, trial, approval, or comparable external evidence is present. If those fields are absent, the system may still rank dataset-local scientific influence, but reports must not describe the score as validated clinical or translational impact.

## Aggregation

Each valid edge contributes to multiple rankings:

- Paper ranking: full edge credit goes to the selected `reference_id`.
- Field ranking: full credit goes to `references[].field` when present.
- Year ranking: full credit goes to `references[].year` when present.
- Author ranking: credit is split equally across reference authors.
- Institution ranking: each author's share is split equally across that author's affiliations.

Example:

```text
reference r1 rank 1 credit = 1.0
r1 has two authors
Author A receives 0.5
Author B receives 0.5
```

If an author has two affiliations, that author's share is split across both affiliations.

## Ranking Score Columns

Each ranking CSV writes report-ready score fields:

- `score`: raw accumulated dataset-local credit. This is the auditable influence score used for sorting.
- `core_score`: top-5 rank-weighted influence. This is the strict ImpactRankings-compatible core signal.
- `expanded_score`: top-20 log-discounted influence. This keeps lower-ranked but still plausible influences instead of hard-cutting after rank 5.
- `context_score`: top-20 log-discounted citation-context bonus. This rewards references whose contexts indicate method/data/baseline support without giving unsupported references extra context credit.
- `graph_score`: local citation-graph PageRank score used as a structural prior.
- `field_year_percentile`: within-field/year percentile based on raw local influence score, when `field` and `year` metadata are available.
- `max_single_edge_share`: share of the raw score contributed by the largest single target-reference edge.
- `final_score`: 0-100 fused paper influence score built from normalized component scores.
- `confidence_flag`: per-paper evidence label: `stable`, `fragile`, or `low_evidence`.
- `confidence_reason`: short audit reason for the per-paper confidence label.
- `normalized_score`: 0-100 score relative to the highest raw score in the same ranking table.
- `percentile`: rank percentile within the same table. `100.000000` means top ranked in that table.

Formula:

```text
core_score uses rank 1-5 weights: 1.0, 0.8, 0.6, 0.4, 0.2
expanded_score uses expanded_weight(rank) = 1 / log2(rank + 1), for rank <= 20
context_score uses expanded_weight(rank) * max(context_weight - 1.0, 0.0)
graph_score uses local PageRank on citation edges inside the input dataset
max_single_edge_share = max(edge_credit for paper) / raw_score
final_score =
  0.40 * normalized_core_score
+ 0.25 * normalized_expanded_score
+ 0.20 * normalized_context_score
+ 0.15 * normalized_graph_score
when field/year metadata are available:
final_score =
  0.36  * normalized_core_score
+ 0.225 * normalized_expanded_score
+ 0.18  * normalized_context_score
+ 0.135 * normalized_graph_score
+ 0.10  * normalized_field_year_percentile
normalized_score = score / max(score in table) * 100
percentile = (N - rank + 1) / N * 100
```

`core_score` is intentionally sparse and high precision. `expanded_score` is less strict and better for seeing medium-strength influence. `context_score` separates citation-context evidence from the raw rank score. `graph_score` keeps a conventional network baseline visible inside the fused score. `field_year_percentile` reduces unfair comparison across fields and publication years when metadata exists; if no usable field/year metadata exists, the final score keeps the four-component formula and does not penalize the dataset. When `final_score` is present, `papers.csv` is sorted by `final_score`; the raw `score` remains available for auditing. `normalized_score` and `percentile` are reporting conveniences. They do not change the ranking and should not be compared across unrelated datasets unless the datasets have the same scope and coverage.

`max_single_edge_share` is an anomaly-risk indicator. A value near `1.0` means the paper's influence score is mostly or entirely supported by one citing target. This can be a real concentrated influence pattern, but it should be reported as fragile evidence when the dataset is small.

Per-paper confidence flags are evidence labels, not score corrections:

- `stable`: the paper has positive influence score plus citation-context or graph support.
- `fragile`: the paper has influence score but weak context/graph support or highly concentrated strict top-k credit.
- `low_evidence`: the paper has no positive raw score or only expanded low-rank evidence without context or graph support.

## Final Paper Impact Score

The `model-score-dataset` command sorts and reports papers by `final_impact_score`. The final score uses only two categories:

1. `true_citation_score`: the true citation-count signal.
2. `effective_citation_score`: the effective citation-count signal.

Formula:

```text
final_impact_score =
  0.95 * true_citation_score
+ 0.05 * effective_citation_score

true_citation_score =
  tied_rank_percentile(log1p(true_citation_raw))

effective_citation_score =
  tied_rank_percentile(log1p(effective_citation_raw))
```

Missing true/effective citation categories contribute 0; they are not imputed and their absence is reported in `final_score_missing_signals`. The default weighting is citation-heavy because expanded public-current-impact checks with OpCitance and NIH iCite show stronger agreement with current RCR and citation-count indicators when the true-citation component dominates, and because effective-citation context coverage is often incomplete.

The scorer also writes sensitivity fields so reports can show whether conclusions depend on the weight choice:

- `score_100_true_00_effective`
- `score_95_true_05_effective`
- `score_90_true_10_effective`
- `score_50_true_50_effective`
- `score_70_true_30_effective`
- `score_80_true_20_effective`
- `score_30_true_70_effective`

`true_citation_raw` is selected from one source per paper, not summed across databases. The current policy collects named compatible source counts, applies the external-title identity guard, then selects the maximum non-null count. Source-specific raw fields remain in the CSV/JSONL so disagreements can be audited.

```text
max(OpenAlex, Semantic Scholar, PubMed/iCite, OpenCitations, Crossref, DataCite, OpenAIRE, Europe PMC, NCBI PubMed, supplied citation_count)
```

The selected source is written to `true_citation_source`. For life-science corpora with better PMID/PMCID coverage than DOI/OpenAlex coverage, PubMed/iCite can be promoted in a documented preprocessing step, but the scoring artifact must disclose the source used for each paper.

`effective_citation_raw` is the weighted count of incoming internal citation edges. The detailed role weights are in [effective-citation-calculation.md](effective-citation-calculation.md). In short, guideline, clinical, method/protocol, dataset/tool, supporting, negative/contrasting citations, and constrained top-5 key-reference selections count as stronger evidence than background-only or unknown edge-only citations. If context and model-selection evidence exist on the same edge, the scorer uses a per-edge maximum instead of summing them.

The main output fields are:

- `final_impact_score`: the report score used for sorting.
- `true_effective_citation_impact_score`: same numeric value as `final_impact_score` before source-quality exclusion.
- `final_score_method`: currently `true_effective_citation_impact_v1`.
- `true_citation_raw`, `true_citation_source`, `true_citation_score`.
- Source-specific true-citation audit fields include `true_citation_openalex_raw`, `true_citation_semantic_scholar_raw`, `true_citation_pubmed_icite_raw`, `true_citation_opencitations_raw`, `true_citation_crossref_raw`, `true_citation_datacite_raw`, `true_citation_openaire_raw`, `true_citation_europepmc_raw`, `true_citation_ncbi_pubmed_raw`, and `true_citation_supplied_raw`.
- `true_citation_identity_status`, `true_citation_identity_similarity`, `true_citation_identity_external_title`, and `true_citation_identity_reason`: external paper-identity audit fields. When an external source title clearly mismatches the local title, source-specific raw counts remain in the CSV, but `true_citation_raw` is left empty and the final score does not use those external citation counts.
- `effective_citation_raw`, `effective_citation_count`, `effective_citation_score`, `effective_citation_coverage`, `effective_citation_evidence_level`.
- `final_score_signal_count`, `final_score_available_signals`, `final_score_missing_signals`.

## Legacy Diagnostic Scores

The scorer still writes `impact_index_v2`, `impact_index_v3`, `model_supported_score`, `evidence_supported_score`, graph, momentum, external-adoption, and model pairwise fields for audit and ablation. These fields do not directly change `final_impact_score`.

Legacy formulas:

```text
impact_index_v3 =
  0.80 * evidence_supported_score
+ 0.20 * model_supported_score

evidence_supported_score =
  0.18 * field_year_normalized_citation_score
+ 0.14 * relative_citation_ratio_score
+ 0.14 * citation_graph_score
+ 0.14 * citation_context_quality_score
+ 0.10 * influential_citation_score
+ 0.10 * temporal_momentum_score
+ 0.20 * external_adoption_score

impact_index_v2 =
  0.30 * model_pairwise_score
+ 0.25 * field_year_normalized_citation_score
+ 0.20 * citation_graph_score
+ 0.15 * citation_context_quality_score
+ 0.10 * temporal_momentum_score
```

All component scores are normalized to 0-100 within the supplied dataset. If a signal is missing for a paper, its weight is redistributed across the available signals inside the same layer. For example, if evidence support has only field/year-normalized citations and Influential Citation evidence, their normalized evidence weights become `0.18 / 0.28` and `0.10 / 0.28`; the model-assisted layer remains separate and is capped by the `0.20 * model_supported_score` outer weight.

The default weights can be replaced by a benchmark-calibrated profile from `calibrate-fusion-weights`. New calibration reports use `impact_index_v3_final_components`: the grid search fits final component weights across `model_pairwise_score`, field/year citations, RCR-like citation ratio, graph score, citation-context quality, Influential Citation evidence, temporal momentum, and external adoption. The loader converts those final component weights back into `impact_index_v3_weights` plus `evidence_first_weights` so the model-assisted and evidence-supported layers remain auditable.

```bash
PYTHONPATH=src python -m local_impact_ranking.cli calibrate-fusion-weights \
  --rankings runs/dev/impact_scores.csv \
  --labels data/dev_pairwise_labels.jsonl \
  --output runs/dev-fusion-calibration

PYTHONPATH=src python -m local_impact_ranking.cli model-score-dataset \
  --input papers.jsonl \
  --output runs/my-impact \
  --base-url http://localhost:8000/v1 \
  --model local-impact-judge \
  --fusion-weights runs/dev-fusion-calibration/fusion_calibration.json
```

The calibration command grid-searches non-negative simplex weights and selects the profile with the best pairwise-label accuracy, using coverage, tie rate, margin, and distance from the default weights as tie-breakers. Reports using calibrated weights should cite the calibration split and keep `fusion.weight_source` with the output artifacts. Legacy v2 `best_weights` files are still accepted for compatibility, but they do not calibrate newer v3 signals such as `relative_citation_ratio_score`, `influential_citation_score`, or `external_adoption_score`.

The calibration input must be a `model-score-dataset` `impact_scores.csv` with all v3 component score columns. Older benchmark index files that contain only `impact_index` are not valid calibration inputs; treating missing component scores as zeros would produce misleading weights.

Component definitions:

- `model_pairwise_score`: the local Qwen-compatible judge's pairwise score, also stored as `impact_index` for audit. `model-score-dataset` uses confidence-weighted Rank Centrality by default so the score is independent of pair order; `elo_rating` remains available as a legacy audit field.
- `field_year_normalized_citation_score`: `log1p(citation_count)` divided by the average `log1p(citation_count)` inside the same `field` and `year` bucket, then scaled to 0-100 with tied-rank percentiles. The log transform and rank-percentile scaling reduce single-paper citation outlier dominance.
- `relative_citation_ratio_score`: an RCR-like local signal. It divides annual citation rate by a supplied expected citation rate (`expected_citation_rate`, `rcr_expected_citation_rate`, or compatible metadata), falling back to a local field/year expected rate only when that bucket has enough papers. It is not a claim of NIH RCR equivalence.
- `citation_graph_score`: local PageRank over references that point to other papers inside the same dataset.
- `citation_context_quality_score`: incoming citation-context support, with method/data/baseline contexts weighted more strongly than background mentions.
- `influential_citation_score`: Semantic Scholar highly influential citation counts or compatible supplied fields (`semantic_scholar_influential_citation_count`, `s2_influential_citation_count`, `influential_citation_count`, `highly_influential_citation_count`), scaled with tied-rank percentiles.
- `temporal_momentum_score`: recent citation share from `metadata.citation_counts_by_year`, `metadata.citations_by_year`, `metadata.yearly_citation_counts`, or compatible top-level fields.
- `external_adoption_score`: optional external evidence combining `external_field_year_percentile_score`, `bip_impact_score`, `patent_impact_score`, `policy_impact_score`, `artifact_adoption_score`, `media_impact_score`, and `award_impact_score`.
- `external_field_year_percentile_score`: external field/year citation percentile from OpenAlex, Semantic Scholar, BIP!, or another calibrated source, accepted as 0-1 or 0-100.
- `bip_impact_score`: BIP! or compatible external impact percentile/score evidence, accepted through fields such as `bip_impact_percentile`, `bip_impact_score`, `bip_impact_class`, `bip_pagerank`, `bip_att_rank`, or `bip_ram`.
- `patent_impact_score`: log-scaled patent citation count scaled inside the supplied dataset with tied-rank percentiles.
- `policy_impact_score`: log-scaled policy citation or policy mention count scaled inside the supplied dataset with tied-rank percentiles.
- `artifact_adoption_score`: log-scaled software/data/model adoption count scaled inside the supplied dataset with tied-rank percentiles, including compatible fields such as `software_mentions`, `github_stars`, `dataset_reuses`, `dataset_downloads`, and `model_downloads`.
- `media_impact_score`: log-scaled media/news/press/altmetric mention count scaled inside the supplied dataset with tied-rank percentiles.
- `award_impact_score`: log-scaled award or prize count scaled inside the supplied dataset with tied-rank percentiles.

The output keeps audit fields: `fusion_signal_count`, `fusion_available_weight`, `fusion_weight_*`, raw component values, `evidence_supported_score`, `model_supported_score`, `credible_evidence_tier`, `missing_high_value_evidence`, and `evidence_quality_flag`. It also writes a quality layer:

- `score_ci_low`, `score_ci_high`, `score_ci_width`: leave-one-signal-out sensitivity interval around the compatibility fusion layer.
- `score_interval_method`: currently `leave_one_signal_out`.
- `dominant_signal` and `max_signal_contribution_share`: the largest single signal contribution to the final score.
- `dominant_score_family`: whether the final score is mainly `evidence_supported` or `model_assisted`.
- `credible_evidence_tier`: `very_high`, `high`, `moderate`, `low`, or `model_only`, based on the number and type of evidence signals.
- `missing_high_value_evidence`: absent or zero high-value external evidence fields, currently external percentile, patent, policy, and artifact adoption evidence.
- `uncertainty_band`: `low`, `medium`, or `high`, based on signal count, interval width, and dominant-signal concentration.
- `reportability_status`: `reportable`, `exploratory`, or `low_confidence`.
- `reportability_reason`: short reason for the reportability label.
- `reportability_basis`: the scoring layer used for the final reportability decision. Current final reports use `impact_index_v3_evidence_first`.
- `legacy_reportability_status`: the earlier v2 signal-count reportability decision retained for audit.
- `validity_confidence`: a separate confidence field capped by the available validation evidence. Without future-window or external labels, rows are marked `validity_status=unvalidated` and cannot be presented as validated impact accuracy.
- `validated_impact_claim`: a run-level model report summary that states whether supplied future-window or external labels support a validated impact claim. This is separate from the numeric `final_impact_score`.
- `impact_credibility_scorecard`: a validation-gate JSON/Markdown artifact that combines label coverage, statistical support, rank alignment, top-rank quality, field-slice stability, and outlier robustness into a report-facing credibility status.
- `impact_claim_certificate`: a run-level JSON/Markdown claim gate that combines score coverage, `validated_impact_claim`, external-evidence readiness, and project readiness into a report status of `validated`, `evidence_supported`, `exploratory`, or `blocked`.
- `model_credibility_card`: a JSON/Markdown model-evidence gate generated by the user workflow from `configs/model_credibility/local-impact-judge-source-task-dpo.json`. It records cross-seed SciImpact accuracy, weak slices, `served_model`, `model_identity_match`, and whether the model can be used only with disclosure or claimed as high credibility. If the served model does not match the profile, the card reports `mismatched_profile` and the user workflow downgrades the run to exploratory use. The current recommended model is moderate credibility, not a standalone oracle.
- `calibration_profile_card`: a JSON/Markdown confidence-calibration gate generated by the user workflow from `configs/calibration/sciimpact-qwen32b-heldout-p50.json`. It records calibration sample size, ECE/MCE/Brier score, `compatible_models`, `calibration_model_identity_match`, and whether calibrated confidence can support report-facing confidence fields. Missing, weak, or mismatched calibration evidence downgrades the user run to exploratory use.

`impact_index_v3` is still dataset-local and diagnostic. It should not be described as global scientific impact unless the input corpus itself is a calibrated global citation/adoption graph.
`final_impact_score` is the two-category citation score used for sorting and reporting. It should be described as dataset-local impact unless the input citation data and validation labels justify a broader claim.

The anomaly report includes `self_citation_risk`, with self-citation edge rate, self-citation credit share, and the top self-cited references. Project readiness warns on high self-citation concentration because it can make a ranking less reportable, but the scorer does not automatically subtract self-citation credit.

Rows also include citation outlier diagnostics: `citation_outlier_flag`, `citation_outlier_ratio`, and `citation_outlier_basis`. These compare raw citation counts against the local field/year median and expose extreme citation-count dominance without replacing field/year normalization or RCR-like normalization.

`model-score-dataset` writes `impact_score_cards.json` and `impact_score_cards.md` for the model-backed path. The cards expose per-paper component scores, raw evidence values, citation outlier diagnostics, missing evidence, confidence, and reportability.

## Audit Artifacts

Every run writes the scoring details needed to reproduce the ranking:

- `reports/scoring_config.json`: scoring version, mode, weights, and context weighting config.
- `audit/scoring_edges.csv`: one row per valid target-reference edge with rank, rank weight, context count, context weight, and final credit.
- `run_manifest.json`: input hash, judge settings, scoring config, environment, and validation summary.

## Limitations

- Scores are dataset-local. Missing target papers or missing candidate references change the ranking.
- The judge can only select from each target paper's candidate references.
- Rank weighting is a modeling choice, not a universal law of scientific impact.
- Context weighting uses transparent lexical cues, so it is interpretable but imperfect.
- `final_score` is a pragmatic weighted fusion for local reporting, not a universal impact metric.
- Benchmark gold metrics evaluate reference selection quality, not global impact.
