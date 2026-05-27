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
