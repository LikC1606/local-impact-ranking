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

Each ranking CSV writes three score fields:

- `score`: raw accumulated dataset-local credit. This is the auditable influence score used for sorting.
- `normalized_score`: 0-100 score relative to the highest raw score in the same ranking table.
- `percentile`: rank percentile within the same table. `100.000000` means top ranked in that table.

Formula:

```text
normalized_score = score / max(score in table) * 100
percentile = (N - rank + 1) / N * 100
```

`normalized_score` and `percentile` are reporting conveniences. They do not change the ranking and should not be compared across unrelated datasets unless the datasets have the same scope and coverage.

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
- Benchmark gold metrics evaluate reference selection quality, not global impact.
