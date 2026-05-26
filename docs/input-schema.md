# Input Schema

The pipeline reads JSONL: one target paper per line. Each line must validate as a `Paper` record.

## Minimal Record

```json
{
  "paper_id": "p1",
  "title": "Target Paper",
  "references": [
    {
      "reference_id": "r1",
      "title": "Candidate Reference"
    }
  ]
}
```

Required fields:

- `paper_id`: stable ID for the target paper.
- `title`: target paper title.
- `references`: candidate references the judge may select from.
- `references[].title`: reference title.

Recommended fields:

- `abstract`: target abstract used by the judge prompt.
- `year`, `venue`, `field`: used in reports and field/year rankings.
- `authors`: target authors and affiliations.
- `references[].reference_id`: stable candidate ID. If missing, normalization creates one, but stable IDs are strongly preferred.
- `references[].authors`, `references[].year`, `references[].venue`, `references[].field`, `references[].doi`, `references[].external_ids`: used for ranking aggregation and auditability.
- `citation_contexts`: citation sentences connecting the target paper to candidate references.
- `metadata`: benchmark labels or extra source-specific fields.

## Authors

```json
{
  "name": "Author A",
  "affiliations": ["Institution X"],
  "ids": {
    "orcid": "0000-0000-0000-0000"
  }
}
```

`ids` is optional and may contain ORCID, Semantic Scholar author IDs, OpenAlex IDs, or local IDs.

## References

```json
{
  "reference_id": "r1",
  "title": "Reference Paper",
  "authors": [{"name": "Author B", "affiliations": ["Institution Y"]}],
  "year": 2020,
  "venue": "Conference",
  "field": "Machine Learning",
  "doi": "10.0000/example",
  "external_ids": {
    "semantic_scholar": "123456",
    "openalex": "W123456"
  },
  "raw_citation": "Author B. Reference Paper. Conference, 2020."
}
```

The judge can only choose references present in `references`. Do not put global citation graph neighbors in `metadata` and expect them to be selectable.

## Citation Contexts

```json
{
  "reference_id": "r1",
  "sentence": "We build on the representation method introduced by Reference Paper.",
  "paragraph": "Full paragraph text is optional.",
  "section": "Methods"
}
```

`reference_id` must match a candidate reference. Contexts help the prompt and `context_weighted` scoring distinguish method/data/baseline support from background-only citations.

## Gold Labels For Benchmark Evaluation

If the dataset has benchmark positives, store them in `metadata`:

```json
{
  "metadata": {
    "gold_reference_ids": ["r1", "r3"]
  }
}
```

The PreScience adapter uses:

```json
{
  "metadata": {
    "prescience_positive_reference_ids": ["r1", "r3"]
  }
}
```

Both keys are supported by `gold_evaluation.json`. These labels evaluate whether the judge selected benchmark-positive references. They do not prove global scientific impact.

## CSV Conversion

Use `convert-csv` when the source data starts as a table:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli convert-csv \
  --input papers.csv \
  --output papers.jsonl
```

Supported CSV columns:

- `paper_id`
- `title`
- `abstract`
- `year`
- `venue`
- `field`
- `authors_json`
- `references_json`
- `citation_contexts_json`
- `metadata_json`

The `*_json` columns must contain JSON arrays or objects using the same schema as JSONL.

## Validation

Run validation before judging:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli validate \
  --input papers.jsonl \
  --output runs/validation
```

Validation reports duplicate paper IDs, empty reference lists, duplicate reference IDs within a target paper, missing titles, and out-of-range years.

## Example

See `examples/context_papers.jsonl` for a small record with citation contexts and author/affiliation metadata.
