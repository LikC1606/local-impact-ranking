# Input Schema

The pipeline reads JSONL: one target paper per line. Each line must validate as a `Paper` record.

For source-specific citation evidence and reconciliation rules, see [citation-data-sources.md](citation-data-sources.md). For the exact role weights and normalization used by `effective_citation_*` fields, see [effective-citation-calculation.md](effective-citation-calculation.md).

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

## External Impact Evidence

For the final `impact_index_v3` score, the strongest optional fields live under `metadata.external_impact`:

```json
{
  "metadata": {
    "external_impact": {
      "field_year_percentile": 0.95,
      "nih_rcr": 2.4,
      "pubmed_citation_count": 120,
      "bip_impact_percentile": 0.90,
      "semantic_scholar_influential_citation_count": 7,
      "patent_citations": 4,
      "policy_citations": 2,
      "clinical_trial_citations": 3,
      "clinical_guideline_citations": 2,
      "drug_approval_mentions": 1,
      "media_mentions": 12,
      "award_count": 1,
      "expected_citation_rate": 5.0,
      "software_mentions": 20,
      "dataset_reuses": 8,
      "protocol_reuses": 4,
      "reagent_or_assay_reuses": 5,
      "model_downloads": 10000
    }
  }
}
```

Accepted compatible names include:

- External percentile: `field_year_percentile`, `field_year_citation_percentile`, `citation_percentile`, `openalex_percentile`.
- Relative citation evidence for life sciences: `relative_citation_ratio`, `nih_rcr`, `rcr`, `icite_rcr`, `relative_citation_ratio_value`.
- PubMed/PMC citation count evidence: `pubmed_citation_count`, `pmid_citation_count`, `pmc_citation_count`.
- BIP impact evidence: `bip_impact_percentile`, `bip_percentile`, `bip_impact_score`, `bip_impact_class`, `bip_pagerank`, `bip_att_rank`, `bip_ram`.
- Semantic Scholar influential citations: `semantic_scholar_influential_citation_count`, `s2_influential_citation_count`, `influential_citation_count`, `highly_influential_citation_count`.
- Patent evidence: `patent_citations`, `patent_citation_count`, `patent_count`.
- Policy/guideline evidence: `policy_citations`, `policy_citation_count`, `policy_mentions`, `guideline_citations`, `clinical_guideline_citations`, `practice_guideline_citations`, `clinical_guideline_mentions`, `guideline_mentions`, `consensus_guideline_citations`.
- Life-science translational evidence: `life_science_translation`, `clinical_trial_citations`, `clinical_trial_mentions`, `clinical_study_citations`, `trial_registry_citations`, `drug_approval_mentions`, `fda_label_mentions`, `ema_label_mentions`, `diagnostic_guideline_mentions`, `therapeutic_guideline_mentions`.
- Media/public-attention evidence: `media_mentions`, `news_mentions`, `press_mentions`, `altmetric_mentions`.
- Award/recognition evidence: `award_count`, `awards`, `best_paper_awards`, `prize_count`.
- Expected citation-rate evidence: `expected_citation_rate`, `rcr_expected_citation_rate`, `field_expected_citation_rate`, `field_citation_rate`.
- Artifact adoption: `artifact_adoption`, `artifact_adoption_count`, `software_mentions`, `software_citations`, `github_stars`, `repository_stars`, `dataset_reuses`, `dataset_citations`, `model_downloads`, `dataset_downloads`, `huggingface_downloads`, `protocol_reuses`, `protocol_citations`, `assay_reuses`, `assay_citations`, `reagent_reuses`, `reagent_citations`, `reagent_or_assay_reuses`, `antibody_reuses`, `plasmid_reuses`, `cell_line_reuses`, `mouse_model_reuses`, `biobank_reuses`.

These fields can also appear directly in `metadata`, `metadata.impact`, `metadata.metrics`, or as top-level extra fields. Percentiles may be 0-1 or 0-100. Count fields are log-scaled before normalization to reduce extreme outlier dominance. `expected_citation_rate` supports the RCR-like relative citation component; if it is absent, the scorer only uses a local field/year expected rate when enough comparable papers are present.

If OpenAlex, BIP!, Semantic Scholar, or another source has already been exported to CSV/JSONL, merge it locally before scoring:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli import-external-evidence \
  --input papers.jsonl \
  --evidence external_evidence.csv \
  --output papers.external.jsonl \
  --summary-output runs/external-import-summary.json
```

The import command matches by `paper_id`, DOI, OpenAlex ID, or Semantic Scholar/S2 ID and writes compatible fields under `metadata.external_impact`.

You can populate citation count, yearly citation history, and field/year percentile from OpenAlex when papers have DOI or OpenAlex IDs:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-openalex \
  --input papers.jsonl \
  --output papers.openalex.jsonl \
  --email you@example.com
```

The command writes `metadata.external_impact.field_year_percentile`, `metadata.external_impact.openalex_cited_by_count`, `metadata.citation_counts_by_year`, and `metadata.external_ids.openalex` when OpenAlex returns a match.

You can populate Semantic Scholar citation and influential-citation evidence when papers have DOI or Semantic Scholar identifiers:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-semantic-scholar \
  --input papers.openalex.jsonl \
  --output papers.semantic-scholar.jsonl
```

The command writes compatible Semantic Scholar identifiers, citation counts, fields of study, and `metadata.external_impact.semantic_scholar_influential_citation_count` when a paper is matched.

You can also populate DOI-level citation evidence from OpenCitations COCI/index v2 when papers have DOI IDs:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-opencitations \
  --input papers.openalex.jsonl \
  --output papers.opencitations.jsonl \
  --materialize-internal-edges
```

The command writes `metadata.external_impact.opencitations_cited_by_count`, `metadata.external_impact.opencitations_reference_count`, and `metadata.opencitations_incoming_edges`. With `--materialize-internal-edges`, DOI-matched incoming citations from other papers in the same dataset are added to the citing paper's `references`, making `citation_graph_score` and PageRank usable even when the original input did not include all internal reference edges.

Run `external-evidence-readiness` after enrichment to decide whether the dataset can support high-credibility reporting:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-readiness \
  --input papers.opencitations.jsonl \
  --output runs/external-readiness \
  --markdown
```

The same report is generated automatically by the full `run` command under `reports/` and by `model-score-dataset` in its output directory. Both paths also write `external_evidence_paper_coverage.csv`, a per-paper checklist with `missing_evidence` labels for rows that lack DOI/OpenAlex identifiers, enrichment, citation counts, field/year percentiles, or usable internal/OpenCitations graph edges. The readiness JSON/Markdown also includes `paper_coverage_summary`, which counts the most common missing-evidence labels so report readers can see the main evidence bottlenecks without opening the CSV.

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

For life-science datasets, citation contexts are also used to record effective citations for each cited paper. The scorer classifies incoming citation contexts into auditable roles:

- `guideline`: clinical guideline, consensus, or practice recommendation citation.
- `clinical`: patient, clinical trial, diagnostic, therapeutic, prognostic, cohort, or survival context.
- `method`: protocol, assay, reagent, antibody, plasmid, cell line, model, sequencing, PCR, ELISA, CRISPR, or other experimental-method reuse.
- `dataset_or_tool`: dataset, database, software, pipeline, code, benchmark, accession, or repository reuse.
- `supporting`: cites the paper as supporting or consistent evidence.
- `negative_or_contrasting`: cites the paper as contradicted, not replicated, conflicting, or refuted.
- `background`: general background mention.
- `unknown`: internal citation edge exists but no citation context was supplied.

If your extractor already provides a citation-intent label, put it in each context as `label`, `role`, `intent`, `citation_intent`, or `category`. Supported values include `background`, `method`, `result`, `supporting`, `clinical`, `guideline`, `dataset`, `tool`, and `negative`.

The output `impact_scores.csv/jsonl` records per-paper fields such as `effective_citation_count`, `effective_citation_score`, `effective_citation_guideline_count`, `effective_citation_clinical_count`, `effective_citation_method_count`, `effective_citation_dataset_or_tool_count`, `effective_citation_background_count`, `effective_citation_unknown_count`, `effective_citation_coverage`, and `effective_citation_evidence_level`.

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
