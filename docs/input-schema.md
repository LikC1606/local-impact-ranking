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
- `metadata.external_ids.pmid`, `metadata.pmid`, or top-level `pmid`: PubMed identifier used by the public NIH iCite/RCR benchmark.
- `metadata.external_ids.doi`, top-level `doi`, `metadata.external_ids.openalex`, `metadata.external_ids.semantic_scholar`, `metadata.external_ids.pmid`, and `metadata.external_ids.pmcid`: identifiers used by citation enrichment and `citation-completeness-audit`.
- `references[].reference_id`: stable candidate ID. If missing, normalization creates one, but stable IDs are strongly preferred.
- `references[].authors`, `references[].year`, `references[].venue`, `references[].field`, `references[].doi`, `references[].external_ids`: used for ranking aggregation and auditability.
- `citation_contexts`: citation sentences connecting the target paper to candidate references.
- `metadata`: benchmark labels or extra source-specific fields.

## Reference Deduplication

Duplicate references are handled within each target paper only. The system never removes the fact that two different papers cite the same work; those are valid incoming citation edges.

During `run` and `judge`, normalization deduplicates repeated entries in one paper's `references` list using conservative keys in this order:

- DOI from `references[].doi` or DOI-like `references[].external_ids`.
- OpenAlex, Semantic Scholar/S2, or PMID values from `references[].external_ids`.
- Normalized `references[].title` plus `references[].year` plus the first reference authors.
- Repeated `references[].reference_id`.

The first occurrence is kept. Missing metadata such as authors or external IDs can be filled from removed duplicates. Source provenance from removed duplicates, such as raw citations or internal-citation match methods, is kept on the retained reference as `deduplicated_reference_evidence`. Any `citation_contexts[].reference_id` pointing to a removed duplicate is remapped to the kept reference ID. When deduplication happens, the paper receives an audit object:

```json
{
  "metadata": {
    "reference_deduplication": {
      "original_reference_count": 2,
      "deduplicated_reference_count": 1,
      "removed_reference_count": 1,
      "duplicate_reference_id_map": {"r2": "r1"}
    }
  }
}
```

## Citation Evidence For Final Impact

For the final `final_impact_score`, only two evidence categories are used: true citation count and effective citation count. True citation counts can live under `metadata.external_impact` or as top-level compatible fields:

```json
{
  "metadata": {
    "external_impact": {
      "openalex_cited_by_count": 132,
      "semantic_scholar_citation_count": 128,
      "pubmed_citation_count": 120,
      "opencitations_cited_by_count": 110,
      "crossref_is_referenced_by_count": 108,
      "crossref_data_citation_count": 2,
      "datacite_citation_count": 5,
      "datacite_events_cited_by_count": 6,
      "openaire_citation_count": 4,
      "wikidata_cited_by_count": 3,
      "ucite_citation_count": 121,
      "nih_occ_citation_count": 119,
      "dimensions_citation_count": 140,
      "scopus_citation_count": 136,
      "web_of_science_citation_count": 130,
      "lens_citation_count": 138,
      "google_scholar_citation_count": 155
    }
  }
}
```

The scorer selects one true citation source per paper from named, compatible sources. It does not sum citation counts across databases. When multiple named sources are present, the current policy selects the maximum non-null count after identity checks, while keeping every raw source count for audit.

```text
max(named source counts, supplied citation_count only when it is the only value or higher)
```

Accepted true-citation names include:

- OpenAlex: `openalex_cited_by_count`, `openalex_citation_count`, `openalex_citations`.
- Semantic Scholar: `semantic_scholar_citation_count`, `s2_citation_count`, `semantic_scholar_citations`.
- PubMed/iCite: `pubmed_citation_count`, `pmid_citation_count`, `pmc_citation_count`, `icite_citation_count`.
- PubMed union/high-recall biomedical imports: `ucite_citation_count`, `nih_occ_citation_count`.
- High-recall external imports: `dimensions_citation_count`, `scopus_citation_count`, `web_of_science_citation_count`, `lens_citation_count`, `google_scholar_citation_count`.
- OpenCitations: `opencitations_cited_by_count`, `opencitations_citation_count`, `opencitations_citations`.
- Crossref: `crossref_cited_by_count`, `crossref_is_referenced_by_count`, `is_referenced_by_count`, `is-referenced-by-count`.
- Crossref Data Citations: `crossref_data_citation_count`, `crossref_data_citations_total_results`.
- DataCite: `datacite_citation_count`, `datacite_cited_by_count`.
- DataCite Events: `datacite_events_cited_by_count`, `datacite_events_citation_count`.
- OpenAIRE: `openaire_citation_count`, `openaire_cited_by_count`.
- Wikidata: `wikidata_cited_by_count`, `wikidata_citation_count`.
- Europe PMC: `europepmc_cited_by_count`, `europepmc_citation_count`.
- NCBI PubMed: `ncbi_pubmed_citedin_count`, `ncbi_pubmed_cited_by_count`.
- Supplied generic count: `citation_count`, `cited_by_count`, `citations`.

Run `citation-completeness-audit` after enrichment to separate citation evidence quality from the score itself. The audit reports per-paper source counts, source conflict, missing DOI/OpenAlex/Semantic Scholar/PMID/PMCID identifiers, and whether a zero citation count is multi-source-confirmed. It cannot certify that every citation has been found; it exposes the residual missing-citation risk.

Effective citations come from dataset-internal reference edges plus citation contexts and optional constrained ImpactRankings-style key-reference selections:

```json
{
  "paper_id": "paper_that_cites",
  "references": [{"reference_id": "cited_paper", "title": "Cited paper"}],
  "citation_contexts": [
    {
      "reference_id": "cited_paper",
      "sentence": "We used this method as the assay protocol.",
      "section": "Methods",
      "role": "method"
    }
  ]
}
```

If a local model has selected key references for a citing paper, store the constrained selections under `metadata.impact_selected_references`:

```json
{
  "paper_id": "paper_that_cites",
  "references": [{"reference_id": "cited_paper", "title": "Cited paper"}],
  "metadata": {
    "impact_selected_references": [
      {"reference_id": "cited_paper", "rank": 1, "confidence": 0.82, "valid": true}
    ]
  }
}
```

Only selections whose `reference_id` exists in the citing paper's `references` list and in the supplied dataset can contribute to `effective_citation_*`. This keeps the ImpactRankings-style signal constrained to real local citation edges.

For diagnostics and ablations, additional external fields can still be stored under `metadata.external_impact`, but they do not directly change `final_impact_score`. Compatible diagnostic names include:

- External percentile: `field_year_percentile`, `field_year_citation_percentile`, `citation_percentile`, `openalex_percentile`.
- Relative citation evidence for life sciences: `relative_citation_ratio`, `nih_rcr`, `rcr`, `icite_rcr`, `relative_citation_ratio_value`.
- BIP impact evidence: `bip_impact_percentile`, `bip_percentile`, `bip_impact_score`, `bip_impact_class`, `bip_pagerank`, `bip_att_rank`, `bip_ram`.
- Semantic Scholar influential citations: `semantic_scholar_influential_citation_count`, `s2_influential_citation_count`, `influential_citation_count`, `highly_influential_citation_count`.
- Patent evidence: `patent_citations`, `patent_citation_count`, `patent_count`.
- Policy/guideline evidence: `policy_citations`, `policy_citation_count`, `policy_mentions`, `guideline_citations`, `clinical_guideline_citations`, `practice_guideline_citations`, `clinical_guideline_mentions`, `guideline_mentions`, `consensus_guideline_citations`.
- Life-science translational evidence: `life_science_translation`, `clinical_trial_citations`, `clinical_trial_mentions`, `clinical_study_citations`, `trial_registry_citations`, `drug_approval_mentions`, `fda_label_mentions`, `ema_label_mentions`, `diagnostic_guideline_mentions`, `therapeutic_guideline_mentions`.
- Media/public-attention evidence: `media_mentions`, `news_mentions`, `press_mentions`, `altmetric_mentions`.
- Award/recognition evidence: `award_count`, `awards`, `best_paper_awards`, `prize_count`.
- Expected citation-rate evidence: `expected_citation_rate`, `rcr_expected_citation_rate`, `field_expected_citation_rate`, `field_citation_rate`.
- Artifact adoption: `artifact_adoption`, `artifact_adoption_count`, `software_mentions`, `software_citations`, `github_stars`, `repository_stars`, `dataset_reuses`, `dataset_citations`, `model_downloads`, `dataset_downloads`, `huggingface_downloads`, `protocol_reuses`, `protocol_citations`, `assay_reuses`, `assay_citations`, `reagent_reuses`, `reagent_citations`, `reagent_or_assay_reuses`, `antibody_reuses`, `plasmid_reuses`, `cell_line_reuses`, `mouse_model_reuses`, `biobank_reuses`.

These fields can also appear directly in `metadata`, `metadata.impact`, `metadata.metrics`, or as top-level extra fields. Percentiles may be 0-1 or 0-100. Count fields are log-scaled before normalization to reduce extreme outlier dominance. `expected_citation_rate` supports the RCR-like diagnostic component; if it is absent, the scorer only uses a local field/year expected rate when enough comparable papers are present.

If OpenAlex, BIP!, Semantic Scholar, uCite, NIH-OCC, Scopus, Web of Science, Dimensions, Lens, Google Scholar, or another source has already been exported to CSV/JSONL, merge it locally before scoring:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-template \
  --input papers.jsonl \
  --output external_evidence_template.csv
```

The template includes stable matching columns (`paper_id`, DOI, OpenAlex ID, Semantic Scholar/S2 ID, PMID, PMCID), supported source-count columns, and optional external title / identity-status columns. Existing input counts are pre-filled; fill the remaining source-count columns from your exported databases, then import the file:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli import-external-evidence \
  --input papers.jsonl \
  --evidence external_evidence_template.csv \
  --output papers.external.jsonl \
  --summary-output runs/external-import-summary.json
```

The import command matches by `paper_id`, DOI, OpenAlex ID, Semantic Scholar/S2 ID, PMID, or PMCID and writes compatible fields under `metadata.external_impact`. Its summary includes `evidence_validation`, which flags rows without match keys, duplicate match keys, invalid numeric cells, and negative numeric cells. Proprietary/manual exports are treated as user-supplied evidence: they improve source coverage and auditability, but they are not summed and do not prove exhaustive citation recovery.

You can populate citation count, yearly citation history, and field/year percentile from OpenAlex when papers have DOI or OpenAlex IDs:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-openalex \
  --input papers.jsonl \
  --output papers.openalex.jsonl \
  --email you@example.com \
  --resume
```

The command writes `metadata.external_impact.field_year_percentile`, `metadata.external_impact.openalex_cited_by_count`, `metadata.citation_counts_by_year`, and `metadata.external_ids.openalex` when OpenAlex returns a match. `--resume` reuses rows already present in the output JSONL, which is useful for interrupted large enrichment runs.

You can populate Semantic Scholar citation and influential-citation evidence when papers have DOI or Semantic Scholar identifiers:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-semantic-scholar \
  --input papers.openalex.jsonl \
  --output papers.semantic-scholar.jsonl \
  --batch-size 100 \
  --resume
```

The command writes compatible Semantic Scholar identifiers, citation counts, fields of study, and `metadata.external_impact.semantic_scholar_influential_citation_count` when a paper is matched. `--batch-size` chunks identifier lookups, and `--resume` reuses rows already present in the output JSONL.

You can populate DOI-level Crossref cited-by evidence:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-crossref \
  --input papers.europepmc.jsonl \
  --output papers.crossref.jsonl \
  --summary-output papers.crossref.summary.json \
  --resume
```

You can populate DataCite citation and usage evidence for DataCite-registered DOI outputs such as Zenodo records:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-datacite \
  --input papers.crossref.jsonl \
  --output papers.datacite.jsonl \
  --summary-output papers.datacite.summary.json \
  --resume
```

The command writes `metadata.external_impact.datacite_citation_count`, `metadata.external_impact.datacite_view_count`, and `metadata.external_impact.datacite_download_count`. `--resume` reuses rows already present in the output JSONL.

You can populate incoming DOI citation events from DataCite Event Data:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-datacite-events \
  --input papers.datacite.jsonl \
  --output papers.datacite-events.jsonl \
  --summary-output papers.datacite-events.summary.json \
  --resume
```

The command writes `metadata.external_impact.datacite_events_cited_by_count`, `metadata.external_impact.datacite_events_references_count`, `metadata.external_impact.datacite_events_cites_count`, and source-specific audit counts. `--resume` reuses rows already present in the output JSONL. Only `references` and `cites` relations are treated as citation evidence.

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-crossref-data-citations \
  --input papers.datacite-events.jsonl \
  --output papers.crossref-data-citations.jsonl \
  --summary-output papers.crossref-data-citations.summary.json \
  --resume
```

The command writes `metadata.external_impact.crossref_data_citation_count` and `metadata.external_impact.crossref_data_citations_total_results`, mainly for data/software DOI reuse evidence. `--resume` reuses rows already present in the output JSONL.

You can populate OpenAIRE citation-impact indicators:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-openaire \
  --input papers.datacite.jsonl \
  --output papers.openaire.jsonl \
  --summary-output papers.openaire.summary.json \
  --resume
```

The command writes `metadata.external_impact.openaire_citation_count`, `metadata.external_impact.openaire_influence`, `metadata.external_impact.openaire_popularity`, and `metadata.external_impact.openaire_impulse`.
`--resume` reuses rows already present in the output JSONL, which is useful for interrupted large enrichment runs.

You can optionally populate public Wikidata citation counts:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-wikidata \
  --input papers.openaire.jsonl \
  --output papers.wikidata.jsonl \
  --summary-output papers.wikidata.summary.json \
  --resume
```

The command writes `metadata.wikidata_entity`, `metadata.wikidata_title`, and `metadata.external_impact.wikidata_cited_by_count`. It is optional because Wikidata Query Service can be rate-limited or slow. `--resume` reuses rows already present in the output JSONL.

You can optionally populate PLOS article usage metrics:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-plos-metrics \
  --input papers.wikidata.jsonl \
  --output papers.plos-metrics.jsonl \
  --summary-output papers.plos-metrics.summary.json \
  --resume
```

The command writes `metadata.plos_title`, `metadata.plos_journal`, `metadata.external_impact.plos_counter_total_all`, and `metadata.external_impact.article_usage_count` for PLOS usage-metric audit evidence. `--resume` reuses rows already present in the output JSONL.

You can populate NIH iCite evidence for life-science records when papers have PMID identifiers:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-icite \
  --input papers.europepmc.jsonl \
  --output papers.icite.jsonl \
  --summary-output papers.icite.summary.json \
  --resume
```

The command writes PubMed/iCite citation counts, RCR, APT, NIH percentile, Human/Animal/Molecular scores, and clinical-citation flags under `metadata.external_impact`. `--resume` reuses rows already present in the output JSONL and excludes their PMIDs from new iCite requests.

Use Europe PMC before iCite when DOI coverage is stronger than PMID coverage:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-europepmc \
  --input papers.openalex.jsonl \
  --output papers.europepmc.jsonl \
  --summary-output papers.europepmc.summary.json \
  --resume
```

The command writes `metadata.external_impact.europepmc_cited_by_count`, can recover `metadata.external_ids.pmid` and `metadata.external_ids.pmcid`, and records `metadata.europepmc_match_method` plus `metadata.europepmc_title`. `--resume` reuses rows already present in the output JSONL. The title is used later as an external-identity check: if the external title clearly mismatches the local paper title, the citation counts remain visible for audit but are excluded from `true_citation_raw`.

You can then add NCBI ELink cited-in counts:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-ncbi-citations \
  --input papers.europepmc.jsonl \
  --output papers.ncbi-citations.jsonl \
  --summary-output papers.ncbi-citations.summary.json \
  --resume
```

The command writes `metadata.external_impact.ncbi_pubmed_citedin_count` and `metadata.external_impact.ncbi_pmc_cited_by_count` when PMID/PMCID identifiers are available. PubMed cited-in can support true-citation scoring after identity checks; PMC cited-by is treated as audit evidence because NCBI PMC links can behave differently from article-level cited-in counts.
`--resume` reuses rows already present in the output JSONL and excludes their PMID/PMCID identifiers from new NCBI ELink requests.

You can also populate DOI-level citation evidence from OpenCitations COCI/index v2 when papers have DOI IDs:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli enrich-opencitations \
  --input papers.openalex.jsonl \
  --output papers.opencitations.jsonl \
  --max-incoming-edges-per-paper 500 \
  --materialize-internal-edges \
  --resume
```

The command writes `metadata.external_impact.opencitations_cited_by_count`, `metadata.external_impact.opencitations_reference_count`, and capped `metadata.opencitations_incoming_edges`. If stored incoming edges are truncated, `metadata.opencitations_incoming_edges_truncated_count` records the omitted edge count for that paper. With `--materialize-internal-edges`, DOI-matched incoming citations from other papers in the same dataset are added to the citing paper's `references`, making `citation_graph_score` and PageRank usable even when the original input did not include all internal reference edges. `--resume` reuses rows already present in the output JSONL and can materialize saved internal incoming edges without refetching that DOI.

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

For life-science datasets, citation contexts and constrained key-reference selections are also used to record effective citations for each cited paper. The scorer classifies incoming citation contexts into auditable roles:

- `guideline`: clinical guideline, consensus, or practice recommendation citation.
- `clinical`: patient, clinical trial, diagnostic, therapeutic, prognostic, cohort, or survival context.
- `method`: protocol, assay, reagent, antibody, plasmid, cell line, model, sequencing, PCR, ELISA, CRISPR, or other experimental-method reuse.
- `dataset_or_tool`: dataset, database, software, pipeline, code, benchmark, accession, or repository reuse.
- `supporting`: cites the paper as supporting or consistent evidence.
- `negative_or_contrasting`: cites the paper as contradicted, not replicated, conflicting, or refuted.
- `background`: general background mention.
- `unknown`: internal citation edge exists but no citation context was supplied.

If your extractor already provides a citation-intent label, put it in each context as `label`, `role`, `intent`, `citation_intent`, or `category`. Supported values include `background`, `method`, `result`, `supporting`, `clinical`, `guideline`, `dataset`, `tool`, and `negative`.

The output `impact_scores.csv/jsonl` records per-paper fields such as `effective_citation_count`, `effective_citation_score`, `effective_citation_context_role_raw`, `effective_citation_impact_selected_raw`, `effective_citation_impact_selected_count`, `effective_citation_impact_selected_top5_count`, `effective_citation_guideline_count`, `effective_citation_clinical_count`, `effective_citation_method_count`, `effective_citation_dataset_or_tool_count`, `effective_citation_background_count`, `effective_citation_unknown_count`, `effective_citation_coverage`, and `effective_citation_evidence_level`.

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
