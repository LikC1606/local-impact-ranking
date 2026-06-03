# Citation Data Sources

This document defines how the project compares and uses external citation data sources. The goal is not to declare one source universally correct, but to make each source's role explicit and auditable.

## Summary

Use multiple citation sources as complementary evidence:

| Source | Best use in this project | Useful fields | Main limitation |
| --- | --- | --- | --- |
| OpenAlex | Broad open citation count, field/year context, yearly citation history, DOI/OpenAlex matching | `cited_by_count`, `referenced_works`, `publication_year`, `doi`, `ids`, `fwci`, PMID/PMCID flags | Citation count and cited-by filters can lag; coverage and metadata quality vary by field |
| Semantic Scholar | Influential-citation signal and broad paper metadata | `citationCount`, `influentialCitationCount`, references/citations endpoints, fields of study | Influential-citation evidence depends on available metadata/full-text signals and should not replace raw counts |
| OpenCitations | Auditable DOI-to-DOI citation edges and self-citation metadata | `/citation-count/{doi}`, `/reference-count/{doi}`, `/citations/{doi}`, `/references/{doi}`, `journal_sc`, `author_sc` | Only DOI-to-DOI open citation links; coverage is lower than broad bibliographic indexes |
| PubMed/iCite | Life-science normalization and translational evidence | RCR, Human/Animal/Molecular scores, APT, PMID/PMCID linkage | Biomedical scope; not a complete cross-field citation graph |
| Crossref | DOI metadata and reference availability checks | DOI, bibliographic metadata, reference lists when deposited | Useful for identity/reference grounding, but not enough alone for influence scoring |
| Crossref Data Citations | Dataset/software DOI reuse and data-citation links | `total-results` from the beta Data Citations API | Usually zero for ordinary journal article DOIs; beta endpoint semantics are narrower than general citations |
| DataCite | Data/software/preprint repository DOI evidence | `citationCount`, `viewCount`, `downloadCount`, resource type | Best for DataCite-registered outputs such as Zenodo; many journal DOIs are absent |
| DataCite Events | Incoming DOI-to-DOI citation events, including Crossref/DataCite event links | `references`, `cites`, source and relation-type counts | Event data can overlap with Crossref/OpenCitations; non-citation relations must stay audit-only |
| OpenAIRE | Open research graph citation-impact indicators | `citationImpact.citationCount`, influence, popularity, impulse | Coverage and indicator semantics vary by source aggregation |
| Wikidata | Publicly editable DOI citation graph | DOI property `P356`, cites-work property `P2860` | Sparse, community-curated, and query endpoint can be slow; optional enrichment only |
| PLOS Metrics | PLOS article usage evidence | `counter_total_all`, `counter_total_month` | PLOS-only and usage-oriented; audit/external-adoption evidence, not direct true-citation evidence |
| uCite / NIH-OCC | PubMed-focused citation union evidence for biomedical records | `ucite_citation_count`, `nih_occ_citation_count` | Biomedical/PubMed-centered; import from user-exported files |
| Scopus / Web of Science / Dimensions / Lens / Google Scholar | High-recall external citation corroboration | `scopus_citation_count`, `web_of_science_citation_count`, `dimensions_citation_count`, `lens_citation_count`, `google_scholar_citation_count` | Proprietary/manual sources may not be publicly reproducible; Google Scholar has no stable official bulk API |

Recommended default:

1. Use OpenAlex for broad `citation_count`, yearly history, field/year percentile, and OpenAlex IDs.
2. Run identity recovery after any source that supplies external titles, especially Europe PMC/OpenAlex, so wrong DOI/PMID evidence extracted from references is quarantined before downstream scoring.
3. Use Crossref for DOI-level `is-referenced-by-count` when DOI coverage exists.
4. Use OpenCitations when DOI coverage is good to materialize internal citation graph edges.
5. Use Semantic Scholar for `semantic_scholar_influential_citation_count`.
6. Use PubMed/iCite fields for life-science datasets when PMID/PMCID coverage exists.
7. Use Crossref Data Citations, DataCite, and DataCite Events for repository/data/software DOI evidence and incoming DOI event evidence.
8. Use OpenAIRE for open-graph citation-impact indicators.
9. Use Wikidata and PLOS Metrics only as optional public corroboration because they are sparse, endpoint-limited, or domain-specific.
10. Import high-recall exported counts from uCite, NIH-OCC, Scopus, Web of Science, Dimensions, Lens, or Google Scholar when available.
11. Keep every raw source field in `metadata.external_impact`; do not silently average incompatible counts.

## Source Comparison Outputs

When comparing sources for a dataset, record:

- `paper_count`: total input papers.
- Identifier coverage:
  - DOI coverage
  - OpenAlex ID coverage
  - Semantic Scholar/S2 ID coverage
  - PMID/PMCID coverage for life sciences
- Citation count coverage by source:
  - OpenAlex `openalex_cited_by_count`
  - Semantic Scholar `semantic_scholar_citation_count`
  - OpenCitations `opencitations_cited_by_count`
  - Crossref `crossref_is_referenced_by_count`
  - Crossref Data Citations `crossref_data_citation_count`
  - DataCite `datacite_citation_count`
  - DataCite Events `datacite_events_cited_by_count`
  - OpenAIRE `openaire_citation_count`
  - Wikidata `wikidata_cited_by_count`
  - PubMed/iCite-compatible `pubmed_citation_count`
  - uCite `ucite_citation_count`
  - NIH-OCC `nih_occ_citation_count`
  - Dimensions `dimensions_citation_count`
  - Scopus `scopus_citation_count`
  - Web of Science `web_of_science_citation_count`
  - Lens `lens_citation_count`
  - Google Scholar `google_scholar_citation_count`
- Graph-edge coverage:
  - internal references already present in input
  - OpenCitations materialized internal edges
  - OpenAlex `referenced_works` materialized internal edges, when available
- Normalization coverage:
  - `field_year_percentile`
  - `relative_citation_ratio` / `nih_rcr`
  - citation history for backtesting
- Meaningful-citation coverage:
  - Semantic Scholar `influentialCitationCount`
  - citation contexts for effective citation classification

The existing command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-readiness \
  --input papers.external.jsonl \
  --output runs/external-readiness \
  --markdown
```

writes the current per-paper evidence checklist and missing-evidence summary. Use it as the first source-comparison report.

When a source is available only as a manual or institutional export, generate a reproducible import template first:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli external-evidence-template \
  --input papers.external.jsonl \
  --output external_evidence_template.csv
```

The template contains the matching identifiers plus source-specific count columns for OpenAlex, Semantic Scholar, PubMed/iCite, uCite, NIH-OCC, Dimensions, Scopus, Web of Science, Lens, Google Scholar, Crossref, OpenCitations, DataCite, OpenAIRE, Wikidata, Europe PMC, and NCBI PubMed. Existing input counts are pre-filled. After filling missing exported counts, merge it with `import-external-evidence` and run the citation completeness audit.

For the narrower question "can this run find all citations?", use the citation completeness audit:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli citation-completeness-audit \
  --input papers.external.jsonl \
  --output runs/citation-completeness \
  --markdown
```

It writes:

- `citation_completeness_audit.json`
- `citation_completeness_audit.md`
- `citation_completeness_papers.csv`

The audit grades each paper as `multi_source_compatible`, `source_conflict`, `single_source_only`, `single_source_zero`, `multi_source_zero`, `not_found_in_sources`, `no_query_id`, or `external_identity_mismatch`. It also writes a 0-1 `completeness_score`, a `coverage_confidence` band, and `recommended_next_source_actions` such as resolving identifiers, adding a second independent source, querying high-recall sources, or using PubMed union sources for life-science records. A zero count is treated as a stronger zero only when at least two independent numeric sources report zero. Single-source zero and no-source cases are kept as missing-risk evidence.

## Reconciliation Policy

### Identifier Matching

Normalize identifiers before comparing counts:

- DOI: lowercase, strip `https://doi.org/`, `http://doi.org/`, and `doi:`.
- OpenAlex: keep the work ID suffix, e.g. `W123`.
- Semantic Scholar: keep S2 paper ID / CorpusID separately.
- PubMed: keep PMID and PMCID separately.

Match priority:

1. `paper_id`
2. DOI
3. OpenAlex ID
4. Semantic Scholar/S2 ID
5. PMID/PMCID
6. strict normalized title only for diagnostics, not high-confidence matching

### Count Disagreements

Do not average citation counts across sources. Citation indexes have different coverage, update schedules, and inclusion rules. Instead:

- Keep source-specific raw fields.
- Use source-specific components where they are strongest.
- Flag large disagreement as an audit issue, not an automatic penalty.

Suggested disagreement diagnostics:

```text
max_source_count = max(non_null_source_counts)
min_source_count = min(non_null_source_counts)
citation_count_spread = max_source_count - min_source_count
citation_count_ratio = max_source_count / max(min_source_count, 1)
```

Interpretation:

- Ratio below 2: usually acceptable source variation.
- Ratio 2-5: disclose as moderate disagreement.
- Ratio above 5: inspect identifiers, duplicates, review articles, retractions, and source coverage.

### Component Mapping

Use each source as follows:

| Project component | Preferred evidence | Fallback evidence |
| --- | --- | --- |
| `field_year_normalized_citation_score` | OpenAlex citation count + field/year bucket | Semantic Scholar or PubMed count if OpenAlex is absent |
| `relative_citation_ratio_score` | iCite/NIH RCR or supplied expected citation rate | local field/year expected citation rate when bucket has enough papers |
| `citation_graph_score` | internal references + OpenCitations/OpenAlex materialized internal edges | none |
| `citation_context_quality_score` | explicit citation contexts and effective-citation roles | edge-only evidence gets weak unknown credit |
| `influential_citation_score` | Semantic Scholar `influentialCitationCount` | compatible user-supplied influential-citation count |
| `external_adoption_score` | policy/guideline, patent, artifact, media, award, BIP, life-science translation fields | user-supplied CSV/JSONL evidence |

## Life-Science Preference

For life sciences, prioritize:

1. PubMed/iCite RCR for article-level field/time normalization.
2. uCite / NIH-OCC citation-union exports when you need a stronger PubMed-centered citation check.
3. PMID/PMCID linkage for biomedical identity confidence.
4. Clinical trial, guideline, drug approval, protocol reuse, reagent/assay reuse, and dataset/tool reuse evidence.
5. Effective citations from citation contexts.
6. OpenAlex and Semantic Scholar as broad coverage signals.
7. OpenCitations for DOI-to-DOI graph edges when DOI coverage is high.

Do not claim clinical impact from citation counts alone. Clinical or translational claims need clinical/guideline/trial/approval evidence or a validated domain-specific benchmark.

For extracted PDF datasets, DOI/PMID values can be accidentally taken from a reference inside the article rather than from the article itself. The scorer therefore performs an external-title identity check when an external title such as `metadata.europepmc_title` is available. Clear title mismatches are reported as `true_citation_identity_status=mismatch`; the source-specific raw citation counts are retained for audit, but they are excluded from `true_citation_raw` and `final_impact_score`.

When `recover-external-identities` is enabled, mismatched rows are handled conservatively: old DOI/PMID/OpenAlex/Semantic Scholar IDs and identity-derived impact fields are moved to `metadata.rejected_external_ids`; a local-title OpenAlex search is accepted only above the high title-similarity threshold; unresolved rows keep no quarantined citation evidence in the score. OpenAlex-derived `field` labels are also removed from unresolved mismatches.

## Current Repository Support

Implemented ingestion paths:

- `enrich-openalex`
- `enrich-europepmc`
- `enrich-crossref`
- `enrich-crossref-data-citations`
- `enrich-icite`
- `enrich-ncbi-citations`
- `recover-external-identities`
- `enrich-opencitations`
- `enrich-semantic-scholar`
- `enrich-datacite`
- `enrich-datacite-events`
- `enrich-openaire`
- `enrich-wikidata`
- `enrich-plos-metrics`
- `import-external-evidence`
- `materialize-internal-citation-edges`
- `external-evidence-readiness`

Relevant output files:

- `impact_scores.csv`
- `impact_scores.jsonl`
- `external_evidence_readiness.json`
- `external_evidence_paper_coverage.csv`
- `citation_completeness_audit.json`
- `citation_completeness_papers.csv`
- `impact_score_cards.md`
- `impact_claim_certificate.md`

## References

- OpenAlex Works API documents `cited_by_count`, `referenced_works`, `fwci`, DOI, PMID, PMCID, and work filtering fields: https://developers.openalex.org/api-reference/works
- Semantic Scholar Academic Graph API provides paper, citation, and influential-citation fields: https://api.semanticscholar.org/api-docs/graph
- Semantic Scholar API overview describes its academic graph, citation metadata, and API key/rate-limit policy: https://www.semanticscholar.org/product/api
- OpenCitations REST API exposes DOI-level citation count, reference count, incoming citations, outgoing references, and self-citation metadata: https://api.opencitations.net/index/v1
- NIH iCite documents Relative Citation Ratio, Human/Animal/Molecular scores, and Approximate Potential to Translate: https://support.icite.nih.gov/hc/en-us/articles/9062490125083-Metrics
- Crossref REST API exposes DOI metadata and `is-referenced-by-count`: https://api.crossref.org/works
- Crossref Data Citations API exposes DOI-level data-citation result counts: https://api.crossref.org/beta/datacitations/
- DataCite REST API exposes DOI metadata, citation counts, view counts, and download counts for DataCite records: https://api.datacite.org/dois
- DataCite Event Data exposes DOI-to-DOI event counts with source and relation-type facets: https://api.datacite.org/events
- OpenAIRE Graph API exposes research products and citation-impact indicators: https://graph.openaire.eu/docs/apis/graph-api/searching-entities/filtering-search-results/
- Wikidata Query Service can count public `P2860` cites-work links after DOI lookup by `P356`: https://query.wikidata.org/
- PLOS Search API exposes article metadata and PLOS usage counters for PLOS DOI records: https://api.plos.org/search
