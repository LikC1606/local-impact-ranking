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

Recommended default:

1. Use OpenAlex for broad `citation_count`, yearly history, field/year percentile, and OpenAlex IDs.
2. Use OpenCitations when DOI coverage is good to materialize internal citation graph edges.
3. Use Semantic Scholar for `semantic_scholar_influential_citation_count`.
4. Use PubMed/iCite fields for life-science datasets when PMID/PMCID coverage exists.
5. Keep every raw source field in `metadata.external_impact`; do not silently average incompatible counts.

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
  - PubMed/iCite-compatible `pubmed_citation_count`
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
2. PMID/PMCID linkage for biomedical identity confidence.
3. Clinical trial, guideline, drug approval, protocol reuse, reagent/assay reuse, and dataset/tool reuse evidence.
4. Effective citations from citation contexts.
5. OpenAlex and Semantic Scholar as broad coverage signals.
6. OpenCitations for DOI-to-DOI graph edges when DOI coverage is high.

Do not claim clinical impact from citation counts alone. Clinical or translational claims need clinical/guideline/trial/approval evidence or a validated domain-specific benchmark.

## Current Repository Support

Implemented ingestion paths:

- `enrich-openalex`
- `enrich-opencitations`
- `enrich-semantic-scholar`
- `import-external-evidence`
- `materialize-internal-citation-edges`
- `external-evidence-readiness`

Relevant output files:

- `impact_scores.csv`
- `impact_scores.jsonl`
- `external_evidence_readiness.json`
- `external_evidence_paper_coverage.csv`
- `impact_score_cards.md`
- `impact_claim_certificate.md`

## References

- OpenAlex Works API documents `cited_by_count`, `referenced_works`, `fwci`, DOI, PMID, PMCID, and work filtering fields: https://developers.openalex.org/api-reference/works
- Semantic Scholar Academic Graph API provides paper, citation, and influential-citation fields: https://api.semanticscholar.org/api-docs/graph
- Semantic Scholar API overview describes its academic graph, citation metadata, and API key/rate-limit policy: https://www.semanticscholar.org/product/api
- OpenCitations REST API exposes DOI-level citation count, reference count, incoming citations, outgoing references, and self-citation metadata: https://api.opencitations.net/index/v1
- NIH iCite documents Relative Citation Ratio, Human/Animal/Molecular scores, and Approximate Potential to Translate: https://support.icite.nih.gov/hc/en-us/articles/9062490125083-Metrics
