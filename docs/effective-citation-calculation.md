# Effective Citation Calculation

This document defines how `effective_citation_*` fields are calculated. The purpose is to make "effective citation" auditable instead of letting a model silently infer it from title or abstract text.

## Definition

An effective citation is an incoming citation from another paper in the supplied dataset that has either an explicit role stronger than a generic background mention or a constrained ImpactRankings-style local-model selection marking it as a key reference.

The unit is an internal citation edge:

```text
citing_paper.references[].reference_id == cited_paper.paper_id
```

The scorer then looks for `citation_contexts` on the citing paper whose `reference_id` matches the cited paper. It can also consume constrained local-model selections from `metadata.impact_selected_references`; those selections are only accepted when the selected `reference_id` exists in the citing paper's real `references` list and in the supplied dataset.

## Input Requirements

Minimum for edge-only evidence:

```json
{
  "paper_id": "trial",
  "title": "Clinical trial using prior assay",
  "references": [
    {
      "reference_id": "assay",
      "title": "Reusable biomarker assay"
    }
  ]
}
```

Recommended for effective citation evidence:

```json
{
  "paper_id": "trial",
  "title": "Clinical trial using prior assay",
  "references": [
    {
      "reference_id": "assay",
      "title": "Reusable biomarker assay"
    }
  ],
  "citation_contexts": [
    {
      "reference_id": "assay",
      "sentence": "Patients were stratified using the assay protocol described by the prior study.",
      "section": "Methods"
    }
  ]
}
```

If a citation-intent extractor is available, it can provide an explicit label:

```json
{
  "reference_id": "assay",
  "sentence": "We used the published assay protocol.",
  "section": "Methods",
  "label": "method"
}
```

Supported explicit label keys:

- `role`
- `label`
- `intent`
- `citation_intent`
- `category`

Supported explicit label values:

- `guideline`
- `clinical`
- `method`
- `dataset`
- `tool`
- `supporting`
- `background`
- `negative`

ImpactRankings-style local-model evidence can be supplied in the citing paper metadata:

```json
{
  "paper_id": "trial",
  "title": "Clinical trial using prior assay",
  "references": [{"reference_id": "assay", "title": "Reusable biomarker assay"}],
  "metadata": {
    "impact_selected_references": [
      {
        "reference_id": "assay",
        "rank": 1,
        "confidence": 0.82,
        "valid": true
      }
    ]
  }
}
```

Compatible metadata keys are `impact_selected_references`, `impactrankings_selected_references`, `llm_selected_references`, `model_selected_references`, `judge_selected_references`, and `selected_references`. Invalid selections, hallucinated IDs, and IDs that are not actual references of the citing paper are ignored.

## Role Classification

Each citation context is assigned one or more roles. If an explicit role label exists, it is used first. Otherwise, rule-based cue matching is applied to the context's `section`, `sentence`, and `paragraph`.

| Role | Typical evidence | Weight |
| --- | --- | ---: |
| `guideline` | guideline, practice recommendation, consensus statement, NCCN/ASCO/ESMO/NICE/WHO/FDA guidance | 2.00 |
| `clinical` | clinical trial, randomized study, patient cohort, treatment, diagnostic, prognostic, survival | 1.80 |
| `method` | protocol, assay, reagent, antibody, plasmid, cell line, mouse model, CRISPR, sequencing, PCR, ELISA | 1.60 |
| `dataset_or_tool` | dataset, database, software, pipeline, code, benchmark, accession, repository | 1.50 |
| `negative_or_contrasting` | contradicted, conflicting, failed to reproduce, not replicated, refuted | 1.10 |
| `supporting` | supports, consistent with, confirmed, demonstrated, reported, found | 1.00 |
| `background` | general literature/background mention | 0.35 |
| `unknown` | internal citation edge exists but no context is supplied | 0.10 |

For life sciences, `biomarker` alone is intentionally not enough to trigger `clinical`. It must appear with stronger patient, trial, diagnostic, treatment, prognostic, cohort, or survival context. This avoids turning broad review sentences into clinical impact evidence.

## Per-Edge Credit

Each internal citation edge receives at most one primary credit. When only citation-context evidence exists, the edge uses the strongest role observed on that edge:

```text
edge_effective_weight = max(role_weight(context_roles_for_edge))
```

If the edge has no citation contexts:

```text
edge_effective_weight = 0.10
edge_role = unknown
```

When constrained local-model key-reference evidence exists for the same edge, the edge uses the larger of the context-role weight and the ImpactRankings-style rank weight:

```text
edge_effective_weight =
  max(context_role_weight, impact_selected_reference_weight)
```

Top-5 model selections use rank weights `1.0`, `0.8`, `0.6`, `0.4`, and `0.2`. Ranks 6-20 receive a smaller log-discounted weight. This per-edge max rule prevents a paper from receiving inflated credit just because the same citing paper has many citation-context sentences or both context and model-selection evidence.

## Per-Paper Raw Score

For each cited paper:

```text
effective_citation_raw =
  sum(max(context_role_weight, impact_selected_reference_weight)
      for all incoming internal citation edges)
```

Example:

```text
Incoming edge 1: guideline -> 2.00
Incoming edge 2: method    -> 1.60
Incoming edge 3: background -> 0.35

effective_citation_raw = 3.95
```

## Per-Paper Effective Citation Count

`effective_citation_count` counts incoming edges with at least one strong context role or a constrained top-5 local-model key-reference selection:

```text
strong roles =
  guideline
  clinical
  method
  dataset_or_tool
  negative_or_contrasting
  supporting
  top_5_impact_selected_reference
```

Background-only and unknown edge-only citations are recorded but do not increase `effective_citation_count`.

Example:

```text
guideline edge  -> counted
method edge     -> counted
background edge -> not counted

effective_citation_count = 2
```

## Normalized Score

The normalized 0-100 score is:

```text
effective_citation_score =
  tied_rank_percentile(log1p(effective_citation_raw))
```

This means:

- Scores are dataset-local.
- Extreme raw values are compressed by `log1p`.
- Ties receive the same percentile.
- Missing or zero raw evidence receives 0.

`effective_citation_score` is one of the two inputs to the final paper impact score:

```text
final_impact_score =
  0.95 * true_citation_score
+ 0.05 * effective_citation_score
```

The output also records sensitivity fields with `1.00/0.00`, `0.95/0.05`, `0.90/0.10`, `0.80/0.20`, `0.70/0.30`, `0.50/0.50`, and `0.30/0.70` true/effective weights. The same raw signal is also retained in legacy diagnostic fields such as `citation_context_quality_score`, but those diagnostic fields do not directly change `final_impact_score`.

## Output Fields

`model-score-dataset` writes these fields to `impact_scores.csv` and `impact_scores.jsonl`:

| Field | Meaning |
| --- | --- |
| `effective_citation_score` | 0-100 dataset-local normalized effective citation score |
| `effective_citation_raw` | raw weighted incoming effective citation score |
| `effective_citation_context_role_raw` | raw context-role-only incoming effective citation score |
| `effective_citation_impact_selected_raw` | raw constrained local-model selected-reference score before per-edge max fusion |
| `effective_citation_edge_count` | number of incoming internal citation edges |
| `effective_citation_context_count` | number of citation contexts attached to incoming edges |
| `effective_citation_count` | count of incoming edges with strong effective roles |
| `effective_citation_impact_selected_count` | incoming internal edges selected by constrained local-model key-reference evidence |
| `effective_citation_impact_selected_top5_count` | selected incoming edges whose model rank is in the strict top 5 |
| `effective_citation_guideline_count` | context observations classified as guideline |
| `effective_citation_clinical_count` | context observations classified as clinical |
| `effective_citation_method_count` | context observations classified as method |
| `effective_citation_dataset_or_tool_count` | context observations classified as dataset/tool |
| `effective_citation_supporting_count` | context observations classified as supporting |
| `effective_citation_negative_count` | context observations classified as negative/contrasting |
| `effective_citation_background_count` | context observations classified as background |
| `effective_citation_unknown_count` | edge-only citations without context |
| `effective_citation_coverage` | `missing`, `edge_only`, `partial_context`, or `contextualized` |
| `effective_citation_evidence_level` | `missing`, `weak_background_or_unknown`, `low`, `moderate`, or `high` |

The same information appears in `impact_score_cards.json` and `impact_score_cards.md`.

## Coverage Labels

```text
missing:
  no incoming internal citation edge

edge_only:
  incoming edge exists, but no citation context

partial_context:
  incoming edges exist, and some but not all have contexts

contextualized:
  incoming edges exist, and context_count >= edge_count
```

## Evidence-Level Labels

```text
missing:
  no incoming edge

high:
  at least one guideline, clinical, method, or dataset/tool context

moderate:
  at least two strong effective citation edges, without high-role evidence

low:
  one strong effective citation edge, without high-role evidence

weak_background_or_unknown:
  only background or unknown edge-only citations
```

## Interpretation Rules

Use these boundaries in reports:

- `effective_citation_score` is not a global impact score.
- `effective_citation_count` is not a replacement for citation count.
- Background and unknown citations are visible but weak.
- Clinical impact should not be claimed unless clinical, guideline, trial, approval, or comparable external evidence exists.
- If `effective_citation_coverage=missing`, the result means the dataset lacks internal citation-context evidence; it does not prove the paper has no external effective citations.

## Worked Example

Input dataset:

```json
{"paper_id":"assay","title":"Reusable biomarker assay","references":[]}
{"paper_id":"trial","title":"Phase 3 trial using assay","references":[{"reference_id":"assay","title":"Reusable biomarker assay"}],"citation_contexts":[{"reference_id":"assay","sentence":"Patients were stratified using the assay protocol described by the prior study.","section":"Methods"}]}
{"paper_id":"guideline","title":"Clinical practice guideline","references":[{"reference_id":"assay","title":"Reusable biomarker assay"}],"citation_contexts":[{"reference_id":"assay","sentence":"The NCCN guideline recommends this diagnostic biomarker assay for patient selection.","section":"Guidelines"}]}
{"paper_id":"review","title":"Background review","references":[{"reference_id":"assay","title":"Reusable biomarker assay"}],"citation_contexts":[{"reference_id":"assay","sentence":"Several previous studies have examined biomarkers in this disease.","section":"Introduction"}]}
```

For `assay`:

```text
trial -> assay:
  roles: clinical + method
  edge weight: 1.80
  counted as effective: yes

guideline -> assay:
  roles: guideline
  edge weight: 2.00
  counted as effective: yes

review -> assay:
  roles: background
  edge weight: 0.35
  counted as effective: no

effective_citation_edge_count = 3
effective_citation_context_count = 3
effective_citation_count = 2
effective_citation_raw = 4.15
effective_citation_coverage = contextualized
effective_citation_evidence_level = high
```

## Implementation Reference

The implementation lives in:

- `src/local_impact_ranking/impact_fusion.py`
  - `_effective_citation_components`
  - `_life_science_context_roles`
  - `_effective_citation_coverage`
  - `_effective_citation_evidence_level`

Regression tests:

- `tests/test_impact_fusion.py::test_compute_impact_index_records_life_science_effective_citations`
- `tests/test_model_impact_index.py::test_model_score_dataset_writes_fused_impact_index_v2_components`
