# Model Citation Role Classification Design

## Goal

Use a local OpenAI-compatible model to classify every existing citation context into one canonical effective-citation role before scoring.

The model does not find citation positions. It only classifies already extracted `citation_contexts`.

## Non-goals

- Do not let the model invent missing citation contexts.
- Do not let the model select references for this feature.
- Do not call the model during final score fusion.
- Do not use keyword rules to decide the final role in the recommended workflow.

## Roles

The model must choose exactly one role:

- `guideline`: guideline, consensus, policy, or clinical recommendation.
- `clinical`: clinical trial, patient cohort, real-world clinical evidence, diagnosis, treatment, or translational clinical use.
- `method`: protocol, assay, reagent, antibody, plasmid, cell line, animal model, sequencing, PCR, ELISA, CRISPR, or other method reuse.
- `dataset_or_tool`: dataset, database, software, tool, platform, benchmark, repository, or reusable resource.
- `negative_or_contrasting`: contradiction, failed replication, limitation, refutation, or contrasting evidence.
- `supporting`: evidence that supports or confirms a claim.
- `background`: related work, literature framing, general introduction, or ordinary background mention.
- `unknown`: not enough context to classify reliably.

## Data Flow

1. Read papers from JSONL.
2. For every `citation_contexts[]` item, build one model-classification request.
3. Include the citing paper title, cited reference title, section, sentence, paragraph, and any source-provided role or label as non-authoritative context.
4. Ask the model to return strict JSON with `role`, `confidence`, and `rationale`.
5. Validate the output.
6. Write the model role back to the context.
7. Run scoring from the enriched JSONL only.

## Output Fields

Each classified context should keep the original context fields and add:

```json
{
  "role": "method",
  "role_confidence": 0.86,
  "role_rationale": "The sentence says the cited work's protocol was used in the Methods section.",
  "role_source": "model",
  "role_model": "local-impact-citation-role-classifier",
  "role_prompt_sha256": "...",
  "role_raw_output": "{...}",
  "role_validation_status": "valid"
}
```

If a context already has `role`, `label`, `intent`, `citation_intent`, or `category`, preserve the source value under `source_role_hint` before writing the model role.

## Validation

- `role` must be in the allowed role list.
- `confidence` must be numeric and clamped to `0..1`.
- Invalid JSON becomes `role=unknown` with `role_validation_status=invalid_json`.
- Illegal role becomes `role=unknown` with `role_validation_status=invalid_role`.
- Empty or missing sentence becomes `role=unknown` without calling the model.

## Determinism

The recommended run uses:

- `temperature=0`.
- fixed model name.
- fixed prompt version.
- fixed role schema.
- per-context `prompt_sha256`.
- JSONL journal/checkpoint file for resume and replay.

Repeated scoring on the same enriched JSONL is deterministic because scoring reads saved roles and does not call the model.

## CLI Shape

Add a command:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli classify-citation-contexts \
  --input papers.jsonl \
  --output papers.model-roles.jsonl \
  --base-url http://localhost:8000/v1 \
  --model local-impact-citation-role-classifier \
  --journal runs/my-impact/citation_role_journal.jsonl \
  --temperature 0 \
  --concurrency 1
```

The command should support resume from the journal so interrupted runs do not reclassify completed contexts.

## Scoring Integration

`impact_fusion` should treat model-written `role` values as the canonical role for effective citation weighting.

For the recommended model-classified workflow, keyword-based role inference is not used to decide the final role. Existing keyword inference may remain for legacy inputs and tests, but model-classified contexts should be scored from the stored model role.

## Audit Requirements

Every model classification must be traceable through:

- paper ID.
- citation context index.
- reference ID.
- prompt hash.
- model name.
- raw model output.
- parsed role.
- validation status.

## Test Plan

- Unit test strict JSON parsing.
- Unit test invalid JSON and illegal roles falling back to `unknown`.
- Unit test original source labels are preserved as hints.
- Unit test scoring uses stored model role rather than keyword inference when `role_source=model`.
- Unit test journal replay avoids duplicate model calls.
- CLI test for writing enriched JSONL with model-classified contexts.
