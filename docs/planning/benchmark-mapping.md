# Benchmark Mapping for Local Impact Ranking

## Goal

Provide public benchmark options when the user has no private dataset, and map each benchmark to the correct part of the local pipeline.

## Recommended Order

### 1. PreScience

Use for:

- Prior work selection.
- Impact prediction / reference importance ranking.
- A large AI-paper citation graph with realistic scale.

Map to pipeline:

- `metadata.prescience_positive_reference_ids` for gold reference-selection evaluation.
- Author `publication_history` as negative/neutral candidate references.
- `citation_contexts` only if full text or citation spans are available.
- `judge` for top-k influential references.
- `baselines` for citation count and graph metrics.

Why first:

- Closest to the current project goal.
- Already aligned with AI papers and citation graph structure.

### 2. SCIDOCA 2025

Use for:

- Citation discovery.
- Citation sentence prediction.
- Masked citation prediction.

Map to pipeline:

- `judge` for candidate reference selection.
- `contexts` for citation-sentence evidence.
- `finetuning` for proxy-positive and pairwise examples.

Why second:

- Good fit for citation-context-aware judging.
- Useful for testing whether the model can identify missing or likely citations.

### 3. SciCite / ACL-ARC / MultiCite

Use for:

- Citation intent classification.
- Background vs method vs result-comparison labeling.

Map to pipeline:

- `proxy_evaluation` for method-context support.
- `finetuning` for auxiliary intent classifier data.
- `judge` prompt conditioning.

Why third:

- Small, clean, and useful for training the model to tell meaningful citations from background citations.

### 4. SciImpact

Use for:

- Broader scientific impact evaluation.
- Pairwise comparison supervision.

Map to pipeline:

- `finetuning` for pairwise preference or ranking data.
- `reports` for broader impact comparison later.

Why fourth:

- Useful, but broader than citation-only influence.

### 5. BIP! DB / OpenCitations

Use for:

- Citation count baseline.
- PageRank baseline.
- Citation graph construction.

Map to pipeline:

- `baselines`.

Why fifth:

- They are not benchmark tasks by themselves, but they are the best comparison anchors.

## Task-to-Pipeline Matrix

| Benchmark | judge | contexts | baselines | proxy_eval | finetuning |
| --- | --- | --- | --- | --- | --- |
| PreScience | yes | optional | yes | yes | yes |
| SCIDOCA 2025 | yes | yes | yes | yes | yes |
| SciCite / ACL-ARC / MultiCite | limited | yes | limited | yes | yes |
| SciImpact | yes | optional | limited | yes | yes |
| BIP! DB / OpenCitations | no | no | yes | yes | yes |

## Practical Dataset Choice

If the user needs one dataset to start:

1. PreScience
2. SCIDOCA 2025
3. SciCite

If the user wants a baseline-only sanity check:

1. BIP! DB
2. OpenCitations
