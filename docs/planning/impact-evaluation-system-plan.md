# Local Paper Impact Evaluation System Plan

## Goal

Build a local, auditable system that accepts a user-provided paper dataset and produces dataset-local influence rankings for papers, authors, and institutions.

The system should preserve the core ImpactRankings idea: for each target paper, identify the most influential references, then propagate credit to referenced papers, authors, and institutions. It should also improve on the public ImpactRankings release by adding constrained candidate selection, citation-context signals, graph baselines, explicit evaluation, and fine-tuning data export.

## Design Position

This project should not attempt an exact reproduction of ImpactRankings because the public repository does not contain the full pipeline. The repo exposes static output artifacts and a high-level method: a DeepSeek-R1-Distill-Llama-8B model selects five important references per AI paper, selected papers are mapped to DBLP, affiliations are mapped through CSRankings, and credit is split across coauthors.

That method is useful as a baseline, but it is underspecified. Missing details include the prompt, sampling parameters, candidate construction, fuzzy matching code, failed-match handling, scoring transform, validation protocol, and any fine-tuning or evaluation data.

The local system should therefore implement two modes:

- `impactrank_compatible`: constrained top-5 reference selection plus rank/flat credit, matching the public method as closely as possible.
- `impact_plus`: top-k reference selection enriched with citation context, citation intent, graph priors, field/year normalization, stability checks, and human-audit evaluation.

## Capability Boundary

The short-term objective is a dataset-local, content-aware impact ranking system for the user's own corpus. It can support claims such as:

- within a defined paper corpus, identify references that appear to be important to citing papers;
- aggregate selected-reference credit into paper, author, and institution rankings;
- compare the resulting rankings against citation count, PageRank, and field/year-normalized baselines;
- audit model outputs, parsing reliability, and proxy evidence for each selected reference.

Global scientific impact is explicitly out of scope for the MVP. The system should not claim to recover true global impact across all fields, years, venues, institutions, or hidden informal influence channels.

Global graph calibration is a future optional extension. It would require:

- external citation graph coverage from OpenAlex, Semantic Scholar, OpenCitations, BIP! DB, or equivalent sources;
- reliable paper, author, and institution entity resolution;
- field/year normalization and uncertainty reporting as first-class outputs;
- stability tests across prompts, models, and bootstrap samples;
- anomaly reports for extreme papers, authors, institutions, missing metadata, and single-edge dominance;
- optionally, human labels or high-quality external labels for key-reference accuracy.

Short-term target:

```text
dataset-local content-aware impact ranking on the user-provided corpus
```

Optional future target:

```text
global graph-calibrated impact ranking
```

## Evaluation Target Levels

The project should separate three evaluation targets instead of treating impact as one undifferentiated objective.

### Level 1: Local Key-Reference Identification

Question:

- Given one target paper and its candidate references, can the system identify which references appear most important to the target paper?

Primary evidence:

- valid constrained selections;
- citation-context support;
- citation intent or method/data/baseline cues;
- optional human or external influential-citation labels when available.

This level evaluates the judge.

### Level 2: Dataset-Local Impact Ranking

Question:

- Within a fixed user-provided corpus, can selected-reference credit be aggregated into plausible paper, author, and institution rankings?

Primary evidence:

- paper, author, and institution ranking outputs;
- baseline comparison against citation count, PageRank, and field/year-normalized citation;
- anomaly and robustness reports.

This level is the short-term MVP target.

### Level 3: Global Graph-Calibrated Impact Ranking

Question:

- After mapping local records to external citation graphs, can the system estimate broader scientific impact beyond the local corpus?

Primary evidence:

- external graph coverage;
- entity-resolution confidence;
- global citation graph baselines;
- local-vs-global ranking comparison;
- coverage and uncertainty reports.

This level is not part of the MVP and should not block implementation for the user's current requirement.

## MVP Success Criteria

The short-term MVP is successful only if it satisfies all of the following conditions on a bounded pilot corpus.

Engineering reliability:

- `parse_success_rate >= 0.98`
- `invalid_selection_rate = 0.0`
- `top_k_compliance_rate = 1.0`
- `empty_selection_rate <= 0.02` for papers with at least one candidate reference
- every run writes audit, rankings, reports, and manifest artifacts

Proxy quality:

- `unsupported_selection_rate` is reported and inspected, not hidden;
- context-aware runs report `context_supported_selection_rate`;
- baseline comparison is generated for citation count and PageRank where data allows.

Ranking safety:

- paper, author, and institution rankings are not reported without corresponding data-quality coverage;
- anomaly report identifies missing metadata and single-edge dominance;
- all MVP outputs are labeled as dataset-local.

Scale checkpoint:

- 100-paper smoke run completes cleanly;
- 1k-paper pilot completes with reports;
- 5k-paper pilot is required before treating the workflow as ready for a larger 50k run.

## Data Readiness Gates

The system should not produce all ranking types for every dataset by default. Ranking scope depends on metadata coverage.

Minimum gate for paper ranking:

- `paper_id` coverage: 100%
- target paper `title` coverage: >= 95%
- candidate reference list coverage: >= 95%
- reference `reference_id` or stable generated ID coverage: 100%
- reference `title` coverage: >= 90% for formal runs

Minimum gate for author ranking:

- all paper-ranking gates pass;
- referenced-paper author coverage: >= 70%;
- author name normalization is enabled;
- missing-author report is generated.

Minimum gate for institution ranking:

- all author-ranking gates pass;
- affiliation coverage: >= 70%;
- institution normalization is enabled;
- missing-affiliation report is generated.

Minimum gate for context-aware ranking:

- citation-context coverage is reported;
- if citation-context coverage is low, the run must be labeled `metadata-only` or `low-context-coverage`;
- context-supported metrics must not be interpreted as correctness when context extraction coverage is incomplete.

If a gate fails, the pipeline may still run, but the corresponding ranking must be labeled exploratory and the missing-data report must be included.

## Scoring Versions

Scoring must be versioned so experiments remain comparable.

`score_v1_rank_weighted`:

```text
edge_score = rank_weight
```

Use for the ImpactRankings-compatible MVP.

`score_v2_context_weighted`:

```text
edge_score = rank_weight * citation_context_weight
```

Use when citation contexts are available and coverage is reported.

`score_v3_confidence_context_weighted`:

```text
edge_score = rank_weight * citation_context_weight * confidence_weight
```

Use only after confidence calibration checks, because raw LLM confidence can be poorly calibrated.

`score_v4_normalized_capped`:

```text
edge_score =
  rank_weight
  * citation_context_weight
  * confidence_weight
  * field_year_normalization
  * contribution_cap
```

Use for robustness and outlier-resistant reporting.

Every run must record the scoring version, weights, and normalization settings in `run_manifest.json`.

## Claim Boundaries

The short-term system may claim:

- dataset-local content-aware impact ranking;
- proxy-evaluated key-reference selection;
- auditable paper, author, and institution ranking within a defined corpus;
- comparison against bibliometric baselines;
- robustness or anomaly findings when corresponding reports are generated.

The short-term system must not claim:

- true global scientific impact;
- authoritative global paper, author, or institution ranking;
- human-validated correctness without human labels;
- complete citation-graph coverage without external graph ingestion;
- fair cross-field comparison without field/year normalization and coverage reporting.

## Evidence Base and Reusable Assets

### Not All Citations Are Equal

Use for:

- Core justification that citation count alone is insufficient.
- Key-reference style supervision when available.
- Features based on how often and where a reference is cited in the full text.

Reason: this line of work argues that only a subset of references carry central intellectual influence. It directly supports the project's decision to identify influential references before aggregating impact.

Transfer risk:

- The original setting may use author-identified or manually annotated key references. In the short-term no-human setting, treat this as method justification and use citation contexts as proxy evidence.

### Identifying Meaningful Citations

Use for:

- Distinguishing meaningful citations from incidental citations.
- Citation-context features such as section, mention frequency, and local textual cues.
- External validation against Semantic Scholar-style influential-citation signals when available.

Reason: this is directly aligned with the system's judge task: not every reference in a bibliography should receive equal credit.

Transfer risk:

- Production influential-citation labels are not a perfect public gold standard. Use them as proxy or external checks, not as absolute truth.

### Content-Aware Bibliometrics

Use for:

- Treating citation edges as weighted edges rather than binary links.
- Justifying the use of citation context to weight impact propagation.
- Designing no-human proxy evaluation when human labels are unavailable.

Reason: content-aware bibliometrics bridges local textual evidence and global bibliometric ranking, which matches this project's intended architecture.

Transfer risk:

- Distant supervision and proxy labels can encode bias. Keep proxy metrics separate from final claims about true scientific impact.

### ImpactRankings

Use for:

- Method inspiration.
- Output concept: paper, author, and institution rankings.
- Optional frontend-compatible CSV shape.

Do not depend on it for:

- Prompt reproduction.
- Inference parameters.
- Evaluation protocol.
- Fine-tuning data.

Reason: the public codebase does not expose enough method detail to support exact reproduction.

### SciImpact

Use for:

- Pairwise scientific impact evaluation.
- Fine-tuning or preference-training data format.
- Benchmark-style validation against multiple impact dimensions.

Reason: SciImpact is directly aligned with scientific impact prediction and includes a public repository. Its pairwise comparison setup maps well to training a judge that decides which of two candidate references is more influential.

Transfer risk:

- Its impact dimensions may include signals outside pure paper-to-paper scholarly influence, such as awards, patents, media, code, datasets, and models. These are useful for general impact, but the local system should keep them separate from citation-derived influence unless the user explicitly enables them.

### Semantic Scholar Highly Influential Citations

Use for:

- Weak labels or external checks for whether a citation is influential.
- Design justification for using citation context rather than raw citation count only.

Reason: Semantic Scholar's influential citation concept distinguishes meaningful citations from incidental citations using context and metadata.

Transfer risk:

- The exact production model is not fully reproducible locally. Treat this as an external signal or validation target, not as the core method.

### SciCite and ACL-ARC

Use for:

- Citation intent classification.
- Training or evaluating a local classifier for background / method / result-comparison citation types.

Reason: Citation intent is a strong proxy for whether a reference is foundational, reused, or merely background.

Transfer risk:

- Most citation-intent datasets are domain-biased and label a small number of coarse categories. They should inform an auxiliary signal, not replace the LLM top-k judge.

### S2ORC

Use for:

- Citation context extraction when full text is available.
- Structured references and citation mentions.

Reason: S2ORC provides large-scale scientific full text with citation spans and bibliography structure.

Transfer risk:

- Coverage varies by publisher and license. The system must work without full text by falling back to title/abstract/reference metadata.

### OpenAlex, DBLP, Crossref, Semantic Scholar APIs

Use for:

- Entity resolution.
- Missing metadata enrichment.
- Author and institution normalization.
- External IDs.
- Citation counts and citation graph edges when available.

Reason:

- OpenAlex provides broad open metadata for works, authors, institutions, venues, and concepts.
- DBLP is strong for computer science publication identity.
- Crossref is useful for DOI lookup.
- Semantic Scholar is useful for citation metadata and influential-citation checks.

Transfer risk:

- API coverage and rate limits can change. All enrichment must be cached locally with request/response audit logs.

### BIP! DB and Graph-Based Bibliometrics

Use for:

- Baselines such as citation count, PageRank, RAM, AttRank, and short-window citation count.

Reason: Graph baselines provide necessary sanity checks. If the LLM ranking is uncorrelated with every graph signal, that needs to be explained by evidence rather than ignored.

Transfer risk:

- Bibliometric indicators often favor older papers and larger fields. Field/year normalization is mandatory for fair comparison.

### H2CGL and Citation Impact Prediction Models

Use for:

- Later-stage graph-learning baseline.

Reason: dynamic citation graph models can predict future impact and capture network structure beyond local reference lists.

Transfer risk:

- Implementation and data requirements are higher than the first production system. Do not put this on the critical path for the MVP.

## Data Model

The canonical input should be JSONL with one target paper per line.

Required fields:

- `paper_id`
- `title`
- `references`

Recommended fields:

- `abstract`
- `year`
- `venue`
- `field`
- `authors`
- `affiliations`
- `doi`
- `external_ids`
- `citation_count`
- `references[].title`
- `references[].authors`
- `references[].year`
- `references[].doi`
- `references[].external_ids`

Optional full-text fields:

- `citation_contexts`
- `section`
- `paragraph`
- `mention_count`

Design reason:

The system must work on the dataset the user controls, but each additional metadata field improves ranking quality, auditability, and downstream aggregation. A minimal schema keeps the pipeline usable; optional fields allow stronger methods.

## Pipeline

### Stage 1: Validate

Inputs:

- Raw JSONL or CSV paper dataset.

Outputs:

- `reports/validation.json`
- `reports/validation.md`

Checks:

- Missing required fields.
- Duplicate `paper_id`.
- Duplicate or missing `reference_id`.
- Empty reference lists.
- Invalid year.
- Missing author or institution coverage.
- External ID coverage.

Reason:

Ranking failures often come from bad metadata rather than model behavior. Validation makes missing coverage explicit before expensive judging.

### Stage 2: Normalize

Inputs:

- Validated paper records.

Outputs:

- `normalized/papers.jsonl`
- `audit/normalization.jsonl`

Operations:

- Normalize whitespace and Unicode variants.
- Canonicalize titles.
- Normalize author names.
- Normalize institution names.
- Generate stable IDs for references without IDs.
- Preserve raw fields.

Reason:

Entity identity drives ranking correctness. Raw values must be retained so every normalization decision can be audited or revised.

### Stage 3: Enrich

Inputs:

- Normalized records.

Outputs:

- `enriched/papers.jsonl`
- `cache/openalex/`
- `cache/dblp/`
- `cache/crossref/`
- `cache/semanticscholar/`
- `audit/enrichment.jsonl`

Provider order:

1. DOI or external ID exact lookup.
2. Title + year + venue lookup.
3. Fuzzy title lookup with conservative threshold.
4. Manual review queue for ambiguous matches.

Reason:

Exact IDs are more reliable than fuzzy title matching. Ambiguous matches should not silently affect rankings.

### Stage 4: Citation Context Extraction

Inputs:

- Full text or structured citation-context fields, when available.

Outputs:

- `contexts/citation_contexts.jsonl`
- `reports/context_coverage.json`

Fields:

- target paper ID.
- reference ID.
- citation sentence.
- surrounding paragraph.
- section.
- mention count.
- local cue words.

Reason:

Citation context separates true influence from background citation. This stage should be optional because many datasets do not include full text.

### Stage 5: Candidate Construction

Inputs:

- Enriched or normalized paper records.
- Optional citation contexts.

Outputs:

- `candidates/judge_inputs.jsonl`

Rules:

- The judge may only select from the target paper's provided references.
- If a paper has too many references, use chunked judging followed by merge reranking.
- References with missing titles are allowed but marked low-information.
- Candidate IDs must be stable and visible in the prompt.

Reason:

Constrained candidate selection prevents hallucinated papers and makes every selection directly traceable.

### Stage 6: LLM Judge

Inputs:

- Candidate records.
- Local OpenAI-compatible model server.

Outputs:

- `audit/judge_outputs.jsonl`
- `reports/judge_health.json`

Prompt requirements:

- Define influence explicitly.
- Rank only candidate reference IDs.
- Return strict JSON.
- Provide short evidence-grounded rationale.
- Assign influence type.
- Assign confidence.

Influence types:

- `method_or_data_reuse`
- `conceptual_foundation`
- `benchmark_or_comparison`
- `background`
- `uncertain`

Default output schema:

```json
{
  "target_paper_id": "paper-id",
  "selected_references": [
    {
      "reference_id": "ref-id",
      "rank": 1,
      "influence_type": "method_or_data_reuse",
      "confidence": 0.86,
      "rationale": "The target paper directly extends this method."
    }
  ]
}
```

Initial model strategy:

- Use the already downloaded `DeepSeek-R1-Distill-Qwen-32B` as the first local baseline.
- Use a stronger 70B+ or MoE local judge on 8x H100 after the pipeline is stable.
- Keep temperature at 0 for production ranking unless running stability experiments.

Reason:

The first version needs reliable, auditable outputs more than clever fine-tuning. Strong zero-shot or few-shot prompting establishes a baseline before spending labels on training.

### Stage 7: Score

Inputs:

- Parsed judge audit records.
- Paper, author, and institution metadata.

Outputs:

- `rankings/papers.csv`
- `rankings/authors.csv`
- `rankings/institutions.csv`
- `rankings/fields.csv`
- `rankings/years.csv`
- `audit/scoring.jsonl`

Compatible scoring:

- rank 1: 1.00
- rank 2: 0.80
- rank 3: 0.60
- rank 4: 0.40
- rank 5: 0.20

Enhanced scoring:

```text
edge_score =
  rank_weight
  * influence_type_weight
  * citation_context_weight
  * confidence_weight
  * field_year_normalization
```

Default influence type weights:

- `method_or_data_reuse`: 1.20
- `conceptual_foundation`: 1.10
- `benchmark_or_comparison`: 0.80
- `background`: 0.20
- `uncertain`: 0.50

Aggregation:

- Add edge score to the referenced paper.
- Split referenced paper credit across authors.
- Split author credit across current or declared affiliations.
- Export both total and normalized scores.

Reason:

The compatible score allows comparison with ImpactRankings-style results. The enhanced score reflects evidence that not all citations are equal and that field/year effects distort raw impact.

### Stage 8: Evaluate

Inputs:

- Judge outputs.
- Ranking outputs.
- Optional human labels.
- Optional benchmark labels.
- Optional graph baselines.

Outputs:

- `reports/evaluation.json`
- `reports/evaluation.md`
- `review/human_audit_sample.csv`

Engineering metrics:

- parse success rate.
- invalid reference rate.
- empty output rate.
- duplicate selected reference rate.
- top-k compliance.
- average selected references.

Human-label metrics:

- Precision@5.
- NDCG@5.
- Recall@5.
- pairwise accuracy.
- Kendall correlation.
- Spearman correlation.

Baseline comparisons:

- citation count.
- PageRank.
- field/year-normalized citation count.
- Semantic Scholar influential-citation signal when available.
- citation-intent-only score.

Stability metrics:

- MVP: top-N overlap and rank agreement across scoring modes.
- MVP: leave-one-target-out top-N retention without rerunning the model.
- Later: top-N overlap across prompts and models.
- Later: bootstrap confidence intervals for paper, author, and institution rankings.

Reason:

Influence has no single perfect label. A credible system needs format reliability, human relevance checks, baseline comparison, and stability analysis.

## Fine-Tuning Plan

Fine-tuning should begin only after a zero-shot or few-shot pipeline has produced audit logs and an initial human-reviewed sample.

### Phase 1: Format and Constraint SFT

Goal:

- Teach the local model to emit strict JSON.
- Reduce invalid reference IDs.
- Remove unnecessary reasoning text.

Training examples:

- Candidate prompt.
- Correct selected reference JSON.
- Repaired outputs from audit logs.

Reason:

Before optimizing judgment quality, the model must reliably follow the interface.

### Phase 2: Pairwise Ranking SFT

Goal:

- Train the model to compare two references for the same target paper.

Input:

```text
target paper + reference A + reference B + optional contexts
```

Output:

```json
{"winner": "A", "reason": "...", "confidence": 0.82}
```

Data sources:

- Human labels from local audit.
- SciImpact-style pairwise examples.
- Strong-model pseudo-labels with human sampling.

Reason:

Pairwise ranking is easier to label and evaluate than direct top-5 generation.

### Phase 3: Preference Optimization

Goal:

- Prefer answers that select valid, important, evidence-grounded references.

Positive examples:

- Valid IDs.
- Correct influence types.
- Specific rationale.
- High agreement with human labels or strong baseline.

Negative examples:

- Hallucinated IDs.
- Background-only references selected as foundational.
- Generic rationales.
- Wrong ordering.

Reason:

Preference training aligns the judge with the actual ranking quality objective better than format-only SFT.

## Implementation Roadmap

### Milestone 1: Current MVP Hardening

Status:

- Basic JSONL loading, normalization, local judge interface, audit log, scoring, export, and tests already exist.

Next tasks:

- Add run manifest with input checksum, model name, prompt version, scoring config, code revision, and inference parameters.
- Add duplicate-selection and top-k compliance metrics.
- Add human audit sample export.
- Add scoring config file.

Exit criteria:

- A `--limit 100` local-model run produces rankings, audit logs, evaluation report, and manifest.

### Milestone 2: Baseline Module

Tasks:

- Implement citation-count baseline.
- Implement PageRank baseline when citation graph edges are available.
- Implement field/year-normalized citation baseline.
- Export baseline rankings beside LLM rankings.
- Add Spearman/Kendall comparison.
- Add log-transformed or percentile citation baselines for outlier-resistant comparison.

Exit criteria:

- Evaluation report compares LLM ranking with at least citation count and PageRank on the same dataset.

### Milestone 3: Citation Context Module

Tasks:

- Define citation context input schema.
- Add context coverage reporting.
- Add citation-intent classifier interface.
- Add simple rule baseline for method/background cue words.
- Add optional LLM prompt fields for citation contexts.

Exit criteria:

- The same dataset can run in metadata-only mode and context-aware mode, with a comparison report.

### Milestone 4: No-Human Proxy Evaluation Workflow

Tasks:

- Compute weak/proxy evidence for each selected reference.
- Use citation context cues to distinguish method/data/baseline citations from background-only citations.
- Compare selected references against citation-count, PageRank, and field/year-normalized citation top-N baselines.
- Report unsupported selections: model-selected references with no citation context and no bibliometric support.
- Report MVP stability across scoring modes and leave-one-target-out perturbations.
- Report stability across prompt variants, model variants, and bootstrap samples when feasible.
- Keep `human_audit_sample.csv` export available for future review, but do not require human labels.

Exit criteria:

- A run can quantify model selections using local proxy signals without human annotators.
- The report clearly separates proxy agreement from true human-validated correctness.

Reason:

The current project has no human annotators. Proxy evaluation does not prove the model is correct, but it gives a scalable local quality signal before expensive fine-tuning or full-dataset runs.

Optional future human review:

- Export stratified audit samples.
- Support human labels in CSV/JSONL.
- Compute P@5, NDCG@5, pairwise accuracy, and annotator agreement where available.

### Milestone 4.5: Anomaly and Robustness Report

Tasks:

- Report extreme paper, author, and institution scores.
- Report missing reference titles, missing authors, missing affiliations, and missing years/fields.
- Report scores dominated by a single selected citation edge.
- Report field/year buckets with unusually high or low normalized scores.
- Add contribution caps for author and institution aggregation experiments.
- Compare raw score, normalized score, and capped score rankings.

Exit criteria:

- A run surfaces the main ways the ranking could be misleading before any result is used for reporting.

Reason:

Impact rankings are especially vulnerable to outliers: old papers, hot fields, large institutions, surveys, datasets, benchmarks, and metadata gaps can dominate scores. The system should expose these risks rather than hide them behind a single ranking table.

### Milestone 5: Fine-Tuning Data Export

Tasks:

- Export SFT JSONL for strict top-k selection.
- Export pairwise comparison JSONL.
- Export DPO/preference JSONL.
- Add LLaMA-Factory-compatible converter.

Exit criteria:

- Human-reviewed or pseudo-labeled examples can be used to train a local judge.

### Milestone 6: Strong Local Model Run

Tasks:

- Serve a stronger local judge on 8x H100.
- Run a 5k paper pilot.
- Run a 50k paper full batch after pilot evaluation.
- Compare 32B baseline vs stronger model.

Exit criteria:

- Reports show quality, stability, runtime, cost, and ranking differences across models.

### Optional Future Milestone: Global Graph Calibration

Tasks:

- Add OpenAlex, Semantic Scholar, OpenCitations, or BIP! DB ingestion for external citation graph coverage.
- Map local paper IDs and references to external work IDs.
- Add author and institution entity-resolution confidence fields.
- Compute global graph baselines on the mapped corpus.
- Fuse local semantic influence with global graph influence.
- Report coverage: mapped papers, mapped references, mapped authors, mapped institutions, and unmapped high-score items.

Exit criteria:

- The system can explicitly distinguish dataset-local impact from graph-calibrated global impact.

Priority:

- Deferred. This milestone is not required for the current user-dataset-local implementation.

## Recommended First Experiments

### Experiment A: 100-Paper Smoke Evaluation

Purpose:

- Verify model output quality and audit structure before expensive runs.

Configuration:

- Model: local 32B baseline.
- top-k: 5.
- temperature: 0.
- repair attempts: 1.
- dataset size: 100 papers.

Metrics:

- parse success rate.
- invalid reference rate.
- empty selection rate.
- average selected references.
- qualitative inspection of 30 examples.

Observed local checkpoint:

- SciCite dev, 100 papers, `DeepSeek-R1-Distill-Qwen-32B`, `temperature=0`, `repair_attempts=1`, `max_tokens=1024`.
- Run directory: `runs/scicite-dev-deepseek32b-100c`.
- `parse_success_rate=1.0`, `invalid_selection_rate=0.0`, `empty_selection_rate=0.0`, `average_selected_references=1.0`, `top_k_compliance_rate=1.0`.
- Proxy evaluation: `context_supported_selection_rate=1.0`, `unsupported_selection_rate=0.0`.
- Compared with the earlier 100-paper run, the single-candidate prompt fix reduced `empty_selection_rate` from `0.03` to `0.0`.

### Experiment B: 1,000-Paper Baseline Comparison

Purpose:

- Check whether LLM influence ranking behaves plausibly against graph baselines.

Configuration:

- Same prompt.
- Add citation count and PageRank if edges are available.

Metrics:

- Spearman correlation.
- top-50 overlap.
- examples where LLM strongly disagrees with citation count.

### Experiment C: No-Human Proxy Review

Purpose:

- Estimate selection plausibility without human annotators.

Configuration:

- Run metadata-only and context-aware prompts.
- Compute proxy support for every selected reference.
- Compare selections with citation count, PageRank, and field/year-normalized top-N baselines.

Metrics:

- context-supported selection rate.
- method/baseline cue selection rate.
- bibliometric top-N agreement.
- unsupported selection rate.
- examples where LLM selects low-citation but context-supported references.

Limitation:

- Proxy review is not a substitute for human labels. It is a triage and debugging tool.

### Experiment D: Context-Aware Ablation

Purpose:

- Test whether citation contexts improve selection quality.

Configuration:

- Run metadata-only prompt.
- Run metadata + citation context prompt.

Metrics:

- NDCG@5 on human labels.
- invalid output rate.
- qualitative improvement on method/data reuse citations.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Entity matching errors dominate ranking | High | Preserve raw IDs, cache enrichment, add ambiguous-match review queue |
| LLM selects plausible but wrong references | High | Constrain candidates, use human audit, compare against citation context |
| Citation count bias favors old papers | Medium | Add field/year normalization and report both raw and normalized scores |
| Institution affiliation is time-varying | Medium | Store affiliation source and year; export uncertainty |
| Full text is unavailable | Medium | Keep metadata-only mode as first-class |
| Strong local model is expensive | Medium | Use 32B for pilot, run stronger model only after quality checks |
| Fine-tuning overfits weak labels | Medium | Separate human labels, weak labels, and pseudo-labels in manifests |
| Dataset-local scores are mistaken for global impact | High | Label local vs global outputs explicitly; require external graph calibration before global claims |
| Extreme papers or institutions dominate rankings | High | Add anomaly reports, normalized scores, and contribution-cap sensitivity runs |

## Reproducibility Requirements

Every run must write:

- input dataset checksum.
- normalized dataset checksum.
- config file copy.
- model path/name/revision.
- serving backend and version.
- inference parameters.
- prompt template version.
- scoring mode and weights.
- code git revision.
- timestamp.
- hardware summary.
- enrichment provider cache versions.

Every model call must log:

- target paper ID.
- candidate reference IDs.
- prompt hash.
- raw model output.
- parsed output.
- repair attempts.
- invalid selections.
- error message if any.

## Immediate Next Implementation Steps

1. Add `run_manifest.json` generation.
2. Expand evaluation metrics for duplicate selection and top-k compliance.
3. Add human audit sample export.
4. Add baseline scoring module for citation count and PageRank.
5. Add field/year normalization utility.
6. Add citation context schema.
7. Add fine-tuning data exporters.
