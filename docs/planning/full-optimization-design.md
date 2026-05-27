# Full Optimization Design

## Goal

Make the project a complete local paper-impact evaluation product: a user can provide a paper dataset, run a local judge or heuristic smoke judge, receive paper/author/institution/field/year influence scores, and inspect whether the run is credible enough to report.

The valid claim is:

```text
dataset-local, content-aware influence inside the supplied corpus
```

The system must not claim global scientific impact unless external citation-graph calibration and entity resolution are added.

## Evidence Behind The Design

- ImpactRankings supplies the high-level workflow: select important references per target paper, then aggregate credit to papers, authors, and institutions. Its public repo does not expose enough prompt, candidate, matching, scoring, or evaluation detail for exact reproduction, so this project implements the idea with stronger audit and evaluation.
- `Not all citations are equal` and Semantic Scholar highly influential citations justify selecting influential references instead of treating every bibliography item equally.
- `Identifying Meaningful Citations`, SciCite, ACL-ARC, and citation-context work justify method/data/baseline context cues and citation-intent-style proxy checks.
- PreScience is the closest public benchmark for prior-work and impact-related reference selection; SciImpact is useful for broader pairwise impact evaluation; SCIDOCA is useful for citation discovery and citation prediction.
- Citation count, PageRank/CiteRank, field/year-normalized citation, BIP! DB, and OpenCitations provide conventional baselines and calibration anchors.
- DORA and the Leiden Manifesto justify explicit claim boundaries, multiple metrics, field context, uncertainty, and avoiding a single overclaimed score.

## Optimized System Shape

1. Data validation
   - Validate IDs, titles, references, duplicate records, and year ranges.
   - Label ranking scope as formal or exploratory based on metadata coverage.

2. Constrained judge
   - For each target paper, the judge can only select from candidate references.
   - Store prompts, candidate IDs, raw outputs, selected IDs, parse status, and errors.
   - Support heuristic smoke, local OpenAI-compatible model, and audit replay.

3. Scoring
   - Keep `score` as raw accumulated local edge credit.
   - Keep `core_score` for strict top-5 ImpactRankings-compatible credit.
   - Keep `expanded_score` for log-discounted top-20 credit so the system is not overly brittle.
   - Keep `context_score` for citation-context support.
   - Keep `graph_score` and `field_year_percentile` as baseline/context components.
   - Use `final_score` as the 0-100 fused reporting score while retaining all components for audit.

4. Evaluation
   - Judge reliability: parse success, invalid selections, empty selections, duplicate selections, top-k compliance.
   - Gold/proxy quality: PreScience or gold-reference hit rate, precision, recall, MRR, NDCG when labels exist.
   - Baseline alignment: compare final ranking with citation count, PageRank, and field/year-normalized citation.
   - Robustness: leave-one-target-out stability, paper-level stability, bootstrap uncertainty.
   - Anomalies: missing metadata, single-edge dominance, concentrated evidence.

5. Model improvement
   - Generate SFT and DPO data from constrained outputs, context hard negatives, method-context contrasts, and benchmark gold positives.
   - Use no-GPU preflight before training.
   - Promote a tuned model only through held-out dev comparison, then one final test comparison.

6. Final quality gate
   - Add a single `project-readiness` report that aggregates existing reports.
   - Output a top-level decision: `reportable`, `needs_improvement`, or `blocked`.
   - Include passed checks, failed checks, blockers, warnings, claim scope, required evidence files, and recommended next actions.
   - This becomes the user's one-command answer to "can I report this run?"

## Readiness Decision Rules

`reportable` requires:

- all required run artifacts exist;
- validation says the pipeline can run;
- parse success >= 0.98;
- invalid selection rate == 0;
- top-k compliance == 1;
- a scoring config and ranking CSV exist;
- no critical anomaly makes the ranking unsupported;
- claim scope remains dataset-local.

`needs_improvement` means the run exists and can be inspected, but has weaknesses such as low context coverage, weak stability, high unsupported-selection rate, no gold labels, or limited fine-tuning data.

`blocked` means a report would be misleading because required artifacts are missing, validation failed, judge outputs are invalid, or no ranking score is available.

## Implementation Increment

Implement the quality gate as a focused addition:

- New module: `project_readiness.py`
- New CLI command: `project-readiness RUN_DIR`
- New outputs: `reports/project_readiness.json` and `reports/project_readiness.md`
- Include these artifacts in `artifact_index.json`
- Call the readiness writer automatically at the end of `run`
- Add unit and CLI tests
- Update README and summary docs

This completes the product loop without starting GPU jobs or inventing an unsupported new metric.
