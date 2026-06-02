#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

test -f README.md
test -f REPRODUCIBILITY.md
test -f pyproject.toml
test -f .gitignore
test -f .github/workflows/tests.yml
test -f docs/input-schema.md
test -f docs/scoring-method.md
test -f docs/evaluation-method.md
test -f docs/citation-data-sources.md
test -f docs/effective-citation-calculation.md
test -f docs/publish-checklist.md
test -f configs/calibration/sciimpact-qwen32b-heldout-p50.json
test -f configs/model_credibility/local-impact-judge-source-task-dpo.json
test -f scripts/run_quickstart.sh
test -f scripts/run_impact_demo.sh
test -f scripts/reproduce_no_gpu.sh
test -f scripts/run_user_impact.sh
test -f scripts/run_deliverable_check.sh
test -f scripts/train_sciimpact_weak_slice_dpo.sh
test -f examples/impact_demo_papers.jsonl
test -f examples/impact_demo_pairwise_judgments.jsonl
test -f examples/impact_demo_future_labels.jsonl

git check-ignore \
  data/.publish-check \
  models/.publish-check \
  runs/.publish-check \
  local-dataset.zip \
  >/dev/null

if git ls-files data models runs | grep -q .; then
  echo "Refusing to publish: data/, models/, or runs/ are tracked by git." >&2
  git ls-files data models runs >&2
  exit 1
fi

grep -q "report-bundle" README.md
grep -q "dataset-local" README.md
grep -q "run_quickstart.sh" README.md
grep -q "run_impact_demo.sh" README.md
grep -q "reproduce_no_gpu.sh" README.md
grep -q "REPRODUCIBILITY.md" README.md
grep -q "docs/citation-data-sources.md" README.md
grep -q "docs/effective-citation-calculation.md" README.md
grep -q "run_user_impact.sh" README.md
grep -q "user_impact_status.json" README.md
grep -q "user_impact_summary.md" README.md
grep -q "offline replayed pairwise judgments" README.md
grep -q "run_deliverable_check.sh" README.md
grep -q "configs/calibration/sciimpact-qwen32b-heldout-p50.json" README.md
grep -q "configs/model_credibility/local-impact-judge-source-task-dpo.json" README.md
grep -q "model_credibility_status" README.md
grep -q "model_credibility_card.md" README.md
grep -q "model_identity_match" README.md
grep -q "calibration_status" README.md
grep -q "calibration_profile_card.md" README.md
grep -q "calibration_model_identity_match" README.md
grep -q "bundle_decision" README.md
grep -q "model-score-preflight" README.md
grep -q "total_model_request_count" README.md
grep -q "apply-citation-cutoff" README.md
grep -q "audit-citation-cutoff" README.md
grep -q "derive-future-impact-labels" README.md
grep -q "evaluate-impact-validity" README.md
grep -q "backtest-impact-validity" README.md
grep -q "impact_credibility_scorecard" README.md
grep -q "pairwise_credibility_scorecard" README.md
grep -q "sciimpact_model_credibility_scorecard" README.md
grep -q "sciimpact_model_credibility_summary" README.md
grep -q "sciimpact_model_candidate_gate" README.md
grep -q "sciimpact_model_weak_slice_error_pack" README.md
grep -q "train_sciimpact_weak_slice_dpo.sh" README.md
grep -q "relative_citation_ratio_score" README.md
grep -q "enrich-semantic-scholar" README.md
grep -q "import-external-evidence" README.md
grep -q -- "--validity-labels" README.md
grep -q -- "--pairwise-judgments" README.md
grep -q -- "--resume-pairwise-judgments" README.md
grep -q -- "--pairwise-journal" README.md
grep -q "rank_centrality" README.md
grep -q "rank-percentile" README.md
grep -q "impact_score_cards.md" README.md
grep -q "impact_claim_certificate.md" README.md
grep -q "Impact Dimension Coverage" README.md
grep -q "self_citation_risk" docs/scoring-method.md
grep -q "citation_outlier_flag" docs/scoring-method.md
grep -q "influential_citation_score" docs/scoring-method.md
grep -q "bip_impact_score" docs/scoring-method.md
grep -q "impact_index_v3_final_components" docs/scoring-method.md
grep -q "validity_confidence" docs/scoring-method.md
grep -q "validated_impact_claim" docs/scoring-method.md
grep -q "impact_credibility_scorecard" docs/scoring-method.md
grep -q "impact_claim_certificate" docs/scoring-method.md
grep -q "model_credibility_card" docs/scoring-method.md
grep -q "mismatched_profile" docs/scoring-method.md
grep -q "calibration_profile_card" docs/scoring-method.md
grep -q "calibration_model_identity_match" docs/scoring-method.md
grep -q "media_impact_score" docs/scoring-method.md
grep -q "award_impact_score" docs/scoring-method.md
grep -q "Rank Centrality" docs/scoring-method.md
grep -q "rank-percentile" docs/scoring-method.md
grep -q "effective-citation-calculation.md" docs/scoring-method.md
grep -q "citation-data-sources.md" docs/scoring-method.md
grep -q "enrich-semantic-scholar" docs/input-schema.md
grep -q "semantic_scholar_influential_citation_count" docs/input-schema.md
grep -q "OpenAlex" docs/citation-data-sources.md
grep -q "Semantic Scholar" docs/citation-data-sources.md
grep -q "OpenCitations" docs/citation-data-sources.md
grep -q "PubMed/iCite" docs/citation-data-sources.md
grep -q "do not silently average incompatible counts" docs/citation-data-sources.md
grep -q "effective_citation_raw" docs/effective-citation-calculation.md
grep -q "edge_effective_weight" docs/effective-citation-calculation.md
grep -q "guideline" docs/effective-citation-calculation.md
grep -q "clinical" docs/effective-citation-calculation.md
grep -q "effective_citation_score" docs/effective-citation-calculation.md
grep -q "spearman_correlation" docs/evaluation-method.md
grep -q "credibility_scorecard" docs/evaluation-method.md
grep -q "outlier robustness" docs/evaluation-method.md
grep -q "pairwise_credibility_scorecard" docs/evaluation-method.md
grep -q "sciimpact_model_credibility_scorecard" docs/evaluation-method.md
grep -q "sciimpact_model_credibility_summary" docs/evaluation-method.md
grep -q "model_credibility_card" docs/evaluation-method.md
grep -q "model_identity_match" docs/evaluation-method.md
grep -q "calibration_profile_card" docs/evaluation-method.md
grep -q "calibration_model_identity_match" docs/evaluation-method.md
grep -q "sciimpact_model_candidate_gate" docs/evaluation-method.md
grep -q "sciimpact_model_weak_slice_error_pack" docs/evaluation-method.md
grep -q "data/" docs/publish-checklist.md
grep -q "models/" docs/publish-checklist.md
grep -q "runs/" docs/publish-checklist.md
grep -q "configs/calibration/" docs/publish-checklist.md
grep -q "reproduce_no_gpu.sh" REPRODUCIBILITY.md
grep -q "no_gpu_offline_reproduction" scripts/reproduce_no_gpu.sh
grep -q "dataset-local paper influence" REPRODUCIBILITY.md

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m pytest -q

echo "Publish readiness checks passed."
