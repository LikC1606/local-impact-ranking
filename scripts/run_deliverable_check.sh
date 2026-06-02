#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/deliverable-check}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_MODEL="${CHECK_MODEL:-0}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge-source-task-dpo}"
export PYTHON_BIN

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

scripts/run_quickstart.sh "${OUTPUT}/quickstart"
scripts/run_impact_demo.sh "${OUTPUT}/impact-demo"
PAIRWISE_JUDGMENTS=examples/impact_demo_pairwise_judgments.jsonl \
VALIDITY_LABELS=examples/impact_demo_future_labels.jsonl \
MODEL_CREDIBILITY_REPORT=configs/model_credibility/local-impact-judge-source-task-dpo.json \
VALIDITY_BOOTSTRAP_SAMPLES=20 \
VALIDITY_PERMUTATION_SAMPLES=20 \
  scripts/run_user_impact.sh examples/impact_demo_papers.jsonl "${OUTPUT}/user-impact"
scripts/run_sciimpact_smoke.sh "${OUTPUT}/sciimpact-smoke"

mkdir -p "${OUTPUT}/backtest"
cat > "${OUTPUT}/backtest/enriched_papers.jsonl" <<'JSONL'
{"paper_id":"b1","title":"Early Method With Later Uptake","field":"AI","references":[],"metadata":{"citation_counts_by_year":{"2021":2,"2022":3,"2023":12},"citation_cutoff_year":2022}}
{"paper_id":"b2","title":"Early Spike With Low Later Uptake","field":"AI","references":[],"metadata":{"citation_counts_by_year":{"2021":8,"2022":7,"2023":1},"citation_cutoff_year":2022}}
JSONL
cat > "${OUTPUT}/backtest/cutoff_papers.jsonl" <<'JSONL'
{"paper_id":"b1","title":"Early Method With Later Uptake","field":"AI","references":[],"metadata":{"citation_counts_by_year":{"2021":2,"2022":3},"citation_cutoff_year":2022}}
{"paper_id":"b2","title":"Early Spike With Low Later Uptake","field":"AI","references":[],"metadata":{"citation_counts_by_year":{"2021":8,"2022":7},"citation_cutoff_year":2022}}
JSONL
cat > "${OUTPUT}/backtest/impact_scores.csv" <<'CSV'
paper_id,final_impact_score
b1,90
b2,10
CSV

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli backtest-impact-validity \
  --enriched-input "${OUTPUT}/backtest/enriched_papers.jsonl" \
  --cutoff-input "${OUTPUT}/backtest/cutoff_papers.jsonl" \
  --rankings "${OUTPUT}/backtest/impact_scores.csv" \
  --output "${OUTPUT}/backtest" \
  --cutoff-year 2022 \
  --future-start-year 2023 \
  --future-end-year 2023 \
  --label-normalization field-year-percentile \
  --bootstrap-samples 10 \
  --permutation-samples 10

test -s "${OUTPUT}/quickstart/run/report_bundle/bundle_manifest.json"
test -s "${OUTPUT}/quickstart/run/reports/project_readiness.json"
test -s "${OUTPUT}/quickstart/run/reports/external_evidence_paper_coverage.csv"
test -s "${OUTPUT}/impact-demo/demo_status.json"
test -s "${OUTPUT}/impact-demo/demo_report.md"
test -s "${OUTPUT}/impact-demo/model-impact/impact_scores.csv"
test -s "${OUTPUT}/impact-demo/model-impact/impact_claim_certificate.md"
test -s "${OUTPUT}/user-impact/user_impact_status.json"
test -s "${OUTPUT}/user-impact/user_impact_summary.md"
test -s "${OUTPUT}/user-impact/model-impact/impact_scores.csv"
test -s "${OUTPUT}/user-impact/model-impact/impact_claim_certificate.md"
test -s "${OUTPUT}/user-impact/model-impact/model_credibility_card.md"
test -s "${OUTPUT}/user-impact/model-impact/calibration_profile_card.md"
test -s "${OUTPUT}/user-impact/model-impact/report_bundle/bundle_manifest.json"
test -s "${OUTPUT}/user-impact/model-impact/report_bundle/model_credibility_card.md"
test -s "${OUTPUT}/user-impact/model-impact/report_bundle/calibration_profile_card.md"
grep -q "User Impact Run Summary" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "Model Credibility" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "model_credibility_status" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "model_identity_match" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "Calibration Credibility" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "calibration_status" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "calibration_model_identity_match" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "can_use_for_report_with_disclosure" "${OUTPUT}/user-impact/user_impact_summary.md"
grep -q "claim_status" "${OUTPUT}/user-impact/user_impact_summary.md"
test -s "${OUTPUT}/sciimpact-smoke/pairwise_eval/pairwise_evaluation.json"
test -s "${OUTPUT}/sciimpact-smoke/pairwise_eval/pairwise_credibility_scorecard.json"
test -s "${OUTPUT}/backtest/reports/citation_cutoff_audit.json"
test -s "${OUTPUT}/backtest/reports/impact_validity_evaluation.json"
test -s "${OUTPUT}/backtest/reports/impact_credibility_scorecard.json"
test -s "${OUTPUT}/backtest/future-labels/future_impact_labels.jsonl"
test -s "${OUTPUT}/backtest/impact_backtest_manifest.json"

model_status="skipped"
if [[ "${CHECK_MODEL}" == "1" ]]; then
  cat > "${OUTPUT}/model_smoke_papers.jsonl" <<'JSONL'
{"paper_id":"m1","title":"Influential Representation Learning Method","abstract":"Introduces a reusable representation learning method with broad downstream adoption.","year":2021,"field":"AI","citation_count":120,"references":[]}
{"paper_id":"m2","title":"Narrow Ablation Study","abstract":"Reports a small ablation for one local model variant.","year":2021,"field":"AI","citation_count":8,"references":[{"reference_id":"m1","title":"Influential Representation Learning Method"}],"citation_contexts":[{"reference_id":"m1","sentence":"We build directly on the representation method introduced by m1.","section":"Methods"}]}
{"paper_id":"m3","title":"Dataset Benchmark Release","abstract":"Releases a dataset used by multiple follow-up benchmark papers.","year":2022,"field":"AI","citation_count":55,"references":[]}
{"paper_id":"m4","title":"Application Note","abstract":"Applies existing methods to a narrow case study.","year":2023,"field":"AI","citation_count":4,"references":[{"reference_id":"m1","title":"Influential Representation Learning Method"},{"reference_id":"m3","title":"Dataset Benchmark Release"}]}
JSONL

  scripts/check_model_server.sh "${BASE_URL}"
  PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-dataset \
    --input "${OUTPUT}/model_smoke_papers.jsonl" \
    --output "${OUTPUT}/model-impact" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --max-pairs 1 \
    --pair-sampling coverage \
    --pair-budget auto \
    --target-paper-coverage 1.0 \
    --bidirectional-pairwise \
    --bidirectional-confidence-margin 0.05 \
    --calibration-report configs/calibration/sciimpact-qwen32b-heldout-p50.json \
    --report-bundle \
    --report-bundle-output "${OUTPUT}/model-impact/report_bundle"

  test -s "${OUTPUT}/model-impact/impact_scores.csv"
  test -s "${OUTPUT}/model-impact/model_impact_report.md"
  test -s "${OUTPUT}/model-impact/external_evidence_paper_coverage.csv"
  test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"
  test -s "${OUTPUT}/model-impact/report_bundle/bundle_manifest.json"
  test -s "${OUTPUT}/model-impact/report_bundle/impact_scores.csv"
  test -s "${OUTPUT}/model-impact/report_bundle/model_impact_report.md"
  grep -q "validated_impact_claim.status" "${OUTPUT}/model-impact/model_impact_report.md"
  grep -q "validity_confidence" "${OUTPUT}/model-impact/impact_scores.csv"
  grep -q "Impact Claim Certificate" "${OUTPUT}/model-impact/impact_claim_certificate.md"
  grep -q "Model Score Dataset" "${OUTPUT}/model-impact/report_bundle/README.md"
  model_status="passed"
fi

cat > "${OUTPUT}/deliverable_status.json" <<JSON
{
  "status": "passed",
  "quickstart": "passed",
  "impact_demo": "passed",
  "user_impact": "passed",
  "sciimpact_fixture": "passed",
  "backtest_evidence": "passed",
  "model_smoke": "${model_status}",
  "claim_boundary": "dataset-local paper influence; not global scientific impact without external calibrated evidence"
}
JSON

echo "Deliverable check complete: ${OUTPUT} (model_smoke=${model_status})"
