#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/reproduce-no-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -e ".[dev]"
fi

PYTHON_BIN="${PYTHON_BIN}" scripts/check_publish_ready.sh
PYTHON_BIN="${PYTHON_BIN}" scripts/run_deliverable_check.sh "${OUTPUT}/deliverable-check"

cat > "${OUTPUT}/reproduce_status.json" <<JSON
{
  "status": "passed",
  "mode": "no_gpu_offline_reproduction",
  "checks": {
    "publish_ready": "passed",
    "deliverable_check": "passed"
  },
  "demo_scores": "${OUTPUT}/deliverable-check/impact-demo/model-impact/impact_scores.csv",
  "user_impact_scores": "${OUTPUT}/deliverable-check/user-impact/model-impact/impact_scores.csv",
  "report_bundle": "${OUTPUT}/deliverable-check/user-impact/model-impact/report_bundle",
  "claim_boundary": "dataset-local paper influence; not global scientific impact without external calibrated evidence"
}
JSON

echo "No-GPU reproduction complete: ${OUTPUT}"
