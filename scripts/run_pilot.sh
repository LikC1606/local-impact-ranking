#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: scripts/run_pilot.sh INPUT_JSONL [OUTPUT_DIR]}"
OUTPUT="${2:-runs/pilot-100}"

LIMIT="${LIMIT:-100}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge}"
TEMPERATURE="${TEMPERATURE:-0}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

AUDIT_PATH="${OUTPUT}/audit/judge_outputs.jsonl"

scripts/check_model_server.sh "${BASE_URL}" >/dev/null

mkdir -p "${OUTPUT}/audit"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli judge \
  --input "${INPUT}" \
  --audit-output "${AUDIT_PATH}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --judge openai-compatible \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --max-retries "${MAX_RETRIES}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --top-k "${TOP_K}" \
  --judge openai-compatible \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --judge-audit "${AUDIT_PATH}" \
  --limit "${LIMIT}" \
  --strict-validation

test -s "${OUTPUT}/audit/scoring_edges.csv"
test -s "${OUTPUT}/audit/prompts.jsonl"
test -s "${OUTPUT}/rankings/fields.csv"
test -s "${OUTPUT}/rankings/years.csv"
test -s "${OUTPUT}/reports/validation.json"
test -s "${OUTPUT}/reports/validation.md"
test -s "${OUTPUT}/reports/artifact_index.json"
test -s "${OUTPUT}/reports/evaluation.json"
test -s "${OUTPUT}/reports/evaluation.md"
test -s "${OUTPUT}/reports/gold_evaluation.json"
test -s "${OUTPUT}/reports/gold_evaluation.md"
test -s "${OUTPUT}/reports/proxy_evaluation.json"
test -s "${OUTPUT}/reports/baseline_comparison.json"
test -s "${OUTPUT}/reports/context_coverage.json"
test -s "${OUTPUT}/reports/data_readiness.json"
test -s "${OUTPUT}/reports/anomaly_report.json"
test -s "${OUTPUT}/reports/scoring_config.json"
test -s "${OUTPUT}/reports/stability_report.json"
test -s "${OUTPUT}/reports/paper_explanations.json"
test -s "${OUTPUT}/reports/entity_explanations.json"
test -s "${OUTPUT}/reports/mvp_summary.json"
test -s "${OUTPUT}/reports/mvp_summary.md"
test -f "${OUTPUT}/run_manifest.json"

echo "Pilot run complete: ${OUTPUT}"
