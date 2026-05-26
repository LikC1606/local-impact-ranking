#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: scripts/compare_model_judges.sh INPUT_JSONL [OUTPUT_DIR]}"
OUTPUT="${2:-runs/model-comparison}"

LIMIT="${LIMIT:-100}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
CANDIDATE_BASE_URL="${CANDIDATE_BASE_URL:-${BASE_URL}}"
BASE_MODEL="${BASE_MODEL:-local-impact-judge}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-local-impact-judge-candidate}"
TEMPERATURE="${TEMPERATURE:-0}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
CREDIT_MODE="${CREDIT_MODE:-rank_weighted}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

BASE_OUTPUT="${OUTPUT}/base"
CANDIDATE_OUTPUT="${OUTPUT}/candidate"
COMPARISON_OUTPUT="${OUTPUT}/comparison"
BASE_AUDIT_PATH="${OUTPUT}/base/audit/judge_outputs.jsonl"
CANDIDATE_AUDIT_PATH="${OUTPUT}/candidate/audit/judge_outputs.jsonl"

scripts/check_model_server.sh "${BASE_URL}" >/dev/null
scripts/check_model_server.sh "${CANDIDATE_BASE_URL}" >/dev/null

mkdir -p "${BASE_OUTPUT}/audit" "${CANDIDATE_OUTPUT}/audit" "${COMPARISON_OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli judge \
  --input "${INPUT}" \
  --audit-output "${BASE_AUDIT_PATH}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --judge openai-compatible \
  --base-url "${BASE_URL}" \
  --model "${BASE_MODEL}" \
  --temperature "${TEMPERATURE}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --max-retries "${MAX_RETRIES}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli judge \
  --input "${INPUT}" \
  --audit-output "${CANDIDATE_AUDIT_PATH}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --judge openai-compatible \
  --base-url "${CANDIDATE_BASE_URL}" \
  --model "${CANDIDATE_MODEL}" \
  --temperature "${TEMPERATURE}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --max-retries "${MAX_RETRIES}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli score \
  --input "${INPUT}" \
  --output "${BASE_OUTPUT}" \
  --judge-audit "${BASE_AUDIT_PATH}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --credit-mode "${CREDIT_MODE}" \
  --strict-validation

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli score \
  --input "${INPUT}" \
  --output "${CANDIDATE_OUTPUT}" \
  --judge-audit "${CANDIDATE_AUDIT_PATH}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --credit-mode "${CREDIT_MODE}" \
  --strict-validation

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli compare-benchmarks \
  "${BASE_OUTPUT}" \
  "${CANDIDATE_OUTPUT}" \
  --output "${COMPARISON_OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli compare-gold-targets \
  "${BASE_OUTPUT}" \
  "${CANDIDATE_OUTPUT}" \
  --output "${COMPARISON_OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-selection-decision \
  "${COMPARISON_OUTPUT}"

test -s "${OUTPUT}/base/reports/gold_evaluation.json"
test -s "${OUTPUT}/candidate/reports/gold_evaluation.json"
test -s "${OUTPUT}/comparison/benchmark_comparison.md"
test -s "${OUTPUT}/comparison/target_comparison.json"
test -s "${OUTPUT}/comparison/target_changes.csv"
test -s "${OUTPUT}/comparison/target_comparison.md"
test -s "${OUTPUT}/comparison/model_selection_decision.json"
test -s "${OUTPUT}/comparison/model_selection_decision.md"

echo "Model judge comparison complete: ${OUTPUT}"
