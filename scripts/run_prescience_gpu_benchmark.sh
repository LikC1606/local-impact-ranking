#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/prescience-gpu-benchmark}"

SAMPLE_PATH="${SAMPLE_PATH:-data/prescience/sample.jsonl}"
LIMIT="${LIMIT:-100}"
MAX_CANDIDATES="${MAX_CANDIDATES:-32}"
TOP_K="${TOP_K:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge}"
TEMPERATURE="${TEMPERATURE:-0}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

INPUT_PATH="${OUTPUT}/prescience.papers.jsonl"
AUDIT_PATH="${OUTPUT}/audit/judge_outputs.jsonl"

if [[ ! -s "${SAMPLE_PATH}" ]]; then
  scripts/download_prescience_sample.sh "${SAMPLE_PATH}"
fi

scripts/check_model_server.sh "${BASE_URL}" >/dev/null

mkdir -p "${OUTPUT}/audit"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-prescience \
  --input "${SAMPLE_PATH}" \
  --output "${INPUT_PATH}" \
  --limit "${LIMIT}" \
  --max-candidates "${MAX_CANDIDATES}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli judge \
  --input "${INPUT_PATH}" \
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

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli score \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT}" \
  --top-k "${TOP_K}" \
  --limit "${LIMIT}" \
  --credit-mode rank_weighted \
  --judge-audit "${AUDIT_PATH}" \
  --strict-validation

test -s "${INPUT_PATH}"
test -s "${OUTPUT}/audit/judge_outputs.jsonl"
test -s "${OUTPUT}/audit/scoring_edges.csv"
test -s "${OUTPUT}/audit/prompts.jsonl"
test -s "${OUTPUT}/rankings/papers.csv"
test -s "${OUTPUT}/rankings/fields.csv"
test -s "${OUTPUT}/rankings/years.csv"
test -s "${OUTPUT}/reports/validation.json"
test -s "${OUTPUT}/reports/evaluation.json"
test -s "${OUTPUT}/reports/evaluation.md"
test -s "${OUTPUT}/reports/gold_evaluation.json"
test -s "${OUTPUT}/reports/gold_evaluation.md"
test -s "${OUTPUT}/reports/proxy_evaluation.json"
test -s "${OUTPUT}/reports/mvp_summary.md"
test -f "${OUTPUT}/run_manifest.json"

echo "PreScience GPU benchmark complete: ${OUTPUT}"
