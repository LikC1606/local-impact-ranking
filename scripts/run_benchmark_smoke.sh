#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/benchmark-smoke}"
INPUT="${INPUT:-data/scicite/dev.papers.jsonl}"
LIMIT="${LIMIT:-5}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

if [[ ! -s "${INPUT}" ]]; then
  echo "Missing benchmark input: ${INPUT}" >&2
  echo "Run scripts/download_scicite.sh data/scicite or set INPUT=/path/to/*.papers.jsonl" >&2
  exit 1
fi

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --top-k 1 \
  --limit "${LIMIT}" \
  --credit-mode context_weighted \
  --strict-validation

test -s "${OUTPUT}/rankings/papers.csv"
test -s "${OUTPUT}/rankings/fields.csv"
test -s "${OUTPUT}/rankings/years.csv"
test -s "${OUTPUT}/audit/judge_outputs.jsonl"
test -s "${OUTPUT}/audit/prompts.jsonl"
test -s "${OUTPUT}/reports/validation.json"
test -s "${OUTPUT}/reports/evaluation.md"
test -s "${OUTPUT}/reports/gold_evaluation.json"
test -s "${OUTPUT}/reports/proxy_evaluation.json"
test -s "${OUTPUT}/reports/mvp_summary.md"
test -s "${OUTPUT}/run_manifest.json"

echo "Benchmark smoke run complete: ${OUTPUT}"
