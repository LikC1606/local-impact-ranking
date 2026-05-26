#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/prescience-smoke}"
SAMPLE_PATH="${SAMPLE_PATH:-data/prescience/sample.jsonl}"
LIMIT="${LIMIT:-100}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50}"
TOP_K="${TOP_K:-5}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

if [[ ! -s "${SAMPLE_PATH}" ]]; then
  scripts/download_prescience_sample.sh "${SAMPLE_PATH}"
fi

mkdir -p "${OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-prescience \
  --input "${SAMPLE_PATH}" \
  --output "${OUTPUT}/prescience.papers.jsonl" \
  --limit "${LIMIT}" \
  --max-candidates "${MAX_CANDIDATES}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input "${OUTPUT}/prescience.papers.jsonl" \
  --output "${OUTPUT}" \
  --top-k "${TOP_K}" \
  --credit-mode rank_weighted \
  --strict-validation

test -s "${OUTPUT}/prescience.papers.jsonl"
test -s "${OUTPUT}/rankings/papers.csv"
test -s "${OUTPUT}/reports/gold_evaluation.json"
test -s "${OUTPUT}/reports/gold_evaluation.md"
test -s "${OUTPUT}/reports/mvp_summary.md"
test -s "${OUTPUT}/reports/artifact_index.json"
test -s "${OUTPUT}/run_manifest.json"

echo "PreScience smoke run complete: ${OUTPUT}"
