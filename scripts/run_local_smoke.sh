#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/local-smoke}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input examples/tiny_papers.jsonl \
  --output "${OUTPUT}/tiny" \
  --top-k 1 \
  --strict-validation

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input examples/context_papers.jsonl \
  --output "${OUTPUT}/context" \
  --top-k 1 \
  --credit-mode context_weighted \
  --strict-validation

test -s "${OUTPUT}/tiny/reports/validation.json"
test -s "${OUTPUT}/tiny/reports/mvp_summary.md"
test -s "${OUTPUT}/context/audit/scoring_edges.csv"
test -s "${OUTPUT}/context/audit/prompts.jsonl"
test -s "${OUTPUT}/context/rankings/fields.csv"
test -s "${OUTPUT}/context/rankings/years.csv"
test -s "${OUTPUT}/context/reports/validation.json"
test -s "${OUTPUT}/context/reports/evaluation.md"
test -s "${OUTPUT}/context/reports/gold_evaluation.json"
test -s "${OUTPUT}/context/reports/mvp_summary.md"
test -s "${OUTPUT}/context/reports/artifact_index.json"
grep -q "1,r1,1.500000" "${OUTPUT}/context/rankings/papers.csv"
grep -q "1.500000" "${OUTPUT}/context/audit/scoring_edges.csv"

echo "Local smoke run complete: ${OUTPUT}"
