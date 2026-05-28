#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/quickstart}"
PYTHON_BIN="${PYTHON_BIN:-python}"

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli doctor \
  --output "${OUTPUT}/doctor"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli run \
  --input examples/context_papers.jsonl \
  --output "${OUTPUT}/run" \
  --top-k 1 \
  --credit-mode context_weighted \
  --strict-validation

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli score-card \
  "${OUTPUT}/run" \
  --output "${OUTPUT}/score_cards"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli entity-resolution-report \
  --input examples/context_papers.jsonl \
  --output "${OUTPUT}/entity_resolution"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli report-bundle \
  "${OUTPUT}/run" \
  --output "${OUTPUT}/run/report_bundle"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli runs-index \
  "${OUTPUT}" \
  --output "${OUTPUT}/runs_index"

test -s "${OUTPUT}/doctor/doctor.json"
test -s "${OUTPUT}/run/rankings/papers.csv"
test -s "${OUTPUT}/run/reports/mvp_summary.md"
test -s "${OUTPUT}/run/reports/project_readiness.md"
test -s "${OUTPUT}/run/reports/score_cards.md"
test -s "${OUTPUT}/run/reports/entity_resolution_report.md"
test -s "${OUTPUT}/score_cards/score_cards.md"
test -s "${OUTPUT}/entity_resolution/entity_resolution_report.md"
test -s "${OUTPUT}/run/report_bundle/README.md"
test -s "${OUTPUT}/run/report_bundle/bundle_manifest.json"
test -s "${OUTPUT}/runs_index/index.md"

echo "Quickstart complete: ${OUTPUT}"
