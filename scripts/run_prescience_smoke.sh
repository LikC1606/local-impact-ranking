#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/prescience-smoke}"
SAMPLE_PATH="${SAMPLE_PATH:-data/prescience/sample.jsonl}"
TARGETS="${TARGETS:-${LIMIT:-100}}"
LIMIT="${LIMIT:-100}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50}"
SHUFFLE_SEED="${SHUFFLE_SEED:-13}"
TOP_K="${TOP_K:-5}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

sample_target_count=0
if [[ -s "${SAMPLE_PATH}" ]]; then
  sample_target_count="$("${PYTHON_BIN}" - "${SAMPLE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
targets = 0
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if "target" in (row.get("roles") or []) or row.get("key_references"):
            targets += 1
print(targets)
PY
)"
fi

if [[ ! -s "${SAMPLE_PATH}" || "${sample_target_count}" -lt "${TARGETS}" ]]; then
  scripts/download_prescience_sample.sh "${SAMPLE_PATH}"
fi

mkdir -p "${OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-prescience \
  --input "${SAMPLE_PATH}" \
  --output "${OUTPUT}/prescience.papers.jsonl" \
  --limit "${LIMIT}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --shuffle-seed "${SHUFFLE_SEED}"

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
test -s "${OUTPUT}/reports/gold_candidate_position.json"
test -s "${OUTPUT}/reports/gold_candidate_position.md"
test -s "${OUTPUT}/reports/gold_baseline_comparison.json"
test -s "${OUTPUT}/reports/gold_baseline_comparison.md"
test -s "${OUTPUT}/reports/mvp_summary.md"
test -s "${OUTPUT}/reports/artifact_index.json"
test -s "${OUTPUT}/run_manifest.json"
PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli report-bundle "${OUTPUT}" --output "${OUTPUT}/report_bundle"
test -s "${OUTPUT}/report_bundle/README.md"
test -s "${OUTPUT}/report_bundle/bundle_manifest.json"

echo "PreScience smoke run complete: ${OUTPUT}"
