#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-dpo-export-smoke}"
INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_train_clean.jsonl}"
LIMIT="${LIMIT:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

if [[ ! -s "${INPUT}" ]]; then
  echo "SciImpact SFT input not found: ${INPUT}" >&2
  echo "Download SciImpact first, then rerun this script." >&2
  exit 1
fi

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}/finetuning"

sed -n "1,${LIMIT}p" "${INPUT}" > "${OUTPUT}/input.jsonl"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks export-sciimpact-dpo \
  --input "${OUTPUT}/input.jsonl" \
  --output "${OUTPUT}/finetuning/sciimpact_dpo_preferences.jsonl"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli validate-dpo-data \
  --train-file "${OUTPUT}/finetuning/sciimpact_dpo_preferences.jsonl" \
  --output "${OUTPUT}/finetuning"

test -s "${OUTPUT}/input.jsonl"
test -s "${OUTPUT}/finetuning/sciimpact_dpo_preferences.jsonl"
test -s "${OUTPUT}/finetuning/sciimpact_dpo_manifest.json"
test -s "${OUTPUT}/finetuning/dpo_data_validation.json"

loaded="$(
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}/finetuning/dpo_data_validation.json"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report["loaded_example_count"])
PY
)"

if [[ "${loaded}" -lt 1 ]]; then
  echo "Expected at least one loaded DPO example" >&2
  exit 1
fi

echo "SciImpact DPO export smoke complete: ${OUTPUT} (loaded=${loaded})"
