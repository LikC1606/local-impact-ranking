#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-model-eval}"
INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge}"
LIMIT="${LIMIT:-200}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

if [[ ! -s "${INPUT}" ]]; then
  echo "SciImpact SFT input not found: ${INPUT}" >&2
  echo "Download SciImpact first, then rerun this script." >&2
  exit 1
fi

scripts/check_model_server.sh "${BASE_URL}"

mkdir -p "${OUTPUT}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli generate-sciimpact-predictions \
  --sft "${INPUT}" \
  --output "${OUTPUT}/predictions.jsonl" \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --limit "${LIMIT}" \
  --resume

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli evaluate-sciimpact-predictions \
  --sft "${INPUT}" \
  --predictions "${OUTPUT}/predictions.jsonl" \
  --output "${OUTPUT}/eval"

test -s "${OUTPUT}/predictions.jsonl"
test -s "${OUTPUT}/sciimpact_prediction_manifest.json"
test -s "${OUTPUT}/eval/sciimpact_model_eval.json"
test -s "${OUTPUT}/eval/sciimpact_model_eval.md"

accuracy="$(
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}/eval/sciimpact_model_eval.json"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{report['summary']['accuracy']:.3f}")
PY
)"

echo "SciImpact model evaluation complete: ${OUTPUT} (accuracy=${accuracy})"
