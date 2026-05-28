#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-sft-baseline}"
INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"
LIMIT="${LIMIT:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

if [[ ! -s "${INPUT}" ]]; then
  echo "SciImpact SFT input not found: ${INPUT}" >&2
  echo "Download SciImpact first, then rerun this script." >&2
  exit 1
fi

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

sed -n "1,${LIMIT}p" "${INPUT}" > "${OUTPUT}/input.jsonl"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-sciimpact \
  --input "${OUTPUT}/input.jsonl" \
  --output "${OUTPUT}/converted"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli sciimpact-text-baseline \
  --input "${OUTPUT}/converted/papers.jsonl" \
  --output "${OUTPUT}/rankings/sciimpact_text_prior.csv"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli evaluate-pairwise-impact \
  --rankings "${OUTPUT}/rankings/sciimpact_text_prior.csv" \
  --labels "${OUTPUT}/converted/pairwise_labels.jsonl" \
  --output "${OUTPUT}/pairwise_eval"

test -s "${OUTPUT}/converted/papers.jsonl"
test -s "${OUTPUT}/converted/pairwise_labels.jsonl"
test -s "${OUTPUT}/converted/manifest.json"
test -s "${OUTPUT}/rankings/sciimpact_text_prior.csv"
test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.json"
test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.md"

accuracy="$(
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}/pairwise_eval/pairwise_evaluation.json"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{report['summary']['pairwise_accuracy']:.3f}")
PY
)"

echo "SciImpact SFT text-prior baseline complete: ${OUTPUT} (pairwise_accuracy=${accuracy})"
