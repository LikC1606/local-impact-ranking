#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-sft-smoke}"
INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"
LIMIT="${LIMIT:-200}"
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

test -s "${OUTPUT}/input.jsonl"
test -s "${OUTPUT}/converted/papers.jsonl"
test -s "${OUTPUT}/converted/pairwise_labels.jsonl"
test -s "${OUTPUT}/converted/manifest.json"
grep -q '"skipped_pair_count"' "${OUTPUT}/converted/manifest.json"

pair_count="$(wc -l < "${OUTPUT}/converted/pairwise_labels.jsonl" | tr -d ' ')"
paper_count="$(wc -l < "${OUTPUT}/converted/papers.jsonl" | tr -d ' ')"

if [[ "${pair_count}" -ne "${LIMIT}" ]]; then
  echo "Expected ${LIMIT} converted pairs, got ${pair_count}" >&2
  exit 1
fi

if [[ "${paper_count}" -lt "${LIMIT}" ]]; then
  echo "Expected at least ${LIMIT} converted items, got ${paper_count}" >&2
  exit 1
fi

echo "SciImpact SFT conversion smoke run complete: ${OUTPUT} (pairs=${pair_count}, items=${paper_count})"
