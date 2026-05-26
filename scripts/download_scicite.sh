#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-data/scicite}"
SCICITE_URL="${SCICITE_URL:-https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"
ARCHIVE_PATH="${DEST_DIR}/scicite.tar.gz"

mkdir -p "${DEST_DIR}/raw"

if [[ ! -s "${ARCHIVE_PATH}" ]]; then
  curl -L "${SCICITE_URL}" -o "${ARCHIVE_PATH}"
fi

tar -xzf "${ARCHIVE_PATH}" -C "${DEST_DIR}/raw"

for split in train dev test; do
  SOURCE_PATH="$(find "${DEST_DIR}/raw" -type f \( -name "${split}.jsonl" -o -name "${split}.json" \) | head -n 1)"
  if [[ -z "${SOURCE_PATH}" ]]; then
    echo "Missing SciCite split: ${split}" >&2
    exit 1
  fi
  PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-scicite \
    --input "${SOURCE_PATH}" \
    --output "${DEST_DIR}/${split}.papers.jsonl"
done

echo "SciCite converted files:"
printf '  %s\n' "${DEST_DIR}/train.papers.jsonl" "${DEST_DIR}/dev.papers.jsonl" "${DEST_DIR}/test.papers.jsonl"
