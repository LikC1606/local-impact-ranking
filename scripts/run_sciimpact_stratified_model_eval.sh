#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-model-eval-stratified}"
INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"
PER_DIMENSION="${PER_DIMENSION:-20}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

SAMPLE_DIR="${OUTPUT}/sample"
SAMPLE_PATH="${SAMPLE_DIR}/sciimpact_sft_stratified.jsonl"

if [[ ! -s "${INPUT}" ]]; then
  echo "SciImpact SFT input not found: ${INPUT}" >&2
  echo "Download SciImpact first, then rerun this script." >&2
  exit 1
fi

mkdir -p "${SAMPLE_DIR}"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli sample-sciimpact-sft \
  --sft "${INPUT}" \
  --output "${SAMPLE_PATH}" \
  --per-dimension "${PER_DIMENSION}" \
  --seed "${SEED}"

sample_count="$(wc -l < "${SAMPLE_PATH}" | tr -d ' ')"

INPUT="${SAMPLE_PATH}" LIMIT="${sample_count}" \
  scripts/run_sciimpact_model_eval.sh "${OUTPUT}/model_eval"

test -s "${SAMPLE_PATH}"
test -s "${SAMPLE_DIR}/sciimpact_stratified_sample_manifest.json"
test -s "${OUTPUT}/model_eval/eval/sciimpact_model_eval.json"
test -s "${OUTPUT}/model_eval/eval/sciimpact_model_eval.md"

echo "SciImpact stratified model evaluation complete: ${OUTPUT} (sample_count=${sample_count})"
