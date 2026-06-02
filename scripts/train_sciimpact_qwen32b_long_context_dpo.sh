#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/deepseek-r1-distill-qwen-32b}"
BASE_ADAPTER="${BASE_ADAPTER:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step}"
TRAIN_FILE="${TRAIN_FILE:-runs/sciimpact-pairwise-dpo-full/train_pairwise_dpo.jsonl}"
OUTPUT="${OUTPUT:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-dpo16k-256step}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16000}"
MAX_STEPS="${MAX_STEPS:-256}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
BETA="${BETA:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SEED="${SEED:-137}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-random}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

test -d "${BASE_ADAPTER}"
test -s "${TRAIN_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli train-dpo \
  --model "${MODEL_PATH}" \
  --base-adapter "${BASE_ADAPTER}" \
  --train-file "${TRAIN_FILE}" \
  --output "${OUTPUT}" \
  --max-examples "${MAX_EXAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --beta "${BETA}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --device-map "${DEVICE_MAP}" \
  --sampling-strategy "${SAMPLING_STRATEGY}"
