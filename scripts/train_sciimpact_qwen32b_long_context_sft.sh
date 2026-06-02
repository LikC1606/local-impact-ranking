#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/deepseek-r1-distill-qwen-32b}"
TRAIN_FILE="${TRAIN_FILE:-runs/sciimpact-pairwise-sft-full/train_pairwise_sft.jsonl}"
OUTPUT="${OUTPUT:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16000}"
MAX_STEPS="${MAX_STEPS:-256}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-131}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-random}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli inspect-sft-tokenization \
  --model "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --output "${OUTPUT}/tokenization-preflight" \
  --max-length "${MAX_LENGTH}" \
  --max-examples "${MAX_EXAMPLES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli train-sft \
  --model "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --output "${OUTPUT}" \
  --max-examples "${MAX_EXAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --device-map "${DEVICE_MAP}" \
  --sampling-strategy "${SAMPLING_STRATEGY}"
