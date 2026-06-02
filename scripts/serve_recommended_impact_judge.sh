#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/deepseek-r1-distill-qwen-32b}"
ADAPTER_PATH="${ADAPTER_PATH:-runs/sciimpact-source-task-sft-dpo/checkpoints/qwen32b-source-task-sft16k4096-dpo16k1024-128step}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-impact-judge-source-task-dpo}"
PORT="${PORT:-8000}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}" \
VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}" \
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${SERVER_HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --enable-lora \
  --lora-modules "${SERVED_MODEL_NAME}=${ADAPTER_PATH}"
