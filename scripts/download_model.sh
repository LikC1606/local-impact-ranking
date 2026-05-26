#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
LOCAL_DIR="${2:-models/deepseek-r1-distill-qwen-32b}"

mkdir -p "${LOCAL_DIR}"

huggingface-cli download "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}" \
  --resume-download
