#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-data/prescience/sample.jsonl}"
TARGETS="${TARGETS:-100}"
MAX_ROWS="${MAX_ROWS:-50000}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

mkdir -p "$(dirname "${OUTPUT}")"

HF_ENDPOINT="${HF_ENDPOINT}" "${PYTHON_BIN}" - "${OUTPUT}" "${TARGETS}" "${MAX_ROWS}" <<'PY'
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

output = Path(sys.argv[1])
target_limit = int(sys.argv[2])
row_limit = int(sys.argv[3])

dataset = load_dataset('allenai/prescience', split='train', streaming=True)
targets = 0
rows = 0
with output.open("w", encoding="utf-8") as handle:
    for row in dataset:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows += 1
        if "target" in (row.get("roles") or []) or row.get("key_references"):
            targets += 1
        if targets >= target_limit or rows >= row_limit:
            break
print(json.dumps({"output": str(output), "rows": rows, "targets": targets}, indent=2), flush=True)
os._exit(0)
PY
