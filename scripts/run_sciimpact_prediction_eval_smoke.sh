#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-prediction-eval-smoke}"
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

PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}/input.jsonl" "${OUTPUT}/oracle_predictions.jsonl"
import json
import sys

input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
    for line_number, line in enumerate(src, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        answer = next(
            message["value"]
            for message in row["conversations"]
            if message.get("from") == "gpt"
        )
        dst.write(json.dumps({"source_line": line_number, "prediction": answer}) + "\n")
PY

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli evaluate-sciimpact-predictions \
  --sft "${OUTPUT}/input.jsonl" \
  --predictions "${OUTPUT}/oracle_predictions.jsonl" \
  --output "${OUTPUT}/model_eval"

test -s "${OUTPUT}/input.jsonl"
test -s "${OUTPUT}/oracle_predictions.jsonl"
test -s "${OUTPUT}/model_eval/sciimpact_model_eval.json"
test -s "${OUTPUT}/model_eval/sciimpact_model_eval.md"
test -s "${OUTPUT}/model_eval/sciimpact_model_credibility_scorecard.json"
test -s "${OUTPUT}/model_eval/sciimpact_model_credibility_scorecard.md"

accuracy="$(
  PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}/model_eval/sciimpact_model_eval.json"
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{report['summary']['accuracy']:.3f}")
PY
)"

if [[ "${accuracy}" != "1.000" ]]; then
  echo "Expected oracle accuracy 1.000, got ${accuracy}" >&2
  exit 1
fi

echo "SciImpact prediction evaluation smoke complete: ${OUTPUT} (accuracy=${accuracy})"
