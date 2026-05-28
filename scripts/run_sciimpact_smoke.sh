#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/sciimpact-smoke}"
INPUT="${INPUT:-examples/sciimpact_pairs.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/air/bin/python3.11}"

mkdir -p "${OUTPUT}/rankings"

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli benchmarks convert-sciimpact \
  --input "${INPUT}" \
  --output "${OUTPUT}/converted"

cat > "${OUTPUT}/rankings/papers.csv" <<'CSV'
rank,id,score,final_score
1,sciimpact-high-1,90.0,90.0
2,sciimpact-policy-1,85.0,85.0
3,doi:10.5555/high-method,80.0,80.0
4,sciimpact-low-1,40.0,40.0
5,sciimpact-method-1,35.0,35.0
6,doi:10.5555/low-method,30.0,30.0
CSV

PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli evaluate-pairwise-impact \
  --rankings "${OUTPUT}/rankings/papers.csv" \
  --labels "${OUTPUT}/converted/pairwise_labels.jsonl" \
  --output "${OUTPUT}/pairwise_eval"

test -s "${OUTPUT}/converted/papers.jsonl"
test -s "${OUTPUT}/converted/pairwise_labels.jsonl"
test -s "${OUTPUT}/converted/manifest.json"
test -s "${OUTPUT}/rankings/papers.csv"
test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.json"
test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.md"

echo "SciImpact pairwise smoke run complete: ${OUTPUT}"
