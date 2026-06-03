#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: scripts/apply_high_recall_evidence.sh INPUT_JSONL EVIDENCE_CSV [OUTPUT_DIR]

Imports a filled external/high-recall citation evidence CSV, audits citation
completeness, and optionally replays existing pairwise judgments to recompute
impact scores without GPU inference.

Environment variables:
  PYTHON_BIN=python
  BASE_URL=http://localhost:8000/v1
  MODEL=local-impact-judge-source-task-dpo
  PAIRWISE_JUDGMENTS=path/to/pairwise_judgments.jsonl
  CALIBRATION_REPORT=configs/calibration/sciimpact-qwen32b-heldout-p50.json
  REPORT_BUNDLE=1

Outputs:
  papers.high_recall.jsonl
  import_summary.json
  citation-completeness/
  model-impact/                 when PAIRWISE_JUDGMENTS is supplied
  high_recall_external_evidence_status.json
USAGE
  exit 0
fi

INPUT="${1:?Usage: scripts/apply_high_recall_evidence.sh INPUT_JSONL EVIDENCE_CSV [OUTPUT_DIR]}"
EVIDENCE="${2:?Usage: scripts/apply_high_recall_evidence.sh INPUT_JSONL EVIDENCE_CSV [OUTPUT_DIR]}"
OUTPUT="${3:-runs/high-recall-evidence}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge-source-task-dpo}"
PAIRWISE_JUDGMENTS="${PAIRWISE_JUDGMENTS:-}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-configs/calibration/sciimpact-qwen32b-heldout-p50.json}"
REPORT_BUNDLE="${REPORT_BUNDLE:-1}"

mkdir -p "${OUTPUT}"

IMPORTED_INPUT="${OUTPUT}/papers.high_recall.jsonl"
IMPORT_SUMMARY="${OUTPUT}/import_summary.json"
CITATION_COMPLETENESS_BEFORE_OUTPUT="${OUTPUT}/citation-completeness-before"
CITATION_COMPLETENESS_OUTPUT="${OUTPUT}/citation-completeness"
MODEL_OUTPUT="${OUTPUT}/model-impact"
STATUS_PATH="${OUTPUT}/high_recall_external_evidence_status.json"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli citation-completeness-audit \
  --input "${INPUT}" \
  --output "${CITATION_COMPLETENESS_BEFORE_OUTPUT}" \
  --markdown > "${OUTPUT}/citation_completeness_before_stdout.md"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli import-external-evidence \
  --input "${INPUT}" \
  --evidence "${EVIDENCE}" \
  --output "${IMPORTED_INPUT}" \
  --summary-output "${IMPORT_SUMMARY}"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli citation-completeness-audit \
  --input "${IMPORTED_INPUT}" \
  --output "${CITATION_COMPLETENESS_OUTPUT}" \
  --markdown > "${OUTPUT}/citation_completeness_stdout.md"

test -s "${OUTPUT}/import_summary.json"
test -s "${IMPORTED_INPUT}"
test -s "${OUTPUT}/citation-completeness-before/citation_completeness_audit.json"
test -s "${OUTPUT}/citation-completeness-before/citation_completeness_audit.md"
test -s "${OUTPUT}/citation-completeness-before/citation_completeness_papers.csv"
test -s "${OUTPUT}/citation-completeness/citation_completeness_audit.json"
test -s "${OUTPUT}/citation-completeness/citation_completeness_audit.md"
test -s "${OUTPUT}/citation-completeness/citation_completeness_papers.csv"

score_status="skipped"
if [[ -n "${PAIRWISE_JUDGMENTS}" ]]; then
  model_score_args=(
    --input "${IMPORTED_INPUT}"
    --output "${MODEL_OUTPUT}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --pairwise-judgments "${PAIRWISE_JUDGMENTS}"
    --calibration-report "${CALIBRATION_REPORT}"
  )
  if [[ "${REPORT_BUNDLE}" == "1" ]]; then
    model_score_args+=(--report-bundle)
  fi
  PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-dataset "${model_score_args[@]}"
  test -s "${MODEL_OUTPUT}/impact_scores.csv"
  test -s "${MODEL_OUTPUT}/impact_claim_certificate.md"
  score_status="replayed"
fi

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - "${STATUS_PATH}" "${INPUT}" "${EVIDENCE}" "${OUTPUT}" "${IMPORTED_INPUT}" "${IMPORT_SUMMARY}" "${CITATION_COMPLETENESS_BEFORE_OUTPUT}" "${CITATION_COMPLETENESS_OUTPUT}" "${PAIRWISE_JUDGMENTS}" "${MODEL_OUTPUT}" "${score_status}" <<'PY'
import json
import sys
from pathlib import Path

(
    status_path,
    input_path,
    evidence_path,
    output_dir,
    imported_input,
    import_summary_path,
    citation_completeness_before_output,
    citation_completeness_output,
    pairwise_judgments,
    model_output,
    score_status,
) = sys.argv[1:]

import_summary = json.loads(Path(import_summary_path).read_text(encoding="utf-8"))
before_audit_path = Path(citation_completeness_before_output) / "citation_completeness_audit.json"
audit_path = Path(citation_completeness_output) / "citation_completeness_audit.json"
before_audit_summary = json.loads(before_audit_path.read_text(encoding="utf-8"))["summary"]
audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))["summary"]

delta_keys = (
    "citation_evidence_coverage",
    "multi_source_coverage",
    "positive_citation_evidence_coverage",
    "mean_completeness_score",
    "papers_with_source_conflict",
    "papers_with_confirmed_multi_source_zero",
)
citation_completeness_delta = {
    key: round(float(audit_summary.get(key) or 0.0) - float(before_audit_summary.get(key) or 0.0), 6)
    for key in delta_keys
}

status = {
    "status": "passed",
    "input": input_path,
    "evidence": evidence_path,
    "output": output_dir,
    "imported_input": imported_input,
    "import_summary": import_summary,
    "citation_completeness_before_output": citation_completeness_before_output,
    "citation_completeness_before_summary": before_audit_summary,
    "citation_completeness_output": citation_completeness_output,
    "citation_completeness_summary": audit_summary,
    "citation_completeness_delta": citation_completeness_delta,
    "pairwise_judgments": pairwise_judgments or None,
    "score_status": score_status,
    "model_output": model_output if score_status == "replayed" else None,
}
Path(status_path).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
PY

test -s "${STATUS_PATH}"

echo "Applied high-recall evidence to ${IMPORTED_INPUT}"
echo "Citation completeness audit: ${CITATION_COMPLETENESS_OUTPUT}"
if [[ "${score_status}" == "replayed" ]]; then
  echo "Replayed model score output: ${MODEL_OUTPUT}"
else
  echo "Model scoring skipped; set PAIRWISE_JUDGMENTS to replay scores."
fi
echo "Status: ${STATUS_PATH}"
