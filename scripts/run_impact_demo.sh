#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-runs/impact-demo}"
PYTHON_BIN="${PYTHON_BIN:-python}"

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-preflight \
  --input examples/impact_demo_papers.jsonl \
  --output "${OUTPUT}/preflight" \
  --max-pairs 6 \
  --pair-budget manual \
  --pair-sampling sequential

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-dataset \
  --input examples/impact_demo_papers.jsonl \
  --output "${OUTPUT}/model-impact" \
  --model offline-demo-judge \
  --pairwise-judgments examples/impact_demo_pairwise_judgments.jsonl \
  --validity-labels examples/impact_demo_future_labels.jsonl \
  --validity-label-column future_impact \
  --validity-bootstrap-samples 20 \
  --validity-permutation-samples 20 \
  --report-bundle \
  --report-bundle-output "${OUTPUT}/model-impact/report_bundle"

test -s "${OUTPUT}/preflight/model_score_preflight.json"
test -s "${OUTPUT}/preflight/model_score_preflight.md"
test -s "${OUTPUT}/model-impact/impact_scores.csv"
test -s "${OUTPUT}/model-impact/impact_scores.jsonl"
test -s "${OUTPUT}/model-impact/model_impact_report.md"
test -s "${OUTPUT}/model-impact/external_evidence_readiness.md"
test -s "${OUTPUT}/model-impact/external_evidence_paper_coverage.csv"
test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"
test -s "${OUTPUT}/model-impact/impact_score_cards.md"
test -s "${OUTPUT}/model-impact/impact_validity/impact_validity_evaluation.json"
test -s "${OUTPUT}/model-impact/impact_validity/impact_credibility_scorecard.json"
test -s "${OUTPUT}/model-impact/report_bundle/bundle_manifest.json"
test -s "${OUTPUT}/model-impact/report_bundle/README.md"
test -s "${OUTPUT}/model-impact/report_bundle/impact_scores.csv"
test -s "${OUTPUT}/model-impact/report_bundle/model_impact_report.md"
grep -q "validated_impact_claim.status" "${OUTPUT}/model-impact/model_impact_report.md"
grep -q "final_impact_score" "${OUTPUT}/model-impact/impact_scores.csv"
grep -q "Impact Claim Certificate" "${OUTPUT}/model-impact/impact_claim_certificate.md"
grep -q "Model Score Dataset" "${OUTPUT}/model-impact/report_bundle/README.md"

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - <<'PY' "${OUTPUT}"
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
impact_dir = root / "model-impact"
scores_path = impact_dir / "impact_scores.csv"
certificate_path = impact_dir / "impact_claim_certificate.json"
validity_path = impact_dir / "impact_validity" / "impact_credibility_scorecard.json"

scores = list(csv.DictReader(scores_path.open(encoding="utf-8")))
certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
validity = json.loads(validity_path.read_text(encoding="utf-8"))
claim = certificate.get("claim", {})
if not claim.get("status") or not claim.get("reason"):
    raise SystemExit("impact claim certificate is missing claim.status or claim.reason")

lines = [
    "# Impact Demo Report",
    "",
    "## What This Demonstrates",
    "",
    "A local paper dataset can be scored into dataset-local impact scores with evidence, confidence, validity boundaries, and a report bundle without calling a model server.",
    "",
    "## Claim Boundary",
    "",
    "- Scope: dataset-local paper influence inside the supplied demo corpus.",
    "- Not claimed: global scientific impact or validated impact accuracy on real deployment data.",
    "",
    "## Claim Certificate",
    "",
    "| Item | Value |",
    "| --- | --- |",
    f"| status | {claim.get('status', '')} |",
    f"| can_report_dataset_local_score | {claim.get('can_report_dataset_local_score', '')} |",
    f"| can_claim_validated_impact_accuracy | {claim.get('can_claim_validated_impact_accuracy', '')} |",
    f"| can_claim_high_credibility | {claim.get('can_claim_high_credibility', '')} |",
    f"| confidence | {claim.get('confidence', '')} |",
    f"| reason | {claim.get('reason', '')} |",
    "",
    "## Top Impact Scores",
    "",
    "| Rank | Paper ID | Final Impact Score | Evidence Tier | Reportability |",
    "| ---: | --- | ---: | --- | --- |",
]
for row in scores[:5]:
    lines.append(
        "| "
        f"{row.get('rank', '')} | "
        f"{row.get('paper_id', '')} | "
        f"{float(row.get('final_impact_score') or 0.0):.3f} | "
        f"{row.get('credible_evidence_tier', '')} | "
        f"{row.get('reportability_status', '')} |"
    )
lines.extend(
    [
        "",
        "## Validation Boundary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| impact_credibility_status | {validity.get('status', '')} |",
        f"| impact_credibility_grade | {validity.get('grade', '')} |",
        f"| can_claim_high_validation_credibility | {validity.get('can_claim_high_validation_credibility', '')} |",
        f"| reason | {validity.get('reason', '')} |",
        "",
        "## Key Artifacts",
        "",
        f"- Scores: `{impact_dir / 'impact_scores.csv'}`",
        f"- Claim certificate: `{impact_dir / 'impact_claim_certificate.md'}`",
        f"- Score cards: `{impact_dir / 'impact_score_cards.md'}`",
        f"- Model impact report: `{impact_dir / 'model_impact_report.md'}`",
        f"- Report bundle: `{impact_dir / 'report_bundle'}`",
        "",
    ]
)
(root / "demo_report.md").write_text("\n".join(lines), encoding="utf-8")
PY

test -s "${OUTPUT}/demo_report.md"
grep -q "Impact Demo Report" "${OUTPUT}/demo_report.md"
grep -q "| status | evidence_supported |" "${OUTPUT}/demo_report.md"
grep -q "Scores have auditable local evidence" "${OUTPUT}/demo_report.md"

cat > "${OUTPUT}/demo_status.json" <<JSON
{
  "status": "passed",
  "mode": "offline_pairwise_replay",
  "input": "examples/impact_demo_papers.jsonl",
  "scores": "${OUTPUT}/model-impact/impact_scores.csv",
  "report": "${OUTPUT}/demo_report.md",
  "claim_certificate": "${OUTPUT}/model-impact/impact_claim_certificate.md",
  "report_bundle": "${OUTPUT}/model-impact/report_bundle",
  "claim_boundary": "demo dataset-local paper influence; not a global scientific-impact claim"
}
JSON

test -s "${OUTPUT}/demo_status.json"

echo "Impact demo complete: ${OUTPUT}"
