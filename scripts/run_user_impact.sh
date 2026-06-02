#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: scripts/run_user_impact.sh INPUT_JSONL [OUTPUT_DIR]

Runs the user-facing paper impact workflow:
  validate -> model-score-preflight -> model-score-dataset -> report bundle

Outputs:
  user_impact_status.json
  user_impact_summary.md
  model-impact/impact_scores.csv
  model-impact/impact_claim_certificate.md
  model-impact/report_bundle/

Common environment variables:
  RUN_PRESET=quick|balanced|full|custom
  BASE_URL=http://localhost:8000/v1
  MODEL=local-impact-judge-source-task-dpo
  MAX_PAIRS=200
  PAIR_BUDGET=auto
  PAIR_SAMPLING=coverage
  TARGET_PAPER_COVERAGE=0.25
  CONCURRENCY=1
  PAIRWISE_JUDGMENTS=path/to/pairwise_judgments.jsonl
  RESUME_PAIRWISE_JUDGMENTS=path/to/pairwise_judgments.jsonl
  PAIRWISE_JOURNAL=path/to/pairwise_judgments.jsonl
  VALIDITY_LABELS=path/to/future_impact_labels.jsonl
  FUSION_WEIGHTS=path/to/fusion_calibration.json
  MODEL_CREDIBILITY_REPORT=configs/model_credibility/local-impact-judge-source-task-dpo.json

Claim boundary:
  Scores are dataset-local paper influence unless external calibrated evidence
  and validity labels support a stronger claim.
USAGE
  exit 0
fi

INPUT="${1:?Usage: scripts/run_user_impact.sh INPUT_JSONL [OUTPUT_DIR]}"
OUTPUT="${2:-runs/user-impact}"

USER_SET_MAX_PAIRS="${MAX_PAIRS+x}"
USER_SET_TARGET_PAPER_COVERAGE="${TARGET_PAPER_COVERAGE+x}"
USER_SET_CONCURRENCY="${CONCURRENCY+x}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PRESET="${RUN_PRESET:-custom}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-local-impact-judge-source-task-dpo}"
API_KEY="${API_KEY:-EMPTY}"
TIMEOUT="${TIMEOUT:-120}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_PAIRS="${MAX_PAIRS:-}"
PAIR_BUDGET="${PAIR_BUDGET:-auto}"
PAIR_SAMPLING="${PAIR_SAMPLING:-coverage}"
TARGET_PAPER_COVERAGE="${TARGET_PAPER_COVERAGE:-0.25}"
COMPARISON_ROUNDS="${COMPARISON_ROUNDS:-1}"
SHUFFLE_PAIRS="${SHUFFLE_PAIRS:-0}"
SEED="${SEED:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
DIRECT_SCORE="${DIRECT_SCORE:-0}"
BIDIRECTIONAL_PAIRWISE="${BIDIRECTIONAL_PAIRWISE:-1}"
BIDIRECTIONAL_CONFIDENCE_MARGIN="${BIDIRECTIONAL_CONFIDENCE_MARGIN:-0.05}"
PAIRWISE_AGGREGATION="${PAIRWISE_AGGREGATION:-rank_centrality}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-configs/calibration/sciimpact-qwen32b-heldout-p50.json}"
MODEL_CREDIBILITY_REPORT="${MODEL_CREDIBILITY_REPORT:-configs/model_credibility/local-impact-judge-source-task-dpo.json}"
FUSION_WEIGHTS="${FUSION_WEIGHTS:-}"
PAIRWISE_JUDGMENTS="${PAIRWISE_JUDGMENTS:-}"
RESUME_PAIRWISE_JUDGMENTS="${RESUME_PAIRWISE_JUDGMENTS:-}"
PAIRWISE_JOURNAL="${PAIRWISE_JOURNAL:-${OUTPUT}/model-impact/pairwise_journal.jsonl}"
VALIDITY_LABELS="${VALIDITY_LABELS:-}"
VALIDITY_LABEL_COLUMN="${VALIDITY_LABEL_COLUMN:-future_impact}"
VALIDITY_BOOTSTRAP_SAMPLES="${VALIDITY_BOOTSTRAP_SAMPLES:-200}"
VALIDITY_PERMUTATION_SAMPLES="${VALIDITY_PERMUTATION_SAMPLES:-200}"
PAIRWISE_STABILITY_SAMPLES="${PAIRWISE_STABILITY_SAMPLES:-80}"
PAIRWISE_STABILITY_TOP_N="${PAIRWISE_STABILITY_TOP_N:-20}"
PAIRWISE_STABILITY_SEED="${PAIRWISE_STABILITY_SEED:-17}"
FINAL_SCORE_STABILITY_SAMPLES="${FINAL_SCORE_STABILITY_SAMPLES:-80}"
FINAL_SCORE_STABILITY_TOP_N="${FINAL_SCORE_STABILITY_TOP_N:-20}"
FINAL_SCORE_STABILITY_SEED="${FINAL_SCORE_STABILITY_SEED:-23}"

case "${RUN_PRESET}" in
  "custom" | "")
    ;;
  "quick")
    if [[ -z "${USER_SET_MAX_PAIRS}" ]]; then
      MAX_PAIRS="50"
    fi
    if [[ -z "${USER_SET_TARGET_PAPER_COVERAGE}" ]]; then
      TARGET_PAPER_COVERAGE="0.10"
    fi
    if [[ -z "${USER_SET_CONCURRENCY}" ]]; then
      CONCURRENCY="1"
    fi
    ;;
  "balanced")
    if [[ -z "${USER_SET_MAX_PAIRS}" ]]; then
      MAX_PAIRS="200"
    fi
    if [[ -z "${USER_SET_TARGET_PAPER_COVERAGE}" ]]; then
      TARGET_PAPER_COVERAGE="0.25"
    fi
    if [[ -z "${USER_SET_CONCURRENCY}" ]]; then
      CONCURRENCY="2"
    fi
    ;;
  "full")
    if [[ -z "${USER_SET_MAX_PAIRS}" ]]; then
      MAX_PAIRS=""
    fi
    if [[ -z "${USER_SET_TARGET_PAPER_COVERAGE}" ]]; then
      TARGET_PAPER_COVERAGE="1.0"
    fi
    if [[ -z "${USER_SET_CONCURRENCY}" ]]; then
      CONCURRENCY="4"
    fi
    ;;
  *)
    echo "Unknown RUN_PRESET: ${RUN_PRESET}. Expected quick, balanced, full, or custom." >&2
    exit 2
    ;;
esac

if [[ -n "${PAIRWISE_JUDGMENTS}" && -n "${RESUME_PAIRWISE_JUDGMENTS}" ]]; then
  echo "PAIRWISE_JUDGMENTS and RESUME_PAIRWISE_JUDGMENTS are mutually exclusive." >&2
  exit 2
fi

mkdir -p "${OUTPUT}"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli validate \
  --input "${INPUT}" \
  --output "${OUTPUT}/validation"

preflight_args=(
  --input "${INPUT}"
  --output "${OUTPUT}/preflight"
  --pair-budget "${PAIR_BUDGET}"
  --target-paper-coverage "${TARGET_PAPER_COVERAGE}"
  --comparison-rounds "${COMPARISON_ROUNDS}"
  --pair-sampling "${PAIR_SAMPLING}"
  --seed "${SEED}"
)

if [[ -n "${MAX_PAIRS}" ]]; then
  preflight_args+=(--max-pairs "${MAX_PAIRS}")
fi
if [[ "${SHUFFLE_PAIRS}" == "1" ]]; then
  preflight_args+=(--shuffle-pairs)
else
  preflight_args+=(--no-shuffle-pairs)
fi
if [[ "${DIRECT_SCORE}" == "1" ]]; then
  preflight_args+=(--direct-score)
else
  preflight_args+=(--no-direct-score)
fi
if [[ "${BIDIRECTIONAL_PAIRWISE}" == "1" ]]; then
  preflight_args+=(--bidirectional-pairwise)
else
  preflight_args+=(--single-direction-pairwise)
fi
if [[ -n "${PAIRWISE_JUDGMENTS}" ]]; then
  preflight_args+=(--resume-pairwise-judgments "${PAIRWISE_JUDGMENTS}")
elif [[ -n "${RESUME_PAIRWISE_JUDGMENTS}" ]]; then
  preflight_args+=(--resume-pairwise-judgments "${RESUME_PAIRWISE_JUDGMENTS}")
fi

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-preflight "${preflight_args[@]}"

if [[ -z "${PAIRWISE_JUDGMENTS}" || "${DIRECT_SCORE}" == "1" ]]; then
  scripts/check_model_server.sh "${BASE_URL}"
fi

score_args=(
  --input "${INPUT}"
  --output "${OUTPUT}/model-impact"
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --api-key "${API_KEY}"
  --timeout "${TIMEOUT}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --pair-budget "${PAIR_BUDGET}"
  --target-paper-coverage "${TARGET_PAPER_COVERAGE}"
  --comparison-rounds "${COMPARISON_ROUNDS}"
  --pair-sampling "${PAIR_SAMPLING}"
  --seed "${SEED}"
  --concurrency "${CONCURRENCY}"
  --bidirectional-confidence-margin "${BIDIRECTIONAL_CONFIDENCE_MARGIN}"
  --pairwise-aggregation "${PAIRWISE_AGGREGATION}"
  --pairwise-stability-samples "${PAIRWISE_STABILITY_SAMPLES}"
  --pairwise-stability-top-n "${PAIRWISE_STABILITY_TOP_N}"
  --pairwise-stability-seed "${PAIRWISE_STABILITY_SEED}"
  --final-score-stability-samples "${FINAL_SCORE_STABILITY_SAMPLES}"
  --final-score-stability-top-n "${FINAL_SCORE_STABILITY_TOP_N}"
  --final-score-stability-seed "${FINAL_SCORE_STABILITY_SEED}"
  --report-bundle
  --report-bundle-output "${OUTPUT}/model-impact/report_bundle"
)

if [[ -n "${MAX_PAIRS}" ]]; then
  score_args+=(--max-pairs "${MAX_PAIRS}")
fi
if [[ "${SHUFFLE_PAIRS}" == "1" ]]; then
  score_args+=(--shuffle-pairs)
else
  score_args+=(--no-shuffle-pairs)
fi
if [[ "${DIRECT_SCORE}" == "1" ]]; then
  score_args+=(--direct-score)
else
  score_args+=(--no-direct-score)
fi
if [[ "${BIDIRECTIONAL_PAIRWISE}" == "1" ]]; then
  score_args+=(--bidirectional-pairwise)
else
  score_args+=(--single-direction-pairwise)
fi
if [[ -n "${CALIBRATION_REPORT}" ]]; then
  score_args+=(--calibration-report "${CALIBRATION_REPORT}")
fi
if [[ -n "${FUSION_WEIGHTS}" ]]; then
  score_args+=(--fusion-weights "${FUSION_WEIGHTS}")
fi
if [[ -n "${PAIRWISE_JUDGMENTS}" ]]; then
  score_args+=(--pairwise-judgments "${PAIRWISE_JUDGMENTS}")
elif [[ -n "${RESUME_PAIRWISE_JUDGMENTS}" ]]; then
  score_args+=(--resume-pairwise-judgments "${RESUME_PAIRWISE_JUDGMENTS}")
  score_args+=(--pairwise-journal "${PAIRWISE_JOURNAL}")
else
  score_args+=(--pairwise-journal "${PAIRWISE_JOURNAL}")
fi
if [[ -n "${VALIDITY_LABELS}" ]]; then
  score_args+=(--validity-labels "${VALIDITY_LABELS}")
  score_args+=(--validity-label-column "${VALIDITY_LABEL_COLUMN}")
  score_args+=(--validity-bootstrap-samples "${VALIDITY_BOOTSTRAP_SAMPLES}")
  score_args+=(--validity-permutation-samples "${VALIDITY_PERMUTATION_SAMPLES}")
fi

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli model-score-dataset "${score_args[@]}"

test -s "${OUTPUT}/validation/validation.json"
test -s "${OUTPUT}/validation/validation.md"
test -s "${OUTPUT}/preflight/model_score_preflight.json"
test -s "${OUTPUT}/preflight/model_score_preflight.md"
test -s "${OUTPUT}/model-impact/impact_scores.csv"
test -s "${OUTPUT}/model-impact/impact_scores.jsonl"
test -s "${OUTPUT}/model-impact/model_impact_report.md"
test -s "${OUTPUT}/model-impact/external_evidence_readiness.md"
test -s "${OUTPUT}/model-impact/external_evidence_paper_coverage.csv"
test -s "${OUTPUT}/model-impact/pairwise_stability.md"
test -s "${OUTPUT}/model-impact/pairwise_stability.csv"
test -s "${OUTPUT}/model-impact/pairwise_stability.json"
test -s "${OUTPUT}/model-impact/final_score_stability.md"
test -s "${OUTPUT}/model-impact/final_score_stability.csv"
test -s "${OUTPUT}/model-impact/final_score_stability.json"
test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"
test -s "${OUTPUT}/model-impact/impact_score_cards.md"
test -s "${OUTPUT}/model-impact/report_bundle/bundle_manifest.json"
test -s "${OUTPUT}/model-impact/report_bundle/README.md"
grep -q "validated_impact_claim.status" "${OUTPUT}/model-impact/model_impact_report.md"
grep -q "final_impact_score" "${OUTPUT}/model-impact/impact_scores.csv"
grep -q "Impact Claim Certificate" "${OUTPUT}/model-impact/impact_claim_certificate.md"
grep -q "Model Score Dataset" "${OUTPUT}/model-impact/report_bundle/README.md"

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" - <<'PY' "${OUTPUT}" "${INPUT}" "${BASE_URL}" "${MODEL}" "${RUN_PRESET}" "${MODEL_CREDIBILITY_REPORT}" "${CALIBRATION_REPORT}"
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from local_impact_ranking.calibration_profile import (
    report_decision_with_calibration_profile,
    write_calibration_profile_card,
)
from local_impact_ranking.model_credibility_profile import (
    report_decision_with_model_credibility,
    write_model_credibility_card,
)

root = Path(sys.argv[1])
impact_dir = root / "model-impact"
scores_path = impact_dir / "impact_scores.csv"
report = json.loads((impact_dir / "model_impact_report.json").read_text(encoding="utf-8"))
certificate = json.loads((impact_dir / "impact_claim_certificate.json").read_text(encoding="utf-8"))
preflight = json.loads((root / "preflight" / "model_score_preflight.json").read_text(encoding="utf-8"))
preflight_pair_plan_quality = preflight.get("pair_plan_quality", {})
validation = json.loads((root / "validation" / "validation.json").read_text(encoding="utf-8"))
external_readiness = json.loads(
    (impact_dir / "external_evidence_readiness.json").read_text(encoding="utf-8")
)
summary = report.get("summary", {})
pairwise_stability = summary.get("pairwise_stability", {})
final_score_stability = summary.get("final_score_stability", {})
fusion_summary = summary.get("fusion", {})
evidence_first_summary = (
    fusion_summary.get("evidence_first", {})
    if isinstance(fusion_summary, dict)
    else {}
)
evidence_signal_audit = (
    evidence_first_summary.get("evidence_signal_audit", {})
    if isinstance(evidence_first_summary, dict)
    else {}
)
claim = certificate.get("claim", {})
validation_summary = validation.get("summary", {})
validation_severity_counts = validation_summary.get("severity_counts", {})
validation_issues = validation.get("issues", [])
scores = list(csv.DictReader(scores_path.open(encoding="utf-8")))
source_quality_counts = Counter(row.get("source_quality_flag") or "unknown" for row in scores)
reportability_counts = Counter(row.get("reportability_status") or "unknown" for row in scores)
top_missing_evidence = (
    external_readiness.get("paper_coverage_summary", {}).get("top_missing_evidence", [])
)
internal_graph = external_readiness.get("graph_evidence", {})
recommended_next_actions = certificate.get("recommended_next_actions", [])
model_credibility_path = Path(sys.argv[6]) if sys.argv[6] else None
calibration_report_path = Path(sys.argv[7]) if sys.argv[7] else None
model_credibility_card = write_model_credibility_card(
    model_credibility_path,
    impact_dir,
    served_model=sys.argv[4],
)
calibration_profile_card = write_calibration_profile_card(
    calibration_report_path,
    impact_dir,
    served_model=sys.argv[4],
)


def cell(value):
    return str(value if value is not None else "").replace("|", "\\|")


def validation_disclosure_suffix(validation_summary):
    severity_counts = validation_summary.get("severity_counts", {})
    error_count = int(severity_counts.get("error", 0) or 0)
    warning_count = int(severity_counts.get("warning", 0) or 0)
    if error_count:
        return (
            f" Input validation has {error_count} error(s) and {warning_count} warning(s); "
            "fix errors before external reporting."
        )
    if warning_count:
        return (
            f" Input validation has {warning_count} warning(s); disclose coverage limits before external reporting."
        )
    return ""


def final_score_stability_blocks_reporting(summary):
    status = (
        (summary.get("final_score_stability") or {}).get("status") or ""
    ).lower()
    return status in {"unstable", "insufficient_samples"}


def final_score_stability_disclosure_suffix(summary):
    stability = summary.get("final_score_stability") or {}
    status = (stability.get("status") or "").lower()
    reason = stability.get("reason") or ""
    if status == "unstable":
        return (
            " Final-score stability is unstable; keep scores exploratory until "
            f"pairwise coverage improves. {reason}"
        )
    if status == "insufficient_samples":
        return (
            " Final-score stability has insufficient samples; rerun stability "
            f"auditing before reporting ranked scores. {reason}"
        )
    if status == "moderate":
        return (
            " Final-score stability is moderate; disclose bootstrap movement "
            "when reporting dataset-local scores."
        )
    return ""


def no_reportable_scores_block_reporting(scores):
    if not scores:
        return True
    return not any(row.get("reportability_status") == "reportable" for row in scores)


def report_decision(summary, claim, validation_summary, scores):
    ranking_status = (summary.get("ranking_quality") or {}).get("status")
    claim_status = claim.get("status")
    validation_suffix = validation_disclosure_suffix(validation_summary)
    stability_suffix = final_score_stability_disclosure_suffix(summary)
    if not claim.get("can_report_dataset_local_score", False) or claim_status == "blocked":
        return {
            "status": "blocked",
            "reason": "The claim certificate blocks reporting dataset-local scores.",
        }
    if final_score_stability_blocks_reporting(summary):
        return {
            "status": "exploratory_only",
            "reason": (
                "Use the scores for exploration until final-score stability improves."
                + stability_suffix
                + validation_suffix
            ),
        }
    if no_reportable_scores_block_reporting(scores):
        return {
            "status": "exploratory_only",
            "reason": (
                "No papers are marked reportable; use the score table for exploration "
                "until evidence coverage or reportability improves."
                + stability_suffix
                + validation_suffix
            ),
        }
    if claim.get("can_claim_validated_impact_accuracy") and claim.get(
        "can_claim_high_credibility"
    ):
        return {
            "status": "validated_report",
            "reason": (
                "Validated impact accuracy and high-credibility evidence are both supported."
                + stability_suffix
                + validation_suffix
            ),
        }
    if ranking_status == "reportable" and claim_status in {
        "evidence_supported",
        "validated",
    }:
        return {
            "status": "report_with_disclosure",
            "reason": (
                "Dataset-local scores can be reported with the claim certificate and evidence-gap disclosures attached."
                + stability_suffix
                + validation_suffix
            ),
        }
    return {
        "status": "exploratory_only",
        "reason": (
            "Use the scores for exploration until ranking quality, validation labels, or external evidence improve."
            + stability_suffix
            + validation_suffix
        ),
    }


decision = report_decision_with_calibration_profile(
    report_decision_with_model_credibility(
        report_decision(summary, claim, validation_summary, scores),
        model_credibility_card,
    ),
    calibration_profile_card,
)
model_credibility_summary = model_credibility_card.get("summary", {})
model_credibility_scorecard = model_credibility_card.get("decision", {})
model_credibility_weak_slices = model_credibility_card.get("weak_slices", [])
calibration_summary = calibration_profile_card.get("summary", {})
calibration_decision = calibration_profile_card.get("decision", {})
status = {
    "status": "passed",
    "input": sys.argv[2],
    "output": str(root),
    "base_url": sys.argv[3],
    "model": sys.argv[4],
    "run_preset": sys.argv[5],
    "scores": str(impact_dir / "impact_scores.csv"),
    "model_impact_report": str(impact_dir / "model_impact_report.md"),
    "claim_certificate": str(impact_dir / "impact_claim_certificate.md"),
    "summary_report": str(root / "user_impact_summary.md"),
    "report_bundle": str(impact_dir / "report_bundle"),
    "model_credibility_card": str(impact_dir / "model_credibility_card.md"),
    "calibration_profile_card": str(impact_dir / "calibration_profile_card.md"),
    "model_credibility_report": str(model_credibility_path)
    if model_credibility_path is not None
    else None,
    "calibration_report": str(calibration_report_path)
    if calibration_report_path is not None
    else None,
    "model_credibility_served_model": model_credibility_card.get("served_model"),
    "model_credibility_status": model_credibility_scorecard.get(
        "status",
        "not_provided",
    ),
    "model_credibility_grade": model_credibility_scorecard.get("grade"),
    "model_credibility_model_identity_match": model_credibility_scorecard.get(
        "model_identity_match"
    ),
    "model_credibility_profile_model_names": model_credibility_scorecard.get(
        "profile_model_names",
        [],
    ),
    "model_credibility_pooled_accuracy": model_credibility_summary.get("pooled_accuracy"),
    "model_credibility_min_seed_accuracy": model_credibility_summary.get(
        "min_seed_accuracy"
    ),
    "model_credibility_evaluation_count": model_credibility_summary.get(
        "evaluation_count"
    ),
    "model_credibility_total_examples": model_credibility_summary.get("total_examples"),
    "model_credibility_significant_vs_random": model_credibility_scorecard.get(
        "significant_vs_random"
    ),
    "model_credibility_can_use_for_report_with_disclosure": model_credibility_scorecard.get(
        "can_use_for_report_with_disclosure"
    ),
    "model_credibility_can_claim_high_model_credibility": model_credibility_scorecard.get(
        "can_claim_high_model_credibility"
    ),
    "model_credibility_standalone_model_claim_allowed": model_credibility_scorecard.get(
        "standalone_model_claim_allowed"
    ),
    "model_credibility_weak_slice_count": len(model_credibility_weak_slices),
    "model_credibility_weak_slices": [
        row.get("slice") for row in model_credibility_weak_slices[:10]
    ],
    "calibration_status": calibration_decision.get("status", "not_provided"),
    "calibration_can_use_calibrated_confidence": calibration_decision.get(
        "can_use_calibrated_confidence"
    ),
    "calibration_model_identity_match": calibration_decision.get(
        "model_identity_match"
    ),
    "calibration_compatible_models": calibration_decision.get(
        "compatible_models",
        [],
    ),
    "calibration_scored_examples": calibration_summary.get("scored_examples"),
    "calibration_ece": calibration_summary.get("ece"),
    "calibration_mce": calibration_summary.get("mce"),
    "calibration_brier_score": calibration_summary.get("brier_score"),
    "paper_count": summary.get("paper_count"),
    "validation_issue_count": validation_summary.get("issue_count", 0),
    "validation_error_count": validation_severity_counts.get("error", 0),
    "validation_warning_count": validation_severity_counts.get("warning", 0),
    "validation_can_run": validation_summary.get("can_run"),
    "judgment_count": summary.get("judgment_count"),
    "preflight_total_model_request_count": preflight.get("request_estimate", {}).get(
        "total_model_request_count"
    ),
    "preflight_pair_plan_quality_status": preflight_pair_plan_quality.get("status"),
    "preflight_pair_plan_connected_component_count": preflight_pair_plan_quality.get(
        "connected_component_count"
    ),
    "preflight_pair_plan_largest_connected_component_coverage": preflight_pair_plan_quality.get(
        "largest_connected_component_coverage"
    ),
    "preflight_pair_plan_min_comparisons_per_paper": preflight_pair_plan_quality.get(
        "min_planned_comparisons_per_paper"
    ),
    "pairwise_judgment_source": summary.get("pairwise_judgment_source"),
    "ranking_quality_status": summary.get("ranking_quality", {}).get("status"),
    "pairwise_stability_status": pairwise_stability.get("status"),
    "pairwise_stability_samples": pairwise_stability.get("samples"),
    "pairwise_stability_mean_top_n_retention_rate": pairwise_stability.get(
        "mean_top_n_retention_rate"
    ),
    "pairwise_stability_unstable_paper_count": pairwise_stability.get(
        "unstable_paper_count"
    ),
    "final_score_stability_status": final_score_stability.get("status"),
    "final_score_stability_samples": final_score_stability.get("samples"),
    "final_score_stability_mean_top_n_retention_rate": final_score_stability.get(
        "mean_top_n_retention_rate"
    ),
    "final_score_stability_unstable_paper_count": final_score_stability.get(
        "unstable_paper_count"
    ),
    "claim_status": claim.get("status"),
    "report_decision": decision.get("status"),
    "report_decision_reason": decision.get("reason"),
    "can_report_dataset_local_score": claim.get("can_report_dataset_local_score"),
    "can_claim_validated_impact_accuracy": claim.get("can_claim_validated_impact_accuracy"),
    "can_claim_high_credibility": claim.get("can_claim_high_credibility"),
    "source_quality_counts": dict(source_quality_counts),
    "reportable_paper_count": reportability_counts.get("reportable", 0),
    "exploratory_paper_count": reportability_counts.get("exploratory", 0),
    "low_confidence_paper_count": reportability_counts.get("low_confidence", 0),
    "excluded_non_article_count": reportability_counts.get("excluded_non_article", 0),
    "evidence_signal_audit": evidence_signal_audit,
    "claim_boundary": "dataset-local paper influence; not global scientific impact without external calibrated evidence",
}
(root / "user_impact_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

lines = [
    "# User Impact Run Summary",
    "",
    "## Claim Boundary",
    "",
    "- Scope: dataset-local paper influence inside the supplied corpus.",
    "- Not claimed: global scientific impact or validated impact accuracy without external calibrated evidence and validity labels.",
    "",
    "## Run Status",
    "",
    "| Item | Value |",
    "| --- | --- |",
    f"| status | {cell(status.get('status'))} |",
    f"| input | `{cell(status.get('input'))}` |",
    f"| model | {cell(status.get('model'))} |",
    f"| run_preset | {cell(status.get('run_preset'))} |",
    f"| paper_count | {cell(status.get('paper_count'))} |",
    f"| validation_issue_count | {cell(status.get('validation_issue_count'))} |",
    f"| validation_error_count | {cell(status.get('validation_error_count'))} |",
    f"| validation_warning_count | {cell(status.get('validation_warning_count'))} |",
    f"| validation_can_run | {cell(status.get('validation_can_run'))} |",
    f"| judgment_count | {cell(status.get('judgment_count'))} |",
    f"| preflight_total_model_request_count | {cell(status.get('preflight_total_model_request_count'))} |",
    f"| preflight_pair_plan_quality_status | {cell(status.get('preflight_pair_plan_quality_status'))} |",
    f"| preflight_pair_plan_connected_component_count | {cell(status.get('preflight_pair_plan_connected_component_count'))} |",
    f"| preflight_pair_plan_largest_connected_component_coverage | {cell(status.get('preflight_pair_plan_largest_connected_component_coverage'))} |",
    f"| preflight_pair_plan_min_comparisons_per_paper | {cell(status.get('preflight_pair_plan_min_comparisons_per_paper'))} |",
    f"| pairwise_judgment_source | {cell((status.get('pairwise_judgment_source') or {}).get('type'))} |",
    f"| ranking_quality_status | {cell(status.get('ranking_quality_status'))} |",
    f"| pairwise_stability_status | {cell(status.get('pairwise_stability_status'))} |",
    f"| pairwise_stability_samples | {cell(status.get('pairwise_stability_samples'))} |",
    f"| pairwise_stability_mean_top_n_retention_rate | {cell(status.get('pairwise_stability_mean_top_n_retention_rate'))} |",
    f"| pairwise_stability_unstable_paper_count | {cell(status.get('pairwise_stability_unstable_paper_count'))} |",
    f"| final_score_stability_status | {cell(status.get('final_score_stability_status'))} |",
    f"| final_score_stability_samples | {cell(status.get('final_score_stability_samples'))} |",
    f"| final_score_stability_mean_top_n_retention_rate | {cell(status.get('final_score_stability_mean_top_n_retention_rate'))} |",
    f"| final_score_stability_unstable_paper_count | {cell(status.get('final_score_stability_unstable_paper_count'))} |",
    f"| claim_status | {cell(status.get('claim_status'))} |",
    f"| report_decision | {cell(status.get('report_decision'))} |",
    f"| report_decision_reason | {cell(status.get('report_decision_reason'))} |",
    f"| can_report_dataset_local_score | {cell(status.get('can_report_dataset_local_score'))} |",
    f"| can_claim_validated_impact_accuracy | {cell(status.get('can_claim_validated_impact_accuracy'))} |",
    f"| can_claim_high_credibility | {cell(status.get('can_claim_high_credibility'))} |",
    f"| reportable_paper_count | {cell(status.get('reportable_paper_count'))} |",
    f"| exploratory_paper_count | {cell(status.get('exploratory_paper_count'))} |",
    f"| low_confidence_paper_count | {cell(status.get('low_confidence_paper_count'))} |",
    f"| excluded_non_article_count | {cell(status.get('excluded_non_article_count'))} |",
    "",
    "## Model Credibility",
    "",
    "| Item | Value |",
    "| --- | --- |",
    f"| model_credibility_report | `{cell(status.get('model_credibility_report'))}` |",
    f"| model_credibility_card | `{cell(status.get('model_credibility_card'))}` |",
    f"| served_model | {cell(status.get('model_credibility_served_model'))} |",
    f"| profile_model_names | {cell(', '.join(status.get('model_credibility_profile_model_names') or []))} |",
    f"| model_identity_match | {cell(status.get('model_credibility_model_identity_match'))} |",
    f"| model_credibility_status | {cell(status.get('model_credibility_status'))} |",
    f"| model_credibility_grade | {cell(status.get('model_credibility_grade'))} |",
    f"| pooled_accuracy | {cell(status.get('model_credibility_pooled_accuracy'))} |",
    f"| min_seed_accuracy | {cell(status.get('model_credibility_min_seed_accuracy'))} |",
    f"| evaluation_count | {cell(status.get('model_credibility_evaluation_count'))} |",
    f"| total_examples | {cell(status.get('model_credibility_total_examples'))} |",
    f"| significant_vs_random | {cell(status.get('model_credibility_significant_vs_random'))} |",
    f"| can_use_for_report_with_disclosure | {cell(status.get('model_credibility_can_use_for_report_with_disclosure'))} |",
    f"| can_claim_high_model_credibility | {cell(status.get('model_credibility_can_claim_high_model_credibility'))} |",
    f"| standalone_model_claim_allowed | {cell(status.get('model_credibility_standalone_model_claim_allowed'))} |",
    f"| weak_slice_count | {cell(status.get('model_credibility_weak_slice_count'))} |",
    "",
    "## Pair Plan Quality",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
    f"| status | {cell(preflight_pair_plan_quality.get('status'))} |",
    f"| reason | {cell(preflight_pair_plan_quality.get('reason'))} |",
    f"| minimum_connected_pair_count | {cell(preflight_pair_plan_quality.get('minimum_connected_pair_count'))} |",
    f"| compared_paper_count | {cell(preflight_pair_plan_quality.get('compared_paper_count'))} |",
    f"| paper_coverage_rate | {cell(preflight_pair_plan_quality.get('paper_coverage_rate'))} |",
    f"| connected_component_count | {cell(preflight_pair_plan_quality.get('connected_component_count'))} |",
    f"| largest_connected_component_coverage | {cell(preflight_pair_plan_quality.get('largest_connected_component_coverage'))} |",
    f"| isolated_paper_count | {cell(preflight_pair_plan_quality.get('isolated_paper_count'))} |",
    f"| min_planned_comparisons_per_paper | {cell(preflight_pair_plan_quality.get('min_planned_comparisons_per_paper'))} |",
    f"| minimum_recommended_comparisons_per_paper | {cell(preflight_pair_plan_quality.get('minimum_recommended_comparisons_per_paper'))} |",
    "",
    "## Pairwise Stability",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
    f"| status | {cell(pairwise_stability.get('status'))} |",
    f"| reason | {cell(pairwise_stability.get('reason'))} |",
    f"| samples | {cell(pairwise_stability.get('samples'))} |",
    f"| top_n | {cell(pairwise_stability.get('top_n'))} |",
    f"| decisive_judgment_count | {cell(pairwise_stability.get('decisive_judgment_count'))} |",
    f"| mean_top_n_retention_rate | {cell(pairwise_stability.get('mean_top_n_retention_rate'))} |",
    f"| stable_paper_count | {cell(pairwise_stability.get('stable_paper_count'))} |",
    f"| unstable_paper_count | {cell(pairwise_stability.get('unstable_paper_count'))} |",
    "",
    "## Final Score Stability",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
    f"| status | {cell(final_score_stability.get('status'))} |",
    f"| reason | {cell(final_score_stability.get('reason'))} |",
    f"| samples | {cell(final_score_stability.get('samples'))} |",
    f"| top_n | {cell(final_score_stability.get('top_n'))} |",
    f"| decisive_judgment_count | {cell(final_score_stability.get('decisive_judgment_count'))} |",
    f"| mean_top_n_retention_rate | {cell(final_score_stability.get('mean_top_n_retention_rate'))} |",
    f"| stable_paper_count | {cell(final_score_stability.get('stable_paper_count'))} |",
    f"| unstable_paper_count | {cell(final_score_stability.get('unstable_paper_count'))} |",
    "",
    "## Calibration Credibility",
    "",
    "| Item | Value |",
    "| --- | --- |",
    f"| calibration_report | `{cell(status.get('calibration_report'))}` |",
    f"| calibration_profile_card | `{cell(status.get('calibration_profile_card'))}` |",
    f"| calibration_status | {cell(status.get('calibration_status'))} |",
    f"| calibration_model_identity_match | {cell(status.get('calibration_model_identity_match'))} |",
    f"| calibration_compatible_models | {cell(', '.join(status.get('calibration_compatible_models') or []))} |",
    f"| can_use_calibrated_confidence | {cell(status.get('calibration_can_use_calibrated_confidence'))} |",
    f"| scored_examples | {cell(status.get('calibration_scored_examples'))} |",
    f"| ece | {cell(status.get('calibration_ece'))} |",
    f"| mce | {cell(status.get('calibration_mce'))} |",
    f"| brier_score | {cell(status.get('calibration_brier_score'))} |",
    "",
    "### Model Weak Slices",
    "",
    "| weak_slices | Pooled Accuracy | Min Seed Accuracy | Covered Examples |",
    "| --- | ---: | ---: | ---: |",
]
for row in model_credibility_weak_slices[:10]:
    lines.append(
        "| "
        f"{cell(row.get('slice'))} | "
        f"{cell(row.get('pooled_accuracy'))} | "
        f"{cell(row.get('min_seed_accuracy'))} | "
        f"{cell(row.get('covered_examples'))} |"
    )
if not model_credibility_weak_slices:
    lines.append("| none_reported |  |  |  |")
lines.extend(
    [
        "",
        "## Validation Issues",
        "",
        "| Severity | Code | Paper ID | Message |",
        "| --- | --- | --- | --- |",
    ]
)
for issue in validation_issues[:10]:
    lines.append(
        "| "
        f"{cell(issue.get('severity'))} | "
        f"{cell(issue.get('code'))} | "
        f"{cell(issue.get('paper_id'))} | "
        f"{cell(issue.get('message'))} |"
    )
if not validation_issues:
    lines.append("| none_reported |  |  |  |")
lines.extend(
    [
        "",
        "## Source Quality",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| article | {source_quality_counts.get('article', 0)} |",
        f"| non_article | {source_quality_counts.get('non_article', 0)} |",
        f"| unknown | {source_quality_counts.get('unknown', 0)} |",
        f"| excluded_non_article | {reportability_counts.get('excluded_non_article', 0)} |",
        "",
        "Records marked `excluded_non_article` are kept in the audit table but receive a final impact score of 0.",
    ]
)
lines.extend(
    [
        "",
        "## Top Impact Scores",
        "",
        "| Rank | Paper ID | Final Impact Score | Evidence Tier | Positive Evidence Signals | Zero Evidence Signals | Reportability |",
        "| ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
)
for row in scores[:10]:
    lines.append(
        "| "
        f"{cell(row.get('rank'))} | "
        f"{cell(row.get('paper_id'))} | "
        f"{float(row.get('final_impact_score') or 0.0):.3f} | "
        f"{cell(row.get('credible_evidence_tier'))} | "
        f"{cell(row.get('positive_evidence_signal_count'))} | "
        f"{cell(row.get('zero_evidence_signal_count'))} | "
        f"{cell(row.get('reportability_status'))} |"
    )
lines.extend(
    [
        "",
        "## Evidence Signal Audit",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean_positive_evidence_signal_count | {cell(evidence_signal_audit.get('mean_positive_evidence_signal_count', 0.0))} |",
        f"| mean_zero_evidence_signal_count | {cell(evidence_signal_audit.get('mean_zero_evidence_signal_count', 0.0))} |",
        f"| papers_with_no_positive_evidence | {cell(evidence_signal_audit.get('papers_with_no_positive_evidence', 0))} |",
        f"| papers_with_zero_only_evidence | {cell(evidence_signal_audit.get('papers_with_zero_only_evidence', 0))} |",
        "",
        "| Evidence count type | Count | Paper count |",
        "| --- | ---: | ---: |",
    ]
)
for count, paper_count in (
    evidence_signal_audit.get("positive_signal_count_distribution", {}) or {}
).items():
    lines.append(f"| positive | {cell(count)} | {cell(paper_count)} |")
for count, paper_count in (
    evidence_signal_audit.get("zero_signal_count_distribution", {}) or {}
).items():
    lines.append(f"| zero | {cell(count)} | {cell(paper_count)} |")
for count, paper_count in (
    evidence_signal_audit.get("available_signal_count_distribution", {}) or {}
).items():
    lines.append(f"| available | {cell(count)} | {cell(paper_count)} |")
if evidence_signal_audit.get("interpretation"):
    lines.extend(["", cell(evidence_signal_audit.get("interpretation"))])
lines.extend(
    [
        "",
        "## Evidence Gaps",
        "",
        "| Missing Evidence | Paper Count | Paper Coverage |",
        "| --- | ---: | ---: |",
    ]
)
for item in top_missing_evidence[:10]:
    lines.append(
        "| "
        f"{cell(item.get('label'))} | "
        f"{cell(item.get('paper_count'))} | "
        f"{cell(item.get('paper_coverage'))} |"
    )
if not top_missing_evidence:
    lines.append("| none_reported | 0 | 0 |")
lines.extend(
    [
        "",
        "## Internal Citation Graph Audit",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| internal_reference_edge_count | {cell(internal_graph.get('internal_reference_edge_count', 0))} |",
        f"| materialized_internal_edge_count | {cell(internal_graph.get('materialized_internal_edge_count', 0))} |",
        f"| internal_citation_audit_coverage | {cell(internal_graph.get('internal_citation_audit_coverage', 0.0))} |",
        f"| openalex_reference_audit_coverage | {cell(internal_graph.get('openalex_reference_audit_coverage', 0.0))} |",
    ]
)
lines.extend(["", "## Recommended Next Actions", ""])
if recommended_next_actions:
    lines.extend(f"- {cell(action)}" for action in recommended_next_actions)
else:
    lines.append("- Review the claim certificate before reporting the score.")
lines.extend(
    [
        "",
        "## Key Artifacts",
        "",
        f"- Scores: `{impact_dir / 'impact_scores.csv'}`",
        f"- Model impact report: `{impact_dir / 'model_impact_report.md'}`",
        f"- Claim certificate: `{impact_dir / 'impact_claim_certificate.md'}`",
        f"- Model credibility card: `{impact_dir / 'model_credibility_card.md'}`",
        f"- Calibration profile card: `{impact_dir / 'calibration_profile_card.md'}`",
        f"- Score cards: `{impact_dir / 'impact_score_cards.md'}`",
        f"- Final score stability: `{impact_dir / 'final_score_stability.md'}`",
        f"- Report bundle: `{impact_dir / 'report_bundle'}`",
        "",
    ]
)
(root / "user_impact_summary.md").write_text("\n".join(lines), encoding="utf-8")
PY

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m local_impact_ranking.cli report-bundle \
  "${OUTPUT}/model-impact" \
  --output "${OUTPUT}/model-impact/report_bundle"

test -s "${OUTPUT}/user_impact_status.json"
test -s "${OUTPUT}/user_impact_summary.md"
test -s "${OUTPUT}/model-impact/model_credibility_card.json"
test -s "${OUTPUT}/model-impact/model_credibility_card.md"
test -s "${OUTPUT}/model-impact/calibration_profile_card.json"
test -s "${OUTPUT}/model-impact/calibration_profile_card.md"
test -s "${OUTPUT}/model-impact/report_bundle/model_credibility_card.md"
test -s "${OUTPUT}/model-impact/report_bundle/calibration_profile_card.md"
grep -q "User Impact Run Summary" "${OUTPUT}/user_impact_summary.md"
grep -q "Evidence Signal Audit" "${OUTPUT}/user_impact_summary.md"
grep -q "Evidence Gaps" "${OUTPUT}/user_impact_summary.md"
grep -q "Recommended Next Actions" "${OUTPUT}/user_impact_summary.md"
grep -q "report_decision" "${OUTPUT}/user_impact_summary.md"
grep -q "Pair Plan Quality" "${OUTPUT}/user_impact_summary.md"
grep -q "preflight_pair_plan_quality_status" "${OUTPUT}/user_impact_summary.md"
grep -q "Pairwise Stability" "${OUTPUT}/user_impact_summary.md"
grep -q "pairwise_stability_status" "${OUTPUT}/user_impact_summary.md"
grep -q "Final Score Stability" "${OUTPUT}/user_impact_summary.md"
grep -q "final_score_stability_status" "${OUTPUT}/user_impact_summary.md"
grep -q "Model Credibility" "${OUTPUT}/user_impact_summary.md"
grep -q "model_credibility_status" "${OUTPUT}/user_impact_summary.md"
grep -q "Calibration Credibility" "${OUTPUT}/user_impact_summary.md"
grep -q "calibration_status" "${OUTPUT}/user_impact_summary.md"
grep -q "calibration_model_identity_match" "${OUTPUT}/user_impact_summary.md"
grep -q "can_use_for_report_with_disclosure" "${OUTPUT}/user_impact_summary.md"
grep -q "Model Credibility Card" "${OUTPUT}/model-impact/model_credibility_card.md"
grep -q "Calibration Profile Card" "${OUTPUT}/model-impact/calibration_profile_card.md"
grep -q "claim_status" "${OUTPUT}/user_impact_summary.md"
grep -q "ranking_quality_status" "${OUTPUT}/user_impact_summary.md"

echo "User impact run complete: ${OUTPUT}"
