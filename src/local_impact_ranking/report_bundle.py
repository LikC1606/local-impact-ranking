from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


REPORT_BUNDLE_FILES = [
    "rankings/papers.csv",
    "rankings/authors.csv",
    "rankings/institutions.csv",
    "rankings/fields.csv",
    "rankings/years.csv",
    "reports/mvp_summary.md",
    "reports/benchmark_summary.json",
    "reports/benchmark_summary.md",
    "reports/project_readiness.json",
    "reports/project_readiness.md",
    "reports/external_evidence_readiness.json",
    "reports/external_evidence_readiness.md",
    "reports/external_evidence_paper_coverage.csv",
    "reports/citation_completeness_audit.json",
    "reports/citation_completeness_audit.md",
    "reports/citation_completeness_papers.csv",
    "reports/impact_validity_evaluation.json",
    "reports/impact_validity_evaluation.md",
    "reports/impact_credibility_scorecard.json",
    "reports/impact_credibility_scorecard.md",
    "reports/citation_cutoff_audit.json",
    "reports/citation_cutoff_audit.md",
    "reports/impact_claim_certificate.json",
    "reports/impact_claim_certificate.md",
    "impact_claim_certificate.json",
    "impact_claim_certificate.md",
    "model_credibility_card.json",
    "model_credibility_card.md",
    "calibration_profile_card.json",
    "calibration_profile_card.md",
    "impact_scores.csv",
    "impact_scores.jsonl",
    "model_impact_report.json",
    "model_impact_report.md",
    "impact_score_cards.json",
    "impact_score_cards.md",
    "pairwise_judgments.jsonl",
    "direct_scores.csv",
    "direct_scores.jsonl",
    "external_evidence_readiness.json",
    "external_evidence_readiness.md",
    "external_evidence_paper_coverage.csv",
    "citation_completeness_audit.json",
    "citation_completeness_audit.md",
    "citation_completeness_papers.csv",
    "impact_validity/impact_validity_evaluation.json",
    "impact_validity/impact_validity_evaluation.md",
    "impact_validity/impact_credibility_scorecard.json",
    "impact_validity/impact_credibility_scorecard.md",
    "reports/evaluation.md",
    "reports/gold_evaluation.json",
    "reports/gold_evaluation.md",
    "reports/gold_baseline_comparison.json",
    "reports/gold_baseline_comparison.md",
    "reports/baseline_comparison.json",
    "reports/paper_evidence_report.csv",
    "reports/score_cards.md",
    "reports/entity_resolution_report.md",
    "reports/paper_stability_report.csv",
    "reports/paper_score_uncertainty_report.csv",
    "reports/scoring_config.json",
    "reports/artifact_index.json",
    "run_manifest.json",
]


def write_report_bundle(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    missing_optional_files: list[str] = []
    for relative_path in REPORT_BUNDLE_FILES:
        source_path = source / relative_path
        destination_path = destination / relative_path
        if not source_path.exists() or not source_path.is_file():
            missing_optional_files.append(relative_path)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied_files.append(relative_path)

    readiness = _read_json(source / "reports" / "project_readiness.json")
    benchmark = _read_json(source / "reports" / "benchmark_summary.json")
    impact_claim = _impact_claim_summary(source)
    model_score = _model_score_summary(source)
    readiness_summary = _readiness_summary(readiness)
    manifest = {
        "run_dir": str(source),
        "claim_scope": readiness.get(
            "claim_scope",
            "dataset-local content-aware impact ranking",
        ),
        "impact_claim": impact_claim,
        "model_credibility": _model_credibility_summary(source),
        "calibration_profile": _calibration_profile_summary(source),
        "readiness": readiness_summary,
        "model_score": model_score,
        "bundle_decision": _display_decision(
            readiness=readiness_summary,
            impact_claim=impact_claim,
            model_score=model_score,
        ),
        "benchmark": _benchmark_summary(benchmark),
        "warnings": readiness.get("warnings", []),
        "copied_files": copied_files,
        "missing_optional_files": missing_optional_files,
        "bundle_files": list(REPORT_BUNDLE_FILES),
    }
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        render_report_bundle_readme(manifest),
        encoding="utf-8",
    )
    return manifest


def render_report_bundle_readme(manifest: dict[str, Any]) -> str:
    readiness = manifest.get("readiness", {})
    benchmark = manifest.get("benchmark", {})
    model_score = manifest.get("model_score", {})
    model_credibility = manifest.get("model_credibility", {})
    calibration_profile = manifest.get("calibration_profile", {})
    decision = manifest.get("bundle_decision") or _display_decision(
        readiness=readiness,
        impact_claim=manifest.get("impact_claim", {}),
        model_score=model_score,
    )
    lines = [
        "# Local Impact Ranking Report Bundle",
        "",
        "This bundle contains report-facing artifacts for a single local impact ranking run.",
        "",
        "## Claim Boundary",
        "",
        f"- Claim scope: {manifest.get('claim_scope', 'dataset-local impact ranking')}",
        "- Scores and rankings are valid only inside the input dataset unless an external calibrated graph is added.",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| readiness_status | {_text(decision.get('status'))} |",
        f"| can_report | {_yes_no(decision.get('can_report'))} |",
        f"| evidence_level | {_text(decision.get('evidence_level'))} |",
        f"| impact_claim_status | {_text(manifest.get('impact_claim', {}).get('status'))} |",
        f"| validated_impact_accuracy | {_yes_no(manifest.get('impact_claim', {}).get('can_claim_validated_impact_accuracy'))} |",
        f"| high_credibility_claim | {_yes_no(manifest.get('impact_claim', {}).get('can_claim_high_credibility'))} |",
        f"| decision_basis | {_text(decision.get('basis'))} |",
        f"| reason | {_text(decision.get('reason'))} |",
        "",
        "## Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {benchmark.get('paper_count', 0)} |",
        f"| hit_rate | {_number(benchmark.get('hit_rate'))} |",
        f"| mean_ndcg | {_number(benchmark.get('mean_ndcg'))} |",
        f"| judge_ndcg_delta_vs_best_baseline | {_number(benchmark.get('judge_ndcg_delta_vs_best_baseline'))} |",
        f"| judge_ndcg_delta_ci95 | {_ci(benchmark.get('judge_ndcg_delta_ci95'))} |",
        f"| statistically_positive_ndcg_delta | {_yes_no(benchmark.get('statistically_positive_ndcg_delta'))} |",
        f"| needs_prompt_or_model_tuning | {_yes_no(benchmark.get('needs_prompt_or_model_tuning'))} |",
        "",
        "## Model Score Dataset",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| model_paper_count | {model_score.get('paper_count', 0)} |",
        f"| judgment_count | {model_score.get('judgment_count', 0)} |",
        f"| parseable_rate | {_number(model_score.get('parseable_rate'))} |",
        f"| model_scoring_method | {_text(model_score.get('scoring_method'))} |",
        f"| ranking_quality_status | {_text(model_score.get('ranking_quality_status'))} |",
        f"| validated_impact_claim_status | {_text(model_score.get('validated_impact_claim_status'))} |",
        f"| impact_credibility_status | {_text(model_score.get('impact_credibility_status'))} |",
        "",
        "## Model Credibility",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| model_credibility_status | {_text(model_credibility.get('status'))} |",
        f"| model_credibility_grade | {_text(model_credibility.get('grade'))} |",
        f"| served_model | {_text(model_credibility.get('served_model'))} |",
        f"| model_identity_match | {_identity_match_text(model_credibility.get('model_identity_match'))} |",
        f"| model_can_use_for_report_with_disclosure | {_yes_no(model_credibility.get('can_use_for_report_with_disclosure'))} |",
        f"| model_can_claim_high_credibility | {_yes_no(model_credibility.get('can_claim_high_model_credibility'))} |",
        f"| pooled_accuracy | {_number(model_credibility.get('pooled_accuracy'))} |",
        f"| min_seed_accuracy | {_number(model_credibility.get('min_seed_accuracy'))} |",
        f"| total_examples | {_text(model_credibility.get('total_examples'))} |",
        "",
        "## Calibration Profile",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| calibration_status | {_text(calibration_profile.get('status'))} |",
        f"| served_model | {_text(calibration_profile.get('served_model'))} |",
        f"| calibration_model_identity_match | {_identity_match_text(calibration_profile.get('model_identity_match'))} |",
        f"| can_use_calibrated_confidence | {_yes_no(calibration_profile.get('can_use_calibrated_confidence'))} |",
        f"| scored_examples | {_text(calibration_profile.get('scored_examples'))} |",
        f"| ece | {_number(calibration_profile.get('ece'))} |",
        f"| mce | {_number(calibration_profile.get('mce'))} |",
        f"| brier_score | {_number(calibration_profile.get('brier_score'))} |",
        "",
        "## Included Files",
        "",
    ]
    copied_files = manifest.get("copied_files", [])
    if copied_files:
        lines.extend(f"- `{path}`" for path in copied_files)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Optional Files", ""])
    missing_files = manifest.get("missing_optional_files", [])
    if missing_files:
        lines.extend(f"- `{path}`" for path in missing_files)
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _readiness_summary(readiness: dict[str, Any]) -> dict[str, Any]:
    decision = readiness.get("decision", {})
    quality = readiness.get("quality", {})
    return {
        "status": decision.get("status"),
        "can_report": decision.get("can_report"),
        "reason": decision.get("reason"),
        "evidence_level": quality.get("evidence_level"),
    }


def _impact_claim_summary(source: Path) -> dict[str, Any]:
    if (source / "model_impact_report.json").exists():
        certificate = _read_json(source / "impact_claim_certificate.json")
        if not certificate:
            certificate = _read_json(source / "reports" / "impact_claim_certificate.json")
    else:
        certificate = _read_json(source / "reports" / "impact_claim_certificate.json")
        if not certificate:
            certificate = _read_json(source / "impact_claim_certificate.json")
    claim = certificate.get("claim", {})
    return {
        "status": claim.get("status"),
        "can_claim_validated_impact_accuracy": claim.get(
            "can_claim_validated_impact_accuracy"
        ),
        "can_claim_high_credibility": claim.get("can_claim_high_credibility"),
        "confidence": claim.get("confidence"),
    }


def _display_decision(
    *,
    readiness: dict[str, Any],
    impact_claim: dict[str, Any],
    model_score: dict[str, Any],
) -> dict[str, Any]:
    if int(model_score.get("paper_count", 0) or 0) > 0:
        claim_status = impact_claim.get("status")
        can_report = claim_status in {"validated", "evidence_supported"}
        return {
            "status": claim_status or model_score.get("ranking_quality_status"),
            "can_report": can_report,
            "evidence_level": "model_score_dataset",
            "basis": "model_score_dataset",
            "reason": (
                "Model-score dataset decision uses the model-score impact claim "
                "certificate rather than full-pipeline project readiness."
            ),
        }
    return {
        "status": readiness.get("status"),
        "can_report": readiness.get("can_report"),
        "evidence_level": readiness.get("evidence_level"),
        "basis": "project_readiness",
        "reason": readiness.get("reason"),
    }


def _model_credibility_summary(source: Path) -> dict[str, Any]:
    card = _read_json(source / "model_credibility_card.json")
    decision = card.get("decision", {})
    summary = card.get("summary", {})
    return {
        "status": decision.get("status"),
        "grade": decision.get("grade"),
        "served_model": card.get("served_model"),
        "model_identity_match": decision.get("model_identity_match"),
        "profile_model_names": decision.get("profile_model_names", []),
        "can_use_for_report_with_disclosure": decision.get(
            "can_use_for_report_with_disclosure"
        ),
        "can_claim_high_model_credibility": decision.get(
            "can_claim_high_model_credibility"
        ),
        "standalone_model_claim_allowed": decision.get(
            "standalone_model_claim_allowed"
        ),
        "pooled_accuracy": summary.get("pooled_accuracy"),
        "min_seed_accuracy": summary.get("min_seed_accuracy"),
        "total_examples": summary.get("total_examples"),
    }


def _calibration_profile_summary(source: Path) -> dict[str, Any]:
    card = _read_json(source / "calibration_profile_card.json")
    decision = card.get("decision", {})
    summary = card.get("summary", {})
    return {
        "status": decision.get("status"),
        "served_model": card.get("served_model"),
        "model_identity_match": decision.get("model_identity_match"),
        "compatible_models": decision.get("compatible_models", []),
        "can_use_calibrated_confidence": decision.get(
            "can_use_calibrated_confidence"
        ),
        "quality_supported": decision.get("quality_supported"),
        "scored_examples": summary.get("scored_examples"),
        "ece": summary.get("ece"),
        "mce": summary.get("mce"),
        "brier_score": summary.get("brier_score"),
    }


def _model_score_summary(source: Path) -> dict[str, Any]:
    report = _read_json(source / "model_impact_report.json")
    validity = _read_json(source / "impact_validity" / "impact_validity_evaluation.json")
    if not validity:
        validity = _read_json(source / "reports" / "impact_validity_evaluation.json")
    scorecard = validity.get("credibility_scorecard", {})
    summary = report.get("summary", {})
    ranking_quality = summary.get("ranking_quality", {})
    validated_claim = summary.get("validated_impact_claim", {})
    return {
        "paper_count": summary.get("paper_count", 0),
        "judgment_count": summary.get("judgment_count", 0),
        "parseable_rate": summary.get("parseable_rate", 0.0),
        "scoring_method": summary.get("scoring_method"),
        "ranking_quality_status": ranking_quality.get("status"),
        "validated_impact_claim_status": validated_claim.get("status"),
        "impact_credibility_status": scorecard.get("status"),
        "impact_credibility_grade": scorecard.get("grade"),
    }


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    gold = benchmark.get("gold_quality", {})
    baseline = benchmark.get("gold_baseline_comparison", {})
    decision = benchmark.get("decision", {})
    return {
        "paper_count": benchmark.get("benchmark", {}).get("paper_count", 0),
        "hit_rate": gold.get("hit_rate", 0.0),
        "mean_ndcg": gold.get("mean_ndcg", 0.0),
        "judge_ndcg_delta_vs_best_baseline": baseline.get(
            "judge_mean_ndcg_delta_vs_best_baseline",
            0.0,
        ),
        "judge_ndcg_delta_ci95": baseline.get("judge_ndcg_delta_ci95", [0.0, 0.0]),
        "statistically_positive_ndcg_delta": baseline.get(
            "statistically_positive_ndcg_delta",
            False,
        ),
        "needs_prompt_or_model_tuning": decision.get(
            "needs_prompt_or_model_tuning",
            False,
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _ci(value: Any) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _identity_match_text(value: Any) -> str:
    if value is None:
        return "not_checked"
    return _yes_no(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
